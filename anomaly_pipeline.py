"""Train a production-oriented session-level network anomaly detector.

Packets are grouped into sessions: consecutive packets between the same pair
of endpoints, split on new TCP connections and on idle gaps. Each session is
described by its own behavior and by its initiator's recent history (how many
of its last sessions failed, went to the same peer, or touched new hosts and
ports), which exposes slow repeated attacks such as password spraying that
look benign one at a time. History is counted in sessions rather than seconds
so it does not depend on how busy the network is.

Whole captures are labeled (clean = 0, attack = 1); several clean and attack
captures from different networks can be combined. Model selection uses
leakage-aware blocks of consecutive sessions, the decision threshold is tuned
on out-of-fold training predictions only, and independently labeled mixed
captures serve as external test sets.

Alerts come from three layers: the classifier (attacks like those it was
trained on), a benign-only anomaly layer (sessions unlike any normal
traffic), and deterministic rules (ARP spoofing). A session alerts when any
layer fires, and `alert_sources` records which.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
    train_test_split,
)

# Wireshark reports the highest decoded layer, so SSL/HTTP/DNS rows must be
# mapped back to their transport protocol.
TCP_PROTOCOLS = frozenset(
    {"TCP", "SSL", "TLSV1", "TLSV1.2", "TLSV1.3", "HTTP", "SSH", "SSHV2",
     "SMTP", "RSYNC", "NBSS", "SMB", "SMB2", "PGSQL", "FTP"}
)
UDP_PROTOCOLS = frozenset(
    {"UDP", "DNS", "NTP", "MDNS", "SSDP", "DHCP", "SNMP", "OPENVPN"}
)

# Columns read from a per-packet ground-truth file. `event_id` is optional;
# when present it enables incident-level metrics, the incident threshold rule,
# and removal of incidents shared between training and test captures.
LABEL_FILE_COLUMNS = ["packet_no", "frame_bytes", "label", "scenario"]
OPTIONAL_LABEL_COLUMNS = ["event_id"]
# Label files used when none are given on the command line. The ornek_*
# packet labels are derived from flow labels by flow_labels_to_packets.py.
DEFAULT_LABELS = {
    Path("data/ornek_temiz.csv"): Path("data/ornek_temiz_paket_etiketleri.csv"),
    Path("data/ornek_saldiri.csv"): Path("data/ornek_saldiri_paket_etiketleri.csv"),
    Path("data/yeni_temiz.csv"): Path("data/yeni_temiz_etiketleri.csv"),
    Path("data/yeni_saldiri.csv"): Path("data/yeni_saldiri_etiketleri.csv"),
    Path("data/ornek_karisik.csv"): Path("data/ornek_karisik_etiketli.csv"),
    Path("data/yeni_karisik.csv"): Path("data/yeni_karisik_etiketleri.csv"),
}

# Host history covers each initiator's most recent sessions, including the
# current one. Sessions this short are typical of probes and failed attempts.
HISTORY_SESSIONS = 20
SHORT_SESSION_PACKETS = 3
# The benign-only anomaly layer's false-positive budget. Its alerts add to
# the classifier's, so it is kept below the classifier's cap.
ANOMALY_FALSE_POSITIVE_RATE = 0.005

SESSION_FEATURE_COLUMNS = [
    "packet_count",
    "duration",
    "packet_rate",
    "total_bytes",
    "forward_packet_ratio",
    "forward_byte_ratio",
    "mean_packet_length",
    "packet_length_std",
    "packet_length_cv",
    "max_packet_length",
    "iat_mean",
    "iat_std",
    "iat_max",
    "tcp_ratio",
    "udp_ratio",
    "icmp_ratio",
    "arp_ratio",
    "dns_ratio",
    "protocol_diversity",
    "syn_flag_rate",
    "syn_ack_ratio",
    "rst_flag_rate",
    "http_error_ratio",
    "max_dns_name_length",
    "max_dns_label_length",
    "max_dns_label_entropy",
    "max_dns_label_digit_ratio",
    "unique_dest_ports",
    "service_port",
]
CONTEXT_FEATURE_COLUMNS = [
    "initiator_recent_sessions",
    "initiator_recent_failed_ratio",
    "initiator_recent_same_responder_ratio",
    "initiator_recent_unique_responders",
    "initiator_recent_unique_ports",
    "initiator_recent_short_ratio",
]
FEATURE_COLUMNS = SESSION_FEATURE_COLUMNS + CONTEXT_FEATURE_COLUMNS

COLUMN_ALIASES = {
    "No.": "packet_number",
    "No": "packet_number",
    "Time": "time",
    "Source": "source_ip",
    "Source_IP": "source_ip",
    "Destination": "destination_ip",
    "Destination_IP": "destination_ip",
    "Protocol": "protocol",
    "Length": "length",
    "Info": "info",
}
REQUIRED_COLUMNS = {
    "time",
    "source_ip",
    "destination_ip",
    "protocol",
    "length",
    "info",
}


def load_capture(
    path: Path,
    capture_id: str,
    label: int | None = None,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """Load one capture; `label` applies to every row, None leaves it unlabeled."""
    frame = pd.read_csv(path, encoding=encoding, low_memory=False)
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.rename(columns=COLUMN_ALIASES)

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    frame["row_position"] = np.arange(1, len(frame) + 1, dtype=np.int64)
    frame["capture_id"] = capture_id
    if label is not None:
        if label not in (0, 1):
            raise ValueError("label must be 0, 1, or None")
        frame["label"] = np.int8(label)

    frame["time"] = pd.to_numeric(frame["time"], errors="coerce")
    frame["length"] = pd.to_numeric(frame["length"], errors="coerce")
    invalid_time_count = int(frame["time"].isna().sum())
    if invalid_time_count:
        raise ValueError(f"{path} contains {invalid_time_count} invalid timestamps")

    frame["length"] = frame["length"].fillna(0).clip(lower=0)
    frame["protocol"] = frame["protocol"].fillna("").astype(str).str.upper()
    frame["info"] = frame["info"].fillna("").astype(str)
    frame["source_ip"] = frame["source_ip"].fillna("UNKNOWN").astype(str)
    frame["destination_ip"] = (
        frame["destination_ip"].fillna("UNKNOWN").astype(str)
    )
    return frame


def load_and_label_data(
    clean_paths: list[Path],
    attack_paths: list[Path],
    encoding: str = "utf-8",
    labels_paths: dict[Path, Path | None] | None = None,
) -> pd.DataFrame:
    """Label and merge captures while retaining independent timeline IDs.

    Attack captures contain only attack scenarios (scans, brute force, DNS
    tunneling, ARP spoofing, lateral SMB, exfiltration), so every row is 1.
    A per-packet label file, when given for a capture, adds `scenario` and
    `event_id` and must agree with the capture's role. Each file keeps its own
    capture ID, so sessions and host history never span two captures.
    """
    labels_paths = labels_paths or {}
    paths = [(path, 0) for path in clean_paths] + [(path, 1) for path in attack_paths]
    capture_ids = [f"{'attack' if label else 'clean'}:{path.stem}" for path, label in paths]
    if len(set(capture_ids)) != len(capture_ids):
        raise ValueError("training capture file names must be unique")
    frames = []
    for (path, label), capture_id in zip(paths, capture_ids):
        labels_path = labels_paths.get(path)
        if labels_path is None:
            frames.append(load_capture(path, capture_id, label=label, encoding=encoding))
            continue
        frame = attach_labels(
            load_capture(path, capture_id, label=None, encoding=encoding),
            path,
            labels_path,
        )
        wrong = int((frame["label"] != label).sum())
        if wrong:
            role = "attack" if label else "clean"
            raise ValueError(
                f"{labels_path} marks {wrong} packets of {role} capture {path} "
                f"as {'benign' if label else 'attack'}"
            )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


def load_mixed_capture(
    path: Path, labels_path: Path | None = None, encoding: str = "utf-8"
) -> pd.DataFrame:
    """Load a mixed capture and, if given, join its per-packet ground truth."""
    frame = load_capture(path, f"mixed:{path.stem}", label=None, encoding=encoding)
    if labels_path is None:
        return frame
    return attach_labels(frame, path, labels_path)


def attach_labels(frame: pd.DataFrame, path: Path, labels_path: Path) -> pd.DataFrame:
    """Join per-packet ground truth onto a loaded capture.

    The label file is matched on Wireshark's `No.` column. Frame lengths must
    agree on every row, which guards against pairing a stale label export with
    a regenerated capture.
    """
    if "packet_number" not in frame.columns:
        raise ValueError(f"{path} has no 'No.' column to join labels on")
    available = pd.read_csv(labels_path, nrows=0).columns
    columns = LABEL_FILE_COLUMNS + [
        column for column in OPTIONAL_LABEL_COLUMNS if column in available
    ]
    labels = pd.read_csv(labels_path, usecols=columns)
    if not labels["packet_no"].is_unique:
        raise ValueError(f"{labels_path} contains duplicate packet_no values")

    merged = frame.merge(
        labels,
        left_on="packet_number",
        right_on="packet_no",
        how="left",
        validate="one_to_one",
    )
    unmatched = int(merged["label"].isna().sum())
    if unmatched or len(labels) != len(frame):
        raise ValueError(
            f"{labels_path} does not cover {path}: {unmatched} capture rows "
            f"unmatched, {len(labels)} label rows vs {len(frame)} packets"
        )
    length_mismatches = int((merged["frame_bytes"] != merged["length"]).sum())
    if length_mismatches:
        raise ValueError(
            f"{length_mismatches} packets differ in length between {path} and "
            f"{labels_path}; the label file belongs to another capture"
        )
    if not merged["label"].isin([0, 1]).all():
        raise ValueError(f"{labels_path} labels must be 0 or 1")

    merged["label"] = merged["label"].astype(np.int8)
    return merged.drop(columns=["packet_no", "frame_bytes"])


def assign_sessions(packets: pd.DataFrame, gap_seconds: float) -> pd.DataFrame:
    """Attach a `session_id` to every packet.

    A session is one connection or transaction: a run of packets between the
    same unordered endpoint pair that ends at an idle gap longer than
    `gap_seconds` or when a new exchange opens (a TCP [SYN], a DNS query, an
    ICMP echo request, or an ARP request). On a busy network the idle gap may
    never occur, so without these splits thousands of repeated attempts,
    queries, or probes would merge into one session and give the model a
    handful of examples instead of thousands. Ports are not used because
    Wireshark omits them on application-layer rows (SSL, HTTP, ...).
    """
    if gap_seconds <= 0:
        raise ValueError("gap_seconds must be greater than zero")

    data = packets.copy()
    source = data["source_ip"]
    destination = data["destination_ip"]
    forward_order = source <= destination
    data["endpoint_a"] = source.where(forward_order, destination)
    data["endpoint_b"] = destination.where(forward_order, source)
    data = data.sort_values(
        ["capture_id", "endpoint_a", "endpoint_b", "time", "row_position"],
        kind="stable",
    )
    pair_keys = ["capture_id", "endpoint_a", "endpoint_b"]
    same_pair = (data[pair_keys] == data[pair_keys].shift()).all(axis=1)
    protocol = data["protocol"]
    info = data["info"]
    opens_exchange = (
        (protocol.eq("TCP") & info.str.contains(r"\[SYN\]", regex=True, na=False))
        | (protocol.eq("DNS") & info.str.match(r"Standard query 0x", na=False))
        # Replies say "(request in N)", so match the request itself.
        | (protocol.str.startswith("ICMP")
           & info.str.contains(r"Echo \(ping\) request", regex=True, na=False))
        | (protocol.eq("ARP") & info.str.startswith("Who has", na=False))
    )
    new_session = (
        ~same_pair | (data["time"].diff() > gap_seconds) | opens_exchange
    )
    session_number = new_session.cumsum()
    data["session_id"] = data["capture_id"] + ":" + session_number.astype(str)
    data = data.drop(columns=["endpoint_a", "endpoint_b"])
    return data.sort_values(["capture_id", "time", "row_position"], kind="stable")


def fit_anomaly_layer(
    train: pd.DataFrame,
    groups: np.ndarray,
    max_false_positive_rate: float = ANOMALY_FALSE_POSITIVE_RATE,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit an Isolation Forest on benign training sessions only.

    The classifier only recognizes attacks resembling those it was trained
    on; this layer flags sessions unlike any normal traffic, whatever the
    attack. Benign captures from different networks differ in size by an
    order of magnitude, so each capture gets equal total weight, and the
    threshold is the strictest per-capture (1 - cap) quantile of anomaly
    scores on benign sessions held out by grouped CV. A pooled quantile would
    fit the largest network and over-alert on the others.
    """
    is_benign = train["label"].to_numpy() == 0
    benign = train.loc[is_benign, FEATURE_COLUMNS]
    captures = train.loc[is_benign, "capture_id"].to_numpy()
    benign_groups = np.asarray(groups)[is_benign]
    _, inverse, counts = np.unique(captures, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    folds = min(3, len(np.unique(benign_groups)))
    if folds < 2:
        raise ValueError("anomaly layer needs at least two benign blocks")

    def forest() -> IsolationForest:
        return IsolationForest(n_estimators=300, random_state=random_state, n_jobs=-1)

    held_out_scores = np.empty(len(benign))
    for fold_train, fold_test in GroupKFold(n_splits=folds).split(benign, groups=benign_groups):
        fold_model = forest().fit(benign.iloc[fold_train], sample_weight=weights[fold_train])
        held_out_scores[fold_test] = -fold_model.score_samples(benign.iloc[fold_test])
    threshold = max(
        float(np.quantile(held_out_scores[captures == capture], 1 - max_false_positive_rate))
        for capture in np.unique(captures)
    )
    return {
        "model": forest().fit(benign, sample_weight=weights),
        "threshold": threshold,
        "max_false_positive_rate": float(max_false_positive_rate),
        "feature_columns": list(FEATURE_COLUMNS),
    }


def score_anomaly_layer(layer: dict[str, Any], sessions: pd.DataFrame) -> np.ndarray:
    """Return True for sessions the benign-only layer considers anomalous."""
    scores = -layer["model"].score_samples(sessions[layer["feature_columns"]])
    return scores >= layer["threshold"]


def rule_alerts(
    packets: pd.DataFrame, sessions: pd.DataFrame, arp_request_window: float = 2.0
) -> pd.DataFrame:
    """Deterministic alerts that complement the model, with a reason per session.

    ARP spoofing looks different on every network (one MAC flooding replies,
    or many MACs each claiming the gateway once), but it always sends replies
    nobody asked for and can rebind an address to a new MAC. A reply is
    unsolicited when no "Who has <ip>?" request for its address appeared in
    the preceding `arp_request_window` seconds; a conflict is a reply binding
    an address to a different MAC than the first one seen in the capture.
    Gratuitous ARP from legitimate failover also triggers this, so on real
    networks known failover pairs may need an allow-list.

    Returns one row per session in `sessions` order: `rule_alert` and
    `rule_reason` ("" when no rule fired).
    """
    arp = packets[packets["protocol"].eq("ARP")].sort_values(
        ["capture_id", "time", "row_position"], kind="stable"
    )
    replies = arp["info"].str.extract(r"^(\d+\.\d+\.\d+\.\d+) is at ([0-9A-Fa-f:]+)")
    requests = arp["info"].str.extract(r"^Who has (\d+\.\d+\.\d+\.\d+)\?", expand=False)

    reasons: dict[str, set[str]] = {}
    last_request: dict[tuple[str, str], float] = {}
    bound_mac: dict[tuple[str, str], str] = {}
    for capture, time, session, asked, claimed, mac in zip(
        arp["capture_id"], arp["time"], arp["session_id"],
        requests, replies[0], replies[1],
    ):
        if isinstance(asked, str):
            last_request[(capture, asked)] = time
            continue
        if not isinstance(claimed, str):
            continue
        key = (capture, claimed)
        if time - last_request.get(key, -np.inf) > arp_request_window:
            reasons.setdefault(session, set()).add("arp_unsolicited_reply")
        if bound_mac.setdefault(key, mac.lower()) != mac.lower():
            reasons.setdefault(session, set()).add("arp_mac_conflict")

    reason = sessions["session_id"].map(
        lambda session_id: ",".join(sorted(reasons.get(session_id, ())))
    )
    return pd.DataFrame(
        {
            "session_id": sessions["session_id"].to_numpy(),
            "rule_alert": reason.ne("").to_numpy(),
            "rule_reason": reason.to_numpy(),
        }
    )


def _shannon_entropy(text: Any) -> float:
    """Bits per character of a string; NaN for missing values."""
    if not isinstance(text, str) or not text:
        return np.nan
    _, counts = np.unique(list(text), return_counts=True)
    shares = counts / counts.sum()
    return float(-(shares * np.log2(shares)).sum())


def _annotate_packets(data: pd.DataFrame) -> pd.DataFrame:
    """Add per-packet protocol, flag, and payload indicators."""
    protocol = data["protocol"]
    info = data["info"]
    data["is_tcp"] = protocol.isin(TCP_PROTOCOLS).astype(np.int8)
    data["is_udp"] = protocol.isin(UDP_PROTOCOLS).astype(np.int8)
    data["is_icmp"] = protocol.str.startswith("ICMP").astype(np.int8)
    data["is_arp"] = protocol.eq("ARP").astype(np.int8)
    data["is_dns"] = protocol.eq("DNS").astype(np.int8)
    data["is_http"] = protocol.eq("HTTP").astype(np.int8)
    # Flags only appear on raw TCP rows. A lone [SYN] is an initiating scan
    # packet; [SYN, ACK] is a response, so unanswered SYNs lower the ratio.
    raw_tcp = protocol.eq("TCP")
    data["is_syn"] = (
        raw_tcp & info.str.contains(r"\[SYN\]", regex=True, na=False)
    ).astype(np.int8)
    data["is_syn_ack"] = (
        raw_tcp & info.str.contains(r"\[SYN, ACK\]", regex=True, na=False)
    ).astype(np.int8)
    data["is_rst"] = (
        raw_tcp & info.str.contains(r"\[[^]]*\bRST\b", regex=True, na=False)
    ).astype(np.int8)
    # 401/404 bursts indicate login brute force and directory enumeration.
    data["is_http_error"] = (
        data["is_http"].astype(bool)
        & info.str.match(r"HTTP/\d\.\d\s+[45]\d\d\b", na=False)
    ).astype(np.int8)
    # Long encoded query names are characteristic of DNS tunneling.
    dns_names = info.str.extract(
        r"Standard query (?:response )?0x[0-9a-fA-F]+\s+"
        r"(?:No such name\s+)?[A-Z]+\s+(\S+)",
        expand=False,
    )
    dns_names = dns_names.where(data["is_dns"].astype(bool))
    data["dns_name_length"] = dns_names.str.len().fillna(0)
    # Tunnels and random-subdomain floods put generated strings in the first
    # label; real host names are short, low-entropy words.
    first_labels = dns_names.str.split(".").str[0]
    data["dns_label_length"] = first_labels.str.len().fillna(0)
    data["dns_label_entropy"] = first_labels.map(_shannon_entropy).fillna(0)
    data["dns_label_digit_ratio"] = (
        first_labels.str.count(r"\d") / first_labels.str.len().replace(0, np.nan)
    ).fillna(0)
    data["is_nxdomain"] = (
        data["is_dns"].astype(bool) & info.str.contains("No such name", na=False)
    ).astype(np.int8)

    # Ports are only printed as "src  >  dst" on raw TCP/UDP rows.
    destination_ports = info.str.extract(r"^\s*\d+\s+>\s+(\d+)\b", expand=False)
    transport_mask = raw_tcp | protocol.eq("UDP")
    data["destination_port"] = pd.to_numeric(
        destination_ports.where(transport_mask), errors="coerce"
    )
    return data


def engineer_session_features(packets: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sessionized packets and add the initiator's recent history."""
    if "session_id" not in packets.columns:
        raise ValueError("call assign_sessions() before engineering features")

    data = _annotate_packets(
        packets.sort_values(["session_id", "time", "row_position"], kind="stable")
    )
    sessions_group = data.groupby("session_id", sort=False)
    # The sender of the first packet is treated as the initiator.
    data["initiator"] = sessions_group["source_ip"].transform("first")
    data["responder"] = sessions_group["destination_ip"].transform("first")
    data["is_forward"] = data["source_ip"].eq(data["initiator"]).astype(np.int8)
    data["forward_bytes"] = data["length"] * data["is_forward"]
    data["forward_port"] = data["destination_port"].where(
        data["is_forward"].astype(bool)
    )
    data["iat"] = sessions_group["time"].diff()

    # The label is optional so unlabeled captures can be scored too. A session
    # is an attack if any of its packets is. Scenario and incident are carried
    # along for evaluation only; they are never model features.
    label_aggregations = (
        {"label": ("label", "max")} if "label" in data.columns else {}
    )
    if "scenario" in data.columns:
        label_aggregations["scenario"] = ("scenario", "first")
    if "event_id" in data.columns and "label" in data.columns:
        data["attack_event_id"] = data["event_id"].where(data["label"] == 1)
        label_aggregations["event_id"] = ("attack_event_id", "first")
    sessions = (
        data.groupby("session_id", as_index=False, sort=False)
        .agg(
            capture_id=("capture_id", "first"),
            initiator=("initiator", "first"),
            responder=("responder", "first"),
            start_time=("time", "min"),
            end_time=("time", "max"),
            packet_count=("time", "size"),
            forward_packets=("is_forward", "sum"),
            total_bytes=("length", "sum"),
            forward_bytes=("forward_bytes", "sum"),
            mean_packet_length=("length", "mean"),
            packet_length_std=("length", "std"),
            max_packet_length=("length", "max"),
            iat_mean=("iat", "mean"),
            iat_std=("iat", "std"),
            iat_max=("iat", "max"),
            tcp_count=("is_tcp", "sum"),
            udp_count=("is_udp", "sum"),
            icmp_count=("is_icmp", "sum"),
            arp_count=("is_arp", "sum"),
            dns_count=("is_dns", "sum"),
            http_count=("is_http", "sum"),
            http_error_count=("is_http_error", "sum"),
            syn_count=("is_syn", "sum"),
            syn_ack_count=("is_syn_ack", "sum"),
            rst_count=("is_rst", "sum"),
            nxdomain_count=("is_nxdomain", "sum"),
            max_dns_name_length=("dns_name_length", "max"),
            max_dns_label_length=("dns_label_length", "max"),
            max_dns_label_entropy=("dns_label_entropy", "max"),
            max_dns_label_digit_ratio=("dns_label_digit_ratio", "max"),
            unique_dest_ports=("destination_port", "nunique"),
            protocol_diversity=("protocol", "nunique"),
            # Zero after fillna means no raw TCP/UDP row revealed the port.
            service_port=("forward_port", "first"),
            **label_aggregations,
        )
    )
    text_columns = ["session_id", "capture_id", "initiator", "responder",
                    "scenario", "event_id"]
    numeric_columns = sessions.columns.difference(text_columns)
    sessions[numeric_columns] = sessions[numeric_columns].fillna(0)

    count = sessions["packet_count"]
    sessions["duration"] = sessions["end_time"] - sessions["start_time"]
    sessions["packet_rate"] = _safe_ratio(count, sessions["duration"])
    sessions["forward_packet_ratio"] = sessions["forward_packets"] / count
    sessions["forward_byte_ratio"] = _safe_ratio(
        sessions["forward_bytes"], sessions["total_bytes"]
    )
    sessions["packet_length_cv"] = _safe_ratio(
        sessions["packet_length_std"], sessions["mean_packet_length"]
    )
    for name in ("tcp", "udp", "icmp", "arp", "dns"):
        sessions[f"{name}_ratio"] = sessions[f"{name}_count"] / count
    sessions["syn_flag_rate"] = _safe_ratio(
        sessions["syn_count"], sessions["tcp_count"]
    )
    sessions["syn_ack_ratio"] = _safe_ratio(
        sessions["syn_ack_count"], sessions["syn_count"]
    )
    sessions["rst_flag_rate"] = _safe_ratio(
        sessions["rst_count"], sessions["tcp_count"]
    )
    sessions["http_error_ratio"] = _safe_ratio(
        sessions["http_error_count"], sessions["http_count"]
    )
    # A failed session was reset, never answered, refused by HTTP, or asked
    # for a name that does not exist.
    sessions["failed"] = (
        (sessions["rst_count"] > 0)
        | ((sessions["syn_count"] > 0) & (sessions["syn_ack_count"] == 0))
        | (sessions["http_error_count"] > 0)
        | (sessions["nxdomain_count"] > 0)
    ).astype(np.int8)
    if "label" in sessions.columns:
        sessions["label"] = sessions["label"].astype(np.int8)

    sessions = add_history_context(sessions, HISTORY_SESSIONS)
    ordered_columns = [
        "capture_id",
        "session_id",
        "initiator",
        "responder",
        "start_time",
        "end_time",
        *FEATURE_COLUMNS,
        *label_aggregations,
    ]
    return sessions[ordered_columns]


def add_history_context(sessions: pd.DataFrame, history: int) -> pd.DataFrame:
    """Describe each initiator's last `history` sessions, including the current.

    A single password attempt looks like a normal login; a run of short,
    failed sessions to one host, or to many hosts and ports, does not. History
    is counted in sessions, not seconds, so the features mean the same on a
    quiet network and a busy one. Only past sessions are used, so the features
    are available at scoring time in a live deployment.
    """
    if history < 1:
        raise ValueError("history must be at least one session")

    result = sessions.sort_values(
        ["capture_id", "start_time", "session_id"], kind="stable"
    ).reset_index(drop=True)
    failed = result["failed"].to_numpy(dtype=float)
    short = (result["packet_count"] <= SHORT_SESSION_PACKETS).to_numpy(dtype=float)
    responders = result["responder"].to_numpy()
    ports = result["service_port"].to_numpy()
    context = {column: np.zeros(len(result)) for column in CONTEXT_FEATURE_COLUMNS}

    for index in result.groupby(["capture_id", "initiator"], sort=False).indices.values():
        for position, row in enumerate(index):
            recent = index[max(0, position - history + 1) : position + 1]
            context["initiator_recent_sessions"][row] = len(recent)
            context["initiator_recent_failed_ratio"][row] = failed[recent].mean()
            context["initiator_recent_same_responder_ratio"][row] = (
                responders[recent] == responders[row]
            ).mean()
            context["initiator_recent_unique_responders"][row] = len(
                set(responders[recent])
            )
            context["initiator_recent_unique_ports"][row] = len(set(ports[recent]))
            context["initiator_recent_short_ratio"][row] = short[recent].mean()

    for column, values in context.items():
        result[column] = values
    return result


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> np.ndarray:
    """Divide numeric series and return zero where the denominator is zero."""
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(len(numerator), dtype=float),
        where=denominator.to_numpy() != 0,
    )


def add_temporal_block_ids(
    sessions: pd.DataFrame, block_sessions: int
) -> pd.DataFrame:
    """Attach block IDs of `block_sessions` consecutive sessions per capture.

    Blocks are counted in sessions rather than seconds because captures differ
    by orders of magnitude in density: one hour holds a few sessions on a quiet
    network and tens of thousands on a busy one.
    """
    if block_sessions < 1:
        raise ValueError("block_sessions must be at least one")
    result = sessions.sort_values(
        ["capture_id", "start_time", "session_id"], kind="stable"
    ).reset_index(drop=True)
    block_number = result.groupby("capture_id").cumcount() // block_sessions
    result["block_id"] = (
        result["capture_id"].astype(str) + ":" + block_number.astype(str)
    )
    return result


def block_stratified_split(
    sessions: pd.DataFrame,
    test_size: float,
    random_state: int,
    block_sessions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole session blocks, stratified by block attack presence."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between zero and one")

    split_frame = add_temporal_block_ids(sessions, block_sessions)
    blocks = split_frame.groupby("block_id", as_index=False).agg(
        block_label=("label", "max")
    )
    class_counts = blocks["block_label"].value_counts()
    if len(class_counts) < 2 or int(class_counts.min()) < 2:
        raise ValueError(
            "Not enough positive and negative temporal blocks for a stratified "
            "split. Try a smaller --block-sessions value."
        )

    train_blocks, test_blocks = train_test_split(
        blocks["block_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=blocks["block_label"],
    )
    train = split_frame[split_frame["block_id"].isin(train_blocks)].copy()
    test = split_frame[split_frame["block_id"].isin(test_blocks)].copy()
    for name, partition in (("training", train), ("test", test)):
        if partition["label"].nunique() < 2:
            raise ValueError(
                f"The {name} partition has one class. Try a smaller "
                "--block-sessions value or another --random-state."
            )
    return train, test


def _build_grouped_cv(
    labels: pd.Series,
    groups: pd.Series,
    requested_splits: int,
    random_state: int,
) -> StratifiedGroupKFold:
    """Create CV with no more folds than positive temporal groups permit."""
    group_labels = pd.DataFrame({"group": groups, "label": labels}).groupby(
        "group"
    )["label"].max()
    positive_groups = int(group_labels.sum())
    negative_groups = int((group_labels == 0).sum())
    splits = min(requested_splits, positive_groups, negative_groups)
    if splits < 2:
        raise ValueError(
            "Training data needs at least two positive and two negative temporal "
            "blocks for grouped cross-validation."
        )
    return StratifiedGroupKFold(
        n_splits=splits, shuffle=True, random_state=random_state
    )


def optimize_random_forest(
    train: pd.DataFrame,
    search_iterations: int,
    cv_splits: int,
    random_state: int,
) -> tuple[RandomForestClassifier, pd.DataFrame, np.ndarray, int]:
    """Tune RF hyperparameters and generate grouped out-of-fold probabilities."""
    if search_iterations < 1:
        raise ValueError("search_iterations must be at least one")
    x_train = train[FEATURE_COLUMNS]
    y_train = train["label"]
    groups = train["block_id"]
    cv = _build_grouped_cv(y_train, groups, cv_splits, random_state)

    base_model = RandomForestClassifier(
        n_jobs=-1,
        random_state=random_state,
    )
    parameter_space = {
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [None, 8, 12, 18, 24],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5, None],
        "class_weight": ["balanced", "balanced_subsample", {0: 1, 1: 4}],
    }
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=parameter_space,
        n_iter=search_iterations,
        scoring="average_precision",
        n_jobs=-1,
        cv=cv,
        random_state=random_state,
        refit=True,
        return_train_score=False,
        verbose=0,
    )
    search.fit(x_train, y_train, groups=groups)

    best_template = clone(search.best_estimator_)
    oof_probabilities = cross_val_predict(
        best_template,
        x_train,
        y_train,
        groups=groups,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    results = pd.DataFrame(search.cv_results_).sort_values(
        "rank_test_score"
    )
    result_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "mean_fit_time",
        "params",
    ]
    return (
        search.best_estimator_,
        results[result_columns],
        oof_probabilities,
        cv.n_splits,
    )


def tune_probability_threshold(
    labels: pd.Series,
    probabilities: np.ndarray,
    max_false_positive_rate: float,
    event_ids: pd.Series | None = None,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    """Pick the OOF threshold that catches the most attacks within an FPR cap.

    Missed attacks cost more than false alarms, so recall is the objective.
    The cap limits the share of benign sessions that raise an alert. It is not
    a precision floor because precision depends on how common attacks are:
    training captures are attack-heavy, so a precision floor that holds in
    training permits far more alerts on a real network where attacks are rare.

    With `event_ids`, recall is counted in incidents: an incident is caught
    when any of its sessions alerts, which is what an analyst needs, and it
    usually allows a higher threshold and fewer false alarms than catching
    every session. Attack sessions without an incident count as their own.
    Ties go to the higher threshold, which raises fewer alerts.
    """
    if not 0 <= max_false_positive_rate < 1:
        raise ValueError("max_false_positive_rate must be in [0, 1)")

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    false_positive_rate, recall, thresholds = roc_curve(
        labels, probabilities, drop_intermediate=False
    )
    positives = int(labels.sum())
    negatives = len(labels) - positives
    true_positives = recall * positives
    flagged = true_positives + false_positive_rate * negatives
    curve = pd.DataFrame(
        {
            "threshold": thresholds,
            "false_positive_rate": false_positive_rate,
            "recall": recall,
            "precision": np.divide(
                true_positives,
                flagged,
                out=np.ones(len(flagged), dtype=float),
                where=flagged != 0,
            ),
        }
    )
    # roc_curve starts with an infinite threshold that flags nothing.
    candidates = curve[np.isfinite(curve["threshold"])].copy()

    rule = "session"
    if event_ids is not None:
        incidents = pd.Series(np.asarray(event_ids, dtype=object))[labels == 1]
        if incidents.notna().any():
            rule = "event"
            incidents = incidents.where(
                incidents.notna(), "session:" + incidents.index.astype(str)
            )
            incident_peak = (
                pd.Series(probabilities[labels == 1], index=incidents.index)
                .groupby(incidents.to_numpy())
                .max()
                .to_numpy()
            )
            candidates["event_recall"] = [
                float((incident_peak >= threshold).mean())
                for threshold in candidates["threshold"]
            ]

    objective = "event_recall" if rule == "event" else "recall"
    eligible = candidates[
        candidates["false_positive_rate"] <= max_false_positive_rate
    ]
    constraint_met = not eligible.empty
    if constraint_met:
        chosen = eligible.sort_values(
            [objective, "threshold"], ascending=False
        ).iloc[0]
    else:
        # Even the strictest threshold exceeds the cap; flag as little as possible.
        chosen = candidates.sort_values("threshold", ascending=False).iloc[0]

    details: dict[str, Any] = {
        "rule": rule,
        "threshold": float(chosen["threshold"]),
        "oof_recall": float(chosen["recall"]),
        "oof_false_positive_rate": float(chosen["false_positive_rate"]),
        "oof_precision": float(chosen["precision"]),
        "max_false_positive_rate": float(max_false_positive_rate),
        "fpr_constraint_met": bool(constraint_met),
    }
    if rule == "event":
        details["oof_event_recall"] = float(chosen["event_recall"])
        details["oof_events"] = int(len(incident_peak))
    return float(chosen["threshold"]), details, curve


def _classification_metrics(
    labels: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate threshold-dependent classification metrics."""
    predictions = (probabilities >= threshold).astype(np.int8)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "confusion_matrix": {
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
        "classification_report": classification_report(
            labels, predictions, output_dict=True, zero_division=0
        ),
    }


def evaluate_model(
    model: RandomForestClassifier,
    test: pd.DataFrame,
    tuned_threshold: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate default and tuned thresholds once on untouched test blocks."""
    labels = test["label"]
    probabilities = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    test_curve = pd.DataFrame(
        {
            "threshold": np.append(thresholds, np.nan),
            "precision": precision,
            "recall": recall,
        }
    )
    metrics = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "default_threshold": _classification_metrics(labels, probabilities, 0.5),
        "tuned_threshold": _classification_metrics(
            labels, probabilities, tuned_threshold
        ),
    }
    if "event_id" in test.columns:
        metrics["tuned_threshold_events"] = evaluate_events(
            test, probabilities, tuned_threshold
        )
    return metrics, test_curve


def evaluate_events(
    sessions: pd.DataFrame, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Incident-level view: which attacks were caught, how fast, at what alert cost.

    An incident is detected when any of its sessions alerts. Detection delay
    is the number of the incident's sessions, in start order, that passed
    before the first alert (0 means the first session alerted).
    """
    flagged = np.asarray(probabilities) >= threshold
    labels = sessions["label"].to_numpy()
    benign = labels == 0
    attack = sessions.loc[labels == 1, ["event_id", "start_time"]].assign(
        flagged=flagged[labels == 1]
    )
    attack = attack[attack["event_id"].notna()].sort_values("start_time", kind="stable")
    per_event = attack.groupby("event_id")["flagged"].agg(
        detected="max",
        sessions="size",
        sessions_before_detection=lambda verdicts: (
            float(np.argmax(verdicts.to_numpy())) if verdicts.any() else np.nan
        ),
    )
    detected = per_event["detected"].astype(bool)
    delays = per_event.loc[detected, "sessions_before_detection"]
    return {
        "attack_events": int(len(per_event)),
        "detected_events": int(detected.sum()),
        "event_recall": float(detected.mean()) if len(per_event) else None,
        "missed_events": sorted(per_event.index[~detected].astype(str)),
        "median_sessions_before_detection": float(delays.median()) if len(delays) else None,
        "max_sessions_before_detection": float(delays.max()) if len(delays) else None,
        "benign_sessions": int(benign.sum()),
        "false_alarm_sessions": int(flagged[benign].sum()),
        "false_positive_rate": float(flagged[benign].mean()) if benign.any() else None,
    }


def train_and_evaluate(
    sessions: pd.DataFrame,
    test_size: float = 0.25,
    random_state: int = 42,
    block_sessions: int = 200,
    search_iterations: int = 24,
    cv_splits: int = 3,
    max_false_positive_rate: float = 0.01,
    threshold_rule: str = "event",
    anomaly_false_positive_rate: float = ANOMALY_FALSE_POSITIVE_RATE,
) -> tuple[
    RandomForestClassifier,
    float,
    dict[str, Any] | None,
    dict[str, Any],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run outer split, grouped model search, OOF threshold tuning, and test.

    The anomaly layer is fitted on the same training partition, so the
    internal test blocks stay untouched by every layer. An
    `anomaly_false_positive_rate` of 0 disables it.
    """
    train, test = block_stratified_split(
        sessions, test_size, random_state, block_sessions
    )
    model, search_results, oof_probabilities, actual_cv_splits = (
        optimize_random_forest(
            train, search_iterations, cv_splits, random_state
        )
    )
    if threshold_rule not in ("event", "session"):
        raise ValueError("threshold_rule must be 'event' or 'session'")
    event_ids = (
        train["event_id"]
        if threshold_rule == "event" and "event_id" in train.columns
        else None
    )
    threshold, threshold_details, validation_curve = tune_probability_threshold(
        train["label"], oof_probabilities, max_false_positive_rate, event_ids
    )
    evaluation, test_curve = evaluate_model(model, test, threshold)
    anomaly_layer = (
        fit_anomaly_layer(
            train, train["block_id"].to_numpy(), anomaly_false_positive_rate,
            random_state,
        )
        if anomaly_false_positive_rate > 0
        else None
    )
    layered = predict_anomalies(
        {"model": model, "feature_columns": FEATURE_COLUMNS,
         "decision_threshold": threshold, "anomaly_layer": anomaly_layer},
        test,
    )
    if "event_id" in test.columns:
        evaluation["layered_events"] = layered_event_metrics(layered)

    metrics = {
        "data_split": {
            "train_sessions": len(train),
            "test_sessions": len(test),
            "train_attack_sessions": int(train["label"].sum()),
            "test_attack_sessions": int(test["label"].sum()),
            "block_sessions": int(block_sessions),
        },
        "model_selection": {
            "scoring": "average_precision",
            "search_iterations": int(search_iterations),
            "grouped_cv_splits": int(actual_cv_splits),
            "best_cv_average_precision": float(
                search_results.iloc[0]["mean_test_score"]
            ),
            "best_parameters": model.get_params(deep=False),
        },
        "threshold_selection": threshold_details,
        "test_evaluation": evaluation,
    }
    curves = pd.concat(
        [
            validation_curve.assign(dataset="training_oof"),
            test_curve.assign(dataset="test"),
        ],
        ignore_index=True,
    )
    return model, threshold, anomaly_layer, metrics, search_results, curves, test


def predict_anomalies(
    model_bundle: dict[str, Any], sessions: pd.DataFrame
) -> pd.DataFrame:
    """Apply a saved model bundle and combine its alert layers.

    `anomaly_probability` and `prediction` are the classifier's score and
    verdict. `anomaly_alert` comes from the benign-only layer when the bundle
    has one, and the rule columns from `add_rule_alerts` when present.
    `alert` is set when any layer fires; `alert_sources` names them.
    """
    features = model_bundle["feature_columns"]
    probabilities = model_bundle["model"].predict_proba(sessions[features])[:, 1]
    result = sessions.copy()
    result["anomaly_probability"] = probabilities
    result["prediction"] = (
        probabilities >= model_bundle["decision_threshold"]
    ).astype(np.int8)

    sources = {"model": result["prediction"].to_numpy().astype(bool)}
    if model_bundle.get("anomaly_layer") is not None:
        anomalous = score_anomaly_layer(model_bundle["anomaly_layer"], result)
        result["anomaly_alert"] = anomalous.astype(np.int8)
        sources["anomaly"] = anomalous
    if "rule_alert" in result.columns:
        sources["rules"] = result["rule_alert"].to_numpy().astype(bool)
    result["alert"] = np.logical_or.reduce(list(sources.values())).astype(np.int8)
    names = pd.DataFrame({name: np.where(fired, name, "") for name, fired in sources.items()})
    result["alert_sources"] = names.apply(
        lambda row: "+".join(name for name in row if name), axis=1
    ).to_numpy()
    return result


def layered_event_metrics(predictions: pd.DataFrame) -> dict[str, Any]:
    """Incident metrics for each alert layer alone and for their combination."""
    verdicts = {"model": predictions["prediction"]}
    if "anomaly_alert" in predictions.columns:
        verdicts["anomaly"] = predictions["anomaly_alert"]
    if "rule_alert" in predictions.columns:
        verdicts["rules"] = predictions["rule_alert"]
    verdicts["combined"] = predictions["alert"]
    return {
        name: evaluate_events(predictions, verdict.to_numpy().astype(float), 0.5)
        for name, verdict in verdicts.items()
    }


def add_rule_alerts(packets: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """Attach `rule_alert` and `rule_reason` to sessions; they are not features."""
    return sessions.merge(
        rule_alerts(packets, sessions), on="session_id", how="left", validate="one_to_one"
    )


def packet_level_predictions(
    packets: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """Give every packet the probability and verdict of its session."""
    columns = ["packet_number", "time", "session_id", "label"]
    columns += [c for c in ("scenario", "event_id") if c in packets.columns]
    verdicts = [c for c in ("prediction", "alert", "alert_sources") if c in predictions.columns]
    return packets[columns].merge(
        predictions[["session_id", "anomaly_probability", *verdicts]],
        on="session_id",
        how="left",
        validate="many_to_one",
    )


def evaluate_packets(
    packet_predictions: pd.DataFrame, threshold: float
) -> dict[str, Any]:
    """Packet-level metrics, comparable across session or window detectors."""
    labels = packet_predictions["label"]
    probabilities = packet_predictions["anomaly_probability"].to_numpy()
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "tuned_threshold": _classification_metrics(labels, probabilities, threshold),
    }


def scenario_flag_rates(packet_predictions: pd.DataFrame) -> pd.DataFrame:
    """Share of each scenario's packets whose session the model flagged.

    For attack scenarios this is per-scenario recall; for benign scenarios it
    is the false-alarm rate. With incidents, it also counts how many of each
    scenario's attack incidents raised at least one alert. The combined
    `alert` is used when present, otherwise the classifier's `prediction`.
    """
    verdict = "alert" if "alert" in packet_predictions.columns else "prediction"
    table = packet_predictions.groupby("scenario", as_index=False).agg(
        label=("label", "max"),
        packets=("label", "size"),
        sessions=("session_id", "nunique"),
        flag_rate=(verdict, "mean"),
    )
    if "event_id" in packet_predictions.columns:
        attacks = packet_predictions[
            packet_predictions["label"].eq(1) & packet_predictions["event_id"].notna()
        ]
        incidents = (
            attacks.groupby(["scenario", "event_id"])[verdict].max()
            .groupby("scenario")
            .agg(events="size", detected_events="sum")
            .reset_index()
        )
        table = table.merge(incidents, on="scenario", how="left")
    return table.sort_values(["label", "flag_rate"], ascending=[False, True])


def drop_shared_events(
    packets: pd.DataFrame, test_labels_paths: list[Path | None]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove training attack packets whose incident also appears in a test.

    A generator can emit the same incident (same `event_id`) into training and
    test captures; keeping it in training would make the test score the model
    on attacks it has already seen.
    """
    test_events: set[str] = set()
    for labels_path in test_labels_paths:
        if labels_path is None:
            continue
        available = pd.read_csv(labels_path, nrows=0).columns
        if "event_id" not in available:
            continue
        labels = pd.read_csv(labels_path, usecols=["label", "event_id"])
        test_events |= set(labels.loc[labels["label"] == 1, "event_id"].dropna())
    if "event_id" not in packets.columns or not test_events:
        return packets, {"shared_events": 0, "removed_packets": 0}

    shared = packets["label"].eq(1) & packets["event_id"].isin(test_events)
    report = {
        "shared_events": int(packets.loc[shared, "event_id"].nunique()),
        "removed_packets": int(shared.sum()),
        "removed_packets_per_capture": (
            packets.loc[shared, "capture_id"].value_counts().to_dict()
        ),
    }
    return packets[~shared.to_numpy()].copy(), report


def _optional_path(value: str) -> Path | None:
    """Parse a CLI path where '' or '-' means "none"."""
    return Path(value) if value not in ("", "-") else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tuned session-level network anomaly detector."
    )
    parser.add_argument(
        "--clean",
        type=Path,
        nargs="+",
        default=[Path("data/ornek_temiz.csv"), Path("data/yeni_temiz.csv")],
        help="Captures containing only benign traffic (labeled 0).",
    )
    parser.add_argument(
        "--clean-labels",
        type=_optional_path,
        nargs="*",
        default=None,
        help=(
            "Per-packet label file for each --clean capture, same order; '-' "
            "for none. Omitted: the bundled label files for known captures."
        ),
    )
    parser.add_argument(
        "--attack",
        type=Path,
        nargs="+",
        default=[Path("data/ornek_saldiri.csv"), Path("data/yeni_saldiri.csv")],
        help="Captures containing only attack traffic (labeled 1).",
    )
    parser.add_argument(
        "--attack-labels",
        type=_optional_path,
        nargs="*",
        default=None,
        help=(
            "Per-packet label file for each --attack capture, same order; '-' "
            "for none. Labels add scenarios and incidents (event_id)."
        ),
    )
    parser.add_argument(
        "--mixed",
        type=Path,
        nargs="*",
        default=[Path("data/ornek_karisik.csv"), Path("data/yeni_karisik.csv")],
        help="Mixed captures scored after training; pass none to skip.",
    )
    parser.add_argument(
        "--mixed-labels",
        type=_optional_path,
        nargs="*",
        default=None,
        help=(
            "Per-packet ground truth for each --mixed file, in the same order "
            "(packet_no, frame_bytes, label, scenario); use '-' to only score "
            "that capture. Omitted: the bundled label files for the default "
            "captures, none for others."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--session-gap",
        type=float,
        default=15.0,
        help=(
            "Idle seconds that end a session. In the sparse clean capture, gaps "
            "between packets of the same endpoint pair are almost all under "
            "15 s or over 120 s."
        ),
    )
    parser.add_argument(
        "--block-sessions",
        type=int,
        default=200,
        help=(
            "Consecutive sessions per block for leakage-safe splitting; much "
            f"larger than the {HISTORY_SESSIONS}-session host history."
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--search-iterations", type=int, default=24)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument(
        "--max-false-positive-rate",
        type=float,
        default=0.01,
        help=(
            "Largest share of benign OOF training sessions allowed to raise an "
            "alert; recall is maximized subject to it. Unlike a precision "
            "floor, it does not depend on how attack-heavy the training data is."
        ),
    )
    parser.add_argument(
        "--anomaly-false-positive-rate",
        type=float,
        default=ANOMALY_FALSE_POSITIVE_RATE,
        help=(
            "Share of held-out benign sessions per training network allowed to "
            "trip the benign-only anomaly layer; 0 disables the layer."
        ),
    )
    parser.add_argument(
        "--threshold-rule",
        choices=["event", "session"],
        default="event",
        help=(
            "Maximize caught incidents (event_id) or caught sessions under the "
            "FPR cap. 'event' falls back to 'session' without incident labels."
        ),
    )
    parser.add_argument(
        "--keep-shared-events",
        action="store_true",
        help=(
            "Keep training attack incidents that also appear in a labeled "
            "--mixed capture. By default they are removed so tests stay "
            "independent; keep them only for a final model trained on all data."
        ),
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="CSV encoding of the Wireshark/tshark exports.",
    )
    parser.add_argument(
        "--save-labeled-packets",
        action="store_true",
        help="Also save the large packet-level merged dataframe.",
    )
    args = parser.parse_args()
    for captures, option in (
        ("clean", "clean_labels"),
        ("attack", "attack_labels"),
        ("mixed", "mixed_labels"),
    ):
        paths = getattr(args, captures)
        labels = getattr(args, option)
        if labels is None:
            labels = [DEFAULT_LABELS.get(path) for path in paths]
            labels = [path if path and path.exists() else None for path in labels]
            setattr(args, option, labels)
        if len(labels) != len(paths):
            parser.error(f"--{option.replace('_', '-')} needs one entry per --{captures} file")
    return args


def _json_default(value: Any) -> Any:
    """Convert NumPy/scikit-learn values into JSON-compatible scalars."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, type):
        return value.__name__
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def evaluate_external_capture(
    path: Path,
    labels_path: Path | None,
    model_bundle: dict[str, Any],
    output_dir: Path,
    encoding: str,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Score one mixed capture and, when labeled, evaluate it.

    Mixed captures never touch training, search, or threshold selection, so
    they serve as independent external test sets.
    """
    name = path.stem
    packets = assign_sessions(
        load_mixed_capture(path, labels_path, encoding),
        model_bundle["session_gap_seconds"],
    )
    sessions = add_rule_alerts(packets, engineer_session_features(packets))
    predictions = predict_anomalies(model_bundle, sessions)
    predictions.to_csv(output_dir / f"{name}_predictions.csv", index=False)
    metrics: dict[str, Any] = {
        "path": str(path),
        "packet_rows": len(packets),
        "total_sessions": len(sessions),
        "alert_sessions": int(predictions["alert"].sum()),
        "alert_sessions_by_source": (
            predictions.loc[predictions["alert"] == 1, "alert_sources"]
            .value_counts().to_dict()
        ),
    }
    if "label" not in sessions.columns:
        return metrics, None

    threshold = model_bundle["decision_threshold"]
    session_evaluation, curve = evaluate_model(
        model_bundle["model"], sessions, threshold
    )
    curve.to_csv(output_dir / f"{name}_precision_recall_curve.csv", index=False)
    packet_predictions = packet_level_predictions(packets, predictions)
    packet_predictions.to_csv(
        output_dir / f"{name}_packet_predictions.csv", index=False
    )
    scenario_rates = scenario_flag_rates(packet_predictions)
    scenario_rates.to_csv(
        output_dir / f"{name}_scenario_flag_rates.csv", index=False
    )
    metrics.update(
        labels_path=str(labels_path),
        attack_packets=int(packets["label"].sum()),
        attack_sessions=int(sessions["label"].sum()),
        session_evaluation=session_evaluation,
        packet_evaluation=evaluate_packets(packet_predictions, threshold),
    )
    if "event_id" in sessions.columns:
        metrics["layered_events"] = layered_event_metrics(predictions)
    return metrics, scenario_rates


def main() -> None:
    args = parse_args()
    training_labels = dict(
        zip(args.clean + args.attack, args.clean_labels + args.attack_labels)
    )
    packets = load_and_label_data(
        args.clean, args.attack, args.encoding, training_labels
    )
    if args.keep_shared_events:
        shared_report = {"kept": True}
    else:
        packets, shared_report = drop_shared_events(packets, args.mixed_labels)
    packets = assign_sessions(packets, args.session_gap)
    sessions = add_rule_alerts(packets, engineer_session_features(packets))
    (
        model, threshold, anomaly_layer, metrics, search_results, pr_curves, test
    ) = train_and_evaluate(
        sessions,
        test_size=args.test_size,
        random_state=args.random_state,
        block_sessions=args.block_sessions,
        search_iterations=args.search_iterations,
        cv_splits=args.cv_splits,
        max_false_positive_rate=args.max_false_positive_rate,
        threshold_rule=args.threshold_rule,
        anomaly_false_positive_rate=args.anomaly_false_positive_rate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(args.output_dir / "session_features.csv", index=False)
    search_results.to_csv(args.output_dir / "hyperparameter_search.csv", index=False)
    pr_curves.to_csv(args.output_dir / "precision_recall_curve.csv", index=False)
    if args.save_labeled_packets:
        packets.to_csv(args.output_dir / "labeled_packets.csv", index=False)

    feature_importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(
        args.output_dir / "feature_importances.csv", index=False
    )

    model_bundle = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "decision_threshold": threshold,
        "anomaly_layer": anomaly_layer,
        "session_gap_seconds": args.session_gap,
        "history_sessions": HISTORY_SESSIONS,
        "encoding": args.encoding,
    }
    joblib.dump(model_bundle, args.output_dir / "random_forest.joblib")

    metrics["dataset"] = {
        "clean_captures": [str(path) for path in args.clean],
        "attack_captures": [str(path) for path in args.attack],
        "packet_rows": len(packets),
        "labeled_attack_packets": int(packets["label"].sum()),
        "total_sessions": len(sessions),
        "total_attack_sessions": int(sessions["label"].sum()),
        "sessions_per_capture": sessions["capture_id"].value_counts().to_dict(),
        "session_gap_seconds": float(args.session_gap),
        "history_sessions": HISTORY_SESSIONS,
        "training_labels": {str(k): str(v) for k, v in training_labels.items()},
        "shared_event_removal": shared_report,
    }

    scenario_tables = {}
    metrics["external_tests"] = {}
    for path, labels_path in zip(args.mixed, args.mixed_labels):
        external_metrics, scenario_rates = evaluate_external_capture(
            path, labels_path, model_bundle, args.output_dir, args.encoding
        )
        metrics["external_tests"][path.stem] = external_metrics
        if scenario_rates is not None:
            scenario_tables[path.stem] = scenario_rates

    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=_json_default)

    predictions = predict_anomalies(model_bundle, test.drop(columns="block_id"))
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)

    print(json.dumps(metrics, indent=2, default=_json_default))
    print("\nFeature importances:")
    print(feature_importance.to_string(index=False))
    for name, scenario_rates in scenario_tables.items():
        print(f"\n{name}: flag rate by scenario (share of packets):")
        print(scenario_rates.to_string(index=False))
    for name, external in metrics["external_tests"].items():
        for layer, events in external.get("layered_events", {}).items():
            print(
                f"{name} [{layer}]: caught {events['detected_events']}/"
                f"{events['attack_events']} incidents, "
                f"{events['false_alarm_sessions']} false-alarm sessions "
                f"(FPR {events['false_positive_rate']:.4f})"
            )
    print(f"\nArtifacts written to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
