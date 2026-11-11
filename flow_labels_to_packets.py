"""Convert flow-level ground truth into per-packet labels for a Wireshark CSV.

Some label files describe flows (5-tuple, start/end time, label, scenario,
event_id) rather than packets. The Wireshark CSV omits ports on
application-layer rows, so ports and absolute times are read from the matching
pcap with tshark. Each packet is assigned to the flow with the same
direction-free 5-tuple whose time range contains it. ARP/ICMP flows carry a
placeholder port, and spoofed ARP replies name the attacker's real address
rather than the claimed one, so those fall back to protocol, time, and at least
one shared endpoint.

The output matches the per-packet label format used by anomaly_pipeline.py:
packet_no, frame_bytes, label, scenario, event_id, flow_id.

Example:
    python flow_labels_to_packets.py --pcap data/trafik_ornek_saldiri.pcap \\
        --flows data/ornek_saldiri_etiketli.csv --csv data/ornek_saldiri.csv \\
        --out data/ornek_saldiri_paket_etiketleri.csv
"""

from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TSHARK = r"C:\Program Files\Wireshark\tshark.exe"
TSHARK_FIELDS = [
    "frame.number", "frame.time_epoch", "ip.src", "ip.dst",
    "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport", "ip.proto",
    "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
]
# Packet times may sit a rounding step outside the millisecond flow bounds.
TIME_TOLERANCE_SECONDS = 0.002


def read_packet_tuples(pcap: Path, tshark: str) -> pd.DataFrame:
    """Return frame number, epoch time, endpoints, ports, and protocol per packet."""
    command = [tshark, "-r", str(pcap), "-T", "fields", "-E", "separator=,",
               "-E", "occurrence=f"]
    for field in TSHARK_FIELDS:
        command += ["-e", field]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    names = ["no", "epoch", "ip_src", "ip_dst", "tcp_sport", "tcp_dport",
             "udp_sport", "udp_dport", "ip_proto", "arp_src", "arp_dst"]
    packets = pd.read_csv(io.StringIO(output), header=None, names=names)

    is_arp = packets["arp_src"].notna()
    packets["src"] = packets["ip_src"].where(~is_arp, packets["arp_src"]).astype(str)
    packets["dst"] = packets["ip_dst"].where(~is_arp, packets["arp_dst"]).astype(str)
    packets["sport"] = packets["tcp_sport"].fillna(packets["udp_sport"]).fillna(0).astype(int)
    packets["dport"] = packets["tcp_dport"].fillna(packets["udp_dport"]).fillna(0).astype(int)
    packets["protocol"] = np.select(
        [is_arp, packets["ip_proto"].eq(6), packets["ip_proto"].eq(17), packets["ip_proto"].eq(1)],
        ["ARP", "TCP", "UDP", "ICMP"],
        default="OTHER",
    )
    return packets


def _flow_key(src: pd.Series, sport: pd.Series, dst: pd.Series, dport: pd.Series,
              protocol: pd.Series) -> np.ndarray:
    left = src + ":" + sport.astype(str)
    right = dst + ":" + dport.astype(str)
    return np.where(left <= right, left + "|" + right, right + "|" + left) + "|" + protocol


def match_packets_to_flows(packets: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    """Assign each packet the flow whose 5-tuple and time range contain it."""
    flows = flows.copy()
    flows["start"] = pd.to_datetime(flows["timestamp_start"], format="ISO8601").astype("int64") / 1e9
    flows["end"] = pd.to_datetime(flows["timestamp_end"], format="ISO8601").astype("int64") / 1e9
    flows["key"] = _flow_key(flows["src_ip"].astype(str), flows["src_port"],
                             flows["dst_ip"].astype(str), flows["dst_port"], flows["protocol"])
    packets = packets.assign(
        key=_flow_key(packets["src"], packets["sport"], packets["dst"], packets["dport"],
                      packets["protocol"])
    )
    label_columns = ["flow_id", "event_id", "label", "scenario"]

    exact = packets[["no", "epoch", "key"]].merge(
        flows[["key", "start", "end", *label_columns]], on="key"
    )
    exact = exact[(exact["epoch"] >= exact["start"] - TIME_TOLERANCE_SECONDS)
                  & (exact["epoch"] <= exact["end"] + TIME_TOLERANCE_SECONDS)]

    remaining = packets[~packets["no"].isin(exact["no"])
                        & packets["protocol"].isin(["ARP", "ICMP"])]
    fallback = remaining[["no", "epoch", "src", "dst", "protocol"]].merge(
        flows[["protocol", "start", "end", "src_ip", "dst_ip", *label_columns]], on="protocol"
    )
    shares_endpoint = (
        fallback["src"].eq(fallback["src_ip"]) | fallback["src"].eq(fallback["dst_ip"])
        | fallback["dst"].eq(fallback["src_ip"]) | fallback["dst"].eq(fallback["dst_ip"])
    )
    fallback = fallback[(fallback["epoch"] >= fallback["start"] - TIME_TOLERANCE_SECONDS)
                        & (fallback["epoch"] <= fallback["end"] + TIME_TOLERANCE_SECONDS)
                        & shares_endpoint]

    candidates = pd.concat([exact, fallback], ignore_index=True)
    ambiguous = candidates.groupby("no")["flow_id"].nunique()
    if (ambiguous > 1).any():
        raise ValueError(f"{int((ambiguous > 1).sum())} packets match more than one flow")
    matched = candidates.drop_duplicates("no")[["no", *label_columns]]
    return packets[["no"]].merge(matched, on="no", how="left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pcap", type=Path, required=True)
    parser.add_argument("--flows", type=Path, required=True, help="Flow-level label CSV.")
    parser.add_argument("--csv", type=Path, required=True,
                        help="Wireshark CSV export of the same pcap.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tshark", default=DEFAULT_TSHARK)
    args = parser.parse_args()

    packets = read_packet_tuples(args.pcap, args.tshark)
    labels = match_packets_to_flows(packets, pd.read_csv(args.flows, low_memory=False))

    wireshark = pd.read_csv(args.csv, usecols=["No.", "Length"])
    if len(wireshark) != len(labels) or not (wireshark["No."].to_numpy() == labels["no"].to_numpy()).all():
        raise ValueError(f"{args.csv} does not list the same packets as {args.pcap}")
    unmatched = int(labels["label"].isna().sum())
    if unmatched:
        raise ValueError(f"{unmatched} packets in {args.pcap} match no flow in {args.flows}")

    labels["frame_bytes"] = wireshark["Length"].to_numpy()
    labels["label"] = labels["label"].astype(int)
    labels = labels.rename(columns={"no": "packet_no"})
    columns = ["packet_no", "frame_bytes", "label", "scenario", "event_id", "flow_id"]
    labels[columns].to_csv(args.out, index=False)
    print(f"{args.out}: {len(labels)} packets, {labels['flow_id'].nunique()} flows, "
          f"labels {labels['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
