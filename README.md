# Session-level network anomaly detection

This project groups the supplied packet logs into sessions, describes each
session by its own behavior and by its initiator's recent history, trains a
tuned Random Forest anomaly detector on clean and attack captures from two
networks without allowing held-out test blocks to influence model or threshold
selection, and then evaluates it on independent, per-packet labeled mixed
captures, counting which attack incidents were caught and at what alert cost.
Alerts come from three layers: the classifier, a benign-only anomaly layer,
and deterministic rules; a session alerts when any of them fires.

## Data and layout

```text
anomaly_pipeline.py        Loading, sessions, features, layers, training, evaluation (CLI)
flow_labels_to_packets.py  Converts flow-level labels to packet-level labels (needs tshark)
scripts/eval_harness.py    Fast fixed-model comparison of the alert layers on both test sets
data/                      Wireshark CSV exports, label files, and the source pcaps
requirements.txt           Python dependencies
```

Two synthetic networks are used. Each has a benign capture, an attack-only
capture, and a mixed capture with per-packet ground truth:

| Network | Benign (train) | Attack (train) | Mixed (test) | Notes |
|---|---|---|---|---|
| `ornek` | `ornek_temiz.csv` | `ornek_saldiri.csv` | `ornek_karisik.csv` | Sparse; CSVs are ~9% samples of the full flow-labeled data |
| `yeni` | `yeni_temiz.csv` | `yeni_saldiri.csv` | `yeni_karisik.csv` | Dense; constant 10 ms packet spacing over 900 s |

CSVs are Wireshark exports (`No.`, `Time`, `Source`, `Destination`,
`Protocol`, `Length`, `Info`); there are no port columns. Label files
(`*_etiketli.csv`, `*_etiketleri.csv`, `*_paket_etiketleri.csv`) hold the
per-packet `label`, `scenario`, and `event_id` (one attack incident). The
`.pcap` files are the sources of the exports and are used only by
`flow_labels_to_packets.py`.

## Results

Test captures are scored per incident: an incident counts as caught when any
of its sessions alerts. Full pipeline run, 20% of blocks held out internally:

| Test capture | Classifier | Anomaly layer | Rules | Combined | False-alarm sessions |
|---|---|---|---|---|---|
| `ornek_karisik` | 94/157 | 48/157 | 10/157 | **122/157** | 12 (FPR 0.74%) |
| `yeni_karisik` | 15/16 | 9/16 | 1/16 | **16/16** | 7 (FPR 0.07%) |

On `ornek_karisik`, SSH brute force (0/10), HTTPS exfiltration (1/10) and RDP
(4/10) are still mostly missed. Results move by about ±5 incidents between
runs and settings. The test captures informed some design choices (ARP rule,
DNS features), so treat them as semi-validation sets, not a final estimate;
a fresh independent mixed capture is needed for that.

## Setup and training

Requires Python 3.10+. `flow_labels_to_packets.py` additionally needs
Wireshark's `tshark` (default `C:\Program Files\Wireshark\tshark.exe`, change
with `--tshark`).

```sh
python -m pip install -r requirements.txt
python anomaly_pipeline.py
```

A full run takes a few minutes, mostly the hyperparameter search.

Defaults train on `data/ornek_temiz.csv` and `data/yeni_temiz.csv` (benign)
plus `data/ornek_saldiri.csv` and `data/yeni_saldiri.csv` (attack), and test
on `data/ornek_karisik.csv` and `data/yeni_karisik.csv` with labels from
`data/ornek_karisik_etiketli.csv` and `data/yeni_karisik_etiketleri.csv`.
Training captures use their per-packet label files (see `DEFAULT_LABELS`) for
scenarios and incidents. Attack incidents that also appear in a test capture
are removed from training. Sessions end on a new TCP connection or after 15
idle seconds, 24 randomized hyperparameter trials run, the threshold catches
the most out-of-fold training incidents while flagging at most 1% of benign
sessions, and outputs go to `artifacts/`.

```sh
python anomaly_pipeline.py \
  --clean data/ornek_temiz.csv data/yeni_temiz.csv \
  --attack data/ornek_saldiri.csv data/yeni_saldiri.csv \
  --mixed data/yeni_karisik.csv \
  --mixed-labels data/yeni_karisik_etiketleri.csv \
  --max-false-positive-rate 0.01
```

Important options:

```text
--clean PATH [PATH ...]         Benign-only captures (labeled 0)
--clean-labels [PATH ...]       Per-packet labels per --clean file, same order; '-' for none
--attack PATH [PATH ...]        Attack-only captures (labeled 1)
--attack-labels [PATH ...]      Per-packet labels per --attack file, same order; '-' for none
--mixed [PATH ...]              Mixed captures used as external test sets
--mixed-labels [PATH ...]       Ground truth per --mixed file, same order; '-' to only score
--threshold-rule event          'event' (incidents caught) or 'session' (sessions caught)
--anomaly-false-positive-rate 0.005  Benign-only layer's FPR budget per network; 0 disables it
--keep-shared-events            Keep training incidents that also appear in a test capture
--session-gap 15                Idle seconds that end a session
--block-sessions 200            Consecutive sessions per leakage-safe split block
--test-size 0.25                Fraction of blocks held out once for test
--search-iterations 24          Random Forest parameter combinations to evaluate
--cv-splits 3                   Grouped CV folds within the training partition
--max-false-positive-rate 0.01  Share of benign OOF sessions allowed to alert
--encoding utf-8                CSV encoding of the Wireshark exports
--save-labeled-packets          Also save the merged packet-level dataframe
--output-dir artifacts          Output directory
```

## Pipeline

1. **Label packets:** Every row of a `--clean` capture receives `0` and every
   row of an `--attack` capture receives `1`; attack captures contain only
   attack scenarios (ICMP/port scans, HTTP brute force and directory
   enumeration, SSH/RDP brute force, DNS tunneling and floods, ARP gateway
   spoofing, lateral SMB, HTTPS exfiltration). A per-packet label file adds
   `scenario` and `event_id` (the incident) and must agree with the capture's
   role. Label files are joined on Wireshark's `No.` column and must match
   every frame length. Each file keeps its own capture ID, so sessions and
   host history never span two files. Both networks contribute benign and
   attack captures, so the model cannot use "which network is this" as a
   shortcut for the label.
2. **Remove shared incidents:** The `ornek_*` generator writes the same
   incidents into the attack and mixed captures (157 of 160 attack
   `event_id`s). Training attack packets whose `event_id` appears in a labeled
   test capture are removed before sessionizing, so tests measure attacks the
   model has not seen. `--keep-shared-events` disables this for a final model
   trained on all data, whose test scores are then optimistic.
3. **Build sessions:** A session is one connection or transaction: packets
   between the same unordered endpoint pair, ending when a new exchange
   opens (a TCP `[SYN]`, a DNS query, an ICMP echo request, or an ARP
   request) or the pair is idle for more than `--session-gap` seconds. On
   busy networks the idle gap may never occur; without these splits
   thousands of attempts, queries, or probes merge into one session and the
   model sees a handful of examples instead of thousands. Ports are not part
   of the key because Wireshark omits them on application-layer rows.
4. **Session behavior:** Packet count, duration, rate, bytes, initiator share
   of packets and bytes, packet-length mean/std/CV/max, IAT mean/std/max,
   transport and ARP/DNS ratios (application protocols such as SSL/TLS, HTTP,
   SSH, DNS, NTP, OpenVPN count toward their TCP/UDP transport), protocol
   diversity, SYN and RST rates, SYN-ACK answer ratio, HTTP 4xx/5xx ratio,
   maximum DNS query-name length, the longest, highest-entropy, and most
   digit-heavy first DNS label (generated names in tunnels and
   random-subdomain floods; real host names are short words), distinct
   destination ports, and service port.
5. **Initiator history:** Over the initiator's last 20 sessions, including the
   current one: how many there are, the share that failed (reset, unanswered
   SYN, HTTP 4xx/5xx, or NXDOMAIN), the share to the same responder, the share
   of short sessions (at most 3 packets), and the number of distinct
   responders and service ports. History is counted in sessions, not seconds,
   so it means the same on a quiet network and a busy one; time-window counts
   learned on a sparse network flagged nearly all traffic on a dense one. Only
   past sessions are used, so the features are available in live scoring. IP
   addresses are used only as grouping keys, never as feature values.
6. **Create an untouched test set:** Blocks of 200 consecutive sessions per
   capture are assigned to training or test. Blocks are counted in sessions
   because one hour holds a few sessions on one network and tens of thousands
   on another. The test blocks are not used by search or threshold tuning.
7. **Optimize the model:** `RandomizedSearchCV` compares tree count/depth,
   split and leaf sizes, feature sampling, and class weighting. Folds use
   `StratifiedGroupKFold`, so one block cannot span train/validation. Average
   precision (PR-AUC) is the optimization score.
8. **Tune the threshold:** The best model generates out-of-fold probabilities
   for training sessions. A missed attack costs more than a false alarm, so
   the threshold catching the most attack incidents while flagging at most
   `--max-false-positive-rate` of benign sessions is selected; ties go to the
   higher threshold. An incident is caught when any of its sessions alerts,
   which is what an analyst needs and usually permits a higher threshold, and
   so fewer false alarms, than catching every session. Without incident
   labels, or with `--threshold-rule session`, sessions are counted instead.
   The cap is on the false-positive rate rather than precision because
   precision depends on how common attacks are: the training data is mostly
   attack sessions, so a precision floor that holds in training allows far
   more alerts on a real network where attacks are rare.
9. **Add the other alert layers:** An Isolation Forest is fitted on the
   benign training sessions only and flags sessions unlike any normal
   traffic, whatever the attack. Each benign capture gets equal weight, and
   its threshold is the strictest per-capture quantile of held-out benign
   scores at `--anomaly-false-positive-rate`, because a pooled threshold fits
   the largest network and over-alerts on the others. A rule flags sessions
   with an unsolicited ARP reply (no request for that address in the
   preceding 2 s) or an ARP reply rebinding an address to a new MAC: ARP
   spoofing looks different on every network but always does one of these.
   `alert` combines the three layers and `alert_sources` names those that
   fired; `rule_reason` explains rule alerts.
10. **Evaluate once:** Metrics report both the default 0.5 threshold and tuned
   threshold on untouched test blocks, including precision, recall, F1, and
   exact TN/FP/FN/TP counts. ROC-AUC and average precision are threshold-free.
   With incident labels, `tuned_threshold_events` adds incidents caught and
   missed, sessions that passed before the first alert, and false-alarm
   sessions with their rate; `layered_events` gives the same for each layer
   and for the combined alert.
11. **External tests on mixed captures:** Each label file is joined to its
   capture on Wireshark's `No.` column; the join fails loudly if any packet is
   unmatched or its frame length differs. A session is an attack if any of
   its packets is. Metrics are reported per session and per packet (each
   packet inherits its session's verdict), the latter being comparable across
   detectors with different units. Mixed captures never influence training,
   search, or threshold selection. Per-scenario tables show the share of
   packets flagged, how many of each scenario's incidents were caught, and
   which benign services raise false alarms.

## Flow-level labels

`data/ornek_temiz_etiketli.csv` and `data/ornek_saldiri_etiketli.csv` label
flows, not packets. `flow_labels_to_packets.py` reads ports and absolute times
from the matching pcap with tshark and assigns each packet to the flow with
the same direction-free 5-tuple whose time range contains it (ARP/ICMP flows
carry a placeholder port, so they match on protocol, time, and a shared
endpoint). It refuses to write a file if any packet is unmatched or matches
two flows. The bundled `data/ornek_*_paket_etiketleri.csv` files were made
with:

```sh
python flow_labels_to_packets.py --pcap data/trafik_ornek_saldiri.pcap \
  --flows data/ornek_saldiri_etiketli.csv --csv data/ornek_saldiri.csv \
  --out data/ornek_saldiri_paket_etiketleri.csv
python flow_labels_to_packets.py --pcap data/trafik_ornek_temiz.pcap \
  --flows data/ornek_temiz_etiketli.csv --csv data/ornek_temiz.csv \
  --out data/ornek_temiz_paket_etiketleri.csv
```

## Outputs

- `artifacts/random_forest.joblib`: model, feature schema, tuned threshold,
  anomaly layer, session gap, history length, and CSV encoding
- `artifacts/metrics.json`: dataset, search, threshold-selection, internal
  test, and per-capture external test metrics
- `artifacts/session_features.csv`: engineered training session dataset
- `artifacts/feature_importances.csv`: ranked Random Forest importances
- `artifacts/hyperparameter_search.csv`: ranked search trials
- `artifacts/precision_recall_curve.csv`: OOF-training and test curve points
- `artifacts/test_predictions.csv`: probabilities and tuned predictions
- `artifacts/<capture>_predictions.csv`: session scores, `alert`, `alert_sources`,
  and `rule_reason` for each mixed capture
- `artifacts/<capture>_packet_predictions.csv`: per-packet verdicts and scenarios
- `artifacts/<capture>_scenario_flag_rates.csv`: flag rate and incidents caught per scenario
- `artifacts/<capture>_precision_recall_curve.csv`: session PR points
- `artifacts/labeled_packets.csv`: optional packet-level merged data

For inference, load packets with `load_capture`, then call
`assign_sessions(packets, bundle["session_gap_seconds"])`,
`engineer_session_features(...)`, `add_rule_alerts(packets, sessions)`, and
`predict_anomalies(bundle, sessions)`. The helper applies the exact saved
feature order, `decision_threshold`, and anomaly layer, and sets `alert`.

## Comparing alert layers quickly

`scripts/eval_harness.py` trains a fixed 300-tree Random Forest (no search),
removes shared incidents, tunes the threshold on out-of-fold predictions, and
prints incidents caught and false alarms per layer and per attack family for
both test captures. Use it to compare feature or layer changes in 1-2 minutes:

```sh
PYTHONPATH=. python scripts/eval_harness.py my_change
```

(PowerShell: `$env:PYTHONPATH="."; python scripts/eval_harness.py my_change`.)

## Production caveats

Every packet of an attack capture is an attack, so attack sessions that
look exactly like normal traffic are still labeled `1`. The `yeni_*` captures
share no incidents, but the generator reuses attacker addresses and port
sequences across scenarios, so the model may partly recognize the attacker
rather than the attack; `yeni_karisik` is optimistic about novel attackers.
After shared incidents are removed, almost no `ornek` attacks remain in
training, so `ornek_karisik` effectively measures detection on a network
whose attacks were learned elsewhere. The `yeni_*` timestamps are synthetic (a
constant 10 ms between packets), so duration, rate, and IAT features carry
little meaning there. The `yeni_*` captures dominate the training sessions,
which pulls the operating point toward that network. Host history needs
enough sessions: the first sessions of an initiator see little context. The
endpoint-pair session key merges concurrent connections between the same two
hosts. Legitimate gratuitous ARP (for example, failover) trips the ARP
rule; known pairs may need an allow-list. The anomaly layer's budget is met
per training network, but a new network has its own normal traffic, so its
false-positive rate there can be higher. The `ornek` CSVs are samples of
larger captures (about 9% of flows), so floods and brute force appear as a
few packets per incident and volume cannot reveal them; slow SSH and RDP
password attempts and HTTPS exfiltration remain the weakest families on that
network. Before deployment, add benign traffic from the target network,
calibrate the false-positive cap to an acceptable alert rate, and monitor
feature drift. Overall accuracy is not informative for this imbalance.
