"""Fixed-model evaluation of the current anomaly_pipeline on both mixed tests.

Shared incidents are removed from training (as the pipeline default), a fixed
Random Forest is fitted, and the incident threshold is chosen on grouped OOF
predictions. Optional layers are used when the pipeline provides them:
`rule_alerts(packets, sessions)` and an `AnomalyLayer`.

Run from the project root: PYTHONPATH=. python scripts/eval_harness.py <tag>
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

import anomaly_pipeline as ap

pd.set_option("display.width", 250)
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
DATA = Path("data")
CLEAN = [DATA / "ornek_temiz.csv", DATA / "yeni_temiz.csv"]
ATTACK = [DATA / "ornek_saldiri.csv", DATA / "yeni_saldiri.csv"]
TESTS = [DATA / "ornek_karisik.csv", DATA / "yeni_karisik.csv"]
FAMILY = {
    "http_login_attack_like": "http_login", "http_login_burst": "http_login",
    "rdp_login_attack_like": "rdp_login", "rdp_login_attempt_pattern": "rdp_login",
    "ssh_password_spray_like": "ssh_login", "ssh_login_attempt_pattern": "ssh_login",
}


def model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  max_features="sqrt", n_jobs=-1, random_state=42)


def main() -> None:
    t0 = time.time()
    labels = {path: ap.DEFAULT_LABELS[path] for path in CLEAN + ATTACK}
    packets = ap.load_and_label_data(CLEAN, ATTACK, labels_paths=labels)
    packets, _ = ap.drop_shared_events(packets, [ap.DEFAULT_LABELS[p] for p in TESTS])
    packets = ap.assign_sessions(packets, 15.0)
    train = ap.engineer_session_features(packets)
    blocks = ap.add_temporal_block_ids(train, 200).set_index("session_id").loc[train["session_id"], "block_id"]
    oof = cross_val_predict(model(), train[ap.FEATURE_COLUMNS], train["label"], groups=blocks.to_numpy(),
                            cv=StratifiedGroupKFold(3, shuffle=True, random_state=42), method="predict_proba")[:, 1]
    threshold, details, _ = ap.tune_probability_threshold(train["label"], oof, 0.01, train["event_id"])
    clf = model().fit(train[ap.FEATURE_COLUMNS], train["label"])
    layer = None
    if hasattr(ap, "fit_anomaly_layer"):
        layer = ap.fit_anomaly_layer(train, blocks.to_numpy())
    print(f"[{TAG}] train sessions={len(train)} attack={int(train.label.sum())} incidents={train.event_id.nunique()} "
          f"threshold={threshold:.3f} oof_event_recall={details.get('oof_event_recall')} ({time.time() - t0:.0f}s)")

    summary, per_family = [], {}
    for path in TESTS:
        name = path.stem
        test_packets = ap.assign_sessions(ap.load_mixed_capture(path, ap.DEFAULT_LABELS[path]), 15.0)
        test = ap.engineer_session_features(test_packets)
        score = clf.predict_proba(test[ap.FEATURE_COLUMNS])[:, 1]
        sources = {"model": score >= threshold}
        if layer is not None:
            sources["anomaly"] = ap.score_anomaly_layer(layer, test)
        if hasattr(ap, "rule_alerts"):
            sources["rules"] = ap.rule_alerts(test_packets, test)["rule_alert"].to_numpy()
        combined = np.zeros(len(test), dtype=bool)
        for verdict in sources.values():
            combined |= verdict
        for source, verdict in [*sources.items(), ("combined", combined)]:
            ev = ap.evaluate_events(test, verdict.astype(float), 0.5)
            summary.append({"test": name, "source": source, "sessions": len(test),
                            "incidents": f"{ev['detected_events']}/{ev['attack_events']}",
                            "event_recall": ev["event_recall"], "false_alarms": ev["false_alarm_sessions"],
                            "FPR": ev["false_positive_rate"]})
        attacks = test[test["label"] == 1].assign(flag=combined[test["label"].to_numpy() == 1])
        attacks["family"] = attacks["scenario"].map(lambda s: FAMILY.get(s, s))
        per_event = attacks.groupby(["family", "event_id"])["flag"].max().groupby("family")
        per_family[name] = (per_event.sum().astype(int).astype(str) + "/" + per_event.size().astype(str))
    print(pd.DataFrame(summary).round(4).to_string(index=False))
    print("\nincidents caught per family (combined):")
    print(pd.DataFrame(per_family).fillna("-").to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
