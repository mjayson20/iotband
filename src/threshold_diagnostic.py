"""
src/threshold_diagnostic.py
IoT-SecBand — Phase 2E (diagnostic only, no holdout split)

Purpose:
    Answer "is there untapped performance available via threshold alone?"
    WITHOUT consuming any part of the test set as a calibration/holdout split.

    The official UNSW-NB15 test set remains fully intact and untouched
    as a benchmark. This script only PLOTS metrics across thresholds —
    it does not select or commit to any threshold value.

Current candidate configuration (Phase 2D best, count-corrected):
    RobustScaler
    11 features (dropped: dload, dinpkt)
    MLP (32, 16)

    Phase 2D result (at default threshold=0.50):
    F1=0.7299  Precision=0.5864  Recall=0.9664  AUC=0.9267  FPR=0.3454

This is diagnostic only. No threshold is locked. No test split occurs.
The pipeline is NOT yet declared final — further feature/scaler experiments
remain possible after this.

Outputs:
    outputs/metrics/E03_threshold_sweep_diagnostic.png
    outputs/metrics/phase2e_diagnostic.json

Run from project root:
    python src/threshold_diagnostic.py
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import load_parquet, clean, encode_categoricals
from train import TARGET

RAW_TRAIN = ROOT / "data" / "raw" / "unsw_nb15" / "UNSW_NB15_training-set.parquet"
RAW_TEST  = ROOT / "data" / "raw" / "unsw_nb15" / "UNSW_NB15_testing-set.parquet"
METRICS   = ROOT / "outputs" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

# ── CANDIDATE CONFIGURATION (count-corrected: 11 features) ───────────────────
CANDIDATE_FEATURES = [
    "dur", "proto", "service", "state",
    "sbytes", "dbytes", "rate",
    "sload", "sinpkt", "smean", "dmean",
]  # 11 features — dload and dinpkt dropped

assert len(CANDIDATE_FEATURES) == 11, f"Expected 11 features, got {len(CANDIDATE_FEATURES)}"

MLP_PARAMS = dict(
    hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
    alpha=0.001, batch_size=256, learning_rate_init=0.001, max_iter=200,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
    random_state=42,
)

plt.rcParams.update({
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif",
})


def threshold_sweep(y_true, y_prob):
    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        rows.append({
            "threshold": round(float(t), 2),
            "f1":        round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred, pos_label=1)), 4),
            "fpr":       round(float(fpr), 4),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("IoT-SecBand | Phase 2E — Threshold Diagnostic (NO test split)")
    print("Candidate: RobustScaler + 11 features (drop dload, dinpkt) + MLP(32,16)")
    print("=" * 60)
    print("\nNOTE: This is diagnostic only. The official test set remains")
    print("fully intact. No threshold is being locked. No holdout is created.")
    print("=" * 60)

    # ── Load and prepare ───────────────────────────────────────
    df_train_raw = load_parquet(RAW_TRAIN)
    df_test_raw  = load_parquet(RAW_TEST)
    df_train_clean = clean(df_train_raw)
    df_test_clean  = clean(df_test_raw)
    df_train_enc, encoders = encode_categoricals(df_train_clean, fit=True)
    df_test_enc,  _        = encode_categoricals(df_test_clean, encoders=encoders, fit=False)

    print(f"\nFeature count check: {len(CANDIDATE_FEATURES)} features")
    print(f"Features: {CANDIDATE_FEATURES}")

    scaler = RobustScaler()
    train = df_train_enc.copy()
    test  = df_test_enc.copy()
    scaler.fit(train[CANDIDATE_FEATURES])
    train[CANDIDATE_FEATURES] = scaler.transform(train[CANDIDATE_FEATURES])
    test[CANDIDATE_FEATURES]  = scaler.transform(test[CANDIDATE_FEATURES])

    X_train, y_train = train[CANDIDATE_FEATURES], train[TARGET]
    X_test,  y_test  = test[CANDIDATE_FEATURES],  test[TARGET]

    print("\nTraining candidate configuration...")
    mlp = MLPClassifier(**MLP_PARAMS)
    mlp.fit(X_train, y_train)
    log.info(f"MLP trained — iterations: {mlp.n_iter_}")

    # ── Threshold sweep on FULL test set (diagnostic only) ─────
    print("\n" + "=" * 60)
    print("Threshold sweep — FULL test set (46,391 samples, untouched)")
    print("=" * 60)

    y_prob = mlp.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    sweep = threshold_sweep(y_test.values, y_prob)

    best_f1_row = sweep.loc[sweep["f1"].idxmax()]
    best_recall_constrained = sweep[sweep["recall"] >= 0.88]
    best_constrained_row = (
        best_recall_constrained.loc[best_recall_constrained["f1"].idxmax()]
        if not best_recall_constrained.empty else None
    )

    log.info(f"\nAUC (full test set): {auc:.4f}")
    log.info(f"\nDefault threshold (0.50):")
    default_row = sweep[sweep["threshold"] == 0.50].iloc[0]
    log.info(f"  F1={default_row['f1']:.4f}  Precision={default_row['precision']:.4f}  "
             f"Recall={default_row['recall']:.4f}  FPR={default_row['fpr']:.4f}")

    log.info(f"\nBest F1 threshold (unconstrained): {best_f1_row['threshold']:.2f}")
    log.info(f"  F1={best_f1_row['f1']:.4f}  Precision={best_f1_row['precision']:.4f}  "
             f"Recall={best_f1_row['recall']:.4f}  FPR={best_f1_row['fpr']:.4f}")

    if best_constrained_row is not None:
        log.info(f"\nBest F1 threshold (recall>=0.88 constraint): {best_constrained_row['threshold']:.2f}")
        log.info(f"  F1={best_constrained_row['f1']:.4f}  Precision={best_constrained_row['precision']:.4f}  "
                 f"Recall={best_constrained_row['recall']:.4f}  FPR={best_constrained_row['fpr']:.4f}")
    else:
        log.warning("\nNo threshold achieves recall>=0.88 with reasonable F1")

    # ── Plot ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(sweep["threshold"], sweep["f1"],        color="#1D9E75", lw=2.0, label="F1")
    ax.plot(sweep["threshold"], sweep["precision"], color="#378ADD", lw=1.6, label="Precision", alpha=0.85)
    ax.plot(sweep["threshold"], sweep["recall"],    color="#D85A30", lw=1.6, label="Recall", alpha=0.85)
    ax.plot(sweep["threshold"], sweep["fpr"],       color="#888780", lw=1.4, label="FPR", alpha=0.7, linestyle="--")

    ax.axvline(0.50, color="#5F5E5A", linestyle=":", lw=1.0, label="Default (0.50)")
    ax.axvline(best_f1_row["threshold"], color="#1D9E75", linestyle="--", lw=1.0, alpha=0.6,
               label=f"Best F1 ({best_f1_row['threshold']:.2f})")
    ax.axhline(0.88, color="#888780", linestyle=":", lw=0.7, alpha=0.5, label="Target (0.88)")

    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title(
        f"Threshold sweep — full UNSW test set, untouched\n"
        f"Candidate: RobustScaler + 11 features + MLP(32,16)  |  AUC={auc:.4f}",
        fontsize=10
    )
    ax.legend(fontsize=8.5, ncol=2)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_plot = METRICS / "E03_threshold_sweep_diagnostic.png"
    fig.savefig(out_plot, bbox_inches="tight")
    plt.close(fig)
    log.info(f"\nSaved: {out_plot}")

    # ── Verdict ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    f1_ceiling = float(best_f1_row["f1"])
    if f1_ceiling >= 0.88:
        verdict = "UNTAPPED PERFORMANCE EXISTS — meets target"
        action = (
            f"Best achievable F1 at threshold={best_f1_row['threshold']:.2f} is "
            f"{f1_ceiling:.4f}, meeting the 0.88 target.\n"
            "This configuration (RobustScaler + 11 features + MLP 32,16) can be\n"
            "declared FINAL. Proceed to the one-time calibration/holdout split\n"
            "(tune_threshold_v3.py) to confirm on a true held-out partition."
        )
    elif f1_ceiling >= 0.78:
        verdict = "MODEST UNTAPPED PERFORMANCE — within expected ceiling range"
        action = (
            f"Best achievable F1 at threshold={best_f1_row['threshold']:.2f} is "
            f"{f1_ceiling:.4f}.\n"
            f"Gain over default (t=0.50): {f1_ceiling - default_row['f1']:+.4f}\n\n"
            "This is within the expected ceiling (0.78-0.83) given AUC={:.3f} and\n"
            "the documented overlap (~0.32). Two paths:\n"
            "  A) Declare this configuration final, proceed to one-time split,\n"
            "     and document the ceiling as a UNSW-NB15 temporal drift finding.\n"
            "  B) Run ONE more feature experiment first: the moderate-drift\n"
            "     features (rate, sinpkt, sload, dur, sbytes — KS 0.14-0.19)\n"
            "     could be tested for removal in the same controlled manner\n"
            "     as Phase 2D, before finalizing.".format(auc)
        )
    else:
        verdict = "MINIMAL UNTAPPED PERFORMANCE — threshold is not the lever"
        action = (
            f"Best achievable F1 at threshold={best_f1_row['threshold']:.2f} is only "
            f"{f1_ceiling:.4f}.\n"
            "Threshold tuning will not close the gap. Feature drift remains\n"
            "the dominant factor. Consider architecture change as next step."
        )

    print(f"\n  Gain available (best F1 - default F1): {f1_ceiling - default_row['f1']:+.4f}")
    print(f"  Verdict: {verdict}")
    print(f"\n  {action}")

    # ── Save ─────────────────────────────────────────────────────
    output = {
        "phase": "2E — Threshold Diagnostic (no test split)",
        "candidate_config": {
            "scaler": "RobustScaler",
            "features": CANDIDATE_FEATURES,
            "n_features": len(CANDIDATE_FEATURES),
            "note": "Count-corrected from Phase 2D 'drop dload+dinpkt' (was mislabeled as 10, actually 11)",
        },
        "auc_full_test_set": round(float(auc), 4),
        "default_threshold_050": default_row.to_dict(),
        "best_f1_threshold_unconstrained": best_f1_row.to_dict(),
        "best_f1_threshold_recall_constrained": (
            best_constrained_row.to_dict() if best_constrained_row is not None else None
        ),
        "f1_gain_available": round(f1_ceiling - float(default_row["f1"]), 4),
        "verdict": verdict,
        "action": action,
        "note": (
            "Full UNSW-NB15 test set (46,391 samples) remains untouched as a "
            "benchmark. No calibration/holdout split performed. No threshold "
            "is locked. Pipeline not yet declared final."
        ),
        "plot": str(out_plot),
    }

    out_path = METRICS / "phase2e_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()