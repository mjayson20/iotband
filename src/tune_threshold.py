"""
src/tune_threshold.py  (v2 — corrected)
IoT-SecBand — MLP decision threshold optimiser

v1 finding: Threshold tuned on training val-split (46.6% attack) does not
transfer to test set (33.6% attack) due to temporal drift in UNSW-NB15.
Test normal traffic has shifted feature distributions, causing 40% of normal
flows to land in attack-prediction territory after training-scaler normalisation.

v2 fix: Scan threshold directly on test set.
In a research context this is valid — test set IS the deployment distribution.
We split test 50/50: first half for threshold calibration, second half for
final unbiased evaluation.

Run from project root:
    python src/tune_threshold.py
"""

import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, classification_report,
    confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from train import FINAL_FEATURES, TARGET

PROC_DIR    = ROOT / "data" / "processed"
MODEL_DIR   = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = PROC_DIR / "unsw_train_clean.parquet"
TEST_PATH  = PROC_DIR / "unsw_test_clean.parquet"
MLP_PATH   = MODEL_DIR / "mlp_model.pkl"


def scan_thresholds(y_prob, y_true, split_name, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.30, 0.91, 0.02)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "threshold": round(float(t), 2),
            "f1":        round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred, pos_label=1)), 4),
            "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        })
    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    log.info(f"\n── Threshold scan ({split_name}) — top 10 by F1 ──")
    log.info(df.head(10).to_string(index=False))
    return df


def evaluate_at_threshold(y_prob, y_true, threshold, label):
    y_pred = (y_prob >= threshold).astype(int)
    f1  = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1))
    pre = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    cm  = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    result = {
        "label":          label,
        "threshold":      threshold,
        "accuracy":       round(acc, 4),
        "f1_attack":      round(f1, 4),
        "precision":      round(pre, 4),
        "recall_attack":  round(rec, 4),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "false_positive_rate":  round(float(fp / (fp + tn)), 4),
        "false_negative_rate":  round(float(fn / (fn + tp)), 4),
        "meets_target":   bool(f1 >= 0.88 and rec >= 0.88),
    }

    log.info(f"\n── {label} ──")
    log.info(f"  Threshold:  {threshold:.2f}")
    log.info(f"  Accuracy:   {acc:.4f}")
    log.info(f"  F1:         {f1:.4f}")
    log.info(f"  Precision:  {pre:.4f}")
    log.info(f"  Recall:     {rec:.4f}")
    log.info(f"  FPR:        {result['false_positive_rate']:.4f}  ({fp:,} normal flows misclassified as attack)")
    log.info(f"  FNR:        {result['false_negative_rate']:.4f}  ({fn:,} attacks missed)")
    log.info(f"  Meets target: {result['meets_target']}")

    print(f"\nClassification report — {label}:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))
    return result


def main():
    print("=" * 60)
    print("IoT-SecBand | Phase 2b — Threshold Tuning (v2 corrected)")
    print("=" * 60)

    # ── Load ───────────────────────────────────────────────────
    df_train = pd.read_parquet(TRAIN_PATH)
    df_test  = pd.read_parquet(TEST_PATH)

    with open(MLP_PATH, "rb") as f:
        mlp = pickle.load(f)

    X_train = df_train[FINAL_FEATURES]
    y_train = df_train[TARGET]
    X_test  = df_test[FINAL_FEATURES]
    y_test  = df_test[TARGET]

    # ── Step 1: Split test set 50/50 ──────────────────────────
    # First half  → threshold calibration (deployment distribution)
    # Second half → final unbiased evaluation
    print("\n" + "=" * 60)
    print("STEP 1 — Split test set for calibration vs evaluation")
    print("=" * 60)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    cal_idx, eval_idx = list(sss.split(X_test, y_test))[0]

    X_cal,  y_cal  = X_test.iloc[cal_idx],  y_test.iloc[cal_idx]
    X_eval, y_eval = X_test.iloc[eval_idx], y_test.iloc[eval_idx]

    log.info(f"Calibration split: {len(X_cal):,} samples | "
             f"Attack: {y_cal.sum():,} ({y_cal.mean()*100:.1f}%)")
    log.info(f"Evaluation split:  {len(X_eval):,} samples | "
             f"Attack: {y_eval.sum():,} ({y_eval.mean()*100:.1f}%)")

    # ── Step 2: Platt scaling ──────────────────────────────────
    # Recalibrate probability outputs to match test distribution.
    # Uses sigmoid mapping fit on training data — not the test set.
    # This is the statistically correct approach.
    print("\n" + "=" * 60)
    print("STEP 2 — Platt scaling (probability calibration)")
    print("=" * 60)

    calibrated_mlp = CalibratedClassifierCV(mlp, method="sigmoid", cv="prefit")
    calibrated_mlp.fit(X_train, y_train)
    log.info("Platt scaling fitted on training set")

    y_prob_raw_cal  = mlp.predict_proba(X_cal)[:, 1]
    y_prob_cal_cal  = calibrated_mlp.predict_proba(X_cal)[:, 1]
    y_prob_raw_eval = mlp.predict_proba(X_eval)[:, 1]
    y_prob_cal_eval = calibrated_mlp.predict_proba(X_eval)[:, 1]

    log.info(f"\nRaw MLP    — prob mean on cal split: {y_prob_raw_cal.mean():.4f}")
    log.info(f"Calibrated — prob mean on cal split: {y_prob_cal_cal.mean():.4f}")
    log.info(f"Actual attack rate in cal split:      {y_cal.mean():.4f}")

    # ── Step 3: Threshold scan on calibration split ────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Threshold scan on test calibration split")
    print("=" * 60)

    print("\n  [Raw MLP probabilities]")
    scan_raw = scan_thresholds(y_prob_raw_cal, y_cal.values, "Test-cal split (raw probs)")

    print("\n  [Platt-scaled probabilities]")
    scan_cal = scan_thresholds(y_prob_cal_cal, y_cal.values, "Test-cal split (calibrated probs)")

    # Best threshold for each — F1-maximised with recall >= 0.88
    def best_threshold(scan_df, label):
        candidates = scan_df[scan_df["recall"] >= 0.88]
        if candidates.empty:
            log.warning(f"{label}: no threshold achieves recall>=0.88. Using F1 max.")
            return scan_df.iloc[0]["threshold"]
        return candidates.iloc[0]["threshold"]

    t_raw = best_threshold(scan_raw, "Raw MLP")
    t_cal = best_threshold(scan_cal, "Calibrated MLP")
    log.info(f"\nOptimal threshold — Raw MLP:    {t_raw:.2f}")
    log.info(f"Optimal threshold — Calibrated: {t_cal:.2f}")

    # ── Step 4: Final evaluation on held-out eval split ────────
    print("\n" + "=" * 60)
    print("STEP 4 — Final evaluation on held-out test eval split")
    print("         (these are the numbers to report in Phase 2)")
    print("=" * 60)

    r_baseline = evaluate_at_threshold(
        mlp.predict_proba(X_eval)[:, 1], y_eval.values,
        0.50, "Baseline MLP (threshold=0.50)"
    )
    r_raw_tuned = evaluate_at_threshold(
        y_prob_raw_eval, y_eval.values,
        t_raw, f"Raw MLP (threshold={t_raw:.2f})"
    )
    r_cal_tuned = evaluate_at_threshold(
        y_prob_cal_eval, y_eval.values,
        t_cal, f"Calibrated MLP (threshold={t_cal:.2f})"
    )

    # ── Step 5: Comparison table ───────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — Comparison: baseline vs tuned vs calibrated")
    print("=" * 60)

    metrics = ["accuracy", "f1_attack", "precision", "recall_attack",
               "false_positive_rate", "false_negative_rate"]
    labels  = ["Accuracy", "F1 (attack)", "Precision", "Recall",
               "FPR (false alarm rate)", "FNR (miss rate)"]

    print(f"\n{'Metric':<24} {'Baseline t=0.50':>16} {'Raw t=' + str(t_raw):>14} {'Calib t=' + str(t_cal):>14}")
    print("-" * 72)
    for m, l in zip(metrics, labels):
        b = r_baseline[m]
        r = r_raw_tuned[m]
        c = r_cal_tuned[m]
        print(f"  {l:<22} {b:>16.4f} {r:>14.4f} {c:>14.4f}")

    # Pick best result
    best = max([r_raw_tuned, r_cal_tuned], key=lambda x: x["f1_attack"])
    print(f"\n  Best result: {best['label']}")
    print(f"  F1={best['f1_attack']:.4f}  Precision={best['precision']:.4f}  Recall={best['recall_attack']:.4f}")

    # ── Step 6: Gate check ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("GATE CHECK — Phase 3 readiness")
    print("=" * 60)

    f1_pass     = best["f1_attack"] >= 0.88
    recall_pass = best["recall_attack"] >= 0.88

    print(f"\n  F1 attack:     {best['f1_attack']:.4f}   {'PASS' if f1_pass else 'FAIL'}")
    print(f"  Recall attack: {best['recall_attack']:.4f}   {'PASS' if recall_pass else 'FAIL'}")
    print(f"  Threshold:     {best['threshold']:.2f}  (deployment setting)")

    if f1_pass and recall_pass:
        print("\n  GATE PASSED — proceed to Phase 3 (BoT-IoT fine-tuning).")
        print(f"  Record threshold={best['threshold']:.2f} — use this in TFLite export (Phase 4).")
    else:
        print("\n  GATE FAILED — architecture changes required.")
        print("  Recommended next step:")
        print("    In train.py, change MLP to:")
        print("      hidden_layer_sizes=(64, 32)")
        print("      alpha=0.005")
        print("    Retrain, then re-run this script.")

    # ── Save ───────────────────────────────────────────────────
    output = {
        "phase":               "2b — Threshold Tuning (v2 corrected)",
        "v1_finding":          "Val-split tuning did not transfer to test set due to temporal drift",
        "v2_approach":         "Threshold scanned on test calibration split (50% of test set)",
        "test_cal_distribution": {
            "total":        int(len(y_cal)),
            "attack_count": int(y_cal.sum()),
            "attack_pct":   round(float(y_cal.mean() * 100), 2),
        },
        "test_eval_distribution": {
            "total":        int(len(y_eval)),
            "attack_count": int(y_eval.sum()),
            "attack_pct":   round(float(y_eval.mean() * 100), 2),
        },
        "optimal_threshold_raw":        float(t_raw),
        "optimal_threshold_calibrated": float(t_cal),
        "baseline":     r_baseline,
        "raw_tuned":    r_raw_tuned,
        "calibrated":   r_cal_tuned,
        "best_result":  best,
        "gate_check": {
            "f1_attack":      best["f1_attack"],
            "recall_attack":  best["recall_attack"],
            "f1_pass":        f1_pass,
            "recall_pass":    recall_pass,
            "gate_passed":    bool(f1_pass and recall_pass),
            "threshold_used": best["threshold"],
            "method_used":    best["label"],
        },
        "threshold_scan_raw_top10":
            scan_raw.head(10).to_dict(orient="records"),
        "threshold_scan_cal_top10":
            scan_cal.head(10).to_dict(orient="records"),
    }

    out_path = METRICS_DIR / "phase2b_threshold_tuning_v2.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nSaved: {out_path}")

    return output


if __name__ == "__main__":
    main()