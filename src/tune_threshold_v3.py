"""
src/tune_threshold_v3.py
IoT-SecBand — Phase 2E: Threshold Calibration on Locked Configuration

LOCKED CONFIGURATION (from Phase 2D, count-corrected):
    RobustScaler
    11 features (dropped: dload, dinpkt)
    MLP (32, 16)

Phase 2D results for this config (at threshold=0.50):
    F1=0.7299  Precision=0.5864  Recall=0.9664  AUC=0.9267
    FPR=0.3454  Overlap=0.3217

Phase 2E diagnostic (no-split, full test set, threshold sweep):
    Best F1 threshold ~0.74 -> F1=0.7788  Precision=0.7394
    Recall=0.8226  FPR=0.1469 (57% relative reduction in false alarms)
    This confirms the predicted ceiling (~0.78-0.83) from AUC/overlap analysis.

DECISION (Phase 2E consensus):
    Freeze this configuration. Do not pursue further feature removal
    (dbytes/dmean carry high MI -- removing them risks losing genuine
    signal, unlike dload/dinpkt which combined high drift with high
    model over-reliance). Do not change architecture -- CV F1~0.90 and
    AUC~0.93 show no capacity problem; the bottleneck is UNSW-NB15
    temporal distribution shift, which architecture cannot fix.

    This script performs the ONE-TIME calibration/holdout split on this
    frozen pipeline to establish the final reported benchmark before
    Phase 3 (BoT-IoT fine-tuning).

METHODOLOGICAL NOTE — READ BEFORE RUNNING:
    This script splits the official UNSW-NB15 test set 50/50 (stratified).
    - Calibration half: used to find optimal threshold
    - Evaluation half:  used for final reporting

    IMPORTANT: If you proceed with this approach, the evaluation half
    becomes your new permanent holdout for this model configuration.
    The original full test set can no longer be treated as an untouched
    benchmark for THIS model, because half of it informed the threshold.

    This must be documented in any report/paper as:
    "Decision threshold was calibrated on a held-out 50% partition of the
    UNSW-NB15 test set; final metrics are reported on the remaining 50%,
    which was not used in training, scaling, or threshold selection."

Outputs:
    outputs/metrics/phase2e_threshold_locked.json
    outputs/metrics/E01_roc_curve.png
    outputs/metrics/E02_pr_curve.png
    outputs/metrics/E03_threshold_vs_metrics.png

Run from project root:
    python src/tune_threshold_v3.py
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
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
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

# ── LOCKED CONFIGURATION ─────────────────────────────────────────────────────
LOCKED_FEATURES = [
    "dur", "proto", "service", "state",
    "sbytes", "dbytes", "rate",
    "sload", "sinpkt", "smean", "dmean",
]  # 11 features — dload and dinpkt dropped per Phase 2D (count-corrected)

assert len(LOCKED_FEATURES) == 11, f"Expected 11 features, got {len(LOCKED_FEATURES)}"

MLP_PARAMS = dict(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=0.001,
    batch_size=256,
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10,
    random_state=42,
)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})


def evaluate_at_threshold(y_prob, y_true, threshold, label):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    f1  = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    pre = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1))
    acc = float(accuracy_score(y_true, y_pred))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    result = {
        "label": label, "threshold": round(float(threshold), 3),
        "accuracy": round(acc, 4), "f1_attack": round(f1, 4),
        "precision": round(pre, 4), "recall_attack": round(rec, 4),
        "fpr": round(fpr, 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "meets_target": bool(f1 >= 0.88 and rec >= 0.88),
    }
    log.info(f"\n── {label} (threshold={threshold:.3f}) ──")
    log.info(f"  F1={f1:.4f}  Precision={pre:.4f}  Recall={rec:.4f}  FPR={fpr:.4f}  Acc={acc:.4f}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))
    return result


def plot_roc(y_true, y_prob, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#378ADD", lw=1.8, label=f"AUC={auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — locked config (11 features, RobustScaler)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {save_path}")
    return float(auc)


def plot_pr(y_true, y_prob, save_path):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(recall, precision, color="#D85A30", lw=1.8)
    ax.axhline(0.88, color="#888780", linestyle="--", lw=0.8, label="Target precision (0.88)")
    ax.axvline(0.88, color="#888780", linestyle=":", lw=0.8, label="Target recall (0.88)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve — locked config")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {save_path}")
    return precision, recall, thresholds


def plot_threshold_sweep(y_true, y_prob, save_path):
    thresholds = np.arange(0.05, 0.96, 0.01)
    f1s, pres, recs = [], [], []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1s.append(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
        pres.append(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        recs.append(recall_score(y_true, y_pred, pos_label=1))

    best_idx = int(np.argmax(f1s))
    best_t   = thresholds[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, f1s,  color="#1D9E75", lw=1.8, label="F1")
    ax.plot(thresholds, pres, color="#378ADD", lw=1.5, label="Precision", alpha=0.8)
    ax.plot(thresholds, recs, color="#D85A30", lw=1.5, label="Recall", alpha=0.8)
    ax.axvline(best_t, color="#888780", linestyle="--", lw=0.9,
               label=f"Best F1 threshold={best_t:.2f}")
    ax.axhline(0.88, color="#888780", linestyle=":", lw=0.8, label="Target (0.88)")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep — F1 / Precision / Recall (calibration split)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {save_path}")

    return float(best_t), float(f1s[best_idx]), float(pres[best_idx]), float(recs[best_idx])


def main():
    print("=" * 60)
    print("IoT-SecBand | Phase 2E — Threshold Calibration (Locked Config)")
    print("RobustScaler + 11 features + MLP(32,16)")
    print("=" * 60)

    # ── Load and prepare ───────────────────────────────────────
    df_train_raw = load_parquet(RAW_TRAIN)
    df_test_raw  = load_parquet(RAW_TEST)
    df_train_clean = clean(df_train_raw)
    df_test_clean  = clean(df_test_raw)
    df_train_enc, encoders = encode_categoricals(df_train_clean, fit=True)
    df_test_enc,  _        = encode_categoricals(df_test_clean, encoders=encoders, fit=False)

    scaler = RobustScaler()
    train = df_train_enc.copy()
    test  = df_test_enc.copy()
    scaler.fit(train[LOCKED_FEATURES])
    train[LOCKED_FEATURES] = scaler.transform(train[LOCKED_FEATURES])
    test[LOCKED_FEATURES]  = scaler.transform(test[LOCKED_FEATURES])

    X_train, y_train = train[LOCKED_FEATURES], train[TARGET]
    X_test,  y_test  = test[LOCKED_FEATURES],  test[TARGET]

    # ── Train locked model ─────────────────────────────────────
    print("\nTraining locked configuration...")
    mlp = MLPClassifier(**MLP_PARAMS)
    mlp.fit(X_train, y_train)
    log.info(f"MLP trained — iterations: {mlp.n_iter_}")

    # ── Step 1: Split test set 50/50 ───────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 — Split test set: calibration (50%) / final eval (50%)")
    print("=" * 60)
    print("\n*** METHODOLOGICAL NOTE ***")
    print("The 'final eval' half below becomes the new permanent holdout")
    print("for this model configuration. Document this in your write-up.")
    print("=" * 60)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
    cal_idx, eval_idx = list(sss.split(X_test, y_test))[0]
    X_cal,  y_cal  = X_test.iloc[cal_idx],  y_test.iloc[cal_idx]
    X_eval, y_eval = X_test.iloc[eval_idx], y_test.iloc[eval_idx]

    log.info(f"Calibration split: {len(X_cal):,} | Attack: {y_cal.mean()*100:.1f}%")
    log.info(f"Final eval split:  {len(X_eval):,} | Attack: {y_eval.mean()*100:.1f}%")

    y_prob_cal  = mlp.predict_proba(X_cal)[:, 1]
    y_prob_eval = mlp.predict_proba(X_eval)[:, 1]

    # ── Step 2: Research artifacts (on calibration split) ──────
    print("\n" + "=" * 60)
    print("STEP 2 — Generating research artifacts (calibration split)")
    print("=" * 60)

    auc = plot_roc(y_cal.values, y_prob_cal, METRICS / "E01_roc_curve.png")
    plot_pr(y_cal.values, y_prob_cal, METRICS / "E02_pr_curve.png")
    best_t, best_f1, best_pre, best_rec = plot_threshold_sweep(
        y_cal.values, y_prob_cal, METRICS / "E03_threshold_vs_metrics.png"
    )

    log.info(f"\nAUC (calibration split): {auc:.4f}")
    log.info(f"Best threshold (max F1): {best_t:.3f}")
    log.info(f"  At this threshold — F1={best_f1:.4f}  Precision={best_pre:.4f}  Recall={best_rec:.4f}")

    # ── Step 3: Evaluate on final holdout at multiple thresholds ──
    print("\n" + "=" * 60)
    print("STEP 3 — Final evaluation on held-out half")
    print("         (THIS IS THE NUMBER TO REPORT)")
    print("=" * 60)

    r_050 = evaluate_at_threshold(y_prob_eval, y_eval.values, 0.50, "Default (t=0.50)")
    r_best = evaluate_at_threshold(y_prob_eval, y_eval.values, best_t, f"Calibrated (t={best_t:.2f})")

    # ── Step 4: Summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Summary: default vs calibrated threshold")
    print("=" * 60)
    print(f"\n{'Metric':<16} {'t=0.50':>10} {'t=' + f'{best_t:.2f}':>10} {'Delta':>10}")
    print("-" * 48)
    for m in ["accuracy", "f1_attack", "precision", "recall_attack", "fpr"]:
        d = r_best[m] - r_050[m]
        print(f"  {m:<14} {r_050[m]:>10.4f} {r_best[m]:>10.4f} {d:>+10.4f}")

    # ── Gate check ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GATE CHECK — Phase 3 readiness")
    print("=" * 60)
    f1_pass = r_best["f1_attack"] >= 0.88
    rec_pass = r_best["recall_attack"] >= 0.88
    print(f"\n  F1 attack:     {r_best['f1_attack']:.4f}   {'PASS' if f1_pass else 'FAIL'}")
    print(f"  Recall attack: {r_best['recall_attack']:.4f}   {'PASS' if rec_pass else 'FAIL'}")
    print(f"  Threshold:     {best_t:.3f}")

    if f1_pass and rec_pass:
        verdict_text = "GATE PASSED (unexpected, given Phase 2E prediction). Proceed to Phase 3."
        verdict_short = "PASSED"
    elif r_best["f1_attack"] >= 0.75:
        verdict_text = (
            "Result consistent with the documented UNSW-NB15 ceiling (~0.78-0.83),\n"
            "  predicted from AUC/overlap analysis and confirmed in Phase 2E.\n"
            "  This is the EXPECTED outcome, not a failure requiring further action.\n"
            "  Action: Record this as the final calibrated benchmark. Document the\n"
            "  ceiling as a UNSW-NB15 temporal drift finding in the research report.\n"
            "  Proceed to Phase 3 (BoT-IoT fine-tuning) with this baseline."
        )
        verdict_short = "AT EXPECTED CEILING — DOCUMENT AND PROCEED TO PHASE 3"
    else:
        verdict_text = (
            "F1 notably below the 0.78-0.83 predicted range — this would be\n"
            "  unexpected given Phase 2E (full test set) results. Before changing\n"
            "  anything, check: did the calibration/eval split happen to be unusually\n"
            "  skewed? Re-run with a different random_state to confirm before\n"
            "  considering architecture changes."
        )
        verdict_short = "BELOW EXPECTED CEILING — VERIFY BEFORE ACTING"

    print(f"\n  {verdict_text}")

    # ── Save ─────────────────────────────────────────────────────
    output = {
        "phase": "2E — Threshold Calibration (Locked Config)",
        "locked_config": {
            "scaler": "RobustScaler",
            "features": LOCKED_FEATURES,
            "n_features": len(LOCKED_FEATURES),
            "mlp_params": {k: str(v) for k, v in MLP_PARAMS.items()},
        },
        "methodological_note": (
            "Threshold calibrated on 50% of official UNSW-NB15 test set. "
            "Final metrics reported on the other 50%, which was not used "
            "in training, scaling, or threshold selection. This 50% becomes "
            "the de facto holdout for this configuration."
        ),
        "calibration_split": {
            "n": int(len(y_cal)), "attack_pct": round(float(y_cal.mean()*100), 2),
            "auc": round(auc, 4), "best_threshold": round(best_t, 3),
            "best_f1": round(best_f1, 4), "best_precision": round(best_pre, 4),
            "best_recall": round(best_rec, 4),
        },
        "final_eval_split": {
            "n": int(len(y_eval)), "attack_pct": round(float(y_eval.mean()*100), 2),
        },
        "default_threshold_result":    r_050,
        "calibrated_threshold_result": r_best,
        "gate_check": {
            "f1_pass": f1_pass, "recall_pass": rec_pass,
            "verdict": verdict_short, "threshold_used": round(best_t, 3),
        },
        "plots": {
            "roc_curve":            str(METRICS / "E01_roc_curve.png"),
            "pr_curve":             str(METRICS / "E02_pr_curve.png"),
            "threshold_sweep":      str(METRICS / "E03_threshold_vs_metrics.png"),
        },
    }

    out_path = METRICS / "phase2e_threshold_locked.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()