"""
src/botiot_evaluate.py
IoT-SecBand — Phase 3 evaluation: frozen MLP on NF-BoT-IoT

Pipeline position: UNSW training → frozen MLP → BoT-IoT alignment → evaluation

Purpose:
    Evaluate the frozen Phase 2 MLP (threshold=0.74) on the aligned BoT-IoT data.
    Run the same diagnostic toolkit used in Phase 2 (probability overlap, feature
    drift, threshold sweep) so results are directly comparable.

    Key comparisons:
        - F1 vs Phase 2 baseline (F1=0.7829 on UNSW holdout)
        - Probability overlap vs Phase 2 (overlap=0.351)
        - Feature drift: UNSW-train vs BoT-IoT per feature

    Red flag check:
        F1 > 0.95 is suspicious, not a win — re-run diagnostics immediately.
        The "fake 99% accuracy" failure mode from early Colab work was caused
        by refitting the scaler. We don't do that here, but extreme imbalance
        (97.7% attack) can still inflate accuracy-based metrics.

Run from project root:
    python src/botiot_evaluate.py

Outputs:
    outputs/metrics/P3_01_probability_overlap.png
    outputs/metrics/P3_02_feature_drift.png
    outputs/metrics/P3_03_threshold_sweep.png
    outputs/metrics/phase3_results.json
"""

import json
import logging
import pickle
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from botiot_preprocess import LOCKED_FEATURES, TARGET

PROC_DIR    = ROOT / "data" / "processed"
MODEL_DIR   = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

PHASE2_BASELINE = {
    "f1_attack":     0.7829,
    "precision":     0.7406,
    "recall_attack": 0.8303,
    "fpr":           0.1474,
    "threshold":     0.74,
    "dataset":       "UNSW-NB15 holdout (Phase 2F)",
}

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "axes.labelsize": 10,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. METRICS AT A GIVEN THRESHOLD
# ─────────────────────────────────────────────────────────────────────────────

def metrics_at_threshold(y_true, y_prob, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "threshold":     threshold,
        "accuracy":      round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_attack":     round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision":     round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_attack": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "fpr":           round(float(fpr), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROBABILITY OVERLAP
# ─────────────────────────────────────────────────────────────────────────────

def probability_overlap(y_true, y_prob, save_path: Path) -> dict:
    prob_normal = y_prob[y_true == 0]
    prob_attack = y_prob[y_true == 1]

    bins      = np.linspace(0, 1, 51)
    hist_n, _ = np.histogram(prob_normal, bins=bins, density=True)
    hist_a, _ = np.histogram(prob_attack, bins=bins, density=True)
    overlap   = float(np.sum(np.minimum(hist_n, hist_a)) * (bins[1] - bins[0]))
    ks_stat, ks_pval = stats.ks_2samp(prob_normal, prob_attack)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(prob_normal, bins=50, alpha=0.55, color="#1D9E75",
            density=True, label=f"Normal (n={len(prob_normal):,})", edgecolor="none")
    ax.hist(prob_attack, bins=50, alpha=0.55, color="#D85A30",
            density=True, label=f"Attack (n={len(prob_attack):,})", edgecolor="none")
    ax.axvline(0.50, color="#888780", linestyle="--", lw=1.0, label="threshold=0.50")
    ax.axvline(0.74, color="#333333", linestyle="--", lw=1.0, label="threshold=0.74 (locked)")
    ax.set_xlabel("P(attack)")
    ax.set_ylabel("Density")
    ax.set_title(f"BoT-IoT probability distributions\nOverlap={overlap:.3f}  KS={ks_stat:.3f}")
    ax.legend(fontsize=9)

    ax2 = axes[1]
    ax2.ecdf(prob_normal, color="#1D9E75", label="Normal", lw=1.5)
    ax2.ecdf(prob_attack, color="#D85A30", label="Attack", lw=1.5)
    ax2.axvline(0.74, color="#333333", linestyle="--", lw=1.0, label="threshold=0.74")
    ax2.set_xlabel("P(attack)")
    ax2.set_ylabel("Cumulative proportion")
    ax2.set_title("Cumulative distributions")
    ax2.legend(fontsize=9)

    fig.suptitle("Phase 3 — BoT-IoT Probability Overlap", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {save_path}")

    verdict = (
        "GOOD"        if overlap < 0.15 else
        "MODERATE"    if overlap < 0.30 else
        "CONCERNING"  if overlap < 0.50 else
        "CRITICAL"
    )
    log.info(f"  Overlap={overlap:.4f}  KS={ks_stat:.4f}  Verdict={verdict}")

    return {
        "overlap_coefficient": round(overlap, 4),
        "ks_statistic":        round(ks_stat, 4),
        "ks_pvalue":           float(ks_pval),
        "mean_prob_normal":    round(float(prob_normal.mean()), 4) if len(prob_normal) > 0 else None,
        "mean_prob_attack":    round(float(prob_attack.mean()), 4),
        "verdict":             verdict,
        "phase2_overlap":      0.351,
        "overlap_change":      round(overlap - 0.351, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE DRIFT: UNSW-NB15 train vs BoT-IoT
# ─────────────────────────────────────────────────────────────────────────────

def feature_drift(df_unsw_train, df_botiot, save_path: Path) -> dict:
    """
    Compare UNSW-NB15 training distribution vs BoT-IoT distribution per feature.
    Uses pre-scaled values so both are in the same coordinate space.
    """
    rows = []
    for feat in LOCKED_FEATURES:
        unsw_vals   = df_unsw_train[feat].dropna().values
        botiot_vals = df_botiot[feat].dropna().values
        ks_stat, ks_pval = stats.ks_2samp(unsw_vals, botiot_vals)
        rows.append({
            "feature":        feat,
            "ks_stat":        round(float(ks_stat), 4),
            "ks_pval":        float(ks_pval),
            "unsw_median":    round(float(np.median(unsw_vals)), 4),
            "botiot_median":  round(float(np.median(botiot_vals)), 4),
            "drift_level":    (
                "severe"   if ks_stat > 0.50 else
                "high"     if ks_stat > 0.25 else
                "moderate" if ks_stat > 0.10 else
                "minimal"
            ),
        })

    drift_df = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)
    log.info("\n  Feature drift (UNSW-NB15 train vs BoT-IoT, post-scale):")
    log.info(drift_df[["feature", "ks_stat", "drift_level",
                         "unsw_median", "botiot_median"]].to_string(index=False))

    colors = [
        "#D85A30" if r > 0.50 else
        "#E89B3A" if r > 0.25 else
        "#3A8AE8" if r > 0.10 else
        "#1D9E75"
        for r in drift_df["ks_stat"]
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(drift_df["feature"], drift_df["ks_stat"],
            color=colors, edgecolor="none", alpha=0.85)
    ax.axvline(0.50, color="#D85A30", linestyle="--", lw=0.9, label="Severe (0.50)")
    ax.axvline(0.25, color="#E89B3A", linestyle="--", lw=0.9, label="High (0.25)")
    ax.axvline(0.10, color="#3A8AE8", linestyle="--", lw=0.9, label="Moderate (0.10)")
    ax.set_xlabel("KS statistic (UNSW-NB15 train vs BoT-IoT)")
    ax.set_title("Phase 3 — Feature Drift: UNSW-NB15 vs BoT-IoT (post-scale)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {save_path}")

    high_drift = drift_df[drift_df["ks_stat"] > 0.25]["feature"].tolist()
    return {
        "drift_table":         drift_df.to_dict(orient="records"),
        "high_drift_features": high_drift,
        "n_high_drift":        len(high_drift),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. THRESHOLD SWEEP
# ─────────────────────────────────────────────────────────────────────────────

def threshold_sweep(y_true, y_prob, save_path: Path) -> dict:
    thresholds = np.arange(0.10, 0.95, 0.01)
    results = [metrics_at_threshold(y_true, y_prob, float(t)) for t in thresholds]

    f1s   = [r["f1_attack"] for r in results]
    precs = [r["precision"] for r in results]
    recs  = [r["recall_attack"] for r in results]
    fprs  = [r["fpr"] for r in results]

    best_idx   = int(np.argmax(f1s))
    best_t     = float(thresholds[best_idx])
    best_f1    = float(f1s[best_idx])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    ax.plot(thresholds, f1s,   color="#1D9E75", lw=1.8, label="F1 (attack)")
    ax.plot(thresholds, precs, color="#3A8AE8", lw=1.4, linestyle="--", label="Precision")
    ax.plot(thresholds, recs,  color="#D85A30", lw=1.4, linestyle="--", label="Recall")
    ax.axvline(0.74,     color="#333333", linestyle=":",  lw=1.2, label="Locked threshold=0.74")
    ax.axvline(best_t,   color="#E89B3A", linestyle="-.", lw=1.2, label=f"Best F1 threshold={best_t:.2f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"BoT-IoT — Threshold vs Metrics\nBest F1={best_f1:.4f} @ t={best_t:.2f}")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(thresholds, fprs, color="#D85A30", lw=1.8, label="FPR")
    ax2.axvline(0.74,   color="#333333", linestyle=":",  lw=1.2, label="Locked threshold=0.74")
    ax2.axvline(best_t, color="#E89B3A", linestyle="-.", lw=1.2, label=f"Best F1 t={best_t:.2f}")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("False Positive Rate")
    ax2.set_title("BoT-IoT — Threshold vs FPR")
    ax2.legend(fontsize=8)

    fig.suptitle("Phase 3 — BoT-IoT Threshold Sweep", fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {save_path}")

    return {
        "best_threshold":    best_t,
        "best_f1":           round(best_f1, 4),
        "at_locked_0_74":    results[int(np.argmin(np.abs(thresholds - 0.74)))],
        "at_best_threshold": results[best_idx],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_botiot_evaluation() -> dict:
    log.info("=" * 60)
    log.info("IoT-SecBand | Phase 3 — BoT-IoT Evaluation")
    log.info("=" * 60)

    # Load processed BoT-IoT
    botiot_path = PROC_DIR / "botiot_clean.parquet"
    if not botiot_path.exists():
        raise FileNotFoundError(
            f"BoT-IoT processed file not found: {botiot_path}\n"
            "Run src/botiot_preprocess.py first."
        )
    df_botiot = pd.read_parquet(botiot_path)
    log.info(f"Loaded BoT-IoT: {df_botiot.shape}")

    # Load UNSW-NB15 training data (for drift comparison)
    df_unsw_train = pd.read_parquet(PROC_DIR / "unsw_train_clean.parquet")
    # Subset to the 11 locked features only
    from train import FINAL_FEATURES as UNSW_13_FEATURES
    unsw_locked = [f for f in LOCKED_FEATURES if f in df_unsw_train.columns]
    df_unsw_train = df_unsw_train[unsw_locked]

    # Load frozen 11-feature MLP (produced by train_locked.py)
    locked_mlp_path = MODEL_DIR / "mlp_model_locked.pkl"
    if not locked_mlp_path.exists():
        raise FileNotFoundError(
            f"Not found: {locked_mlp_path}\n"
            "Run: python src/train_locked.py"
        )
    with open(locked_mlp_path, "rb") as f:
        mlp = pickle.load(f)
    log.info("Loaded locked 11-feature MLP")

    # Load locked scaler (same RobustScaler the MLP was trained with)
    scaler_pkl = MODEL_DIR / "scaler_locked.pkl"
    if not scaler_pkl.exists():
        raise FileNotFoundError(
            f"Not found: {scaler_pkl}\n"
            "Run: python src/train_locked.py"
        )
    with open(scaler_pkl, "rb") as f:
        scaler = pickle.load(f)

    X_botiot_raw = df_botiot[LOCKED_FEATURES].values.astype(float)
    X_botiot = scaler.transform(X_botiot_raw)
    y_botiot = df_botiot[TARGET].values

    log.info(f"BoT-IoT label dist: normal={int((y_botiot==0).sum()):,}  attack={int((y_botiot==1).sum()):,}")

    # Predict
    y_prob = mlp.predict_proba(X_botiot)[:, 1]

    # ── Evaluate at locked threshold 0.74 ──
    log.info("\n── Metrics at locked threshold=0.74 ──")
    m_locked = metrics_at_threshold(y_botiot, y_prob, 0.74)
    log.info(f"  F1={m_locked['f1_attack']}  Precision={m_locked['precision']}  "
             f"Recall={m_locked['recall_attack']}  FPR={m_locked['fpr']}")

    log.info("\n── Metrics at default threshold=0.50 ──")
    m_default = metrics_at_threshold(y_botiot, y_prob, 0.50)
    log.info(f"  F1={m_default['f1_attack']}  Precision={m_default['precision']}  "
             f"Recall={m_default['recall_attack']}  FPR={m_default['fpr']}")

    # ── Classification report ──
    y_pred_locked = (y_prob >= 0.74).astype(int)
    log.info("\n── Classification Report (threshold=0.74) ──")
    log.info("\n" + classification_report(y_botiot, y_pred_locked,
                                          target_names=["Normal", "Attack"]))

    # ── Probability overlap diagnostic ──
    log.info("\n── Probability Overlap ──")
    overlap_result = probability_overlap(
        y_botiot, y_prob,
        save_path=METRICS_DIR / "P3_01_probability_overlap.png",
    )

    # ── Feature drift diagnostic ──
    log.info("\n── Feature Drift ──")
    drift_result = feature_drift(
        df_unsw_train,
        df_botiot[LOCKED_FEATURES],
        save_path=METRICS_DIR / "P3_02_feature_drift.png",
    )

    # ── Threshold sweep ──
    log.info("\n── Threshold Sweep ──")
    sweep_result = threshold_sweep(
        y_botiot, y_prob,
        save_path=METRICS_DIR / "P3_03_threshold_sweep.png",
    )

    # ── Red flag check ──
    red_flags = []
    if m_locked["accuracy"] > 0.97:
        red_flags.append(
            f"Accuracy={m_locked['accuracy']:.4f} is suspiciously high. "
            "Likely caused by extreme class imbalance (97.7% attack). "
            "Use F1/recall, not accuracy."
        )
    if m_locked["f1_attack"] > 0.95:
        red_flags.append(
            f"F1={m_locked['f1_attack']:.4f} > 0.95. "
            "This matches the 'fake 99% accuracy' failure pattern. "
            "Verify the scaler was NOT refit on BoT-IoT data."
        )
    if overlap_result["overlap_coefficient"] < 0.05:
        red_flags.append(
            "Overlap < 0.05: suspiciously perfect separation. "
            "Check for data leakage or preprocessing errors."
        )

    # ── Comparison vs Phase 2 baseline ──
    delta_f1  = round(m_locked["f1_attack"] - PHASE2_BASELINE["f1_attack"], 4)
    delta_fpr = round(m_locked["fpr"] - PHASE2_BASELINE["fpr"], 4)
    comparison = {
        "phase2_baseline":     PHASE2_BASELINE,
        "botiot_at_0_74":      m_locked,
        "delta_f1":            delta_f1,
        "delta_fpr":           delta_fpr,
        "interpretation": (
            "IMPROVED — BoT-IoT traffic better aligned with UNSW model" if delta_f1 > 0.05 else
            "SIMILAR — comparable generalisation across datasets"        if abs(delta_f1) <= 0.05 else
            "DEGRADED — significant distribution shift from UNSW to BoT-IoT"
        ),
    }

    log.info("\n── Phase 2 vs Phase 3 Comparison ──")
    log.info(f"  UNSW holdout F1 = {PHASE2_BASELINE['f1_attack']:.4f}")
    log.info(f"  BoT-IoT F1      = {m_locked['f1_attack']:.4f}  (delta={delta_f1:+.4f})")
    log.info(f"  Interpretation: {comparison['interpretation']}")

    if red_flags:
        log.warning("\n⚠ RED FLAGS:")
        for flag in red_flags:
            log.warning(f"  • {flag}")

    output = {
        "phase":               "3 — BoT-IoT Evaluation",
        "features":            LOCKED_FEATURES,
        "botiot_shape":        list(df_botiot.shape),
        "at_locked_threshold": m_locked,
        "at_default_threshold": m_default,
        "probability_overlap": overlap_result,
        "feature_drift":       drift_result,
        "threshold_sweep":     sweep_result,
        "comparison":          comparison,
        "red_flags":           red_flags,
    }

    out_path = METRICS_DIR / "phase3_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\n✓ Full results: {out_path}")

    return output


if __name__ == "__main__":
    results = run_botiot_evaluation()
    print("\n── Summary ──")
    print(f"  F1 @ 0.74:    {results['at_locked_threshold']['f1_attack']}")
    print(f"  Recall:       {results['at_locked_threshold']['recall_attack']}")
    print(f"  FPR:          {results['at_locked_threshold']['fpr']}")
    print(f"  Overlap:      {results['probability_overlap']['overlap_coefficient']}")
    print(f"  vs Phase 2:   {results['comparison']['interpretation']}")
    if results["red_flags"]:
        print("\n  ⚠ RED FLAGS:")
        for f in results["red_flags"]:
            print(f"    • {f}")
