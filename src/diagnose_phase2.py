"""
src/diagnose_phase2.py
IoT-SecBand — Phase 2 diagnostic: probability overlap + feature drift

Purpose:
    Answer the fundamental question before committing to calibration:
    "Can the MLP separate attack from normal in the TEST set's feature space?"

    If yes  → calibration + threshold tuning is valid (run tune_threshold.py v2)
    If no   → feature drift is the root cause, retraining approach must change

Two diagnostics run:
    1. Probability overlap analysis
       - Plot P(attack) distributions for normal vs attack in test set
       - Compute overlap coefficient (0=perfect separation, 1=total overlap)
       - Overlap > 0.30 means calibration will not rescue F1

    2. Feature drift analysis
       - Compare each feature distribution: train-normal vs test-normal
       - Uses KS statistic (0=identical, 1=completely different)
       - High-drift features are causing normal flows to appear attack-like
       - These must be addressed in preprocessing before BoT-IoT fine-tuning

Run from project root:
    python src/diagnose_phase2.py

Outputs:
    outputs/metrics/D01_probability_overlap.png
    outputs/metrics/D02_feature_drift.png
    outputs/metrics/phase2_diagnostic.json
"""

import sys
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from train import FINAL_FEATURES, TARGET

PROC_DIR    = ROOT / "data" / "processed"
MODEL_DIR   = ROOT / "models"
METRICS_DIR = ROOT / "outputs" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
    "axes.labelsize": 10,
})


# ── Diagnostic 1: Probability Overlap ─────────────────────────────────────────

def probability_overlap_analysis(mlp, X_test, y_test):
    """
    Compute P(attack) for every test sample, split by true label.
    Overlap coefficient: proportion of the two distributions that overlap.

    Interpretation:
        overlap < 0.15  — excellent separation, calibration will work well
        0.15–0.30       — moderate overlap, calibration may help
        0.30–0.50       — significant overlap, calibration has limited effect
        > 0.50          — heavy overlap, model cannot separate in test space
                          feature drift is the root cause, retraining needed
    """
    y_prob = mlp.predict_proba(X_test)[:, 1]
    y_true = y_test.values

    prob_normal = y_prob[y_true == 0]
    prob_attack = y_prob[y_true == 1]

    # Overlap coefficient via histogram intersection
    bins      = np.linspace(0, 1, 51)
    hist_n, _ = np.histogram(prob_normal, bins=bins, density=True)
    hist_a, _ = np.histogram(prob_attack, bins=bins, density=True)
    bin_width = bins[1] - bins[0]
    overlap   = float(np.sum(np.minimum(hist_n, hist_a)) * bin_width)

    # KS test: are normal and attack probability distributions different?
    ks_stat, ks_pval = stats.ks_2samp(prob_normal, prob_attack)

    # Stats
    log.info("\n── Probability Overlap Analysis ──")
    log.info(f"  Normal flows  — mean P(attack): {prob_normal.mean():.4f}  std: {prob_normal.std():.4f}")
    log.info(f"  Attack flows  — mean P(attack): {prob_attack.mean():.4f}  std: {prob_attack.std():.4f}")
    log.info(f"  Overlap coefficient: {overlap:.4f}")
    log.info(f"  KS statistic:        {ks_stat:.4f}  (p={ks_pval:.2e})")

    if overlap < 0.15:
        verdict = "GOOD — clear separation. Calibration + threshold tuning will work."
    elif overlap < 0.30:
        verdict = "MODERATE — partial overlap. Calibration worth trying, expect partial improvement."
    elif overlap < 0.50:
        verdict = "CONCERNING — significant overlap. Calibration has limited effect. Check feature drift."
    else:
        verdict = "CRITICAL — heavy overlap. Model cannot separate classes in test space. Retraining required."

    log.info(f"  Verdict: {verdict}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: histogram of P(attack) by true class
    ax = axes[0]
    ax.hist(prob_normal, bins=50, alpha=0.55, color="#1D9E75",
            density=True, label=f"True Normal (n={len(prob_normal):,})", edgecolor="none")
    ax.hist(prob_attack, bins=50, alpha=0.55, color="#D85A30",
            density=True, label=f"True Attack (n={len(prob_attack):,})", edgecolor="none")
    ax.axvline(0.50, color="#888780", linestyle="--", lw=1.0, label="Default threshold=0.50")
    ax.set_xlabel("MLP predicted P(attack)")
    ax.set_ylabel("Density")
    ax.set_title(f"Probability distributions — test set\nOverlap={overlap:.3f}  KS={ks_stat:.3f}")
    ax.legend(fontsize=9)

    # Right: cumulative distributions
    ax2 = axes[1]
    ax2.ecdf(prob_normal, color="#1D9E75", label="True Normal", lw=1.5)
    ax2.ecdf(prob_attack, color="#D85A30", label="True Attack", lw=1.5)
    ax2.axvline(0.50, color="#888780", linestyle="--", lw=1.0)
    ax2.set_xlabel("MLP predicted P(attack)")
    ax2.set_ylabel("Cumulative proportion")
    ax2.set_title("Cumulative probability distributions")
    ax2.legend(fontsize=9)

    fig.suptitle("Diagnostic 1 — Probability Overlap Analysis", fontsize=11, y=1.01)
    fig.tight_layout()
    out = METRICS_DIR / "D01_probability_overlap.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out}")

    return {
        "overlap_coefficient": round(overlap, 4),
        "ks_statistic":        round(ks_stat, 4),
        "ks_pvalue":           float(ks_pval),
        "mean_prob_normal":    round(float(prob_normal.mean()), 4),
        "mean_prob_attack":    round(float(prob_attack.mean()), 4),
        "std_prob_normal":     round(float(prob_normal.std()), 4),
        "std_prob_attack":     round(float(prob_attack.std()), 4),
        "verdict":             verdict,
        "calibration_viable":  bool(overlap < 0.30),
        "plot":                str(out),
    }


# ── Diagnostic 2: Feature Drift ───────────────────────────────────────────────

def feature_drift_analysis(df_train, df_test):
    """
    Compare each feature's distribution for NORMAL flows only:
    train-normal vs test-normal.

    If a feature has drifted significantly for normal traffic:
    - The training scaler maps test-normal values into a region the model
      learned as attack-like during training.
    - This is the root cause of false positives on the test set.

    KS statistic per feature:
        < 0.10  — minimal drift, not a concern
        0.10–0.25 — moderate drift, worth monitoring
        0.25–0.50 — significant drift, likely contributing to FP rate
        > 0.50  — severe drift, this feature is a primary FP cause
    """
    train_normal = df_train[df_train[TARGET] == 0][FINAL_FEATURES]
    test_normal  = df_test[df_test[TARGET] == 0][FINAL_FEATURES]
    train_attack = df_train[df_train[TARGET] == 1][FINAL_FEATURES]
    test_attack  = df_test[df_test[TARGET] == 1][FINAL_FEATURES]

    log.info(f"\n── Feature Drift Analysis ──")
    log.info(f"  Train normal: {len(train_normal):,} | Test normal: {len(test_normal):,}")
    log.info(f"  Train attack: {len(train_attack):,} | Test attack: {len(test_attack):,}")

    drift_rows = []
    for feat in FINAL_FEATURES:
        ks_normal_stat, ks_normal_p = stats.ks_2samp(
            train_normal[feat].dropna(), test_normal[feat].dropna()
        )
        ks_attack_stat, ks_attack_p = stats.ks_2samp(
            train_attack[feat].dropna(), test_attack[feat].dropna()
        )
        drift_rows.append({
            "feature":               feat,
            "ks_normal":             round(float(ks_normal_stat), 4),
            "ks_normal_pval":        float(ks_normal_p),
            "ks_attack":             round(float(ks_attack_stat), 4),
            "ks_attack_pval":        float(ks_attack_p),
            "normal_drift_level":    (
                "severe"   if ks_normal_stat > 0.50 else
                "high"     if ks_normal_stat > 0.25 else
                "moderate" if ks_normal_stat > 0.10 else
                "minimal"
            ),
            "train_normal_mean":     round(float(train_normal[feat].mean()), 4),
            "test_normal_mean":      round(float(test_normal[feat].mean()), 4),
            "mean_shift_pct":        round(
                abs(test_normal[feat].mean() - train_normal[feat].mean())
                / (abs(train_normal[feat].mean()) + 1e-9) * 100, 2
            ),
        })

    drift_df = pd.DataFrame(drift_rows).sort_values("ks_normal", ascending=False)

    log.info("\n  Feature drift (normal traffic, train vs test) — sorted by KS stat:")
    log.info(drift_df[["feature", "ks_normal", "normal_drift_level",
                         "train_normal_mean", "test_normal_mean",
                         "mean_shift_pct"]].to_string(index=False))

    high_drift = drift_df[drift_df["ks_normal"] > 0.25]["feature"].tolist()
    if high_drift:
        log.warning(f"\n  High-drift features (KS > 0.25): {high_drift}")
        log.warning("  These features are mapping test-normal flows into attack-like regions.")
        log.warning("  Consider robust scaling or dropping before BoT-IoT fine-tuning.")

    # Plot: KS statistic per feature (normal drift)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = [
        "#D85A30" if r > 0.50 else
        "#E89B3A" if r > 0.25 else
        "#3A8AE8" if r > 0.10 else
        "#1D9E75"
        for r in drift_df["ks_normal"]
    ]

    ax = axes[0]
    ax.barh(drift_df["feature"], drift_df["ks_normal"],
            color=colors, edgecolor="none", alpha=0.85)
    ax.axvline(0.25, color="#D85A30", linestyle="--", lw=0.9, label="High drift (0.25)")
    ax.axvline(0.10, color="#E89B3A", linestyle="--", lw=0.9, label="Moderate drift (0.10)")
    ax.set_xlabel("KS statistic (train-normal vs test-normal)")
    ax.set_title("Normal traffic drift per feature")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.barh(drift_df["feature"], drift_df["ks_attack"],
             color="#378ADD", edgecolor="none", alpha=0.75)
    ax2.axvline(0.25, color="#D85A30", linestyle="--", lw=0.9, label="High drift (0.25)")
    ax2.set_xlabel("KS statistic (train-attack vs test-attack)")
    ax2.set_title("Attack traffic drift per feature")
    ax2.legend(fontsize=8)

    fig.suptitle("Diagnostic 2 — Feature Drift: Train vs Test by Class", fontsize=11, y=1.01)
    fig.tight_layout()
    out = METRICS_DIR / "D02_feature_drift.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out}")

    return {
        "drift_table":          drift_df.to_dict(orient="records"),
        "high_drift_features":  high_drift,
        "n_high_drift":         len(high_drift),
        "plot":                 str(out),
    }


# ── Decision Logic ─────────────────────────────────────────────────────────────

def gate_decision(overlap_result, drift_result):
    overlap   = overlap_result["overlap_coefficient"]
    n_drift   = drift_result["n_high_drift"]
    high_feat = drift_result["high_drift_features"]

    print("\n" + "=" * 60)
    print("DIAGNOSTIC VERDICT")
    print("=" * 60)
    print(f"\n  Probability overlap: {overlap:.4f}")
    print(f"  High-drift features: {n_drift} — {high_feat}")

    if overlap < 0.15 and n_drift == 0:
        decision = "CALIBRATE"
        action   = (
            "Model separates classes well. No significant feature drift.\n"
            "Run tune_threshold.py v2 — calibration will fix F1.\n"
            "Gate will pass after calibration."
        )
    elif overlap < 0.30 and n_drift <= 3:
        decision = "CALIBRATE_THEN_ASSESS"
        action   = (
            "Moderate overlap and/or low feature drift.\n"
            "Run tune_threshold.py v2. If F1 >= 0.88 — gate passes.\n"
            "If F1 < 0.85 after calibration — investigate drift features\n"
            "before proceeding to BoT-IoT fine-tuning."
        )
    elif overlap >= 0.30 and n_drift <= 2:
        decision = "RETRAIN_WITH_ROBUST_SCALING"
        action   = (
            f"Significant probability overlap ({overlap:.3f}) despite low feature drift.\n"
            "Model capacity may be insufficient. Retrain MLP with:\n"
            "  hidden_layer_sizes=(64, 32)\n"
            "  alpha=0.005\n"
            "Then re-run this diagnostic."
        )
    else:
        decision = "FIX_FEATURES_THEN_RETRAIN"
        action   = (
            f"High overlap ({overlap:.3f}) + {n_drift} drifted features.\n"
            f"Drifted: {high_feat}\n"
            "Feature drift is causing test-normal flows to appear attack-like.\n"
            "Actions before retraining:\n"
            "  1. Apply RobustScaler instead of StandardScaler for high-drift features\n"
            "  2. Or drop the highest-drift features from FINAL_FEATURES\n"
            "  3. Rerun Phase 1 preprocess → retrain → re-diagnose\n"
            "Do NOT proceed to BoT-IoT fine-tuning until this is resolved."
        )

    print(f"\n  Decision: {decision}")
    print(f"\n  Action:\n  {action}")

    return {"decision": decision, "action": action}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("IoT-SecBand | Phase 2 — Diagnostic Analysis")
    print("=" * 60)

    df_train = pd.read_parquet(PROC_DIR / "unsw_train_clean.parquet")
    df_test  = pd.read_parquet(PROC_DIR / "unsw_test_clean.parquet")

    with open(MODEL_DIR / "mlp_model.pkl", "rb") as f:
        mlp = pickle.load(f)

    X_test = df_test[FINAL_FEATURES]
    y_test = df_test[TARGET]

    print("\n" + "=" * 60)
    print("DIAGNOSTIC 1 — Probability Overlap")
    print("=" * 60)
    overlap_result = probability_overlap_analysis(mlp, X_test, y_test)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC 2 — Feature Drift (train vs test, per class)")
    print("=" * 60)
    drift_result = feature_drift_analysis(df_train, df_test)

    verdict = gate_decision(overlap_result, drift_result)

    output = {
        "phase":               "2 — Diagnostic",
        "probability_overlap": overlap_result,
        "feature_drift":       drift_result,
        "verdict":             verdict,
    }

    out_path = METRICS_DIR / "phase2_diagnostic.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nFull results saved: {out_path}")
    log.info(f"Plots: {METRICS_DIR}/D01_probability_overlap.png")
    log.info(f"       {METRICS_DIR}/D02_feature_drift.png")

    return output


if __name__ == "__main__":
    main()