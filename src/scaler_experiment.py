"""
src/scaler_experiment.py  (v2 — feature drop experiment)
IoT-SecBand — Phase 2D: Feature Drop Experiment

Context:
    Phase 2C confirmed RobustScaler is better (overlap 0.417 → 0.351, AUC +0.012)
    but F1 only improved marginally (+0.007). Root cause is not scaler alone.

    Feature importance plots revealed:
        dload  — DT importance 0.49 (dominant), RF 0.175 (highest), drift 69%
        dinpkt — RF importance 0.150 (3rd), DT ~0, drift 123%
        dbytes — MI 0.44 (2nd highest), tree importance low, drift 178%
        dmean  — MI 0.33 (3rd), tree importance low, drift 153%

    dbytes and dmean have high MI but low tree reliance → keep (real signal)
    dload and dinpkt have high tree reliance AND high drift → over-anchoring risk

Hypothesis:
    Tree models and MLP have over-anchored to dload (and secondarily dinpkt),
    features that shift significantly between UNSW-NB15 train/test periods.
    Removing them forces the model to rely on more stable features,
    reducing false positives on test-period normal traffic.

Experiments:
    Run 1: RobustScaler, 13 features (baseline from Phase 2C)
    Run 2: RobustScaler, 11 features (drop dload)
    Run 3: RobustScaler, 10 features (drop dload + dinpkt)

Key metric added: False Positive Rate (FPR)
    FPR = FP / (FP + TN) = normal flows misclassified as attack
    Current FPR ≈ 0.40 (40% of normal test flows flagged as attack)
    This is the direct measurement of the failure mode.

Success criteria (from combined model analysis):
    Strong evidence dload is culprit:  overlap→0.20, precision→0.75, F1→0.82+
    Strong evidence it is a symptom:   overlap→0.33, precision→0.58, F1→0.73

Run from project root:
    python src/scaler_experiment.py
"""

import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, roc_auc_score, confusion_matrix
)
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import load_parquet, clean, encode_categoricals, TARGET_COLUMN
from train import TARGET

RAW_TRAIN = ROOT / "data" / "raw" / "unsw_nb15" / "UNSW_NB15_training-set.parquet"
RAW_TEST  = ROOT / "data" / "raw" / "unsw_nb15" / "UNSW_NB15_testing-set.parquet"
METRICS   = ROOT / "outputs" / "metrics"
METRICS.mkdir(parents=True, exist_ok=True)

# ── Feature sets ───────────────────────────────────────────────────────────────

FEATURES_13 = [
    "dur", "proto", "service", "state",
    "sbytes", "dbytes", "rate",
    "sload", "dload", "sinpkt", "dinpkt", "smean", "dmean",
]

FEATURES_11 = [                        # drop dload (DT dominant, drift 69%)
    "dur", "proto", "service", "state",
    "sbytes", "dbytes", "rate",
    "sload", "sinpkt", "dinpkt", "smean", "dmean",
]

FEATURES_10 = [                        # drop dload + dinpkt (RF 3rd, drift 123%)
    "dur", "proto", "service", "state",
    "sbytes", "dbytes", "rate",
    "sload", "sinpkt", "smean", "dmean",
]

EXPERIMENTS = [
    ("13 features (baseline)",    FEATURES_13),
    ("11 features (drop dload)",  FEATURES_11),
    ("10 features (drop dload+dinpkt)", FEATURES_10),
]

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


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob, label):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    f1   = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    pre  = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred, pos_label=1)
    acc  = accuracy_score(y_true, y_pred)
    auc  = roc_auc_score(y_true, y_prob)

    # Probability overlap
    prob_n = y_prob[y_true == 0]
    prob_a = y_prob[y_true == 1]
    bins   = np.linspace(0, 1, 51)
    hn, _  = np.histogram(prob_n, bins=bins, density=True)
    ha, _  = np.histogram(prob_a, bins=bins, density=True)
    overlap = float(np.sum(np.minimum(hn, ha)) * (bins[1] - bins[0]))
    ks_stat, _ = stats.ks_2samp(prob_n, prob_a)

    return {
        "label":          label,
        "accuracy":       round(float(acc),     4),
        "f1_attack":      round(float(f1),      4),
        "precision":      round(float(pre),     4),
        "recall_attack":  round(float(rec),     4),
        "roc_auc":        round(float(auc),     4),
        "fpr":            round(float(fpr),     4),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "normal_flagged_as_attack": int(fp),
        "total_normal":             int(fp + tn),
        "overlap":        round(overlap,        4),
        "ks_stat":        round(float(ks_stat), 4),
        "meets_target":   bool(f1 >= 0.88 and rec >= 0.88),
    }


# ── Single run ─────────────────────────────────────────────────────────────────

def run_experiment(name, features, df_train_enc, df_test_enc):
    log.info(f"\n{'='*55}")
    log.info(f"Experiment: {name}")
    log.info(f"Features ({len(features)}): {features}")
    log.info(f"{'='*55}")

    train = df_train_enc.copy()
    test  = df_test_enc.copy()

    scaler = RobustScaler()
    scaler.fit(train[features])
    train[features] = scaler.transform(train[features])
    test[features]  = scaler.transform(test[features])

    X_train, y_train = train[features], train[TARGET]
    X_test,  y_test  = test[features],  test[TARGET]

    mlp = MLPClassifier(**MLP_PARAMS)
    mlp.fit(X_train, y_train)
    log.info(f"MLP trained — iterations: {mlp.n_iter_}")

    y_pred = mlp.predict(X_test)
    y_prob = mlp.predict_proba(X_test)[:, 1]

    result = compute_metrics(y_test.values, y_pred, y_prob, name)

    log.info(
        f"  f1={result['f1_attack']:.4f}  pre={result['precision']:.4f}  "
        f"rec={result['recall_attack']:.4f}  fpr={result['fpr']:.4f}  "
        f"auc={result['roc_auc']:.4f}  overlap={result['overlap']:.4f}"
    )
    log.info(
        f"  Normal flows flagged as attack: "
        f"{result['normal_flagged_as_attack']:,} / {result['total_normal']:,} "
        f"({result['fpr']*100:.1f}%)"
    )
    return result, mlp


# ── Verdict logic ──────────────────────────────────────────────────────────────

def verdict(results: list[dict]) -> dict:
    base  = results[0]   # 13-feature baseline
    exp_a = results[1]   # 11-feature (drop dload)
    exp_b = results[2]   # 10-feature (drop dload + dinpkt)

    best = max(results[1:], key=lambda r: r["f1_attack"])

    overlap_drop  = base["overlap"]   - best["overlap"]
    precision_gain = best["precision"] - base["precision"]
    f1_gain        = best["f1_attack"] - base["f1_attack"]
    fpr_drop       = base["fpr"]       - best["fpr"]

    # Success criteria from combined model analysis
    strong_culprit = (
        overlap_drop   > 0.15  and   # 0.351 → < 0.20
        precision_gain > 0.15  and   # 0.55  → > 0.70
        f1_gain        > 0.10        # 0.71  → > 0.81
    )
    symptom_only = (
        overlap_drop   < 0.03  and
        precision_gain < 0.05  and
        f1_gain        < 0.03
    )
    fpr_halved = fpr_drop > (base["fpr"] * 0.40)   # 40%+ reduction in FPR

    if strong_culprit:
        decision = "DLOAD IS THE CULPRIT"
        action = (
            f"Dropping {best['label'].split('(')[1].rstrip(')')} meaningfully resolves "
            f"the false positive problem.\n"
            f"Update FINAL_FEATURES in train.py to use {best['label']}.\n"
            f"Run diagnose_phase2.py to confirm overlap < 0.25.\n"
            f"Then run tune_threshold v2 — gate check should pass."
        )
    elif symptom_only:
        decision = "DLOAD IS A SYMPTOM — ROOT CAUSE LIES ELSEWHERE"
        action = (
            "Feature removal did not move the metrics meaningfully.\n"
            "The distribution shift is pervasive across features, not localised.\n"
            "Next steps:\n"
            "  1. Increase MLP to hidden_layer_sizes=(64, 32), alpha=0.005\n"
            "  2. Retrain and re-diagnose\n"
            "  3. If still failing, the UNSW train/test temporal gap is too\n"
            "     large for this feature set — reconsider feature engineering."
        )
    elif fpr_halved:
        decision = "PARTIAL — FPR IMPROVING, CONTINUE THIS PATH"
        action = (
            f"FPR dropped {fpr_drop*100:.1f}pp — false alarm rate is improving.\n"
            f"F1 and precision gains are modest but the direction is correct.\n"
            f"Run diagnose_phase2.py on the best variant.\n"
            f"If overlap < 0.30, run tune_threshold v2 — threshold optimisation\n"
            f"may close the remaining F1 gap."
        )
    else:
        decision = "MARGINAL IMPROVEMENT — MIXED SIGNAL"
        action = (
            "Some metrics improved, some did not. Results are ambiguous.\n"
            "Run diagnose_phase2.py on the best variant before deciding next step.\n"
            "Do not move to BoT-IoT fine-tuning yet."
        )

    return {
        "best_variant":     best["label"],
        "f1_gain":          round(f1_gain, 4),
        "precision_gain":   round(precision_gain, 4),
        "fpr_reduction":    round(fpr_drop, 4),
        "fpr_reduction_pct":round(fpr_drop / base["fpr"] * 100, 1),
        "overlap_drop":     round(overlap_drop, 4),
        "decision":         decision,
        "action":           action,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("IoT-SecBand | Phase 2D — Feature Drop Experiment")
    print("Scaler: RobustScaler (fixed from Phase 2C)")
    print("Variable: feature set only")
    print("=" * 60)

    # Load and prepare data up to (not including) normalisation
    print("\nStep 1 — Load and encode raw data")
    df_train_raw = load_parquet(RAW_TRAIN)
    df_test_raw  = load_parquet(RAW_TEST)
    df_train_clean = clean(df_train_raw)
    df_test_clean  = clean(df_test_raw)
    df_train_enc, encoders = encode_categoricals(df_train_clean, fit=True)
    df_test_enc,  _        = encode_categoricals(df_test_clean, encoders=encoders, fit=False)
    log.info(f"Train: {df_train_enc.shape} | Test: {df_test_enc.shape}")

    # Run experiments
    all_results = []
    for name, features in EXPERIMENTS:
        result, _ = run_experiment(name, features, df_train_enc, df_test_enc)
        all_results.append(result)

    # Print comparison table
    print("\n" + "=" * 60)
    print("RESULTS — All experiments (RobustScaler, MLP 32→16)")
    print("=" * 60)

    col_w = 34
    metrics = [
        ("F1 (attack)",        "f1_attack"),
        ("Precision",          "precision"),
        ("Recall (attack)",    "recall_attack"),
        ("ROC-AUC",            "roc_auc"),
        ("FPR (false alarm %)",None),          # special handling
        ("Prob overlap",       "overlap"),
        ("KS statistic",       "ks_stat"),
        ("Meets target",       "meets_target"),
    ]

    header = f"\n{'Metric':<26}" + "".join(
        f"{r['label'][:col_w]:>{col_w}}" for r in all_results
    )
    print(header)
    print("-" * (26 + col_w * len(all_results)))

    for label, key in metrics:
        row = f"  {label:<24}"
        for r in all_results:
            if key is None:   # FPR special case
                val = f"{r['fpr']*100:.1f}%  ({r['normal_flagged_as_attack']:,} flows)"
                row += f"{val:>{col_w}}"
            elif key == "meets_target":
                row += f"{'YES' if r[key] else 'NO':>{col_w}}"
            else:
                row += f"{r[key]:>{col_w}.4f}"
        print(row)

    # Delta rows vs baseline
    print("\n  Deltas vs 13-feature baseline:")
    base = all_results[0]
    for r in all_results[1:]:
        f1d  = r["f1_attack"]  - base["f1_attack"]
        pred = r["precision"]  - base["precision"]
        fprd = base["fpr"]     - r["fpr"]
        ovld = base["overlap"] - r["overlap"]
        print(
            f"  {r['label'][:50]:<50} | "
            f"F1 {f1d:+.4f}  Pre {pred:+.4f}  "
            f"FPR {-fprd:+.4f} ({fprd/base['fpr']*100:.1f}% reduction)  "
            f"Overlap {ovld:+.4f}"
        )

    # Verdict
    v = verdict(all_results)
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"\n  Decision: {v['decision']}")
    print(f"\n  Best variant: {v['best_variant']}")
    print(f"  F1 gain:          {v['f1_gain']:+.4f}")
    print(f"  Precision gain:   {v['precision_gain']:+.4f}")
    print(f"  FPR reduction:    {v['fpr_reduction']:+.4f}  ({v['fpr_reduction_pct']:.1f}%)")
    print(f"  Overlap drop:     {v['overlap_drop']:+.4f}")
    print(f"\n  Action:\n  {v['action']}")

    # Save
    output = {
        "phase":       "2D — Feature Drop Experiment",
        "scaler":      "RobustScaler",
        "experiments": all_results,
        "verdict":     v,
    }
    out_path = METRICS / "phase2d_feature_drop.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()