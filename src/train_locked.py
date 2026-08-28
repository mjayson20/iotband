"""
src/train_locked.py
IoT-SecBand — Train and save the locked 11-feature MLP for Phase 3

The mlp_model.pkl produced by Phase 2 notebook uses 13 features.
The locked Phase 2E configuration (F1=0.7829 benchmark) uses 11 features
with RobustScaler — that model was trained inside tune_threshold_v3.py but
never persisted. This script reproduces it exactly and saves it.

Locked config (identical to tune_threshold_v3.py):
    Scaler:   RobustScaler — fit on UNSW-NB15 training data only
    Features: dur, proto, service, state, sbytes, dbytes, rate,
              sload, sinpkt, smean, dmean  (11 features)
    MLP:      hidden_layer_sizes=(32,16), relu, adam, alpha=0.001,
              early_stopping=True, random_state=42

Outputs:
    models/mlp_model_locked.pkl     — the 11-feature MLP for Phase 3
    (scaler_params_locked.json already exists from save_locked_scaler.py)

Run from project root:
    python src/train_locked.py
"""

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, recall_score, roc_auc_score, confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import load_parquet, clean, encode_categoricals
from train import TARGET

PROC_DIR  = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Locked 11-feature set (identical to tune_threshold_v3.py) ────────────────
LOCKED_FEATURES = [
    "dur", "proto", "service", "state",
    "sbytes", "dbytes", "rate",
    "sload", "sinpkt", "smean", "dmean",
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


def main():
    print("=" * 60)
    print("IoT-SecBand | Training locked 11-feature MLP for Phase 3")
    print("=" * 60)

    # ── Load raw UNSW-NB15 data and apply the same pipeline as tune_threshold_v3.py ──
    # We must NOT use the processed parquet — it was StandardScaler-normalized in Phase 1.
    # The locked config uses RobustScaler on the raw (clean + encoded) features.
    raw_train = ROOT / "data" / "raw" / "unsw_nb15" / "UNSW_NB15_training-set.parquet"
    raw_test  = ROOT / "data" / "raw" / "unsw_nb15" / "UNSW_NB15_testing-set.parquet"

    if not raw_train.exists():
        raise FileNotFoundError(
            f"Not found: {raw_train}\n"
            "Place UNSW_NB15_training-set.parquet in data/raw/unsw_nb15/"
        )

    df_train_raw = load_parquet(raw_train)
    df_test_raw  = load_parquet(raw_test)
    df_train_clean = clean(df_train_raw)
    df_test_clean  = clean(df_test_raw)
    df_train_enc, encoders = encode_categoricals(df_train_clean, fit=True)
    df_test_enc, _         = encode_categoricals(df_test_clean, encoders=encoders, fit=False)
    log.info(f"Train: {df_train_enc.shape}  Test: {df_test_enc.shape}")

    X_train = df_train_enc[LOCKED_FEATURES]
    y_train = df_train_enc[TARGET]
    X_test  = df_test_enc[LOCKED_FEATURES]
    y_test  = df_test_enc[TARGET]
    # and stores these exact values. We refit here from the processed data to get
    # a sklearn scaler object for transform(), ensuring identical results.
    log.info("Fitting RobustScaler on training features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    log.info(f"  Scaler fitted on {X_train_scaled.shape[0]:,} rows, {X_train_scaled.shape[1]} features")

    # ── Train MLP ─────────────────────────────────────────────────────────────
    log.info("Training locked MLP(32,16)...")
    mlp = MLPClassifier(**MLP_PARAMS)
    mlp.fit(X_train_scaled, y_train)
    log.info(f"  Training complete — iterations: {mlp.n_iter_}")

    # ── Evaluate at threshold=0.74 (locked benchmark) ─────────────────────────
    y_prob = mlp.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= 0.74).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    f1  = f1_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    auc = roc_auc_score(y_test, y_prob)

    log.info(f"\n── Verification at threshold=0.74 ──")
    log.info(f"  F1={f1:.4f}  Recall={rec:.4f}  FPR={fpr:.4f}  AUC={auc:.4f}")
    log.info(f"  Phase 2E reference: F1=0.7829  Recall=0.8303  FPR=0.1474")

    # Accept a small tolerance — the model is trained on the full test set
    # here (not the 50/50 split used in Phase 2E), so F1 will differ slightly.
    tol = 0.03
    ref_f1 = 0.7299  # Phase 2E default-threshold F1 on full test set
    if abs(f1_score(y_test, (y_prob >= 0.50).astype(int), pos_label=1) - ref_f1) > tol:
        log.warning(
            f"F1@0.50 differs from Phase 2E reference by >{tol:.0%}. "
            "This may indicate a data mismatch — verify processed parquet files."
        )
    else:
        log.info("  ✓ F1@0.50 matches Phase 2E reference within tolerance")

    # ── Save model ────────────────────────────────────────────────────────────
    out_path = MODEL_DIR / "mlp_model_locked.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(mlp, f)
    log.info(f"\n✓ Saved: {out_path}")

    # Also save the scaler so botiot_evaluate.py can use it directly
    scaler_pkl = MODEL_DIR / "scaler_locked.pkl"
    with open(scaler_pkl, "wb") as f:
        pickle.dump(scaler, f)
    log.info(f"✓ Saved: {scaler_pkl}")

    print("\n── Done ──")
    print(f"  mlp_model_locked.pkl  — use this for Phase 3 evaluation")
    print(f"  scaler_locked.pkl     — apply to BoT-IoT before inference")
    print(f"  F1@0.74 = {f1:.4f}  (full test set, vs 0.7829 on 50% holdout)")


if __name__ == "__main__":
    main()
