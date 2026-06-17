"""
src/save_locked_scaler.py
IoT-SecBand — Persist the canonical Phase 2 scaler artifact

Problem fixed:
    models/scaler_params.json (written by run_preprocess.py) was fit on
    preprocess.py's EDGE_FEATURES list, which still has 17 columns —
    including sloss, dloss, spkts, dpkts (dropped in Phase 1 EDA) and
    dload, dinpkt (dropped in Phase 2D). The scaler that actually produced
    the frozen Phase 2 benchmark (F1=0.7829) was fit on exactly 11 features
    inside tune_threshold_v3.py / scaler_experiment.py and was never saved
    to disk — it only existed transiently inside those scripts.

This script:
    1. Re-derives that exact 11-feature RobustScaler from raw UNSW-NB15
       training data (same cleaning/encoding steps as the locked pipeline).
    2. Saves it to models/scaler_params_locked.json — the ONE file Phase 3
       should load when aligning BoT-IoT to this configuration.
    3. Leaves models/scaler_params.json untouched (historical Phase 1
       artifact, 17 features) so nothing else that may reference it breaks,
       but it should NOT be used for Phase 3.

Run from project root:
    python src/save_locked_scaler.py
"""

import sys
import json
import logging
from pathlib import Path
from sklearn.preprocessing import RobustScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocess import load_parquet, clean, encode_categoricals

RAW_TRAIN = ROOT / "data" / "raw" / "unsw_nb15" / "UNSW_NB15_training-set.parquet"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── LOCKED 11-FEATURE SET — must match train.py / tune_threshold_v3.py ──────
LOCKED_FEATURES = [
    "dur", "proto", "service", "state",
    "sbytes", "dbytes", "rate",
    "sload", "sinpkt", "smean", "dmean",
]
assert len(LOCKED_FEATURES) == 11, f"Expected 11 features, got {len(LOCKED_FEATURES)}"


def main():
    print("=" * 60)
    print("IoT-SecBand | Persisting canonical Phase 2 scaler")
    print(f"Locked features ({len(LOCKED_FEATURES)}): {LOCKED_FEATURES}")
    print("=" * 60)

    df_train_raw = load_parquet(RAW_TRAIN)
    df_train_clean = clean(df_train_raw)
    df_train_enc, _ = encode_categoricals(df_train_clean, fit=True)

    missing = [f for f in LOCKED_FEATURES if f not in df_train_enc.columns]
    if missing:
        raise ValueError(f"Missing locked features in cleaned data: {missing}")

    scaler = RobustScaler()
    scaler.fit(df_train_enc[LOCKED_FEATURES])

    scaler_params = {
        "scaler_type":  "RobustScaler",
        "fitted_on":    "UNSW_NB15_training-set.parquet (post-clean, post-dedup, post-encode)",
        "features":     LOCKED_FEATURES,
        "n_features":   len(LOCKED_FEATURES),
        "center":       scaler.center_.tolist(),   # median per feature
        "scale":        scaler.scale_.tolist(),    # IQR per feature
        "usage_note": (
            "This is the canonical scaler for the frozen Phase 2 configuration "
            "(F1=0.7829 on UNSW-NB15 holdout). Apply via scaler.transform() to "
            "BoT-IoT data ALIGNED to these exact 11 feature names, in this order, "
            "WITHOUT refitting. Refitting on BoT-IoT hides distribution shift "
            "instead of exposing it."
        ),
    }

    out_path = MODEL_DIR / "scaler_params_locked.json"
    with open(out_path, "w") as f:
        json.dump(scaler_params, f, indent=2)

    log.info(f"\nSaved: {out_path}")
    log.info(f"Center (median) per feature:")
    for feat, c, s in zip(LOCKED_FEATURES, scaler.center_, scaler.scale_):
        log.info(f"  {feat:<10} center={c:>10.4f}  scale={s:>10.4f}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nUse models/scaler_params_locked.json for Phase 3 BoT-IoT alignment.")
    print(f"Do NOT use models/scaler_params.json (17-feature, pre-Phase-2D artifact).")


if __name__ == "__main__":
    main()