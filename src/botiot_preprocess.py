"""
src/botiot_preprocess.py
IoT-SecBand — NF-BoT-IoT preprocessing module (Phase 3)

Pipeline position: UNSW training → [frozen model] → BoT-IoT alignment → evaluation

Purpose:
    Align NF-BoT-IoT's NetFlow schema to the 11 locked UNSW-NB15 features,
    apply the UNSW-NB15-fitted RobustScaler and label encoders WITHOUT refitting,
    and produce a processed parquet file ready for evaluation.

Key rules enforced here:
    - NEVER refit the scaler on BoT-IoT. Load locked center/scale from
      models/scaler_params_locked.json and apply directly.
    - NEVER refit the label encoders. Load classes from models/label_encoders.json
      and map unseen categories to class 0.
    - Missing features (service, state) are filled with the UNSW-NB15 training
      mode value, clearly documented as a deliberate approximation.
    - The label column is binarised: 0=Normal, 1=Attack.

NF-BoT-IoT raw columns:
    L4_SRC_PORT, L4_DST_PORT, PROTOCOL, L7_PROTO
    IN_BYTES, OUT_BYTES, IN_PKTS, OUT_PKTS
    TCP_FLAGS, FLOW_DURATION_MILLISECONDS
    Label (0/1), Attack (category string)

Locked UNSW-NB15 feature set (11):
    dur, proto, service, state, sbytes, dbytes,
    rate, sload, sinpkt, smean, dmean

Column mapping:
    FLOW_DURATION_MILLISECONDS / 1000  -> dur      (ms -> s)
    PROTOCOL                           -> proto     (numeric, map via label encoder)
    IN_BYTES                           -> sbytes
    OUT_BYTES                          -> dbytes
    IN_BYTES / OUT_PKTS (approx)       -> smean     (mean src pkt size)
    OUT_BYTES / OUT_PKTS (approx)      -> dmean     (mean dst pkt size)
    IN_BYTES * 8 / dur                 -> sload     (src bits/sec)
    IN_PKTS / dur                      -> rate      (pkts/sec, approx)
    dur / IN_PKTS                      -> sinpkt    (inter-pkt interval ms)
    service                            -> MISSING — filled with UNSW-NB15 mode
    state                              -> MISSING — filled with UNSW-NB15 mode

Run from project root:
    python src/botiot_preprocess.py

Outputs:
    data/processed/botiot_clean.parquet
    outputs/metrics/phase3_alignment.json
"""

import json
import logging
import pickle
import sys
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

RAW_PATH    = ROOT / "data" / "raw" / "unsw_nb15" / "NF-BoT-IoT.parquet"
OUTPUT_PATH = ROOT / "data" / "processed" / "botiot_clean.parquet"
SCALER_PATH = ROOT / "models" / "scaler_params_locked.json"
ENCODER_PATH = ROOT / "models" / "label_encoders.json"
METRICS_DIR  = ROOT / "outputs" / "metrics"

# Locked feature order — must match scaler exactly
LOCKED_FEATURES = [
    "dur", "proto", "service", "state",
    "sbytes", "dbytes", "rate", "sload",
    "sinpkt", "smean", "dmean",
]

TARGET = "label"

# UNSW-NB15 training mode values for missing features
# (proto=113 is 'tcp' in the encoder, service=0 is '-', state=2 is 'FIN')
# These are the values that appear most in UNSW-NB15 training data.
UNSW_SERVICE_MODE = "-"    # most common service in UNSW-NB15 train
UNSW_STATE_MODE   = "FIN"  # most common state in UNSW-NB15 train


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_botiot(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    log.info(f"Loaded {path.name}: shape={df.shape}")
    log.info(f"  Columns: {list(df.columns)}")
    log.info(f"  Label dist: {df['Label'].value_counts().to_dict()}")
    log.info(f"  Attack dist: {df['Attack'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING — derive UNSW-like features from NetFlow columns
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive the 11 locked UNSW-NB15 features from NF-BoT-IoT's NetFlow columns.
    All derived columns use safe division (avoid div-by-zero with small epsilon).
    """
    df = df.copy()
    eps = 1e-9  # avoid division by zero

    # --- dur: flow duration in seconds ---
    # FLOW_DURATION_MILLISECONDS is in ms; UNSW-NB15 dur is in seconds
    df["dur"] = df["FLOW_DURATION_MILLISECONDS"] / 1000.0
    df["dur"] = df["dur"].clip(lower=0)

    # --- proto: use PROTOCOL field directly ---
    # NF-BoT-IoT uses numeric IANA protocol numbers (6=TCP, 17=UDP, 1=ICMP)
    # UNSW-NB15 used string proto names then label-encoded.
    # We map numeric -> string using IANA names matching UNSW-NB15's encoder classes,
    # then fall back to 'other' (mapped to class 0 as unseen category).
    IANA_MAP = {
        1: "icmp", 6: "tcp", 17: "udp", 41: "ipv6",
        47: "gre", 50: "esp", 51: "ah", 58: "ipv6",
        132: "sctp",
    }
    df["proto_str"] = df["PROTOCOL"].map(IANA_MAP).fillna("other")

    # --- service: not available in NetFlow — fill with UNSW-NB15 mode ---
    # Document clearly: this is an approximation. All flows get service='-'
    # which was the most common value in UNSW training data.
    df["service"] = UNSW_SERVICE_MODE
    log.info(f"  'service' not in NF-BoT-IoT — filled with UNSW mode: '{UNSW_SERVICE_MODE}'")

    # --- state: not available in NetFlow — fill with UNSW-NB15 mode ---
    df["state"] = UNSW_STATE_MODE
    log.info(f"  'state' not in NF-BoT-IoT — filled with UNSW mode: '{UNSW_STATE_MODE}'")

    # --- sbytes / dbytes ---
    df["sbytes"] = df["IN_BYTES"].clip(lower=0).astype(float)
    df["dbytes"] = df["OUT_BYTES"].clip(lower=0).astype(float)

    # --- rate: incoming packets per second ---
    df["rate"] = df["IN_PKTS"] / (df["dur"] + eps)
    df["rate"] = df["rate"].clip(lower=0)

    # --- sload: source bits per second ---
    df["sload"] = (df["sbytes"] * 8) / (df["dur"] + eps)
    df["sload"] = df["sload"].clip(lower=0)

    # --- sinpkt: mean inter-packet interval in ms (source direction) ---
    # UNSW sinpkt = dur(s) / spkts * 1000  (ms between source packets)
    df["sinpkt"] = (df["dur"] * 1000) / (df["IN_PKTS"].clip(lower=1) + eps)
    df["sinpkt"] = df["sinpkt"].clip(lower=0)

    # --- smean: mean source packet size in bytes ---
    df["smean"] = df["sbytes"] / (df["IN_PKTS"].clip(lower=1) + eps)
    df["smean"] = df["smean"].clip(lower=0)

    # --- dmean: mean destination packet size in bytes ---
    df["dmean"] = df["dbytes"] / (df["OUT_PKTS"].clip(lower=1) + eps)
    df["dmean"] = df["dmean"].clip(lower=0)

    # --- label: binarise (Label column is already 0/1) ---
    df[TARGET] = df["Label"].astype(int)

    log.info("Feature engineering complete.")
    log.info(f"  Derived: dur, proto_str, sbytes, dbytes, rate, sload, sinpkt, smean, dmean")
    log.info(f"  Filled:  service='{UNSW_SERVICE_MODE}', state='{UNSW_STATE_MODE}'")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(
        subset=["dur", "sbytes", "dbytes", "rate", "smean", "dmean", TARGET]
    )
    dropped = before - len(df)
    log.info(f"Deduplication: {before:,} -> {len(df):,} rows (dropped {dropped:,})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENCODE CATEGORICALS using locked UNSW-NB15 encoders
# ─────────────────────────────────────────────────────────────────────────────

def apply_locked_encoders(df: pd.DataFrame, encoder_path: Path) -> pd.DataFrame:
    """
    Apply the UNSW-NB15 label encoders to proto, service, state.
    Unseen categories map to class 0 — matches Phase 1 design.
    Does NOT refit.
    """
    with open(encoder_path) as f:
        encoder_classes = json.load(f)

    df = df.copy()

    # proto: map string name -> integer using UNSW-NB15 encoder classes
    proto_classes = encoder_classes["proto"]
    proto_map = {name: idx for idx, name in enumerate(proto_classes)}
    unseen_proto = set(df["proto_str"].unique()) - set(proto_classes)
    if unseen_proto:
        log.warning(f"  proto: {len(unseen_proto)} unseen values -> class 0: {list(unseen_proto)[:8]}")
    df["proto"] = df["proto_str"].map(proto_map).fillna(0).astype(int)

    # service: all values are UNSW_SERVICE_MODE ('-')
    service_classes = encoder_classes["service"]
    service_map = {name: idx for idx, name in enumerate(service_classes)}
    df["service"] = df["service"].map(service_map).fillna(0).astype(int)

    # state: all values are UNSW_STATE_MODE ('FIN')
    state_classes = encoder_classes["state"]
    state_map = {name: idx for idx, name in enumerate(state_classes)}
    df["state"] = df["state"].map(state_map).fillna(0).astype(int)

    log.info(f"  proto  encoded: {df['proto'].value_counts().head(5).to_dict()}")
    log.info(f"  service encoded: {df['service'].value_counts().to_dict()}")
    log.info(f"  state  encoded: {df['state'].value_counts().to_dict()}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. APPLY LOCKED SCALER (no refitting)
# ─────────────────────────────────────────────────────────────────────────────

def apply_locked_scaler(df: pd.DataFrame, scaler_path: Path) -> pd.DataFrame:
    """
    Apply the UNSW-NB15-fitted RobustScaler to the 11 locked features.
    Formula: scaled = (x - center) / scale

    Uses scaler_params_locked.json which was fit on raw UNSW-NB15 training
    features (pre-StandardScaler, post-clean/encode) by save_locked_scaler.py.

    This intentionally exposes distribution shift rather than hiding it.
    """
    with open(scaler_path) as f:
        scaler_params = json.load(f)

    features = scaler_params["features"]
    center   = np.array(scaler_params["center"])
    scale    = np.array(scaler_params["scale"])

    assert features == LOCKED_FEATURES, (
        f"Scaler feature order mismatch!\n"
        f"  Expected: {LOCKED_FEATURES}\n  Got: {features}"
    )

    df = df.copy()
    X_scaled = (df[LOCKED_FEATURES].values.astype(float) - center) / scale
    df[LOCKED_FEATURES] = X_scaled

    log.info(f"Applied locked RobustScaler (JSON) to {len(features)} features.")
    log.info("  Post-scale range check (expect ~[-5, 5] for well-aligned data):")
    for i, feat in enumerate(features):
        col = X_scaled[:, i]
        flag = " ⚠ HIGH DRIFT" if (col.max() > 20 or col.min() < -20) else ""
        log.info(f"    {feat:10s}  min={col.min():8.2f}  max={col.max():8.2f}"
                 f"  median={np.median(col):6.2f}{flag}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_botiot_pipeline(
    raw_path:     Path = RAW_PATH,
    output_path:  Path = OUTPUT_PATH,
    scaler_path:  Path = SCALER_PATH,
    encoder_path: Path = ENCODER_PATH,
) -> dict:
    """
    Full NF-BoT-IoT preprocessing pipeline.

    Reads:  data/raw/unsw_nb15/NF-BoT-IoT.parquet
    Writes: data/processed/botiot_clean.parquet

    Returns alignment report dict.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info("IoT-SecBand | Phase 3 — NF-BoT-IoT Preprocessing")
    log.info("=" * 60)

    # Step 1: Load
    df = load_botiot(raw_path)
    raw_shape = df.shape

    # Step 2: Feature engineering
    log.info("\n── Step 1: Feature Engineering ──")
    df = engineer_features(df)

    # Step 3: Select only needed columns
    # At this point proto is still 'proto_str'; the locked encoder step renames it.
    # Build the list using the engineered names before encoding.
    pre_encode_cols = [
        "dur", "proto_str", "service", "state",
        "sbytes", "dbytes", "rate", "sload",
        "sinpkt", "smean", "dmean", TARGET,
    ]
    df = df[pre_encode_cols].copy()

    # Step 4: Handle nulls / infinities
    log.info("\n── Step 2: Null / Inf handling ──")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    null_counts = df.isnull().sum()
    if null_counts.any():
        log.warning(f"Null values after engineering:\n{null_counts[null_counts > 0].to_string()}")
        for col in LOCKED_FEATURES:
            if df[col].isnull().any():
                fill = df[col].median()
                df[col].fillna(fill, inplace=True)
                log.info(f"  {col}: filled nulls with median={fill:.4f}")
    else:
        log.info("  No nulls found.")

    # Step 5: Deduplicate
    log.info("\n── Step 3: Deduplication ──")
    df = deduplicate(df)

    # Step 6: Class balance report
    log.info("\n── Step 4: Class Balance ──")
    label_counts = df[TARGET].value_counts().to_dict()
    n_normal = label_counts.get(0, 0)
    n_attack = label_counts.get(1, 0)
    imbalance_ratio = (n_attack / n_normal) if n_normal > 0 else float("inf")
    log.info(f"  Normal: {n_normal:,}  Attack: {n_attack:,}  Ratio: {imbalance_ratio:.1f}:1")
    if imbalance_ratio > 10:
        pct_attack = n_attack / (n_normal + n_attack) * 100
        log.warning(
            f"  ⚠ Extreme imbalance ({imbalance_ratio:.0f}:1). "
            f"A classifier predicting all-attack gets ~{pct_attack:.1f}% accuracy. "
            "Use F1/recall, never raw accuracy."
        )

    # Step 7: Encode categoricals with locked UNSW encoders
    log.info("\n── Step 5: Categorical Encoding (locked UNSW-NB15 encoders) ──")
    df = apply_locked_encoders(df, encoder_path)

    # Step 7: Apply locked scaler
    # NOTE: Scaling is intentionally deferred to botiot_evaluate.py, which uses
    # scaler_locked.pkl directly. This keeps the saved parquet in interpretable
    # (unscaled) units for inspection, and ensures a single source of truth for scaling.
    log.info("\n── Step 6: Scaling — deferred to evaluation step ──")
    log.info("  The saved parquet contains unscaled (raw-engineered + encoded) features.")
    log.info("  botiot_evaluate.py applies scaler_locked.pkl before inference.")

    # Save
    df.to_parquet(output_path, index=False)
    log.info(f"\n✓ Saved: {output_path}  shape={df.shape}")

    # Alignment report
    report = {
        "raw_shape":             list(raw_shape),
        "processed_shape":       list(df.shape),
        "n_normal":              int(n_normal),
        "n_attack":              int(n_attack),
        "imbalance_ratio":       round(float(imbalance_ratio), 2),
        "features_derived":      ["dur", "sbytes", "dbytes", "rate", "sload", "sinpkt", "smean", "dmean"],
        "features_approx":       ["proto"],
        "features_missing_filled": {
            "service": f"filled with UNSW-NB15 mode='{UNSW_SERVICE_MODE}'",
            "state":   f"filled with UNSW-NB15 mode='{UNSW_STATE_MODE}'",
        },
        "scaler":  "NOT applied here — scaler_locked.pkl applied in botiot_evaluate.py before inference",
        "encoders": "locked UNSW-NB15 label encoders — NOT refit on BoT-IoT",
        "output_path": str(output_path),
    }

    report_path = METRICS_DIR / "phase3_alignment.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"✓ Alignment report: {report_path}")

    return report


if __name__ == "__main__":
    report = run_botiot_pipeline()
    print("\n── Alignment Report ──")
    print(json.dumps(report, indent=2))
