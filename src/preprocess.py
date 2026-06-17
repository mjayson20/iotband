"""
src/preprocess.py
IoT-SecBand — UNSW-NB15 preprocessing module

Pipeline position: UNSW preprocessing → feature selection → model training
Input:  data/raw/unsw_nb15/{train,test}.parquet
Output: data/processed/unsw_{train,test}_clean.parquet

All logic is importable. No side effects on import.
Notebooks call these functions; they do not duplicate this logic.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def _to_json_serializable(obj):
    """Recursively convert common numpy / Path types to Python built-ins for JSON.

    This is conservative and safe for downstream callers that expect native types
    (e.g. notebooks that call `json.dump(summary, ...)`).
    """
    import numpy as _np
    from pathlib import Path as _Path

    # Dicts, lists, tuples
    if isinstance(obj, dict):
        return { _to_json_serializable(k): _to_json_serializable(v) for k, v in obj.items() }
    if isinstance(obj, list):
        return [ _to_json_serializable(i) for i in obj ]
    if isinstance(obj, tuple):
        return [ _to_json_serializable(i) for i in obj ]

    # Numpy scalar types -> native
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)

    # Path -> str
    if isinstance(obj, _Path):
        return str(obj)

    # Objects exposing tolist (e.g., numpy arrays)
    if hasattr(obj, "tolist"):
        try:
            return _to_json_serializable(obj.tolist())
        except Exception:
            pass

    return obj

# ─────────────────────────────────────────────
# 1. FEATURE SETS
# ─────────────────────────────────────────────

# Justified by: directly derivable from ESP32/nRF52840 header-level capture.
# No payload inspection required.
EDGE_FEATURES = [
    "dur",        # flow duration (s)
    "proto",      # protocol — categorical, encode
    "service",    # inferred from port — categorical, encode
    "state",      # TCP/UDP state flags — categorical, encode
    "spkts",      # source packet count
    "dpkts",      # destination packet count
    "sbytes",     # source byte total
    "dbytes",     # destination byte total
    "rate",       # packets/sec
    "sload",      # source bits/sec
    "dload",      # destination bits/sec
    "sloss",      # source packet loss count
    "dloss",      # destination packet loss count
    "sinpkt",     # source inter-packet interval (ms)
    "dinpkt",     # destination inter-packet interval (ms)
    "smean",      # mean source packet size — keep: running avg feasible on MCU
    "dmean",      # mean dest packet size — same
]

# Dropped: require deep packet inspection, application-layer parsing,
# persistent multi-flow lookup tables, or are FTP/HTTP specific.
DROPPED_FEATURES = [
    "stcpb", "dtcpb",            # TCP sequence numbers — high cardinality, no IDS signal
    "swin", "dwin",              # TCP window — only meaningful over TCP, adds branching
    "sjit", "djit",              # Jitter — expensive rolling stddev on MCU
    "tcprtt", "synack", "ackdat",# TCP timing — requires paired packet state machine
    "trans_depth",               # HTTP transaction depth — payload inspection
    "response_body_len",         # HTTP body length — deep packet inspection
    "ct_src_dport_ltm",          # Connection count per dst port — persistent RAM table
    "ct_dst_sport_ltm",          # Connection count per src port — same
    "is_ftp_login",              # FTP-specific — irrelevant to BLE/WiFi wearable
    "ct_ftp_cmd",                # FTP command count — same
    "ct_flw_http_mthd",          # HTTP method count — application-layer
    "is_sm_ips_ports",           # Same IP/port flag — marginal, high compute
    "attack_cat",                # Multi-class label — not used (binary only)
]

CATEGORICAL_FEATURES = ["proto", "service", "state"]
TARGET_COLUMN = "label"

# ─────────────────────────────────────────────
# 2. LOAD
# ─────────────────────────────────────────────

def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a parquet file and log shape + class distribution."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_parquet(path)
    log.info(f"Loaded: {path.name} — shape={df.shape}")
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts()
        log.info(f"Class distribution:\n{counts.to_string()}")
    return df


# ─────────────────────────────────────────────
# 3. AUDIT
# ─────────────────────────────────────────────

def audit_columns(df: pd.DataFrame) -> dict:
    """
    Audit raw columns against expected feature sets.
    Returns a report dict with missing/unexpected columns.
    """
    present = set(df.columns)
    expected = set(EDGE_FEATURES + [TARGET_COLUMN])
    dropped  = set(DROPPED_FEATURES)

    report = {
        "present_expected":   sorted(present & expected),
        "missing_expected":   sorted(expected - present),
        "present_dropped":    sorted(present & dropped),
        "unexpected_columns": sorted(present - expected - dropped),
        "total_columns":      len(df.columns),
    }

    if report["missing_expected"]:
        log.warning(f"Missing expected features: {report['missing_expected']}")
    if report["unexpected_columns"]:
        log.info(f"Unrecognised columns (will be dropped): {report['unexpected_columns']}")

    return report


# ─────────────────────────────────────────────
# 4. CLEAN
# ─────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    1. Select edge-justified features + target only.
    2. Drop duplicates.
    3. Handle nulls — numeric: median; categorical: mode.
    4. Clip inf values.
    5. Cast target to int.
    """
    # Select only what we need
    cols_to_keep = [c for c in EDGE_FEATURES + [TARGET_COLUMN] if c in df.columns]
    df = df[cols_to_keep].copy()
    log.info(f"Selected {len(cols_to_keep)} columns (incl. target)")

    # Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    log.info(f"Dropped {before - len(df)} duplicate rows")

    # Clip infinities before null handling
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Null handling
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if not null_cols.empty:
        log.info(f"Null values found:\n{null_cols.to_string()}")
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                log.info(f"  {col}: filled {null_cols.get(col, 0)} nulls with median={median_val:.4f}")
        for col in CATEGORICAL_FEATURES:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
                log.info(f"  {col}: filled nulls with mode='{mode_val}'")
    else:
        log.info("No null values found")

    # Cast target
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    log.info(f"Clean complete — shape={df.shape}")
    return df


# ─────────────────────────────────────────────
# 5. ENCODE CATEGORICALS
# ─────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    encoders: dict | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical features.

    Args:
        df:       DataFrame (post-clean).
        encoders: Existing encoder dict (pass from training when encoding test set).
        fit:      If True, fit new encoders. If False, transform only using provided encoders.

    Returns:
        (encoded_df, encoders_dict)

    Usage:
        # Training set:
        df_train, encoders = encode_categoricals(df_train, fit=True)
        # Test set (must use same encoders):
        df_test, _ = encode_categoricals(df_test, encoders=encoders, fit=False)
    """
    if encoders is None:
        encoders = {}

    df = df.copy()

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            log.warning(f"Categorical column '{col}' not found — skipping")
            continue

        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            log.info(f"Encoded '{col}' — {len(le.classes_)} classes: {list(le.classes_[:8])}")
        else:
            if col not in encoders:
                raise ValueError(f"No encoder found for '{col}'. Was fit=True used on training set?")
            le = encoders[col]
            # Handle unseen categories by mapping to the most frequent class (index 0)
            known = set(le.classes_)
            unseen = set(df[col].astype(str).unique()) - known
            if unseen:
                log.warning(f"'{col}' has {len(unseen)} unseen categories — mapping to class 0")
            df[col] = df[col].astype(str).apply(
                lambda x: le.transform([x])[0] if x in known else 0
            )

    return df, encoders


# ─────────────────────────────────────────────
# 6. NORMALIZE
# ─────────────────────────────────────────────

def normalize(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Z-score normalize all numeric features except target.

    IMPORTANT for cross-dataset use: fit scaler on UNSW-NB15 train only.
    Apply the SAME scaler to BoT-IoT without refitting.
    This forces BoT-IoT into UNSW-NB15's value space, exposing distribution shift
    as a detectable signal rather than hiding it.

    Returns:
        (normalized_df, scaler)
    """
    df = df.copy()
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c != TARGET_COLUMN
    ]

    if fit:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        log.info(f"Fitted StandardScaler on {len(numeric_cols)} numeric features")
    else:
        if scaler is None:
            raise ValueError("Scaler required when fit=False. Pass scaler from training set.")
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        log.info(f"Applied existing scaler to {len(numeric_cols)} numeric features")

    return df, scaler


# ─────────────────────────────────────────────
# 7. CLASS BALANCE REPORT
# ─────────────────────────────────────────────

def class_balance_report(df: pd.DataFrame) -> dict:
    """
    Report class distribution and imbalance ratio.
    Imbalance ratio > 3:1 → recommend SMOTE on training set only.
    """
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    total  = len(df)
    ratio  = counts.max() / counts.min() if counts.min() > 0 else float("inf")

    report = {
        "counts":          {int(k): int(v) for k, v in counts.to_dict().items()},
        "percentages":     {int(k): float(round(v, 2)) for k, v in (counts / total * 100).to_dict().items()},
        "imbalance_ratio": float(round(ratio, 2)),
        "recommend_smote": bool(ratio > 3.0),
    }

    log.info(f"Class balance — counts: {report['counts']}, ratio: {ratio:.2f}")
    if report["recommend_smote"]:
        log.warning("Imbalance ratio > 3:1 — apply SMOTE on training set only, never on test/validation")
    return report


# ─────────────────────────────────────────────
# 8. FULL PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(
    train_path: str | Path,
    test_path:  str | Path,
    output_dir: str | Path,
    artifacts_dir: str | Path = Path("models"),
) -> dict:
    """
    Execute full UNSW-NB15 preprocessing pipeline.

    Reads:
        data/raw/unsw_nb15/train.parquet
        data/raw/unsw_nb15/test.parquet

    Writes:
        data/processed/unsw_train_clean.parquet
        data/processed/unsw_test_clean.parquet
        models/label_encoders.json   (class lists for each categorical)
        models/scaler_params.json    (mean + scale for each numeric feature)

    Returns:
        dict with paths, shapes, balance report, and artifact paths
    """
    output_dir    = Path(output_dir)
    artifacts_dir = Path(artifacts_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────
    log.info("=" * 50)
    log.info("STEP 1/6 — Load raw data")
    df_train = load_parquet(train_path)
    df_test  = load_parquet(test_path)

    # ── Audit ─────────────────────────────────
    log.info("=" * 50)
    log.info("STEP 2/6 — Audit columns")
    audit_report = audit_columns(df_train)
    log.info(f"Audit: {len(audit_report['present_expected'])} expected features present")

    # ── Clean ─────────────────────────────────
    log.info("=" * 50)
    log.info("STEP 3/6 — Clean")
    df_train = clean(df_train)
    df_test  = clean(df_test)

    # ── Class balance ─────────────────────────
    log.info("=" * 50)
    log.info("STEP 4/6 — Class balance")
    balance = class_balance_report(df_train)

    # ── Encode ────────────────────────────────
    log.info("=" * 50)
    log.info("STEP 5/6 — Encode categoricals")
    df_train, encoders = encode_categoricals(df_train, fit=True)
    df_test,  _        = encode_categoricals(df_test, encoders=encoders, fit=False)

    # ── Normalize ─────────────────────────────
    log.info("=" * 50)
    log.info("STEP 6/6 — Normalize")
    df_train, scaler = normalize(df_train, fit=True)
    df_test,  _      = normalize(df_test, scaler=scaler, fit=False)

    # ── Save processed data ───────────────────
    train_out = output_dir / "unsw_train_clean.parquet"
    test_out  = output_dir / "unsw_test_clean.parquet"
    df_train.to_parquet(train_out, index=False)
    df_test.to_parquet(test_out, index=False)
    log.info(f"Saved: {train_out}")
    log.info(f"Saved: {test_out}")

    # ── Save scaler params (for BoT-IoT alignment later) ─────
    numeric_cols = [c for c in df_train.columns if c != TARGET_COLUMN
                    and df_train[c].dtype != object]
    scaler_params = {
        "features": numeric_cols,
        "mean":     scaler.mean_.tolist(),
        "scale":    scaler.scale_.tolist(),
    }
    scaler_path = artifacts_dir / "scaler_params.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler_params, f, indent=2)
    log.info(f"Saved scaler params: {scaler_path}")

    # ── Save encoder class lists ──────────────
    encoder_classes = {col: le.classes_.tolist() for col, le in encoders.items()}
    encoder_path = artifacts_dir / "label_encoders.json"
    with open(encoder_path, "w") as f:
        json.dump(encoder_classes, f, indent=2)
    log.info(f"Saved encoder classes: {encoder_path}")

    result = {
        "train_output":    str(train_out),
        "test_output":     str(test_out),
        "train_shape":     df_train.shape,
        "test_shape":      df_test.shape,
        "features_used":   [c for c in df_train.columns if c != TARGET_COLUMN],
        "balance_report":  balance,
        "scaler_path":     str(scaler_path),
        "encoder_path":    str(encoder_path),
        "audit":           audit_report,
    }

    # Ensure any numpy / Path scalars are converted to native Python types
    return _to_json_serializable(result)
