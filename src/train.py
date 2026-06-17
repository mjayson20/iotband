"""
src/train.py
IoT-SecBand — Model training module

Pipeline position: UNSW preprocessing → feature selection → model training
Input:  data/processed/unsw_train_clean.parquet
        data/processed/unsw_test_clean.parquet
Output: models/dt_model.pkl
        models/rf_model.pkl
        models/mlp_model.pkl
        outputs/metrics/phase2_results.json

Feature set locked at Phase 1 output: 13 edge-justified features.
No payload-derived features. No SMOTE (imbalance ratio 1.14 — not required).
"""

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, roc_auc_score,
    classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# LOCKED FEATURE SET — Phase 1 output
# DO NOT modify without re-running Phase 1
# ─────────────────────────────────────────────

FINAL_FEATURES = [
    "dur",
    "proto",
    "service",
    "state",
    "sbytes",
    "dbytes",
    "rate",
    "sload",
    "dload",
    "sinpkt",
    "dinpkt",
    "smean",
    "dmean",
]

TARGET = "label"

# ─────────────────────────────────────────────
# MODEL DEFINITIONS
# Ordered by complexity: DT → RF → MLP
# Start with DT to establish a fast interpretable baseline.
# MLP is the TinyML candidate — it must pass size + latency checks.
# ─────────────────────────────────────────────

def get_models() -> dict:
    """
    Return model definitions with TinyML-aware hyperparameters.

    Decision Tree:
      max_depth=10 — prevents overfitting; deeper trees overfit UNSW-NB15 noise.
      Easily converted to rule-based firmware logic if needed.

    Random Forest:
      n_estimators=50 — balance between accuracy and ensemble size.
      max_depth=10 — same constraint as DT.
      NOTE: RF cannot be deployed to nRF52840 directly.
      Use as accuracy ceiling benchmark only.

    MLP:
      Two hidden layers (32, 16) — fits within ~200KB INT8 quantized.
      relu activation — fast on Cortex-M4, no exp() computation.
      max_iter=200 — sufficient for convergence on 89K samples.
      This is the TinyML deployment candidate.
    """
    return {
        "decision_tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "mlp": MLPClassifier(
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
        ),
    }


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_splits(
    train_path: str | Path,
    test_path:  str | Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load processed parquet files and select final 13-feature set.

    Returns:
        X_train, y_train, X_test, y_test
    """
    train_path = Path(train_path)
    test_path  = Path(test_path)

    df_train = pd.read_parquet(train_path)
    df_test  = pd.read_parquet(test_path)
    log.info(f"Train loaded: {df_train.shape} | Test loaded: {df_test.shape}")

    # Validate all required features are present
    missing = [f for f in FINAL_FEATURES if f not in df_train.columns]
    if missing:
        raise ValueError(
            f"Missing features in training data: {missing}\n"
            f"Available: {list(df_train.columns)}"
        )

    X_train = df_train[FINAL_FEATURES]
    y_train = df_train[TARGET]
    X_test  = df_test[FINAL_FEATURES]
    y_test  = df_test[TARGET]

    log.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    log.info(f"Train label dist: {y_train.value_counts().to_dict()}")
    log.info(f"Test  label dist: {y_test.value_counts().to_dict()}")

    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """
    Full evaluation on held-out test set.

    Metrics reported:
    - Accuracy       — overall correctness (misleading if imbalanced)
    - F1 (attack)    — primary metric: harmonic mean of precision/recall on attack class
    - Precision      — of predicted attacks, how many are real attacks
    - Recall         — of real attacks, how many did we catch
    - ROC-AUC        — discrimination ability across thresholds
    - Confusion matrix

    For IDS systems: Recall (attack) is more important than Precision.
    A missed attack (false negative) is worse than a false alarm.
    Target: Recall >= 0.88, F1 >= 0.88.
    """
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    acc       = float(accuracy_score(y_test, y_pred))
    f1        = float(f1_score(y_test, y_pred, pos_label=1))
    precision = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    recall    = float(recall_score(y_test, y_pred, pos_label=1))
    cm        = confusion_matrix(y_test, y_pred).tolist()
    auc       = float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    result = {
        "model":         model_name,
        "accuracy":      round(acc, 4),
        "f1_attack":     round(f1, 4),
        "precision":     round(precision, 4),
        "recall_attack": round(recall, 4),
        "roc_auc":       round(auc, 4) if auc else None,
        "confusion_matrix": cm,
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "false_negative_rate": round(float(fn / (fn + tp)), 4) if (fn + tp) > 0 else 0,
        "meets_target": bool(f1 >= 0.88 and recall >= 0.88),
    }

    formatted_auc = f"{auc:.4f}" if auc is not None else "N/A"
    log.info(f"{model_name} | acc={acc:.4f} f1={f1:.4f} recall={recall:.4f} auc={formatted_auc}")
    if not result["meets_target"]:
        log.warning(f"{model_name} does not meet F1>=0.88 and Recall>=0.88 targets")

    return result


def cross_validate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    cv_folds: int = 5,
) -> dict:
    """
    Stratified 5-fold cross-validation on training set.
    Detects overfitting: if CV score << test score, model memorised training data.
    Reports mean ± std for accuracy and F1.
    """
    log.info(f"Running {cv_folds}-fold CV for {model_name}...")
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        scoring=["accuracy", "f1"],
        return_train_score=True,
        n_jobs=-1,
    )

    result = {
        "cv_accuracy_mean":    round(float(cv_results["test_accuracy"].mean()), 4),
        "cv_accuracy_std":     round(float(cv_results["test_accuracy"].std()), 4),
        "cv_f1_mean":          round(float(cv_results["test_f1"].mean()), 4),
        "cv_f1_std":           round(float(cv_results["test_f1"].std()), 4),
        "train_accuracy_mean": round(float(cv_results["train_accuracy"].mean()), 4),
        "train_f1_mean":       round(float(cv_results["train_f1"].mean()), 4),
        "overfit_gap_f1":      round(
            float(cv_results["train_f1"].mean() - cv_results["test_f1"].mean()), 4
        ),
    }

    log.info(
        f"{model_name} CV | "
        f"acc={result['cv_accuracy_mean']:.4f}±{result['cv_accuracy_std']:.4f} "
        f"f1={result['cv_f1_mean']:.4f}±{result['cv_f1_std']:.4f} "
        f"overfit_gap={result['overfit_gap_f1']:.4f}"
    )
    if result["overfit_gap_f1"] > 0.05:
        log.warning(f"{model_name}: overfit gap > 0.05 — model may not generalise to BoT-IoT")

    return result


# ─────────────────────────────────────────────
# MODEL PERSISTENCE
# ─────────────────────────────────────────────

def save_model(model, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    log.info(f"Model saved: {path}")


def load_model(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    log.info(f"Model loaded: {path}")
    return model


# ─────────────────────────────────────────────
# FULL TRAINING PIPELINE
# ─────────────────────────────────────────────

def run_training(
    train_path:   str | Path,
    test_path:    str | Path,
    models_dir:   str | Path,
    metrics_dir:  str | Path,
) -> dict:
    """
    Train all three models, evaluate, save.

    Reads:
        data/processed/unsw_train_clean.parquet
        data/processed/unsw_test_clean.parquet

    Writes:
        models/dt_model.pkl
        models/rf_model.pkl
        models/mlp_model.pkl
        outputs/metrics/phase2_results.json
    """
    models_dir  = Path(models_dir)
    metrics_dir = Path(metrics_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, y_train, X_test, y_test = load_splits(train_path, test_path)

    models      = get_models()
    all_results = {}

    for name, model in models.items():
        log.info("=" * 55)
        log.info(f"Training: {name}")
        log.info("=" * 55)

        # Cross-validate first — detect overfit before test evaluation
        cv_result = cross_validate_model(model, X_train, y_train, name)

        # Fit on full training set
        model.fit(X_train, y_train)

        # Evaluate on held-out test set
        test_result = evaluate(model, X_test, y_test, name)

        # Save model
        model_path = models_dir / f"{name}_model.pkl"
        save_model(model, model_path)

        all_results[name] = {
            **test_result,
            **cv_result,
            "model_path": str(model_path),
        }

    # Save metrics
    output = {
        "phase":        "2 — Model Training (UNSW-NB15)",
        "features":     FINAL_FEATURES,
        "n_features":   len(FINAL_FEATURES),
        "train_size":   int(len(X_train)),
        "test_size":    int(len(X_test)),
        "results":      all_results,
        "best_model":   max(
            all_results,
            key=lambda k: all_results[k]["f1_attack"]
        ),
    }

    metrics_path = metrics_dir / "phase2_results.json"
    with open(metrics_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info(f"Metrics saved: {metrics_path}")

    return output
