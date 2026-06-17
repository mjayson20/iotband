"""
src/evaluate.py
IoT-SecBand — Evaluation visualisation module

Pipeline position: runs after train.py, produces metric plots.
Input:  trained models (pkl) + test split
Output: outputs/metrics/*.png

All functions accept fitted model + test data and a save_dir.
Call from notebook only.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, RocCurveDisplay,
)
import logging

log = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "font.family":       "sans-serif",
})

PALETTE = {"decision_tree": "#1D9E75", "random_forest": "#378ADD", "mlp": "#D85A30"}


def _save(fig: plt.Figure, save_dir: Path, filename: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out}")
    return out


# ─────────────────────────────────────────────
# CONFUSION MATRIX
# ─────────────────────────────────────────────

def plot_confusion_matrices(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: Path,
) -> Path:
    """
    Side-by-side confusion matrices for all three models.
    Focus on false negatives (bottom-left) — missed attacks.
    For IDS: FN is more costly than FP.
    """
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4))
    if len(models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Normal", "Attack"],
        )
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")

    fig.suptitle("Confusion matrices — UNSW-NB15 test set", fontsize=11, y=1.02)
    fig.tight_layout()
    return _save(fig, save_dir, "06_confusion_matrices.png")


# ─────────────────────────────────────────────
# ROC CURVES
# ─────────────────────────────────────────────

def plot_roc_curves(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: Path,
) -> Path:
    """
    Overlaid ROC curves for all models.
    AUC > 0.95 required before proceeding to BoT-IoT fine-tuning.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            log.warning(f"{name} has no predict_proba — skipping ROC")
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        color = PALETTE.get(name, "#888780")
        ax.plot(fpr, tpr, color=color, lw=1.5,
                label=f"{name.replace('_', ' ').title()} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves — UNSW-NB15 test set", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    return _save(fig, save_dir, "07_roc_curves.png")


# ─────────────────────────────────────────────
# METRIC COMPARISON BAR CHART
# ─────────────────────────────────────────────

def plot_metric_comparison(results: dict, save_dir: Path) -> Path:
    """
    Grouped bar chart comparing Accuracy, F1 (attack), Recall (attack), AUC.
    Highlights whether any model meets the F1 >= 0.88, Recall >= 0.88 target.
    """
    metrics = ["accuracy", "f1_attack", "recall_attack", "roc_auc"]
    labels  = ["Accuracy", "F1 (attack)", "Recall (attack)", "ROC-AUC"]
    model_names = list(results.keys())

    x = np.arange(len(metrics))
    width = 0.22
    offsets = np.linspace(-(len(model_names)-1)/2, (len(model_names)-1)/2, len(model_names)) * width

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, name in enumerate(model_names):
        vals   = [results[name].get(m, 0) or 0 for m in metrics]
        color  = PALETTE.get(name, "#888780")
        bars   = ax.bar(x + offsets[i], vals, width * 0.88, label=name.replace("_", " ").title(),
                        color=color, alpha=0.85, edgecolor="none")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7.5,
            )

    # Target line
    ax.axhline(0.88, color="#D85A30", linestyle="--", lw=0.9, alpha=0.7, label="Target (0.88)")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — UNSW-NB15 test set", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, save_dir, "08_metric_comparison.png")


# ─────────────────────────────────────────────
# FEATURE IMPORTANCE (DT + RF only)
# ─────────────────────────────────────────────

def plot_feature_importance(
    models: dict,
    feature_names: list[str],
    save_dir: Path,
) -> Path | None:
    """
    Feature importances from Decision Tree and Random Forest.
    Compare against mutual information scores from Phase 1 (plot 05).
    Features with near-zero importance in both models → safe to drop in Phase 3.
    """
    tree_models = {k: v for k, v in models.items() if hasattr(v, "feature_importances_")}
    if not tree_models:
        log.warning("No tree-based models found — skipping feature importance plot")
        return None

    fig, axes = plt.subplots(1, len(tree_models), figsize=(7 * len(tree_models), 5))
    if len(tree_models) == 1:
        axes = [axes]

    for ax, (name, model) in zip(axes, tree_models.items()):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        color = PALETTE.get(name, "#888780")
        ax.barh(
            [feature_names[i] for i in indices],
            importances[indices],
            color=color, alpha=0.8, edgecolor="none",
        )
        ax.set_title(name.replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Importance")

    fig.suptitle("Feature importances (tree-based models)", fontsize=11, y=1.01)
    fig.tight_layout()
    return _save(fig, save_dir, "09_feature_importance.png")


# ─────────────────────────────────────────────
# CV OVERFIT DIAGNOSTIC
# ─────────────────────────────────────────────

def plot_overfit_diagnostic(results: dict, save_dir: Path) -> Path:
    """
    Train F1 vs CV F1 per model — visualises overfitting gap.
    Gap > 0.05 means model memorised training data and will
    degrade on BoT-IoT. This is the root cause of your previous 99% issue.
    """
    model_names  = list(results.keys())
    train_f1s    = [results[n].get("train_f1_mean", 0) for n in model_names]
    cv_f1s       = [results[n].get("cv_f1_mean", 0) for n in model_names]

    x = np.arange(len(model_names))
    width = 0.32

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width/2, train_f1s, width, label="Train F1",
           color="#378ADD", alpha=0.8, edgecolor="none")
    ax.bar(x + width/2, cv_f1s,    width, label="CV F1 (5-fold)",
           color="#1D9E75", alpha=0.8, edgecolor="none")

    # Annotate gap
    for i, (tr, cv) in enumerate(zip(train_f1s, cv_f1s)):
        gap = tr - cv
        color = "#D85A30" if gap > 0.05 else "#5F5E5A"
        ax.text(i, max(tr, cv) + 0.01, f"gap={gap:.3f}",
                ha="center", fontsize=8.5, color=color)

    ax.axhline(0.88, color="#D85A30", linestyle="--", lw=0.9, alpha=0.7, label="Target (0.88)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ").title() for n in model_names])
    ax.set_ylim(0.7, 1.05)
    ax.set_ylabel("F1 score (attack class)")
    ax.set_title("Overfitting diagnostic: train F1 vs CV F1", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save(fig, save_dir, "10_overfit_diagnostic.png")


# ─────────────────────────────────────────────
# RUN ALL EVALUATION PLOTS
# ─────────────────────────────────────────────

def run_evaluation_plots(
    models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    results: dict,
    feature_names: list[str],
    save_dir: str | Path,
) -> dict:
    """
    Run all evaluation plots for Phase 2.
    Input:  fitted model dict + test split + results dict from train.py
    Output: outputs/metrics/*.png
    """
    save_dir = Path(save_dir)
    paths = {}

    paths["confusion_matrices"] = str(
        plot_confusion_matrices(models, X_test, y_test, save_dir))
    paths["roc_curves"] = str(
        plot_roc_curves(models, X_test, y_test, save_dir))
    paths["metric_comparison"] = str(
        plot_metric_comparison(results, save_dir))
    fi_path = plot_feature_importance(models, feature_names, save_dir)
    if fi_path:
        paths["feature_importance"] = str(fi_path)
    paths["overfit_diagnostic"] = str(
        plot_overfit_diagnostic(results, save_dir))

    log.info(f"Evaluation complete — {len(paths)} plots saved to {save_dir}")
    return paths
