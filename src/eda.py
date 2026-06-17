"""
src/eda.py
IoT-SecBand — Exploratory Data Analysis module

Pipeline position: runs after preprocess.py, before model training.
Input:  data/processed/unsw_train_clean.parquet
Output: outputs/eda/*.png  (all plots saved, not shown inline)

All functions accept a DataFrame and a save_dir Path.
Call from notebook; do not duplicate logic there.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import logging

log = logging.getLogger(__name__)

TARGET = "label"
PALETTE = {0: "#1D9E75", 1: "#D85A30"}   # teal=normal, coral=attack

plt.rcParams.update({
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.labelsize":   11,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "font.family":      "sans-serif",
})


def _save(fig: plt.Figure, save_dir: Path, filename: str) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / filename
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out}")
    return out


# ─────────────────────────────────────────────
# 1. CLASS DISTRIBUTION
# ─────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, save_dir: Path) -> Path:
    """Bar chart of Normal vs Attack counts and percentages."""
    counts = df[TARGET].value_counts().sort_index()
    labels = {0: "Normal", 1: "Attack"}
    total  = len(df)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        [labels[i] for i in counts.index],
        counts.values,
        color=[PALETTE[i] for i in counts.index],
        width=0.5,
        edgecolor="none",
    )
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{count:,}\n({count/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title("Class distribution — UNSW-NB15 training set", fontsize=11, pad=10)
    ax.set_ylabel("Sample count")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylim(0, counts.max() * 1.15)
    return _save(fig, save_dir, "01_class_distribution.png")


# ─────────────────────────────────────────────
# 2. FEATURE DISTRIBUTIONS (Normal vs Attack)
# ─────────────────────────────────────────────

def plot_feature_distributions(
    df: pd.DataFrame,
    save_dir: Path,
    numeric_features: list[str] | None = None,
    n_cols: int = 4,
) -> Path:
    """
    KDE plots per numeric feature, split by class.
    Reveals which features separate Normal from Attack traffic.
    """
    if numeric_features is None:
        numeric_features = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c != TARGET
        ]

    n_rows = int(np.ceil(len(numeric_features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
    axes = axes.flatten()

    for i, col in enumerate(numeric_features):
        ax = axes[i]
        for label, color in PALETTE.items():
            subset = df[df[TARGET] == label][col].dropna()
            subset_clipped = subset.clip(subset.quantile(0.01), subset.quantile(0.99))
            ax.hist(
                subset_clipped, bins=40, alpha=0.55, color=color,
                label=("Normal" if label == 0 else "Attack"),
                density=True, edgecolor="none",
            )
        ax.set_title(col, fontsize=9, pad=4)
        ax.set_xlabel("")
        ax.set_ylabel("")
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(len(numeric_features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature distributions: Normal vs Attack (clipped 1–99 pct)", fontsize=11, y=1.01)
    fig.tight_layout()
    return _save(fig, save_dir, "02_feature_distributions.png")


# ─────────────────────────────────────────────
# 3. CORRELATION HEATMAP (attack class only)
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, save_dir: Path) -> Path:
    """
    Pearson correlation heatmap of numeric features.
    Use this to detect highly correlated feature pairs (|r| > 0.90)
    — candidates for removal to reduce model input size.
    """
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax,
        cmap="RdYlGn", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".1f", annot_kws={"size": 7},
        linewidths=0.3, linecolor="white",
        square=True,
    )
    ax.set_title("Feature correlation matrix (lower triangle)", fontsize=11, pad=10)
    fig.tight_layout()
    return _save(fig, save_dir, "03_correlation_heatmap.png")


# ─────────────────────────────────────────────
# 4. OUTLIER DETECTION (IQR method)
# ─────────────────────────────────────────────

def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    IQR-based outlier count per numeric feature.
    Returns a DataFrame sorted by outlier_pct descending.
    Features with >10% outliers should be inspected before training.
    """
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
    rows = []
    for col in numeric_cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append({
            "feature":      col,
            "q1":           round(q1, 4),
            "q3":           round(q3, 4),
            "iqr":          round(iqr, 4),
            "n_outliers":   int(n_out),
            "outlier_pct":  round(n_out / len(df) * 100, 2),
        })
    report = pd.DataFrame(rows).sort_values("outlier_pct", ascending=False)
    log.info(f"Outlier report — top 5:\n{report.head().to_string(index=False)}")
    return report


def plot_outlier_summary(df: pd.DataFrame, save_dir: Path) -> Path:
    """Horizontal bar chart of outlier % per feature."""
    report = outlier_report(df)

    fig, ax = plt.subplots(figsize=(7, max(4, len(report) * 0.35)))
    colors = ["#D85A30" if p > 10 else "#1D9E75" for p in report["outlier_pct"]]
    ax.barh(report["feature"], report["outlier_pct"], color=colors, edgecolor="none")
    ax.axvline(10, color="#888780", linestyle="--", linewidth=0.8, label=">10% threshold")
    ax.set_xlabel("Outlier %")
    ax.set_title("Outlier % per feature (IQR method)", fontsize=11)
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_dir, "04_outlier_summary.png")


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE PROXY (mutual info)
# ─────────────────────────────────────────────

def plot_mutual_info(df: pd.DataFrame, save_dir: Path) -> Path:
    """
    Mutual information score of each feature vs target label.
    Higher MI = more relevant to attack/normal classification.
    Use to validate your feature selection before training.
    """
    from sklearn.feature_selection import mutual_info_classif

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]
    X = df[numeric_cols].fillna(0)
    y = df[TARGET]

    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({"feature": numeric_cols, "mi_score": mi_scores})
    mi_df = mi_df.sort_values("mi_score", ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(4, len(mi_df) * 0.38)))
    colors = [
        "#1D9E75" if s > mi_df["mi_score"].median() else "#B4B2A9"
        for s in mi_df["mi_score"]
    ]
    ax.barh(mi_df["feature"], mi_df["mi_score"], color=colors, edgecolor="none")
    ax.axvline(
        mi_df["mi_score"].median(),
        color="#D85A30", linestyle="--", linewidth=0.8, label="Median MI"
    )
    ax.set_xlabel("Mutual information score")
    ax.set_title("Feature relevance to target (mutual information)", fontsize=11)
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_dir, "05_mutual_info.png")


# ─────────────────────────────────────────────
# 6. RUN ALL EDA
# ─────────────────────────────────────────────

def run_eda(df: pd.DataFrame, save_dir: str | Path) -> dict:
    """
    Run full EDA suite on cleaned training data.
    Input:  data/processed/unsw_train_clean.parquet
    Output: outputs/eda/*.png

    Returns dict of output file paths.
    """
    save_dir = Path(save_dir)
    log.info(f"Running EDA — saving to {save_dir}")

    paths = {
        "class_distribution":    str(plot_class_distribution(df, save_dir)),
        "feature_distributions": str(plot_feature_distributions(df, save_dir)),
        "correlation_heatmap":   str(plot_correlation_heatmap(df, save_dir)),
        "outlier_summary":       str(plot_outlier_summary(df, save_dir)),
        "mutual_info":           str(plot_mutual_info(df, save_dir)),
    }

    log.info(f"EDA complete — {len(paths)} plots saved to {save_dir}")
    return paths
