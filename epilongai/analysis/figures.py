"""
Phase R — Manuscript-ready figures and tables.

All figures are generated at 300 DPI, suitable for Nature, Genome Biology,
Scientific Reports, and similar journals.

Styling:
    - Clean white background
    - Sans-serif fonts (Helvetica/Arial)
    - Colour palette accessible to colour-blind readers
    - Consistent axis labelling and legend placement
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

# ── Journal-quality styling ──────────────────────────────────────────────
JOURNAL_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.transparent": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Colour-blind–friendly palette (IBM design library)
PALETTE = ["#648FFF", "#DC267F", "#FE6100", "#785EF0", "#FFB000"]


def apply_journal_style() -> None:
    """Apply journal-quality matplotlib style globally."""
    plt.rcParams.update(JOURNAL_RC)
    sns.set_palette(PALETTE)


# =====================================================================
# Pipeline Workflow Diagram (text-based for code generation)
# =====================================================================
PIPELINE_DIAGRAM = """
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  ONT Methyl   │────▶│  Ingest &    │────▶│  Genomic     │
│  Files (.bed) │     │  Validate    │     │  Windowing   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────┐
                    │                             │                     │
              ┌─────▼─────┐              ┌───────▼──────┐     ┌───────▼──────┐
              │ Methylation│              │  DNA Sequence │     │   Variant    │
              │ Features   │              │  Encoding     │     │   Features   │
              └─────┬─────┘              └───────┬──────┘     └───────┬──────┘
                    │                             │                     │
                    └──────────────┬──────────────┘─────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │  Multimodal     │
                          │  Deep Learning  │
                          │  Model          │
                          └────────┬────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
              ┌─────▼─────┐ ┌─────▼─────┐ ┌─────▼─────┐
              │ Prediction │ │  Interpret │ │  Clinical │
              │ & Scoring  │ │  (SHAP)   │ │  Report   │
              └────────────┘ └───────────┘ └───────────┘
"""


# =====================================================================
# Model Architecture Diagram
# =====================================================================
def plot_model_architecture(
    save_path: str | Path | None = None,
) -> None:
    """
    Generate a visual model architecture diagram using matplotlib.
    """
    apply_journal_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box_style = dict(boxstyle="round,pad=0.4", facecolor="#E8EEF7", edgecolor="#4C72B0", linewidth=1.5)
    arrow_kw = dict(arrowstyle="->", color="#4C72B0", linewidth=1.5)

    # Input boxes
    boxes = [
        (1.5, 9, "DNA Sequence\n(one-hot, L bp)"),
        (5.0, 9, "Methylation\nFeatures"),
        (8.5, 9, "Variant\nFeatures"),
    ]
    for x, y, text in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=9, bbox=box_style)

    # Encoders
    encoders = [
        (1.5, 7, "1D CNN /\nTransformer"),
        (5.0, 7, "MLP\nEncoder"),
        (8.5, 7, "MLP\nEncoder"),
    ]
    for x, y, text in encoders:
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#D4E6F1", edgecolor="#2980B9", linewidth=1.5))

    # Arrows to encoders
    for x in [1.5, 5.0, 8.5]:
        ax.annotate("", xy=(x, 7.6), xytext=(x, 8.4), arrowprops=arrow_kw)

    # Fusion
    ax.text(5.0, 5, "Fusion Layer\n(concatenate / cross-attention)", ha="center", va="center",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FADBD8", edgecolor="#E74C3C", linewidth=1.5))

    for x in [1.5, 5.0, 8.5]:
        ax.annotate("", xy=(5.0, 5.6), xytext=(x, 6.4), arrowprops=arrow_kw)

    # Output
    ax.text(5.0, 3, "Classification Head\n(PTB vs FTB)", ha="center", va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#D5F5E3", edgecolor="#27AE60", linewidth=1.5))
    ax.annotate("", xy=(5.0, 3.6), xytext=(5.0, 4.4), arrowprops=arrow_kw)

    # Outputs
    outputs = [
        (2.5, 1.2, "Predictions"),
        (5.0, 1.2, "SHAP\nImportance"),
        (7.5, 1.2, "Risk\nScores"),
    ]
    for x, y, text in outputs:
        ax.text(x, y, text, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF9E7", edgecolor="#F39C12", linewidth=1))
        ax.annotate("", xy=(x, 1.8), xytext=(5.0, 2.4), arrowprops=arrow_kw)

    ax.set_title("EpiLongAI — Multimodal Model Architecture", fontsize=13, fontweight="bold", pad=15)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved architecture diagram to {save_path}")
    plt.close(fig)


# =====================================================================
# Cohort Summary Table
# =====================================================================
def generate_cohort_table(
    metadata: pd.DataFrame,
    group_column: str = "group",
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate a cohort summary table (Table 1 in manuscripts).
    """
    table_rows: list[dict[str, Any]] = {}

    overall = {"Group": "Overall", "N": len(metadata)}
    for grp, sub in metadata.groupby(group_column):
        table_rows[grp] = {"Group": grp, "N": len(sub)}

        # Add categorical summaries
        for col in ["source", "batch", "phenotype"]:
            if col in metadata.columns:
                counts = sub[col].value_counts()
                table_rows[grp][col] = "; ".join(f"{k}: {v}" for k, v in counts.items())

    table = pd.DataFrame([overall] + list(table_rows.values()))

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        logger.info(f"Saved cohort table to {output_path}")

    return table


# =====================================================================
# Performance Table
# =====================================================================
def generate_performance_table(
    results: dict[str, dict[str, float]],
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generate a model performance comparison table.

    Parameters
    ----------
    results : dict
        model_name → {metric_name: value}
    """
    rows = []
    for model, metrics in results.items():
        row = {"Model": model}
        row.update(metrics)
        rows.append(row)

    table = pd.DataFrame(rows)

    # Round numeric columns
    for col in table.select_dtypes(include=[np.number]).columns:
        table[col] = table[col].round(4)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        logger.info(f"Saved performance table to {output_path}")

    return table


# =====================================================================
# Top Predictive Regions Table
# =====================================================================
def generate_regions_table(
    top_regions: pd.DataFrame,
    output_path: str | Path | None = None,
    caption: str = "Top predictive genomic regions identified by EpiLongAI.",
) -> pd.DataFrame:
    """Format top regions for supplementary table."""
    table = top_regions.copy()
    table = table.round(4)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_path, index=False)
        logger.info(f"Saved regions table to {output_path}")

    return table


# =====================================================================
# Multi-panel Figure Generation
# =====================================================================
def generate_manuscript_figure_1(
    history: dict[str, list[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    save_path: str | Path | None = None,
) -> None:
    """
    Generate a multi-panel Figure 1 with:
        A) Training curves
        B) ROC curve
        C) Confusion matrix
        D) Feature importance (placeholder — requires SHAP values)

    Example figure caption:
        "Figure 1. EpiLongAI model performance for preterm birth prediction.
        (A) Training and validation loss over epochs. (B) Receiver operating
        characteristic curve on the held-out test set (AUC = X.XX).
        (C) Confusion matrix showing classification performance.
        (D) Top methylation features ranked by SHAP importance."
    """
    from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

    apply_journal_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # A) Training curves
    ax = axes[0, 0]
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train", color=PALETTE[0])
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val", color=PALETTE[1])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("A", fontweight="bold", loc="left", fontsize=14)
    ax.legend()

    # B) ROC
    ax = axes[0, 1]
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, color=PALETTE[0])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_title("B", fontweight="bold", loc="left", fontsize=14)

    # C) Confusion matrix
    ax = axes[1, 0]
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, cmap="Blues", ax=ax
    )
    ax.set_title("C", fontweight="bold", loc="left", fontsize=14)

    # D) Placeholder for SHAP
    ax = axes[1, 1]
    ax.text(0.5, 0.5, "Feature importance\n(see interpret module)",
            ha="center", va="center", fontsize=12, style="italic", color="gray")
    ax.set_title("D", fontweight="bold", loc="left", fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout(pad=2.0)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved Figure 1 to {save_path}")
    plt.close(fig)


# =====================================================================
# Example Figure Captions
# =====================================================================
FIGURE_CAPTIONS = {
    "figure_1": (
        "Figure 1. EpiLongAI model performance for preterm birth prediction. "
        "(A) Training and validation loss over epochs showing convergence. "
        "(B) Receiver operating characteristic curve on the held-out test set. "
        "(C) Confusion matrix for binary classification (PTB vs FTB). "
        "(D) Top methylation features ranked by mean absolute SHAP value."
    ),
    "figure_2": (
        "Figure 2. Genomic regions predictive of preterm birth. "
        "(A) Manhattan-style plot of per-window prediction scores across chromosomes. "
        "(B) Differentially methylated regions identified by case-control comparison. "
        "(C) Gene-level enrichment analysis of top predictive regions."
    ),
    "supplementary_figure_1": (
        "Supplementary Figure 1. Benchmarking EpiLongAI against baseline models. "
        "(A) ROC-AUC comparison across 5-fold cross-validation. "
        "(B) Calibration curves for each model. "
        "(C) Precision-recall curves."
    ),
}
