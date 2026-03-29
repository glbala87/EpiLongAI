"""
Publication-ready plotting utilities for training results.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_curve

# Use a clean style suitable for manuscripts
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot and optionally save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, cmap="Blues", ax=ax
    )
    ax.set_title("Confusion Matrix")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved confusion matrix to {save_path}")
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Plot and optionally save a ROC curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_title("ROC Curve")
    ax.legend()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved ROC curve to {save_path}")
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved PR curve to {save_path}")
    plt.close(fig)


def plot_training_history(
    history: dict[str, list[float]],
    save_path: str | Path | None = None,
) -> None:
    """Plot training/validation loss and metric curves."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    # Primary metric (e.g. F1 or accuracy)
    metric_keys = [k for k in history if k.startswith("val_") and k != "val_loss"]
    for k in metric_keys[:3]:
        axes[1].plot(history[k], label=k)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_title("Validation Metrics")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        logger.info(f"Saved training history to {save_path}")
    plt.close(fig)
