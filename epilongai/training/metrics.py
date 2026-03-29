"""
Metrics computation for classification and regression tasks.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    num_classes: int = 2,
) -> dict[str, float]:
    """Compute classification metrics."""
    metrics: dict[str, float] = {}

    if num_classes == 2:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["precision"] = precision_score(y_true, y_pred, zero_division=0)
        metrics["recall"] = recall_score(y_true, y_pred, zero_division=0)
        metrics["f1"] = f1_score(y_true, y_pred, zero_division=0)
        if y_prob is not None:
            prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob.ravel()
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, prob)
            except ValueError:
                metrics["roc_auc"] = float("nan")
            try:
                metrics["pr_auc"] = average_precision_score(y_true, prob)
            except ValueError:
                metrics["pr_auc"] = float("nan")
    else:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
        metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute regression metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
