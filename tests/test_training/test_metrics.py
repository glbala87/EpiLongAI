"""Tests for metrics computation."""

import numpy as np

from epilongai.training.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)


class TestClassificationMetrics:
    def test_binary(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_prob = np.array([0.1, 0.6, 0.9, 0.8])
        m = compute_classification_metrics(y_true, y_pred, y_prob, num_classes=2)
        assert "accuracy" in m
        assert "f1" in m
        assert "roc_auc" in m
        assert 0 <= m["accuracy"] <= 1

    def test_multiclass(self):
        y_true = np.array([0, 1, 2, 0])
        y_pred = np.array([0, 1, 1, 0])
        m = compute_classification_metrics(y_true, y_pred, num_classes=3)
        assert "macro_f1" in m
        assert "weighted_f1" in m


class TestRegressionMetrics:
    def test_basic(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.2, 2.8])
        m = compute_regression_metrics(y_true, y_pred)
        assert "mae" in m
        assert "rmse" in m
        assert "r2" in m
