"""Tests for benchmarking framework."""

import numpy as np
import pytest

from epilongai.analysis.benchmark import (
    compute_calibration,
    cross_validate,
    delong_test,
    get_baseline_models,
)


@pytest.fixture
def classification_data():
    np.random.seed(42)
    X = np.random.randn(100, 7).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


class TestCrossValidate:
    def test_basic(self, classification_data):
        X, y = classification_data
        from sklearn.linear_model import LogisticRegression
        metrics = cross_validate(
            X, y,
            model_factory=lambda: LogisticRegression(max_iter=200),
            n_folds=3,
        )
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert len(metrics["accuracy"]) == 3

    def test_all_baselines_run(self, classification_data):
        X, y = classification_data
        baselines = get_baseline_models()
        for name, factory in baselines.items():
            metrics = cross_validate(X, y, factory, n_folds=2)
            assert "accuracy" in metrics, f"{name} failed"


class TestDelongTest:
    def test_identical_models(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.3, 0.8, 0.9, 0.2, 0.7])
        result = delong_test(y_true, y_prob, y_prob)
        assert result["auc_a"] == pytest.approx(result["auc_b"])
        # p-value should be ~1 for identical models
        assert result["p_value"] > 0.5

    def test_different_models(self):
        np.random.seed(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_good = np.concatenate([np.random.uniform(0, 0.4, 50), np.random.uniform(0.6, 1, 50)])
        y_bad = np.random.uniform(0, 1, 100)
        result = delong_test(y_true, y_good, y_bad)
        assert result["auc_a"] > result["auc_b"]


class TestCalibration:
    def test_basic(self):
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7, 0.2, 0.6, 0.8, 0.1])
        cal = compute_calibration(y_true, y_prob, n_bins=3)
        assert "fraction_positive" in cal
        assert "mean_predicted" in cal
        assert len(cal["fraction_positive"]) > 0
