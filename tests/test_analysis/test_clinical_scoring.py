"""Tests for clinical scoring module."""

import numpy as np
import pandas as pd
import pytest

from epilongai.analysis.clinical_scoring import (
    compute_risk_scores,
    generate_clinical_report,
    identify_top_regions,
)


@pytest.fixture
def window_predictions():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "sample_id": [f"s{i % 5}" for i in range(n)],
        "chr": ["chr1"] * n,
        "window_start": range(0, n * 1000, 1000),
        "window_end": range(1000, (n + 1) * 1000, 1000),
        "prob_positive": np.random.rand(n),
        "mean_beta": np.random.rand(n),
        "mean_coverage": np.random.randint(10, 50, n).astype(float),
        "n_cpgs": np.random.randint(3, 20, n),
    })


class TestComputeRiskScores:
    def test_basic(self, window_predictions):
        scores = compute_risk_scores(window_predictions)
        assert "risk_score" in scores.columns
        assert "risk_category" in scores.columns
        assert "confidence" in scores.columns
        assert len(scores) == 5  # 5 unique samples
        assert all(0 <= scores["risk_score"]) and all(scores["risk_score"] <= 1)

    def test_median_aggregation(self, window_predictions):
        scores = compute_risk_scores(window_predictions, aggregation="median")
        assert len(scores) == 5

    def test_weighted_aggregation(self, window_predictions):
        scores = compute_risk_scores(window_predictions, aggregation="weighted")
        assert len(scores) == 5


class TestIdentifyTopRegions:
    def test_basic(self, window_predictions):
        top = identify_top_regions(window_predictions, "s0", top_k=5)
        assert len(top) == 5
        # Should be sorted descending by prob_positive
        assert top["prob_positive"].is_monotonic_decreasing

    def test_missing_sample(self, window_predictions):
        top = identify_top_regions(window_predictions, "nonexistent")
        assert len(top) == 0


class TestGenerateClinicalReport:
    def test_basic(self, window_predictions, tmp_path):
        top = identify_top_regions(window_predictions, "s0", top_k=3)
        report = generate_clinical_report(
            sample_id="s0",
            risk_score=0.75,
            risk_category="high",
            confidence=0.85,
            top_regions=top,
            model_info={"model_type": "baseline_mlp", "version": "0.1.0"},
            output_path=tmp_path / "report.txt",
        )
        assert "s0" in report
        assert "0.75" in report
        assert "HIGH" in report
        assert "RESEARCH USE ONLY" in report
        assert (tmp_path / "report.txt").exists()

    def test_low_risk(self, window_predictions):
        top = identify_top_regions(window_predictions, "s0", top_k=3)
        report = generate_clinical_report(
            sample_id="s0",
            risk_score=0.15,
            risk_category="low",
            confidence=0.9,
            top_regions=top,
            model_info={},
        )
        assert "low" in report.lower()
