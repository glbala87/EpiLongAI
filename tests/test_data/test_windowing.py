"""Tests for genomic windowing module."""

import numpy as np
import pandas as pd
import pytest

from epilongai.data.windowing import compute_window_features_fast, generate_windows


@pytest.fixture
def methylation_df() -> pd.DataFrame:
    """Simple methylation data for testing."""
    return pd.DataFrame({
        "sample_id": ["s1"] * 10 + ["s2"] * 10,
        "chr": ["chr1"] * 20,
        "start": list(range(0, 1000, 100)) * 2,
        "beta": np.random.rand(20),
        "coverage": np.random.randint(5, 50, 20),
    })


class TestGenerateWindows:
    def test_basic(self):
        windows = generate_windows(["chr1"], {"chr1": 5000}, window_size=1000, stride=1000)
        assert len(windows) == 5
        assert list(windows.columns) == ["chr", "window_start", "window_end"]

    def test_overlapping(self):
        windows = generate_windows(["chr1"], {"chr1": 2000}, window_size=1000, stride=500)
        assert len(windows) == 4


class TestWindowFeaturesFast:
    def test_basic(self, methylation_df: pd.DataFrame):
        result = compute_window_features_fast(
            methylation_df, window_size=1000, stride=1000, min_cpgs=1
        )
        assert "mean_beta" in result.columns
        assert "n_cpgs" in result.columns
        assert len(result) > 0

    def test_min_cpgs_filter(self, methylation_df: pd.DataFrame):
        full = compute_window_features_fast(methylation_df, min_cpgs=1)
        filtered = compute_window_features_fast(methylation_df, min_cpgs=100)
        assert len(filtered) <= len(full)
