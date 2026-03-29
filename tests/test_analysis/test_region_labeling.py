"""Tests for DMR-style region labeling."""

import numpy as np
import pandas as pd
import pytest

from epilongai.analysis.region_labeling import compare_windows


@pytest.fixture
def case_control_data():
    """Create windowed data with clear case/control difference."""
    np.random.seed(42)
    n_windows = 20
    n_samples_per_group = 10

    records = []
    for i in range(n_windows):
        for j in range(n_samples_per_group):
            # Cases have higher methylation in first 5 windows
            if i < 5:
                beta = np.random.normal(0.8, 0.05) if j < 5 else np.random.normal(0.3, 0.05)
            else:
                beta = np.random.normal(0.5, 0.1)
            records.append({
                "sample_id": f"case_{j}" if j < 5 else f"ctrl_{j - 5}",
                "chr": "chr1",
                "window_start": i * 1000,
                "window_end": (i + 1) * 1000,
                "mean_beta": np.clip(beta, 0, 1),
            })

    windows = pd.DataFrame(records)
    metadata = pd.DataFrame({
        "sample_id": [f"case_{j}" for j in range(5)] + [f"ctrl_{j}" for j in range(5)],
        "group": ["PTB"] * 5 + ["FTB"] * 5,
    })
    return windows, metadata


class TestCompareWindows:
    def test_basic_output(self, case_control_data):
        windows, metadata = case_control_data
        result = compare_windows(windows, metadata)
        assert "delta_beta" in result.columns
        assert "pvalue" in result.columns
        assert "label" in result.columns
        assert set(result["label"].unique()).issubset({"hyper", "hypo", "unchanged", "insufficient_data"})

    def test_detects_hyper(self, case_control_data):
        windows, metadata = case_control_data
        result = compare_windows(
            windows, metadata,
            delta_beta_threshold=0.1,
            pvalue_threshold=0.05,
        )
        # First 5 windows should be hypermethylated in cases
        hyper = result[result["label"] == "hyper"]
        assert len(hyper) > 0

    def test_custom_thresholds(self, case_control_data):
        windows, metadata = case_control_data
        # Very strict threshold → fewer significant windows
        strict = compare_windows(windows, metadata, delta_beta_threshold=0.9)
        lenient = compare_windows(windows, metadata, delta_beta_threshold=0.01)
        strict_sig = len(strict[strict["label"].isin(["hyper", "hypo"])])
        lenient_sig = len(lenient[lenient["label"].isin(["hyper", "hypo"])])
        assert lenient_sig >= strict_sig
