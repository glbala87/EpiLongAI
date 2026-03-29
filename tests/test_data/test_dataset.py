"""Tests for Dataset and DataLoader classes."""

import numpy as np
import pandas as pd
import pytest

from epilongai.data.dataset import (
    MethylationDataset,
    encode_sequence_onehot,
    encode_sequence_tokenized,
    split_dataset,
)


class TestSequenceEncoding:
    def test_onehot_shape(self):
        enc = encode_sequence_onehot("ACGTN", max_len=10)
        assert enc.shape == (5, 10)
        assert enc[:, 0].sum() == 1.0  # only one base active per position

    def test_tokenized(self):
        enc = encode_sequence_tokenized("ACG", max_len=5)
        assert enc.shape == (5,)
        assert enc[0] == 0  # A
        assert enc[1] == 1  # C
        assert enc[3] == 4  # N (padding)


@pytest.fixture
def sample_windows() -> tuple[pd.DataFrame, np.ndarray]:
    n = 100
    df = pd.DataFrame({
        "sample_id": [f"s{i % 5}" for i in range(n)],
        "chr": ["chr1"] * n,
        "window_start": range(0, n * 1000, 1000),
        "window_end": range(1000, (n + 1) * 1000, 1000),
        "mean_beta": np.random.rand(n),
        "median_beta": np.random.rand(n),
        "var_beta": np.random.rand(n) * 0.1,
        "n_cpgs": np.random.randint(3, 20, n),
        "mean_coverage": np.random.randint(10, 50, n).astype(float),
        "frac_high_meth": np.random.rand(n),
        "frac_low_meth": np.random.rand(n),
    })
    labels = np.random.randint(0, 2, n)
    return df, labels


class TestMethylationDataset:
    def test_len(self, sample_windows):
        df, labels = sample_windows
        ds = MethylationDataset(df, labels, mode="methylation")
        assert len(ds) == 100

    def test_getitem(self, sample_windows):
        df, labels = sample_windows
        ds = MethylationDataset(df, labels, mode="methylation")
        item = ds[0]
        assert "methylation" in item
        assert "label" in item
        assert item["methylation"].shape == (7,)

    def test_num_features(self, sample_windows):
        df, labels = sample_windows
        ds = MethylationDataset(df, labels, mode="methylation")
        assert ds.num_features == 7


class TestSplitDataset:
    def test_basic_split(self, sample_windows):
        df, labels = sample_windows
        # Drop sample_id to use index-based splitting (avoids per-sample stratify issue)
        df_no_sample = df.drop(columns=["sample_id"])
        w_tr, w_va, w_te, y_tr, y_va, y_te = split_dataset(df_no_sample, labels)
        assert len(w_tr) + len(w_va) + len(w_te) == 100
        assert len(y_tr) == len(w_tr)
