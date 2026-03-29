"""
Phase D — PyTorch Dataset and DataLoader for ONT methylation training.

Supports data modes:
    1. methylation  — numeric window features only
    2. sequence     — one-hot or tokenized DNA sequence
    3. multimodal   — both sequence + methylation features
    4. variants     — methylation + variant features
    5. full         — sequence + methylation + variants

Each sample represents one genomic window.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# ── Constants ────────────────────────────────────────────────────────────
NUCLEOTIDE_MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
NUM_NUCLEOTIDES = 5

METHYLATION_FEATURE_COLS = [
    "mean_beta",
    "median_beta",
    "var_beta",
    "n_cpgs",
    "mean_coverage",
    "frac_high_meth",
    "frac_low_meth",
]

VARIANT_FEATURE_COLS = [
    "n_variants",
    "n_snps",
    "n_indels",
    "mean_dosage",
    "max_dosage",
    "mean_af",
]


# ── Sequence encoding helpers ────────────────────────────────────────────
def encode_sequence_onehot(seq: str, max_len: int) -> np.ndarray:
    """Encode a DNA string as a (NUM_NUCLEOTIDES, max_len) one-hot matrix."""
    arr = np.zeros((NUM_NUCLEOTIDES, max_len), dtype=np.float32)
    for i, base in enumerate(seq[:max_len]):
        idx = NUCLEOTIDE_MAP.get(base.upper(), 4)
        arr[idx, i] = 1.0
    return arr


def encode_sequence_tokenized(seq: str, max_len: int) -> np.ndarray:
    """Encode a DNA string as an integer token array of shape (max_len,)."""
    tokens = np.full(max_len, NUCLEOTIDE_MAP["N"], dtype=np.int64)
    for i, base in enumerate(seq[:max_len]):
        tokens[i] = NUCLEOTIDE_MAP.get(base.upper(), 4)
    return tokens


# ── Dataset ──────────────────────────────────────────────────────────────
class MethylationDataset(Dataset):
    """
    PyTorch Dataset for windowed ONT methylation data.

    Parameters
    ----------
    windows : DataFrame
        Windowed feature table (from windowing module).
    labels : array-like or None
        Integer labels aligned with windows rows.
    mode : str
        One of 'methylation', 'sequence', 'multimodal', 'variants', 'full'.
    sequence_encoding : str
        'onehot' or 'tokenized'.
    max_sequence_length : int
        Pad/truncate sequences to this length.
    feature_cols : list[str] or None
        Override which numeric columns to use as methylation features.
    variant_feature_cols : list[str] or None
        Override which columns to use as variant features.
    """

    def __init__(
        self,
        windows: pd.DataFrame,
        labels: np.ndarray | None = None,
        mode: Literal["methylation", "sequence", "multimodal", "variants", "full"] = "methylation",
        sequence_encoding: str = "onehot",
        max_sequence_length: int = 1000,
        feature_cols: list[str] | None = None,
        variant_feature_cols: list[str] | None = None,
    ) -> None:
        self.mode = mode
        self.encoding = sequence_encoding
        self.max_seq_len = max_sequence_length
        self.feature_cols = feature_cols or [c for c in METHYLATION_FEATURE_COLS if c in windows.columns]
        self.variant_feature_cols = variant_feature_cols or [c for c in VARIANT_FEATURE_COLS if c in windows.columns]

        # Methylation features → float tensor
        if mode in ("methylation", "multimodal", "variants", "full"):
            self.features = windows[self.feature_cols].fillna(0).values.astype(np.float32)
        else:
            self.features = None

        # Sequences
        if mode in ("sequence", "multimodal", "full"):
            if "sequence" not in windows.columns:
                raise ValueError(f"Mode '{mode}' requires a 'sequence' column in windows DataFrame")
            self.sequences = windows["sequence"].tolist()
        else:
            self.sequences = None

        # Variant features
        if mode in ("variants", "full") and self.variant_feature_cols:
            self.variant_features = windows[self.variant_feature_cols].fillna(0).values.astype(np.float32)
        else:
            self.variant_features = None

        # Labels
        self.labels = labels

        # Coordinates for interpretability / traceability
        coord_cols = [c for c in ["sample_id", "chr", "window_start", "window_end"] if c in windows.columns]
        self.coords = windows[coord_cols].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {}

        if self.features is not None:
            item["methylation"] = torch.from_numpy(self.features[idx])

        if self.sequences is not None:
            seq = self.sequences[idx]
            if self.encoding == "onehot":
                item["sequence"] = torch.from_numpy(
                    encode_sequence_onehot(seq, self.max_seq_len)
                )
            else:
                item["sequence"] = torch.from_numpy(
                    encode_sequence_tokenized(seq, self.max_seq_len)
                )

        if self.variant_features is not None:
            item["variants"] = torch.from_numpy(self.variant_features[idx])

        if self.labels is not None:
            item["label"] = torch.tensor(self.labels[idx], dtype=torch.long)

        item["idx"] = idx
        return item

    @property
    def num_features(self) -> int:
        """Number of numeric methylation features."""
        return len(self.feature_cols)

    @property
    def num_variant_features(self) -> int:
        """Number of variant features."""
        return len(self.variant_feature_cols) if self.variant_features is not None else 0


# ── Collate function ─────────────────────────────────────────────────────
def methylation_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate that handles variable keys gracefully."""
    collated: dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        if isinstance(vals[0], torch.Tensor):
            collated[k] = torch.stack(vals)
        else:
            collated[k] = vals
    return collated


# ── Data splitting ───────────────────────────────────────────────────────
def split_dataset(
    windows: pd.DataFrame,
    labels: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    stratify: bool = True,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split windows + labels into train / val / test sets.

    When ``sample_id`` is present, splits are done at the sample level
    to prevent data leakage between windows from the same sample.
    """
    if "sample_id" in windows.columns:
        return _split_by_sample(windows, labels, test_size, val_size, stratify, random_seed)

    strat = labels if stratify else None
    idx = np.arange(len(windows))
    idx_train_val, idx_test = train_test_split(
        idx, test_size=test_size, stratify=strat, random_state=random_seed
    )
    strat_tv = labels[idx_train_val] if stratify else None
    relative_val = val_size / (1 - test_size)
    idx_train, idx_val = train_test_split(
        idx_train_val, test_size=relative_val, stratify=strat_tv, random_state=random_seed
    )

    logger.info(f"Split: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    return (
        windows.iloc[idx_train].reset_index(drop=True),
        windows.iloc[idx_val].reset_index(drop=True),
        windows.iloc[idx_test].reset_index(drop=True),
        labels[idx_train],
        labels[idx_val],
        labels[idx_test],
    )


def _split_by_sample(
    windows: pd.DataFrame,
    labels: np.ndarray,
    test_size: float,
    val_size: float,
    stratify: bool,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Split at sample level to prevent leakage."""
    sample_labels = windows.groupby("sample_id").apply(
        lambda g: labels[g.index[0]], include_groups=False
    )
    samples = sample_labels.index.values
    slabels = sample_labels.values

    strat = slabels if stratify else None
    s_train_val, s_test = train_test_split(
        samples, test_size=test_size, stratify=strat, random_state=seed
    )
    strat_tv = slabels[np.isin(samples, s_train_val)] if stratify else None
    relative_val = val_size / (1 - test_size)
    s_train, s_val = train_test_split(
        s_train_val, test_size=relative_val, stratify=strat_tv, random_state=seed
    )

    mask_train = windows["sample_id"].isin(s_train)
    mask_val = windows["sample_id"].isin(s_val)
    mask_test = windows["sample_id"].isin(s_test)

    logger.info(
        f"Sample-level split: train={mask_train.sum()} ({len(s_train)} samples), "
        f"val={mask_val.sum()} ({len(s_val)} samples), "
        f"test={mask_test.sum()} ({len(s_test)} samples)"
    )
    return (
        windows[mask_train].reset_index(drop=True),
        windows[mask_val].reset_index(drop=True),
        windows[mask_test].reset_index(drop=True),
        labels[mask_train.values],
        labels[mask_val.values],
        labels[mask_test.values],
    )


# ── DataLoader factory ───────────────────────────────────────────────────
def build_dataloaders(
    train_ds: MethylationDataset,
    val_ds: MethylationDataset,
    test_ds: MethylationDataset | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    """Build DataLoaders for each split."""
    loaders: dict[str, DataLoader] = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=methylation_collate,
            drop_last=False,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=methylation_collate,
        ),
    }
    if test_ds is not None:
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=methylation_collate,
        )
    return loaders
