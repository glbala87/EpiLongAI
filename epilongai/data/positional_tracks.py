"""
Phase I — Position-aware methylation track modelling.

Instead of collapsing each window into summary statistics, this module
preserves the *spatial structure* of methylation within each window by
binning CpG sites into fixed-length position bins.

Why this can outperform summary statistics:
    - Local methylation patterns (e.g. gradients, bimodal islands) carry
      biological signal that means/variances erase.
    - 1D CNNs can learn spatial motifs directly from positional tracks.
    - Particularly relevant for ONT data where read-level phasing may
      reveal allele-specific methylation structure.

Trade-off: sparsity vs resolution.
    - More bins → finer spatial resolution but sparser signal per bin.
    - Fewer bins → denser features but coarser positional information.
    - A good default is window_size / 10 bins (e.g. 100 bins for 1 kb).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


def build_positional_tracks(
    meth_df: pd.DataFrame,
    window_size: int = 1000,
    stride: int = 1000,
    n_bins: int = 100,
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build fixed-length positional methylation tracks for each window.

    Parameters
    ----------
    meth_df : DataFrame
        Per-CpG data with columns: sample_id (optional), chr, start, beta, coverage.
    window_size : int
        Window size in bp.
    stride : int
        Stride in bp.
    n_bins : int
        Number of positional bins per window.
    features : list[str] or None
        Which per-bin features to compute.  Default: ['mean_beta', 'cpg_count', 'mean_coverage'].

    Returns
    -------
    window_index : DataFrame
        One row per window with (sample_id, chr, window_start, window_end).
    tracks : np.ndarray
        Shape (n_windows, n_channels, n_bins) — ready for 1D CNN input.
    """
    if features is None:
        features = ["mean_beta", "cpg_count", "mean_coverage"]

    df = meth_df.copy()
    df["window_start"] = (df["start"] // stride) * stride
    df["window_end"] = df["window_start"] + window_size

    # Compute bin index within each window
    bin_size = window_size / n_bins
    df["bin_idx"] = ((df["start"] - df["window_start"]) / bin_size).astype(int).clip(0, n_bins - 1)

    has_sample = "sample_id" in df.columns
    group_cols = (["sample_id"] if has_sample else []) + ["chr", "window_start", "window_end"]

    # Group by window + bin
    grouped = df.groupby(group_cols + ["bin_idx"])

    # Compute per-bin aggregates
    agg_dict: dict[str, Any] = {}
    if "mean_beta" in features:
        agg_dict["beta"] = "mean"
    if "cpg_count" in features:
        agg_dict["start"] = "count"  # count CpGs in this bin
    if "mean_coverage" in features and "coverage" in df.columns:
        agg_dict["coverage"] = "mean"

    bin_stats = grouped.agg(agg_dict).reset_index()
    # Rename
    rename_map = {"beta": "mean_beta", "start": "cpg_count", "coverage": "mean_coverage"}
    bin_stats = bin_stats.rename(columns={k: v for k, v in rename_map.items() if k in bin_stats.columns})

    # Build window index
    window_groups = df.groupby(group_cols).size().reset_index(name="_n")
    window_index = window_groups[group_cols].reset_index(drop=True)
    n_windows = len(window_index)

    # Determine channels
    channels = [f for f in features if f in bin_stats.columns]
    n_channels = len(channels)

    # Allocate track array
    tracks = np.zeros((n_windows, n_channels, n_bins), dtype=np.float32)

    # Build mapping from group_cols → window index
    win_key_to_idx = {
        tuple(row[col] for col in group_cols): i
        for i, (_, row) in enumerate(window_index.iterrows())
    }

    for _, row in bin_stats.iterrows():
        key = tuple(row[col] for col in group_cols)
        widx = win_key_to_idx.get(key)
        if widx is None:
            continue
        bidx = int(row["bin_idx"])
        for ci, ch in enumerate(channels):
            val = row.get(ch, 0.0)
            tracks[widx, ci, bidx] = val if not np.isnan(val) else 0.0

    logger.info(
        f"Built positional tracks: {n_windows} windows × {n_channels} channels × {n_bins} bins"
    )
    return window_index, tracks


class PositionalTrackDataset:
    """
    Wraps positional tracks as a PyTorch-compatible dataset.

    Integrates with the existing MethylationDataset by providing
    a ``__getitem__`` that returns the track tensor.
    """

    def __init__(
        self,
        tracks: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        import torch
        self.tracks = torch.from_numpy(tracks)
        self.labels = torch.from_numpy(labels).long() if labels is not None else None

    def __len__(self) -> int:
        return self.tracks.shape[0]

    def __getitem__(self, idx: int) -> dict:
        item = {"sequence": self.tracks[idx]}  # reuse "sequence" key for CNN input
        if self.labels is not None:
            item["label"] = self.labels[idx]
        item["idx"] = idx
        return item
