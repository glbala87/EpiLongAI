"""
Phase C — Genomic windowing for ONT methylation modelling.

Converts sparse per-CpG methylation calls into fixed-size genomic windows
with summary features suitable for machine learning.

Output schema per window:
    sample_id, chr, window_start, window_end,
    mean_beta, median_beta, var_beta, n_cpgs, mean_coverage,
    frac_high_meth, frac_low_meth
    [optional: sequence]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from epilongai.utils.config import load_config
from epilongai.utils.logging import setup_logging


# ── Window generation ────────────────────────────────────────────────────
def generate_windows(
    chromosomes: list[str],
    chrom_sizes: dict[str, int],
    window_size: int = 1000,
    stride: int = 1000,
) -> pd.DataFrame:
    """Generate a DataFrame of genomic windows for requested chromosomes."""
    records: list[dict[str, Any]] = []
    for chrom in chromosomes:
        max_pos = chrom_sizes.get(chrom, 0)
        if max_pos == 0:
            continue
        for start in range(0, max_pos, stride):
            end = start + window_size
            records.append({"chr": chrom, "window_start": start, "window_end": end})
    windows = pd.DataFrame(records)
    logger.info(f"Generated {len(windows):,} windows across {len(chromosomes)} chromosomes")
    return windows


def _infer_chrom_sizes(df: pd.DataFrame) -> dict[str, int]:
    """Infer chromosome sizes from the data (max position per chrom)."""
    sizes = df.groupby("chr")["start"].max().to_dict()
    # Add padding so last window captures trailing sites
    return {k: int(v) + 1 for k, v in sizes.items()}


# ── Per-window feature extraction ────────────────────────────────────────
def compute_window_features(
    meth_df: pd.DataFrame,
    windows: pd.DataFrame,
    high_meth_threshold: float = 0.8,
    low_meth_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Assign methylation calls to windows and compute summary statistics.

    Parameters
    ----------
    meth_df : DataFrame
        Must contain columns: chr, start, beta.  Optionally: coverage, sample_id.
    windows : DataFrame
        Must contain: chr, window_start, window_end.

    Returns
    -------
    DataFrame with one row per (sample_id, window) with computed features.
    """
    has_sample = "sample_id" in meth_df.columns
    results: list[dict[str, Any]] = []

    groups = meth_df.groupby("chr") if not has_sample else meth_df.groupby(["sample_id", "chr"])

    for group_key, grp in groups:
        if has_sample:
            sample_id, chrom = group_key
        else:
            chrom = group_key
            sample_id = "single"

        chrom_windows = windows[windows["chr"] == chrom]
        if chrom_windows.empty:
            continue

        positions = grp["start"].values
        betas = grp["beta"].values.astype(float)
        coverages = grp["coverage"].values.astype(float) if "coverage" in grp.columns else None

        for _, win in chrom_windows.iterrows():
            ws, we = int(win["window_start"]), int(win["window_end"])
            mask = (positions >= ws) & (positions < we)
            n_cpgs = int(mask.sum())

            row: dict[str, Any] = {
                "chr": chrom,
                "window_start": ws,
                "window_end": we,
            }
            if has_sample:
                row["sample_id"] = sample_id

            if n_cpgs == 0:
                row.update({
                    "mean_beta": np.nan,
                    "median_beta": np.nan,
                    "var_beta": np.nan,
                    "n_cpgs": 0,
                    "mean_coverage": np.nan,
                    "frac_high_meth": np.nan,
                    "frac_low_meth": np.nan,
                })
            else:
                b = betas[mask]
                row["mean_beta"] = float(np.nanmean(b))
                row["median_beta"] = float(np.nanmedian(b))
                row["var_beta"] = float(np.nanvar(b))
                row["n_cpgs"] = n_cpgs
                row["mean_coverage"] = float(np.nanmean(coverages[mask])) if coverages is not None else np.nan
                row["frac_high_meth"] = float(np.sum(b >= high_meth_threshold) / n_cpgs)
                row["frac_low_meth"] = float(np.sum(b <= low_meth_threshold) / n_cpgs)

            results.append(row)

    feat_df = pd.DataFrame(results)
    logger.info(f"Computed features for {len(feat_df):,} windows")
    return feat_df


# ── Optimised vectorised version for large datasets ──────────────────────
def compute_window_features_fast(
    meth_df: pd.DataFrame,
    window_size: int = 1000,
    stride: int = 1000,
    min_cpgs: int = 3,
    high_meth_threshold: float = 0.8,
    low_meth_threshold: float = 0.2,
) -> pd.DataFrame:
    """
    Vectorised windowing — assigns each site to its window via integer
    division, then groups.  Much faster than the iterative approach.
    """
    df = meth_df.copy()
    df["window_start"] = (df["start"] // stride) * stride
    df["window_end"] = df["window_start"] + window_size

    has_sample = "sample_id" in df.columns
    group_cols = ["sample_id", "chr", "window_start", "window_end"] if has_sample else ["chr", "window_start", "window_end"]

    agg: dict[str, Any] = {
        "beta": ["mean", "median", "var", "count"],
    }
    if "coverage" in df.columns:
        agg["coverage"] = "mean"

    grouped = df.groupby(group_cols).agg(agg).reset_index()
    # Flatten multi-level columns
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in grouped.columns
    ]

    rename = {
        "beta_mean": "mean_beta",
        "beta_median": "median_beta",
        "beta_var": "var_beta",
        "beta_count": "n_cpgs",
        "coverage_mean": "mean_coverage",
    }
    grouped = grouped.rename(columns=rename)

    # Fraction high / low methylated
    def _fracs(sub: pd.DataFrame) -> pd.Series:
        b = sub["beta"].dropna()
        n = len(b)
        if n == 0:
            return pd.Series({"frac_high_meth": np.nan, "frac_low_meth": np.nan})
        return pd.Series({
            "frac_high_meth": float((b >= high_meth_threshold).sum() / n),
            "frac_low_meth": float((b <= low_meth_threshold).sum() / n),
        })

    fracs = df.groupby(group_cols).apply(_fracs, include_groups=False).reset_index()
    grouped = grouped.merge(fracs, on=group_cols, how="left")

    # Filter windows with too few CpGs
    n_before = len(grouped)
    grouped = grouped[grouped["n_cpgs"] >= min_cpgs].copy()
    logger.info(f"Kept {len(grouped):,}/{n_before:,} windows with ≥{min_cpgs} CpGs")

    return grouped


# ── Optional FASTA sequence extraction ───────────────────────────────────
def extract_sequences(
    windows: pd.DataFrame,
    fasta_path: str | Path,
) -> pd.Series:
    """
    Extract the reference DNA sequence for each window from a FASTA file.
    Returns a Series of uppercase strings aligned with the windows index.
    """
    from pyfaidx import Fasta

    fa = Fasta(str(fasta_path))
    seqs: list[str] = []
    for _, row in windows.iterrows():
        chrom = str(row["chr"])
        ws = int(row["window_start"])
        we = int(row["window_end"])
        try:
            seq = str(fa[chrom][ws:we]).upper()
        except (KeyError, ValueError):
            seq = "N" * (we - ws)
        seqs.append(seq)
    logger.info(f"Extracted sequences for {len(seqs):,} windows")
    return pd.Series(seqs, index=windows.index, name="sequence")


# ── CLI entry-point ──────────────────────────────────────────────────────
def run_windowing(
    input_path: str,
    config_path: str,
    output_dir: str,
    fasta_path: str | None = None,
) -> None:
    """Orchestrate genomic windowing (called from CLI)."""
    cfg = load_config(config_path)
    win_cfg = cfg.get("windowing", {})
    log_cfg = cfg.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("log_file"))

    window_size = win_cfg.get("window_size", 1000)
    stride = win_cfg.get("stride", window_size)
    min_cpgs = win_cfg.get("min_cpgs_per_window", 3)

    # Load methylation data
    p = Path(input_path)
    if p.suffix == ".parquet":
        meth = pd.read_parquet(p)
    else:
        meth = pd.read_csv(p, sep="\t")
    logger.info(f"Loaded {len(meth):,} methylation records from {p.name}")

    # Filter chromosomes if specified
    chroms = win_cfg.get("chromosomes")
    if chroms:
        meth = meth[meth["chr"].isin(chroms)]

    # Compute windowed features
    feat = compute_window_features_fast(
        meth,
        window_size=window_size,
        stride=stride,
        min_cpgs=min_cpgs,
    )

    # Optional sequence extraction
    if fasta_path or win_cfg.get("extract_sequence"):
        fasta = fasta_path or win_cfg.get("fasta_path")
        if fasta:
            feat["sequence"] = extract_sequences(feat, fasta)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "windows.parquet"
    feat.to_parquet(out_path, index=False)
    logger.info(f"Saved windowed features to {out_path}")
