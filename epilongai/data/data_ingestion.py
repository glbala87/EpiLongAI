"""
Phase B — Data ingestion for ONT methylation files.

Supports:
    1. bedMethyl-like files (standard modkit/nanopolish output)
    2. Tab-delimited files with chr, pos, N, X columns
    3. Optional sample metadata file

Key behaviours:
    - Chunked reading for large files
    - Automatic column-name standardisation
    - Beta computation from N and X when beta column is absent
    - Coordinate validation
    - Malformed-row warnings (not silent drops)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from epilongai.utils.config import load_config
from epilongai.utils.logging import setup_logging

# ── Column-name aliases (lower-cased) → canonical names ──────────────────
_ALIASES: dict[str, list[str]] = {
    "chr": ["chr", "chrom", "chromosome", "#chrom", "contig"],
    "start": ["start", "pos", "position", "chromstart"],
    "end": ["end", "chromend"],
    "strand": ["strand"],
    "coverage": ["coverage", "n", "total", "valid_cov", "n_valid", "depth"],
    "modified_count": ["modified_count", "x", "n_mod", "mod_count", "methylated"],
    "beta": ["beta", "freq", "methylation_frequency", "percent_modified", "frac"],
}


# ── Helpers ──────────────────────────────────────────────────────────────
def _build_rename_map(columns: list[str]) -> dict[str, str]:
    """Build a mapping from raw column names → canonical names."""
    lower_cols = {c.lower().strip(): c for c in columns}
    rename = {}
    for canonical, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in lower_cols:
                rename[lower_cols[alias]] = canonical
                break
    return rename


def _detect_format(path: Path) -> str:
    """Heuristic: peek at first line to decide bedMethyl vs generic tabular."""
    with open(path) as fh:
        header = fh.readline().strip().lower()
    if header.startswith("#") or "chrom" in header:
        return "bedmethyl"
    return "tabular"


def _validate_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with invalid genomic coordinates and warn."""
    n_before = len(df)
    # chr must be a non-empty string
    mask = df["chr"].astype(str).str.strip().astype(bool)
    # start must be non-negative integer
    mask &= pd.to_numeric(df["start"], errors="coerce").notna()
    df = df.loc[mask].copy()
    df["start"] = df["start"].astype(int)
    if "end" in df.columns:
        df["end"] = pd.to_numeric(df["end"], errors="coerce").fillna(df["start"] + 1).astype(int)
    else:
        df["end"] = df["start"] + 1
    dropped = n_before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with invalid coordinates")
    return df


def _compute_beta(df: pd.DataFrame) -> pd.DataFrame:
    """Compute beta = modified_count / coverage when beta column is absent."""
    if "beta" in df.columns and df["beta"].notna().any():
        # If beta values look like percentages (0-100), rescale to 0-1
        if df["beta"].max() > 1.5:
            logger.info("Beta values appear to be percentages — rescaling to [0, 1]")
            df["beta"] = df["beta"] / 100.0
        return df
    if "coverage" in df.columns and "modified_count" in df.columns:
        logger.info("Computing beta from modified_count / coverage")
        cov = pd.to_numeric(df["coverage"], errors="coerce")
        mod = pd.to_numeric(df["modified_count"], errors="coerce")
        df["beta"] = (mod / cov).where(cov > 0)
    else:
        logger.warning("Cannot compute beta: missing coverage and/or modified_count columns")
    return df


def _apply_coverage_filter(df: pd.DataFrame, min_coverage: int) -> pd.DataFrame:
    """Filter out sites below minimum coverage."""
    if "coverage" not in df.columns:
        return df
    df["coverage"] = pd.to_numeric(df["coverage"], errors="coerce")
    n_before = len(df)
    df = df[df["coverage"] >= min_coverage].copy()
    dropped = n_before - len(df)
    if dropped:
        logger.info(f"Filtered {dropped} sites below min_coverage={min_coverage}")
    return df


# ── Single-file parser ───────────────────────────────────────────────────
def parse_methylation_file(
    path: str | Path,
    *,
    file_format: str = "auto",
    separator: str = "\t",
    chunk_size: int = 500_000,
    min_coverage: int = 5,
    column_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Parse a single ONT methylation file into a standardised DataFrame.

    Returns columns: chr, start, end, strand (optional), coverage, modified_count, beta
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Methylation file not found: {path}")

    if file_format == "auto":
        file_format = _detect_format(path)
    logger.info(f"Parsing {path.name} as format={file_format}")

    # Read in chunks to limit memory
    chunks: list[pd.DataFrame] = []
    reader = pd.read_csv(
        path,
        sep=separator,
        comment="#",
        chunksize=chunk_size,
        low_memory=False,
        dtype=str,  # read everything as string first for safety
    )
    for i, chunk in enumerate(reader):
        # Rename columns
        rename_map = _build_rename_map(chunk.columns.tolist())
        if column_overrides:
            rename_map.update(column_overrides)
        chunk = chunk.rename(columns=rename_map)

        # Keep only known columns that exist
        keep = [c for c in ["chr", "start", "end", "strand", "coverage", "modified_count", "beta"] if c in chunk.columns]
        chunk = chunk[keep]
        chunks.append(chunk)
        if (i + 1) % 10 == 0:
            logger.debug(f"  … read {(i + 1) * chunk_size:,} rows")

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"  Total raw rows: {len(df):,}")

    # Numeric coercion for key columns
    for col in ["coverage", "modified_count", "beta"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _validate_coordinates(df)
    df = _apply_coverage_filter(df, min_coverage)
    df = _compute_beta(df)

    logger.info(f"  Rows after QC: {len(df):,}")
    return df


# ── Metadata parser ──────────────────────────────────────────────────────
def parse_metadata(path: str | Path) -> pd.DataFrame:
    """Parse sample metadata (TSV or CSV)."""
    path = Path(path)
    sep = "," if path.suffix == ".csv" else "\t"
    meta = pd.read_csv(path, sep=sep)
    required = {"sample_id"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"Metadata missing required columns: {missing}")
    logger.info(f"Loaded metadata with {len(meta)} samples, columns: {list(meta.columns)}")
    return meta


# ── Multi-sample merger ──────────────────────────────────────────────────
def merge_samples(
    input_dir: str | Path,
    *,
    metadata: pd.DataFrame | None = None,
    glob_pattern: str = "*.bed*",
    **parse_kwargs: Any,
) -> pd.DataFrame:
    """
    Parse all methylation files in a directory and merge into one DataFrame
    with a ``sample_id`` column derived from the file stem.
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(glob_pattern))
    if not files:
        # Try broader patterns
        for pat in ["*.tsv", "*.txt", "*.csv", "*.bed*"]:
            files = sorted(input_dir.glob(pat))
            if files:
                break
    if not files:
        raise FileNotFoundError(f"No methylation files found in {input_dir}")

    logger.info(f"Found {len(files)} methylation files in {input_dir}")

    frames: list[pd.DataFrame] = []
    for f in files:
        sample_id = f.stem.split(".")[0]
        try:
            df = parse_methylation_file(f, **parse_kwargs)
            df.insert(0, "sample_id", sample_id)
            frames.append(df)
        except Exception:
            logger.exception(f"Failed to parse {f.name} — skipping")

    merged = pd.concat(frames, ignore_index=True)
    logger.info(f"Merged dataset: {len(merged):,} rows across {merged['sample_id'].nunique()} samples")

    if metadata is not None:
        merged = merged.merge(metadata, on="sample_id", how="left")
        unmatched = merged["sample_id"][merged.get("group", pd.Series(dtype=str)).isna()].nunique()
        if unmatched:
            logger.warning(f"{unmatched} samples had no matching metadata")

    return merged


# ── CLI entry-point ──────────────────────────────────────────────────────
def run_ingestion(
    input_dir: str,
    metadata_path: str | None,
    config_path: str,
    output_dir: str,
) -> None:
    """Orchestrate the full ingestion pipeline (called from CLI)."""
    cfg = load_config(config_path)
    ing_cfg = cfg.get("ingestion", {})
    log_cfg = cfg.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("log_file"))

    metadata = parse_metadata(metadata_path) if metadata_path else None

    merged = merge_samples(
        input_dir,
        metadata=metadata,
        file_format=ing_cfg.get("file_format", "auto"),
        separator=ing_cfg.get("separator", "\t"),
        chunk_size=ing_cfg.get("chunk_size", 500_000),
        min_coverage=ing_cfg.get("min_coverage", 5),
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "methylation_merged.parquet"
    merged.to_parquet(out_path, index=False)
    logger.info(f"Saved merged methylation data to {out_path}")
