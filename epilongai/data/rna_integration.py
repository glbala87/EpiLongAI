"""
Phase P — RNA-seq expression data integration with ONT methylation.

BIOLOGICAL RELATIONSHIP BETWEEN METHYLATION AND EXPRESSION
=============================================================
1. **Promoter methylation silences genes**: Hypermethylation of CpG islands
   at gene promoters typically represses transcription.  This is a core
   mechanism in cancer, development, and preterm-birth–related pathways.

2. **Gene body methylation correlates positively with expression**: Unlike
   promoters, methylation within gene bodies often tracks with active
   transcription — possibly by suppressing spurious intragenic transcription.

3. **Enhancer methylation reflects cell-type identity**: Demethylated enhancers
   mark active regulatory elements.  Combining enhancer methylation with
   expression of target genes provides a powerful readout of regulatory state.

4. **Methylation is the *cause*; expression is the *effect***: Modeling both
   jointly lets the network learn causal relationships that neither modality
   alone captures.

WHEN MULTIMODAL (METH + RNA) LEARNING IMPROVES PERFORMANCE
=============================================================
- When the label depends on *regulatory disruption* (e.g. PTB driven by
  aberrant gene regulation), multimodal > either unimodal.
- When cohorts are small, adding RNA-seq provides complementary signal that
  regularises the methylation model.
- When there are *trans* effects: a distal methylation change affects a
  gene's expression far away — the model needs both modalities to connect them.
- Diminishing returns if methylation and expression are highly redundant
  for the specific phenotype.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger


# =====================================================================
# Expression Data Parsing
# =====================================================================
def parse_expression_file(
    path: str | Path,
    id_column: str = "gene_id",
    sample_columns: list[str] | None = None,
    separator: str = "\t",
) -> pd.DataFrame:
    """
    Parse a gene-level expression matrix.

    Expected format: rows = genes, columns = samples (wide format).
    Returns a long-format DataFrame: sample_id, gene_id, expression.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=separator)

    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found. Available: {list(df.columns[:10])}")

    if sample_columns is None:
        # Assume all non-id columns are samples
        sample_columns = [c for c in df.columns if c != id_column]

    long = df.melt(
        id_vars=[id_column],
        value_vars=sample_columns,
        var_name="sample_id",
        value_name="expression",
    )
    long = long.rename(columns={id_column: "gene_id"})
    logger.info(f"Parsed expression: {long['gene_id'].nunique()} genes × {long['sample_id'].nunique()} samples")
    return long


# =====================================================================
# Expression Normalisation
# =====================================================================
def normalize_expression(
    expr: pd.DataFrame,
    method: str = "log2_tpm",
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """
    Normalise expression values.

    Methods:
        - 'log2_tpm': log2(expression + pseudocount) — assumes input is TPM
        - 'zscore': per-gene z-score across samples
        - 'quantile': rank-based inverse-normal transform per gene
    """
    df = expr.copy()

    if method == "log2_tpm":
        df["expression"] = np.log2(df["expression"].clip(lower=0) + pseudocount)
    elif method == "zscore":
        stats = df.groupby("gene_id")["expression"].agg(["mean", "std"])
        df = df.merge(stats, on="gene_id")
        df["expression"] = (df["expression"] - df["mean"]) / df["std"].replace(0, 1)
        df = df.drop(columns=["mean", "std"])
    elif method == "quantile":
        from scipy.stats import norm, rankdata
        def _qnorm(x: pd.Series) -> pd.Series:
            r = rankdata(x)
            return pd.Series(norm.ppf(r / (len(r) + 1)), index=x.index)
        df["expression"] = df.groupby("gene_id")["expression"].transform(_qnorm)
    else:
        raise ValueError(f"Unknown normalisation method: {method}")

    logger.info(f"Normalised expression using method='{method}'")
    return df


# =====================================================================
# Map Expression to Genomic Windows
# =====================================================================
def load_gene_annotations(
    gtf_path: str | Path | None = None,
    bed_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load gene coordinate annotations.

    Supports GTF (extracts gene records) or simple BED format:
        chr, start, end, gene_id
    """
    if bed_path:
        genes = pd.read_csv(bed_path, sep="\t", names=["chr", "start", "end", "gene_id"])
    elif gtf_path:
        records = []
        with open(gtf_path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 9 or parts[2] != "gene":
                    continue
                attrs = parts[8]
                gene_id = _extract_gtf_attr(attrs, "gene_id")
                if gene_id:
                    records.append({
                        "chr": parts[0],
                        "start": int(parts[3]),
                        "end": int(parts[4]),
                        "gene_id": gene_id,
                    })
        genes = pd.DataFrame(records)
    else:
        raise ValueError("Must provide gtf_path or bed_path")

    logger.info(f"Loaded annotations for {len(genes)} genes")
    return genes


def _extract_gtf_attr(attrs: str, key: str) -> str | None:
    """Extract a value from GTF attribute string."""
    for field in attrs.split(";"):
        field = field.strip()
        if field.startswith(key):
            return field.split('"')[1] if '"' in field else field.split(" ")[-1]
    return None


def map_expression_to_windows(
    expression: pd.DataFrame,
    gene_annotations: pd.DataFrame,
    window_size: int = 1000,
    stride: int = 1000,
    aggregation: str = "max",
) -> pd.DataFrame:
    """
    Map gene-level expression to genomic windows.

    Strategy: for each window, find genes whose body overlaps, and assign
    the expression value using the specified aggregation.

    Parameters
    ----------
    expression : DataFrame
        Long format: sample_id, gene_id, expression.
    gene_annotations : DataFrame
        chr, start, end, gene_id.
    aggregation : str
        'max', 'mean', or 'sum' when multiple genes overlap a window.

    Returns
    -------
    DataFrame with: sample_id, chr, window_start, window_end, expression.
    """
    # Assign genes to windows (gene TSS → window)
    genes = gene_annotations.copy()
    genes["window_start"] = (genes["start"] // stride) * stride
    genes["window_end"] = genes["window_start"] + window_size

    # Merge with expression
    merged = expression.merge(
        genes[["gene_id", "chr", "window_start", "window_end"]],
        on="gene_id",
        how="inner",
    )

    has_sample = "sample_id" in merged.columns
    group_cols = (["sample_id"] if has_sample else []) + ["chr", "window_start", "window_end"]

    agg_fn = {"max": "max", "mean": "mean", "sum": "sum"}[aggregation]
    result = merged.groupby(group_cols)["expression"].agg(agg_fn).reset_index()

    logger.info(f"Mapped expression to {len(result):,} window-sample pairs")
    return result


# =====================================================================
# Multi-omics Feature Merger
# =====================================================================
def merge_omics_features(
    methylation_windows: pd.DataFrame,
    expression_windows: pd.DataFrame | None = None,
    variant_windows: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Merge methylation, expression, and variant features into a single
    DataFrame aligned by (sample_id, chr, window_start, window_end).
    """
    merged = methylation_windows.copy()
    join_cols = [c for c in ["sample_id", "chr", "window_start", "window_end"] if c in merged.columns]

    if expression_windows is not None:
        merged = merged.merge(expression_windows, on=join_cols, how="left", suffixes=("", "_rna"))
        logger.info(f"Merged expression features; shape: {merged.shape}")

    if variant_windows is not None:
        merged = merged.merge(variant_windows, on=join_cols, how="left", suffixes=("", "_var"))
        logger.info(f"Merged variant features; shape: {merged.shape}")

    return merged


# =====================================================================
# RNA-seq Encoder (Model Branch)
# =====================================================================
class RNASeqEncoder(nn.Module):
    """
    MLP encoder for per-window expression features.

    Can encode a single expression value or multiple expression-derived
    features (e.g. expression + expression_zscore + is_highly_expressed).
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [32, 16]
        layers: list[nn.Module] = []
        in_d = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
