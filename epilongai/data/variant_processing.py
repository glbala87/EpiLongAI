"""
Phase N — Variant-aware modeling: VCF parsing and variant feature encoding.

HOW VARIANTS INFLUENCE METHYLATION PATTERNS
=============================================
1. **meQTLs (methylation quantitative trait loci)**: SNPs directly alter
   CpG dinucleotides (C→T at CpG sites) or disrupt transcription factor
   binding, changing local methylation.  ~20% of CpGs have a detectable
   meQTL in blood.

2. **Structural variants**: Large deletions/duplications can remove or
   duplicate entire CpG islands, fundamentally altering regional methylation.

3. **Repeat-element polymorphisms**: Alu/LINE insertions carry their own
   CpGs; polymorphic insertions create inter-individual methylation
   differences invisible without variant data.

4. **Gene-environment interaction**: Variants modulate how environmental
   exposures (e.g. smoking, nutrition) affect methylation — relevant for
   preterm birth.

BEST PRACTICES FOR SPARSE VARIANT SIGNALS
==========================================
- Most windows contain 0–2 variants → highly sparse input.
- Use embedding representations rather than dense one-hot to handle
  sparsity efficiently.
- Apply L1/elastic-net regularisation or attention-based gating so the
  model can learn to *ignore* variant channels when they carry no signal.
- For structural variants, encode presence/absence + size rather than
  exact breakpoints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


# =====================================================================
# VCF Parsing
# =====================================================================
def parse_vcf(
    vcf_path: str | Path,
    sample_id: str | None = None,
    min_qual: float = 20.0,
    variant_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Parse a VCF file into a flat DataFrame of variants.

    Parameters
    ----------
    vcf_path : path
        VCF or VCF.gz file.
    sample_id : str or None
        If multi-sample VCF, extract this sample.  If None, use first sample.
    min_qual : float
        Minimum QUAL to keep a variant.
    variant_types : list or None
        Filter to these types: 'SNP', 'INDEL', 'SV'.  None → keep all.

    Returns
    -------
    DataFrame with columns:
        chr, pos, ref, alt, qual, variant_type, genotype (0/1/2),
        allele_freq (if INFO/AF present), sample_id
    """
    path = Path(vcf_path)
    if not path.exists():
        raise FileNotFoundError(f"VCF not found: {path}")

    records: list[dict[str, Any]] = []
    sample_col_idx: int | None = None
    sample_name: str = sample_id or ""

    open_fn = _open_vcf(path)
    with open_fn(str(path), "rt") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.strip().split("\t")
                if len(header) > 9:
                    samples = header[9:]
                    if sample_id and sample_id in samples:
                        sample_col_idx = 9 + samples.index(sample_id)
                    else:
                        sample_col_idx = 9
                        sample_name = samples[0] if samples else "unknown"
                continue

            fields = line.strip().split("\t")
            if len(fields) < 8:
                continue

            chrom, pos_str, _, ref, alt, qual_str = fields[0], fields[1], fields[2], fields[3], fields[4], fields[5]

            # Quality filter
            try:
                qual = float(qual_str) if qual_str != "." else 0.0
            except ValueError:
                qual = 0.0
            if qual < min_qual and qual_str != ".":
                continue

            # Determine variant type
            vtype = _classify_variant(ref, alt)
            if variant_types and vtype not in variant_types:
                continue

            # Parse genotype
            gt = "0/0"
            if sample_col_idx is not None and len(fields) > sample_col_idx:
                gt_field = fields[sample_col_idx].split(":")[0]
                gt = gt_field.replace("|", "/")

            dosage = _genotype_to_dosage(gt)

            # Parse allele frequency from INFO
            af = _parse_af(fields[7])

            records.append({
                "chr": chrom,
                "pos": int(pos_str),
                "ref": ref,
                "alt": alt,
                "qual": qual,
                "variant_type": vtype,
                "genotype": gt,
                "dosage": dosage,
                "allele_freq": af,
                "sample_id": sample_name,
            })

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df):,} variants from {path.name} (sample={sample_name})")
    return df


def _open_vcf(path: Path):
    """Return appropriate open function for .vcf or .vcf.gz."""
    if path.suffix == ".gz" or str(path).endswith(".vcf.gz"):
        import gzip
        return gzip.open
    return open


def _classify_variant(ref: str, alt: str) -> str:
    """Classify variant type."""
    alts = alt.split(",")
    alt0 = alts[0]
    if alt0.startswith("<"):
        return "SV"
    if len(ref) == 1 and len(alt0) == 1:
        return "SNP"
    return "INDEL"


def _genotype_to_dosage(gt: str) -> int:
    """Convert genotype string to allele dosage (0, 1, or 2)."""
    parts = gt.replace("|", "/").split("/")
    try:
        return sum(1 for a in parts if a not in ("0", "."))
    except Exception:
        return 0


def _parse_af(info: str) -> float | None:
    """Extract AF from INFO field."""
    for field in info.split(";"):
        if field.startswith("AF="):
            try:
                return float(field.split("=")[1].split(",")[0])
            except (ValueError, IndexError):
                return None
    return None


# =====================================================================
# Map Variants to Genomic Windows
# =====================================================================
def map_variants_to_windows(
    variants: pd.DataFrame,
    window_size: int = 1000,
    stride: int = 1000,
) -> pd.DataFrame:
    """
    Assign each variant to its genomic window and compute per-window
    variant features.

    Returns one row per (sample_id, window) with columns:
        chr, window_start, window_end, n_variants, n_snps, n_indels,
        mean_dosage, max_dosage, has_coding_variant, mean_af
    """
    df = variants.copy()
    df["window_start"] = (df["pos"] // stride) * stride
    df["window_end"] = df["window_start"] + window_size

    has_sample = "sample_id" in df.columns
    group_cols = (["sample_id"] if has_sample else []) + ["chr", "window_start", "window_end"]

    agg = df.groupby(group_cols).agg(
        n_variants=("pos", "count"),
        n_snps=("variant_type", lambda x: (x == "SNP").sum()),
        n_indels=("variant_type", lambda x: (x == "INDEL").sum()),
        mean_dosage=("dosage", "mean"),
        max_dosage=("dosage", "max"),
        mean_af=("allele_freq", lambda x: x.dropna().mean() if x.dropna().any() else 0.0),
    ).reset_index()

    logger.info(f"Mapped variants to {len(agg):,} windows")
    return agg


# =====================================================================
# Variant Encoding for Model Input
# =====================================================================
VARIANT_FEATURE_COLS = [
    "n_variants", "n_snps", "n_indels",
    "mean_dosage", "max_dosage", "mean_af",
]


def encode_variants_for_model(
    variant_windows: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """
    Convert variant window features to a numeric array for model input.

    Returns array of shape (n_windows, n_features).
    """
    cols = feature_cols or VARIANT_FEATURE_COLS
    available = [c for c in cols if c in variant_windows.columns]
    return variant_windows[available].fillna(0).values.astype(np.float32)


# =====================================================================
# Batch VCF Processing
# =====================================================================
def process_vcf_directory(
    vcf_dir: str | Path,
    window_size: int = 1000,
    stride: int = 1000,
    glob_pattern: str = "*.vcf*",
) -> pd.DataFrame:
    """
    Parse all VCF files in a directory and produce windowed variant features.
    """
    vcf_dir = Path(vcf_dir)
    files = sorted(vcf_dir.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(f"No VCF files found in {vcf_dir}")

    logger.info(f"Processing {len(files)} VCF files from {vcf_dir}")

    frames: list[pd.DataFrame] = []
    for f in files:
        sample_id = f.stem.split(".")[0]
        try:
            variants = parse_vcf(f, sample_id=sample_id)
            if not variants.empty:
                windowed = map_variants_to_windows(variants, window_size, stride)
                frames.append(windowed)
        except Exception:
            logger.exception(f"Failed to process {f.name} — skipping")

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True)
    logger.info(f"Total variant windows across all samples: {len(merged):,}")
    return merged
