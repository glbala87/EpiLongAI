#!/usr/bin/env python3
"""
Generate synthetic ONT methylation, VCF, expression, and metadata files
for end-to-end pipeline testing.

Usage:
    python scripts/generate_synthetic_data.py --output data/synthetic --n-samples 20
    snakemake --cores 4 -s workflow/Snakefile --config raw_dir=data/synthetic/methylation metadata=data/synthetic/metadata.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_methylation_file(
    sample_id: str,
    output_dir: Path,
    n_sites: int = 5000,
    chromosomes: list[str] | None = None,
    group: str = "FTB",
    seed: int = 42,
) -> Path:
    """Generate a synthetic bedMethyl-like file for one sample."""
    rng = np.random.default_rng(seed)
    chroms = chromosomes or [f"chr{i}" for i in range(1, 23)]

    # Generate clustered CpG positions (realistic: CpGs cluster in islands)
    n_clusters = n_sites // 10
    cluster_starts = rng.integers(1000, 10_000_000, size=n_clusters)
    cluster_chroms = rng.choice(chroms, size=n_clusters)

    records = []
    for ci in range(n_clusters):
        base_pos = int(cluster_starts[ci])
        chrom = cluster_chroms[ci]
        for offset in range(10):
            pos = base_pos + offset * int(rng.integers(50, 200))
            coverage = int(rng.integers(5, 100))
            # PTB samples get slightly higher methylation (simulates biological signal)
            if group == "PTB":
                beta = float(rng.beta(6, 3))  # mean ~0.67
            else:
                beta = float(rng.beta(3, 3))  # mean ~0.50
            modified = int(round(beta * coverage))
            records.append({
                "chr": chrom,
                "start": pos,
                "end": pos + 1,
                "strand": rng.choice(["+", "-"]),
                "coverage": coverage,
                "modified_count": modified,
                "beta": round(beta, 4),
            })

    df = pd.DataFrame(records).sort_values(["chr", "start"]).reset_index(drop=True)
    out_path = output_dir / f"{sample_id}.bed"
    df.to_csv(out_path, sep="\t", index=False)
    return out_path


def generate_vcf_file(
    sample_id: str,
    output_dir: Path,
    n_variants: int = 500,
    chromosomes: list[str] | None = None,
    seed: int = 42,
) -> Path:
    """Generate a synthetic VCF file for one sample."""
    rng = np.random.default_rng(seed)
    chroms = chromosomes or [f"chr{i}" for i in range(1, 23)]
    bases = ["A", "C", "G", "T"]

    lines = [
        "##fileformat=VCFv4.2",
        '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">',
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_id}",
    ]

    for _ in range(n_variants):
        chrom = rng.choice(chroms)
        pos = rng.integers(1000, 250_000_000)
        ref = rng.choice(bases)
        alt = rng.choice([b for b in bases if b != ref])
        qual = rng.integers(20, 99)
        af = round(rng.uniform(0.001, 0.5), 4)
        gt = rng.choice(["0/0", "0/1", "1/1"], p=[0.5, 0.35, 0.15])
        lines.append(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t{qual}\tPASS\tAF={af}\tGT\t{gt}")

    out_path = output_dir / f"{sample_id}.vcf"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def generate_expression_matrix(
    sample_ids: list[str],
    output_dir: Path,
    n_genes: int = 200,
    seed: int = 42,
) -> Path:
    """Generate a synthetic gene expression matrix (TPM-like)."""
    rng = np.random.default_rng(seed)
    gene_ids = [f"GENE_{i:04d}" for i in range(n_genes)]
    data = {"gene_id": gene_ids}
    for sid in sample_ids:
        data[sid] = np.abs(rng.normal(50, 30, n_genes)).round(2)
    df = pd.DataFrame(data)
    out_path = output_dir / "expression_matrix.tsv"
    df.to_csv(out_path, sep="\t", index=False)
    return out_path


def generate_gene_annotations(
    output_dir: Path,
    n_genes: int = 200,
    chromosomes: list[str] | None = None,
    seed: int = 42,
) -> Path:
    """Generate a synthetic gene annotation BED file."""
    rng = np.random.default_rng(seed)
    chroms = chromosomes or [f"chr{i}" for i in range(1, 23)]
    records = []
    for i in range(n_genes):
        chrom = rng.choice(chroms)
        start = rng.integers(1000, 250_000_000)
        end = start + rng.integers(1000, 50000)
        records.append({"chr": chrom, "start": start, "end": end, "gene_id": f"GENE_{i:04d}"})
    df = pd.DataFrame(records)
    out_path = output_dir / "gene_annotations.bed"
    df.to_csv(out_path, sep="\t", index=False, header=False)
    return out_path


def generate_metadata(
    sample_ids: list[str],
    groups: list[str],
    output_dir: Path,
) -> Path:
    """Generate sample metadata TSV."""
    meta = pd.DataFrame({
        "sample_id": sample_ids,
        "group": groups,
        "phenotype": ["preterm_birth" if g == "PTB" else "full_term_birth" for g in groups],
        "batch": [f"batch{i % 3 + 1}" for i in range(len(sample_ids))],
        "source": ["cord_blood" if i % 2 == 0 else "maternal_blood" for i in range(len(sample_ids))],
    })
    out_path = output_dir / "metadata.tsv"
    meta.to_csv(out_path, sep="\t", index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data for EpiLongAI")
    parser.add_argument("--output", default="data/synthetic", help="Output directory")
    parser.add_argument("--n-samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--n-sites", type=int, default=5000, help="Methylation sites per sample")
    parser.add_argument("--n-variants", type=int, default=500, help="Variants per sample")
    parser.add_argument("--n-genes", type=int, default=200, help="Number of genes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    out = Path(args.output)
    meth_dir = out / "methylation"
    vcf_dir = out / "vcf"
    meth_dir.mkdir(parents=True, exist_ok=True)
    vcf_dir.mkdir(parents=True, exist_ok=True)

    # Create balanced cohort
    n_ptb = args.n_samples // 2
    n_ftb = args.n_samples - n_ptb
    sample_ids = [f"PTB_{i:03d}" for i in range(n_ptb)] + [f"FTB_{i:03d}" for i in range(n_ftb)]
    groups = ["PTB"] * n_ptb + ["FTB"] * n_ftb

    print(f"Generating synthetic data for {args.n_samples} samples → {out}")

    # Methylation files
    for i, (sid, grp) in enumerate(zip(sample_ids, groups)):
        generate_methylation_file(
            sid, meth_dir, n_sites=args.n_sites, group=grp, seed=args.seed + i
        )
    print(f"  ✓ {args.n_samples} methylation files → {meth_dir}")

    # VCF files
    for i, sid in enumerate(sample_ids):
        generate_vcf_file(sid, vcf_dir, n_variants=args.n_variants, seed=args.seed + 1000 + i)
    print(f"  ✓ {args.n_samples} VCF files → {vcf_dir}")

    # Expression matrix
    generate_expression_matrix(sample_ids, out, n_genes=args.n_genes, seed=args.seed)
    print(f"  ✓ Expression matrix → {out / 'expression_matrix.tsv'}")

    # Gene annotations
    generate_gene_annotations(out, n_genes=args.n_genes, seed=args.seed)
    print(f"  ✓ Gene annotations → {out / 'gene_annotations.bed'}")

    # Metadata
    generate_metadata(sample_ids, groups, out)
    print(f"  ✓ Metadata → {out / 'metadata.tsv'}")

    print(f"\nDone! To run the pipeline:")
    print(f"  snakemake --cores 4 -s workflow/Snakefile \\")
    print(f"    --config raw_dir={meth_dir} metadata={out / 'metadata.tsv'}")


if __name__ == "__main__":
    main()
