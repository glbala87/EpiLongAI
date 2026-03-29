"""Tests for RNA-seq integration module."""

from pathlib import Path

import pandas as pd
import pytest

from epilongai.data.rna_integration import (
    map_expression_to_windows,
    merge_omics_features,
    normalize_expression,
    parse_expression_file,
)


@pytest.fixture
def expression_file(tmp_path: Path) -> Path:
    """Create a minimal gene expression matrix."""
    content = "gene_id\tsample1\tsample2\tsample3\n"
    content += "GENE_A\t100.0\t200.0\t50.0\n"
    content += "GENE_B\t0.5\t1.0\t0.0\n"
    content += "GENE_C\t500.0\t600.0\t450.0\n"
    p = tmp_path / "expression.tsv"
    p.write_text(content)
    return p


@pytest.fixture
def gene_bed(tmp_path: Path) -> Path:
    """Create a minimal gene annotation BED."""
    content = "chr1\t1000\t2000\tGENE_A\n"
    content += "chr1\t5000\t6000\tGENE_B\n"
    content += "chr2\t3000\t4000\tGENE_C\n"
    p = tmp_path / "genes.bed"
    p.write_text(content)
    return p


class TestParseExpression:
    def test_basic(self, expression_file: Path):
        df = parse_expression_file(expression_file)
        assert "gene_id" in df.columns
        assert "sample_id" in df.columns
        assert "expression" in df.columns
        assert df["gene_id"].nunique() == 3
        assert df["sample_id"].nunique() == 3
        assert len(df) == 9  # 3 genes x 3 samples


class TestNormalizeExpression:
    def test_log2_tpm(self):
        df = pd.DataFrame({
            "gene_id": ["A", "A", "B", "B"],
            "sample_id": ["s1", "s2", "s1", "s2"],
            "expression": [100.0, 200.0, 0.0, 50.0],
        })
        result = normalize_expression(df, method="log2_tpm")
        assert all(result["expression"] >= 0)  # log2(0 + 1) = 0, so >= 0

    def test_zscore(self):
        df = pd.DataFrame({
            "gene_id": ["A", "A", "A"],
            "sample_id": ["s1", "s2", "s3"],
            "expression": [10.0, 20.0, 30.0],
        })
        result = normalize_expression(df, method="zscore")
        # z-score should have mean ~0
        assert abs(result["expression"].mean()) < 1e-10


class TestMapExpressionToWindows:
    def test_basic(self, expression_file: Path, gene_bed: Path):
        from epilongai.data.rna_integration import load_gene_annotations

        expr = parse_expression_file(expression_file)
        genes = load_gene_annotations(bed_path=gene_bed)
        result = map_expression_to_windows(expr, genes, window_size=1000, stride=1000)
        assert "expression" in result.columns
        assert "chr" in result.columns
        assert len(result) > 0


class TestMergeOmicsFeatures:
    def test_merge_expression(self):
        meth = pd.DataFrame({
            "sample_id": ["s1", "s1"],
            "chr": ["chr1", "chr1"],
            "window_start": [0, 1000],
            "window_end": [1000, 2000],
            "mean_beta": [0.5, 0.8],
        })
        expr = pd.DataFrame({
            "sample_id": ["s1"],
            "chr": ["chr1"],
            "window_start": [0],
            "window_end": [1000],
            "expression": [100.0],
        })
        merged = merge_omics_features(meth, expression_windows=expr)
        assert "expression" in merged.columns
        assert len(merged) == 2  # left join keeps all meth rows
