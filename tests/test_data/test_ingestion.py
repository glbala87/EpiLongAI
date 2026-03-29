"""Tests for data ingestion module."""

from pathlib import Path

import pandas as pd
import pytest

from epilongai.data.data_ingestion import (
    _build_rename_map,
    _compute_beta,
    _validate_coordinates,
    parse_methylation_file,
)


@pytest.fixture
def sample_bedmethyl(tmp_path: Path) -> Path:
    """Create a minimal bedMethyl-like test file."""
    content = "chrom\tstart\tend\tstrand\tcoverage\tmod_count\n"
    content += "chr1\t1000\t1001\t+\t20\t15\n"
    content += "chr1\t2000\t2001\t+\t10\t3\n"
    content += "chr1\t3000\t3001\t-\t30\t25\n"
    p = tmp_path / "sample.bed"
    p.write_text(content)
    return p


@pytest.fixture
def sample_tabular(tmp_path: Path) -> Path:
    """Create a minimal tab-delimited methylation file."""
    content = "chr\tpos\tN\tX\n"
    content += "chr1\t1000\t20\t15\n"
    content += "chr1\t2000\t10\t3\n"
    content += "chrX\t5000\t50\t40\n"
    p = tmp_path / "sample.tsv"
    p.write_text(content)
    return p


class TestRenameMap:
    def test_standard_columns(self):
        cols = ["chrom", "start", "end", "coverage", "mod_count"]
        rmap = _build_rename_map(cols)
        assert rmap["chrom"] == "chr"
        assert rmap["coverage"] == "coverage"

    def test_pos_column(self):
        cols = ["chr", "pos", "N", "X"]
        rmap = _build_rename_map(cols)
        assert rmap["pos"] == "start"
        assert rmap["N"] == "coverage"
        assert rmap["X"] == "modified_count"


class TestValidateCoordinates:
    def test_valid(self):
        df = pd.DataFrame({"chr": ["chr1", "chr2"], "start": [100, 200]})
        result = _validate_coordinates(df)
        assert len(result) == 2

    def test_invalid_chr(self):
        df = pd.DataFrame({"chr": ["chr1", ""], "start": [100, 200]})
        result = _validate_coordinates(df)
        assert len(result) == 1


class TestComputeBeta:
    def test_from_coverage_and_modified(self):
        df = pd.DataFrame({"coverage": [20, 10], "modified_count": [15, 3]})
        result = _compute_beta(df)
        assert "beta" in result.columns
        assert abs(result["beta"].iloc[0] - 0.75) < 1e-6

    def test_existing_beta_preserved(self):
        df = pd.DataFrame({"beta": [0.5, 0.8]})
        result = _compute_beta(df)
        assert result["beta"].iloc[0] == 0.5


class TestParseFile:
    def test_parse_bedmethyl(self, sample_bedmethyl: Path):
        df = parse_methylation_file(sample_bedmethyl, min_coverage=1)
        assert len(df) == 3
        assert "beta" in df.columns
        assert "chr" in df.columns

    def test_parse_tabular(self, sample_tabular: Path):
        df = parse_methylation_file(sample_tabular, min_coverage=1)
        assert len(df) == 3
        assert "beta" in df.columns

    def test_coverage_filter(self, sample_tabular: Path):
        df = parse_methylation_file(sample_tabular, min_coverage=15)
        assert len(df) == 2  # only rows with coverage >= 15
