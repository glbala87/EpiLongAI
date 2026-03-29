"""Tests for variant processing module."""

from pathlib import Path

import pandas as pd
import pytest

from epilongai.data.variant_processing import (
    _classify_variant,
    _genotype_to_dosage,
    _parse_af,
    encode_variants_for_model,
    map_variants_to_windows,
    parse_vcf,
)


@pytest.fixture
def sample_vcf(tmp_path: Path) -> Path:
    """Create a minimal VCF test file."""
    content = """##fileformat=VCFv4.2
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1
chr1\t1000\t.\tA\tG\t30\tPASS\tAF=0.05\tGT\t0/1
chr1\t2000\t.\tC\tT\t50\tPASS\tAF=0.10\tGT\t1/1
chr1\t3000\t.\tAG\tA\t25\tPASS\tAF=0.02\tGT\t0/1
chr2\t5000\t.\tT\tC\t10\tPASS\t.\tGT\t0/0
"""
    p = tmp_path / "sample.vcf"
    p.write_text(content)
    return p


class TestClassifyVariant:
    def test_snp(self):
        assert _classify_variant("A", "G") == "SNP"

    def test_indel(self):
        assert _classify_variant("AG", "A") == "INDEL"

    def test_sv(self):
        assert _classify_variant("A", "<DEL>") == "SV"


class TestGenotypeToDosage:
    def test_ref(self):
        assert _genotype_to_dosage("0/0") == 0

    def test_het(self):
        assert _genotype_to_dosage("0/1") == 1

    def test_hom_alt(self):
        assert _genotype_to_dosage("1/1") == 2

    def test_phased(self):
        assert _genotype_to_dosage("0|1") == 1


class TestParseAF:
    def test_present(self):
        assert _parse_af("AF=0.05;DP=100") == pytest.approx(0.05)

    def test_missing(self):
        assert _parse_af("DP=100") is None

    def test_dot(self):
        assert _parse_af(".") is None


class TestParseVcf:
    def test_basic(self, sample_vcf: Path):
        df = parse_vcf(sample_vcf, min_qual=0)
        assert len(df) == 4
        assert "chr" in df.columns
        assert "dosage" in df.columns
        assert "variant_type" in df.columns

    def test_quality_filter(self, sample_vcf: Path):
        df = parse_vcf(sample_vcf, min_qual=20)
        assert len(df) == 3  # QUAL=10 row filtered

    def test_type_filter(self, sample_vcf: Path):
        df = parse_vcf(sample_vcf, min_qual=0, variant_types=["SNP"])
        assert all(df["variant_type"] == "SNP")


class TestMapVariantsToWindows:
    def test_basic(self):
        variants = pd.DataFrame({
            "sample_id": ["s1"] * 3,
            "chr": ["chr1"] * 3,
            "pos": [100, 200, 1500],
            "variant_type": ["SNP", "SNP", "INDEL"],
            "dosage": [1, 2, 1],
            "allele_freq": [0.05, 0.10, None],
        })
        result = map_variants_to_windows(variants, window_size=1000, stride=1000)
        assert "n_variants" in result.columns
        assert "mean_dosage" in result.columns
        assert len(result) == 2  # two windows: 0-1000, 1000-2000


class TestEncodeVariants:
    def test_basic(self):
        df = pd.DataFrame({
            "n_variants": [3, 0],
            "n_snps": [2, 0],
            "n_indels": [1, 0],
            "mean_dosage": [1.0, 0.0],
            "max_dosage": [2, 0],
            "mean_af": [0.05, 0.0],
        })
        arr = encode_variants_for_model(df)
        assert arr.shape == (2, 6)
