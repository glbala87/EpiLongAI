"""Tests for population-aware model."""

import numpy as np
import pytest
import torch

from epilongai.models.baseline_mlp import BaselineMLP
from epilongai.models.population_aware import (
    PopulationAwareModel,
    PopulationConditionedHead,
    PopulationEmbedding,
    encode_allele_frequencies,
    evaluate_per_population,
)


class TestPopulationEmbedding:
    def test_forward_no_af(self):
        emb = PopulationEmbedding(n_populations=5, embed_dim=8)
        pop_id = torch.tensor([0, 2, 4])
        out = emb(pop_id)
        assert out.shape == (3, 8)

    def test_forward_with_af(self):
        emb = PopulationEmbedding(n_populations=5, embed_dim=8, n_af_features=3)
        pop_id = torch.tensor([0, 1])
        af = torch.randn(2, 3)
        out = emb(pop_id, af)
        assert out.shape == (2, 11)  # 8 + 3


class TestPopulationConditionedHead:
    def test_concatenate(self):
        head = PopulationConditionedHead(
            feature_dim=32, population_dim=8, num_classes=2, conditioning="concatenate"
        )
        features = torch.randn(4, 32)
        pop_emb = torch.randn(4, 8)
        out = head(features, pop_emb)
        assert out["logits"].shape == (4, 1)
        assert out["probs"].shape == (4, 1)

    def test_film(self):
        head = PopulationConditionedHead(
            feature_dim=32, population_dim=8, num_classes=2, conditioning="film"
        )
        features = torch.randn(4, 32)
        pop_emb = torch.randn(4, 8)
        out = head(features, pop_emb)
        assert out["logits"].shape == (4, 1)


class TestPopulationAwareModel:
    def test_wraps_baseline_mlp(self):
        backbone = BaselineMLP(input_dim=7, hidden_dims=[32, 16], num_classes=2)
        model = PopulationAwareModel(
            backbone=backbone,
            backbone_embed_dim=16,
            n_populations=5,
        )
        pop_id = torch.tensor([0, 1, 2, 3])
        x = torch.randn(4, 7)
        out = model(population_id=pop_id, x=x)
        assert "logits" in out
        assert "backbone_embed" in out
        assert "population_embed" in out

    def test_with_allele_freq(self):
        backbone = BaselineMLP(input_dim=7, hidden_dims=[16], num_classes=2)
        model = PopulationAwareModel(
            backbone=backbone,
            backbone_embed_dim=16,
            n_populations=3,
            n_af_features=4,
        )
        pop_id = torch.tensor([0, 1])
        af = torch.randn(2, 4)
        x = torch.randn(2, 7)
        out = model(population_id=pop_id, allele_freq_features=af, x=x)
        assert out["logits"].shape[0] == 2


class TestEncodeAlleleFrequencies:
    def test_basic(self):
        af = encode_allele_frequencies({"gnomAD_AF": 0.05, "QGP_AF": 0.12})
        assert af.shape == (7,)
        assert af[0] == pytest.approx(0.05)
        assert af[1] == pytest.approx(0.12)
        assert af[2] == 0.0  # missing source

    def test_custom_sources(self):
        af = encode_allele_frequencies({"A": 0.1, "B": 0.2}, sources=["A", "B", "C"])
        assert af.shape == (3,)


class TestEvaluatePerPopulation:
    def test_basic(self):
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8, 0.6])
        pop_ids = np.array([0, 0, 0, 1, 1, 1])
        results = evaluate_per_population(y_true, y_pred, y_prob, pop_ids)
        assert "0" in results
        assert "1" in results
        assert "accuracy" in results["0"]
        assert results["0"]["n_samples"] == 3
