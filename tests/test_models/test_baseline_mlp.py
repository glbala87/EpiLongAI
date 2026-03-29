"""Tests for baseline MLP model."""

import torch

from epilongai.models.baseline_mlp import BaselineMLP


class TestBaselineMLP:
    def test_forward_binary(self):
        model = BaselineMLP(input_dim=7, hidden_dims=[32, 16], num_classes=2)
        x = torch.randn(4, 7)
        out = model(x)
        assert out["logits"].shape == (4, 1)
        assert out["probs"].shape == (4, 1)
        assert out["embed"].shape[0] == 4

    def test_forward_multiclass(self):
        model = BaselineMLP(input_dim=7, hidden_dims=[32, 16], num_classes=5)
        x = torch.randn(4, 7)
        out = model(x)
        assert out["logits"].shape == (4, 5)
        assert out["probs"].shape == (4, 5)

    def test_forward_regression(self):
        model = BaselineMLP(input_dim=7, hidden_dims=[32], num_classes=1, task="regression")
        x = torch.randn(4, 7)
        out = model(x)
        assert out["logits"].shape == (4, 1)
        assert "probs" not in out

    def test_from_config(self):
        cfg = {
            "mlp": {"hidden_dims": [64, 32], "dropout": 0.5, "batch_norm": True, "activation": "gelu"},
            "num_classes": 2,
            "task": "classification",
        }
        model = BaselineMLP.from_config(cfg, input_dim=7)
        x = torch.randn(2, 7)
        out = model(x)
        assert out["logits"].shape == (2, 1)
