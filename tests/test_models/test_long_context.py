"""Tests for long-context SSM model."""

import torch

from epilongai.models.long_context_model import (
    LongContextDataset,
    LongContextGenomicModel,
    MambaBlock,
    S6Core,
    long_context_collate,
)


class TestS6Core:
    def test_forward_shape(self):
        core = S6Core(d_model=32, d_state=4)
        x = torch.randn(2, 100, 32)
        out = core(x)
        assert out.shape == (2, 100, 32)

    def test_deterministic(self):
        core = S6Core(d_model=16, d_state=4)
        x = torch.randn(1, 50, 16)
        core.eval()
        with torch.no_grad():
            y1 = core(x)
            y2 = core(x)
        assert torch.allclose(y1, y2)


class TestMambaBlock:
    def test_forward_shape(self):
        block = MambaBlock(d_model=32, d_state=4, expand=2)
        x = torch.randn(2, 100, 32)
        out = block(x)
        assert out.shape == (2, 100, 32)

    def test_residual_connection(self):
        """Output should not be identical to input (residual + transform)."""
        block = MambaBlock(d_model=16, d_state=4)
        x = torch.randn(1, 20, 16)
        out = block(x)
        assert not torch.allclose(x, out)


class TestLongContextGenomicModel:
    def test_forward_onehot_only(self):
        model = LongContextGenomicModel(
            n_input_channels=5, d_model=32, n_layers=2, d_state=4,
            num_classes=2, gradient_checkpointing=False,
        )
        x = torch.randn(2, 5, 200)
        out = model(x)
        assert out["logits"].shape == (2, 1)
        assert out["probs"].shape == (2, 1)
        assert out["embed"].shape[0] == 2

    def test_forward_with_methylation_track(self):
        model = LongContextGenomicModel(
            n_input_channels=8, d_model=32, n_layers=2, d_state=4,
            num_classes=2, gradient_checkpointing=False,
        )
        seq = torch.randn(2, 5, 200)
        meth = torch.randn(2, 3, 200)
        out = model(seq, methylation_track=meth)
        assert out["logits"].shape == (2, 1)

    def test_forward_multiclass(self):
        model = LongContextGenomicModel(
            n_input_channels=5, d_model=32, n_layers=1, d_state=4,
            num_classes=4, gradient_checkpointing=False,
        )
        x = torch.randn(2, 5, 100)
        out = model(x)
        assert out["logits"].shape == (2, 4)
        assert out["probs"].shape == (2, 4)

    def test_forward_regression(self):
        model = LongContextGenomicModel(
            n_input_channels=5, d_model=32, n_layers=1, d_state=4,
            num_classes=1, task="regression", gradient_checkpointing=False,
        )
        x = torch.randn(2, 5, 100)
        out = model(x)
        assert out["logits"].shape == (2, 1)
        assert "probs" not in out

    def test_from_config(self):
        cfg = {
            "num_classes": 2,
            "task": "classification",
            "long_context": {
                "n_input_channels": 5,
                "d_model": 32,
                "n_layers": 2,
                "d_state": 4,
                "gradient_checkpointing": False,
            },
        }
        model = LongContextGenomicModel.from_config(cfg)
        x = torch.randn(1, 5, 100)
        out = model(x)
        assert "logits" in out

    def test_cls_pooling(self):
        model = LongContextGenomicModel(
            n_input_channels=5, d_model=32, n_layers=1, d_state=4,
            pool="cls", gradient_checkpointing=False,
        )
        x = torch.randn(2, 5, 100)
        out = model(x)
        assert out["logits"].shape == (2, 1)

    def test_tokenized_input(self):
        """Model should handle (B, L) integer tokens."""
        model = LongContextGenomicModel(
            n_input_channels=5, d_model=32, n_layers=1, d_state=4,
            gradient_checkpointing=False,
        )
        x = torch.randint(0, 5, (2, 100))
        out = model(x)
        assert out["logits"].shape == (2, 1)


class TestLongContextDataset:
    def test_basic(self):
        regions = [
            {
                "sequence": "ACGT" * 25,
                "meth_positions": [10, 20, 30],
                "meth_betas": [0.5, 0.8, 0.2],
                "meth_coverages": [20, 15, 30],
                "region_length": 100,
            }
        ]
        ds = LongContextDataset(regions, labels=[1], max_length=100)
        assert len(ds) == 1
        item = ds[0]
        assert item["sequence"].shape == (5, 100)
        assert item["methylation_track"].shape == (3, 100)
        assert item["label"].item() == 1

    def test_collate(self):
        regions = [
            {"sequence": "A" * 50, "meth_positions": [5], "meth_betas": [0.9], "meth_coverages": [10], "region_length": 50},
            {"sequence": "C" * 50, "meth_positions": [10], "meth_betas": [0.1], "meth_coverages": [20], "region_length": 50},
        ]
        ds = LongContextDataset(regions, labels=[0, 1], max_length=50)
        batch = long_context_collate([ds[0], ds[1]])
        assert batch["sequence"].shape == (2, 5, 50)
        assert batch["label"].shape == (2,)
