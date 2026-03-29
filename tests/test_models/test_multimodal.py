"""Tests for multimodal model."""

import torch

from epilongai.models.multimodal_model import MultimodalModel, SequenceCNN


class TestSequenceCNN:
    def test_forward(self):
        cnn = SequenceCNN(in_channels=5, channels=[16, 32])
        x = torch.randn(4, 5, 500)
        out = cnn(x)
        assert out.shape == (4, 32)


class TestMultimodalModel:
    def test_multimodal_concat(self):
        model = MultimodalModel(
            use_sequence=True,
            sequence_encoder_type="cnn",
            use_methylation=True,
            methylation_input_dim=7,
            fusion_method="concatenate",
            num_classes=2,
        )
        seq = torch.randn(4, 5, 500)
        meth = torch.randn(4, 7)
        out = model(sequence=seq, methylation=meth)
        assert "logits" in out
        assert "probs" in out
        assert "fused" in out

    def test_methylation_only(self):
        model = MultimodalModel(
            use_sequence=False,
            use_methylation=True,
            methylation_input_dim=7,
            num_classes=2,
        )
        meth = torch.randn(4, 7)
        out = model(methylation=meth)
        assert out["logits"].shape[0] == 4

    def test_sequence_only(self):
        model = MultimodalModel(
            use_sequence=True,
            sequence_encoder_type="cnn",
            use_methylation=False,
            num_classes=2,
        )
        seq = torch.randn(4, 5, 500)
        out = model(sequence=seq)
        assert out["logits"].shape[0] == 4
