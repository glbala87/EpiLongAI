"""
Phase F — Multimodal deep learning model: DNA sequence + ONT methylation.

Architecture:
    ┌──────────────┐     ┌──────────────┐
    │  Sequence     │     │ Methylation  │
    │  Encoder      │     │ Encoder      │
    │ (1D CNN /     │     │ (MLP)        │
    │  Transformer) │     │              │
    └──────┬───────┘     └──────┬───────┘
           │                     │
           └────────┬────────────┘
                    │
              ┌─────▼─────┐
              │  Fusion    │
              │ (concat /  │
              │  cross-att)│
              └─────┬─────┘
                    │
              ┌─────▼─────┐
              │  Output    │
              │  Head      │
              └────────────┘

Design notes:
    - Each branch can be toggled on/off via config → graceful fallback
      to unimodal when one modality is missing.
    - Intermediate embeddings are returned for downstream interpretability.

Fusion trade-offs:
    - **Concatenation (late fusion)**: simple, robust, works well with
      limited data. Each branch learns independently.
    - **Cross-attention (mid fusion)**: richer interaction between
      modalities, but needs more data to train and is harder to interpret.
    For ONT methylation studies with typical cohort sizes (50–500 samples),
    concatenation is the recommended starting point.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from loguru import logger


# ── Sequence encoder: 1D CNN ─────────────────────────────────────────────
class SequenceCNN(nn.Module):
    """1D CNN operating on one-hot encoded DNA sequences."""

    def __init__(
        self,
        in_channels: int = 5,
        channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        channels = channels or [64, 128, 256]
        kernel_sizes = kernel_sizes or [7, 5, 3]

        layers: list[nn.Module] = []
        in_c = in_channels
        for out_c, ks in zip(channels, kernel_sizes):
            layers.extend([
                nn.Conv1d(in_c, out_c, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(out_c),
                nn.ReLU(),
                nn.MaxPool1d(pool_size),
            ])
            in_c = out_c
        layers.append(nn.AdaptiveAvgPool1d(1))
        self.net = nn.Sequential(*layers)
        self.out_dim = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len) → (batch, out_dim)"""
        return self.net(x).squeeze(-1)


# ── Sequence encoder: lightweight Transformer ────────────────────────────
class SequenceTransformer(nn.Module):
    """Transformer encoder on tokenized DNA sequences."""

    def __init__(
        self,
        vocab_size: int = 5,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        max_len: int = 2000,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) integer tokens → (batch, d_model)"""
        seq_len = x.size(1)
        emb = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        encoded = self.encoder(emb)
        return encoded.mean(dim=1)  # global average pooling


# ── Methylation MLP encoder ─────────────────────────────────────────────
class MethylationEncoder(nn.Module):
    """MLP encoder for numeric methylation features."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 64]
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


# ── Cross-attention fusion ───────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """Lightweight cross-attention between two embedding vectors."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.scale = math.sqrt(dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        """Each input is (batch, dim). Returns fused (batch, dim)."""
        q = self.q_proj(emb_a).unsqueeze(1)   # (B,1,D)
        k = self.k_proj(emb_b).unsqueeze(1)
        v = self.v_proj(emb_b).unsqueeze(1)
        attn = torch.softmax(q @ k.transpose(-2, -1) / self.scale, dim=-1)
        out = (attn @ v).squeeze(1)
        return self.out_proj(out + emb_a)  # residual


# ── Full multimodal model ────────────────────────────────────────────────
class MultimodalModel(nn.Module):
    """
    Multimodal model combining DNA sequence, methylation features, and
    optionally genomic variant features.

    Toggle modalities via ``use_sequence``, ``use_methylation``, ``use_variants``.
    """

    def __init__(
        self,
        # Sequence branch config
        use_sequence: bool = True,
        sequence_encoder_type: str = "cnn",
        sequence_encoder_kwargs: dict[str, Any] | None = None,
        # Methylation branch config
        use_methylation: bool = True,
        methylation_input_dim: int = 7,
        methylation_encoder_kwargs: dict[str, Any] | None = None,
        # Variant branch config (Phase N)
        use_variants: bool = False,
        variant_input_dim: int = 6,
        variant_encoder_kwargs: dict[str, Any] | None = None,
        # Fusion
        fusion_method: str = "concatenate",
        fusion_hidden_dim: int = 128,
        # Output
        num_classes: int = 2,
        task: str = "classification",
    ) -> None:
        super().__init__()
        self.use_sequence = use_sequence
        self.use_methylation = use_methylation
        self.use_variants = use_variants
        self.task = task
        self._num_classes = num_classes

        # Build branches
        combined_dim = 0

        if use_sequence:
            seq_kw = sequence_encoder_kwargs or {}
            if sequence_encoder_type == "cnn":
                self.seq_encoder = SequenceCNN(**seq_kw)
            else:
                self.seq_encoder = SequenceTransformer(**seq_kw)
            combined_dim += self.seq_encoder.out_dim
        else:
            self.seq_encoder = None

        if use_methylation:
            meth_kw = methylation_encoder_kwargs or {}
            self.meth_encoder = MethylationEncoder(input_dim=methylation_input_dim, **meth_kw)
            combined_dim += self.meth_encoder.out_dim
        else:
            self.meth_encoder = None

        if use_variants:
            var_kw = variant_encoder_kwargs or {}
            self.variant_encoder = MethylationEncoder(
                input_dim=variant_input_dim,
                hidden_dims=var_kw.get("hidden_dims", [64, 32]),
                dropout=var_kw.get("dropout", 0.2),
            )
            combined_dim += self.variant_encoder.out_dim
        else:
            self.variant_encoder = None

        # Fusion
        self.fusion_method = fusion_method
        if fusion_method == "cross_attention" and use_sequence and use_methylation:
            seq_dim = self.seq_encoder.out_dim
            meth_dim = self.meth_encoder.out_dim
            self.seq_proj = nn.Linear(seq_dim, fusion_hidden_dim)
            self.meth_proj = nn.Linear(meth_dim, fusion_hidden_dim)
            self.cross_attn = CrossAttentionFusion(fusion_hidden_dim)
            # Variants still concatenated after cross-attention
            cross_dim = fusion_hidden_dim
            if use_variants:
                cross_dim += self.variant_encoder.out_dim
            combined_dim = cross_dim
        else:
            self.cross_attn = None

        # Output head
        out_dim = 1 if (num_classes == 2 and task == "classification") or task == "regression" else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden_dim, out_dim),
        )

        logger.debug(
            f"MultimodalModel: seq={use_sequence}, meth={use_methylation}, "
            f"var={use_variants}, fusion={fusion_method}, classes={num_classes}, task={task}"
        )

    def forward(
        self,
        sequence: torch.Tensor | None = None,
        methylation: torch.Tensor | None = None,
        variants: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        embeddings: dict[str, torch.Tensor] = {}

        if self.use_sequence and sequence is not None:
            seq_emb = self.seq_encoder(sequence)
            embeddings["seq_embed"] = seq_emb
        else:
            seq_emb = None

        if self.use_methylation and methylation is not None:
            meth_emb = self.meth_encoder(methylation)
            embeddings["meth_embed"] = meth_emb
        else:
            meth_emb = None

        if self.use_variants and variants is not None:
            var_emb = self.variant_encoder(variants)
            embeddings["variant_embed"] = var_emb
        else:
            var_emb = None

        # Fusion
        if self.cross_attn is not None and seq_emb is not None and meth_emb is not None:
            seq_proj = self.seq_proj(seq_emb)
            meth_proj = self.meth_proj(meth_emb)
            fused = self.cross_attn(seq_proj, meth_proj)
            # Append variant embedding after cross-attention
            if var_emb is not None:
                fused = torch.cat([fused, var_emb], dim=-1)
        else:
            parts = [e for e in [seq_emb, meth_emb, var_emb] if e is not None]
            if not parts:
                raise ValueError("At least one modality must provide input")
            fused = torch.cat(parts, dim=-1)

        embeddings["fused"] = fused
        logits = self.classifier(fused)

        out: dict[str, torch.Tensor] = {"logits": logits, **embeddings}
        if self.task == "classification":
            if self._num_classes == 2:
                out["probs"] = torch.sigmoid(logits)
            else:
                out["probs"] = torch.softmax(logits, dim=-1)
        return out

    @classmethod
    def from_config(cls, cfg: dict[str, Any], methylation_input_dim: int) -> MultimodalModel:
        """Instantiate from the model section of train.yaml."""
        mm = cfg.get("multimodal", {})
        ds_cfg = cfg  # parent model config
        seq_type = mm.get("sequence_encoder", "cnn")
        seq_kw = mm.get(seq_type, {})
        meth_kw = mm.get("methylation_encoder", {})
        fusion = mm.get("fusion", {})

        mode = ds_cfg.get("_dataset_mode", "multimodal")
        use_seq = mode in ("sequence", "multimodal", "full")
        use_meth = mode in ("methylation", "multimodal", "variants", "full")
        use_var = mode in ("variants", "full")

        var_kw = mm.get("variant_encoder", {})
        var_dim = ds_cfg.get("_variant_input_dim", 6)

        return cls(
            use_sequence=use_seq,
            sequence_encoder_type=seq_type,
            sequence_encoder_kwargs=seq_kw,
            use_methylation=use_meth,
            methylation_input_dim=methylation_input_dim,
            methylation_encoder_kwargs=meth_kw,
            use_variants=use_var,
            variant_input_dim=var_dim,
            variant_encoder_kwargs=var_kw,
            fusion_method=fusion.get("method", "concatenate"),
            fusion_hidden_dim=fusion.get("hidden_dim", 128),
            num_classes=ds_cfg.get("num_classes", 2),
            task=ds_cfg.get("task", "classification"),
        )
