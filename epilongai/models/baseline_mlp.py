"""
Phase E — Baseline MLP for methylation classification / regression.

A configurable multi-layer perceptron that operates on window-level
numeric methylation features.  Serves as the strong baseline before
moving to CNN or multimodal architectures.

When to use this baseline:
    - Small to moderate cohort sizes (< 500 samples)
    - When sequence information is unavailable
    - As a reference point for more complex models
    - For quick hypothesis testing

Extension path:
    - Replace with 1D CNN operating on positional methylation tracks
    - Feed methylation branch of the multimodal model
    - Add attention over window features across chromosomes
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from loguru import logger


class BaselineMLP(nn.Module):
    """
    Configurable MLP for methylation-window classification or regression.

    Parameters
    ----------
    input_dim : int
        Number of input features per window.
    hidden_dims : list[int]
        Sizes of hidden layers.
    num_classes : int
        Output dimension.  Use 1 for binary classification or regression.
    dropout : float
        Dropout probability between layers.
    batch_norm : bool
        Whether to apply batch normalisation.
    activation : str
        'relu', 'gelu', or 'leaky_relu'.
    task : str
        'classification' or 'regression'.  Controls final activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        num_classes: int = 2,
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
        task: str = "classification",
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.task = task
        act_fn = {"relu": nn.ReLU, "gelu": nn.GELU, "leaky_relu": nn.LeakyReLU}[activation]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.backbone = nn.Sequential(*layers)

        # Output head
        out_dim = 1 if (num_classes == 2 and task == "classification") else num_classes
        if task == "regression":
            out_dim = 1
        self.head = nn.Linear(in_dim, out_dim)
        self._num_classes = num_classes

        logger.debug(
            f"BaselineMLP: input={input_dim}, hidden={hidden_dims}, "
            f"out={out_dim}, task={task}"
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Returns dict with keys:
            logits  — raw output
            probs   — probabilities (classification only)
            embed   — last hidden representation
        """
        embed = self.backbone(x)
        logits = self.head(embed)

        out: dict[str, torch.Tensor] = {"logits": logits, "embed": embed}

        if self.task == "classification":
            if self._num_classes == 2:
                out["probs"] = torch.sigmoid(logits)
            else:
                out["probs"] = torch.softmax(logits, dim=-1)

        return out

    @classmethod
    def from_config(cls, cfg: dict[str, Any], input_dim: int) -> BaselineMLP:
        """Instantiate from a model config dict."""
        mlp_cfg = cfg.get("mlp", {})
        return cls(
            input_dim=input_dim,
            hidden_dims=mlp_cfg.get("hidden_dims", [256, 128, 64]),
            num_classes=cfg.get("num_classes", 2),
            dropout=mlp_cfg.get("dropout", 0.3),
            batch_norm=mlp_cfg.get("batch_norm", True),
            activation=mlp_cfg.get("activation", "relu"),
            task=cfg.get("task", "classification"),
        )
