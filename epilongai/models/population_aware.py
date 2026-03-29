"""
Phase O — Population-aware modeling extension.

WHY POPULATION-AWARE MODELING IMPROVES PREDICTION
===================================================
1. **Methylation varies by ancestry**: Baseline methylation levels at thousands
   of CpG sites differ systematically between populations due to genetic
   background, environment, and gene-environment interactions.  A model
   trained on European-only data may misclassify Middle-Eastern or South-Asian
   samples simply because their "normal" methylation looks unusual to the model.

2. **Allele frequencies are population-specific**: meQTLs that drive
   methylation variation have different frequencies in QGP vs gnomAD.
   Encoding population-specific allele frequencies lets the model learn
   population-calibrated effects.

3. **Clinical relevance**: Preterm birth rates vary 2–3× across populations.
   A population-aware model can output calibrated risk scores that account
   for population-specific baselines.

HOW TO AVOID BIAS AND OVERFITTING
===================================
- **Never use population as a label** for prediction — use it as a *covariate*
  that the model conditions on, not a target.
- **Stratified evaluation**: Always report metrics per population.  An overall
  AUC of 0.85 may hide 0.95 for one population and 0.65 for another.
- **Regularise the population embedding**: Keep embedding dim small (8–16)
  and apply weight decay.  The embedding should capture coarse ancestry
  structure, not memorise individual samples.
- **Use population as a fairness constraint**: Monitor that false-positive
  and false-negative rates are balanced across groups.
- **Validation on held-out populations**: If feasible, hold out an entire
  population from training to test generalisation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


# =====================================================================
# Population Embedding Layer
# =====================================================================
class PopulationEmbedding(nn.Module):
    """
    Learnable embedding for discrete population labels.

    Optionally concatenates continuous allele-frequency features.
    """

    def __init__(
        self,
        n_populations: int = 10,
        embed_dim: int = 16,
        n_af_features: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_populations, embed_dim)
        self.n_af_features = n_af_features
        total_dim = embed_dim + n_af_features
        self.proj = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.LayerNorm(total_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out_dim = total_dim

    def forward(
        self,
        population_id: torch.Tensor,
        allele_freq_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        population_id : (batch,) long tensor
        allele_freq_features : (batch, n_af_features) or None

        Returns
        -------
        (batch, out_dim)
        """
        emb = self.embedding(population_id)
        if allele_freq_features is not None and self.n_af_features > 0:
            emb = torch.cat([emb, allele_freq_features], dim=-1)
        return self.proj(emb)


# =====================================================================
# Population-Conditioned Prediction Head
# =====================================================================
class PopulationConditionedHead(nn.Module):
    """
    Output head that conditions predictions on population context.

    Two strategies:
    1. **concatenate**: Append population embedding to feature vector before
       the final classifier.  Simple and effective.
    2. **film** (Feature-wise Linear Modulation): Population embedding
       generates scale/shift parameters that modulate the feature vector.
       More expressive for learning population-specific transformations.
    """

    def __init__(
        self,
        feature_dim: int,
        population_dim: int,
        num_classes: int = 2,
        task: str = "classification",
        conditioning: str = "concatenate",
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.task = task
        self._num_classes = num_classes
        self.conditioning = conditioning

        if conditioning == "film":
            self.film_scale = nn.Linear(population_dim, feature_dim)
            self.film_shift = nn.Linear(population_dim, feature_dim)
            input_dim = feature_dim
        else:
            input_dim = feature_dim + population_dim

        out_dim = 1 if (num_classes == 2 and task == "classification") or task == "regression" else num_classes
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        population_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.conditioning == "film":
            gamma = self.film_scale(population_emb)
            beta = self.film_shift(population_emb)
            features = features * (1 + gamma) + beta
        else:
            features = torch.cat([features, population_emb], dim=-1)

        logits = self.classifier(features)
        out: dict[str, torch.Tensor] = {"logits": logits}
        if self.task == "classification":
            if self._num_classes == 2:
                out["probs"] = torch.sigmoid(logits)
            else:
                out["probs"] = torch.softmax(logits, dim=-1)
        return out


# =====================================================================
# Full Population-Aware Model Wrapper
# =====================================================================
class PopulationAwareModel(nn.Module):
    """
    Wraps any backbone (BaselineMLP, MultimodalModel, LongContextModel)
    and adds population-conditioned prediction.

    Parameters
    ----------
    backbone : nn.Module
        Any model that returns a dict with 'embed' or 'fused' key.
    backbone_embed_dim : int
        Dimension of the backbone's embedding output.
    n_populations : int
        Number of distinct population labels.
    pop_embed_dim : int
        Size of population embedding.
    n_af_features : int
        Number of continuous allele-frequency features.
    conditioning : str
        'concatenate' or 'film'.
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_embed_dim: int,
        n_populations: int = 10,
        pop_embed_dim: int = 16,
        n_af_features: int = 0,
        num_classes: int = 2,
        task: str = "classification",
        conditioning: str = "concatenate",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pop_embedding = PopulationEmbedding(
            n_populations=n_populations,
            embed_dim=pop_embed_dim,
            n_af_features=n_af_features,
        )
        self.head = PopulationConditionedHead(
            feature_dim=backbone_embed_dim,
            population_dim=self.pop_embedding.out_dim,
            num_classes=num_classes,
            task=task,
            conditioning=conditioning,
        )
        self.task = task
        self._num_classes = num_classes

        logger.info(
            f"PopulationAwareModel: backbone_dim={backbone_embed_dim}, "
            f"populations={n_populations}, conditioning={conditioning}"
        )

    def forward(
        self,
        population_id: torch.Tensor,
        allele_freq_features: torch.Tensor | None = None,
        **backbone_kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Run backbone to get embedding
        backbone_out = self.backbone(**backbone_kwargs)
        embed = backbone_out.get("fused", backbone_out.get("embed"))
        if embed is None:
            raise ValueError("Backbone must return 'embed' or 'fused' key")

        # Population embedding
        pop_emb = self.pop_embedding(population_id, allele_freq_features)

        # Conditioned prediction
        out = self.head(embed, pop_emb)
        out["backbone_embed"] = embed
        out["population_embed"] = pop_emb
        return out


# =====================================================================
# Allele Frequency Encoding
# =====================================================================
def encode_allele_frequencies(
    af_data: dict[str, float],
    sources: list[str] | None = None,
) -> np.ndarray:
    """
    Encode allele frequency data from population databases.

    Parameters
    ----------
    af_data : dict
        Keys are source names (e.g. 'gnomAD_AF', 'QGP_AF', 'AF_eas', 'AF_sas'),
        values are frequency floats.
    sources : list[str] or None
        Which sources to include.  None → all available.

    Returns
    -------
    1D float32 array of allele frequencies (0 for missing sources).
    """
    if sources is None:
        sources = ["gnomAD_AF", "QGP_AF", "AF_eas", "AF_sas", "AF_amr", "AF_afr", "AF_eur"]
    return np.array([af_data.get(s, 0.0) for s in sources], dtype=np.float32)


# =====================================================================
# Population-Stratified Evaluation
# =====================================================================
def evaluate_per_population(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    population_ids: np.ndarray,
    population_names: dict[int, str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute metrics stratified by population.

    Returns dict mapping population_name → metrics dict.
    """
    from epilongai.training.metrics import compute_classification_metrics

    results: dict[str, dict[str, float]] = {}
    for pop_id in np.unique(population_ids):
        mask = population_ids == pop_id
        pop_name = population_names.get(pop_id, str(pop_id)) if population_names else str(pop_id)

        prob = y_prob[mask] if y_prob is not None else None
        metrics = compute_classification_metrics(
            y_true[mask], y_pred[mask], prob, num_classes=len(np.unique(y_true))
        )
        metrics["n_samples"] = int(mask.sum())
        results[pop_name] = metrics
        logger.info(f"Population '{pop_name}': {metrics}")

    return results
