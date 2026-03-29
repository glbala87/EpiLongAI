"""
Phase J — Interpretability tools for trained ONT methylation models.

Provides:
    1. SHAP explanations for numeric methylation features (baseline MLP)
    2. Integrated gradients / saliency for sequence branch (multimodal)
    3. Feature importance summary plots
    4. Top-predictive genomic window identification

Cautions for biological interpretation:
    - Feature importance ≠ causation.  A highly ranked window may correlate
      with the phenotype without being mechanistically involved.
    - SHAP values are model-dependent — a different architecture may rank
      features differently.
    - Always cross-reference top regions with known biology (gene
      annotations, regulatory databases) before drawing conclusions.
    - Consider batch effects: if top features correlate with batch rather
      than phenotype, the model may be learning confounders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from loguru import logger

from epilongai.data.dataset import (
    MethylationDataset,
    methylation_collate,
    split_dataset,
)
from epilongai.models.baseline_mlp import BaselineMLP
from epilongai.models.multimodal_model import MultimodalModel
from epilongai.utils.config import load_config
from epilongai.utils.logging import setup_logging
from epilongai.utils.seed import set_seed


# ── SHAP for methylation features ────────────────────────────────────────
def shap_explain_methylation(
    model: BaselineMLP,
    background_data: torch.Tensor,
    explain_data: torch.Tensor,
    feature_names: list[str],
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute SHAP values for the methylation MLP branch.

    Returns shap_values array of shape (n_explain, n_features).
    """
    import shap

    model.eval()
    model.to(device)

    def model_fn(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(x, dtype=torch.float32, device=device)
            out = model(t)
            if "probs" in out:
                return out["probs"].cpu().numpy()
            return out["logits"].cpu().numpy()

    bg = background_data.numpy()
    ex = explain_data.numpy()

    explainer = shap.KernelExplainer(model_fn, bg)
    shap_values = explainer.shap_values(ex)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    logger.info(f"SHAP values computed for {len(ex)} samples × {len(feature_names)} features")
    return np.array(shap_values)


def plot_shap_summary(
    shap_values: np.ndarray,
    feature_names: list[str],
    save_path: str | Path | None = None,
) -> None:
    """Publication-ready SHAP summary bar plot."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, mean_abs[order], color="#4C72B0", edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([feature_names[i] for i in order])
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved SHAP plot to {save_path}")
    plt.close(fig)


# ── Integrated gradients for sequence branch ─────────────────────────────
def integrated_gradients_sequence(
    model: torch.nn.Module,
    input_seq: torch.Tensor,
    target_class: int = 1,
    n_steps: int = 50,
    device: str = "cpu",
) -> np.ndarray:
    """
    Compute integrated gradients for a sequence input tensor.

    Parameters
    ----------
    model : nn.Module
        Must accept ``sequence=...`` keyword argument.
    input_seq : Tensor
        Shape (1, channels, seq_len) for CNN or (1, seq_len) for transformer.
    target_class : int
        Class index to explain.
    n_steps : int
        Number of interpolation steps.

    Returns
    -------
    attributions : np.ndarray
        Same shape as input_seq (without batch dim).
    """
    model.eval()
    model.to(device)
    input_seq = input_seq.to(device)

    baseline = torch.zeros_like(input_seq)
    scaled_inputs = [
        baseline + (float(i) / n_steps) * (input_seq - baseline)
        for i in range(1, n_steps + 1)
    ]

    grads: list[torch.Tensor] = []
    for si in scaled_inputs:
        si = si.detach().requires_grad_(True)
        out = model(sequence=si)
        logits = out["logits"]
        if logits.shape[-1] > 1:
            score = logits[0, target_class]
        else:
            score = logits[0, 0]
        score.backward()
        grads.append(si.grad.detach().clone())

    avg_grad = torch.stack(grads).mean(dim=0)
    attributions = (input_seq - baseline) * avg_grad
    return attributions.squeeze(0).cpu().numpy()


# ── Top predictive windows ──────────────────────────────────────────────
def rank_predictive_windows(
    model: torch.nn.Module,
    dataset: MethylationDataset,
    top_k: int = 50,
    device: str = "cpu",
) -> pd.DataFrame:
    """
    Rank genomic windows by prediction confidence (probability for positive class).
    """
    model.eval()
    model.to(device)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False, collate_fn=methylation_collate
    )

    scores: list[float] = []
    indices: list[int] = []

    with torch.no_grad():
        for batch in loader:
            kwargs: dict[str, Any] = {}
            if "methylation" in batch:
                kwargs["methylation"] = batch["methylation"].to(device)
            if "sequence" in batch:
                kwargs["sequence"] = batch["sequence"].to(device)

            if isinstance(model, BaselineMLP):
                out = model(kwargs.get("methylation", kwargs.get("sequence")))
            else:
                out = model(**kwargs)

            if "probs" in out:
                prob = out["probs"].cpu().numpy().ravel()
            else:
                prob = out["logits"].cpu().numpy().ravel()
            scores.extend(prob.tolist())
            indices.extend(batch["idx"])

    df = dataset.coords.iloc[indices].copy()
    df["score"] = scores
    df = df.sort_values("score", ascending=False).head(top_k).reset_index(drop=True)
    logger.info(f"Top {top_k} predictive windows identified")
    return df


# ── CLI entry-point ──────────────────────────────────────────────────────
def run_interpretation(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    top_k: int = 50,
) -> None:
    """Generate interpretability reports."""
    cfg = load_config(config_path)
    log_cfg = cfg.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=cfg.get("output", {}).get("log_file"))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    ds_cfg = cfg.get("dataset", {})
    train_cfg = cfg.get("training", {})

    set_seed(train_cfg.get("seed", 42))

    # Load data
    windows_path = Path(data_cfg["windows_path"])
    windows = pd.read_parquet(windows_path) if windows_path.suffix == ".parquet" else pd.read_csv(windows_path, sep="\t")

    label_col = data_cfg.get("label_column", "group")
    label_map = data_cfg.get("label_map", {"FTB": 0, "PTB": 1})
    if label_col in windows.columns:
        labels = windows[label_col].map(label_map).values.astype(np.int64)
    else:
        meta = pd.read_csv(data_cfg["metadata_path"], sep="\t")
        sample_label = dict(zip(meta["sample_id"], meta[label_col].map(label_map)))
        labels = windows["sample_id"].map(sample_label).values.astype(np.int64)

    split_cfg = data_cfg.get("split", {})
    _, _, w_test, _, _, y_test = split_dataset(
        windows, labels,
        test_size=split_cfg.get("test_size", 0.15),
        val_size=split_cfg.get("val_size", 0.15),
        stratify=split_cfg.get("stratify", True),
        random_seed=split_cfg.get("random_seed", 42),
    )

    mode = ds_cfg.get("mode", "methylation")
    test_ds = MethylationDataset(
        w_test, y_test, mode=mode,
        sequence_encoding=ds_cfg.get("sequence_encoding", "onehot"),
        max_sequence_length=ds_cfg.get("max_sequence_length", 1000),
    )

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtype = model_cfg.get("type", "baseline_mlp")
    if mtype == "baseline_mlp":
        model = BaselineMLP.from_config(model_cfg, input_dim=test_ds.num_features)
    else:
        model_cfg["_dataset_mode"] = mode
        model = MultimodalModel.from_config(model_cfg, methylation_input_dim=test_ds.num_features)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── SHAP (methylation features) ──────────────────────────────────────
    if mode in ("methylation", "multimodal") and isinstance(model, BaselineMLP):
        bg_size = min(100, len(test_ds))
        bg_data = torch.from_numpy(test_ds.features[:bg_size])
        ex_data = torch.from_numpy(test_ds.features)

        try:
            sv = shap_explain_methylation(
                model, bg_data, ex_data, test_ds.feature_cols, device=device
            )
            plot_shap_summary(sv, test_ds.feature_cols, save_path=out / "shap_summary.png")

            # Export feature importance CSV
            mean_abs = np.abs(sv).mean(axis=0)
            fi_df = pd.DataFrame({
                "feature": test_ds.feature_cols,
                "mean_abs_shap": mean_abs,
            }).sort_values("mean_abs_shap", ascending=False)
            fi_df.to_csv(out / "feature_importance.csv", index=False)
        except Exception:
            logger.exception("SHAP computation failed — skipping")

    # ── Top predictive windows ───────────────────────────────────────────
    top_windows = rank_predictive_windows(model, test_ds, top_k=top_k, device=device)
    top_windows.to_csv(out / "top_predictive_windows.csv", index=False)

    logger.info(f"Interpretability outputs saved to {out}")
