"""
Phase G — Training entry-point.

Orchestrates data loading, model construction, and training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from epilongai.data.dataset import (
    MethylationDataset,
    build_dataloaders,
    split_dataset,
)
from epilongai.models.baseline_mlp import BaselineMLP
from epilongai.models.long_context_model import LongContextGenomicModel
from epilongai.models.multimodal_model import MultimodalModel
from epilongai.models.population_aware import PopulationAwareModel
from epilongai.training.plotting import plot_training_history
from epilongai.training.trainer import Trainer
from epilongai.utils.config import load_config
from epilongai.utils.logging import setup_logging
from epilongai.utils.seed import set_seed


def run_training(config_path: str, resume_path: str | None = None) -> None:
    """Full training pipeline driven by a YAML config."""
    cfg = load_config(config_path)
    log_cfg = cfg.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=cfg.get("output", {}).get("log_file"))

    train_cfg = cfg["training"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    ds_cfg = cfg.get("dataset", {})

    set_seed(train_cfg.get("seed", 42))

    # ── Load data ────────────────────────────────────────────────────────
    windows_path = Path(data_cfg["windows_path"])
    if windows_path.suffix == ".parquet":
        windows = pd.read_parquet(windows_path)
    else:
        windows = pd.read_csv(windows_path, sep="\t")
    logger.info(f"Loaded {len(windows):,} windows from {windows_path}")

    # Load metadata & assign labels
    meta = None
    if data_cfg.get("metadata_path"):
        meta = pd.read_csv(data_cfg["metadata_path"], sep="\t")

    label_col = data_cfg.get("label_column", "group")
    label_map = data_cfg.get("label_map", {"FTB": 0, "PTB": 1})

    if label_col in windows.columns:
        labels = windows[label_col].map(label_map).values.astype(np.int64)
    elif meta is not None and label_col in meta.columns and "sample_id" in windows.columns:
        sample_label = dict(zip(meta["sample_id"], meta[label_col].map(label_map)))
        labels = windows["sample_id"].map(sample_label).values.astype(np.int64)
    else:
        raise ValueError(f"Cannot find label column '{label_col}' in windows or metadata")

    # ── Split ────────────────────────────────────────────────────────────
    split_cfg = data_cfg.get("split", {})
    w_train, w_val, w_test, y_train, y_val, y_test = split_dataset(
        windows,
        labels,
        test_size=split_cfg.get("test_size", 0.15),
        val_size=split_cfg.get("val_size", 0.15),
        stratify=split_cfg.get("stratify", True),
        random_seed=split_cfg.get("random_seed", 42),
    )

    # ── Datasets ─────────────────────────────────────────────────────────
    mode = ds_cfg.get("mode", "methylation")
    seq_enc = ds_cfg.get("sequence_encoding", "onehot")
    max_seq = ds_cfg.get("max_sequence_length", 1000)

    ds_kwargs = dict(mode=mode, sequence_encoding=seq_enc, max_sequence_length=max_seq)
    train_ds = MethylationDataset(w_train, y_train, **ds_kwargs)
    val_ds = MethylationDataset(w_val, y_val, **ds_kwargs)
    test_ds = MethylationDataset(w_test, y_test, **ds_kwargs)

    loaders = build_dataloaders(
        train_ds, val_ds, test_ds, batch_size=train_cfg.get("batch_size", 64)
    )

    # ── Model ────────────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtype = model_cfg.get("type", "baseline_mlp")

    if mtype == "baseline_mlp":
        model = BaselineMLP.from_config(model_cfg, input_dim=train_ds.num_features)
    elif mtype == "multimodal":
        model_cfg["_dataset_mode"] = mode
        model_cfg["_variant_input_dim"] = train_ds.num_variant_features
        model = MultimodalModel.from_config(model_cfg, methylation_input_dim=train_ds.num_features)
    elif mtype == "long_context":
        model = LongContextGenomicModel.from_config(model_cfg)
    else:
        raise ValueError(f"Unknown model type: {mtype}")

    # Wrap with population-aware head if enabled
    pop_cfg = model_cfg.get("population", {})
    if pop_cfg.get("enabled", False):
        # Determine backbone embedding dimension
        if mtype == "baseline_mlp":
            backbone_dim = model_cfg.get("mlp", {}).get("hidden_dims", [256, 128, 64])[-1]
        elif mtype == "long_context":
            backbone_dim = model_cfg.get("long_context", {}).get("d_model", 256)
        else:
            fusion_dim = model_cfg.get("multimodal", {}).get("fusion", {}).get("hidden_dim", 128)
            backbone_dim = fusion_dim  # fused embedding size
        model = PopulationAwareModel(
            backbone=model,
            backbone_embed_dim=backbone_dim,
            n_populations=pop_cfg.get("n_populations", 10),
            pop_embed_dim=pop_cfg.get("embed_dim", 16),
            n_af_features=pop_cfg.get("n_af_features", 0),
            num_classes=model_cfg.get("num_classes", 2),
            task=model_cfg.get("task", "classification"),
            conditioning=pop_cfg.get("conditioning", "concatenate"),
        )

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {mtype}, parameters: {total_params:,}, device: {device}")

    # ── Trainer ──────────────────────────────────────────────────────────
    # Merge checkpointing config into training config for Trainer
    full_train_cfg = {**train_cfg, "checkpointing": cfg.get("checkpointing", {})}

    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        cfg=full_train_cfg,
        model_cfg=model_cfg,
        device=device,
    )

    if resume_path:
        trainer.load_checkpoint(resume_path)

    history = trainer.fit()

    # ── Save plots ───────────────────────────────────────────────────────
    plots_dir = Path(cfg.get("output", {}).get("plots_dir", "results/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_training_history(history, save_path=plots_dir / "training_history.png")
    logger.info("Training complete.")
