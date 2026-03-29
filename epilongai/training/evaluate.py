"""
Phase G — Model evaluation on a held-out split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger

from epilongai.data.dataset import (
    MethylationDataset,
    build_dataloaders,
    methylation_collate,
    split_dataset,
)
from epilongai.models.baseline_mlp import BaselineMLP
from epilongai.models.long_context_model import LongContextGenomicModel
from epilongai.models.multimodal_model import MultimodalModel
from epilongai.training.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from epilongai.training.plotting import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from epilongai.utils.config import load_config
from epilongai.utils.logging import setup_logging
from epilongai.utils.seed import set_seed


def run_evaluation(
    config_path: str,
    checkpoint_path: str,
    split: str = "test",
) -> dict:
    """Evaluate a trained model and generate reports."""
    cfg = load_config(config_path)
    log_cfg = cfg.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("output", {}).get("log_file"))

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    ds_cfg = cfg.get("dataset", {})
    train_cfg = cfg["training"]

    set_seed(train_cfg.get("seed", 42))

    # ── Load data & reproduce split ──────────────────────────────────────
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
    w_train, w_val, w_test, y_train, y_val, y_test = split_dataset(
        windows, labels,
        test_size=split_cfg.get("test_size", 0.15),
        val_size=split_cfg.get("val_size", 0.15),
        stratify=split_cfg.get("stratify", True),
        random_seed=split_cfg.get("random_seed", 42),
    )

    if split == "test":
        eval_windows, eval_labels = w_test, y_test
    elif split == "val":
        eval_windows, eval_labels = w_val, y_val
    else:
        eval_windows, eval_labels = w_train, y_train

    mode = ds_cfg.get("mode", "methylation")
    eval_ds = MethylationDataset(
        eval_windows, eval_labels,
        mode=mode,
        sequence_encoding=ds_cfg.get("sequence_encoding", "onehot"),
        max_sequence_length=ds_cfg.get("max_sequence_length", 1000),
    )
    from torch.utils.data import DataLoader
    loader = DataLoader(eval_ds, batch_size=train_cfg.get("batch_size", 64), shuffle=False, collate_fn=methylation_collate)

    # ── Load model ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtype = model_cfg.get("type", "baseline_mlp")

    if mtype == "baseline_mlp":
        model = BaselineMLP.from_config(model_cfg, input_dim=eval_ds.num_features)
    elif mtype == "long_context":
        model = LongContextGenomicModel.from_config(model_cfg)
    else:
        model_cfg["_dataset_mode"] = mode
        model_cfg["_variant_input_dim"] = eval_ds.num_variant_features
        model = MultimodalModel.from_config(model_cfg, methylation_input_dim=eval_ds.num_features)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path}")

    # ── Inference ────────────────────────────────────────────────────────
    task = model_cfg.get("task", "classification")
    num_classes = model_cfg.get("num_classes", 2)
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            kwargs = {}
            if "methylation" in batch:
                kwargs["methylation"] = batch["methylation"].to(device)
            if "sequence" in batch:
                kwargs["sequence"] = batch["sequence"].to(device)
            if "variants" in batch:
                kwargs["variants"] = batch["variants"].to(device)
            if "methylation_track" in batch:
                kwargs["methylation_track"] = batch["methylation_track"].to(device)

            if isinstance(model, BaselineMLP):
                out = model(kwargs.get("methylation", kwargs.get("sequence")))
            elif isinstance(model, LongContextGenomicModel):
                out = model(kwargs.get("sequence"), methylation_track=kwargs.get("methylation_track"))
            else:
                out = model(**kwargs)

            all_labels.append(batch["label"].numpy())
            if task == "classification":
                if num_classes == 2:
                    probs = out["probs"].cpu().numpy().ravel()
                    preds = (probs >= 0.5).astype(int)
                    all_probs.append(probs)
                else:
                    probs = out["probs"].cpu().numpy()
                    preds = probs.argmax(axis=1)
                    all_probs.append(probs)
                all_preds.append(preds)
            else:
                all_preds.append(out["logits"].cpu().numpy().ravel())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    # ── Metrics ──────────────────────────────────────────────────────────
    if task == "classification":
        y_prob = np.concatenate(all_probs) if all_probs else None
        metrics = compute_classification_metrics(y_true, y_pred, y_prob, num_classes)
    else:
        metrics = compute_regression_metrics(y_true, y_pred)
        y_prob = None

    logger.info(f"Evaluation metrics ({split}): {metrics}")

    # ── Plots ────────────────────────────────────────────────────────────
    plots_dir = Path(cfg.get("output", {}).get("plots_dir", "results/plots"))
    plots_dir.mkdir(parents=True, exist_ok=True)

    inv_label_map = {v: k for k, v in label_map.items()}
    class_names = [inv_label_map.get(i, str(i)) for i in range(num_classes)]

    if task == "classification":
        plot_confusion_matrix(y_true, y_pred, labels=class_names, save_path=plots_dir / f"cm_{split}.png")
        if num_classes == 2 and y_prob is not None:
            plot_roc_curve(y_true, y_prob, save_path=plots_dir / f"roc_{split}.png")
            plot_pr_curve(y_true, y_prob, save_path=plots_dir / f"pr_{split}.png")

    # Save metrics JSON
    results_dir = Path(cfg.get("output", {}).get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = results_dir / f"metrics_{split}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    return metrics
