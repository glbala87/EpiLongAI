"""
Phase K — Inference pipeline for trained ONT methylation models.

Usage:
    epilongai predict \\
        --input-dir data/new_samples/ \\
        --checkpoint checkpoints/best.pt \\
        --config configs/train.yaml \\
        --output results/predictions/

Produces:
    - window_predictions.csv   (per-window scores)
    - sample_predictions.csv   (aggregated per-sample scores)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from epilongai.data.data_ingestion import merge_samples
from epilongai.data.dataset import MethylationDataset, methylation_collate
from epilongai.data.windowing import compute_window_features_fast, extract_sequences
from epilongai.models.baseline_mlp import BaselineMLP
from epilongai.models.long_context_model import LongContextGenomicModel
from epilongai.models.multimodal_model import MultimodalModel
from epilongai.utils.config import load_config
from epilongai.utils.logging import setup_logging
from epilongai.utils.seed import set_seed


def run_prediction(
    input_dir: str,
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    fasta_path: str | None = None,
) -> None:
    """End-to-end inference on new methylation samples."""
    cfg = load_config(config_path)
    log_cfg = cfg.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("output", {}).get("log_file"))

    train_cfg = cfg.get("training", {})
    model_cfg = cfg["model"]
    ds_cfg = cfg.get("dataset", {})
    data_cfg = cfg.get("data", {})

    set_seed(train_cfg.get("seed", 42))

    # ── Ingest new data (same pipeline as training) ──────────────────────
    logger.info(f"Ingesting new samples from {input_dir}")
    ing_cfg = cfg.get("ingestion", {})  # may not exist in train.yaml
    meth = merge_samples(
        input_dir,
        file_format=ing_cfg.get("file_format", "auto"),
        separator=ing_cfg.get("separator", "\t"),
        chunk_size=ing_cfg.get("chunk_size", 500_000),
        min_coverage=ing_cfg.get("min_coverage", 5),
    )

    # ── Windowing ────────────────────────────────────────────────────────
    win_cfg = cfg.get("windowing", {})
    windows = compute_window_features_fast(
        meth,
        window_size=win_cfg.get("window_size", 1000),
        stride=win_cfg.get("stride", 1000),
        min_cpgs=win_cfg.get("min_cpgs_per_window", 3),
    )

    if fasta_path and ds_cfg.get("mode") in ("sequence", "multimodal"):
        windows["sequence"] = extract_sequences(windows, fasta_path)

    logger.info(f"Created {len(windows):,} windows for prediction")

    # ── Build dataset ────────────────────────────────────────────────────
    mode = ds_cfg.get("mode", "methylation")
    pred_ds = MethylationDataset(
        windows,
        labels=None,
        mode=mode,
        sequence_encoding=ds_cfg.get("sequence_encoding", "onehot"),
        max_sequence_length=ds_cfg.get("max_sequence_length", 1000),
    )

    loader = torch.utils.data.DataLoader(
        pred_ds,
        batch_size=train_cfg.get("batch_size", 64),
        shuffle=False,
        collate_fn=methylation_collate,
    )

    # ── Load model ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtype = model_cfg.get("type", "baseline_mlp")

    if mtype == "baseline_mlp":
        model = BaselineMLP.from_config(model_cfg, input_dim=pred_ds.num_features)
    elif mtype == "long_context":
        model = LongContextGenomicModel.from_config(model_cfg)
    else:
        model_cfg["_dataset_mode"] = mode
        model_cfg["_variant_input_dim"] = pred_ds.num_variant_features
        model = MultimodalModel.from_config(model_cfg, methylation_input_dim=pred_ds.num_features)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    # ── Inference ────────────────────────────────────────────────────────
    task = model_cfg.get("task", "classification")
    num_classes = model_cfg.get("num_classes", 2)
    all_probs, all_preds, all_idxs = [], [], []

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

            if task == "classification":
                if num_classes == 2:
                    probs = out["probs"].cpu().numpy().ravel()
                    preds = (probs >= 0.5).astype(int)
                else:
                    probs = out["probs"].cpu().numpy()
                    preds = probs.argmax(axis=1)
                all_probs.append(probs)
                all_preds.append(preds)
            else:
                all_preds.append(out["logits"].cpu().numpy().ravel())

            all_idxs.extend(batch["idx"])

    # ── Assemble results ─────────────────────────────────────────────────
    inv_label_map = {v: k for k, v in data_cfg.get("label_map", {0: "FTB", 1: "PTB"}).items()}

    window_results = pred_ds.coords.copy()
    y_pred = np.concatenate(all_preds)
    window_results["predicted_class"] = y_pred
    if task == "classification":
        window_results["predicted_label"] = [inv_label_map.get(int(p), str(p)) for p in y_pred]
        if all_probs:
            y_prob = np.concatenate(all_probs)
            if y_prob.ndim == 1:
                window_results["prob_positive"] = y_prob
            else:
                for i in range(y_prob.shape[1]):
                    window_results[f"prob_class_{i}"] = y_prob[:, i]
    else:
        window_results["predicted_value"] = y_pred

    # ── Sample-level aggregation ─────────────────────────────────────────
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    window_results.to_csv(out_path / "window_predictions.csv", index=False)
    logger.info(f"Saved window predictions to {out_path / 'window_predictions.csv'}")

    if "sample_id" in window_results.columns and task == "classification":
        if "prob_positive" in window_results.columns:
            sample_agg = window_results.groupby("sample_id").agg(
                mean_prob=("prob_positive", "mean"),
                median_prob=("prob_positive", "median"),
                n_windows=("prob_positive", "count"),
                frac_positive=("predicted_class", "mean"),
            ).reset_index()
            sample_agg["predicted_label"] = sample_agg["mean_prob"].apply(
                lambda p: inv_label_map.get(1, "positive") if p >= 0.5 else inv_label_map.get(0, "negative")
            )
            sample_agg.to_csv(out_path / "sample_predictions.csv", index=False)
            logger.info(f"Saved sample predictions to {out_path / 'sample_predictions.csv'}")

    # ── Log metadata ─────────────────────────────────────────────────────
    pred_meta = {
        "checkpoint": checkpoint_path,
        "config": config_path,
        "model_type": mtype,
        "task": task,
        "n_windows": len(window_results),
        "epoch": ckpt.get("epoch", "unknown"),
    }
    with open(out_path / "prediction_meta.json", "w") as f:
        json.dump(pred_meta, f, indent=2)

    logger.info("Prediction complete.")
