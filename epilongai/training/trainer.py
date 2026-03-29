"""
Phase G — Production-ready training loop for ONT methylation models.

Supports:
    - Binary / multiclass classification and regression
    - Checkpointing (best + periodic)
    - Early stopping
    - LR schedulers (cosine, step, plateau)
    - Weighted loss for class imbalance
    - Mixed-precision training via torch.amp
    - Per-epoch metric logging
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from epilongai.training.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)


class Trainer:
    """
    General-purpose trainer for EpiLongAI models.

    Parameters
    ----------
    model : nn.Module
    train_loader, val_loader : DataLoader
    cfg : dict
        Training section of the YAML config.
    model_cfg : dict
        Model section (to determine task, num_classes).
    device : str | torch.device
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = torch.device(device)
        self.task = model_cfg.get("task", "classification")
        self.num_classes = model_cfg.get("num_classes", 2)

        # Optimiser
        opt_name = cfg.get("optimizer", "adamw")
        lr = cfg.get("learning_rate", 1e-3)
        wd = cfg.get("weight_decay", 1e-4)
        if opt_name == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
        else:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

        # Loss
        self.criterion = self._build_loss(cfg, model_cfg)

        # Scheduler
        self.scheduler = self._build_scheduler(cfg)

        # Mixed precision
        self.use_amp = cfg.get("mixed_precision", False) and self.device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        # Gradient clipping
        self.grad_clip = cfg.get("gradient_clip", 0.0)

        # Early stopping
        es = cfg.get("early_stopping", {})
        self.es_enabled = es.get("enabled", True)
        self.es_patience = es.get("patience", 15)
        self.es_metric = es.get("metric", "val_f1")
        self.es_mode = es.get("mode", "max")

        # Checkpointing
        ck = cfg.get("checkpointing", model_cfg.get("checkpointing", {}))
        self.ckpt_dir = Path(ck.get("save_dir", "checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = ck.get("save_best", True)
        self.save_every = ck.get("save_every_n_epochs", 10)

        # History
        self.history: dict[str, list[float]] = {}
        self.best_metric: float = -float("inf") if self.es_mode == "max" else float("inf")
        self.epochs_no_improve = 0
        self.current_epoch = 0

    # ── Loss construction ────────────────────────────────────────────────
    def _build_loss(self, cfg: dict, model_cfg: dict) -> nn.Module:
        if self.task == "regression":
            return nn.MSELoss()
        cw = cfg.get("class_weights")
        weight = None
        if cw == "balanced":
            # Will be computed from training data on first epoch
            self._need_class_weight = True
        elif isinstance(cw, list):
            weight = torch.tensor(cw, dtype=torch.float32, device=self.device)
            self._need_class_weight = False
        else:
            self._need_class_weight = False

        if self.num_classes == 2:
            loss = nn.BCEWithLogitsLoss(pos_weight=weight[1:2] if weight is not None else None)
        else:
            loss = nn.CrossEntropyLoss(weight=weight)
        return loss

    def _compute_balanced_weights(self) -> None:
        """Compute class weights from training labels."""
        all_labels: list[int] = []
        for batch in self.train_loader:
            all_labels.extend(batch["label"].numpy().tolist())
        labels_arr = np.array(all_labels)
        classes = np.unique(labels_arr)
        counts = np.bincount(labels_arr, minlength=len(classes)).astype(float)
        weights = len(labels_arr) / (len(classes) * counts)
        wt = torch.tensor(weights, dtype=torch.float32, device=self.device)
        if self.num_classes == 2:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=wt[1:2])
        else:
            self.criterion = nn.CrossEntropyLoss(weight=wt)
        logger.info(f"Computed balanced class weights: {weights.tolist()}")

    # ── Scheduler ────────────────────────────────────────────────────────
    def _build_scheduler(self, cfg: dict) -> Any:
        sch = cfg.get("scheduler", {})
        stype = sch.get("type", "none")
        if stype == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=sch.get("T_max", cfg.get("epochs", 100))
            )
        elif stype == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=sch.get("step_size", 30), gamma=sch.get("gamma", 0.1)
            )
        elif stype == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode=self.es_mode, patience=sch.get("patience", 10)
            )
        return None

    # ── Training loop ────────────────────────────────────────────────────
    def fit(self, epochs: int | None = None) -> dict[str, list[float]]:
        """Run the full training loop."""
        epochs = epochs or self.cfg.get("epochs", 100)

        if getattr(self, "_need_class_weight", False):
            self._compute_balanced_weights()
            self._need_class_weight = False

        for epoch in range(1, epochs + 1):
            self.current_epoch = epoch
            train_loss = self._train_epoch()
            val_metrics = self._validate()

            # Log
            self._append_history("train_loss", train_loss)
            for k, v in val_metrics.items():
                self._append_history(f"val_{k}", v)

            lr = self.optimizer.param_groups[0]["lr"]
            primary = val_metrics.get(self.es_metric.replace("val_", ""), 0.0)
            logger.info(
                f"Epoch {epoch}/{epochs} — loss: {train_loss:.4f}, "
                f"val_loss: {val_metrics.get('loss', 0):.4f}, "
                f"{self.es_metric}: {primary:.4f}, lr: {lr:.2e}"
            )

            # Checkpoint
            if self.save_best:
                improved = (
                    (self.es_mode == "max" and primary > self.best_metric)
                    or (self.es_mode == "min" and primary < self.best_metric)
                )
                if improved:
                    self.best_metric = primary
                    self._save_checkpoint("best.pt")
                    self.epochs_no_improve = 0
                else:
                    self.epochs_no_improve += 1

            if epoch % self.save_every == 0:
                self._save_checkpoint(f"epoch_{epoch}.pt")

            # Scheduler step
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(primary)
                else:
                    self.scheduler.step()

            # Early stopping
            if self.es_enabled and self.epochs_no_improve >= self.es_patience:
                logger.info(f"Early stopping at epoch {epoch} (no improvement for {self.es_patience} epochs)")
                break

        return self.history

    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad()
            loss = self._forward_loss(batch)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if self.grad_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _validate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        for batch in self.val_loader:
            loss = self._forward_loss(batch)
            total_loss += loss.item()

            out = self._forward(batch)
            labels = batch["label"].numpy()
            all_labels.append(labels)

            if self.task == "classification":
                if self.num_classes == 2:
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

        metrics: dict[str, float] = {"loss": total_loss / max(len(self.val_loader), 1)}

        if self.task == "classification":
            y_prob = np.concatenate(all_probs) if all_probs else None
            metrics.update(compute_classification_metrics(y_true, y_pred, y_prob, self.num_classes))
        else:
            metrics.update(compute_regression_metrics(y_true, y_pred))

        return metrics

    # ── Helpers ──────────────────────────────────────────────────────────
    def _forward(self, batch: dict) -> dict[str, torch.Tensor]:
        """Route batch through the model based on available keys."""
        from epilongai.models.baseline_mlp import BaselineMLP
        from epilongai.models.long_context_model import LongContextGenomicModel
        from epilongai.models.population_aware import PopulationAwareModel

        kwargs: dict[str, Any] = {}
        if "methylation" in batch:
            kwargs["methylation"] = batch["methylation"].to(self.device)
        if "sequence" in batch:
            kwargs["sequence"] = batch["sequence"].to(self.device)
        if "variants" in batch:
            kwargs["variants"] = batch["variants"].to(self.device)
        if "methylation_track" in batch:
            kwargs["methylation_track"] = batch["methylation_track"].to(self.device)

        # Unwrap PopulationAwareModel to check backbone type
        model = self.model
        if isinstance(model, PopulationAwareModel):
            backbone = model.backbone
        else:
            backbone = model

        # BaselineMLP expects a single positional tensor
        if isinstance(backbone, BaselineMLP):
            x = kwargs.get("methylation", kwargs.get("sequence"))
            if isinstance(model, PopulationAwareModel):
                pop_id = batch.get("population_id", torch.zeros(x.size(0), dtype=torch.long)).to(self.device)
                af = batch.get("allele_freq_features")
                if af is not None:
                    af = af.to(self.device)
                return model(population_id=pop_id, allele_freq_features=af, x=x)
            return backbone(x)

        # LongContextGenomicModel expects (x, methylation_track)
        if isinstance(backbone, LongContextGenomicModel):
            seq = kwargs.get("sequence")
            meth_track = kwargs.get("methylation_track")
            if isinstance(model, PopulationAwareModel):
                pop_id = batch.get("population_id", torch.zeros(seq.size(0), dtype=torch.long)).to(self.device)
                af = batch.get("allele_freq_features")
                if af is not None:
                    af = af.to(self.device)
                return model(population_id=pop_id, allele_freq_features=af, x=seq, methylation_track=meth_track)
            return backbone(seq, methylation_track=meth_track)

        # MultimodalModel and PopulationAwareModel wrapping it
        if isinstance(model, PopulationAwareModel):
            pop_id = batch.get("population_id", torch.zeros(batch["label"].size(0), dtype=torch.long)).to(self.device)
            af = batch.get("allele_freq_features")
            if af is not None:
                af = af.to(self.device)
            return model(population_id=pop_id, allele_freq_features=af, **kwargs)

        return model(**kwargs)

    def _forward_loss(self, batch: dict) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            out = self._forward(batch)
            labels = batch["label"].to(self.device)
            logits = out["logits"]
            if self.task == "regression":
                return self.criterion(logits.squeeze(), labels.float())
            if self.num_classes == 2:
                return self.criterion(logits.squeeze(), labels.float())
            return self.criterion(logits, labels)

    def _save_checkpoint(self, filename: str) -> None:
        path = self.ckpt_dir / filename
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "history": self.history,
        }, path)
        logger.info(f"Saved checkpoint: {path}")

    def _append_history(self, key: str, value: float) -> None:
        self.history.setdefault(key, []).append(value)

    def load_checkpoint(self, path: str | Path) -> None:
        """Resume training from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_metric = ckpt.get("best_metric", self.best_metric)
        self.history = ckpt.get("history", {})
        self.current_epoch = ckpt.get("epoch", 0)
        logger.info(f"Resumed from {path} (epoch {self.current_epoch})")
