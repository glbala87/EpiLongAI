"""
Model registry and versioning.

Tracks model versions, configs, metrics, and data lineage so every
prediction can be traced back to the exact checkpoint, config, and
dataset that produced it.

Registry layout on disk:
    model_registry/
    ├── registry.json            # index of all registered models
    └── v001_baseline_mlp/
        ├── checkpoint.pt        # model weights
        ├── config.yaml          # frozen training config
        ├── metrics.json         # evaluation metrics
        └── meta.json            # version, timestamp, git hash, data hash
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


class ModelRegistry:
    """
    File-based model registry for tracking trained model versions.

    Usage:
        registry = ModelRegistry("model_registry")
        version = registry.register(
            checkpoint_path="checkpoints/best.pt",
            config_path="configs/train.yaml",
            metrics={"roc_auc": 0.87, "f1": 0.82},
            data_path="data/windows/windows.parquet",
            description="Baseline MLP, PTB cohort v1",
        )
        info = registry.get(version)
        registry.list_models()
    """

    def __init__(self, root: str | Path = "model_registry") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "registry.json"
        self._index = self._load_index()

    def _load_index(self) -> list[dict[str, Any]]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return []

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self._index, indent=2, default=str))

    def _next_version(self) -> str:
        existing = [e["version"] for e in self._index]
        if not existing:
            return "v001"
        nums = [int(v.lstrip("v")) for v in existing if v.startswith("v")]
        return f"v{max(nums) + 1:03d}"

    @staticmethod
    def _git_hash() -> str | None:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return None

    @staticmethod
    def _file_hash(path: Path, algorithm: str = "sha256") -> str:
        h = hashlib.new(algorithm)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]

    def register(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        metrics: dict[str, float] | None = None,
        data_path: str | Path | None = None,
        description: str = "",
        tags: list[str] | None = None,
    ) -> str:
        """
        Register a trained model version.

        Returns the version string (e.g. 'v001').
        """
        checkpoint_path = Path(checkpoint_path)
        config_path = Path(config_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        version = self._next_version()
        model_type = "unknown"

        # Read config to get model type
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_type = cfg.get("model", {}).get("type", "unknown")

        # Create version directory
        version_dir = self.root / f"{version}_{model_type}"
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifacts
        shutil.copy2(checkpoint_path, version_dir / "checkpoint.pt")
        shutil.copy2(config_path, version_dir / "config.yaml")

        if metrics:
            (version_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        # Build metadata
        meta: dict[str, Any] = {
            "version": version,
            "model_type": model_type,
            "description": description,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat(),
            "git_hash": self._git_hash(),
            "checkpoint_hash": self._file_hash(checkpoint_path),
            "config_hash": self._file_hash(config_path),
        }
        if data_path:
            dp = Path(data_path)
            if dp.exists():
                meta["data_hash"] = self._file_hash(dp)
                meta["data_path"] = str(dp)

        (version_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

        # Update index
        entry = {
            "version": version,
            "model_type": model_type,
            "description": description,
            "timestamp": meta["timestamp"],
            "metrics": metrics or {},
            "directory": str(version_dir),
            "tags": tags or [],
        }
        self._index.append(entry)
        self._save_index()

        logger.info(f"Registered model {version} ({model_type}) → {version_dir}")
        return version

    def get(self, version: str) -> dict[str, Any]:
        """Retrieve full metadata for a model version."""
        for entry in self._index:
            if entry["version"] == version:
                version_dir = Path(entry["directory"])
                meta_path = version_dir / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    meta["metrics"] = entry.get("metrics", {})
                    return meta
                return entry
        raise KeyError(f"Version {version} not found in registry")

    def get_checkpoint_path(self, version: str) -> Path:
        """Get the checkpoint file path for a version."""
        for entry in self._index:
            if entry["version"] == version:
                return Path(entry["directory"]) / "checkpoint.pt"
        raise KeyError(f"Version {version} not found")

    def get_best(self, metric: str = "roc_auc", mode: str = "max") -> dict[str, Any]:
        """Find the version with the best value for a metric."""
        best_entry = None
        best_val = -float("inf") if mode == "max" else float("inf")
        for entry in self._index:
            val = entry.get("metrics", {}).get(metric)
            if val is None:
                continue
            if (mode == "max" and val > best_val) or (mode == "min" and val < best_val):
                best_val = val
                best_entry = entry
        if best_entry is None:
            raise ValueError(f"No models with metric '{metric}' in registry")
        return best_entry

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models."""
        return self._index.copy()

    def delete(self, version: str) -> None:
        """Delete a model version from the registry."""
        for i, entry in enumerate(self._index):
            if entry["version"] == version:
                version_dir = Path(entry["directory"])
                if version_dir.exists():
                    shutil.rmtree(version_dir)
                self._index.pop(i)
                self._save_index()
                logger.info(f"Deleted model {version}")
                return
        raise KeyError(f"Version {version} not found")

    def __repr__(self) -> str:
        return f"ModelRegistry(root={self.root}, models={len(self._index)})"
