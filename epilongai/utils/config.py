"""Configuration loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger


def load_config(path: str | Path, validate: bool = True) -> dict[str, Any]:
    """
    Load a YAML configuration file and return as a dict.

    If validate=True (default), runs Pydantic validation to catch errors early.
    Set validate=False to skip validation (e.g. for partial or non-standard configs).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if validate:
        try:
            from epilongai.utils.schemas import validate_config
            validate_config(path)
        except Exception as e:
            logger.warning(f"Config validation warning for {path}: {e}")

    logger.info(f"Loaded config from {path}")
    return cfg


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


def get_nested(cfg: dict, dotpath: str, default: Any = None) -> Any:
    """Retrieve a value from a nested dict using dot notation, e.g. 'training.batch_size'."""
    keys = dotpath.split(".")
    node = cfg
    for key in keys:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node
