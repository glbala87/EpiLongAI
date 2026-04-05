"""Configuration loading and validation utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# Directories that should never be accessed by CLI path arguments.
_SENSITIVE_DIRS: tuple[str, ...] = (
    "/etc",
    "/proc",
    "/sys",
    "/dev",
    "/var/run",
    "/private/etc",          # macOS equivalent of /etc
)


def sanitize_path(path: str | Path, must_exist: bool = False) -> Path:
    """Resolve *path* and guard against directory-traversal attacks.

    The function:
    1. Resolves the path to an absolute, symlink-free form.
    2. Rejects paths whose *original* text contains ``..`` components that
       would escape the current working directory.
    3. Rejects paths that land inside system-sensitive directories.

    Parameters
    ----------
    path:
        A file-system path (string or ``Path``).
    must_exist:
        If ``True``, raise ``FileNotFoundError`` when the resolved path does
        not exist on disk.

    Returns
    -------
    Path
        The resolved, validated ``Path`` object.

    Raises
    ------
    ValueError
        If the path contains a traversal attempt or targets a sensitive
        directory.
    FileNotFoundError
        If *must_exist* is ``True`` and the path does not exist.
    """
    raw = str(path)
    resolved = Path(raw).resolve()

    # Check for ".." components that escape the working directory.
    cwd = Path.cwd().resolve()
    # Normalise the raw input *without* resolving symlinks so we can inspect
    # the literal components the caller supplied.
    parts = Path(raw).parts
    if ".." in parts:
        # Walk the components to see if they ever leave cwd
        probe = cwd
        for part in Path(raw).parts:
            probe = (probe / part).resolve()
        # After walking, the result must still be under cwd (or equal to it)
        try:
            probe.relative_to(cwd)
        except ValueError:
            raise ValueError(
                f"Path traversal detected: '{raw}' escapes the working directory"
            )

    # Block sensitive directories.
    resolved_str = str(resolved)
    for sensitive in _SENSITIVE_DIRS:
        if resolved_str == sensitive or resolved_str.startswith(sensitive + "/"):
            raise ValueError(
                f"Access denied: path '{raw}' resolves into sensitive directory {sensitive}"
            )

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")

    return resolved


def _auto_cast(value: str) -> bool | int | float | str:
    """Cast a string value to bool, int, float, or leave as str."""
    low = value.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    # Try int first (stricter), then float.
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_env_overrides(cfg: dict) -> dict:
    """Apply ``EPILONGAI_``-prefixed environment variables as config overrides.

    Naming convention (case-insensitive on the key side):
    * ``EPILONGAI_TRAINING__BATCH_SIZE=64``  ->  ``cfg["training"]["batch_size"] = 64``
    * Double underscore ``__`` separates nested dict keys.
    * Single underscore ``_`` is kept as-is within a key segment.
    * Values are auto-cast: ``"true"``/``"false"`` -> bool, numeric strings
      -> int/float, otherwise str.

    Returns a **new** dict (the original is not mutated).
    """
    prefix = "EPILONGAI_"
    result = _deep_copy_dict(cfg)

    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        # Strip the prefix and lower-case for dict keys.
        raw_key = env_key[len(prefix):]
        # Skip internal vars like EPILONGAI_CONFIG / EPILONGAI_CHECKPOINT that
        # are not config-override keys (they don't contain __ for nesting and
        # are used by the serve command for env passthrough).
        if "__" not in raw_key:
            continue
        segments = [seg.lower() for seg in raw_key.split("__")]
        node = result
        for seg in segments[:-1]:
            if seg not in node or not isinstance(node[seg], dict):
                node[seg] = {}
            node = node[seg]
        node[segments[-1]] = _auto_cast(env_val)
        logger.debug(f"Env override: {env_key} -> {'.'.join(segments)} = {node[segments[-1]]!r}")

    return result


def _deep_copy_dict(d: dict) -> dict:
    """Recursively copy a dict of dicts (avoids importing copy)."""
    out: dict = {}
    for k, v in d.items():
        out[k] = _deep_copy_dict(v) if isinstance(v, dict) else v
    return out


def load_config(path: str | Path, validate: bool = True) -> dict[str, Any]:
    """
    Load a YAML configuration file and return as a dict.

    If validate=True (default), runs Pydantic validation to catch errors early.
    Set validate=False to skip validation (e.g. for partial or non-standard configs).

    After loading, any ``EPILONGAI_``-prefixed environment variables are applied
    as overrides (see :func:`apply_env_overrides`).
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

    # Apply environment-variable overrides.
    cfg = apply_env_overrides(cfg)

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
