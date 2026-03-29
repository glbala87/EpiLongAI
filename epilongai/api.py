"""
Phase T — FastAPI service for EpiLongAI model deployment.

Endpoints:
    POST /predict_methylation   — predict from methylation file
    POST /predict_sample        — predict from pre-windowed features
    POST /predict_variant       — predict with variant integration
    GET  /health                — health check
    GET  /model_info            — model metadata

GPU inference is supported automatically when CUDA is available.
"""

from __future__ import annotations

import os
import secrets
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel

# ── App setup ────────────────────────────────────────────────────────────
app = FastAPI(
    title="EpiLongAI",
    description="ONT methylation deep learning prediction API",
    version="0.1.0",
)

# Global state (loaded on startup)
_state: dict[str, Any] = {}


# =====================================================================
# Authentication
# =====================================================================
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# API keys loaded from environment variable (comma-separated) or file
def _load_api_keys() -> set[str]:
    """Load valid API keys from EPILONGAI_API_KEYS env var or .api_keys file."""
    keys: set[str] = set()
    # From environment
    env_keys = os.environ.get("EPILONGAI_API_KEYS", "")
    if env_keys:
        keys.update(k.strip() for k in env_keys.split(",") if k.strip())
    # From file
    keyfile = Path(os.environ.get("EPILONGAI_API_KEYS_FILE", ".api_keys"))
    if keyfile.exists():
        keys.update(line.strip() for line in keyfile.read_text().splitlines() if line.strip() and not line.startswith("#"))
    return keys


def _auth_enabled() -> bool:
    return os.environ.get("EPILONGAI_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")


async def verify_api_key(api_key: str | None = Depends(_API_KEY_HEADER)) -> str | None:
    """Dependency that verifies the API key if auth is enabled."""
    if not _auth_enabled():
        return None  # auth disabled — allow all
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")
    valid_keys = _load_api_keys()
    if not valid_keys:
        logger.warning("Auth enabled but no API keys configured — rejecting all requests")
        raise HTTPException(status_code=500, detail="Server has no API keys configured")
    # Constant-time comparison
    if not any(secrets.compare_digest(api_key, k) for k in valid_keys):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key


# =====================================================================
# Rate Limiting
# =====================================================================
class RateLimiter:
    """Simple in-memory sliding-window rate limiter."""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def check(self, client_id: str) -> bool:
        now = time.time()
        window_start = now - self.window
        # Prune old entries
        self._requests[client_id] = [t for t in self._requests[client_id] if t > window_start]
        if len(self._requests[client_id]) >= self.max_requests:
            return False
        self._requests[client_id].append(now)
        return True


_rate_limiter = RateLimiter(
    max_requests=int(os.environ.get("EPILONGAI_RATE_LIMIT", "60")),
    window_seconds=60,
)


async def rate_limit(request: Request) -> None:
    """Rate-limiting dependency."""
    client = request.client.host if request.client else "unknown"
    if not _rate_limiter.check(client):
        raise HTTPException(status_code=429, detail="Rate limit exceeded — try again later")


# =====================================================================
# Input Validation
# =====================================================================
MAX_UPLOAD_SIZE = int(os.environ.get("EPILONGAI_MAX_UPLOAD_MB", "500")) * 1024 * 1024  # bytes


async def validate_upload_size(file: UploadFile) -> bytes:
    """Read and validate file upload size."""
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / 1e6:.1f}MB). Max: {MAX_UPLOAD_SIZE / 1e6:.0f}MB",
        )
    return content


class PredictionRequest(BaseModel):
    """Request body for /predict_sample."""
    features: list[list[float]]  # list of feature vectors
    sample_ids: list[str] | None = None


class PredictionResponse(BaseModel):
    """Response body for predictions."""
    predictions: list[dict[str, Any]]
    model_info: dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ── Startup ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_load_model() -> None:
    """Load model and config on application startup."""
    import os

    config_path = os.environ.get("EPILONGAI_CONFIG", "configs/train.yaml")
    checkpoint_path = os.environ.get("EPILONGAI_CHECKPOINT", "checkpoints/best.pt")

    try:
        from epilongai.utils.config import load_config
        cfg = load_config(config_path)
        _state["config"] = cfg
        _state["config_path"] = config_path

        model_cfg = cfg["model"]
        ds_cfg = cfg.get("dataset", {})
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _state["device"] = device

        # Determine input dim from config
        from epilongai.data.dataset import METHYLATION_FEATURE_COLS
        input_dim = len(METHYLATION_FEATURE_COLS)

        mtype = model_cfg.get("type", "baseline_mlp")
        if mtype == "baseline_mlp":
            from epilongai.models.baseline_mlp import BaselineMLP
            model = BaselineMLP.from_config(model_cfg, input_dim=input_dim)
        elif mtype == "long_context":
            from epilongai.models.long_context_model import LongContextGenomicModel
            model = LongContextGenomicModel.from_config(model_cfg)
        else:
            from epilongai.models.multimodal_model import MultimodalModel
            model_cfg["_dataset_mode"] = ds_cfg.get("mode", "methylation")
            model = MultimodalModel.from_config(model_cfg, methylation_input_dim=input_dim)

        if Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            _state["epoch"] = ckpt.get("epoch", "unknown")
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path} — using random weights")
            _state["epoch"] = "none"

        model.to(device)
        model.eval()
        _state["model"] = model
        _state["model_type"] = mtype
        _state["loaded"] = True
        logger.info(f"Model ready on {device}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        _state["loaded"] = False


# ── Endpoints ────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if _state.get("loaded") else "model_not_loaded",
        model_loaded=_state.get("loaded", False),
        device=_state.get("device", "unknown"),
    )


@app.get("/model_info", dependencies=[Depends(verify_api_key), Depends(rate_limit)])
async def model_info() -> dict:
    return {
        "model_type": _state.get("model_type", "unknown"),
        "device": _state.get("device", "unknown"),
        "epoch": _state.get("epoch", "unknown"),
        "config_path": _state.get("config_path", "unknown"),
    }


@app.post("/predict_sample", response_model=PredictionResponse, dependencies=[Depends(verify_api_key), Depends(rate_limit)])
async def predict_sample(request: PredictionRequest) -> PredictionResponse:
    """Predict from pre-computed feature vectors."""
    if not _state.get("loaded"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = _state["model"]
    device = _state["device"]
    cfg = _state["config"]

    features = torch.tensor(request.features, dtype=torch.float32, device=device)

    with torch.no_grad():
        from epilongai.models.baseline_mlp import BaselineMLP
        if isinstance(model, BaselineMLP):
            out = model(features)
        else:
            out = model(methylation=features)

    predictions = []
    num_classes = cfg["model"].get("num_classes", 2)
    label_map = cfg.get("data", {}).get("label_map", {})
    inv_map = {v: k for k, v in label_map.items()}

    if "probs" in out:
        probs = out["probs"].cpu().numpy()
        if num_classes == 2:
            for i, p in enumerate(probs.ravel()):
                pred_class = 1 if p >= 0.5 else 0
                predictions.append({
                    "sample_id": request.sample_ids[i] if request.sample_ids else f"sample_{i}",
                    "predicted_class": pred_class,
                    "predicted_label": inv_map.get(pred_class, str(pred_class)),
                    "probability": float(p),
                })
        else:
            for i, row in enumerate(probs):
                pred_class = int(row.argmax())
                predictions.append({
                    "sample_id": request.sample_ids[i] if request.sample_ids else f"sample_{i}",
                    "predicted_class": pred_class,
                    "predicted_label": inv_map.get(pred_class, str(pred_class)),
                    "probabilities": row.tolist(),
                })
    else:
        for i, val in enumerate(out["logits"].cpu().numpy().ravel()):
            predictions.append({
                "sample_id": request.sample_ids[i] if request.sample_ids else f"sample_{i}",
                "predicted_value": float(val),
            })

    return PredictionResponse(
        predictions=predictions,
        model_info={"model_type": _state["model_type"], "device": device},
    )


@app.post("/predict_methylation", dependencies=[Depends(verify_api_key), Depends(rate_limit)])
async def predict_methylation(
    file: UploadFile = File(...),
) -> JSONResponse:
    """
    Predict from a raw methylation file.
    Runs full pipeline: ingest → window → predict.
    """
    if not _state.get("loaded"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate and save uploaded file to temp
    content = await validate_upload_size(file)
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="wb") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from epilongai.data.data_ingestion import parse_methylation_file
        from epilongai.data.dataset import METHYLATION_FEATURE_COLS
        from epilongai.data.windowing import compute_window_features_fast

        cfg = _state["config"]
        ing_cfg = cfg.get("ingestion", {})
        win_cfg = cfg.get("windowing", {})

        meth = parse_methylation_file(
            tmp_path,
            min_coverage=ing_cfg.get("min_coverage", 5),
        )
        windows = compute_window_features_fast(
            meth,
            window_size=win_cfg.get("window_size", 1000),
            stride=win_cfg.get("stride", 1000),
            min_cpgs=win_cfg.get("min_cpgs_per_window", 3),
        )

        feature_cols = [c for c in METHYLATION_FEATURE_COLS if c in windows.columns]
        features = torch.tensor(
            windows[feature_cols].fillna(0).values,
            dtype=torch.float32,
            device=_state["device"],
        )

        model = _state["model"]
        with torch.no_grad():
            from epilongai.models.baseline_mlp import BaselineMLP
            if isinstance(model, BaselineMLP):
                out = model(features)
            else:
                out = model(methylation=features)

        if "probs" in out:
            probs = out["probs"].cpu().numpy().ravel()
            mean_score = float(probs.mean())
            label_map = cfg.get("data", {}).get("label_map", {})
            inv_map = {v: k for k, v in label_map.items()}
            predicted_label = inv_map.get(1, "positive") if mean_score >= 0.5 else inv_map.get(0, "negative")
        else:
            mean_score = float(out["logits"].cpu().numpy().mean())
            predicted_label = "N/A"

        return JSONResponse({
            "filename": file.filename,
            "n_sites": len(meth),
            "n_windows": len(windows),
            "mean_risk_score": round(mean_score, 4),
            "predicted_label": predicted_label,
        })

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/predict_variant", dependencies=[Depends(verify_api_key), Depends(rate_limit)])
async def predict_variant(
    methylation_file: UploadFile = File(...),
    vcf_file: UploadFile = File(...),
) -> JSONResponse:
    """
    Predict using methylation + variant data.
    """
    if not _state.get("loaded"):
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate and save uploaded files
    meth_content = await validate_upload_size(methylation_file)
    vcf_content = await validate_upload_size(vcf_file)

    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="wb") as tmp1:
        tmp1.write(meth_content)
        meth_path = tmp1.name

    with tempfile.NamedTemporaryFile(suffix=".vcf", delete=False, mode="wb") as tmp2:
        tmp2.write(vcf_content)
        vcf_path = tmp2.name

    try:
        from epilongai.data.data_ingestion import parse_methylation_file
        from epilongai.data.dataset import METHYLATION_FEATURE_COLS, VARIANT_FEATURE_COLS
        from epilongai.data.variant_processing import map_variants_to_windows, parse_vcf
        from epilongai.data.windowing import compute_window_features_fast

        cfg = _state["config"]

        meth = parse_methylation_file(meth_path)
        windows = compute_window_features_fast(meth)

        variants = parse_vcf(vcf_path)
        var_windows = map_variants_to_windows(variants)

        # Merge
        join_cols = [c for c in ["chr", "window_start", "window_end"] if c in windows.columns]
        merged = windows.merge(var_windows, on=join_cols, how="left").fillna(0)

        meth_cols = [c for c in METHYLATION_FEATURE_COLS if c in merged.columns]
        var_cols = [c for c in VARIANT_FEATURE_COLS if c in merged.columns]

        meth_features = torch.tensor(merged[meth_cols].values, dtype=torch.float32, device=_state["device"])
        var_features = torch.tensor(merged[var_cols].values, dtype=torch.float32, device=_state["device"]) if var_cols else None

        model = _state["model"]
        with torch.no_grad():
            from epilongai.models.baseline_mlp import BaselineMLP
            if isinstance(model, BaselineMLP):
                out = model(meth_features)
            else:
                kwargs: dict[str, Any] = {"methylation": meth_features}
                if var_features is not None:
                    kwargs["variants"] = var_features
                out = model(**kwargs)

        if "probs" in out:
            mean_score = float(out["probs"].cpu().numpy().ravel().mean())
        else:
            mean_score = float(out["logits"].cpu().numpy().mean())

        label_map = cfg.get("data", {}).get("label_map", {})
        inv_map = {v: k for k, v in label_map.items()}

        return JSONResponse({
            "n_methylation_sites": len(meth),
            "n_variants": len(variants),
            "n_windows": len(merged),
            "mean_risk_score": round(mean_score, 4),
            "predicted_label": inv_map.get(1, "positive") if mean_score >= 0.5 else inv_map.get(0, "negative"),
        })

    finally:
        Path(meth_path).unlink(missing_ok=True)
        Path(vcf_path).unlink(missing_ok=True)
