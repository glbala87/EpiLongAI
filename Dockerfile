# =============================================================================
# EpiLongAI — Docker deployment
# =============================================================================
# Multi-stage build: slim runtime image
#
# Build:
#   docker build -t epilongai .
#
# Run (CPU):
#   docker run -p 8000:8000 \
#     -v $(pwd)/checkpoints:/app/checkpoints \
#     -v $(pwd)/configs:/app/configs \
#     epilongai
#
# Run (GPU — requires nvidia-docker):
#   docker run --gpus all -p 8000:8000 \
#     -v $(pwd)/checkpoints:/app/checkpoints \
#     -v $(pwd)/configs:/app/configs \
#     epilongai
# =============================================================================

FROM python:3.11-slim AS base

# OCI image metadata
LABEL org.opencontainers.image.title="EpiLongAI"
LABEL org.opencontainers.image.description="Deep learning pipeline for phenotype prediction from ONT methylation data"
LABEL org.opencontainers.image.source="https://github.com/epilongai/epilongai"
LABEL org.opencontainers.image.vendor="EpiLongAI"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.licenses="MIT"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-api.txt

# Copy application
COPY . .
RUN pip install --no-cache-dir -e .

# Create non-root user
RUN groupadd --gid 1000 epilongai && \
    useradd --uid 1000 --gid epilongai --shell /bin/bash --create-home epilongai && \
    chown -R epilongai:epilongai /app

# Environment variables
ENV EPILONGAI_CONFIG=configs/train.yaml
ENV EPILONGAI_CHECKPOINT=checkpoints/best.pt
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Switch to non-root user
USER epilongai

# Health check — verifies the API is up AND the model is loaded
HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=60s \
    CMD python -c "\
import urllib.request, json, sys; \
r = urllib.request.urlopen('http://localhost:8000/health'); \
d = json.loads(r.read()); \
sys.exit(0 if d.get('model_loaded') else 1)"

# Run
CMD ["uvicorn", "epilongai.api:app", "--host", "0.0.0.0", "--port", "8000"]
