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

# Environment variables
ENV EPILONGAI_CONFIG=configs/train.yaml
ENV EPILONGAI_CHECKPOINT=checkpoints/best.pt
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run
CMD ["uvicorn", "epilongai.api:app", "--host", "0.0.0.0", "--port", "8000"]
