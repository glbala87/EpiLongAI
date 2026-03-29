# EpiLongAI

**Production-grade deep learning pipeline for Oxford Nanopore (ONT) methylation data analysis and phenotype prediction.**

[![CI](https://github.com/glbala87/EpiLongAI/actions/workflows/ci.yml/badge.svg)](https://github.com/glbala87/EpiLongAI/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

EpiLongAI is an end-to-end pipeline for training deep learning models on ONT sequencing data integrated with methylation features. It supports **any binary, multiclass, or regression phenotype** prediction from methylation data — including but not limited to preterm birth, cancer subtypes, neurological disorders, aging, and environmental exposure effects. The pipeline is phenotype-agnostic: define your labels in a metadata file and the entire pipeline adapts.

### Key Features

- **Multi-format ingestion** -- bedMethyl, tab-delimited methylation, VCF, RNA-seq expression
- **Genomic windowing** -- configurable window size, vectorised feature extraction
- **Three model architectures**:
  - Baseline MLP (methylation features)
  - Multimodal CNN/Transformer (sequence + methylation + variants)
  - Long-context Mamba SSM (50kb--1Mb sequences)
- **Population-aware modeling** -- learnable population embeddings with FiLM conditioning
- **Production-ready** -- Pydantic config validation, API key auth, rate limiting, Docker, CI/CD
- **Manuscript-ready** -- publication-quality figures, benchmarking framework, clinical risk reports
- **Reproducible** -- Snakemake workflow, YAML configs, seed control, model registry

---

## Architecture

```
ONT Methylation Files        VCF Files          RNA-seq
        |                       |                  |
        v                       v                  v
   +---------+            +-----------+      +----------+
   | Ingest  |            | Variants  |      | RNA-seq  |
   | & QC    |            | Processing|      | Integrate|
   +---------+            +-----------+      +----------+
        |                       |                  |
        v                       v                  v
   +---------+         +-----------------------------------+
   | Genomic |-------->| PyTorch Dataset (train/val/test)  |
   | Windows |         | Sample-level split (no leakage)   |
   +---------+         +-----------------------------------+
                                |
          +---------------------+----------------------+
          |                     |                      |
   +------v------+     +-------v--------+    +--------v---------+
   | Sequence    |     | Methylation    |    | Variant          |
   | Branch      |     | Branch (MLP)   |    | Branch (MLP)     |
   | (CNN/SSM)   |     |                |    |                  |
   +------+------+     +-------+--------+    +--------+---------+
          |                     |                      |
          +----------+----------+----------+-----------+
                     |
              +------v------+
              |   Fusion    |
              | (concat /   |
              | cross-attn) |
              +------+------+
                     |
              +------v------+
              | Output Head |
              | PTB vs FTB  |
              +------+------+
                     |
       +-------------+-------------+
       |             |             |
  +----v----+  +-----v-----+  +---v--------+
  | Predict |  | Interpret  |  | Clinical   |
  | & Score |  | (SHAP/IG)  |  | Reports    |
  +----+----+  +-----+-----+  +---+--------+
       |             |             |
       v             v             v
   CSV/JSON    Figures/CSV    PDF/TXT Reports
```

---

## Installation

### From source (recommended)

```bash
git clone https://github.com/glbala87/EpiLongAI.git
cd EpiLongAI
pip install -e .
```

### With optional dependencies

```bash
pip install -e ".[dev]"          # development (pytest, ruff, mypy)
pip install -e ".[api]"          # FastAPI deployment server
pip install -e ".[workflow]"     # Snakemake workflow engine
pip install -e ".[dev,api]"      # everything
```

### Optional: CUDA Mamba kernel (3x faster long-context model)

```bash
pip install mamba-ssm            # requires CUDA toolkit
```

### Verify installation

```bash
epilongai --help
pytest tests/ -v
```

---

## Quick Start

### 1. Generate synthetic test data

```bash
python scripts/generate_synthetic_data.py \
    --output data/synthetic \
    --n-samples 20 \
    --n-sites 2000
```

### 2. Run the pipeline

```bash
# Ingest methylation files
epilongai ingest \
    --input-dir data/synthetic/methylation \
    --metadata data/synthetic/metadata.tsv \
    --config configs/default.yaml \
    --output data/processed/

# Create genomic windows
epilongai window \
    --input data/processed/methylation_merged.parquet \
    --config configs/default.yaml \
    --output data/windows/

# Train model
epilongai train --config configs/train.yaml

# Evaluate
epilongai evaluate \
    --config configs/train.yaml \
    --checkpoint checkpoints/best.pt \
    --split test

# Predict on new samples
epilongai predict \
    --input-dir data/synthetic/methylation \
    --checkpoint checkpoints/best.pt \
    --config configs/train.yaml \
    --output results/predictions/
```

### Or run everything with Snakemake

```bash
snakemake --cores 4 -s workflow/Snakefile
```

---

## CLI Commands

| Command | Description |
|---|---|
| `epilongai ingest` | Parse and validate ONT methylation files |
| `epilongai window` | Create fixed-size genomic windows with features |
| `epilongai train` | Train a deep learning model |
| `epilongai evaluate` | Evaluate on held-out test set |
| `epilongai predict` | Run inference on new samples |
| `epilongai interpret` | Generate SHAP explanations and top regions |
| `epilongai benchmark` | Compare against baseline models (LR, RF, GBM) |
| `epilongai variants` | Parse VCF files into per-window features |
| `epilongai rna-integrate` | Integrate RNA-seq expression data |
| `epilongai label-regions` | DMR-style case/control window labeling |
| `epilongai report` | Generate clinical risk reports |
| `epilongai register` | Register a trained model in the registry |
| `epilongai models` | List all registered model versions |
| `epilongai serve` | Start the FastAPI prediction server |

---

## Model Types

Configure in `configs/train.yaml` under `model.type`:

### Baseline MLP (`baseline_mlp`)

Best for small-to-medium cohorts with methylation features only.

```yaml
model:
  type: baseline_mlp
  mlp:
    hidden_dims: [256, 128, 64]
    dropout: 0.3
    batch_norm: true
```

### Multimodal (`multimodal`)

Combines DNA sequence (1D CNN or Transformer) + methylation (MLP) + optional variant features.

```yaml
model:
  type: multimodal
dataset:
  mode: multimodal        # or: full (adds variants)
```

### Long-Context Mamba SSM (`long_context`)

State-space model for 50kb--1Mb sequences. Linear memory scaling. Auto-uses CUDA kernel when available.

```yaml
model:
  type: long_context
  long_context:
    d_model: 256
    n_layers: 6
    d_state: 16
    gradient_checkpointing: true
```

### Population-Aware (any backbone)

Wrap any model with population conditioning. Supports concatenation or FiLM.

```yaml
model:
  population:
    enabled: true
    n_populations: 10
    embed_dim: 16
    conditioning: film     # or: concatenate
```

---

## Project Structure

```
EpiLongAI/
├── configs/
│   ├── default.yaml              # Ingestion, windowing, labeling params
│   ├── train.yaml                # Model, training, data split params
│   └── metadata_template.tsv     # Example metadata format
├── epilongai/
│   ├── cli.py                    # Typer CLI (14 commands)
│   ├── api.py                    # FastAPI server with auth & rate limiting
│   ├── data/
│   │   ├── data_ingestion.py     # bedMethyl/tabular parsing, chunked I/O
│   │   ├── windowing.py          # Genomic windows, vectorised features
│   │   ├── dataset.py            # PyTorch Dataset/DataLoader, 5 modes
│   │   ├── positional_tracks.py  # Position-aware methylation for CNN
│   │   ├── variant_processing.py # VCF parsing, variant-to-window mapping
│   │   └── rna_integration.py    # RNA-seq expression integration
│   ├── models/
│   │   ├── baseline_mlp.py       # Configurable MLP
│   │   ├── multimodal_model.py   # Seq + Meth + Variants fusion
│   │   ├── long_context_model.py # Mamba SSM (50kb-1Mb)
│   │   └── population_aware.py   # Population embeddings + FiLM
│   ├── training/
│   │   ├── trainer.py            # Training loop, AMP, early stopping
│   │   ├── train.py              # Training orchestrator
│   │   ├── evaluate.py           # Evaluation with plots
│   │   ├── predict.py            # Inference pipeline
│   │   ├── metrics.py            # Classification + regression metrics
│   │   └── plotting.py           # Publication-ready matplotlib plots
│   ├── analysis/
│   │   ├── interpret.py          # SHAP, integrated gradients
│   │   ├── benchmark.py          # Cross-validation, DeLong test
│   │   ├── figures.py            # Journal-quality figures & tables
│   │   ├── clinical_scoring.py   # Risk scores & patient reports
│   │   └── region_labeling.py    # DMR-style hyper/hypo labeling
│   └── utils/
│       ├── config.py             # YAML loading with auto-validation
│       ├── schemas.py            # Pydantic config validation (20+ models)
│       ├── logging.py            # Loguru setup
│       ├── seed.py               # Reproducibility seeding
│       └── model_registry.py     # Model versioning & lineage tracking
├── workflow/
│   ├── Snakefile                 # Full pipeline DAG (11 rules)
│   ├── config.yaml               # Workflow path configuration
│   └── envs/epilongai.yaml       # Conda environment spec
├── scripts/
│   └── generate_synthetic_data.py # Synthetic data for testing
├── tests/                         # 87 tests across 13 test files
├── Dockerfile                     # GPU-ready Docker deployment
├── pyproject.toml                 # Package config & dependencies
├── requirements.txt               # pip dependencies
├── requirements-api.txt           # API server dependencies
├── .github/workflows/ci.yml      # GitHub Actions CI/CD
└── .gitignore
```

---

## Data Format

### Methylation files

Tab-delimited with any of these column names (auto-detected):

| Column | Aliases |
|---|---|
| Chromosome | `chr`, `chrom`, `chromosome`, `#chrom` |
| Position | `start`, `pos`, `position` |
| Coverage | `coverage`, `N`, `total`, `depth` |
| Modified count | `modified_count`, `X`, `n_mod` |
| Beta value | `beta`, `freq`, `methylation_frequency` |

### Metadata file

TSV with at minimum:

```
sample_id    group    phenotype       batch    source
sample_001   PTB      preterm_birth   batch1   cord_blood
sample_002   FTB      full_term_birth batch1   maternal_blood
```

### VCF files (optional)

Standard VCF v4.x format. Supports `.vcf` and `.vcf.gz`.

---

## Configuration

All parameters are controlled via YAML configs. No magic numbers in code.

### `configs/default.yaml` -- Data processing

```yaml
ingestion:
  min_coverage: 5
  chunk_size: 500000

windowing:
  window_size: 1000
  stride: 1000
  min_cpgs_per_window: 3

labeling:
  case_label: "PTB"
  control_label: "FTB"
  delta_beta_threshold: 0.1
  pvalue_threshold: 0.05
```

### `configs/train.yaml` -- Model & training

```yaml
model:
  type: baseline_mlp          # baseline_mlp | multimodal | long_context
  task: classification
  num_classes: 2

training:
  epochs: 100
  batch_size: 64
  learning_rate: 0.001
  optimizer: adamw
  early_stopping:
    enabled: true
    patience: 15
    metric: val_f1
  mixed_precision: true
  class_weights: balanced
```

Config validation is automatic via Pydantic -- typos and invalid values are caught at load time.

---

## API Deployment

### Local server

```bash
epilongai serve --port 8000 --checkpoint checkpoints/best.pt
```

### Docker

```bash
docker build -t epilongai .

# CPU
docker run -p 8000:8000 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    epilongai

# GPU
docker run --gpus all -p 8000:8000 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    epilongai
```

### Authentication

```bash
# Enable auth via environment
EPILONGAI_AUTH_ENABLED=true \
EPILONGAI_API_KEYS=my-secret-key-1,my-secret-key-2 \
epilongai serve
```

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check (no auth required) |
| `GET` | `/model_info` | Model metadata |
| `POST` | `/predict_sample` | Predict from feature vectors (JSON) |
| `POST` | `/predict_methylation` | Predict from methylation file upload |
| `POST` | `/predict_variant` | Predict from methylation + VCF upload |

### Example request

```bash
curl -X POST http://localhost:8000/predict_sample \
    -H "Content-Type: application/json" \
    -H "X-API-Key: my-secret-key-1" \
    -d '{
        "features": [[0.65, 0.60, 0.02, 8, 25.0, 0.3, 0.1]],
        "sample_ids": ["patient_001"]
    }'
```

---

## Snakemake Workflow

Run the full pipeline as a reproducible DAG:

```bash
# Full pipeline
snakemake --cores 4 -s workflow/Snakefile

# Dry run (preview)
snakemake --cores 4 -s workflow/Snakefile -n

# Only preprocessing
snakemake --cores 4 -s workflow/Snakefile preprocess

# Only through training
snakemake --cores 4 -s workflow/Snakefile train

# Visualise the DAG
snakemake -s workflow/Snakefile --dag | dot -Tpng > dag.png

# Override data paths
snakemake --cores 4 -s workflow/Snakefile \
    --config raw_dir=my/data metadata=my/metadata.tsv
```

**DAG:**

```
ingest -> window -> train_model -> evaluate  -> figures
                                -> interpret
                                -> predict   -> reports
                                -> benchmark
                 -> label_regions (optional)
```

---

## Benchmarking

Compare your model against classical baselines:

```bash
epilongai benchmark \
    --config configs/train.yaml \
    --output results/benchmark/ \
    --folds 5
```

**Baselines included:**
- Logistic Regression (balanced)
- Random Forest (200 trees)
- Gradient Boosting (200 trees)

**Statistical testing:** DeLong test for AUC comparison between models.

---

## Clinical Reports

Generate per-patient risk assessments:

```bash
epilongai report \
    --predictions results/predictions/window_predictions.csv \
    --checkpoint checkpoints/best.pt \
    --output results/reports/
```

**Output per sample:**

```
======================================================================
EpiLongAI -- Methylation Risk Assessment Report
======================================================================

DISCLAIMER: This report is for RESEARCH USE ONLY.

RISK ASSESSMENT
  Risk Score:      0.79 (0 = low, 1 = high)
  Risk Category:   HIGH
  Confidence:      0.73

TOP CONTRIBUTING GENOMIC REGIONS
   1. chr11:4372000-4373000  score=0.9976
   2. chr13:8227000-8228000  score=0.9973
   ...
======================================================================
```

---

## Model Registry

Track model versions with full lineage:

```bash
# Register after training
epilongai register \
    --checkpoint checkpoints/best.pt \
    --config configs/train.yaml \
    --metrics results/metrics_test.json \
    --description "Baseline MLP v1, cord blood cohort"

# List all models
epilongai models

# Output:
# v001  baseline_mlp  2026-03-29  roc_auc=0.935  f1=0.819  Baseline MLP v1
```

Each version stores: checkpoint, frozen config, metrics, git hash, data hash.

---

## End-to-End Test Results

Tested on 20 synthetic samples (10 PTB, 10 FTB):

| Metric | Value |
|---|---|
| Accuracy | 0.872 |
| Precision | 0.777 |
| Recall | 0.865 |
| F1 | 0.819 |
| **ROC-AUC** | **0.935** |
| PR-AUC | 0.871 |

Pipeline produced: training curves, ROC/PR/CM plots, sample predictions, top predictive windows, 20 individual clinical reports, model registered in versioned registry.

---

## Testing

```bash
# Run all 87 tests
pytest tests/ -v

# By module
pytest tests/test_models/ -v       # model architectures
pytest tests/test_data/ -v         # data pipeline
pytest tests/test_analysis/ -v     # analysis modules
pytest tests/test_training/ -v     # metrics

# With coverage
pytest tests/ --cov=epilongai --cov-report=html

# Lint
ruff check epilongai/ tests/
```

---

## Biological Context

### Why ONT methylation?

- DNA methylation at CpG sites is a stable epigenetic mark measurable from blood, tissue, or any biological sample
- ONT long reads detect methylation natively during sequencing — no bisulfite conversion needed
- Long reads (10–100kb) preserve haplotype and long-range methylation structure
- Methylation is implicated in hundreds of phenotypes: cancer, preterm birth, neurodegeneration, aging, metabolic disease, immune disorders, environmental exposures

### Example use cases

| Phenotype | Labels | Sample type |
|---|---|---|
| Preterm birth | PTB vs FTB | Cord/maternal blood |
| Cancer subtyping | Tumor type A/B/C | Tumor biopsy |
| Aging clock | Chronological age (regression) | Blood |
| Smoking exposure | Smoker vs non-smoker | Blood/lung tissue |
| Neurological disease | Case vs control | Brain tissue / CSF |
| Immune cell deconvolution | Cell type proportions (regression) | PBMCs |

### Why deep learning?

- Traditional approaches (limma, DMR tools) test one region at a time
- Deep learning captures non-linear interactions between methylation regions
- Multimodal models integrate sequence context, variant effects, and expression
- Long-context SSMs can model regulatory relationships across 100kb+ distances

### Limitations

- Model predictions are statistical associations, not causal mechanisms
- Performance depends on cohort size, population composition, and sample processing
- Clinical deployment requires prospective validation in independent cohorts
- Batch effects from different ONT chemistries/basecallers can confound results

---

## Citation

If you use EpiLongAI in your research, please cite:

```bibtex
@software{epilongai2026,
  title = {EpiLongAI: Deep Learning Pipeline for Oxford Nanopore Methylation-Based Phenotype Prediction},
  year = {2026},
  url = {https://github.com/glbala87/EpiLongAI}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Run tests (`pytest tests/ -v`)
4. Run linting (`ruff check epilongai/ tests/`)
5. Commit and push
6. Open a Pull Request

CI runs automatically on all PRs: tests on Python 3.10/3.11/3.12, ruff linting, mypy type checking.
