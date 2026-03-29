# EpiLongAI — Developer Guide

## What is this project?
A production-grade deep learning pipeline for predicting preterm birth (PTB) from Oxford Nanopore (ONT) methylation data. Designed for hospital/research environments.

## Quick start
```bash
pip install -e ".[dev]"          # install with dev dependencies
pip install -e ".[api]"          # install with API server
epilongai --help                 # see all commands
pytest -v                        # run tests
```

## Project layout
```
epilongai/
  data/           → ingestion, windowing, dataset, variants, RNA-seq, positional tracks
  models/         → baseline_mlp, multimodal, long_context (Mamba SSM), population_aware
  training/       → trainer, train, evaluate, predict, metrics, plotting
  analysis/       → interpret (SHAP), benchmark, figures, clinical_scoring, region_labeling
  utils/          → config, logging, seed
  cli.py          → Typer CLI entry points
  api.py          → FastAPI deployment server
configs/          → YAML configs for ingestion + training
workflow/         → Snakemake pipeline
tests/            → pytest suite mirroring package structure
```

## Key commands
```bash
epilongai ingest          # parse ONT methylation files
epilongai window          # create genomic windows
epilongai train           # train model
epilongai evaluate        # evaluate on test set
epilongai predict         # inference on new samples
epilongai interpret       # SHAP / feature importance
epilongai benchmark       # compare against baselines
epilongai variants        # process VCF files
epilongai rna-integrate   # integrate RNA-seq data
epilongai label-regions   # DMR-style case/control labeling
epilongai report          # generate clinical risk reports
epilongai serve           # start FastAPI server
```

## Snakemake workflow
```bash
snakemake --cores 4 -s workflow/Snakefile        # full pipeline
snakemake --cores 4 -s workflow/Snakefile -n      # dry run
```

## Code conventions
- Python 3.10+, type hints throughout
- loguru for logging (not stdlib logging)
- YAML configs for all parameters — no magic numbers in code
- All models return `dict[str, Tensor]` with at least `logits` key
- Binary classification models return `probs` via sigmoid
- Sample-level splits prevent data leakage between windows from same sample

## Model types (set in `configs/train.yaml` → `model.type`)
- `baseline_mlp` — methylation feature MLP
- `multimodal` — sequence CNN/Transformer + methylation MLP + optional variants
- `long_context` — Mamba SSM for 50kb–1Mb sequences

## Testing
```bash
pytest tests/ -v                    # all tests
pytest tests/test_models/ -v        # model tests only
pytest tests/test_data/ -v          # data pipeline tests
pytest tests/test_analysis/ -v      # analysis module tests
```
