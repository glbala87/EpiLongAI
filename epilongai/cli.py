"""EpiLongAI command-line interface."""

import typer

app = typer.Typer(
    name="epilongai",
    help="Deep learning pipeline for ONT methylation data analysis.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Subcommand: ingest
# ---------------------------------------------------------------------------
@app.command()
def ingest(
    input_dir: str = typer.Option(..., "--input-dir", "-i", help="Directory with methylation files"),
    metadata: str = typer.Option(None, "--metadata", "-m", help="Sample metadata TSV/CSV"),
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="YAML config path"),
    output: str = typer.Option("data/processed", "--output", "-o", help="Output directory"),
) -> None:
    """Parse and validate ONT methylation files."""
    from epilongai.data.data_ingestion import run_ingestion

    run_ingestion(input_dir=input_dir, metadata_path=metadata, config_path=config, output_dir=output)


# ---------------------------------------------------------------------------
# Subcommand: window
# ---------------------------------------------------------------------------
@app.command()
def window(
    input_path: str = typer.Option(..., "--input", "-i", help="Ingested methylation parquet/CSV"),
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="YAML config path"),
    output: str = typer.Option("data/windows", "--output", "-o", help="Output directory"),
    fasta: str = typer.Option(None, "--fasta", "-f", help="Reference genome FASTA"),
) -> None:
    """Create genomic windows from methylation data."""
    from epilongai.data.windowing import run_windowing

    run_windowing(input_path=input_path, config_path=config, output_dir=output, fasta_path=fasta)


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------
@app.command()
def train(
    config: str = typer.Option("configs/train.yaml", "--config", "-c", help="Training YAML config"),
    resume: str = typer.Option(None, "--resume", "-r", help="Checkpoint path to resume from"),
) -> None:
    """Train a model on windowed methylation data."""
    from epilongai.training.train import run_training

    run_training(config_path=config, resume_path=resume)


# ---------------------------------------------------------------------------
# Subcommand: evaluate
# ---------------------------------------------------------------------------
@app.command()
def evaluate(
    config: str = typer.Option("configs/train.yaml", "--config", "-c", help="Training YAML config"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-k", help="Model checkpoint path"),
    split: str = typer.Option("test", "--split", "-s", help="Data split to evaluate on"),
) -> None:
    """Evaluate a trained model."""
    from epilongai.training.evaluate import run_evaluation

    run_evaluation(config_path=config, checkpoint_path=checkpoint, split=split)


# ---------------------------------------------------------------------------
# Subcommand: predict
# ---------------------------------------------------------------------------
@app.command()
def predict(
    input_dir: str = typer.Option(..., "--input-dir", "-i", help="Directory with new methylation files"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-k", help="Model checkpoint path"),
    config: str = typer.Option("configs/train.yaml", "--config", "-c", help="YAML config"),
    output: str = typer.Option("results/predictions", "--output", "-o", help="Output directory"),
    fasta: str = typer.Option(None, "--fasta", "-f", help="Reference FASTA"),
) -> None:
    """Run inference on new samples."""
    from epilongai.training.predict import run_prediction

    run_prediction(
        input_dir=input_dir,
        checkpoint_path=checkpoint,
        config_path=config,
        output_dir=output,
        fasta_path=fasta,
    )


# ---------------------------------------------------------------------------
# Subcommand: interpret
# ---------------------------------------------------------------------------
@app.command()
def interpret(
    checkpoint: str = typer.Option(..., "--checkpoint", "-k", help="Model checkpoint path"),
    config: str = typer.Option("configs/train.yaml", "--config", "-c", help="YAML config"),
    output: str = typer.Option("results/interpretability", "--output", "-o", help="Output dir"),
    top_k: int = typer.Option(50, "--top-k", help="Number of top features/regions to report"),
) -> None:
    """Generate interpretability reports for a trained model."""
    from epilongai.analysis.interpret import run_interpretation

    run_interpretation(
        checkpoint_path=checkpoint, config_path=config, output_dir=output, top_k=top_k
    )


# ---------------------------------------------------------------------------
# Subcommand: label-regions
# ---------------------------------------------------------------------------
@app.command("label-regions")
def label_regions(
    windows: str = typer.Option(..., "--windows", "-w", help="Windowed features file"),
    metadata: str = typer.Option(..., "--metadata", "-m", help="Metadata with group column"),
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="YAML config"),
    output: str = typer.Option("data/labels", "--output", "-o", help="Output directory"),
) -> None:
    """Derive DMR-style labels by comparing case vs control windows."""
    from epilongai.analysis.region_labeling import run_region_labeling

    run_region_labeling(
        windows_path=windows, metadata_path=metadata, config_path=config, output_dir=output
    )


# ---------------------------------------------------------------------------
# Subcommand: variants
# ---------------------------------------------------------------------------
@app.command()
def variants(
    vcf_dir: str = typer.Option(..., "--vcf-dir", "-v", help="Directory with VCF files"),
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="YAML config"),
    output: str = typer.Option("data/processed", "--output", "-o", help="Output directory"),
) -> None:
    """Parse VCF files and compute per-window variant features."""
    from pathlib import Path

    from epilongai.data.variant_processing import process_vcf_directory
    from epilongai.utils.config import load_config
    from epilongai.utils.logging import setup_logging

    cfg = load_config(config)
    log_cfg = cfg.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("log_file"))

    win_cfg = cfg.get("windowing", {})
    result = process_vcf_directory(
        vcf_dir,
        window_size=win_cfg.get("window_size", 1000),
        stride=win_cfg.get("stride", 1000),
    )

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "variant_windows.parquet"
    result.to_parquet(out_path, index=False)


# ---------------------------------------------------------------------------
# Subcommand: rna-integrate
# ---------------------------------------------------------------------------
@app.command("rna-integrate")
def rna_integrate(
    expression: str = typer.Option(..., "--expression", "-e", help="Gene expression matrix (TSV)"),
    windows: str = typer.Option(..., "--windows", "-w", help="Windowed methylation features"),
    gene_bed: str = typer.Option(None, "--gene-bed", help="Gene annotations BED (chr, start, end, gene_id)"),
    gene_gtf: str = typer.Option(None, "--gene-gtf", help="Gene annotations GTF"),
    config: str = typer.Option("configs/default.yaml", "--config", "-c", help="YAML config"),
    output: str = typer.Option("data/processed", "--output", "-o", help="Output directory"),
    normalize: str = typer.Option("log2_tpm", "--normalize", help="Normalisation method: log2_tpm | zscore | quantile"),
) -> None:
    """Integrate RNA-seq expression data with methylation windows."""
    from pathlib import Path

    import pandas as pd

    from epilongai.data.rna_integration import (
        load_gene_annotations,
        map_expression_to_windows,
        merge_omics_features,
        normalize_expression,
        parse_expression_file,
    )
    from epilongai.utils.config import load_config
    from epilongai.utils.logging import setup_logging

    cfg = load_config(config)
    log_cfg = cfg.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("log_file"))

    win_cfg = cfg.get("windowing", {})

    # Parse expression
    expr = parse_expression_file(expression)
    expr = normalize_expression(expr, method=normalize)

    # Load gene annotations
    genes = load_gene_annotations(gtf_path=gene_gtf, bed_path=gene_bed)

    # Map to windows
    expr_windows = map_expression_to_windows(
        expr, genes,
        window_size=win_cfg.get("window_size", 1000),
        stride=win_cfg.get("stride", 1000),
    )

    # Merge with methylation windows
    wp = Path(windows)
    meth_windows = pd.read_parquet(wp) if wp.suffix == ".parquet" else pd.read_csv(wp, sep="\t")
    merged = merge_omics_features(meth_windows, expression_windows=expr_windows)

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "multiomics_windows.parquet"
    merged.to_parquet(out_path, index=False)


# ---------------------------------------------------------------------------
# Subcommand: benchmark
# ---------------------------------------------------------------------------
@app.command()
def benchmark(
    config: str = typer.Option("configs/train.yaml", "--config", "-c", help="YAML config"),
    output: str = typer.Option("results/benchmark", "--output", "-o", help="Output directory"),
    n_folds: int = typer.Option(5, "--folds", help="Number of CV folds"),
) -> None:
    """Run benchmarking against baseline models."""
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from epilongai.analysis.benchmark import plot_benchmark_comparison, run_benchmark
    from epilongai.data.dataset import METHYLATION_FEATURE_COLS
    from epilongai.utils.config import load_config

    cfg = load_config(config)
    data_cfg = cfg["data"]
    wp = Path(data_cfg["windows_path"])
    windows = pd.read_parquet(wp) if wp.suffix == ".parquet" else pd.read_csv(wp, sep="\t")

    label_col = data_cfg.get("label_column", "group")
    label_map = data_cfg.get("label_map", {"FTB": 0, "PTB": 1})
    if label_col in windows.columns:
        labels = windows[label_col].map(label_map).values.astype(np.int64)
    else:
        meta = pd.read_csv(data_cfg["metadata_path"], sep="\t")
        sl = dict(zip(meta["sample_id"], meta[label_col].map(label_map)))
        labels = windows["sample_id"].map(sl).values.astype(np.int64)

    feat_cols = [c for c in METHYLATION_FEATURE_COLS if c in windows.columns]
    X = windows[feat_cols].fillna(0).values.astype(np.float32)

    results = run_benchmark(X, labels, n_folds=n_folds, output_dir=output)
    plot_benchmark_comparison(results, metric="roc_auc", save_path=f"{output}/benchmark_roc_auc.png")
    plot_benchmark_comparison(results, metric="f1", save_path=f"{output}/benchmark_f1.png")


# ---------------------------------------------------------------------------
# Subcommand: report
# ---------------------------------------------------------------------------
@app.command()
def report(
    predictions: str = typer.Option(..., "--predictions", "-p", help="Window predictions CSV"),
    model_checkpoint: str = typer.Option(..., "--checkpoint", "-k", help="Model checkpoint"),
    output: str = typer.Option("results/reports", "--output", "-o", help="Output directory"),
) -> None:
    """Generate clinical risk reports for predicted samples."""
    import pandas as pd

    from epilongai.analysis.clinical_scoring import compute_risk_scores, generate_batch_reports

    preds = pd.read_csv(predictions)
    scores = compute_risk_scores(preds)
    model_info = {"model_type": "EpiLongAI", "checkpoint": model_checkpoint, "version": "0.1.0"}
    generate_batch_reports(scores, preds, model_info, output)


# ---------------------------------------------------------------------------
# Subcommand: serve
# ---------------------------------------------------------------------------
@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", help="Bind port"),
    config: str = typer.Option("configs/train.yaml", "--config", "-c", help="YAML config"),
    checkpoint: str = typer.Option("checkpoints/best.pt", "--checkpoint", "-k", help="Model checkpoint"),
) -> None:
    """Start the FastAPI prediction server."""
    import os
    os.environ["EPILONGAI_CONFIG"] = config
    os.environ["EPILONGAI_CHECKPOINT"] = checkpoint

    import uvicorn
    uvicorn.run("epilongai.api:app", host=host, port=port, reload=False)


# ---------------------------------------------------------------------------
# Subcommand: register
# ---------------------------------------------------------------------------
@app.command()
def register(
    checkpoint: str = typer.Option(..., "--checkpoint", "-k", help="Model checkpoint path"),
    config: str = typer.Option("configs/train.yaml", "--config", "-c", help="Training config"),
    metrics_file: str = typer.Option(None, "--metrics", help="Metrics JSON file"),
    data_path: str = typer.Option(None, "--data", help="Data file used for training"),
    description: str = typer.Option("", "--description", "-d", help="Model description"),
    registry_dir: str = typer.Option("model_registry", "--registry", help="Registry directory"),
) -> None:
    """Register a trained model in the model registry."""
    import json as _json

    from epilongai.utils.model_registry import ModelRegistry

    reg = ModelRegistry(registry_dir)
    metrics = None
    if metrics_file:
        with open(metrics_file) as f:
            metrics = _json.load(f)
    version = reg.register(
        checkpoint_path=checkpoint,
        config_path=config,
        metrics=metrics,
        data_path=data_path,
        description=description,
    )
    typer.echo(f"Registered model {version}")


# ---------------------------------------------------------------------------
# Subcommand: models
# ---------------------------------------------------------------------------
@app.command()
def models(
    registry_dir: str = typer.Option("model_registry", "--registry", help="Registry directory"),
) -> None:
    """List all registered models."""
    from epilongai.utils.model_registry import ModelRegistry

    reg = ModelRegistry(registry_dir)
    entries = reg.list_models()
    if not entries:
        typer.echo("No models registered.")
        return
    for e in entries:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in e.get("metrics", {}).items())
        typer.echo(f"  {e['version']}  {e['model_type']:<15}  {e['timestamp'][:19]}  {metrics_str}  {e.get('description', '')}")


if __name__ == "__main__":
    app()
