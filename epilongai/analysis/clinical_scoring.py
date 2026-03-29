"""
Phase S — Clinical scoring system for ONT methylation models.

Converts model predictions into clinically interpretable risk reports.

LIMITATIONS OF MODEL PREDICTIONS (must be communicated to clinicians)
=====================================================================
1. **Not a diagnostic test**: This model identifies statistical associations
   between methylation patterns and preterm birth.  It does NOT establish
   causation or replace clinical judgement.

2. **Population sensitivity**: Model performance may vary by population,
   gestational age at sampling, and sample type (cord vs maternal blood).
   Always report the population the model was trained on.

3. **Calibration matters**: A "70% risk" prediction is only meaningful if
   the model is well-calibrated.  Recalibrate on local cohorts before
   clinical deployment.

4. **Research use only**: Until validated in prospective clinical trials,
   outputs should be treated as research findings, not clinical recommendations.

5. **Batch effects**: Different ONT library preps, flow cells, and basecallers
   can shift methylation distributions.  The model expects data processed
   with the same pipeline as training data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


# =====================================================================
# Risk Score Computation
# =====================================================================
def compute_risk_scores(
    window_predictions: pd.DataFrame,
    prob_column: str = "prob_positive",
    aggregation: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate window-level predictions to sample-level risk scores.

    Parameters
    ----------
    window_predictions : DataFrame
        Must have sample_id and a probability column.
    prob_column : str
        Column with predicted probabilities.
    aggregation : str
        'mean', 'median', or 'weighted' (weighted by coverage if available).

    Returns
    -------
    DataFrame with sample_id, risk_score (0–1), risk_category, confidence.
    """
    if "sample_id" not in window_predictions.columns:
        raise ValueError("window_predictions must have 'sample_id' column")

    if aggregation == "weighted" and "mean_coverage" in window_predictions.columns:
        def _weighted_mean(group: pd.DataFrame) -> float:
            w = group["mean_coverage"].fillna(1)
            return float(np.average(group[prob_column], weights=w))
        scores = window_predictions.groupby("sample_id").apply(
            _weighted_mean, include_groups=False
        ).reset_index(name="risk_score")
    elif aggregation == "median":
        scores = window_predictions.groupby("sample_id")[prob_column].median().reset_index(name="risk_score")
    else:
        scores = window_predictions.groupby("sample_id")[prob_column].mean().reset_index(name="risk_score")

    # Additional stats
    stats = window_predictions.groupby("sample_id").agg(
        n_windows=(prob_column, "count"),
        score_std=(prob_column, "std"),
        frac_high_risk=(prob_column, lambda x: (x >= 0.5).mean()),
    ).reset_index()

    scores = scores.merge(stats, on="sample_id")

    # Risk category
    scores["risk_category"] = pd.cut(
        scores["risk_score"],
        bins=[0, 0.3, 0.7, 1.0],
        labels=["low", "intermediate", "high"],
    )

    # Confidence (inverse of score variability)
    scores["confidence"] = 1.0 - scores["score_std"].fillna(0).clip(0, 1)

    logger.info(f"Computed risk scores for {len(scores)} samples")
    return scores


# =====================================================================
# Top Contributing Regions
# =====================================================================
def identify_top_regions(
    window_predictions: pd.DataFrame,
    sample_id: str,
    top_k: int = 10,
    prob_column: str = "prob_positive",
) -> pd.DataFrame:
    """
    For a specific sample, find the top-k windows contributing to risk.
    """
    sample = window_predictions[window_predictions["sample_id"] == sample_id].copy()
    if sample.empty:
        logger.warning(f"No windows found for sample {sample_id}")
        return pd.DataFrame()

    top = sample.nlargest(top_k, prob_column)
    cols = [c for c in ["chr", "window_start", "window_end", prob_column,
                         "mean_beta", "n_cpgs"] if c in top.columns]
    return top[cols].reset_index(drop=True)


# =====================================================================
# Clinical Report Generation
# =====================================================================
def generate_clinical_report(
    sample_id: str,
    risk_score: float,
    risk_category: str,
    confidence: float,
    top_regions: pd.DataFrame,
    model_info: dict[str, Any],
    output_path: str | Path | None = None,
) -> str:
    """
    Generate a structured clinical report as text (CSV-exportable).

    Parameters
    ----------
    sample_id : str
    risk_score : float (0–1)
    risk_category : str
    confidence : float (0–1)
    top_regions : DataFrame
    model_info : dict with model_type, training_date, version, etc.
    output_path : path or None

    Returns
    -------
    Report as a formatted string.
    """
    report_lines = [
        "=" * 70,
        "EpiLongAI — Methylation Risk Assessment Report",
        "=" * 70,
        "",
        "DISCLAIMER: This report is for RESEARCH USE ONLY.",
        "It is NOT a diagnostic test and should NOT be used for clinical",
        "decision-making without independent validation.",
        "",
        "-" * 70,
        "SAMPLE INFORMATION",
        "-" * 70,
        f"  Sample ID:       {sample_id}",
        f"  Report Date:     {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "-" * 70,
        "RISK ASSESSMENT",
        "-" * 70,
        f"  Risk Score:      {risk_score:.4f} (scale: 0 = low risk, 1 = high risk)",
        f"  Risk Category:   {risk_category.upper()}",
        f"  Confidence:      {confidence:.2f}",
        "",
    ]

    # Risk interpretation
    if risk_category == "high":
        report_lines.append("  Interpretation:  Methylation profile is consistent with elevated")
        report_lines.append("                   preterm birth risk based on model predictions.")
    elif risk_category == "intermediate":
        report_lines.append("  Interpretation:  Methylation profile shows intermediate risk signal.")
        report_lines.append("                   Further investigation may be warranted.")
    else:
        report_lines.append("  Interpretation:  Methylation profile is consistent with low")
        report_lines.append("                   preterm birth risk based on model predictions.")

    report_lines.extend([
        "",
        "-" * 70,
        "TOP CONTRIBUTING GENOMIC REGIONS",
        "-" * 70,
    ])

    if not top_regions.empty:
        for i, (_, row) in enumerate(top_regions.iterrows(), 1):
            chrom = row.get("chr", "?")
            ws = row.get("window_start", "?")
            we = row.get("window_end", "?")
            prob = row.get("prob_positive", row.get("score", None))
            beta = row.get("mean_beta", None)
            try:
                prob_str = f"score={float(prob):.4f}" if prob is not None else ""
                beta_str = f"mean_beta={float(beta):.3f}" if beta is not None else ""
                report_lines.append(f"  {i:>2}. {chrom}:{ws}-{we}  {prob_str}  {beta_str}".rstrip())
            except (TypeError, ValueError):
                report_lines.append(f"  {i:>2}. {chrom}:{ws}-{we}")
    else:
        report_lines.append("  No regions available.")

    report_lines.extend([
        "",
        "-" * 70,
        "MODEL INFORMATION",
        "-" * 70,
        f"  Model Type:      {model_info.get('model_type', 'unknown')}",
        f"  Model Version:   {model_info.get('version', 'unknown')}",
        f"  Training Date:   {model_info.get('training_date', 'unknown')}",
        f"  Checkpoint:      {model_info.get('checkpoint', 'unknown')}",
        "",
        "=" * 70,
    ])

    report = "\n".join(report_lines)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Saved clinical report to {output_path}")

    return report


# =====================================================================
# Batch Report Generation
# =====================================================================
def generate_batch_reports(
    risk_scores: pd.DataFrame,
    window_predictions: pd.DataFrame,
    model_info: dict[str, Any],
    output_dir: str | Path,
    top_k: int = 10,
) -> None:
    """
    Generate individual reports for all samples and a summary CSV.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Summary CSV
    risk_scores.to_csv(out / "risk_summary.csv", index=False)
    logger.info(f"Saved risk summary to {out / 'risk_summary.csv'}")

    # Individual reports
    reports_dir = out / "individual_reports"
    reports_dir.mkdir(exist_ok=True)

    for _, row in risk_scores.iterrows():
        sid = row["sample_id"]
        top = identify_top_regions(window_predictions, sid, top_k=top_k)
        generate_clinical_report(
            sample_id=sid,
            risk_score=row["risk_score"],
            risk_category=str(row["risk_category"]),
            confidence=row["confidence"],
            top_regions=top,
            model_info=model_info,
            output_path=reports_dir / f"{sid}_report.txt",
        )

    logger.info(f"Generated {len(risk_scores)} individual reports in {reports_dir}")
