"""
Phase Q — Rigorous benchmarking framework for genomic AI models.

HOW TO INTERPRET RESULTS
=========================
- **ROC-AUC**: Measures discrimination — can the model separate PTB from FTB?
  ≥0.80 is clinically interesting; ≥0.90 is strong.
- **PR-AUC**: More informative than ROC-AUC when classes are imbalanced (common
  in preterm birth where PTB may be 10–15% of samples).
- **F1-score**: Harmonic mean of precision and recall.  Report macro-F1 for
  fairness across classes.
- **Calibration**: Does the model's predicted probability match observed
  frequency?  Critical for clinical use — a well-calibrated model saying
  "70% risk" should be correct 70% of the time.

WHAT REVIEWERS EXPECT IN BENCHMARKING (Nature, Genome Biology, etc.)
=====================================================================
1. **Same data splits** across all models (not re-split per model).
2. **Cross-validation** (5-fold or 10-fold) with confidence intervals.
3. **Simple baselines** (logistic regression, random forest) to show
   that the DL model adds value beyond feature engineering.
4. **State-of-the-art comparisons** (Enformer, Nucleotide Transformer)
   to position your model in the field.
5. **Ablation studies**: Which modality/component drives performance?
6. **Statistical significance**: Paired test (DeLong for AUC, McNemar
   for accuracy) between your model and the best baseline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from epilongai.training.metrics import compute_classification_metrics


# =====================================================================
# Cross-Validation
# =====================================================================
def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Any,
    n_folds: int = 5,
    random_seed: int = 42,
) -> dict[str, list[float]]:
    """
    Run stratified k-fold cross-validation.

    Parameters
    ----------
    X : (n_samples, n_features) array
    y : (n_samples,) label array
    model_factory : callable
        Returns a fresh sklearn-compatible model with fit/predict/predict_proba.
    n_folds : int
    random_seed : int

    Returns
    -------
    dict mapping metric_name → list of per-fold values
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    all_metrics: dict[str, list[float]] = {}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model = model_factory()
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None

        n_classes = len(np.unique(y))
        metrics = compute_classification_metrics(y_te, y_pred, y_prob, n_classes)

        for k, v in metrics.items():
            all_metrics.setdefault(k, []).append(v)

        logger.info(f"  Fold {fold + 1}/{n_folds}: {metrics}")

    return all_metrics


# =====================================================================
# Baseline Models
# =====================================================================
def get_baseline_models() -> dict[str, Any]:
    """Return a dict of baseline model factories."""
    return {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=10, class_weight="balanced", random_state=42
        ),
        "gradient_boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=5, random_state=42
        ),
    }


# =====================================================================
# Run Full Benchmark
# =====================================================================
def run_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    model_names: list[str] | None = None,
    n_folds: int = 5,
    random_seed: int = 42,
    output_dir: str | Path = "results/benchmark",
) -> pd.DataFrame:
    """
    Run cross-validated benchmarking across multiple baselines.

    Returns a summary DataFrame and saves results to output_dir.
    """
    baselines = get_baseline_models()
    if model_names:
        baselines = {k: v for k, v in baselines.items() if k in model_names}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for name, factory in baselines.items():
        logger.info(f"Benchmarking: {name}")
        fold_metrics = cross_validate(X, y, factory, n_folds, random_seed)

        for metric, values in fold_metrics.items():
            rows.append({
                "model": name,
                "metric": metric,
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values,
            })

    results = pd.DataFrame(rows)

    # Save
    results.to_csv(out / "benchmark_results.csv", index=False)
    logger.info(f"Benchmark results saved to {out / 'benchmark_results.csv'}")

    # Summary table
    summary = results.pivot_table(
        index="model", columns="metric", values="mean"
    ).round(4)
    summary.to_csv(out / "benchmark_summary.csv")
    logger.info(f"Summary table:\n{summary.to_string()}")

    return results


# =====================================================================
# DeLong Test for AUC Comparison
# =====================================================================
def delong_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
) -> dict[str, float]:
    """
    Approximate DeLong test for comparing two ROC-AUCs.

    Uses the fast O(n log n) algorithm.

    Returns dict with auc_a, auc_b, z_statistic, p_value.
    """
    from scipy.stats import norm

    auc_a = roc_auc_score(y_true, y_prob_a)
    auc_b = roc_auc_score(y_true, y_prob_b)

    # Structural components
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1

    pos_a = y_prob_a[y_true == 1]
    neg_a = y_prob_a[y_true == 0]
    pos_b = y_prob_b[y_true == 1]
    neg_b = y_prob_b[y_true == 0]

    # Placement values
    v10_a = np.array([np.mean(pos_a > n) + 0.5 * np.mean(pos_a == n) for n in neg_a])
    v01_a = np.array([np.mean(neg_a < p) + 0.5 * np.mean(neg_a == p) for p in pos_a])
    v10_b = np.array([np.mean(pos_b > n) + 0.5 * np.mean(pos_b == n) for n in neg_b])
    v01_b = np.array([np.mean(neg_b < p) + 0.5 * np.mean(neg_b == p) for p in pos_b])

    # Variance of AUC difference
    s10 = np.cov(v10_a, v10_b)[0, 1] if len(v10_a) > 1 else 0
    s01 = np.cov(v01_a, v01_b)[0, 1] if len(v01_a) > 1 else 0
    var_a = np.var(v10_a) / n0 + np.var(v01_a) / n1
    var_b = np.var(v10_b) / n0 + np.var(v01_b) / n1
    covar = s10 / n0 + s01 / n1
    var_diff = var_a + var_b - 2 * covar

    z = (auc_a - auc_b) / max(np.sqrt(var_diff), 1e-10)
    p = 2 * norm.sf(abs(z))

    return {"auc_a": auc_a, "auc_b": auc_b, "z_statistic": z, "p_value": p}


# =====================================================================
# Calibration Curve
# =====================================================================
def compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute calibration curve data."""
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    return {
        "fraction_positive": fraction_positive,
        "mean_predicted": mean_predicted,
    }


# =====================================================================
# Benchmark Plotting
# =====================================================================
def plot_benchmark_comparison(
    results: pd.DataFrame,
    metric: str = "roc_auc",
    save_path: str | Path | None = None,
) -> None:
    """Bar chart comparing models on a single metric with error bars."""
    import matplotlib.pyplot as plt

    sub = results[results["metric"] == metric].copy()
    if sub.empty:
        logger.warning(f"No results for metric '{metric}'")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    models = sub["model"].values
    means = sub["mean"].values
    stds = sub["std"].values

    bars = ax.barh(models, means, xerr=stds, color="#4C72B0", edgecolor="white", capsize=4)
    ax.set_xlabel(metric.replace("_", " ").upper())
    ax.set_title(f"Model Comparison — {metric.upper()}")
    ax.set_xlim(0, 1.05)

    for bar, m, s in zip(bars, means, stds):
        ax.text(m + s + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{m:.3f}±{s:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved benchmark plot to {save_path}")
    plt.close(fig)


def plot_calibration_curve(
    calibrations: dict[str, dict[str, np.ndarray]],
    save_path: str | Path | None = None,
) -> None:
    """Plot calibration curves for multiple models."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")

    for name, data in calibrations.items():
        ax.plot(data["mean_predicted"], data["fraction_positive"],
                "o-", label=name, markersize=4)

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
        logger.info(f"Saved calibration plot to {save_path}")
    plt.close(fig)
