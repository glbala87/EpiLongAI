"""
Phase H — DMR-style region labeling from case/control comparisons.

Compares windowed methylation between two groups (e.g. PTB vs FTB)
and assigns labels:
    - hyper   : delta_beta ≥ +threshold AND p < alpha
    - hypo    : delta_beta ≤ -threshold AND p < alpha
    - unchanged : otherwise

Important caveats (for manuscripts):
    - These labels are *exploratory*. They are not validated DMRs.
    - Multiple-testing correction is not applied by default — users should
      treat results as hypothesis-generating.
    - Small cohort sizes limit statistical power; consider permutation
      testing for validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from epilongai.utils.config import load_config
from epilongai.utils.logging import setup_logging


def compare_windows(
    windows: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str = "group",
    case_label: str = "PTB",
    control_label: str = "FTB",
    feature: str = "mean_beta",
    delta_beta_threshold: float = 0.1,
    pvalue_threshold: float = 0.05,
    test_method: str = "mannwhitneyu",
) -> pd.DataFrame:
    """
    Compare methylation windows between case and control groups.

    Returns a table with one row per window containing:
        chr, window_start, window_end, mean_case, mean_control,
        delta_beta, pvalue, label
    """
    # Merge group info
    if group_column not in windows.columns and "sample_id" in windows.columns:
        meta_map = dict(zip(metadata["sample_id"], metadata[group_column]))
        windows = windows.copy()
        windows["_group"] = windows["sample_id"].map(meta_map)
    elif group_column in windows.columns:
        windows = windows.copy()
        windows["_group"] = windows[group_column]
    else:
        raise ValueError(f"Cannot find group column '{group_column}'")

    case_df = windows[windows["_group"] == case_label]
    ctrl_df = windows[windows["_group"] == control_label]

    logger.info(
        f"Case ({case_label}): {case_df['sample_id'].nunique() if 'sample_id' in case_df.columns else '?'} samples, "
        f"Control ({control_label}): {ctrl_df['sample_id'].nunique() if 'sample_id' in ctrl_df.columns else '?'} samples"
    )

    window_cols = ["chr", "window_start", "window_end"]
    case_agg = case_df.groupby(window_cols)[feature].agg(["mean", "std", list]).rename(
        columns={"mean": "mean_case", "std": "std_case", "list": "values_case"}
    )
    ctrl_agg = ctrl_df.groupby(window_cols)[feature].agg(["mean", "std", list]).rename(
        columns={"mean": "mean_control", "std": "std_control", "list": "values_control"}
    )

    merged = case_agg.join(ctrl_agg, how="outer").reset_index()
    merged["delta_beta"] = merged["mean_case"] - merged["mean_control"]

    # Statistical test
    test_fn = stats.mannwhitneyu if test_method == "mannwhitneyu" else stats.ttest_ind
    pvalues: list[float] = []
    for _, row in merged.iterrows():
        vals_c = row.get("values_case")
        vals_ctrl = row.get("values_control")
        if (
            vals_c is None or vals_ctrl is None
            or not isinstance(vals_c, list) or not isinstance(vals_ctrl, list)
            or len(vals_c) < 2 or len(vals_ctrl) < 2
        ):
            pvalues.append(np.nan)
            continue
        a = np.array([x for x in vals_c if not np.isnan(x)])
        b = np.array([x for x in vals_ctrl if not np.isnan(x)])
        if len(a) < 2 or len(b) < 2:
            pvalues.append(np.nan)
            continue
        try:
            _, p = test_fn(a, b, alternative="two-sided")
            pvalues.append(p)
        except Exception:
            pvalues.append(np.nan)

    merged["pvalue"] = pvalues

    # Assign labels
    def _label(row: pd.Series) -> str:
        db = row["delta_beta"]
        p = row["pvalue"]
        if pd.isna(db) or pd.isna(p):
            return "insufficient_data"
        if p < pvalue_threshold:
            if db >= delta_beta_threshold:
                return "hyper"
            elif db <= -delta_beta_threshold:
                return "hypo"
        return "unchanged"

    merged["label"] = merged.apply(_label, axis=1)

    # Clean up
    result = merged[["chr", "window_start", "window_end", "mean_case", "mean_control",
                      "delta_beta", "pvalue", "label"]].copy()

    counts = result["label"].value_counts()
    logger.info(f"Region labels: {counts.to_dict()}")
    logger.info(
        f"Thresholds used — delta_beta: ±{delta_beta_threshold}, "
        f"p-value: {pvalue_threshold}, test: {test_method}"
    )

    return result


def run_region_labeling(
    windows_path: str,
    metadata_path: str,
    config_path: str,
    output_dir: str,
) -> None:
    """CLI entry-point for region labeling."""
    cfg = load_config(config_path)
    lab_cfg = cfg.get("labeling", {})
    log_cfg = cfg.get("logging", {})
    setup_logging(level=log_cfg.get("level", "INFO"), log_file=log_cfg.get("log_file"))

    windows = pd.read_parquet(windows_path) if windows_path.endswith(".parquet") else pd.read_csv(windows_path, sep="\t")
    metadata = pd.read_csv(metadata_path, sep="\t")

    result = compare_windows(
        windows,
        metadata,
        group_column=lab_cfg.get("group_column", "group"),
        case_label=lab_cfg.get("case_label", "PTB"),
        control_label=lab_cfg.get("control_label", "FTB"),
        delta_beta_threshold=lab_cfg.get("delta_beta_threshold", 0.1),
        pvalue_threshold=lab_cfg.get("pvalue_threshold", 0.05),
        test_method=lab_cfg.get("test_method", "mannwhitneyu"),
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / "region_labels.tsv"
    result.to_csv(out_path, sep="\t", index=False)
    logger.info(f"Saved region labels to {out_path}")
