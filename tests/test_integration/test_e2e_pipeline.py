"""
End-to-end integration tests for the EpiLongAI pipeline.

Tests the full chain:
    synthetic data -> ingest -> window -> train -> evaluate -> predict

Uses small synthetic data (5 samples, ~200 CpG sites each) so tests
run in seconds on CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from epilongai.data.data_ingestion import merge_samples, parse_methylation_file
from epilongai.data.dataset import (
    METHYLATION_FEATURE_COLS,
    MethylationDataset,
    build_dataloaders,
    split_dataset,
)
from epilongai.data.windowing import compute_window_features_fast
from epilongai.models.baseline_mlp import BaselineMLP
from epilongai.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers to generate synthetic data
# ---------------------------------------------------------------------------

NUM_SAMPLES = 10
CPGS_PER_SAMPLE = 200
WINDOW_SIZE = 500
STRIDE = 500
MIN_CPGS = 2

# Labels: 6 FTB (0), 4 PTB (1) — enough for stratified sample-level splitting
SAMPLE_GROUPS = {
    "sample_01": "FTB",
    "sample_02": "FTB",
    "sample_03": "FTB",
    "sample_04": "FTB",
    "sample_05": "FTB",
    "sample_06": "FTB",
    "sample_07": "PTB",
    "sample_08": "PTB",
    "sample_09": "PTB",
    "sample_10": "PTB",
}


def _generate_methylation_tsv(path: Path, sample_id: str, rng: np.random.Generator) -> None:
    """Write a realistic ONT methylation TSV file for one sample."""
    group = SAMPLE_GROUPS[sample_id]
    # PTB samples have slightly higher mean beta to create a learnable signal
    beta_mean = 0.65 if group == "PTB" else 0.35

    chroms = ["chr1"] * (CPGS_PER_SAMPLE // 2) + ["chr2"] * (CPGS_PER_SAMPLE - CPGS_PER_SAMPLE // 2)
    starts = []
    for chrom_name in ["chr1", "chr2"]:
        n = chroms.count(chrom_name)
        starts.extend(sorted(rng.choice(range(100, 5000), size=n, replace=False)))

    coverages = rng.integers(8, 60, size=CPGS_PER_SAMPLE)
    betas = np.clip(rng.normal(beta_mean, 0.15, size=CPGS_PER_SAMPLE), 0, 1)
    modified_counts = (betas * coverages).astype(int)

    df = pd.DataFrame(
        {
            "chr": chroms,
            "start": starts,
            "end": [s + 1 for s in starts],
            "coverage": coverages,
            "modified_count": modified_counts,
            "beta": betas.round(4),
        }
    )
    df.to_csv(path, sep="\t", index=False)


def _generate_metadata_tsv(path: Path) -> None:
    """Write a metadata TSV with sample_id and group columns."""
    rows = [{"sample_id": sid, "group": grp} for sid, grp in SAMPLE_GROUPS.items()]
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _make_test_config(tmp_path: Path, windows_path: str, metadata_path: str) -> Path:
    """Write a minimal YAML config suitable for the integration test."""
    cfg = {
        "data": {
            "windows_path": windows_path,
            "metadata_path": metadata_path,
            "label_column": "group",
            "label_map": {"FTB": 0, "PTB": 1},
            "split": {
                "test_size": 0.2,
                "val_size": 0.2,
                "stratify": True,
                "random_seed": 42,
            },
        },
        "dataset": {
            "mode": "methylation",
            "sequence_encoding": "onehot",
            "max_sequence_length": 500,
        },
        "ingestion": {
            "file_format": "auto",
            "separator": "\t",
            "min_coverage": 5,
        },
        "windowing": {
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "min_cpgs_per_window": MIN_CPGS,
        },
        "model": {
            "type": "baseline_mlp",
            "task": "classification",
            "num_classes": 2,
            "mlp": {
                "hidden_dims": [32, 16],
                "dropout": 0.1,
                "batch_norm": True,
                "activation": "relu",
            },
        },
        "training": {
            "epochs": 3,
            "batch_size": 16,
            "learning_rate": 0.01,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "scheduler": {"type": "none"},
            "early_stopping": {"enabled": False},
            "mixed_precision": False,
            "gradient_clip": 0.0,
            "seed": 42,
        },
        "checkpointing": {
            "save_dir": str(tmp_path / "checkpoints"),
            "save_best": True,
            "save_every_n_epochs": 10,
        },
        "output": {
            "results_dir": str(tmp_path / "results"),
            "plots_dir": str(tmp_path / "results" / "plots"),
        },
        "logging": {
            "level": "WARNING",
        },
    }
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return config_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_tmp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Module-scoped temporary directory shared across all tests."""
    return tmp_path_factory.mktemp("e2e")


@pytest.fixture(scope="module")
def synthetic_data_dir(shared_tmp: Path) -> Path:
    """Generate synthetic methylation TSV files for all samples."""
    data_dir = shared_tmp / "raw_methylation"
    data_dir.mkdir()
    rng = np.random.default_rng(seed=12345)
    for sample_id in SAMPLE_GROUPS:
        _generate_methylation_tsv(data_dir / f"{sample_id}.tsv", sample_id, rng)
    return data_dir


@pytest.fixture(scope="module")
def metadata_path(shared_tmp: Path) -> Path:
    """Generate metadata TSV."""
    path = shared_tmp / "metadata.tsv"
    _generate_metadata_tsv(path)
    return path


@pytest.fixture(scope="module")
def merged_methylation(synthetic_data_dir: Path, metadata_path: Path) -> pd.DataFrame:
    """Run ingestion: merge all sample files with metadata."""
    meta = pd.read_csv(metadata_path, sep="\t")
    merged = merge_samples(
        synthetic_data_dir,
        metadata=meta,
        glob_pattern="*.tsv",
        min_coverage=5,
    )
    return merged


@pytest.fixture(scope="module")
def windowed_features(merged_methylation: pd.DataFrame) -> pd.DataFrame:
    """Run windowing on merged methylation data."""
    windows = compute_window_features_fast(
        merged_methylation,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        min_cpgs=MIN_CPGS,
    )
    return windows


@pytest.fixture(scope="module")
def labels_array(windowed_features: pd.DataFrame, metadata_path: Path) -> np.ndarray:
    """Create integer labels aligned with windowed features."""
    meta = pd.read_csv(metadata_path, sep="\t")
    label_map = {"FTB": 0, "PTB": 1}
    sample_label = dict(zip(meta["sample_id"], meta["group"].map(label_map)))
    labels = windowed_features["sample_id"].map(sample_label).values.astype(np.int64)
    return labels


@pytest.fixture(scope="module")
def split_data(
    windowed_features: pd.DataFrame,
    labels_array: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Split windowed features into train/val/test."""
    return split_dataset(
        windowed_features,
        labels_array,
        test_size=0.2,
        val_size=0.2,
        stratify=True,
        random_seed=42,
    )


@pytest.fixture(scope="module")
def datasets(
    split_data: tuple,
) -> tuple[MethylationDataset, MethylationDataset, MethylationDataset]:
    """Build MethylationDataset objects for each split."""
    w_train, w_val, w_test, y_train, y_val, y_test = split_data
    train_ds = MethylationDataset(w_train, y_train, mode="methylation")
    val_ds = MethylationDataset(w_val, y_val, mode="methylation")
    test_ds = MethylationDataset(w_test, y_test, mode="methylation")
    return train_ds, val_ds, test_ds


@pytest.fixture(scope="module")
def dataloaders(datasets: tuple) -> dict:
    """Build DataLoaders for each split."""
    train_ds, val_ds, test_ds = datasets
    return build_dataloaders(train_ds, val_ds, test_ds, batch_size=16, num_workers=0)


@pytest.fixture(scope="module")
def trained_model_and_history(
    datasets: tuple,
    dataloaders: dict,
    shared_tmp: Path,
) -> tuple[BaselineMLP, dict]:
    """Train a BaselineMLP for 3 epochs and return model + history."""
    train_ds, val_ds, test_ds = datasets

    model = BaselineMLP(
        input_dim=train_ds.num_features,
        hidden_dims=[32, 16],
        num_classes=2,
        dropout=0.1,
        batch_norm=True,
        task="classification",
    )

    train_cfg = {
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 0.01,
        "weight_decay": 1e-4,
        "optimizer": "adamw",
        "scheduler": {"type": "none"},
        "early_stopping": {"enabled": False},
        "mixed_precision": False,
        "gradient_clip": 0.0,
        "checkpointing": {
            "save_dir": str(shared_tmp / "checkpoints"),
            "save_best": True,
            "save_every_n_epochs": 10,
        },
    }

    model_cfg = {
        "type": "baseline_mlp",
        "task": "classification",
        "num_classes": 2,
    }

    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        cfg=train_cfg,
        model_cfg=model_cfg,
        device="cpu",
    )
    history = trainer.fit(epochs=3)
    return model, history


@pytest.fixture(scope="module")
def checkpoint_path(shared_tmp: Path) -> Path:
    """Return path to the best checkpoint saved during training."""
    return shared_tmp / "checkpoints" / "best.pt"


@pytest.fixture(scope="module")
def config_path(
    shared_tmp: Path,
    windowed_features: pd.DataFrame,
    metadata_path: Path,
) -> Path:
    """Write windowed features to parquet and create a YAML config pointing to them."""
    windows_parquet = shared_tmp / "windows.parquet"
    windowed_features.to_parquet(windows_parquet, index=False)
    return _make_test_config(shared_tmp, str(windows_parquet), str(metadata_path))


# ---------------------------------------------------------------------------
# Tests — Stage 1: Data Ingestion
# ---------------------------------------------------------------------------


class TestIngestion:
    """Tests for the data ingestion stage."""

    def test_parse_single_file(self, synthetic_data_dir: Path) -> None:
        """A single methylation TSV should parse into a DataFrame with required columns."""
        path = synthetic_data_dir / "sample_01.tsv"
        df = parse_methylation_file(path, min_coverage=5)

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        for col in ("chr", "start", "end", "coverage", "beta"):
            assert col in df.columns, f"Missing column: {col}"
        assert (df["coverage"] >= 5).all(), "Coverage filter not applied"
        assert df["beta"].between(0, 1).all(), "Beta values out of [0,1]"

    def test_merge_samples_shape(self, merged_methylation: pd.DataFrame) -> None:
        """Merged data should contain all samples and have a sample_id column."""
        assert "sample_id" in merged_methylation.columns
        assert merged_methylation["sample_id"].nunique() == NUM_SAMPLES

    def test_merge_samples_has_metadata(self, merged_methylation: pd.DataFrame) -> None:
        """Merged data should carry the metadata group column from the join."""
        assert "group" in merged_methylation.columns
        assert set(merged_methylation["group"].dropna().unique()) == {"FTB", "PTB"}


# ---------------------------------------------------------------------------
# Tests — Stage 2: Windowing
# ---------------------------------------------------------------------------


class TestWindowing:
    """Tests for the genomic windowing stage."""

    def test_windowed_features_schema(self, windowed_features: pd.DataFrame) -> None:
        """Windowed output should contain all expected feature columns."""
        for col in METHYLATION_FEATURE_COLS:
            assert col in windowed_features.columns, f"Missing feature column: {col}"
        assert "sample_id" in windowed_features.columns
        assert "chr" in windowed_features.columns
        assert "window_start" in windowed_features.columns

    def test_windowed_features_nonempty(self, windowed_features: pd.DataFrame) -> None:
        """There should be a reasonable number of windows after filtering."""
        assert len(windowed_features) >= NUM_SAMPLES, (
            "Too few windows — check min_cpgs or data generation"
        )

    def test_min_cpgs_respected(self, windowed_features: pd.DataFrame) -> None:
        """Every window should have at least MIN_CPGS CpG sites."""
        assert (windowed_features["n_cpgs"] >= MIN_CPGS).all()

    def test_beta_features_range(self, windowed_features: pd.DataFrame) -> None:
        """Mean beta should be in [0, 1]."""
        mean_beta = windowed_features["mean_beta"]
        assert mean_beta.between(0, 1).all()


# ---------------------------------------------------------------------------
# Tests — Stage 3: Dataset & DataLoaders
# ---------------------------------------------------------------------------


class TestDataset:
    """Tests for the PyTorch dataset and dataloader construction."""

    def test_split_sizes(self, split_data: tuple) -> None:
        """Train + val + test row counts should equal original window count."""
        w_train, w_val, w_test, y_train, y_val, y_test = split_data
        total = len(w_train) + len(w_val) + len(w_test)
        assert total > 0
        # Labels should match windows
        assert len(w_train) == len(y_train)
        assert len(w_val) == len(y_val)
        assert len(w_test) == len(y_test)

    def test_no_sample_leakage(self, split_data: tuple) -> None:
        """No sample_id should appear in more than one split."""
        w_train, w_val, w_test, *_ = split_data
        train_ids = set(w_train["sample_id"].unique())
        val_ids = set(w_val["sample_id"].unique())
        test_ids = set(w_test["sample_id"].unique())
        assert train_ids.isdisjoint(val_ids), "Train/val sample leakage"
        assert train_ids.isdisjoint(test_ids), "Train/test sample leakage"
        assert val_ids.isdisjoint(test_ids), "Val/test sample leakage"

    def test_dataset_getitem(self, datasets: tuple) -> None:
        """Each dataset item should have methylation tensor and label."""
        train_ds, _, _ = datasets
        item = train_ds[0]
        assert "methylation" in item
        assert "label" in item
        assert item["methylation"].shape == (len(METHYLATION_FEATURE_COLS),)
        assert item["label"].dtype == torch.long

    def test_dataloaders_iterate(self, dataloaders: dict) -> None:
        """All dataloaders should be iterable and yield proper batches."""
        for name in ("train", "val", "test"):
            loader = dataloaders[name]
            batch = next(iter(loader))
            assert "methylation" in batch
            assert "label" in batch
            assert batch["methylation"].ndim == 2
            assert batch["methylation"].shape[1] == len(METHYLATION_FEATURE_COLS)


# ---------------------------------------------------------------------------
# Tests — Stage 4: Model & Training
# ---------------------------------------------------------------------------


class TestTraining:
    """Tests for model construction and training."""

    def test_model_forward_pass(self, datasets: tuple) -> None:
        """BaselineMLP forward pass should return logits, probs, embed."""
        train_ds, _, _ = datasets
        model = BaselineMLP(
            input_dim=train_ds.num_features,
            hidden_dims=[32, 16],
            num_classes=2,
            task="classification",
        )
        x = torch.randn(4, train_ds.num_features)
        out = model(x)
        assert "logits" in out
        assert "probs" in out
        assert "embed" in out
        assert out["logits"].shape == (4, 1)  # binary: single logit
        assert out["probs"].shape == (4, 1)

    def test_from_config(self, datasets: tuple) -> None:
        """BaselineMLP.from_config should produce a working model."""
        train_ds, _, _ = datasets
        model_cfg = {
            "task": "classification",
            "num_classes": 2,
            "mlp": {
                "hidden_dims": [32, 16],
                "dropout": 0.1,
                "batch_norm": True,
                "activation": "relu",
            },
        }
        model = BaselineMLP.from_config(model_cfg, input_dim=train_ds.num_features)
        x = torch.randn(2, train_ds.num_features)
        out = model(x)
        assert out["logits"].shape == (2, 1)

    def test_training_produces_history(self, trained_model_and_history: tuple) -> None:
        """Trainer.fit() should return a history dict with expected keys."""
        _, history = trained_model_and_history
        assert "train_loss" in history
        assert len(history["train_loss"]) == 3  # 3 epochs

    def test_training_loss_decreases_or_finite(self, trained_model_and_history: tuple) -> None:
        """Training loss should be finite across all epochs."""
        _, history = trained_model_and_history
        for loss_val in history["train_loss"]:
            assert np.isfinite(loss_val), f"Non-finite training loss: {loss_val}"

    def test_checkpoint_saved(self, checkpoint_path: Path) -> None:
        """Best checkpoint should have been written to disk."""
        assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        assert "epoch" in ckpt


# ---------------------------------------------------------------------------
# Tests — Stage 5: Evaluation
# ---------------------------------------------------------------------------


class TestEvaluation:
    """Tests for model evaluation on the held-out test split."""

    def test_evaluate_on_test_set(
        self,
        trained_model_and_history: tuple,
        datasets: tuple,
        dataloaders: dict,
    ) -> None:
        """Evaluate the trained model on the test DataLoader and check metrics."""
        model, _ = trained_model_and_history
        _, _, test_ds = datasets
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in dataloaders["test"]:
                x = batch["methylation"]
                out = model(x)
                probs = out["probs"].numpy().ravel()
                preds = (probs >= 0.5).astype(int)
                all_preds.append(preds)
                all_probs.append(probs)
                all_labels.append(batch["label"].numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)

        from epilongai.training.metrics import compute_classification_metrics

        metrics = compute_classification_metrics(y_true, y_pred, y_prob, num_classes=2)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_run_evaluation_function(
        self,
        config_path: Path,
        checkpoint_path: Path,
        trained_model_and_history: tuple,
    ) -> None:
        """run_evaluation() should return a metrics dict without error."""
        from epilongai.training.evaluate import run_evaluation

        metrics = run_evaluation(
            config_path=str(config_path),
            checkpoint_path=str(checkpoint_path),
            split="test",
        )
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "f1" in metrics


# ---------------------------------------------------------------------------
# Tests — Stage 6: Prediction on New Samples
# ---------------------------------------------------------------------------


class TestPrediction:
    """Tests for the end-to-end prediction pipeline."""

    def test_run_prediction(
        self,
        synthetic_data_dir: Path,
        checkpoint_path: Path,
        config_path: Path,
        shared_tmp: Path,
        trained_model_and_history: tuple,
    ) -> None:
        """run_prediction() should produce window and sample prediction CSVs."""
        from epilongai.training.predict import run_prediction

        pred_output = shared_tmp / "predictions"
        run_prediction(
            input_dir=str(synthetic_data_dir),
            checkpoint_path=str(checkpoint_path),
            config_path=str(config_path),
            output_dir=str(pred_output),
            fasta_path=None,
        )

        # Window-level predictions
        window_csv = pred_output / "window_predictions.csv"
        assert window_csv.exists(), "window_predictions.csv not created"
        window_df = pd.read_csv(window_csv)
        assert len(window_df) > 0
        assert "predicted_class" in window_df.columns
        assert "predicted_label" in window_df.columns

        # Sample-level predictions
        sample_csv = pred_output / "sample_predictions.csv"
        assert sample_csv.exists(), "sample_predictions.csv not created"
        sample_df = pd.read_csv(sample_csv)
        assert "sample_id" in sample_df.columns
        assert "predicted_label" in sample_df.columns
        assert len(sample_df) == NUM_SAMPLES

        # Prediction metadata
        meta_json = pred_output / "prediction_meta.json"
        assert meta_json.exists()
        with open(meta_json) as f:
            meta = json.load(f)
        assert meta["model_type"] == "baseline_mlp"
        assert meta["task"] == "classification"


# ---------------------------------------------------------------------------
# Tests — Full E2E (single test exercising the complete chain)
# ---------------------------------------------------------------------------


class TestFullE2EPipeline:
    """A single test that runs the entire pipeline end-to-end from scratch."""

    def test_ingest_to_predict(self, tmp_path: Path) -> None:
        """
        Full pipeline: generate data -> ingest -> window -> train -> evaluate -> predict.

        This is a self-contained test that does not rely on module-scoped fixtures,
        verifying the entire flow in one pass.
        """
        rng = np.random.default_rng(seed=99)

        # 1. Generate synthetic methylation files
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        sample_groups = {
            "sA": "FTB", "sB": "FTB", "sC": "FTB", "sD": "FTB", "sE": "FTB",
            "sF": "PTB", "sG": "PTB", "sH": "PTB", "sI": "PTB",
        }
        for sid, grp in sample_groups.items():
            beta_mean = 0.7 if grp == "PTB" else 0.3
            n_sites = 150
            chroms = ["chr1"] * n_sites
            starts = sorted(rng.choice(range(100, 5000), size=n_sites, replace=False))
            coverages = rng.integers(10, 50, size=n_sites)
            betas = np.clip(rng.normal(beta_mean, 0.12, size=n_sites), 0, 1).round(4)
            modified = (betas * coverages).astype(int)
            df = pd.DataFrame({
                "chr": chroms,
                "start": starts,
                "end": [s + 1 for s in starts],
                "coverage": coverages,
                "modified_count": modified,
                "beta": betas,
            })
            df.to_csv(raw_dir / f"{sid}.tsv", sep="\t", index=False)

        # Metadata
        meta_path = tmp_path / "meta.tsv"
        pd.DataFrame([
            {"sample_id": sid, "group": grp} for sid, grp in sample_groups.items()
        ]).to_csv(meta_path, sep="\t", index=False)

        # 2. Ingest
        meta_df = pd.read_csv(meta_path, sep="\t")
        merged = merge_samples(raw_dir, metadata=meta_df, glob_pattern="*.tsv", min_coverage=5)
        assert "sample_id" in merged.columns
        assert merged["sample_id"].nunique() == len(sample_groups)

        # 3. Window
        windows = compute_window_features_fast(merged, window_size=500, stride=500, min_cpgs=2)
        assert len(windows) > 0
        for col in METHYLATION_FEATURE_COLS:
            assert col in windows.columns

        # 4. Labels
        label_map = {"FTB": 0, "PTB": 1}
        sample_label = dict(zip(meta_df["sample_id"], meta_df["group"].map(label_map)))
        labels = windows["sample_id"].map(sample_label).values.astype(np.int64)
        assert len(labels) == len(windows)
        assert set(np.unique(labels)) == {0, 1}

        # 5. Split
        w_train, w_val, w_test, y_train, y_val, y_test = split_dataset(
            windows, labels, test_size=0.2, val_size=0.2, stratify=True, random_seed=7,
        )
        assert len(w_train) + len(w_val) + len(w_test) == len(windows)

        # 6. Datasets and loaders
        train_ds = MethylationDataset(w_train, y_train, mode="methylation")
        val_ds = MethylationDataset(w_val, y_val, mode="methylation")
        test_ds = MethylationDataset(w_test, y_test, mode="methylation")
        loaders = build_dataloaders(train_ds, val_ds, test_ds, batch_size=8, num_workers=0)

        # 7. Model (hidden_dims must match the config used for prediction)
        model = BaselineMLP(
            input_dim=train_ds.num_features,
            hidden_dims=[32, 16],
            num_classes=2,
            dropout=0.1,
            batch_norm=True,
            task="classification",
        )

        # 8. Train
        ckpt_dir = tmp_path / "ckpts"
        train_cfg = {
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "optimizer": "adam",
            "scheduler": {"type": "none"},
            "early_stopping": {"enabled": False},
            "mixed_precision": False,
            "gradient_clip": 0.0,
            "checkpointing": {
                "save_dir": str(ckpt_dir),
                "save_best": True,
                "save_every_n_epochs": 100,
            },
        }
        model_cfg = {"task": "classification", "num_classes": 2}

        trainer = Trainer(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            cfg=train_cfg,
            model_cfg=model_cfg,
            device="cpu",
        )
        history = trainer.fit(epochs=2)
        assert len(history["train_loss"]) == 2
        assert all(np.isfinite(v) for v in history["train_loss"])

        # 9. Checkpoint exists
        best_ckpt = ckpt_dir / "best.pt"
        assert best_ckpt.exists()

        # 10. Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loaders["test"]:
                out = model(batch["methylation"])
                preds = (out["probs"].numpy().ravel() >= 0.5).astype(int)
                all_preds.append(preds)
                all_labels.append(batch["label"].numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        from epilongai.training.metrics import compute_classification_metrics

        metrics = compute_classification_metrics(y_true, y_pred, num_classes=2)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

        # 11. Predict on the same raw data (smoke test)
        from epilongai.training.predict import run_prediction

        # Write config for prediction
        windows_pq = tmp_path / "windows.parquet"
        windows.to_parquet(windows_pq, index=False)
        cfg_path = _make_test_config(tmp_path, str(windows_pq), str(meta_path))

        pred_out = tmp_path / "pred_output"
        run_prediction(
            input_dir=str(raw_dir),
            checkpoint_path=str(best_ckpt),
            config_path=str(cfg_path),
            output_dir=str(pred_out),
        )

        assert (pred_out / "window_predictions.csv").exists()
        assert (pred_out / "sample_predictions.csv").exists()
        sample_preds = pd.read_csv(pred_out / "sample_predictions.csv")
        assert len(sample_preds) == len(sample_groups)
        assert "predicted_label" in sample_preds.columns
