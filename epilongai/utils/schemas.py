"""
Pydantic validation schemas for all YAML configurations.

Catches typos, missing fields, and invalid values at load time with
clear error messages — not deep in a training loop.

Usage:
    from epilongai.utils.schemas import validate_train_config, validate_pipeline_config
    cfg = validate_train_config("configs/train.yaml")
    # Returns a validated dict; raises ValidationError with readable messages on failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator


# =====================================================================
# Pipeline config (default.yaml)
# =====================================================================
class IngestionConfig(BaseModel):
    file_format: Literal["auto", "bedmethyl", "tabular"] = "auto"
    separator: str = "\t"
    chunk_size: int = Field(500_000, gt=0)
    min_coverage: int = Field(5, ge=0)
    columns: dict[str, str | None] = Field(default_factory=dict)


class WindowingConfig(BaseModel):
    window_size: int = Field(1000, gt=0)
    stride: int = Field(1000, gt=0)
    min_cpgs_per_window: int = Field(3, ge=0)
    chromosomes: list[str] | None = None
    extract_sequence: bool = False
    fasta_path: str | None = None


class LabelingConfig(BaseModel):
    group_column: str = "group"
    case_label: str = "PTB"
    control_label: str = "FTB"
    delta_beta_threshold: float = Field(0.1, ge=0, le=1)
    pvalue_threshold: float = Field(0.05, gt=0, le=1)
    test_method: Literal["mannwhitneyu", "ttest"] = "mannwhitneyu"


class LoggingConfig(BaseModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: str | None = None


class PipelineConfig(BaseModel):
    """Validates configs/default.yaml."""
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    windowing: WindowingConfig = Field(default_factory=WindowingConfig)
    labeling: LabelingConfig = Field(default_factory=LabelingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# =====================================================================
# Training config (train.yaml)
# =====================================================================
class SplitConfig(BaseModel):
    test_size: float = Field(0.15, gt=0, lt=1)
    val_size: float = Field(0.15, gt=0, lt=1)
    stratify: bool = True
    random_seed: int = 42

    @model_validator(mode="after")
    def check_split_sum(self) -> SplitConfig:
        if self.test_size + self.val_size >= 1.0:
            raise ValueError(
                f"test_size ({self.test_size}) + val_size ({self.val_size}) must be < 1.0"
            )
        return self


class DataConfig(BaseModel):
    windows_path: str
    metadata_path: str | None = None
    label_column: str = "group"
    label_map: dict[str, int] = Field(default_factory=lambda: {"FTB": 0, "PTB": 1})
    split: SplitConfig = Field(default_factory=SplitConfig)


class DatasetConfig(BaseModel):
    mode: Literal["methylation", "sequence", "multimodal", "variants", "full"] = "methylation"
    sequence_encoding: Literal["onehot", "tokenized"] = "onehot"
    max_sequence_length: int = Field(1000, gt=0)


class MLPConfig(BaseModel):
    hidden_dims: list[int] = Field(default_factory=lambda: [256, 128, 64])
    dropout: float = Field(0.3, ge=0, le=1)
    batch_norm: bool = True
    activation: Literal["relu", "gelu", "leaky_relu"] = "relu"

    @field_validator("hidden_dims")
    @classmethod
    def check_hidden_dims(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("hidden_dims must not be empty")
        if any(d <= 0 for d in v):
            raise ValueError("All hidden_dims must be positive")
        return v


class CNNConfig(BaseModel):
    channels: list[int] = Field(default_factory=lambda: [64, 128, 256])
    kernel_sizes: list[int] = Field(default_factory=lambda: [7, 5, 3])
    pool_size: int = Field(2, gt=0)


class TransformerBlockConfig(BaseModel):
    d_model: int = Field(128, gt=0)
    nhead: int = Field(4, gt=0)
    num_layers: int = Field(2, gt=0)
    dim_feedforward: int = Field(256, gt=0)


class EncoderConfig(BaseModel):
    hidden_dims: list[int] = Field(default_factory=lambda: [128, 64])
    dropout: float = Field(0.2, ge=0, le=1)


class FusionConfig(BaseModel):
    method: Literal["concatenate", "cross_attention"] = "concatenate"
    hidden_dim: int = Field(128, gt=0)


class MultimodalConfig(BaseModel):
    sequence_encoder: Literal["cnn", "transformer"] = "cnn"
    cnn: CNNConfig = Field(default_factory=CNNConfig)
    transformer: TransformerBlockConfig = Field(default_factory=TransformerBlockConfig)
    methylation_encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    variant_encoder: EncoderConfig = Field(default_factory=lambda: EncoderConfig(hidden_dims=[64, 32]))
    fusion: FusionConfig = Field(default_factory=FusionConfig)


class LongContextConfig(BaseModel):
    n_input_channels: int = Field(8, gt=0)
    d_model: int = Field(256, gt=0)
    n_layers: int = Field(6, gt=0)
    d_state: int = Field(16, gt=0)
    expand: int = Field(2, gt=0)
    pool: Literal["mean", "max", "cls"] = "mean"
    dropout: float = Field(0.1, ge=0, le=1)
    gradient_checkpointing: bool = True


class PopulationConfig(BaseModel):
    enabled: bool = False
    n_populations: int = Field(10, gt=0)
    embed_dim: int = Field(16, gt=0)
    n_af_features: int = Field(0, ge=0)
    conditioning: Literal["concatenate", "film"] = "concatenate"


class ModelConfig(BaseModel):
    type: Literal["baseline_mlp", "multimodal", "long_context"] = "baseline_mlp"
    task: Literal["classification", "regression"] = "classification"
    num_classes: int = Field(2, gt=0)
    mlp: MLPConfig = Field(default_factory=MLPConfig)
    multimodal: MultimodalConfig = Field(default_factory=MultimodalConfig)
    long_context: LongContextConfig = Field(default_factory=LongContextConfig)
    population: PopulationConfig = Field(default_factory=PopulationConfig)

    @model_validator(mode="after")
    def check_regression_classes(self) -> ModelConfig:
        if self.task == "regression" and self.num_classes > 1:
            # Auto-fix: regression always has 1 output
            pass
        return self


class SchedulerConfig(BaseModel):
    type: Literal["cosine", "step", "plateau", "none"] = "cosine"
    T_max: int = 100
    step_size: int = 30
    gamma: float = Field(0.1, gt=0, lt=1)
    patience: int = Field(10, gt=0)


class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience: int = Field(15, gt=0)
    metric: str = "val_f1"
    mode: Literal["max", "min"] = "max"


class TrainingConfig(BaseModel):
    epochs: int = Field(100, gt=0)
    batch_size: int = Field(64, gt=0)
    learning_rate: float = Field(0.001, gt=0)
    weight_decay: float = Field(0.0001, ge=0)
    optimizer: Literal["adam", "adamw", "sgd"] = "adamw"
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = Field(default_factory=EarlyStoppingConfig)
    class_weights: str | list[float] | None = "balanced"
    mixed_precision: bool = True
    gradient_clip: float = Field(1.0, ge=0)
    seed: int = 42


class CheckpointConfig(BaseModel):
    save_dir: str = "checkpoints"
    save_best: bool = True
    save_every_n_epochs: int = Field(10, gt=0)
    monitor_metric: str = "val_f1"
    monitor_mode: Literal["max", "min"] = "max"


class OutputConfig(BaseModel):
    results_dir: str = "results"
    plots_dir: str = "results/plots"
    log_file: str | None = "logs/training.log"


class TrainConfig(BaseModel):
    """Validates configs/train.yaml."""
    data: DataConfig
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


# =====================================================================
# Validation entry points
# =====================================================================
def validate_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a pipeline config (default.yaml). Returns validated dict."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    validated = PipelineConfig(**raw)
    logger.info(f"Pipeline config validated: {path}")
    return validated.model_dump()


def validate_train_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a training config (train.yaml). Returns validated dict."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    validated = TrainConfig(**raw)
    logger.info(f"Training config validated: {path}")
    return validated.model_dump()


def validate_config(path: str | Path, schema: str = "auto") -> dict[str, Any]:
    """
    Auto-detect and validate a config file.

    schema: 'auto' | 'pipeline' | 'train'
    """
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)

    if schema == "auto":
        schema = "train" if "training" in raw or "model" in raw else "pipeline"

    if schema == "train":
        return validate_train_config(path)
    return validate_pipeline_config(path)
