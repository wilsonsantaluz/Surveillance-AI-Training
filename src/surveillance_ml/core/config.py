"""Typed configuration loading and validation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    root: Path
    version: str
    image_size: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    imbalance_ratio: float


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: Path
    seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    scheduler_factor: float
    scheduler_patience: int
    early_stopping_patience: int
    num_workers: int
    cpu_threads: int
    mixed_precision: bool


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    pretrained: bool
    strategy: str


@dataclass(frozen=True)
class TrackingConfig:
    uri: str
    experiment_name: str


@dataclass(frozen=True)
class AppConfig:
    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    tracking: TrackingConfig


def load_config(path: Path) -> AppConfig:
    """Load a YAML configuration, validating values before any training begins."""
    with path.open(encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {path}")

    try:
        dataset_values = dict(raw["dataset"])
        training_values = dict(raw["training"])
        model_values = dict(raw["model"])
        tracking_values = dict(raw["tracking"])
        dataset_values["root"] = Path(dataset_values["root"])
        training_values["output_dir"] = Path(training_values["output_dir"])
        config = AppConfig(
            dataset=DatasetConfig(**dataset_values),
            training=TrainingConfig(**training_values),
            model=ModelConfig(**model_values),
            tracking=TrackingConfig(**tracking_values),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing or invalid configuration field: {error}") from error

    if config.dataset.image_size < 32:
        raise ValueError("dataset.image_size must be at least 32")
    fractions = (
        config.dataset.train_fraction,
        config.dataset.validation_fraction,
        config.dataset.test_fraction,
    )
    if any(fraction <= 0 or fraction >= 1 for fraction in fractions):
        raise ValueError("All dataset split fractions must be between 0 and 1")
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Dataset split fractions must sum to 1")
    if config.dataset.imbalance_ratio < 1:
        raise ValueError("dataset.imbalance_ratio must be at least 1")
    if config.training.batch_size < 1 or config.training.epochs < 1:
        raise ValueError("training.batch_size and training.epochs must be positive")
    if config.training.learning_rate <= 0 or config.training.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    if config.training.optimizer not in {"adam", "adamw", "sgd"}:
        raise ValueError("training.optimizer must be one of: adam, adamw, sgd")
    if config.training.num_workers < 0 or config.training.cpu_threads < 1:
        raise ValueError("Worker count must be non-negative and CPU threads positive")
    if not 0 < config.training.scheduler_factor < 1:
        raise ValueError("training.scheduler_factor must be between 0 and 1")
    if config.training.scheduler_patience < 0 or config.training.early_stopping_patience < 1:
        raise ValueError("Scheduler patience must be non-negative and early stopping positive")
    if config.model.architecture != "resnet18":
        raise ValueError("Only the resnet18 architecture is supported in Phase 1")
    if config.model.strategy not in {"head", "partial", "full"}:
        raise ValueError("model.strategy must be one of: head, partial, full")
    if not config.tracking.experiment_name.strip():
        raise ValueError("tracking.experiment_name must not be empty")
    return config
