"""Typed, validated configuration for the independent YOLO detection pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DetectionDatasetConfig:
    root: Path
    version: str


@dataclass(frozen=True)
class DetectionModelConfig:
    architecture: str
    pretrained: bool


@dataclass(frozen=True)
class DetectionTrainingConfig:
    output_dir: Path
    image_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    patience: int
    seed: int
    device: str
    workers: int
    cpu_threads: int
    mixed_precision: bool


@dataclass(frozen=True)
class DetectionEvaluationConfig:
    confidence_threshold: float
    metric_confidence_threshold: float
    nms_iou_threshold: float
    max_detections: int


@dataclass(frozen=True)
class DetectionTrackingConfig:
    uri: str
    experiment_name: str


@dataclass(frozen=True)
class DetectionExportConfig:
    model_id: str
    version: str
    opset: int


@dataclass(frozen=True)
class DetectionConfig:
    dataset: DetectionDatasetConfig
    model: DetectionModelConfig
    training: DetectionTrainingConfig
    evaluation: DetectionEvaluationConfig
    tracking: DetectionTrackingConfig
    export: DetectionExportConfig


def load_detection_config(path: Path) -> DetectionConfig:
    """Load a detection YAML file and reject unsupported or unsafe values."""
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid detection configuration: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError(f"Detection configuration must be a YAML mapping: {path}")
    try:
        dataset_values = dict(raw["dataset"])
        model_values = dict(raw["model"])
        training_values = dict(raw["training"])
        evaluation_values = dict(raw["evaluation"])
        tracking_values = dict(raw["tracking"])
        export_values = dict(raw["export"])
        dataset_values["root"] = Path(dataset_values["root"])
        training_values["output_dir"] = Path(training_values["output_dir"])
        config = DetectionConfig(
            dataset=DetectionDatasetConfig(**dataset_values),
            model=DetectionModelConfig(**model_values),
            training=DetectionTrainingConfig(**training_values),
            evaluation=DetectionEvaluationConfig(**evaluation_values),
            tracking=DetectionTrackingConfig(**tracking_values),
            export=DetectionExportConfig(**export_values),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing or invalid detection configuration field: {error}") from error

    if config.model.architecture not in {"yolo11n", "yolo11s"}:
        raise ValueError("model.architecture must be yolo11n or yolo11s")
    if config.training.image_size < 64 or config.training.image_size > 2048:
        raise ValueError("training.image_size must be between 64 and 2048")
    if config.training.image_size % 32:
        raise ValueError("training.image_size must be divisible by 32")
    if config.training.epochs < 1 or config.training.batch_size < 1:
        raise ValueError("training.epochs and training.batch_size must be positive")
    if config.training.learning_rate <= 0 or config.training.patience < 0:
        raise ValueError("training.learning_rate must be positive and patience non-negative")
    if config.training.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("training.device must be auto, cpu, or cuda")
    if config.training.workers < 0 or config.training.cpu_threads < 1:
        raise ValueError("training.workers must be non-negative and cpu_threads positive")
    if config.training.optimizer not in {"AdamW", "Adam", "SGD"}:
        raise ValueError("training.optimizer must be AdamW, Adam, or SGD")
    thresholds = (
        config.evaluation.confidence_threshold,
        config.evaluation.metric_confidence_threshold,
        config.evaluation.nms_iou_threshold,
    )
    if any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("Detection confidence and IoU thresholds must be between zero and one")
    if config.evaluation.max_detections < 1:
        raise ValueError("evaluation.max_detections must be positive")
    if config.export.opset < 17:
        raise ValueError("export.opset must be at least 17")
    for label, value in (
        ("dataset.version", config.dataset.version),
        ("tracking.experiment_name", config.tracking.experiment_name),
        ("export.model_id", config.export.model_id),
        ("export.version", config.export.version),
    ):
        if not value.strip():
            raise ValueError(f"{label} must not be empty")
    model_identifier = config.export.model_id + config.export.version
    if any(character in model_identifier for character in ("/", "\\", "..")):
        raise ValueError("export model_id and version must not contain path components")
    return config
