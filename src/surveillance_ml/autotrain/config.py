"""Typed configuration and validation for AutoTrain."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    output_root: Path


@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    format: str
    version: str
    source: str
    license: str
    provenance: str
    license_review_required: bool
    license_review_status: str


@dataclass(frozen=True)
class ModelConfig:
    architecture: str
    pretrained: bool
    image_size: int


@dataclass(frozen=True)
class TrainingConfig:
    device: str
    smoke_epochs: int
    pilot_epochs: int
    max_total_epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    patience: int
    workers: int
    cpu_threads: int
    mixed_precision: bool
    max_runtime_minutes: float


@dataclass(frozen=True)
class AuditConfig:
    verify_images: bool
    verify_annotations: bool
    detect_exact_duplicates: bool
    detect_perceptual_duplicates: bool
    verify_source_group_splits: bool
    generate_contact_sheets: bool
    contact_sheet_samples_per_category: int
    perceptual_hamming_threshold: int
    minimum_resolution: int
    small_box_area_fraction: float
    large_box_area_fraction: float
    maximum_negative_fraction: float
    require_manual_annotation_review: bool


@dataclass(frozen=True)
class TrackingConfig:
    mlflow_enabled: bool
    uri: str
    experiment_name: str
    dvc_enabled: bool


@dataclass(frozen=True)
class QualityGateConfig:
    min_validation_map50: float | None
    min_firearm_ap50: float | None
    min_knife_ap50: float | None
    min_validation_precision: float | None
    min_validation_recall: float | None
    max_false_positives_per_hour: float | None
    require_source_split_integrity: bool
    require_manifest_validation: bool
    require_python_dotnet_parity: bool
    require_manual_deployment_approval: bool


@dataclass(frozen=True)
class ExportConfig:
    format: str
    model_id: str
    version: str
    opset: int
    export_candidate: bool
    auto_deploy: bool


@dataclass(frozen=True)
class AutoTrainConfig:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    audit: AuditConfig
    tracking: TrackingConfig
    quality_gates: QualityGateConfig
    export: ExportConfig
    repository_root: Path
    configuration_path: Path

    @property
    def fingerprint(self) -> str:
        """Return a stable hash of all execution settings, excluding file location."""
        values = asdict(self)
        values.pop("configuration_path")
        values.pop("repository_root")
        values["tracking"]["uri"] = os.environ.get(
            "MLFLOW_TRACKING_URI",
            values["tracking"]["uri"],
        )
        payload = json.dumps(
            _jsonable(values),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def snapshot(self) -> dict[str, Any]:
        """Return a portable YAML-safe configuration snapshot."""
        values = {
            key: value
            for key, value in asdict(self).items()
            if key not in {"repository_root", "configuration_path"}
        }
        values["tracking"]["uri"] = os.environ.get(
            "MLFLOW_TRACKING_URI",
            values["tracking"]["uri"],
        )
        return {
            **_jsonable(values),
            "_repository_root": str(self.repository_root),
        }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _repository_root(path: Path, raw: dict[str, Any]) -> Path:
    configured = raw.get("_repository_root")
    if configured is not None:
        if not isinstance(configured, str):
            raise ValueError("_repository_root must be a path string")
        return Path(configured).resolve()
    for parent in path.resolve().parents:
        if (parent / ".git").exists():
            return parent
    return path.resolve().parent.parent


def _resolve_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _construct_config(raw: Any, path: Path) -> AutoTrainConfig:
    if not isinstance(raw, dict):
        raise ValueError("AutoTrain configuration must be a YAML mapping")
    root = _repository_root(path, raw)
    try:
        experiment_values = dict(raw["experiment"])
        experiment_values["output_root"] = _resolve_path(
            root,
            experiment_values["output_root"],
            "experiment.output_root",
        )
        dataset_values = dict(raw["dataset"])
        dataset_values["path"] = _resolve_path(root, dataset_values["path"], "dataset.path")
        config = AutoTrainConfig(
            experiment=ExperimentConfig(**experiment_values),
            dataset=DatasetConfig(**dataset_values),
            model=ModelConfig(**dict(raw["model"])),
            training=TrainingConfig(**dict(raw["training"])),
            audit=AuditConfig(**dict(raw["audit"])),
            tracking=TrackingConfig(**dict(raw["tracking"])),
            quality_gates=QualityGateConfig(**dict(raw["quality_gates"])),
            export=ExportConfig(**dict(raw["export"])),
            repository_root=root,
            configuration_path=path.resolve(),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Missing or invalid AutoTrain configuration field: {error}") from error
    _validate_config(config)
    return config


def _validate_config(config: AutoTrainConfig) -> None:
    if not config.experiment.name.strip():
        raise ValueError("experiment.name must not be empty")
    if config.experiment.seed < 0:
        raise ValueError("experiment.seed must be non-negative")
    if config.dataset.format != "yolo" or config.dataset.source != "local":
        raise ValueError(
            "AutoTrain currently supports only an already-downloaded local YOLO dataset"
        )
    if not config.dataset.version.strip() or not config.dataset.license_review_status.strip():
        raise ValueError("dataset.version and license_review_status must not be empty")
    if config.dataset.license_review_status not in {"unreviewed", "approved", "restricted"}:
        raise ValueError(
            "dataset.license_review_status must be unreviewed, approved, or restricted"
        )
    if config.model.architecture not in {"yolo11n", "yolo11s"}:
        raise ValueError("model.architecture must be yolo11n or yolo11s")
    if config.model.image_size < 64 or config.model.image_size > 2048:
        raise ValueError("model.image_size must be between 64 and 2048")
    if config.model.image_size % 32:
        raise ValueError("model.image_size must be divisible by 32")
    training = config.training
    if training.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("training.device must be auto, cpu, or cuda")
    if not 1 <= training.smoke_epochs < training.pilot_epochs <= training.max_total_epochs:
        raise ValueError("Training epochs must satisfy 1 <= smoke < pilot <= max_total_epochs")
    if training.batch_size < 1 or training.workers < 0 or training.cpu_threads < 1:
        raise ValueError(
            "batch_size must be positive, workers non-negative, and cpu_threads positive"
        )
    if training.learning_rate <= 0 or training.patience < 0:
        raise ValueError("learning_rate must be positive and patience non-negative")
    if not 0 < training.max_runtime_minutes < float("inf"):
        raise ValueError("training.max_runtime_minutes must be finite and positive")
    if training.optimizer not in {"AdamW", "Adam", "SGD"}:
        raise ValueError("training.optimizer must be AdamW, Adam, or SGD")

    audit = config.audit
    if audit.contact_sheet_samples_per_category < 1 or audit.minimum_resolution < 1:
        raise ValueError("Contact-sheet sample count and minimum resolution must be positive")
    if not 0 <= audit.perceptual_hamming_threshold <= 16:
        raise ValueError("audit.perceptual_hamming_threshold must be between zero and 16")
    if not 0 < audit.small_box_area_fraction < audit.large_box_area_fraction <= 1:
        raise ValueError("Box area thresholds must satisfy 0 < small < large <= 1")
    if not 0 <= audit.maximum_negative_fraction <= 1:
        raise ValueError("audit.maximum_negative_fraction must be between zero and one")

    tracking = config.tracking
    if not tracking.mlflow_enabled:
        raise ValueError("MLflow tracking is required for every AutoTrain run")
    if not tracking.uri.strip() or not tracking.experiment_name.strip():
        raise ValueError("tracking.uri and tracking.experiment_name must not be empty")
    if not config.export.format == "onnx":
        raise ValueError("Only the existing ONNX exporter is supported")
    if (
        config.export.opset < 17
        or not config.export.model_id.strip()
        or not config.export.version.strip()
    ):
        raise ValueError("Export requires opset >= 17 and non-empty model id/version")
    if config.export.auto_deploy:
        raise ValueError("AutoTrain never deploys models automatically")
    for name in (
        "min_validation_map50",
        "min_firearm_ap50",
        "min_knife_ap50",
        "min_validation_precision",
        "min_validation_recall",
    ):
        threshold = getattr(config.quality_gates, name)
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError(f"quality_gates.{name} must be between zero and one or null")
    false_positives = config.quality_gates.max_false_positives_per_hour
    if false_positives is not None and false_positives < 0:
        raise ValueError("quality_gates.max_false_positives_per_hour must be non-negative or null")


def load_autotrain_config(path: Path) -> AutoTrainConfig:
    """Load and validate an AutoTrain YAML configuration."""
    try:
        raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FileNotFoundError(f"AutoTrain configuration does not exist: {path}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid AutoTrain YAML configuration: {path}") from error
    return _construct_config(raw, path)


def save_config_snapshot(config: AutoTrainConfig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        yaml.safe_dump(config.snapshot(), sort_keys=False),
        encoding="utf-8",
    )


def resolve_tracking_uri(uri: str, repository_root: Path) -> str:
    """Apply the standard environment override and anchor relative SQLite paths."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", uri)
    prefix = "sqlite:///"
    if uri.startswith(prefix):
        sqlite_path = Path(uri[len(prefix) :])
        if not sqlite_path.is_absolute():
            return prefix + (repository_root / sqlite_path).resolve().as_posix()
    return uri
