"""Reproducible Ultralytics detector fine-tuning with local MLflow tracking."""

import csv
import json
import logging
import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

from surveillance_ml.core.device import seed_everything
from surveillance_ml.core.reproducibility import (
    configuration_fingerprint,
    git_commit_sha,
    package_version,
    runtime_versions,
)
from surveillance_ml.detection import DETECTION_CLASS_NAMES
from surveillance_ml.detection.config import DetectionConfig
from surveillance_ml.detection.dataset import DetectionDatasetReport, validate_yolo_dataset
from surveillance_ml.detection.evaluation import (
    evaluate_detection_checkpoint,
    write_metrics_report,
)
from surveillance_ml.tracking.mlflow_tracker import configure_tracking

LOGGER = logging.getLogger(__name__)


class _SingleRunDetectionTrainer(DetectionTrainer):
    """Keep Ultralytics plugins from creating a second experiment run."""

    def __init__(
        self,
        cfg: str | None = None,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict[str, Any] | None = None,
    ) -> None:
        if cfg is None:
            super().__init__(overrides=overrides, _callbacks=_callbacks)
        else:
            super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        for event, callbacks in self.callbacks.items():
            self.callbacks[event] = [
                callback
                for callback in callbacks
                if callback.__module__ != "ultralytics.utils.callbacks.mlflow"
            ]


@dataclass(frozen=True)
class DetectionTrainingOutcome:
    checkpoint_path: Path
    results_dir: Path
    mlflow_run_id: str | None
    epochs_completed: int
    training_seconds: float
    dataset_fingerprint: str
    stop_reason: str | None = None


def _resolve_device(requested: str) -> tuple[str, str]:
    if requested == "cpu":
        return "cpu", "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return "0", "cuda:0"
    if torch.cuda.is_available():
        return "0", "cuda:0"
    return "cpu", "cpu"


def _log_epoch_metrics(results_csv: Path) -> int:
    if not results_csv.is_file():
        raise FileNotFoundError(f"Ultralytics did not write training history: {results_csv}")
    epoch_count = 0
    with results_csv.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            epoch_text = row.get("epoch")
            step = int(float(epoch_text)) + 1 if epoch_text is not None else epoch_count + 1
            metrics: dict[str, float] = {}
            for name, value in row.items():
                if name == "epoch" or value is None or not value.strip():
                    continue
                try:
                    metric_value = float(value)
                except ValueError:
                    continue
                safe_name = (
                    name.strip()
                    .replace("/", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("-", "_")
                    .replace(" ", "_")
                )
                metrics[f"epoch_{safe_name}"] = metric_value
            if metrics:
                mlflow.log_metrics(metrics, step=step)
            epoch_count += 1
    if epoch_count == 0:
        raise ValueError(f"Ultralytics training history is empty: {results_csv}")
    return epoch_count


def _write_provenance(
    config: DetectionConfig,
    report: DetectionDatasetReport,
    *,
    run_id: str,
    device_name: str,
    results_dir: Path,
    epoch_count: int,
    training_seconds: float,
) -> Path:
    commit = git_commit_sha(Path(__file__).resolve().parents[3])
    versions = runtime_versions()
    versions["ultralytics"] = package_version("ultralytics")
    provenance: dict[str, Any] = {
        "architecture": config.model.architecture,
        "class_names": list(DETECTION_CLASS_NAMES),
        "pretrained": config.model.pretrained,
        "weights_source": (
            f"{config.model.architecture}.pt"
            if config.model.pretrained
            else f"{config.model.architecture}.yaml (random initialization)"
        ),
        "dataset_version": config.dataset.version,
        "dataset_fingerprint": report.dataset_fingerprint,
        "configuration_fingerprint": configuration_fingerprint(config),
        "mlflow_run_id": run_id,
        "git_commit_sha": commit,
        "runtime_versions": versions,
        "device": device_name,
        "epochs_completed": epoch_count,
        "training_seconds": training_seconds,
        "split_counts": report.split_counts,
        "license_notice": "Ultralytics code is AGPL-3.0; review model-weight terms separately.",
    }
    destination = results_dir / "provenance.json"
    destination.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return destination


def train_detector(
    config: DetectionConfig,
    *,
    output_dir: Path | None = None,
    run_name: str | None = None,
    resume_from: Path | None = None,
    stop_check: Callable[[], str | None] | None = None,
    stop_after_epochs: int | None = None,
) -> DetectionTrainingOutcome:
    """Train the configured detector, select its validation-best checkpoint, and log artifacts."""
    if output_dir is not None:
        config = replace(config, training=replace(config.training, output_dir=output_dir))
    if stop_after_epochs is not None and not 1 <= stop_after_epochs <= config.training.epochs:
        raise ValueError("stop_after_epochs must be between one and the configured epoch target")
    report = validate_yolo_dataset(config.dataset.root)
    seed_everything(config.training.seed)
    torch.set_num_threads(config.training.cpu_threads)
    training_device, device_name = _resolve_device(config.training.device)
    output_root = config.training.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config.tracking.uri)
    configure_tracking(tracking_uri, config.tracking.experiment_name)

    active = mlflow.active_run()
    with mlflow.start_run(
        run_name=run_name or f"{config.model.architecture}-{config.dataset.version}",
        nested=active is not None,
    ) as run:
        commit = git_commit_sha(Path(__file__).resolve().parents[3])
        versions = runtime_versions()
        versions["ultralytics"] = package_version("ultralytics")
        mlflow.set_tags(
            {
                "task": "object-detection",
                "architecture": config.model.architecture,
                "dataset_fingerprint": report.dataset_fingerprint,
                "git_commit_sha": commit or "unavailable",
                "license": "AGPL-3.0",
            }
        )
        mlflow.log_params(
            {
                "architecture": config.model.architecture,
                "pretrained": config.model.pretrained,
                "weights_source": (
                    f"{config.model.architecture}.pt"
                    if config.model.pretrained
                    else f"{config.model.architecture}.yaml"
                ),
                "dataset_version": config.dataset.version,
                "dataset_fingerprint": report.dataset_fingerprint,
                "configuration_fingerprint": configuration_fingerprint(config),
                "image_size": config.training.image_size,
                "epochs": config.training.epochs,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "optimizer": config.training.optimizer,
                "patience": config.training.patience,
                "seed": config.training.seed,
                "device": device_name,
                "class_names": ",".join(DETECTION_CLASS_NAMES),
            }
        )
        for name, version in versions.items():
            mlflow.set_tag(f"runtime_{name}", version)
        mlflow.log_dict(report.to_dict(), "data/validation.json")
        mlflow.log_artifact(str(config.dataset.root / "dataset.yaml"), artifact_path="data")

        model_reference = (
            f"{config.model.architecture}.pt"
            if config.model.pretrained
            else f"{config.model.architecture}.yaml"
        )
        if resume_from is not None:
            if not resume_from.is_file():
                raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_from}")
            model = YOLO(str(resume_from))
        else:
            model = YOLO(model_reference)
        stop_reasons: list[str] = []
        if stop_check is not None or stop_after_epochs is not None:

            def stop_after_epoch(trainer: DetectionTrainer) -> None:
                reason = stop_check() if stop_check is not None else None
                if reason is not None:
                    stop_reasons.append(reason)
                    trainer.stop = True
                elif stop_after_epochs is not None and trainer.epoch + 1 >= stop_after_epochs:
                    stop_reasons.append("stage_boundary")
                    trainer.stop = True

            model.add_callback("on_train_epoch_end", stop_after_epoch)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=".ultralytics-dataset-",
            dir=output_root,
            encoding="utf-8",
            delete=False,
        ) as temporary_yaml:
            yaml.safe_dump(
                {
                    "path": str(config.dataset.root.resolve()),
                    "train": "images/train",
                    "val": "images/val",
                    "test": "images/test",
                    "nc": len(DETECTION_CLASS_NAMES),
                    "names": list(DETECTION_CLASS_NAMES),
                },
                sort_keys=False,
                stream=temporary_yaml,
            )
        training_dataset_yaml = Path(temporary_yaml.name)
        label_root = config.dataset.root / "labels"
        existing_label_caches = set(label_root.rglob("*.cache"))
        started = time.perf_counter()
        try:
            model.train(
                trainer=_SingleRunDetectionTrainer,
                data=str(training_dataset_yaml),
                imgsz=config.training.image_size,
                epochs=config.training.epochs,
                batch=config.training.batch_size,
                workers=config.training.workers,
                device=training_device,
                seed=config.training.seed,
                deterministic=True,
                optimizer=config.training.optimizer,
                lr0=config.training.learning_rate,
                patience=config.training.patience,
                pretrained=config.model.pretrained,
                resume=resume_from is not None,
                amp=config.training.mixed_precision and training_device != "cpu",
                cache=False,
                plots=False,
                val=True,
                save=True,
                project=str(output_root.parent),
                name=output_root.name,
                exist_ok=True,
                verbose=False,
            )
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA ran out of memory; lower batch size or image size and retry."
            ) from error
        except RuntimeError as error:
            if "cuda out of memory" in str(error).lower():
                torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA ran out of memory; lower batch size or image size and retry."
                ) from error
            raise
        finally:
            training_dataset_yaml.unlink(missing_ok=True)
            for cache_path in label_root.rglob("*.cache"):
                if cache_path not in existing_label_caches:
                    cache_path.unlink()
        training_seconds = time.perf_counter() - started

        trainer = model.trainer
        if trainer is None:
            raise RuntimeError("Ultralytics did not attach its trainer after fitting")
        results_dir = Path(trainer.save_dir).resolve()
        checkpoint_path = Path(trainer.best).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Best validation checkpoint was not saved: {checkpoint_path}")
        epoch_count = _log_epoch_metrics(results_dir / "results.csv")
        provenance_path = _write_provenance(
            config,
            report,
            run_id=run.info.run_id,
            device_name=device_name,
            results_dir=results_dir,
            epoch_count=epoch_count,
            training_seconds=training_seconds,
        )
        mlflow.log_metrics(
            {
                "training_seconds": training_seconds,
                "epochs_completed": float(epoch_count),
            }
        )
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(results_dir / "results.csv"), artifact_path="training")
        mlflow.log_artifact(str(provenance_path), artifact_path="provenance")
        LOGGER.info(
            "Completed detector run %s, best checkpoint %s, elapsed %.2fs",
            run.info.run_id,
            checkpoint_path,
            training_seconds,
        )
        return DetectionTrainingOutcome(
            checkpoint_path=checkpoint_path,
            results_dir=results_dir,
            mlflow_run_id=run.info.run_id,
            epochs_completed=epoch_count,
            training_seconds=training_seconds,
            dataset_fingerprint=report.dataset_fingerprint,
            stop_reason=(
                stop_reasons[0]
                if stop_reasons
                else "early_stopping"
                if epoch_count < config.training.epochs
                else None
            ),
        )


def evaluate_and_log_detector(
    config: DetectionConfig,
    checkpoint_path: Path,
    *,
    split: str,
    output_path: Path,
    parent_run_id: str | None = None,
    collect_samples: bool = False,
) -> dict[str, Any]:
    """Evaluate one split and persist metrics in a dedicated MLflow evaluation run."""
    dataset_report = validate_yolo_dataset(config.dataset.root)
    _, device_name = _resolve_device(config.training.device)
    metrics = evaluate_detection_checkpoint(
        checkpoint_path,
        dataset_report,
        split=split,
        image_size=config.training.image_size,
        confidence_threshold=config.evaluation.metric_confidence_threshold,
        operating_confidence_threshold=config.evaluation.confidence_threshold,
        nms_iou_threshold=config.evaluation.nms_iou_threshold,
        max_detections=config.evaluation.max_detections,
        device=device_name,
        collect_samples=collect_samples,
    )
    write_metrics_report(metrics, output_path)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config.tracking.uri)
    configure_tracking(tracking_uri, config.tracking.experiment_name)
    active = mlflow.active_run()
    with mlflow.start_run(
        run_name=f"{config.model.architecture}-{split}-evaluation",
        nested=active is not None,
    ) as run:
        mlflow.set_tags(
            {
                "task": "object-detection-evaluation",
                "architecture": config.model.architecture,
                "dataset_fingerprint": dataset_report.dataset_fingerprint,
                "split": split,
                "parent_training_run_id": parent_run_id or "unavailable",
            }
        )
        mlflow.log_params(
            {
                "checkpoint_sha256": _sha256_file(checkpoint_path),
                "image_size": config.training.image_size,
                "confidence_threshold": config.evaluation.confidence_threshold,
                "metric_confidence_threshold": config.evaluation.metric_confidence_threshold,
                "nms_iou_threshold": config.evaluation.nms_iou_threshold,
                "device": device_name,
            }
        )
        metrics_to_log = {
            name: value
            for name, value in metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and name not in {"sample_count"}
        }
        mlflow.log_metrics({name: float(value) for name, value in metrics_to_log.items()})
        for class_name, class_metrics in metrics["per_class"].items():
            mlflow.log_metrics(
                {
                    f"{class_name}_{name}": float(value)
                    for name, value in class_metrics.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            )
        mlflow.log_artifact(str(output_path), artifact_path="evaluation")
    metrics["mlflow_run_id"] = run.info.run_id
    write_metrics_report(metrics, output_path)
    return metrics


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
