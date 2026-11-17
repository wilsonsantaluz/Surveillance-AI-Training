"""Reusable training, validation, optional test evaluation, and MLflow logging."""

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from surveillance_ml.core.config import AppConfig
from surveillance_ml.core.device import seed_everything, select_device
from surveillance_ml.core.reproducibility import (
    configuration_fingerprint,
    dataset_fingerprint,
    git_commit_sha,
    runtime_versions,
    split_fingerprint,
)
from surveillance_ml.data.dataset import (
    CLASS_NAMES,
    ImageRecord,
    SceneImageDataset,
    discover_images,
)
from surveillance_ml.data.splitting import DatasetSplits, class_distribution, split_records
from surveillance_ml.models.classifier import build_resnet18
from surveillance_ml.tracking.mlflow_tracker import configure_tracking
from surveillance_ml.training.evaluate import evaluate_model, metric_values
from surveillance_ml.training.metrics import ClassificationMetrics
from surveillance_ml.training.trainer import TrainingResult, load_best_checkpoint, train_model

LOGGER = logging.getLogger(__name__)
EpochCallback = Callable[[int, dict[str, float]], None]


@dataclass(frozen=True)
class TrainingOutcome:
    checkpoint_path: Path
    result: TrainingResult
    validation_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics | None
    mlflow_run_id: str
    dataset_fingerprint: str
    split_fingerprint: str
    configuration_fingerprint: str


def _distribution_summary(
    splits: DatasetSplits,
    imbalance_ratio: float,
    include_test: bool,
) -> dict[str, dict[str, object]]:
    names = ("train", "validation", "test") if include_test else ("train", "validation")
    return {
        name: class_distribution(getattr(splits, name), imbalance_ratio)
        for name in names
    }


def _split_counts(splits: DatasetSplits, include_test: bool) -> dict[str, dict[str, int]]:
    names = ("train", "validation", "test") if include_test else ("train", "validation")
    return {
        name: {
            "images": len(getattr(splits, name)),
            "source_groups": len({record.source_id for record in getattr(splits, name)}),
        }
        for name in names
    }


def _create_loaders(
    splits: DatasetSplits,
    config: AppConfig,
    include_test: bool,
) -> tuple[
    DataLoader[tuple[torch.Tensor, int]],
    DataLoader[tuple[torch.Tensor, int]],
    DataLoader[tuple[torch.Tensor, int]] | None,
]:
    image_size = config.dataset.image_size
    generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(
        SceneImageDataset(splits.train, image_size, augment=True),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        SceneImageDataset(splits.validation, image_size),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    test_loader = None
    if include_test:
        test_loader = DataLoader(
            SceneImageDataset(splits.test, image_size),
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
        )
    return train_loader, validation_loader, test_loader


def _log_training(
    config: AppConfig,
    splits: DatasetSplits,
    all_records: list[ImageRecord],
    model: torch.nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    validation_loader: DataLoader[tuple[torch.Tensor, int]],
    test_loader: DataLoader[tuple[torch.Tensor, int]] | None,
    device: torch.device,
    output_dir: Path,
    data_hash: str,
    split_hash: str,
    config_hash: str,
    epoch_callback: EpochCallback | None,
    tags: dict[str, str] | None,
) -> TrainingOutcome:
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("Training metrics must be logged inside an active MLflow run")
    experiment = mlflow.get_experiment(active_run.info.experiment_id)
    experiment_id = experiment.experiment_id if experiment else active_run.info.experiment_id
    commit_sha = git_commit_sha(Path(__file__).resolve().parents[3])
    versions = runtime_versions()
    if tags:
        mlflow.set_tags(tags)
    mlflow.set_tags(
        {
            "dataset_fingerprint": data_hash,
            "split_fingerprint": split_hash,
            "configuration_fingerprint": config_hash,
            "git_commit_sha": commit_sha or "unavailable",
        }
    )
    params: dict[str, str | int | float | bool] = {
        "experiment_id": experiment_id,
        "architecture": config.model.architecture,
        "pretrained": config.model.pretrained,
        "weights": (
            "torchvision:ResNet18_Weights.IMAGENET1K_V1"
            if config.model.pretrained
            else "random-initialization"
        ),
        "fine_tuning_strategy": config.model.strategy,
        "dataset_version": config.dataset.version,
        "dataset_size": len(all_records),
        "dataset_fingerprint": data_hash,
        "split_fingerprint": split_hash,
        "configuration_fingerprint": config_hash,
        "image_size": config.dataset.image_size,
        "seed": config.training.seed,
        "optimizer": config.training.optimizer,
        "learning_rate": config.training.learning_rate,
        "batch_size": config.training.batch_size,
        "configured_epochs": config.training.epochs,
        "weight_decay": config.training.weight_decay,
        "train_fraction": config.dataset.train_fraction,
        "validation_fraction": config.dataset.validation_fraction,
        "test_fraction": config.dataset.test_fraction,
        "device": str(device),
        "mixed_precision_enabled": config.training.mixed_precision and device.type == "cuda",
        "class_names": ",".join(CLASS_NAMES),
    }
    params.update({f"version_{name}": value for name, value in versions.items()})
    mlflow.log_params(params)
    mlflow.log_dict(
        _distribution_summary(
            splits,
            config.dataset.imbalance_ratio,
            test_loader is not None,
        ),
        "data/class_distribution.json",
    )
    mlflow.log_dict(_split_counts(splits, test_loader is not None), "data/split_summary.json")

    def log_epoch(epoch: int, values: dict[str, float]) -> None:
        mlflow.log_metrics(values, step=epoch)
        if epoch_callback is not None:
            epoch_callback(epoch, values)
        LOGGER.info("epoch %d: %s", epoch, values)

    result = train_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=device,
        output_dir=output_dir,
        class_names=CLASS_NAMES,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        optimizer_name=config.training.optimizer,
        weight_decay=config.training.weight_decay,
        scheduler_factor=config.training.scheduler_factor,
        scheduler_patience=config.training.scheduler_patience,
        early_stopping_patience=config.training.early_stopping_patience,
        mixed_precision=config.training.mixed_precision,
        epoch_callback=log_epoch,
    )
    checkpoint_path = output_dir / "best.pt"
    checkpoint = load_best_checkpoint(model, checkpoint_path, device)
    validation_metrics = result.best_validation_metrics
    test_metrics = (
        evaluate_model(model, test_loader, device, len(CLASS_NAMES))
        if test_loader is not None
        else None
    )
    mlflow.log_metrics(metric_values(validation_metrics, "best_validation_"))
    if test_metrics is not None:
        mlflow.log_metrics(metric_values(test_metrics, "test_"))
        mlflow.log_dict(test_metrics.to_dict(), "evaluation/test_metrics.json")
    mlflow.log_metrics(
        {
            "training_seconds": result.elapsed_seconds,
            "best_epoch": float(result.best_epoch),
            "best_validation_loss": result.best_validation_loss,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    validation_report = {
        "run_id": active_run.info.run_id,
        "experiment_id": experiment_id,
        "architecture": config.model.architecture,
        "dataset_version": config.dataset.version,
        "dataset_fingerprint": data_hash,
        "split_fingerprint": split_hash,
        "configuration_fingerprint": config_hash,
        "git_commit_sha": commit_sha,
        "runtime_versions": versions,
        "device": str(device),
        "seed": config.training.seed,
        "best_epoch": result.best_epoch,
        "best_validation_loss": result.best_validation_loss,
        "best_validation_metrics": validation_metrics.to_dict(),
        "training_seconds": result.elapsed_seconds,
        "test_evaluated": test_metrics is not None,
        "test_metrics": test_metrics.to_dict() if test_metrics is not None else None,
        "class_names": list(CLASS_NAMES),
        "checkpoint_epoch": checkpoint.get("epoch"),
    }
    validation_path = output_dir / "validation.json"
    validation_path.write_text(json.dumps(validation_report, indent=2) + "\n", encoding="utf-8")
    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(result.history, indent=2) + "\n", encoding="utf-8")
    mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
    mlflow.log_artifact(str(validation_path), artifact_path="evaluation")
    mlflow.log_artifact(str(history_path), artifact_path="evaluation")
    if test_metrics is not None:
        evaluation_path = output_dir / "evaluation.json"
        evaluation_path.write_text(json.dumps(validation_report, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(evaluation_path), artifact_path="evaluation")
    LOGGER.info(
        "Completed MLflow run %s; best epoch %d; validation metrics %s; test metrics %s",
        active_run.info.run_id,
        result.best_epoch,
        validation_metrics.to_dict(),
        test_metrics.to_dict() if test_metrics is not None else "not evaluated",
    )
    return TrainingOutcome(
        checkpoint_path=checkpoint_path,
        result=result,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        mlflow_run_id=active_run.info.run_id,
        dataset_fingerprint=data_hash,
        split_fingerprint=split_hash,
        configuration_fingerprint=config_hash,
    )


def run_training_experiment(
    config: AppConfig,
    *,
    evaluate_test: bool = True,
    output_dir: Path | None = None,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
    epoch_callback: EpochCallback | None = None,
    dataset_fingerprint_value: str | None = None,
) -> TrainingOutcome:
    """Train and validate; access held-out test images only when explicitly requested."""
    if output_dir is not None:
        config = replace(config, training=replace(config.training, output_dir=output_dir))
    seed_everything(config.training.seed)
    torch.set_num_threads(config.training.cpu_threads)
    device = select_device()
    records = discover_images(config.dataset.root)
    splits = split_records(
        records,
        config.training.seed,
        (
            config.dataset.train_fraction,
            config.dataset.validation_fraction,
            config.dataset.test_fraction,
        ),
    )
    distributions = _distribution_summary(
        splits,
        config.dataset.imbalance_ratio,
        include_test=evaluate_test,
    )
    for name, distribution in distributions.items():
        LOGGER.info("%s distribution: %s", name, distribution)

    train_loader, validation_loader, test_loader = _create_loaders(
        splits,
        config,
        include_test=evaluate_test,
    )
    model = build_resnet18(
        len(CLASS_NAMES),
        strategy=config.model.strategy,
        pretrained=config.model.pretrained,
    )
    config.training.output_dir.mkdir(parents=True, exist_ok=True)
    data_hash = dataset_fingerprint_value or dataset_fingerprint(config.dataset.root)
    split_hash = split_fingerprint(
        {
            "train": splits.train,
            "validation": splits.validation,
            "test": splits.test,
        },
        config.dataset.root,
    )
    config_hash = configuration_fingerprint(config)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config.tracking.uri)
    configure_tracking(tracking_uri, config.tracking.experiment_name)

    active_run = mlflow.active_run()
    if active_run is None:
        with mlflow.start_run(run_name=run_name):
            return _log_training(
                config,
                splits,
                records,
                model,
                train_loader,
                validation_loader,
                test_loader,
                device,
                config.training.output_dir,
                data_hash,
                split_hash,
                config_hash,
                epoch_callback,
                tags,
            )
    return _log_training(
        config,
        splits,
        records,
        model,
        train_loader,
        validation_loader,
        test_loader,
        device,
        config.training.output_dir,
        data_hash,
        split_hash,
        config_hash,
        epoch_callback,
        tags,
    )


def run_training(config: AppConfig) -> Path:
    """Backward-compatible Phase 1 entry point returning the best checkpoint path."""
    return run_training_experiment(config).checkpoint_path
