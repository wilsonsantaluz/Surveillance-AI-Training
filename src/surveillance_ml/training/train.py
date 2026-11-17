"""End-to-end configured ResNet18 training, test evaluation, and MLflow logging."""

import json
import logging
import os
from pathlib import Path

import mlflow
import torch
from torch.utils.data import DataLoader

from surveillance_ml.core.config import AppConfig
from surveillance_ml.core.device import seed_everything, select_device
from surveillance_ml.data.dataset import CLASS_NAMES, SceneImageDataset, discover_images
from surveillance_ml.data.splitting import class_distribution, split_records
from surveillance_ml.models.classifier import build_resnet18
from surveillance_ml.tracking.mlflow_tracker import configure_tracking
from surveillance_ml.training.evaluate import evaluate_model, metric_values
from surveillance_ml.training.trainer import load_best_checkpoint, train_model

LOGGER = logging.getLogger(__name__)


def run_training(config: AppConfig) -> Path:
    """Run reproducible training and return the persisted best checkpoint path."""
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
    distributions = {
        name: class_distribution(getattr(splits, name), config.dataset.imbalance_ratio)
        for name in ("train", "validation", "test")
    }
    for name, distribution in distributions.items():
        LOGGER.info("%s distribution: %s", name, distribution)

    image_size = config.dataset.image_size
    train_dataset = SceneImageDataset(splits.train, image_size, augment=True)
    validation_dataset = SceneImageDataset(splits.validation, image_size)
    test_dataset = SceneImageDataset(splits.test, image_size)
    generator = torch.Generator().manual_seed(config.training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    model = build_resnet18(
        len(CLASS_NAMES),
        strategy=config.model.strategy,
        pretrained=config.model.pretrained,
    )
    config.training.output_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config.tracking.uri)
    experiment_id = configure_tracking(tracking_uri, config.tracking.experiment_name)
    checkpoint_path = config.training.output_dir / "best.pt"

    with mlflow.start_run() as active_run:
        mlflow.log_params(
            {
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
                "dataset_size": len(records),
                "image_size": image_size,
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
                "mixed_precision_enabled": config.training.mixed_precision
                and device.type == "cuda",
                "class_names": ",".join(CLASS_NAMES),
            }
        )
        mlflow.log_dict(distributions, "data/class_distribution.json")
        mlflow.log_dict(
            {
                name: {
                    "images": len(getattr(splits, name)),
                    "source_groups": len(
                        {record.source_id for record in getattr(splits, name)}
                    ),
                }
                for name in ("train", "validation", "test")
            },
            "data/split_summary.json",
        )

        def log_epoch(epoch: int, values: dict[str, float]) -> None:
            mlflow.log_metrics(values, step=epoch)
            LOGGER.info("epoch %d: %s", epoch, values)

        result = train_model(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            device=device,
            output_dir=config.training.output_dir,
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
        checkpoint = load_best_checkpoint(model, checkpoint_path, device)
        test_metrics = evaluate_model(model, test_loader, device, len(CLASS_NAMES))
        mlflow.log_metrics(metric_values(test_metrics, "test_"))
        mlflow.log_metrics(
            {
                "training_seconds": result.elapsed_seconds,
                "best_epoch": float(result.best_epoch),
                "best_validation_loss": result.best_validation_loss,
            }
        )
        report = {
            "run_id": active_run.info.run_id,
            "experiment_id": experiment_id,
            "architecture": config.model.architecture,
            "dataset_version": config.dataset.version,
            "device": str(device),
            "seed": config.training.seed,
            "best_epoch": result.best_epoch,
            "best_validation_loss": result.best_validation_loss,
            "training_seconds": result.elapsed_seconds,
            "test_metrics": test_metrics.to_dict(),
            "class_names": list(CLASS_NAMES),
            "checkpoint_epoch": checkpoint.get("epoch"),
        }
        report_path = config.training.output_dir / "evaluation.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        history_path = config.training.output_dir / "history.json"
        history_path.write_text(json.dumps(result.history, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(report_path), artifact_path="evaluation")
        mlflow.log_artifact(str(history_path), artifact_path="evaluation")
        mlflow.log_dict(test_metrics.to_dict(), "evaluation/test_metrics.json")
        LOGGER.info(
            "Completed MLflow run %s; best epoch %d; test metrics %s",
            active_run.info.run_id,
            result.best_epoch,
            test_metrics.to_dict(),
        )
    return checkpoint_path
