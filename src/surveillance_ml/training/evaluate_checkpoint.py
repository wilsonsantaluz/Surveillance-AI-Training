"""Evaluate one persisted best checkpoint on the fixed held-out test split."""

import json
from pathlib import Path

from torch.utils.data import DataLoader

from surveillance_ml.core.config import AppConfig
from surveillance_ml.core.device import seed_everything, select_device
from surveillance_ml.core.reproducibility import (
    configuration_fingerprint,
    dataset_fingerprint,
    split_fingerprint,
)
from surveillance_ml.data.dataset import CLASS_NAMES, SceneImageDataset, discover_images
from surveillance_ml.data.splitting import split_records
from surveillance_ml.models.classifier import build_resnet18
from surveillance_ml.training.evaluate import evaluate_model
from surveillance_ml.training.trainer import load_best_checkpoint


def evaluate_checkpoint(
    config: AppConfig,
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Evaluate the test split and write a machine-readable report."""
    seed_everything(config.training.seed)
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
    test_loader = DataLoader(
        SceneImageDataset(splits.test, config.dataset.image_size),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    model = build_resnet18(
        len(CLASS_NAMES),
        strategy=config.model.strategy,
        pretrained=False,
    )
    checkpoint = load_best_checkpoint(model, checkpoint_path, device)
    checkpoint_classes = checkpoint.get("class_names")
    if checkpoint_classes != list(CLASS_NAMES):
        raise ValueError("Checkpoint class labels do not match the dataset class schema")
    model.to(device)
    metrics = evaluate_model(model, test_loader, device, len(CLASS_NAMES))
    data_hash = dataset_fingerprint(config.dataset.root)
    split_hash = split_fingerprint(
        {"train": splits.train, "validation": splits.validation, "test": splits.test},
        config.dataset.root,
    )
    report: dict[str, object] = {
        "dataset_version": config.dataset.version,
        "dataset_fingerprint": data_hash,
        "split_fingerprint": split_hash,
        "configuration_fingerprint": configuration_fingerprint(config),
        "seed": config.training.seed,
        "class_names": list(CLASS_NAMES),
        "test_samples": len(splits.test),
        "test_metrics": metrics.to_dict(),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "device": str(device),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
