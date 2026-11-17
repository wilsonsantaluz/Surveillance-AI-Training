"""Evaluation helpers for validation and held-out test datasets."""

import torch
from torch import nn
from torch.utils.data import DataLoader

from surveillance_ml.training.metrics import ClassificationMetrics, classification_metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    num_classes: int,
) -> ClassificationMetrics:
    """Evaluate one full dataset pass and return sample-weighted loss and metrics."""
    model.eval()
    loss_function = nn.CrossEntropyLoss(reduction="sum")
    targets: list[int] = []
    predictions: list[int] = []
    total_loss = 0.0
    total = 0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        total_loss += float(loss_function(logits, labels).item())
        total += labels.numel()
        targets.extend(labels.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())
    if total == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    return classification_metrics(targets, predictions, total_loss / total, num_classes)


def metric_values(metrics: ClassificationMetrics, prefix: str = "") -> dict[str, float]:
    """Flatten scalar classification metrics for MLflow logging."""
    return {
        f"{prefix}{name}": float(value)
        for name, value in (
            ("loss", metrics.loss),
            ("accuracy", metrics.accuracy),
            ("precision", metrics.precision),
            ("recall", metrics.recall),
            ("f1", metrics.f1),
        )
    }
