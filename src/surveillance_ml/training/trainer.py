"""Epoch-based supervised training with early stopping and best-checkpoint saving."""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from surveillance_ml.training.evaluate import evaluate_model, metric_values
from surveillance_ml.training.metrics import ClassificationMetrics


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_loss: float
    best_validation_metrics: ClassificationMetrics
    elapsed_seconds: float
    history: list[dict[str, float]]


def _build_optimizer(
    name: str,
    parameters: list[nn.Parameter],
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            parameters, lr=learning_rate, weight_decay=weight_decay, momentum=0.9
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader[tuple[torch.Tensor, int]],
    validation_loader: DataLoader[tuple[torch.Tensor, int]],
    device: torch.device,
    output_dir: Path,
    class_names: tuple[str, ...],
    epochs: int,
    learning_rate: float,
    optimizer_name: str,
    weight_decay: float,
    scheduler_factor: float,
    scheduler_patience: int,
    early_stopping_patience: int,
    mixed_precision: bool,
    epoch_callback: Callable[[int, dict[str, float]], None] | None = None,
) -> TrainingResult:
    """Train, validate every epoch, and persist the lowest-validation-loss weights."""
    if epochs < 1 or early_stopping_patience < 1:
        raise ValueError("epochs and early_stopping_patience must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("The selected model strategy left no trainable parameters")
    optimizer = _build_optimizer(optimizer_name, parameters, learning_rate, weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )
    use_amp = mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_function = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_epoch = 0
    best_validation: ClassificationMetrics | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    checkpoint_path = output_dir / "best.pt"
    started_at = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = loss_function(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * labels.numel()
            total += labels.numel()
        if total == 0:
            raise ValueError("Cannot train on an empty dataset")
        training_loss = total_loss / total
        validation = evaluate_model(model, validation_loader, device, len(class_names))
        scheduler.step(validation.loss)
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        values = {
            "train_loss": training_loss,
            **metric_values(validation, "validation_"),
            "learning_rate": current_learning_rate,
        }
        history.append(values)
        if validation.loss < best_loss:
            best_loss = validation.loss
            best_epoch = epoch
            best_validation = validation
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_loss": validation.loss,
                    "class_names": list(class_names),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                if epoch_callback is not None:
                    epoch_callback(epoch, values)
                break
        if epoch_callback is not None:
            epoch_callback(epoch, values)

    elapsed = time.perf_counter() - started_at
    if best_validation is None:
        raise RuntimeError("Training completed without a best validation checkpoint")
    return TrainingResult(
        best_epoch=best_epoch,
        best_validation_loss=best_loss,
        best_validation_metrics=best_validation,
        elapsed_seconds=elapsed,
        history=history,
    )


def load_best_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict[str, object]:
    """Load a project-generated checkpoint and update a model in place."""
    payload: dict[str, object] = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError(f"Invalid checkpoint structure: {checkpoint_path}")
    model.load_state_dict(state_dict)
    return payload
