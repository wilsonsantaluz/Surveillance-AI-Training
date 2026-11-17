"""Classification metrics computed without external metric services."""

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ClassificationMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def classification_metrics(
    targets: list[int],
    predictions: list[int],
    average_loss: float,
    num_classes: int,
) -> ClassificationMetrics:
    """Compute accuracy, macro precision/recall/F1, and a fixed-size confusion matrix."""
    if len(targets) != len(predictions) or not targets:
        raise ValueError("Targets and predictions must have the same non-zero length")
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for target, prediction in zip(targets, predictions, strict=True):
        if not 0 <= target < num_classes or not 0 <= prediction < num_classes:
            raise ValueError("Class index is outside the configured class range")
        matrix[target, prediction] += 1

    precisions: list[float] = []
    recalls: list[float] = []
    f1_scores: list[float] = []
    for class_index in range(num_classes):
        true_positive = int(matrix[class_index, class_index])
        predicted_positive = int(matrix[:, class_index].sum())
        actual_positive = int(matrix[class_index, :].sum())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / actual_positive if actual_positive else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    accuracy = sum(
        target == prediction for target, prediction in zip(targets, predictions, strict=True)
    ) / len(targets)
    return ClassificationMetrics(
        loss=average_loss,
        accuracy=accuracy,
        precision=sum(precisions) / num_classes,
        recall=sum(recalls) / num_classes,
        f1=sum(f1_scores) / num_classes,
        confusion_matrix=matrix.tolist(),
    )
