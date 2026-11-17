"""Transparent detection metrics and error counts for held-out evaluation."""

import hashlib
from dataclasses import dataclass
from typing import Any

DEFAULT_IOU_THRESHOLDS = tuple(value / 100 for value in range(50, 100, 5))


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    box: Box


@dataclass(frozen=True)
class EvaluationSample:
    sample_id: str
    ground_truths: tuple[Detection, ...]
    predictions: tuple[Detection, ...]


def intersection_over_union(left: Box, right: Box) -> float:
    """Return box IoU for xyxy boxes, including degenerate non-overlap."""
    width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = width * height
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _average_precision(true_positives: list[int], false_positives: list[int], total: int) -> float:
    if total == 0:
        return 0.0
    cumulative_true: list[int] = []
    cumulative_false: list[int] = []
    for true_positive, false_positive in zip(true_positives, false_positives, strict=True):
        cumulative_true.append(true_positive + (cumulative_true[-1] if cumulative_true else 0))
        cumulative_false.append(
            false_positive + (cumulative_false[-1] if cumulative_false else 0)
        )
    recalls = [value / total for value in cumulative_true]
    precisions = [
        true / (true + false)
        for true, false in zip(cumulative_true, cumulative_false, strict=True)
    ]
    return sum(
        max(
            (
                precision
                for precision, recall in zip(precisions, recalls, strict=True)
                if recall >= threshold
            ),
            default=0.0,
        )
        for threshold in (index / 100 for index in range(101))
    ) / 101


def _class_matches(
    samples: tuple[EvaluationSample, ...],
    class_id: int,
    iou_threshold: float,
) -> tuple[list[int], list[int], int]:
    candidates = sorted(
        (
            (prediction.confidence, sample_index, prediction)
            for sample_index, sample in enumerate(samples)
            for prediction in sample.predictions
            if prediction.class_id == class_id
        ),
        key=lambda item: (-item[0], item[1]),
    )
    matched: set[tuple[int, int]] = set()
    true_positives: list[int] = []
    false_positives: list[int] = []
    total_ground_truths = sum(
        truth.class_id == class_id for sample in samples for truth in sample.ground_truths
    )
    for _, sample_index, prediction in candidates:
        truths = samples[sample_index].ground_truths
        best_index, best_iou = max(
            (
                (index, intersection_over_union(prediction.box, truth.box))
                for index, truth in enumerate(truths)
                if truth.class_id == class_id and (sample_index, index) not in matched
            ),
            key=lambda item: item[1],
            default=(-1, 0.0),
        )
        if best_index >= 0 and best_iou >= iou_threshold:
            matched.add((sample_index, best_index))
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)
    return true_positives, false_positives, total_ground_truths


def _confusion_matrix(
    samples: tuple[EvaluationSample, ...],
    class_count: int,
    confidence_threshold: float,
    iou_threshold: float,
) -> tuple[list[list[int]], list[int], list[int]]:
    matrix = [[0 for _ in range(class_count + 1)] for _ in range(class_count + 1)]
    false_positives_by_sample: list[int] = []
    false_negatives_by_sample: list[int] = []
    for sample in samples:
        predictions = [
            item for item in sample.predictions if item.confidence >= confidence_threshold
        ]
        pairs = sorted(
            (
                (intersection_over_union(truth.box, prediction.box), truth_index, pred_index)
                for truth_index, truth in enumerate(sample.ground_truths)
                for pred_index, prediction in enumerate(predictions)
            ),
            reverse=True,
        )
        matched_truths: set[int] = set()
        matched_predictions: set[int] = set()
        for overlap, truth_index, pred_index in pairs:
            if overlap < iou_threshold:
                break
            if truth_index in matched_truths or pred_index in matched_predictions:
                continue
            truth = sample.ground_truths[truth_index]
            prediction = predictions[pred_index]
            matrix[truth.class_id][prediction.class_id] += 1
            matched_truths.add(truth_index)
            matched_predictions.add(pred_index)
        false_negatives = len(sample.ground_truths) - len(matched_truths)
        false_positives = len(predictions) - len(matched_predictions)
        for index, truth in enumerate(sample.ground_truths):
            if index not in matched_truths:
                matrix[truth.class_id][class_count] += 1
        for index, prediction in enumerate(predictions):
            if index not in matched_predictions:
                matrix[class_count][prediction.class_id] += 1
        false_positives_by_sample.append(false_positives)
        false_negatives_by_sample.append(false_negatives)
    return matrix, false_positives_by_sample, false_negatives_by_sample


def calculate_detection_metrics(
    samples: list[EvaluationSample],
    class_names: tuple[str, ...],
    *,
    confidence_threshold: float,
    iou_thresholds: tuple[float, ...] = DEFAULT_IOU_THRESHOLDS,
) -> dict[str, Any]:
    """Calculate AP@[.50:.95], AP50, operating-point P/R, confusion, and errors."""
    if not samples:
        raise ValueError("Cannot evaluate an empty detection split")
    if len(class_names) < 1 or len(set(class_names)) != len(class_names):
        raise ValueError("class_names must be non-empty and unique")
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between zero and one")
    if not iou_thresholds or any(not 0 < value <= 1 for value in iou_thresholds):
        raise ValueError("IoU thresholds must be non-empty and between zero and one")
    for sample in samples:
        for detection in (*sample.ground_truths, *sample.predictions):
            if detection.class_id < 0 or detection.class_id >= len(class_names):
                raise ValueError("Detection class id is outside class_names")
            if not 0 <= detection.confidence <= 1:
                raise ValueError("Detection confidence must be between zero and one")

    sample_tuple = tuple(samples)
    per_class: dict[str, dict[str, Any]] = {}
    all_ap50: list[float] = []
    all_ap: list[float] = []
    true_positives_total = 0
    false_positives_total = 0
    false_negatives_total = 0
    for class_id, class_name in enumerate(class_names):
        aps: list[float] = []
        for threshold in iou_thresholds:
            true, false, total = _class_matches(sample_tuple, class_id, threshold)
            aps.append(_average_precision(true, false, total))
        true, false, total = _class_matches(sample_tuple, class_id, 0.5)
        predictions = sorted(
            (
                item
                for sample in sample_tuple
                for item in sample.predictions
                if item.class_id == class_id
            ),
            key=lambda item: item.confidence,
            reverse=True,
        )
        selected = [
            (true_positive, false_positive)
            for true_positive, false_positive, prediction in zip(
                true, false, predictions, strict=True
            )
            if prediction.confidence >= confidence_threshold
        ]
        tp = sum(true_positive for true_positive, _ in selected)
        fp = sum(false_positive for _, false_positive in selected)
        fn = total - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total if total else 0.0
        cumulative_true = 0
        cumulative_false = 0
        precision_curve: list[float] = []
        recall_curve: list[float] = []
        confidence_curve: list[float] = []
        for true_positive, false_positive, prediction in zip(
            true,
            false,
            predictions,
            strict=True,
        ):
            cumulative_true += true_positive
            cumulative_false += false_positive
            confidence_curve.append(prediction.confidence)
            precision_curve.append(
                cumulative_true / (cumulative_true + cumulative_false)
                if cumulative_true + cumulative_false
                else 0.0
            )
            recall_curve.append(cumulative_true / total if total else 0.0)
        per_class[class_name] = {
            "ground_truth_objects": total,
            "precision": precision,
            "recall": recall,
            "average_precision_50": aps[0],
            "average_precision_50_95": sum(aps) / len(aps),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision_recall_curve": {
                "confidence_thresholds": confidence_curve,
                "precision": precision_curve,
                "recall": recall_curve,
            },
        }
        all_ap50.append(aps[0])
        all_ap.append(sum(aps) / len(aps))
        true_positives_total += tp
        false_positives_total += fp
        false_negatives_total += fn

    matrix, fp_counts, fn_counts = _confusion_matrix(
        sample_tuple,
        len(class_names),
        confidence_threshold,
        0.5,
    )
    total_predictions = true_positives_total + false_positives_total
    total_ground_truths = true_positives_total + false_negatives_total
    return {
        "sample_count": len(sample_tuple),
        "confidence_threshold": confidence_threshold,
        "iou_thresholds": list(iou_thresholds),
        "precision": (
            true_positives_total / total_predictions if total_predictions else 0.0
        ),
        "recall": (
            true_positives_total / total_ground_truths if total_ground_truths else 0.0
        ),
        "mAP50": sum(all_ap50) / len(all_ap50),
        "mAP50_95": sum(all_ap) / len(all_ap),
        "true_positives": true_positives_total,
        "false_positives": false_positives_total,
        "false_negatives": false_negatives_total,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": [*class_names, "background"],
            "rows_true_columns_predicted": matrix,
        },
        "error_analysis": {
            "false_positive_samples": [
                {"sample_id": sample_tuple[index].sample_id, "count": count}
                for index, count in sorted(
                    enumerate(fp_counts), key=lambda item: item[1], reverse=True
                )
                if count
            ],
            "false_negative_samples": [
                {"sample_id": sample_tuple[index].sample_id, "count": count}
                for index, count in sorted(
                    enumerate(fn_counts), key=lambda item: item[1], reverse=True
                )
                if count
            ],
        },
    }


def safe_sample_id(relative_path: str) -> str:
    """Create a stable non-reversible identifier without exposing source filenames."""
    return hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
