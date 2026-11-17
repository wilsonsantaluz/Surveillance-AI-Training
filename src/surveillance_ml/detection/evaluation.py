"""YOLO prediction and held-out detection evaluation."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from surveillance_ml.detection import DETECTION_CLASS_NAMES
from surveillance_ml.detection.dataset import DetectionDatasetReport, DetectionRecord
from surveillance_ml.detection.metrics import (
    Box,
    Detection,
    EvaluationSample,
    calculate_detection_metrics,
    safe_sample_id,
)


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _read_ground_truths(
    record: DetectionRecord,
    image_width: int,
    image_height: int,
) -> tuple[Detection, ...]:
    detections: list[Detection] = []
    for line in record.label_path.read_text(encoding="utf-8").splitlines():
        class_id_text, x_text, y_text, width_text, height_text = line.split()
        x_center, y_center, width, height = (
            float(x_text),
            float(y_text),
            float(width_text),
            float(height_text),
        )
        detections.append(
            Detection(
                class_id=int(class_id_text),
                confidence=1.0,
                box=Box(
                    (x_center - width / 2) * image_width,
                    (y_center - height / 2) * image_height,
                    (x_center + width / 2) * image_width,
                    (y_center + height / 2) * image_height,
                ),
            )
        )
    return tuple(detections)


def _predict_image(
    model: YOLO,
    image_path: Path,
    *,
    image_size: int,
    confidence_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    device: str,
) -> tuple[Detection, ...]:
    results = cast(
        list[Results],
        model.predict(
        source=str(image_path),
        imgsz=image_size,
        conf=confidence_threshold,
        iou=nms_iou_threshold,
        max_det=max_detections,
        device=device,
        verbose=False,
        save=False,
        stream=False,
        ),
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one prediction for {image_path}, received {len(results)}")
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return ()
    coordinates = _to_numpy(boxes.xyxy)
    confidences = _to_numpy(boxes.conf)
    class_ids = _to_numpy(boxes.cls)
    return tuple(
        Detection(
            class_id=int(class_id),
            confidence=float(confidence),
            box=Box(*(float(value) for value in coordinate)),
        )
        for coordinate, confidence, class_id in zip(
            coordinates, confidences, class_ids, strict=True
        )
    )


def evaluate_detection_checkpoint(
    checkpoint_path: Path,
    dataset_report: DetectionDatasetReport,
    *,
    split: str,
    image_size: int,
    confidence_threshold: float,
    operating_confidence_threshold: float,
    nms_iou_threshold: float,
    max_detections: int,
    device: str,
) -> dict[str, Any]:
    """Evaluate one checkpoint on exactly one validated dataset partition."""
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be train, val, or test")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Detector checkpoint does not exist: {checkpoint_path}")
    model = YOLO(str(checkpoint_path))
    samples: list[EvaluationSample] = []
    for record in dataset_report.records:
        if record.split != split:
            continue
        with Image.open(record.image_path) as image:
            image_width, image_height = image.size
        predictions = _predict_image(
            model,
            record.image_path,
            image_size=image_size,
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_detections=max_detections,
            device=device,
        )
        relative = record.image_path.relative_to(dataset_report.root / "images" / split)
        samples.append(
            EvaluationSample(
                sample_id=safe_sample_id(f"{split}/{relative.as_posix()}"),
                ground_truths=_read_ground_truths(record, image_width, image_height),
                predictions=predictions,
            )
        )
    metrics = calculate_detection_metrics(
        samples,
        DETECTION_CLASS_NAMES,
        confidence_threshold=operating_confidence_threshold,
    )
    metrics["dataset_fingerprint"] = dataset_report.dataset_fingerprint
    metrics["checkpoint"] = checkpoint_path.name
    metrics["split"] = split
    metrics["image_size"] = image_size
    metrics["prediction_confidence_threshold"] = confidence_threshold
    metrics["operating_confidence_threshold"] = operating_confidence_threshold
    metrics["nms_iou_threshold"] = nms_iou_threshold
    return metrics


def write_metrics_report(metrics: dict[str, Any], output_path: Path) -> None:
    """Persist evaluation data as plain JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
