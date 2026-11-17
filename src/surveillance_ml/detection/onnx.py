"""YOLO ONNX export, contract validation, preprocessing, decoding, and parity."""

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

from surveillance_ml.detection import DETECTION_CLASS_NAMES
from surveillance_ml.detection.manifest import (
    DetectionManifest,
    load_detection_manifest,
    sha256_file,
    verify_detection_model_hash,
)
from surveillance_ml.detection.metrics import Box, intersection_over_union


def _to_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


@dataclass(frozen=True)
class DetectionPrediction:
    class_id: int
    class_name: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

    @property
    def xyxy(self) -> Box:
        return Box(self.x, self.y, self.x + self.width, self.y + self.height)


@dataclass(frozen=True)
class DetectionInference:
    detections: tuple[DetectionPrediction, ...]
    elapsed_seconds: float
    image_width: int
    image_height: int


def _letterbox_rgb(
    image: np.ndarray,
    target_height: int,
    target_width: int,
) -> tuple[np.ndarray, float, int, int]:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected an HWC uint8 RGB image")
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("Image dimensions must be positive")
    scale = min(target_height / height, target_width / width)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))
    pad_width = target_width - resized_width
    pad_height = target_height - resized_height
    left = int(round(pad_width / 2 - 0.1))
    right = int(round(pad_width / 2 + 0.1))
    top = int(round(pad_height / 2 - 0.1))
    bottom = int(round(pad_height / 2 + 0.1))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR_EXACT,
    )
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, scale, left, top


def _prepare_rgb(image: np.ndarray, height: int, width: int) -> tuple[np.ndarray, float, int, int]:
    letterboxed, scale, left, top = _letterbox_rgb(image, height, width)
    tensor = letterboxed.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(tensor), scale, left, top


def _nms(
    candidates: list[DetectionPrediction],
    iou_threshold: float,
    max_detections: int,
) -> tuple[DetectionPrediction, ...]:
    remaining = sorted(candidates, key=lambda item: item.confidence, reverse=True)
    retained: list[DetectionPrediction] = []
    while remaining and len(retained) < max_detections:
        best = remaining.pop(0)
        retained.append(best)
        remaining = [
            candidate
            for candidate in remaining
            if candidate.class_id != best.class_id
            or intersection_over_union(best.xyxy, candidate.xyxy) <= iou_threshold
        ]
    return tuple(retained)


def _decode_output(
    output: np.ndarray,
    manifest: DetectionManifest,
    *,
    scale: float,
    pad_left: int,
    pad_top: int,
    image_width: int,
    image_height: int,
) -> tuple[DetectionPrediction, ...]:
    if output.shape != manifest.output_shape or not np.isfinite(output).all():
        raise RuntimeError(f"Unexpected or non-finite YOLO output shape: {output.shape}")
    rows = output[0].T
    threshold = float(manifest.postprocessing["confidence_threshold"])
    candidates: list[DetectionPrediction] = []
    for class_id, class_name in enumerate(manifest.class_names):
        scores = rows[:, 4 + class_id]
        for candidate_index in np.flatnonzero(scores >= threshold):
            center_x, center_y, box_width, box_height = (
                float(value) for value in rows[candidate_index, :4]
            )
            confidence = float(scores[candidate_index])
            if box_width <= 0 or box_height <= 0:
                continue
            x1 = (center_x - box_width / 2 - pad_left) / scale
            y1 = (center_y - box_height / 2 - pad_top) / scale
            x2 = (center_x + box_width / 2 - pad_left) / scale
            y2 = (center_y + box_height / 2 - pad_top) / scale
            x1 = min(max(x1, 0.0), float(image_width))
            y1 = min(max(y1, 0.0), float(image_height))
            x2 = min(max(x2, 0.0), float(image_width))
            y2 = min(max(y2, 0.0), float(image_height))
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append(
                DetectionPrediction(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                )
            )
    return _nms(
        candidates,
        float(manifest.postprocessing["iou_threshold"]),
        int(manifest.postprocessing["max_detections"]),
    )


class OnnxDetector:
    """CPU ONNX Runtime for a manifest-verified single-image YOLO detector."""

    def __init__(
        self,
        model_path: Path,
        manifest_path: Path,
        *,
        intra_op_threads: int = 0,
    ) -> None:
        if intra_op_threads < 0:
            raise ValueError("intra_op_threads must be non-negative")
        self.model_path = model_path
        self.manifest = load_detection_manifest(manifest_path)
        verify_detection_model_hash(model_path, self.manifest)
        options = ort.SessionOptions()
        if intra_op_threads:
            options.intra_op_num_threads = intra_op_threads
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._validate_runtime_contract()

    def _validate_runtime_contract(self) -> None:
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("ONNX detector must have exactly one input and one output")
        model_input, model_output = inputs[0], outputs[0]
        if model_input.name != self.manifest.input_name or model_input.type != "tensor(float)":
            raise ValueError("ONNX input does not match the detection manifest")
        if model_output.name != self.manifest.output_name or model_output.type != "tensor(float)":
            raise ValueError("ONNX output does not match the detection manifest")
        if tuple(model_input.shape) != self.manifest.input_shape:
            raise ValueError("ONNX input shape does not match the detection manifest")
        if tuple(model_output.shape) != self.manifest.output_shape:
            raise ValueError("ONNX output shape does not match the detection manifest")

    def predict_rgb(self, image: np.ndarray) -> DetectionInference:
        """Preprocess one RGB uint8 frame, infer, decode, and restore source coordinates."""
        height, width = image.shape[:2] if image.ndim >= 2 else (0, 0)
        input_height, input_width = self.manifest.input_shape[2:]
        batch, scale, left, top = _prepare_rgb(image, input_height, input_width)
        started = time.perf_counter()
        outputs = self._session.run(
            [self.manifest.output_name],
            {self.manifest.input_name: batch},
        )
        elapsed = time.perf_counter() - started
        detections = _decode_output(
            outputs[0],
            self.manifest,
            scale=scale,
            pad_left=left,
            pad_top=top,
            image_width=width,
            image_height=height,
        )
        return DetectionInference(detections, elapsed, width, height)

    def predict_image(self, image_path: Path) -> DetectionInference:
        """Read an image as RGB and run ONNX inference."""
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return self.predict_rgb(rgb)


def _model_output(model: YOLO, tensor: torch.Tensor) -> np.ndarray:
    torch_model = model.model
    if not isinstance(torch_model, torch.nn.Module):
        raise RuntimeError("YOLO checkpoint does not contain a loaded PyTorch model")
    with torch.inference_mode():
        output: Any = torch_model(tensor)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise RuntimeError("PyTorch detector returned an unsupported output")
    return output.cpu().numpy()


def _reference_predictions(
    model: YOLO,
    image: np.ndarray,
    manifest: DetectionManifest,
) -> tuple[DetectionPrediction, ...]:
    results = cast(
        list[Results],
        model.predict(
            source=np.ascontiguousarray(image[:, :, ::-1]),
            imgsz=manifest.input_shape[2],
            conf=float(manifest.postprocessing["confidence_threshold"]),
            iou=float(manifest.postprocessing["iou_threshold"]),
            max_det=int(manifest.postprocessing["max_detections"]),
            device="cpu",
            verbose=False,
            save=False,
            stream=False,
            rect=False,
        ),
    )
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return ()
    return tuple(
        DetectionPrediction(
            int(class_id),
            manifest.class_names[int(class_id)],
            float(confidence),
            float(x1),
            float(y1),
            float(x2 - x1),
            float(y2 - y1),
        )
        for (x1, y1, x2, y2), confidence, class_id in zip(
            _to_numpy(boxes.xyxy),
            _to_numpy(boxes.conf),
            _to_numpy(boxes.cls),
            strict=True,
        )
    )


def _compare_detections(
    expected: tuple[DetectionPrediction, ...],
    actual: tuple[DetectionPrediction, ...],
) -> dict[str, Any]:
    unmatched = list(actual)
    ious: list[float] = []
    confidence_differences: list[float] = []
    for reference in expected:
        options = [
            (index, item, intersection_over_union(reference.xyxy, item.xyxy))
            for index, item in enumerate(unmatched)
            if item.class_id == reference.class_id
        ]
        if not options:
            continue
        index, candidate, overlap = max(options, key=lambda item: item[2])
        unmatched.pop(index)
        ious.append(overlap)
        confidence_differences.append(abs(reference.confidence - candidate.confidence))
    matched = len(ious)
    passed = (
        matched == len(expected) == len(actual)
        and all(value >= 0.99 for value in ious)
        and all(value <= 1e-3 for value in confidence_differences)
    )
    return {
        "passed": passed,
        "expected_detections": len(expected),
        "actual_detections": len(actual),
        "matched_detections": matched,
        "minimum_matched_iou": min(ious, default=1.0),
        "maximum_confidence_difference": max(confidence_differences, default=0.0),
    }


def export_detector(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    model_id: str,
    model_version: str,
    image_size: int,
    opset: int = 17,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    max_detections: int = 300,
    dataset_fingerprint: str | None = None,
    mlflow_run_id: str | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Export a YOLO checkpoint, verify its graph, then require raw and box parity."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Detector checkpoint does not exist: {checkpoint_path}")
    if image_size < 64 or image_size % 32:
        raise ValueError("image_size must be at least 64 and divisible by 32")
    if opset < 17:
        raise ValueError("ONNX opset must be at least 17")
    if not 0 <= confidence_threshold <= 1 or not 0 <= iou_threshold <= 1:
        raise ValueError("Confidence and IoU thresholds must be between zero and one")
    if max_detections < 1:
        raise ValueError("max_detections must be positive")
    model_path = output_dir / "model.onnx"
    manifest_path = output_dir / "manifest.json"
    if model_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite an existing detection bundle: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    temporary_model = output_dir / ".model.onnx.tmp"
    temporary_manifest = output_dir / ".manifest.json.tmp"
    with tempfile.TemporaryDirectory(prefix=".yolo-export-", dir=output_dir) as temporary_dir:
        export_checkpoint = Path(temporary_dir) / checkpoint_path.name
        shutil.copyfile(checkpoint_path, export_checkpoint)
        model = YOLO(str(export_checkpoint))
        names = tuple(model.names[index] for index in range(len(model.names)))
        if names != DETECTION_CLASS_NAMES:
            raise ValueError(f"Checkpoint classes must be ordered as {DETECTION_CLASS_NAMES}")
        exported_path = Path(
            model.export(
                format="onnx",
                imgsz=image_size,
                batch=1,
                dynamic=False,
                nms=False,
                opset=opset,
                simplify=False,
                device="cpu",
                verbose=False,
            )
        )
        if not exported_path.is_file():
            raise FileNotFoundError(f"Ultralytics did not write the ONNX export: {exported_path}")
        graph = onnx.load(str(exported_path))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])
        if session.get_providers()[0] != "CPUExecutionProvider":
            raise RuntimeError("ONNX Runtime did not initialize CPUExecutionProvider")
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Exported detector must have exactly one input and one output")
        input_name, output_name = inputs[0].name, outputs[0].name
        input_shape = tuple(int(value) for value in inputs[0].shape)
        output_shape = tuple(int(value) for value in outputs[0].shape)
        expected_output_channels = 4 + len(DETECTION_CLASS_NAMES)
        if (
            input_shape != (1, 3, image_size, image_size)
            or len(output_shape) != 3
            or output_shape[0] != 1
            or output_shape[1] != expected_output_channels
        ):
            raise ValueError(
                f"Exported graph has unsupported shapes: input={input_shape}, "
                f"output={output_shape}"
            )
        shutil.copyfile(exported_path, temporary_model)
        del session

    preprocessing = {
        "color_space": "RGB",
        "layout": "NCHW",
        "resize_mode": "letterbox",
        "resample": "opencv_inter_linear_exact",
        "padding_value": 114,
        "normalization": "divide_by_255",
        "width": image_size,
        "height": image_size,
    }
    output_contract = {
        "layout": "batch_channels_candidates",
        "box_format": "cxcywh_pixels_in_letterboxed_input",
        "class_scores": "sigmoid_probabilities_without_objectness",
    }
    postprocessing = {
        "nms": "class_aware_greedy",
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "max_detections": max_detections,
    }
    provenance_path = checkpoint_path.parent.parent / "provenance.json"
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if isinstance(provenance, dict):
            dataset_fingerprint = dataset_fingerprint or provenance.get("dataset_fingerprint")
            mlflow_run_id = mlflow_run_id or provenance.get("mlflow_run_id")

    torch_model = model.model
    if not isinstance(torch_model, torch.nn.Module):
        raise RuntimeError("YOLO checkpoint does not contain a loaded PyTorch model")
    model_yaml = getattr(torch_model, "yaml", None)
    if not isinstance(model_yaml, dict):
        raise RuntimeError("YOLO checkpoint model does not expose its architecture config")
    scale = model_yaml.get("scale") or "n"
    architecture = f"yolo11{scale}"
    if architecture not in {"yolo11n", "yolo11s"}:
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    manifest = DetectionManifest(
        schema_version=1,
        model_id=model_id,
        model_version=model_version,
        model_file=model_path.name,
        model_sha256=sha256_file(temporary_model),
        checkpoint_sha256=sha256_file(checkpoint_path),
        architecture=architecture,
        class_names=names,
        input_name=input_name,
        input_shape=input_shape,
        output_name=output_name,
        output_shape=output_shape,
        opset=opset,
        preprocessing=preprocessing,
        output_contract=output_contract,
        postprocessing=postprocessing,
        dataset_fingerprint=(
            dataset_fingerprint if isinstance(dataset_fingerprint, str) else None
        ),
        mlflow_run_id=mlflow_run_id if isinstance(mlflow_run_id, str) else None,
        source_license="AGPL-3.0",
    )
    temporary_manifest.write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    try:
        os.replace(temporary_model, model_path)
        os.replace(temporary_manifest, manifest_path)
        detector = OnnxDetector(model_path, manifest_path)
        torch_model.eval()
        rng = np.random.default_rng(2027)
        parity_cases: list[dict[str, Any]] = []
        for shape in (
            (max(32, image_size // 2), image_size + 16, 3),
            (image_size + 16, max(32, image_size // 2), 3),
        ):
            image = rng.integers(0, 256, size=shape, dtype=np.uint8)
            batch, _, _, _ = _prepare_rgb(image, image_size, image_size)
            torch_output = _model_output(model, torch.from_numpy(batch))
            onnx_output = detector._session.run(
                [manifest.output_name],
                {manifest.input_name: batch},
            )[0]
            if torch_output.shape != onnx_output.shape or not np.allclose(
                torch_output,
                onnx_output,
                rtol=1e-3,
                atol=1e-4,
            ):
                raise RuntimeError("PyTorch/ONNX raw detector output parity check failed")
            runtime_predictions = detector.predict_rgb(image).detections
            reference_predictions = _reference_predictions(model, image, manifest)
            parity = _compare_detections(reference_predictions, runtime_predictions)
            if not parity["passed"]:
                raise RuntimeError(f"PyTorch/ONNX detection parity check failed: {parity}")
            parity_cases.append(
                {
                    "raw_output_max_abs_error": float(
                        np.max(np.abs(torch_output - onnx_output), initial=0.0)
                    ),
                    **parity,
                }
            )
    except Exception:
        model_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    finally:
        temporary_model.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return model_path, manifest_path, {"passed": True, "cases": parity_cases}
