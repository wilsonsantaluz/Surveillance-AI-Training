"""Versioned ONNX contract for object detectors, independent of classifier bundles."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from surveillance_ml.detection import DETECTION_CLASS_NAMES


@dataclass(frozen=True)
class DetectionManifest:
    schema_version: int
    model_id: str
    model_version: str
    model_file: str
    model_sha256: str
    checkpoint_sha256: str
    architecture: str
    class_names: tuple[str, ...]
    input_name: str
    input_shape: tuple[int, int, int, int]
    output_name: str
    output_shape: tuple[int, int, int]
    opset: int
    preprocessing: dict[str, Any]
    output_contract: dict[str, Any]
    postprocessing: dict[str, Any]
    dataset_fingerprint: str | None
    mlflow_run_id: str | None
    source_license: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "model_file": self.model_file,
            "model_sha256": self.model_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "architecture": self.architecture,
            "class_names": list(self.class_names),
            "input": {
                "name": self.input_name,
                "shape": list(self.input_shape),
                "dtype": "float32",
            },
            "output": {
                "name": self.output_name,
                "shape": list(self.output_shape),
                "dtype": "float32",
            },
            "opset": self.opset,
            "preprocessing": self.preprocessing,
            "output_contract": self.output_contract,
            "postprocessing": self.postprocessing,
            "dataset_fingerprint": self.dataset_fingerprint,
            "mlflow_run_id": self.mlflow_run_id,
            "source_license": self.source_license,
        }


def sha256_file(path: Path) -> str:
    """Hash a model or checkpoint in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_detection_manifest(path: Path) -> DetectionManifest:
    """Load and validate the supported single-image YOLO ONNX contract."""
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read detection manifest: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError("Detection manifest must be a JSON object")
    try:
        input_config = raw["input"]
        output_config = raw["output"]
        names = raw["class_names"]
        if not isinstance(input_config, dict) or not isinstance(output_config, dict):
            raise TypeError("input and output must be objects")
        if not isinstance(names, list) or not all(isinstance(item, str) for item in names):
            raise TypeError("class_names must be a list of strings")
        manifest = DetectionManifest(
            schema_version=raw["schema_version"],
            model_id=raw["model_id"],
            model_version=raw["model_version"],
            model_file=raw["model_file"],
            model_sha256=raw["model_sha256"],
            checkpoint_sha256=raw["checkpoint_sha256"],
            architecture=raw["architecture"],
            class_names=tuple(names),
            input_name=input_config["name"],
            input_shape=tuple(input_config["shape"]),
            output_name=output_config["name"],
            output_shape=tuple(output_config["shape"]),
            opset=raw["opset"],
            preprocessing=raw["preprocessing"],
            output_contract=raw["output_contract"],
            postprocessing=raw["postprocessing"],
            dataset_fingerprint=raw.get("dataset_fingerprint"),
            mlflow_run_id=raw.get("mlflow_run_id"),
            source_license=raw["source_license"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid detection manifest fields: {error}") from error

    if not all(
        isinstance(value, str)
        for value in (
            manifest.model_id,
            manifest.model_version,
            manifest.model_file,
            manifest.model_sha256,
            manifest.checkpoint_sha256,
            manifest.architecture,
            manifest.input_name,
            manifest.output_name,
            manifest.source_license,
        )
    ):
        raise ValueError("Detection manifest identifiers and names must be strings")
    if manifest.dataset_fingerprint is not None and not isinstance(
        manifest.dataset_fingerprint, str
    ):
        raise ValueError("dataset_fingerprint must be a string or null")
    if manifest.mlflow_run_id is not None and not isinstance(manifest.mlflow_run_id, str):
        raise ValueError("mlflow_run_id must be a string or null")
    if (
        not isinstance(manifest.schema_version, int)
        or isinstance(manifest.schema_version, bool)
        or manifest.schema_version != 1
    ):
        raise ValueError(f"Unsupported detection manifest schema: {manifest.schema_version}")
    if manifest.architecture not in {"yolo11n", "yolo11s"}:
        raise ValueError(f"Unsupported detector architecture: {manifest.architecture}")
    if not manifest.model_id or not manifest.model_version:
        raise ValueError("model_id and model_version must not be empty")
    if Path(manifest.model_file).name != manifest.model_file or manifest.model_file in {".", ".."}:
        raise ValueError("model_file must be a plain filename")
    if (
        manifest.class_names != DETECTION_CLASS_NAMES
        or len(manifest.class_names) != 2
        or len(set(manifest.class_names)) != 2
        or any(not name for name in manifest.class_names)
    ):
        raise ValueError("class_names must contain exactly two unique, non-empty values")
    if (
        len(manifest.input_shape) != 4
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in manifest.input_shape
        )
        or manifest.input_shape[0] != 1
        or manifest.input_shape[1] != 3
        or manifest.input_shape[2] < 64
        or manifest.input_shape[3] < 64
        or manifest.input_shape[2] % 32
        or manifest.input_shape[3] % 32
    ):
        raise ValueError("input shape must be [1, 3, H, W] with H and W multiples of 32")
    if (
        len(manifest.output_shape) != 3
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in manifest.output_shape
        )
        or manifest.output_shape[0] != 1
        or manifest.output_shape[1] != 4 + len(manifest.class_names)
        or manifest.output_shape[2] < 1
    ):
        raise ValueError("output shape must be [1, 4 + class_count, candidates]")
    if (
        not isinstance(manifest.opset, int)
        or isinstance(manifest.opset, bool)
        or manifest.opset < 17
    ):
        raise ValueError("ONNX opset must be at least 17")
    for digest in (manifest.model_sha256, manifest.checkpoint_sha256):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("Model and checkpoint hashes must be SHA-256 hex strings")
    expected_preprocessing = {
        "color_space": "RGB",
        "layout": "NCHW",
        "resize_mode": "letterbox",
        "resample": "opencv_inter_linear_exact",
        "padding_value": 114,
        "normalization": "divide_by_255",
        "width": manifest.input_shape[3],
        "height": manifest.input_shape[2],
    }
    if (
        not isinstance(manifest.preprocessing, dict)
        or manifest.preprocessing != expected_preprocessing
    ):
        raise ValueError("Unsupported or inconsistent detection preprocessing contract")
    if not isinstance(manifest.output_contract, dict) or manifest.output_contract != {
        "layout": "batch_channels_candidates",
        "box_format": "cxcywh_pixels_in_letterboxed_input",
        "class_scores": "sigmoid_probabilities_without_objectness",
    }:
        raise ValueError("Unsupported raw detector output contract")
    postprocessing = manifest.postprocessing
    if not isinstance(postprocessing, dict) or postprocessing.get("nms") != "class_aware_greedy":
        raise ValueError("Unsupported detector postprocessing contract")
    for key in ("confidence_threshold", "iou_threshold"):
        value = postprocessing.get(key)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Invalid postprocessing {key}")
    maximum = postprocessing.get("max_detections")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
        raise ValueError("Invalid postprocessing max_detections")
    if manifest.source_license != "AGPL-3.0":
        raise ValueError("Unsupported or missing detector source license notice")
    return manifest


def verify_detection_model_hash(model_path: Path, manifest: DetectionManifest) -> None:
    """Reject model files whose bytes do not match the manifest."""
    if model_path.name != manifest.model_file:
        raise ValueError("ONNX model filename does not match the detection manifest")
    if sha256_file(model_path) != manifest.model_sha256:
        raise ValueError(f"Detection ONNX model hash does not match manifest: {model_path}")
