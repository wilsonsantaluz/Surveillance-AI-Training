"""Validated deployment manifest format for exported classifiers."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from surveillance_ml.core.image_preprocessing import IMAGENET_MEAN, IMAGENET_STD

type JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]


@dataclass(frozen=True)
class ModelManifest:
    schema_version: int
    model_file: str
    model_sha256: str
    checkpoint_sha256: str
    architecture: str
    class_names: tuple[str, ...]
    input_name: str
    output_name: str
    input_shape: tuple[int | str, ...]
    opset: int
    preprocessing: dict[str, JsonValue]
    dataset_fingerprint: str | None
    mlflow_run_id: str | None

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
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
            "output": {"name": self.output_name, "shape": ["batch", len(self.class_names)]},
            "opset": self.opset,
            "preprocessing": self.preprocessing,
            "dataset_fingerprint": self.dataset_fingerprint,
            "mlflow_run_id": self.mlflow_run_id,
        }


def sha256_file(path: Path) -> str:
    """Hash a file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> ModelManifest:
    """Load a manifest and reject unsupported or malformed deployment contracts."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Unable to read model manifest: {path}") from error
    if not isinstance(raw, dict):
        raise ValueError("Model manifest must be a JSON object")
    try:
        input_config = raw["input"]
        if not isinstance(input_config, dict):
            raise TypeError("input must be an object")
        output_config = raw["output"]
        if not isinstance(output_config, dict):
            raise TypeError("output must be an object")
        classes = raw["class_names"]
        shape = input_config["shape"]
        preprocessing = raw["preprocessing"]
        if not isinstance(classes, list) or not all(isinstance(item, str) for item in classes):
            raise TypeError("class_names must be a list of strings")
        if not isinstance(shape, list) or not all(isinstance(item, (int, str)) for item in shape):
            raise TypeError("input.shape must contain integers and dimension names")
        if not isinstance(preprocessing, dict):
            raise TypeError("preprocessing must be an object")
        if input_config.get("dtype") != "float32":
            raise TypeError("input.dtype must be float32")
        if output_config.get("shape") != ["batch", len(classes)]:
            raise TypeError("output.shape must match the dynamic batch and class count")
        manifest = ModelManifest(
            schema_version=raw["schema_version"],
            model_file=raw["model_file"],
            model_sha256=raw["model_sha256"],
            checkpoint_sha256=raw["checkpoint_sha256"],
            architecture=raw["architecture"],
            class_names=tuple(classes),
            input_name=input_config["name"],
            output_name=output_config["name"],
            input_shape=tuple(shape),
            opset=raw["opset"],
            preprocessing=preprocessing,
            dataset_fingerprint=raw.get("dataset_fingerprint"),
            mlflow_run_id=raw.get("mlflow_run_id"),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid model manifest fields: {error}") from error
    if (
        not isinstance(manifest.schema_version, int)
        or isinstance(manifest.schema_version, bool)
        or manifest.schema_version != 1
    ):
        raise ValueError(f"Unsupported manifest schema version: {manifest.schema_version}")
    if not isinstance(manifest.model_file, str) or (
        not manifest.model_file
        or Path(manifest.model_file).name != manifest.model_file
        or manifest.model_file in {".", ".."}
    ):
        raise ValueError("model_file must be a plain filename")
    if not all(
        isinstance(value, str)
        for value in (
            manifest.model_sha256,
            manifest.checkpoint_sha256,
            manifest.architecture,
            manifest.input_name,
            manifest.output_name,
        )
    ):
        raise ValueError("Manifest names, architecture, and digests must be strings")
    if (
        len(manifest.class_names) < 2
        or len(set(manifest.class_names)) != len(manifest.class_names)
        or any(not name for name in manifest.class_names)
    ):
        raise ValueError("class_names must contain unique, non-empty class names")
    if len(manifest.input_shape) != 4:
        raise ValueError("input.shape must have the form [dynamic_batch, 3, height, width]")
    if manifest.input_shape[0] != "batch":
        raise ValueError("input.shape must use a dynamic 'batch' dimension")
    channels, height, width = manifest.input_shape[1:]
    if (
        not isinstance(channels, int)
        or not isinstance(height, int)
        or not isinstance(width, int)
        or channels != 3
    ):
        raise ValueError("input.shape must have the form [dynamic_batch, 3, height, width]")
    if height < 32 or width < 32 or height > 4096 or width > 4096:
        raise ValueError("Input image dimensions must be between 32 and 4096 pixels")
    if (
        manifest.architecture != "resnet18"
        or not isinstance(manifest.opset, int)
        or isinstance(manifest.opset, bool)
        or manifest.opset < 1
    ):
        raise ValueError("Unsupported architecture or ONNX opset")
    hashes = (manifest.model_sha256, manifest.checkpoint_sha256)
    if any(
        len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
        for digest in hashes
    ):
        raise ValueError("Model and checkpoint hashes must be SHA-256 hex strings")
    expected_preprocessing: dict[str, JsonValue] = {
        "color_space": "RGB",
        "layout": "NCHW",
        "resize_mode": "fit",
        "resample": "bilinear",
        "normalization": "imagenet",
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "width": width,
        "height": height,
    }
    if manifest.preprocessing != expected_preprocessing:
        raise ValueError("Unsupported or inconsistent image preprocessing contract")
    return manifest


def verify_model_hash(model_path: Path, manifest: ModelManifest) -> None:
    """Reject an ONNX file that does not match the manifest's recorded digest."""
    if model_path.name != manifest.model_file:
        raise ValueError("Model filename does not match the deployment manifest")
    if sha256_file(model_path) != manifest.model_sha256:
        raise ValueError(f"ONNX model hash does not match its manifest: {model_path}")
