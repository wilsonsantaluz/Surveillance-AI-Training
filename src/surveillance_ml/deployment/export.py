"""Export and validate a ResNet18 checkpoint as an ONNX deployment bundle."""

import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from surveillance_ml.core.image_preprocessing import IMAGENET_MEAN, IMAGENET_STD
from surveillance_ml.data.dataset import CLASS_NAMES
from surveillance_ml.deployment.manifest import JsonValue, ModelManifest, sha256_file
from surveillance_ml.models.classifier import build_resnet18

INPUT_NAME = "images"
OUTPUT_NAME = "logits"
DEFAULT_OPSET = 17


def _checkpoint_class_names(payload: dict[str, object]) -> tuple[str, ...]:
    names = payload.get("class_names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("Checkpoint is missing a valid class_names list")
    class_names = tuple(names)
    if class_names != CLASS_NAMES:
        raise ValueError(f"Checkpoint classes must match the supported order: {CLASS_NAMES}")
    return class_names


def export_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    image_size: int | None = None,
    opset: int = DEFAULT_OPSET,
) -> tuple[Path, Path, ModelManifest]:
    """Export a checkpoint, verify its ONNX graph, and measure PyTorch/ORT parity."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not isinstance(opset, int) or isinstance(opset, bool) or opset < 17:
        raise ValueError("ONNX opset must be at least 17")
    payload: dict[str, object] = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    state_dict = payload.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Checkpoint has no model_state_dict")
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    resolved_image_size = image_size if image_size is not None else metadata.get("image_size")
    if (
        not isinstance(resolved_image_size, int)
        or isinstance(resolved_image_size, bool)
        or resolved_image_size < 32
    ):
        raise ValueError("Provide image_size or export a checkpoint containing metadata.image_size")
    architecture = metadata.get("architecture", "resnet18")
    if architecture != "resnet18":
        raise ValueError(f"Unsupported checkpoint architecture: {architecture}")
    class_names = _checkpoint_class_names(payload)
    model = build_resnet18(len(class_names), strategy="full", pretrained=False)
    model.load_state_dict(state_dict)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.onnx"
    manifest_path = output_dir / "manifest.json"
    temporary_model_path = output_dir / ".model.onnx.tmp"
    temporary_manifest_path = output_dir / ".manifest.json.tmp"
    try:
        example = torch.zeros(1, 3, resolved_image_size, resolved_image_size, dtype=torch.float32)
        torch.onnx.export(
            model,
            (example,),
            str(temporary_model_path),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={
                INPUT_NAME: {0: "batch"},
                OUTPUT_NAME: {0: "batch"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
        graph = onnx.load(str(temporary_model_path))
        onnx.checker.check_model(graph)
        session = ort.InferenceSession(
            str(temporary_model_path),
            providers=["CPUExecutionProvider"],
        )
        if session.get_providers()[0] != "CPUExecutionProvider":
            raise RuntimeError("ONNX Runtime did not initialize the CPU execution provider")
        generator = torch.Generator(device="cpu").manual_seed(0)
        parity_input = torch.randn(
            2,
            3,
            resolved_image_size,
            resolved_image_size,
            generator=generator,
        )
        with torch.inference_mode():
            torch_output = model(parity_input).cpu().numpy()
        runtime_output = session.run(
            [OUTPUT_NAME],
            {INPUT_NAME: parity_input.numpy()},
        )[0]
        if torch_output.shape != runtime_output.shape or not np.allclose(
            torch_output,
            runtime_output,
            rtol=1e-3,
            atol=1e-4,
        ):
            max_error = float(np.max(np.abs(torch_output - runtime_output)))
            raise RuntimeError(f"PyTorch/ONNX parity check failed (max absolute error {max_error})")

        model_digest = sha256_file(temporary_model_path)
        preprocessing: dict[str, JsonValue] = {
            "color_space": "RGB",
            "layout": "NCHW",
            "resize_mode": "fit",
            "resample": "bilinear",
            "normalization": "imagenet",
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
            "width": resolved_image_size,
            "height": resolved_image_size,
        }
        dataset_fingerprint = metadata.get("dataset_fingerprint")
        run_id = metadata.get("mlflow_run_id")
        manifest = ModelManifest(
            schema_version=1,
            model_file=model_path.name,
            model_sha256=model_digest,
            checkpoint_sha256=sha256_file(checkpoint_path),
            architecture="resnet18",
            class_names=class_names,
            input_name=INPUT_NAME,
            output_name=OUTPUT_NAME,
            input_shape=("batch", 3, resolved_image_size, resolved_image_size),
            opset=opset,
            preprocessing=preprocessing,
            dataset_fingerprint=(
                dataset_fingerprint if isinstance(dataset_fingerprint, str) else None
            ),
            mlflow_run_id=run_id if isinstance(run_id, str) else None,
        )
        temporary_manifest_path.write_text(
            json.dumps(manifest.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_model_path, model_path)
        os.replace(temporary_manifest_path, manifest_path)
    finally:
        temporary_model_path.unlink(missing_ok=True)
        temporary_manifest_path.unlink(missing_ok=True)
    return model_path, manifest_path, manifest
