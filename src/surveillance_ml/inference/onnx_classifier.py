"""Validated ONNX Runtime inference for scene-classification bundles."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from surveillance_ml.core.image_preprocessing import preprocess_image
from surveillance_ml.deployment.manifest import ModelManifest, load_manifest, verify_model_hash


@dataclass(frozen=True)
class Prediction:
    class_name: str
    confidence: float
    probabilities: dict[str, float]


class OnnxClassifier:
    """Loads a hash-verified model bundle and produces named class probabilities."""

    def __init__(self, model_path: Path, manifest_path: Path, intra_op_threads: int = 0) -> None:
        self.model_path = model_path
        self.manifest: ModelManifest = load_manifest(manifest_path)
        verify_model_hash(model_path, self.manifest)
        if intra_op_threads < 0:
            raise ValueError("intra_op_threads must be non-negative")
        self.intra_op_threads = intra_op_threads
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
            raise ValueError("ONNX classifier must have exactly one input and one output")
        model_input = inputs[0]
        model_output = outputs[0]
        if model_input.name != self.manifest.input_name or model_input.type != "tensor(float)":
            raise ValueError("ONNX input does not match the deployment manifest")
        if model_output.name != self.manifest.output_name or model_output.type != "tensor(float)":
            raise ValueError("ONNX output does not match the deployment manifest")
        expected = list(self.manifest.input_shape)
        actual = model_input.shape
        if actual[1:] != expected[1:] or actual[0] not in {None, "batch", "unk__0"}:
            raise ValueError("ONNX input shape does not match the deployment manifest")
        output_shape = model_output.shape
        if (
            output_shape[1:] != [len(self.manifest.class_names)]
            or output_shape[0] not in {None, "batch", "unk__0"}
        ):
            raise ValueError("ONNX output shape does not match the deployment manifest")

    def predict_tensor(self, batch: np.ndarray) -> list[Prediction]:
        """Run model inference on already-normalized float32 NCHW inputs."""
        expected_height = int(self.manifest.input_shape[2])
        expected_width = int(self.manifest.input_shape[3])
        if (
            batch.dtype != np.float32
            or batch.ndim != 4
            or batch.shape[0] < 1
            or batch.shape[1:] != (3, expected_height, expected_width)
            or not np.isfinite(batch).all()
        ):
            raise ValueError(
                f"Expected finite float32 NCHW inputs shaped [N, 3, {expected_height}, "
                f"{expected_width}]"
            )
        logits = self._session.run(
            [self.manifest.output_name],
            {self.manifest.input_name: np.ascontiguousarray(batch)},
        )[0]
        if logits.shape != (batch.shape[0], len(self.manifest.class_names)):
            raise RuntimeError(f"Unexpected ONNX output shape: {logits.shape}")
        if not np.isfinite(logits).all():
            raise RuntimeError("ONNX model returned non-finite logits")
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return [
            Prediction(
                class_name=self.manifest.class_names[int(index)],
                confidence=float(row[index]),
                probabilities={
                    class_name: float(row[class_index])
                    for class_index, class_name in enumerate(self.manifest.class_names)
                },
            )
            for row in probabilities
            for index in [np.argmax(row)]
        ]

    def predict_image(self, image: Image.Image) -> Prediction:
        """Preprocess one RGB image according to the exported manifest and infer."""
        height = int(self.manifest.input_shape[2])
        width = int(self.manifest.input_shape[3])
        batch = preprocess_image(image, width, height)[None, ...]
        return self.predict_tensor(batch)[0]
