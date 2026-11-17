from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from surveillance_ml.api.app import create_app
from surveillance_ml.deployment.benchmark import benchmark_image, write_benchmark_report
from surveillance_ml.deployment.export import export_checkpoint
from surveillance_ml.deployment.manifest import load_manifest, verify_model_hash
from surveillance_ml.inference.onnx_classifier import OnnxClassifier
from surveillance_ml.models.classifier import build_resnet18

API_KEY = "test-key-with-more-than-thirty-two-characters"


@pytest.fixture(scope="module")
def exported_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("onnx-bundle")
    torch.manual_seed(23)
    model = build_resnet18(num_classes=3, strategy="full", pretrained=False)
    checkpoint_path = directory / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "validation_loss": 1.0,
            "class_names": ["empty_scene", "single_person", "multiple_people"],
            "metadata": {"architecture": "resnet18", "image_size": 32},
        },
        checkpoint_path,
    )
    model_path, manifest_path, _ = export_checkpoint(checkpoint_path, directory / "onnx")
    return model_path, manifest_path


def test_export_runtime_manifest_and_probabilities(
    exported_bundle: tuple[Path, Path],
) -> None:
    model_path, manifest_path = exported_bundle
    manifest = load_manifest(manifest_path)
    verify_model_hash(model_path, manifest)
    classifier = OnnxClassifier(model_path, manifest_path)
    batch = np.zeros((2, 3, 32, 32), dtype=np.float32)
    predictions = classifier.predict_tensor(batch)
    assert len(predictions) == 2
    for prediction in predictions:
        assert prediction.class_name in manifest.class_names
        assert sum(prediction.probabilities.values()) == pytest.approx(1.0)
        assert prediction.confidence == max(prediction.probabilities.values())


def test_legacy_checkpoint_requires_image_size(tmp_path: Path) -> None:
    model = build_resnet18(num_classes=3, pretrained=False)
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ["empty_scene", "single_person", "multiple_people"],
        },
        checkpoint,
    )
    with pytest.raises(ValueError, match="Provide image_size"):
        export_checkpoint(checkpoint, tmp_path / "export")


def test_runtime_rejects_modified_onnx_file(
    exported_bundle: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    model_path, manifest_path = exported_bundle
    modified_model = tmp_path / model_path.name
    modified_model.write_bytes(model_path.read_bytes() + b"modified")
    with pytest.raises(ValueError, match="hash does not match"):
        OnnxClassifier(modified_model, manifest_path)


def test_manifest_rejects_model_path_traversal(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
        {
          "schema_version": 1,
          "model_file": "../model.onnx",
          "model_sha256": "0",
          "checkpoint_sha256": "0",
          "architecture": "resnet18",
          "class_names": ["empty_scene", "single_person", "multiple_people"],
          "input": {"name": "images", "shape": ["batch", 3, 32, 32], "dtype": "float32"},
          "output": {"name": "logits", "shape": ["batch", 3]},
          "opset": 17,
          "preprocessing": {}
        }
        """,
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="plain filename"):
        load_manifest(manifest_path)


def test_api_auth_upload_validation_and_prediction(
    exported_bundle: tuple[Path, Path],
) -> None:
    model_path, manifest_path = exported_bundle
    app = create_app(model_path, manifest_path, API_KEY, max_upload_bytes=1024)
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"

        image_bytes = _png_bytes()
        unauthorized = client.post(
            "/predict",
            files={"file": ("scene.png", image_bytes, "image/png")},
        )
        assert unauthorized.status_code == 401

        prediction = client.post(
            "/predict",
            headers={"X-API-Key": API_KEY},
            files={"file": ("scene.png", image_bytes, "image/png")},
        )
        assert prediction.status_code == 200
        assert prediction.json()["class_name"] in {
            "empty_scene",
            "single_person",
            "multiple_people",
        }

        too_large = client.post(
            "/predict",
            headers={"X-API-Key": API_KEY},
            files={"file": ("large.png", b"x" * 2048, "image/png")},
        )
        assert too_large.status_code == 413

        oversized_request = client.post(
            "/predict",
            headers={"X-API-Key": API_KEY},
            files={"file": ("very-large.png", b"x" * (70 * 1024), "image/png")},
        )
        assert oversized_request.status_code == 413

        corrupt = client.post(
            "/predict",
            headers={"X-API-Key": API_KEY},
            files={"file": ("corrupt.png", b"not an image", "image/png")},
        )
        assert corrupt.status_code == 400


def test_benchmark_writes_observed_latency_report(
    exported_bundle: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    model_path, manifest_path = exported_bundle
    classifier = OnnxClassifier(model_path, manifest_path)
    image_path = tmp_path / "sample.png"
    Image.new("RGB", (48, 36), color=(30, 90, 150)).save(image_path)
    report = benchmark_image(classifier, image_path, warmup=0, iterations=2)
    output_path = tmp_path / "benchmark.json"
    write_benchmark_report(report, output_path)
    assert report["measured_iterations"] == 2
    assert report["throughput_images_per_second"] > 0
    assert output_path.is_file()
    assert "preprocessing excluded" in report["timing_scope"]


def test_api_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="at least 32 characters"):
        create_app(api_key="too-short")
    with pytest.raises(ValueError, match="upload size"):
        create_app(api_key=API_KEY, max_upload_bytes=0)


def _png_bytes() -> bytes:
    from io import BytesIO

    image = Image.new("RGB", (48, 36), color=(120, 80, 20))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()
