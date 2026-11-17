import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from surveillance_ml.detection.manifest import (
    DetectionManifest,
    load_detection_manifest,
)
from surveillance_ml.detection.onnx import (
    DetectionPrediction,
    _decode_output,
    _letterbox_rgb,
    _nms,
    _reference_predictions,
)


def _manifest() -> DetectionManifest:
    return DetectionManifest(
        schema_version=1,
        model_id="test-detector",
        model_version="test-v1",
        model_file="model.onnx",
        model_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        architecture="yolo11n",
        class_names=("firearm", "knife"),
        input_name="images",
        input_shape=(1, 3, 64, 64),
        output_name="output0",
        output_shape=(1, 6, 3),
        opset=17,
        preprocessing={
            "color_space": "RGB",
            "layout": "NCHW",
            "resize_mode": "letterbox",
            "resample": "opencv_inter_linear_exact",
            "padding_value": 114,
            "normalization": "divide_by_255",
            "width": 64,
            "height": 64,
        },
        output_contract={
            "layout": "batch_channels_candidates",
            "box_format": "cxcywh_pixels_in_letterboxed_input",
            "class_scores": "sigmoid_probabilities_without_objectness",
        },
        postprocessing={
            "nms": "class_aware_greedy",
            "confidence_threshold": 0.25,
            "iou_threshold": 0.5,
            "max_detections": 10,
        },
        dataset_fingerprint=None,
        mlflow_run_id=None,
        source_license="AGPL-3.0",
    )


def test_detection_manifest_round_trips_and_rejects_wrong_contract(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = _manifest().to_dict()
    path.write_text(json.dumps(payload), encoding="utf-8")

    manifest = load_detection_manifest(path)
    assert manifest.class_names == ("firearm", "knife")

    payload["output_contract"]["box_format"] = "xyxy"
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_detection_manifest(path)
    except ValueError as error:
        assert "output contract" in str(error)
    else:
        raise AssertionError("Invalid detection output contract was accepted")


def test_decoder_restores_coordinates_and_applies_class_aware_nms() -> None:
    output = np.array(
        [
            [
                [32, 33, 32],
                [32, 32, 32],
                [20, 20, 20],
                [20, 20, 20],
                [0.9, 0.8, 0.1],
                [0.1, 0.1, 0.7],
            ]
        ],
        dtype=np.float32,
    )

    predictions = _decode_output(
        output,
        _manifest(),
        scale=0.64,
        pad_left=0,
        pad_top=16,
        image_width=100,
        image_height=50,
    )

    assert [prediction.class_name for prediction in predictions] == ["firearm", "knife"]
    assert predictions[0].x == 34.375
    assert predictions[0].y == 9.375
    assert predictions[0].width == 31.25
    assert predictions[0].height == 31.25
    assert all(0 <= prediction.x <= 100 for prediction in predictions)
    assert all(0 <= prediction.y <= 50 for prediction in predictions)


def test_reference_predictions_use_fixed_letterbox_geometry() -> None:
    call_arguments: dict[str, object] = {}

    class Model:
        def predict(self, **kwargs: object) -> list[SimpleNamespace]:
            call_arguments.update(kwargs)
            return [SimpleNamespace(boxes=None)]

    _reference_predictions(
        Model(),  # type: ignore[arg-type]
        np.zeros((32, 64, 3), dtype=np.uint8),
        _manifest(),
    )

    assert call_arguments["rect"] is False


def test_decoder_includes_threshold_boundary_and_keeps_classes_separate() -> None:
    manifest = replace(_manifest(), output_shape=(1, 6, 4))
    output = np.array(
        [
            [
                [32, 32, 8, 48],
                [32, 32, 8, 48],
                [20, 20, 10, 10],
                [20, 20, 10, 10],
                [0.25, 0.249, 0.3, 0.0],
                [0.25, 0.8, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )

    predictions = _decode_output(
        output,
        manifest,
        scale=1,
        pad_left=0,
        pad_top=0,
        image_width=64,
        image_height=64,
    )

    assert [item.class_name for item in predictions] == ["knife", "firearm", "firearm"]
    np.testing.assert_allclose(
        [item.confidence for item in predictions],
        [0.8, 0.3, 0.25],
        rtol=0,
        atol=1e-7,
    )


def test_nms_suppresses_same_class_only_and_retains_iou_boundary() -> None:
    candidates = [
        DetectionPrediction(0, "firearm", 0.9, 0, 0, 15, 10),
        DetectionPrediction(0, "firearm", 0.85, 1, 0, 15, 10),
        DetectionPrediction(0, "firearm", 0.8, 5, 0, 15, 10),
        DetectionPrediction(1, "knife", 0.7, 0, 0, 15, 10),
    ]

    predictions = _nms(candidates, iou_threshold=0.5, max_detections=10)

    assert [item.confidence for item in predictions] == [0.9, 0.8, 0.7]


def test_letterbox_uses_bit_exact_linear_interpolation() -> None:
    image = np.array(
        [
            [[0, 0, 0], [100, 100, 100]],
            [[200, 200, 200], [254, 254, 254]],
        ],
        dtype=np.uint8,
    )

    resized, scale, pad_left, pad_top = _letterbox_rgb(image, 3, 3)

    assert (scale, pad_left, pad_top) == (1.5, 0, 0)
    np.testing.assert_array_equal(
        resized[:, :, 0],
        np.array([[0, 50, 100], [100, 139, 177], [200, 227, 254]], dtype=np.uint8),
    )
