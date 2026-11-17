import json
from pathlib import Path

import numpy as np

from surveillance_ml.detection.manifest import (
    DetectionManifest,
    load_detection_manifest,
)
from surveillance_ml.detection.onnx import _decode_output


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
            "resample": "opencv_inter_linear",
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
