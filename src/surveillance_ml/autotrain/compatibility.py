"""Actual-detection parity checks between Python ONNX Runtime and the .NET detector."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from surveillance_ml.detection import DETECTION_CLASS_NAMES
from surveillance_ml.detection.dataset import DetectionDatasetReport
from surveillance_ml.detection.metrics import Box, intersection_over_union
from surveillance_ml.detection.onnx import OnnxDetector


def _prediction(value: Any) -> tuple[int, float, Box]:
    box = value["box"]
    return (
        int(value["class_id"]),
        float(value["confidence"]),
        Box(
            float(box["x"]),
            float(box["y"]),
            float(box["x"]) + float(box["width"]),
            float(box["y"]) + float(box["height"]),
        ),
    )


def _compare(
    expected: list[tuple[int, float, Box]],
    actual: list[tuple[int, float, Box]],
) -> dict[str, Any]:
    unmatched = list(actual)
    matches: list[dict[str, float | int]] = []
    for class_id, confidence, box in expected:
        choices = [
            (index, item, intersection_over_union(box, item[2]))
            for index, item in enumerate(unmatched)
            if item[0] == class_id
        ]
        if not choices:
            continue
        index, item, overlap = max(choices, key=lambda value: value[2])
        unmatched.pop(index)
        matches.append(
            {
                "class_id": class_id,
                "iou": overlap,
                "confidence_difference": abs(confidence - item[1]),
            }
        )
    passed = (
        len(matches) == len(expected) == len(actual)
        and all(float(item["iou"]) >= 0.99 for item in matches)
        and all(float(item["confidence_difference"]) <= 1e-3 for item in matches)
    )
    return {
        "passed": passed,
        "python_detection_count": len(expected),
        "dotnet_detection_count": len(actual),
        "matched_detection_count": len(matches),
        "minimum_matched_iou": min(
            (float(item["iou"]) for item in matches),
            default=1.0,
        ),
        "maximum_confidence_difference": max(
            (float(item["confidence_difference"]) for item in matches),
            default=0.0,
        ),
        "matches": matches,
    }


def run_python_dotnet_parity(
    model_path: Path,
    manifest_path: Path,
    dataset_report: DetectionDatasetReport,
    repository_root: Path,
    *,
    maximum_images: int = 8,
) -> dict[str, Any]:
    """Compare non-empty detections from validation images; never pass empty-only checks."""
    executable = shutil.which("dotnet")
    if executable is None:
        return {"passed": False, "status": "blocked", "reason": "dotnet executable was not found"}
    detector = OnnxDetector(model_path, manifest_path)
    candidates: list[tuple[Path, set[int]]] = []
    for record in sorted(
        (item for item in dataset_report.records if item.split == "val"),
        key=lambda item: item.image_path.as_posix(),
    ):
        inference = detector.predict_image(record.image_path)
        classes = {item.class_id for item in inference.detections}
        if classes:
            candidates.append((record.image_path, classes))
    selected: list[Path] = []
    for class_id in range(len(DETECTION_CLASS_NAMES)):
        match = next(
            (path for path, classes in candidates if class_id in classes),
            None,
        )
        if match is None:
            return {
                "passed": False,
                "status": "blocked_no_nonempty_detection",
                "reason": (
                    f"No non-empty Python ONNX detection for "
                    f"{DETECTION_CLASS_NAMES[class_id]} on validation images."
                ),
                "python_detected_classes": sorted(
                    {DETECTION_CLASS_NAMES[value] for _, classes in candidates for value in classes}
                ),
            }
        if match not in selected:
            selected.append(match)
    for path, _ in candidates:
        if len(selected) >= maximum_images:
            break
        if path not in selected:
            selected.append(path)
    sample_names = {record.image_path: record for record in dataset_report.records}
    project = repository_root / "dotnet" / "Detector" / "Detector.csproj"
    command = [
        executable,
        "run",
        "--project",
        str(project),
        "--configuration",
        "Release",
        "--no-restore",
        "--",
        str(model_path.resolve()),
        str(manifest_path.resolve()),
        "autotrain-parity",
        *(str(path.resolve()) for path in selected),
    ]
    try:
        process = subprocess.run(
            command,
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "status": "failed",
            "reason": ".NET parity command exceeded 300 seconds.",
        }
    except OSError as error:
        return {"passed": False, "status": "failed", "reason": str(error)}
    if process.returncode != 0:
        return {
            "passed": False,
            "status": "failed",
            "reason": "The standalone .NET detector returned a non-zero exit code.",
            "exit_code": process.returncode,
            "stderr": process.stderr[-4000:],
        }
    try:
        frames = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        return {
            "passed": False,
            "status": "failed",
            "reason": f"The .NET detector returned invalid JSON: {error}",
            "stdout": process.stdout[-4000:],
        }
    if not isinstance(frames, list) or len(frames) != len(selected):
        return {
            "passed": False,
            "status": "failed",
            "reason": "The .NET detector returned an unexpected number of frames.",
        }

    checks: list[dict[str, Any]] = []
    for image_path, frame_result in zip(selected, frames, strict=True):
        if not isinstance(frame_result, dict) or not isinstance(frame_result.get("frame"), dict):
            return {"passed": False, "status": "failed", "reason": "Malformed .NET frame result."}
        dotnet_detections = frame_result["frame"].get("detections")
        if not isinstance(dotnet_detections, list):
            return {"passed": False, "status": "failed", "reason": "Missing .NET detections."}
        python_detections = detector.predict_image(image_path).detections
        expected = [(item.class_id, item.confidence, item.xyxy) for item in python_detections]
        actual = [_prediction(item) for item in dotnet_detections]
        comparison = _compare(expected, actual)
        checks.append(
            {
                "sample_id": sample_names[image_path].source_id,
                "relative_path": image_path.relative_to(
                    dataset_report.root / "images" / "val"
                ).as_posix(),
                **comparison,
            }
        )
    passed = bool(checks) and all(bool(item["passed"]) for item in checks)
    return {
        "passed": passed,
        "status": "passed" if passed else "failed",
        "comparison_count": len(checks),
        "nonempty_python_detection_count": sum(
            int(item["python_detection_count"]) for item in checks
        ),
        "required_classes": list(DETECTION_CLASS_NAMES),
        "checks": checks,
    }
