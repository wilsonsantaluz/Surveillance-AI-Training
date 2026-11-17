"""Reproducible Markdown, JSON, PR-curve, and annotated error reports."""

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from surveillance_ml.detection import DETECTION_CLASS_NAMES
from surveillance_ml.detection.metrics import Box, intersection_over_union


def _box(value: Any) -> Box:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"Expected an xyxy box, received {value!r}")
    return Box(*(float(coordinate) for coordinate in value))


def _sample_errors(
    sample: dict[str, Any],
    confidence_threshold: float,
) -> tuple[list[dict[str, Any]], set[int], set[int]]:
    truths = sample["ground_truths"]
    predictions = [
        item for item in sample["predictions"] if float(item["confidence"]) >= confidence_threshold
    ]
    pairs = sorted(
        (
            (
                intersection_over_union(_box(truth["box"]), _box(prediction["box"])),
                truth_index,
                prediction_index,
            )
            for truth_index, truth in enumerate(truths)
            for prediction_index, prediction in enumerate(predictions)
            if truth["class_id"] == prediction["class_id"]
        ),
        reverse=True,
    )
    matched_truths: set[int] = set()
    matched_predictions: set[int] = set()
    errors: list[dict[str, Any]] = []
    for overlap, truth_index, prediction_index in pairs:
        if overlap < 0.5:
            break
        if truth_index in matched_truths or prediction_index in matched_predictions:
            continue
        matched_truths.add(truth_index)
        matched_predictions.add(prediction_index)
    for index, truth in enumerate(truths):
        if index not in matched_truths:
            errors.append(
                {
                    "kind": f"missed_{truth['class_name']}",
                    "true_class": truth["class_name"],
                    "predicted_class": None,
                    "confidence": None,
                    "ground_truth_box": truth["box"],
                    "prediction_box": None,
                    "iou": None,
                }
            )
    for index, prediction in enumerate(predictions):
        if index not in matched_predictions:
            errors.append(
                {
                    "kind": f"false_positive_{prediction['class_name']}",
                    "true_class": None,
                    "predicted_class": prediction["class_name"],
                    "confidence": float(prediction["confidence"]),
                    "ground_truth_box": None,
                    "prediction_box": prediction["box"],
                    "iou": None,
                }
            )
    return errors, matched_truths, matched_predictions


def _draw_error_image(
    source: Path,
    destination: Path,
    sample: dict[str, Any],
    errors: list[dict[str, Any]],
    confidence_threshold: float,
) -> None:
    with Image.open(source) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    _, matched_truths, matched_predictions = _sample_errors(sample, confidence_threshold)
    for index, truth in enumerate(sample["ground_truths"]):
        box = _box(truth["box"])
        color = "#238636" if index in matched_truths else "#d1242f"
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=3)
        label = f"GT {truth['class_name']}"
        draw.text((box.x1, max(0, box.y1 - 12)), label, fill=color, font=font)
    predictions = [
        item for item in sample["predictions"] if float(item["confidence"]) >= confidence_threshold
    ]
    for index, prediction in enumerate(predictions):
        box = _box(prediction["box"])
        color = "#1f6feb" if index in matched_predictions else "#cf222e"
        draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline=color, width=2)
        draw.text(
            (box.x1, min(canvas.height - 12, box.y2 + 2)),
            f"Pred {prediction['class_name']} {float(prediction['confidence']):.3f}",
            fill=color,
            font=font,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, quality=92)


def _write_precision_recall_plot(metrics: dict[str, Any], destination: Path) -> None:
    width, height = 720, 520
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, plot_width, plot_height = 72, 48, 600, 400
    draw.line(
        (left, top, left, top + plot_height, left + plot_width, top + plot_height),
        fill="black",
        width=2,
    )
    for step in range(6):
        x = left + int(plot_width * step / 5)
        y = top + int(plot_height * step / 5)
        draw.line((x, top, x, top + plot_height), fill="#eeeeee")
        draw.line((left, y, left + plot_width, y), fill="#eeeeee")
        draw.text((x - 3, top + plot_height + 8), f"{step / 5:.1f}", fill="black", font=font)
        draw.text((left - 30, y - 4), f"{1 - step / 5:.1f}", fill="black", font=font)
    draw.text((left + plot_width // 2 - 25, height - 24), "Recall", fill="black", font=font)
    draw.text((8, 12), "Precision", fill="black", font=font)
    colors = ("#d1242f", "#1f6feb")
    for color, class_name in zip(colors, DETECTION_CLASS_NAMES, strict=True):
        curve = metrics.get("per_class", {}).get(class_name, {}).get("precision_recall_curve", {})
        recall = curve.get("recall", [])
        precision = curve.get("precision", [])
        points = [
            (
                left + float(x) * plot_width,
                top + (1 - float(y)) * plot_height,
            )
            for x, y in zip(recall, precision, strict=True)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        draw.text(
            (left + 12, top + 12 + 16 * DETECTION_CLASS_NAMES.index(class_name)),
            class_name,
            fill=color,
            font=font,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def generate_error_reports(
    dataset_root: Path,
    run_dir: Path,
    metrics: dict[str, Any],
    *,
    maximum_images: int = 100,
) -> dict[str, Any]:
    """Write per-class error records and review images for an evaluation."""
    destination = run_dir / "error-analysis"
    destination.mkdir(parents=True, exist_ok=True)
    confidence_threshold = float(metrics.get("operating_confidence_threshold", 0.25))
    samples = metrics.get("sample_details", [])
    entries: list[dict[str, Any]] = []
    annotated_count = 0
    for sample in samples:
        errors, _, _ = _sample_errors(sample, confidence_threshold)
        if not errors:
            continue
        row = {
            "sample_id": sample["sample_id"],
            "relative_path": sample["relative_path"],
            "split": metrics.get("split", "val"),
            "errors": errors,
        }
        entries.append(row)
    entries.sort(
        key=lambda item: (
            -max(
                (
                    float(error["confidence"])
                    for error in item["errors"]
                    if error["confidence"] is not None
                ),
                default=0.0,
            ),
            item["sample_id"],
        )
    )
    for item in entries[:maximum_images]:
        image_path = dataset_root / "images" / str(item["split"]) / str(item["relative_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Evaluation image for error report is missing: {image_path}")
        sample = next(sample for sample in samples if sample["sample_id"] == item["sample_id"])
        image_destination = (
            destination / "images" / f"{item['sample_id']}{image_path.suffix.lower()}"
        )
        _draw_error_image(
            image_path,
            image_destination,
            sample,
            item["errors"],
            confidence_threshold,
        )
        item["review_image"] = image_destination.relative_to(run_dir).as_posix()
        annotated_count += 1
    report = {
        "split": metrics.get("split"),
        "operating_confidence_threshold": confidence_threshold,
        "iou_threshold": 0.5,
        "sample_error_count": len(entries),
        "annotated_image_count": annotated_count,
        "omitted_image_count": max(0, len(entries) - annotated_count),
        "samples": entries,
    }
    (destination / "error-analysis.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with (destination / "error-analysis.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "sample_id",
                "relative_path",
                "split",
                "kind",
                "true_class",
                "predicted_class",
                "confidence",
                "iou",
                "review_image",
            ),
        )
        writer.writeheader()
        for sample in entries:
            for error in sample["errors"]:
                writer.writerow(
                    {
                        "sample_id": sample["sample_id"],
                        "relative_path": sample["relative_path"],
                        "split": sample["split"],
                        **error,
                        "review_image": sample.get("review_image", ""),
                    }
                )
    _write_precision_recall_plot(metrics, destination / "precision-recall.png")
    markdown = [
        "# Detection error analysis",
        "",
        f"- Split: `{report['split']}`",
        f"- Operating confidence threshold: {confidence_threshold}",
        f"- Images with errors: {len(entries)}",
        f"- Annotated images: {annotated_count}",
        "",
        "## Highest-confidence false positives and missed objects",
        "",
    ]
    for sample in entries:
        kinds = ", ".join(sorted({error["kind"] for error in sample["errors"]}))
        link = sample.get("review_image", "")
        markdown.append(
            f"- `{sample['sample_id']}` (`{sample['relative_path']}`): {kinds}"
            + (f" — [review image]({link})" if link else "")
        )
    (destination / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return report


def write_autotrain_report(
    run_dir: Path,
    *,
    state: dict[str, Any],
    audit: dict[str, Any],
    environment: dict[str, Any] | None,
    metrics: dict[str, Any] | None,
    qualification: dict[str, Any] | None,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "dataset_audit": audit,
        "environment": environment,
        "validation_metrics": metrics,
        "qualification": qualification,
    }
    json_path = run_dir / "autotrain-report.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# AutoTrain report: {state['run_id']}",
        "",
        f"- Status: **{state['status']}**",
        f"- Current stage: `{state['current_stage']}`",
        f"- Dataset fingerprint: `{state.get('dataset_fingerprint') or 'unavailable'}`",
        f"- Audit: **{'passed' if audit.get('valid') else 'blocked'}**",
        f"- Images: {audit.get('image_count', 0)}",
        f"- Class boxes: `{json.dumps(audit.get('class_counts', {}), sort_keys=True)}`",
        "- Qualification: **"
        f"{(qualification or {}).get('qualification_status', 'not evaluated')}**",
        "- Deployment: **not automatic; manual approval is required**",
        "",
    ]
    if environment:
        lines.extend(
            [
                "## Environment",
                "",
                f"- Python / PyTorch: {environment.get('python')} / {environment.get('torch')}",
                f"- Device: {environment.get('resolved_device')}",
                f"- CUDA available: {environment.get('cuda_available')}",
                "",
            ]
        )
    if metrics:
        lines.extend(
            [
                "## Validation metrics",
                "",
                f"- Precision / recall: {metrics.get('precision')} / {metrics.get('recall')}",
                f"- mAP50 / mAP50-95: {metrics.get('mAP50')} / {metrics.get('mAP50_95')}",
                "",
                "| Class | AP50 | AP50-95 | Precision | Recall |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for class_name, values in metrics.get("per_class", {}).items():
            lines.append(
                f"| {class_name} | {values.get('average_precision_50')} "
                f"| {values.get('average_precision_50_95')} "
                f"| {values.get('precision')} | {values.get('recall')} |"
            )
        lines.extend(
            ["", "See `error-analysis/README.md` for review images and sample-level errors.", ""]
        )
    if qualification:
        lines.extend(["## Qualification gates", ""])
        for check in qualification.get("checks", []):
            lines.append(f"- `{check['name']}`: **{check['status']}** — {check['reason']}")
        lines.append("")
    lines.extend(
        [
            "## Dataset audit",
            "",
            "See [dataset-audit.md](dataset-audit.md) and its contact sheets. Audit warnings "
            "do not certify source licensing or model accuracy.",
            "",
        ]
    )
    markdown_path = run_dir / "autotrain-report.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path
