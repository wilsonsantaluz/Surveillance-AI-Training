"""Dataset integrity, class balance, duplicate, provenance, and annotation review."""

import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from surveillance_ml.autotrain.config import AutoTrainConfig
from surveillance_ml.detection import DETECTION_CLASS_NAMES
from surveillance_ml.detection.dataset import (
    DetectionDatasetReport,
    DetectionRecord,
    validate_yolo_dataset,
)


@dataclass(frozen=True)
class AuditIssue:
    severity: str
    code: str
    message: str
    sample_id: str | None = None
    relative_path: str | None = None
    details: dict[str, Any] | None = None


class DuplicateSample(TypedDict):
    path: str
    split: str


class DuplicateGroup(TypedDict):
    samples: list[DuplicateSample]
    splits: list[str]


def _duplicate_details(group: DuplicateGroup) -> dict[str, Any]:
    return {
        "samples": [dict(sample) for sample in group["samples"]],
        "splits": group["splits"],
    }


@dataclass(frozen=True)
class AuditResult:
    valid: bool
    dataset_root: str
    dataset_fingerprint: str | None
    image_count: int
    positive_image_count: int
    negative_image_count: int
    class_counts: dict[str, int]
    split_counts: dict[str, dict[str, Any]]
    imbalance_ratio: float | None
    exact_duplicate_groups: tuple[DuplicateGroup, ...]
    perceptual_duplicate_groups: tuple[DuplicateGroup, ...]
    source_split_integrity: bool
    license_review_status: str
    issues: tuple[AuditIssue, ...]
    contact_sheets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sample_id(record: DetectionRecord, root: Path) -> str:
    relative = record.image_path.relative_to(root).as_posix()
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]


def _issue(
    severity: str,
    code: str,
    message: str,
    record: DetectionRecord | None = None,
    root: Path | None = None,
    details: dict[str, Any] | None = None,
) -> AuditIssue:
    return AuditIssue(
        severity=severity,
        code=code,
        message=message,
        sample_id=_sample_id(record, root) if record is not None and root is not None else None,
        relative_path=(
            record.image_path.relative_to(root).as_posix()
            if record is not None and root is not None
            else None
        ),
        details=details,
    )


def _coordinate_rows(record: DetectionRecord) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line in record.label_path.read_text(encoding="utf-8").splitlines():
        class_text, x_text, y_text, width_text, height_text = line.split()
        rows.append(
            (
                int(class_text),
                float(x_text),
                float(y_text),
                float(width_text),
                float(height_text),
            )
        )
    return rows


def _perceptual_hash(image_path: Path) -> int:
    with Image.open(image_path) as image:
        pixels = np.asarray(image.convert("L").resize((32, 32)), dtype=np.float64)
    coordinates = np.arange(32, dtype=np.float64)
    basis = np.cos((2 * coordinates[:, None] + 1) * coordinates[None, :] * math.pi / 64)
    basis[:, 0] *= 1 / math.sqrt(2)
    basis *= math.sqrt(2 / 32)
    transformed = basis.T @ pixels @ basis
    low_frequency = transformed[:8, :8].flatten()[1:]
    median = float(np.median(low_frequency))
    result = 0
    for value in low_frequency:
        result = (result << 1) | int(value > median)
    return result


def _perceptual_groups(
    entries: list[tuple[str, str, int]],
    maximum_distance: int,
) -> list[DuplicateGroup]:
    parents = list(range(len(entries)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    buckets: dict[int, set[int]] = defaultdict(set)
    band_bits = 16
    band_mask = (1 << band_bits) - 1
    for index, (_, _, fingerprint) in enumerate(entries):
        candidates: set[int] = set()
        for band in range(64 // band_bits):
            key = (band << band_bits) | ((fingerprint >> (band * band_bits)) & band_mask)
            candidates.update(buckets[key])
            buckets[key].add(index)
        for candidate in candidates:
            if (fingerprint ^ entries[candidate][2]).bit_count() <= maximum_distance:
                union(index, candidate)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(entries)):
        components[find(index)].append(index)
    groups: list[DuplicateGroup] = []
    for indices in components.values():
        if len(indices) < 2:
            continue
        groups.append(
            {
                "samples": [
                    {"path": entries[index][0], "split": entries[index][1]} for index in indices
                ],
                "splits": sorted({entries[index][1] for index in indices}),
            }
        )
    return sorted(groups, key=lambda group: group["samples"][0]["path"])


def _save_contact_sheets(
    report: DetectionDatasetReport,
    root: Path,
    destination: Path,
    config: AutoTrainConfig,
    records_by_category: dict[str, list[DetectionRecord]],
) -> tuple[str, ...]:
    destination.mkdir(parents=True, exist_ok=True)
    sheet_paths: list[str] = []
    rng = random.Random(config.experiment.seed)
    for category, values in sorted(records_by_category.items()):
        records = sorted(values, key=lambda item: item.image_path.as_posix())
        rng.shuffle(records)
        records = records[: config.audit.contact_sheet_samples_per_category]
        for page_index in range(0, len(records), 16):
            page = records[page_index : page_index + 16]
            canvas = Image.new("RGB", (4 * 320, 4 * 250), "white")
            draw = ImageDraw.Draw(canvas)
            font = ImageFont.load_default()
            for tile_index, record in enumerate(page):
                column, row = tile_index % 4, tile_index // 4
                x_offset, y_offset = column * 320, row * 250
                with Image.open(record.image_path) as source:
                    image = source.convert("RGB")
                    source_width, source_height = image.size
                    image.thumbnail((304, 208), Image.Resampling.LANCZOS)
                    image_x = x_offset + (304 - image.width) // 2 + 8
                    image_y = y_offset + 8
                    canvas.paste(image, (image_x, image_y))
                scale = min(304 / source_width, 208 / source_height)
                boxes = _coordinate_rows(record)
                for class_id, x_center, y_center, width, height in boxes:
                    left = image_x + (x_center - width / 2) * source_width * scale
                    top = image_y + (y_center - height / 2) * source_height * scale
                    right = image_x + (x_center + width / 2) * source_width * scale
                    bottom = image_y + (y_center + height / 2) * source_height * scale
                    color = ("#e63946", "#167d4a")[class_id]
                    draw.rectangle((left, top, right, bottom), outline=color, width=2)
                    draw.text(
                        (left, max(y_offset + 8, top - 12)),
                        DETECTION_CLASS_NAMES[class_id],
                        fill=color,
                        font=font,
                    )
                relative = record.image_path.relative_to(root).as_posix()
                draw.text(
                    (x_offset + 8, y_offset + 220),
                    f"{record.split} | {_sample_id(record, root)}",
                    fill="black",
                    font=font,
                )
                draw.text(
                    (x_offset + 8, y_offset + 233),
                    relative[:42],
                    fill="black",
                    font=font,
                )
            filename = f"{category}-{page_index // 16 + 1:03d}.jpg"
            relative_sheet = (destination / filename).relative_to(destination.parent).as_posix()
            canvas.save(destination / filename, quality=90)
            sheet_paths.append(relative_sheet)
    return tuple(sheet_paths)


def _write_reports(
    run_dir: Path,
    result: AuditResult,
    report: DetectionDatasetReport | None,
    records_by_category: dict[str, list[DetectionRecord]],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dataset-audit.json").write_text(
        json.dumps(result.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    with (run_dir / "annotation-issues.csv").open("w", newline="", encoding="utf-8") as file:
        issues_writer = csv.DictWriter(
            file,
            fieldnames=("severity", "code", "sample_id", "relative_path", "message", "details"),
        )
        issues_writer.writeheader()
        for item in result.issues:
            issues_writer.writerow(
                {
                    **asdict(item),
                    "details": json.dumps(item.details, sort_keys=True),
                }
            )
    with (run_dir / "class-distribution.csv").open("w", newline="", encoding="utf-8") as file:
        distribution_writer = csv.writer(file)
        distribution_writer.writerow(("split", "class", "boxes", "images", "negative_images"))
        for split, values in report.split_counts.items() if report is not None else ():
            for class_name in DETECTION_CLASS_NAMES:
                distribution_writer.writerow(
                    (
                        split,
                        class_name,
                        values["objects"].get(class_name, 0),
                        values["images"],
                        values["negative_images"],
                    )
                )
    lines = [
        "# AutoTrain dataset audit",
        "",
        f"- Dataset root: `{result.dataset_root}`",
        f"- Dataset fingerprint: `{result.dataset_fingerprint or 'unavailable'}`",
        f"- Audit status: **{'passed' if result.valid else 'blocked'}**",
        f"- Images: {result.image_count}",
        f"- Positive / negative images: {result.positive_image_count} / "
        f"{result.negative_image_count}",
        f"- Class boxes: {json.dumps(result.class_counts, sort_keys=True)}",
        f"- License review status: `{result.license_review_status}`",
        f"- Source-split integrity: **{result.source_split_integrity}**",
        "",
        "## Findings",
        "",
    ]
    if not result.issues:
        lines.append("No audit issues were found.")
    else:
        for item in result.issues:
            lines.append(f"- **{item.severity.upper()} `{item.code}`:** {item.message}")
    lines.extend(["", "## Annotation review sheets", ""])
    if result.contact_sheets:
        lines.extend(f"- [{Path(item).name}]({item})" for item in result.contact_sheets)
    else:
        lines.append("No review sheets were generated because the dataset did not validate.")
    if records_by_category:
        lines.extend(
            [
                "",
                "Contact sheets are deterministic samples with original annotations overlaid. "
                "They do not establish licensing or model accuracy.",
            ]
        )
    (run_dir / "dataset-audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_dataset(config: AutoTrainConfig, run_dir: Path) -> AuditResult:
    """Audit an existing YOLO dataset and persist machine- and human-readable reports."""
    root = config.dataset.path.resolve()
    issues: list[AuditIssue] = []
    if not root.is_dir():
        issues.append(
            AuditIssue(
                "error",
                "dataset_missing",
                f"Dataset directory does not exist: {root}",
            )
        )
        result = AuditResult(
            False,
            str(root),
            None,
            0,
            0,
            0,
            {},
            {},
            None,
            (),
            (),
            False,
            config.dataset.license_review_status,
            tuple(issues),
            (),
        )
        _write_reports(run_dir, result, None, {})
        return result

    report: DetectionDatasetReport | None
    try:
        report = validate_yolo_dataset(root)
    except (FileNotFoundError, ValueError, OSError) as error:
        report = None
        message = str(error)
        code = "dataset_validation_failed"
        if "appears in both" in message:
            code = "source_group_leakage"
        elif "normalized inside" in message or "Non-finite" in message:
            code = "invalid_coordinates"
        elif "Corrupt or unreadable image" in message:
            code = "corrupt_image"
        elif "pair mismatch" in message:
            code = "missing_or_orphan_annotation"
        elif "Unsupported image format" in message:
            code = "unsupported_image_format"
        issues.append(AuditIssue("error", code, message))

    if (
        config.dataset.license_review_required
        and config.dataset.license_review_status != "approved"
    ):
        issues.append(
            AuditIssue(
                "warning",
                "license_review_pending",
                "License review is not approved; deployment approval will be blocked.",
                details={
                    "license": config.dataset.license,
                    "provenance": config.dataset.provenance,
                },
            )
        )
    if not config.dataset.provenance.strip():
        issues.append(
            AuditIssue("warning", "provenance_missing", "Dataset provenance is not documented.")
        )

    if report is None:
        result = AuditResult(
            False,
            str(root),
            None,
            0,
            0,
            0,
            {},
            {},
            None,
            (),
            (),
            False,
            config.dataset.license_review_status,
            tuple(issues),
            (),
        )
        _write_reports(run_dir, result, None, {})
        return result

    issues_by_record: dict[Path, list[str]] = defaultdict(list)
    category_records: dict[str, list[DetectionRecord]] = defaultdict(list)
    image_hashes: dict[str, list[tuple[DetectionRecord, str]]] = defaultdict(list)
    perceptual_entries: list[tuple[str, str, int]] = []
    filename_locations: dict[str, list[str]] = defaultdict(list)
    for record in report.records:
        relative = record.image_path.relative_to(root).as_posix()
        filename_locations[record.image_path.name.casefold()].append(relative)
        if config.audit.verify_images:
            digest = hashlib.sha256(record.image_path.read_bytes()).hexdigest()
            image_hashes[digest].append((record, relative))
            with Image.open(record.image_path) as image:
                width, height = image.size
            if max(width, height) < config.audit.minimum_resolution:
                issues.append(
                    _issue(
                        "warning",
                        "low_resolution",
                        f"Image resolution is {width}x{height}.",
                        record,
                        root,
                    )
                )
                issues_by_record[record.image_path].append("low_resolution")
                category_records["low-resolution"].append(record)
        if config.audit.detect_perceptual_duplicates:
            perceptual_entries.append(
                (
                    relative,
                    record.split,
                    _perceptual_hash(record.image_path),
                )
            )
        rows = _coordinate_rows(record) if config.audit.verify_annotations else []
        seen_annotations: set[tuple[int, float, float, float, float]] = set()
        for row in rows:
            class_id, _, _, box_width, box_height = row
            area = box_width * box_height
            if row in seen_annotations:
                issues.append(
                    _issue(
                        "warning",
                        "duplicate_annotation",
                        "Duplicate bounding-box row appears in one label file.",
                        record,
                        root,
                    )
                )
                issues_by_record[record.image_path].append("duplicate_annotation")
            seen_annotations.add(row)
            if area < config.audit.small_box_area_fraction:
                issues.append(
                    _issue(
                        "warning",
                        "suspiciously_small_box",
                        f"Bounding-box area fraction is {area:.8g}.",
                        record,
                        root,
                    )
                )
                issues_by_record[record.image_path].append("suspiciously_small_box")
                category_records["small-box"].append(record)
            if area > config.audit.large_box_area_fraction:
                issues.append(
                    _issue(
                        "warning",
                        "suspiciously_large_box",
                        f"Bounding-box area fraction is {area:.8g}.",
                        record,
                        root,
                    )
                )
                issues_by_record[record.image_path].append("suspiciously_large_box")
                category_records["large-box"].append(record)
            category_records[DETECTION_CLASS_NAMES[class_id]].append(record)
        if not record.class_ids:
            category_records["negative"].append(record)
        if issues_by_record[record.image_path]:
            category_records["anomalies"].append(record)

    exact_groups: list[DuplicateGroup] = []
    if config.audit.detect_exact_duplicates:
        for duplicate_records in image_hashes.values():
            if len(duplicate_records) < 2:
                continue
            splits = sorted({record.split for record, _ in duplicate_records})
            group: DuplicateGroup = {
                "samples": [
                    {"path": relative, "split": record.split}
                    for record, relative in duplicate_records
                ],
                "splits": splits,
            }
            exact_groups.append(group)
            cross_split = len(splits) > 1
            issues.append(
                AuditIssue(
                    "error" if cross_split else "warning",
                    "exact_duplicate_cross_split" if cross_split else "exact_duplicate",
                    "Byte-identical images occur across splits."
                    if cross_split
                    else "Byte-identical images occur more than once in one split.",
                    details=_duplicate_details(group),
                )
            )
    for filename, file_paths in filename_locations.items():
        if len(file_paths) > 1:
            issues.append(
                AuditIssue(
                    "warning",
                    "duplicate_filename",
                    f"Filename {filename!r} occurs at multiple dataset paths.",
                    details={"paths": sorted(file_paths)},
                )
            )

    perceptual_groups: list[DuplicateGroup] = []
    if config.audit.detect_perceptual_duplicates:
        perceptual_groups = _perceptual_groups(
            perceptual_entries,
            config.audit.perceptual_hamming_threshold,
        )
        exact_duplicate_path_groups = {
            frozenset(sample["path"] for sample in group["samples"]) for group in exact_groups
        }
        for group in perceptual_groups:
            duplicate_paths = frozenset(sample["path"] for sample in group["samples"])
            if duplicate_paths in exact_duplicate_path_groups:
                continue
            cross_split = len(group["splits"]) > 1
            issues.append(
                AuditIssue(
                    "error" if cross_split else "warning",
                    "perceptual_duplicate_cross_split" if cross_split else "perceptual_duplicate",
                    "Perceptually similar images occur across splits."
                    if cross_split
                    else "Perceptually similar images occur in one split.",
                    details=_duplicate_details(group),
                )
            )

    total_images = max(1, report.image_count)
    negative_fraction = report.negative_image_count / total_images
    if negative_fraction > config.audit.maximum_negative_fraction:
        issues.append(
            AuditIssue(
                "warning",
                "high_negative_fraction",
                f"Negative-image fraction {negative_fraction:.3f} exceeds configured "
                f"warning threshold {config.audit.maximum_negative_fraction:.3f}.",
            )
        )
    if report.is_imbalanced:
        issues.append(
            AuditIssue(
                "warning",
                "class_imbalance",
                f"Class object-count imbalance ratio is {report.imbalance_ratio:.3f}.",
                details=report.object_counts,
            )
        )
    if not all(
        (
            config.audit.verify_source_group_splits,
            config.audit.detect_exact_duplicates,
            config.audit.detect_perceptual_duplicates,
        )
    ):
        issues.append(
            AuditIssue(
                "warning",
                "source_split_check_disabled",
                "At least one source-group or duplicate check was disabled in configuration.",
            )
        )
    source_integrity = all(
        (
            config.audit.verify_source_group_splits,
            config.audit.detect_exact_duplicates,
            config.audit.detect_perceptual_duplicates,
        )
    ) and not any(
        item.code
        in {
            "source_group_leakage",
            "exact_duplicate_cross_split",
            "perceptual_duplicate_cross_split",
        }
        and item.severity == "error"
        for item in issues
    )
    if config.audit.generate_contact_sheets:
        sheet_paths = _save_contact_sheets(
            report,
            root,
            run_dir / "annotation-review",
            config,
            category_records,
        )
    else:
        sheet_paths = ()
    result = AuditResult(
        valid=not any(item.severity == "error" for item in issues),
        dataset_root=str(root),
        dataset_fingerprint=report.dataset_fingerprint,
        image_count=report.image_count,
        positive_image_count=report.positive_image_count,
        negative_image_count=report.negative_image_count,
        class_counts=report.object_counts,
        split_counts=report.split_counts,
        imbalance_ratio=report.imbalance_ratio,
        exact_duplicate_groups=tuple(exact_groups),
        perceptual_duplicate_groups=tuple(perceptual_groups),
        source_split_integrity=source_integrity,
        license_review_status=config.dataset.license_review_status,
        issues=tuple(issues),
        contact_sheets=sheet_paths,
    )
    _write_reports(run_dir, result, report, category_records)
    return result
