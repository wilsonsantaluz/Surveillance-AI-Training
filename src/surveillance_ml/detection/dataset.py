"""YOLO-format detection dataset validation, provenance, and source-group splitting."""

import hashlib
import json
import math
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError

from surveillance_ml.detection import DETECTION_CLASS_NAMES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DetectionRecord:
    image_path: Path
    label_path: Path
    split: str
    source_id: str
    class_ids: tuple[int, ...]


@dataclass(frozen=True)
class DetectionDatasetReport:
    root: Path
    dataset_fingerprint: str
    image_count: int
    positive_image_count: int
    negative_image_count: int
    object_counts: dict[str, int]
    imbalance_ratio: float
    is_imbalanced: bool
    split_counts: dict[str, dict[str, Any]]
    records: tuple[DetectionRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_fingerprint": self.dataset_fingerprint,
            "image_count": self.image_count,
            "positive_image_count": self.positive_image_count,
            "negative_image_count": self.negative_image_count,
            "object_counts": self.object_counts,
            "imbalance_ratio": self.imbalance_ratio,
            "is_imbalanced": self.is_imbalanced,
            "split_counts": self.split_counts,
        }


def _read_dataset_yaml(root: Path) -> tuple[Path, tuple[str, ...]]:
    config_path = root / "dataset.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Detection dataset is missing {config_path}")
    try:
        raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid dataset YAML: {config_path}") from error
    if not isinstance(raw, dict):
        raise ValueError("dataset.yaml must contain a mapping")

    names = raw.get("names")
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError("dataset.yaml names keys must be contiguous integers from zero")
        try:
            class_names = tuple(names[index] for index in range(len(names)))
        except KeyError as error:
            raise ValueError(
                "dataset.yaml names keys must be contiguous integers from zero"
            ) from error
    elif isinstance(names, list) and all(isinstance(name, str) for name in names):
        class_names = tuple(names)
    else:
        raise ValueError("dataset.yaml names must be an ordered list or contiguous integer mapping")
    if class_names != DETECTION_CLASS_NAMES:
        raise ValueError(
            f"Class order must be {DETECTION_CLASS_NAMES}, received {class_names}"
        )
    if raw.get("nc", len(class_names)) != len(class_names):
        raise ValueError("dataset.yaml nc does not match the class names")

    declared_root = raw.get("path", ".")
    if not isinstance(declared_root, str):
        raise ValueError("dataset.yaml path must be a string")
    data_root = (root / declared_root).resolve()
    if data_root != root.resolve():
        raise ValueError("dataset.yaml path must resolve to the configured dataset root")
    for split in SPLITS:
        expected = f"images/{split}"
        if raw.get(split) != expected:
            raise ValueError(f"dataset.yaml {split} must be '{expected}'")
    return config_path, class_names


def _source_id(image_path: Path, image_root: Path) -> str:
    relative = image_path.relative_to(image_root)
    if len(relative.parts) > 1:
        return relative.parts[0]
    return f"sample:{image_root.name}:{relative.as_posix()}"


def _validate_label(
    label_path: Path,
    class_count: int,
) -> tuple[int, ...]:
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"Unable to read annotation {label_path}") from error
    class_ids: list[int] = []
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(
                f"Expected 'class x_center y_center width height' at {label_path}:{line_number}"
            )
        try:
            class_id = int(fields[0])
            x_center, y_center, width, height = (float(value) for value in fields[1:])
        except ValueError as error:
            raise ValueError(f"Invalid YOLO annotation at {label_path}:{line_number}") from error
        if class_id < 0 or class_id >= class_count:
            raise ValueError(f"Class id out of range at {label_path}:{line_number}")
        if not all(math.isfinite(value) for value in (x_center, y_center, width, height)):
            raise ValueError(f"Non-finite box coordinates at {label_path}:{line_number}")
        if width <= 0 or height <= 0 or not (
            0 <= x_center <= 1
            and 0 <= y_center <= 1
            and x_center - width / 2 >= 0
            and y_center - height / 2 >= 0
            and x_center + width / 2 <= 1
            and y_center + height / 2 <= 1
        ):
            raise ValueError(f"Box must be positive and normalized inside the image: {label_path}")
        class_ids.append(class_id)
    return tuple(class_ids)


def _fingerprint(root: Path, files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def validate_yolo_dataset(
    root: Path,
    *,
    imbalance_threshold: float = 1.5,
) -> DetectionDatasetReport:
    """Validate labels, image bytes, required splits, and source leakage."""
    if not math.isfinite(imbalance_threshold) or imbalance_threshold < 1:
        raise ValueError("imbalance_threshold must be at least one")
    root = root.resolve()
    config_path, class_names = _read_dataset_yaml(root)
    records: list[DetectionRecord] = []
    split_counts: dict[str, dict[str, Any]] = {}
    seen_sources: dict[str, str] = {}
    hashed_files = [config_path]
    object_counts: Counter[str] = Counter()
    positive_count = 0

    for split in SPLITS:
        image_root = root / "images" / split
        label_root = root / "labels" / split
        if not image_root.is_dir() or not label_root.is_dir():
            raise FileNotFoundError(f"Dataset split {split} needs image and label directories")
        image_files = [path for path in image_root.rglob("*") if path.is_file()]
        unsupported_images = [
            path for path in image_files if path.suffix.lower() not in IMAGE_EXTENSIONS
        ]
        if unsupported_images:
            raise ValueError(f"Unsupported image format: {unsupported_images[0]}")
        image_paths = sorted(image_files)
        if not image_paths:
            raise ValueError(f"Dataset split {split} contains no supported images")
        image_relatives = {path.relative_to(image_root).with_suffix(".txt") for path in image_paths}
        if len(image_relatives) != len(image_paths):
            raise ValueError(f"Image filenames collide on annotation paths in split {split}")
        all_label_files = [
            path
            for path in label_root.rglob("*")
            if path.is_file() and path.suffix.lower() != ".cache"
        ]
        unsupported_labels = [path for path in all_label_files if path.suffix.lower() != ".txt"]
        if unsupported_labels:
            raise ValueError(f"Unsupported annotation format: {unsupported_labels[0]}")
        label_paths = {
            path.relative_to(label_root)
            for path in all_label_files
        }
        missing_labels = image_relatives - label_paths
        orphan_labels = label_paths - image_relatives
        if missing_labels or orphan_labels:
            raise ValueError(
                f"Image/label pair mismatch in {split}: "
                f"{len(missing_labels)} missing labels, {len(orphan_labels)} orphan labels"
            )

        per_class = Counter[str]()
        split_sources: set[str] = set()
        split_positive = 0
        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"Corrupt or unreadable image: {image_path}") from error
            relative = image_path.relative_to(image_root)
            label_path = label_root / relative.with_suffix(".txt")
            class_ids = _validate_label(label_path, len(class_names))
            source_id = _source_id(image_path, image_root)
            previous_split = seen_sources.setdefault(source_id, split)
            if previous_split != split:
                raise ValueError(
                    f"Source group {source_id!r} appears in both {previous_split} and {split}"
                )
            split_sources.add(source_id)
            split_positive += bool(class_ids)
            for class_id in class_ids:
                per_class[class_names[class_id]] += 1
                object_counts[class_names[class_id]] += 1
            records.append(DetectionRecord(image_path, label_path, split, source_id, class_ids))
            hashed_files.extend((image_path, label_path))
        split_counts[split] = {
            "images": len(image_paths),
            "positive_images": split_positive,
            "negative_images": len(image_paths) - split_positive,
            "source_groups": len(split_sources),
            "objects": {name: per_class.get(name, 0) for name in class_names},
        }
        if any(per_class.get(name, 0) == 0 for name in class_names):
            raise ValueError(f"Each split must contain labeled objects from both classes: {split}")
        positive_count += split_positive

    nonzero_counts = [object_counts.get(name, 0) for name in class_names]
    imbalance_ratio = max(nonzero_counts) / min(nonzero_counts)
    return DetectionDatasetReport(
        root=root,
        dataset_fingerprint=_fingerprint(root, hashed_files),
        image_count=len(records),
        positive_image_count=positive_count,
        negative_image_count=len(records) - positive_count,
        object_counts={name: object_counts.get(name, 0) for name in class_names},
        imbalance_ratio=imbalance_ratio,
        is_imbalanced=imbalance_ratio > imbalance_threshold,
        split_counts=split_counts,
        records=tuple(records),
    )


def split_yolo_sources(
    source_root: Path,
    output_root: Path,
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.7, 0.2, 0.1),
) -> Path:
    """Copy complete source directories into deterministic train/val/test partitions."""
    if len(fractions) != 3 or any(not math.isfinite(value) or value <= 0 for value in fractions):
        raise ValueError("Exactly three positive split fractions are required")
    if abs(sum(fractions) - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1")
    source_images = source_root / "images"
    source_labels = source_root / "labels"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError("Raw dataset needs images/<source_id> and labels/<source_id>")
    groups = sorted(path for path in source_images.iterdir() if path.is_dir())
    if len(groups) < 3:
        raise ValueError("At least three source groups are required for a three-way split")
    image_group_names = {path.name for path in groups}
    label_group_names = {path.name for path in source_labels.iterdir() if path.is_dir()}
    if image_group_names != label_group_names:
        raise ValueError("Image and label source groups do not match")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset directory: {output_root}")

    rng = random.Random(seed)
    rng.shuffle(groups)
    remaining = len(groups) - len(SPLITS)
    desired = [remaining * fraction for fraction in fractions]
    extras = [int(value) for value in desired]
    for index in sorted(range(3), key=lambda item: desired[item] - extras[item], reverse=True)[
        : remaining - sum(extras)
    ]:
        extras[index] += 1
    counts = [1 + value for value in extras]
    first_boundary = counts[0]
    second_boundary = counts[0] + counts[1]
    grouped = {
        "train": groups[:first_boundary],
        "val": groups[first_boundary:second_boundary],
        "test": groups[second_boundary:],
    }
    if any(not members for members in grouped.values()):
        raise ValueError("Source count and split fractions must produce three non-empty splits")

    for split, members in grouped.items():
        for source in members:
            source_id = source.name
            image_source = source_images / source_id
            label_source = source_labels / source_id
            images = sorted(
                path for path in image_source.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not images:
                raise ValueError(f"Source group {source_id!r} contains no supported images")
            expected_labels = {
                path.relative_to(image_source).with_suffix(".txt") for path in images
            }
            actual_labels = {
                path.relative_to(label_source)
                for path in label_source.rglob("*.txt")
                if path.is_file()
            }
            if expected_labels != actual_labels:
                raise ValueError(f"Image/label pair mismatch in source group {source_id!r}")
            for image_path in images:
                relative = image_path.relative_to(image_source)
                label_path = label_source / relative.with_suffix(".txt")
                if not label_path.is_file():
                    raise ValueError(f"Missing annotation for {image_path}")
                image_target = output_root / "images" / split / source_id / relative
                label_target = (
                    output_root / "labels" / split / source_id / relative.with_suffix(".txt")
                )
                image_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, image_target)
                shutil.copy2(label_path, label_target)

    output_root.mkdir(parents=True, exist_ok=True)
    config = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(DETECTION_CLASS_NAMES),
        "names": list(DETECTION_CLASS_NAMES),
    }
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    validate_yolo_dataset(output_root)
    return output_root / "dataset.yaml"


def write_validation_report(report: DetectionDatasetReport, destination: Path) -> None:
    """Persist the validation summary without image or annotation contents."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
