"""Import the Zenodo Dangerous Items Pascal VOC release into grouped YOLO splits."""

import hashlib
import json
import math
import random
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import numpy as np
from PIL import Image, UnidentifiedImageError

from surveillance_ml.detection import DETECTION_CLASS_NAMES
from surveillance_ml.detection.dataset import validate_yolo_dataset

ZENODO_RECORD_URL = "https://zenodo.org/records/11582054"
ZENODO_API_URL = "https://zenodo.org/api/records/11582054"
ZENODO_DOI = "10.5281/zenodo.11582054"
VOC_CLASS_MAP = {"gun": 0, "rifle": 0, "knife": 1, "machete": 1}
UNSUPPORTED_VOC_CLASSES = {"baseball bat"}


@dataclass(frozen=True)
class _VocObject:
    source_name: str
    class_id: int | None
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    difficult: bool

    def signature(self) -> tuple[int, float, float, float, float] | None:
        if self.class_id is None:
            return None
        return (self.class_id, self.xmin, self.ymin, self.xmax, self.ymax)

    def to_yolo(self, width: int, height: int) -> str:
        if self.class_id is None:
            raise ValueError("An unsupported VOC class cannot be converted to YOLO")
        center_x = (self.xmin + self.xmax) / (2 * width)
        center_y = (self.ymin + self.ymax) / (2 * height)
        box_width = (self.xmax - self.xmin) / width
        box_height = (self.ymax - self.ymin) / height
        x_values = _format_normalized_box(center_x, box_width)
        y_values = _format_normalized_box(center_y, box_height)
        return (
            f"{self.class_id} {x_values[0]} {y_values[0]} "
            f"{x_values[1]} {y_values[1]}"
        )


@dataclass
class _VocRecord:
    stem: str
    image_member: str
    xml_member: str
    original_split: str
    width: int
    height: int
    objects: tuple[_VocObject, ...]
    source_metadata: dict[str, str]
    image_sha256: str
    perceptual_hash: int
    excluded_reason: str | None = None

    @property
    def class_ids(self) -> set[int]:
        return {obj.class_id for obj in self.objects if obj.class_id is not None}

    @property
    def signature(self) -> tuple[tuple[int, float, float, float, float], ...]:
        return tuple(
            sorted(
                (obj.class_id, obj.xmin, obj.ymin, obj.xmax, obj.ymax)
                for obj in self.objects
                if obj.class_id is not None
            )
        )


@dataclass
class _HashNode:
    value: int
    record_indexes: list[int] = field(default_factory=list)
    children: dict[int, "_HashNode"] = field(default_factory=dict)


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self._parents = list(range(count))

    def find(self, value: int) -> int:
        while self._parents[value] != value:
            self._parents[value] = self._parents[self._parents[value]]
            value = self._parents[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parents[right_root] = left_root


def _format_normalized_box(center: float, length: float) -> tuple[str, str]:
    precision = 12
    center_text = f"{center:.{precision}f}"
    length_text = f"{length:.{precision}f}"
    serialized_center = float(center_text)
    serialized_length = float(length_text)
    if (
        serialized_center - serialized_length / 2 < 0
        or serialized_center + serialized_length / 2 > 1
    ):
        serialized_length = max(0.0, serialized_length - 10**-precision)
        length_text = f"{serialized_length:.{precision}f}"
    return center_text, length_text


def _perceptual_hash(image_bytes: bytes) -> int:
    import cv2

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        with Image.open(BytesIO(image_bytes)) as image:
            grayscale = np.asarray(
                image.convert("L").resize((32, 32), Image.Resampling.LANCZOS),
                dtype=np.float32,
            )
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError("Dataset contains an unreadable image") from error
    low_frequencies = cv2.dct(grayscale)[:8, :8].reshape(-1)
    median = float(np.median(low_frequencies[1:]))
    return sum(int(value > median) << bit for bit, value in enumerate(low_frequencies))


def _query_hash_tree(node: _HashNode | None, value: int, radius: int) -> list[int]:
    matches: list[int] = []
    pending = [node] if node is not None else []
    while pending:
        current = pending.pop()
        distance = (current.value ^ value).bit_count()
        if distance <= radius:
            matches.extend(current.record_indexes)
        minimum = max(0, distance - radius)
        maximum = distance + radius
        pending.extend(
            child
            for edge, child in current.children.items()
            if minimum <= edge <= maximum
        )
    return matches


def _insert_hash_tree(node: _HashNode | None, value: int, record_index: int) -> _HashNode:
    if node is None:
        return _HashNode(value, [record_index])
    current = node
    while True:
        distance = (current.value ^ value).bit_count()
        if distance == 0:
            current.record_indexes.append(record_index)
            return node
        child = current.children.get(distance)
        if child is None:
            current.children[distance] = _HashNode(value, [record_index])
            return node
        current = child


def _source_groups(records: list[_VocRecord], distance: int) -> list[list[_VocRecord]]:
    sets = _DisjointSet(len(records))
    tree: _HashNode | None = None
    for index, record in enumerate(records):
        for matching_index in _query_hash_tree(tree, record.perceptual_hash, distance):
            sets.union(index, matching_index)
        tree = _insert_hash_tree(tree, record.perceptual_hash, index)
    grouped: dict[int, list[_VocRecord]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[sets.find(index)].append(record)
    return sorted(grouped.values(), key=lambda group: min(record.stem for record in group))


def _read_voc_records(archive: ZipFile) -> tuple[list[_VocRecord], dict[str, set[str]]]:
    members = [name for name in archive.namelist() if not name.endswith("/")]
    split_member = next(
        (name for name in members if name.endswith("/datasplit/train.txt")),
        None,
    )
    if split_member is None:
        raise ValueError("Archive is missing datasplit/train.txt")
    root = split_member[: -len("/datasplit/train.txt")]
    split_ids = {
        split: set(archive.read(f"{root}/datasplit/{split}.txt").decode("utf-8-sig").split())
        for split in ("train", "test")
    }

    image_members = {
        PurePosixPath(name).stem: name
        for name in members
        if name.startswith(f"{root}/images/") and PurePosixPath(name).suffix.lower() == ".jpg"
    }
    annotation_members = {
        PurePosixPath(name).stem: name
        for name in members
        if name.startswith(f"{root}/annotations/")
        and PurePosixPath(name).suffix.lower() == ".xml"
    }
    if len(image_members) != sum(
        name.startswith(f"{root}/images/") and PurePosixPath(name).suffix.lower() == ".jpg"
        for name in members
    ):
        raise ValueError("Dataset has duplicate JPEG image identifiers")
    if len(annotation_members) != sum(
        name.startswith(f"{root}/annotations/")
        and PurePosixPath(name).suffix.lower() == ".xml"
        for name in members
    ):
        raise ValueError("Dataset has duplicate annotation identifiers")
    if set(image_members) != set(annotation_members):
        raise ValueError("Image and Pascal VOC annotation identifiers do not match")
    all_ids = set(image_members)
    if (
        split_ids["train"] & split_ids["test"]
        or split_ids["train"] | split_ids["test"] != all_ids
    ):
        raise ValueError("Publisher train/test lists must be disjoint and cover all images")

    records: list[_VocRecord] = []
    for stem in sorted(all_ids):
        xml_member = annotation_members[stem]
        try:
            annotation = ET.fromstring(archive.read(xml_member))
            width = int(annotation.findtext("size/width") or "0")
            height = int(annotation.findtext("size/height") or "0")
        except (ET.ParseError, ValueError) as error:
            raise ValueError(f"Invalid Pascal VOC annotation: {xml_member}") from error
        if width < 1 or height < 1:
            raise ValueError(f"Invalid image dimensions in {xml_member}")

        source = annotation.find("source")
        source_metadata = {
            key: (source.findtext(key) or "").strip() if source is not None else ""
            for key in ("database", "annotation", "image", "flickrid")
        }
        objects: list[_VocObject] = []
        invalid_box = False
        has_difficult = False
        for item in annotation.findall("object"):
            source_name = (item.findtext("name") or "").strip()
            normalized_name = source_name.casefold()
            if normalized_name in VOC_CLASS_MAP:
                class_id: int | None = VOC_CLASS_MAP[normalized_name]
            elif normalized_name in UNSUPPORTED_VOC_CLASSES:
                class_id = None
            else:
                raise ValueError(f"Unknown VOC class {source_name!r} in {xml_member}")
            box = item.find("bndbox")
            if box is None:
                invalid_box = True
                continue
            try:
                xmin, ymin, xmax, ymax = (
                    float(box.findtext(key) or "nan")
                    for key in ("xmin", "ymin", "xmax", "ymax")
                )
            except ValueError as error:
                raise ValueError(f"Non-numeric VOC box in {xml_member}") from error
            if not all(math.isfinite(value) for value in (xmin, ymin, xmax, ymax)) or not (
                0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height
            ):
                invalid_box = True
            difficult = (item.findtext("difficult") or "0").strip() == "1"
            has_difficult = has_difficult or difficult
            objects.append(
                _VocObject(source_name, class_id, xmin, ymin, xmax, ymax, difficult)
            )
        if invalid_box:
            excluded_reason = "invalid_box"
        elif has_difficult:
            excluded_reason = "difficult_annotation"
        elif any(item.class_id is None for item in objects):
            excluded_reason = "unsupported_baseball_bat"
        elif not any(item.class_id is not None for item in objects):
            excluded_reason = "no_target_objects"
        else:
            excluded_reason = None

        image_member = image_members[stem]
        image_bytes = archive.read(image_member)
        with Image.open(BytesIO(image_bytes)) as image:
            if image.size != (width, height):
                raise ValueError(f"Image dimensions disagree with annotation: {image_member}")
        records.append(
            _VocRecord(
                stem=stem,
                image_member=image_member,
                xml_member=xml_member,
                original_split="train" if stem in split_ids["train"] else "test",
                width=width,
                height=height,
                objects=tuple(objects),
                source_metadata=source_metadata,
                image_sha256=hashlib.sha256(image_bytes).hexdigest(),
                perceptual_hash=_perceptual_hash(image_bytes),
                excluded_reason=excluded_reason,
            )
        )
    return records, split_ids


def _validation_groups(
    groups: list[dict[str, Any]],
    fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(groups) < 4:
        raise ValueError("At least four training source groups are required")
    rng = random.Random(seed)
    shuffled = groups.copy()
    rng.shuffle(shuffled)
    validation_count = min(
        len(shuffled) - 2,
        max(len(DETECTION_CLASS_NAMES), round(len(shuffled) * fraction)),
    )
    validation: list[dict[str, Any]] = []
    for class_id in range(len(DETECTION_CLASS_NAMES)):
        if any(class_id in group["class_ids"] for group in validation):
            continue
        candidates = [
            group
            for group in shuffled
            if group not in validation and class_id in group["class_ids"]
        ]
        if not candidates:
            raise ValueError(f"Training source groups do not contain class id {class_id}")
        validation.append(candidates[0])
    validation.extend(group for group in shuffled if group not in validation)
    selected = validation[:validation_count]
    selected_ids = {group["group_id"] for group in selected}
    training = [group for group in shuffled if group["group_id"] not in selected_ids]
    for class_id in range(len(DETECTION_CLASS_NAMES)):
        if not any(class_id in group["class_ids"] for group in selected):
            raise ValueError(f"Validation split has no {DETECTION_CLASS_NAMES[class_id]} objects")
        if not any(class_id in group["class_ids"] for group in training):
            raise ValueError(f"Training split has no {DETECTION_CLASS_NAMES[class_id]} objects")
    return training, selected


def import_zenodo_pascal_voc(
    archive_path: Path,
    output_root: Path,
    *,
    seed: int = 42,
    validation_fraction: float = 0.15,
    perceptual_distance: int = 2,
) -> dict[str, Any]:
    """Convert the published archive, deduplicate, and keep perceptual groups split-safe."""
    if not math.isfinite(validation_fraction) or not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between zero and one half")
    if not 0 <= perceptual_distance <= 8:
        raise ValueError("perceptual_distance must be between zero and eight")
    archive_path = archive_path.resolve()
    output_root = output_root.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive does not exist: {archive_path}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite dataset output: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}-",
        dir=output_root.parent,
    ) as temp:
        temporary_root = Path(temp)
        with ZipFile(archive_path) as archive:
            records, split_ids = _read_voc_records(archive)
            groups: list[dict[str, Any]] = []
            excluded: list[dict[str, str]] = []
            for members in _source_groups(records, perceptual_distance):
                group_id = hashlib.sha256(
                    "\n".join(sorted(record.stem for record in members)).encode("utf-8")
                ).hexdigest()[:16]
                reasons = {record.excluded_reason for record in members if record.excluded_reason}
                duplicate_signatures: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
                for record in members:
                    duplicate_signatures[record.image_sha256].add(record.signature)
                if any(len(signatures) > 1 for signatures in duplicate_signatures.values()):
                    reasons.add("duplicate_annotation_conflict")
                if reasons:
                    reason = ",".join(sorted(reasons))
                    excluded.extend(
                        {"image_id": record.stem, "reason": reason, "source_group": group_id}
                        for record in members
                    )
                    continue

                split = (
                    "test"
                    if any(record.stem in split_ids["test"] for record in members)
                    else "train"
                )
                canonical_by_hash: dict[str, _VocRecord] = {}
                for record in members:
                    previous = canonical_by_hash.get(record.image_sha256)
                    if previous is None or (
                        record.original_split == split and previous.original_split != split
                    ) or (
                        record.original_split == previous.original_split
                        and record.stem < previous.stem
                    ):
                        canonical_by_hash[record.image_sha256] = record
                canonical_ids = {record.stem for record in canonical_by_hash.values()}
                for record in members:
                    if record.stem not in canonical_ids:
                        excluded.append(
                            {
                                "image_id": record.stem,
                                "reason": "exact_duplicate",
                                "source_group": group_id,
                            }
                        )
                retained = list(canonical_by_hash.values())
                groups.append(
                    {
                        "group_id": group_id,
                        "split": split,
                        "records": retained,
                        "class_ids": set().union(*(record.class_ids for record in retained)),
                    }
                )

            test_groups = [group for group in groups if group["split"] == "test"]
            original_train_groups = [group for group in groups if group["split"] == "train"]
            train_groups, validation_groups = _validation_groups(
                original_train_groups,
                validation_fraction,
                seed,
            )
            assignments = {
                group["group_id"]: split
                for split, split_groups in (
                    ("train", train_groups),
                    ("val", validation_groups),
                    ("test", test_groups),
                )
                for group in split_groups
            }
            split_groups = {
                split: [
                    group
                    for group in groups
                    if assignments.get(group["group_id"]) == split
                ]
                for split in ("train", "val", "test")
            }

            converted_records: list[dict[str, Any]] = []
            split_counts: dict[str, dict[str, int]] = {}
            for split, selected_groups in split_groups.items():
                image_count = 0
                object_counts: Counter[str] = Counter()
                for group in selected_groups:
                    group_id = group["group_id"]
                    for record in sorted(group["records"], key=lambda item: item.stem):
                        image_bytes = archive.read(record.image_member)
                        image_path = (
                            temporary_root / "images" / split / group_id / f"{record.stem}.jpg"
                        )
                        label_path = (
                            temporary_root / "labels" / split / group_id / f"{record.stem}.txt"
                        )
                        image_path.parent.mkdir(parents=True, exist_ok=True)
                        label_path.parent.mkdir(parents=True, exist_ok=True)
                        image_path.write_bytes(image_bytes)
                        label_lines = [
                            item.to_yolo(record.width, record.height)
                            for item in record.objects
                            if item.class_id is not None
                        ]
                        label_path.write_text(
                            "\n".join(label_lines) + "\n",
                            encoding="utf-8",
                        )
                        for item in record.objects:
                            if item.class_id is not None:
                                object_counts[DETECTION_CLASS_NAMES[item.class_id]] += 1
                        image_count += 1
                        converted_records.append(
                            {
                                "image_id": record.stem,
                                "source_image": record.image_member,
                                "source_annotation": record.xml_member,
                                "original_split": record.original_split,
                                "split": split,
                                "source_group": group_id,
                                "image_sha256": record.image_sha256,
                                "perceptual_hash": f"{record.perceptual_hash:016x}",
                                "source_metadata": record.source_metadata,
                                "source_classes": sorted(
                                    {item.source_name for item in record.objects}
                                ),
                                "target_classes": sorted(
                                    DETECTION_CLASS_NAMES[class_id]
                                    for class_id in record.class_ids
                                ),
                            }
                        )
                split_counts[split] = {
                    "images": image_count,
                    "firearm_boxes": object_counts["firearm"],
                    "knife_boxes": object_counts["knife"],
                    "source_groups": len(selected_groups),
                }

        dataset_config = {
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "nc": len(DETECTION_CLASS_NAMES),
            "names": list(DETECTION_CLASS_NAMES),
        }
        (temporary_root / "dataset.yaml").write_text(
            json.dumps(dataset_config, indent=2) + "\n",
            encoding="utf-8",
        )
        archive_hash = hashlib.sha256()
        archive_md5 = hashlib.md5(usedforsecurity=False)
        with archive_path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                archive_hash.update(block)
                archive_md5.update(block)
        provenance = {
            "source": {
                "record_url": ZENODO_RECORD_URL,
                "api_url": ZENODO_API_URL,
                "doi": ZENODO_DOI,
                "license": "CC BY 4.0 (Zenodo record metadata)",
                "archive": archive_path.name,
                "archive_sha256": archive_hash.hexdigest(),
                "archive_md5": archive_md5.hexdigest(),
            },
            "conversion": {
                "format": "Pascal VOC XML to YOLO detection labels",
                "class_mapping": {
                    "Gun": "firearm",
                    "Rifle": "firearm",
                    "Knife": "knife",
                    "Machete": "knife",
                },
                "unsupported_class_policy": (
                    "Exclude the entire perceptual source group if it contains Baseball Bat."
                ),
                "coordinate_policy": (
                    "Use annotation coordinates as zero-based half-open pixel extents; "
                    "retain zero-valued xmin/ymin and do not subtract one."
                ),
                "source_group_policy": (
                    f"Connect records with 64-bit DCT perceptual-hash Hamming distance "
                    f"<= {perceptual_distance}; assign each connected group to the "
                    "publisher test split if any member is test, otherwise keep it in "
                    "the publisher train pool. Exact duplicate images with conflicting "
                    "target boxes are excluded."
                ),
                "validation_policy": (
                    f"Seed {seed}; select {validation_fraction:.6g} of eligible publisher "
                    "training source groups for validation."
                ),
                "external_source_identifiers": (
                    "Pascal VOC source fields identify COCO2017/Flickr, but all 4,000 "
                    "flickrid values are NULL; original per-image URLs and license IDs "
                    "are unavailable in this archive."
                ),
            },
            "source_image_count": len(records),
            "publisher_split_counts": {
                "train": len(split_ids["train"]),
                "test": len(split_ids["test"]),
            },
            "converted_split_counts": split_counts,
            "excluded_records": excluded,
            "converted_records": converted_records,
        }
        (temporary_root / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n",
            encoding="utf-8",
        )

        report = validate_yolo_dataset(temporary_root)
        validation = report.to_dict()
        (temporary_root / "validation-report.json").write_text(
            json.dumps(validation, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_root.replace(output_root)
    return {
        "output_root": str(output_root),
        "source_image_count": len(records),
        "converted_split_counts": split_counts,
        "excluded_reason_counts": dict(Counter(item["reason"] for item in excluded)),
        "validation": validation,
    }
