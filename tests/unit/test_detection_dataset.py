import shutil
from pathlib import Path

import pytest
import yaml

from surveillance_ml.detection.dataset import validate_yolo_dataset
from surveillance_ml.detection.synthetic import generate_synthetic_detection_dataset


def test_synthetic_detection_data_is_valid_and_source_split(tmp_path: Path) -> None:
    dataset_yaml = generate_synthetic_detection_dataset(
        tmp_path / "detection",
        source_count=10,
        image_size=64,
        seed=17,
    )
    report = validate_yolo_dataset(dataset_yaml.parent)

    assert report.image_count == 30
    assert report.positive_image_count == 20
    assert report.negative_image_count == 10
    assert all(report.object_counts[name] > 0 for name in ("firearm", "knife"))
    assert all(report.split_counts[name]["images"] > 0 for name in ("train", "val", "test"))


def test_source_group_leakage_is_rejected(tmp_path: Path) -> None:
    root = generate_synthetic_detection_dataset(
        tmp_path / "detection",
        source_count=9,
        image_size=64,
        seed=9,
    ).parent
    source_id = next((root / "images" / "train").iterdir()).name
    shutil.copytree(
        root / "images" / "train" / source_id,
        root / "images" / "val" / source_id,
    )
    shutil.copytree(
        root / "labels" / "train" / source_id,
        root / "labels" / "val" / source_id,
    )

    with pytest.raises(ValueError, match="appears in both"):
        validate_yolo_dataset(root)


def test_invalid_normalized_annotation_is_rejected(tmp_path: Path) -> None:
    root = generate_synthetic_detection_dataset(
        tmp_path / "detection",
        source_count=9,
        image_size=64,
        seed=11,
    ).parent
    annotation = next((root / "labels" / "train").rglob("*.txt"))
    annotation.write_text("0 1.2 0.5 0.1 0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="normalized inside"):
        validate_yolo_dataset(root)


def test_source_split_is_reproducible(tmp_path: Path) -> None:
    first = generate_synthetic_detection_dataset(
        tmp_path / "first",
        source_count=9,
        image_size=64,
        seed=3,
    ).parent
    second = generate_synthetic_detection_dataset(
        tmp_path / "second",
        source_count=9,
        image_size=64,
        seed=3,
    ).parent

    def assignments(root: Path) -> dict[str, list[str]]:
        return {
            split: sorted(path.name for path in (root / "images" / split).iterdir())
            for split in ("train", "val", "test")
        }

    assert assignments(first) == assignments(second)


def test_unsupported_image_and_noncontiguous_class_mapping_are_rejected(
    tmp_path: Path,
) -> None:
    root = generate_synthetic_detection_dataset(
        tmp_path / "detection",
        source_count=9,
        image_size=64,
        seed=21,
    ).parent
    (root / "images" / "train" / "unsupported.gif").write_bytes(b"not an image")

    with pytest.raises(ValueError, match="Unsupported image format"):
        validate_yolo_dataset(root)

    (root / "images" / "train" / "unsupported.gif").unlink()
    dataset_yaml = root / "dataset.yaml"
    configuration = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    configuration["names"] = {0: "firearm", 2: "knife"}
    dataset_yaml.write_text(yaml.safe_dump(configuration), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous integers"):
        validate_yolo_dataset(root)
