from pathlib import Path

import pytest
import torch
from PIL import Image

from surveillance_ml.data.dataset import CLASS_NAMES, SceneImageDataset, discover_images
from surveillance_ml.data.splitting import class_distribution, split_records
from surveillance_ml.data.synthetic import generate_synthetic_dataset


def test_synthetic_dataset_is_loadable_and_splits_by_source(tmp_path: Path) -> None:
    root = generate_synthetic_dataset(tmp_path / "demo", samples_per_class=6, seed=7)
    records = discover_images(root)
    split = split_records(records, seed=9, fractions=(0.6, 0.2, 0.2))

    assert len(records) == 18
    assert {record.class_name for record in records} == set(CLASS_NAMES)
    assert not ({record.source_id for record in split.train} & {r.source_id for r in split.test})
    assert set(class_distribution(split.train, 1.5)["counts"]) == set(CLASS_NAMES)


def test_source_directories_stay_in_one_split(tmp_path: Path) -> None:
    root = tmp_path / "grouped"
    for class_name in CLASS_NAMES:
        for source_index in range(4):
            source = root / class_name / f"video-{source_index}"
            source.mkdir(parents=True)
            for frame_index in range(2):
                Image.new("RGB", (40, 40), (source_index * 20, frame_index * 40, 10)).save(
                    source / f"frame-{frame_index}.png"
                )
    records = discover_images(root)
    split = split_records(records, seed=3, fractions=(0.5, 0.25, 0.25))
    source_to_split: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        for record in getattr(split, split_name):
            previous = source_to_split.setdefault(record.source_id, split_name)
            assert previous == split_name


def test_corrupt_image_fails_with_path(tmp_path: Path) -> None:
    for class_name in CLASS_NAMES:
        class_dir = tmp_path / class_name
        class_dir.mkdir(parents=True)
        (class_dir / "bad.png").write_bytes(b"not an image")

    with pytest.raises(ValueError, match="Corrupt or unreadable image"):
        discover_images(tmp_path)


def test_image_content_must_match_extension(tmp_path: Path) -> None:
    for class_name in CLASS_NAMES:
        (tmp_path / class_name).mkdir(parents=True)
    Image.new("RGB", (40, 40)).save(tmp_path / CLASS_NAMES[0] / "misnamed.jpg", format="PNG")

    with pytest.raises(ValueError, match="does not match its file extension"):
        discover_images(tmp_path)


def test_dataset_normalizes_to_tensor_and_augmentation_is_opt_in(tmp_path: Path) -> None:
    root = generate_synthetic_dataset(tmp_path / "demo", samples_per_class=3)
    record = discover_images(root)[0]
    tensor, label = SceneImageDataset([record], image_size=32)[0]

    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32
    assert 0 <= label < len(CLASS_NAMES)
