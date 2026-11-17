"""Seeded geometric fixtures for testing object-detection pipelines only."""

import random
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from surveillance_ml.detection import DETECTION_CLASS_NAMES
from surveillance_ml.detection.dataset import split_yolo_sources


def _draw_fixture(
    image: Image.Image,
    class_name: str | None,
    rng: random.Random,
) -> tuple[float, float, float, float] | None:
    if class_name is None:
        return None
    draw = ImageDraw.Draw(image)
    width, height = image.size
    box_width = rng.randint(width // 4, width // 3)
    box_height = rng.randint(height // 8, height // 5)
    left = rng.randint(width // 6, width - box_width - width // 6)
    top = rng.randint(height // 4, height - box_height - height // 4)
    right = left + box_width
    bottom = top + box_height
    color = (35, 115, 205) if class_name == "firearm" else (230, 130, 35)

    if class_name == "firearm":
        middle = (top + bottom) // 2
        draw.rectangle((left, top, right - 8, bottom), fill=color)
        draw.rectangle((right - 8, middle - 3, right + 10, middle + 3), fill=color)
        draw.polygon(
            ((left + 8, bottom - 2), (left + 22, bottom - 2), (left + 18, bottom + 12)),
            fill=color,
        )
        x1, y1, x2, y2 = left, top, right + 10, bottom + 12
    elif class_name == "knife":
        middle = (top + bottom) // 2
        draw.rectangle((left, middle - 3, left + box_width // 3, middle + 3), fill=color)
        draw.polygon(
            (
                (left + box_width // 3, top),
                (right, middle),
                (left + box_width // 3, bottom),
            ),
            fill=color,
        )
        x1, y1, x2, y2 = left, top, right, bottom
    else:
        raise ValueError(f"Unsupported synthetic class: {class_name}")

    return (
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def generate_synthetic_detection_dataset(
    output_root: Path,
    *,
    source_count: int = 12,
    image_size: int = 128,
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.7, 0.2, 0.1),
) -> Path:
    """Create colored geometric positives and blank negatives, grouped by source."""
    if source_count < 3:
        raise ValueError("At least three source groups are required")
    if image_size < 64 or image_size > 2048:
        raise ValueError("image_size must be between 64 and 2048 pixels")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset directory: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory(prefix=".detection-source-", dir=output_root.parent) as temp:
        source_root = Path(temp)
        for source_index in range(source_count):
            source_id = f"source_{source_index:04d}"
            image_dir = source_root / "images" / source_id
            label_dir = source_root / "labels" / source_id
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            class_order: list[str | None] = [None, *DETECTION_CLASS_NAMES]
            rng.shuffle(class_order)
            for frame_index, class_name in enumerate(class_order):
                width = image_size + rng.randrange(0, max(1, image_size // 8))
                height = image_size + rng.randrange(0, max(1, image_size // 8))
                background = tuple(rng.randint(185, 235) for _ in range(3))
                image = Image.new("RGB", (width, height), background)
                normalized_box = _draw_fixture(image, class_name, rng)
                image_name = f"frame_{frame_index:03d}.png"
                image.save(image_dir / image_name)
                label_path = label_dir / f"frame_{frame_index:03d}.txt"
                if class_name is None or normalized_box is None:
                    label_path.write_text("", encoding="utf-8")
                    continue
                class_id = DETECTION_CLASS_NAMES.index(class_name)
                coordinates = " ".join(f"{value:.8f}" for value in normalized_box)
                label_path.write_text(f"{class_id} {coordinates}\n", encoding="utf-8")

        return split_yolo_sources(
            source_root,
            output_root,
            seed=seed,
            fractions=fractions,
        )
