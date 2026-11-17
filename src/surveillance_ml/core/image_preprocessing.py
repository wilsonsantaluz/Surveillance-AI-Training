"""Shared deterministic RGB image preprocessing for training and inference."""

from typing import Final

import numpy as np
from PIL import Image, ImageOps

IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)


def resize_rgb_image(image: Image.Image, width: int, height: int) -> Image.Image:
    """Center-crop to the target aspect ratio, then resize with Pillow bilinear."""
    if width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive")
    return ImageOps.fit(
        image.convert("RGB"),
        (width, height),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )


def normalize_rgb_image(image: Image.Image) -> np.ndarray:
    """Convert a resized RGB image into a normalized float32 NCHW array."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / np.float32(255.0)
    normalized = (pixels - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD,
        dtype=np.float32,
    )
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)


def preprocess_image(image: Image.Image, width: int, height: int) -> np.ndarray:
    """Apply the training pipeline's deterministic resize and normalization."""
    resized = resize_rgb_image(image, width, height)
    return normalize_rgb_image(resized)
