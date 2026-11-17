"""Stable experiment and dataset fingerprints plus runtime provenance."""

import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from surveillance_ml.data.dataset import SUPPORTED_EXTENSIONS, ImageRecord

LOGGER = logging.getLogger(__name__)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def configuration_fingerprint(config: object) -> str:
    """Hash the canonical JSON representation of a configuration."""
    payload = json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dataset_fingerprint(root: Path) -> str:
    """Hash sorted relative image paths and file contents without storing image data."""
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"No supported image files found under {root}")
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as image_file:
            for block in iter(lambda: image_file.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def split_fingerprint(splits: dict[str, list[ImageRecord]], root: Path) -> str:
    """Hash split membership and labels while excluding absolute local paths."""
    digest = hashlib.sha256()
    for split_name in sorted(splits):
        digest.update(split_name.encode("utf-8"))
        digest.update(b"\n")
        entries = sorted(
            (record.path.relative_to(root).as_posix(), record.class_name, record.source_id)
            for record in splits[split_name]
        )
        for path, class_name, source_id in entries:
            digest.update(f"{path}\0{class_name}\0{source_id}\n".encode())
    return digest.hexdigest()


def git_commit_sha(repository_root: Path) -> str | None:
    """Return the current repository commit when Git metadata is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        LOGGER.warning("Could not read Git revision: %s", error)
        return None
    return result.stdout.strip()


def package_version(package_name: str) -> str:
    """Return an installed package version or an explicit not-installed marker."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def runtime_versions() -> dict[str, str]:
    """Collect environment versions relevant to training and MLOps."""
    return {
        "python": sys.version.split()[0],
        "torch": package_version("torch"),
        "torchvision": package_version("torchvision"),
        "optuna": package_version("optuna"),
        "mlflow": package_version("mlflow"),
        "dvc": package_version("dvc"),
    }
