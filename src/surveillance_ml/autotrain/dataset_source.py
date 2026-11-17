"""Explicit local-only dataset resolution with a pluggable source interface."""

from pathlib import Path
from typing import Protocol

from surveillance_ml.autotrain.config import DatasetConfig


class DatasetSource(Protocol):
    """Resolve a configured dataset without silently downloading remote data."""

    def resolve(self, config: DatasetConfig) -> Path: ...


class LocalDatasetSource:
    """Use a dataset that already exists on the local filesystem."""

    def resolve(self, config: DatasetConfig) -> Path:
        if config.source != "local":
            raise ValueError(f"LocalDatasetSource cannot resolve source {config.source!r}")
        if not config.path.is_dir():
            raise FileNotFoundError(f"Local dataset directory does not exist: {config.path}")
        return config.path.resolve()


def resolve_dataset(config: DatasetConfig, source: DatasetSource | None = None) -> Path:
    """Resolve a dataset through an explicit source implementation."""
    provider = source or LocalDatasetSource()
    return provider.resolve(config)
