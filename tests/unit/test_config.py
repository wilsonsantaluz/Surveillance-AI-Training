from pathlib import Path

import pytest

from surveillance_ml.core.config import load_config


def test_checked_in_classification_config_is_valid() -> None:
    config = load_config(Path("configs/classification.yaml"))

    assert config.model.architecture == "resnet18"
    assert config.training.batch_size > 0
    split_total = (
        config.dataset.train_fraction
        + config.dataset.validation_fraction
        + config.dataset.test_fraction
    )
    assert split_total == 1


def test_invalid_split_fractions_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
dataset:
  root: data
  version: test
  image_size: 64
  train_fraction: 0.7
  validation_fraction: 0.2
  test_fraction: 0.2
  imbalance_ratio: 1.5
training:
  output_dir: artifacts
  seed: 1
  batch_size: 4
  epochs: 1
  learning_rate: 0.001
  optimizer: adamw
  weight_decay: 0.0
  scheduler_factor: 0.5
  scheduler_patience: 1
  early_stopping_patience: 2
  num_workers: 0
  cpu_threads: 1
  mixed_precision: false
model:
  architecture: resnet18
  pretrained: false
  strategy: head
tracking:
  uri: file:./mlruns
  experiment_name: test
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must sum to 1"):
        load_config(path)
