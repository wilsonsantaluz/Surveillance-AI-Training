# Surveillance AI Training Lab

A reproducible computer-vision training foundation built with PyTorch, TorchVision, and local MLflow. Phase 1 provides a folder-based image dataset, deterministic group-aware splits, a configurable ResNet18 classifier, evaluation, and an explicitly synthetic CPU demo.

## Requirements

Python 3.12 or newer (up to 3.14), plus [uv](https://docs.astral.sh/uv/). Dependencies are resolved in `uv.lock`.

```powershell
uv sync --extra dev
uv run python scripts/generate_demo_dataset.py --output-dir data/demo --samples-per-class 24
uv run python scripts/train_classifier.py --config configs/classification.yaml --no-pretrained
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000` to inspect the local experiment. The run writes the best validation-loss checkpoint and evaluation report under `artifacts/`. The checked-in configuration enables TorchVision's public ResNet18 ImageNet weights. Pass `--no-pretrained` for an offline synthetic demo; this initializes the same ResNet18 architecture randomly. Synthetic metrics are not a real-world model result. To use the standard transfer-learning configuration, omit `--no-pretrained` and provide a correctly labeled dataset in the documented folder format.

## Dataset format

Each class directory contains supported image files (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`). Nested directories are treated as source groups: put frames from one video under one source directory so they cannot cross split boundaries. Directly stored files are treated as distinct sources. The loader verifies image contents before training and rejects missing classes, unsupported formats, and corrupt images.

```text
data/my_dataset/
├── empty_scene/
├── single_person/
└── multiple_people/
```

Use the generator only for development and pipeline checks. Its simple colored silhouettes are not representative of camera footage and provide no evidence of real-world accuracy. Do not commit generated datasets, private images, or trained checkpoints.

## Training configuration

`configs/classification.yaml` configures the dataset/output paths, image size, split fractions, seed, model weights and fine-tuning strategy (`head`, `partial`, or `full`), optimizer, batch size, learning rate, scheduler, early stopping, and MLflow tracking. The default local backend uses SQLite; override it with `MLFLOW_TRACKING_URI` if needed. Augmentation runs only in the training dataset. CPU is selected when CUDA is unavailable; mixed precision is enabled only on CUDA. Use `--no-pretrained` to avoid the public TorchVision weight download.

```powershell
uv run python scripts/train_classifier.py --config configs/classification.yaml
uv run pytest
uv run ruff check .
uv run mypy src
```

Training evaluates validation data each epoch, saves the best checkpoint by validation loss, then reports test loss, accuracy, macro precision/recall/F1, and confusion matrix. MLflow records configuration, per-epoch and test metrics, elapsed time, device, and checkpoint/report artifacts.

## Scope and limitations

Phase 1 is an image classifier, not a person detector or a deployment-ready surveillance system. It does not include public-dataset downloads, DVC, Optuna, YOLO, ONNX, an inference API, or LLM workflows. Those are later phases. The synthetic dataset is generated locally and has no external data license. Validate labels, provenance, and licensing for any real dataset before use.

See [docs/training.md](docs/training.md) for run details and [docs/model_card.md](docs/model_card.md) for intended use and limitations.
