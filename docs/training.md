# Training and evaluation

The Phase 1 pipeline is configured in `configs/classification.yaml`. Generate development data with `scripts/generate_demo_dataset.py`; this generator creates labels from the number of silhouettes it draws and does not label external images. For real data, create the three required class directories and verify the source and license before training.

The dataset loader checks supported image extensions and decodes each image before making splits. A class directory may contain nested source directories; all images in the same first-level source directory share a group ID and remain together in one split. Images directly inside a class directory use their filename stem as a source ID. The split routine uses a fixed seed and approximate class balancing. Each class must have at least three source groups and must appear in all three splits; impossible assignments fail before training.

Images are resized to the configured square size and normalized with ImageNet channel statistics. Random horizontal mirroring and brightness changes are applied only to the training dataset. The model supports:

- `head`: freeze the backbone and train the final linear classifier.
- `partial`: train `layer4` and the final linear classifier.
- `full`: fine-tune all parameters.

TorchVision's public `ResNet18_Weights.IMAGENET1K_V1` weights are loaded when `model.pretrained` is true. This may require downloading weights the first time. The checked-in configuration enables transfer learning; pass `--no-pretrained` to make the synthetic demo independent of network access. The trainer uses CUDA when available, otherwise CPU; automatic mixed precision is limited to CUDA. It validates each epoch, reduces learning rate on validation-loss plateaus, applies early stopping, and saves the best validation-loss weights to `artifacts/classifier/best.pt`.

The final best checkpoint is evaluated on the held-out test split. MLflow uses a local SQLite database by default (`sqlite:///mlflow.db`) and stores configuration parameters, split class distributions, per-epoch metrics, test metrics, training duration, checkpoint, and JSON reports. Start the UI with:

```powershell
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

The seed controls Python, NumPy, PyTorch, and the shuffled training loader. GPU kernels and dependency/hardware differences can still prevent bitwise-identical results. Image files and local filesystem paths are not uploaded to MLflow.
