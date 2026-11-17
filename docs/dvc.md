# DVC dataset and pipeline versioning

DVC uses its default local cache under `.dvc/cache`; no remote storage or account is required. The repository's `.dvc/config` disables analytics. `data/`, `artifacts/`, MLflow databases, and Optuna databases remain ignored by Git. DVC stage outputs are content-addressed in the local cache, while `dvc.yaml`, `dvc.lock`, and parameters are suitable for Git.

## Reproduce the demo pipeline

```bash
uv sync --extra dev
uv run dvc repro
uv run dvc status
uv run dvc metrics show
```

The pipeline runs four real stages:

1. `generate_dataset` uses the existing seeded synthetic generator.
2. `validate_dataset` decodes every image and records class counts and a SHA-256 dataset fingerprint.
3. `train_model` reuses the shared ResNet18 trainer, produces a validation-selected checkpoint, and explicitly skips test evaluation.
4. `evaluate_and_report_metrics` loads that checkpoint and evaluates the fixed held-out test split once. Its JSON output is the DVC metrics file.

`params.yaml` controls the dataset generator, validation threshold, preprocessing size, and training. DVC detects script/config dependencies and parameter changes and reruns the affected downstream stages. To check this, change `training.epochs`, run `uv run dvc status`, then `uv run dvc repro`; restore the parameter and rerun if you want the checked-in demo defaults back.

To keep a locally sourced dataset in DVC rather than Git, use `uv run dvc add data/my_dataset` and review the generated `.dvc` file before committing it. Change the dataset root and adjust the generation stage for that source: the checked-in `generate_dataset` stage intentionally regenerates synthetic data and will overwrite its configured output. Verify the dataset's license and labels before use.

## Optional remote storage

Local cache is the default. A filesystem remote can be configured without cloud credentials:

```bash
uv run dvc remote add -d local /path/to/dvc-storage
uv run dvc push
uv run dvc pull
```

Keep the path appropriate for the team and machine. Do not configure credentials in tracked files.
