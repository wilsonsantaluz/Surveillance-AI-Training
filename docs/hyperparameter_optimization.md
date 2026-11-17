# Hyperparameter optimization

Optuna trials reuse `run_training_experiment` and the existing trainer; no second training loop is maintained. The default objective is validation macro-F1. Trials share one seed, dataset fingerprint, and deterministic source-group split. The test loader is not constructed, test images are not decoded, and test metrics are not logged during a trial.

Each trial has an isolated `artifacts/optimization/<study>/trial_NNN/` directory, a persistent Optuna SQLite study, and an MLflow run tagged with its study name, trial number, status, Git revision, dataset fingerprint, split fingerprint, configuration fingerprint, and runtime versions. Completed trials record best-checkpoint validation metrics; pruned runs retain a `PRUNED` tag and failed runs retain a `FAILED` tag and error type. A median pruner reports validation values once per epoch. Study execution is sequential and clears temporary Python/CUDA resources after every trial.

## Run a small local study

First generate the dataset if it is not already present:

```bash
uv run python scripts/generate_demo_dataset.py --output-dir data/demo --samples-per-class 24
uv run python scripts/optimize_classifier.py --config configs/optimization_demo.yaml --no-pretrained
```

The demo performs two one-epoch CPU-compatible trials, uses randomly initialized ResNet18, and stores its study in `optuna-demo.db`. Rerunning the command resumes the named study and adds two more trials. It writes the actual MLflow comparison report under `artifacts/reports/`.

The regular study configuration is `configs/optimization.yaml`. It uses public pretrained weights and a modest 12-trial search; the weight download may require network access. `--n_jobs` is intentionally fixed to one for bounded CPU resource use. Search ranges, optimizer choices, strategy choices, scheduler options, timeout, seed, and pruning thresholds are validated from YAML.

## Finalize only after selection

Once trials complete, retrain the best validation-selected parameters and evaluate the held-out test split once:

```bash
uv run python scripts/finalize_classifier.py --config configs/optimization_demo.yaml
```

Finalization is a separate MLflow run. Its test metrics must not be used to change the study's winning parameters.
