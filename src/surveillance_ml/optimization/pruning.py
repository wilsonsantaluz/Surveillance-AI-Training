"""Validation-only Optuna pruning callbacks."""

import optuna


def report_validation_metric(
    trial: optuna.Trial,
    metric_name: str,
    epoch: int,
    values: dict[str, float],
) -> None:
    """Report one validation metric and prune when the configured pruner requests it."""
    metric_keys = {"val_macro_f1": "validation_f1", "val_loss": "validation_loss"}
    if metric_name not in metric_keys:
        raise ValueError(f"Unsupported Optuna objective metric: {metric_name}")
    metric_value = values.get(metric_keys[metric_name])
    if metric_value is None:
        raise ValueError(f"Training epoch did not report {metric_keys[metric_name]}")
    trial.report(metric_value, step=epoch)
    if trial.should_prune():
        raise optuna.TrialPruned(f"Pruned after validation epoch {epoch}")
