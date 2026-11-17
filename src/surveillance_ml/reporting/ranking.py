"""Stable ranking of compatible completed Optuna trial results."""

import math
from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class TrialSummary:
    run_id: str
    trial_number: int | None
    status: str
    parameters: dict[str, Any]
    architecture: str | None
    strategy: str | None
    learning_rate: float | None
    batch_size: int | None
    optimizer: str | None
    weight_decay: float | None
    validation_loss: float | None
    validation_accuracy: float | None
    validation_f1: float | None
    training_seconds: float | None
    objective_value: float | None
    compatible: bool = True
    incompatibility_reasons: tuple[str, ...] = ()
    rank: int | None = None


def _finite_metric(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def rank_trials(
    trials: list[TrialSummary],
    metric: str,
    direction: str,
) -> list[TrialSummary]:
    """Rank only compatible completed trials with an observed objective metric."""
    if metric not in {"val_macro_f1", "val_loss"}:
        raise ValueError(f"Unsupported ranking metric: {metric}")
    if direction not in {"maximize", "minimize"}:
        raise ValueError(f"Unsupported ranking direction: {direction}")
    scored: list[tuple[float, int, TrialSummary]] = []
    for trial in trials:
        if trial.status != "COMPLETE" or not trial.compatible:
            continue
        value = trial.validation_f1 if metric == "val_macro_f1" else trial.validation_loss
        score = _finite_metric(value)
        if score is not None:
            scored.append((score, trial.trial_number or 0, trial))
    if direction == "maximize":
        scored.sort(key=lambda item: (-item[0], item[1]))
    else:
        scored.sort(key=lambda item: (item[0], item[1]))
    ranked = [replace(trial, rank=rank) for rank, (_, _, trial) in enumerate(scored, start=1)]
    ranked_ids = {trial.run_id for trial in ranked}
    unranked = [replace(trial, rank=None) for trial in trials if trial.run_id not in ranked_ids]
    return ranked + sorted(unranked, key=lambda trial: trial.trial_number or -1)
