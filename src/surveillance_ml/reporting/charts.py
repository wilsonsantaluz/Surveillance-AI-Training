"""Charts generated only from recorded MLflow trial results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from surveillance_ml.reporting.ranking import TrialSummary


def create_comparison_charts(
    trials: list[TrialSummary],
    output_dir: Path,
    metric: str,
    direction: str,
) -> list[Path]:
    """Write observed validation and resource charts, omitting unavailable views."""
    metric_label = "Validation macro-F1" if metric == "val_macro_f1" else "Validation loss"
    metric_attribute = "validation_f1" if metric == "val_macro_f1" else "validation_loss"
    scored: list[tuple[TrialSummary, float]] = []
    for trial in trials:
        value = getattr(trial, metric_attribute)
        if trial.status == "COMPLETE" and trial.compatible and value is not None:
            scored.append((trial, value))
    if not scored:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    numbers = [trial.trial_number or 0 for trial, _value in scored]
    metric_values = [value for _trial, value in scored]

    def save(name: str) -> None:
        path = output_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=140)
        plt.close()
        paths.append(path)

    plt.figure(figsize=(6, 4))
    plt.plot(numbers, metric_values, marker="o")
    plt.xlabel("Trial")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} by trial")
    save(f"{metric}_by_trial.png")

    history: list[float] = []
    for value in metric_values:
        if not history:
            history.append(value)
        else:
            history.append(
                max(history[-1], value)
                if direction == "maximize"
                else min(history[-1], value)
            )
    plt.figure(figsize=(6, 4))
    plt.plot(numbers, history, marker="o")
    plt.xlabel("Trial")
    plt.ylabel(f"Best {metric_label.lower()} so far")
    plt.title("Optimization history")
    save("optimization_history.png")

    learning_rate_scores = [
        (trial.learning_rate, value)
        for trial, value in scored
        if trial.learning_rate is not None
    ]
    if learning_rate_scores:
        plt.figure(figsize=(6, 4))
        plt.scatter(
            [learning_rate for learning_rate, _value in learning_rate_scores],
            [value for _learning_rate, value in learning_rate_scores],
        )
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel(metric_label)
        plt.title(f"Learning rate vs {metric_label.lower()}")
        save(f"learning_rate_vs_{metric}.png")

    timed_scores = [
        (trial.training_seconds, value)
        for trial, value in scored
        if trial.training_seconds is not None
    ]
    if timed_scores:
        plt.figure(figsize=(6, 4))
        plt.scatter(
            [duration for duration, _value in timed_scores],
            [value for _duration, value in timed_scores],
        )
        plt.xlabel("Training duration (seconds)")
        plt.ylabel(metric_label)
        plt.title(f"Training time vs {metric_label.lower()}")
        save(f"training_time_vs_{metric}.png")

    strategy_scores: dict[str, list[float]] = {}
    for trial, value in scored:
        if trial.strategy is not None:
            strategy_scores.setdefault(trial.strategy, []).append(value)
    if strategy_scores:
        strategies = sorted(strategy_scores)
        means = [sum(strategy_scores[name]) / len(strategy_scores[name]) for name in strategies]
        plt.figure(figsize=(6, 4))
        plt.bar(strategies, means)
        plt.xlabel("Fine-tuning strategy")
        plt.ylabel(f"Mean {metric_label.lower()}")
        plt.title("Fine-tuning strategy comparison")
        save(f"strategy_vs_{metric}.png")
    return paths
