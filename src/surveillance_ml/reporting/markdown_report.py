"""Markdown rendering for machine-generated experiment comparisons."""

from surveillance_ml.reporting.ranking import TrialSummary


def render_markdown_report(
    study_name: str,
    metric: str,
    direction: str,
    ranked_trials: list[TrialSummary],
    best: TrialSummary | None,
    trial_counts: dict[str, int],
    dataset_fingerprint: str | None,
    warnings: list[str],
    limitations: list[str],
) -> str:
    """Render a concise report from actual MLflow trial records."""
    lines = [
        f"# Experiment comparison: {study_name}",
        "",
        f"Objective: `{metric}` ({direction}).",
        "",
        f"Trial counts: {trial_counts.get('COMPLETE', 0)} complete, "
        f"{trial_counts.get('PRUNED', 0)} pruned, {trial_counts.get('FAILED', 0)} failed.",
        "",
        f"Dataset fingerprint: `{dataset_fingerprint or 'unavailable'}`.",
        "",
    ]
    if best is None:
        lines.extend(["No compatible completed trial with the requested metric was recorded.", ""])
    else:
        value = best.validation_f1 if metric == "val_macro_f1" else best.validation_loss
        lines.extend(
            [
                f"Best trial: `{best.trial_number}` with `{metric}={value:.6f}`.",
                f"Selected parameters: `{best.parameters}`.",
                "",
            ]
        )
    lines.extend(
        [
            "| Rank | Trial | Status | Strategy | Learning rate | Batch | Val loss | "
            "Val accuracy | Val F1 | Duration (s) |",
            "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for trial in ranked_trials:
        def display(value: float | int | None, precision: int = 4) -> str:
            return f"{value:.{precision}f}" if isinstance(value, float) else (
                str(value) if value is not None else "N/A"
            )

        lines.append(
            f"| {trial.rank if trial.rank is not None else 'N/A'} "
            f"| {trial.trial_number if trial.trial_number is not None else 'N/A'} "
            f"| {trial.status} | {trial.strategy or 'N/A'} "
            f"| {display(trial.learning_rate, 6)} | {display(trial.batch_size, 0)} "
            f"| {display(trial.validation_loss)} | {display(trial.validation_accuracy)} "
            f"| {display(trial.validation_f1)} | {display(trial.training_seconds, 2)} |"
        )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in limitations)
    lines.append("")
    return "\n".join(lines)
