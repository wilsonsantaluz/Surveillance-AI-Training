"""MLflow-backed experiment comparison and machine-readable report generation."""

import csv
import json
import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from surveillance_ml.reporting.charts import create_comparison_charts
from surveillance_ml.reporting.markdown_report import render_markdown_report
from surveillance_ml.reporting.ranking import TrialSummary, rank_trials

LOGGER = logging.getLogger(__name__)
COMPATIBILITY_FIELDS = (
    "dataset_fingerprint",
    "split_fingerprint",
    "dataset_version",
    "class_names",
    "image_size",
    "seed",
    "train_fraction",
    "validation_fraction",
    "test_fraction",
    "architecture",
    "pretrained",
    "weights",
    "device",
    "version_torch",
    "version_torchvision",
)
LIMITATIONS = [
    "Validation scores select configurations; held-out test metrics are not used in ranking.",
    "Small synthetic datasets demonstrate pipeline behavior, not real-world model quality.",
    "A ranking is descriptive and is not evidence of statistically significant differences.",
]


def _get_compatibility_values(run: Any) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for field in COMPATIBILITY_FIELDS:
        value = run.data.tags.get(field)
        if value is None:
            value = run.data.params.get(field)
        values[field] = value
    return values


def _float_metric(run: Any, name: str) -> float | None:
    value = run.data.metrics.get(name)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _as_trial_summary(run: Any) -> TrialSummary:
    params = run.data.params
    tags = run.data.tags
    try:
        trial_number = int(tags["trial_number"])
    except (KeyError, TypeError, ValueError):
        trial_number = None

    def param_float(name: str) -> float | None:
        value = params.get(name)
        try:
            return float(value) if value is not None else None
        except ValueError:
            return None

    def param_int(name: str) -> int | None:
        value = params.get(name)
        try:
            return int(value) if value is not None else None
        except ValueError:
            return None

    return TrialSummary(
        run_id=run.info.run_id,
        trial_number=trial_number,
        status=tags.get("trial_status", run.info.status).upper(),
        parameters={
            name: params.get(name)
            for name in (
                "learning_rate",
                "weight_decay",
                "batch_size",
                "optimizer",
                "fine_tuning_strategy",
                "scheduler_factor",
                "scheduler_patience",
            )
        },
        architecture=params.get("architecture"),
        strategy=params.get("fine_tuning_strategy"),
        learning_rate=param_float("learning_rate"),
        batch_size=param_int("batch_size"),
        optimizer=params.get("optimizer"),
        weight_decay=param_float("weight_decay"),
        validation_loss=_float_metric(run, "best_validation_loss"),
        validation_accuracy=_float_metric(run, "best_validation_accuracy"),
        validation_f1=_float_metric(run, "best_validation_f1"),
        training_seconds=_float_metric(run, "training_seconds"),
        objective_value=_float_metric(run, "objective_value"),
    )


def _compare_compatibility(trials: list[TrialSummary], runs: dict[str, Any]) -> list[TrialSummary]:
    valid_reference: dict[str, str | None] | None = None
    reference_run_id: str | None = None
    for trial in sorted(trials, key=lambda item: item.trial_number or -1):
        values = _get_compatibility_values(runs[trial.run_id])
        if all(values[field] is not None for field in COMPATIBILITY_FIELDS):
            valid_reference = values
            reference_run_id = trial.run_id
            break
    if valid_reference is None:
        reason = "No trial has complete dataset, label, split, and evaluation metadata."
        return [
            replace(trial, compatible=False, incompatibility_reasons=(reason,))
            for trial in trials
        ]

    result: list[TrialSummary] = []
    for trial in trials:
        values = _get_compatibility_values(runs[trial.run_id])
        reasons = tuple(
            f"{field} differs from reference trial"
            for field in COMPATIBILITY_FIELDS
            if values[field] is None or values[field] != valid_reference[field]
        )
        result.append(replace(trial, compatible=not reasons, incompatibility_reasons=reasons))
    LOGGER.info(
        "Using MLflow run %s as the dataset/split compatibility reference",
        reference_run_id,
    )
    return result


def _trial_counts(trials: list[TrialSummary]) -> dict[str, int]:
    counts = {"COMPLETE": 0, "PRUNED": 0, "FAILED": 0, "RUNNING": 0}
    for trial in trials:
        status = trial.status.upper()
        if status in counts:
            counts[status] += 1
        else:
            counts["FAILED"] += 1
    return counts


def _report_rows(trials: list[TrialSummary]) -> list[dict[str, object]]:
    return [
        {
            "rank": trial.rank,
            "trial_number": trial.trial_number,
            "run_id": trial.run_id,
            "status": trial.status,
            "compatible": trial.compatible,
            "incompatibility_reasons": list(trial.incompatibility_reasons),
            "architecture": trial.architecture,
            "strategy": trial.strategy,
            "learning_rate": trial.learning_rate,
            "batch_size": trial.batch_size,
            "optimizer": trial.optimizer,
            "weight_decay": trial.weight_decay,
            "validation_loss": trial.validation_loss,
            "validation_accuracy": trial.validation_accuracy,
            "validation_f1": trial.validation_f1,
            "training_seconds": trial.training_seconds,
            "objective_value": trial.objective_value,
            "parameters": trial.parameters,
        }
        for trial in trials
    ]


def generate_comparison_report(
    study_name: str,
    experiment_name: str,
    tracking_uri: str,
    metric: str,
    direction: str,
    output_dir: Path = Path("artifacts/reports"),
    reproducibility: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare only compatible MLflow trial runs and log the resulting reports."""
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")
    matched_runs = [
        run
        for run in client.search_runs(
            [experiment.experiment_id],
            filter_string="tags.run_type = 'optuna_trial'",
            order_by=["attributes.start_time ASC"],
            max_results=10000,
        )
        if run.data.tags.get("study_name") == study_name
    ]
    run_map = {run.info.run_id: run for run in matched_runs}
    trials = _compare_compatibility(
        [_as_trial_summary(run) for run in matched_runs],
        run_map,
    ) if matched_runs else []
    ranked = rank_trials(trials, metric, direction)
    best = next((trial for trial in ranked if trial.rank == 1), None)
    counts = _trial_counts(trials)
    warnings = [
        f"Trial {trial.trial_number}: {'; '.join(trial.incompatibility_reasons)}"
        for trial in ranked
        if not trial.compatible
    ]
    warnings.extend(
        f"Trial {trial.trial_number} has no recorded {metric} and was not ranked."
        for trial in ranked
        if trial.compatible and trial.status == "COMPLETE" and trial.rank is None
    )
    reference = next(
        (run_map[trial.run_id] for trial in ranked if trial.compatible),
        None,
    )
    dataset_fingerprint = (
        reference.data.tags.get("dataset_fingerprint")
        or reference.data.params.get("dataset_fingerprint")
        if reference is not None
        else None
    )
    best_score = None
    if best is not None:
        best_score = (
            best.validation_f1 if metric == "val_macro_f1" else best.validation_loss
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = output_dir / "charts"
    chart_paths = create_comparison_charts(ranked, charts_dir, metric, direction)
    rows = _report_rows(ranked)
    report: dict[str, Any] = {
        "study_name": study_name,
        "experiment_name": experiment_name,
        "objective": {"metric": metric, "direction": direction},
        "best_trial": (
            {
                "number": best.trial_number,
                "run_id": best.run_id,
                "score": best.validation_f1 if metric == "val_macro_f1" else best.validation_loss,
                "parameters": best.parameters,
            }
            if best is not None
            else None
        ),
        "best_validation_score": best_score,
        "selected_parameters": best.parameters if best is not None else None,
        "trial_counts": counts,
        "ranked_trials": rows,
        "dataset": {
            "fingerprint": dataset_fingerprint,
            "version": reference.data.params.get("dataset_version") if reference else None,
            "class_names": reference.data.params.get("class_names") if reference else None,
        },
        "reproducibility": reproducibility or {},
        "warnings": warnings,
        "limitations": LIMITATIONS,
        "charts": [path.name for path in chart_paths],
    }

    json_path = output_dir / "experiment_comparison.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    csv_path = output_dir / "experiment_comparison.csv"
    csv_fields = [
        "rank",
        "trial_number",
        "run_id",
        "status",
        "compatible",
        "architecture",
        "strategy",
        "learning_rate",
        "batch_size",
        "optimizer",
        "weight_decay",
        "validation_loss",
        "validation_accuracy",
        "validation_f1",
        "training_seconds",
        "objective_value",
        "parameters",
        "incompatibility_reasons",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        for trial, row in zip(ranked, rows, strict=True):
            csv_row = dict(row)
            csv_row["parameters"] = json.dumps(row["parameters"], sort_keys=True)
            csv_row["incompatibility_reasons"] = "; ".join(
                trial.incompatibility_reasons
            )
            writer.writerow(csv_row)
    markdown_path = output_dir / "experiment_comparison.md"
    markdown_path.write_text(
        render_markdown_report(
            study_name,
            metric,
            direction,
            ranked,
            best,
            counts,
            dataset_fingerprint,
            warnings,
            LIMITATIONS,
        ),
        encoding="utf-8",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{study_name}-comparison-report"):
        mlflow.set_tags({"run_type": "experiment_comparison", "study_name": study_name})
        mlflow.log_params(
            {
                "study_name": study_name,
                "objective_metric": metric,
                "direction": direction,
                "trial_count": len(trials),
                "completed_count": counts["COMPLETE"],
                "pruned_count": counts["PRUNED"],
                "failed_count": counts["FAILED"],
            }
        )
        if best is not None:
            if best_score is not None:
                mlflow.log_metric("best_validation_score", best_score)
        for artifact in [json_path, csv_path, markdown_path, *chart_paths]:
            mlflow.log_artifact(str(artifact), artifact_path="comparison")
    return report
