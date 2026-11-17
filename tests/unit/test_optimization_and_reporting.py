from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest
from mlflow.tracking import MlflowClient

from surveillance_ml.core.config import load_config
from surveillance_ml.data.dataset import CLASS_NAMES, discover_images
from surveillance_ml.data.splitting import split_records
from surveillance_ml.data.synthetic import generate_synthetic_dataset
from surveillance_ml.optimization.objective import ClassifierObjective
from surveillance_ml.optimization.pruning import report_validation_metric
from surveillance_ml.optimization.search_space import SearchSpace
from surveillance_ml.optimization.study import (
    _validate_study_context,
    load_optimization_settings,
)
from surveillance_ml.reporting.charts import create_comparison_charts
from surveillance_ml.reporting.comparison import (
    COMPATIBILITY_FIELDS,
    _compare_compatibility,
    generate_comparison_report,
)
from surveillance_ml.reporting.ranking import TrialSummary, rank_trials
from surveillance_ml.tracking.mlflow_tracker import configure_tracking
from surveillance_ml.training.train import _create_loaders


def _trial(
    run_id: str,
    number: int,
    *,
    status: str = "COMPLETE",
    score: float | None = None,
    compatible: bool = True,
) -> TrialSummary:
    return TrialSummary(
        run_id=run_id,
        trial_number=number,
        status=status,
        parameters={},
        architecture="resnet18",
        strategy="head",
        learning_rate=0.001,
        batch_size=4,
        optimizer="adamw",
        weight_decay=0.0,
        validation_loss=1.0,
        validation_accuracy=0.5,
        validation_f1=score,
        training_seconds=1.0,
        objective_value=score,
        compatible=compatible,
    )


def test_search_space_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="requires 0 < min < max"):
        SearchSpace.from_mapping(
            {
                "learning_rate": {"min": 0.1, "max": 0.1},
                "weight_decay": {"min": 0.0001, "max": 0.01},
                "batch_size": {"choices": [4]},
                "optimizer": {"choices": ["adamw"]},
                "fine_tuning_strategy": {"choices": ["head"]},
                "scheduler_factor": {"choices": [0.5]},
                "scheduler_patience": {"choices": [1]},
            }
        )


def test_pruning_uses_only_the_reported_validation_metric() -> None:
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.ThresholdPruner(lower=0.5),
    )

    def objective(trial: optuna.Trial) -> float:
        report_validation_metric(
            trial,
            "val_macro_f1",
            1,
            {"validation_f1": 0.1},
        )
        return 0.1

    study.optimize(objective, n_trials=1)

    assert study.trials[0].state == optuna.trial.TrialState.PRUNED


def test_study_resume_rejects_changed_dataset_or_search_context() -> None:
    study = optuna.create_study()
    study.set_user_attr("dataset_fingerprint", "original-data")
    study.set_user_attr("study_context_fingerprint", "original-context")

    with pytest.raises(ValueError, match="different dataset"):
        _validate_study_context(study, "changed-data", "original-context")
    with pytest.raises(ValueError, match="different search/training context"):
        _validate_study_context(study, "original-data", "changed-context")
    _validate_study_context(study, "original-data", "original-context")


def test_ranking_is_stable_and_excludes_incompatible_or_unscored_trials() -> None:
    trials = [
        _trial("later-tie", 3, score=0.8),
        _trial("first-tie", 1, score=0.8),
        _trial("missing", 2),
        _trial("incompatible", 4, score=0.9, compatible=False),
        _trial("failed", 5, score=1.0, status="FAILED"),
    ]

    ranked = rank_trials(trials, "val_macro_f1", "maximize")

    assert [(trial.trial_number, trial.rank) for trial in ranked[:2]] == [(1, 1), (3, 2)]
    assert all(trial.rank is None for trial in ranked[2:])


def test_loss_objective_generates_loss_charts(tmp_path: Path) -> None:
    paths = create_comparison_charts(
        [_trial("run", 0, score=0.8)],
        tmp_path,
        metric="val_loss",
        direction="minimize",
    )

    assert "val_loss_by_trial.png" in {path.name for path in paths}
    assert "learning_rate_vs_val_loss.png" in {path.name for path in paths}


def test_compatibility_requires_matching_dataset_and_split_metadata() -> None:
    first, second = _trial("a", 0, score=0.5), _trial("b", 1, score=0.6)
    shared = {field: f"value-{field}" for field in COMPATIBILITY_FIELDS}
    first_run = SimpleNamespace(
        data=SimpleNamespace(tags=shared, params={}),
    )
    second_run = SimpleNamespace(
        data=SimpleNamespace(
            tags={**shared, "split_fingerprint": "different-split"},
            params={},
        ),
    )

    compared = _compare_compatibility(
        [first, second],
        {first.run_id: first_run, second.run_id: second_run},
    )

    assert compared[0].compatible
    assert not compared[1].compatible
    assert compared[1].incompatibility_reasons == (
        "split_fingerprint differs from reference trial",
    )


def test_comparison_report_without_trials_has_no_best_score(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    configure_tracking(tracking_uri, "empty-comparison-test")

    report = generate_comparison_report(
        study_name="empty-study",
        experiment_name="empty-comparison-test",
        tracking_uri=tracking_uri,
        metric="val_macro_f1",
        direction="maximize",
        output_dir=tmp_path / "reports",
    )

    assert report["best_trial"] is None
    assert report["best_validation_score"] is None
    assert report["trial_counts"]["COMPLETE"] == 0
    assert (tmp_path / "reports" / "experiment_comparison.json").is_file()
    assert report["charts"] == []


def test_failed_trial_is_tagged_in_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    settings = load_optimization_settings(Path("configs/optimization_demo.yaml"))
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    experiment_name = "failed-trial-test"
    configure_tracking(tracking_uri, experiment_name)
    config = load_config(settings.base_config)
    config = replace(
        config,
        tracking=replace(
            config.tracking,
            uri=tracking_uri,
            experiment_name=experiment_name,
        ),
    )

    def fail_training(*_args: object, **_kwargs: object) -> object:
        raise ValueError("expected test failure")

    monkeypatch.setattr(
        "surveillance_ml.optimization.objective.run_training_experiment",
        fail_training,
    )
    objective = ClassifierObjective(
        base_config=config,
        search_space=settings.search_space,
        study_name="failed-study",
        metric=settings.metric,
        output_dir=tmp_path / "trials",
        dataset_fingerprint="dataset-hash",
        provenance_tags={"git_commit_sha": "test", "dvc_lock_fingerprint": "test"},
    )
    study = optuna.create_study(
        study_name="failed-study",
        storage=f"sqlite:///{(tmp_path / 'optuna.db').as_posix()}",
        direction="maximize",
    )
    study.optimize(objective, n_trials=1, catch=(ValueError,))

    experiment = MlflowClient(tracking_uri=tracking_uri).get_experiment_by_name(
        experiment_name
    )
    assert experiment is not None
    runs = MlflowClient(tracking_uri=tracking_uri).search_runs([experiment.experiment_id])

    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert len(runs) == 1
    assert runs[0].data.tags["trial_status"] == "FAILED"
    assert runs[0].data.tags["failure_type"] == "ValueError"


def test_validation_only_loaders_do_not_construct_a_test_loader(tmp_path: Path) -> None:
    root = generate_synthetic_dataset(tmp_path / "demo", samples_per_class=6, seed=2)
    records = discover_images(root)
    splits = split_records(records, seed=5, fractions=(0.6, 0.2, 0.2))
    config = load_config(Path("configs/classification.yaml"))
    config = replace(config, dataset=replace(config.dataset, root=root))

    train_loader, validation_loader, test_loader = _create_loaders(
        splits,
        config,
        include_test=False,
    )

    assert train_loader.dataset.records
    assert validation_loader.dataset.records
    assert test_loader is None
    assert set(CLASS_NAMES) == {record.class_name for record in records}
