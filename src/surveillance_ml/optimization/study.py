"""Persistent Optuna study setup and reproducible local optimization."""

import gc
import hashlib
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import optuna
import torch
import yaml
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.study import StudyDirection

from surveillance_ml.core.config import AppConfig, load_config
from surveillance_ml.core.device import seed_everything
from surveillance_ml.core.reproducibility import (
    configuration_fingerprint,
    dataset_fingerprint,
    git_commit_sha,
    runtime_versions,
)
from surveillance_ml.data.dataset import CLASS_NAMES
from surveillance_ml.optimization.objective import ClassifierObjective
from surveillance_ml.optimization.search_space import SearchSpace
from surveillance_ml.tracking.mlflow_tracker import configure_tracking

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationSettings:
    study_name: str
    direction: str
    metric: str
    n_trials: int
    timeout_seconds: int | None
    seed: int
    storage: str
    load_if_exists: bool
    output_dir: Path
    base_config: Path
    epochs_per_trial: int
    pretrained: bool
    tracking_uri: str
    experiment_name: str
    pruner_startup_trials: int
    pruner_warmup_steps: int
    pruner_interval_steps: int
    search_space: SearchSpace


def load_optimization_settings(path: Path) -> OptimizationSettings:
    """Parse and validate a YAML optimization configuration."""
    with path.open(encoding="utf-8") as file:
        raw: Any = yaml.safe_load(file)
    if not isinstance(raw, dict) or not isinstance(raw.get("optimization"), dict):
        raise ValueError("Optimization config must contain an 'optimization' mapping")
    values = raw["optimization"]
    pruner = values.get("pruner", {})
    if not isinstance(pruner, dict):
        raise ValueError("optimization.pruner must be a mapping")
    try:
        settings = OptimizationSettings(
            study_name=str(values["study_name"]),
            direction=str(values["direction"]),
            metric=str(values["metric"]),
            n_trials=int(values["n_trials"]),
            timeout_seconds=(
                int(values["timeout_seconds"])
                if values.get("timeout_seconds") is not None
                else None
            ),
            seed=int(values["seed"]),
            storage=str(values["storage"]),
            load_if_exists=bool(values.get("load_if_exists", True)),
            output_dir=Path(values["output_dir"]),
            base_config=Path(values["base_config"]),
            epochs_per_trial=int(values["epochs_per_trial"]),
            pretrained=bool(values["pretrained"]),
            tracking_uri=str(values["tracking_uri"]),
            experiment_name=str(values["experiment_name"]),
            pruner_startup_trials=int(pruner.get("startup_trials", 2)),
            pruner_warmup_steps=int(pruner.get("warmup_steps", 1)),
            pruner_interval_steps=int(pruner.get("interval_steps", 1)),
            search_space=SearchSpace.from_mapping(raw["search_space"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid optimization configuration: {error}") from error

    if not settings.study_name.strip() or not settings.experiment_name.strip():
        raise ValueError("Study and MLflow experiment names must not be empty")
    if settings.direction not in {"maximize", "minimize"}:
        raise ValueError("optimization.direction must be maximize or minimize")
    if settings.metric not in {"val_macro_f1", "val_loss"}:
        raise ValueError("optimization.metric must be val_macro_f1 or val_loss")
    if settings.metric == "val_macro_f1" and settings.direction != "maximize":
        raise ValueError("val_macro_f1 must be maximized")
    if settings.metric == "val_loss" and settings.direction != "minimize":
        raise ValueError("val_loss must be minimized")
    if settings.n_trials < 1 or settings.epochs_per_trial < 1:
        raise ValueError("n_trials and epochs_per_trial must be positive")
    if settings.timeout_seconds is not None and settings.timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive or null")
    if (
        settings.pruner_startup_trials < 0
        or settings.pruner_warmup_steps < 0
        or settings.pruner_interval_steps < 1
    ):
        raise ValueError("Pruner startup/warmup must be non-negative and interval positive")
    if not settings.storage.startswith(("sqlite:///", "postgresql://", "mysql://")):
        raise ValueError("Optimization storage must use a supported SQLAlchemy database URI")
    return settings


def _dvc_lock_fingerprint(repository_root: Path) -> str | None:
    lock_path = repository_root / "dvc.lock"
    if not lock_path.is_file():
        return None
    return hashlib.sha256(lock_path.read_bytes()).hexdigest()


def _release_trial_resources(_study: optuna.Study, _trial: optuna.trial.FrozenTrial) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _base_training_config(
    settings: OptimizationSettings,
    pretrained_override: bool | None,
) -> AppConfig:
    config = load_config(settings.base_config)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", settings.tracking_uri)
    return replace(
        config,
        training=replace(
            config.training,
            epochs=settings.epochs_per_trial,
            seed=settings.seed,
            output_dir=settings.output_dir,
        ),
        model=replace(
            config.model,
            pretrained=(
                pretrained_override
                if pretrained_override is not None
                else settings.pretrained
            ),
        ),
        tracking=replace(
            config.tracking,
            uri=tracking_uri,
            experiment_name=settings.experiment_name,
        ),
    )


def _study_context_fingerprint(
    settings: OptimizationSettings,
    config: AppConfig,
    data_hash: str,
) -> str:
    """Fingerprint the fixed data and training context shared by comparable trials."""
    return configuration_fingerprint(
        {
            "dataset_fingerprint": data_hash,
            "dataset_version": config.dataset.version,
            "class_names": CLASS_NAMES,
            "image_size": config.dataset.image_size,
            "split_fractions": (
                config.dataset.train_fraction,
                config.dataset.validation_fraction,
                config.dataset.test_fraction,
            ),
            "seed": settings.seed,
            "architecture": config.model.architecture,
            "pretrained": config.model.pretrained,
            "epochs_per_trial": settings.epochs_per_trial,
            "early_stopping_patience": config.training.early_stopping_patience,
            "num_workers": config.training.num_workers,
            "cpu_threads": config.training.cpu_threads,
            "mixed_precision": config.training.mixed_precision,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "metric": settings.metric,
            "direction": settings.direction,
            "search_space": settings.search_space,
            "runtime_versions": runtime_versions(),
            "tracking_uri": config.tracking.uri,
            "experiment_name": config.tracking.experiment_name,
        }
    )


def _validate_study_context(
    study: optuna.Study,
    data_hash: str,
    context_fingerprint: str,
) -> None:
    previous_data_hash = study.user_attrs.get("dataset_fingerprint")
    if previous_data_hash is not None and previous_data_hash != data_hash:
        raise ValueError(
            "Cannot resume this Optuna study with a different dataset; use a new study name"
        )
    previous_context = study.user_attrs.get("study_context_fingerprint")
    if previous_context is not None and previous_context != context_fingerprint:
        raise ValueError(
            "Cannot resume this Optuna study with a different search/training context; "
            "use a new study name"
        )


def run_optimization(
    config_path: Path,
    study_name_override: str | None = None,
    pretrained_override: bool | None = None,
) -> optuna.Study:
    """Run or resume a sequential validation-only Optuna study."""
    settings = load_optimization_settings(config_path)
    if study_name_override:
        settings = replace(settings, study_name=study_name_override)
    base_config = _base_training_config(settings, pretrained_override)
    seed_everything(settings.seed)
    data_hash = dataset_fingerprint(base_config.dataset.root)
    repository_root = Path(__file__).resolve().parents[3]
    commit_sha = git_commit_sha(repository_root)
    dvc_revision = _dvc_lock_fingerprint(repository_root)
    fingerprint = configuration_fingerprint(
        {
            "base_config": base_config,
            "optimization": settings,
        }
    )
    context_fingerprint = _study_context_fingerprint(settings, base_config, data_hash)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", settings.tracking_uri)
    experiment_id = configure_tracking(tracking_uri, settings.experiment_name)
    study = optuna.create_study(
        study_name=settings.study_name,
        direction=(
            StudyDirection.MAXIMIZE
            if settings.direction == "maximize"
            else StudyDirection.MINIMIZE
        ),
        storage=settings.storage,
        load_if_exists=settings.load_if_exists,
        sampler=TPESampler(seed=settings.seed),
        pruner=MedianPruner(
            n_startup_trials=settings.pruner_startup_trials,
            n_warmup_steps=settings.pruner_warmup_steps,
            interval_steps=settings.pruner_interval_steps,
        ),
    )
    _validate_study_context(study, data_hash, context_fingerprint)
    study.set_user_attr("dataset_fingerprint", data_hash)
    study.set_user_attr("configuration_fingerprint", fingerprint)
    study.set_user_attr("study_context_fingerprint", context_fingerprint)
    study.set_user_attr("pretrained", base_config.model.pretrained)
    study.set_user_attr("git_commit_sha", commit_sha)
    study.set_user_attr("dvc_lock_fingerprint", dvc_revision)
    study.set_user_attr("runtime_versions", runtime_versions())
    study.set_user_attr("mlflow_experiment_id", experiment_id)

    objective = ClassifierObjective(
        base_config=base_config,
        search_space=settings.search_space,
        study_name=settings.study_name,
        metric=settings.metric,
        output_dir=settings.output_dir,
        dataset_fingerprint=data_hash,
        provenance_tags={
            "git_commit_sha": commit_sha or "unavailable",
            "dvc_lock_fingerprint": dvc_revision or "unavailable",
        },
    )
    LOGGER.info(
        "Starting study %s with %d new trials on %s; dataset=%s",
        settings.study_name,
        settings.n_trials,
        "CUDA" if torch.cuda.is_available() else "CPU",
        data_hash,
    )
    study.optimize(
        objective,
        n_trials=settings.n_trials,
        timeout=settings.timeout_seconds,
        n_jobs=1,
        callbacks=[_release_trial_resources],
        catch=(Exception,),
    )

    from surveillance_ml.reporting.comparison import generate_comparison_report

    generate_comparison_report(
        study_name=settings.study_name,
        experiment_name=settings.experiment_name,
        tracking_uri=tracking_uri,
        metric=settings.metric,
        direction=settings.direction,
        output_dir=Path("artifacts/reports"),
        reproducibility={
            "dataset_fingerprint": data_hash,
            "configuration_fingerprint": fingerprint,
            "git_commit_sha": commit_sha,
            "dvc_lock_fingerprint": dvc_revision,
            "runtime_versions": runtime_versions(),
            "random_seed": settings.seed,
        },
    )
    completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    LOGGER.info(
        "Study %s now has %d trials (%d complete, %d pruned, %d failed)",
        settings.study_name,
        len(study.trials),
        completed,
        sum(trial.state == optuna.trial.TrialState.PRUNED for trial in study.trials),
        sum(trial.state == optuna.trial.TrialState.FAIL for trial in study.trials),
    )
    return study


def finalize_best_trial(config_path: Path, output_dir: Path | None = None) -> Path:
    """Retrain the best validation-selected parameters and evaluate test once."""
    settings = load_optimization_settings(config_path)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", settings.tracking_uri)
    study = optuna.load_study(
        study_name=settings.study_name,
        storage=settings.storage,
        sampler=TPESampler(seed=settings.seed),
    )
    saved_pretrained = study.user_attrs.get("pretrained")
    pretrained_override = (
        saved_pretrained if isinstance(saved_pretrained, bool) else None
    )
    base_config = _base_training_config(settings, pretrained_override)
    data_hash = dataset_fingerprint(base_config.dataset.root)
    context_fingerprint = _study_context_fingerprint(settings, base_config, data_hash)
    _validate_study_context(study, data_hash, context_fingerprint)
    if not any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        raise ValueError(f"Study '{settings.study_name}' has no completed trial to finalize")
    best = study.best_trial
    training = replace(
        base_config.training,
        batch_size=int(best.params["batch_size"]),
        learning_rate=float(best.params["learning_rate"]),
        optimizer=str(best.params["optimizer"]),
        weight_decay=float(best.params["weight_decay"]),
        scheduler_factor=float(best.params["scheduler_factor"]),
        scheduler_patience=int(best.params["scheduler_patience"]),
    )
    model_config = replace(
        base_config.model,
        strategy=str(best.params["fine_tuning_strategy"]),
    )
    tracking = replace(base_config.tracking, uri=tracking_uri)
    config = replace(
        base_config,
        training=training,
        model=model_config,
        tracking=tracking,
    )
    final_output = output_dir or Path("artifacts/finalized") / settings.study_name
    from surveillance_ml.training.train import run_training_experiment

    outcome = run_training_experiment(
        config,
        evaluate_test=True,
        output_dir=final_output,
        run_name=f"{settings.study_name}-finalized",
        tags={
            "run_type": "finalized_model",
            "finalized_from_study": settings.study_name,
            "best_trial_number": str(best.number),
            "selection_metric": settings.metric,
            "selection_dataset_fingerprint": str(
                study.user_attrs.get("dataset_fingerprint", "unavailable")
            ),
        },
        dataset_fingerprint_value=str(study.user_attrs.get("dataset_fingerprint", "")) or None,
    )
    return outcome.checkpoint_path
