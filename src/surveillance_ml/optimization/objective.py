"""MLflow-linked Optuna objective reusing the shared PyTorch training loop."""

import logging
from dataclasses import replace
from pathlib import Path

import mlflow
import optuna

from surveillance_ml.core.config import AppConfig
from surveillance_ml.optimization.pruning import report_validation_metric
from surveillance_ml.optimization.search_space import SearchSpace
from surveillance_ml.training.train import run_training_experiment

LOGGER = logging.getLogger(__name__)


class ClassifierObjective:
    """Train one isolated trial and return its best-checkpoint validation score."""

    def __init__(
        self,
        base_config: AppConfig,
        search_space: SearchSpace,
        study_name: str,
        metric: str,
        output_dir: Path,
        dataset_fingerprint: str,
        provenance_tags: dict[str, str],
    ) -> None:
        self.base_config = base_config
        self.search_space = search_space
        self.study_name = study_name
        self.metric = metric
        self.output_dir = output_dir
        self.dataset_fingerprint = dataset_fingerprint
        self.provenance_tags = provenance_tags

    def __call__(self, trial: optuna.Trial) -> float:
        parameters = self.search_space.sample(trial)
        trial_config = replace(
            self.base_config,
            model=replace(
                self.base_config.model,
                strategy=str(parameters["fine_tuning_strategy"]),
            ),
            training=replace(
                self.base_config.training,
                batch_size=int(parameters["batch_size"]),
                learning_rate=float(parameters["learning_rate"]),
                optimizer=str(parameters["optimizer"]),
                weight_decay=float(parameters["weight_decay"]),
                scheduler_factor=float(parameters["scheduler_factor"]),
                scheduler_patience=int(parameters["scheduler_patience"]),
            ),
        )
        trial_output = self.output_dir / f"trial_{trial.number:03d}"
        run_name = f"{self.study_name}-trial-{trial.number:03d}"
        tags = {
            "run_type": "optuna_trial",
            "study_name": self.study_name,
            "trial_number": str(trial.number),
            "trial_status": "RUNNING",
            "dataset_fingerprint": self.dataset_fingerprint,
        }
        did_prune = False

        with mlflow.start_run(run_name=run_name) as active_run:
            mlflow.set_tags({**tags, **self.provenance_tags})
            mlflow.log_params(
                {
                    "study_name": self.study_name,
                    "trial_number": trial.number,
                    **parameters,
                }
            )

            def on_epoch(epoch: int, values: dict[str, float]) -> None:
                report_validation_metric(trial, self.metric, epoch, values)

            try:
                outcome = run_training_experiment(
                    trial_config,
                    evaluate_test=False,
                    output_dir=trial_output,
                    epoch_callback=on_epoch,
                    dataset_fingerprint_value=self.dataset_fingerprint,
                )
            except optuna.TrialPruned as error:
                mlflow.set_tag("trial_status", "PRUNED")
                trial.set_user_attr("mlflow_run_id", active_run.info.run_id)
                trial.set_user_attr("artifact_dir", trial_output.as_posix())
                did_prune = True
                LOGGER.info("Optuna trial %d pruned: %s", trial.number, error)
            except Exception as error:
                mlflow.set_tag("trial_status", "FAILED")
                mlflow.set_tag("failure_type", type(error).__name__)
                trial.set_user_attr("mlflow_run_id", active_run.info.run_id)
                trial.set_user_attr("artifact_dir", trial_output.as_posix())
                LOGGER.exception("Optuna trial %d failed", trial.number)
                raise
            else:
                score = (
                    outcome.validation_metrics.f1
                    if self.metric == "val_macro_f1"
                    else outcome.validation_metrics.loss
                )
                mlflow.log_metric("objective_value", score)
                mlflow.set_tag("trial_status", "COMPLETE")
                trial.set_user_attr("mlflow_run_id", active_run.info.run_id)
                trial.set_user_attr("artifact_dir", trial_output.as_posix())
                trial.set_user_attr("dataset_fingerprint", self.dataset_fingerprint)
                trial.set_user_attr("best_epoch", outcome.result.best_epoch)
                LOGGER.info(
                    "Optuna trial %d completed with %s=%.6f",
                    trial.number,
                    self.metric,
                    score,
                )
                return float(score)

        if did_prune:
            raise optuna.TrialPruned(f"Trial {trial.number} pruned")
        raise RuntimeError(f"Trial {trial.number} ended without a score")
