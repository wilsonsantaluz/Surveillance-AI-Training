"""Local-capable MLflow experiment tracking."""

import mlflow


def configure_tracking(uri: str, experiment_name: str) -> str:
    """Select a tracking backend and return the active experiment ID."""
    mlflow.set_tracking_uri(uri)
    experiment = mlflow.set_experiment(experiment_name)
    return experiment.experiment_id
