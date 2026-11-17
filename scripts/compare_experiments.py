import argparse
import json
import os
from pathlib import Path

import optuna

from surveillance_ml.optimization.study import load_optimization_settings
from surveillance_ml.reporting.comparison import generate_comparison_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare recorded Optuna trials from MLflow")
    parser.add_argument("--config", type=Path, default=Path("configs/optimization.yaml"))
    parser.add_argument("--study")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reports"))
    args = parser.parse_args()
    settings = load_optimization_settings(args.config)
    study_name = args.study or settings.study_name
    study = optuna.load_study(study_name=study_name, storage=settings.storage)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", settings.tracking_uri)
    report = generate_comparison_report(
        study_name=study_name,
        experiment_name=settings.experiment_name,
        tracking_uri=tracking_uri,
        metric=settings.metric,
        direction=settings.direction,
        output_dir=args.output_dir,
        reproducibility=study.user_attrs,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
