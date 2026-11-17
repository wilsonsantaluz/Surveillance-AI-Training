import argparse
import json
import logging
from pathlib import Path

from surveillance_ml.detection.config import load_detection_config
from surveillance_ml.detection.training import train_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the isolated YOLO detector.")
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    outcome = train_detector(
        load_detection_config(args.config),
        output_dir=args.output_dir,
        run_name=args.run_name,
    )
    print(
        json.dumps(
            {
                "checkpoint": str(outcome.checkpoint_path),
                "results_dir": str(outcome.results_dir),
                "mlflow_run_id": outcome.mlflow_run_id,
                "epochs_completed": outcome.epochs_completed,
                "training_seconds": outcome.training_seconds,
                "dataset_fingerprint": outcome.dataset_fingerprint,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
