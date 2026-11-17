import argparse
import json
from pathlib import Path

from surveillance_ml.detection.config import load_detection_config
from surveillance_ml.detection.training import evaluate_and_log_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO checkpoint on one dataset split.")
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=Path("artifacts/detection/evaluation.json"))
    parser.add_argument("--parent-run-id")
    args = parser.parse_args()
    metrics = evaluate_and_log_detector(
        load_detection_config(args.config),
        args.checkpoint,
        split=args.split,
        output_path=args.output,
        parent_run_id=args.parent_run_id,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
