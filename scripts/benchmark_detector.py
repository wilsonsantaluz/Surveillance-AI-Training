import argparse
import json
import os
from pathlib import Path

import mlflow

from surveillance_ml.detection.config import load_detection_config
from surveillance_ml.detection.onnx import OnnxDetector
from surveillance_ml.detection.video import run_video_inference
from surveillance_ml.tracking.mlflow_tracker import configure_tracking


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure ONNX detector video latency and throughput."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/detection/video-benchmark.json")
    )
    parser.add_argument("--target-fps", type=float)
    parser.add_argument("--mlflow-run-id")
    args = parser.parse_args()

    detector = OnnxDetector(args.model, args.manifest)
    report = run_video_inference(
        detector,
        args.video,
        target_fps=args.target_fps,
    )
    report["model_sha256"] = detector.manifest.model_sha256
    report["model_id"] = detector.manifest.model_id
    report["model_version"] = detector.manifest.model_version
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.mlflow_run_id:
        config = load_detection_config(args.config)
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config.tracking.uri)
        configure_tracking(tracking_uri, config.tracking.experiment_name)
        client = mlflow.MlflowClient()
        client.log_artifact(args.mlflow_run_id, str(args.output), artifact_path="benchmarks")
        for name, value in (
            ("video_throughput_fps", report["throughput_fps"]),
            ("video_latency_p50_ms", report["end_to_end_latency_ms"]["p50"]),
            ("video_latency_p95_ms", report["end_to_end_latency_ms"]["p95"]),
            ("video_dropped_frames", report["dropped_frames"]),
        ):
            client.log_metric(args.mlflow_run_id, name, float(value))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
