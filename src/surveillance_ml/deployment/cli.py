"""Command-line entry points for ONNX export and benchmarking."""

import argparse
import json
import logging
import os
from pathlib import Path


def export_main() -> None:
    from surveillance_ml.deployment.export import export_checkpoint

    parser = argparse.ArgumentParser(description="Export a classifier checkpoint to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/onnx"))
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--mlflow-run-id")
    args = parser.parse_args()
    model_path, manifest_path, manifest = export_checkpoint(
        args.checkpoint,
        args.output_dir,
        image_size=args.image_size,
        opset=args.opset,
    )
    if args.mlflow_run_id:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        client.log_artifact(args.mlflow_run_id, str(model_path), artifact_path="onnx")
        client.log_artifact(args.mlflow_run_id, str(manifest_path), artifact_path="onnx")
    print(
        json.dumps(
            {
                "model": str(model_path),
                "manifest": str(manifest_path),
                "model_sha256": manifest.model_sha256,
                "class_names": list(manifest.class_names),
            },
            indent=2,
        )
    )


def benchmark_main() -> None:
    from surveillance_ml.deployment.benchmark import benchmark_image, write_benchmark_report
    from surveillance_ml.inference.onnx_classifier import OnnxClassifier

    parser = argparse.ArgumentParser(description="Benchmark ONNX Runtime on one image")
    parser.add_argument("--model", type=Path, default=Path("artifacts/onnx/model.onnx"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/onnx/manifest.json"),
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/onnx/benchmark.json"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--mlflow-run-id")
    args = parser.parse_args()
    classifier = OnnxClassifier(args.model, args.manifest)
    report = benchmark_image(classifier, args.image, args.warmup, args.iterations)
    write_benchmark_report(report, args.output, args.mlflow_run_id)
    print(json.dumps(report, indent=2))


def serve_main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    if not os.environ.get("SURVEILLANCE_API_KEY"):
        raise SystemExit("Set SURVEILLANCE_API_KEY to a secret of at least 32 characters")
    uvicorn.run(
        "surveillance_ml.api.app:create_app",
        factory=True,
        host=os.environ.get("SURVEILLANCE_HOST", "127.0.0.1"),
        port=int(os.environ.get("SURVEILLANCE_PORT", "8000")),
    )
