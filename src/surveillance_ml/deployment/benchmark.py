"""Repeatable ONNX Runtime latency measurements for one input image."""

import json
import platform
import statistics
import time
from pathlib import Path
from typing import cast

import numpy as np
import onnxruntime as ort
from PIL import Image

from surveillance_ml.core.image_preprocessing import preprocess_image
from surveillance_ml.inference.onnx_classifier import OnnxClassifier


def benchmark_image(
    classifier: OnnxClassifier,
    image_path: Path,
    warmup: int = 5,
    iterations: int = 50,
) -> dict[str, object]:
    """Measure model-only CPU latency and report actual observed timings."""
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    with Image.open(image_path) as image:
        height = int(classifier.manifest.input_shape[2])
        width = int(classifier.manifest.input_shape[3])
        batch = preprocess_image(image, width, height)[None, ...]
    for _ in range(warmup):
        classifier.predict_tensor(batch)
    latencies_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter()
        classifier.predict_tensor(batch)
        latencies_ms.append((time.perf_counter() - started) * 1000)
    mean_latency = statistics.fmean(latencies_ms)
    return {
        "model_sha256": classifier.manifest.model_sha256,
        "image": image_path.name,
        "input_shape": list(batch.shape),
        "provider": "CPUExecutionProvider",
        "onnxruntime_version": ort.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "intra_op_threads": classifier.intra_op_threads,
        "warmup_iterations": warmup,
        "measured_iterations": iterations,
        "latency_ms": {
            "mean": mean_latency,
            "median": statistics.median(latencies_ms),
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "minimum": min(latencies_ms),
            "maximum": max(latencies_ms),
        },
        "throughput_images_per_second": 1000 / mean_latency,
        "timing_scope": "ONNX inference only; image decoding and preprocessing excluded",
    }


def write_benchmark_report(
    report: dict[str, object],
    output_path: Path,
    mlflow_run_id: str | None = None,
) -> None:
    """Persist measured benchmark data and optionally attach it to an MLflow run."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if mlflow_run_id is None:
        return
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    latency = cast(dict[str, float], report["latency_ms"])
    for metric, value in {
        "onnx_latency_mean_ms": latency["mean"],
        "onnx_latency_p50_ms": latency["p50"],
        "onnx_latency_p95_ms": latency["p95"],
        "onnx_throughput_images_per_second": cast(float, report["throughput_images_per_second"]),
    }.items():
        client.log_metric(mlflow_run_id, metric, value)
    client.log_artifact(mlflow_run_id, str(output_path), artifact_path="onnx_benchmarks")
