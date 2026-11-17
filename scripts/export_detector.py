import argparse
import json
from pathlib import Path

import mlflow

from surveillance_ml.detection.config import load_detection_config
from surveillance_ml.detection.dataset import validate_yolo_dataset
from surveillance_ml.detection.onnx import export_detector
from surveillance_ml.tracking.mlflow_tracker import configure_tracking


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and verify a detection ONNX bundle.")
    parser.add_argument("--config", type=Path, default=Path("configs/detection.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/detection/onnx"))
    args = parser.parse_args()
    config = load_detection_config(args.config)
    report = validate_yolo_dataset(config.dataset.root)
    provenance_path = args.checkpoint.parent.parent / "provenance.json"
    parent_run_id = None
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if isinstance(provenance, dict) and isinstance(provenance.get("mlflow_run_id"), str):
            parent_run_id = provenance["mlflow_run_id"]

    model_path, manifest_path, parity = export_detector(
        args.checkpoint,
        args.output_dir,
        model_id=config.export.model_id,
        model_version=config.export.version,
        image_size=config.training.image_size,
        opset=config.export.opset,
        confidence_threshold=config.evaluation.confidence_threshold,
        iou_threshold=config.evaluation.nms_iou_threshold,
        max_detections=config.evaluation.max_detections,
        dataset_fingerprint=report.dataset_fingerprint,
        mlflow_run_id=parent_run_id,
    )
    parity_path = args.output_dir / "parity.json"
    parity_path.write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    tracking_uri = config.tracking.uri
    configure_tracking(tracking_uri, config.tracking.experiment_name)
    with mlflow.start_run(
        run_name=f"{config.export.model_id}-{config.export.version}-export"
    ) as run:
        mlflow.set_tags(
            {
                "task": "object-detection-export",
                "architecture": config.model.architecture,
                "source_training_run_id": parent_run_id or "unavailable",
                "dataset_fingerprint": report.dataset_fingerprint,
            }
        )
        mlflow.log_params(
            {
                "model_id": config.export.model_id,
                "model_version": config.export.version,
                "opset": config.export.opset,
                "checkpoint": args.checkpoint.name,
            }
        )
        mlflow.log_artifact(str(model_path), artifact_path="onnx")
        mlflow.log_artifact(str(manifest_path), artifact_path="onnx")
        mlflow.log_artifact(str(parity_path), artifact_path="onnx")
    print(
        json.dumps(
            {
                "model": str(model_path),
                "manifest": str(manifest_path),
                "parity": parity,
                "mlflow_export_run_id": run.info.run_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
