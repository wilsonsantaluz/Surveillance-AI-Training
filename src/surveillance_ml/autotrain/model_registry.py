"""Local candidate-bundle verification and cataloging; this never deploys a model."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from surveillance_ml.detection.manifest import (
    load_detection_manifest,
    verify_detection_model_hash,
)


def _write_json(destination: Path, payload: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def package_candidate(
    candidate_dir: Path,
    *,
    validation: dict[str, Any],
    benchmark: dict[str, Any],
    qualification: dict[str, Any],
    export_parity: dict[str, Any],
    dotnet_parity: dict[str, Any] | None,
) -> dict[str, Any]:
    """Verify exported files and write the candidate's qualification artifacts."""
    manifest_path = candidate_dir / "manifest.json"
    model_path = candidate_dir / "model.onnx"
    manifest = load_detection_manifest(manifest_path)
    verify_detection_model_hash(model_path, manifest)
    _write_json(candidate_dir / "validation.json", validation)
    _write_json(candidate_dir / "benchmark.json", benchmark)
    _write_json(candidate_dir / "export-parity.json", export_parity)
    if dotnet_parity is not None:
        _write_json(candidate_dir / "python-dotnet-parity.json", dotnet_parity)
    _write_json(candidate_dir / "qualification.json", qualification)
    record = {
        "model_id": manifest.model_id,
        "model_version": manifest.model_version,
        "architecture": manifest.architecture,
        "model_file": model_path.name,
        "manifest_file": manifest_path.name,
        "model_sha256": manifest.model_sha256,
        "checkpoint_sha256": manifest.checkpoint_sha256,
        "dataset_fingerprint": manifest.dataset_fingerprint,
        "mlflow_run_id": manifest.mlflow_run_id,
        "qualification_status": qualification.get("qualification_status"),
        "deployment_approved": False,
    }
    _write_json(candidate_dir / "registry-record.json", record)
    return record
