"""Hardware checks and cumulative training-stage execution using Phase 4 APIs."""

import csv
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from surveillance_ml.autotrain.config import AutoTrainConfig, resolve_tracking_uri
from surveillance_ml.detection.config import (
    DetectionConfig,
    load_detection_config,
)
from surveillance_ml.detection.dataset import validate_yolo_dataset
from surveillance_ml.detection.training import DetectionTrainingOutcome, train_detector


@dataclass(frozen=True)
class EnvironmentReport:
    python: str
    torch: str
    torchvision: str
    ultralytics: str
    cuda_available: bool
    cuda_device: str | None
    resolved_device: str
    cpu_count: int
    disk_free_bytes: int
    pretrained_weights: str
    output_writable: bool
    dvc: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "python": self.python,
            "torch": self.torch,
            "torchvision": self.torchvision,
            "ultralytics": self.ultralytics,
            "cuda_available": self.cuda_available,
            "cuda_device": self.cuda_device,
            "resolved_device": self.resolved_device,
            "cpu_count": self.cpu_count,
            "disk_free_bytes": self.disk_free_bytes,
            "pretrained_weights": self.pretrained_weights,
            "output_writable": self.output_writable,
            "dvc": self.dvc,
        }


def _dvc_status(config: AutoTrainConfig) -> dict[str, Any]:
    if not config.tracking.dvc_enabled:
        return {"enabled": False, "status": "disabled"}
    executable = shutil.which("dvc")
    if executable is None:
        return {"enabled": True, "status": "not-installed"}
    result: dict[str, Any] = {"enabled": True, "status": "unknown"}
    for filename in (".dvc/config", "dvc.yaml", "dvc.lock", "pipelines/detection/dvc.yaml"):
        path = config.repository_root / filename
        if path.is_file():
            import hashlib

            result[f"{filename}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        process = subprocess.run(
            [executable, "status", "--json"],
            cwd=config.repository_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {**result, "status": "unavailable", "message": str(error)}
    result["status"] = "clean" if process.returncode == 0 else "changes-or-error"
    result["exit_code"] = process.returncode
    result["stdout"] = process.stdout.strip()[:4000]
    result["stderr"] = process.stderr.strip()[:4000]
    return result


def check_environment(config: AutoTrainConfig, run_dir: Path) -> EnvironmentReport:
    """Verify runtime, device, model-weight load, writable output, and DVC context."""
    if sys.version_info < (3, 12) or sys.version_info >= (3, 15):
        raise RuntimeError("AutoTrain requires Python >=3.12,<3.15")
    run_dir.mkdir(parents=True, exist_ok=True)
    test_file = run_dir / ".write-check"
    try:
        test_file.write_text("ok", encoding="utf-8")
    except OSError as error:
        raise RuntimeError(f"AutoTrain output directory is not writable: {run_dir}") from error
    finally:
        test_file.unlink(missing_ok=True)
    cuda_available = torch.cuda.is_available()
    if config.training.device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but PyTorch reports that it is unavailable")
    resolved_device = (
        "cuda:0"
        if config.training.device == "cuda" or (config.training.device == "auto" and cuda_available)
        else "cpu"
    )
    cuda_device = torch.cuda.get_device_name(0) if cuda_available else None
    weight_reference = (
        f"{config.model.architecture}.pt"
        if config.model.pretrained
        else f"{config.model.architecture}.yaml"
    )
    try:
        YOLO(weight_reference)
    except Exception as error:
        raise RuntimeError(
            f"Unable to load model initialization {weight_reference}: {error}"
        ) from error
    disk_free = shutil.disk_usage(run_dir).free
    try:
        import torchvision

        torchvision_version = torchvision.__version__
    except ImportError:
        torchvision_version = "not-installed"
    return EnvironmentReport(
        python=sys.version.split()[0],
        torch=torch.__version__,
        torchvision=torchvision_version,
        ultralytics=__import__("ultralytics").__version__,
        cuda_available=cuda_available,
        cuda_device=cuda_device,
        resolved_device=resolved_device,
        cpu_count=os.cpu_count() or 1,
        disk_free_bytes=disk_free,
        pretrained_weights=weight_reference,
        output_writable=True,
        dvc=_dvc_status(config),
    )


def _detection_config(
    config: AutoTrainConfig,
    run_dir: Path,
    epochs: int,
) -> DetectionConfig:
    base = load_detection_config(config.repository_root / "configs" / "detection.yaml")
    return replace(
        base,
        dataset=replace(
            base.dataset,
            root=config.dataset.path,
            version=config.dataset.version,
        ),
        model=replace(
            base.model,
            architecture=config.model.architecture,
            pretrained=config.model.pretrained,
        ),
        training=replace(
            base.training,
            output_dir=run_dir / "training",
            image_size=config.model.image_size,
            epochs=epochs,
            batch_size=config.training.batch_size,
            learning_rate=config.training.learning_rate,
            optimizer=config.training.optimizer,
            patience=config.training.patience,
            seed=config.experiment.seed,
            device=config.training.device,
            workers=config.training.workers,
            cpu_threads=config.training.cpu_threads,
            mixed_precision=config.training.mixed_precision,
        ),
        tracking=replace(
            base.tracking,
            uri=resolve_tracking_uri(config.tracking.uri, config.repository_root),
            experiment_name=config.tracking.experiment_name,
        ),
        export=replace(
            base.export,
            model_id=config.export.model_id,
            version=config.export.version,
            opset=config.export.opset,
        ),
    )


def _completed_epochs(results_csv: Path) -> int:
    if not results_csv.is_file():
        return 0
    with results_csv.open(newline="", encoding="utf-8") as file:
        return sum(1 for _ in csv.DictReader(file))


def _recover_outcome(
    config: AutoTrainConfig,
    results_dir: Path,
    stage_records: list[dict[str, Any]],
) -> DetectionTrainingOutcome:
    best_checkpoint = results_dir / "weights" / "best.pt"
    last_checkpoint = results_dir / "weights" / "last.pt"
    if not best_checkpoint.is_file() or not last_checkpoint.is_file():
        raise FileNotFoundError("Cannot recover completed training without best.pt and last.pt")
    previous_run = next(
        (
            str(item["mlflow_run_id"])
            for item in reversed(stage_records)
            if item.get("mlflow_run_id")
        ),
        None,
    )
    return DetectionTrainingOutcome(
        checkpoint_path=best_checkpoint,
        results_dir=results_dir,
        mlflow_run_id=previous_run,
        epochs_completed=_completed_epochs(results_dir / "results.csv"),
        training_seconds=0.0,
        dataset_fingerprint=validate_yolo_dataset(config.dataset.path).dataset_fingerprint,
    )


def run_training_stages(
    config: AutoTrainConfig,
    run_dir: Path,
    *,
    completed_stages: set[str],
    completed_stage_records: list[dict[str, Any]],
    stop_check: Callable[[], str | None],
    on_stage_start: Callable[[str], None],
    on_stage_complete: Callable[[str, DetectionTrainingOutcome, int], None],
) -> tuple[DetectionTrainingOutcome, str | None]:
    """Run smoke, pilot, and main stages at cumulative epoch milestones."""
    milestones = (
        ("smoke", config.training.smoke_epochs),
        ("pilot", config.training.pilot_epochs),
        ("main", config.training.max_total_epochs),
    )
    completed_order = [name for name, _ in milestones if name in completed_stages]
    if completed_order != [name for name, _ in milestones[: len(completed_order)]]:
        raise ValueError("Completed training stages are not a valid cumulative prefix")
    latest: DetectionTrainingOutcome | None = None
    resume_checkpoint: Path | None = None
    if completed_stages:
        last_stage = next(name for name, _ in reversed(milestones) if name in completed_stages)
        resume_checkpoint = run_dir / "training" / "weights" / "last.pt"
        if not resume_checkpoint.is_file():
            raise FileNotFoundError(
                f"Completed training stage {last_stage!r} has no resumable last checkpoint"
            )
    else:
        interrupted_checkpoint = run_dir / "training" / "weights" / "last.pt"
        if interrupted_checkpoint.is_file():
            resume_checkpoint = interrupted_checkpoint
    for stage_name, total_epochs in milestones:
        if stage_name in completed_stages:
            continue
        results_dir = run_dir / "training"
        if (
            resume_checkpoint is not None
            and _completed_epochs(results_dir / "results.csv") >= total_epochs
        ):
            latest = _recover_outcome(config, results_dir, completed_stage_records)
            on_stage_complete(stage_name, latest, total_epochs)
            resume_checkpoint = latest.results_dir / "weights" / "last.pt"
            continue
        on_stage_start(stage_name)
        detection_config = _detection_config(
            config,
            run_dir,
            config.training.max_total_epochs,
        )
        latest = train_detector(
            detection_config,
            run_name=f"{config.experiment.name}-{stage_name}",
            resume_from=resume_checkpoint,
            stop_check=stop_check,
            stop_after_epochs=total_epochs,
        )
        last_checkpoint = latest.results_dir / "weights" / "last.pt"
        if not latest.checkpoint_path.is_file() or not last_checkpoint.is_file():
            raise FileNotFoundError("YOLO training did not persist both best and last checkpoints")
        on_stage_complete(stage_name, latest, total_epochs)
        if latest.stop_reason in {"runtime_limit", "operator_requested"}:
            return latest, latest.stop_reason
        resume_checkpoint = last_checkpoint
        if latest.stop_reason == "stage_boundary":
            if stage_name == "main":
                return latest, None
            continue
        if latest.epochs_completed < total_epochs:
            return latest, "early_stopping"
    if latest is None:
        previous = next(
            (
                item
                for item in reversed(completed_stage_records)
                if item.get("stage") in completed_stages
            ),
            None,
        )
        if previous is None:
            raise RuntimeError("No training stage was run or resumable stage metadata is missing")
        latest = DetectionTrainingOutcome(
            checkpoint_path=Path(str(previous["best_checkpoint"])),
            results_dir=Path(str(previous["results_dir"])),
            mlflow_run_id=str(previous["mlflow_run_id"]),
            epochs_completed=int(previous["epochs_completed"]),
            training_seconds=float(
                previous.get("cumulative_training_seconds", previous["training_seconds"])
            ),
            dataset_fingerprint=str(previous["dataset_fingerprint"]),
            stop_reason=(
                str(previous["stop_reason"]) if previous.get("stop_reason") is not None else None
            ),
        )
    recovered_stop_reason = (
        latest.stop_reason if latest.stop_reason != "stage_boundary" else None
    )
    return latest, recovered_stop_reason


def runtime_stop_check(
    run_dir: Path, started: float, maximum_seconds: float
) -> Callable[[], str | None]:
    def check() -> str | None:
        from surveillance_ml.autotrain.state import cancellation_reason

        requested = cancellation_reason(run_dir)
        if requested is not None:
            return requested
        if time.monotonic() - started >= maximum_seconds:
            return "runtime_limit"
        return None

    return check


def write_environment_report(report: EnvironmentReport, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
