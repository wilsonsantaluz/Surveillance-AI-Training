"""AutoTrain workflow composition around the existing detection pipeline."""

import json
import shutil
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from surveillance_ml.autotrain.compatibility import run_python_dotnet_parity
from surveillance_ml.autotrain.config import (
    AutoTrainConfig,
    load_autotrain_config,
    resolve_tracking_uri,
    save_config_snapshot,
)
from surveillance_ml.autotrain.dataset_audit import AuditResult, audit_dataset
from surveillance_ml.autotrain.dataset_source import resolve_dataset
from surveillance_ml.autotrain.evaluation_gate import (
    GateEvaluation,
    evaluate_pre_export_gates,
    finalize_qualification,
)
from surveillance_ml.autotrain.experiment_runner import (
    EnvironmentReport,
    check_environment,
    run_training_stages,
    runtime_stop_check,
    write_environment_report,
)
from surveillance_ml.autotrain.model_registry import package_candidate
from surveillance_ml.autotrain.reporting import (
    generate_error_reports,
    write_autotrain_report,
)
from surveillance_ml.autotrain.state import (
    AutoTrainState,
    RunLock,
    StateStore,
    cancellation_reason,
    clear_cancellation,
    request_cancellation,
    run_directory,
    utc_now,
)
from surveillance_ml.detection.config import DetectionConfig
from surveillance_ml.detection.dataset import DetectionDatasetReport, validate_yolo_dataset
from surveillance_ml.detection.manifest import load_detection_manifest, verify_detection_model_hash
from surveillance_ml.detection.onnx import OnnxDetector, export_detector
from surveillance_ml.detection.training import evaluate_and_log_detector
from surveillance_ml.tracking.mlflow_tracker import configure_tracking


@contextmanager
def _tracking_action(
    config: AutoTrainConfig,
    state: AutoTrainState,
    store: StateStore,
    action: str,
) -> Iterator[Any]:
    import mlflow

    configure_tracking(
        resolve_tracking_uri(config.tracking.uri, config.repository_root),
        config.tracking.experiment_name,
    )
    active = mlflow.active_run()
    with mlflow.start_run(
        run_name=f"{config.experiment.name}-{action}-{state.run_id}",
        nested=active is not None,
    ) as run:
        state.mlflow_run_ids.append(run.info.run_id)
        state.updated_at = utc_now()
        store.save(state)
        mlflow.set_tags(
            {
                "task": "autotrain",
                "autotrain_run_id": state.run_id,
                "autotrain_action": action,
                "dataset_version": config.dataset.version,
                "dataset_source": config.dataset.source,
                "dataset_license": config.dataset.license,
                "license_review_status": config.dataset.license_review_status,
                "configuration_fingerprint": config.fingerprint,
            }
        )
        mlflow.log_params(
            {
                "experiment_name": config.experiment.name,
                "model_architecture": config.model.architecture,
                "pretrained": config.model.pretrained,
                "image_size": config.model.image_size,
                "seed": config.experiment.seed,
                "smoke_epochs": config.training.smoke_epochs,
                "pilot_epochs": config.training.pilot_epochs,
                "max_total_epochs": config.training.max_total_epochs,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
                "optimizer": config.training.optimizer,
                "configuration_fingerprint": config.fingerprint,
            }
        )
        yield run


def _new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def _new_state(run_id: str, config: AutoTrainConfig) -> AutoTrainState:
    now = utc_now()
    return AutoTrainState(
        run_id=run_id,
        status="CREATED",
        current_stage="created",
        completed_stages=[],
        started_at=now,
        updated_at=now,
        ended_at=None,
        config_fingerprint=config.fingerprint,
        dataset_fingerprint=None,
    )


def _save_report(
    run_dir: Path,
    store: StateStore,
    state: AutoTrainState,
    audit: AuditResult | None,
    environment: EnvironmentReport | None,
    metrics: dict[str, Any] | None,
    qualification: GateEvaluation | None,
) -> None:
    audit_data: dict[str, Any] = (
        audit.to_dict() if audit is not None else _read_json(run_dir / "dataset-audit.json", {})
    )
    environment_data = (
        environment.to_dict()
        if environment is not None
        else _read_json(
            run_dir / "environment.json",
            None,
        )
    )
    report_path = write_autotrain_report(
        run_dir,
        state=state.to_dict(),
        audit=audit_data,
        environment=environment_data,
        metrics=metrics or _read_json(run_dir / "validation.json", None),
        qualification=(
            qualification.to_dict()
            if qualification is not None
            else _read_json(run_dir / "qualification.json", None)
        ),
    )
    store.save(state)
    state.execution["report_path"] = str(report_path)
    store.save(state)


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.is_file():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON artifact: {path}") from error


def _log_file(path: Path, artifact_path: str | None = None) -> None:
    import mlflow

    if path.is_file():
        mlflow.log_artifact(str(path), artifact_path=artifact_path)


def _log_directory(path: Path, artifact_path: str) -> None:
    import mlflow

    if path.is_dir() and any(path.iterdir()):
        mlflow.log_artifacts(str(path), artifact_path=artifact_path)


def _environment(config: AutoTrainConfig, run_dir: Path) -> EnvironmentReport:
    report = check_environment(config, run_dir)
    write_environment_report(report, run_dir / "environment.json")
    return report


def _benchmark_candidate(
    model_path: Path,
    manifest_path: Path,
    dataset_report: DetectionDatasetReport,
    *,
    maximum_images: int = 20,
) -> dict[str, Any]:
    import platform
    import statistics
    import time

    detector = OnnxDetector(model_path, manifest_path)
    records = sorted(
        (record for record in dataset_report.records if record.split == "val"),
        key=lambda record: record.image_path.as_posix(),
    )[:maximum_images]
    if not records:
        raise ValueError("No validation images are available for the candidate benchmark")
    detector.predict_image(records[0].image_path)
    elapsed: list[float] = []
    detection_counts: list[int] = []
    for record in records:
        started = time.perf_counter()
        result = detector.predict_image(record.image_path)
        elapsed.append(time.perf_counter() - started)
        detection_counts.append(len(result.detections))
    ordered = sorted(elapsed)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "measurement": (
            "single-image Python ONNX Runtime CPU inference, including image decode "
            "and preprocessing"
        ),
        "images": len(elapsed),
        "mean_seconds": statistics.fmean(elapsed),
        "median_seconds": statistics.median(elapsed),
        "p95_seconds": ordered[p95_index],
        "detections_per_image": detection_counts,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor() or "unknown",
        },
        "real_time_claim": False,
    }


def _evaluation_config(config: AutoTrainConfig) -> DetectionConfig:
    from dataclasses import replace

    from surveillance_ml.detection.config import load_detection_config

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
            image_size=config.model.image_size,
            device=config.training.device,
            workers=config.training.workers,
            cpu_threads=config.training.cpu_threads,
        ),
        tracking=replace(
            base.tracking,
            uri=resolve_tracking_uri(config.tracking.uri, config.repository_root),
            experiment_name=config.tracking.experiment_name,
        ),
    )


def _execute(
    config: AutoTrainConfig,
    store: StateStore,
    state: AutoTrainState,
    *,
    audit_only: bool,
) -> AutoTrainState:
    run_dir = store.run_dir
    audit: AuditResult | None = None
    environment: EnvironmentReport | None = None
    metrics: dict[str, Any] | None = None
    qualification: GateEvaluation | None = None
    with _tracking_action(config, state, store, "audit" if audit_only else "run"):
        state.status = "AUDITING"
        state.current_stage = "dataset_audit"
        state.updated_at = utc_now()
        store.save(state)
        resolved_root = resolve_dataset(config.dataset)
        if resolved_root != config.dataset.path.resolve():
            raise RuntimeError("Dataset source resolved to an unexpected path")
        audit = audit_dataset(config, run_dir)
        if (
            state.dataset_fingerprint is not None
            and audit.dataset_fingerprint != state.dataset_fingerprint
        ):
            raise ValueError(
                "Dataset fingerprint changed since this run was created; refusing to resume."
            )
        state.dataset_fingerprint = audit.dataset_fingerprint
        state.audit_passed = audit.valid
        state.completed_stages = (
            state.completed_stages
            if "audit" in state.completed_stages
            else [*state.completed_stages, "audit"]
        )
        import mlflow

        mlflow.set_tag("dataset_fingerprint", audit.dataset_fingerprint or "unavailable")
        mlflow.log_metrics(
            {
                "dataset_image_count": float(audit.image_count),
                "dataset_positive_image_count": float(audit.positive_image_count),
                "dataset_negative_image_count": float(audit.negative_image_count),
                **{
                    f"dataset_boxes_{name}": float(count)
                    for name, count in audit.class_counts.items()
                },
            }
        )
        mlflow.log_dict(audit.to_dict(), "dataset/dataset-audit.json")
        _log_file(run_dir / "dataset-audit.md", "dataset")
        _log_file(run_dir / "class-distribution.csv", "dataset")
        _log_file(run_dir / "annotation-issues.csv", "dataset")
        _log_directory(run_dir / "annotation-review", "annotation-review")
        if not audit.valid:
            state.failed_stage = "dataset_audit"
            state.status = "FAILED"
            state.stop_reason = "dataset_integrity_failure"
            state.ended_at = utc_now()
            state.updated_at = state.ended_at
            store.save(state)
            _save_report(run_dir, store, state, audit, None, None, None)
            return state
        if audit_only:
            state.status = "COMPLETED"
            state.current_stage = "audit"
            state.stop_reason = "audit_only"
            state.ended_at = utc_now()
            state.updated_at = state.ended_at
            _save_report(run_dir, store, state, audit, None, None, None)
            _log_file(run_dir / "autotrain-report.md", "reports")
            return state

        if config.audit.require_manual_annotation_review:
            review = state.annotation_review or {}
            decision = review.get("decision")
            if decision != "approved":
                if decision == "rejected":
                    state.status = "CANCELLED"
                    state.stop_reason = "annotation_review_rejected"
                    state.ended_at = utc_now()
                    state.updated_at = state.ended_at
                else:
                    state.status = "WAITING_FOR_REVIEW"
                    state.current_stage = "annotation_review"
                    state.stop_reason = "human_annotation_review_required"
                    state.updated_at = utc_now()
                _save_report(run_dir, store, state, audit, environment, None, None)
                _log_file(run_dir / "autotrain-report.md", "reports")
                return state
        environment = _environment(config, run_dir)
        mlflow.log_dict(environment.to_dict(), "execution/environment.json")
        if "environment" not in state.completed_stages:
            state.completed_stages.append("environment")
        state.execution["environment_path"] = str(run_dir / "environment.json")
        clear_cancellation(run_dir)
        state.stop_reason = None
        invocation_started = time.monotonic()
        stop_check = runtime_stop_check(
            run_dir,
            invocation_started,
            config.training.max_runtime_minutes * 60,
        )

        def on_stage_start(name: str) -> None:
            state.status = {
                "smoke": "SMOKE_TRAINING",
                "pilot": "PILOT_TRAINING",
                "main": "MAIN_TRAINING",
            }[name]
            state.current_stage = name
            state.failed_stage = None
            state.updated_at = utc_now()
            store.save(state)
            mlflow.log_param("current_training_stage", name)

        def on_stage_complete(name: str, outcome: Any, target_epochs: int) -> None:
            state.checkpoint_paths[name] = str(outcome.checkpoint_path)
            state.checkpoint_paths["last"] = str(outcome.results_dir / "weights" / "last.pt")
            training_stages = state.execution.setdefault("training_stages", [])
            cumulative_training_seconds = (
                sum(float(item.get("training_seconds", 0.0)) for item in training_stages)
                + outcome.training_seconds
            )
            stage_details = {
                "stage": name,
                "target_total_epochs": target_epochs,
                "epochs_completed": outcome.epochs_completed,
                "training_seconds": outcome.training_seconds,
                "cumulative_training_seconds": cumulative_training_seconds,
                "best_checkpoint": str(outcome.checkpoint_path),
                "last_checkpoint": state.checkpoint_paths["last"],
                "results_dir": str(outcome.results_dir),
                "dataset_fingerprint": outcome.dataset_fingerprint,
                "stop_reason": outcome.stop_reason,
                "mlflow_run_id": outcome.mlflow_run_id,
            }
            training_stages.append(stage_details)
            if outcome.mlflow_run_id and outcome.mlflow_run_id not in state.mlflow_run_ids:
                state.mlflow_run_ids.append(outcome.mlflow_run_id)
            if outcome.stop_reason not in {"runtime_limit", "operator_requested"}:
                if name not in state.completed_stages:
                    state.completed_stages.append(name)
            elapsed_epochs = max(1, outcome.epochs_completed)
            measured_seconds_per_epoch = cumulative_training_seconds / elapsed_epochs
            state.execution["measured_seconds_per_epoch"] = measured_seconds_per_epoch
            state.execution["estimated_seconds_remaining_at_max_epochs"] = (
                measured_seconds_per_epoch
                * max(0, config.training.max_total_epochs - outcome.epochs_completed)
            )
            state.updated_at = utc_now()
            store.save(state)

        try:
            outcome, stop_reason = run_training_stages(
                config,
                run_dir,
                completed_stages=set(state.completed_stages).intersection(
                    {"smoke", "pilot", "main"}
                ),
                completed_stage_records=state.execution.get("training_stages", []),
                stop_check=stop_check,
                on_stage_start=on_stage_start,
                on_stage_complete=on_stage_complete,
            )
        except Exception as error:
            state.failed_stage = state.current_stage
            state.status = "FAILED"
            state.stop_reason = f"training_failed: {error}"
            state.ended_at = utc_now()
            state.updated_at = state.ended_at
            store.save(state)
            _save_report(run_dir, store, state, audit, environment, None, None)
            raise
        if stop_reason in {"runtime_limit", "operator_requested"}:
            state.status = "CANCELLED"
            state.stop_reason = stop_reason
            state.updated_at = utc_now()
            state.ended_at = state.updated_at
            _save_report(run_dir, store, state, audit, environment, None, None)
            _log_file(run_dir / "autotrain-report.md", "reports")
            return state

        state.status = "EVALUATING"
        state.current_stage = "validation_evaluation"
        state.updated_at = utc_now()
        store.save(state)
        evaluation_config = _evaluation_config(config)
        validation_path = run_dir / "validation.json"
        metrics = evaluate_and_log_detector(
            evaluation_config,
            outcome.checkpoint_path,
            split="val",
            output_path=validation_path,
            parent_run_id=outcome.mlflow_run_id,
            collect_samples=True,
        )
        mlflow.log_metrics(
            {
                "validation_precision": float(metrics["precision"]),
                "validation_recall": float(metrics["recall"]),
                "validation_map50": float(metrics["mAP50"]),
                "validation_map50_95": float(metrics["mAP50_95"]),
            }
        )
        generate_error_reports(config.dataset.path, run_dir, metrics)
        _log_file(validation_path, "evaluation")
        _log_directory(run_dir / "error-analysis", "error-analysis")
        if "validation_evaluation" not in state.completed_stages:
            state.completed_stages.append("validation_evaluation")
        state.execution["validation_evaluation"] = str(validation_path)
        pre_export = evaluate_pre_export_gates(config, audit, metrics)
        qualification = pre_export
        if (
            config.export.export_candidate
            and pre_export.performance_gates_passed
            and pre_export.dataset_integrity_passed
        ):
            state.status = "EXPORTING"
            state.current_stage = "candidate_export"
            state.updated_at = utc_now()
            store.save(state)
            candidate_dir = run_dir / "candidate"
            model_path, manifest_path, export_parity = export_detector(
                outcome.checkpoint_path,
                candidate_dir,
                model_id=config.export.model_id,
                model_version=config.export.version,
                image_size=config.model.image_size,
                opset=config.export.opset,
                dataset_fingerprint=audit.dataset_fingerprint,
                mlflow_run_id=outcome.mlflow_run_id,
            )
            manifest = load_detection_manifest(manifest_path)
            verify_detection_model_hash(model_path, manifest)
            manifest_validated = True
            dotnet_parity: dict[str, Any] | None = None
            if config.quality_gates.require_python_dotnet_parity:
                dataset_report = validate_yolo_dataset(config.dataset.path)
                dotnet_parity = run_python_dotnet_parity(
                    model_path,
                    manifest_path,
                    dataset_report,
                    config.repository_root,
                )
                (run_dir / "python-dotnet-parity.json").write_text(
                    json.dumps(dotnet_parity, indent=2) + "\n",
                    encoding="utf-8",
                )
            else:
                dotnet_parity = {"passed": False, "status": "not_required"}
            parity_passed = (
                bool(dotnet_parity.get("passed"))
                if config.quality_gates.require_python_dotnet_parity
                else True
            )
            qualification = finalize_qualification(
                config,
                pre_export,
                manifest_validated=manifest_validated,
                python_dotnet_parity=parity_passed,
            )
            benchmark = _benchmark_candidate(
                model_path,
                manifest_path,
                validate_yolo_dataset(config.dataset.path),
            )
            if qualification.candidate_qualified:
                state.status = "EVALUATING"
                state.current_stage = "final_test_evaluation"
                state.updated_at = utc_now()
                store.save(state)
                test_metrics = evaluate_and_log_detector(
                    evaluation_config,
                    outcome.checkpoint_path,
                    split="test",
                    output_path=run_dir / "test-evaluation.json",
                    parent_run_id=outcome.mlflow_run_id,
                    collect_samples=True,
                )
                generate_error_reports(
                    config.dataset.path,
                    run_dir / "test-review",
                    test_metrics,
                )
                (run_dir / "test-review" / "test-evaluation.json").write_text(
                    json.dumps(test_metrics, indent=2) + "\n",
                    encoding="utf-8",
                )
                shutil.copyfile(
                    run_dir / "test-review" / "test-evaluation.json",
                    candidate_dir / "test-evaluation.json",
                )
                if "final_test_evaluation" not in state.completed_stages:
                    state.completed_stages.append("final_test_evaluation")
                state.execution["test_evaluation"] = str(run_dir / "test-evaluation.json")
            package_candidate(
                candidate_dir,
                validation=metrics,
                benchmark=benchmark,
                qualification=qualification.to_dict(),
                export_parity=export_parity,
                dotnet_parity=dotnet_parity,
            )
            if "candidate_export" not in state.completed_stages:
                state.completed_stages.append("candidate_export")
            state.execution["candidate_dir"] = str(candidate_dir)
            state.qualification_status = qualification.qualification_status
            mlflow.log_dict(qualification.to_dict(), "qualification/qualification.json")
            _log_directory(candidate_dir, "candidate")
        else:
            qualification = finalize_qualification(
                config,
                pre_export,
                manifest_validated=None,
                python_dotnet_parity=None,
            )
            state.qualification_status = qualification.qualification_status
            mlflow.log_dict(qualification.to_dict(), "qualification/qualification.json")

        (run_dir / "qualification.json").write_text(
            json.dumps(qualification.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        state.status = "COMPLETED"
        state.current_stage = "completed"
        state.stop_reason = (
            stop_reason or qualification.stopping_reason or "training_and_validation_completed"
        )
        state.ended_at = utc_now()
        state.updated_at = state.ended_at
        _save_report(run_dir, store, state, audit, environment, metrics, qualification)
        _log_file(run_dir / "autotrain-report.md", "reports")
        _log_file(run_dir / "autotrain-report.json", "reports")
        return state


def create_run(config: AutoTrainConfig) -> tuple[str, Path]:
    run_id = _new_run_id()
    run_dir = run_directory(config.experiment.output_root, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    store = StateStore(run_dir)
    store.create(_new_state(run_id, config))
    save_config_snapshot(config, run_dir / "autotrain-config.yaml")
    return run_id, run_dir


def run_autotrain(config: AutoTrainConfig) -> AutoTrainState:
    """Create and execute an AutoTrain run; manual review may pause it safely."""
    run_id, run_dir = create_run(config)
    store = StateStore(run_dir)
    with RunLock(run_dir):
        state = store.load()
        try:
            return _execute(config, store, state, audit_only=False)
        except Exception as error:
            if state.status not in {"FAILED", "CANCELLED"}:
                state.status = "FAILED"
                state.failed_stage = state.current_stage
                state.stop_reason = f"run_failed: {error}"
                state.ended_at = utc_now()
                state.updated_at = state.ended_at
                store.save(state)
            raise


def audit_autotrain(config: AutoTrainConfig) -> AutoTrainState:
    """Run a dataset-only audit and create an immutable audit run."""
    _, run_dir = create_run(config)
    store = StateStore(run_dir)
    with RunLock(run_dir):
        state = store.load()
        return _execute(config, store, state, audit_only=True)


def _config_for_run(config: AutoTrainConfig | None, run_dir: Path) -> AutoTrainConfig:
    snapshot = run_dir / "autotrain-config.yaml"
    saved = load_autotrain_config(snapshot)
    if config is not None and config.fingerprint != saved.fingerprint:
        raise ValueError("Provided configuration differs from the immutable run snapshot.")
    return config or saved


def resume_autotrain(
    run_id: str,
    output_root: Path,
    config: AutoTrainConfig | None = None,
) -> AutoTrainState:
    """Resume a paused or interrupted run only when config and data still match."""
    run_dir = run_directory(output_root, run_id)
    store = StateStore(run_dir)
    with RunLock(run_dir):
        state = store.load()
        if state.status == "COMPLETED":
            raise ValueError("Completed AutoTrain runs cannot be resumed.")
        resolved_config = _config_for_run(config, run_dir)
        if state.config_fingerprint != resolved_config.fingerprint:
            raise ValueError("Configuration fingerprint changed; refusing to resume.")
        state.status = "AUDITING"
        state.updated_at = utc_now()
        store.save(state)
        clear_cancellation(run_dir)
        try:
            return _execute(resolved_config, store, state, audit_only=False)
        except Exception as error:
            if state.status not in {"FAILED", "CANCELLED"}:
                state.status = "FAILED"
                state.failed_stage = state.current_stage
                state.stop_reason = f"resume_failed: {error}"
                state.ended_at = utc_now()
                state.updated_at = state.ended_at
                store.save(state)
            raise


def record_annotation_review(
    run_id: str,
    output_root: Path,
    *,
    decision: str,
    reviewer: str,
    notes: str,
) -> AutoTrainState:
    if decision not in {"approved", "rejected"}:
        raise ValueError("Annotation review decision must be approved or rejected")
    if not reviewer.strip():
        raise ValueError("Reviewer name must not be empty")
    run_dir = run_directory(output_root, run_id)
    store = StateStore(run_dir)
    with RunLock(run_dir):
        state = store.load()
        if state.status != "WAITING_FOR_REVIEW":
            raise ValueError("This run is not waiting for annotation review.")
        state.annotation_review = {
            "decision": decision,
            "reviewer": reviewer.strip(),
            "notes": notes.strip(),
            "reviewed_at": utc_now(),
        }
        if decision == "rejected":
            state.status = "CANCELLED"
            state.stop_reason = "annotation_review_rejected"
            state.ended_at = utc_now()
        state.updated_at = utc_now()
        store.save(state)
        _save_report(
            run_dir,
            store,
            state,
            None,
            None,
            None,
            None,
        )
        return state


def cancel_autotrain(run_id: str, output_root: Path) -> None:
    run_dir = run_directory(output_root, run_id)
    state = StateStore(run_dir).load()
    if state.status in {"COMPLETED", "FAILED", "CANCELLED"}:
        raise ValueError(f"Cannot cancel a run in status {state.status}.")
    request_cancellation(run_dir)


def load_run_state(run_id: str, output_root: Path) -> AutoTrainState:
    return StateStore(run_directory(output_root, run_id)).load()


def load_run_report(run_id: str, output_root: Path) -> dict[str, Any]:
    run_dir = run_directory(output_root, run_id)
    report = _read_json(run_dir / "autotrain-report.json", None)
    if not isinstance(report, dict):
        raise FileNotFoundError(f"AutoTrain report does not exist for run {run_id}")
    return report


def dry_run_plan(config: AutoTrainConfig) -> dict[str, Any]:
    return {
        "experiment": config.experiment.name,
        "dataset": str(config.dataset.path),
        "dataset_source": config.dataset.source,
        "configuration_fingerprint": config.fingerprint,
        "stages": [
            {"name": "audit", "epochs": 0},
            {"name": "smoke", "total_epochs": config.training.smoke_epochs},
            {"name": "pilot", "total_epochs": config.training.pilot_epochs},
            {"name": "main", "total_epochs": config.training.max_total_epochs},
            {"name": "validation_gates", "epochs": 0},
            {"name": "candidate_export", "enabled": config.export.export_candidate},
            {"name": "automatic_deployment", "enabled": False},
        ],
        "manual_annotation_review_required": config.audit.require_manual_annotation_review,
        "license_review_status": config.dataset.license_review_status,
        "quality_gate_thresholds": {
            key: value
            for key, value in config.quality_gates.__dict__.items()
            if key.startswith("min_") or key == "max_false_positives_per_hour"
        },
    }


def current_cancellation_reason(run_id: str, output_root: Path) -> str | None:
    return cancellation_reason(run_directory(output_root, run_id))
