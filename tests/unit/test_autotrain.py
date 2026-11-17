import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from surveillance_ml.autotrain.cli import EXIT_SUCCESS, main
from surveillance_ml.autotrain.config import load_autotrain_config
from surveillance_ml.autotrain.dataset_audit import AuditResult, audit_dataset
from surveillance_ml.autotrain.evaluation_gate import evaluate_pre_export_gates
from surveillance_ml.autotrain.experiment_runner import run_training_stages
from surveillance_ml.autotrain.reporting import _sample_errors, write_autotrain_report
from surveillance_ml.autotrain.state import (
    AutoTrainState,
    StateStore,
    cancellation_reason,
    request_cancellation,
    run_directory,
    utc_now,
)
from surveillance_ml.detection.dataset import validate_yolo_dataset
from surveillance_ml.detection.synthetic import generate_synthetic_detection_dataset
from surveillance_ml.detection.training import DetectionTrainingOutcome

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "autotrain.yaml"


def _config(dataset_path: Path):
    config = load_autotrain_config(CONFIG_PATH)
    return replace(
        config,
        dataset=replace(config.dataset, path=dataset_path),
        audit=replace(
            config.audit,
            minimum_resolution=32,
            detect_perceptual_duplicates=False,
            generate_contact_sheets=True,
        ),
    )


def _state(run_id: str) -> AutoTrainState:
    now = utc_now()
    return AutoTrainState(
        run_id=run_id,
        status="CREATED",
        current_stage="created",
        completed_stages=[],
        started_at=now,
        updated_at=now,
        ended_at=None,
        config_fingerprint="config",
        dataset_fingerprint=None,
    )


def test_config_dry_run_does_not_require_a_dataset(capsys: pytest.CaptureFixture[str]) -> None:
    result = main(["run", "--dry-run", "--config", str(CONFIG_PATH)])

    assert result == EXIT_SUCCESS
    plan = json.loads(capsys.readouterr().out)
    assert plan["stages"][-1] == {"name": "automatic_deployment", "enabled": False}


def test_tracking_environment_override_is_part_of_run_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_autotrain_config(CONFIG_PATH)
    original_fingerprint = config.fingerprint
    override = "sqlite:///different-tracking.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", override)

    assert config.fingerprint != original_fingerprint
    assert config.snapshot()["tracking"]["uri"] == override


def test_audit_reports_distribution_and_creates_contact_sheets(tmp_path: Path) -> None:
    dataset_root = generate_synthetic_detection_dataset(
        tmp_path / "dataset",
        source_count=9,
        image_size=64,
        seed=7,
    ).parent

    result = audit_dataset(_config(dataset_root), tmp_path / "run")

    assert result.valid
    assert result.image_count == 27
    assert result.positive_image_count + result.negative_image_count == result.image_count
    assert result.class_counts["firearm"] > 0
    assert result.class_counts["knife"] > 0
    assert result.contact_sheets
    assert list((tmp_path / "run" / "annotation-review").glob("*.jpg"))


def test_audit_blocks_exact_cross_split_duplicate(tmp_path: Path) -> None:
    dataset_root = generate_synthetic_detection_dataset(
        tmp_path / "dataset",
        source_count=9,
        image_size=64,
        seed=13,
    ).parent
    report = validate_yolo_dataset(dataset_root)
    source = next(item for item in report.records if item.split == "train" and item.class_ids)
    relative = source.image_path.relative_to(dataset_root / "images" / "train")
    image_destination = dataset_root / "images" / "val" / "duplicate-source" / relative.name
    label_destination = (
        dataset_root / "labels" / "val" / "duplicate-source" / relative.with_suffix(".txt").name
    )
    image_destination.parent.mkdir(parents=True)
    label_destination.parent.mkdir(parents=True, exist_ok=True)
    image_destination.write_bytes(source.image_path.read_bytes())
    label_destination.write_text(source.label_path.read_text(encoding="utf-8"), encoding="utf-8")

    result = audit_dataset(_config(dataset_root), tmp_path / "run")

    assert not result.valid
    assert not result.source_split_integrity
    assert any(issue.code == "exact_duplicate_cross_split" for issue in result.issues)


def test_unconfigured_performance_thresholds_never_pass(tmp_path: Path) -> None:
    config = _config(tmp_path / "dataset")
    audit = AuditResult(
        valid=True,
        dataset_root=str(tmp_path / "dataset"),
        dataset_fingerprint="dataset",
        image_count=3,
        positive_image_count=3,
        negative_image_count=0,
        class_counts={"firearm": 1, "knife": 1},
        split_counts={},
        imbalance_ratio=1.0,
        exact_duplicate_groups=(),
        perceptual_duplicate_groups=(),
        source_split_integrity=True,
        license_review_status="approved",
        issues=(),
        contact_sheets=(),
    )
    metrics = {
        "precision": 1.0,
        "recall": 1.0,
        "mAP50": 1.0,
        "per_class": {
            "firearm": {"average_precision_50": 1.0},
            "knife": {"average_precision_50": 1.0},
        },
    }

    gates = evaluate_pre_export_gates(config, audit, metrics)

    assert not gates.performance_thresholds_configured
    assert not gates.performance_gates_passed
    assert not gates.candidate_qualified


def test_still_image_metrics_cannot_pass_false_positive_rate_gate(tmp_path: Path) -> None:
    config = _config(tmp_path / "dataset")
    config = replace(
        config,
        quality_gates=replace(
            config.quality_gates,
            min_validation_map50=0.5,
            min_firearm_ap50=0.5,
            min_knife_ap50=0.5,
            min_validation_precision=0.5,
            min_validation_recall=0.5,
            max_false_positives_per_hour=1.0,
        ),
    )
    audit = AuditResult(
        valid=True,
        dataset_root=str(tmp_path / "dataset"),
        dataset_fingerprint="dataset",
        image_count=3,
        positive_image_count=3,
        negative_image_count=0,
        class_counts={"firearm": 1, "knife": 1},
        split_counts={},
        imbalance_ratio=1.0,
        exact_duplicate_groups=(),
        perceptual_duplicate_groups=(),
        source_split_integrity=True,
        license_review_status="approved",
        issues=(),
        contact_sheets=(),
    )
    metrics = {
        "precision": 1.0,
        "recall": 1.0,
        "mAP50": 1.0,
        "per_class": {
            "firearm": {"average_precision_50": 1.0},
            "knife": {"average_precision_50": 1.0},
        },
    }

    gates = evaluate_pre_export_gates(config, audit, metrics)

    false_positive_gate = next(
        check for check in gates.checks if check.name == "false_positives_per_hour"
    )
    assert false_positive_gate.status == "not_evaluated"
    assert not gates.performance_gates_passed


def test_run_state_is_atomic_and_cancellation_is_observable(tmp_path: Path) -> None:
    run_dir = run_directory(tmp_path, "run-1")
    store = StateStore(run_dir)
    store.create(_state("run-1"))

    assert store.load().run_id == "run-1"
    request_cancellation(run_dir)
    assert cancellation_reason(run_dir) == "operator_requested"
    with pytest.raises(ValueError, match="run_id must contain"):
        run_directory(tmp_path, "..")


def test_review_pause_and_report_are_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = generate_synthetic_detection_dataset(
        tmp_path / "dataset",
        source_count=9,
        image_size=64,
        seed=19,
    ).parent
    config = _config(dataset_root)
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    config = replace(
        config,
        experiment=replace(config.experiment, output_root=tmp_path / "runs"),
        tracking=replace(
            config.tracking,
            uri=tracking_uri,
            experiment_name=f"autotrain-test-{tmp_path.name}",
            dvc_enabled=False,
        ),
    )

    from surveillance_ml.autotrain.orchestrator import record_annotation_review, run_autotrain

    state = run_autotrain(config)
    assert state.status == "WAITING_FOR_REVIEW"
    assert state.current_stage == "annotation_review"
    assert (config.experiment.output_root / state.run_id / "autotrain-report.json").is_file()

    reviewed = record_annotation_review(
        state.run_id,
        config.experiment.output_root,
        decision="approved",
        reviewer="test reviewer",
        notes="Contact sheets reviewed.",
    )
    assert reviewed.status == "WAITING_FOR_REVIEW"
    assert reviewed.annotation_review is not None
    assert reviewed.annotation_review["decision"] == "approved"


def test_sample_errors_count_class_mismatch_as_false_positive_and_negative() -> None:
    sample = {
        "ground_truths": [{"class_id": 0, "class_name": "firearm", "box": [0, 0, 10, 10]}],
        "predictions": [
            {
                "class_id": 1,
                "class_name": "knife",
                "confidence": 0.9,
                "box": [0, 0, 10, 10],
            }
        ],
    }

    errors, matched_truths, matched_predictions = _sample_errors(sample, 0.25)

    assert matched_truths == set()
    assert matched_predictions == set()
    assert {error["kind"] for error in errors} == {"missed_firearm", "false_positive_knife"}


def test_report_writer_persists_json_and_markdown(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    markdown_path = write_autotrain_report(
        run_dir,
        state={
            "run_id": "run-1",
            "status": "WAITING_FOR_REVIEW",
            "current_stage": "annotation_review",
            "dataset_fingerprint": None,
        },
        audit={"valid": True, "image_count": 0, "class_counts": {}},
        environment=None,
        metrics=None,
        qualification=None,
    )

    assert markdown_path.is_file()
    assert "WAITING_FOR_REVIEW" in markdown_path.read_text(encoding="utf-8")
    assert (run_dir / "autotrain-report.json").is_file()


def test_training_milestones_resume_one_cumulative_optimizer_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_autotrain_config(CONFIG_PATH)
    config = replace(
        config,
        training=replace(
            config.training,
            smoke_epochs=1,
            pilot_epochs=2,
            max_total_epochs=3,
        ),
    )
    calls: list[tuple[int, int, Path | None]] = []

    def fake_detection_config(_config, _run_dir, epochs):
        return SimpleNamespace(training=SimpleNamespace(epochs=epochs))

    def fake_train(
        detection_config,
        *,
        resume_from,
        stop_after_epochs,
        **_kwargs,
    ):
        results_dir = tmp_path / "run" / "training"
        weights = results_dir / "weights"
        weights.mkdir(parents=True, exist_ok=True)
        history = results_dir / "results.csv"
        completed = 0
        if history.is_file():
            completed = len(history.read_text(encoding="utf-8").splitlines()) - 1
        mode = "a" if history.exists() else "w"
        with history.open(mode, encoding="utf-8") as file:
            if mode == "w":
                file.write("epoch,metric\n")
            for epoch in range(completed, stop_after_epochs):
                file.write(f"{epoch},1.0\n")
        best = weights / "best.pt"
        last = weights / "last.pt"
        best.write_bytes(b"best")
        last.write_bytes(b"last")
        calls.append((detection_config.training.epochs, stop_after_epochs, resume_from))
        return DetectionTrainingOutcome(
            checkpoint_path=best,
            results_dir=results_dir,
            mlflow_run_id=f"mlflow-{len(calls)}",
            epochs_completed=stop_after_epochs,
            training_seconds=1.0,
            dataset_fingerprint="dataset",
            stop_reason="stage_boundary",
        )

    monkeypatch.setattr(
        "surveillance_ml.autotrain.experiment_runner._detection_config",
        fake_detection_config,
    )
    monkeypatch.setattr("surveillance_ml.autotrain.experiment_runner.train_detector", fake_train)
    completed: list[str] = []

    outcome, stop_reason = run_training_stages(
        config,
        tmp_path / "run",
        completed_stages=set(),
        completed_stage_records=[],
        stop_check=lambda: None,
        on_stage_start=lambda _name: None,
        on_stage_complete=lambda name, _outcome, _epochs: completed.append(name),
    )

    assert calls == [
        (3, 1, None),
        (3, 2, tmp_path / "run" / "training" / "weights" / "last.pt"),
        (3, 3, tmp_path / "run" / "training" / "weights" / "last.pt"),
    ]
    assert completed == ["smoke", "pilot", "main"]
    assert outcome.epochs_completed == 3
    assert stop_reason is None
