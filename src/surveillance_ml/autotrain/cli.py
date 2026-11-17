"""Command-line entry point for AutoTrain."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from surveillance_ml.autotrain.config import load_autotrain_config
from surveillance_ml.autotrain.orchestrator import (
    audit_autotrain,
    cancel_autotrain,
    dry_run_plan,
    load_run_report,
    load_run_state,
    record_annotation_review,
    resume_autotrain,
    run_autotrain,
)

EXIT_SUCCESS = 0
EXIT_CONFIGURATION = 2
EXIT_DATASET_BLOCKED = 3
EXIT_TRAINING_FAILED = 4
EXIT_NOT_QUALIFIED = 5
EXIT_WAITING_FOR_REVIEW = 6
EXIT_CANCELLED = 7


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and train YOLO detection candidates safely."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser(
        "audit", help="Audit a local YOLO dataset and create review sheets."
    )
    audit.add_argument("--config", type=Path, default=Path("configs/autotrain.yaml"))

    run = commands.add_parser("run", help="Run the staged AutoTrain workflow.")
    run.add_argument("--config", type=Path, default=Path("configs/autotrain.yaml"))
    run.add_argument("--dry-run", action="store_true", help="Validate and display stages only.")

    status = commands.add_parser("status", help="Show durable status for a run.")
    status.add_argument("--run-id", required=True)
    status.add_argument("--config", type=Path, default=Path("configs/autotrain.yaml"))

    resume = commands.add_parser("resume", help="Safely resume a paused or interrupted run.")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--config", type=Path)

    review = commands.add_parser("review", help="Record a human annotation-review decision.")
    review.add_argument("--run-id", required=True)
    review.add_argument("--decision", choices=("approved", "rejected"), required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--notes", default="")
    review.add_argument("--config", type=Path, default=Path("configs/autotrain.yaml"))

    cancel = commands.add_parser("cancel", help="Request cancellation after the current epoch.")
    cancel.add_argument("--run-id", required=True)
    cancel.add_argument("--config", type=Path, default=Path("configs/autotrain.yaml"))

    report = commands.add_parser("report", help="Print the saved JSON engineering report.")
    report.add_argument("--run-id", required=True)
    report.add_argument("--config", type=Path, default=Path("configs/autotrain.yaml"))
    return parser


def _state_exit_code(state: Any) -> int:
    if state.status == "WAITING_FOR_REVIEW":
        return EXIT_WAITING_FOR_REVIEW
    if state.status == "CANCELLED":
        return EXIT_CANCELLED
    if state.status == "FAILED":
        return (
            EXIT_DATASET_BLOCKED if state.failed_stage == "dataset_audit" else EXIT_TRAINING_FAILED
        )
    if state.status == "COMPLETED" and state.qualification_status not in {
        "manual_approval_required",
        "deployment_blocked",
    }:
        return EXIT_NOT_QUALIFIED
    return EXIT_SUCCESS


def _state_payload(state: Any) -> dict[str, Any]:
    return {
        "run_id": state.run_id,
        "status": state.status,
        "current_stage": state.current_stage,
        "completed_stages": state.completed_stages,
        "qualification_status": state.qualification_status,
        "stop_reason": state.stop_reason,
        "dataset_fingerprint": state.dataset_fingerprint,
        "mlflow_run_ids": state.mlflow_run_ids,
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parser().parse_args(argv)
    try:
        if args.command == "resume":
            config = load_autotrain_config(args.config) if args.config is not None else None
            if config is None:
                default = load_autotrain_config(Path("configs/autotrain.yaml"))
                output_root = default.experiment.output_root
            else:
                output_root = config.experiment.output_root
            state = resume_autotrain(args.run_id, output_root, config)
            print(json.dumps(_state_payload(state), indent=2))
            return _state_exit_code(state)

        config = load_autotrain_config(args.config)
        if args.command == "audit":
            state = audit_autotrain(config)
            print(json.dumps(_state_payload(state), indent=2))
            return EXIT_SUCCESS if state.status == "COMPLETED" else EXIT_DATASET_BLOCKED
        if args.command == "run":
            if args.dry_run:
                print(json.dumps(dry_run_plan(config), indent=2))
                return EXIT_SUCCESS
            state = run_autotrain(config)
            print(json.dumps(_state_payload(state), indent=2))
            return _state_exit_code(state)
        if args.command == "status":
            state = load_run_state(args.run_id, config.experiment.output_root)
            print(json.dumps(_state_payload(state), indent=2))
            return EXIT_SUCCESS
        if args.command == "review":
            state = record_annotation_review(
                args.run_id,
                config.experiment.output_root,
                decision=args.decision,
                reviewer=args.reviewer,
                notes=args.notes,
            )
            print(json.dumps(_state_payload(state), indent=2))
            return EXIT_SUCCESS if state.status != "CANCELLED" else EXIT_CANCELLED
        if args.command == "cancel":
            cancel_autotrain(args.run_id, config.experiment.output_root)
            print(json.dumps({"run_id": args.run_id, "cancellation_requested": True}, indent=2))
            return EXIT_SUCCESS
        if args.command == "report":
            print(json.dumps(load_run_report(args.run_id, config.experiment.output_root), indent=2))
            return EXIT_SUCCESS
    except (OSError, ValueError, RuntimeError) as error:
        print(f"AutoTrain {args.command} failed: {error}", file=sys.stderr)
        if args.command in {"audit", "run", "resume"} and "dataset" in str(error).lower():
            return EXIT_DATASET_BLOCKED
        return EXIT_CONFIGURATION
    return EXIT_CONFIGURATION
