"""Atomic run state, resumability metadata, cancellation, and process locking."""

import json
import os
import re
import tempfile
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED"}
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class AutoTrainState:
    run_id: str
    status: str
    current_stage: str
    completed_stages: list[str]
    started_at: str
    updated_at: str
    ended_at: str | None
    config_fingerprint: str
    dataset_fingerprint: str | None
    failed_stage: str | None = None
    stop_reason: str | None = None
    checkpoint_paths: dict[str, str] = field(default_factory=dict)
    mlflow_run_ids: list[str] = field(default_factory=list)
    annotation_review: dict[str, Any] | None = None
    audit_passed: bool | None = None
    qualification_status: str = "not_evaluated"
    execution: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> "AutoTrainState":
        if not isinstance(raw, dict):
            raise ValueError("AutoTrain state must be a JSON object")
        try:
            state = cls(**raw)
        except TypeError as error:
            raise ValueError(f"Invalid AutoTrain state fields: {error}") from error
        if not _RUN_ID_PATTERN.fullmatch(state.run_id):
            raise ValueError("AutoTrain state contains an invalid run id")
        if state.status not in {
            "CREATED",
            "AUDITING",
            "WAITING_FOR_REVIEW",
            "SMOKE_TRAINING",
            "PILOT_TRAINING",
            "MAIN_TRAINING",
            "EVALUATING",
            "EXPORTING",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
        }:
            raise ValueError(f"Unknown AutoTrain state status: {state.status}")
        return state


class StateStore:
    """Persist a single run's state with atomic replacement."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "state.json"

    def create(self, state: AutoTrainState) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"AutoTrain state already exists: {self.path}")
        self.save(state)

    def load(self) -> AutoTrainState:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Unable to read AutoTrain state: {self.path}") from error
        return AutoTrainState.from_dict(raw)

    def save(self, state: AutoTrainState) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".state-",
            suffix=".tmp",
            dir=self.run_dir,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
                json.dump(state.to_dict(), file, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def set_status(
        self,
        state: AutoTrainState,
        status: str,
        *,
        stage: str | None = None,
        reason: str | None = None,
    ) -> AutoTrainState:
        if state.status in TERMINAL_STATUSES and status not in {
            "AUDITING",
            "SMOKE_TRAINING",
            "PILOT_TRAINING",
            "MAIN_TRAINING",
            "EVALUATING",
            "EXPORTING",
            "CANCELLED",
        }:
            raise ValueError(f"Cannot transition from terminal state {state.status} to {status}")
        state.status = status
        state.current_stage = stage or state.current_stage
        state.stop_reason = reason
        state.updated_at = utc_now()
        state.ended_at = utc_now() if status in TERMINAL_STATUSES else None
        self.save(state)
        return state


class RunLock(AbstractContextManager["RunLock"]):
    """Hold an OS process lock that is released automatically on process exit."""

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / ".run.lock"
        self._file: Any = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self._file.write(b"\0")
            self._file.flush()
        self._file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl = __import__("fcntl")

                fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as error:
            self._file.close()
            self._file = None
            raise RuntimeError(
                f"AutoTrain run is already locked: {self.path.parent.name}"
            ) from error
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._file is None:
            return
        if os.name == "nt":
            import msvcrt

            self._file.seek(0)
            msvcrt.locking(self._file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl = __import__("fcntl")

            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()
        self._file = None


def run_directory(output_root: Path, run_id: str) -> Path:
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must contain only letters, numbers, '.', '_' or '-'")
    root = output_root.resolve()
    result = (root / run_id).resolve()
    if result.parent != root:
        raise ValueError("run_id must identify a direct child of the configured output root")
    return result


def request_cancellation(run_dir: Path, reason: str = "operator_requested") -> None:
    """Create an atomic cancellation marker without waiting for the run lock."""
    destination = run_dir / "cancel.json"
    temporary = run_dir / ".cancel.tmp"
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps({"requested_at": utc_now(), "reason": reason}, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def cancellation_reason(run_dir: Path) -> str | None:
    marker = run_dir / "cancel.json"
    if not marker.is_file():
        return None
    try:
        raw = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid cancellation request: {marker}") from error
    reason = raw.get("reason") if isinstance(raw, dict) else None
    return reason if isinstance(reason, str) else "operator_requested"


def clear_cancellation(run_dir: Path) -> None:
    (run_dir / "cancel.json").unlink(missing_ok=True)
