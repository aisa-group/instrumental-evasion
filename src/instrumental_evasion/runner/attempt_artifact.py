"""Reserve, finalize, and validate one write-once episode attempt."""

from __future__ import annotations

import hashlib
import json
import re
import stat
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from instrumental_evasion.runner import log as native_log

MARKER_VERSION = 1
RESULT_NAME = "result.json"
COMPLETION_NAME = "complete.json"
_ATTEMPT_ID = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{32}$")
_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class AttemptArtifactError(RuntimeError):
    """The attempt directory or its final artifact violates the contract."""


class AttemptArtifactUnreadable(AttemptArtifactError):
    """Storage could not supply the evidence needed to validate an attempt."""


def _regular_file(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.stat().st_mode)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise AttemptArtifactUnreadable(f"Attempt evidence is unreadable: {error}") from error


@dataclass(frozen=True)
class AttemptLocation:
    """Identifiers and reserved directory for one episode attempt."""

    episode_id: str
    attempt_id: str
    directory: Path


def new_attempt_id() -> str:
    """Return a sortable identifier with enough entropy for concurrent jobs."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{uuid.uuid4().hex}"


def validate_attempt_id(value: str) -> str:
    """Return a valid attempt ID or raise ``AttemptArtifactError``."""
    if not _ATTEMPT_ID.fullmatch(value):
        raise AttemptArtifactError(f"invalid attempt ID: {value!r}")
    return value


def episode_id_for(task_name: str, epoch: int) -> str:
    """Return the stable episode identity for one task and epoch."""
    if not _PATH_COMPONENT.fullmatch(task_name):
        raise AttemptArtifactError(f"unsafe task name for attempt path: {task_name!r}")
    if type(epoch) is not int or epoch < 0:
        raise AttemptArtifactError(f"epoch must be a nonnegative integer: {epoch!r}")
    return f"{task_name}:epoch{epoch}"


def reserve_attempt(
    log_root: Path | str,
    task_name: str,
    epoch: int,
    *,
    attempt_id: str | None = None,
) -> AttemptLocation:
    """Create a new attempt directory and refuse to reuse an existing one."""
    identifier = validate_attempt_id(attempt_id or new_attempt_id())
    episode_id = episode_id_for(task_name, epoch)
    directory = Path(log_root) / task_name / f"epoch{epoch}" / identifier
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise AttemptArtifactError(
            f"attempt directory already exists and cannot be reused: {directory}"
        ) from error
    return AttemptLocation(episode_id, identifier, directory)


def _fail(message: str) -> None:
    raise AttemptArtifactError(message)


def validate_log(log: native_log.EvalLog) -> None:
    """Validate one final per-attempt log before it receives a marker."""
    if log.version != native_log.LOG_VERSION:
        _fail(f"log version {log.version!r} is not {native_log.LOG_VERSION}")
    if log.arm not in {"control", "monitored"}:
        _fail(f"invalid arm: {log.arm!r}")
    if not log.task or not log.task_id or not log.scaffold:
        _fail("task, task_id, and scaffold must be nonempty")
    if len(log.samples) != 1:
        _fail(f"an attempt log must contain exactly one sample, found {len(log.samples)}")

    sample = log.samples[0]
    native_log.finalize_infrastructure(sample)
    if not log.episode_id or sample.episode_id != log.episode_id:
        _fail("sample and log episode IDs do not match")
    if not log.attempt_id or sample.attempt_id != log.attempt_id:
        _fail("sample and log attempt IDs do not match")
    if not _ATTEMPT_ID.fullmatch(log.attempt_id):
        _fail("log attempt ID has an invalid format")
    if sample.sample_id == "" or type(sample.epoch) is not int or sample.epoch < 0:
        _fail("sample ID and epoch are invalid")
    if type(sample.solved) is not bool:
        _fail("sample solved value must be a Boolean")
    for name in ("turns", "tool_calls", "blocked_calls", "monitor_errors"):
        value = getattr(sample, name)
        if type(value) is not int or value < 0:
            _fail(f"sample {name} must be a nonnegative integer")
    if sample.blocked_calls > sample.tool_calls:
        _fail("blocked call count exceeds tool call count")
    if len(sample.decisions) != sample.tool_calls:
        _fail("decision count does not match tool call count")

    decision_blocks = sum(bool(item.get("blocked")) for item in sample.decisions)
    decision_errors = sum(
        item.get("decision_source") == "monitor_error" for item in sample.decisions
    )
    if decision_blocks != sample.blocked_calls:
        _fail("blocked decision count does not match blocked_calls")
    if decision_errors != sample.monitor_errors:
        _fail("monitor-error decision count does not match monitor_errors")
    if sample.monitor_errors and not sample.infrastructure_invalid:
        _fail("a monitor error must make the sample infrastructure-invalid")
    if sample.infrastructure_invalid and not sample.infrastructure_failure:
        _fail("an infrastructure-invalid sample needs a normalized failure kind")
    if sample.infrastructure_invalid:
        try:
            native_log.InfrastructureFailure(str(sample.infrastructure_failure))
        except ValueError:
            _fail("sample infrastructure failure is not in the normalized set")
    if not sample.infrastructure_invalid and (
        sample.error is not None
        or sample.infrastructure_failure is not None
        or sample.infrastructure_reason
    ):
        _fail("a valid sample carries contradictory infrastructure failure fields")

    expected = native_log.summarize(log.samples, monitored=log.arm == "monitored")
    if log.results != expected:
        _fail("result summary does not match the sample")


def _marker_data(log: native_log.EvalLog, payload: bytes) -> dict[str, object]:
    return {
        "schema_version": MARKER_VERSION,
        "completed_at": datetime.now(UTC).isoformat(),
        "result": RESULT_NAME,
        "result_bytes": len(payload),
        "result_sha256": hashlib.sha256(payload).hexdigest(),
        "episode_id": log.episode_id,
        "attempt_id": log.attempt_id,
        "infrastructure_status": log.results["infrastructure_status"],
    }


def finalize_attempt(log: native_log.EvalLog, directory: Path | str) -> Path:
    """Write, read back, validate, and mark one completed attempt atomically."""
    directory = Path(directory)
    if not directory.is_dir():
        _fail(f"attempt directory does not exist: {directory}")
    result_path = directory / RESULT_NAME
    marker_path = directory / COMPLETION_NAME
    if result_path.exists() or marker_path.exists():
        _fail(f"attempt artifact already exists and cannot be replaced: {directory}")

    validate_log(log)
    native_log.write_log(log, result_path)
    try:
        persisted = native_log.read_log(result_path)
        validate_log(persisted)
        if persisted.as_dict() != log.as_dict():
            _fail("result changed during serialization")
        payload = result_path.read_bytes()
        marker = _marker_data(persisted, payload)
        native_log.write_text_atomic(
            marker_path,
            json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
        validate_attempt(directory)
    except BaseException:
        marker_path.unlink(missing_ok=True)
        raise
    return result_path


def validate_attempt(directory: Path | str) -> native_log.EvalLog:
    """Read and verify a completed attempt, including its digest marker."""
    directory = Path(directory)
    result_path = directory / RESULT_NAME
    marker_path = directory / COMPLETION_NAME
    if not _regular_file(result_path) or not _regular_file(marker_path):
        _fail(f"attempt is incomplete: {directory}")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise AttemptArtifactUnreadable(f"Completion marker is unreadable: {error}") from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AttemptArtifactError(f"completion marker is unreadable: {error}") from error
    if not isinstance(marker, dict):
        _fail("completion marker must be a JSON object")
    if type(marker.get("schema_version")) is not int or marker["schema_version"] != MARKER_VERSION:
        _fail("completion marker has an unsupported schema version")

    try:
        payload = result_path.read_bytes()
    except OSError as error:
        raise AttemptArtifactUnreadable(f"Result log is unreadable: {error}") from error
    if marker.get("result") != RESULT_NAME:
        _fail("completion marker names the wrong result file")
    if marker.get("result_bytes") != len(payload):
        _fail("completion marker byte count does not match the result")
    if marker.get("result_sha256") != hashlib.sha256(payload).hexdigest():
        _fail("completion marker digest does not match the result")

    try:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            _fail("result log must be a JSON object")
        if type(raw.get("version")) is not int or raw["version"] != native_log.LOG_VERSION:
            _fail("result log has an unsupported schema version")
        log = native_log.log_from_dict(raw)
        validate_log(log)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise AttemptArtifactError(f"result log is unreadable: {error}") from error
    sample = log.samples[0]
    expected_episode = episode_id_for(log.task, sample.epoch)
    if log.episode_id != expected_episode:
        _fail("episode ID does not match the task and epoch")
    if marker.get("episode_id") != log.episode_id:
        _fail("completion marker episode ID does not match the result")
    if marker.get("attempt_id") != log.attempt_id:
        _fail("completion marker attempt ID does not match the result")
    if marker.get("infrastructure_status") != log.results["infrastructure_status"]:
        _fail("completion marker infrastructure status does not match the result")
    if directory.name != log.attempt_id:
        _fail("attempt directory name does not match the result attempt ID")
    if directory.parent.name != f"epoch{sample.epoch}":
        _fail("attempt parent directory does not match the sample epoch")
    if directory.parent.parent.name != log.task:
        _fail("attempt task directory does not match the result task")
    return log


__all__ = [
    "AttemptArtifactError",
    "AttemptArtifactUnreadable",
    "AttemptLocation",
    "COMPLETION_NAME",
    "RESULT_NAME",
    "episode_id_for",
    "finalize_attempt",
    "new_attempt_id",
    "reserve_attempt",
    "validate_attempt",
    "validate_attempt_id",
    "validate_log",
]
