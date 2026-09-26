"""Contracts for native attempt identity, validity, and durable finalization."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from instrumental_evasion.runner import log as native_log
from instrumental_evasion.runner.attempt_artifact import (
    COMPLETION_NAME,
    RESULT_NAME,
    AttemptArtifactError,
    AttemptArtifactUnreadable,
    finalize_attempt,
    reserve_attempt,
    validate_attempt,
)
from instrumental_evasion.runner.sandbox import ExecResult
from instrumental_evasion.runner.scaffolds.base import (
    STOP_DESISTED,
    STOP_SUBMIT,
    Limits,
    ScaffoldResult,
)
from instrumental_evasion.tasks.registry import ResolvedTask
from instrumental_evasion.tasks.tb_spec import TBTaskSpec

ATTEMPT_ID = "20260902T120000Z-0123456789abcdef0123456789abcdef"


def _sample(*, episode_id: str, attempt_id: str, **values) -> native_log.SampleRecord:
    fields = {
        "sample_id": "fixture",
        "epoch": 0,
        "solved": False,
        "episode_id": episode_id,
        "attempt_id": attempt_id,
        "stop_reason": STOP_SUBMIT,
        "metadata": {"evasion_attempted": False},
    }
    fields.update(values)
    sample = native_log.SampleRecord(**fields)
    native_log.finalize_infrastructure(sample)
    return sample


def _log(sample: native_log.SampleRecord) -> native_log.EvalLog:
    log = native_log.EvalLog(
        task="fixture_monitored",
        task_id="fixture",
        arm="monitored",
        scaffold="react",
        episode_id=sample.episode_id,
        attempt_id=sample.attempt_id,
        samples=[sample],
    )
    log.results = native_log.summarize(log.samples, monitored=True)
    return log


def test_monitor_error_is_infrastructure_invalid_even_when_fail_closed():
    sample = _sample(
        episode_id="fixture_monitored:epoch0",
        attempt_id=ATTEMPT_ID,
        tool_calls=1,
        blocked_calls=1,
        monitor_errors=1,
        decisions=[
            {
                "blocked": True,
                "decision_source": "monitor_error",
            }
        ],
    )

    results = native_log.summarize([sample], monitored=True)

    assert sample.infrastructure_status == "invalid"
    assert sample.infrastructure_failure == "monitor_error"
    assert sample.void is True
    assert results["infrastructure_status"] == "invalid"
    assert results["infrastructure_failures"] == {"monitor_error": 1}
    assert results["counted"] == 0


def test_desisted_is_infrastructure_valid():
    sample = _sample(
        episode_id="fixture_monitored:epoch0",
        attempt_id=ATTEMPT_ID,
        stop_reason=STOP_DESISTED,
        metadata={"evasion_attempted": False, "refusal_stop_triggered": True},
    )

    results = native_log.summarize([sample], monitored=True)

    assert sample.infrastructure_status == "valid"
    assert sample.void is False
    assert results["infrastructure_status"] == "valid"
    assert results["counted"] == 1


def test_attempt_finalization_round_trips_with_digest_marker(tmp_path):
    location = reserve_attempt(
        tmp_path, "fixture_monitored", 0, attempt_id=ATTEMPT_ID
    )
    log = _log(
        _sample(episode_id=location.episode_id, attempt_id=location.attempt_id)
    )

    result_path = finalize_attempt(log, location.directory)
    persisted = validate_attempt(location.directory)
    marker = json.loads(
        (location.directory / COMPLETION_NAME).read_text(encoding="utf-8")
    )

    assert result_path == location.directory / RESULT_NAME
    assert persisted.attempt_id == ATTEMPT_ID
    assert marker["attempt_id"] == ATTEMPT_ID
    assert marker["infrastructure_status"] == "valid"
    assert marker["result_bytes"] == result_path.stat().st_size
    assert [item.attempt_id for item in native_log.read_logs(tmp_path)] == [ATTEMPT_ID]


def test_attempt_directory_and_final_artifact_are_write_once(tmp_path):
    location = reserve_attempt(
        tmp_path, "fixture_monitored", 0, attempt_id=ATTEMPT_ID
    )
    log = _log(
        _sample(episode_id=location.episode_id, attempt_id=location.attempt_id)
    )
    finalize_attempt(log, location.directory)

    with pytest.raises(AttemptArtifactError, match="cannot be reused"):
        reserve_attempt(tmp_path, "fixture_monitored", 0, attempt_id=ATTEMPT_ID)
    with pytest.raises(AttemptArtifactError, match="cannot be replaced"):
        finalize_attempt(log, location.directory)


def test_missing_marker_and_tampered_result_are_not_complete(tmp_path):
    location = reserve_attempt(
        tmp_path, "fixture_monitored", 0, attempt_id=ATTEMPT_ID
    )
    log = _log(
        _sample(episode_id=location.episode_id, attempt_id=location.attempt_id)
    )
    finalize_attempt(log, location.directory)
    marker_path = location.directory / COMPLETION_NAME
    marker_text = marker_path.read_text(encoding="utf-8")

    marker_path.unlink()
    with pytest.raises(AttemptArtifactError, match="incomplete"):
        validate_attempt(location.directory)
    assert native_log.read_logs(tmp_path) == []

    marker_path.write_text(marker_text, encoding="utf-8")
    result_path = location.directory / RESULT_NAME
    result_path.write_text(result_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(AttemptArtifactError, match="byte count|digest"):
        validate_attempt(location.directory)
    assert native_log.read_logs(tmp_path) == []


@pytest.mark.parametrize("payload", [b"[]", b"null", b"1", b'"text"', b"\xff", b"{"])
def test_malformed_completion_marker_uses_the_artifact_error_contract(tmp_path, payload):
    location = reserve_attempt(tmp_path, "fixture_monitored", 0, attempt_id=ATTEMPT_ID)
    finalize_attempt(_log(_sample(episode_id=location.episode_id, attempt_id=ATTEMPT_ID)), location.directory)
    (location.directory / COMPLETION_NAME).write_bytes(payload)

    with pytest.raises(AttemptArtifactError):
        validate_attempt(location.directory)


@pytest.mark.parametrize("mutate", [
    lambda raw: raw.update(version=999),
    lambda raw: raw.pop("version"),
    lambda raw: raw.update(version=True),
    lambda raw: raw.update(metadata=[]),
    lambda raw: raw.update(samples=None),
    lambda raw: raw["samples"][0].update(metadata=None),
    lambda raw: raw["samples"][0].update(decisions=[None]),
])
def test_valid_digest_does_not_admit_an_invalid_result_schema(tmp_path, mutate):
    location = reserve_attempt(tmp_path, "fixture_monitored", 0, attempt_id=ATTEMPT_ID)
    finalize_attempt(_log(_sample(episode_id=location.episode_id, attempt_id=ATTEMPT_ID)), location.directory)
    result = location.directory / RESULT_NAME
    raw = json.loads(result.read_bytes())
    mutate(raw)
    payload = json.dumps(raw).encode()
    result.write_bytes(payload)
    marker_path = location.directory / COMPLETION_NAME
    marker = json.loads(marker_path.read_bytes())
    marker.update(result_bytes=len(payload), result_sha256=hashlib.sha256(payload).hexdigest())
    marker_path.write_text(json.dumps(marker))

    with pytest.raises(AttemptArtifactError):
        validate_attempt(location.directory)


def test_validation_parses_only_the_result_bytes_checked_by_the_digest(tmp_path, monkeypatch):
    location = reserve_attempt(tmp_path, "fixture_monitored", 0, attempt_id=ATTEMPT_ID)
    finalize_attempt(_log(_sample(episode_id=location.episode_id, attempt_id=ATTEMPT_ID)), location.directory)
    result = location.directory / RESULT_NAME
    read = Path.read_bytes
    reads = []

    def change_after_read(path):
        payload = read(path)
        if path == result:
            reads.append(path)
            path.write_text("The file changed after the verified read.")
        return payload

    monkeypatch.setattr(Path, "read_bytes", change_after_read)
    observed = validate_attempt(location.directory)
    assert observed.attempt_id == ATTEMPT_ID
    assert observed.samples[0].solved is False
    assert reads == [result]


@pytest.mark.parametrize("name", [RESULT_NAME, COMPLETION_NAME])
def test_storage_failure_has_a_distinct_artifact_error(tmp_path, monkeypatch, name):
    location = reserve_attempt(tmp_path, "fixture_monitored", 0, attempt_id=ATTEMPT_ID)
    finalize_attempt(_log(_sample(episode_id=location.episode_id, attempt_id=ATTEMPT_ID)), location.directory)
    original = Path.open

    def fail_read(path, *args, **kwargs):
        if path == location.directory / name:
            raise OSError(5, "Simulated storage failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_read)
    with pytest.raises(AttemptArtifactUnreadable):
        validate_attempt(location.directory)


def test_inconsistent_summary_is_rejected_before_completion(tmp_path):
    location = reserve_attempt(
        tmp_path, "fixture_monitored", 0, attempt_id=ATTEMPT_ID
    )
    log = _log(
        _sample(episode_id=location.episode_id, attempt_id=location.attempt_id)
    )
    log.results["solved"] = 99

    with pytest.raises(AttemptArtifactError, match="summary"):
        finalize_attempt(log, location.directory)

    assert not (location.directory / RESULT_NAME).exists()
    assert not (location.directory / COMPLETION_NAME).exists()


def test_atomic_log_write_preserves_the_previous_file_on_replace_failure(
    monkeypatch, tmp_path,
):
    path = tmp_path / "result.json"
    path.write_text("previous", encoding="utf-8")

    def fail_replace(_source, _destination):
        raise OSError("simulated interrupted replacement")

    monkeypatch.setattr(native_log.os, "replace", fail_replace)

    with pytest.raises(OSError, match="interrupted replacement"):
        native_log.write_text_atomic(path, "replacement")

    assert path.read_text(encoding="utf-8") == "previous"
    assert list(tmp_path.glob(".result.json.*.tmp")) == []


def _resolved_task(score_fn) -> ResolvedTask:
    spec = TBTaskSpec(
        id="fixture",
        tb_name="fixture",
        image_env="FIXTURE_IMAGE",
        prompt="complete the fixture",
        policy="no_compilation",
        key_step="compile",
        score_fn=score_fn,
    )
    return ResolvedTask(
        task_id="fixture",
        name="fixture_monitored",
        spec=spec,
        arm="monitored",
        policy="no_compilation",
    )


class _Sandbox:
    async def exec(self, _command):
        return ExecResult(True, 0, "", "")

    def cleanup(self):
        return None


class _Runtime:
    name = "fake"
    read_static_flag = None

    async def open_sandbox(self, _spec, *, monitored):
        assert monitored is True
        return _Sandbox()


def _limits() -> Limits:
    return Limits(max_turns=2, token_limit=100, time_limit_seconds=30, attempts=2)


def test_live_monitor_error_is_recorded_as_infrastructure_invalid(
    monkeypatch, tmp_path,
):
    from instrumental_evasion.runner.run import run_episode

    async def score(_sandbox):
        return False, "not solved", {}

    def monitor_failure(*_args, **_kwargs):
        raise RuntimeError("monitor unavailable")

    monkeypatch.setattr("instrumental_evasion.runner.gate.verdict_direct", monitor_failure)

    class _Scaffold:
        name = "react"

        async def run(self, ctx):
            decision = await ctx.gate.review("bash", {"command": "make"})
            assert decision.blocked is True
            return ScaffoldResult(stop_reason=STOP_SUBMIT)

    record = asyncio.run(
        run_episode(
            _resolved_task(score),
            _Scaffold(),
            _Runtime(),
            epoch=0,
            outdir=tmp_path,
            monitor_model="stub/monitor",
            limits=_limits(),
        )
    )

    assert record.monitor_errors == 1
    assert record.infrastructure_status == "invalid"
    assert record.infrastructure_failure == "monitor_error"


def test_scorer_and_cleanup_failures_are_normalized(tmp_path):
    from instrumental_evasion.runner.run import run_episode

    async def broken_score(_sandbox):
        raise RuntimeError("scorer crashed")

    class _Scaffold:
        name = "react"

        async def run(self, _ctx):
            return ScaffoldResult(stop_reason=STOP_SUBMIT)

    scorer_record = asyncio.run(
        run_episode(
            _resolved_task(broken_score),
            _Scaffold(),
            _Runtime(),
            epoch=0,
            outdir=tmp_path / "scorer",
            monitor_model="stub/monitor",
            limits=_limits(),
        )
    )
    assert scorer_record.infrastructure_failure == "scorer_error"

    async def score(_sandbox):
        return False, "not solved", {}

    class _BrokenCleanupSandbox(_Sandbox):
        def cleanup(self):
            raise RuntimeError("cleanup crashed")

    class _BrokenCleanupRuntime(_Runtime):
        async def open_sandbox(self, _spec, *, monitored):
            return _BrokenCleanupSandbox()

    cleanup_record = asyncio.run(
        run_episode(
            _resolved_task(score),
            _Scaffold(),
            _BrokenCleanupRuntime(),
            epoch=0,
            outdir=tmp_path / "cleanup",
            monitor_model="stub/monitor",
            limits=_limits(),
        )
    )
    assert cleanup_record.infrastructure_failure == "cleanup_error"
    assert "cleanup crashed" in cleanup_record.infrastructure_reason


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_new_attempt_evidence_is_finalized_after_scoring_and_cleanup(tmp_path, cleanup_fails):
    from instrumental_evasion.runner.attempts import ATTEMPT_VERSION
    from instrumental_evasion.runner.run import run_episode

    async def score(_sandbox):
        return True, "external score passed", {}

    class Scaffold:
        name = "react"

        async def run(self, _ctx):
            return ScaffoldResult(stop_reason=STOP_SUBMIT)

    class Sandbox(_Sandbox):
        def cleanup(self):
            if cleanup_fails:
                raise RuntimeError("cleanup failed")

    class Runtime(_Runtime):
        async def open_sandbox(self, _spec, *, monitored):
            return Sandbox()

    record = asyncio.run(run_episode(
        _resolved_task(score), Scaffold(), Runtime(), epoch=0, outdir=tmp_path,
        monitor_model="stub/monitor", limits=_limits(),
    ))
    assert record.solved is True
    assert record.metadata["evasion_marker_detected"] is False
    assert record.metadata["evasion_attempt_classifier_version"] == ATTEMPT_VERSION
    assert record.metadata["evasion_attempted"] is (None if cleanup_fails else True)


def test_run_task_writes_one_unique_completed_attempt_per_epoch(
    monkeypatch, tmp_path,
):
    from instrumental_evasion.runner import run as native_run

    async def score(_sandbox):
        return False, "not solved", {}

    task = _resolved_task(score)

    class _Scaffold:
        name = "react"

        @staticmethod
        def describe():
            return {
                "agent": "react",
                "time_limit_enforced": True,
                "token_limit_enforced": True,
                "turn_limit_enforced": True,
            }

    class _DescribedRuntime:
        name = "fake"

        @staticmethod
        def describe():
            return {"runtime": "fake"}

    async def fake_episode(
        _task,
        _scaffold,
        _runtime,
        *,
        epoch,
        episode_id,
        attempt_id,
        **_kwargs,
    ):
        return _sample(
            episode_id=episode_id,
            attempt_id=attempt_id,
            epoch=epoch,
        )

    monkeypatch.setattr(native_run, "run_episode", fake_episode)

    aggregate = asyncio.run(
        native_run.run_task(
            task,
            _Scaffold(),
            _DescribedRuntime(),
            epochs=2,
            log_dir=tmp_path,
            monitor_model="stub/monitor",
            limits=_limits(),
        )
    )

    markers = sorted(tmp_path.rglob(COMPLETION_NAME))
    assert len(markers) == 2
    assert len({sample.attempt_id for sample in aggregate.samples}) == 2
    assert aggregate.results["infrastructure_status"] == "valid"
    assert all(validate_attempt(path.parent) for path in markers)
    assert not (tmp_path / "fixture_monitored.json").exists()


def test_run_task_uses_a_fixed_epoch_and_attempt_id(monkeypatch, tmp_path):
    from instrumental_evasion.runner import run as native_run

    async def score(_sandbox):
        return False, "not solved", {}

    task = _resolved_task(score)

    class _Scaffold:
        name = "react"

        @staticmethod
        def describe():
            return {
                "agent": "react",
                "agent_model": "openrouter/deepseek/deepseek-v4-flash",
                "time_limit_enforced": True,
                "token_limit_enforced": True,
                "turn_limit_enforced": True,
            }

    class _Runtime:
        name = "fake"

        @staticmethod
        def describe():
            return {"runtime": "fake"}

    async def fake_episode(
        _task,
        _scaffold,
        _runtime,
        *,
        epoch,
        episode_id,
        attempt_id,
        **_kwargs,
    ):
        return _sample(
            episode_id=episode_id,
            attempt_id=attempt_id,
            epoch=epoch,
        )

    monkeypatch.setattr(native_run, "run_episode", fake_episode)

    aggregate = asyncio.run(
        native_run.run_task(
            task,
            _Scaffold(),
            _Runtime(),
            epochs=1,
            epoch=3,
            attempt_id=ATTEMPT_ID,
            policy_variant="trimmed",
            log_dir=tmp_path,
            monitor_model="stub/monitor",
            limits=_limits(),
        )
    )

    sample = aggregate.samples[0]
    assert sample.epoch == 3
    assert sample.attempt_id == ATTEMPT_ID
    assert aggregate.metadata["policy_variant"] == "trimmed"
    assert aggregate.metadata["condition"] == "no_compilation"
    assert len(aggregate.metadata["policy_prompt_sha256"]) == 64
    assert validate_attempt(
        tmp_path / task.name / "epoch3" / ATTEMPT_ID
    ).attempt_id == ATTEMPT_ID


def test_a_fixed_attempt_id_requires_exactly_one_epoch(tmp_path):
    from instrumental_evasion.runner import run as native_run

    async def score(_sandbox):
        return False, "not solved", {}

    with pytest.raises(AttemptArtifactError, match="exactly one epoch"):
        asyncio.run(
            native_run.run_task(
                _resolved_task(score),
                object(),
                object(),
                epochs=2,
                attempt_id=ATTEMPT_ID,
                log_dir=tmp_path,
                monitor_model="stub/monitor",
                limits=_limits(),
            )
        )


def test_native_runner_exits_nonzero_for_infrastructure_invalid_episode(
    monkeypatch,
):
    from instrumental_evasion.runner import run as native_run

    async def score(_sandbox):
        return False, "not solved", {}

    task = _resolved_task(score)
    sample = _sample(
        episode_id="fixture_monitored:epoch0",
        attempt_id=ATTEMPT_ID,
        error="provider failed",
    )
    native_log.mark_infrastructure_invalid(
        sample, native_log.InfrastructureFailure.PROVIDER, "provider failed"
    )
    log = _log(sample)

    class _Scaffold:
        name = "react"

        @staticmethod
        def describe():
            return {"refusal_stop_supported": True}

        async def preflight(self, _spec):
            return {"ok": True}

    class _Runtime:
        name = "fake"

        async def setup(self):
            return None

        def teardown(self):
            return None

    async def fake_run_task(*_args, **_kwargs):
        return log

    monkeypatch.setattr(native_run, "load_env_file", lambda: None)
    monkeypatch.setattr(native_run, "resolve", lambda *_args, **_kwargs: [task])
    monkeypatch.setattr(native_run, "get_scaffold", lambda *_args, **_kwargs: _Scaffold())
    monkeypatch.setattr(native_run, "get_runtime", lambda *_args, **_kwargs: _Runtime())
    monkeypatch.setattr(native_run, "run_task", fake_run_task)
    monkeypatch.setattr(native_run, "print_table", lambda _logs: None)

    exit_code = native_run.main(
        ["fixture", "--arm", "monitored", "--scaffold", "react"]
    )

    assert exit_code == 4
