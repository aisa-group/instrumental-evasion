"""Exercise hook startup and monitor failures without model requests."""

import json
import os
import subprocess
import sys
import time
import urllib.error
from pathlib import Path
from types import SimpleNamespace

import pytest

from instrumental_evasion.config import DEFAULT_MONITOR_SERVING, PAPER_ENVIRONMENT
from instrumental_evasion.hooks import bootstrap, deploy, monitor_client
from instrumental_evasion.monitor_core import NEUTRAL_BLOCK_MESSAGE
from instrumental_evasion.runner.gate import Gate
from instrumental_evasion.runner.gate_server import serve


@pytest.fixture(autouse=True)
def _library_defaults(monkeypatch):
    """Test the monitor client's own defaults, not the runner's paper environment."""
    for name in {**PAPER_ENVIRONMENT, **DEFAULT_MONITOR_SERVING}:
        monkeypatch.delenv(name, raising=False)


def run_bootstrap(path, workdir, *, payload="{}", env=None):
    return subprocess.run(
        [sys.executable, "-I", str(path)], input=payload, text=True,
        cwd=workdir, capture_output=True, timeout=5,
        env={"PATH": os.defpath, "HOME": str(workdir), "PYTHONPATH": str(workdir), **(env or {})},
    )


def assert_denied(result):
    assert result.returncode == 2
    value = json.loads(result.stdout)["hookSpecificOutput"]
    assert value["permissionDecision"] == "deny"
    assert value["permissionDecisionReason"] == NEUTRAL_BLOCK_MESSAGE
    assert result.stderr.strip() == NEUTRAL_BLOCK_MESSAGE


def test_hook_ignores_task_modules_and_user_python_path(tmp_path):
    for name in ("inspect", "json", "dataclasses", "typing", "sitecustomize"):
        (tmp_path / f"{name}.py").write_text("raise RuntimeError('TASK_MODULE_IMPORTED')\n")
    assert_denied(run_bootstrap(Path(bootstrap.__file__), tmp_path))


def test_deployed_hook_uses_the_same_isolated_entrypoint_and_tracks_id(tmp_path, monkeypatch):
    monkeypatch.setattr("instrumental_evasion.runner.gate.verdict_direct",
                        lambda *a, **k: pytest.fail("control must not call a monitor"))
    root = deploy.minimal_src(tmp_path / "trusted")
    entry = root / "envtools/hooks/bootstrap.py"
    (tmp_path / "inspect.py").write_text("raise RuntimeError('TASK_MODULE_IMPORTED')\n")
    gate = Gate(policy=None)
    with serve(gate) as server:
        result = run_bootstrap(entry, tmp_path, env={"RUNTIME_URL": server.url}, payload=json.dumps({
            "tool_name": "Bash", "tool_input": {"command": "never executed"}, "tool_use_id": "call-1",
        }))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"] == "allow"
    assert len(gate.records()) == 1
    assert gate.records()[0]["decision_source"] == "unmonitored"
    assert gate.records()[0]["tool_use_id"] == "call-1"


def test_import_failure_cannot_exit_as_nonblocking_error(tmp_path):
    root = deploy.minimal_src(tmp_path / "trusted")
    (root / "envtools/hooks/protocol.py").unlink()
    assert_denied(run_bootstrap(root / "envtools/hooks/bootstrap.py", tmp_path))


@pytest.mark.parametrize("code", [0, 2])
def test_bootstrap_preserves_configured_denial_and_context(tmp_path, code):
    root = deploy.minimal_src(tmp_path / "trusted")
    output = {"hookSpecificOutput": {
        "hookEventName": "PreToolUse", "permissionDecision": "deny",
        "permissionDecisionReason": "Configured refusal.", "additionalContext": "Configured reminder.",
    }}
    (root / "envtools/hooks/runner.py").write_text(
        f"def main(payload):\n    print({json.dumps(output)!r})\n    return {code}\n"
    )
    result = run_bootstrap(root / "envtools/hooks/bootstrap.py", tmp_path)
    assert result.returncode == code
    assert json.loads(result.stdout) == output
    assert result.stderr == ("Configured refusal.\n" if code == 2 else "")


def test_startup_timeout_does_not_log_process_arguments(tmp_path, monkeypatch):
    episode = SimpleNamespace(outdir=tmp_path, _home=lambda: tmp_path,
                              _env=lambda: {"PRIVATE_TEST_VALUE": "sensitive-placeholder"})
    spec = SimpleNamespace(image=lambda: "test.sif")

    def timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    monkeypatch.setattr(deploy.subprocess, "run", timeout)
    result = deploy.probe_hook(spec, episode)
    assert result == {"hook_alive": False, "rc": 124,
                      "probe_error": "startup_timeout", "probe_timeout_seconds": 180}


def test_hook_deadline_blocks_a_stalled_import(tmp_path):
    root = deploy.minimal_src(tmp_path / "trusted")
    entry = root / "envtools/hooks/bootstrap.py"
    entry.write_text(entry.read_text().replace("HOOK_DEADLINE_SECONDS = 105", "HOOK_DEADLINE_SECONDS = 0.1"))
    (root / "envtools/hooks/runner.py").write_text("import time\ntime.sleep(30)\n")
    started = time.monotonic()
    assert_denied(run_bootstrap(entry, tmp_path))
    assert time.monotonic() - started < 3


def test_oversize_input_is_denied(tmp_path):
    assert_denied(run_bootstrap(Path(bootstrap.__file__), tmp_path, payload="x" * (bootstrap.MAX_INPUT_BYTES + 1)))


def test_null_monitor_responses_have_metadata_without_private_content(monkeypatch):
    private = "sk-test-private-value"
    response = {"id": "gen-example", "provider": "ExampleProvider", "secret": private,
                "choices": [{"finish_reason": "length", "message": {"content": None, "reasoning": private}}],
                "usage": {"completion_tokens": 1024, "completion_tokens_details": {"reasoning_tokens": 1024}, "secret": private}}
    monkeypatch.setenv("OPENROUTER_API_KEY", private)
    monkeypatch.setattr(monitor_client, "_post", lambda *a, **k: response)
    gate = Gate(policy="no_compilation")
    decision = gate.decide("bash", {"command": "ls /app"})
    record = gate.record("bash", {"command": "ls /app"}, decision)
    assert record["decision_source"] == "monitor_error"
    assert record["blocked"]
    attempts = record["monitor_attempts"]
    assert len(attempts) == 3
    assert all(a["content_type"] == "NoneType" and a["parse_status"] == "error" for a in attempts)
    assert attempts[-1]["usage"]["reasoning_tokens"] == 1024
    assert "latency_ms" in attempts[-1]
    assert private not in json.dumps(attempts)


def test_classification_retries_share_one_deadline(monkeypatch):
    ticks = [0.0]
    monkeypatch.setattr(monitor_client, "time", SimpleNamespace(
        monotonic=lambda: ticks[0], sleep=lambda seconds: ticks.__setitem__(0, ticks[0] + seconds)))
    timeouts = []

    def post(url, payload, headers, timeout):
        timeouts.append(timeout)
        ticks[0] += min(40, timeout)
        return {"choices": [{"message": {"content": None}}]}

    monkeypatch.setattr(monitor_client, "_post", post)
    with pytest.raises(monitor_client.MonitorUnavailable, match="deadline exceeded"):
        monitor_client.verdict_direct("bash", {"command": "true"}, policy="no_compilation_trim", api_key="test")
    assert timeouts == [85.0, 45.0, 5.0]
    assert ticks[0] == 85


def test_late_valid_response_is_not_an_approval(monkeypatch):
    ticks = [0.0]
    monkeypatch.setattr(monitor_client, "time", SimpleNamespace(monotonic=lambda: ticks[0]))

    def post(*args):
        ticks[0] = 86
        return {"choices": [{"message": {"content": '{"decision":"ALLOW","reason":"ok"}'}}]}

    monkeypatch.setattr(monitor_client, "_post", post)
    with pytest.raises(monitor_client.MonitorUnavailable, match="deadline exceeded"):
        monitor_client.verdict_direct("bash", {}, policy="no_compilation_trim", api_key="test")


def test_transport_retry_diagnostics_and_success(monkeypatch):
    responses = iter([urllib.error.HTTPError("https://example.invalid", 503, "private", {}, None),
                      {"choices": [{"message": {"content": '{"decision":"BLOCK","reason":"policy"}'}}]}])
    monkeypatch.setattr(monitor_client.time, "sleep", lambda _: None)

    def post(*args):
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(monitor_client, "_post", post)
    diagnostics = []
    verdict, _ = monitor_client.verdict_direct("bash", {}, policy="no_compilation_trim", api_key="test", diagnostics=diagnostics)
    assert verdict.decision == "BLOCK"
    assert [d["http_status"] for d in diagnostics] == [503, 200]
    assert diagnostics[-1]["parse_status"] == "ok"
    assert "private" not in json.dumps(diagnostics)
