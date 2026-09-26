"""Test Codex failure handling with synthetic event producers and no providers."""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from instrumental_evasion.runner.codex_sessions import SessionCapture
from instrumental_evasion.runner.gate import Gate
from instrumental_evasion.runner.sandbox import ExecResult
from instrumental_evasion.runner.scaffolds import cli_process, codex
from instrumental_evasion.runner.scaffolds.base import EpisodeContext, Limits


def _events(*, completed=True, thread="synthetic-thread", text="Progress."):
    events = [
        {"type": "thread.started", "thread_id": thread},
        {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
    ]
    if completed:
        events.append({"type": "turn.completed", "usage": {"input_tokens": 2, "output_tokens": 1}})
    return "\n".join(json.dumps(event) for event in events)


@pytest.fixture
def local_codex(tmp_path, monkeypatch):
    scaffold = codex.CodexScaffold("openrouter/deepseek/deepseek-v4-pro-0813", cli_timeout=0.5)
    ctx = EpisodeContext(
        spec=SimpleNamespace(workdir="/workdir"),
        sandbox=SimpleNamespace(_directory=str(tmp_path)),
        gate=Gate(policy=None),
        prompt="Synthetic task",
        system_prompt="Synthetic instructions",
        limits=Limits(max_turns=3, token_limit=100, time_limit_seconds=10, attempts=3),
        outdir=tmp_path,
        monitored=False,
        score=AsyncMock(return_value=(False, "", {})),
    )
    monkeypatch.setattr(scaffold, "_ensure_gate_server", lambda ctx: "unused")
    monkeypatch.setattr(scaffold, "_ensure_proxy", lambda ctx: "unused")
    monkeypatch.setattr(scaffold, "_codex_home", lambda ctx: tmp_path / "private-home")
    monkeypatch.setattr(scaffold, "_hooks_file", lambda ctx: tmp_path / "hooks.json")
    monkeypatch.setattr(scaffold, "_env", lambda *args: {})
    monkeypatch.setattr(scaffold, "_probe_hook", lambda *args, **kwargs: {"hook_alive": True})
    return scaffold, ctx


def test_timeout_retains_partial_messages_and_previous_usage(local_codex, monkeypatch):
    scaffold, ctx = local_codex

    def command(ctx, home, hooks, env, prompt, thread):
        if thread is None:
            code = f"print({_events()!r})"
        else:
            code = f"import time; print({_events(completed=False, text='Partial progress.')!r}, flush=True); time.sleep(60)"
        return [sys.executable, "-c", code]

    monkeypatch.setattr(scaffold, "_command", command)
    result = asyncio.run(scaffold.run(ctx))
    assert result.stop_reason == "time_limit"
    assert result.turns == 2
    assert result.usage.total == 3
    assert result.extra["cli_timed_out"]
    assert result.error is None
    assert result.extra["refusal_turns_observed"] == 1
    assert [m.content for m in result.messages if m.role == "assistant"] == ["Progress.", "Partial progress."]
    assert "Partial progress." in (ctx.outdir / "stdout.txt").read_text()
    assert ctx.gate.records() == []


@pytest.mark.parametrize("tokens,expected,partial", [(110, "token_limit", False), (60, "time_limit", False), (60, "time_limit", True)])
def test_live_usage_stops_invocation_and_survives_timeout(local_codex, monkeypatch, tokens, expected, partial):
    scaffold, ctx = local_codex
    scaffold.token_budget_watchdog = True
    # Leave the child enough time to write its session record before the CLI timeout.
    scaffold.timeout = 2
    monkeypatch.setattr(codex, "cli_version", lambda binary: "codex-cli 0.153.3")
    home = ctx.outdir / "private-home"
    (home / "sessions").mkdir(parents=True)
    usage_record = json.dumps({"type": "token_usage_record", "payload": {
        "thread_id": "synthetic-thread", "thread_token_usage": {
            "input_tokens": tokens - 10, "cached_input_tokens": 20, "output_tokens": 10, "total_tokens": tokens,
        },
    }}) + "\n"
    path = home / "sessions/rollout-task.jsonl"
    stdout = '{"type":"item' if partial else _events(completed=False)
    code = (
        "import pathlib,time; "
        f"pathlib.Path({str(path)!r}).write_text({usage_record!r}); "
        f"print({stdout!r}, flush=True); time.sleep(60)"
    )
    monkeypatch.setattr(scaffold, "_command", lambda *args: [sys.executable, "-c", code])
    result = asyncio.run(scaffold.run(ctx))
    assert result.stop_reason == expected
    assert result.usage.total == tokens
    assert result.usage.cache_read_tokens == 20
    assert result.extra["token_budget_diagnostics"]["revision"] == "response_completion_hook_latency_v3"
    assert result.extra["token_budget_diagnostics"]["observed_tokens"] == tokens
    assert result.error is None
    assert result.extra["refusal_turns_observed"] == 0
    assert ctx.gate.records() == []


def test_resume_failure_stops_without_resetting_the_thread(local_codex, monkeypatch):
    scaffold, ctx = local_codex
    threads = []
    error = "Error: thread/resume: thread/resume failed: list_turns is not supported yet (code -32601)"

    def command(ctx, home, hooks, env, prompt, thread):
        threads.append(thread)
        if thread is None:
            code = f"print({_events()!r})"
        else:
            code = f"import sys; print({error!r}, file=sys.stderr); sys.exit(1)"
        return [sys.executable, "-c", code]

    monkeypatch.setattr(scaffold, "_command", command)
    # A later successful score must not hide the CLI failure.
    ctx.score.side_effect = [(False, "", {}), (True, "", {})]
    result = asyncio.run(scaffold.run(ctx))
    assert result.stop_reason == "error"
    assert result.extra["cli_failure_kind"] == "resume_history_unsupported"
    assert error in result.error
    assert threads == [None, "synthetic-thread", "synthetic-thread", "synthetic-thread"]
    assert len(result.extra["resume_retries"]) == 2
    assert ctx.score.await_count == 1
    assert (ctx.outdir / "stderr.txt").read_text().count(error) == 3


def test_resume_retry_preserves_prompt_history_usage_and_refusal_count(local_codex, monkeypatch):
    scaffold, ctx = local_codex
    calls = []

    def command(ctx, home, hooks, env, prompt, thread):
        calls.append((prompt, thread))
        if len(calls) == 2:
            code = f"import sys; print({codex.codex_resume.ERROR!r}, file=sys.stderr); sys.exit(1)"
        else:
            code = f"print({_events(text='I cannot help with that.')!r})"
        return [sys.executable, "-c", code]

    monkeypatch.setattr(scaffold, "_command", command)
    ctx.score.side_effect = [(False, "", {}), (True, "", {})]
    result = asyncio.run(scaffold.run(ctx))
    assert result.stop_reason == "submit"
    assert result.error is None
    assert len(calls) == 3
    assert calls[1] == calls[2]
    assert calls[1][1] == "synthetic-thread"
    assert len(result.messages) == 4
    assert result.turns == 2
    assert result.usage.total == 3
    assert result.extra["refusal_turns_observed"] == 1
    assert result.extra["resume_retries"] == [
        {"rung": 1, "thread_id": "synthetic-thread", "retry": 1, "delay_seconds": 0.5}
    ]
    assert ctx.score.await_count == 2
    assert codex.codex_resume.ERROR in (ctx.outdir / "stderr.txt").read_text()


@pytest.mark.parametrize("activity", ["stdout", "gate", "gate_during_delay"])
def test_resume_error_after_activity_is_not_retried(local_codex, monkeypatch, activity):
    scaffold, ctx = local_codex
    calls = []
    records = []
    monkeypatch.setattr(ctx.gate, "records", lambda: records)

    if activity == "gate_during_delay":
        async def new_activity(delay):
            records.append(object())
        monkeypatch.setattr(codex, "asyncio", SimpleNamespace(sleep=new_activity))

    def command(ctx, home, hooks, env, prompt, thread):
        calls.append(thread)
        if thread is None:
            code = f"print({_events()!r})"
        else:
            if activity == "gate":
                records.append(object())
            code = f"import sys; print({codex.codex_resume.ERROR!r}, file=sys.stderr); "
            if activity == "stdout":
                code += f"print({_events(completed=False)!r}); "
            code += "sys.exit(1)"
        return [sys.executable, "-c", code]

    monkeypatch.setattr(scaffold, "_command", command)
    result = asyncio.run(scaffold.run(ctx))
    assert result.stop_reason == "error"
    assert calls == [None, "synthetic-thread"]
    assert result.extra["resume_retries"] == []
    ctx.score.assert_awaited_once()


def test_resume_retry_delay_cannot_extend_episode_deadline(local_codex, monkeypatch):
    scaffold, ctx = local_codex
    calls = []
    now = [0.0]
    # Replace only this module's clock. asyncio and process capture retain real time.
    monkeypatch.setattr(codex, "time", SimpleNamespace(monotonic=lambda: now[0]))

    async def delayed_scheduler(delay):
        now[0] = ctx.limits.time_limit_seconds + 1

    monkeypatch.setattr(codex, "asyncio", SimpleNamespace(sleep=delayed_scheduler))

    def command(ctx, home, hooks, env, prompt, thread):
        calls.append(thread)
        code = f"print({_events()!r})" if thread is None else (
            f"import sys; print({codex.codex_resume.ERROR!r}, file=sys.stderr); sys.exit(1)"
        )
        return [sys.executable, "-c", code]

    monkeypatch.setattr(scaffold, "_command", command)
    result = asyncio.run(scaffold.run(ctx))
    assert result.stop_reason == "time_limit"
    assert calls == [None, "synthetic-thread"]
    assert result.extra["cli_timed_out"]
    ctx.score.assert_awaited_once()


@pytest.mark.parametrize("recover", [False, True])
def test_modal_resume_retries_keep_the_same_thread_and_all_error_logs(local_codex, monkeypatch, recover):
    scaffold, ctx = local_codex
    scaffold.timeout = 10
    gate_count = [0]
    commands = []

    async def execute(command, **kwargs):
        if command[0] != codex.CONTAINER_BIN:
            return ExecResult(True, 0, "", "")
        if "Run the shell command" in command[-1]:
            gate_count[0] += 1
            return ExecResult(True, 0, "", "")
        commands.append(command)
        if len(commands) == 2 or (len(commands) > 2 and not recover):
            return ExecResult(False, 1, "", codex.codex_resume.ERROR + "\n")
        return ExecResult(True, 0, _events(), "")

    ctx.sandbox = SimpleNamespace(
        start_remote_gate=AsyncMock(return_value=("http://gate", "synthetic-token")),
        allow_hosts=AsyncMock(), stage_path=AsyncMock(), prepare_user=AsyncMock(), write_file=AsyncMock(),
        remote_gate_count=AsyncMock(side_effect=lambda: gate_count[0]), exec=execute,
    )
    ctx.spec.container_env = {}
    hooks = ctx.outdir / "hooks.json"
    hooks.write_text("{}")
    monkeypatch.setattr(scaffold, "_effective_egress_allow", lambda ctx: ())
    monkeypatch.setattr(scaffold, "_api_provider", None)
    monkeypatch.setattr(scaffold, "_capture_secrets", lambda *args: ())

    def home(ctx, path):
        path.mkdir()
        return path

    monkeypatch.setattr(scaffold, "_codex_home", home)
    ctx.score.side_effect = [(False, "", {}), (True, "", {})]
    result = asyncio.run(scaffold._run_modal(ctx, SessionCapture(ctx.outdir)))
    assert result.stop_reason == ("submit" if recover else "error")
    assert len(commands) == (3 if recover else 4)
    assert all(command == commands[1] for command in commands[2:])
    assert commands[1][commands[1].index("resume") + 1] == "synthetic-thread"
    assert len(result.extra["resume_retries"]) == (1 if recover else 2)
    assert (ctx.outdir / "stderr.txt").read_text().count(codex.codex_resume.ERROR) == (1 if recover else 3)
    assert ctx.score.await_count == (2 if recover else 1)


@pytest.mark.parametrize("oversized", [False, True])
def test_malformed_stream_cannot_be_hidden_by_a_successful_score(local_codex, monkeypatch, oversized):
    scaffold, ctx = local_codex
    ctx.score.return_value = (True, "", {})
    monkeypatch.setattr(cli_process, "_EVENT_BYTES", 256)
    stdout = _events(text="x" * 1024) if oversized else _events(completed=False)
    monkeypatch.setattr(scaffold, "_command", lambda *args: [sys.executable, "-c", f"print({stdout!r})"])
    result = asyncio.run(scaffold.run(ctx))
    assert result.stop_reason == "error"
    assert result.extra["void_reason"] == "refusal_stream_invalid"
    ctx.score.assert_not_awaited()


def test_valid_control_success_still_uses_the_external_score(local_codex, monkeypatch):
    scaffold, ctx = local_codex
    ctx.score.return_value = (True, "", {})
    monkeypatch.setattr(scaffold, "_command", lambda *args: [sys.executable, "-c", f"print({_events()!r})"])
    result = asyncio.run(scaffold.run(ctx))
    assert result.stop_reason == "submit"
    assert result.usage.total == 3
    assert result.error is None
    ctx.score.assert_awaited_once()
    assert ctx.gate.records() == []


def test_probe_timeout_keeps_sanitized_stage_evidence(tmp_path, monkeypatch):
    scaffold = codex.CodexScaffold()
    ctx = EpisodeContext(
        spec=SimpleNamespace(workdir="/workdir"), sandbox=None, gate=Gate(policy=None),
        prompt="", system_prompt="", limits=Limits(1, 100, 10, 1),
        outdir=tmp_path, monitored=False, score=AsyncMock(),
    )
    secret = "synthetic-probe-token-123456789"
    code = (
        f"import sys,time; print({_events(completed=False)!r}, flush=True); "
        f"print('waiting ' + {secret!r}, file=sys.stderr, flush=True); time.sleep(60)"
    )
    monkeypatch.setattr(scaffold, "_command", lambda *args, **kwargs: [sys.executable, "-c", code])
    run = codex.run_cli_process

    def bounded_run(*args, **kwargs):
        kwargs["timeout"] = 0.5
        return run(*args, **kwargs)

    monkeypatch.setattr(codex, "run_cli_process", bounded_run)
    result = scaffold._probe_hook(ctx, tmp_path / "home", tmp_path / "hooks", {}, secrets=(secret,))
    assert not result["hook_alive"]
    assert result["reason"] == "probe timed out"
    assert result["thread_started"]
    assert not result["turn_completed"]
    assert result["gate_calls"] == 0
    assert result["timed_out"]
    assert not (tmp_path / "stdout.txt").exists()
    for name in ["probe.stdout.txt", "probe.stderr.txt", "probe.json"]:
        assert (tmp_path / name).exists()
        assert secret not in (tmp_path / name).read_text()
    assert "waiting [REDACTED CREDENTIAL]" in (tmp_path / "probe.stderr.txt").read_text()


def test_provider_startup_error_is_not_reported_as_a_dead_hook(tmp_path, monkeypatch):
    scaffold = codex.CodexScaffold()
    ctx = EpisodeContext(
        spec=SimpleNamespace(workdir="/workdir"), sandbox=None, gate=Gate(policy=None),
        prompt="", system_prompt="", limits=Limits(1, 100, 10, 1),
        outdir=tmp_path, monitored=False, score=AsyncMock(),
    )
    events = '\n'.join(json.dumps(event) for event in [
        {"type": "thread.started", "thread_id": "synthetic-thread"},
        {"type": "turn.failed", "error": {"message": "unexpected status 502 Bad Gateway"}},
    ])
    command = [sys.executable, "-c", f"import sys; print({events!r}); sys.exit(1)"]
    monkeypatch.setattr(scaffold, "_command", lambda *args, **kwargs: command)
    result = scaffold._probe_hook(ctx, tmp_path / "home", tmp_path / "hooks", {})
    assert not result["hook_alive"]
    assert result["gate_calls"] == 0
    assert result["void_reason"] == "cli_startup_error"
    assert "502 Bad Gateway" in result["reason"]
