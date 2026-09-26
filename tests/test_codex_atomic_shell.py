"""Check the opt-in atomic MCP command and hook contracts without processes."""

import asyncio
import io
import json
import tomllib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from instrumental_evasion.runner import codex_atomic_shell as candidate


@pytest.mark.parametrize("arguments", [
    None, {}, {"command": ""}, {"command": "\0"}, {"command": 1},
    {"command": "true", "stdin": "injected"}, {"command": "x" * 32_769},
    {"command": "é" * 20_000},
])
def test_command_validation_rejects_unsupported_inputs(arguments):
    with pytest.raises(ValueError):
        candidate.validate_arguments(arguments)


def test_tool_surface_has_one_command_and_no_input_session():
    response = candidate.handle_request({"id": 1, "method": "tools/list"})
    assert response["result"]["tools"] == [candidate.TOOL]
    assert candidate.TOOL["inputSchema"]["required"] == ["command"]
    assert set(candidate.TOOL["inputSchema"]["properties"]) == {"command"}
    assert "features.shell_tool=false" in candidate.CONFIGURATION


def test_unknown_tool_cannot_start_a_process(monkeypatch):
    def unexpected(*args):
        raise AssertionError("An unavailable tool must not execute.")

    monkeypatch.setattr(candidate, "execute_command", unexpected)
    response = candidate.handle_request({
        "id": 1, "method": "tools/call", "params": {"name": "write_stdin", "arguments": {"chars": "x"}},
    })
    assert response["error"]["code"] == -32602


def test_mcp_dispatch_preserves_the_reviewable_command(monkeypatch):
    commands = []

    def execute(command):
        commands.append(command)
        return {"returncode": 0, "stdout": "ok", "stderr": ""}

    monkeypatch.setattr(candidate, "execute_command", execute)
    response = candidate.handle_request({
        "id": 1, "method": "tools/call",
        "params": {"name": "run_command", "arguments": {"command": "printf ok"}},
    })
    assert commands == ["printf ok"]
    assert response["result"]["isError"] is False


@pytest.mark.parametrize("returncode", [0, 1, 2])
def test_hook_adapter_preserves_call_identity_and_fails_closed(monkeypatch, capsys, returncode):
    request = {"hook_event_name": "PreToolUse", "tool_name": candidate.MCP_TOOL,
               "tool_input": {"command": "printf ok"}, "tool_use_id": "expected-id"}
    monkeypatch.setattr(candidate.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(request).encode())))
    received = []

    def run(command, **kwargs):
        received.append(json.loads(kwargs["input"]))
        return SimpleNamespace(returncode=returncode, stdout="{}", stderr="")

    monkeypatch.setattr(candidate.subprocess, "run", run)
    assert candidate.run_hook() == (returncode if returncode in {0, 2} else 2)
    capsys.readouterr()
    assert received == [dict(request, tool_name="Bash")]


def test_hook_timeout_fails_closed(monkeypatch, capsys):
    request = {"tool_name": candidate.MCP_TOOL, "tool_input": {"command": "true"}}
    monkeypatch.setattr(candidate.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(request).encode())))

    def timeout(*args, **kwargs):
        raise TimeoutError("Synthetic timeout.")

    monkeypatch.setattr(candidate.subprocess, "run", timeout)
    assert candidate.run_hook() == 2
    assert "Tool execution denied." in capsys.readouterr().err


@pytest.mark.parametrize("tool", ["spawn_agent", "Agent", "multi_agent_v1__spawn_agent", "multi_agent_v1.spawn_agent"])
def test_delegation_is_unavailable_before_monitoring(monkeypatch, capsys, tool):
    request = {"tool_name": tool, "tool_input": {"message": "Test"}}
    monkeypatch.setattr(candidate.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(request).encode())))

    def unexpected(*args, **kwargs):
        raise AssertionError("Unavailable delegation must not reach the monitor.")

    monkeypatch.setattr(candidate.subprocess, "run", unexpected)
    assert candidate.run_hook() == 2
    assert json.loads(capsys.readouterr().out)["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_scaffold_requires_explicit_candidate_selection(monkeypatch, tmp_path):
    from instrumental_evasion.runner.scaffolds.codex import CodexScaffold

    monkeypatch.delenv("CODEX_TOOL_SCOPE", raising=False)
    ctx = SimpleNamespace(spec=SimpleNamespace(workdir="/app", container_env={}), system_prompt="", outdir=tmp_path)
    native = CodexScaffold()
    assert native.tool_scope == "native"
    assert not any("mcp_servers.terminal" in argument for argument in native._run_command(ctx, "test", None))
    assert "/opt/terminal-tools.py" not in native._hooks_file(ctx).read_text()


def test_managed_requirements_ignore_writable_hook_sources():
    requirements = tomllib.loads(candidate.managed_requirements())
    assert requirements["allow_managed_hooks_only"] is True
    assert requirements["features"]["hooks"] is True
    assert requirements["hooks"]["PreToolUse"] == candidate.hook_configuration()["hooks"]["PreToolUse"]
    assert requirements["mcp_servers"] == {"terminal": {"identity": {"command": "/opt/pyrt/bin/python3"}}}


@pytest.mark.parametrize("thread", [None, "existing-thread"])
def test_scaffold_uses_the_qualified_candidate_config_on_each_invocation(tmp_path, thread):
    from instrumental_evasion.runner.scaffolds.codex import CodexScaffold

    ctx = SimpleNamespace(spec=SimpleNamespace(workdir="/app", container_env={}), system_prompt="", outdir=tmp_path)
    scaffold = CodexScaffold(tool_scope=candidate.CONDITION)
    command = scaffold._run_command(ctx, "test", thread)
    assert all(value in command for value in candidate.CONFIGURATION)
    assert json.loads(scaffold._hooks_file(ctx).read_text()) == candidate.hook_configuration()


def test_candidate_rejects_an_unqualified_runtime():
    from instrumental_evasion.runner.scaffolds.codex import CodexScaffold

    scaffold = CodexScaffold(tool_scope=candidate.CONDITION)
    with pytest.raises(RuntimeError, match="requires the Modal runtime"):
        asyncio.run(scaffold.run(SimpleNamespace(sandbox=object())))


def test_modal_scaffold_stages_the_candidate_module_read_only(tmp_path, monkeypatch):
    from instrumental_evasion.runner.gate import Gate
    from instrumental_evasion.runner.sandbox import ExecResult
    from instrumental_evasion.runner.scaffolds.base import EpisodeContext, Limits
    from instrumental_evasion.runner.scaffolds.codex import CodexScaffold

    stream = '\n'.join(json.dumps(event) for event in [
        {"type": "thread.started", "thread_id": "test-thread"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Done."}},
        {"type": "turn.completed"},
    ])
    sandbox = SimpleNamespace(
        start_remote_gate=AsyncMock(return_value=("http://gate", "synthetic-token")),
        allow_hosts=AsyncMock(), stage_path=AsyncMock(), prepare_user=AsyncMock(), write_file=AsyncMock(),
        remote_gate_count=AsyncMock(side_effect=[0, 1, 1]),
        exec=AsyncMock(return_value=ExecResult(True, 0, stream, "")),
    )
    ctx = EpisodeContext(
        spec=SimpleNamespace(workdir="/app", container_env={}, task_egress_hosts_for=lambda **kwargs: ()),
        sandbox=sandbox, gate=Gate(policy=None), prompt="Test", system_prompt="",
        limits=Limits(1, 100, 10, 1), outdir=tmp_path, monitored=False,
        score=AsyncMock(return_value=(True, "", {})),
    )
    scaffold = CodexScaffold(tool_scope=candidate.CONDITION)
    monkeypatch.setattr(scaffold, "_codex_home", lambda _ctx, path: path)
    result = asyncio.run(scaffold.run(ctx))
    assert result.error is None
    stages = sandbox.stage_path.await_args_list
    assert stages[0].args[1] == candidate.REMOTE_MODULE
    assert stages[0].kwargs == {"read_only": True}
    assert ctx.gate.records() == []


@pytest.mark.parametrize("probe_alive", [True, False])
def test_modal_redacts_staged_auth_before_retaining_any_output(tmp_path, monkeypatch, probe_alive):
    from instrumental_evasion.runner.gate import Gate
    from instrumental_evasion.runner.sandbox import ExecResult
    from instrumental_evasion.runner.scaffolds.base import EpisodeContext, Limits
    from instrumental_evasion.runner.scaffolds.codex import CodexScaffold

    secret = 'synthetic-token-with-"quotes"-and-\n-newline'
    gate_token = "synthetic-gate-token-0987654321"
    stream = '\n'.join(json.dumps(event) for event in [
        {"type": "thread.started", "thread_id": "test-thread"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": secret + gate_token}},
        {"type": "turn.completed"},
    ])
    sandbox = SimpleNamespace(
        start_remote_gate=AsyncMock(return_value=("http://gate", gate_token)),
        allow_hosts=AsyncMock(), stage_path=AsyncMock(), prepare_user=AsyncMock(), write_file=AsyncMock(),
        remote_gate_count=AsyncMock(side_effect=[0, int(probe_alive), 1]),
        exec=AsyncMock(return_value=ExecResult(True, 0, stream, secret + gate_token)),
    )
    ctx = EpisodeContext(
        spec=SimpleNamespace(workdir="/app", container_env={}, task_egress_hosts_for=lambda **kwargs: ()),
        sandbox=sandbox, gate=Gate(policy=None), prompt="Test", system_prompt="",
        limits=Limits(1, 100, 10, 1), outdir=tmp_path, monitored=False,
        score=AsyncMock(return_value=(True, "", {})),
    )
    scaffold = CodexScaffold(tool_scope=candidate.CONDITION)

    def home(_ctx, path):
        path.mkdir()
        (path / "auth.json").write_text(json.dumps({"tokens": {"access_token": secret}}))
        return path

    monkeypatch.setattr(scaffold, "_codex_home", home)
    result = asyncio.run(scaffold.run(ctx))
    retained = json.dumps({"extra": result.extra, "messages": [m.content for m in result.messages], "error": result.error})
    retained += ''.join(path.read_text() for path in tmp_path.iterdir() if path.is_file())
    assert json.dumps(secret)[1:-1] not in retained
    assert gate_token not in retained
    assert "[REDACTED CREDENTIAL]" in retained
