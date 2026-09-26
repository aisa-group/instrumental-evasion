"""Opt-in Modal and Claude Code configuration: tool scope, resource caps, egress."""

import asyncio
from types import SimpleNamespace

import pytest

from instrumental_evasion.hooks.deploy import claude_tool_arguments
from instrumental_evasion.runner import modal as runtime_module
from instrumental_evasion.runner.sandbox import ExecResult


def test_claude_tool_restriction_requires_explicit_selection(monkeypatch):
    monkeypatch.delenv("CLAUDE_TOOL_SCOPE", raising=False)
    assert claude_tool_arguments() == []
    monkeypatch.setenv("CLAUDE_TOOL_SCOPE", "configured")
    assert claude_tool_arguments() == ["--tools", "Bash,Edit,Write,Read,Glob,Grep"]
    monkeypatch.setenv("CLAUDE_TOOL_SCOPE", "misspelled")
    with pytest.raises(ValueError, match="CLAUDE_TOOL_SCOPE"):
        claude_tool_arguments()


@pytest.mark.parametrize("name,value", [
    ("MODAL_SANDBOX_CPU", "nan"), ("MODAL_SANDBOX_CPU", "0"),
    ("MODAL_SANDBOX_MEMORY_MIB", "-1"), ("MODAL_SANDBOX_LIFETIME_SECONDS", "0"),
])
def test_invalid_resource_caps_fail_before_creation(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        runtime_module.ModalRuntime()


def test_configured_task_and_sidecar_keep_network_and_resource_caps(monkeypatch, tmp_path):
    monkeypatch.setenv("MODAL_NETWORK_POLICY", "configured")
    monkeypatch.setenv("MODAL_SANDBOX_CPU", "1")
    monkeypatch.setenv("MODAL_SANDBOX_MEMORY_MIB", "2048")
    monkeypatch.setenv("MODAL_SANDBOX_LIFETIME_SECONDS", "1200")
    monkeypatch.delenv("MODAL_IMAGE_MANIFEST", raising=False)
    creations, updates = [], []

    def create(*args, **kwargs):
        creations.append(kwargs)
        return SimpleNamespace(
            _experimental_set_outbound_network_policy=lambda **policy: updates.append(policy),
            tunnels=lambda **kwargs: {8000: SimpleNamespace(host="fixture.example", port=443)},
        )

    async def execute(*args, **kwargs):
        return ExecResult(True, 0, "", "")

    async def image(*args):
        return runtime_module.ModalImage(object())

    async def target(*args):
        return object()

    monkeypatch.setattr(runtime_module.modal.Sandbox, "create", create)
    monkeypatch.setattr(runtime_module.ModalSandbox, "exec", execute)
    runtime = runtime_module.ModalRuntime()
    runtime._app = object()
    monkeypatch.setattr(runtime, "_agent_image", image)
    monkeypatch.setattr(runtime, "_target_image", target)
    spec = SimpleNamespace(network_for=lambda **kwargs: "proxy", workdir="/app",
                           container_env={"HTTPS_PROXY": "http://controller.invalid"},
                           task_egress_hosts_for=lambda **kwargs: ("hf.co",))
    sidecar = SimpleNamespace(service_port=8000, control_auth=False,
                              start=("service", "--port", "{port}"), docker_cmd=())

    async def run():
        sandbox = await runtime.open_sandbox(spec, monitored=True)
        await sandbox.allow_hosts(["api.anthropic.com"])
        await runtime.start_sidecar(sidecar, sandbox, tmp_path)

    asyncio.run(run())
    assert creations[0]["outbound_domain_allowlist"] == ["*.hf.co"]
    assert "HTTPS_PROXY" not in creations[0]["env"]
    assert creations[1]["outbound_domain_allowlist"] == [runtime_module._CLOSED_DOMAIN_SENTINEL]
    assert runtime_module._CLOSED_DOMAIN_SENTINEL == "closed.invalid"
    assert set(updates[-1]["outbound_domain_allowlist"]) == {"*.hf.co", "*.api.anthropic.com", "fixture.example"}
    assert all(row["cpu"] == (1, 1) and row["memory"] == (2048, 2048)
               and row["timeout"] == 1200 for row in creations)
