"""Require verified disposal even when a gate or provider operation fails."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from instrumental_evasion.runner import modal as runtime


def handle(*, code=137, error=None):
    return SimpleNamespace(
        terminate=SimpleNamespace(aio=AsyncMock(side_effect=error)),
        poll=SimpleNamespace(aio=AsyncMock(return_value=code)),
    )


@pytest.mark.parametrize("owner", ["task", "gate", "sidecar"])
@pytest.mark.parametrize("failure", [None, "still_running", "provider_error"])
def test_cleanup_verifies_each_resource(owner, failure):
    remote = handle(
        code=None if failure == "still_running" else 137,
        error=RuntimeError("provider unavailable") if failure == "provider_error" else None,
    )
    if owner == "task":
        operation = runtime.ModalSandbox(remote, "/app", "none", None).cleanup
    elif owner == "gate":
        operation = runtime.ModalRemoteGate(remote, "https://gate.invalid", "test-token").stop
    else:
        operation = runtime.ModalSidecar(remote, "fixture.invalid").stop
    if failure:
        with pytest.raises(RuntimeError):
            asyncio.run(operation())
    else:
        asyncio.run(operation())
    remote.terminate.aio.assert_awaited_once_with(wait=True)
    if failure != "provider_error":
        remote.poll.aio.assert_awaited_once()


def test_gate_cleanup_failure_still_disposes_task_home():
    remote = handle()
    sandbox = runtime.ModalSandbox(remote, "/app", "none", None)
    sandbox.remote_gate = SimpleNamespace(stop=AsyncMock(side_effect=RuntimeError("gate unavailable")))
    with pytest.raises(RuntimeError, match="gate unavailable"):
        asyncio.run(sandbox.cleanup())
    remote.terminate.aio.assert_awaited_once_with(wait=True)
    remote.poll.aio.assert_awaited_once()


@pytest.mark.parametrize("stalled", ["terminate", "poll"])
def test_cleanup_has_one_bounded_deadline(monkeypatch, stalled):
    remote = handle()

    async def wait_forever(**kwargs):
        await asyncio.Event().wait()

    getattr(remote, stalled).aio.side_effect = wait_forever
    monkeypatch.setattr(runtime, "_CLEANUP_TIMEOUT", 0.01)
    with pytest.raises(TimeoutError):
        asyncio.run(runtime._terminate_sandbox(remote))


def test_cancelled_sidecar_startup_disposes_created_resource(monkeypatch, tmp_path):
    remote = handle()
    remote.tunnels = lambda **kwargs: (_ for _ in ()).throw(asyncio.CancelledError())
    monkeypatch.setattr(runtime.modal.Sandbox, "create", lambda *args, **kwargs: remote)
    runner = runtime.ModalRuntime()
    runner._app = object()
    monkeypatch.setattr(runner, "_target_image", AsyncMock(return_value=object()))
    spec = SimpleNamespace(service_port=8000, control_auth=False, start=("service",), docker_cmd=())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(runner.start_sidecar(spec, SimpleNamespace(), tmp_path))
    remote.terminate.aio.assert_awaited_once_with(wait=True)
    remote.poll.aio.assert_awaited_once()
