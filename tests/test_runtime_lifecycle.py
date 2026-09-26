"""Keep ownership of sandbox resources until setup succeeds."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.mark.parametrize("error", [RuntimeError("mount failed"), asyncio.CancelledError()])
def test_modal_mount_failure_terminates_partial_sandbox(monkeypatch, error):
    from instrumental_evasion.runner import modal as runtime_module

    stopped = []
    sandbox = SimpleNamespace(
        mount_image=lambda *args: (_ for _ in ()).throw(error),
        terminate=SimpleNamespace(aio=AsyncMock(side_effect=lambda **kwargs: stopped.append(kwargs))),
        poll=SimpleNamespace(aio=AsyncMock(return_value=137)),
    )
    monkeypatch.setattr(runtime_module.modal.Sandbox, "create", lambda *args, **kwargs: sandbox)
    runtime = runtime_module.ModalRuntime(scaffold=SimpleNamespace(name="codex", modal_runtime_image=True))
    runtime._app = object()
    runtime._prepared_images = SimpleNamespace(resolve=lambda key: (object(), None))

    async def image(spec):
        return runtime_module.ModalImage(object())

    monkeypatch.setattr(runtime, "_agent_image", image)
    spec = SimpleNamespace(network_for=lambda **kwargs: "blackhole", container_env={}, workdir="/app")
    with pytest.raises(type(error)):
        asyncio.run(runtime.open_sandbox(spec, monitored=True))
    assert stopped == [{"wait": True}]


def test_apptainer_setup_cancellation_removes_state(monkeypatch, tmp_path):
    from instrumental_evasion.runner import sandbox as sandbox_module

    image = tmp_path / "image.sif"
    image.touch()
    stopped = []

    class Sandbox:
        def __init__(self, *args):
            pass

        async def verify(self):
            raise asyncio.CancelledError()

        def cleanup(self):
            stopped.append(True)

    monkeypatch.setattr(sandbox_module, "AppSandbox", Sandbox)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(sandbox_module.open_app_sandbox(str(image), "/app"))
    assert stopped == [True]
