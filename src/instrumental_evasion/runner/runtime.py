"""The container runtime, as a plugin -- Apptainer, Docker, or Modal.

The runner has three independent axes: the task (`TBTaskSpec`), the agent
harness (`runner.scaffolds`), and the container runtime. A scaffold drives the
agent; a runtime is what a container *is*. Both are plugins so that switching
one does not touch the other.

    from instrumental_evasion.runner.runtime import get_runtime
    runtime = get_runtime("modal")      # or "apptainer" / "docker"

Apptainer needs no daemon, but run rootless it erases a privilege boundary (no
setuid in the user namespace) and has no multi-container network. Rootless
Docker restores both -- real UID separation and a bridge network -- so tasks
that ship Docker images and multi-container setups run unchanged. Modal runs the
same registry images in remote sandboxes and needs no local container runtime.

A runtime owns exactly the runtime-specific parts of an episode:

  * `open_sandbox(spec)` -- the agent's container, as a `SandboxLike`
    (exec/read_file/write_file) plus `cleanup()`;
  * `start_sidecar(sidecar, agent_sandbox, outdir)` -- the task's service
    container, reachable from the agent (a localhost port on Apptainer, a bridge
    hostname on Docker, a tunnel on Modal), or None for a task without one;
  * `read_static_flag(sidecar, argv)` -- run a short read-only command in a
    fresh container of the sidecar's image to read a value baked into it, at
    score time;
  * `setup()`/`teardown()` -- once-per-run lifecycle (Docker starts and stops its
    rootless daemon here; Apptainer does nothing).

Everything else -- the gate, scoring via `spec.score_fn`, attempt
classification, the log -- is the runner's and is identical across runtimes,
which is what keeps numbers from different runtimes comparable.
"""

from __future__ import annotations

import contextvars
from typing import Any, Awaitable, Callable, Protocol

# Set per episode by the runner to the active runtime's reader, so a score_fn
# (which is handed only the agent sandbox) can read a value baked into a
# sidecar's image without knowing which runtime is live. Falls back to the
# Apptainer reader when the runner has not set one.
_flag_reader: contextvars.ContextVar[
    "Callable[[Any, tuple[str, ...]], Awaitable[str]] | None"
] = contextvars.ContextVar("flag_reader", default=None)


def set_flag_reader(fn: "Callable[[Any, tuple[str, ...]], Awaitable[str]]") -> None:
    _flag_reader.set(fn)


async def _apptainer_read_flag(image: str, argv: tuple[str, ...]) -> str:
    import asyncio

    from instrumental_evasion.apptainer_core import CONTAINMENT_FLAGS

    args = ["apptainer", "exec", *CONTAINMENT_FLAGS, image, *argv]
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    out, _ = await proc.communicate()
    return out.decode("utf-8", "replace").strip()


async def read_static_flag(sidecar: Any, argv: tuple[str, ...]) -> str:
    """Read a static flag from a sidecar's target image via the active runtime.

    Takes the whole `Sidecar` (not a bare image string) because Apptainer wants
    its SIF path and Docker wants its registry ref -- the runtime picks. Defaults
    to a one-shot contained apptainer exec so the specs work under the default
    runtime and in a plain unit test.
    """
    fn = _flag_reader.get()
    if fn is not None:
        return await fn(sidecar, argv)
    return await _apptainer_read_flag(sidecar.image(), argv)


class Runtime(Protocol):
    """One container runtime. One instance per run, reused across episodes."""

    name: str

    async def setup(self) -> None:
        """Once-per-run start-up (Docker: launch the rootless daemon)."""
        ...

    def teardown(self) -> None:
        """Once-per-run tear-down (Docker: stop the daemon)."""
        ...

    def describe(self) -> dict[str, Any]:
        """Facts for the eval log's metadata; must include `runtime`."""
        ...

    async def open_sandbox(self, spec: Any, *, monitored: bool) -> Any:
        """A verified, ready sandbox for one episode. Caller must `cleanup()`."""
        ...

    async def start_sidecar(self, sidecar: Any, agent_sandbox: Any, outdir: Any) -> Any:
        """Start the task's sidecar reachable from `agent_sandbox`, or return None."""
        ...

    async def read_static_flag(self, sidecar: Any, argv: tuple[str, ...]) -> str:
        """Read a static flag from a sidecar's target image in a throwaway container."""
        ...


# --------------------------------------------------------------------------- #
# Apptainer: thin delegation to `runner.sandbox` and `runner.sidecar`.
# --------------------------------------------------------------------------- #
class ApptainerRuntime:
    name = "apptainer"

    async def setup(self) -> None:
        return None

    def teardown(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"runtime": "apptainer"}

    async def open_sandbox(self, spec: Any, *, monitored: bool) -> Any:
        from instrumental_evasion.runner.sandbox import open_app_sandbox

        return await open_app_sandbox(
            spec.image(),
            spec.workdir,
            spec.network_for(monitored=monitored),
            spec.container_env,
        )

    async def start_sidecar(self, sidecar: Any, agent_sandbox: Any, outdir: Any) -> Any:
        from instrumental_evasion.runner.sidecar import start_sidecar

        return await start_sidecar(sidecar, outdir)

    async def read_static_flag(self, sidecar: Any, argv: tuple[str, ...]) -> str:
        return await _apptainer_read_flag(sidecar.image(), argv)


def _apptainer(**options: Any) -> Runtime:
    return ApptainerRuntime()


def _docker(**options: Any) -> Runtime:
    from instrumental_evasion.runner.docker import DockerRuntime

    return DockerRuntime(**options)


def _modal(**options: Any) -> Runtime:
    from instrumental_evasion.runner.modal import ModalRuntime

    return ModalRuntime(**options)


_FACTORIES: dict[str, Callable[..., Runtime]] = {
    "apptainer": _apptainer,
    "docker": _docker,
    "modal": _modal,
}

DEFAULT_RUNTIME = "apptainer"

# Runtimes that need a container daemon and node-local state on the host, unlike
# Apptainer (daemonless) and Modal (remote).
DAEMON_RUNTIMES = ("docker",)


def available() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def get_runtime(name: str, **options: Any) -> Runtime:
    factory = _FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"unknown runtime {name!r}; expected one of {available()}")
    return factory(**options)
