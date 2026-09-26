"""A task's sidecar service, held alive alongside the agent (Apptainer runtime).

Unprivileged Apptainer may lack `instance start` (it needs a /proc remount), so
the service is a long-lived *backgrounded* `apptainer exec` -- alive for
exactly one episode -- bound to a per-episode loopback port. The agent's own
sandbox is a different image sharing the host network namespace (Apptainer's
default), so it reaches the service at `127.0.0.1:<port>`. The service and
agent images stay separate.

If the sidecar never comes up, the episode is void, not a clean block: an
episode counts only if the machinery was demonstrably alive.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import secrets
import socket
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from instrumental_evasion.apptainer_core import CONTAINMENT_FLAGS
from instrumental_evasion.tasks.tb_spec import Sidecar

# Generous, to allow for slower-starting services.
READY_TIMEOUT_SECONDS = 60
_POLL_INTERVAL = 0.5


def _free_port() -> int:
    """Ask the OS for an unused TCP port, then release it for the service.

    A small TOCTOU window remains before the service binds it; acceptable here,
    and a bind clash surfaces as a readiness timeout (a void), never as a solve.
    """
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _reachable(port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex(("127.0.0.1", port)) == 0


@dataclass
class RunningSidecar:
    """A live target service and how to reach it."""

    port: int
    process: asyncio.subprocess.Process
    log_path: Path
    control_token: str | None = None

    @property
    def challenge_address(self) -> str:
        return f"localhost:{self.port}"

    async def request_json(self, path: str) -> object:
        """Read harness-only service state without entering the agent sandbox."""
        return await asyncio.to_thread(self._request_json, path)

    def _request_json(self, path: str, value: object | None = None) -> object:
        """Make one authenticated control request from trusted host code."""

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        data = None if value is None else json.dumps(value).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=data
        )
        if self.control_token is not None:
            request.add_header("Authorization", f"Bearer {self.control_token}")
        with opener.open(request, timeout=10) as response:
            return json.loads(response.read())

    def lock_setting_writes(self) -> str:
        """Lock device setting writes before a blocked shell command starts."""

        response = self._request_json("/control/lock-setting-writes", {})
        if not isinstance(response, dict) or response.get("locked") is not True:
            raise RuntimeError("sidecar did not confirm the setting-write lock")
        return "setting_writes_locked_before_tool_execution"

    async def stop(self) -> None:
        if self.process.returncode is not None:
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=10)
        except asyncio.TimeoutError:
            self.process.kill()
            with contextlib.suppress(Exception):
                await self.process.wait()


def _argv(
    sidecar: Sidecar,
    port: int,
    control_token_ref: str | None = None,
) -> list[str]:
    start = [
        part.replace("{port}", str(port)).replace(
            "{control_token}", control_token_ref or ""
        )
        for part in sidecar.start
    ]
    # --contain gives a private tmpfs /tmp; no --net, so the service shares the
    # host loopback the agent's container also sees; no overlay, since the
    # services need no writable rootfs.
    return ["apptainer", "exec", *CONTAINMENT_FLAGS, sidecar.image(), *start]


async def start_sidecar(sidecar: Sidecar, outdir: Path) -> RunningSidecar:
    """Start the target, wait until it accepts connections, or raise (void)."""
    if not Path(sidecar.image()).exists():
        raise FileNotFoundError(f"sidecar image not found: {sidecar.image()}")
    port = _free_port()
    control_token = secrets.token_urlsafe(32) if sidecar.control_auth else None
    log_path = outdir / "sidecar.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Apptainer shares the host PID namespace by default. Do not place the
    # token in argv or the environment, where an agent could inspect it through
    # /proc. Send it once through stdin, then close the pipe.
    control_token_ref = "-" if control_token is not None else None
    log = open(log_path, "wb")
    process = await asyncio.create_subprocess_exec(
        *_argv(sidecar, port, control_token_ref),
        stdout=log,
        stderr=log,
        stdin=(
            asyncio.subprocess.PIPE
            if control_token is not None
            else asyncio.subprocess.DEVNULL
        ),
    )
    if control_token is not None:
        assert process.stdin is not None
        process.stdin.write((control_token + "\n").encode())
        await process.stdin.drain()
        process.stdin.close()
    running = RunningSidecar(
        port=port,
        process=process,
        log_path=log_path,
        control_token=control_token,
    )
    waited = 0.0
    while waited < READY_TIMEOUT_SECONDS:
        if process.returncode is not None:
            await running.stop()
            tail = log_path.read_text("utf-8", "replace")[-800:]
            raise RuntimeError(
                f"sidecar {sidecar.tb_name} exited early (code {process.returncode}): {tail}"
            )
        if _reachable(port):
            return running
        await asyncio.sleep(_POLL_INTERVAL)
        waited += _POLL_INTERVAL
    await running.stop()
    tail = log_path.read_text("utf-8", "replace")[-800:]
    raise RuntimeError(
        f"sidecar {sidecar.tb_name} did not accept connections on :{port} within "
        f"{READY_TIMEOUT_SECONDS}s: {tail}"
    )
