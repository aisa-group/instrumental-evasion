"""The Docker runtime: rootless Docker, for importing benchmark images unchanged.

Chosen with `--runtime docker`. It provides the two things rootless Apptainer
cannot (see `runner.runtime`): real UID separation and a real multi-container
bridge network. That lets a task run an upstream Docker image, with its target
service in a second container on the same bridge, without rewriting it.

Two host constraints are handled in `setup()`:

  * rootless dockerd needs to mount its own /proc, which fails on hosts that
    mount lxcfs over /proc, and its state must sit on node-local disk because
    network filesystems break rootlesskit's flock. `setup()` puts
    XDG_RUNTIME_DIR and the data-root in a node-local directory.
  * the daemon needs the host's HTTP(S) proxy to pull from a registry. The
    agent's container gets the proxy only when the task's network mode grants
    egress.

The agent runs on the host and drives the container with `docker exec`, so the
runner keeps talking to its model API on the host network. Only the agent's
tool calls enter the container.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import tempfile
import uuid
from pathlib import Path
from typing import Any

from instrumental_evasion.apptainer_core import (
    NET_PROXY,
    PROXY_VARS,
    state_root,
)
from instrumental_evasion.runner.sandbox import ExecResult, run_process

_LOOPBACK = "localhost,127.0.0.1,::1"
_READY_TIMEOUT = 90  # seconds, for the daemon and for a target service


def _docker_state_dir() -> Path:
    """Node-local dir for the rootless daemon's runtime dir and data-root.

    Uses the batch scheduler's per-job scratch directory when one is set.
    """
    return Path(state_root()) / f"ie-docker-{os.getpid()}"


class _DockerMixin:
    """`exec`/`read_file`/`write_file` over a running container via the CLI."""

    name: str  # container name
    workdir: str

    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult:
        del timeout_retry
        args = ["docker", "exec", "-i", "-w", cwd or self.workdir]
        if user:
            args += ["-u", user]
        for key, value in (env or {}).items():
            args += ["-e", f"{key}={value}"]
        args += [self.name, *cmd]
        return await run_process(args, input=input, timeout=timeout, concurrency=concurrency)

    async def write_file(self, file: str, contents: str | bytes) -> None:
        path = file if file.startswith("/") else f"{self.workdir}/{file}"
        data = contents.encode() if isinstance(contents, str) else contents
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(data)
            host = handle.name
        try:
            # ensure the parent exists, then copy in
            await run_process(["docker", "exec", self.name, "mkdir", "-p", str(Path(path).parent)])
            result = await run_process(["docker", "cp", host, f"{self.name}:{path}"])
            if not result.success:
                raise RuntimeError(f"docker cp into container failed: {result.stderr.strip()}")
        finally:
            os.unlink(host)

    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        path = file if file.startswith("/") else f"{self.workdir}/{file}"
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "f"
            result = await run_process(["docker", "cp", f"{self.name}:{path}", str(dest)])
            if not result.success or not dest.exists():
                raise FileNotFoundError(file)
            if text:
                with open(dest, "r", newline="", encoding="utf-8") as handle:
                    return handle.read()
            return dest.read_bytes()


class DockerSandbox(_DockerMixin):
    """The agent's container: `docker run -d ... tail -f /dev/null`, exec'd into."""

    def __init__(
        self,
        image: str,
        workdir: str,
        network: str,
        extra_env: dict[str, str] | None,
        *,
        net_mode: str,
    ) -> None:
        self.image = image
        self.workdir = workdir
        self.network = network  # the docker network name the agent joined
        self.extra_env = extra_env or {}
        self.net_mode = net_mode
        self.name = f"ie_agent_{uuid.uuid4().hex[:12]}"

    async def start(self) -> None:
        args = ["docker", "run", "-d", "--name", self.name, "-w", self.workdir,
                "--network", self.network]
        for key, value in self.extra_env.items():
            args += ["-e", f"{key}={value}"]
        # A container we exec into needs a foreground process to stay alive;
        # tail exists in the Debian-based images we import.
        args += [self.image, "tail", "-f", "/dev/null"]
        result = await run_process(args, timeout=180)
        if not result.success:
            raise RuntimeError(f"docker run (agent) failed: {result.stderr.strip()}")

    async def verify(self) -> None:
        probe = await self.exec(["true"], cwd="/")
        if not probe.success:
            raise RuntimeError(f"container not execable: {probe.stderr.strip()}")

    def cleanup(self) -> None:
        # Fire-and-forget; the daemon (and its whole state dir) is torn down at
        # run end anyway, so a stray container cannot outlive the run.
        os.system(f"docker rm -f {self.name} >/dev/null 2>&1")
        if self.network:
            os.system(f"docker network rm {self.network} >/dev/null 2>&1")


class DockerSidecar:
    """A target service on the agent's bridge, addressed by the hostname target_env."""

    def __init__(self, name: str, address: str, control_token: str | None = None) -> None:
        self.name = name
        self.challenge_address = address
        self.control_token = control_token

    async def stop(self) -> None:
        await run_process(["docker", "rm", "-f", self.name])

    async def request_json(self, path: str) -> object:
        """Query the service from its own container, outside agent control."""
        header = (
            f"r.add_header('Authorization','Bearer {self.control_token}');"
            if self.control_token is not None
            else ""
        )
        code = (
            "import urllib.request;"
            "o=urllib.request.build_opener(urllib.request.ProxyHandler({}));"
            f"r=urllib.request.Request('http://127.0.0.1:8080{path}');"
            f"{header}"
            "print(o.open(r,timeout=10)"
            ".read().decode())"
        )
        result = await run_process(["docker", "exec", self.name, "python", "-c", code])
        if not result.success:
            raise RuntimeError(f"sidecar query failed: {result.stderr.strip()}")
        return json.loads(result.stdout)


class DockerRuntime:
    name = "docker"

    def __init__(self, **_options: Any) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._state: Path | None = None
        self._owns_daemon = False

    # ---- lifecycle -------------------------------------------------------- #
    async def setup(self) -> None:
        # Reuse a daemon an outer wrapper already started, if one answers.
        if os.environ.get("DOCKER_HOST") and await self._daemon_up():
            return
        self._state = _docker_state_dir()
        xdg = self._state / "xdg"
        data = self._state / "data"
        for d in (xdg, data):
            d.mkdir(parents=True, exist_ok=True)
        sock = xdg / "docker.sock"
        os.environ["XDG_RUNTIME_DIR"] = str(xdg)
        os.environ["DOCKER_HOST"] = f"unix://{sock}"
        # localhost must never go through the proxy; the daemon DOES need the
        # proxy to pull images from a registry.
        os.environ["no_proxy"] = _LOOPBACK + ",localnet"
        os.environ["NO_PROXY"] = os.environ["no_proxy"]
        env = dict(os.environ)
        for lower in ("http_proxy", "https_proxy"):
            if env.get(lower):
                env[lower.upper()] = env[lower]
        log = open(self._state / "dockerd.log", "wb")
        self._proc = await asyncio.create_subprocess_exec(
            "dockerd-rootless.sh", "--data-root", str(data), "--host", os.environ["DOCKER_HOST"],
            stdout=log, stderr=log, stdin=asyncio.subprocess.DEVNULL, env=env,
            start_new_session=True,
        )
        self._owns_daemon = True
        waited = 0.0
        while waited < _READY_TIMEOUT:
            if self._proc.returncode is not None:
                tail = (self._state / "dockerd.log").read_text("utf-8", "replace")[-1200:]
                raise RuntimeError(f"rootless dockerd exited early: {tail}")
            if await self._daemon_up():
                return
            await asyncio.sleep(1.5)
            waited += 1.5
        raise RuntimeError(
            "rootless dockerd did not become ready; it needs a working rootless "
            "Docker installation (dockerd-rootless.sh) and node-local state."
        )

    def teardown(self) -> None:
        if self._proc is not None and self._proc.returncode is None:
            # The process has its own session, so this stops rootlesskit,
            # dockerd and slirp without killing other processes owned by the
            # same user on the same host.
            try:
                os.killpg(self._proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if self._state is not None:
            shutil.rmtree(self._state, ignore_errors=True)

    async def _daemon_up(self) -> bool:
        result = await run_process(["docker", "version"], timeout=15)
        return result.success

    def describe(self) -> dict[str, Any]:
        return {"runtime": "docker", "runtime_rootless": True}

    # ---- per-episode ------------------------------------------------------ #
    def _network_for(self, net_mode: str) -> tuple[str, bool]:
        """(network_name, internal?) for a per-episode bridge."""
        internal = net_mode != NET_PROXY  # only the proxy mode gets egress
        return f"ie_net_{uuid.uuid4().hex[:12]}", internal

    async def open_sandbox(self, spec: Any, *, monitored: bool) -> DockerSandbox:
        net_mode = spec.network_for(monitored=monitored)
        network, internal = self._network_for(net_mode)
        create = ["docker", "network", "create"]
        if internal:
            create.append("--internal")
        create.append(network)
        result = await run_process(create, timeout=60)
        if not result.success:
            raise RuntimeError(f"docker network create failed: {result.stderr.strip()}")
        env: dict[str, str] = dict(spec.container_env or {})
        if net_mode == NET_PROXY:
            for key in PROXY_VARS:
                if os.environ.get(key):
                    env.setdefault(key, os.environ[key])
        image = getattr(spec, "docker_image", None)
        if not image:
            await run_process(["docker", "network", "rm", network])
            raise RuntimeError(
                f"task {spec.id!r} has no docker_image; it cannot run under "
                "--runtime docker (only Apptainer has its SIF)."
            )
        sandbox = DockerSandbox(image, spec.workdir, network, env, net_mode=net_mode)
        try:
            await sandbox.start()
            await sandbox.verify()
        except Exception:
            sandbox.cleanup()
            raise
        return sandbox

    async def start_sidecar(self, sidecar: Any, agent_sandbox: Any, outdir: Any) -> DockerSidecar:
        import secrets

        target_image = getattr(sidecar, "docker_image", None)
        if not target_image:
            raise RuntimeError(f"sidecar {sidecar.tb_name!r} has no docker_image")
        name = f"ie_target_{uuid.uuid4().hex[:12]}"
        args = ["docker", "run", "-d", "--name", name,
                "--network", agent_sandbox.network, "--network-alias", "target_env"]
        control_token = secrets.token_urlsafe(32) if sidecar.control_auth else None
        args.append(target_image)
        if sidecar.docker_cmd:
            args += [
                part.replace("{control_token}", control_token or "")
                for part in sidecar.docker_cmd
            ]
        result = await run_process(args, timeout=180)
        if not result.success:
            raise RuntimeError(f"docker run (target) failed: {result.stderr.strip()}")
        # Readiness: the agent's container proves it can reach target_env:port
        # over the bridge (via bash's /dev/tcp). This also validates DNS and the
        # network, not just that the target booted.
        port = sidecar.service_port
        waited = 0.0
        while waited < _READY_TIMEOUT:
            probe = await agent_sandbox.exec(
                ["bash", "-lc", f"exec 3<>/dev/tcp/target_env/{port}"], timeout=8
            )
            if probe.success:
                return DockerSidecar(name, f"target_env:{port}", control_token)
            # target died?
            alive = await run_process(["docker", "inspect", "-f", "{{.State.Running}}", name])
            if alive.stdout.strip() != "true":
                logs = await run_process(["docker", "logs", "--tail", "40", name])
                await run_process(["docker", "rm", "-f", name])
                raise RuntimeError(f"target {sidecar.tb_name} exited early: {logs.stdout[-800:]}")
            await asyncio.sleep(1.5)
            waited += 1.5
        await run_process(["docker", "rm", "-f", name])
        raise RuntimeError(f"target {sidecar.tb_name} not reachable on target_env:{port} in {_READY_TIMEOUT}s")

    async def read_static_flag(self, sidecar: Any, argv: tuple[str, ...]) -> str:
        image = getattr(sidecar, "docker_image", None)
        if not image or not argv:
            return ""
        # Bypass any image ENTRYPOINT and run argv directly.
        args = ["docker", "run", "--rm", "--entrypoint", argv[0], image, *argv[1:]]
        result = await run_process(args, timeout=120)
        return result.stdout.strip()
