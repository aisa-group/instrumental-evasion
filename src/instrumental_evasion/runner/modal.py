"""The Modal runtime: task commands run in a remote Modal Sandbox.

The harness stays local (including the monitor and model client); only task
commands execute in a short-lived Modal Sandbox. Registry-backed tasks import
their upstream image directly. SIF-only tasks are unpacked into a deterministic
local cache and copied into a Modal image layer, so the same customized image
the Apptainer runtime uses also runs remotely. Modal credentials are consumed by
the Modal SDK on the host and are never copied into a task container.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

from instrumental_evasion.apptainer_core import NET_NONE, NET_PROXY, PROXY_VARS
from instrumental_evasion.runner.gate import LLM_CLASSIFIER
from instrumental_evasion.runner.modal_images import (
    PreparedImageManifest,
    scaffold_runtime_key,
    sidecar_image_key,
    task_image_key,
)
from instrumental_evasion.runner.sandbox import MAX_EXEC_OUTPUT_SIZE, ExecResult

_APP_NAME = "ie-sandboxes"
# A single CLI invocation can legitimately spend most of the episode's
# wall-clock limit waiting on the model backend. Modal counts that as idle even
# though the process inside the Sandbox is still alive, so a shorter lifetime
# would turn a slow episode into a void (missing sandbox).
_SANDBOX_TIMEOUT = 60 * 60 * 4
_CLOSED_DOMAIN_SENTINEL = "closed.invalid"
_SIF_CACHE_ENV = "MODAL_SIF_CACHE"
_SIF_CACHE_DEFAULT = "/tmp/ie-modal-sif"
_SIF_IGNORE = (
    "dev/**",
    "proc/**",
    "sys/**",
    "run/**",
    ".singularity.d/**",
    ".exec",
    ".run",
    ".shell",
    ".test",
    "singularity",
)
_GATE_PORT = 8787
_GATE_DECISIONS = "/root/decisions.jsonl"
_GATE_START_ATTEMPTS = 2
_GATE_READY_TIMEOUT = 30
_CLEANUP_TIMEOUT = 60
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_WRITE_CHUNK = 512 * 1024
_TASK_ROOT = "/opt/task-rootfs"


async def _terminate_sandbox(sandbox: modal.Sandbox) -> int:
    """Terminate a sandbox and verify its exit within one cleanup deadline.

    Raise on an unavailable or incomplete cleanup so the runner can record an
    infrastructure failure. A termination request alone does not prove cleanup.
    """
    async with asyncio.timeout(_CLEANUP_TIMEOUT):
        await sandbox.terminate.aio(wait=True)
        code = await sandbox.poll.aio()
    if code is None:
        raise RuntimeError("Modal sandbox termination did not reach a terminal state.")
    return code


@dataclass(frozen=True)
class ModalSandboxLimits:
    """Optional resource caps shared by task, gate, and sidecar sandboxes."""

    lifetime_seconds: int = _SANDBOX_TIMEOUT
    cpu: float | None = None
    memory_mib: int | None = None

    @classmethod
    def from_env(cls) -> "ModalSandboxLimits":
        lifetime = int(os.environ.get("MODAL_SANDBOX_LIFETIME_SECONDS", str(_SANDBOX_TIMEOUT)))
        cpu_value = os.environ.get("MODAL_SANDBOX_CPU")
        memory_value = os.environ.get("MODAL_SANDBOX_MEMORY_MIB")
        cpu = float(cpu_value) if cpu_value else None
        memory = int(memory_value) if memory_value else None
        if not 60 <= lifetime <= _SANDBOX_TIMEOUT:
            raise ValueError("Modal sandbox lifetime must be between 60 and 14400 seconds")
        if cpu is not None and not 0.125 <= cpu <= 16:
            raise ValueError("Modal sandbox CPU cap must be between 0.125 and 16")
        if memory is not None and not 128 <= memory <= 65536:
            raise ValueError("Modal sandbox memory cap must be between 128 and 65536 MiB")
        return cls(lifetime, cpu, memory)

    def create_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "timeout": self.lifetime_seconds, "idle_timeout": self.lifetime_seconds,
        }
        if self.cpu is not None:
            options["cpu"] = (self.cpu, self.cpu)
        if self.memory_mib is not None:
            options["memory"] = (self.memory_mib, self.memory_mib)
        return options


def _native_task_image(obj: Any) -> modal.Image | None:
    """Build a task image from its recipe as a Modal image, or return None.

    Copying a large Python SIF as tens of thousands of local files makes Modal's
    image builder spend minutes hashing and uploading metadata. Images that are
    only a public base plus pinned packages are therefore expressed as the
    equivalent Modal recipe. Every other image goes through the exact-SIF path
    in `_image_for`.
    """
    task_id = getattr(obj, "id", None)
    if str(task_id).startswith("tauc_tau_"):
        # The tau-bench adapter stages a pure-Python, JSON-backed CLI at
        # setup, so a slim Python base is the whole image.
        return modal.Image.from_registry("python:3.12-slim-bookworm")
    return None


def _trim(value: str) -> str:
    raw = value.encode("utf-8", "replace")
    if len(raw) <= MAX_EXEC_OUTPUT_SIZE:
        return value
    return raw[:MAX_EXEC_OUTPUT_SIZE].decode("utf-8", "replace") + "\n<output truncated>"


def _sif_offset(path: Path) -> int:
    """Return the embedded SquashFS byte offset reported by Apptainer."""
    listed = subprocess.run(
        ["apptainer", "sif", "list", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in listed.splitlines():
        if "FS (Squashfs" not in line:
            continue
        match = re.search(r"\|(\d+)-(\d+)\|FS\(Squashfs", line.replace(" ", ""))
        if match:
            return int(match.group(1))
    raise RuntimeError(f"could not find a SquashFS partition in {path}")


def _sif_rootfs(path: Path) -> Path:
    """Unpack a SIF once, keyed by the exact local file identity.

    `apptainer build --sandbox` needs a proc mount, which some hosts do not
    allow. The SIF data partition is ordinary SquashFS, so unsquashfs can
    extract it without a user namespace or privileged mount.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Modal fallback SIF not found: {path}")
    stat = path.stat()
    identity = f"{path.resolve()}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
    key = hashlib.sha256(identity).hexdigest()[:20]
    cache = Path(os.environ.get(_SIF_CACHE_ENV, _SIF_CACHE_DEFAULT)).expanduser()
    rootfs = cache / key / "rootfs"
    ready = cache / key / ".ready"
    if ready.is_file():
        return rootfs

    cache.mkdir(parents=True, exist_ok=True)
    staging = cache / f".{key}.{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        subprocess.run(
            [
                "unsquashfs",
                "-o",
                str(_sif_offset(path)),
                "-d",
                str(staging / "rootfs"),
                str(path),
            ],
            check=True,
        )
        destination = cache / key
        try:
            staging.rename(destination)
        except FileExistsError:  # another worker completed the same extraction
            shutil.rmtree(staging, ignore_errors=True)
        ready.touch(exist_ok=True)
        return rootfs
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _sif_symlink_archive(rootfs: Path) -> Path:
    """Create a cached archive that restores symlinks lost by Modal's copy."""
    archive = rootfs.parent / "symlinks.tar"
    if archive.is_file():
        return archive
    temporary = rootfs.parent / f".symlinks.{os.getpid()}.tar"
    skipped_roots = {"dev", "proc", "sys", "run", ".singularity.d"}
    copied_root_directories = {"bin", "lib", "lib64", "sbin"}
    try:
        with tarfile.open(temporary, "w") as output:
            for directory, dirnames, filenames in os.walk(rootfs, followlinks=False):
                relative_dir = Path(directory).relative_to(rootfs)
                if relative_dir.parts and relative_dir.parts[0] in skipped_roots:
                    dirnames[:] = []
                    continue
                for name in [*dirnames, *filenames]:
                    source = Path(directory) / name
                    if not source.is_symlink():
                        continue
                    relative = source.relative_to(rootfs)
                    if relative.as_posix() in _SIF_IGNORE:
                        continue
                    if (
                        len(relative.parts) == 1
                        and relative.name in copied_root_directories
                    ):
                        continue
                    stat = source.lstat()
                    member = tarfile.TarInfo(relative.as_posix())
                    member.type = tarfile.SYMTYPE
                    member.linkname = os.readlink(source)
                    member.mode = stat.st_mode & 0o7777
                    member.mtime = int(stat.st_mtime)
                    output.addfile(member)
        try:
            temporary.replace(archive)
        except FileNotFoundError:
            pass
        return archive
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class ModalImage:
    image: modal.Image
    task_root: str | None = None
    runtime_embedded: bool = False


def _image_for(obj: Any, *, prefer_sif: bool = False) -> ModalImage:
    """Resolve either registry provenance or the task's authoritative SIF."""
    registry = getattr(obj, "docker_image", None)
    if registry and not prefer_sif:
        # The runtime always supplies the process to execute. A retained OCI
        # ENTRYPOINT would receive that process as arguments instead (and can
        # turn a one-shot scorer command into a long-lived server). Clear it
        # for deterministic Sandbox commands.
        return ModalImage(modal.Image.from_registry(str(registry)).entrypoint([]))
    native = None if prefer_sif else _native_task_image(obj)
    if native is not None and not prefer_sif:
        return ModalImage(native)
    image_fn = getattr(obj, "image", None)
    if not callable(image_fn):
        raise RuntimeError(
            f"{getattr(obj, 'id', getattr(obj, 'tb_name', 'task'))!r} has neither "
            "docker_image nor a SIF image()"
        )
    rootfs = _sif_rootfs(Path(image_fn()))
    # Use the extracted SIF as the container root; Modal supplies the real
    # procfs at runtime. Running it under PRoot on another base image would mix
    # base-image executables with task libraries and produce invalid glibc
    # combinations.
    image = modal.Image.from_scratch().add_local_dir(
        rootfs, "/", copy=True, ignore=_SIF_IGNORE
    )
    # Modal omits top-level directory symlinks. Materialize Debian's merged-/usr
    # views from this same SIF so /bin/sh and its libraries exist before RUN.
    for source_name, destination in (
        ("usr/bin", "/bin"),
        ("usr/sbin", "/sbin"),
        ("usr/lib", "/lib"),
        ("usr/lib64", "/lib64"),
    ):
        source = rootfs / source_name
        if source.is_dir():
            image = image.add_local_dir(source, destination, copy=True)
    # A scratch image must contain the ELF interpreter before its first RUN.
    # Modal omits the top-level /lib64 directory symlink from a local-dir
    # layer, so install the SIF's real loader at the interpreter path.
    loader = (rootfs / "usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2").resolve()
    if not loader.is_file():
        raise RuntimeError(f"SIF ELF interpreter not found: {loader}")
    image = image.add_local_file(
        loader, "/lib64/ld-linux-x86-64.so.2", copy=True
    )
    # Modal dereferences other symlinks. Restore all of them in one bounded
    # layer rather than creating hundreds of slow builder layers.
    image = image.add_local_file(
        _sif_symlink_archive(rootfs), "/tmp/ie-symlinks.tar", copy=True
    ).run_commands(
        "tar -xpf /tmp/ie-symlinks.tar -C / && "
        "rm /tmp/ie-symlinks.tar"
    )
    image = image.run_commands(
        "mkdir -p /tmp /proc /dev /sys /run && chmod 1777 /tmp"
    )
    return ModalImage(image)


class ModalSandbox:
    def __init__(
        self,
        sandbox: modal.Sandbox,
        workdir: str,
        net_mode: str,
        app: modal.App,
        task_root: str | None = None,
        limits: ModalSandboxLimits | None = None,
        allowed_hosts: tuple[str, ...] = (),
        domain_suffixes: bool = False,
    ) -> None:
        self._sandbox = sandbox
        self._modal_app = app
        self._limits = limits or ModalSandboxLimits()
        self.task_root = task_root
        self.workdir = workdir
        self.net_mode = net_mode
        self.sidecar: ModalSidecar | None = None
        self.remote_gate: ModalRemoteGate | None = None
        self._allowed_hosts: set[str] = set(allowed_hosts)
        self._domain_suffixes = domain_suffixes

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
        inside_task: bool = True,
    ) -> ExecResult:
        del timeout_retry, concurrency
        task_user = str(user) if self.task_root and inside_task and str(user).isdigit() else None
        if user and task_user is None:
            if str(user).isdigit():
                launcher = (
                    "import os,sys; uid=int(sys.argv[1]); "
                    "os.setgroups([]); os.setgid(uid); os.setuid(uid); "
                    "os.execvpe(sys.argv[2],sys.argv[2:],os.environ)"
                )
                python = "/opt/pyrt/bin/python3"
                # Python starts before the UID drop. Ignore the agent's Python
                # environment and user site so startup cannot load its code.
                cmd = [python, "-I", "-c", launcher, str(user), *cmd]
            else:
                cmd = ["su", "-s", "/bin/sh", user, "-c", shlex.join(cmd)]

        def run() -> ExecResult:
            try:
                workdir = cwd or self.workdir
                command = list(cmd)
                remote_workdir = workdir
                if self.task_root and inside_task:
                    command = [
                        "proot",
                        "-R",
                        self.task_root,
                        "-w",
                        workdir,
                        "/bin/sh",
                        "-c",
                        'exec "$@"',
                        "sh",
                        *command,
                    ]
                    if task_user is not None:
                        # Drop privileges before PRoot starts. A setuid call
                        # inside PRoot breaks its tracing and prevents exec.
                        command = [
                            "setpriv",
                            f"--reuid={task_user}",
                            f"--regid={task_user}",
                            "--clear-groups",
                            *command,
                        ]
                    remote_workdir = "/"
                process = self._sandbox.exec(
                    *command, workdir=remote_workdir, env=env or {}, timeout=timeout
                )
                stdin_error: modal.exception.ConflictError | None = None
                try:
                    if input is not None:
                        payload = input.encode() if isinstance(input, str) else input
                        for index in range(0, len(payload), _WRITE_CHUNK):
                            process.stdin.write(payload[index : index + _WRITE_CHUNK])
                            process.stdin.drain()
                    # Some CLIs wait for stdin to close even when the prompt is
                    # in argv. Send EOF for every command to avoid that wait.
                    process.stdin.write_eof()
                    process.stdin.drain()
                except modal.exception.ConflictError as error:
                    # The process can exit before Modal accepts the next stdin
                    # write. Preserve its output and exit code for diagnosis.
                    stdin_error = error
                stdout = process.stdout.read()
                stderr = process.stderr.read()
                code = process.wait()
                if stdin_error is not None and input is not None:
                    return ExecResult(
                        False,
                        code if code != 0 else 1,
                        _trim(stdout),
                        _trim(stderr) or str(stdin_error),
                    )
                # Modal returns -1 when the per-exec timeout terminates a
                # process (rather than always raising TimeoutError). Normalize
                # both provider representations to the sandbox contract's 124
                # so scaffolds record a time limit, not a non-void CLI error.
                if code == -1:
                    return ExecResult(
                        False,
                        124,
                        _trim(stdout),
                        _trim(stderr) or f"command timed out after {timeout}s",
                    )
                return ExecResult(
                    success=code == 0,
                    returncode=code,
                    stdout=_trim(stdout),
                    stderr=_trim(stderr),
                )
            except modal.exception.TimeoutError:
                return ExecResult(False, 124, "", f"command timed out after {timeout}s")

        return await asyncio.to_thread(run)

    async def write_file(self, file: str, contents: str | bytes) -> None:
        path = file if file.startswith("/") else f"{self.workdir}/{file}"
        physical_path = f"{self.task_root}{path}" if self.task_root else path
        payload = contents.encode() if isinstance(contents, str) else contents
        parent = shlex.quote(str(physical_path.rsplit("/", 1)[0]))
        quoted = shlex.quote(physical_path)
        result = await self.exec(
            ["sh", "-c", f"mkdir -p {parent} && cat > {quoted}"],
            input=payload,
            cwd="/",
            inside_task=False,
        )
        if not result.success:
            raise RuntimeError(f"could not write {path}: {result.stderr.strip()}")

    async def stage_path(
        self, local: Path | str, remote: str, *, read_only: bool = False
    ) -> None:
        """Copy a local file/tree into the already-running remote sandbox.

        CLI packages and subscription homes cannot be baked into a shared task
        image. A tar stream preserves executable bits and symlinks, and the
        optional root-owned read-only mode gives hooks the same integrity as an
        Apptainer ``:ro`` bind.
        """
        source = Path(local)
        if not source.exists():
            raise FileNotFoundError(source)
        if source.is_file():
            mode = source.stat().st_mode & 0o777
            target = f"{self.task_root}{remote}" if self.task_root else remote
            # Stream large CLI binaries through the filesystem API. Sending
            # each chunk through stdin adds a remote drain round trip.
            await asyncio.to_thread(
                self._sandbox.filesystem.copy_from_local, source, target
            )
            result = await self.exec(
                ["chmod", oct(mode)[2:], target], cwd="/", inside_task=False
            )
        else:
            with tempfile.SpooledTemporaryFile(max_size=64 * 1024**2) as archive:
                with tarfile.open(fileobj=archive, mode="w") as tar:
                    tar.add(source, arcname=".", recursive=True)
                archive.seek(0)
                payload = archive.read()
            remote_tar = f"/tmp/.asset-{secrets.token_hex(8)}.tar"
            await self.write_file(remote_tar, payload)
            target = f"{self.task_root}{remote}" if self.task_root else remote
            physical_tar = (
                f"{self.task_root}{remote_tar}" if self.task_root else remote_tar
            )
            result = await self.exec(
                [
                    "sh", "-c",
                    f"mkdir -p {shlex.quote(target)} && "
                    f"tar -xpf {shlex.quote(physical_tar)} -C {shlex.quote(target)} && "
                    f"rm -f {shlex.quote(physical_tar)}",
                ],
                cwd="/", timeout=900, inside_task=False,
            )
        if not result.success:
            raise RuntimeError(f"could not stage {source} at {remote}: {result.stderr}")
        if read_only:
            target = f"{self.task_root}{remote}" if self.task_root else remote
            locked = await self.exec(
                ["chmod", "-R", "a-w", target],
                cwd="/",
                timeout=300,
                inside_task=False,
            )
            if not locked.success:
                raise RuntimeError(f"could not protect staged path {remote}: {locked.stderr}")

    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        """Read bounded file bytes without passing binary data through UTF-8.

        Decode only when the caller requests text. Reject oversized files
        rather than return truncated scorer evidence.
        """
        path = file if file.startswith("/") else f"{self.workdir}/{file}"
        command = ["head", "-c", str(MAX_EXEC_OUTPUT_SIZE + 1), "--", path]
        if self.task_root:
            command = ["proot", "-R", self.task_root, "-w", "/", *command]
        process = await self._sandbox.exec.aio(
            *command, workdir="/", timeout=60, text=False
        )
        contents, _stderr, code = await asyncio.gather(
            process.stdout.read.aio(), process.stderr.read.aio(), process.wait.aio()
        )
        if code in (-1, 124, 126, 127):
            raise RuntimeError(f"Modal file read failed with exit code {code}.")
        if code != 0:
            raise FileNotFoundError(file)
        if len(contents) > MAX_EXEC_OUTPUT_SIZE:
            raise ValueError("Modal file exceeds the read size limit.")
        return contents.decode("utf-8") if text else contents

    async def allow_sidecar(self, host: str) -> None:
        if self.net_mode in (NET_NONE,):
            raise RuntimeError("a sidecar cannot be attached to a no-network sandbox")
        self._allowed_hosts.add(host)
        await asyncio.to_thread(
            self._sandbox._experimental_set_outbound_network_policy,
            outbound_domain_allowlist=sorted(self._allowed_hosts),
        )

    async def allow_hosts(self, hosts: tuple[str, ...] | list[str]) -> None:
        """Add destinations without dropping an existing sidecar/gate tunnel."""
        self._allowed_hosts.update(
            f"*.{host}" if self._domain_suffixes and not host.startswith("*.") else host
            for host in hosts if host
        )
        await asyncio.to_thread(
            self._sandbox._experimental_set_outbound_network_policy,
            outbound_domain_allowlist=sorted(self._allowed_hosts),
        )

    async def prepare_user(self, *paths: str, uid: int = 1000) -> None:
        """Give the unprivileged CLI identity only its home and task workspace."""
        result = await self.exec(
            ["chown", "-R", f"{uid}:{uid}", *paths], cwd="/", timeout=600
        )
        if not result.success:
            raise RuntimeError(f"could not prepare Modal CLI uid {uid}: {result.stderr}")

    async def start_remote_gate(self, gate: Any) -> tuple[str, str]:
        """Start the hidden policy/monitor process in a separate Modal Sandbox."""
        if self.remote_gate is not None:
            return self.remote_gate.url, self.remote_gate.token
        env = {
            "PYTHONPATH": "/opt/gate/src",
            "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY", ""),
            "MONITOR_HISTORY": "1" if gate.history.enabled else "0",
            "MONITOR_HISTORY_MAX_CALLS": str(gate.history.max_calls),
            "MONITOR_HISTORY_MAX_ARG_CHARS": str(gate.history.max_arg_chars),
            "MONITOR_HISTORY_OUTCOMES": "1" if gate.history.include_outcomes else "0",
        }
        image = modal.Image.from_registry("python:3.12-slim").add_local_dir(
            _SOURCE_ROOT, "/opt/gate/src", copy=True
        )
        command = (
            "python",
            "-m",
            "instrumental_evasion.hooks.server",
            "--policy",
            gate.policy or "__control__",
            "--model",
            gate.model or "",
            "--classifier",
            gate.classifier,
            *(
                (
                    "--regex-ruleset-version",
                    gate.regex_ruleset_version,
                )
                if gate.regex_ruleset_version is not None
                else ()
            ),
            "--host",
            "0.0.0.0",
            "--port",
            str(_GATE_PORT),
            "--token",
        )
        failures: list[str] = []
        for attempt in range(1, _GATE_START_ATTEMPTS + 1):
            token = secrets.token_urlsafe(32)
            command_with_token = (*command, token, "--decisions", _GATE_DECISIONS)
            sandbox = await asyncio.to_thread(
                modal.Sandbox.create,
                *command_with_token,
                app=self._modal_app,
                image=image,
                env=env,
                **self._limits.create_options(),
                encrypted_ports=[_GATE_PORT],
                outbound_domain_allowlist=(
                    ["openrouter.ai"]
                    if gate.monitored and gate.classifier == LLM_CLASSIFIER
                    else [_CLOSED_DOMAIN_SENTINEL]
                ),
            )
            try:
                tunnels = await asyncio.to_thread(sandbox.tunnels, timeout=90)
                tunnel = tunnels[_GATE_PORT]
                remote = ModalRemoteGate(
                    sandbox=sandbox,
                    url=f"https://{tunnel.host}:{tunnel.port}",
                    token=token,
                )
                await remote.wait_ready(timeout=_GATE_READY_TIMEOUT)
                self.remote_gate = remote
                await self.allow_hosts([tunnel.host])
                return remote.url, remote.token
            except BaseException as error:
                failures.append(f"attempt {attempt}: {type(error).__name__}: {error}")
                await _terminate_sandbox(sandbox)
                if not isinstance(error, Exception):
                    raise
        raise RuntimeError("remote gate failed readiness checks; " + " | ".join(failures))

    async def sync_remote_gate(self, gate: Any) -> None:
        if self.remote_gate is None:
            return
        await self.remote_gate.sync(gate)
        self.remote_gate = None

    async def remote_gate_count(self) -> int:
        return 0 if self.remote_gate is None else len(await self.remote_gate.records())

    async def remote_gate_healthy_since(self, count: int) -> bool:
        """Return true after the gate records a non-error readiness decision."""
        if self.remote_gate is None:
            return False
        records = await self.remote_gate.records()
        new_records = records[count:]
        return bool(new_records) and all(
            record.get("decision_source") != "monitor_error" for record in new_records
        )

    async def cleanup(self) -> None:
        try:
            if self.remote_gate is not None:
                await self.remote_gate.stop()
                self.remote_gate = None
        finally:
            # A gate failure must not prevent disposal of the credential home.
            await _terminate_sandbox(self._sandbox)


class ModalRemoteGate:
    def __init__(self, sandbox: modal.Sandbox, url: str, token: str) -> None:
        self._sandbox = sandbox
        self.url = url
        self.token = token

    async def wait_ready(self, *, timeout: int) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            returncode = await asyncio.to_thread(self._sandbox.poll)
            if returncode is not None:
                diagnostics = await self.diagnostics(returncode=returncode)
                raise RuntimeError(f"gate exited before readiness; {diagnostics}")
            try:
                await asyncio.to_thread(self._health_request)
                return
            except Exception as error:  # tunnel and server may become ready separately
                last_error = error
                await asyncio.sleep(0.5)
        diagnostics = await self.diagnostics()
        raise TimeoutError(
            f"gate was not healthy within {timeout}s: {last_error}; {diagnostics}"
        )

    def _health_request(self) -> None:
        request = urllib.request.Request(
            f"{self.url}/health",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read())
        if response.status != 200 or payload != {"ok": True}:
            raise RuntimeError(f"unexpected gate health response: {response.status} {payload}")

    async def diagnostics(self, *, returncode: int | None = None) -> str:
        if returncode is None:
            returncode = await asyncio.to_thread(self._sandbox.poll)
        if returncode is None:
            return "gate process is still running"
        stdout = await asyncio.to_thread(self._sandbox.stdout.read)
        stderr = await asyncio.to_thread(self._sandbox.stderr.read)
        return (
            f"returncode={returncode}; stdout={stdout[-1000:]!r}; "
            f"stderr={stderr[-2000:]!r}"
        )

    async def sync(self, gate: Any) -> None:
        try:
            gate.import_records(await self.records())
        finally:
            await self.stop()

    async def records(self) -> list[dict[str, Any]]:
        process = await asyncio.to_thread(
            self._sandbox.exec, "cat", _GATE_DECISIONS, timeout=30
        )
        raw = await asyncio.to_thread(process.stdout.read)
        await asyncio.to_thread(process.wait)
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    async def stop(self) -> None:
        await _terminate_sandbox(self._sandbox)


class ModalSidecar:
    def __init__(
        self,
        sandbox: modal.Sandbox,
        address: str,
        control_token: str | None = None,
    ) -> None:
        self._sandbox = sandbox
        self.challenge_address = address
        self.endpoint_scheme = "https"
        self.control_token = control_token

    async def stop(self) -> None:
        await _terminate_sandbox(self._sandbox)

    async def request_json(self, path: str) -> object:
        import json
        import urllib.request

        request = urllib.request.Request(f"https://{self.challenge_address}{path}")
        if self.control_token is not None:
            request.add_header("Authorization", f"Bearer {self.control_token}")
        return await asyncio.to_thread(
            lambda: json.loads(urllib.request.urlopen(request, timeout=10).read())
        )


class ModalRuntime:
    name = "modal"

    def __init__(self, *, scaffold: Any | None = None, **_options: Any) -> None:
        self._app: modal.App | None = None
        self._scaffold = scaffold
        self._limits = ModalSandboxLimits.from_env()
        self._network_policy = os.environ.get("MODAL_NETWORK_POLICY", "legacy")
        if self._network_policy not in {"legacy", "configured"}:
            raise ValueError("MODAL_NETWORK_POLICY must be legacy or configured")
        self._prepared_images = PreparedImageManifest.from_env()

    async def setup(self) -> None:
        if not os.environ.get("MODAL_TOKEN_ID") or not os.environ.get("MODAL_TOKEN_SECRET"):
            raise RuntimeError(
                "Modal credentials are missing; set MODAL_TOKEN_ID and "
                "MODAL_TOKEN_SECRET (the runner loads the root .env)"
            )
        self._app = await asyncio.to_thread(
            modal.App.lookup, _APP_NAME, create_if_missing=True
        )

    def teardown(self) -> None:
        return None

    def describe(self) -> dict[str, Any]:
        return {
            "runtime": "modal",
            "runtime_remote": True,
            "runtime_app": _APP_NAME,
            "runtime_proxy_mode": (
                "configured_domain_allowlist" if self._network_policy == "configured"
                else "direct_egress"
            ),
            "runtime_network_policy": self._network_policy,
            "runtime_prepared_images": self._prepared_images is not None,
            "runtime_sandbox_lifetime_seconds": self._limits.lifetime_seconds,
            "runtime_sandbox_cpu_cap": self._limits.cpu,
            "runtime_sandbox_memory_mib_cap": self._limits.memory_mib,
            "runtime_cleanup": "verified_termination_v1",
            "runtime_image_manifest": (
                self._prepared_images.path.name if self._prepared_images else ""
            ),
        }

    async def _agent_image(self, spec: Any) -> ModalImage:
        """Resolve a prepared image, or build one lazily from the task spec."""
        if self._prepared_images is not None:
            scaffold_name = getattr(self._scaffold, "name", "react")
            key = task_image_key(spec, scaffold_name)
            image, task_root = self._prepared_images.resolve(key)
            return ModalImage(
                image, task_root,
                runtime_embedded=self._prepared_images.images[key].runtime_embedded,
            )
        resolved = await asyncio.to_thread(_image_for, spec)
        image = resolved.image
        if self._scaffold is not None and hasattr(self._scaffold, "modal_image"):
            image = self._scaffold.modal_image(image, root=resolved.task_root or "")
        return ModalImage(image, resolved.task_root)

    async def _target_image(self, sidecar: Any) -> modal.Image:
        """Resolve the immutable target image used by startup and scoring."""
        if self._prepared_images is not None:
            image, _ = self._prepared_images.resolve(sidecar_image_key(sidecar))
            return image
        return (await asyncio.to_thread(_image_for, sidecar)).image

    async def open_sandbox(self, spec: Any, *, monitored: bool) -> ModalSandbox:
        if self._app is None:
            raise RuntimeError("ModalRuntime.setup() was not called")
        net_mode = spec.network_for(monitored=monitored)
        configured_hosts = (
            tuple(f"*.{host}" for host in spec.task_egress_hosts_for(monitored=monitored))
            if self._network_policy == "configured" else ()
        )
        env = dict(spec.container_env or {})
        # Proxy variables point at a proxy on the local host, which Modal cannot
        # reach. The `configured` network policy enforces the task's declared
        # egress hosts with Modal's outbound domain allowlist instead; the
        # `legacy` policy gives proxy-mode tasks direct egress.
        if net_mode == NET_PROXY:
            for key in PROXY_VARS:
                env.pop(key, None)
        resolved = await self._agent_image(spec)
        image = resolved.image
        outbound_domains = (
            list(configured_hosts) or [_CLOSED_DOMAIN_SENTINEL]
            if self._network_policy == "configured"
            else ([_CLOSED_DOMAIN_SENTINEL] if net_mode != NET_PROXY else None)
        )
        sandbox = await asyncio.to_thread(
            modal.Sandbox.create,
            "sleep",
            "infinity",
            app=self._app,
            image=image,
            env=env,
            workdir=spec.workdir,
            **self._limits.create_options(),
            # A nonresolving sentinel starts effectively closed and, unlike
            # block_network (or an empty list that the SDK serializes as
            # absent), can later be extended with sidecar and gate tunnel
            # hostnames.
            outbound_domain_allowlist=outbound_domains,
        )
        ready = False
        try:
            if self._prepared_images is not None and self._scaffold is not None:
                runtime_key = scaffold_runtime_key(getattr(self._scaffold, "name", ""))
                if hasattr(self._scaffold, "modal_runtime_image") and not resolved.runtime_embedded:
                    runtime_image, _ = self._prepared_images.resolve(runtime_key)
                    mount_path = f"{resolved.task_root or ''}/opt"
                    await asyncio.to_thread(sandbox.mount_image, mount_path, runtime_image)
            if resolved.task_root:
                # Modal omits empty directories from local layers. Workdirs are
                # often intentionally empty before setup (e.g. broken-python), and
                # a chroot also needs its conventional volatile mountpoints even
                # when no files are baked under them.
                required = [
                    f"{resolved.task_root}{spec.workdir}",
                    *(
                        f"{resolved.task_root}/{name}"
                        for name in ("tmp", "proc", "dev", "sys", "run")
                    ),
                ]
                preparation = (
                    f"mkdir -p {' '.join(shlex.quote(path) for path in required)} && "
                    # Some imported SIFs link resolv.conf into /run, which is
                    # intentionally omitted from the image. Use Modal's resolver
                    # file inside the chroot; the outbound domain policy still
                    # controls which resolved destinations are reachable.
                    f"rm -f {shlex.quote(resolved.task_root + '/etc/resolv.conf')} && "
                    f"cp -L /etc/resolv.conf "
                    f"{shlex.quote(resolved.task_root + '/etc/resolv.conf')} && "
                    f"for device in 'null 1 3' 'zero 1 5' 'random 1 8' 'urandom 1 9'; do "
                    "set -- $device; path="
                    f"{shlex.quote(resolved.task_root)}/dev/$1; "
                    "test -e \"$path\" || mknod -m 666 \"$path\" c \"$2\" \"$3\"; "
                    "done"
                )
                mkdir = await asyncio.to_thread(
                    sandbox.exec, "sh", "-c", preparation, timeout=60
                )
                await asyncio.to_thread(mkdir.stdout.read)
                mkdir_stderr = await asyncio.to_thread(mkdir.stderr.read)
                mkdir_code = await asyncio.to_thread(mkdir.wait)
                if mkdir_code != 0:
                    raise RuntimeError(
                        f"could not prepare SIF chroot directories: {mkdir_stderr.strip()}"
                    )
            wrapped = ModalSandbox(
                sandbox, spec.workdir, net_mode, self._app, task_root=resolved.task_root,
                limits=self._limits,
                allowed_hosts=configured_hosts,
                domain_suffixes=self._network_policy == "configured",
            )
            probe = await wrapped.exec(["true"], cwd="/", timeout=180)
            if not probe.success:
                raise RuntimeError(f"Modal sandbox is not execable: {probe.stderr.strip()}")
            ready = True
            return wrapped
        finally:
            # Retain ownership until mounting, preparation, and exec verification finish.
            if not ready:
                await _terminate_sandbox(sandbox)

    async def start_sidecar(
        self, sidecar: Any, agent_sandbox: ModalSandbox, outdir: Any
    ) -> ModalSidecar:
        del outdir
        if self._app is None:
            raise RuntimeError("ModalRuntime.setup() was not called")
        import secrets

        port = int(sidecar.service_port)
        control_token = secrets.token_urlsafe(32) if sidecar.control_auth else None
        # Sidecar commands can contain embedded Python/JSON with ordinary
        # braces (the ToolSandbox service does). Only the explicit placeholders
        # are template syntax; str.format() would treat every other brace as a
        # replacement field.
        command = tuple(
            part.replace("{port}", str(port)).replace(
                "{control_token}", control_token or ""
            )
            for part in sidecar.start
        )
        if command:
            # Modal tunnels bind the container port, so services must listen on
            # all interfaces rather than the Apptainer localhost address.
            command = tuple(part.replace("127.0.0.1", "0.0.0.0") for part in command)
        elif sidecar.docker_cmd:
            command = tuple(
                part.replace("{control_token}", control_token or "")
                for part in sidecar.docker_cmd
            )
        target_image = await self._target_image(sidecar)
        sandbox = await asyncio.to_thread(
            modal.Sandbox.create,
            *command,
            app=self._app,
            image=target_image,
            **self._limits.create_options(),
            encrypted_ports=[port],
            **({"outbound_domain_allowlist": [_CLOSED_DOMAIN_SENTINEL]}
               if self._network_policy == "configured" else {}),
        )
        try:
            tunnels = await asyncio.to_thread(sandbox.tunnels, timeout=90)
            tunnel = tunnels[port]
            await agent_sandbox.allow_sidecar(tunnel.host)
            return ModalSidecar(
                sandbox,
                f"{tunnel.host}:{tunnel.port}",
                control_token,
            )
        except BaseException:
            await _terminate_sandbox(sandbox)
            raise

    async def read_static_flag(self, sidecar: Any, argv: tuple[str, ...]) -> str:
        if not argv:
            return ""
        if self._app is None:
            raise RuntimeError("ModalRuntime.setup() was not called")
        target_image = await self._target_image(sidecar)
        sandbox = await asyncio.to_thread(
            modal.Sandbox.create,
            *argv,
            app=self._app,
            image=target_image,
            timeout=180,
        )
        try:
            stdout = await asyncio.to_thread(sandbox.stdout.read)
            await asyncio.to_thread(sandbox.wait)
            return stdout.strip()
        finally:
            await _terminate_sandbox(sandbox)
