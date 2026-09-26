"""Apptainer sandboxes for the runner, plus shared subprocess plumbing.

`instrumental_evasion.apptainer_core` holds the container mechanics. This module
adds an async subprocess runner with a concurrency bound and an output cap, and
wraps the Apptainer layout in the small `exec`/`read_file`/`write_file` surface
that `instrumental_evasion.tasks.tb_spec.SandboxLike` describes. Every runtime
implements that surface, so a `score_fn` runs unchanged on any of them.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

from instrumental_evasion.apptainer_core import (
    NET_BLACKHOLE,
    AppLayout,
    verify_apptainer_available,
)

# A tool result larger than this is truncated rather than dropped: an agent that
# cats a binary should get a truncated result and keep going, not lose the
# episode.
MAX_EXEC_OUTPUT_SIZE = 10 * 1024**2

# A bound on concurrent `apptainer exec` processes. Without one, a batch of
# episodes each spawning execs will exhaust the node's file descriptors and
# processes, and the failure surfaces as unexplained non-zero exits inside
# containers -- which read as task failures rather than as harness failures.
_DEFAULT_CONCURRENCY = max(2, min(32, (os.cpu_count() or 4) * 2))
_semaphore: asyncio.Semaphore | None = None
_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _process_semaphore() -> asyncio.Semaphore:
    """One semaphore per event loop, created lazily.

    Module-level construction would bind it to whatever loop happened to be
    current at import time; `asyncio.run` makes a new loop per call, and a
    semaphore from a dead loop raises on acquire.
    """
    global _semaphore, _semaphore_loop
    loop = asyncio.get_running_loop()
    if _semaphore is None or _semaphore_loop is not loop:
        _semaphore = asyncio.Semaphore(
            int(os.environ.get("NATIVE_MAX_SUBPROCESSES", "") or _DEFAULT_CONCURRENCY)
        )
        _semaphore_loop = loop
    return _semaphore


@dataclass
class ExecResult:
    """The result of one command."""

    success: bool
    returncode: int
    stdout: str
    stderr: str


def _decode(raw: bytes, limit: int | None) -> str:
    if limit is not None and len(raw) > limit:
        raw = raw[:limit]
        return raw.decode("utf-8", errors="replace") + "\n<output truncated>"
    return raw.decode("utf-8", errors="replace")


async def run_process(
    args: list[str],
    *,
    input: str | bytes | None = None,
    timeout: int | None = None,
    output_limit: int | None = MAX_EXEC_OUTPUT_SIZE,
    concurrency: bool = True,
) -> ExecResult:
    """Run a command, returning its result rather than raising on failure.

    A timeout is a RESULT, not an exception: a command the agent ran that hung is
    an ordinary thing for an agent to do, and it should get a tool result saying
    so and be able to try something else. Only the harness's own calls treat a
    timeout as fatal, and they check `returncode`.
    """

    async def _run() -> ExecResult:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE if input is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        payload = input.encode() if isinstance(input, str) else input
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(payload), timeout=timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            # Reap it, or the child is left as a zombie for the run's lifetime.
            try:
                await process.communicate()
            except Exception:  # noqa: BLE001
                pass
            return ExecResult(
                success=False,
                returncode=124,
                stdout="",
                stderr=f"command timed out after {timeout}s",
            )
        return ExecResult(
            success=process.returncode == 0,
            returncode=process.returncode or 0,
            stdout=_decode(stdout, output_limit),
            stderr=_decode(stderr, output_limit),
        )

    if not concurrency:
        return await _run()
    async with _process_semaphore():
        return await _run()


class _SandboxMixin:
    """`exec`/`read_file`/`write_file` over an Apptainer layout."""

    def _describe(self) -> str:
        raise NotImplementedError

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
        del user, timeout_retry
        return await run_process(
            self.command(cmd, cwd, env),  # type: ignore[attr-defined]
            input=input,
            timeout=timeout,
            concurrency=concurrency,
        )

    async def write_file(self, file: str, contents: str | bytes) -> None:
        host = self.host_path(file)  # type: ignore[attr-defined]
        if host is None:
            raise PermissionError(f"{self._describe()}: {file}")
        host.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(contents, str):
            host.write_text(contents, encoding="utf-8")
        else:
            host.write_bytes(contents)

    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        host = self.host_path(file)  # type: ignore[attr-defined]
        if host is None:
            raise PermissionError(f"{self._describe()}: {file}")
        if not host.exists():
            raise FileNotFoundError(file)
        if text:
            # newline="" preserves line endings.
            with open(host, "r", newline="", encoding="utf-8") as handle:
                return handle.read()
        return host.read_bytes()


class AppSandbox(_SandboxMixin, AppLayout):
    """The Apptainer layout (`AppLayout`) with the async sandbox surface."""

    def _describe(self) -> str:
        return f"this sandbox can only read or write under {self.workdir}"

    async def seed(self) -> None:
        """Copy the image's workdir into the host bind, before the first exec."""
        result = await run_process(self.seed_command())
        if not result.success:
            raise RuntimeError(
                f"failed to seed the workdir from image: {result.stderr.strip()}"
            )

    async def verify(self) -> None:
        """Prove the runtime, the image and the overlay before an episode counts.

        The overlay is not optional: tasks are graded on state outside the
        working directory (broken_python repairs site-packages), so a host whose
        kernel refuses an unprivileged overlay silently turns those tasks into
        guaranteed zeros. An episode counts only if the machinery was
        demonstrably alive, so a failed check raises instead.
        """
        verify_apptainer_available(self.image)
        result = await run_process(self.command(["true"], cwd="/"), timeout=180)
        if not result.success:
            raise RuntimeError(
                f"apptainer could not start a container from {self.image} with an "
                f"overlay: {result.stderr.strip() or 'no output'}"
            )


async def open_app_sandbox(
    image: str,
    workdir: str,
    net: str = NET_BLACKHOLE,
    extra_env: dict[str, str] | None = None,
) -> AppSandbox:
    """A verified, seeded sandbox for one episode. The caller must `cleanup()`."""
    if not Path(image).exists():
        raise FileNotFoundError(
            f"Apptainer image not found: {image}. Build it with "
            "scripts/build_images.sh"
        )
    sandbox = AppSandbox(image, workdir, net, extra_env)
    try:
        await sandbox.verify()
        await sandbox.seed()
    except BaseException:
        # Cancellation during setup must release the disposable state too.
        sandbox.cleanup()
        raise
    return sandbox
