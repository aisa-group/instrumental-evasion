"""Capture a local CLI invocation and stop its process group on failure."""

from __future__ import annotations

import os
import re
import selectors
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterator

CAPTURE_VERSION = "streamed_process_group_v2"
_CHUNK_BYTES = 64 * 1024
_TAIL_BYTES = 4096
_EVENT_BYTES = 8 * 1024 * 1024
_TERMINATION_SECONDS = 5.0


@dataclass(frozen=True)
class CLIProcessResult:
    """Offsets select one invocation from an append-only, sanitized stdout file."""

    returncode: int
    timed_out: bool
    seconds: float
    stdout_path: Path
    stdout_start: int
    stdout_end: int
    stdout_tail: str
    stderr_tail: str
    termination_reason: str | None = None

    def stdout_lines(self) -> Iterator[str]:
        """Read JSONL with bounded line memory; oversized events remain invalid.

        The full sanitized event stays in the file. Yield invalid JSON for an
        oversized line so a parser cannot silently accept an incomplete stream.
        """
        with self.stdout_path.open("rb") as handle:
            handle.seek(self.stdout_start)
            while handle.tell() < self.stdout_end:
                remaining = self.stdout_end - handle.tell()
                line = handle.readline(min(_EVENT_BYTES + 1, remaining))
                if not line:
                    raise OSError("Captured CLI output ended before its recorded size")
                if len(line) > _EVENT_BYTES:
                    while not line.endswith(b"\n") and handle.tell() < self.stdout_end:
                        line = handle.readline(min(_CHUNK_BYTES, self.stdout_end - handle.tell()))
                        if not line:
                            raise OSError("Captured CLI output ended before its recorded size")
                    yield "[CLI event exceeds the parser size limit]\n"
                else:
                    yield line.decode("utf-8", errors="replace")


class _Capture:
    """Redact known credentials before writing, including across read boundaries."""

    def __init__(self, handle: BinaryIO, secrets: tuple[bytes, ...]) -> None:
        self.handle = handle
        self.pattern = (
            re.compile(b"|".join(re.escape(value) for value in secrets))
            if secrets else None
        )
        self.overlap = max((len(value) - 1 for value in secrets), default=0)
        self.pending = b""
        self.tail = b""

    def write(self, chunk: bytes, *, final: bool = False) -> None:
        data = self.pending + chunk
        end = len(data) if final else max(0, len(data) - self.overlap)
        if self.pattern is not None:
            for match in self.pattern.finditer(data):
                if match.start() < end < match.end():
                    end = match.start()
                    break
            safe = self.pattern.sub(b"[REDACTED CREDENTIAL]", data[:end])
        else:
            safe = data[:end]
        self.pending = data[end:]
        written = 0
        while written < len(safe):
            count = self.handle.write(safe[written:])
            if not count:
                raise OSError("Could not write captured CLI output")
            written += count
        self.tail = (self.tail + safe)[-_TAIL_BYTES:]


def _group_has_live_processes(group_id: int, proc_root: Path = Path("/proc")) -> bool:
    """Check Linux process-group members without waiting for orphan reaping."""
    for entry in proc_root.iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            # A process can exit during the scan. Other users' entries can be
            # hidden; this invocation's processes run under our host identity.
            continue
        if int(fields[2]) == group_id and fields[0] not in {"Z", "X", "x"}:
            return True
    return False


def _kill_group(process: subprocess.Popen) -> None:
    """Stop and verify the process group within one bounded cleanup deadline."""
    deadline = time.monotonic() + _TERMINATION_SECONDS
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=max(0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        # TimeoutExpired includes argv, which can contain provider credentials.
        raise RuntimeError("CLI process did not exit after group termination") from None
    # Waiting for an exited parent says nothing about its orphaned children.
    # SIGKILL delivery is asynchronous, so confirm that no group member can run.
    # Zombies need their new parent to reap them; they hold no execution pipes.
    while _group_has_live_processes(process.pid):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("CLI process group did not stop before the cleanup deadline")
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        time.sleep(min(0.01, remaining))


def run_cli_process(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout: float,
    secrets: tuple[str, ...] = (),
    stop_requested: Callable[[], str | None] | None = None,
) -> CLIProcessResult:
    """Stream sanitized output to disk without buffering the whole invocation.

    The caller must supply a command that enters the configured sandbox.
    This function never constructs or executes an agent command itself.
    Timeout or interruption kills the dedicated process group. Startup and
    capture errors propagate; partial output remains available for diagnosis.
    """
    if timeout <= 0:
        raise ValueError("CLI timeout must be positive")
    secret_bytes = tuple(sorted(
        {s.encode("utf-8") for s in secrets if s}, key=len, reverse=True
    ))
    started = time.monotonic()
    timed_out = False
    termination_reason = None
    # Open the trusted output paths before starting a process. A logging failure
    # must not leave an invocation running without a reader.
    with (
        stdout_path.open("ab", buffering=0) as stdout,
        stderr_path.open("ab", buffering=0) as stderr,
    ):
        os.fchmod(stdout.fileno(), 0o600)
        os.fchmod(stderr.fileno(), 0o600)
        stdout_start = stdout.tell()
        captures = (_Capture(stdout, secret_bytes), _Capture(stderr, secret_bytes))
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None and process.stderr is not None
        cleanup_attempted = False
        try:
            with selectors.DefaultSelector() as selector:
                for pipe, capture in zip((process.stdout, process.stderr), captures, strict=True):
                    os.set_blocking(pipe.fileno(), False)
                    selector.register(pipe, selectors.EVENT_READ, capture)
                while selector.get_map() or process.poll() is None:
                    remaining = timeout - (time.monotonic() - started)
                    termination_reason = stop_requested() if stop_requested else None
                    if remaining <= 0 or termination_reason is not None:
                        timed_out = remaining <= 0
                        cleanup_attempted = True
                        _kill_group(process)
                        # Read only bytes already in the pipes. An escaped
                        # descendant must not keep timeout cleanup waiting.
                        for key in list(selector.get_map().values()):
                            for _ in range(32):
                                try:
                                    chunk = os.read(key.fd, _CHUNK_BYTES)
                                except BlockingIOError:
                                    break
                                if not chunk:
                                    break
                                key.data.write(chunk)
                        break
                    for key, _ in selector.select(min(remaining, 0.1)):
                        try:
                            chunk = os.read(key.fd, _CHUNK_BYTES)
                        except BlockingIOError:
                            continue
                        if chunk:
                            key.data.write(chunk)
                        else:
                            selector.unregister(key.fileobj)
                process.wait()
        except BaseException:
            # A cleanup failure already consumed its bounded deadline. Preserve
            # that failure instead of starting a second termination attempt.
            if not cleanup_attempted:
                _kill_group(process)
            raise
        finally:
            process.stdout.close()
            process.stderr.close()
            for capture in captures:
                capture.write(b"", final=True)
            # Keep consecutive invocations separate even if one ends mid-event.
            stdout_end = stdout.tell()
            stdout.write(b"\n")
        return CLIProcessResult(
            returncode=process.returncode,
            timed_out=timed_out,
            seconds=round(time.monotonic() - started, 3),
            stdout_path=stdout_path,
            stdout_start=stdout_start,
            stdout_end=stdout_end,
            stdout_tail=captures[0].tail.decode("utf-8", errors="replace"),
            stderr_tail=captures[1].tail.decode("utf-8", errors="replace"),
            termination_reason=termination_reason,
        )
