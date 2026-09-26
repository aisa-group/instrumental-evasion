"""Observe session usage while a local Codex invocation is running."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from instrumental_evasion.runner.codex_sessions import read_session_files
from instrumental_evasion.runner.types import ModelUsage

VERSION = "session_usage_watchdog_v1"
REVISION = "response_completion_hook_latency_v3"
TELEMETRY_GRACE_SECONDS = 30
# A Codex stdout tool-start can precede the corresponding session record while
# its next PreToolUse hook is still waiting on the monitor.  The hook itself has
# a 105-second deadline, so the ordinary 30-second response-usage grace is too
# short for this cross-stream parity check.  Keep the check fail-closed, but do
# not kill a healthy invocation before the bounded hook can return.
TOOL_RESPONSE_GRACE_SECONDS = 150


class TelemetryError(ValueError):
    """Carry a fixed diagnostic code without session content or credentials."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class SessionTokenBudget:
    """Read bounded session increments without changing the agent's prompt.

    The CLI reports usage after a response. Polling can include an in-flight
    request beyond the threshold. Session files are telemetry, not a security
    boundary against an agent that can modify its own CLI home.
    """

    def __init__(self, home: Path, limit: int, stdout_path: Path | None = None):
        self.home = str(home)
        self.limit = limit
        self.excluded = {row["path"] for row in read_session_files(self.home)["files"]}
        self.offsets: dict[str, int] = {}
        self.pending: dict[str, bytes] = {}
        self.threads: dict[str, ModelUsage] = {}
        self.error: str | None = None
        self.error_code: str | None = None
        self.io_errno: int | None = None
        self.stdout_path = stdout_path
        self.stdout_offset = stdout_path.stat().st_size if stdout_path and stdout_path.exists() else 0
        self.stdout_pending = b""
        self.thread_started_at: float | None = None
        self.pending_usage: set[str] = set()
        self.usage_due_at: dict[str, float] = {}
        self.stdout_tool_starts = 0
        self.session_tool_calls = 0
        self.missing_call_since: float | None = None

    @property
    def usage(self) -> ModelUsage:
        total = ModelUsage()
        for usage in self.threads.values():
            total = total + usage
        return total

    def poll(self) -> str | None:
        """Return a stop reason at the threshold or on invalid telemetry."""
        if self.error is not None:
            return "token_usage_error"
        try:
            self._read_stdout()
            self._read()
            now = time.monotonic()
            if self.thread_started_at is not None and not self.offsets:
                if now - self.thread_started_at >= TELEMETRY_GRACE_SECONDS:
                    raise TelemetryError("session_missing")
            if any(now - started >= TELEMETRY_GRACE_SECONDS for started in self.usage_due_at.values()):
                raise TelemetryError("completed_response_usage_missing")
            if self.stdout_tool_starts > self.session_tool_calls:
                if self.missing_call_since is None:
                    self.missing_call_since = now
                elif now - self.missing_call_since >= TOOL_RESPONSE_GRACE_SECONDS:
                    raise TelemetryError("tool_response_missing")
            else:
                self.missing_call_since = None
        except TelemetryError as error:
            return self._fail(error.code)
        except OSError as error:
            self.io_errno = error.errno
            return self._fail("session_io_error")
        except (ValueError, KeyError, TypeError):
            return self._fail("invalid_session_record")
        return "token_limit" if self.usage.total >= self.limit else None

    def _fail(self, code: str) -> str:
        self.error_code = code
        self.error = f"Codex token usage telemetry is unavailable or invalid ({code})."
        return "token_usage_error"

    def finish_invocation(self) -> str | None:
        """Require usage for all response items before another invocation starts."""
        reason = self.poll()
        if reason is not None:
            return reason
        if self.thread_started_at is not None and not self.threads:
            return self._fail("usage_missing_at_exit")
        if self.pending_usage:
            return self._fail("usage_missing_at_exit")
        return None

    def diagnostics(self) -> dict:
        """Return counters and fixed codes, without retaining raw session data."""
        return {"revision": REVISION, "error_code": self.error_code, "io_errno": self.io_errno,
                "limit": self.limit, "observed_tokens": self.usage.total,
                "session_files": len(self.offsets), "threads": len(self.threads),
                "pending_usage_sessions": len(self.pending_usage),
                "completed_responses_awaiting_usage": len(self.usage_due_at),
                "stdout_tool_starts": self.stdout_tool_starts,
                "session_tool_calls": self.session_tool_calls}

    def _read_stdout(self) -> None:
        if self.stdout_path is None or not self.stdout_path.exists():
            return
        with self.stdout_path.open("rb") as stream:
            stream.seek(self.stdout_offset)
            data = stream.read(256 * 1024)
        self.stdout_offset += len(data)
        lines = (self.stdout_pending + data).split(b"\n")
        self.stdout_pending = lines.pop()
        if len(self.stdout_pending) > 8 * 1024 * 1024:
            raise TelemetryError("stdout_record_too_large")
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                # A timeout can leave a partial stdout event. The invocation
                # parser owns stream validity; this reader only detects progress.
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") == "thread.started" and self.thread_started_at is None:
                self.thread_started_at = time.monotonic()
            item = event.get("item", {})
            if not isinstance(item, dict):
                continue
            if event.get("type") == "item.started" and item.get("type") in {"command_execution", "mcp_tool_call"}:
                # Compare record counts, not arrival times. Tool startup can
                # appear long after its response usage was written.
                self.stdout_tool_starts += 1

    def _read(self) -> None:
        inventory = read_session_files(self.home)["files"]
        present = {row["path"] for row in inventory}
        if self.offsets.keys() - present:
            raise TelemetryError("session_disappeared")
        for entry in inventory:
            path, size = entry["path"], entry["size"]
            if path in self.excluded:
                continue
            offset = self.offsets.get(path, 0)
            if size < offset:
                raise TelemetryError("session_truncated")
            pending = self.pending.get(path, b"")
            while offset < size:
                chunk = read_session_files(self.home, path, offset)
                data = base64.b64decode(chunk["data"], validate=True)
                if not data:
                    raise TelemetryError("session_read_stalled")
                offset += len(data)
                lines = (pending + data).split(b"\n")
                pending = lines.pop()
                if len(pending) > 8 * 1024 * 1024:
                    raise TelemetryError("session_record_too_large")
                for line in lines:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise TelemetryError("invalid_session_record")
                    payload = record.get("payload", {})
                    if record.get("type") == "response_item" and isinstance(payload, dict):
                        if payload.get("type") in {"function_call", "custom_tool_call"}:
                            self.session_tool_calls += 1
                            self.pending_usage.add(path)
                        elif payload.get("role") == "assistant":
                            self.pending_usage.add(path)
                    # Codex writes response items while the provider is still
                    # streaming. Its token_count/task_complete events follow
                    # response completion; only then can usage be overdue.
                    if (record.get("type") == "event_msg" and isinstance(payload, dict)
                            and payload.get("type") in {"token_count", "task_complete"}
                            and path in self.pending_usage):
                        self.usage_due_at.setdefault(path, time.monotonic())
                    if record.get("type") != "token_usage_record":
                        continue
                    payload = record["payload"]
                    if not isinstance(payload, dict):
                        raise TelemetryError("invalid_usage_payload")
                    thread = payload["thread_id"]
                    if not isinstance(thread, str) or not thread:
                        raise TelemetryError("invalid_usage_thread")
                    raw = payload["thread_token_usage"]
                    if not isinstance(raw, dict) or not {"input_tokens", "output_tokens", "total_tokens"} <= raw.keys():
                        raise TelemetryError("usage_counters_missing")
                    values = [raw.get(key, 0) for key in (
                        "input_tokens", "output_tokens", "cached_input_tokens",
                        "cache_write_input_tokens",
                    )]
                    if not all(type(value) is int and value >= 0 for value in values):
                        raise TelemetryError("invalid_usage_counter")
                    input_tokens, output_tokens, cached, written = values
                    if type(raw["total_tokens"]) is not int or raw["total_tokens"] != input_tokens + output_tokens:
                        raise TelemetryError("inconsistent_usage_total")
                    if cached + written > input_tokens:
                        raise TelemetryError("invalid_cache_usage")
                    usage = ModelUsage(input_tokens - cached - written, output_tokens, cached, written)
                    if usage.total < self.threads.get(thread, ModelUsage()).total:
                        raise TelemetryError("usage_decreased")
                    self.threads[thread] = usage
                    self.pending_usage.discard(path)
                    self.usage_due_at.pop(path, None)
            self.offsets[path] = offset
            self.pending[path] = pending
