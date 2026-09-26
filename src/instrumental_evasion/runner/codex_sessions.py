"""Export original Codex session files before disposable storage is removed."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VERSION = "codex_session_export_v2_credential_free_inputs"


def read_session_files(home: str, relative: str | None = None, offset: int = 0) -> dict:
    """List session files or read one bounded chunk without following links.

    This function also runs through the sandbox's trusted Python interpreter.
    Keep its imports and limits local so the remote copy is self-contained.
    """
    import base64
    import os
    import re
    import stat

    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    maximum_file_bytes = 32 * 1024 * 1024
    chunk_bytes = 256 * 1024
    filename = re.compile(r"rollout-[A-Za-z0-9_.:-]+\.jsonl\Z")
    root = os.open(home, directory_flags)
    try:
        if relative is not None:
            parts = relative.split("/")
            if (len(parts) > 8 or parts[0] not in {"sessions", "archived_sessions"}
                    or any(part in {"", ".", ".."} for part in parts)
                    or not filename.fullmatch(parts[-1]) or offset < 0):
                raise ValueError("Invalid session path or offset.")
            parent = os.dup(root)
            try:
                for part in parts[:-1]:
                    child = os.open(part, directory_flags, dir_fd=parent)
                    os.close(parent)
                    parent = child
                descriptor = os.open(parts[-1], file_flags, dir_fd=parent)
                with os.fdopen(descriptor, "rb") as stream:
                    info = os.fstat(stream.fileno())
                    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                            or info.st_size > maximum_file_bytes):
                        raise ValueError("Session is not a bounded private regular file.")
                    stream.seek(offset)
                    data = stream.read(chunk_bytes)
                    return {"size": info.st_size, "data": base64.b64encode(data).decode("ascii")}
            finally:
                os.close(parent)

        files = []
        entries_seen = 0
        total_bytes = 0

        def visit(descriptor: int, prefix: str, depth: int) -> None:
            nonlocal entries_seen, total_bytes
            if depth > 6:
                raise ValueError("Session directory is too deep.")
            for name in sorted(os.listdir(descriptor)):
                entries_seen += 1
                if entries_seen > 4096:
                    raise ValueError("Session directory has too many entries.")
                info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                path = prefix + "/" + name
                if stat.S_ISDIR(info.st_mode):
                    child = os.open(name, directory_flags, dir_fd=descriptor)
                    try:
                        visit(child, path, depth + 1)
                    finally:
                        os.close(child)
                elif filename.fullmatch(name):
                    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                            or info.st_size > maximum_file_bytes):
                        raise ValueError("Session is not a bounded private regular file.")
                    total_bytes += info.st_size
                    if len(files) >= 64 or total_bytes > 128 * 1024 * 1024:
                        raise ValueError("Session export exceeds its size limit.")
                    files.append({"path": path, "size": info.st_size})

        for name in ("sessions", "archived_sessions"):
            try:
                descriptor = os.open(name, directory_flags, dir_fd=root)
            except FileNotFoundError:
                continue
            try:
                visit(descriptor, name, 0)
            finally:
                os.close(descriptor)
        return {"files": files}
    finally:
        os.close(root)


def session_summary(data: bytes) -> dict[str, Any]:
    """Count stored fields without decoding or exposing encrypted reasoning."""
    summary: dict[str, Any] = {
        "records": 0, "invalid_lines": 0, "session_id": None,
        "prompt_messages": 0, "reasoning_summaries": 0, "encrypted_reasoning": 0,
    }
    for line in data.splitlines():
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Not an object.")
            payload = record.get("payload", {})
            if not isinstance(payload, dict):
                raise ValueError("Not an object.")
        except (ValueError, UnicodeError):
            summary["invalid_lines"] += 1
            continue
        summary["records"] += 1
        if record.get("type") == "session_meta":
            summary["session_id"] = payload.get("id")
        if record.get("type") == "response_item":
            if payload.get("type") == "message" and payload.get("role") in {"user", "developer", "system"}:
                summary["prompt_messages"] += 1
            if payload.get("type") == "reasoning":
                summary["reasoning_summaries"] += bool(payload.get("summary"))
                summary["encrypted_reasoning"] += bool(payload.get("encrypted_content"))
    return summary


@dataclass
class SessionCapture:
    """Own one episode's export state; never retain its credential files."""

    outdir: Path
    home: Path | None = None
    remote: bool = False
    secrets: tuple[str, ...] = ()
    task_threads: set[str] = field(default_factory=set)

    async def collect(self, sandbox: Any) -> dict[str, Any]:
        """Export available files on every exit and report missing or unsafe data.

        Collection status is separate from the task score and stop reason.
        Source files remain unchanged. Files containing known credentials are
        excluded in full. Export a separate subset of input messages that do not
        contain known credentials. Never reconstruct missing or malformed data.
        """
        report: dict[str, Any] = {
            "version": VERSION, "status": "not_started", "files": [],
            "task_thread_ids": sorted(self.task_threads),
        }

        async def read(relative: str | None = None, offset: int = 0) -> dict:
            if not self.remote:
                return read_session_files(str(self.home), relative, offset)
            from instrumental_evasion.hooks.deploy import CONTAINER_PY

            script = inspect.getsource(read_session_files) + (
                "\nimport json,sys\n"
                "print(json.dumps(read_session_files(*json.loads(sys.argv[1]))))\n"
            )
            result = await sandbox.exec(
                [CONTAINER_PY, "-I", "-c", script, json.dumps([str(self.home), relative, offset])],
                cwd="/", timeout=60,
            )
            if not result.success:
                raise RuntimeError("Sandbox session read failed.")
            return json.loads(result.stdout)

        try:
            if self.home is not None:
                inventory = await read()
                credentials = tuple(secret.encode() for secret in self.secrets if secret)
                for entry in inventory["files"]:
                    relative, size = entry["path"], entry["size"]
                    parts = Path(relative).parts
                    if (not 0 <= size <= 32 * 1024 * 1024 or len(parts) > 8
                            or parts[0] not in {"sessions", "archived_sessions"}
                            or any(p in {"", ".", ".."} for p in relative.split("/"))):
                        raise ValueError("Invalid session inventory.")
                    data = bytearray()
                    while len(data) < size:
                        chunk = await read(relative, len(data))
                        decoded = base64.b64decode(chunk["data"], validate=True)
                        if chunk["size"] != size or not decoded or len(data) + len(decoded) > size:
                            raise ValueError("Session changed during collection.")
                        data.extend(decoded)
                    item = {"source": relative, "bytes": size, **session_summary(bytes(data))}
                    item["role"] = "task" if item["session_id"] in self.task_threads else "probe_or_unmatched"
                    if any(secret in data for secret in credentials):
                        item["status"] = "excluded_credential"
                        safe_lines = []
                        source_lines = []
                        for number, line in enumerate(data.splitlines(), start=1):
                            try:
                                event = json.loads(line)
                                payload = event.get("payload", {})
                                is_input = (
                                    event.get("type") == "response_item"
                                    and payload.get("type") == "message"
                                    and payload.get("role") in {"user", "developer", "system"}
                                )
                                decoded = json.dumps(event, ensure_ascii=False).encode()
                            except (ValueError, AttributeError, UnicodeError):
                                continue
                            if is_input and not any(
                                secret in line or secret in decoded for secret in credentials
                            ):
                                safe_lines.append(bytes(line))
                                source_lines.append(number)
                        if safe_lines:
                            subset = b"\n".join(safe_lines) + b"\n"
                            destination = self.outdir / "codex-sessions" / (relative + ".inputs.jsonl")
                            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                            with destination.open("xb") as stream:
                                destination.chmod(0o600)
                                stream.write(subset)
                            report["files"].append({
                                "source": relative, "source_lines": source_lines,
                                "export_kind": "credential_free_input_subset_v1",
                                "session_id": item["session_id"], "role": item["role"],
                                "status": "collected", "bytes": len(subset),
                                "path": str(destination.relative_to(self.outdir)),
                                "sha256": hashlib.sha256(subset).hexdigest(),
                            })
                    else:
                        destination = self.outdir / "codex-sessions" / relative
                        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                        with destination.open("xb") as stream:
                            destination.chmod(0o600)
                            stream.write(data)
                        item.update(status="collected", path=str(destination.relative_to(self.outdir)),
                                    sha256=hashlib.sha256(data).hexdigest())
                    report["files"].append(item)
                report["status"] = "collected" if report["files"] else "missing"
                if any(item["status"] != "collected" for item in report["files"]):
                    report["status"] = "incomplete"
        except Exception as error:
            # Exception messages can contain agent-controlled paths or data.
            report.update(status="error", error_type=type(error).__name__)
        self.outdir.mkdir(parents=True, exist_ok=True)
        (self.outdir / "codex-sessions.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
