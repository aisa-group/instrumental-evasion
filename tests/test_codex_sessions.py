"""Verify original session export, secret exclusion, and lifecycle coverage."""

import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from instrumental_evasion.runner.codex_sessions import SessionCapture, read_session_files
from instrumental_evasion.runner.scaffolds.base import ScaffoldResult
from instrumental_evasion.runner.scaffolds.codex import CodexScaffold


def make_session(home, *, text="Synthetic prompt", size=0):
    path = home / "sessions/2026/01/01/rollout-2026-01-01T00-00-00-test.jsonl"
    path.parent.mkdir(parents=True)
    records = [
        {"type": "session_meta", "payload": {"id": "task-thread"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": text}},
        {"type": "response_item", "payload": {"type": "reasoning", "summary": [{"text": "Synthetic summary"}], "encrypted_content": "opaque-test-data" + "x" * size}},
    ]
    path.write_bytes(b"\n".join(json.dumps(r).encode() for r in records) + b"\n")
    return path


@pytest.mark.parametrize("remote", [False, True])
def test_preserves_original_bytes_and_counts_fields(tmp_path, remote):
    home, out = tmp_path / "home", tmp_path / "out"
    path = make_session(home, size=300_000)
    (home / "auth.json").write_text('{"token":"must-not-export"}')

    class Sandbox:
        async def exec(self, command, **kwargs):
            result = subprocess.run([sys.executable, *command[1:]], capture_output=True, text=True, check=True)
            return SimpleNamespace(success=True, stdout=result.stdout)

    capture = SessionCapture(out, home, remote=remote, task_threads={"task-thread"})
    report = asyncio.run(capture.collect(Sandbox()))
    assert report["status"] == "collected"
    assert len(report["files"]) == 1
    item = report["files"][0]
    assert item["role"] == "task"
    assert item["prompt_messages"] == item["reasoning_summaries"] == item["encrypted_reasoning"] == 1
    assert (out / item["path"]).read_bytes() == path.read_bytes()
    assert (out / item["path"]).stat().st_mode & 0o777 == 0o600
    assert not list(out.rglob("auth.json"))


def test_credentials_exclude_whole_file_without_changing_source(tmp_path):
    home = tmp_path / "home"
    path = make_session(home, text="Synthetic secret=private-token")
    original = path.read_bytes()
    capture = SessionCapture(tmp_path / "out", home, secrets=("private-token",))
    report = asyncio.run(capture.collect(None))
    assert report["status"] == "incomplete"
    assert report["files"][0]["status"] == "excluded_credential"
    assert not list(capture.outdir.rglob("*.jsonl"))
    assert path.read_bytes() == original
    assert "private-token" not in (capture.outdir / "codex-sessions.json").read_text()


def test_credential_exclusion_preserves_only_safe_input_records(tmp_path):
    home = tmp_path / "home"
    path = make_session(home, text="Synthetic secret=private-token")
    safe = {"type": "response_item", "payload": {
        "type": "message", "role": "developer",
        "content": [{"type": "input_text", "text": "Guardrail reason: denied."}],
    }}
    assistant = {"type": "response_item", "payload": {
        "type": "message", "role": "assistant", "content": "A claimed delivery.",
    }}
    # A JSON escape must not hide a known credential from the subset filter.
    escaped = json.dumps(safe).replace("Guardrail reason: denied.", "private-\\u0074oken")
    with path.open("a") as stream:
        stream.write(json.dumps(safe) + "\n" + json.dumps(assistant) + "\n" + escaped + "\n")
    original = path.read_bytes()
    capture = SessionCapture(tmp_path / "out", home, secrets=("private-token",),
                             task_threads={"task-thread"})
    report = asyncio.run(capture.collect(None))
    assert report["status"] == "incomplete"
    excluded = [item for item in report["files"] if item["status"] == "excluded_credential"]
    assert len(excluded) == 1 and "path" not in excluded[0]
    subset = [item for item in report["files"] if item["status"] == "collected"]
    assert len(subset) == 1
    item = subset[0]
    assert item["role"] == "task" and item["source_lines"] == [4]
    assert item["export_kind"] == "credential_free_input_subset_v1"
    saved = (capture.outdir / item["path"]).read_bytes()
    assert [json.loads(line) for line in saved.splitlines()] == [safe]
    assert b"private-token" not in saved
    assert path.read_bytes() == original


@pytest.mark.parametrize("link_kind", ["file_symlink", "directory_symlink", "hardlink", "fifo"])
def test_rejects_unsafe_session_storage(tmp_path, link_kind):
    home = tmp_path / "home"
    path = make_session(home)
    secret = tmp_path / "secret"
    secret.write_text("private-data")
    if link_kind == "directory_symlink":
        path.parent.rename(path.parent.with_name("real"))
        path.parent.symlink_to(path.parent.with_name("real"), target_is_directory=True)
        with pytest.raises(OSError):
            read_session_files(str(home), str(path.relative_to(home)))
        return
    path.unlink()
    if link_kind == "file_symlink":
        path.symlink_to(secret)
    elif link_kind == "hardlink":
        os.link(secret, path)
    else:
        os.mkfifo(path)
    with pytest.raises((ValueError, OSError)):
        read_session_files(str(home))


@pytest.mark.parametrize("relative", ["../auth.json", "sessions/../auth.json", "/sessions/rollout-test.jsonl"])
def test_rejects_path_traversal(tmp_path, relative):
    with pytest.raises(ValueError):
        read_session_files(str(tmp_path), relative)


def test_missing_sessions_are_reported_without_fabrication(tmp_path):
    report = asyncio.run(SessionCapture(tmp_path / "out", tmp_path).collect(None))
    assert report["status"] == "missing"
    assert not report["files"]


def test_summary_request_is_explicit_and_recorded(monkeypatch):
    monkeypatch.delenv("CODEX_REASONING_SUMMARY", raising=False)
    ctx = SimpleNamespace(spec=SimpleNamespace(workdir="/workspace"), system_prompt="")
    default = CodexScaffold()
    assert not any("model_reasoning_summary" in x for x in default._run_command(ctx, "Task", None))
    monkeypatch.setenv("CODEX_REASONING_SUMMARY", "detailed")
    scaffold = CodexScaffold()
    for thread in (None, "existing-thread"):
        assert 'model_reasoning_summary="detailed"' in scaffold._run_command(ctx, "Task", thread)
    assert scaffold.describe()["agent_reasoning_summary"] == "detailed"
    with pytest.raises(ValueError, match="reasoning summary"):
        CodexScaffold(reasoning_summary="invalid")


@pytest.mark.parametrize("ending", ["submit", "desisted", "time_limit", "error", "exception", "cancel"])
def test_exports_on_each_scaffold_exit(tmp_path, monkeypatch, ending):
    home = tmp_path / "home"
    path = make_session(home)
    ctx = SimpleNamespace(outdir=tmp_path / "out", sandbox=SimpleNamespace())
    scaffold = CodexScaffold()

    async def run_local(ctx, capture):
        capture.home = home
        capture.task_threads.add("task-thread")
        if ending == "exception":
            raise RuntimeError("Synthetic failure")
        if ending == "cancel":
            raise asyncio.CancelledError()
        return ScaffoldResult(stop_reason=ending)

    monkeypatch.setattr(scaffold, "_run_local", run_local)
    if ending in {"exception", "cancel"}:
        with pytest.raises(RuntimeError if ending == "exception" else asyncio.CancelledError):
            asyncio.run(scaffold.run(ctx))
    else:
        result = asyncio.run(scaffold.run(ctx))
        assert result.stop_reason == ending
        assert result.extra["codex_session_export"]["status"] == "collected"
    manifest = json.loads((ctx.outdir / "codex-sessions.json").read_text())
    assert (ctx.outdir / manifest["files"][0]["path"]).read_bytes() == path.read_bytes()
