"""Exercise CLI capture and cleanup with synthetic local processes, without models."""

import io
import json
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from instrumental_evasion.runner.scaffolds import cli_process
from instrumental_evasion.runner.scaffolds.cli_process import run_cli_process
from instrumental_evasion.runner.scaffolds.codex import CodexScaffold


def _run(tmp_path, code, **kwargs):
    return run_cli_process(
        [sys.executable, "-c", code],
        stdout_path=tmp_path / "stdout.txt",
        stderr_path=tmp_path / "stderr.txt",
        timeout=kwargs.pop("timeout", 5),
        **kwargs,
    )


def _assert_stopped(pid):
    status = Path(f"/proc/{pid}/stat")
    try:
        state = status.read_text().rsplit(") ", 1)[1].split()[0]
    except FileNotFoundError:
        return
    assert state in {"Z", "X", "x"}


def test_output_reaches_disk_before_process_exit_and_stdin_is_closed(tmp_path):
    release = tmp_path / "release"
    code = (
        "import sys,time,pathlib; assert sys.stdin.read() == ''; "
        "print('started', flush=True)\n"
        f"while not pathlib.Path({str(release)!r}).exists(): time.sleep(0.01)\n"
        "print('finished', flush=True)"
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run, tmp_path, code)
        try:
            deadline = time.monotonic() + 3
            output = tmp_path / "stdout.txt"
            while time.monotonic() < deadline:
                if output.exists() and b"started\n" in output.read_bytes():
                    break
                time.sleep(0.01)
            else:
                pytest.fail("CLI output was not persisted while the process ran")
            assert not future.done()
        finally:
            release.touch()
        result = future.result(timeout=5)
    assert result.returncode == 0
    assert not result.timed_out
    assert list(result.stdout_lines()) == ["started\n", "finished\n"]
    assert output.stat().st_mode & 0o777 == 0o600


def test_timeout_preserves_output_and_kills_parent_and_child(tmp_path):
    child_pid = tmp_path / "child.pid"
    parent_pid = tmp_path / "parent.pid"
    secret = "synthetic-provider-token-123456789"
    child = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(child_pid)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    code = (
        "import os,pathlib,subprocess,sys,time; "
        f"pathlib.Path({str(parent_pid)!r}).write_text(str(os.getpid())); "
        f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
        f"print({secret!r}, flush=True); print('partial error', file=sys.stderr, flush=True); "
        "time.sleep(60)"
    )
    result = _run(tmp_path, code, timeout=1, secrets=(secret,))
    assert result.timed_out
    assert result.returncode == -signal.SIGKILL
    assert "partial error" in result.stderr_tail
    assert "[REDACTED CREDENTIAL]" in result.stdout_tail
    assert secret not in (tmp_path / "stdout.txt").read_text()
    assert child_pid.exists()
    _assert_stopped(int(parent_pid.read_text()))
    _assert_stopped(int(child_pid.read_text()))


def test_timeout_does_not_wait_for_child_pipe_after_parent_exits(tmp_path):
    child = tmp_path / "child.pid"
    code = (
        "import subprocess,sys; "
        "p=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
        f"open({str(child)!r}, 'w').write(str(p.pid)); print('parent done', flush=True)"
    )
    result = _run(tmp_path, code, timeout=0.5)
    assert result.timed_out
    assert "parent done" in result.stdout_tail
    _assert_stopped(int(child.read_text()))


def test_exited_parent_does_not_skip_live_group_members(monkeypatch):
    live = iter([True, True, False])
    kills, sleeps, waits = [], [], []
    process = SimpleNamespace(pid=12345, wait=lambda timeout: waits.append(timeout))
    monkeypatch.setattr(cli_process, "_group_has_live_processes", lambda pid: next(live))
    monkeypatch.setattr(cli_process.os, "killpg", lambda pid, sig: kills.append((pid, sig)))
    monkeypatch.setattr(cli_process.time, "sleep", sleeps.append)

    cli_process._kill_group(process)

    assert len(waits) == 1
    assert kills == [(process.pid, signal.SIGKILL)] * 3
    assert sleeps == [0.01, 0.01]


def test_live_process_group_has_a_bounded_cleanup_deadline(monkeypatch):
    clock = [0.0]
    process = SimpleNamespace(pid=12345, wait=lambda timeout: None)
    monkeypatch.setattr(cli_process, "_TERMINATION_SECONDS", 0.02)
    monkeypatch.setattr(cli_process, "_group_has_live_processes", lambda pid: True)
    monkeypatch.setattr(cli_process.os, "killpg", lambda pid, sig: None)
    monkeypatch.setattr(cli_process, "time", SimpleNamespace(
        monotonic=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
    ))

    with pytest.raises(RuntimeError, match="process group did not stop"):
        cli_process._kill_group(process)

    assert clock[0] == 0.02


def test_cleanup_failure_does_not_start_a_second_cleanup_attempt(tmp_path, monkeypatch):
    kill_group = cli_process._kill_group
    attempts = []

    def fail_after_cleanup(process):
        attempts.append(process.pid)
        kill_group(process)
        raise RuntimeError("synthetic cleanup failure")

    monkeypatch.setattr(cli_process, "_kill_group", fail_after_cleanup)
    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        _run(tmp_path, "import time; time.sleep(60)", timeout=0.1)
    assert len(attempts) == 1


@pytest.mark.parametrize("state, live", [("R", True), ("S", True), ("D", True), ("Z", False), ("X", False)])
def test_group_status_checks_members_and_ignores_zombies(tmp_path, state, live):
    member = tmp_path / "123"
    member.mkdir()
    (member / "stat").write_text(f"123 (name with ) parentheses) {state} 1 456 456")
    unrelated = tmp_path / "789"
    unrelated.mkdir()
    (unrelated / "stat").write_text("789 (unrelated) R 1 789 789")

    assert cli_process._group_has_live_processes(456, tmp_path) is live


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_capture_error_or_interruption_stops_the_invocation(tmp_path, monkeypatch, error_type):
    pid_path = tmp_path / "pid"
    code = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(os.getpid())); "
        "print('ready', flush=True); time.sleep(60)"
    )
    write = cli_process._Capture.write

    def fail_on_output(self, chunk, **kwargs):
        if chunk:
            raise error_type("synthetic capture failure")
        return write(self, chunk, **kwargs)

    monkeypatch.setattr(cli_process._Capture, "write", fail_on_output)
    with pytest.raises(error_type, match="synthetic capture failure"):
        _run(tmp_path, code)
    _assert_stopped(int(pid_path.read_text()))


@pytest.mark.parametrize("split", range(1, 55))
def test_redaction_covers_chunk_boundaries(split):
    secret = b"synthetic-provider-token-123456789"
    data = b"prefix " + secret + b" suffix " + secret
    output = io.BytesIO()
    capture = cli_process._Capture(output, (secret,))
    capture.write(data[:split])
    capture.write(data[split:])
    capture.write(b"", final=True)
    assert output.getvalue() == data.replace(secret, b"[REDACTED CREDENTIAL]")
    assert secret not in capture.tail


def test_invocation_offsets_do_not_include_previous_or_later_output(tmp_path):
    first = _run(tmp_path, "print('first')")
    second = _run(tmp_path, "print('second')")
    assert list(first.stdout_lines()) == ["first\n"]
    assert list(second.stdout_lines()) == ["second\n"]


def test_truncated_capture_raises_instead_of_waiting_at_eof(tmp_path):
    result = _run(tmp_path, "print('captured output')")
    (tmp_path / "stdout.txt").write_text("")
    with pytest.raises(OSError, match="before its recorded size"):
        list(result.stdout_lines())


def test_short_writes_preserve_all_sanitized_output():
    class ShortWriter(io.BytesIO):
        def write(self, data):
            return super().write(data[:3])

    output = ShortWriter()
    capture = cli_process._Capture(output, (b"synthetic-token",))
    capture.write(b"before synthetic-token after", final=True)
    assert output.getvalue() == b"before [REDACTED CREDENTIAL] after"


def test_large_event_is_retained_but_cannot_be_silently_accepted(tmp_path, monkeypatch):
    monkeypatch.setattr(cli_process, "_EVENT_BYTES", 256)
    events = [
        {"type": "thread.started", "thread_id": "synthetic-thread"},
        {"type": "item.completed", "item": {"type": "command_execution", "output": "x" * 1024}},
        {"type": "turn.completed"},
    ]
    payload = "\n".join(json.dumps(e) for e in events)
    result = _run(tmp_path, f"print({payload!r})")
    assert payload in (tmp_path / "stdout.txt").read_text()
    observation = CodexScaffold._invocation_from_lines(result.stdout_lines())
    assert observation.malformed
    assert not observation.valid


def test_codex_preserves_unicode_separators_inside_json_strings():
    text = "Progress before\u2028and after\u2029the separators."
    events = [
        {"type": "thread.started", "thread_id": "synthetic-thread"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": text}},
        {"type": "turn.completed"},
    ]
    observation = CodexScaffold._invocation_from_stdout(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events)
    )
    assert observation.valid
    assert observation.assistant_texts == (text,)


def test_staged_and_environment_credentials_are_included_for_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "synthetic-openrouter-key")
    (tmp_path / "auth.json").write_text(json.dumps({
        "tokens": {"access_token": 'synthetic-token-with-"-quotes'},
    }))
    secrets = CodexScaffold._capture_secrets(tmp_path, {"RUNTIME_TOKEN": "synthetic-gate-token"})
    assert "synthetic-openrouter-key" in secrets
    assert "synthetic-gate-token" in secrets
    assert 'synthetic-token-with-"-quotes' in secrets
    assert 'synthetic-token-with-\\"-quotes' in secrets
