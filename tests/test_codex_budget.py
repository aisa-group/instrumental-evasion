"""Check cumulative usage, partial records, and unsafe session telemetry."""

import json

import pytest

from instrumental_evasion.runner import codex_budget
from instrumental_evasion.runner.codex_budget import SessionTokenBudget


def record(total, thread="task"):
    return json.dumps({"type": "token_usage_record", "payload": {
        "thread_id": thread, "thread_token_usage": {
            "input_tokens": total - 10, "output_tokens": 10, "total_tokens": total,
            "cached_input_tokens": 20, "cache_write_input_tokens": 5,
        },
    }}) + "\n"


def test_probe_excluded_partial_usage_counted_and_resume_not_double_counted(tmp_path):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    (sessions / "rollout-probe.jsonl").write_text(record(1000, "probe"))
    budget = SessionTokenBudget(tmp_path, 200)
    task = sessions / "rollout-task.jsonl"
    data = record(100)
    task.write_text(data[:-3])
    assert budget.poll() is None
    assert budget.usage.total == 0
    with task.open("a") as stream:
        stream.write(data[-3:])
    assert budget.poll() is None
    assert budget.usage.total == 100
    assert budget.usage.cache_read_tokens == 20
    with task.open("a") as stream:
        stream.write(record(150) + record(201))
    assert budget.poll() == "token_limit"
    assert budget.usage.total == 201
    assert budget.poll() == "token_limit"
    assert budget.usage.total == 201


@pytest.mark.parametrize("failure", ["malformed", "truncated", "missing", "symlink", "decreased"])
def test_invalid_telemetry_requests_stop(tmp_path, failure):
    (tmp_path / "sessions").mkdir()
    budget = SessionTokenBudget(tmp_path, 1000)
    path = tmp_path / "sessions/rollout-task.jsonl"
    path.write_text(record(100))
    assert budget.poll() is None
    if failure == "malformed":
        with path.open("a") as stream:
            stream.write("not JSON\n")
    elif failure == "truncated":
        path.write_text("")
    elif failure == "missing":
        path.unlink()
    elif failure == "symlink":
        path.unlink()
        path.symlink_to(tmp_path / "unrelated-private-file")
    else:
        with path.open("a") as stream:
            stream.write(record(80))
    assert budget.poll() == "token_usage_error"
    assert budget.error
    assert budget.usage.total == 100


@pytest.mark.parametrize("raw", [{}, {"input_tokens": 90, "output_tokens": 10},
                                 {"input_tokens": 90, "output_tokens": 10, "total_tokens": 999}])
def test_missing_or_inconsistent_counters_stop(tmp_path, raw):
    (tmp_path / "sessions").mkdir()
    budget = SessionTokenBudget(tmp_path, 1000)
    (tmp_path / "sessions/rollout-task.jsonl").write_text(json.dumps({
        "type": "token_usage_record", "payload": {"thread_id": "task", "thread_token_usage": raw},
    }) + "\n")
    assert budget.poll() == "token_usage_error"


@pytest.mark.parametrize("failure", ["session_missing", "usage_missing", "response_missing"])
def test_missing_telemetry_after_observed_progress_stops(tmp_path, monkeypatch, failure):
    now = [100.0]
    monkeypatch.setattr(codex_budget.time, "monotonic", lambda: now[0])
    stdout = tmp_path / "stdout.txt"
    (tmp_path / "sessions").mkdir()
    budget = SessionTokenBudget(tmp_path, 1000, stdout)
    stdout.write_text('{"type":"thread.started","thread_id":"task"}\n')
    if failure == "usage_missing":
        (tmp_path / "sessions/rollout-task.jsonl").write_text(json.dumps({
            "type": "response_item", "payload": {"type": "function_call"},
        }) + '\n{"type":"event_msg","payload":{"type":"token_count"}}\n')
    elif failure == "response_missing":
        (tmp_path / "sessions/rollout-task.jsonl").write_text('{"type":"session_meta","payload":{"id":"task"}}\n')
        with stdout.open("a") as stream:
            stream.write('{"type":"item.started","item":{"type":"command_execution"}}\n')
    assert budget.poll() is None
    grace = (
        codex_budget.TOOL_RESPONSE_GRACE_SECONDS
        if failure == "response_missing"
        else codex_budget.TELEMETRY_GRACE_SECONDS
    )
    now[0] += grace + 1
    assert budget.poll() == "token_usage_error"
    assert budget.error_code == {"session_missing": "session_missing",
                                 "usage_missing": "completed_response_usage_missing",
                                 "response_missing": "tool_response_missing"}[failure]


def test_tool_session_parity_allows_bounded_hook_latency(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(codex_budget.time, "monotonic", lambda: now[0])
    stdout = tmp_path / "stdout.txt"
    (tmp_path / "sessions").mkdir()
    budget = SessionTokenBudget(tmp_path, 1000, stdout)
    session = tmp_path / "sessions/rollout-task.jsonl"
    session.write_text('{"type":"session_meta","payload":{"id":"task"}}\n')
    stdout.write_text(
        '{"type":"thread.started","thread_id":"task"}\n'
        '{"type":"item.started","item":{"type":"command_execution"}}\n'
    )

    assert budget.poll() is None
    now[0] += codex_budget.TELEMETRY_GRACE_SECONDS + 1
    assert budget.poll() is None
    with session.open("a") as stream:
        stream.write('{"type":"response_item","payload":{"type":"function_call"}}\n')
        stream.write(record(100))
    assert budget.poll() is None
    assert budget.diagnostics()["stdout_tool_starts"] == 1
    assert budget.diagnostics()["session_tool_calls"] == 1


def test_streamed_items_do_not_start_usage_deadline(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(codex_budget.time, "monotonic", lambda: now[0])
    (tmp_path / "sessions").mkdir()
    budget = SessionTokenBudget(tmp_path, 200)
    path = tmp_path / "sessions/rollout-task.jsonl"
    path.write_text('{"type":"response_item","payload":{"type":"function_call"}}\n')
    assert budget.poll() is None
    now[0] += 120
    assert budget.poll() is None
    with path.open("a") as stream:
        stream.write(record(201))
    assert budget.poll() == "token_limit"


def test_one_session_cannot_clear_another_sessions_missing_usage(tmp_path, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(codex_budget.time, "monotonic", lambda: now[0])
    (tmp_path / "sessions").mkdir()
    budget = SessionTokenBudget(tmp_path, 1000)
    (tmp_path / "sessions/rollout-a.jsonl").write_text(
        '{"type":"response_item","payload":{"type":"function_call"}}\n'
        '{"type":"event_msg","payload":{"type":"token_count"}}\n')
    (tmp_path / "sessions/rollout-b.jsonl").write_text(record(100, "other"))
    assert budget.poll() is None
    now[0] += 31
    assert budget.poll() == "token_usage_error"
    assert budget.error_code == "completed_response_usage_missing"
    assert budget.diagnostics()["observed_tokens"] == 100


def test_invocation_exit_requires_usage_without_waiting_for_deadline(tmp_path):
    (tmp_path / "sessions").mkdir()
    budget = SessionTokenBudget(tmp_path, 1000)
    (tmp_path / "sessions/rollout-task.jsonl").write_text(
        '{"type":"response_item","payload":{"type":"message","role":"assistant"}}\n')
    assert budget.poll() is None
    assert budget.finish_invocation() == "token_usage_error"
    assert budget.error_code == "usage_missing_at_exit"
    assert budget.poll() == "token_usage_error"


def test_diagnostics_do_not_include_filesystem_error_content(tmp_path, monkeypatch):
    budget = SessionTokenBudget(tmp_path, 1000)
    def fail(*args):
        raise OSError(5, "private-provider-key-and-session-content")
    monkeypatch.setattr(codex_budget, "read_session_files", fail)
    assert budget.poll() == "token_usage_error"
    assert budget.error_code == "session_io_error"
    assert budget.diagnostics()["io_errno"] == 5
    assert "private-provider" not in json.dumps(budget.diagnostics()) + budget.error


@pytest.mark.parametrize("delay", [0.1, 2, 60])
def test_slow_tool_does_not_require_new_usage_until_another_response(tmp_path, monkeypatch, delay):
    now = [100.0]
    monkeypatch.setattr(codex_budget.time, "monotonic", lambda: now[0])
    (tmp_path / "sessions").mkdir()
    stdout = tmp_path / "stdout.txt"
    budget = SessionTokenBudget(tmp_path, 1000, stdout)
    path = tmp_path / "sessions/rollout-task.jsonl"
    path.write_text('{"type":"response_item","payload":{"type":"function_call"}}\n' + record(100))
    assert budget.poll() is None
    now[0] += delay
    stdout.write_text('{"type":"item.started","item":{"type":"command_execution"}}\n')
    assert budget.poll() is None
    now[0] += 600
    assert budget.poll() is None
