"""Audit diagnostics preserve exclusion while distinguishing parser failures."""
import json

import pytest

from instrumental_evasion.runner import log as native_log
from instrumental_evasion.runner import run


def write_trace(path, *, error=False, content="done"):
    events = [
        {"type": "assistant", "message": {"content": [{
            "type": "tool_use", "id": "call1", "name": "Bash",
            "input": {"command": "echo checked"},
        }]}},
        {"type": "user", "message": {"content": [{
            "type": "tool_result", "tool_use_id": "call1",
            "is_error": error, "content": content,
        }]}},
    ]
    (path / "stdout.txt").write_text("\n".join(json.dumps(e) for e in events))


@pytest.mark.parametrize("case, expected", [
    ("unmatched", "execution_without_decision"),
    ("denied_without_decision", "denial_without_decision"),
    ("blocked_execution", "blocked_calls_with_successful_results"),
    ("malformed", "malformed_stdout_lines"),
])
def test_exclusions_remain_invalid_and_report_evidence_category(tmp_path, case, expected):
    write_trace(tmp_path, error=case == "denied_without_decision",
                content="This tool call was denied." if case == "denied_without_decision" else "done")
    decisions = []
    if case == "blocked_execution":
        decisions = [{"tool_name": "bash", "tool_input": {"command": "echo checked"}, "blocked": True}]
    if case == "malformed":
        (tmp_path / "stdout.txt").write_text("not json\n")
    record = native_log.SampleRecord(sample_id="test", epoch=0, solved=False)
    run.audit_claude_interception(record, tmp_path, decisions)
    assert record.infrastructure_status == "invalid"
    assert record.metadata["interception_audit_failure"] == {
        "kind": "evidence_exclusion", "counts": {expected: 1},
    }
    assert f"{expected}=1" in record.infrastructure_reason
    assert "ValueError" not in record.infrastructure_reason


def test_actual_audit_exception_is_distinct_and_does_not_copy_sensitive_text(tmp_path, monkeypatch):
    write_trace(tmp_path)
    def broken(*_):
        raise ValueError("private trace content")
    monkeypatch.setattr(run, "audit_claude_trace", broken)
    record = native_log.SampleRecord(sample_id="test", epoch=0, solved=False)
    run.audit_claude_interception(record, tmp_path, [])
    assert record.infrastructure_status == "invalid"
    assert record.metadata["interception_audit_failure"] == {
        "kind": "audit_exception", "error_type": "ValueError",
    }
    assert "private trace content" not in str(record)


def test_matched_trace_has_no_failure_diagnostic(tmp_path):
    write_trace(tmp_path)
    record = native_log.SampleRecord(sample_id="test", epoch=0, solved=False)
    run.audit_claude_interception(record, tmp_path, [
        {"tool_name": "bash", "tool_input": {"command": "echo checked"}, "blocked": False},
    ])
    assert not record.metadata["interception_audit"]["requires_exclusion"]
    assert "interception_audit_failure" not in record.metadata
    assert record.infrastructure_status != "invalid"
