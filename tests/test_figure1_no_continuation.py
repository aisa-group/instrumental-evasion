"""Regression checks for retrospective continuation censoring."""

import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "figure1_cutoff", Path(__file__).resolve().parents[1] / "scripts/analyze_figure1_no_continuation.py"
)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def episode(tmp_path, *, completed=True, continued=False, witness=None,
            missing_users=False, unlogged_timeout=False):
    users = [] if missing_users else [{"role": "user", "content": "Complete the task."}]
    # Quoting the continuation in assistant text must not establish a cutoff.
    users.append({"role": "assistant", "content": audit.CONTINUATION})
    if continued:
        users.append({"role": "user", "content": audit.CONTINUATION})
    sample = {
        "epoch": 0, "solved": completed, "infrastructure_status": "valid",
        "messages": users, "metadata": {"rungs_used": int(continued) + int(unlogged_timeout)},
        "stop_reason": "time_limit" if unlogged_timeout else "submit",
    }
    raw = json.dumps({"attempt_id": "example", "samples": [sample]}).encode()
    (tmp_path / "result.json").write_bytes(raw)
    row = {"attempt_id": "example", "epoch": 0, "result_path": "result.json",
           "result_sha256": audit.digest(raw)}
    reviews = []
    if witness:
        lines = [json.dumps({"type": "item.completed"}), json.dumps({"type": "turn.completed"}),
                 json.dumps({"type": "item.completed"}), json.dumps({"type": "turn.completed"})]
        stdout = ("\n".join(lines) + "\n").encode()
        (tmp_path / "stdout.txt").write_bytes(stdout)
        row.update(stdout_root="source", stdout_path="stdout.txt", stdout_sha256=audit.digest(stdout))
        reviews.append({"original_positive": True, "qualifying_stdout_lines": [witness],
                        "line_sha256": {str(witness): audit.digest(lines[witness - 1].encode())}})
    return audit.analyze_episode(row, reviews, {"archive": tmp_path, "source": tmp_path})


@pytest.mark.parametrize("continued,witness,full,partial", [
    (False, None, True, True),
    (True, None, False, False),
    (True, 1, False, True),
    (True, 3, False, False),
])
def test_first_scorer_and_effect_timing(tmp_path, continued, witness, full, partial):
    result = episode(tmp_path, continued=continued, witness=witness)
    assert result["task_completed_before_continuation"] is full
    assert result["success_before_continuation"] is partial


def test_missing_negative_trace_preserves_unknown_timing(tmp_path):
    result = episode(tmp_path, completed=False, missing_users=True)
    assert result["continuation_observation"] == "missing_user_messages"
    assert result["continuation_count"] is None
    assert result["success_before_continuation"] is False


def test_missing_positive_trace_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="lacks user-message timing"):
        episode(tmp_path, completed=True, missing_users=True)


def test_unlogged_final_timeout_remains_unsuccessful(tmp_path):
    result = episode(tmp_path, completed=False, unlogged_timeout=True)
    assert result["unlogged_final_timed_out_invocation"] is True
    assert result["success_before_continuation"] is False
