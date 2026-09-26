"""Check the narrow eligibility rules for retrying a saved Codex thread."""

import pytest

from instrumental_evasion.runner.codex_resume import ERROR, retry_delay


@pytest.mark.parametrize("changes,expected", [
    ({}, 0.5),
    ({"retries": 1}, 1.0),
    ({"retries": 2}, None),
    ({"thread_id": None}, None),
    ({"returncode": 0}, None),
    ({"returncode": 124}, None),
    ({"stderr": "Error: connection reset"}, None),
    ({"stderr": "prefix " + ERROR}, None),
    ({"stdout_empty": False}, None),
    ({"gate_unchanged": False}, None),
    ({"interrupted": True}, None),
    ({"remaining_seconds": 1.5}, None),
    ({"remaining_seconds": -1}, None),
])
def test_only_idle_resume_startup_failures_allow_bounded_retry(changes, expected):
    values = dict(thread_id="saved-thread", returncode=1, stderr=ERROR + "\n",
                  stdout_empty=True, gate_unchanged=True, interrupted=False,
                  retries=0, remaining_seconds=30)
    assert retry_delay(**(values | changes)) == expected
