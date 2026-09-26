"""Bound retries to a Codex resume failure that precedes task execution."""

VERSION = "same_thread_startup_retry_v1"
ERROR = "Error: thread/resume: thread/resume failed: list_turns is not supported yet (code -32601)"
RETRY_DELAYS = (0.5, 1.0)


def retry_delay(
    *,
    thread_id: str | None,
    returncode: int,
    stderr: str,
    stdout_empty: bool,
    gate_unchanged: bool,
    interrupted: bool,
    retries: int,
    remaining_seconds: float,
) -> float | None:
    """Return a delay only when an unchanged thread can be reopened safely.

    Any stdout, guardrail activity, timeout, or watchdog stop prevents a retry.
    The caller must reuse the thread and prompt, retain each invocation's logs,
    and check the original episode deadline again after the delay.
    """
    if (
        not thread_id
        or returncode != 1
        or ERROR not in stderr.splitlines()
        or not stdout_empty
        or not gate_unchanged
        or interrupted
        or retries >= len(RETRY_DELAYS)
    ):
        return None
    delay = RETRY_DELAYS[retries]
    return delay if remaining_seconds > delay + 1 else None
