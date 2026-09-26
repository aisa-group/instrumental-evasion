"""Start the hook with isolated imports and a deadline below the CLI timeout.

Run this trusted file with Python's -I option. It imports the hook only after
installing the deadline and adding the trusted package root. No task directory
or user site directory belongs on the import path.
"""

from __future__ import annotations

import os
import signal
import sys

HOOK_DEADLINE_SECONDS = 105
MAX_INPUT_BYTES = 1 << 20

# This fallback cannot depend on imports that may have failed. Keep its text
# equal to the default denial in monitor_core; a regression test checks it.
FALLBACK_MESSAGE = "This tool call was denied."
FALLBACK_JSON = (
    '{"hookSpecificOutput":{"hookEventName":"PreToolUse",'
    '"permissionDecision":"deny",'
    '"permissionDecisionReason":"This tool call was denied."}}\n'
)


def _deadline_expired(signum: int, frame: object) -> None:
    # BaseException prevents the HTTP retry code from swallowing the deadline.
    raise SystemExit(2)


def main() -> int:
    """Return a validated hook decision, or a stable denial on any failure."""
    try:
        if not sys.flags.isolated:
            raise RuntimeError("the hook requires isolated Python")
        signal.signal(signal.SIGALRM, _deadline_expired)
        signal.setitimer(signal.ITIMER_REAL, HOOK_DEADLINE_SECONDS)

        import contextlib
        import io
        import json
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        # Capture output until validation succeeds. Import diagnostics must not
        # become an apparent approval on the CLI's stdout channel.
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            from instrumental_evasion.hooks.pre_tool_use import main as run_hook
            from instrumental_evasion.hooks.protocol import validate_payload

            payload = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
            if len(payload) > MAX_INPUT_BYTES:
                raise ValueError("hook input exceeds its size limit")
            code = run_hook(payload.decode("utf-8"))
            rendered = output.getvalue()
            value = json.loads(rendered)
            validate_payload(value)
            reason = value["hookSpecificOutput"]["permissionDecisionReason"]
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("missing hook decision reason")
            if code not in (0, 2):
                raise ValueError("unexpected hook exit code")
            if code == 2 and value["hookSpecificOutput"]["permissionDecision"] != "deny":
                raise ValueError("conflicting hook decision")
        sys.stdout.write(rendered)
        if code == 2:
            sys.stderr.write(reason + "\n")
        return code
    except BaseException:
        os.write(1, FALLBACK_JSON.encode())
        os.write(2, (FALLBACK_MESSAGE + "\n").encode())
        return 2
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    raise SystemExit(main())
