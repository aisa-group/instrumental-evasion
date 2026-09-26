"""Emit one diagnostic PreToolUse context marker without making a decision."""

from __future__ import annotations

import json

from instrumental_evasion.hooks.pre_tool_use import PROBE_ADDITIONAL_CONTEXT
from instrumental_evasion.hooks.protocol import HOOK_EVENT


def main() -> int:
    payload = {
        "hookSpecificOutput": {
            "hookEventName": HOOK_EVENT,
            "additionalContext": PROBE_ADDITIONAL_CONTEXT,
        }
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
