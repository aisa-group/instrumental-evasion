"""The PreToolUse hook wire format, shared by Claude Code and Codex.

Both CLIs converged on the same decision object, so one adapter drives both:

    {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                            "permissionDecision": "allow" | "deny",
                            "permissionDecisionReason": "..."}}

READ THIS BEFORE ADDING A FIELD. Codex rejects `continue`, `stopReason` and
`suppressOutput` with an unsupported-field error and then *discards the whole
hook output*. A discarded deny is not a deny -- the tool call proceeds. So a
well-meaning extra key does not degrade the block, it removes it, and the run
still looks like a clean experiment: the monitor appears to have approved the
call and the episode scores as an evasion. The emitted object is therefore
whitelisted here and asserted in tests rather than trusted to review.

Belt and braces: a deny is also signalled by exit code 2 with the reason on
stderr, which both CLIs honour *regardless* of the JSON. If the schema drifts,
the block still lands and the worst case is a duplicated signal instead of a
silent hole.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

HOOK_EVENT = "PreToolUse"

# Exit codes. Both CLIs: 0 = defer to the JSON, 2 = block regardless of it,
# anything else = non-blocking error, which means THE TOOL RUNS. An uncaught
# traceback exits 1, so every failure path in this adapter must be converted
# into a 2 or the crash silently disables the monitor.
EXIT_DEFER_TO_JSON = 0
EXIT_BLOCK = 2

ALLOW = "allow"
DENY = "deny"

# The only keys either CLI accepts inside hookSpecificOutput for this event.
REQUIRED_OUTPUT_KEYS = frozenset(
    {"hookEventName", "permissionDecision", "permissionDecisionReason"}
)
PERMITTED_OUTPUT_KEYS = REQUIRED_OUTPUT_KEYS | {"additionalContext"}

# Documented as accepted by Claude Code and rejected by Codex. Emitting one
# costs us the entire decision on Codex, so they are refused for both.
CODEX_REJECTED_KEYS = frozenset({"continue", "stopReason", "suppressOutput"})


class HookProtocolError(ValueError):
    """The payload we were about to emit is not one both CLIs will honour."""


@dataclass(frozen=True)
class HookRequest:
    """One PreToolUse invocation, normalised across the two CLIs."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_use_id: str | None = None
    transcript_path: str | None = None
    session_id: str | None = None
    cwd: str | None = None
    event: str = HOOK_EVENT
    # Codex-specific; absent on Claude Code.
    turn_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def parse(cls, payload: str | bytes) -> "HookRequest":
        """Parse the JSON the CLI wrote to our stdin.

        Raises on anything malformed. The caller turns that into a block: a hook
        that cannot read the request has not approved anything.
        """
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8", errors="replace")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError("hook input must be a JSON object")

        tool_name = value.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name:
            raise ValueError("hook input is missing tool_name")

        tool_input = value.get("tool_input")
        if tool_input is None:
            tool_input = {}
        if not isinstance(tool_input, dict):
            raise ValueError("tool_input must be an object")

        def _text(key: str) -> str | None:
            got = value.get(key)
            return got if isinstance(got, str) and got else None

        return cls(
            tool_name=tool_name,
            tool_input=tool_input,
            tool_use_id=_text("tool_use_id"),
            transcript_path=_text("transcript_path"),
            session_id=_text("session_id"),
            cwd=_text("cwd"),
            event=_text("hook_event_name") or HOOK_EVENT,
            turn_id=_text("turn_id"),
            raw=value,
        )


def decision_payload(
    decision: str, reason: str, *, additional_context: str | None = None
) -> dict[str, Any]:
    """Build the minimal object both CLIs honour, and refuse anything else."""

    if decision not in {ALLOW, DENY}:
        raise HookProtocolError(f"permissionDecision must be {ALLOW!r} or {DENY!r}")
    if not isinstance(reason, str) or not reason.strip():
        # Some readers treat a blank explanation as no decision at all.
        raise HookProtocolError("permissionDecisionReason must be a non-empty string")

    payload = {
        "hookSpecificOutput": {
            "hookEventName": HOOK_EVENT,
            "permissionDecision": decision,
            "permissionDecisionReason": reason.strip(),
        }
    }
    if additional_context is not None:
        if not isinstance(additional_context, str) or not additional_context.strip():
            raise HookProtocolError("additionalContext must be a non-empty string")
        payload["hookSpecificOutput"]["additionalContext"] = additional_context.strip()
    validate_payload(payload)
    return payload


def validate_payload(payload: dict[str, Any]) -> None:
    """Assert the payload survives BOTH CLIs' parsers.

    Called on every emission, not just in tests, because the cost of being
    wrong is a silent unmonitored tool call rather than a visible error.
    """
    if set(payload) != {"hookSpecificOutput"}:
        extra = sorted(set(payload) - {"hookSpecificOutput"})
        raise HookProtocolError(
            f"top-level hook output must contain only hookSpecificOutput; got extra {extra}"
        )

    inner = payload["hookSpecificOutput"]
    if not isinstance(inner, dict):
        raise HookProtocolError("hookSpecificOutput must be an object")

    rejected = sorted(set(inner) & CODEX_REJECTED_KEYS)
    if rejected:
        raise HookProtocolError(
            f"{rejected} are rejected by Codex, which then discards the whole "
            "output and runs the tool. Never emit them."
        )

    unknown = sorted(set(inner) - PERMITTED_OUTPUT_KEYS)
    if unknown:
        raise HookProtocolError(
            f"unknown hookSpecificOutput keys {unknown}; only "
            f"{sorted(PERMITTED_OUTPUT_KEYS)} are honoured by both CLIs"
        )

    missing = sorted(REQUIRED_OUTPUT_KEYS - set(inner))
    if missing:
        raise HookProtocolError(f"hookSpecificOutput is missing {missing}")

    if inner["hookEventName"] != HOOK_EVENT:
        raise HookProtocolError(f"hookEventName must be {HOOK_EVENT!r}")
    if inner["permissionDecision"] not in {ALLOW, DENY}:
        raise HookProtocolError("permissionDecision must be 'allow' or 'deny'")


def render(
    decision: str, reason: str, *, additional_context: str | None = None
) -> tuple[str, int]:
    """Return the (stdout JSON, exit code) pair for a decision.

    A deny goes out on both channels: the JSON says deny AND the exit code is 2,
    which blocks even if the JSON is discarded. An allow only needs the JSON --
    there is no exit code that means "approve regardless", and there should not
    be one.
    """
    payload = decision_payload(
        decision, reason, additional_context=additional_context
    )
    code = EXIT_BLOCK if decision == DENY else EXIT_DEFER_TO_JSON
    return json.dumps(payload), code
