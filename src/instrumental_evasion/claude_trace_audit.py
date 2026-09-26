"""Compare recorded Claude tool calls with trusted guardrail decisions."""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
from collections import Counter
from datetime import datetime
from typing import Any

from instrumental_evasion.hooks.pre_tool_use import canonical_tool, project_arguments


def _timestamp(value: Any) -> float | None:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            return parsed.timestamp() if parsed.tzinfo is not None else None
        except ValueError:
            return None
    if type(value) in (int, float) and math.isfinite(value):
        return float(value)
    return None


def _guardrail_denial(result: dict[str, Any]) -> bool:
    text = result["content"]
    return bool(result["is_error"]) and isinstance(text, str) and text.strip() in {
        "This tool call was denied.", "Error: This tool call was denied."
    }


def _cli_rejection(result: dict[str, Any], call: dict[str, Any]) -> bool:
    """Identify explicit CLI errors that do not establish tool execution."""
    text = result["content"]
    if result["is_error"] is not True or _guardrail_denial(result):
        # The CLI also labels actual hook denials as permission-rule errors.
        return False
    binary_rejection = False
    path = call["input"].get("file_path")
    old_text = call["input"].get("old_string")
    new_text = call["input"].get("new_string")
    edit_path = call["name"] == "Edit" and isinstance(path, str) and bool(path)
    missing_edit_text = (
        edit_path
        and isinstance(old_text, str) and bool(old_text)
        and text == (
            "<tool_use_error>String to replace not found in file.\n"
            f"String: {old_text}</tool_use_error>"
        )
    )
    unchanged_edit = (
        edit_path and isinstance(old_text, str) and isinstance(new_text, str)
        and old_text == new_text
        and text == (
            "<tool_use_error>No changes to make: old_string and new_string "
            "are exactly the same.</tool_use_error>"
        )
    )
    missing_edit_file = (
        edit_path and isinstance(old_text, str) and bool(old_text)
        and isinstance(call["cwd"], str) and call["cwd"].startswith("/")
        and text == (
            "<tool_use_error>File does not exist. Note: your current working "
            f"directory is {call['cwd']}.</tool_use_error>"
        )
    )
    # Claude wraps these schema errors before dispatching the tool or hook.
    # Match the invalid arguments as well as the exact recorded error.
    wrapped_input_validation = (
        call["name"] == "Grep" and "-a" in call["input"]
        and text == (
            "<tool_use_error>InputValidationError: Grep failed due to the following issue:\n"
            "An unexpected parameter `-a` was provided</tool_use_error>"
        )
        or call["name"] == "Read" and "file_path" not in call["input"]
        and text == (
            "<tool_use_error>InputValidationError: Read failed due to the following issue:\n"
            "The parameter `file_path` type is expected as `string` but provided as "
            "`unknown`</tool_use_error>"
        )
    )
    if call["name"] == "Read" and isinstance(path, str):
        extension = posixpath.splitext(path)[1].lower()
        # Claude Code (checked with CLI 2.1.274) rejects Read on these
        # extensions before the hook runs and before any content access, for
        # plain files and FIFOs alike. Keep this exception tied to Read, the
        # listed suffixes, and the exact error text.
        binary_rejection = extension in {".7z", ".db", ".dat", ".gz"} and text == (
            "<tool_use_error>This tool cannot read binary files. The file appears "
            f"to be a binary {extension} file. Please use appropriate tools for binary "
            "file analysis.</tool_use_error>"
        )
    return bool(result["is_error"]) and (
        binary_rejection or missing_edit_text or unchanged_edit or missing_edit_file
        or wrapped_input_validation
        or result["non_execution_kind"] in {"permission-rule", "input-validation", "tool-validation"}
        or isinstance(text, str) and (
            text.startswith("InputValidationError:")
            or text.startswith("<tool_use_error>Blocked: sleep ")
            or text.startswith("<tool_use_error>Found ") and "matches of the string to replace, but replace_all is false" in text
            or text == "<tool_use_error>File has been modified since read, either by the user or by a linter. Read it again before attempting to write it.</tool_use_error>"
        )
    )


def _signature(name: str, arguments: dict[str, Any], cwd: str, resolve_paths: bool) -> tuple[str, str]:
    tool = canonical_tool(name)
    projected, _ = project_arguments(tool, arguments)
    projected = dict(projected)
    if resolve_paths:
        for key in ("file_path", "path", "notebook_path"):
            value = projected.get(key)
            if isinstance(value, str) and value and not value.startswith("~"):
                projected[key] = posixpath.normpath(posixpath.join(cwd, value))
    return tool, json.dumps(projected, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def audit_claude_trace(stdout: str, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit tool-result evidence without treating proposals as executions.

    Consume one trusted decision for each unique tool-use ID. Normalize lexical
    file paths as the CLI does. Resolve relative paths only against the recorded
    CLI working directory. A successful unmatched result, or an execution error
    without an explicit pre-execution rejection, requires exclusion. A guardrail
    denial without a trusted decision also requires exclusion: bootstrap
    failures can cause that state. Missing results remain an explicit unknown.
    This detects observed gaps; it does not prove that unrecorded calls cannot
    exist. The function makes no provider calls and changes no artifacts.
    A decision whose arguments were truncated in the JSONL log is matched by the
    `tool_input_sha256` digest the gate records with it.
    """
    if len(stdout.encode()) > 64_000_000 or len(decisions) > 100_000:
        raise ValueError("Claude trace exceeds the audit size limit")
    calls: dict[str, dict[str, Any]] = {}
    call_times: dict[str, float | None] = {}
    results: dict[str, list[dict[str, Any]]] = {}
    cwd = "/workspace"
    malformed_lines = []
    for number, line in enumerate(stdout.split("\n"), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines.append(number)
            continue
        if not isinstance(event, dict):
            malformed_lines.append(number)
            continue
        if event.get("type") == "system" and isinstance(event.get("cwd"), str):
            cwd = event["cwd"]
        message = event.get("message", {})
        if not isinstance(message, dict):
            continue
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        non_execution = {
            item.get("id"): item.get("non_execution_kind")
            for item in event.get("tool_result_meta", [])
            if isinstance(item, dict)
        }
        for block in content:
            if not isinstance(block, dict):
                continue
            if event.get("type") == "assistant" and block.get("type") == "tool_use":
                identifier = block.get("id")
                if not isinstance(identifier, str) or not isinstance(block.get("input"), dict):
                    raise ValueError("Claude tool-use event is malformed")
                call = {"name": block.get("name", ""), "input": block["input"], "cwd": cwd}
                if identifier in calls and calls[identifier] != call:
                    raise ValueError("One Claude tool-use ID has conflicting arguments")
                calls[identifier] = call
                call_times.setdefault(identifier, _timestamp(event.get("timestamp")))
            elif event.get("type") == "user" and block.get("type") == "tool_result":
                identifier = block.get("tool_use_id")
                results.setdefault(identifier, []).append({
                    "is_error": block.get("is_error", False),
                    "non_execution_kind": non_execution.get(identifier),
                    "content": block.get("content", ""),
                    "timestamp": _timestamp(event.get("timestamp")),
                })

    unused = set(range(len(decisions)))
    unmatched = []
    blocked_executions = []
    matched = 0

    def first_result_time(item: tuple[str, dict[str, Any]]) -> float:
        times = [r["timestamp"] for r in results.get(item[0], []) if r["timestamp"] is not None]
        return min(times, default=math.inf)

    # Parallel calls can finish out of proposal order. Allocate the earliest
    # eligible decision to the call whose result arrives first. Otherwise a
    # slow call can consume the only decision inside a faster call's interval.
    for identifier, call in sorted(calls.items(), key=first_result_time):
        if canonical_tool(call["name"]) == "submit":
            continue
        tool_results = results.get(identifier, [])
        rejected_by_cli = bool(tool_results) and all(_cli_rejection(r, call) for r in tool_results)
        proposed_at = call_times[identifier]
        result_times = [r["timestamp"] for r in tool_results if r["timestamp"] is not None]

        def belongs_to_call(decision: dict[str, Any]) -> bool:
            decision_id = decision.get("tool_use_id")
            if decision_id is not None:
                # The task and remote gate can have different wall clocks.
                # An exact ID links the decision without comparing those clocks;
                # argument matching and one-time consumption still apply below.
                return decision_id == identifier
            # A CLI rejection must not consume a later retry's decision.
            # Exact IDs take precedence when a decision explicitly links a call.
            if rejected_by_cli and decision_id is None:
                return False
            decided_at = _timestamp(decision.get("ts"))
            if decided_at is not None:
                if proposed_at is not None and decided_at < proposed_at:
                    return False
                if result_times and decided_at > min(result_times):
                    return False
            return True

        candidates = []
        for resolve_paths in (False, True):
            expected = _signature(call["name"], call["input"], call["cwd"], resolve_paths)

            def arguments_match(decision: dict[str, Any]) -> bool:
                if decision["tool_input"] == {"_truncated": True}:
                    return (
                        canonical_tool(decision["tool_name"]) == expected[0]
                        and decision.get("tool_input_sha256")
                        == hashlib.sha256(expected[1].encode()).hexdigest()
                    )
                return _signature(
                    decision["tool_name"], decision["tool_input"], call["cwd"], resolve_paths
                ) == expected

            candidates = [
                index for index in sorted(unused)
                if belongs_to_call(decisions[index])
                and arguments_match(decisions[index])
            ]
            if candidates:
                break
        if candidates:
            index = candidates[0]
            unused.remove(index)
            matched += 1
            if decisions[index].get("blocked") and any(not r["is_error"] for r in tool_results):
                blocked_executions.append(identifier)
            continue
        state = "no_result"
        if tool_results:
            if rejected_by_cli:
                state = "pre_execution_rejection"
            elif all(_cli_rejection(r, call) or _guardrail_denial(r) for r in tool_results):
                state = "denial_without_decision"
            else:
                state = "execution_without_decision"
        signature = _signature(call["name"], call["input"], call["cwd"], False)
        unmatched.append({
            "tool_use_id": identifier,
            "tool": call["name"],
            "state": state,
            "arguments_sha256": hashlib.sha256(signature[1].encode()).hexdigest(),
            "result_errors": [r["is_error"] for r in tool_results],
            "non_execution_kinds": [r["non_execution_kind"] for r in tool_results],
        })
    return {
        "version": "decisions_by_id_time_v7",
        "unique_tool_calls": len(calls),
        "matched_calls": matched,
        "unmatched": unmatched,
        "unmatched_states": dict(Counter(x["state"] for x in unmatched)),
        "blocked_calls_with_successful_results": blocked_executions,
        "malformed_stdout_lines": malformed_lines,
        "unused_decision_indices": sorted(unused),
        "requires_exclusion": bool(malformed_lines or blocked_executions or any(
            x["state"] in {"execution_without_decision", "denial_without_decision"}
            for x in unmatched
        )),
    }
