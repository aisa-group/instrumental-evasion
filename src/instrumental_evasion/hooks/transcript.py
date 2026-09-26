"""Recover the assistant message that accompanied a tool call.

The hook input does not carry that message -- both CLIs pass only
`transcript_path` -- so when the monitor is to see it
(HOOK_MONITOR_INCLUDE_MESSAGE=1) it is read back out of the transcript. The
module also recovers a Codex call's working directory the same way.

Doing it by `tool_use_id` rather than "the most recent assistant turn" matters
when a model emits several tool calls in one message: the hook fires once per
call, and the naive read would attribute a later call's context to an earlier
one. Myopia is preserved by construction, because we take exactly the one
message and nothing else -- no task, no history, no earlier decisions.
"""

from __future__ import annotations

import json
import os
import stat
from collections import deque
from typing import Any, Iterator

# Transcripts grow for the whole episode and are re-read on every tool call.
# Scanning only the tail keeps that cost flat; the accompanying message is
# always within the last few entries.
MAX_TAIL_LINES = 400
MAX_CONTEXT_TAIL_BYTES = 2 * 1024 * 1024


def codex_execution_directory(
    transcript_path: str | None, tool_use_id: str | None,
    session_cwd: str | None, command: str,
) -> str:
    """Recover one exec_command directory from its matching CLI rollout entry.

    Codex 0.153.3 omits workdir from the hook payload. Read a bounded tail and
    require an exact call ID and command match. Missing or ambiguous evidence
    raises ValueError or OSError; callers must deny the call. The rollout is mutable CLI
    state, not an authenticated security boundary. No other transcript content
    is returned or disclosed to the monitor.
    """
    if not transcript_path or not tool_use_id or not isinstance(command, str):
        raise ValueError("Execution directory requires a matching CLI tool call.")
    if not session_cwd or not os.path.isabs(session_cwd):
        raise ValueError("Execution directory requires an absolute session directory.")
    descriptor = os.open(transcript_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("The CLI transcript must be a regular file.")
        offset = max(0, info.st_size - MAX_CONTEXT_TAIL_BYTES)
        stream.seek(offset)
        data = stream.read(MAX_CONTEXT_TAIL_BYTES)
    lines = data.splitlines()
    if offset:
        lines = lines[1:]
    matches = []
    for line in lines:
        try:
            entry = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(entry, dict) or entry.get("type") != "response_item":
            continue
        payload = entry.get("payload")
        if (isinstance(payload, dict) and payload.get("type") == "function_call"
                and payload.get("call_id") == tool_use_id):
            matches.append(payload)
    if len(matches) != 1 or matches[0].get("name") != "exec_command":
        raise ValueError("Execution directory has no unique supported CLI tool call.")
    arguments = json.loads(matches[0]["arguments"])
    if not isinstance(arguments, dict) or arguments.get("cmd") != command:
        raise ValueError("CLI command and hook command do not match.")
    workdir = arguments.get("workdir")
    if workdir is None:
        workdir = session_cwd
    if (not isinstance(workdir, str) or not workdir or len(workdir) > 4096
            or any(ord(character) < 32 for character in workdir)):
        raise ValueError("CLI working directory is malformed.")
    candidate = os.path.join(session_cwd, workdir)
    directory = os.path.realpath(candidate, strict=True)
    if not os.path.isdir(directory):
        raise ValueError("CLI working directory is not a directory.")
    return directory


def _tail_lines(path: str, limit: int = MAX_TAIL_LINES) -> list[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return list(deque(handle, maxlen=limit))


def _entries(path: str) -> Iterator[dict[str, Any]]:
    for line in _tail_lines(path):
        line = line.strip()
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue  # a partially flushed final line is normal, not an error
        if isinstance(value, dict):
            yield value


def _message_of(entry: dict[str, Any]) -> dict[str, Any] | None:
    """Return the assistant message body of an entry, whichever shape it uses.

    Claude Code nests it under `message`; Codex entries have been seen both
    nested and flat, so accept either rather than silently returning no context.
    """
    message = entry.get("message")
    if isinstance(message, dict):
        role = message.get("role") or entry.get("role") or entry.get("type")
        return message if role == "assistant" else None
    if (entry.get("role") or entry.get("type")) == "assistant":
        return entry
    return None


def _blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    content = message.get("content")
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _text_of(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    parts = [
        block.get("text", "")
        for block in _blocks(message)
        if block.get("type") in {"text", "output_text"}
    ]
    return "\n".join(part for part in parts if part).strip()


def _calls_tool(message: dict[str, Any], tool_use_id: str) -> bool:
    for block in _blocks(message):
        if block.get("type") not in {"tool_use", "function_call", "tool_call"}:
            continue
        if tool_use_id in {block.get("id"), block.get("call_id"), block.get("tool_use_id")}:
            return True
    return False


def assistant_message(
    transcript_path: str | None, tool_use_id: str | None = None
) -> str | None:
    """The text of the assistant turn that issued `tool_use_id`.

    Returns None when the transcript is missing or unreadable. That is not an
    error: the monitor's contract already allows an absent message, and
    refusing every tool call because a log file was not flushed yet would be
    its own kind of broken experiment.
    """
    if not transcript_path or not os.path.exists(transcript_path):
        return None

    try:
        entries = list(_entries(transcript_path))
    except OSError:
        return None

    messages = [message for message in map(_message_of, entries) if message]

    if tool_use_id:
        for message in reversed(messages):
            if _calls_tool(message, tool_use_id):
                return _text_of(message) or None

    # No id match: either the CLI omitted tool_use_id or the transcript had not
    # been written when the hook fired. The latest assistant turn is the best
    # available answer, and is what the naive implementation would give anyway.
    for message in reversed(messages):
        text = _text_of(message)
        if text:
            return text
    return None
