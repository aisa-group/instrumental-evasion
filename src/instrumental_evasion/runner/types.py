"""Message, tool-call and usage types for the runner.

Deliberately small and provider-neutral. Every scaffold and every model backend
convert to and from these, so a transcript written by one scaffold reads the
same as one written by another.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """One tool call proposed by the model."""

    id: str
    function: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "function": self.function, "arguments": self.arguments}


@dataclass
class ChatMessage:
    """One message in the conversation.

    `role` is one of system/user/assistant/tool. An assistant message may carry
    `tool_calls`; a tool message carries the `tool_call_id` it answers.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    # Set on a tool message whose call the gate refused, so the transcript says
    # which results are refusals without re-matching the block text.
    blocked: bool = False

    @property
    def text(self) -> str:
        return self.content

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            record["tool_calls"] = [call.as_dict() for call in self.tool_calls]
        if self.tool_call_id is not None:
            record["tool_call_id"] = self.tool_call_id
        if self.blocked:
            record["blocked"] = True
        return record


@dataclass
class ModelUsage:
    """Token usage for one generation."""

    input_tokens: int = 0
    output_tokens: int = 0
    # Cache reads dominate a long rollout, and the token limit counts them. A
    # limit that ignored them would fire far later than it appears to.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )

    def __add__(self, other: "ModelUsage") -> "ModelUsage":
        return ModelUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "total_tokens": self.total,
        }


@dataclass
class ModelOutput:
    """One generation."""

    message: ChatMessage
    usage: ModelUsage = field(default_factory=ModelUsage)
    stop_reason: str = "stop"
    model: str = ""

    @property
    def completion(self) -> str:
        return self.message.content


@dataclass
class ToolDef:
    """A tool the agent may call.

    `parameters` is a JSON Schema object. `execute` is an async callable taking
    the parsed arguments and returning the tool result as a string.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    execute: Any

    def as_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def as_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


def parse_arguments(raw: Any) -> dict[str, Any]:
    """Coerce a provider's arguments field to a dict, tolerating bad JSON.

    A model that emits malformed arguments should get a tool error it can react
    to, not take the episode down -- and the raw text has to survive into the
    decision record, because that is what the evasion markers read.
    """
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {"_unparsed": str(raw)}
    return value if isinstance(value, dict) else {"_value": value}
