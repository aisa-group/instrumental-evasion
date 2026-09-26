"""Model access for the ReAct scaffold, chosen by the provider prefix of a model string.

    openrouter/<vendor>/<model>   OpenAI-compatible HTTP against OpenRouter.
    openai/<model>                the same client against api.openai.com.
    anthropic/<model>             the Anthropic SDK, with ANTHROPIC_API_KEY.

The CLI scaffolds (Codex, Claude Code) do not use this module: the vendor CLI
talks to its backend itself, so its token usage is invisible to the runner and
wall clock bounds those episodes. The monitor has its own client
(`hooks.monitor_client`).

Retries live here rather than at the call site because a 429 in the middle of
a rollout must not end the episode: a truncated monitored episode would read
as a clean block.
"""

from __future__ import annotations

import asyncio
import os
import random
from typing import Any

from instrumental_evasion.runner.types import (
    ChatMessage,
    ModelOutput,
    ModelUsage,
    ToolCall,
    ToolDef,
    parse_arguments,
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_MAX_RETRIES = 5
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 520, 524}


class ModelUnavailable(RuntimeError):
    """The model could not be reached, or did not answer, after retries."""


def _status_of(error: Exception) -> int | None:
    return getattr(error, "status_code", None) or getattr(
        getattr(error, "response", None), "status_code", None
    )


async def _with_retries(call, *, retries: int, what: str):
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return await call()
        except Exception as error:  # noqa: BLE001 - re-raised below
            status = _status_of(error)
            last = error
            if status is not None and status not in RETRY_STATUS:
                raise ModelUnavailable(f"{what}: HTTP {status}: {error}") from error
            if attempt == retries - 1:
                break
            # Jittered backoff: a whole batch retrying in lockstep after a shared
            # rate limit just reproduces the rate limit.
            delay = min(2**attempt * 2, 60) * (0.5 + random.random())
            await asyncio.sleep(delay)
    raise ModelUnavailable(f"{what}: giving up after {retries} tries: {last}")


class Model:
    """One model, addressed as `provider/name`."""

    def __init__(self, spec: str) -> None:
        provider, _, name = spec.partition("/")
        if not name:
            raise ValueError(
                f"model {spec!r} has no provider prefix; expected "
                "'openrouter/<vendor>/<model>', 'openai/<model>' or 'anthropic/<model>'"
            )
        self.spec = spec
        self.provider = provider
        self.name = name

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 300,
    ) -> ModelOutput:
        raise NotImplementedError


class OpenAICompatibleModel(Model):
    """OpenRouter and OpenAI, over the OpenAI chat-completions schema."""

    def __init__(self, spec: str) -> None:
        super().__init__(spec)
        from openai import AsyncOpenAI

        if self.provider == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise ModelUnavailable("OPENROUTER_API_KEY is not set")
            self._client = AsyncOpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
            self._headers = {
                "HTTP-Referer": "https://github.com/local/instrumental_evasion",
                "X-Title": "instrumental_evasion",
            }
        else:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ModelUnavailable("OPENAI_API_KEY is not set")
            self._client = AsyncOpenAI(api_key=key)
            self._headers = {}

    @staticmethod
    def _to_wire(messages: list[ChatMessage]) -> list[dict[str, Any]]:
        import json

        wire: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "tool":
                wire.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content,
                    }
                )
            elif message.role == "assistant" and message.tool_calls:
                wire.append(
                    {
                        "role": "assistant",
                        "content": message.content or None,
                        "tool_calls": [
                            {
                                "id": call.id,
                                "type": "function",
                                "function": {
                                    "name": call.function,
                                    "arguments": json.dumps(call.arguments),
                                },
                            }
                            for call in message.tool_calls
                        ],
                    }
                )
            else:
                wire.append({"role": message.role, "content": message.content})
        return wire

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 300,
    ) -> ModelOutput:
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": self._to_wire(messages),
            "timeout": timeout,
        }
        if tools:
            payload["tools"] = [tool.as_openai() for tool in tools]
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self._headers:
            payload["extra_headers"] = self._headers

        async def call():
            return await self._client.chat.completions.create(**payload)

        response = await _with_retries(
            call, retries=DEFAULT_MAX_RETRIES, what=f"generate({self.spec})"
        )

        choice = response.choices[0] if response.choices else None
        raw = getattr(choice, "message", None)
        calls = [
            ToolCall(
                id=call.id or f"call_{index}",
                function=call.function.name,
                arguments=parse_arguments(call.function.arguments),
            )
            for index, call in enumerate(getattr(raw, "tool_calls", None) or [])
        ]
        usage = getattr(response, "usage", None)
        details = getattr(usage, "prompt_tokens_details", None)
        # The two providers report caching under opposite conventions, and
        # ModelUsage.total assumes Anthropic's:
        #   Anthropic -- input_tokens excludes cache_read/cache_creation; the
        #                three are disjoint and summing them is the prompt.
        #   OpenAI    -- prompt_tokens is the whole prompt and
        #                prompt_tokens_details.cached_tokens is the subset of it
        #                that was a cache hit.
        # Passing prompt_tokens through as input_tokens would count every cache
        # read twice and roughly halve the effective token limit. Normalise to
        # the disjoint convention so total counts the prompt once.
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        cached_tokens = getattr(details, "cached_tokens", 0) or 0
        cached_tokens = min(cached_tokens, prompt_tokens)
        return ModelOutput(
            message=ChatMessage(
                role="assistant",
                content=getattr(raw, "content", "") or "",
                tool_calls=calls,
            ),
            usage=ModelUsage(
                input_tokens=prompt_tokens - cached_tokens,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                cache_read_tokens=cached_tokens,
            ),
            stop_reason=getattr(choice, "finish_reason", "stop") or "stop",
            model=self.spec,
        )


class AnthropicModel(Model):
    """The Anthropic SDK, for API-key runs against Claude models."""

    def __init__(self, spec: str) -> None:
        super().__init__(spec)
        from anthropic import AsyncAnthropic

        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ModelUnavailable("ANTHROPIC_API_KEY is not set")
        self._client = AsyncAnthropic(api_key=key)

    @staticmethod
    def _to_wire(messages: list[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        wire: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
                # Consecutive tool results belong in ONE user message, or the
                # API rejects the turn order.
                if wire and wire[-1]["role"] == "user" and isinstance(
                    wire[-1]["content"], list
                ):
                    wire[-1]["content"].append(block)
                else:
                    wire.append({"role": "user", "content": [block]})
            elif message.role == "assistant":
                content: list[dict[str, Any]] = []
                if message.content:
                    content.append({"type": "text", "text": message.content})
                for call in message.tool_calls:
                    content.append(
                        {
                            "type": "tool_use",
                            "id": call.id,
                            "name": call.function,
                            "input": call.arguments,
                        }
                    )
                wire.append({"role": "assistant", "content": content or [{"type": "text", "text": ""}]})
            else:
                wire.append({"role": "user", "content": message.content})
        return system, wire

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        tools: list[ToolDef] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 300,
    ) -> ModelOutput:
        system, wire = self._to_wire(messages)
        payload: dict[str, Any] = {
            "model": self.name,
            "messages": wire,
            "max_tokens": max_tokens or 8192,
            "timeout": timeout,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [tool.as_anthropic() for tool in tools]
        if temperature is not None:
            payload["temperature"] = temperature

        async def call():
            return await self._client.messages.create(**payload)

        response = await _with_retries(
            call, retries=DEFAULT_MAX_RETRIES, what=f"generate({self.spec})"
        )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )
        calls = [
            ToolCall(id=block.id, function=block.name, arguments=dict(block.input or {}))
            for block in response.content
            if getattr(block, "type", "") == "tool_use"
        ]
        usage = getattr(response, "usage", None)
        return ModelOutput(
            message=ChatMessage(role="assistant", content=text, tool_calls=calls),
            usage=ModelUsage(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
                cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
            ),
            stop_reason=getattr(response, "stop_reason", "stop") or "stop",
            model=self.spec,
        )


_PROVIDERS = {
    "openrouter": OpenAICompatibleModel,
    "openai": OpenAICompatibleModel,
    "anthropic": AnthropicModel,
}


def get_model(spec: str) -> Model:
    """Resolve `provider/name` to a client. Raises on an unknown provider."""
    provider = spec.partition("/")[0]
    factory = _PROVIDERS.get(provider)
    if factory is None:
        raise ValueError(
            f"unknown model provider {provider!r} in {spec!r}; "
            f"expected one of {sorted(_PROVIDERS)}"
        )
    return factory(spec)
