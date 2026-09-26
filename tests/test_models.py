"""Model resolution and usage accounting for the ReAct scaffold (`runner.models`)."""

from __future__ import annotations

import asyncio

import pytest

from instrumental_evasion.runner.models import (
    AnthropicModel,
    ModelUnavailable,
    OpenAICompatibleModel,
    get_model,
)
from instrumental_evasion.runner.types import ChatMessage

_KEYS = ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY")


@pytest.fixture
def dummy_keys(monkeypatch):
    for name in _KEYS:
        monkeypatch.setenv(name, "test-key")


@pytest.mark.parametrize(
    ("spec", "cls", "provider", "name"),
    [
        ("openrouter/deepseek/deepseek-v4-flash", OpenAICompatibleModel, "openrouter",
         "deepseek/deepseek-v4-flash"),
        ("openai/gpt-4o-mini", OpenAICompatibleModel, "openai", "gpt-4o-mini"),
        ("anthropic/claude-sonnet-4-5", AnthropicModel, "anthropic", "claude-sonnet-4-5"),
    ],
)
def test_get_model_maps_each_provider_prefix(dummy_keys, spec, cls, provider, name) -> None:
    model = get_model(spec)
    assert type(model) is cls
    assert (model.spec, model.provider, model.name) == (spec, provider, name)


def test_openrouter_uses_the_openrouter_endpoint(dummy_keys) -> None:
    model = get_model("openrouter/deepseek/deepseek-v4-flash")
    assert "openrouter.ai" in str(model._client.base_url)


@pytest.mark.parametrize("spec", ["google/gemini-2.5-flash", "vertex/x", "gpt-4o"])
def test_get_model_rejects_an_unknown_provider(dummy_keys, spec) -> None:
    with pytest.raises(ValueError, match="unknown model provider"):
        get_model(spec)


@pytest.mark.parametrize("spec", ["openrouter", "anthropic"])
def test_get_model_rejects_a_spec_without_a_model_name(dummy_keys, spec) -> None:
    with pytest.raises(ValueError, match="no provider prefix"):
        get_model(spec)


@pytest.mark.parametrize(
    ("spec", "variable"),
    [
        ("openrouter/deepseek/deepseek-v4-flash", "OPENROUTER_API_KEY"),
        ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
        ("anthropic/claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
    ],
)
def test_a_missing_key_is_reported_by_name(monkeypatch, spec, variable) -> None:
    for name in _KEYS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ModelUnavailable, match=variable):
        get_model(spec)


# Cache accounting. The token limit counts cache reads, so each provider's usage
# is normalised to one convention first. OpenAI-style usage reports
# prompt_tokens as the whole prompt with cached_tokens a subset of it.


class _Details:
    def __init__(self, cached: int) -> None:
        self.cached_tokens = cached


class _Usage:
    def __init__(self, prompt: int, completion: int, cached: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.prompt_tokens_details = _Details(cached)


class _Message:
    content = "done"
    tool_calls: list = []


class _Choice:
    message = _Message()
    finish_reason = "stop"


class _Response:
    def __init__(self, usage: _Usage) -> None:
        self.choices = [_Choice()]
        self.usage = usage


def _openrouter_model_with(usage: _Usage, monkeypatch) -> OpenAICompatibleModel:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    model = OpenAICompatibleModel("openrouter/deepseek/deepseek-v4-flash")

    class _Completions:
        async def create(self, **_kwargs):
            return _Response(usage)

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    model._client = _Client()
    return model


def test_openai_cached_tokens_are_not_counted_twice(monkeypatch) -> None:
    """prompt_tokens already contains cached_tokens, so the total must not re-add it."""
    model = _openrouter_model_with(_Usage(prompt=1000, completion=50, cached=900), monkeypatch)
    out = asyncio.run(model.generate([ChatMessage("user", "hi")]))

    assert out.usage.cache_read_tokens == 900
    assert out.usage.input_tokens == 100
    assert out.usage.total == 1050


def test_openai_usage_without_caching_is_unchanged(monkeypatch) -> None:
    model = _openrouter_model_with(_Usage(prompt=1000, completion=50, cached=0), monkeypatch)
    out = asyncio.run(model.generate([ChatMessage("user", "hi")]))

    assert out.usage.input_tokens == 1000
    assert out.usage.total == 1050


def test_openai_cached_over_prompt_is_clamped(monkeypatch) -> None:
    """A provider reporting cached > prompt must never make input_tokens negative."""
    model = _openrouter_model_with(_Usage(prompt=100, completion=10, cached=900), monkeypatch)
    out = asyncio.run(model.generate([ChatMessage("user", "hi")]))

    assert out.usage.input_tokens == 0
    assert out.usage.total == 110
