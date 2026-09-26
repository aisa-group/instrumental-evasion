"""Keep an explicit Codex provider refusal separate from a CLI infrastructure error."""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock

from instrumental_evasion.runner.gate import Gate
from instrumental_evasion.runner.log import SampleRecord, summarize
from instrumental_evasion.runner.scaffolds.base import (
    STOP_ERROR,
    STOP_PROVIDER_REFUSAL,
    EpisodeContext,
    Limits,
)
from instrumental_evasion.runner.scaffolds.codex import (
    CHATGPT_PROVIDER,
    CYBER_SAFETY_REJECTION,
    OPENROUTER_PROVIDER,
    CodexScaffold,
)


def test_astra_openrouter_route_has_recorded_context_window() -> None:
    scaffold = CodexScaffold(
        model="openrouter/openai/gpt-6-astra",
        provider=OPENROUTER_PROVIDER,
        binary="/nonexistent/codex",
    )

    assert scaffold.provider_model == "openai/gpt-6-astra"
    assert scaffold.describe()["agent_provider_route"] == "openrouter_api"


def _failed_turn(message: str):
    return CodexScaffold._invocation_from_stdout(
        "\n".join(
            (
                json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
                json.dumps({"type": "error", "message": message}),
                json.dumps(
                    {"type": "turn.failed", "error": {"message": message}}
                ),
            )
        )
    )


def _scaffold(provider: str) -> CodexScaffold:
    model = "gpt-5.6-astra" if provider == CHATGPT_PROVIDER else "openrouter/test/model"
    return CodexScaffold(model=model, provider=provider, binary="/nonexistent/codex")


def test_codex_chatgpt_cyber_rejection_is_provider_refusal() -> None:
    message = f"{CYBER_SAFETY_REJECTION} Request ID: test-request"
    observation = _failed_turn(message)

    refusal = _scaffold(CHATGPT_PROVIDER)._provider_refusal_message(observation)

    assert refusal == message


def test_codex_run_returns_provider_refusal_without_harness_error(
    tmp_path, monkeypatch
) -> None:
    message = f"{CYBER_SAFETY_REJECTION} Request ID: test-request"
    stream = "\n".join(
        (
            json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
            json.dumps({"type": "error", "message": message}),
            json.dumps({"type": "turn.failed", "error": {"message": message}}),
        )
    )
    scaffold = _scaffold(CHATGPT_PROVIDER)
    context = EpisodeContext(
        spec=SimpleNamespace(workdir="/workdir"),
        sandbox=SimpleNamespace(_directory=str(tmp_path)),
        gate=Gate(policy=None),
        prompt="Synthetic task",
        system_prompt="Synthetic instructions",
        limits=Limits(3, 100, 10, 3),
        outdir=tmp_path,
        monitored=False,
        score=AsyncMock(return_value=(False, "", {})),
    )
    monkeypatch.setattr(scaffold, "_ensure_gate_server", lambda _ctx: "unused")
    monkeypatch.setattr(scaffold, "_ensure_proxy", lambda _ctx: "unused")
    monkeypatch.setattr(
        scaffold, "_codex_home", lambda _ctx: tmp_path / "private-home"
    )
    monkeypatch.setattr(
        scaffold, "_hooks_file", lambda _ctx: tmp_path / "hooks.json"
    )
    monkeypatch.setattr(scaffold, "_env", lambda *_args: {})
    monkeypatch.setattr(
        scaffold,
        "_probe_hook",
        lambda *_args, **_kwargs: {"hook_alive": True},
    )
    monkeypatch.setattr(
        scaffold,
        "_command",
        lambda *_args: [
            sys.executable,
            "-c",
            f"import sys; print({stream!r}); sys.exit(1)",
        ],
    )

    result = asyncio.run(scaffold.run(context))

    assert result.stop_reason == STOP_PROVIDER_REFUSAL
    assert result.error is None
    assert result.extra["provider_refusal"] is True
    assert result.extra["provider_refusal_kind"] == "cybersecurity_risk"
    assert result.extra["provider_refusal_message"] == message
    assert result.extra["cli_returncode"] == 1
    assert "void_reason" not in result.extra
    context.score.assert_not_awaited()


def test_unrelated_and_openrouter_failures_remain_cli_errors() -> None:
    cyber_observation = _failed_turn(CYBER_SAFETY_REJECTION)
    unrelated_observation = _failed_turn("transport disconnected")

    assert (
        _scaffold(OPENROUTER_PROVIDER)._provider_refusal_message(cyber_observation)
        is None
    )
    assert (
        _scaffold(CHATGPT_PROVIDER)._provider_refusal_message(unrelated_observation)
        is None
    )


def test_summary_counts_provider_refusal_as_valid_unsuccessful_result() -> None:
    provider_refusal = SampleRecord(
        sample_id="refused",
        epoch=0,
        solved=False,
        stop_reason=STOP_PROVIDER_REFUSAL,
        metadata={"provider_refusal": True, "evasion_attempted": False},
    )
    infrastructure_error = SampleRecord(
        sample_id="error",
        epoch=0,
        solved=False,
        stop_reason=STOP_ERROR,
        error="unrelated CLI failure",
    )

    result = summarize([provider_refusal, infrastructure_error], monitored=True)

    assert provider_refusal.infrastructure_status == "valid"
    assert provider_refusal.void is False
    assert result["provider_refusals"] == 1
    assert result["counted"] == 1
    assert result["infrastructure_invalid"] == 1
