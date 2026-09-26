"""Keep an explicit provider refusal separate from a CLI infrastructure error."""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from instrumental_evasion.runner.scaffolds.claude_code import ClaudeCodeScaffold, _provider_refusal


def refusal_events():
    return [
        {"type": "system", "subtype": "init", "session_id": "one"},
        {"type": "system", "subtype": "model_refusal_no_fallback", "session_id": "one",
         "original_model": "claude-opus-5", "api_refusal_category": "cyber"},
        {"type": "assistant", "session_id": "one", "is_api_error_message": True,
         "error": "invalid_request", "message": {"model": "<synthetic>",
         "stop_reason": "refusal", "stop_details": {"type": "refusal", "category": "cyber"},
         "content": [{"type": "text", "text": "The provider refused."}]}},
        {"type": "result", "session_id": "one", "is_error": True, "subtype": "success",
         "stop_reason": "refusal", "terminal_reason": "api_error", "api_error_status": None},
    ]


@pytest.mark.parametrize("variation", ["exact", "missing_notice", "wrong_model", "http_error",
    "fallback", "mixed_session", "missing_terminal", "duplicate_terminal", "other_error",
    "wrong_category", "other_model", "malformed", "terminal_error", "terminal_errors", "missing_category"])
def test_only_complete_matching_no_fallback_refusals_are_valid(variation):
    events = refusal_events()
    if variation == "missing_notice":
        events.pop(1)
    elif variation == "wrong_model":
        events[1]["original_model"] = "claude-opus-4-8"
    elif variation == "http_error":
        events[-1]["api_error_status"] = 429
    elif variation == "fallback":
        events.insert(1, {"type": "system", "subtype": "model_fallback"})
    elif variation == "mixed_session":
        events[-1]["session_id"] = "other"
    elif variation == "missing_terminal":
        events.pop()
    elif variation == "duplicate_terminal":
        events.append(copy.deepcopy(events[-1]))
    elif variation == "other_error":
        events.insert(1, {"type": "system", "subtype": "api_error"})
    elif variation == "terminal_error":
        events[-1]["error"] = "unrelated failure"
    elif variation == "terminal_errors":
        events[-1]["errors"] = ["unrelated failure"]
    elif variation == "missing_category":
        events[1].pop("api_refusal_category")
        events[2]["message"]["stop_details"].pop("category")
    elif variation == "wrong_category":
        events[2]["message"]["stop_details"]["category"] = "other"
    elif variation == "other_model":
        events.insert(1, {"type": "assistant", "message": {
            "model": "claude-opus-4-8", "content": [{"type": "text", "text": "Other model."}]}})
    stdout = "\n".join(json.dumps(e) for e in events)
    if variation == "malformed":
        stdout += "\nnot JSON"
    observation = ClaudeCodeScaffold._invocation_from_stdout(stdout)
    assert _provider_refusal(stdout, observation, "claude-opus-5") is (variation == "exact")


@pytest.mark.parametrize("runtime", ["local", "modal"])
def test_provider_refusal_stops_without_another_invocation_or_score(monkeypatch, tmp_path, runtime):
    from instrumental_evasion import subscription
    from instrumental_evasion.hooks import deploy

    # The loop-stop extension is local-only; this test covers the default (off).
    monkeypatch.delenv("AGENT_LOOP_STOP", raising=False)
    stdout = "\n".join(json.dumps(e) for e in refusal_events())
    records = []
    calls = []

    def probe(*_args):
        records.append({"decision_source": "monitor"})
        return {"hook_alive": True}

    def invoke(*_args, **_kwargs):
        calls.append(True)
        return {"stdout": stdout, "rc": 1}

    async def score():
        pytest.fail("The runner must score after the scaffold stops for a provider refusal.")

    async def noop(*_args, **_kwargs):
        pass

    scaffold = ClaudeCodeScaffold(model="claude-opus-5")
    monkeypatch.setattr(deploy, "probe_hook", probe)
    monkeypatch.setattr(scaffold, "_episode", lambda _ctx: SimpleNamespace(
        run=invoke, _home=lambda: tmp_path, _binds=lambda *_args: [],
        _env=lambda: {}, command=lambda *_args, **_kwargs: [],
    ))
    ctx = SimpleNamespace(sandbox=SimpleNamespace(), spec=SimpleNamespace(),
        gate=SimpleNamespace(records=lambda: records), prompt="task", system_prompt="",
        limits=SimpleNamespace(max_turns=3, time_limit_seconds=60), outdir=tmp_path, score=score)
    if runtime == "modal":
        async def start(*_args):
            return "http://gate", "test-token"

        async def count():
            return 0

        async def hook(*_args, **_kwargs):
            return True, None, ""

        async def execute(command, **_kwargs):
            if command[:2] == ["/usr/local/bin/claude", "-p"]:
                invoke()
                return SimpleNamespace(returncode=1, stdout=stdout, stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        ctx.sandbox = SimpleNamespace(start_remote_gate=start, allow_hosts=noop, prepare_user=noop, stage_path=noop,
            exec=execute, remote_gate_count=count)
        ctx.spec = SimpleNamespace(workdir="/app", container_env={})
        ctx.gate.policy = "test"
        monkeypatch.setattr(deploy, "Episode", lambda **_kwargs: SimpleNamespace(_settings_json=lambda: "{}"))
        monkeypatch.setattr(subscription, "anthropic_credential", lambda: SimpleNamespace(expired=False))
        monkeypatch.setattr(scaffold, "_effective_egress_allow", lambda _ctx: ())
        monkeypatch.setattr(scaffold, "_stage_modal_cli", noop)
        monkeypatch.setattr(scaffold, "_stage_modal_home", noop)
        monkeypatch.setattr(scaffold, "_probe_modal_hook", hook)
    outcome = asyncio.run(scaffold.run(ctx))
    assert outcome.stop_reason == "provider_refusal"
    assert outcome.error is None
    assert outcome.extra["provider_refusal"] is True
    assert calls == [True]
