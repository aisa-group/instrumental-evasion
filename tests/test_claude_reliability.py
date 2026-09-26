"""Claude Code stream parsing, model-fallback handling and the interception audit."""

import copy
import json

import pytest

from instrumental_evasion.claude_trace_audit import audit_claude_trace
from instrumental_evasion.runner.scaffolds.claude_code import ClaudeCodeScaffold


def notification_stream():
    return [
        {"type": "system", "subtype": "task_notification", "session_id": "session"},
        {"type": "system", "subtype": "init", "session_id": "session"},
        {"type": "result", "subtype": "success", "is_error": False,
         "num_turns": 0, "result": "", "session_id": "session"},
        {"type": "system", "subtype": "init", "session_id": "session"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "Done."}]}},
        {"type": "result", "subtype": "success", "is_error": False,
         "num_turns": 1, "result": "Done.", "session_id": "session"},
    ]


def parse(events):
    return ClaudeCodeScaffold._invocation_from_stdout(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events)
    )


def test_trusted_cli_settings_disable_model_switching_without_changing_the_hook(tmp_path):
    from instrumental_evasion.hooks.deploy import CONTAINER_HOOK, CONTAINER_PY, Episode

    episode = Episode(None, "no_compilation", "claude-opus-5", "http://gate", tmp_path)
    settings = json.loads(episode._settings_json())
    assert settings["switchModelsOnFlag"] is False
    assert settings["hooks"]["PreToolUse"] == [{
        "matcher": "*", "hooks": [{"type": "command",
        "command": f"{CONTAINER_PY} -I {CONTAINER_HOOK} || exit 2", "timeout": 120}],
    }]


def test_claude_metadata_records_requested_model_and_fallback_settings(monkeypatch):
    from instrumental_evasion.runner.scaffolds import claude_code

    monkeypatch.setattr(claude_code, "cli_version", lambda _binary: "pinned-cli")
    metadata = ClaudeCodeScaffold(model="claude-opus-5").describe()
    assert metadata["agent_model"] == "claude-opus-5"
    assert metadata["claude_cli_settings"] == {"switchModelsOnFlag": False}
    assert metadata["claude_model_fallback_policy"] == "invalidate_observed_fallback_v1"


def test_background_notification_result_does_not_end_user_invocation():
    observation = parse(notification_stream())
    assert observation.valid
    assert observation.assistant_texts == ("Done.",)


@pytest.mark.parametrize("case", [
    "missing_final", "missing_init", "wrong_session", "nonempty_result",
    "error_result", "nonzero_turns", "another_final", "missing_notification",
])
def test_notification_exception_does_not_accept_other_multiple_results(case):
    events = notification_stream()
    if case == "missing_final":
        events.pop()
    elif case == "missing_init":
        events.pop(3)
    elif case == "wrong_session":
        events[3]["session_id"] = "another"
    elif case == "nonempty_result":
        events[2]["result"] = "Not a notification."
    elif case == "error_result":
        events[2]["is_error"] = True
    elif case == "nonzero_turns":
        events[2]["num_turns"] = 1
    elif case == "another_final":
        events.append(copy.deepcopy(events[-1]))
    else:
        events.pop(0)
    assert not parse(events).valid


def test_unicode_inside_json_and_actual_malformed_records():
    events = notification_stream()[3:]
    events[1]["message"]["content"][0]["text"] = "a\u0085b\u2028c\u2029d"
    assert parse(events).assistant_texts == ("a\u0085b\u2028c\u2029d",)
    assert parse(events).valid
    text = "\n".join(json.dumps(e) for e in events)
    assert not ClaudeCodeScaffold._invocation_from_stdout(text + "\nnot JSON").valid


@pytest.mark.parametrize("event", [
    {"type": "system", "subtype": "model_refusal_fallback", "trigger": "refusal",
     "original_model": "claude-opus-5", "fallback_model": "claude-opus-4-8"},
    {"type": "system", "subtype": "model_fallback", "trigger": "overloaded",
     "original_model": "claude-opus-5", "fallback_model": "claude-opus-4-8"},
    {"type": "system", "subtype": "model_consent_fallback",
     "originalModel": "claude-opus-5", "fallbackModel": "claude-opus-4-8"},
    {"type": "assistant", "message": {"model": "claude-opus-4-8", "content": [
        {"type": "fallback", "from": {"model": "claude-opus-5"},
         "to": {"model": "claude-opus-4-8"}},
    ]}},
])
@pytest.mark.parametrize("final_error", [False, True])
def test_model_fallback_is_invalid_even_when_the_cli_reports_success(event, final_error):
    events = notification_stream()[3:]
    events.insert(1, event)
    events[-1]["is_error"] = final_error
    observation = parse(events)
    assert not observation.valid
    assert observation.stream_error == "Claude reported a model fallback"


def test_fallback_text_in_an_assistant_message_is_not_a_model_switch():
    events = notification_stream()[3:]
    events[1]["message"]["content"][0]["text"] = "The file contains model_refusal_fallback."
    assert parse(events).valid


@pytest.mark.parametrize("runtime", ["local", "modal"])
def test_scaffold_stops_after_fallback_before_accepting_a_score(monkeypatch, tmp_path, runtime):
    import asyncio
    from types import SimpleNamespace

    from instrumental_evasion.hooks import deploy

    # The loop-stop extension is local-only; this test covers the default (off).
    monkeypatch.delenv("AGENT_LOOP_STOP", raising=False)
    events = notification_stream()[3:]
    events.insert(1, {"type": "system", "subtype": "model_refusal_fallback"})
    stdout = "\n".join(json.dumps(event) for event in events)
    records = []
    calls = []

    def probe(*_args):
        records.append({"decision_source": "monitor"})
        return {"hook_alive": True}

    def invoke(*_args, **_kwargs):
        calls.append("invoked")
        return {"stdout": stdout, "rc": 0}

    async def score():
        pytest.fail("The scaffold must reject fallback before accepting a score.")

    scaffold = ClaudeCodeScaffold(model="claude-opus-5")
    monkeypatch.setattr(deploy, "probe_hook", probe)
    episode = SimpleNamespace(run=invoke, _home=lambda: tmp_path,
                              _binds=lambda *_args: [], _env=lambda: {},
                              command=lambda *_args, **_kwargs: [])
    monkeypatch.setattr(scaffold, "_episode", lambda _ctx: episode)
    ctx = SimpleNamespace(sandbox=SimpleNamespace(), spec=SimpleNamespace(),
                          gate=SimpleNamespace(records=lambda: records), prompt="task",
                          limits=SimpleNamespace(max_turns=3, time_limit_seconds=60),
                          outdir=tmp_path, score=score, system_prompt="")
    if runtime == "modal":
        from instrumental_evasion import subscription

        async def noop(*_args, **_kwargs):
            pass

        async def start_gate(*_args):
            return "http://gate", "test-token"

        async def count():
            return 0

        async def hook(*_args, **_kwargs):
            return True, None, ""

        async def execute(command, **_kwargs):
            if command[0] == "/usr/local/bin/claude" and command[1] == "-p":
                response = invoke()
                return SimpleNamespace(returncode=response["rc"], stdout=response["stdout"], stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        ctx.sandbox = SimpleNamespace(start_remote_gate=start_gate, allow_hosts=noop,
                                      prepare_user=noop, stage_path=noop, exec=execute,
                                      remote_gate_count=count)
        ctx.spec = SimpleNamespace(workdir="/app", container_env={})
        ctx.gate.policy = "test-policy"
        ctx.system_prompt = ""
        monkeypatch.setattr(deploy, "Episode", lambda **_kwargs: SimpleNamespace(_settings_json=lambda: "{}"))
        monkeypatch.setattr(subscription, "anthropic_credential", lambda: SimpleNamespace(expired=False))
        monkeypatch.setattr(scaffold, "_effective_egress_allow", lambda _ctx: ())
        monkeypatch.setattr(scaffold, "_stage_modal_cli", noop)
        monkeypatch.setattr(scaffold, "_stage_modal_home", noop)
        monkeypatch.setattr(scaffold, "_probe_modal_hook", hook)
    outcome = asyncio.run(scaffold.run(ctx))
    assert outcome.stop_reason == "error"
    assert outcome.error.endswith("Claude reported a model fallback")
    assert calls == ["invoked"]


def background_turns(count=3):
    events = []
    for index in range(count):
        events.append({"type": "system", "subtype": "init", "session_id": "session"})
        if index == 1:
            events.extend({"type": "system", "subtype": subtype, "session_id": "session"}
                          for subtype in ("background_tasks_changed", "task_updated", "task_notification"))
        text = f"Completed step {index}."
        events.append({"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}})
        events.append({"type": "result", "subtype": "success", "is_error": False,
                       "num_turns": 1, "result": text, "session_id": "session", "stop_reason": "end_turn"})
    return events


@pytest.mark.parametrize("count", [2, 5])
def test_background_turns_keep_all_progress_in_one_invocation(count):
    observation = parse(background_turns(count))
    assert observation.valid
    assert observation.assistant_texts == tuple(f"Completed step {i}." for i in range(count))


def test_fallback_in_an_earlier_background_turn_stays_invalid():
    events = background_turns(3)
    events.insert(2, {"type": "system", "subtype": "model_refusal_fallback"})
    observation = parse(events)
    assert not observation.valid
    assert observation.stream_error == "Claude reported a model fallback"


@pytest.mark.parametrize("failure", [
    "no_notification", "missing_completion", "missing_init", "duplicate_result",
    "new_session", "error_result", "missing_assistant", "trailing_assistant", "malformed",
])
def test_background_turns_fail_closed_on_incomplete_or_unexplained_output(failure):
    events = background_turns()
    if failure == "no_notification":
        events = [e for e in events if e.get("subtype") != "task_notification"]
    elif failure == "missing_completion":
        events.pop()
    elif failure == "missing_init":
        events.pop(3)
    elif failure == "duplicate_result":
        events.insert(3, copy.deepcopy(events[2]))
    elif failure == "new_session":
        events[-1]["session_id"] = "another"
    elif failure == "error_result":
        events[2]["is_error"] = True
    elif failure == "missing_assistant":
        events.pop(1)
    elif failure == "trailing_assistant":
        events.append(copy.deepcopy(events[-2]))
    else:
        events.insert(2, {"type": "assistant", "message": {"content": None}})
    assert not parse(events).valid


def test_assistant_output_after_final_result_requires_another_completed_turn():
    events = background_turns(1)
    events.append(copy.deepcopy(events[-2]))
    assert not parse(events).valid


def proposal(identifier, timestamp=None):
    return {"type": "assistant", "timestamp": timestamp,
            "message": {"content": [{"type": "tool_use", "id": identifier,
            "name": "Write", "input": {"file_path": "/app/file", "content": "same text"}}]}}


def result(identifier, *, error=False, content="written", timestamp=None):
    return {"type": "user", "timestamp": timestamp,
            "message": {"content": [{"type": "tool_result", "tool_use_id": identifier,
            "is_error": error, "content": content}]}}


def audit(events, decisions):
    return audit_claude_trace("\n".join(json.dumps(e) for e in events), decisions)


def allowed(**extra):
    return {"tool_name": "Write", "tool_input": {"file_path": "/app/file", "content": "same text"},
            "blocked": False, **extra}


def test_stale_file_rejection_does_not_consume_successful_retry_decision():
    events = [proposal("first"), result("first", error=True, content=(
        "<tool_use_error>File has been modified since read, either by the user or by a linter. "
        "Read it again before attempting to write it.</tool_use_error>")),
        proposal("second"), result("second")]
    report = audit(events, [allowed()])
    assert report["matched_calls"] == 1
    assert report["unmatched_states"] == {"pre_execution_rejection": 1}
    assert not report["requires_exclusion"]


def test_later_identical_decision_cannot_justify_earlier_execution():
    events = [
        proposal("first", "2026-01-01T00:00:15Z"),
        result("first", timestamp="2026-01-01T00:00:16Z"),
        proposal("second", "2026-01-01T00:00:43Z"),
        result("second", timestamp="2026-01-01T00:00:45Z"),
    ]
    from datetime import datetime
    decision_time = datetime.fromisoformat("2026-01-01T00:00:44Z").timestamp()
    report = audit(events, [allowed(ts=decision_time)])
    assert report["requires_exclusion"]
    assert report["unmatched"][0]["tool_use_id"] == "first"


def test_parallel_calls_match_decisions_in_result_order():
    events = [proposal("slow", 1), proposal("fast", 2),
              result("fast", timestamp=4), result("slow", timestamp=10)]
    report = audit(events, [allowed(ts=3), allowed(ts=8)])
    assert report["matched_calls"] == 2
    assert not report["requires_exclusion"]


def test_exact_tool_id_cannot_be_consumed_by_another_identical_call():
    report = audit([proposal("first"), result("first"), proposal("second"), result("second")],
                   [allowed(tool_use_id="second")])
    assert report["requires_exclusion"]
    assert report["matched_calls"] == 1
    assert report["unmatched"][0]["tool_use_id"] == "first"


def test_error_result_without_explicit_validation_does_not_hide_execution():
    report = audit([proposal("first"), result("first", error=True, content="disk failure")], [])
    assert report["requires_exclusion"]


def test_hook_denial_labeled_permission_rule_still_consumes_its_decision():
    denied = result("first", error=True, content="This tool call was denied.")
    denied["tool_result_meta"] = [{"id": "first", "non_execution_kind": "permission-rule"}]
    report = audit([proposal("first"), denied, proposal("second"), result("second")],
                   [allowed(blocked=True), allowed()])
    assert report["matched_calls"] == 2
    assert not report["requires_exclusion"]
    assert report["unmatched"] == []


def test_bootstrap_denial_without_a_decision_is_infrastructure_failure():
    report = audit([proposal("first"), result("first", error=True, content="This tool call was denied.")], [])
    assert report["requires_exclusion"]
    assert report["unmatched_states"] == {"denial_without_decision": 1}


@pytest.mark.parametrize("path", ["/app/./file", "/app/data/../file", "/app//file", "./file"])
def test_cli_path_normalization_preserves_denial_decision_matching(path):
    event = proposal("first")
    event["message"]["content"][0]["input"]["file_path"] = path
    events = [{"type": "system", "cwd": "/app"}, event,
              result("first", error=True, content="This tool call was denied.")]
    report = audit(events, [allowed(blocked=True)])
    assert report["matched_calls"] == 1
    assert not report["requires_exclusion"]


def test_large_arguments_remain_auditable_after_remote_log_truncation(tmp_path):
    from instrumental_evasion.runner.gate import Gate

    event = proposal("first")
    arguments = event["message"]["content"][0]["input"]
    arguments["content"] = "long file\n" * 2000
    path = tmp_path / "decisions.jsonl"
    gate = Gate(policy=None, decisions_path=path)
    gate.record("Write", arguments, gate.decide("Write", arguments), tool_use_id="first")
    decision = json.loads(path.read_text())
    assert decision["tool_input"] == {"_truncated": True}
    assert not audit([event, result("first")], [decision])["requires_exclusion"]
    arguments["content"] += "changed"
    assert audit([event, result("first")], [decision])["requires_exclusion"]
    decision.pop("tool_input_sha256")
    assert audit([event, result("first")], [decision])["requires_exclusion"]


@pytest.mark.parametrize("monitor_error", [False, True])
def test_hook_probe_requires_a_healthy_trusted_decision(tmp_path, monkeypatch, monitor_error):
    import asyncio
    from types import SimpleNamespace

    from instrumental_evasion.hooks import deploy
    from instrumental_evasion.runner.gate import BLOCK, SOURCE_ERROR, Decision, Gate

    gate = Gate(policy=None)
    scaffold = ClaudeCodeScaffold(model="test")
    monkeypatch.setattr(scaffold, "_episode", lambda ctx: object())

    def probe(*args):
        if monitor_error:
            gate.record("bash", {"command": "probe"}, Decision(BLOCK, "unavailable", SOURCE_ERROR))
        return {"hook_alive": True, "rc": 2}

    monkeypatch.setattr(deploy, "probe_hook", probe)
    ctx = SimpleNamespace(sandbox=object(), spec=object(), gate=gate,
                          limits=SimpleNamespace(time_limit_seconds=30))
    result = asyncio.run(scaffold.run(ctx))
    assert result.turns == 0
    assert result.stop_reason == "error"
    assert result.extra["probe_error"] == "no_healthy_gate_decision"


@pytest.mark.parametrize("evidence", ["missing_decision", "hook_failure", "matching_decision", "missing_stdout", "control"])
def test_runner_checks_interception_after_sync_and_preserves_score(tmp_path, monkeypatch, evidence):
    import asyncio

    from instrumental_evasion.runner.gate import ALLOW, SOURCE_MONITOR, Decision
    from instrumental_evasion.runner.run import run_episode
    from instrumental_evasion.runner.sandbox import ExecResult
    from instrumental_evasion.runner.scaffolds.base import STOP_SUBMIT, Limits, ScaffoldResult
    from instrumental_evasion.tasks.registry import ResolvedTask
    from instrumental_evasion.tasks.tb_spec import TBTaskSpec

    monkeypatch.setattr("instrumental_evasion.runner.gate.verdict_direct",
                        lambda *a, **k: pytest.fail("no provider requests"))

    async def score(_sandbox):
        return True, "authoritative result", {}

    spec = TBTaskSpec(id="fake", tb_name="fake", image_env="X", prompt="do it",
                      policy="no_compilation", key_step="compile", score_fn=score)
    monitored = evidence != "control"
    task = ResolvedTask("fake", "fake", spec, "monitored" if monitored else "control",
                        "no_compilation" if monitored else None)

    class Sandbox:
        cleaned = False

        async def exec(self, _cmd):
            return ExecResult(True, 0, "", "")

        async def sync_remote_gate(self, gate):
            if evidence == "matching_decision":
                gate.record("Write", allowed()["tool_input"],
                            Decision(ALLOW, "test", SOURCE_MONITOR), tool_use_id="first")

        async def cleanup(self):
            self.cleaned = True

    sandbox = Sandbox()

    class Runtime:
        name = "fake"
        read_static_flag = None

        async def open_sandbox(self, _spec, *, monitored):
            return sandbox

    class Scaffold:
        name = "claude_code"

        async def run(self, ctx):
            if evidence != "missing_stdout":
                response = result("first", error=True, content="This tool call was denied.") if evidence == "hook_failure" else result("first")
                events = [proposal("first"), response]
                (ctx.outdir / "stdout.txt").write_text("\n".join(json.dumps(e) for e in events))
            return ScaffoldResult(STOP_SUBMIT, turns=1)

    record = asyncio.run(run_episode(task, Scaffold(), Runtime(), epoch=0, outdir=tmp_path,
                         monitor_model="stub", limits=Limits(2, 1000, 30, 2)))
    assert record.solved
    assert sandbox.cleaned
    assert record.infrastructure_invalid == (evidence in {"missing_decision", "hook_failure", "missing_stdout"})
    if record.infrastructure_invalid:
        assert record.infrastructure_failure == "interception_error"
    if not monitored:
        assert record.decisions == []
        assert "interception_audit" not in record.metadata
