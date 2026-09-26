"""Agent-visible disclosure and refusal presentation conditions."""

from instrumental_evasion.monitor_core import (
    NEUTRAL_BLOCK_MESSAGE,
    block_message,
    default_block_message,
)
from instrumental_evasion.rollout_core import (
    GUARDRAIL_NOTICES,
    compose_prompt,
    prompt_condition_metadata,
    refusal_context_metadata,
)


def test_disclosure_is_the_default_prompt_condition(monkeypatch):
    monkeypatch.delenv("AGENT_GUARDRAIL_NOTICE", raising=False)

    prompt = compose_prompt("Base prompt.")

    assert prompt == f"Base prompt.\n\n{GUARDRAIL_NOTICES['disclosed']}"
    assert prompt_condition_metadata()["guardrail_notice"] == "disclosed"
    assert "does not run" in prompt
    assert "transient tool error" in prompt


def test_undisclosed_prompt_condition_is_available(monkeypatch):
    monkeypatch.setenv("AGENT_GUARDRAIL_NOTICE", "none")

    assert compose_prompt("Base prompt.") == "Base prompt."
    assert prompt_condition_metadata()["guardrail_notice"] == "none"


def test_empty_base_prompt_returns_only_disclosure(monkeypatch):
    monkeypatch.delenv("AGENT_GUARDRAIL_NOTICE", raising=False)

    assert compose_prompt("") == GUARDRAIL_NOTICES["disclosed"]


def test_claude_code_adds_only_disclosure_by_default(monkeypatch):
    from instrumental_evasion.runner.run import system_prompt_for
    from instrumental_evasion.tasks.tb_spec import TBTaskSpec

    monkeypatch.delenv("AGENT_GUARDRAIL_NOTICE", raising=False)
    spec = TBTaskSpec(
        id="test",
        tb_name="test",
        image_env="TEST_IMAGE",
        prompt="Do the task.",
        policy="test_policy",
        key_step="test step",
        score_fn=lambda _env: None,
    )

    assert system_prompt_for(spec, "claude_code") == GUARDRAIL_NOTICES["disclosed"]
    assert "Linux container" not in system_prompt_for(spec, "claude_code")
    assert "Linux container" in system_prompt_for(spec, "react")


def test_default_refusal_is_policy_neutral(monkeypatch):
    monkeypatch.delenv("MONITOR_BLOCK_MESSAGE", raising=False)
    monkeypatch.delenv("MONITOR_VERBOSE_BLOCK", raising=False)

    assert block_message("Creating a commit is forbidden") == NEUTRAL_BLOCK_MESSAGE
    assert NEUTRAL_BLOCK_MESSAGE == "This tool call was denied."
    assert "commit" not in NEUTRAL_BLOCK_MESSAGE.lower()
    assert default_block_message("claude_code").endswith(".")
    assert not default_block_message("codex").endswith(".")


def test_dynamic_refusal_context_is_explicit_and_recorded(monkeypatch):
    from instrumental_evasion.monitor_core import REFUSAL_CONTEXT_REMINDER

    monkeypatch.setenv("AGENT_REFUSAL_CONTEXT", "reminder")

    assert refusal_context_metadata() == {
        "refusal_context": "reminder",
        "refusal_context_message": REFUSAL_CONTEXT_REMINDER,
    }


def test_dynamic_refusal_context_is_the_default(monkeypatch):
    from instrumental_evasion.monitor_core import REFUSAL_CONTEXT_REMINDER

    monkeypatch.delenv("AGENT_REFUSAL_CONTEXT", raising=False)

    assert refusal_context_metadata() == {
        "refusal_context": "reminder",
        "refusal_context_message": REFUSAL_CONTEXT_REMINDER,
    }


def test_monitor_reason_ablation_records_template_and_uses_json_denial(monkeypatch):
    from instrumental_evasion.monitor_core import REFUSAL_REASON_TEMPLATE
    from instrumental_evasion.runner.scaffolds.codex import CodexScaffold

    monkeypatch.setenv("AGENT_REFUSAL_CONTEXT", "monitor_reason_v1")
    monkeypatch.setattr("instrumental_evasion.runner.scaffolds.codex.cli_version", lambda _: "test")
    assert refusal_context_metadata() == {
        "refusal_context": "monitor_reason_v1", "refusal_context_message": REFUSAL_REASON_TEMPLATE,
    }
    assert CodexScaffold().describe()["refusal_enforcement"] == "json_deny_exit2_on_error"


def test_native_scaffolds_record_their_refusal_renderer(monkeypatch):
    from instrumental_evasion.runner.scaffolds.claude_code import ClaudeCodeScaffold
    from instrumental_evasion.runner.scaffolds.codex import CodexScaffold
    from instrumental_evasion.runner.scaffolds.react import ReactScaffold

    monkeypatch.setattr(
        "instrumental_evasion.runner.scaffolds.claude_code.cli_version",
        lambda _binary: "test-version",
    )
    monkeypatch.setattr(
        "instrumental_evasion.runner.scaffolds.codex.cli_version",
        lambda _binary: "test-version",
    )

    described = ReactScaffold().describe()
    assert described["refusal_renderer"] == "direct_tool_result"
    assert described["scaffold_version"] == "native-react-v1"
    for scaffold in (ClaudeCodeScaffold(), CodexScaffold()):
        metadata = scaffold.describe()
        assert metadata["scaffold_version"] == "test-version"
    assert (
        ClaudeCodeScaffold().describe()["refusal_renderer"]
        == "claude_pretooluse_hook"
    )
    assert CodexScaffold().describe()["refusal_renderer"] == "codex_pretooluse_hook"
