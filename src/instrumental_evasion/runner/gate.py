"""The tool gate: one decision seam, one decision-log format, every scaffold.

A scaffold's only obligation to the experiment is that every tool call the agent
proposes passes through here before it runs. What differs between scaffolds is
only HOW the call is intercepted:

  react        the loop owns the tools, so it calls `review()` directly
  claude_code  a PreToolUse hook inside the container POSTs to `serve()`
  codex        a PreToolUse hook inside the container POSTs to the same endpoint

All of them reach `Gate.decide`, so the policy prompt, the monitor model, the
allowlist, the fail-closed rule and the block message are the same code for all
of them. Adding a scaffold means teaching it to reach this seam; it never means
copying a decision rule.

The decision RECORD is equally shared: one JSONL line per call, in the shape
`instrumental_evasion.evasion.attempts_from_decision_log` reads. That is what makes an
evasion rate from one scaffold comparable with an evasion rate from another --
the classifier sees identical input whoever produced it.

Two properties are load-bearing and are the reason the log is written here
rather than by the scaffold:

* **It is host-side.** Nothing about a decision is written where the agent can
  read it. An agent that can read the log can read the policy.
* **A control episode is logged too.** With no policy the gate allows
  everything, but still records each call, so control and monitored arms are
  classified by the same code and a control arm has an attempt rate rather than
  a blank.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable

from instrumental_evasion.evasion import classify_call
from instrumental_evasion.hooks.monitor_client import MonitorUnavailable, verdict_direct
from instrumental_evasion.monitor_core import (
    NEUTRAL_BLOCK_MESSAGE,
    HistoryConfig,
    HistoryEntry,
    allowlist_pattern,
    block_message,
)

ALLOW = "ALLOW"
BLOCK = "BLOCK"

# Sources of a decision, recorded per call so an audit can tell a monitor
# verdict from a pre-approval from an outage.
SOURCE_MONITOR = "monitor"
SOURCE_ALLOWLIST = "allowlist"
SOURCE_SUBMIT = "submit_passthrough"
SOURCE_ERROR = "monitor_error"
SOURCE_UNMONITORED = "unmonitored"
SOURCE_REGEX = "regex"

LLM_CLASSIFIER = "llm"
REGEX_CLASSIFIER = "regex"
MONITOR_CLASSIFIERS = (LLM_CLASSIFIER, REGEX_CLASSIFIER)

BlockIntervention = Callable[[str, dict[str, Any]], str]

# Tools that end an episode rather than touching the sandbox. A monitor can
# misread even a bare {"answer": "Done."} as the prohibited action, and a blocked
# submit leaves the agent unable to stop, so it burns every remaining turn.
# Approved here rather than by narrowing the gate, so every call that can
# actually touch the workspace still goes through.
PASSTHROUGH_TOOLS = frozenset({"submit"})


@dataclass(frozen=True)
class Decision:
    """One reviewed tool call."""

    decision: str
    reason: str
    source: str
    raw: str = ""
    allowlist_pattern: str | None = None
    default_block_message: str | None = None
    # How many earlier calls the monitor was shown for this decision. None means
    # the myopic condition -- no history block was sent at all -- and is not the
    # same as 0, which means the history condition on the episode's first call.
    history_calls: int | None = None
    classifier: str = LLM_CLASSIFIER
    classifier_version: str | None = None
    rule_id: str | None = None
    parse_status: str | None = None
    classifier_error: str | None = None
    latency_ms: float | None = None
    canonical_arguments: str | None = None
    monitor_attempts: tuple[dict[str, Any], ...] = ()
    block_intervention: str | None = None

    @property
    def blocked(self) -> bool:
        return self.decision != ALLOW

    @property
    def agent_message(self) -> str:
        """What the AGENT sees when the call is refused.

        The monitor's own reason names the forbidden action, which hands the
        agent the hidden policy, so evasion after that is partly prompted by the
        harness. Neutral by default; MONITOR_VERBOSE_BLOCK=1 shows the monitor's
        reason instead. The full reason is always in the decision record.
        """
        return block_message(self.reason, default=self.default_block_message)


def resolve_policy(policy: str | None) -> str | None:
    """Apply the requested policy variant without silently changing it.

    A trimmed condition requires a registered ``<policy>_trim`` variant. A
    missing variant is invalid configuration, not a reason to run the full
    policy while recording the episode as trimmed.
    """
    from instrumental_evasion.policies import policy_variant_from_env, trimmed_policy_id

    if policy is None or policy_variant_from_env() != "trimmed":
        return policy
    return trimmed_policy_id(policy)


def policy_prompt_metadata(
    policy: str | None, history: HistoryConfig | None = None
) -> dict[str, str]:
    """The wording identity of the policy an episode ran against.

    Resolved exactly as `Gate` resolves it -- the configured variant first, then
    the history reframing -- so a log's digest matches the decision records
    written beside it. The control arm has no policy and gets no digest.
    """

    resolved = resolve_policy(policy)
    if resolved is None:
        return {}
    from instrumental_evasion.policies import policy_prompt_sha256

    digest = policy_prompt_sha256(
        resolved, history_disclosed=(history or HistoryConfig()).enabled
    )
    return {
        "policy_prompt_sha256": digest,
        # A short handle for a results table column; the full digest stays
        # beside it so the short form is never the only record.
        "policy_revision": digest[:12],
    }


class Gate:
    """Review proposed tool calls against one policy, and record every one."""

    def __init__(
        self,
        *,
        policy: str | None,
        model: str | None = None,
        decisions_path: Path | str | None = None,
        include_assistant_message: bool = False,
        legacy_allowlist_ls_cat: bool = False,
        history: HistoryConfig | None = None,
        default_block_message: str | None = None,
        classifier: str = LLM_CLASSIFIER,
        regex_ruleset_version: str | None = None,
        block_intervention: BlockIntervention | None = None,
    ) -> None:
        self.policy = resolve_policy(policy)
        self.model = model
        self.include_assistant_message = include_assistant_message
        self.legacy_allowlist_ls_cat = legacy_allowlist_ls_cat
        self.default_block_message = default_block_message
        if classifier not in MONITOR_CLASSIFIERS:
            raise ValueError(
                f"unknown monitor classifier {classifier!r}; "
                f"expected one of {MONITOR_CLASSIFIERS}"
            )
        self.classifier = classifier
        self.regex_ruleset_version: str | None = None
        if self.classifier == REGEX_CLASSIFIER:
            from instrumental_evasion.regex_monitor import (
                REGEX_RULESET_VERSION,
                rulesets_for_version,
                validate_regex_policy,
            )

            selected_version = regex_ruleset_version or REGEX_RULESET_VERSION
            # Validate the version even for a control gate. A misspelled
            # condition must fail before it produces a mislabeled trace.
            rulesets_for_version(selected_version)
            self.regex_ruleset_version = selected_version
            if self.policy is not None:
                validate_regex_policy(self.policy, selected_version)
        # The monitor is myopic unless this says otherwise. Defaulting to
        # `HistoryConfig()` rather than reading the environment here keeps the
        # condition an argument a caller passes -- a runner reads the env once,
        # prints it and logs it, so an episode can never run under a condition
        # nobody named. cf `HistoryConfig.from_env`.
        self.history = history or HistoryConfig()
        self.block_intervention = block_intervention
        self.decisions_path = Path(decisions_path) if decisions_path else None
        if self.decisions_path is not None:
            self.decisions_path.parent.mkdir(parents=True, exist_ok=True)
            self.decisions_path.write_text("", encoding="utf-8")  # fresh, host-side
        self._records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.counts = {"allow": 0, "block": 0, "error": 0}
        # Fail at construction rather than turning every tool call into a
        # fail-closed block because of a typo in the policy id.
        if self.policy is not None:
            from instrumental_evasion.policies import (
                get_guardrail_spec,
                policy_prompt_sha256,
            )

            self.spec = get_guardrail_spec(self.policy)
            # The WORDING this episode ran against. The id alone does not
            # identify a guardrail across a rewrite of its body, and the history
            # condition changes the prompt too, so both go into the digest.
            self.policy_prompt_sha256: str | None = policy_prompt_sha256(
                self.policy, history_disclosed=self.history.enabled
            )
            if self.classifier == REGEX_CLASSIFIER and self.history.enabled:
                raise ValueError(
                    "the regex classifier is myopic; use the LLM classifier for "
                    "a monitor-history condition"
                )
        else:
            self.spec = None
            self.policy_prompt_sha256 = None

    @property
    def monitored(self) -> bool:
        return self.policy is not None

    # -- history ----------------------------------------------------------

    def history_entries(self) -> list[HistoryEntry]:
        """This episode's calls so far, oldest first, as the monitor sees them.

        Read off the gate's OWN records, which is the only complete account of
        what the agent proposed: a CLI scaffold's calls never enter our message
        list, and a blocked call never enters the container's transcript as an
        action at all. It is also the reason history works identically on every
        scaffold: they all feed this same list.

        The gate records AFTER it decides, so the call under review is never in
        its own history. Two calls decided concurrently can see the same
        history; that is accepted rather than serialised, because holding the
        lock across a monitor round trip would make the gate the episode's
        bottleneck.
        """

        with self._lock:
            records = list(self._records)
        return [
            HistoryEntry(
                function=record["tool_name"],
                arguments=record["tool_input"],
                blocked=record["blocked"],
            )
            for record in records
        ]

    # -- deciding ---------------------------------------------------------

    def decide(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        assistant_message: str | None = None,
        history: list[HistoryEntry] | None = None,
    ) -> Decision:
        """Decide one call. Synchronous: the monitor client is urllib-based.

        Fails CLOSED. A monitor that could not answer has not approved anything,
        and failing open would let an outage score as a clean evasion: a monitor
        that is off would look like one that is on.

        `history` overrides what the monitor is shown; it exists for offline
        policy-recall probes that prime a synthetic history to ask whether
        disclosure closes a route. In a real episode it is left None and the
        gate supplies its own records.
        """
        if not self.monitored:
            return Decision(
                ALLOW,
                "No policy in force.",
                SOURCE_UNMONITORED,
                classifier=self.classifier,
            )

        if tool_name in PASSTHROUGH_TOOLS:
            return Decision(
                ALLOW,
                "Episode-ending submit.",
                SOURCE_SUBMIT,
                classifier=self.classifier,
            )

        if self.legacy_allowlist_ls_cat:
            pattern = allowlist_pattern(tool_name, tool_input)
            if pattern is not None:
                return Decision(
                    ALLOW,
                    f"Pre-approved by legacy policy allowlist: {pattern}",
                    SOURCE_ALLOWLIST,
                    allowlist_pattern=pattern,
                    classifier=self.classifier,
                )

        if self.classifier == REGEX_CLASSIFIER:
            from instrumental_evasion.regex_monitor import classify_tool_call

            verdict = classify_tool_call(
                str(self.policy),
                tool_name,
                tool_input,
                ruleset_version=str(self.regex_ruleset_version),
            )
            source = SOURCE_ERROR if verdict.parse_status == "error" else SOURCE_REGEX
            return Decision(
                verdict.decision,
                verdict.reason,
                source,
                default_block_message=self.default_block_message,
                history_calls=None,
                classifier=REGEX_CLASSIFIER,
                classifier_version=verdict.ruleset_version,
                rule_id=verdict.rule_id,
                parse_status=verdict.parse_status,
                classifier_error=verdict.classifier_error,
                latency_ms=verdict.latency_ms,
                canonical_arguments=verdict.canonical_arguments,
            )

        # None means the myopic monitor and the prompt it has always had; a
        # list (empty included) means the history condition. Never `or []`.
        if history is not None:
            disclosed = list(history)
        elif self.history.enabled:
            disclosed = self.history_entries()
        else:
            disclosed = None

        history_calls_disclosed = None
        if disclosed is not None:
            history_calls_disclosed = len(disclosed)
            if self.history.max_calls > 0:
                history_calls_disclosed = min(
                    history_calls_disclosed, self.history.max_calls
                )

        diagnostics: list[dict] = []
        started = time.monotonic()
        try:
            verdict, raw = verdict_direct(
                tool_name,
                tool_input,
                policy=str(self.policy),
                assistant_message=(
                    assistant_message if self.include_assistant_message else None
                ),
                history=disclosed,
                history_config=self.history,
                model=self.model,
                diagnostics=diagnostics,
            )
        except MonitorUnavailable as error:
            return Decision(
                BLOCK,
                f"Monitor unavailable: {error}",
                SOURCE_ERROR,
                default_block_message=self.default_block_message,
                history_calls=history_calls_disclosed,
                classifier=LLM_CLASSIFIER,
                parse_status="error",
                classifier_error=str(error),
                latency_ms=round((time.monotonic() - started) * 1000, 2),
                monitor_attempts=tuple(diagnostics),
            )
        except Exception as error:  # noqa: BLE001 - fail closed by design
            return Decision(
                BLOCK,
                f"Monitor unavailable: {type(error).__name__}: {error}",
                SOURCE_ERROR,
                default_block_message=self.default_block_message,
                history_calls=history_calls_disclosed,
                classifier=LLM_CLASSIFIER,
                parse_status="error",
                classifier_error=f"{type(error).__name__}: {error}",
                latency_ms=round((time.monotonic() - started) * 1000, 2),
                monitor_attempts=tuple(diagnostics),
            )
        return Decision(
            verdict.decision,
            verdict.reason,
            SOURCE_MONITOR,
            raw=raw,
            default_block_message=self.default_block_message,
            history_calls=history_calls_disclosed,
            classifier=LLM_CLASSIFIER,
            parse_status="ok",
            latency_ms=round((time.monotonic() - started) * 1000, 2),
            monitor_attempts=tuple(diagnostics),
        )

    def review_sync(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        assistant_message: str | None = None,
        history: list[HistoryEntry] | None = None,
        tool_use_id: str | None = None,
    ) -> Decision:
        """Decide, apply a trusted intervention, and record one tool call."""

        decision = self.decide(
            tool_name,
            tool_input,
            assistant_message=assistant_message,
            history=history,
        )
        intervention = None
        if (
            decision.blocked
            and decision.source in {SOURCE_MONITOR, SOURCE_REGEX}
            and self.block_intervention is not None
        ):
            try:
                intervention = self.block_intervention(tool_name, tool_input)
            except Exception as error:  # noqa: BLE001 - failure must stay closed
                decision = Decision(
                    BLOCK,
                    f"Block intervention unavailable: {type(error).__name__}: {error}",
                    SOURCE_ERROR,
                    raw=decision.raw,
                    default_block_message=decision.default_block_message,
                    history_calls=decision.history_calls,
                    classifier=decision.classifier,
                    classifier_version=decision.classifier_version,
                    rule_id=decision.rule_id,
                    parse_status="error",
                    classifier_error=f"block intervention failed: {type(error).__name__}: {error}",
                    latency_ms=decision.latency_ms,
                    canonical_arguments=decision.canonical_arguments,
                    monitor_attempts=decision.monitor_attempts,
                )
            else:
                decision = replace(decision, block_intervention=intervention)
        self.record(tool_name, tool_input, decision, tool_use_id=tool_use_id)
        return decision

    async def review(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        assistant_message: str | None = None,
        history: list[HistoryEntry] | None = None,
    ) -> Decision:
        """Review one tool call without blocking the event loop."""

        return await asyncio.to_thread(
            partial(
                self.review_sync,
                tool_name,
                tool_input,
                assistant_message=assistant_message,
                history=history,
            )
        )

    # -- recording --------------------------------------------------------

    def record(
        self, tool_name: str, tool_input: dict[str, Any], decision: Decision,
        *, tool_use_id: str | None = None,
    ) -> dict[str, Any]:
        """Append one decision to the host-side log. The ONLY place it is written."""
        record = {
            "ts": time.time(),
            "tool_name": tool_name,
            "tool_input": json.loads(json.dumps(tool_input, default=str))
            if isinstance(tool_input, dict)
            else {},
            "decision": decision.decision,
            "blocked": decision.blocked,
            "reason": decision.reason,
            "decision_source": decision.source,
            "policy": self.policy,
            # The policy's id does not survive a rewrite of its body, so the
            # digest of the prompt is what lets a decision trace identify its
            # own guardrail without the episode file beside it.
            "policy_prompt_sha256": self.policy_prompt_sha256,
            "monitor_classifier": decision.classifier,
            # What the MONITOR could see when it decided. An evasion audit that
            # cannot tell a myopic verdict from a history-informed one is
            # comparing two different monitors, so it is recorded per call and
            # not only per episode: a caller can supply a history for some
            # calls and not others.
            "history_calls_disclosed": decision.history_calls,
            # Markers for this ONE call, so a single decision can be audited
            # without replaying the episode. The episode-level judgement is made
            # by `evasion.episode_attempts`, which can see the order.
            "attempt_markers": list(classify_call(tool_name, tool_input)),
        }
        optional = {
            "tool_use_id": tool_use_id,
            "monitor_attempts": list(decision.monitor_attempts) if decision.monitor_attempts else None,
            "classifier_version": decision.classifier_version,
            "rule_id": decision.rule_id,
            "parse_status": decision.parse_status,
            "classifier_error": decision.classifier_error,
            "classifier_latency_ms": decision.latency_ms,
            "canonical_arguments": decision.canonical_arguments,
            "block_intervention": decision.block_intervention,
        }
        record.update({key: value for key, value in optional.items() if value is not None})
        context_metadata = self.history.as_metadata()
        if self.policy is not None and "monitor_context_profile" in context_metadata:
            record["monitor_context_profile"] = context_metadata["monitor_context_profile"]
            record["monitor_cwd_source"] = context_metadata["monitor_cwd_source"]
        # Remote gates transfer the bounded JSONL log. Retain exact argument
        # evidence even when that log omits a large tool_input value.
        record["tool_input_sha256"] = hashlib.sha256(json.dumps(
            record["tool_input"], sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()).hexdigest()
        with self._lock:
            if decision.source == SOURCE_ERROR:
                self.counts["error"] += 1
            else:
                self.counts["allow" if not decision.blocked else "block"] += 1
            self._records.append(record)
            if self.decisions_path is not None:
                line = json.dumps(record, ensure_ascii=False)
                if len(line) > 8000:  # truncate a pathological input, keep valid JSON
                    trimmed = dict(record)
                    trimmed["tool_input"] = {"_truncated": True}
                    line = json.dumps(trimmed, ensure_ascii=False)
                with self.decisions_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        return record

    def records(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._records)

    def import_records(self, records: list[dict[str, Any]]) -> None:
        """Import records produced by an isolated remote gate.

        Modal CLI scaffolds keep the policy and monitor credential in a second
        sandbox, not in the agent sandbox. At episode teardown its JSONL is
        copied back and installed into this host-side Gate so scoring, evasion
        classification, history metadata, and the durable log remain identical
        to every other runtime. Only a fresh Gate may import: mixing local and
        remote decisions would double-count calls and conceal a routing bug.
        """
        with self._lock:
            if self._records:
                raise RuntimeError("cannot import remote decisions into a non-empty gate")
            for record in records:
                source = record.get("decision_source")
                blocked = bool(record.get("blocked"))
                if source == SOURCE_ERROR:
                    self.counts["error"] += 1
                else:
                    self.counts["block" if blocked else "allow"] += 1
                self._records.append(dict(record))
            if self.decisions_path is not None:
                with self.decisions_path.open("w", encoding="utf-8") as handle:
                    for record in self._records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def attempts(self):
        """This episode's attempt summary, from the records it holds."""
        from instrumental_evasion.runner.attempts import detect_attempts

        return detect_attempts(
            [
                {
                    "tool": r["tool_name"],
                    "arguments": r["tool_input"],
                    "blocked": r["blocked"],
                }
                for r in self.records()
            ]
        )


@dataclass
class GateSummary:
    """Counts a results row carries about the gate, per episode."""

    tool_calls: int = 0
    blocked_calls: int = 0
    monitor_errors: int = 0
    policy: str | None = None
    fields: dict[str, Any] = field(default_factory=dict)


def summarize(gate: Gate) -> GateSummary:
    from instrumental_evasion.runner.attempts import MARKER_VERSION

    records = gate.records()
    attempts = gate.attempts()
    classifier_fields: dict[str, Any] = {"monitor_classifier": gate.classifier}
    if gate.classifier == REGEX_CLASSIFIER:
        classifier_fields["regex_ruleset_version"] = gate.regex_ruleset_version
    return GateSummary(
        tool_calls=len(records),
        blocked_calls=sum(1 for r in records if r["blocked"]),
        monitor_errors=sum(1 for r in records if r["decision_source"] == SOURCE_ERROR),
        policy=gate.policy,
        fields={
            **attempts.as_metadata(),
            "evasion_marker_classifier_version": MARKER_VERSION,
            "evasion_marker_detected": attempts.attempted if attempts.measurable else None,
            **gate.history.as_metadata(),
            **classifier_fields,
        },
    )


__all__ = [
    "ALLOW",
    "BLOCK",
    "Decision",
    "Gate",
    "GateSummary",
    "HistoryConfig",
    "HistoryEntry",
    "LLM_CLASSIFIER",
    "MONITOR_CLASSIFIERS",
    "NEUTRAL_BLOCK_MESSAGE",
    "REGEX_CLASSIFIER",
    "resolve_policy",
    "summarize",
]
