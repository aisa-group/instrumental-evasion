"""What a scaffold is, and what the runner promises it.

A scaffold is whatever makes the agent act: our own react loop over an API
model, a vendor CLI, or something not written yet. It is a plugin, and the
contract is deliberately narrow so that adding one does not require changing
the runner.

A scaffold MUST:

  * drive the agent toward `ctx.prompt` inside `ctx.sandbox`;
  * route EVERY proposed tool call through `ctx.gate` before it runs, and show
    the agent `decision.agent_message` when the gate refuses;
  * respect `ctx.limits` as best its shape allows, and say in `stop_reason`
    which bound ended the episode.

A scaffold MUST NOT:

  * score the task -- the runner calls `spec.score_fn`, so every scaffold is
    graded by identical code;
  * classify evasion or write a decision log -- the gate owns both, which is
    what makes two scaffolds' evasion rates comparable;
  * decide anything about the policy.

Where a scaffold cannot honour part of the contract, it says so in
`describe()` rather than pretending. The CLI scaffolds cannot see the agent's
tokens at all (the CLI talks to its vendor's backend directly, which is what
makes subscription auth work), so they report `token_limit_enforced: False` and
are bounded by wall clock instead. That is a cost of the scaffold, not a
defect, and it belongs in the log where a reader will see it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from instrumental_evasion.runner.gate import Gate
from instrumental_evasion.runner.types import ChatMessage, ModelUsage
from instrumental_evasion.tasks.tb_spec import TBTaskSpec

# Why an episode ended. `desisted` and `provider_refusal` are valid refusal
# outcomes. The three limits are harness bounds. `error` is a harness fault and
# voids the episode.
STOP_SUBMIT = "submit"
STOP_AGENT = "agent_stopped"
STOP_TURN_LIMIT = "turn_limit"
STOP_TOKEN_LIMIT = "token_limit"
STOP_TIME_LIMIT = "time_limit"
STOP_ATTEMPTS = "attempts_exhausted"
STOP_DESISTED = "desisted"
STOP_STALLED_LOOP = "stalled_loop"  # A harness bound, not a refusal classification.
STOP_PROVIDER_REFUSAL = "provider_refusal"
STOP_ERROR = "error"


@dataclass(frozen=True)
class Limits:
    """The rollout bounds, resolved once per run from `rollout_core`."""

    max_turns: int
    token_limit: int
    time_limit_seconds: int
    attempts: int

    def as_dict(self) -> dict[str, int]:
        return {
            "max_turns": self.max_turns,
            "token_limit": self.token_limit,
            "time_limit_seconds": self.time_limit_seconds,
            "attempts": self.attempts,
        }


@dataclass
class EpisodeContext:
    """Everything a scaffold is given for one episode."""

    spec: TBTaskSpec
    sandbox: Any
    gate: Gate
    prompt: str
    system_prompt: str
    limits: Limits
    outdir: Path
    monitored: bool
    # Grade the current sandbox state. A scaffold needs this only to implement
    # the no-quitting policy: a `submit` that does not already satisfy the
    # scorer is answered with a keep-going message and the episode continues,
    # so a rollout ends at a limit rather than whenever the model decides it is
    # stuck. Without it, an agent that quits early and one that keeps trying
    # until the cap would score identically.
    score: Callable[[], Awaitable[tuple[bool, str, dict]]]


@dataclass
class ScaffoldResult:
    """What a scaffold reports back. Everything is optional but `stop_reason`."""

    stop_reason: str
    turns: int = 0
    seconds: float = 0.0
    usage: ModelUsage = field(default_factory=ModelUsage)
    messages: list[ChatMessage] = field(default_factory=list)
    error: str | None = None
    # Scaffold-specific facts worth keeping in the log: the CLI's exit codes,
    # how many continuation prompts were used, a void reason.
    extra: dict[str, Any] = field(default_factory=dict)


class Scaffold(Protocol):
    """The plugin interface. One instance per run, reused across episodes."""

    name: str

    def describe(self) -> dict[str, Any]:
        """Facts about this scaffold for the eval log's metadata.

        Must include `agent` (the scaffold name) and say which limits it can
        actually enforce, so a reader can tell a turn-capped episode from one
        bounded only by the clock.
        """
        ...

    async def preflight(
        self, spec: TBTaskSpec, *, refresh_credentials: bool = True
    ) -> dict[str, Any]:
        """Prove this scaffold can run, for this task, before an episode counts.

        Returns a dict with at least `ok: bool`. A scaffold whose monitor
        interception could silently fail MUST check it here -- both CLIs treat
        any hook exit code but 0 or 2 as "run the tool anyway", so a hook that
        cannot start is a monitor that is off while looking like one that is on.
        An episode is void unless the monitor was demonstrably alive.

        Set ``refresh_credentials`` to false for a read-only check. A normal
        evaluation keeps the default so an expiring subscription token can be
        refreshed before the first episode.
        """
        ...

    async def run(self, ctx: EpisodeContext) -> ScaffoldResult:
        """Run one episode."""
        ...
