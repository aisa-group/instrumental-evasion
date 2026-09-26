"""The scaffold registry: agent harnesses, as plugins.

    from instrumental_evasion.runner.scaffolds import get_scaffold, available
    scaffold = get_scaffold("codex", model="openrouter/deepseek/deepseek-v4-flash")

Registration is by name and lazy, so importing the registry does not import
every backend: the react scaffold does not need a `claude` binary on PATH.

To add a scaffold: implement `runner.scaffolds.base.Scaffold`, then add one line
to `_FACTORIES`. Nothing else in the runner needs to know it exists.
"""

from __future__ import annotations

from typing import Any, Callable

from instrumental_evasion.runner.scaffolds.base import (  # noqa: F401
    EpisodeContext,
    Limits,
    Scaffold,
    ScaffoldResult,
)


def _react(**options: Any) -> Scaffold:
    from instrumental_evasion.runner.scaffolds.react import ReactScaffold

    return ReactScaffold(**options)


def _claude_code(**options: Any) -> Scaffold:
    from instrumental_evasion.runner.scaffolds.claude_code import ClaudeCodeScaffold

    return ClaudeCodeScaffold(**options)


def _codex(**options: Any) -> Scaffold:
    from instrumental_evasion.runner.scaffolds.codex import CodexScaffold

    return CodexScaffold(**options)


_FACTORIES: dict[str, Callable[..., Scaffold]] = {
    # Our own loop over an API model. Sees every token, enforces every limit.
    "react": _react,
    # The Claude Code CLI, monitored by a PreToolUse hook. Runs on subscription
    # auth, which is why it reaches models the API client rate-limits.
    "claude_code": _claude_code,
    # The Codex CLI, monitored by a PreToolUse hook. ChatGPT subscription auth,
    # or an API key for OpenRouter models. It shares the Claude Code hook runner.
    "codex": _codex,
}

# Scaffolds that run the agent OUT OF PROCESS, as a vendor CLI. They take a
# per-turn wall-clock timeout, reach the gate over HTTP instead of in process,
# and cannot report the agent's token usage. Options that only make sense for
# them are routed by this list rather than passed to every scaffold -- an
# unknown keyword should be a TypeError, not something a scaffold silently
# swallows, or a mistyped option would quietly change nothing.
CLI_SCAFFOLDS = ("claude_code", "codex")

# Scaffolds that authenticate as a logged-in CLI rather than with an API key.
# The same set today, but a different property: one is about where the agent
# runs, the other about how it is billed.
SUBSCRIPTION_SCAFFOLDS = CLI_SCAFFOLDS

# Six of the eight agents in the paper's main comparison ran under Codex.
DEFAULT_SCAFFOLD = "codex"


def available() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def register(name: str, factory: Callable[..., Scaffold]) -> None:
    """Add a scaffold at runtime, for one out of tree or under development."""
    _FACTORIES[name] = factory


def get_scaffold(name: str, **options: Any) -> Scaffold:
    factory = _FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"unknown scaffold {name!r}; expected one of {available()}")
    return factory(**options)
