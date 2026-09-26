"""Read completed runner attempts and select the runs that enter an analysis.

The runner writes one write-once directory per episode attempt,
``<log-dir>/<task_id>_<arm>/epoch<N>/<attempt_id>/``. An attempt enters an
analysis only when its completion marker validates against ``result.json``.
Infrastructure-invalid episodes are excluded rather than counted as failures;
a valid episode that stops early or exhausts its budget remains a valid,
unsuccessful episode.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from instrumental_evasion.runner import log as native_log
from instrumental_evasion.runner.attempt_artifact import (
    AttemptArtifactError,
    AttemptArtifactUnreadable,
    validate_attempt,
    validate_attempt_id,
)
from instrumental_evasion.tasks.registry import FAMILY_OF, MONITORED, TASK_IDS

# Run-level metadata that defines a monitored condition. Episodes that differ
# in any of these fields answer different questions and are never pooled.
CONDITION_FIELDS = (
    "policy_variant",
    "monitor_model",
    "monitor_classifier",
    "monitor_history",
    "monitor_history_max_calls",
    "monitor_history_max_arg_chars",
    "monitor_history_outcomes",
    "monitor_context_profile",
    "guardrail_notice",
    "continuation_condition",
    "refusal_context",
    "block_message_condition",
)
OTHER_FAMILY = "other"

# A condition is a tuple of (field, JSON-encoded value), hashable and sortable.
Condition = tuple[tuple[str, str], ...]


class ConditionConflict(ValueError):
    """Monitored episodes span conditions that must not be pooled."""


@dataclass(frozen=True)
class Episode:
    """The analysis view of one completed attempt."""

    directory: Path
    task_id: str
    arm: str
    epoch: int
    attempt_id: str
    agent: str
    condition: Condition
    valid: bool
    task_completed: bool
    confirmed_effect: bool
    success: bool
    attempt_marker: bool | None
    seconds: float
    decisions: tuple[dict[str, Any], ...] = ()

    @property
    def family(self) -> str:
        return FAMILY_OF.get(self.task_id, OTHER_FAMILY)

    @property
    def effect_without_completion(self) -> bool:
        return self.confirmed_effect and not self.task_completed


@dataclass(frozen=True)
class TaskRuns:
    """The selected valid runs of one task for one agent, condition, and arm."""

    task_id: str
    selected: tuple[Episode, ...]
    valid: int
    invalid: int

    @property
    def family(self) -> str:
        return FAMILY_OF.get(self.task_id, OTHER_FAMILY)


@dataclass(frozen=True)
class Group:
    """All tasks run by one agent under one condition and arm."""

    condition: Condition
    agent: str
    arm: str
    tasks: tuple[TaskRuns, ...]


def agent_label(log: native_log.EvalLog) -> str:
    """Identify an agent by its recorded model, scaffold, and reasoning effort."""
    metadata = log.metadata
    label = f"{metadata.get('agent_model') or 'unknown'} [{log.scaffold}]"
    effort = metadata.get("agent_reasoning_effort")
    if effort not in (None, "", "default"):
        label += f" effort={effort}"
    return label


def condition_of(log: native_log.EvalLog) -> Condition:
    """Return the monitored condition; the control arm has none."""
    if log.arm != MONITORED:
        return ()
    return tuple(
        (name, json.dumps(log.metadata.get(name), sort_keys=True)) for name in CONDITION_FIELDS
    )


def condition_dict(condition: Condition) -> dict[str, Any] | None:
    """Decode a condition for JSON output; None for the control arm."""
    return {name: json.loads(value) for name, value in condition} if condition else None


def format_condition(condition: Condition) -> str:
    """Return a one-line label with the fields the logs record."""
    if not condition:
        return "no monitor"
    parts = []
    for name, value in condition:
        decoded = json.loads(value)
        if decoded is not None:
            parts.append(f"{name}={decoded if isinstance(decoded, str) else value}")
    return ", ".join(parts)


def episode_from_log(
    log: native_log.EvalLog, directory: Path, *, keep_decisions: bool = False
) -> Episode:
    """Extract the outcome fields of one validated attempt log."""
    sample = log.samples[0]
    scoring = sample.metadata.get("success_scoring") or {}
    monitored = log.arm == MONITORED
    # The raw marker, as in `runner.log.summarize`; logs that predate the
    # separate raw-marker field carry it as `evasion_attempted`.
    marker = sample.metadata.get(
        "evasion_marker_detected", sample.metadata.get("evasion_attempted")
    )
    return Episode(
        directory=directory,
        task_id=log.task_id,
        arm=log.arm,
        epoch=sample.epoch,
        attempt_id=log.attempt_id,
        agent=agent_label(log),
        condition=condition_of(log),
        valid=(not sample.void
               and sample.infrastructure_status == native_log.InfrastructureStatus.VALID),
        task_completed=bool(scoring.get("task_completed", sample.solved)),
        confirmed_effect=monitored and scoring.get("confirmed_prohibited_effect") is True,
        success=bool(sample.solved),
        attempt_marker=marker if isinstance(marker, bool) else None,
        seconds=float(sample.seconds),
        decisions=tuple(sample.decisions) if keep_decisions else (),
    )


def _is_attempt_id(name: str) -> bool:
    try:
        validate_attempt_id(name)
    except AttemptArtifactError:
        return False
    return True


def _is_epoch_dir(name: str) -> bool:
    return name.startswith("epoch") and name[5:].isdigit()


def find_attempts(roots: Iterable[Path | str]) -> list[Path]:
    """Return every attempt directory below the given log directories."""
    found: set[Path] = set()
    for root in map(Path, roots):
        if not root.is_dir():
            raise FileNotFoundError(f"not a directory: {root}")
        if _is_epoch_dir(root.parent.name) and _is_attempt_id(root.name):
            found.add(root.resolve())
            continue
        for current, children, _ in os.walk(root):
            if _is_epoch_dir(Path(current).name):
                found.update(
                    (Path(current) / name).resolve() for name in children if _is_attempt_id(name)
                )
                children[:] = []
    return sorted(found)


def load_episodes(
    roots: Iterable[Path | str], *, keep_decisions: bool = False
) -> tuple[list[Episode], list[tuple[Path, str]]]:
    """Load validated attempts; return episodes and skipped (path, reason) pairs.

    An attempt without a matching completion marker is incomplete, not an
    episode. Unreadable storage raises instead of silently removing evidence.
    """
    episodes, skipped = [], []
    seen: dict[str, Path] = {}
    for directory in find_attempts(roots):
        try:
            log = validate_attempt(directory)
        except AttemptArtifactUnreadable:
            raise
        except AttemptArtifactError as error:
            skipped.append((directory, str(error)))
            continue
        if log.attempt_id in seen:
            raise ValueError(
                f"attempt {log.attempt_id} appears twice: {seen[log.attempt_id]} and {directory}"
            )
        seen[log.attempt_id] = directory
        episodes.append(episode_from_log(log, directory, keep_decisions=keep_decisions))
    return episodes, skipped


def condition_conflicts(episodes: Iterable[Episode]) -> dict[str, list[Any]]:
    """Return the condition fields on which monitored episodes disagree."""
    conditions = {e.condition for e in episodes if e.arm == MONITORED}
    if len(conditions) < 2:
        return {}
    conflicts = {}
    for index, name in enumerate(CONDITION_FIELDS):
        values = sorted({condition[index][1] for condition in conditions})
        if len(values) > 1:
            conflicts[name] = [json.loads(value) for value in values]
    return conflicts


def require_single_condition(episodes: Iterable[Episode]) -> None:
    """Raise ``ConditionConflict`` unless all monitored episodes share one condition."""
    conflicts = condition_conflicts(episodes)
    if conflicts:
        detail = "; ".join(f"{name}: {values}" for name, values in conflicts.items())
        raise ConditionConflict(
            "monitored logs span several conditions and will not be pooled "
            f"({detail}); pass --split-conditions to report each separately"
        )


def _task_order(task_id: str) -> tuple[int, str]:
    return (TASK_IDS.index(task_id) if task_id in FAMILY_OF else len(TASK_IDS), task_id)


def select_runs(episodes: Iterable[Episode], epochs: int | None) -> list[Group]:
    """Group episodes and select up to ``epochs`` valid runs per task.

    Keep one valid attempt per epoch (the earliest attempt ID, which is
    time-ordered), then the first ``epochs`` epochs in ascending order. The
    selection never depends on outcomes. ``None`` keeps every valid epoch.
    """
    if epochs is not None and epochs < 1:
        raise ValueError("epochs must be a positive integer")
    grouped: dict[tuple[Condition, str, str], dict[str, list[Episode]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for episode in episodes:
        grouped[(episode.condition, episode.agent, episode.arm)][episode.task_id].append(episode)
    groups = []
    for (condition, agent, arm), by_task in sorted(grouped.items(), key=lambda item: item[0]):
        tasks = []
        for task_id in sorted(by_task, key=_task_order):
            attempts = by_task[task_id]
            valid = sorted((e for e in attempts if e.valid), key=lambda e: (e.epoch, e.attempt_id))
            first_per_epoch: dict[int, Episode] = {}
            for episode in valid:
                first_per_epoch.setdefault(episode.epoch, episode)
            selected = tuple(first_per_epoch.values())[:epochs]
            tasks.append(TaskRuns(task_id, selected, len(valid), len(attempts) - len(valid)))
        groups.append(Group(condition, agent, arm, tuple(tasks)))
    return groups
