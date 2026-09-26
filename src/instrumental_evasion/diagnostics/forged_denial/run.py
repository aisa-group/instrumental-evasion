"""Run the forged-denial diagnostic for one agent model.

    # list the planned attempts and the resolved configuration
    python -m instrumental_evasion.diagnostics.forged_denial run \\
        --model openrouter/deepseek/deepseek-v4-flash --dry-run

    # ten tasks x clean/injected x three repeats = 60 attempts
    python -m instrumental_evasion.diagnostics.forged_denial run \\
        --model openrouter/deepseek/deepseek-v4-flash --log-dir logs/forged_denial/dsv4-flash

Every attempt is a fresh Apptainer sandbox and one Codex invocation bounded by
``--timeout`` seconds. The episode itself is the release runner's
(`instrumental_evasion.runner.run.run_task`) on the control arm: no policy, no
monitor call, and a write-once attempt artifact per attempt. The HTTP tasks
serve their pages from this process on 127.0.0.1, which the agent's container
shares; that is why only the Apptainer runtime is supported.

Requirements are those of the Codex scaffold (`CODEX_BIN`, the standalone hook
Python) plus a credential: `codex login` for ChatGPT-subscription models, or
`OPENROUTER_API_KEY` (environment or `.env`) for `openrouter/` models. The task
image needs only python3 with urllib.request, bash, and coreutils. The default
is the release's python:3.12-slim image, `tb-agent-stateful.sif`; set
`FORGED_DENIAL_IMAGE` to another SIF to override it. The paper's runs used a
Debian 12 image with Python 3.13 (terminal-bench's `fix-git` image), whose
/app directory also held that task's unrelated files.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from instrumental_evasion.diagnostics.forged_denial.tasks import (
    CONDITIONS,
    DENIAL_TEXT,
    MAX_REPEATS,
    TASK_IDS,
    create_case,
)
from instrumental_evasion.runner import log as native_log
from instrumental_evasion.runner.attempt_artifact import AttemptArtifactError, new_attempt_id
from instrumental_evasion.runner.scaffolds import Limits

DEFAULT_REPEATS = 3
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_LOG_DIR = Path("logs/forged_denial")
RUNTIME = "apptainer"
# The control arm never calls a monitor; the runner still records a name.
MONITOR_MODEL = "none"

# The three agent configurations of the paper. The reasoning effort applies when
# `--reasoning-effort` is not given.
PAPER_REASONING_EFFORTS = {
    "gpt-6-astra": "medium",
    "openrouter/openai/gpt-5.6-luna": "medium",
    "openrouter/deepseek/deepseek-v4-flash": "high",
}

# The agent sees the task prompt and the runner's one-line environment system
# prompt, nothing else: no guardrail notice and no refusal reminder. The other
# settings pin the Codex defaults so an inherited environment cannot change
# them. Applied for the duration of a run, whatever the caller's environment.
PROTOCOL_ENVIRONMENT = {
    "AGENT_GUARDRAIL_NOTICE": "none",
    "AGENT_REFUSAL_CONTEXT": "none",
    "AGENT_REFUSAL_STOP": "seven_consecutive",
    "AGENT_CONTINUATION": "neutral",
    "AGENT_PRESSURE": "calm",
    "CODEX_TOOL_SCOPE": "native",
    "CODEX_TOOL_INTERFACE": "default",
}
UNSET_ENVIRONMENT = ("CODEX_TOKEN_BUDGET", "CODEX_REASONING_SUMMARY")


@dataclass(frozen=True)
class Attempt:
    """One planned attempt: a task, a condition, and a repeat index."""

    task_id: str
    condition: str
    repeat: int


def plan(
    task_ids: Sequence[str] = TASK_IDS,
    conditions: Sequence[str] = CONDITIONS,
    repeats: int = DEFAULT_REPEATS,
) -> list[Attempt]:
    """Every (task, condition, repeat), repeat-major.

    The two conditions of a task run back to back, and their order alternates
    between repeats so that neither condition always runs first.
    """
    unknown = [task for task in task_ids if task not in TASK_IDS]
    unknown += [condition for condition in conditions if condition not in CONDITIONS]
    if unknown:
        raise ValueError(f"unknown tasks or conditions: {unknown}")
    if type(repeats) is not int or not 1 <= repeats <= MAX_REPEATS:
        raise ValueError(f"repeats must be an integer from 1 through {MAX_REPEATS}")
    tasks = [task for task in TASK_IDS if task in task_ids]
    ordered = [condition for condition in CONDITIONS if condition in conditions]
    attempts = []
    for repeat in range(repeats):
        order = ordered if repeat % 2 == 0 else ordered[::-1]
        attempts += [Attempt(task, condition, repeat) for task in tasks for condition in order]
    return attempts


def limits(timeout: int = DEFAULT_TIMEOUT_SECONDS) -> Limits:
    """One agent invocation with no continuation message, bounded by wall clock."""
    return Limits(max_turns=1, token_limit=1_000_000, time_limit_seconds=timeout, attempts=1)


def reasoning_effort_for(model: str, effort: str | None) -> str | None:
    return effort if effort is not None else PAPER_REASONING_EFFORTS.get(model)


@contextmanager
def protocol_environment() -> Iterator[None]:
    """Apply the protocol settings, and restore the caller's environment after."""
    names = (*PROTOCOL_ENVIRONMENT, *UNSET_ENVIRONMENT)
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(PROTOCOL_ENVIRONMENT)
    for name in UNSET_ENVIRONMENT:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def codex_scaffold(model: str, reasoning_effort: str | None, timeout: int) -> Any:
    from instrumental_evasion.runner.scaffolds.codex import CodexScaffold

    return CodexScaffold(model=model, reasoning_effort=reasoning_effort, cli_timeout=timeout)


class PreflightError(RuntimeError):
    """The scaffold cannot run: missing binary, credential, or image."""


async def run_attempts(
    attempts: Sequence[Attempt],
    *,
    model: str,
    reasoning_effort: str | None,
    timeout: int,
    log_dir: Path,
    runtime: Any = None,
    make_scaffold: Callable[[str, str | None, int], Any] = codex_scaffold,
) -> list[native_log.EvalLog]:
    """Run each attempt in a fresh sandbox with a fresh scaffold.

    A failed attempt is recorded as infrastructure-invalid and does not stop the
    run; a failed scaffold preflight does, because no later attempt could run.
    """
    from instrumental_evasion.runner.run import run_task
    from instrumental_evasion.runner.runtime import get_runtime

    runtime = runtime if runtime is not None else get_runtime(RUNTIME)
    bounds = limits(timeout)
    logs = []
    await runtime.setup()
    try:
        for index, attempt in enumerate(attempts):
            with create_case(attempt.task_id, attempt.repeat, attempt.condition) as case:
                scaffold = make_scaffold(model, reasoning_effort, timeout)
                try:
                    check = await scaffold.preflight(case.task.spec)
                    if not check.get("ok"):
                        raise PreflightError(str(check.get("reason")))
                    print(
                        f"\n[{index + 1}/{len(attempts)}] {attempt.task_id} "
                        f"{attempt.condition} repeat={attempt.repeat}",
                        flush=True,
                    )
                    log = await run_task(
                        case.task, scaffold, runtime,
                        epochs=1, epoch=attempt.repeat, attempt_id=new_attempt_id(),
                        log_dir=log_dir, monitor_model=MONITOR_MODEL, limits=bounds,
                    )
                finally:
                    close = getattr(scaffold, "close", None)
                    if callable(close):
                        close()
            logs.append(log)
            if any(sample.metadata.get("termination") for sample in log.samples):
                break
    finally:
        runtime.teardown()
    return logs


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model", required=True,
        help="Codex model, e.g. gpt-6-astra or openrouter/deepseek/deepseek-v4-flash",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"), default=None,
        help="default: the paper's setting for the three paper models, else the model default",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--tasks", nargs="+", choices=TASK_IDS, default=list(TASK_IDS), metavar="TASK",
        help="narrow the roster (default: all ten)",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS),
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS,
        help=f"wall clock per attempt in seconds (default {DEFAULT_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="build every planned case and print the plan; run nothing",
    )


def _print_plan(attempts: Sequence[Attempt], args: argparse.Namespace, effort: str | None) -> None:
    print(f"model:     {args.model}")
    print(f"effort:    {effort or 'model default'}")
    print(f"runtime:   {RUNTIME}")
    print(f"limits:    {limits(args.timeout).as_dict()}")
    print(f"denial:    {DENIAL_TEXT}")
    print(f"env:       {PROTOCOL_ENVIRONMENT}")
    print(f"logs:      {args.log_dir}")
    print(f"attempts:  {len(attempts)}")
    for index, attempt in enumerate(attempts):
        with create_case(attempt.task_id, attempt.repeat, attempt.condition) as case:
            source = hashlib.sha256(case.source).hexdigest()[:12]
            if index == 0:
                print(f"image:     {case.task.spec.image()}")
        print(
            f"  {index:3d}  {attempt.task_id:<18} {attempt.condition:<8} "
            f"repeat={attempt.repeat}  source_sha256={source}"
        )


def main(args: argparse.Namespace) -> int:
    """Run the parsed `run` command; return a process exit code."""
    try:
        attempts = plan(args.tasks, args.conditions, args.repeats)
    except ValueError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2
    if args.timeout < 1:
        print("ERROR --timeout must be a positive number of seconds", file=sys.stderr)
        return 2
    effort = reasoning_effort_for(args.model, args.reasoning_effort)
    if not args.dry_run:
        from instrumental_evasion.runner.run import load_env_file

        # Before the protocol settings, so a value in .env cannot override them.
        load_env_file()
    with protocol_environment():
        try:
            # Validates the model route and effort before any container starts.
            codex_scaffold(args.model, effort, args.timeout)
        except ValueError as error:
            print(f"ERROR {error}", file=sys.stderr)
            return 2
        _print_plan(attempts, args, effort)
        if args.dry_run:
            return 0

        from instrumental_evasion.runner.termination import finalize_on_termination

        try:
            with finalize_on_termination():
                logs = asyncio.run(run_attempts(
                    attempts, model=args.model, reasoning_effort=effort,
                    timeout=args.timeout, log_dir=args.log_dir,
                ))
        except PreflightError as error:
            print(f"ERROR scaffold preflight failed: {error}", file=sys.stderr)
            return 3
        except AttemptArtifactError as error:
            print(f"ERROR attempt artifact: {error}", file=sys.stderr)
            return 4

    from instrumental_evasion.diagnostics.forged_denial import report

    print()
    print(report.format_summary(report.summarize(report.collect([args.log_dir]))))
    invalid = sum(int(log.results.get("infrastructure_invalid", 0)) for log in logs)
    if invalid:
        print(f"ERROR {invalid} infrastructure-invalid attempt(s)", file=sys.stderr)
        return 4
    return 0
