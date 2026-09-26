"""The upstream verifier must be able to find its own answer key.

Upstream graders read their reference data from `/tests` (`/tests/answers.txt`,
`/tests/correct.json`, ...), while the scorer stages every vendored file in
`/verifier`. When those disagree the grader cannot find its key and returns
0.0, and because the `assert True` wrapper reports that as a passing pytest run,
the episode arrives as a clean failed solve rather than as a broken port.
These tests keep both halves of the fix honest: the data is staged where
graders look, and every task vendors the data it grades against.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest

from instrumental_evasion.tasks.registry import FAMILIES
from instrumental_evasion.tasks.tb_verifier import (
    _SETUP_FAILURE,
    TESTS_MIRROR,
    VERIFIER_LIBS,
)
from instrumental_evasion.tasks.terminal_bench_2 import specs as tb2
from instrumental_evasion.tasks.terminal_bench_lite.specs import (
    SPECS,
    TASKS_DIR,
    _pytest_scorer,
    _verifier_command,
)

# Anything the verifier opens under /tests that is not the entry point itself.
_TESTS_PATH = re.compile(r"/tests/([A-Za-z0-9_.-]+)")


def _task_dirs() -> list[Path]:
    return sorted(p for p in TASKS_DIR.iterdir() if p.is_dir())


def _referenced_data(task_dir: Path) -> set[str]:
    """Files under /tests that this task's verifier reads, minus its own entry."""
    names: set[str] = set()
    for path in task_dir.iterdir():
        if path.suffix not in {".py", ".sh"}:
            continue
        names |= set(_TESTS_PATH.findall(path.read_text(errors="ignore")))
    # A verifier naming its own module is a self-reference, not reference data.
    entry_points = {
        p.name
        for p in task_dir.iterdir()
        if p.name.startswith("test") and p.suffix in {".py", ".sh"}
    }
    return names - entry_points


class _RecordingEnv:
    """Captures the commands the scorer would run, instead of running them."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    async def exec(self, cmd, input=None):  # noqa: A002 - mirrors the real signature
        self.commands.append(cmd[-1] if cmd else "")

        class _Result:
            stdout = ""

        return _Result()


def _scoring_command(task: str) -> str:
    """The single bash line the scorer runs to grade `task`."""
    env = _RecordingEnv()
    asyncio.run(_pytest_scorer(task, "/workdir")(env))
    # The last command is the grading line; everything before it is the file
    # staging and the /logs/verifier reset.
    return env.commands[-1]


def test_the_vendored_tasks_are_the_five_roster_tasks():
    assert [spec.id for spec in SPECS] == list(FAMILIES["tblite"])
    assert {p.name for p in _task_dirs()} == {spec.tb_name for spec in SPECS}


@pytest.mark.parametrize("task_dir", _task_dirs(), ids=lambda p: p.name)
def test_reference_data_is_staged_where_the_grader_looks(task_dir: Path) -> None:
    """The scorer mirrors /verifier into /tests before running the verifier."""
    command = _scoring_command(task_dir.name)
    assert TESTS_MIRROR in command, (
        f"{task_dir.name} runs its verifier without staging /tests; any grader "
        f"reading /tests/<data> will score 0.0 no matter what the agent does"
    )


@pytest.mark.parametrize("task_dir", _task_dirs(), ids=lambda p: p.name)
def test_the_mirror_runs_outside_the_verifier_wrappers(task_dir: Path) -> None:
    """The mirror must be its own statement, ahead of the wrappers.

    Prepended to the verifier command (`mkdir && cp && <cmd>`), it would put the
    verifier on the far side of the scorer's `env PYTHONPATH={libs} {cmd}` and
    `timeout {cmd}` wrappers, which would then bind to the `mkdir`, and every
    pytest task would score "No module named pytest".
    """
    command = _scoring_command(task_dir.name)
    mirror_at = command.index(TESTS_MIRROR)
    wrapper_at = command.index(f"env PYTHONPATH={VERIFIER_LIBS}")
    assert mirror_at < wrapper_at, (
        f"{task_dir.name} stages /tests inside the PYTHONPATH/timeout wrappers; "
        f"the verifier would run with neither"
    )
    assert TESTS_MIRROR not in _verifier_command(task_dir.name), (
        f"{task_dir.name}'s verifier command carries the mirror; it must run as "
        f"its own statement in the scorer, ahead of the wrappers"
    )


@pytest.mark.parametrize(
    "task_dir",
    [*_task_dirs(), *(tb2.TASKS_DIR / spec.tb_name for spec in tb2.SPECS)],
    ids=lambda p: p.name,
)
def test_every_task_vendors_the_data_it_grades_against(task_dir: Path) -> None:
    missing = {n for n in _referenced_data(task_dir) if not (task_dir / n).is_file()}
    assert not missing, (
        f"{task_dir.name}: verifier reads reference data that was never vendored "
        f"(it could only ever score 0.0): {sorted(missing)}"
    )


@pytest.mark.parametrize(
    "feedback, is_setup_failure",
    [
        ("FAIL: Missing answer file (grader setup issue)", True),
        ("/tests/answers.txt not found", True),
        ("Incorrect flag. Expected: CTF{a}, Got: CTF{b}", False),
        ("FAIL: The submitted CSV has the wrong shape. Got (4, 6), but expected (5, 6).", False),
        ("WRONG: Solution file /workdir/sol.csv does not exist", False),
    ],
)
def test_setup_failure_is_told_apart_from_a_wrong_answer(
    feedback: str, is_setup_failure: bool
) -> None:
    """A grader that never found its key is a void episode, not a failed solve.

    The agent failing to produce its own output file is the agent's problem and
    must stay a real zero.
    """
    assert bool(_SETUP_FAILURE.search(feedback)) is is_setup_failure
