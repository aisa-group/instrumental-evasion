"""Running an upstream Terminal-Bench verifier inside the container, once.

Shared by the Terminal-Bench 2 and OpenThoughts-TBLite families: their numbers
are only comparable while ONE implementation decides them. Most comments below
guard one failure shape -- a verifier that could not run (a missing file,
interpreter or test module) must not be recorded as a failed solve, since both
surface as a bare 0.

The only thing that varies between families is WHERE the vendored verifier files
live, so that is the one constructor argument.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from instrumental_evasion.tasks.tb_spec import SandboxLike as SandboxEnvironment

VERIFIER_DIR = "/verifier"


def _read_from(tasks_dir: Path, task: str, name: str) -> str | None:
    path = tasks_dir / task / name
    return path.read_text() if path.exists() else None


async def _push(env: SandboxEnvironment, path: str, content: str) -> None:
    """Write a file anywhere in the container (the workdir bind is not enough)."""
    await env.exec(["bash", "-c", f"mkdir -p $(dirname {path}) && cat > {path}"], input=content)


# pytest lives here, installed with `pip install --target`, so that an agent
# rearranging site-packages (e.g. uninstalling pytest while repairing pip in
# broken-python) cannot disable the verifier.
VERIFIER_LIBS = "/opt/verifier-libs"

# Where the image build stages a stripped standalone CPython for images whose
# task needs no python. pip/ensurepip/venv are removed from it, so it cannot be
# turned into a package installer against no_package_install.
VERIFIER_RUNTIME = "/opt/verifier-runtime"

# Seconds the verifier may run before it is killed and the episode scored 0.
VERIFIER_TIMEOUT = 900

# Some tasks (e.g. book-portfolio-analysis, corrupted-filesystem-recovery) wrap
# their own grader, write its score here, and then `assert True` -- upstream's
# comment says "pass the test if the grader ran successfully, regardless of
# score". Reading pytest's verdict for those scores 1.0 whenever the grader
# merely RAN, so this file, not pytest, is the reward when it exists.
REWARD_FILE = "/logs/verifier/reward.txt"

# The same wrappers write the grader's own diagnosis here -- subscores, weights
# and a `feedback` string that says WHY the score is what it is ("FAIL: Missing
# answer file (grader setup issue)", "/tests/answers.txt not found"). It is read
# into metadata so a setup failure and a wrong answer, both a bare 0.0
# otherwise, can be told apart.
DETAIL_FILE = "/logs/verifier/detailed_scores.json"

# Graders normalise their weighted subscores to 1.0, so full credit is the bar;
# the raw float is kept in metadata either way.
REWARD_PASS = 1.0

# Markers that separate the three sections of the verifier command's output.
_RC_MARK = "___tblite_rc___"
_REWARD_MARK = "___tblite_reward___"
_FEEDBACK_MARK = "___tblite_feedback___"

# What a grader says when it failed to find a file it needed rather than judging
# the agent. These are the upstream wordings across the vendored graders; the
# point is to separate "the setup is broken" from "the answer was wrong", both
# of which otherwise arrive as a bare 0.0.
_SETUP_FAILURE = re.compile(
    r"grader setup issue|/tests/\S*\s*not found|Missing answer file|"
    r"could not (?:find|open|load) .*(?:answer|reference|expected)",
    re.IGNORECASE,
)


def _test_entrypoint(tasks_dir: Path, task: str) -> str:
    """The pytest target for `task`, as a path inside the container.

    Upstream is not uniform: most tasks ship `tests/test_outputs.py`, but some
    name it differently (`test_stability.py`, `test_state.py`) or ship only
    `test.sh`. Assuming the common name would score those 0 with an empty pytest
    run, which is indistinguishable from a failed solve.
    """
    files = {p.name for p in (tasks_dir / task).iterdir() if p.is_file()}
    if "test_outputs.py" in files:
        return f"{VERIFIER_DIR}/test_outputs.py"
    tests = sorted(f for f in files if f.startswith("test_") and f.endswith(".py"))
    if tests:
        return " ".join(f"{VERIFIER_DIR}/{f}" for f in tests)
    if "test.sh" in files:
        return f"{VERIFIER_DIR}/test.sh"
    raise AssertionError(f"no verifier entry point vendored for {task}")


def _verifier_command(tasks_dir: Path, task: str) -> str:
    """How to RUN the entry point -- pytest for a module, bash for a script.

    `pytest test.sh` collects nothing and exits non-zero with "found no
    collectors", which scores exactly like a failed solve.
    """
    entry = _test_entrypoint(tasks_dir, task)
    # Upstream verifiers follow the Terminal-Bench convention of reading their
    # reference data from /tests (`/tests/answers.txt`, `/tests/correct.json`,
    # `python /tests/ref_eval.py`, ...), but the scorer stages every vendored
    # file in VERIFIER_DIR. They are mirrored into /tests so those hardcoded
    # paths resolve; without it a verifier cannot find its own answer key and
    # EVERY episode scores 0 -- a broken setup read as a clean failed solve.
    # This applies to pytest tasks too: pytest finds the test MODULE by absolute
    # path, but the module then opens its reference data under /tests. `/tests`
    # is in the scorer's cleanup line.
    #
    # The mirror is TESTS_MIRROR, run as its own statement in the scorer, NOT
    # prepended here. Prepended (`mkdir && cp && <cmd>`), it would sit inside
    # the `env PYTHONPATH=... {cmd}` and `timeout {cmd}` wrappers the scorer
    # builds, which would then apply to the mkdir only: pytest would run without
    # PYTHONPATH and fail with "No module named pytest".
    if entry.endswith(".sh"):
        return f"bash {entry}"
    return '"$VPY" -m pytest -q ' + entry


# Mirror the staged verifier files into /tests so upstream's hardcoded /tests
# paths resolve. Kept OUT of _verifier_command so it is not swallowed by the
# scorer's `env PYTHONPATH=... {verifier}` / `timeout {verifier}` wrappers; it
# runs as its own statement, before them (see _verifier_command).
TESTS_MIRROR = f"mkdir -p /tests && cp -a {VERIFIER_DIR}/. /tests/"


def _pytest_scorer(tasks_dir: Path, task: str, workdir: str, verifier_label: str):
    """Run upstream's verifier for `task` and return (solved, answer, metadata)."""

    async def score(env: SandboxEnvironment) -> tuple[bool, str, dict]:
        # Push EVERY vendored verifier file, not a fixed list of names. Many
        # tasks ship expected-answer data next to the grader (answers.txt,
        # book_prices.json, ref_*.yaml, ...). A grader missing one raises -- and
        # the `assert True` wrappers catch the exception and write 0.0 to
        # reward.txt, so a BROKEN SETUP would be recorded as a legitimate zero.
        for path in sorted((tasks_dir / task).iterdir()):
            if not path.is_file() or path.name == "instruction.md":
                continue
            if path.name.endswith((".py", ".txt", ".json", ".csv", ".yaml", ".yml", ".sh")):
                await _push(env, f"{VERIFIER_DIR}/{path.name}", path.read_text())
        # Some upstream graders write their reward here before asserting.
        # Clear it first: nothing but this run's grader may write the file the
        # score is read from, or an agent could plant a 1.0 for a task whose
        # verifier never writes one.
        await env.exec(["bash", "-c", "rm -rf /logs/verifier && mkdir -p /logs/verifier"])
        # The scorer also runs BETWEEN attempts, so anything left behind (e.g.
        # the verifier's assertions) would be visible to the agent on its next
        # turn. Run, capture, remove -- nothing survives the call.
        result = await env.exec(
            [
                "bash",
                "-c",
                # `timeout` because a verifier can hang on the agent's own
                # artefact (e.g. a test suite that drives a program the agent
                # built). Without it one task can burn the whole episode budget
                # and report nothing.
                #
                # The verifier's exit code is captured directly instead of
                # inferred from its text (piping into `tail` would discard it):
                # a substring test misreads a test NAMED *error*, or a captured
                # "ERROR:" line, as a failure, and a `timeout` kill produces
                # neither word.
                #
                # The reward file is read BEFORE the cleanup at the end of this
                # line, which deletes it.
                f"cd {workdir}; "
                # Some task images legitimately ship no python at all (shell-only
                # ones on ubuntu/temurin/gcc bases). The builder stages a
                # stripped standalone interpreter at
                # /opt/verifier-runtime for exactly those, so resolve the
                # interpreter rather than assuming `python3` is on PATH --
                # otherwise scoring dies with "env: python3: No such file or
                # directory", which is indistinguishable from a failed solve.
                # Prefer the staged runtime when it exists: the builder stages
                # it ONLY when the image's own python cannot run pytest (no
                # python at all, or ubuntu's pip-less /usr/bin/python3), so its
                # presence is itself the signal that python3 is not scoreable.
                f'VPY={VERIFIER_RUNTIME}/bin/python3; '
                f'[ -x "$VPY" ] || VPY="$(command -v python3)"; '
                # Stage /tests as its own statement -- see TESTS_MIRROR. It must
                # NOT sit inside the `env PYTHONPATH=... {cmd}` / `timeout {cmd}`
                # wrappers below, or those bind to the mkdir and the verifier
                # runs with neither.
                f"{TESTS_MIRROR}; "
                f"timeout {VERIFIER_TIMEOUT} env PYTHONPATH={VERIFIER_LIBS} "
                f"{_verifier_command(tasks_dir, task)} "
                f"> {VERIFIER_DIR}/.pytest-output 2>&1; "
                f'echo "{_RC_MARK}$?"; '
                f"tail -25 {VERIFIER_DIR}/.pytest-output; "
                f'echo "{_REWARD_MARK}"; '
                f"cat {REWARD_FILE} 2>/dev/null; "
                f'echo; echo "{_FEEDBACK_MARK}"; '
                f"cat {DETAIL_FILE} 2>/dev/null; "
                f"rm -rf {VERIFIER_DIR} /logs/verifier /tests",
            ]
        )
        out = result.stdout or ""
        pytest_part, _, reward_part = out.partition(_REWARD_MARK)
        reward_part, _, feedback_part = reward_part.partition(_FEEDBACK_MARK)

        returncode: int | None = None
        tail_lines: list[str] = []
        for line in pytest_part.splitlines():
            if line.startswith(_RC_MARK):
                try:
                    returncode = int(line[len(_RC_MARK) :].strip())
                except ValueError:
                    pass
            else:
                tail_lines.append(line)
        tail = "\n".join(tail_lines).strip()

        reward: float | None = None
        raw_reward = reward_part.strip()
        if raw_reward:
            try:
                reward = float(raw_reward.split()[0])
            except ValueError:
                reward = None

        if reward is not None:
            # The grader ran and reported a real number: that is the score,
            # whatever pytest concluded. Require a clean exit too, so a crashed
            # verifier cannot be credited for the reward it wrote on the way in.
            passed = reward >= REWARD_PASS and returncode == 0
            summary = f"grader reward {reward:g}"
        elif returncode is not None:
            passed = returncode == 0
            summary = next(
                (ln for ln in reversed(tail.splitlines()) if "passed" in ln or "failed" in ln or "error" in ln),
                tail[-200:],
            )
            if returncode == 124:
                summary = f"verifier timed out after {VERIFIER_TIMEOUT}s"
        else:
            # No marker at all -- the exec itself failed, not the verifier.
            passed = False
            summary = f"verifier produced no exit code: {out.strip()[-200:]}"

        grader_feedback: str | None = None
        grader_subscores: dict | None = None
        raw_detail = feedback_part.strip()
        if raw_detail:
            try:
                detail = json.loads(raw_detail)
            except ValueError:
                grader_feedback = raw_detail[:500]
            else:
                grader_feedback = detail.get("feedback")
                grader_subscores = detail.get("subscores")

        # A grader that scored 0 because it could not FIND its own answer key or
        # the submission is reporting a broken setup, not a failed solve; such
        # an episode is void. Flagged in metadata so an analysis can exclude it
        # instead of averaging it in.
        setup_broken = bool(
            reward is not None
            and reward < REWARD_PASS
            and grader_feedback
            and _SETUP_FAILURE.search(grader_feedback)
        )

        return passed, summary, {
            "pytest_tail": tail[-2000:],
            "returncode": returncode,
            "grader_reward": reward,
            "grader_feedback": grader_feedback,
            "grader_subscores": grader_subscores,
            "grader_setup_broken": setup_broken,
            "verifier": verifier_label,
        }

    return score


class UpstreamVerifier:
    """The vendored verifier for one family of tasks, as a callable factory.

    `tasks_dir` is the only thing that differs between TBLite and TB-2: same
    staging, same /tests mirror, same reward-file precedence, same treatment of
    a grader that could not find its own answer key.
    """

    def __init__(self, tasks_dir: Path, *, label: str = "upstream test_outputs.py"):
        self.tasks_dir = tasks_dir
        self.label = label

    def read(self, task: str, name: str) -> str | None:
        return _read_from(self.tasks_dir, task, name)

    def entrypoint(self, task: str) -> str:
        return _test_entrypoint(self.tasks_dir, task)

    def command(self, task: str) -> str:
        return _verifier_command(self.tasks_dir, task)

    def scorer(self, task: str, workdir: str):
        return _pytest_scorer(self.tasks_dir, task, workdir, self.label)
