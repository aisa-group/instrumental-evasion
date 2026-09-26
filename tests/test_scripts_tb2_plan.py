"""Terminal-Bench 2 build plans resolve for each of the paper's nine tasks.

No image is built and no network is touched: the plan is rendered from the
vendored verifier files with a placeholder docker ref and checked for shape.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves fields via sys.modules
    spec.loader.exec_module(module)
    return module


tb2 = _load("build_tb2_image")


def test_paper_roster_is_the_nine_tasks():
    assert len(tb2.PAPER_TASKS) == 9
    # Every paper task has vendored verifier files.
    for task in tb2.PAPER_TASKS:
        assert (tb2.VENDORED / task / "test_outputs.py").is_file()


@pytest.mark.parametrize("task", ["bn-fit-modify", "build-pmars", "cobol-modernization",
                                  "constraints-scheduling", "distribution-search",
                                  "fix-code-vulnerability", "log-summary-date-ranges",
                                  "multi-source-data-merger", "sqlite-db-truncate"])
def test_plan_renders_and_is_valid_bash(task, tmp_path):
    out = tmp_path / f"tb2-{task}.sif"
    script = tb2.render(task, f"placeholder/{task}:tag", out, apt="", runs=[])
    assert task in script
    assert f"placeholder/{task}:tag" in script
    assert "verifier runtime" in script
    # Every requirement list carries pytest so the scorer can collect.
    reqs = tb2.verifier_requirements(task)
    assert any(r.lower().startswith("pytest") for r in reqs)
    # The generated script is syntactically valid bash.
    if shutil.which("bash"):
        plan = tmp_path / "plan.sh"
        plan.write_text(script)
        subprocess.run(["bash", "-n", str(plan)], check=True)


def test_scipy_is_pinned_for_bn_fit_modify():
    reqs = tb2.verifier_requirements("bn-fit-modify")
    assert "scipy==1.15.3" in reqs
    assert "pandas" in reqs


def test_fix_code_vulnerability_uses_test_sh_pins():
    reqs = tb2.verifier_requirements("fix-code-vulnerability")
    assert "pytest==8.4.1" in reqs
    # bottle is the agent's own file, never a dependency to install.
    assert not any(r.lower().startswith("bottle") for r in reqs)


def test_non_paper_task_is_rejected():
    with pytest.raises(SystemExit):
        tb2.main(["fix-git", "--docker-ref", "x", "--print-plan"])


def test_list_prints_the_nine(capsys):
    assert tb2.main(["--list"]) == 0
    printed = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert sorted(printed) == sorted(tb2.PAPER_TASKS)
