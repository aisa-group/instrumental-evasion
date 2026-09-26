"""OpenThoughts-TBLite build plans render from an upstream Dockerfile.

No image is built and no network is touched. Upstream Dockerfiles are not
vendored in the release, so the plan machinery is exercised against small
fixture task directories laid out the way an upstream clone is.
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


tbl = _load("build_tblite_image")


def _task_dir(root: Path, name: str, dockerfile: str, test_sh: str | None = None) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    if test_sh is not None:
        (task / "tests").mkdir()
        (task / "tests" / "test.sh").write_text(test_sh)
    return task


def test_paper_roster_is_the_five_tasks():
    assert len(tbl.PAPER_TASKS) == 5
    assert "broken-python" in tbl.PAPER_TASKS


def test_plain_base_plan(tmp_path):
    dockerfile = (
        "FROM python:3.11-slim-bookworm\n"
        "WORKDIR /app\n"
        "COPY data/ /app/data/\n"
        "RUN apt-get update && apt-get install -y e2fsprogs\n"
        "ENV FOO=bar\n"
        'CMD ["sleep", "infinity"]\n'
    )
    test_sh = "uv pip install pandas pytest==8.3.4\npython -m pytest\n"
    task = _task_dir(tmp_path, "broken-python", dockerfile, test_sh)
    (task / "environment" / "data").mkdir()
    (task / "environment" / "data" / "x.txt").write_text("hi")

    plan = tbl.make_plan(task)
    assert plan.base == "python:3.11-slim-bookworm"
    assert plan.env.get("FOO") == "bar"

    script = tbl.render_script(plan, task, tmp_path / "tb-broken-python.sif")
    assert "installing the unprivileged apt shim" in script
    assert "verifier runtime" in script
    # verifier deps come from the upstream test.sh, pytest included.
    reqs = tbl.verifier_requirements(task)
    assert "pandas" in reqs and "pytest==8.3.4" in reqs
    if shutil.which("bash"):
        out = tmp_path / "plan.sh"
        out.write_text(script)
        subprocess.run(["bash", "-n", str(out)], check=True)


def test_deadsnakes_base_is_substituted(tmp_path):
    dockerfile = (
        "FROM ubuntu:22.04\n"
        "RUN apt-get update && apt-get install -y software-properties-common\n"
        "RUN add-apt-repository ppa:deadsnakes/ppa && apt-get install -y python3.11\n"
        "RUN apt-get install -y e2fsprogs\n"
    )
    task = _task_dir(tmp_path, "corrupted-filesystem-recovery", dockerfile)
    plan = tbl.make_plan(task)
    assert plan.base == "python:3.11-slim-bookworm"
    assert plan.base_original == "ubuntu:22.04"
    kinds = [(s.kind, s.value) for s in plan.steps]
    # The deadsnakes boilerplate RUN lines are dropped; the extra package stays.
    dropped = [v for k, v in kinds if k == "SKIP" and "deadsnakes" in v.lower()]
    kept = [v for k, v in kinds if k == "RUN" and "e2fsprogs" in v]
    assert dropped
    assert kept


def test_roster_restriction(tmp_path):
    dockerfile = "FROM python:3.11-slim-bookworm\n"
    # A directory named outside the paper's five is refused before any build.
    _task_dir(tmp_path, "some-other-task", dockerfile)
    with pytest.raises(SystemExit):
        tbl.main([str(tmp_path / "some-other-task"), "--plan"])
    # A paper task renders its plan and returns 0 (no build under --plan).
    _task_dir(tmp_path, "broken-python", dockerfile)
    assert tbl.main([str(tmp_path / "broken-python"), "--plan"]) == 0
