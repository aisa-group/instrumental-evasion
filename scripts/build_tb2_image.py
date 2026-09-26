#!/usr/bin/env python3
"""Build an Apptainer SIF for a Terminal-Bench 2 task, ready to be SCORED.

Terminal-Bench 2 publishes a prebuilt image per task (`environment.docker_image`
in task.toml), so there is no Dockerfile to replay -- the rootfs is pulled and
the vendored verifier is staged into it. Everything after the rootfs exists is
the same problem the TBLite builder solves, so the verifier-runtime staging is
lifted from there rather than rewritten.

Why the staging is not optional: the scorer runs the vendored verifier INSIDE
the container, and the ways an image cannot run it are not obvious.

  * debian/ubuntu-slim bases may ship no python at all;
  * a system python3 can ship WITHOUT pip, so an install silently no-ops and the
    image looks fine until scoring reports "No module named pytest" --
    indistinguishable from a failed solve;
  * a verifier's third-party imports (pandas, scipy, ...) are present when
    terminal-bench grades WITH internet and absent in a sealed container, and a
    ModuleNotFoundError at collection time also scores like a failed solve.

All three are the same false zero, so the build FAILS LOUDLY instead of
publishing an unscoreable image: every requirement is probed by distribution
name after installation, and a missing one aborts.

pytest and the verifier's deps go to /opt/verifier-libs, on PYTHONPATH only for
the scorer -- the agent's own environment is untouched, so a policy like
no_package_install still means something.

Usage:
    scripts/build_tb2_image.py <task> --upstream <tb2-clone> [--out SIF] [--apt "pkg pkg"]
    scripts/build_tb2_image.py <task> --docker-ref <ref> --print-plan
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tblite_image import _standalone_python_hint  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
VENDORED = REPO / "src/instrumental_evasion/tasks/terminal_bench_2/tasks"

# The paper's nine Terminal-Bench 2 tasks. The builder handles only these; the
# vendored verifier and the resolved image basename exist for exactly this set.
PAPER_TASKS = (
    "bn-fit-modify", "build-pmars", "cobol-modernization", "constraints-scheduling",
    "distribution-search", "fix-code-vulnerability", "log-summary-date-ranges",
    "multi-source-data-merger", "sqlite-db-truncate",
)

# Standard-library and in-image names that must NOT be pip-installed. `bottle`
# is the AGENT'S file in fix-code-vulnerability -- installing the real bottle
# would score the agent against a package it never edited.
_NOT_A_DEP = {
    "pathlib", "subprocess", "re", "os", "json", "hashlib", "sys", "time",
    "shutil", "datetime", "csv", "collections", "itertools", "math", "string",
    "tempfile", "textwrap", "typing", "unittest", "warnings", "glob", "io",
    "struct", "binascii", "sqlite3", "configparser", "pytest", "bottle",
}

# import name -> distribution name, where they differ.
_DIST = {"skimage": "scikit-image", "PIL": "Pillow", "yaml": "PyYAML",
         "cv2": "opencv-python-headless", "sklearn": "scikit-learn"}

_IMPORT = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
_PIP = re.compile(r"(?:uv\s+)?pip\s+install\s+([^\n]*)")

# Verifier dependencies whose LATEST release is broken for what a verifier here
# actually calls. Pinned narrowly, with the evidence, because an unpinned
# install silently turns a broken library into a failed solve. scipy: recent
# releases raise inside `scipy.stats.kstest` (ndtr() argument count), so
# bn-fit-modify's distribution test fails while its other tests pass, which
# reads as a nearly-right answer rather than an unusable verifier. 1.15.3 was
# checked against the same call and returns a p-value normally.
_PINS = {"scipy": "scipy==1.15.3"}


def verifier_requirements(task: str) -> list[str]:
    """Everything the vendored verifier needs, from its imports AND its test.sh.

    Imports are the load-bearing half: TB-2 verifiers mostly do NOT pip-install
    their own dependencies, because upstream grades inside the task image with
    internet available. A sealed container makes an unstaged import a zero that
    reads like a failed solve.
    """
    d = VENDORED / task
    reqs: list[str] = []
    for test in sorted(d.glob("test_*.py")):
        for name in _IMPORT.findall(test.read_text(errors="replace")):
            if name in _NOT_A_DEP:
                continue
            reqs.append(_DIST.get(name, name))
    sh = d / "test.sh"
    if sh.is_file():
        text = re.sub(r"\\\s*\n\s*", " ", sh.read_text(errors="replace"))
        for m in _PIP.finditer(text):
            for tok in m.group(1).split():
                tok = tok.strip("\"'")
                if not tok or tok.startswith(("-", "$", ".", "/")):
                    continue
                if tok in ("&&", "||", ";", "install", "pip", "uv"):
                    continue
                reqs.append(tok)
    seen: dict[str, str] = {}
    for r in reqs:
        seen.setdefault(re.split(r"[=<>!~\[]", r, 1)[0].lower(), r)
    out = list(seen.values())
    if not any(r.lower().startswith("pytest") for r in out):
        out = ["pytest", *out]
    return [_PINS.get(re.split(r"[=<>!~\[]", r, 1)[0].lower(), r) for r in out]


def docker_ref(task: str, upstream: Path) -> str:
    toml = upstream / task / "task.toml"
    if not toml.is_file():
        raise SystemExit(f"{task}: no task.toml at {toml}")
    m = re.search(r'^\s*docker_image\s*=\s*"([^"]+)"', toml.read_text(), re.M)
    if not m:
        raise SystemExit(f"{task}: task.toml declares no environment.docker_image")
    return m.group(1)


def _apt_block(apt: str) -> str:
    if not apt:
        return 'echo "    nothing to bake"'
    # dpkg install needs root and fakeroot is unavailable unprivileged, so
    # download the .debs and unpack them with `dpkg-deb -x`, which lays files
    # into the rootfs without running maintainer scripts. `apt-get download`
    # does NOT resolve dependencies, so a metapackage like build-essential lands
    # an empty shell and the image looks built while gcc is absent. apt-cache
    # depends --recurse is the resolution step; the filters drop the alternatives
    # and virtual lines that have no downloadable .deb behind them.
    return (
        'apptainer exec --writable \\\n'
        '    --env http_proxy="${http_proxy:-}" --env https_proxy="${https_proxy:-}" \\\n'
        '    "$SANDBOX" bash -lc \'\n'
        '        set -e\n'
        '        export DEBIAN_FRONTEND=noninteractive\n'
        '        apt-get update\n'
        f'        PKGS=$(apt-cache depends --recurse --no-recommends --no-suggests \\\n'
        '            --no-conflicts --no-breaks --no-replaces --no-enhances -q '
        f'{apt} \\\n'
        '            | grep "^[a-zA-Z0-9]" | sort -u)\n'
        # An unresolvable name makes apt-cache print nothing, and `echo "" | wc
        # -l` is 1, so a naive count reports "1 package" and the build publishes
        # an image WITHOUT the package it was asked to bake. Abort here instead.
        '        [ -n "$PKGS" ] || { echo "APT COULD NOT RESOLVE THE REQUESTED PACKAGES" >&2; exit 1; }\n'
        '        echo "    resolved $(echo $PKGS | wc -w) package(s)"\n'
        # Only the MISSING ones. Unpacking a .deb that is already installed
        # rewrites files the running toolchain is using -- unpacking libc6 over
        # itself mid-loop kills the dpkg-deb doing the unpacking.
        '        NEW=""\n'
        '        for pkg in $PKGS; do\n'
        '            dpkg-query -s "$pkg" 2>/dev/null | grep -q "^Status:.*ok installed" || NEW="$NEW $pkg"\n'
        '        done\n'
        '        [ -n "$NEW" ] || { echo "    every dependency already present"; exit 0; }\n'
        '        echo "    downloading $(echo $NEW | wc -w) missing package(s)"\n'
        '        apt-get download $NEW 2>/dev/null || true\n'
        '        ls *.deb >/dev/null 2>&1 || { echo "no .debs downloaded" >&2; exit 1; }\n'
        # `dpkg-deb -x` writes a plain directory wherever the .deb carries one,
        # which on a usr-merged image REPLACES the /lib -> usr/lib symlink and
        # takes every library under it out of existence. Piping the filesystem
        # tarball through tar with --keep-directory-symlink lays the files down
        # THROUGH the existing symlinks instead; --fsys-tarfile normalises the
        # compression for us.
        '        for deb in *.deb; do\n'
        '            dpkg-deb --fsys-tarfile "$deb" | tar --keep-directory-symlink -xf - -C /\n'
        '        done\n'
        '        echo "    unpacked $(ls *.deb | wc -l) .deb(s)"\n'
        '        cd / && rm -rf "$tmp" /var/lib/apt/lists/*\n'
        '    \''
    )


def render(task: str, ref: str, out_sif: Path, apt: str, runs: list[str] | None = None) -> str:
    reqs = verifier_requirements(task)
    req_args = " ".join(shlex.quote(r) for r in reqs)
    names = sorted({re.split(r"[=<>!~\[]", r, 1)[0].strip().lower() for r in reqs})
    names_literal = ", ".join('"%s"' % n for n in names)
    src_py = _standalone_python_hint()
    q = shlex.quote(str(out_sif))
    return TEMPLATE.format(
        task=task, ref=ref, apt=apt or "(none)", apt_block=_apt_block(apt),
        run_block=_run_block(runs or []),
        reqs=" ".join(reqs), req_args=req_args, names_literal=names_literal,
        src_py=src_py, q=q, out_sif=out_sif,
    )


def _run_block(runs: list[str]) -> str:
    """Build-time commands inside the sandbox, where the network still exists.

    The escape hatch for anything the honest route fetches that is not a Debian
    package (a CRAN package, a Debian SOURCE package). Baking it at build time
    keeps the task's key step (fitting, compiling) exactly where it was and
    removes only the download the sealed sandbox cannot do. Each command runs
    with `set -e`, so a failed fetch fails the BUILD rather than publishing an
    image that silently cannot be solved.
    """
    if not runs:
        return 'echo "    nothing to run"'
    lines = []
    for cmd in runs:
        lines.append('apptainer exec --writable --no-home \\')
        lines.append('    --env http_proxy="${http_proxy:-}" --env https_proxy="${https_proxy:-}" \\')
        lines.append('    --env no_proxy="${no_proxy:-}" \\')
        lines.append('    "$SANDBOX" bash -lc ' + shlex.quote(
            "set -e; export TMPDIR=/var/tmp HOME=/root DEBIAN_FRONTEND=noninteractive; " + cmd))
    return "\n".join(lines)


TEMPLATE = r"""#!/usr/bin/env bash
# Generated by scripts/build_tb2_image.py for {task}. Do not edit in place.
set -euo pipefail

SCRATCH="${{IE_BUILD_SCRATCH:-/tmp}}"
[[ -d "$SCRATCH" && -w "$SCRATCH" ]] || {{ echo "scratch not writable: $SCRATCH" >&2; exit 1; }}
WORK="$(mktemp -d "$SCRATCH/tb2-{task}-XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
# Cache and tmp on LOCAL disk: an apptainer cache on a networked filesystem
# takes a lock that never returns, and the build hangs with no output at all.
export APPTAINER_CACHEDIR="$WORK/cache" APPTAINER_TMPDIR="$WORK/tmp" TMPDIR="$WORK/tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$TMPDIR"
SANDBOX="$WORK/rootfs"

echo "==> [1/5] pulling {ref} into a sandbox"
apptainer build --sandbox "$SANDBOX" docker://{ref}

echo "==> [2/5] baking apt packages: {apt}"
{apt_block}

echo "==> [2b/5] build-time fetches"
{run_block}

echo "==> [3/5] verifier runtime"
echo "    requirements: {reqs}"
PYPROBE='import sys; sys.path.insert(0, "/opt/verifier-libs"); import importlib.metadata as _m; import pytest; [_m.distribution(_n) for _n in [{names_literal}]]; print(pytest.__version__)'
vpy_ok=0
if apptainer exec --writable --no-home "$SANDBOX" sh -c "command -v python3" >/dev/null 2>&1; then
    apptainer exec --writable --no-home \
        --env http_proxy="${{http_proxy:-}}" --env https_proxy="${{https_proxy:-}}" \
        "$SANDBOX" sh -c 'export TMPDIR=/var/tmp HOME=/root PIP_USER=0 PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1; python3 -m pip install --target /opt/verifier-libs {req_args}' >/dev/null 2>&1 || true
    if apptainer exec --writable --no-home "$SANDBOX" python3 -c "$PYPROBE" >/dev/null 2>&1; then
        vpy_ok=1; echo "    image python3 can run the verifier from /opt/verifier-libs"
    else
        echo "    image python3 cannot; staging a standalone interpreter"
    fi
else
    echo "    image has no python3; staging a standalone interpreter"
fi
if [[ "$vpy_ok" != "1" ]]; then
    SRC_PY="{src_py}"
    [[ -x "$SRC_PY/bin/python3" ]] || {{ echo "no standalone python to stage (SRC_PY=$SRC_PY); set IE_STANDALONE_PYTHON" >&2; exit 1; }}
    mkdir -p "$SANDBOX/opt/verifier-runtime"
    cp -a "$SRC_PY/." "$SANDBOX/opt/verifier-runtime/"
    apptainer exec --writable --no-home \
        --env http_proxy="${{http_proxy:-}}" --env https_proxy="${{https_proxy:-}}" \
        "$SANDBOX" sh -c 'export TMPDIR=/var/tmp HOME=/root; /opt/verifier-runtime/bin/python3 -m pip install --target /opt/verifier-libs {req_args}' 2>&1 | tail -2
    # pip/ensurepip/venv are stripped so the staged interpreter cannot be turned
    # into a package installer against no_package_install.
    rm -rf "$SANDBOX/opt/verifier-runtime/lib/"python*/site-packages/pip* \
           "$SANDBOX/opt/verifier-runtime/lib/"python*/site-packages/setuptools* \
           "$SANDBOX/opt/verifier-runtime/lib/"python*/ensurepip \
           "$SANDBOX/opt/verifier-runtime/lib/"python*/venv \
           "$SANDBOX/opt/verifier-runtime/bin/"pip* 2>/dev/null || true
    apptainer exec --writable --no-home "$SANDBOX" /opt/verifier-runtime/bin/python3 -c "$PYPROBE" >/dev/null 2>&1 \
        || {{ echo "VERIFIER RUNTIME BROKEN: staged interpreter cannot import every requirement" >&2; exit 1; }}
    echo "    standalone interpreter staged, pip stripped"
fi

echo "==> [4/5] rootfs integrity check"
for probe in /bin/sh /bin/ls /bin/cat; do
    [[ -e "$SANDBOX$probe" ]] || {{ echo "ROOTFS CORRUPTED: $probe missing" >&2; exit 1; }}
done
apptainer exec --writable --no-home "$SANDBOX" /bin/sh -c true 2>/dev/null \
    || {{ echo "ROOTFS CORRUPTED: /bin/sh will not execute" >&2; exit 1; }}
echo "    rootfs intact"

echo "==> [5/5] packing and publishing"
apptainer build "$WORK/image.sif" "$SANDBOX"
mkdir -p "$(dirname {q})"
cp "$WORK/image.sif" {q}.tmp.$$
mv -f {q}.tmp.$$ {q}
echo "published {out_sif}"
apptainer exec --contain --no-home {q} sh -lc 'echo container-ok' || true
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", nargs="?", help="one of the paper's nine TB-2 tasks (see --list)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--upstream", type=Path,
                    default=Path(os.environ["TB2_REPO"]) if os.environ.get("TB2_REPO") else None,
                    help="local clone of terminal-bench-2 (or set $TB2_REPO)")
    ap.add_argument("--docker-ref", default=None,
                    help="task image ref, instead of reading it from the clone's task.toml")
    ap.add_argument("--apt", default="", help="extra apt packages to bake in")
    ap.add_argument("--run", action="append", default=[],
                    help="a command to run inside the sandbox at build time, when "
                         "the network still exists (repeatable)")
    ap.add_argument("--print-plan", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the paper's TB-2 tasks and exit")
    args = ap.parse_args(argv)

    if args.list:
        print("\n".join(PAPER_TASKS))
        return 0
    if not args.task:
        ap.error("a task is required (or pass --list)")
    if args.task not in PAPER_TASKS:
        raise SystemExit(f"{args.task!r} is not one of the paper's TB-2 tasks: {', '.join(PAPER_TASKS)}")

    image_dir = os.path.expanduser(
        os.environ.get("IE_IMAGE_DIR") or "~/.cache/instrumental-evasion/images")
    out = args.out or Path(image_dir) / f"tb2-{args.task}.sif"

    if args.docker_ref:
        ref = args.docker_ref
    elif args.upstream:
        ref = docker_ref(args.task, args.upstream)
    else:
        raise SystemExit("give --upstream (or $TB2_REPO), or --docker-ref for a dry plan")

    script = render(args.task, ref, out, args.apt, args.run)
    if args.print_plan:
        print(script)
        return 0
    return subprocess.run(["bash", "-s"], input=script, text=True).returncode


if __name__ == "__main__":
    sys.exit(main())
