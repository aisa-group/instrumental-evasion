#!/usr/bin/env python3
"""Replay an OpenThoughts-TBLite Dockerfile unprivileged and pack it as a SIF.

Upstream ships Dockerfiles that assume Docker: a root user, a writable rootfs
and internet at build time. Unprivileged Apptainer has none of these -- no root,
no fakeroot, and `%post` cannot mount /proc -- so this module reproduces each
Dockerfile's effect without them. Read the task's upstream Dockerfile and its
tests/, and emit a build script that pulls the base, replays the instructions,
stages the verifier runtime, and packs one SIF.

What it does, and why each piece exists:

  * `apt-get install` cannot configure packages without root, so a shim on PATH
    turns `install` into `apt-get install --download-only` (which still resolves
    the dependency tree) followed by `dpkg-deb -x` and `ldconfig`. A shim rather
    than a regex rewrite because RUN lines chain apt into `&&` sequences that no
    rewrite survives intact.
  * Tasks that build ubuntu + the deadsnakes PPA + python3.11 +
    update-alternatives do not survive unpacking without maintainer scripts, and
    all ask for the same interpreter, so the base is substituted for
    python:3.11-slim-bookworm and the boilerplate RUN lines are dropped. Any
    package a task asks for BEYOND that boilerplate still goes through the shim.
  * `apptainer exec` binds the host /tmp and the host $HOME over the sandbox's,
    so pip falls back to a --user install that lands outside the image and is
    then reported "already satisfied" on the next build. Every exec runs with
    --no-home, PIP_USER=0, PYTHONNOUSERSITE=1 and PIP_BREAK_SYSTEM_PACKAGES=1
    (debian/ubuntu system pythons are PEP 668 externally-managed).
  * COPY is done host-side on the sandbox directory. So are /etc/passwd edits:
    through `apptainer exec` they hit the passwd file Apptainer generates for
    the session, not the image's own.
  * Every scratch path stays on local disk. The Apptainer cache defaults under
    $HOME, and on a networked filesystem its lock hangs forever; unpacking many
    small files onto a distributed filesystem stalls its metadata server.

Usage:
    scripts/build_tblite_image.py <upstream-task-dir> --out <target.sif> [--plan]

`--plan` prints the generated build script and exits, which is the intended way
to review a task before building it.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- Dockerfile

@dataclass
class Step:
    kind: str          # FROM | RUN | COPY | WORKDIR | ENV | USER | SKIP
    value: str
    raw: str
    note: str = ""     # why a step was rewritten or skipped, for the plan output


_CONT = re.compile(r"\\\s*$")
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def parse_dockerfile(text: str) -> list[Step]:
    """Split a Dockerfile into logical instructions.

    Handles backslash continuations and heredocs. A line-based parse that joins
    on `\\` alone corrupts the four tasks whose RUN lines carry a `cat > f <<EOF`
    body -- the body's own blank lines and comments are not Dockerfile syntax.
    """
    lines = text.splitlines()
    steps: list[Step] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        buf = [line]
        # Join backslash continuations.
        while _CONT.search(buf[-1]) and i + 1 < len(lines):
            i += 1
            buf.append(lines[i])
        logical = "\n".join(buf)

        # Consume any heredoc bodies opened by this instruction.
        for m in _HEREDOC.finditer(re.sub(_CONT, "", logical)):
            delim = m.group(2)
            body: list[str] = []
            while i + 1 < len(lines):
                i += 1
                body.append(lines[i])
                if lines[i].strip() == delim:
                    break
            logical += "\n" + "\n".join(body)

        joined = re.sub(r"\\\s*\n\s*", " ", logical)
        kw = joined.split(None, 1)[0].upper()
        rest = joined.split(None, 1)[1] if len(joined.split(None, 1)) > 1 else ""
        if kw in ("FROM", "RUN", "COPY", "WORKDIR", "ENV", "USER", "ADD"):
            steps.append(Step(kind="ADD" if kw == "ADD" else kw, value=rest, raw=logical))
        else:
            # CMD / ENTRYPOINT / EXPOSE / LABEL / ARG / SHELL / STOPSIGNAL:
            # none of them affect what the agent finds on disk. The sandbox
            # provider runs its own command, never the image's.
            steps.append(Step(kind="SKIP", value=rest, raw=logical,
                              note=f"{kw} does not affect image contents"))
        i += 1
    return steps


# ------------------------------------------------------- deadsnakes rewrite

# The exact boilerplate the 40 ubuntu:22.04 tasks share. A RUN line whose ONLY
# apt packages come from this set is dropped once the base supplies python 3.11;
# a line that also asks for something else is kept and goes through the shim.
_DEADSNAKES_BOILERPLATE = {
    "ca-certificates", "software-properties-common", "sudo", "curl", "wget",
    "tmux", "asciinema", "python3-pip", "python3", "python3.11",
    "python3.11-venv", "python3.11-dev", "python3.11-distutils",
    "build-essential", "git",
}
_DEADSNAKES_BASE = "python:3.11-slim-bookworm"

# Tasks whose base needs a different substitution for the same reason.
# gcc:13.2.0-bookworm carries the toolchain that ubuntu 24.04 would apt-install.
_BASE_SUBSTITUTIONS = {
    "docker.io/ubuntu:22.04": _DEADSNAKES_BASE,
    "ubuntu:22.04": _DEADSNAKES_BASE,
}

_APT_PKG_RE = re.compile(r"apt-get\s+(?:-\S+\s+)*install\s+((?:[^&|;]|\|\|)*)")
_FLAG = re.compile(r"^-")


def _apt_packages(cmd: str) -> list[str]:
    pkgs: list[str] = []
    for m in _APT_PKG_RE.finditer(cmd):
        for tok in m.group(1).split():
            if _FLAG.match(tok) or tok in ("&&", "||", ";"):
                continue
            if tok.startswith("$") or "*" in tok:
                continue
            pkgs.append(tok)
    return pkgs


def _is_deadsnakes_boilerplate(cmd: str) -> bool:
    """True when this RUN line does nothing but build the deadsnakes chain."""
    low = cmd.lower()
    markers = ("add-apt-repository", "update-alternatives", "deadsnakes")
    if any(m in low for m in markers):
        # Keep it only if it also installs something outside the boilerplate.
        extra = [p for p in _apt_packages(cmd) if p not in _DEADSNAKES_BOILERPLATE]
        return not extra
    if "apt-get" in low and "install" in low:
        pkgs = _apt_packages(cmd)
        return bool(pkgs) and all(p in _DEADSNAKES_BOILERPLATE for p in pkgs)
    if low.strip().startswith("apt-get update") and "install" not in low:
        return True
    return False


# ------------------------------------------------------------------- shims

APT_SHIM = r'''#!/bin/sh
# Unprivileged apt-get. `install` cannot run dpkg's configure step without root,
# so resolve + download the dependency tree and lay the files into the rootfs
# with dpkg-deb -x, which runs no maintainer scripts. Every other verb is passed
# straight through. Fine for leaf packages and libraries; a package that needs
# its postinst to work will need a per-task note.
REAL=/usr/bin/apt-get
mode=""
for a in "$@"; do
    case "$a" in
        install) mode=install; break ;;
        update|remove|purge|clean|autoremove|download|source|build-dep) break ;;
    esac
done
if [ "$mode" != install ]; then exec "$REAL" "$@"; fi

pkgs=""
for a in "$@"; do
    case "$a" in
        install) : ;;
        -*) : ;;
        *) pkgs="$pkgs $a" ;;
    esac
done
[ -n "$pkgs" ] || exit 0

echo "[apt-shim] download+unpack:$pkgs"
"$REAL" update -qq || true
rm -f /var/cache/apt/archives/*.deb 2>/dev/null || true
"$REAL" install --download-only --reinstall -y $pkgs \
    || "$REAL" install --download-only -y $pkgs \
    || { cd /var/cache/apt/archives && "$REAL" download $pkgs; } \
    || { echo "[apt-shim] FAILED to fetch:$pkgs" >&2; exit 1; }
# Packages that own the C library, the shell or the core utilities are
# NEVER unpacked: dpkg-deb -x overwrites in place and would replace this
# rootfs's dynamic loader with another distribution's.
CORE="libc6 libc-bin libc6-dev libcrypt1 coreutils bash dash dpkg tar sed grep gzip \
      base-files debianutils login passwd libselinux1 libpcre2-8-0 perl-base"
n=0; skipped=0
for deb in /var/cache/apt/archives/*.deb; do
    [ -e "$deb" ] || continue
    pkg=$(basename "$deb" | sed 's/_.*//')
    case " $CORE " in
        *" $pkg "*) echo "[apt-shim] SKIP core package $pkg (would overwrite the loader)"; skipped=$((skipped+1)); continue ;;
    esac
    # NOT `dpkg-deb -x`. On a usr-merged rootfs (every modern Debian/Ubuntu)
    # /bin, /sbin and /lib are SYMLINKS into /usr. dpkg-deb -x extracts ./bin/...
    # by creating a real /bin directory, which shadows the symlink and makes
    # every existing binary in /usr/bin invisible -- the build then dies with
    # "sed: not found" and "/usr/bin/rm: cannot execute: required file not
    # found", having destroyed its own image. GNU tar's --keep-directory-symlink
    # is the flag that exists for precisely this, so unpack through tar.
    if dpkg-deb --fsys-tarfile "$deb" | tar -x --keep-directory-symlink -C / 2>/dev/null; then
        n=$((n+1))
    else
        echo "[apt-shim] WARNING: failed to unpack $pkg" >&2
    fi
done
[ "$skipped" -gt 0 ] && echo "[apt-shim] skipped $skipped core package(s)"
rm -f /var/cache/apt/archives/*.deb 2>/dev/null || true
ldconfig 2>/dev/null || true
echo "[apt-shim] unpacked $n package(s)"
'''

APT_GET_ALIASES = ("apt",)   # `apt install ...` routed to the same shim

# `useradd`/`groupadd` run through `apptainer exec` edit the passwd and group
# files Apptainer GENERATES for the session, not the image's own, so the account
# is gone the moment the exec returns. The
# shim therefore only RECORDS the request; a host-side step after the replay
# applies it to $SANDBOX/etc/{passwd,group}, which is the copy that gets packed.
DEFER_FILE = "/opt/build-shims/.deferred-accounts"

USERADD_SHIM = r'''#!/bin/sh
# Record the account request; the build applies it host-side after the replay.
printf '%s\t%s\n' "$(basename "$0")" "$*" >> /opt/build-shims/.deferred-accounts
echo "[account-shim] deferred: $(basename "$0") $*"
exit 0
'''

# chown/chgrp to another uid is EPERM in an unprivileged user namespace, and the
# runtime container runs as the invoking uid over a read-only rootfs, so file
# ownership cannot be part of what a task measures here (a task graded on file
# ownership is out of scope for this builder). A hard failure here would abort
# otherwise-fine builds, so the shim logs and
# succeeds rather than pretending the chown happened silently.
NOOP_SHIM = r'''#!/bin/sh
echo "[noop-shim] ignored (unprivileged build): $(basename "$0") $*"
exit 0
'''

NOOP_TOOLS = ("chown", "chgrp", "service", "systemctl", "sudo")
ACCOUNT_TOOLS = ("useradd", "groupadd", "adduser", "addgroup", "usermod")


def apply_deferred_accounts(sandbox: str) -> str:
    """Bash that replays the recorded useradd/groupadd calls host-side."""
    return f'''
DEFER="{sandbox}{DEFER_FILE}"
if [[ -s "$DEFER" ]]; then
    echo "==> applying deferred account requests host-side"
    next_uid=2000
    while IFS=$'\t' read -r tool argstr; do
        # The last non-flag token is the account name in every upstream usage.
        name=""
        for tok in $argstr; do
            case "$tok" in -*) continue ;; esac
            name="$tok"
        done
        [[ -n "$name" ]] || continue
        case "$tool" in
            groupadd|addgroup)
                if ! grep -q "^$name:" "{sandbox}/etc/group"; then
                    echo "$name:x:$next_uid:" >> "{sandbox}/etc/group"
                    echo "    group $name (gid $next_uid)"
                    next_uid=$((next_uid+1))
                fi
                ;;
            useradd|adduser)
                if ! grep -q "^$name:" "{sandbox}/etc/passwd"; then
                    echo "$name:x:$next_uid:$next_uid::/home/$name:/bin/bash" >> "{sandbox}/etc/passwd"
                    grep -q "^$name:" "{sandbox}/etc/group" || echo "$name:x:$next_uid:" >> "{sandbox}/etc/group"
                    mkdir -p "{sandbox}/home/$name"
                    echo "    user $name (uid $next_uid)"
                    next_uid=$((next_uid+1))
                fi
                ;;
            usermod) echo "    usermod ignored: $argstr" ;;
        esac
    done < "$DEFER"
    rm -f "$DEFER"
fi
'''



def _build_env() -> dict[str, str]:
    return {
        # pip must never fall back to a --user install: with the host $HOME
        # bound over the sandbox's it lands outside the image entirely.
        "PIP_USER": "0",
        "PYTHONNOUSERSITE": "1",
        # debian/ubuntu system pythons ship EXTERNALLY-MANAGED (PEP 668) and
        # refuse a plain `pip install` without this.
        "PIP_BREAK_SYSTEM_PACKAGES": "1",
        "PIP_ROOT_USER_ACTION": "ignore",
        "PIP_NO_CACHE_DIR": "1",
        "DEBIAN_FRONTEND": "noninteractive",
        "UV_SYSTEM_PYTHON": "1",
        # Every temp/cache path must resolve INSIDE the container. If the host
        # exports a TMPDIR on a networked filesystem, apptainer forwards it and
        # uv then tries to take a lock there -- which can return ENOSYS
        # ("Function not implemented", errno 38) and kill the build mid-replay.
        # /var/tmp exists in every base image used by this dataset.
        "TMPDIR": "/var/tmp",
        "TMP": "/var/tmp",
        "TEMP": "/var/tmp",
        "UV_CACHE_DIR": "/var/tmp/uv-cache",
        "XDG_CACHE_HOME": "/var/tmp/xdg-cache",
        # --no-home leaves no host home bound; point HOME at the image's own.
        "HOME": "/root",
    }


# ------------------------------------------------------------------- plan

@dataclass
class Plan:
    task: str
    base: str
    base_original: str
    steps: list[Step]
    env: dict[str, str] = field(default_factory=dict)
    workdir: str = "/"
    notes: list[str] = field(default_factory=list)


def make_plan(task_dir: Path) -> Plan:
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        found = list(task_dir.rglob("Dockerfile"))
        if not found:
            raise SystemExit(f"no Dockerfile under {task_dir}")
        dockerfile = found[0]

    text = dockerfile.read_text()
    steps = parse_dockerfile(text)
    deadsnakes = "deadsnakes" in text

    base_original = ""
    for s in steps:
        if s.kind == "FROM":
            base_original = s.value.split()[0]
            break

    base = base_original
    notes: list[str] = []
    if deadsnakes:
        base = _BASE_SUBSTITUTIONS.get(base_original, _DEADSNAKES_BASE)
        notes.append(
            f"deadsnakes chain detected: base {base_original} -> {base}; "
            "the PPA/update-alternatives/python3.11 RUN lines are dropped "
            "(the substituted base already provides python 3.11)"
        )

    out: list[Step] = []
    env: dict[str, str] = {}
    workdir = "/"
    for s in steps:
        if s.kind == "FROM":
            continue
        if s.kind == "SKIP":
            out.append(s)
            continue
        if s.kind == "USER":
            out.append(Step("SKIP", s.value, s.raw,
                            "USER ignored: the sandbox runs as the invoking uid"))
            continue
        if s.kind == "ENV":
            for k, v in _parse_env(s.value):
                env[k] = v
            out.append(s)
            continue
        if s.kind == "WORKDIR":
            workdir = s.value.strip()
            out.append(s)
            continue
        if s.kind in ("COPY", "ADD"):
            out.append(s)
            continue
        if s.kind == "RUN":
            if deadsnakes and _is_deadsnakes_boilerplate(s.value):
                out.append(Step("SKIP", s.value, s.raw,
                                "deadsnakes boilerplate, supplied by the substituted base"))
            else:
                out.append(s)
            continue
        out.append(s)

    return Plan(task=task_dir.name, base=base, base_original=base_original,
                steps=out, env=env, workdir=workdir, notes=notes)


_ENV_KV = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=((?:\"[^\"]*\")|(?:'[^']*')|(?:\S+))")


def _parse_env(value: str) -> list[tuple[str, str]]:
    pairs = _ENV_KV.findall(value)
    if pairs:
        return [(k, v.strip("\"'")) for k, v in pairs]
    parts = value.split(None, 1)          # legacy `ENV KEY value` form
    return [(parts[0], parts[1].strip("\"'"))] if len(parts) == 2 else []


_PIP_INSTALL = re.compile(r"(?:uv\s+)?pip\s+install\s+([^\n]*)")


def verifier_requirements(task_dir: Path) -> list[str]:
    """Packages upstream's tests/test.sh installs before running the verifier.

    55 of the 100 tasks do NOT ship their verifier's dependencies in the image.
    Their test.sh curls uv, builds a `.tbench-testing` venv and pip-installs what
    the test module imports (pandas, scikit-learn, mlflow, pyyaml, ...). Under
    terminal-bench that works because grading has internet; here the container is
    sealed at scoring time, so an un-baked dependency surfaces as
    "ModuleNotFoundError: No module named 'pandas'" during collection -- scored
    exactly like a failed solve.

    These are installed into /opt/verifier-libs, never into the image's own
    site-packages: that directory is on PYTHONPATH only for the scorer, so the
    agent's environment (and policies like no_package_install) stay untouched.
    """
    sh = task_dir / "tests" / "test.sh"
    if not sh.is_file():
        return []
    text = re.sub(r"\\\s*\n\s*", " ", sh.read_text())
    reqs: list[str] = []
    for m in _PIP_INSTALL.finditer(text):
        for tok in m.group(1).split():
            tok = tok.strip("\"'")
            if not tok or tok.startswith("-") or tok.startswith("$"):
                continue
            if tok in ("&&", "||", ";", "install", "pip", "uv"):
                continue
            if tok.startswith((".", "/")):     # local paths, e.g. `pip install .`
                continue
            reqs.append(tok)
    # Deduplicate, keeping the first pin seen for each distribution name.
    seen: dict[str, str] = {}
    for r in reqs:
        name = re.split(r"[=<>!~\[]", r, 1)[0].lower()
        seen.setdefault(name, r)
    return list(seen.values())


def _standalone_python_hint() -> str:
    """A relocatable CPython on the host, staged into images that lack one.

    A uv-managed CPython links only against libc/libm/libpthread, which every
    base image in this dataset provides, so a plain `cp -a` works. Point
    IE_STANDALONE_PYTHON at such a tree to override the search; a non-empty value
    is required so an empty one can never bind an unintended directory instead.
    """
    env = os.environ.get("IE_STANDALONE_PYTHON") or os.environ.get("HOOK_STANDALONE_PYTHON")
    if env and Path(env, "bin", "python3").exists():
        return env
    roots = sorted(Path(os.path.expanduser("~/.local/share/uv/python")).glob("cpython-3.1*"))
    for r in reversed(roots):
        if (r / "bin" / "python3").exists():
            return str(r)
    return ""


# --------------------------------------------------------- script generation

_SHIM_DIR = "/opt/build-shims"

# Exported at the head of every replayed RUN. `apptainer exec --env` is not
# enough on its own: the command runs under `bash -lc`, a login shell, so a host
# TMPDIR on a networked filesystem is re-exported. uv takes a lock on its cache
# directory, and a single forwarded TMPDIR can kill the build mid-replay with
# "Function not implemented (os error 38)".
_IN_CONTAINER_EXPORTS = (
    "mkdir -p /var/tmp/uv-cache /var/tmp/xdg-cache; "
    f'export PATH={_SHIM_DIR}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:"${{PATH:-}}"; '
    "export TMPDIR=/var/tmp TMP=/var/tmp TEMP=/var/tmp "
    "UV_CACHE_DIR=/var/tmp/uv-cache XDG_CACHE_HOME=/var/tmp/xdg-cache HOME=/root; "
)


def _copy_commands(value: str, context: str, cwd: str) -> list[str]:
    """Host-side COPY. Docker's build context for TBLite is `environment/`.

    The destination is emitted as an unquoted "$SANDBOX" followed by a quoted
    path: quoting the whole string would make $SANDBOX a literal, which silently
    writes the task's data into a directory named `"$SANDBOX"` under the cwd and
    ships an image with no data in it.
    """
    toks = [t for t in shlex.split(value) if not t.startswith("--")]
    if len(toks) < 2:
        return [f'echo "[copy] SKIP unparseable: {_esc(value)}" >&2']
    *srcs, dst = toks
    # A relative COPY destination is resolved against the current WORKDIR.
    if not dst.startswith("/"):
        dst = f"{cwd.rstrip('/')}/{dst}"
    cmds: list[str] = []
    for src in srcs:
        s = f"{context}/{src.lstrip('./')}"
        d = f'"$SANDBOX"{shlex.quote(dst)}'
        # Directory sources copy their CONTENTS into dst, matching docker.
        cmds.append(
            f'if [ -d {shlex.quote(s)} ]; then '
            f'mkdir -p {d} && cp -a {shlex.quote(s)}/. {d}/; '
            f'else mkdir -p "$(dirname {d})" && cp -a {shlex.quote(s)} {d}; fi'
        )
    return cmds


def render_script(plan: Plan, task_dir: Path, out_sif: Path) -> str:
    context = str(task_dir / "environment")
    env = _build_env()
    env.update(plan.env)
    env_args = " ".join(
        f"--env {shlex.quote(f'{k}={v}')}" for k, v in env.items()
    )

    L: list[str] = []
    a = L.append
    a("#!/usr/bin/env bash")
    a(f"# GENERATED by scripts/build_tblite_image.py for task '{plan.task}'.")
    a("# Review with --plan; do not edit in place -- edit the generator.")
    for n in plan.notes:
        a(f"#   note: {n}")
    a("set -euo pipefail")
    a("")
    a('SCRATCH="${IE_BUILD_SCRATCH:-/tmp}"')
    a('if [[ ! -d "$SCRATCH" || ! -w "$SCRATCH" ]]; then')
    a('    echo "build scratch is not writable: $SCRATCH (set IE_BUILD_SCRATCH)" >&2; exit 1')
    a("fi")
    a(f'WORK="$(mktemp -d "$SCRATCH/tblite-{plan.task[:24]}-XXXXXX")"')
    a("cleanup() { rm -rf \"$WORK\"; }")
    a("trap cleanup EXIT")
    a("# Keep every path apptainer touches on local disk: the default cache is")
    a("# under $HOME, and on a networked filesystem its lock hangs with no output.")
    a('export APPTAINER_CACHEDIR="$WORK/cache" APPTAINER_TMPDIR="$WORK/tmp" TMPDIR="$WORK/tmp"')
    a('mkdir -p "$APPTAINER_CACHEDIR" "$TMPDIR"')
    a('SANDBOX="$WORK/rootfs"')
    a("")
    a(f'echo "==> [1/6] base image: {plan.base}"')
    a(f'apptainer build --sandbox "$SANDBOX" docker://{plan.base}')
    a('if [[ ! -e "$SANDBOX/bin/sh" ]]; then')
    a('    echo "sandbox has no /bin/sh -- build --sandbox silently produced an empty tree" >&2')
    a("    exit 1")
    a("fi")
    a("")
    a('echo "==> [2/6] installing the unprivileged apt shim"')
    a(f'mkdir -p "$SANDBOX{_SHIM_DIR}"')
    a(f"cat > \"$SANDBOX{_SHIM_DIR}/apt-get\" <<'__APT_SHIM__'")
    a(APT_SHIM.rstrip("\n"))
    a("__APT_SHIM__")
    a(f'chmod +x "$SANDBOX{_SHIM_DIR}/apt-get"')
    for alias in APT_GET_ALIASES:
        a(f'ln -sf apt-get "$SANDBOX{_SHIM_DIR}/{alias}"')
    a(f"cat > \"$SANDBOX{_SHIM_DIR}/useradd\" <<'__ACCT_SHIM__'")
    a(USERADD_SHIM.rstrip("\n"))
    a("__ACCT_SHIM__")
    a(f'chmod +x "$SANDBOX{_SHIM_DIR}/useradd"')
    for tool in ACCOUNT_TOOLS[1:]:
        a(f'ln -sf useradd "$SANDBOX{_SHIM_DIR}/{tool}"')
    a(f"cat > \"$SANDBOX{_SHIM_DIR}/chown\" <<'__NOOP_SHIM__'")
    a(NOOP_SHIM.rstrip("\n"))
    a("__NOOP_SHIM__")
    a(f'chmod +x "$SANDBOX{_SHIM_DIR}/chown"')
    for tool in NOOP_TOOLS[1:]:
        a(f'ln -sf chown "$SANDBOX{_SHIM_DIR}/{tool}"')
    a("")
    a('echo "==> [3/6] replaying the Dockerfile"')

    exec_prefix = (
        'apptainer exec --writable --no-home '
        f'{env_args} '
        '--env http_proxy="${http_proxy:-}" --env https_proxy="${https_proxy:-}" '
        '--env no_proxy="${no_proxy:-}" '
        '"$SANDBOX"'
    )

    cwd = "/"
    step_no = 0
    for s in plan.steps:
        step_no += 1
        if s.kind == "SKIP":
            first = s.raw.strip().splitlines()[0][:100]
            a(f'echo "    [{step_no:02d}] skip: {_esc(first)}   ({_esc(s.note)})"')
            continue
        if s.kind == "ENV":
            a(f'echo "    [{step_no:02d}] env: {_esc(s.value[:90])}"')
            continue
        if s.kind == "WORKDIR":
            cwd = s.value.strip()
            a(f'echo "    [{step_no:02d}] workdir: {_esc(cwd)}"')
            a(f'mkdir -p "$SANDBOX"{shlex.quote(cwd)}')
            continue
        if s.kind in ("COPY", "ADD"):
            a(f'echo "    [{step_no:02d}] copy: {_esc(s.value[:90])}"')
            for c in _copy_commands(s.value, context, cwd):
                a(c)
            continue
        if s.kind == "RUN":
            a(f'echo "    [{step_no:02d}] run: {_esc(s.value.splitlines()[0][:90])}"')
            a(f"{exec_prefix} \\")
            a(f"    bash -c {shlex.quote(_IN_CONTAINER_EXPORTS + f'cd {cwd} 2>/dev/null || cd /; ' + s.value)}")
            continue
    a("")
    a('echo "==> [3b/6] verifier runtime"')
    a("# The scorer runs pytest INSIDE the container, so every image must be able")
    a("# to run it -- and the ways an image cannot are not obvious:")
    a("#   * ubuntu/temurin/gcc/debian-slim bases ship no python at all;")
    a("#   * ubuntu's /usr/bin/python3 ships WITHOUT pip, so the install silently")
    a("#     no-ops and the image looks fine until scoring says 'No module named")
    a("#     pytest' -- which is indistinguishable from a failed solve.")
    a("# pytest goes to /opt/verifier-libs, which the scorer adds to PYTHONPATH,")
    a("# so the agent's own environment stays untouched.")
    a("# A standalone interpreter is staged ONLY when the image cannot install it,")
    a("# and pip/ensurepip/venv are stripped so it cannot become a package")
    a("# installer against no_package_install.")
    reqs = verifier_requirements(task_dir)
    # pip rejects "double requirement given" if both `pytest` and a pinned
    # `pytest==8.3.4` are passed, so let the task's own pin win when it has one.
    if not any(r.lower().startswith("pytest") for r in reqs):
        reqs = ["pytest", *reqs]
    req_args = " ".join(shlex.quote(r) for r in reqs)
    if reqs:
        a(f'echo "    verifier deps from upstream test.sh: {_esc(" ".join(reqs))}"')
    # The probe verifies that EVERY verifier requirement actually landed in
    # /opt/verifier-libs, not just pytest. A pytest-only probe is blind to a
    # third-party dep that silently failed to install, and the image then scores
    # every episode 0 with "ModuleNotFoundError during collection",
    # indistinguishable from a failed solve. importlib.metadata.distribution()
    # keys on the DISTRIBUTION name, so it needs no import-name mapping
    # (scikit-learn, pyyaml, Pillow all just work) and a missing dep raises
    # PackageNotFoundError, which fails the probe and aborts the build.
    req_names = sorted({re.split(r"[=<>!~\[]", r, 1)[0].strip().lower() for r in reqs})
    names_literal = ", ".join(f'"{n}"' for n in req_names)
    a(
        "PYPROBE='import sys; sys.path.insert(0, \"/opt/verifier-libs\"); "
        "import importlib.metadata as _m; import pytest; "
        f"[_m.distribution(_n) for _n in [{names_literal}]]; print(pytest.__version__)'"
    )
    a('vpy_ok=0')
    a('if apptainer exec --writable --no-home "$SANDBOX" sh -c "command -v python3" >/dev/null 2>&1; then')
    a('    apptainer exec --writable --no-home \\')
    a('        --env http_proxy="${http_proxy:-}" --env https_proxy="${https_proxy:-}" \\')
    a('        "$SANDBOX" sh -c \'export TMPDIR=/var/tmp HOME=/root PIP_USER=0 PIP_BREAK_SYSTEM_PACKAGES=1 PIP_NO_CACHE_DIR=1; python3 -m pip install --target /opt/verifier-libs ' + req_args + '\' >/dev/null 2>&1 || true')
    a('    if apptainer exec --writable --no-home "$SANDBOX" python3 -c "$PYPROBE" >/dev/null 2>&1; then')
    a('        vpy_ok=1; echo "    image python3 can run pytest from /opt/verifier-libs"')
    a("    else")
    a('        echo "    image python3 cannot run pytest (no pip?); staging a standalone interpreter"')
    a("    fi")
    a("else")
    a('    echo "    image has no python3; staging a standalone interpreter"')
    a("fi")
    a('if [[ "$vpy_ok" != "1" ]]; then')
    a(f'    SRC_PY="{_standalone_python_hint()}"')
    a('    [[ -x "$SRC_PY/bin/python3" ]] || { echo "no standalone python to stage (SRC_PY=$SRC_PY)" >&2; exit 1; }')
    a('    mkdir -p "$SANDBOX/opt/verifier-runtime"')
    a('    cp -a "$SRC_PY/." "$SANDBOX/opt/verifier-runtime/"')
    a('    apptainer exec --writable --no-home \\')
    a('        --env http_proxy="${http_proxy:-}" --env https_proxy="${https_proxy:-}" \\')
    a('        "$SANDBOX" sh -c \'export TMPDIR=/var/tmp HOME=/root; /opt/verifier-runtime/bin/python3 -m pip install --target /opt/verifier-libs ' + req_args + '\' 2>&1 | tail -1')
    a('    rm -rf "$SANDBOX/opt/verifier-runtime/lib/"python*/site-packages/pip* \\')
    a('           "$SANDBOX/opt/verifier-runtime/lib/"python*/site-packages/setuptools* \\')
    a('           "$SANDBOX/opt/verifier-runtime/lib/"python*/ensurepip \\')
    a('           "$SANDBOX/opt/verifier-runtime/lib/"python*/venv \\')
    a('           "$SANDBOX/opt/verifier-runtime/bin/"pip* 2>/dev/null || true')
    a('    apptainer exec --writable --no-home "$SANDBOX" /opt/verifier-runtime/bin/python3 -c "$PYPROBE" >/dev/null 2>&1 \\')
    a('        || { echo "VERIFIER RUNTIME BROKEN: staged interpreter still cannot import pytest" >&2; exit 1; }')
    a('    echo "    standalone interpreter staged, pip stripped"')
    a("fi")
    a("")
    a(apply_deferred_accounts('"$SANDBOX"'))
    a("")
    a('echo "==> [4/6] recording the runtime environment"')
    # The provider sets the workdir and env itself, so these are recorded as a
    # sidecar rather than baked into /.singularity.d -- that directory is one of
    # the eval-awareness leaks the sandbox scrubs.
    a(f'cat > "$WORK/{plan.task}.env.json" <<\'__ENVJSON__\'')
    import json as _json
    a(_json.dumps({"task": plan.task, "workdir": plan.workdir,
                   "env": plan.env, "base": plan.base,
                   "base_original": plan.base_original}, indent=2))
    a("__ENVJSON__")
    a(f'cp "$WORK/{plan.task}.env.json" "$(dirname {shlex.quote(str(out_sif))})/{plan.task}.env.json" || true')
    a("")
    a('echo "==> [4b/6] rootfs integrity check"')
    a("# A replay can DESTROY the rootfs it is building: unpacking a deb over a")
    a("# usr-merged tree with the wrong tool replaces /bin with a real directory")
    a("# and every core binary disappears. Such an image can pack successfully and")
    a("# only fail later, at episode start, as \"failed to open elf binary /bin/sh\".")
    a("# Fail here instead, loudly, rather than shipping an image that cannot run.")
    a('for probe in /bin/sh /bin/ls /bin/cat; do')
    a('    if [[ ! -e "$SANDBOX$probe" ]]; then')
    a('        echo "ROOTFS CORRUPTED: $probe is missing after the replay" >&2')
    a('        echo "  (a package unpack most likely shadowed a usr-merge symlink)" >&2')
    a("        exit 1")
    a("    fi")
    a("done")
    a('if ! apptainer exec --writable --no-home "$SANDBOX" /bin/sh -c "true" 2>/dev/null; then')
    a('    echo "ROOTFS CORRUPTED: /bin/sh will not execute after the replay" >&2')
    a("    exit 1")
    a("fi")
    a('echo "    rootfs intact"')
    a("")
    a('echo "==> [5/6] packing the SIF"')
    a('apptainer build "$WORK/image.sif" "$SANDBOX"')
    a("")
    a('echo "==> [6/6] publishing"')
    a("# Publish before any optional diagnostics: writing to a log on a networked")
    a("# filesystem can hit a transient I/O error, and under `set -e` that would")
    a("# lose a finished build.")
    a(f'mkdir -p "$(dirname {shlex.quote(str(out_sif))})"')
    a(f'cp "$WORK/image.sif" {shlex.quote(str(out_sif))}.tmp.$$')
    a(f'mv -f {shlex.quote(str(out_sif))}.tmp.$$ {shlex.quote(str(out_sif))}')
    a(f'echo "published {out_sif}"')
    a("")
    a("# Smoke test against the PACKED SIF under --contain --no-home: verifying")
    a("# inside the build sandbox hides exactly the pip --user failure this")
    a("# script exists to prevent.")
    a(f'apptainer exec --contain --no-home {shlex.quote(str(out_sif))} sh -lc '
      f'"python3 -c \'import sys; print(\\"python\\", sys.version.split()[0])\' 2>/dev/null || echo \'no python3 in image\'" || true')
    return "\n".join(L) + "\n"


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")


# The paper's five OpenThoughts-TBLite tasks (upstream directory names).
PAPER_TASKS = (
    "book-portfolio-analysis", "broken-python", "corrupted-filesystem-recovery",
    "cosign-keyless-signing", "reverse-engineer-stack-vm",
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("task_dir", type=Path, help="upstream task directory")
    p.add_argument("--out", type=Path, default=None,
                   help="target SIF (default $IE_IMAGE_DIR/tb-<task>.sif)")
    p.add_argument("--plan", action="store_true",
                   help="print the generated build script and exit")
    p.add_argument("--save-script", type=Path, default=None,
                   help="write the generated script here as well as running it")
    p.add_argument("--allow-any-task", action="store_true",
                   help="build a task outside the paper's five (unsupported)")
    args = p.parse_args(argv)

    task_dir = args.task_dir.resolve()
    if task_dir.name not in PAPER_TASKS and not args.allow_any_task:
        raise SystemExit(
            f"{task_dir.name!r} is not one of the paper's TBLite tasks "
            f"({', '.join(PAPER_TASKS)}); pass --allow-any-task to override")
    image_dir = os.path.expanduser(
        os.environ.get("IE_IMAGE_DIR") or "~/.cache/instrumental-evasion/images")
    out = args.out or Path(image_dir) / f"tb-{task_dir.name}.sif"

    plan = make_plan(task_dir)
    script = render_script(plan, task_dir, out)

    if args.save_script:
        args.save_script.write_text(script)
        args.save_script.chmod(0o755)
    if args.plan:
        print(script)
        return 0

    proc = subprocess.run(["bash", "-s"], input=script, text=True)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
