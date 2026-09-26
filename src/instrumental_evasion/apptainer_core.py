"""Apptainer container mechanics, shared by every scaffold.

Nothing here runs a subprocess. It holds the part that has to be identical
whoever is driving -- the containment flags, the bind layout, the identity
scrub, the network modes and the state directories -- and hands back an argv
for the caller to execute.

`AppLayout`: the image's own workdir is copied to a host directory and bound
back over itself, plus a writable overlay for state outside the workdir.

Three constraints shape it:

* **A SIF file, not an unpacked directory.** An unpacked tree is ~30k small
  files, which is slow on shared filesystems; one SIF copies in seconds.
* **No instances.** `apptainer instance start` needs a `/proc` mount and a PID
  namespace that unprivileged hosts often do not permit, so the layout is
  stateless: each exec is its own `apptainer exec` and all state lives in host
  binds and the overlay.
* **Containment must be explicit.** By default Apptainer bind-mounts the host
  cwd and `$HOME` and inherits the host environment, so host files and API keys
  would be visible inside the container. `--contain`, `--no-home` and
  `--cleanenv` close that off.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from logging import getLogger
from pathlib import Path

logger = getLogger(__name__)

DEFAULT_WORKDIR = "/app"

# Host cwd, $HOME and the host environment are all mounted or inherited by
# default. Without these the agent can read host files and the real API keys.
CONTAINMENT_FLAGS = ("--contain", "--no-home", "--cleanenv")

# --cleanenv drops the host environment, including any outbound HTTP(S) proxy
# configuration. In the `proxy` network mode only these proxy settings are
# forwarded (see network_env); credentials are passed per-command instead.
PROXY_VARS = ("http_proxy", "https_proxy", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")

# Docker runs these images as root, whose PATH includes the sbin directories; we
# run as an ordinary uid, whose PATH does not. Without this an agent typing
# `e2fsck` gets "command not found" and concludes the tool is missing, when it is
# installed at /usr/sbin/e2fsck.
APPEND_PATH = "/usr/local/sbin:/usr/sbin:/sbin"

# apptainer/singularity inject their own APPTAINER_NAME=<image>.sif,
# APPTAINER_CONTAINER=<path>, SINGULARITY_NAME, ... into the container at runtime,
# regardless of --cleanenv. The image filename IS the task name, so those vars
# tell the agent which evaluation it is in. Wrap the in-container command so every
# APPTAINER_*/SINGULARITY_* var is unset before the real program (and anything it
# spawns, e.g. the hook) runs. The real command is passed through argv ("$@"), so
# arbitrary prompt text is never re-parsed by the shell -- no quoting hazard.
# Also drops PS1/PROMPT_COMMAND, which apptainer sets to the literal "Apptainer> "
# prompt -- another name tell. LD_LIBRARY_PATH=/.singularity.d/libs and the
# /.singularity.d directory itself still say "a container runtime" (not the task),
# and are load-bearing / structural, so they are left.
_IDENTITY_SCRUB = (
    "unset PS1 PROMPT_COMMAND 2>/dev/null; "
    "for v in $(env | sed -n "
    r"'s/^\(APPTAINER[A-Za-z0-9_]*\)=.*/\1/p;s/^\(SINGULARITY[A-Za-z0-9_]*\)=.*/\1/p'"
    "); do unset \"$v\"; done; exec \"$@\""
)


def scrub_identity(command: list[str]) -> list[str]:
    """Wrap an in-container command so the runtime's identity vars are unset first.

    Returns a `/bin/sh -c` invocation that unsets every APPTAINER_*/SINGULARITY_*
    variable and then execs `command` unchanged (via argv, so no re-quoting).
    """
    return ["/bin/sh", "-c", _IDENTITY_SCRUB, "sh", *command]


def state_root() -> str:
    """Return a local-disk directory for sandbox state."""
    override = os.environ.get("IE_STATE_DIR", "").strip()
    candidates = [override] if override else ["/tmp", tempfile.gettempdir()]
    for candidate in candidates:
        if candidate and os.path.isdir(candidate) and os.access(candidate, os.W_OK):
            return candidate
    return tempfile.gettempdir()


def verify_apptainer_available(image: str) -> None:
    """Raise a named error if the runtime or the image is missing.

    Called once per task rather than once per episode. Without it a missing
    binary surfaces only as a bare FileNotFoundError on every exec and a missing
    image as a failure inside each episode -- both of which produce a full run of
    zeros that reads like a measurement rather than like a broken harness. An
    episode is void unless the machinery was demonstrably alive.
    """
    if shutil.which("apptainer") is None:
        raise RuntimeError(
            "The 'apptainer' executable is not on PATH. Install Apptainer or "
            "choose another --runtime."
        )
    if not Path(image).exists():
        raise FileNotFoundError(
            f"Apptainer image not found: {image}. Build it with "
            "scripts/build_images.sh"
        )


# --------------------------------------------------------------------------
# network modes (the /app layout)

# Whether a task container may reach the network. OFF by default.
#
# The benchmark tasks are public: the instruction, the reference solution and
# the verifier all live in an upstream repo. With egress an agent can clone that
# repo and read the answer instead of solving the task, so the measured "solve"
# would be a lookup. No task in this set needs the network.
#
# Set TB_ALLOW_NETWORK=1 only for a task whose honest route genuinely needs the
# network.
ALLOW_NETWORK_ENV = "TB_ALLOW_NETWORK"

# Pointing the proxy at the discard port makes an attempted fetch fail at once
# with "connection refused" rather than hang until a network timeout. The agent
# learns the route is closed on its first try, and a contained episode is not
# silently converted into a timed-out one -- which would look like a monitor
# result and is not.
BLACKHOLE_PROXY = "http://127.0.0.1:9"

#   blackhole  proxy vars point at the discard port. The DEFAULT. It is an
#              ENVIRONMENT cut, not a system one: a process that sets its own
#              proxy variables (e.g. in a Python subprocess) can still reach
#              whatever the host network allows. Treat it as "no accidental
#              egress", never as containment.
#   proxy      the host's proxy settings are forwarded. For tasks whose honest
#              route genuinely needs the network, and for CLI scaffolds, whose
#              agent has to reach its vendor's API (run those behind the egress
#              allowlist).
#   none       `--net --network=none`: a real network namespace with only `lo`.
#              The proxy stops resolving, so there is no route out at any layer
#              the agent can reach. Works unprivileged (tested with apptainer
#              1.1.6).
NET_BLACKHOLE = "blackhole"
NET_PROXY = "proxy"
NET_NONE = "none"
NET_MODES = (NET_BLACKHOLE, NET_PROXY, NET_NONE)

# Flags that give a container its own empty network namespace.
ISOLATED_NET_FLAGS = ("--net", "--network=none")

# Loopback must bypass the proxy either way: some tasks have the agent start a
# local server and then talk to it, and a proxy may refuse even
# `Host: localhost:8080`.
LOOPBACK_HOSTS = "localhost,127.0.0.1,::1"

_warned_network = False


def network_allowed() -> bool:
    return os.environ.get(ALLOW_NETWORK_ENV, "").strip().lower() in ("1", "true", "yes")


def network_env(mode: str = NET_BLACKHOLE) -> dict[str, str]:
    """Proxy settings handed to the task container, for one network mode."""
    global _warned_network
    if mode == NET_NONE:
        # Nothing resolves inside an empty netns; leaving the proxy vars unset
        # keeps failures honest ("name resolution") rather than pointing the
        # agent at a discard port that does not exist here either.
        return {"no_proxy": LOOPBACK_HOSTS, "NO_PROXY": LOOPBACK_HOSTS}
    if mode == NET_PROXY or network_allowed():
        if not _warned_network:
            logger.warning(
                "Task containers can reach the network (%s). Upstream task repos "
                "are public, so solves from this run may be lookups.",
                f"network mode {mode!r}"
                if mode == NET_PROXY
                else f"{ALLOW_NETWORK_ENV} is set",
            )
            _warned_network = True
        forwarded = {key: os.environ[key] for key in PROXY_VARS if os.environ.get(key)}
        forwarded.setdefault("no_proxy", LOOPBACK_HOSTS)
        forwarded.setdefault("NO_PROXY", LOOPBACK_HOSTS)
        return forwarded
    return {
        "http_proxy": BLACKHOLE_PROXY,
        "https_proxy": BLACKHOLE_PROXY,
        "HTTP_PROXY": BLACKHOLE_PROXY,
        "HTTPS_PROXY": BLACKHOLE_PROXY,
        "no_proxy": LOOPBACK_HOSTS,
        "NO_PROXY": LOOPBACK_HOSTS,
    }


# Separator packing the task working directory into a sandbox config string,
# since different terminal-bench images use different roots (/app, /workspace).
class AppLayout:
    """The terminal-bench layout: a host-backed workdir plus a writable overlay.

    A terminal-bench image ships its whole task under its workdir and expects it
    to be writable so the agent can drop `solution.txt` there. A read-only
    `apptainer exec` cannot write the image's rootfs, and `--writable-tmpfs`
    would not survive to the next stateless exec. So at episode setup the image's
    workdir is copied to a per-episode host directory on local disk and bound
    over the workdir for every later exec: writes persist across execs and stay
    isolated per episode.

    Holds no async: `seed_command()` returns the argv, and the caller runs it.
    """

    def __init__(
        self,
        image: str,
        workdir: str = DEFAULT_WORKDIR,
        net: str = NET_BLACKHOLE,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.image = image
        self.workdir = workdir
        self.net = net
        # Per-task container env injected by the harness
        # (TBTaskSpec.container_env). Applied to every exec, below caller
        # overrides but above nothing the agent controls.
        self.extra_env = extra_env or {}
        # All state lives here, on local disk: a shared filesystem stalls on
        # the many small writes the image's workdir contains.
        self._directory = tempfile.mkdtemp(prefix="apptainer-tb-", dir=state_root())
        self._app = Path(self._directory) / "app"
        self._app.mkdir(parents=True, exist_ok=True)
        # A writable overlay over the whole rootfs. The workdir bind alone is not
        # enough: some tasks are graded on state outside the working directory
        # (broken-python repairs site-packages), and a read-only SIF refuses
        # those writes. `--overlay <dir>` works unprivileged and, unlike
        # --writable-tmpfs, survives to the next stateless exec. It does not
        # grant root: chown and mount still fail, so tasks needing real
        # privilege remain out of scope.
        self._overlay = Path(self._directory) / "overlay"
        self._overlay.mkdir(parents=True, exist_ok=True)

    def seed_command(self) -> list[str]:
        """The argv that copies the image's workdir into the host bind directory.

        Run before the first exec: binding a host directory over the workdir masks
        the image's copy, so without this the agent loses the task's own files.
        """
        return [
            "apptainer",
            "exec",
            *CONTAINMENT_FLAGS,
            "--bind",
            f"{self._app}:/seed",
            self.image,
            "bash",
            "-lc",
            # Some images do not ship the workdir at all -- the task's
            # setup_fn creates it from fixtures. A missing workdir is
            # therefore a no-op empty seed, not a failure; a cp that fails for a
            # real reason on an image that does ship content still errors.
            f"if [ -d {self.workdir} ]; then cp -a {self.workdir}/. /seed/; fi",
        ]

    def cleanup(self) -> None:
        shutil.rmtree(self._directory, ignore_errors=True)

    def host_path(self, file: str) -> Path | None:
        """Map a container path under the workdir to its host path."""
        path = Path(file if file.startswith("/") else f"{self.workdir}/{file}")
        try:
            relative = path.relative_to(self.workdir)
        except ValueError:
            return None
        return self._app / relative

    def command(
        self, cmd: list[str], cwd: str | None = None, env: dict[str, str] | None = None
    ) -> list[str]:
        args = ["apptainer", "exec", *CONTAINMENT_FLAGS]
        if self.net == NET_NONE:
            args += [*ISOLATED_NET_FLAGS]
        args += ["--overlay", str(self._overlay)]
        args += ["--env", f"APPEND_PATH={APPEND_PATH}"]
        args += ["--bind", f"{self._app}:{self.workdir}"]
        args += ["--pwd", cwd or self.workdir]
        # --cleanenv drops the host environment, including its proxy settings.
        # What goes back in is decided by network_env(): a blackholed proxy by
        # default. Caller overrides win.
        for key, value in {**network_env(self.net), **self.extra_env, **(env or {})}.items():
            args += ["--env", f"{key}={value}"]
        args += [self.image, *cmd]
        return args
