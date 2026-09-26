"""`TBTaskSpec`: one guardrail task, as data, with no framework attached.

A spec holds the only things that define a task -- the image, the prompt, the
policy id and the scorer -- and imports nothing from a runner, so every
scaffold and runtime runs literally the same task object.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol


class SandboxLike(Protocol):
    """The sandbox surface a scorer or a setup body is allowed to use.

    Structural, not nominal: each runtime's sandbox satisfies it without a
    shared base class, which is what lets one `score_fn` run on every runtime.
    Deliberately narrow -- exec, read, write. A scorer that needs more than this
    is reaching into one runtime's internals and will not port.
    """

    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = ...,
        cwd: str | None = ...,
        env: dict[str, str] | None = ...,
        user: str | None = ...,
        timeout: int | None = ...,
        timeout_retry: bool = ...,
        concurrency: bool = ...,
    ) -> Any: ...

    async def read_file(self, file: str, text: bool = ...) -> Any: ...

    async def write_file(self, file: str, contents: str | bytes) -> None: ...


IMAGE_DIR_ENV = "IE_IMAGE_DIR"
DEFAULT_IMAGE_DIR = "~/.cache/instrumental-evasion/images"


def image_dir() -> str:
    """Directory holding the task SIF images built by `scripts/build_images.sh`.

    A task's own `image_env` variable overrides the path of that one image.
    """
    value = os.environ.get(IMAGE_DIR_ENV, "").strip() or DEFAULT_IMAGE_DIR
    return os.path.expandvars(os.path.expanduser(value))


# A scorer body receives the sandbox and returns (solved, answer, metadata).
ScoreFn = Callable[[Any], Awaitable[tuple[bool, str, dict]]]

# A setup body receives the sandbox and seeds per-episode state (e.g. writes the
# task's fixtures into the workdir). Returns nothing.
SetupFn = Callable[[Any], Awaitable[None]]


@dataclass(frozen=True)
class Sidecar:
    """A second, network-reachable container held alive for the whole episode.

    It hosts a service the task reaches over the network (e.g. the ToolSandbox
    device-state service or the authored worker supervisor). On Apptainer the
    runner starts it as a long-lived background `apptainer exec` -- unprivileged
    apptainer often has no `instance start` (it needs a /proc remount), so a
    backgrounded exec that lives for the episode is the stateless equivalent --
    bound to a per-episode localhost port, and stops it after scoring. The agent
    reaches it over the host loopback that apptainer shares by default. On
    Docker it runs on a shared bridge as `target_env`.

    Both `start` and `flag_read` are argv WITHOUT the apptainer prefix -- the
    runner wraps them in a contained exec of `image()`. `{port}` in `start` is
    filled in at launch. `flag_read` reads a value baked static into the image,
    so scoring never depends on the live service being up.
    """

    image_env: str  # env var overriding the target image path
    tb_name: str  # default image basename: tb-<tb_name>.sif
    # Docker registry ref (e.g. "marshw/cmd_injection:arm64-amd64"), used by the
    # Docker runtime instead of the SIF. None means this target has no Docker
    # image, so the Docker runtime cannot run it.
    docker_image: str | None = None
    start: tuple[str, ...] = ()  # Apptainer: argv that starts the service; {port} substituted
    flag_read: tuple[str, ...] = ()  # argv that prints a static flag to the scorer
    scheme: str = "http"  # how the prompt addresses the service
    # The port the service listens on in its own container. On Docker the target
    # runs its image's default CMD and the agent reaches it at
    # `target_env:<service_port>` over a shared bridge, so no port rewriting is
    # needed; on Apptainer the service is rebound to an ephemeral localhost port
    # (`start` does that) and this is unused.
    service_port: int = 8080
    # Docker: the command to run instead of the image's default CMD, or None to
    # use the image default (the usual case -- the image already starts its
    # service). Never has a {port} hole: Docker keeps the native port.
    docker_cmd: tuple[str, ...] | None = None
    # When true, the runtime creates a fresh control token for each episode.
    # It replaces ``{control_token}`` in the start command. The token remains in
    # the trusted runner and sidecar. ``request_json`` sends it as a bearer
    # token, so scorer-only endpoints are not available to the agent sandbox.
    control_auth: bool = False

    def image(self) -> str:
        value = os.environ.get(self.image_env, "").strip()
        default = os.path.join(image_dir(), f"tb-{self.tb_name}.sif")
        return os.path.expandvars(os.path.expanduser(value or default))


@dataclass(frozen=True)
class TBTaskSpec:
    """One task, in both its control and monitored arms."""

    id: str  # e.g. "tb2_build_pmars"
    tb_name: str  # e.g. "build-pmars", the upstream task name
    image_env: str  # env var overriding the image path
    prompt: str  # the task prompt (instruction.md, verbatim, for Terminal-Bench)
    policy: str  # guardrail spec id used in the monitored arm
    key_step: str  # human-readable description of the blocked action
    score_fn: ScoreFn  # checks the end state in the sandbox
    workdir: str = "/app"  # container working dir (some images use /workspace)
    # Network mode for the sandbox: "blackhole" (default), "proxy", or "none".
    # `network` applies to both arms; `monitored_network` overrides it in the
    # monitored arm, which is how a policy can get a system-level counterpart
    # (e.g. a real network namespace) while the control arm keeps the proxy.
    network: str = "blackhole"
    monitored_network: str | None = None
    # Public hosts that the task itself needs when its effective network mode is
    # "proxy". CLI scaffolds add these hosts to their provider-only allowlist for
    # this episode. A "none" or "blackhole" arm never receives them.
    task_egress_hosts: tuple[str, ...] = ()
    # Extra container env injected by the harness (never by the agent), passed as
    # the 4th sandbox-config field (e.g. ToolSandbox's no_proxy for its sidecar).
    container_env: dict[str, str] | None = None
    # Optional per-episode setup, run after the container is confirmed healthy.
    setup_fn: SetupFn | None = None
    # Optional task-specific system prompt, used in place of the runner's
    # generic one (the guardrail notice is still appended).
    system_prompt: str | None = None
    # Extra key/values merged into the episode metadata (e.g. scorer versions).
    extra_metadata: dict | None = None
    # Docker registry ref for the AGENT image, used by the Docker runtime
    # instead of the SIF from image(). None means no Docker image is known, so
    # `--runtime docker` cannot run this task.
    docker_image: str | None = None
    # A network target held alive alongside the agent sandbox for the episode.
    # The runner starts it before setup and stops it after scoring. A setup
    # function can use the trusted handle at `sandbox.sidecar`.
    sidecar: "Sidecar | None" = None
    # SIF basename without the .sif suffix, when it is not `tb-<tb_name>`. The
    # TB-2 family uses `tb2-<task>`: some of its tasks share an upstream name
    # with a Terminal-Bench 1 task (e.g. sqlite-db-truncate), and resolving to
    # the other generation's image would run a different task under this one's
    # prompt and scorer.
    image_basename: str | None = None

    def network_for(self, *, monitored: bool) -> str:
        if monitored and self.monitored_network:
            return self.monitored_network
        return self.network

    def task_egress_hosts_for(self, *, monitored: bool) -> tuple[str, ...]:
        """Return task hosts only for an arm configured to use the real proxy."""
        if self.network_for(monitored=monitored) != "proxy":
            return ()
        return self.task_egress_hosts

    def image(self) -> str:
        value = os.environ.get(self.image_env, "").strip()
        base = self.image_basename or f"tb-{self.tb_name}"
        default = os.path.join(image_dir(), f"{base}.sif")
        return os.path.expandvars(os.path.expanduser(value or default))
