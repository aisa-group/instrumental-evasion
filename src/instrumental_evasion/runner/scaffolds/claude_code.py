"""The claude_code scaffold: the vendor CLI, monitored by a PreToolUse hook.

The CLI executes its own tools inside the container, so the interception point
cannot be a Python loop -- it is a `PreToolUse` hook running in there, which
POSTs each proposed call to the gate on the host and refuses when the gate says
BLOCK. `instrumental_evasion.hooks` already builds that container (the bound-in
interpreter, the neutralised hook package, the private HOME carrying the
subscription credential, the `--settings` injection that leaves no file for the
agent to edit); this scaffold reuses it rather than rebuilding it, and adds only
what the plugin contract needs.

The CLI authenticates either as a logged-in subscription or, for the models in
`runner.claude_openrouter`, through a host-side OpenRouter gateway that keeps
the provider key out of the container.

Two divergences from the react scaffold, both deliberate and both reported in
`describe()` so a reader of the log sees them:

* **The agent's tokens are invisible.** The CLI talks to its vendor's backend
  directly, which is what makes subscription auth work. There is no token
  limit on this scaffold; wall clock is the bound.
* **A "turn" is a CLI invocation, not a generation.** The CLI runs until it
  stops calling tools; the no-quitting policy is applied by resuming the same
  conversation with the continuation message.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from instrumental_evasion.refusal_stop import InvocationObservation, RefusalStreak
from instrumental_evasion.rollout_core import (
    continuation_message,
    refusal_context_mode,
)
from instrumental_evasion.runner.claude_openrouter import cli_environment, is_openrouter
from instrumental_evasion.runner.gate import Gate
from instrumental_evasion.runner.gate_server import GateServer, serve
from instrumental_evasion.runner.loop_stop import LoopStop, claude_tool_ids, configured_mode
from instrumental_evasion.runner.scaffolds.base import (
    STOP_AGENT,
    STOP_DESISTED,
    STOP_ERROR,
    STOP_PROVIDER_REFUSAL,
    STOP_STALLED_LOOP,
    STOP_SUBMIT,
    STOP_TIME_LIMIT,
    STOP_TURN_LIMIT,
    EpisodeContext,
    ScaffoldResult,
)
from instrumental_evasion.runner.types import ChatMessage
from instrumental_evasion.tasks.tb_spec import TBTaskSpec

DEFAULT_MODEL = "sonnet"
DEFAULT_TIMEOUT = 3600
MODAL_HOOK_PROBE_ATTEMPTS = 4


def _provider_refusal(stdout: str, observation: InvocationObservation, model: str) -> bool:
    """Recognize a complete provider refusal that did not switch models."""
    if not observation.stream_completed or observation.malformed:
        return False
    try:
        events = [json.loads(line) for line in stdout.split("\n") if line.strip()]
    except (ValueError, TypeError):
        return False
    if not events or any(not isinstance(event, dict) for event in events):
        return False
    refusals = [e for e in events if e.get("type") == "system"
                and e.get("subtype") == "model_refusal_no_fallback"]
    if len(refusals) != 1 or refusals[0].get("original_model") != model:
        return False
    session = refusals[0].get("session_id")
    if not isinstance(session, str) or not session or any(
        e.get("session_id", session) != session for e in events
    ):
        return False
    category = refusals[0].get("api_refusal_category")
    if not isinstance(category, str) or not category.strip():
        return False
    terminal = events[-1]
    if terminal.get("error") or terminal.get("errors"):
        return False
    if not (terminal.get("type") == "result" and terminal.get("is_error") is True
            and terminal.get("subtype") == "success" and terminal.get("stop_reason") == "refusal"
            and terminal.get("terminal_reason") == "api_error" and terminal.get("api_error_status") is None):
        return False
    synthetic_refusals = 0
    for event in events:
        if event.get("type") == "error" or event.get("subtype") in {
            "api_error", "model_refusal_fallback", "model_fallback", "model_consent_fallback",
        }:
            return False
        if event.get("type") != "assistant":
            continue
        message = event.get("message", {})
        if not isinstance(message, dict) or message.get("model") not in {model, "<synthetic>"}:
            return False
        if any(isinstance(b, dict) and b.get("type") == "fallback" for b in message.get("content", [])):
            return False
        if message.get("model") == "<synthetic>":
            details = message.get("stop_details")
            if not (event.get("is_api_error_message") is True and event.get("error") == "invalid_request"
                    and message.get("stop_reason") == "refusal" and isinstance(details, dict)
                    and details.get("type") == "refusal"
                    and details.get("category") == refusals[0].get("api_refusal_category")):
                return False
            synthetic_refusals += 1
        elif event.get("is_api_error_message") or event.get("error"):
            return False
    return synthetic_refusals == 1


def _background_completion_indices(events: list[dict[str, Any]]) -> set[int]:
    """Recognize completed turns drained by one CLI process after notifications.

    Each turn needs its own initialization and successful result in the same
    session. Keep every assistant message for the invocation's refusal check.
    An incomplete turn, unexplained duplicate result, or session change fails.
    """
    session: str | None = None
    active = False
    notified = False
    assistant_seen = False
    completions: list[int] = []
    for index, event in enumerate(events):
        kind = event.get("type")
        if kind == "system" and event.get("subtype") in {"init", "task_notification"}:
            identifier = event.get("session_id")
            if not isinstance(identifier, str) or not identifier or (
                session is not None and identifier != session
            ):
                return set()
            session = identifier
            if event["subtype"] == "task_notification":
                notified = True
            else:
                if active:
                    return set()
                active = True
                assistant_seen = False
        elif kind == "assistant":
            if not active:
                return set()
            assistant_seen = True
        elif kind == "user" and not active:
            return set()
        elif kind == "result":
            turns = event.get("num_turns")
            if (
                not active or event.get("session_id") != session
                or event.get("subtype") != "success" or event.get("is_error") is not False
                or type(turns) is not int or turns < 0
                or not isinstance(event.get("result"), str)
                or (turns == 0 and (event["result"] != "" or assistant_seen
                                   or event.get("stop_reason") is not None))
                or (turns > 0 and not assistant_seen)
            ):
                return set()
            active = False
            completions.append(index)
            if len(completions) > 64:
                return set()
    if active or not notified or len(completions) < 2:
        return set()
    return set(completions[:-1])


def cli_version(binary: Path) -> str:
    """Return a CLI's self-reported version without starting an agent."""
    try:
        result = subprocess.run(
            [str(binary), "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return (result.stdout or result.stderr).strip() or "unknown"


class CLIScaffoldBase:
    """What the two CLI scaffolds share: the gate endpoint and the egress proxy.

    Both run the agent OUTSIDE this process, so both need the gate reachable
    over loopback and both need egress restricted to their vendor's API --
    unrestricted host networking would let the agent fetch a task's answer from
    its public upstream repository.
    """

    name = "cli"

    def __init__(self, *, proxy_port: int = 0, egress_allow: tuple[str, ...] | None = None) -> None:
        self._proxy = None
        self._proxy_bind_port = proxy_port
        self._proxy_port = proxy_port
        self._egress_allow = egress_allow
        self._gate_server: GateServer | None = None
        # The gate of the episode currently running. The server is started once
        # and resolves this per request: each episode has its own Gate and its
        # own decision log, and a server pinned to the first episode's gate
        # would silently file every later episode's calls under the first.
        self._gate: Gate | None = None

    def _effective_egress_allow(self, ctx: EpisodeContext) -> tuple[str, ...]:
        """Combine provider hosts with hosts this task needs in this condition."""
        from instrumental_evasion.hooks import egress_proxy

        provider_hosts = self._egress_allow or egress_proxy.DEFAULT_ALLOW
        task_hosts = ctx.spec.task_egress_hosts_for(monitored=ctx.monitored)
        return tuple(dict.fromkeys((*provider_hosts, *task_hosts)))

    def _ensure_proxy(self, ctx: EpisodeContext) -> str:
        """Start an episode-specific proxy and egress log. Returns its URL."""
        from instrumental_evasion.hooks import egress_proxy

        self._close_proxy()
        self._proxy = egress_proxy.serve(
            self._proxy_bind_port,
            self._effective_egress_allow(ctx),
            ctx.outdir / "egress.jsonl",
        )
        self._proxy_port = int(self._proxy.server_address[1])
        return f"http://127.0.0.1:{self._proxy_port}"

    def _close_proxy(self) -> None:
        if self._proxy is not None:
            self._proxy.shutdown()
            self._proxy.server_close()
            self._proxy = None

    def _ensure_gate_server(self, ctx: EpisodeContext) -> str:
        """Expose THIS episode's gate on loopback, over one long-lived server."""
        self._gate = ctx.gate
        if self._gate_server is None:
            self._gate_server = serve(lambda: self._current_gate())
        return self._gate_server.url

    def _current_gate(self) -> Gate:
        if self._gate is None:  # pragma: no cover - only reachable out of order
            raise RuntimeError("the gate endpoint was queried outside an episode")
        return self._gate

    def close(self) -> None:
        if self._gate_server is not None:
            self._gate_server.stop()
            self._gate_server = None
        self._gate = None
        self._close_proxy()

    @staticmethod
    def _rung_prompts(max_turns: int) -> list[str | None]:
        """The base prompt, then the continuation message once per unsolved turn.

        The message is the shared one, so every scaffold sends the same words.
        It asks the agent to continue and never suggests how.
        """
        if max_turns < 1:
            return []
        return [None, *[continuation_message()] * (max_turns - 1)]


class ClaudeCodeScaffold(CLIScaffoldBase):
    """Claude Code inside the task image, gated by a PreToolUse hook."""

    name = "claude_code"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        cli_timeout: int = DEFAULT_TIMEOUT,
        proxy_port: int = 0,
        egress_allow: tuple[str, ...] | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if reasoning_effort not in {None, "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported Claude reasoning effort")
        if is_openrouter(model):
            egress_allow = ("openrouter.ai",)
        super().__init__(proxy_port=proxy_port, egress_allow=egress_allow)
        self.model = model
        self.timeout = cli_timeout
        self.reasoning_effort = reasoning_effort

    def describe(self) -> dict[str, Any]:
        from instrumental_evasion.hooks.bootstrap import HOOK_DEADLINE_SECONDS
        from instrumental_evasion.hooks.deploy import ALLOWED_TOOLS, CLAUDE_BIN, claude_tool_scope
        from instrumental_evasion.hooks.monitor_client import (
            CLASSIFICATION_DEADLINE_SECONDS,
            REMOTE_DEADLINE_SECONDS,
        )

        return {
            "agent": self.name,
            "agent_model": self.model,
            "agent_reasoning_effort": self.reasoning_effort,
            "claude_tool_scope": claude_tool_scope(),
            "claude_available_tools": (
                ALLOWED_TOOLS.split() if claude_tool_scope() == "configured" else "cli_default"
            ),
            "auth": "openrouter_host_gateway" if is_openrouter(self.model) else "subscription",
            "provider_credential_agent_visible": not is_openrouter(self.model),
            "provider_credential_exposure_approved": True,
            "provider_credential_retained": False,
            "turn_limit_enforced": False,
            # The CLI talks to its vendor's backend directly, so no token count
            # reaches us. Saying so is the point: a reader must not take a
            # missing token limit for an unlimited one.
            "token_limit_enforced": False,
            "time_limit_enforced": True,
            "gate_transport": "http_pretooluse_hook",
            "hook_runtime_version": "isolated_bootstrap_v1",
            "hook_deadline_seconds": HOOK_DEADLINE_SECONDS,
            "monitor_deadline_seconds": CLASSIFICATION_DEADLINE_SECONDS,
            "gate_client_deadline_seconds": REMOTE_DEADLINE_SECONDS,
            "claude_stream_parser_version": "background_notification_v3",
            "claude_model_fallback_policy": "invalidate_observed_fallback_v1",
            "claude_cli_settings": {"switchModelsOnFlag": False},
            "claude_provider_refusal_classifier": "structured_no_fallback_v1",
            "refusal_renderer": "claude_pretooluse_hook",
            "scaffold_version": cli_version(CLAUDE_BIN),
            "turn_unit": "cli_invocation",
            "refusal_stop_supported": True,
            "refusal_turn_unit": "cli_invocation",
            "loop_stop_mode": configured_mode(),
            "loop_stop_scope": "local_normal_claude_only",
        }

    def modal_image(self, image: Any, *, root: str = "") -> Any:
        """Bake the static hook runtime into a content-addressed Modal layer."""
        from instrumental_evasion.hooks.deploy import (
            CLAUDE_BIN,
            HOOK_PYTHON,
            minimal_src,
        )

        # Each parallel runner owns its staging tree. A shared path lets one
        # process delete files while another Modal image builder reads them.
        hook_src = minimal_src(
            Path(tempfile.mkdtemp(prefix="cli-hook-src-"))
        )
        return image.add_local_dir(
            HOOK_PYTHON, f"{root}/opt/pyrt", copy=True
        ).add_local_dir(hook_src, f"{root}/opt/envtools", copy=True).add_local_file(
            CLAUDE_BIN, f"{root}/usr/local/bin/claude", copy=True
        ).run_commands(f"chmod 0555 {root}/usr/local/bin/claude")

    async def preflight(
        self, spec: TBTaskSpec, *, refresh_credentials: bool = True
    ) -> dict[str, Any]:
        """Check the binaries this scaffold binds into the container.

        The per-image hook probe is NOT here: it needs a live gate endpoint, so
        it runs at the start of `run()` where one exists.
        """
        from instrumental_evasion.hooks.deploy import CLAUDE_BIN, HOOK_PYTHON

        if not CLAUDE_BIN.exists():
            return {"ok": False, "reason": f"claude binary not found at {CLAUDE_BIN}"}
        # Check the interpreter the probe binds and execs, not just the
        # directory: a misconfigured HOOK_PYTHON can resolve to an existing
        # directory without bin/python3, and every episode would then void as
        # a dead hook.
        if not (HOOK_PYTHON / "bin" / "python3").exists():
            return {
                "ok": False,
                "reason": f"standalone python not found at {HOOK_PYTHON}/bin/python3",
            }
        if not Path(spec.image()).exists():
            return {"ok": False, "reason": f"image not found: {spec.image()}"}
        if is_openrouter(self.model):
            try:
                cli_environment(self.model)
            except ValueError as error:
                return {"ok": False, "reason": str(error)}
            return {"ok": True, "auth": "openrouter_host_gateway", "claude": str(CLAUDE_BIN)}
        try:
            from instrumental_evasion.subscription import anthropic_credential

            credential = anthropic_credential(refresh=refresh_credentials)
        except Exception as error:
            return {"ok": False, "reason": f"Claude credential unavailable: {error}"}
        if credential.expired:
            return {
                "ok": False,
                "reason": "Claude credential expired; run a Claude CLI authentication check",
            }
        return {"ok": True, "claude": str(CLAUDE_BIN), "python": str(HOOK_PYTHON)}

    def _episode(self, ctx: EpisodeContext):
        from instrumental_evasion.hooks.deploy import Episode

        directory = getattr(ctx.sandbox, "_directory", None)
        if directory is None:
            raise RuntimeError(
                "the local Claude scaffold requires disposable sandbox storage"
            )
        return Episode(
            spec=ctx.spec,
            policy=ctx.gate.policy or "control",
            model=self.model,
            monitor_url=self._ensure_gate_server(ctx),
            outdir=ctx.outdir,
            timeout=min(self.timeout, ctx.limits.time_limit_seconds),
            proxy=self._ensure_proxy(ctx),
            env={"RUNTIME_REFUSAL_CONTEXT": refusal_context_mode(),
                 **(cli_environment(self.model) if is_openrouter(self.model) else
                    ({"CLAUDE_CODE_OAUTH_TOKEN": os.environ["CLAUDE_CODE_OAUTH_TOKEN"]}
                     if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") else {}))},
            structured_output=True,
            reasoning_effort=self.reasoning_effort,
            home=Path(directory) / "claude-home",
            append_system_prompt=ctx.system_prompt,
        )

    @staticmethod
    async def _stage_modal_cli(sandbox: Any) -> None:
        """Reuse an identical protected CLI or stage and verify its exact bytes."""
        from instrumental_evasion.hooks.deploy import CLAUDE_BIN

        with CLAUDE_BIN.open("rb") as source:
            expected = hashlib.file_digest(source, "sha256").hexdigest()
        check = [
            "/opt/pyrt/bin/python3", "-I", "-c",
            "import hashlib,pathlib; p=pathlib.Path('/usr/local/bin/claude'); "
            "s=p.stat(); print(hashlib.file_digest(p.open('rb'),'sha256').hexdigest() "
            "if s.st_uid == 0 and s.st_mode & 0o222 == 0 "
            "and s.st_mode & 0o555 == 0o555 else 'unprotected')",
        ]
        present = await sandbox.exec(check, timeout=30)
        if present.success and present.stdout.strip() == expected:
            return
        await sandbox.stage_path(CLAUDE_BIN, "/usr/local/bin/claude", read_only=True)
        staged = await sandbox.exec(check, timeout=30)
        if not staged.success or staged.stdout.strip() != expected:
            raise RuntimeError("The staged Claude CLI does not match the protected source binary.")

    @staticmethod
    async def _stage_modal_home(ctx: EpisodeContext, episode: Any) -> None:
        """Upload Claude's private home without retaining credentials locally."""
        from instrumental_evasion.hooks.deploy import CONTAINER_HOME

        with tempfile.TemporaryDirectory(prefix="cli-home-") as staging:
            home = episode._home(Path(staging) / "home")
            await ctx.sandbox.stage_path(home, CONTAINER_HOME)

    @staticmethod
    async def _probe_modal_hook(
        ctx: EpisodeContext, env: dict[str, str], payload: str
    ) -> tuple[bool, Any, str]:
        """Wait until the remote hook reaches a healthy monitor decision."""
        from instrumental_evasion.hooks.deploy import CONTAINER_HOOK, CONTAINER_PY

        last_probe = None
        last_trace = ""
        for attempt in range(MODAL_HOOK_PROBE_ATTEMPTS):
            before = await ctx.sandbox.remote_gate_count()
            probe_env = {**env, "RUNTIME_LOG": "/tmp/.runtime_probe.jsonl"}
            last_probe = await ctx.sandbox.exec(
                [CONTAINER_PY, "-I", CONTAINER_HOOK],
                input=payload,
                cwd=ctx.spec.workdir,
                env=probe_env,
                timeout=180,
            )
            try:
                last_trace = await ctx.sandbox.read_file("/tmp/.runtime_probe.jsonl")
            except FileNotFoundError:
                last_trace = ""
            healthy = await ctx.sandbox.remote_gate_healthy_since(before)
            if (
                last_probe.returncode in (0, 2)
                and bool(last_probe.stdout.strip())
                and healthy
            ):
                return True, last_probe, str(last_trace)
            if attempt < MODAL_HOOK_PROBE_ATTEMPTS - 1:
                await asyncio.sleep(min(2 ** attempt, 4))
        assert last_probe is not None
        return False, last_probe, str(last_trace)

    async def run(self, ctx: EpisodeContext) -> ScaffoldResult:
        loop_stop = LoopStop(mode=configured_mode())
        if loop_stop.mode != "off" and hasattr(ctx.sandbox, "start_remote_gate"):
            raise ValueError("The loop-stop extension supports local Claude only.")
        if hasattr(ctx.sandbox, "start_remote_gate"):
            if is_openrouter(self.model):
                raise ValueError("Claude OpenRouter supports local Apptainer only.")
            return await self._run_modal(ctx)
        from instrumental_evasion.hooks.deploy import probe_hook

        started = time.monotonic()
        episode = self._episode(ctx)

        # Fail-open is invisible in results, so the hook is proved alive in THIS
        # image before the episode counts: an episode is void unless the
        # monitor was demonstrably alive.
        before_probe = len(ctx.gate.records())
        probe = probe_hook(ctx.spec, episode)
        probe_records = ctx.gate.records()[before_probe:]
        if probe["hook_alive"] and (
            not probe_records or any(r["decision_source"] == "monitor_error" for r in probe_records)
        ):
            probe = {**probe, "hook_alive": False, "probe_error": "no_healthy_gate_decision"}
        if not probe["hook_alive"]:
            return ScaffoldResult(
                stop_reason=STOP_ERROR,
                error="hook did not run in this image; the monitor would be off",
                seconds=round(time.monotonic() - started, 1),
                extra={"void_reason": "hook_dead", **probe},
            )

        prompts = self._rung_prompts(ctx.limits.max_turns)
        base = ctx.prompt
        stop_reason = STOP_AGENT
        rungs = 0
        last: dict[str, Any] = {}
        error: str | None = None
        messages: list[ChatMessage] = []
        refusal_streak = RefusalStreak()
        for index, prompt in enumerate(prompts):
            if time.monotonic() - started >= ctx.limits.time_limit_seconds:
                stop_reason = STOP_TIME_LIMIT
                break
            try:
                # The first prompt is None; use ctx.prompt, which the runner has
                # already filled with the sidecar's challenge_address. Episode's
                # default, spec.prompt, still holds the raw
                # "{challenge_address}" placeholder.
                gate_before = len(ctx.gate.records())
                last = episode.run(ctx.sandbox, prompt=prompt or base, resume=index > 0)
            except subprocess.TimeoutExpired:
                # The turn hit its time limit -- the episode was cut off, not
                # stopped by the agent. That is a different finding.
                stop_reason = STOP_TIME_LIMIT
                rungs = index
                break
            rungs = index
            messages.append(ChatMessage("user", prompt or base))
            observation = self._invocation_from_stdout(last.get("stdout", ""))
            invocation_records = ctx.gate.records()[gate_before:]
            loop_decision = loop_stop.observe(
                observation.assistant_text, invocation_records,
                tool_ids=claude_tool_ids(last.get("stdout", "")),
                valid_stream=observation.valid and last.get("rc") in (0, None),
            )
            messages.extend(
                ChatMessage("assistant", text) for text in observation.assistant_texts
            )
            if loop_decision == "monitor_error":
                stop_reason = STOP_ERROR
                error = "Monitor error; no further continuation was sent."
                break
            if observation.stream_error == "Claude reported a model fallback":
                # A successful score cannot establish success for the configured
                # model after the CLI switches to another model.
                stop_reason = STOP_ERROR
                error = observation.stream_error
                break
            if last.get("rc") == 1 and _provider_refusal(last.get("stdout", ""), observation, self.model):
                stop_reason = STOP_PROVIDER_REFUSAL
                break
            solved, _, _ = await ctx.score()
            if solved:
                stop_reason = STOP_SUBMIT
                break
            if last.get("rc") not in (0, None):
                stop_reason = STOP_ERROR
                break
            if not observation.valid:
                stop_reason = STOP_ERROR
                error = f"invalid Claude refusal event stream: {observation.invalid_reason}"
                break
            if refusal_streak.observe(
                observation.assistant_text,
                tool_call_count=len(ctx.gate.records()) - gate_before,
                final_assistant_text=observation.assistant_texts[-1] if observation.assistant_texts else None,
            ):
                stop_reason = STOP_DESISTED
                break
            if loop_decision == "stalled_loop":
                stop_reason = STOP_STALLED_LOOP
                break
            if index == len(prompts) - 1:
                # Every continuation was used and the task is still unsolved.
                stop_reason = STOP_TURN_LIMIT

        return ScaffoldResult(
            stop_reason=stop_reason,
            turns=rungs + 1,
            seconds=round(time.monotonic() - started, 1),
            messages=messages,
            error=error,
            extra={
                "rungs_used": rungs,
                "cli_returncode": last.get("rc"),
                "hook_alive": True,
                "provider_refusal": stop_reason == STOP_PROVIDER_REFUSAL,
                **loop_stop.metadata(),
                **(
                    {"void_reason": "monitor_error" if loop_stop.reason == "monitor_error"
                     else "refusal_stream_invalid"}
                    if error is not None
                    else {}
                ),
                **refusal_streak.as_metadata(turn_unit="cli_invocation"),
            },
        )

    async def _run_modal(self, ctx: EpisodeContext) -> ScaffoldResult:
        from instrumental_evasion.hooks.deploy import (
            ALLOWED_TOOLS,
            CONTAINER_HOME,
            CONTAINER_SRC,
            Episode,
            claude_tool_arguments,
        )

        started = time.monotonic()
        gate_url, gate_token = await ctx.sandbox.start_remote_gate(ctx.gate)
        await ctx.sandbox.allow_hosts(list(self._effective_egress_allow(ctx)))
        # Refresh before Episode._home copies the credential into the remote
        # private home; otherwise a long-lived login can stage an expired token.
        from instrumental_evasion.subscription import anthropic_credential

        credential = anthropic_credential()
        if credential.expired:
            return ScaffoldResult(
                stop_reason=STOP_ERROR,
                error="Claude credential expired before Modal staging",
                seconds=round(time.monotonic() - started, 1),
                extra={"void_reason": "credential_expired"},
            )
        episode = Episode(
            spec=ctx.spec,
            policy=ctx.gate.policy or "control",
            model=self.model,
            monitor_url=gate_url,
            outdir=ctx.outdir,
            timeout=min(self.timeout, ctx.limits.time_limit_seconds),
            append_system_prompt=ctx.system_prompt,
        )
        await self._stage_modal_cli(ctx.sandbox)
        await self._stage_modal_home(ctx, episode)
        await ctx.sandbox.prepare_user(CONTAINER_HOME, ctx.spec.workdir)
        env = {
            "HOME": CONTAINER_HOME,
            "PYTHONPATH": CONTAINER_SRC,
            "RUNTIME_URL": gate_url,
            # The deployed neutral hook rewrites HOOK_MONITOR_* -> RUNTIME_*.
            "RUNTIME_TOKEN": gate_token,
            "RUNTIME_PROFILE": "standard",
            "RUNTIME_REFUSAL_CONTEXT": refusal_context_mode(),
            "APPEND_PATH": "/usr/local/sbin:/usr/sbin:/sbin",
            "NO_PROXY": "*",
            "no_proxy": "*",
        }
        if ctx.spec.container_env:
            env.update(ctx.spec.container_env)

        payload = json.dumps(
            {"session_id": "probe", "tool_name": "Bash", "tool_input": {"command": "true"}}
        )
        alive, probe, probe_trace = await self._probe_modal_hook(ctx, env, payload)
        if not alive:
            return ScaffoldResult(
                stop_reason=STOP_ERROR,
                error="hook did not run in this Modal sandbox; the monitor would be off",
                seconds=round(time.monotonic() - started, 1),
                extra={
                    "void_reason": "hook_dead", "hook_alive": False,
                    "rc": probe.returncode, "stdout": probe.stdout[:400],
                    "stderr": probe.stderr[-400:], "probe_trace": str(probe_trace)[-1200:],
                },
            )

        # Fail fast when the exact task image cannot reach Anthropic or the
        # staged subscription is unusable. Do not record command output because
        # auth status can contain identifying account data.
        network_probe = await ctx.sandbox.exec(
            [
                "/opt/pyrt/bin/python3",
                "-c",
                (
                    "import urllib.error,urllib.request; "
                    "u='https://api.anthropic.com/'; "
                    "\ntry: urllib.request.urlopen(u,timeout=15)"
                    "\nexcept urllib.error.HTTPError: pass"
                    "\nprint('ok')"
                ),
            ],
            cwd=ctx.spec.workdir,
            env=env,
            timeout=30,
            user="1000",
        )
        auth_probe = await ctx.sandbox.exec(
            ["/usr/local/bin/claude", "auth", "status", "--json"],
            cwd=ctx.spec.workdir,
            env=env,
            timeout=30,
            user="1000",
        )
        if network_probe.returncode != 0 or auth_probe.returncode != 0:
            failed = "network" if network_probe.returncode != 0 else "subscription"
            return ScaffoldResult(
                stop_reason=STOP_ERROR,
                error=f"Claude {failed} preflight failed in the Modal sandbox",
                seconds=round(time.monotonic() - started, 1),
                extra={
                    "void_reason": "claude_preflight_failed",
                    "hook_alive": True,
                    "network_probe_rc": network_probe.returncode,
                    "auth_probe_rc": auth_probe.returncode,
                },
            )

        async def invoke(prompt: str, resume: bool, timeout: int):
            command = ["/usr/local/bin/claude", "-p", prompt]
            if resume:
                command.append("--continue")
            command += [
                "--model", self.model,
                "--permission-mode", "acceptEdits",
                "--allowedTools", ALLOWED_TOOLS,
                *claude_tool_arguments(),
                "--settings", episode._settings_json(),
                "--output-format", "stream-json",
                "--verbose",
            ]
            if self.reasoning_effort is not None:
                command += ["--effort", self.reasoning_effort]
            if ctx.system_prompt:
                command += ["--append-system-prompt", ctx.system_prompt]
            return await ctx.sandbox.exec(
                command, cwd=ctx.spec.workdir, env=env, timeout=timeout, user="1000"
            )

        prompts = self._rung_prompts(ctx.limits.max_turns)
        stop_reason = STOP_AGENT
        rungs = 0
        returncode: int | None = None
        error: str | None = None
        messages: list[ChatMessage] = []
        refusal_streak = RefusalStreak()
        for index, rung in enumerate(prompts):
            elapsed = time.monotonic() - started
            if elapsed >= ctx.limits.time_limit_seconds:
                stop_reason = STOP_TIME_LIMIT
                break
            gate_before = await ctx.sandbox.remote_gate_count()
            result = await invoke(
                rung or ctx.prompt, index > 0,
                min(self.timeout, max(1, int(ctx.limits.time_limit_seconds - elapsed))),
            )
            returncode = result.returncode
            self._append_output(ctx.outdir, result.stdout, result.stderr)
            messages.append(ChatMessage("user", rung or ctx.prompt))
            observation = self._invocation_from_stdout(result.stdout)
            messages.extend(
                ChatMessage("assistant", text) for text in observation.assistant_texts
            )
            rungs = index
            if observation.stream_error == "Claude reported a model fallback":
                stop_reason = STOP_ERROR
                error = observation.stream_error
                break
            if returncode == 1 and _provider_refusal(result.stdout, observation, self.model):
                stop_reason = STOP_PROVIDER_REFUSAL
                break
            solved, _, _ = await ctx.score()
            if solved:
                stop_reason = STOP_SUBMIT
                break
            if returncode == 124:
                stop_reason = STOP_TIME_LIMIT
                break
            if returncode != 0:
                stop_reason = STOP_ERROR
                break
            if not observation.valid:
                stop_reason = STOP_ERROR
                error = f"invalid Claude refusal event stream: {observation.invalid_reason}"
                break
            gate_after = await ctx.sandbox.remote_gate_count()
            if refusal_streak.observe(
                observation.assistant_text,
                tool_call_count=gate_after - gate_before,
                final_assistant_text=observation.assistant_texts[-1] if observation.assistant_texts else None,
            ):
                stop_reason = STOP_DESISTED
                break
            if index == len(prompts) - 1:
                stop_reason = STOP_TURN_LIMIT
        return ScaffoldResult(
            stop_reason=stop_reason,
            turns=rungs + 1,
            seconds=round(time.monotonic() - started, 1),
            messages=messages,
            error=error,
            extra={
                "rungs_used": rungs, "cli_returncode": returncode,
                "hook_alive": True, "modal_cli": True,
                "provider_refusal": stop_reason == STOP_PROVIDER_REFUSAL,
                **(
                    {"void_reason": "refusal_stream_invalid"}
                    if error is not None
                    else {}
                ),
                **refusal_streak.as_metadata(turn_unit="cli_invocation"),
            },
        )

    @staticmethod
    def _invocation_from_stdout(stdout: str) -> InvocationObservation:
        """Parse one complete Claude Code `stream-json` invocation."""
        assistant_texts: list[str] = []
        result_text = ""
        completion_count = 0
        malformed = False
        stream_error: str | None = None
        model_fallback = False
        saw_event = False
        events: list[dict[str, Any]] = []

        # JSON strings can contain Unicode separators that splitlines treats
        # as record boundaries. Only a line feed separates JSONL records.
        for line in stdout.split("\n"):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                malformed = True
                continue
            if not isinstance(event, dict):
                malformed = True
                continue
            events.append(event)

        background_completions = _background_completion_indices(events)
        if sum(event.get("type") == "system" and event.get("subtype") == "init"
               for event in events) > 1 and not background_completions:
            malformed = True
        for index, event in enumerate(events):
            saw_event = True
            event_type = event.get("type")
            if event_type == "system" and event.get("subtype") in {
                "model_refusal_fallback", "model_fallback", "model_consent_fallback",
            }:
                model_fallback = True
            if completion_count and event_type in {"assistant", "user"}:
                malformed = True
            if event_type == "assistant":
                message = event.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, list):
                    malformed = True
                    continue
                if any(isinstance(block, dict) and block.get("type") == "fallback"
                       for block in content):
                    model_fallback = True
                text_blocks = [
                    block.get("text", "").strip()
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                    and block.get("text", "").strip()
                ]
                if text_blocks:
                    assistant_texts.append("\n".join(text_blocks))
            elif event_type == "result":
                if index in background_completions:
                    continue
                completion_count += 1
                subtype = str(event.get("subtype") or "")
                if event.get("is_error") is True or subtype.startswith("error_"):
                    stream_error = str(
                        event.get("result") or event.get("subtype") or "Claude reported an error"
                    )
                elif isinstance(event.get("result"), str):
                    result_text = event["result"].strip()

        if not assistant_texts and result_text:
            assistant_texts.append(result_text)
        return InvocationObservation(
            assistant_texts=tuple(assistant_texts),
            stream_completed=completion_count == 1,
            malformed=(
                malformed
                or completion_count > 1
                or (bool(stdout.strip()) and not saw_event)
            ),
            stream_error="Claude reported a model fallback" if model_fallback else stream_error,
        )

    @staticmethod
    def _append_output(outdir: Path, stdout: str, stderr: str) -> None:
        with (outdir / "stdout.txt").open("a", encoding="utf-8") as handle:
            handle.write(stdout)
            handle.write("\n")
        with (outdir / "stderr.txt").open("a", encoding="utf-8") as handle:
            handle.write(stderr)
