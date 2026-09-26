"""Serve a `Gate` over loopback HTTP, for scaffolds that run out of process.

The react scaffold owns its tools and calls the gate directly. A CLI scaffold
does not: Claude Code and Codex execute their own tools inside the container,
so the interception point is a `PreToolUse` hook running in there, and it needs
a way to ask.

Keeping the gate on the HOST and answering over loopback is what preserves the
property the in-process path has for free: the policy text, the monitor model
choice and OPENROUTER_API_KEY never enter the sandbox, and neither does the
decision log. The agent can see that something is refusing calls; it cannot read
what, and it cannot edit the record of what it tried.

The wire format is the monitor's own response schema --
`{"decision": "ALLOW"|"BLOCK", "reason": "..."}` -- so an in-container client
parses one format whether it is talking to this server or (in a smoke test) to
the model directly. A new scaffold needs an HTTP POST and nothing else.

Bind to loopback and reach it from the container over the host network. It is an
unauthenticated local endpoint unless a token is given, and it is not hardened
for anything beyond a single evaluation host.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

from instrumental_evasion.runner.gate import SOURCE_ERROR, Gate

MAX_BODY_BYTES = 1 << 20
logger = logging.getLogger(__name__)

# Either a Gate or something that returns the CURRENT one. The callable form
# matters for a multi-episode run: the server outlives one episode, but each
# episode has its own Gate and its own decision log, and a server pinned to the
# first one would file every later episode's calls under the first -- leaving
# those episodes looking like they made no tool calls at all.
GateSource = Gate | Callable[[], Gate]


def _resolve(source: GateSource) -> Gate:
    return source() if callable(source) else source


def _handler_for(source: GateSource, token: str | None) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _reply(
            self, code: int, body: dict[str, Any], *,
            tool_use_id: str | None = None, review_started: float | None = None,
        ) -> None:
            raw = json.dumps(body).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError) as error:
                # A CLI can cancel a parallel hook after another call fails.
                # The recorded verdict remains valid evidence of classification,
                # but it does not prove that the client received the response.
                self.close_connection = True
                event = {
                    "event": "gate_response_delivery_failed",
                    "http_status": code,
                    "error_type": type(error).__name__,
                    "tool_use_id_sha256": hashlib.sha256(tool_use_id.encode()).hexdigest()
                    if tool_use_id is not None else None,
                    "review_elapsed_ms": round((time.monotonic() - review_started) * 1000, 2)
                    if review_started is not None else None,
                }
                logger.warning("%s", json.dumps(event, sort_keys=True))

        def _authorized(self) -> bool:
            return not token or self.headers.get("Authorization") == f"Bearer {token}"

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
            if not self._authorized():
                self._reply(401, {"ok": False})
                return
            if self.path != "/health":
                self._reply(404, {"ok": False})
                return
            self._reply(200, {"ok": True})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's contract
            if not self._authorized():
                self._reply(
                    401, {"decision": "BLOCK", "reason": "Unauthorized monitor caller."}
                )
                return

            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_BODY_BYTES:
                self._reply(
                    400, {"decision": "BLOCK", "reason": "Malformed monitor request."}
                )
                return

            try:
                request = json.loads(
                    self.rfile.read(length).decode("utf-8", errors="replace")
                )
                tool_name = request["tool_name"]
                tool_input = request.get("tool_input") or {}
                message = request.get("assistant_message")
                tool_use_id = request.get("tool_use_id")
                if tool_use_id is not None and (
                    not isinstance(tool_use_id, str) or not 1 <= len(tool_use_id) <= 256
                ):
                    raise ValueError("invalid tool-use identifier")
            except Exception:  # noqa: BLE001
                self._reply(
                    400, {"decision": "BLOCK", "reason": "Malformed monitor request."}
                )
                return

            gate = _resolve(source)
            review_started = time.monotonic()
            decision = gate.review_sync(
                tool_name,
                tool_input,
                assistant_message=message,
                tool_use_id=tool_use_id,
            )

            # `message` is what the AGENT is shown on a refusal. It is computed
            # HOST-side so no container-side client has to carry the block text:
            # the verbose variant names the forbidden action, and a client that
            # holds that string holds a hint about the policy. Every hook client
            # uses this trusted field, so custom and verbose conditions are
            # identical across CLI scaffolds.
            if decision.source == SOURCE_ERROR:
                # 503, not 200: a 200 makes an upstream failure indistinguishable
                # from a real BLOCK in the caller's log, and "the monitor blocked
                # it" and "the monitor never answered" are different findings.
                # The status code makes the hook record
                # decision_source=monitor_error, and it still blocks either way.
                self._reply(
                    503,
                    {
                        "decision": "BLOCK",
                        "reason": "Monitor unavailable.",
                        "message": decision.agent_message,
                    },
                    tool_use_id=tool_use_id,
                    review_started=review_started,
                )
                return
            self._reply(
                200,
                {
                    "decision": decision.decision,
                    "reason": decision.reason,
                    "message": decision.agent_message if decision.blocked else "",
                },
                tool_use_id=tool_use_id,
                review_started=review_started,
            )

        def log_message(self, fmt: str, *args) -> None:  # quieter than the default
            return

    return Handler


class GateServer:
    """A running gate endpoint. `url` is what a container-side client is given."""

    def __init__(self, server: ThreadingHTTPServer, thread: threading.Thread) -> None:
        self._server = server
        self._thread = thread

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)

    def __enter__(self) -> "GateServer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()


def serve(
    gate: GateSource,
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    token: str | None = None,
) -> GateServer:
    """Start the gate endpoint on a background thread. Port 0 picks a free one.

    `gate` may be a callable, resolved per request, so one long-lived server can
    serve a run whose episodes each have their own Gate.
    """
    server = ThreadingHTTPServer((host, port), _handler_for(gate, token))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return GateServer(server, thread)
