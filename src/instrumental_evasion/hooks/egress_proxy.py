"""A host-side allowlisting proxy, so the agent CLI can reach its API and nothing else.

The CLI scaffolds run the vendor CLI *inside* the task container, which means
the container needs a route to its API host. The task set needs the opposite:
the benchmark tasks are public -- instruction, reference solution and verifier
all sit in an upstream repo -- and with open egress an agent can fetch that repo
and read the answer instead of solving the task, turning a lookup into a "solve".

Both hold at once only on a host whose direct egress is blocked, so that the
upstream proxy (EGRESS_UPSTREAM) is the only way out. There, pointing the
container's proxy variables at this process is a real cut, not a request: an
agent that unsets `http_proxy` gets no network at all, and an agent that keeps
it can reach the allowlist and nothing else. On a host with direct egress the
allowlist is advisory.

CONNECT is forwarded to the upstream proxy only for an allowlisted host. Plain
HTTP and any other host are refused with 403 and recorded, so a task that fails
for want of a host says so in the log instead of failing silently.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from socketserver import BaseRequestHandler, ThreadingTCPServer

# Hosts the CLI itself needs. Deliberately short: every entry is a hole in the
# containment, so a host is added only when a refusal in the log proves it is
# needed. Suffix match on the hostname, so "api.anthropic.com" also covers a
# regional alias but never "api.anthropic.com.evil.test".
DEFAULT_ALLOW = (
    "api.anthropic.com",
    "statsig.anthropic.com",
)

# Hosts a given model PROVIDER needs, so a scaffold opens the holes its own
# backend requires and no others. Derived from the model string rather than
# hardcoded per scaffold: a CLI on OpenRouter and a CLI on a subscription need
# different hosts, and giving either the union would reopen the route to
# whatever the other one talks to.
PROVIDER_HOSTS = {
    "openrouter": ("openrouter.ai",),
    "anthropic": ("api.anthropic.com", "statsig.anthropic.com"),
    "openai": ("api.openai.com",),
}


def hosts_for(model: str) -> tuple[str, ...]:
    """The allowlist entries one model string implies.

    Unknown providers get nothing: a scaffold that cannot reach its backend
    fails loudly at its preflight, which is better than a silent hole.
    """
    provider = (model or "").split("/", 1)[0].strip().lower()
    return PROVIDER_HOSTS.get(provider, ())


UPSTREAM = os.environ.get("EGRESS_UPSTREAM", "proxy:8080")
BUFSIZE = 65536


def _allowed(host: str, allow: tuple[str, ...]) -> bool:
    host = host.lower().strip().rstrip(".")
    return any(host == a or host.endswith("." + a) for a in allow)


class _Handler(BaseRequestHandler):
    allow: tuple[str, ...] = DEFAULT_ALLOW
    logfile: Path | None = None
    lock = threading.Lock()

    def _record(self, verdict: str, target: str, method: str) -> None:
        if not self.logfile:
            return
        row = {"ts": time.time(), "verdict": verdict, "target": target, "method": method}
        with self.lock:
            with self.logfile.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")

    def handle(self) -> None:
        sock = self.request
        sock.settimeout(120)
        try:
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = sock.recv(BUFSIZE)
                if not chunk:
                    return
                head += chunk
                if len(head) > 1 << 16:
                    return
        except OSError:
            return

        first = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        parts = first.split()
        method = parts[0] if parts else ""
        target = parts[1] if len(parts) > 1 else ""

        # Only CONNECT is ever forwarded. A plain-HTTP request would let the
        # upstream proxy see and cache the URL, and every host the CLI needs is
        # HTTPS anyway.
        if method != "CONNECT":
            self._record("refused", target, method or "?")
            sock.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            return

        host = target.rsplit(":", 1)[0].strip("[]")
        if not _allowed(host, self.allow):
            self._record("refused", target, method)
            sock.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n\r\n")
            return

        up_host, _, up_port = UPSTREAM.partition(":")
        try:
            upstream = socket.create_connection((up_host, int(up_port or 8080)), timeout=30)
        except OSError:
            self._record("upstream_error", target, method)
            sock.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            return

        self._record("allowed", target, method)
        try:
            upstream.sendall(head)
            self._pump(sock, upstream)
        finally:
            upstream.close()

    def _pump(self, a: socket.socket, b: socket.socket) -> None:
        """Relay until a peer closes or episode cleanup closes the proxy.

        Header and connection timeouts must not become stream idle limits.
        The episode runner owns the total duration and closes active tunnels.
        """

        def copy(src: socket.socket, dst: socket.socket) -> None:
            try:
                while True:
                    data = src.recv(BUFSIZE)
                    if not data:
                        break
                    dst.sendall(data)
            except OSError:
                pass
            finally:
                try:
                    dst.shutdown(socket.SHUT_WR)
                except OSError:
                    pass

        with self.server.tunnel_sockets(a, b) as active:
            if not active:
                return
            a.settimeout(None)
            b.settimeout(None)
            t = threading.Thread(target=copy, args=(a, b), daemon=True)
            t.start()
            copy(b, a)
            t.join(timeout=5)
            if t.is_alive():
                # Do not leave the other relay direction outside cleanup's
                # socket registry after the handler finishes.
                for connection in (a, b):
                    try:
                        connection.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                t.join(timeout=1)


class _Server(ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, *args, **kwargs):
        self._tunnel_lock = threading.Lock()
        self._tunnel_sockets: set[socket.socket] = set()
        self._closing = False
        super().__init__(*args, **kwargs)

    @contextmanager
    def tunnel_sockets(self, *connections: socket.socket):
        """Track relay sockets so cleanup can interrupt blocking reads and writes."""
        with self._tunnel_lock:
            active = not self._closing
            if active:
                self._tunnel_sockets.update(connections)
        try:
            yield active
        finally:
            with self._tunnel_lock:
                self._tunnel_sockets.difference_update(connections)

    def server_close(self) -> None:
        with self._tunnel_lock:
            self._closing = True
            connections = tuple(self._tunnel_sockets)
        for connection in connections:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        super().server_close()


def serve(port: int, allow: tuple[str, ...], logfile: Path | None) -> _Server:
    """Start the proxy. Port 0 binds a free port; read it back off the server.

    Several of these can run at once on one host (one per episode), so a
    fixed port is a collision.
    """
    handler = type("Handler", (_Handler,), {"allow": allow, "logfile": logfile})
    server = _Server(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--allow", action="append", default=[])
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()
    allow = tuple(args.allow) or DEFAULT_ALLOW
    server = serve(args.port, allow, args.log)
    print(f"egress proxy on 127.0.0.1:{args.port} -> {UPSTREAM}; allow={list(allow)}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
