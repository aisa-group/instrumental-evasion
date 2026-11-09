"""A model-scoped Responses gateway that keeps the OpenRouter key host-side."""

from __future__ import annotations

import hashlib
import json
import secrets
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

VERSION = "openrouter_responses_host_gateway_v2_provider_pin"
TOKEN_ENV = "CODEX_OPENROUTER_GATEWAY_TOKEN"


def prepare_request(path: str, body: dict, model: str, provider: str | None = None) -> dict:
    if path != "/v1/responses" or not isinstance(body, dict):
        raise ValueError("Only the Responses endpoint is permitted")
    if body.get("model") != model or "models" in body or "route" in body:
        raise ValueError("Only the configured model is permitted")
    if body.get("background"):
        raise ValueError("Asynchronous responses are disabled")
    if provider:
        return {**body, "provider": {"only": [provider], "allow_fallbacks": False,
                                     "require_parameters": False}}
    return body


@contextmanager
def gateway(key: str, log_path: Path, *, model: str,
            upstream: str = "https://openrouter.ai/api", client_factory=httpx.Client,
            provider: str | None = None, capture_path: Path | None = None):
    """Forward requests unchanged; log digests and usage, never headers/prompts."""
    token = secrets.token_urlsafe(32)
    lock = threading.Lock()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(value):
        with lock:
            with log_path.open("a") as stream:
                stream.write(json.dumps({"time": time.time(), "version": VERSION, **value}) + "\n")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            if self.headers.get("Authorization") != "Bearer " + token:
                self.send_error(403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16 * 1024 * 1024:
                    raise ValueError("invalid request size")
                raw = self.rfile.read(size)
                body = prepare_request(self.path, json.loads(raw), model, provider)
            except (ValueError, TypeError, AttributeError):
                record({"event": "request_rejected"})
                self.send_error(400)
                return
            meta = {"model": model, "request_sha256": hashlib.sha256(raw).hexdigest()}
            if provider:
                meta.update(provider_only=[provider], allow_fallbacks=False,
                            request_parameters=sorted(body), requested_include=body.get("include"),
                            reasoning_settings=body.get("reasoning"),
                            tool_types=[tool.get("type") for tool in body.get("tools", [])],
                            forwarded_request_sha256=hashlib.sha256(json.dumps(body).encode()).hexdigest())
            started = False
            try:
                with client_factory(timeout=httpx.Timeout(300, connect=30)) as client:
                    with client.stream("POST", upstream + self.path, json=body,
                                       headers={"Authorization": "Bearer " + key,
                                                "Content-Type": "application/json"}) as response:
                        record({**meta, "status": response.status_code})
                        self.send_response(response.status_code)
                        self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                        self.send_header("Connection", "close")
                        self.end_headers()
                        started = True
                        pending = b""
                        for chunk in response.iter_bytes():
                            self.wfile.write(chunk)
                            self.wfile.flush()
                            pending += chunk
                            while b"\n" in pending:
                                line, pending = pending.split(b"\n", 1)
                                if not line.startswith(b"data:"):
                                    continue
                                try:
                                    event = json.loads(line[5:])
                                except ValueError:
                                    continue
                                if capture_path is not None:
                                    with lock:
                                        with capture_path.open("a") as stream:
                                            capture_path.chmod(0o600)
                                            stream.write(json.dumps({"response_event": event}) + "\n")
                                result = event.get("response") or {}
                                if result.get("usage") or result.get("id"):
                                    record({**meta, "event": event.get("type"),
                                            "response_id": result.get("id"),
                                            "served_model": result.get("model"),
                                            "served_provider": result.get("provider") or event.get("provider"),
                                            "usage": result.get("usage")})
                        if pending.strip().startswith(b"{"):
                            try:
                                result = json.loads(pending)
                                if capture_path is not None:
                                    with lock:
                                        with capture_path.open("a") as stream:
                                            capture_path.chmod(0o600)
                                            stream.write(json.dumps({"response_event": result}) + "\n")
                                record({**meta, "response_id": result.get("id"),
                                        "served_model": result.get("model"), "usage": result.get("usage")})
                            except (ValueError, AttributeError):
                                pass
            except Exception as error:
                record({**meta, "error_type": type(error).__name__})
                if not started:
                    self.send_error(502)
            finally:
                self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {"base_url": f"http://127.0.0.1:{server.server_port}/v1", TOKEN_ENV: token}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
