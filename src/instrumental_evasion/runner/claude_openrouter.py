"""Forward Claude's Messages requests without exposing the provider key."""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

MODEL = "openrouter/anthropic/claude-sonnet-5"
API_MODEL = "anthropic/claude-sonnet-5"
OPUS_MODEL = "openrouter/anthropic/claude-opus-5"
MODELS = {MODEL: API_MODEL, OPUS_MODEL: "anthropic/claude-opus-5",
          "openrouter/anthropic/claude-fable-5.1": "anthropic/claude-fable-5.1",
          "openrouter/anthropic/claude-opus-5.5": "anthropic/claude-opus-5.5"}
VERSION = "claude_openrouter_messages_v1"


def is_openrouter(model: str) -> bool:
    return model in MODELS


def cli_environment(model: str) -> dict[str, str]:
    if not is_openrouter(model):
        return {}
    api_model = MODELS[model]
    url = os.environ.get("CLAUDE_OPENROUTER_PROXY_URL", "")
    token = os.environ.get("CLAUDE_OPENROUTER_PROXY_TOKEN", "")
    if not url.startswith("http://127.0.0.1:") or not token:
        raise ValueError("The host OpenRouter gateway is unavailable.")
    return {
        "ANTHROPIC_BASE_URL": url, "ANTHROPIC_AUTH_TOKEN": token,
        "ANTHROPIC_API_KEY": "", "CLAUDE_CODE_OAUTH_TOKEN": "",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": api_model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": api_model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": api_model,
        "ANTHROPIC_SMALL_FAST_MODEL": api_model,
        "CLAUDE_CODE_SUBAGENT_MODEL": api_model,
    }


def prepare_request(path: str, data: dict, *, model: str = MODEL,
                    reasoning_effort: str | None = None) -> tuple[str, dict]:
    if model not in MODELS:
        raise ValueError("Unsupported gateway model.")
    api_model = MODELS[model]
    endpoint = path.split("?", 1)[0]
    if endpoint not in {"/v1/messages", "/v1/messages/count_tokens"}:
        raise ValueError("The gateway permits only Messages endpoints.")
    if data.get("model") not in {api_model, api_model.removeprefix("anthropic/"), model}:
        raise ValueError("The gateway permits only its configured model.")
    if "models" in data or "route" in data:
        raise ValueError("Model fallback is disabled.")
    if reasoning_effort is not None:
        if reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Unsupported gateway reasoning effort.")
        if endpoint == "/v1/messages" and data.get("output_config", {}).get("effort") != reasoning_effort:
            raise ValueError("Claude request does not match the frozen reasoning effort.")
        if any("output_config" in message for message in data.get("messages", [])):
            raise ValueError("Per-message effort overrides are disabled for a frozen effort sweep.")
    return endpoint, {**data, "model": api_model,
                      "provider": {"only": ["Anthropic"], "allow_fallbacks": False}}


@contextmanager
def gateway(key: str, log_path: Path, *, upstream: str = "https://openrouter.ai/api", model: str = MODEL,
            reasoning_effort: str | None = None):
    """Serve one worker. Retain request digests and usage, never prompts or keys."""
    if model not in MODELS:
        raise ValueError("Unsupported gateway model.")
    api_model = MODELS[model]
    token = secrets.token_urlsafe(32)
    lock = threading.Lock()
    def record(value):
        with lock:
            with log_path.open("a") as stream:
                stream.write(json.dumps({"time": time.time(), **value}) + "\n")
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
                    raise ValueError("Invalid request size.")
                raw = self.rfile.read(size)
                endpoint, data = prepare_request(self.path, json.loads(raw), model=model,
                                                 reasoning_effort=reasoning_effort)
            except (ValueError, TypeError, AttributeError):
                record({"event": "request_rejected", "expected_effort": reasoning_effort})
                self.send_error(400)
                return
            headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
                       "anthropic-version": self.headers.get("anthropic-version", "2023-06-01")}
            if self.headers.get("anthropic-beta"):
                headers["anthropic-beta"] = self.headers["anthropic-beta"]
            meta = {"endpoint": endpoint, "model": api_model,
                    "request_sha256": hashlib.sha256(raw).hexdigest()}
            if reasoning_effort is not None:
                meta["reasoning_effort"] = data.get("output_config", {}).get("effort")
            started = False
            try:
                with httpx.Client(timeout=httpx.Timeout(180, connect=30)) as client:
                    with client.stream("POST", upstream + endpoint, json=data, headers=headers) as response:
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
                                msg = event.get("message", {})
                                usage = event.get("usage") or msg.get("usage")
                                if usage or msg.get("id"):
                                    record({**meta, "event": event.get("type"),
                                            "response_id": msg.get("id"), "served_model": msg.get("model"),
                                            "usage": usage})
                        if pending.strip().startswith(b"{"):
                            try:
                                value = json.loads(pending)
                                record({**meta, "response_id": value.get("id"),
                                        "served_model": value.get("model"), "usage": value.get("usage"),
                                        "error_type": value.get("error", {}).get("type")})
                            except (ValueError, AttributeError):
                                pass
            except Exception as error:
                record({**meta, "error_type": type(error).__name__})
                if not started:
                    self.send_error(502)
            finally:
                self.close_connection = True
    log_path.parent.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield {"CLAUDE_OPENROUTER_PROXY_URL": f"http://127.0.0.1:{server.server_port}",
               "CLAUDE_OPENROUTER_PROXY_TOKEN": token}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
