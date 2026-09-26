"""Check the Codex token budget against a local fake provider with no executable tools.

Opt-in: set CODEX_BUDGET_PROBE_BIN to a Codex CLI binary; skipped otherwise.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from instrumental_evasion.runner.codex_budget import SessionTokenBudget
from instrumental_evasion.runner.scaffolds.cli_process import run_cli_process


@pytest.mark.parametrize("mode", ["normal", "slow_response", "missing_usage"])
def test_pinned_cli_stops_during_invocation_at_token_budget(tmp_path, monkeypatch, mode):
    binary = os.environ.get("CODEX_BUDGET_PROBE_BIN")
    if not binary:
        pytest.skip("Set CODEX_BUDGET_PROBE_BIN for the offline CLI compatibility check.")
    requests = []
    if mode != "normal":
        from instrumental_evasion.runner import codex_budget
        monkeypatch.setattr(codex_budget, "TELEMETRY_GRACE_SECONDS", 0.5)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            index = len(requests)
            if mode == "missing_usage" and index > 1:
                time.sleep(1.5)
            # This unknown function cannot execute a shell command or change a file.
            item = {"id": f"fc_{index}", "type": "function_call",
                    "call_id": f"call_{index}", "name": "nonexistent_budget_probe",
                    "arguments": "{}", "status": "completed"}
            if index > 6:
                item = {"id": "msg_end", "type": "message", "role": "assistant",
                        "status": "completed", "content": [
                            {"type": "output_text", "text": "Done.", "annotations": []}]}
            response = {"id": f"resp_{index}", "object": "response",
                        "created_at": int(time.time()), "status": "completed",
                        "model": "diagnostic", "output": [item],
                        "usage": {"input_tokens": 90, "output_tokens": 10,
                                  "total_tokens": 100, "input_tokens_details": {"cached_tokens": 80}}}
            if mode == "missing_usage":
                response.pop("usage")
            events = [
                ("response.created", {"response": dict(response, status="in_progress", output=[])}),
                ("response.output_item.added", {"output_index": 0, "item": item}),
                ("response.output_item.done", {"output_index": 0, "item": item}),
                ("response.completed", {"response": response}),
            ]
            chunks = [("event: " + kind + "\ndata: " + json.dumps(
                dict(data, type=kind, sequence_number=i)) + "\n\n").encode()
                for i, (kind, data) in enumerate(events)]
            raw = b"".join(chunks)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(b"".join(chunks[:-1]))
                self.wfile.flush()
                if mode == "slow_response":
                    time.sleep(1.5)
                self.wfile.write(chunks[-1])
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

    home = tmp_path / "home"
    home.mkdir()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    command = [str(Path(binary).resolve()), "exec", "--json", "--strict-config",
               "--ignore-user-config", "--ignore-rules", "--skip-git-repo-check",
               "--sandbox", "read-only", "-C", str(tmp_path), "--model", "diagnostic"]
    settings = {
        "model_provider": "offline",
        "model_providers.offline.name": "Offline budget test",
        "model_providers.offline.base_url": f"http://127.0.0.1:{server.server_port}/v1",
        "model_providers.offline.wire_api": "responses",
        "features.shell_tool": False,
    }
    for key, value in settings.items():
        command += ["-c", f"{key}={json.dumps(value)}"]
    command += ["Offline diagnostic. Do not execute commands or access files."]
    budget = SessionTokenBudget(home, 250, tmp_path / "stdout.txt")
    try:
        result = run_cli_process(
            ["/usr/bin/env", "-i", "PATH=" + os.environ["PATH"], "HOME=" + str(home),
             "CODEX_HOME=" + str(home), "NO_PROXY=*", *command],
            stdout_path=tmp_path / "stdout.txt", stderr_path=tmp_path / "stderr.txt",
            timeout=30, stop_requested=budget.poll,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    if mode == "missing_usage":
        assert result.termination_reason == "token_usage_error", (result.stdout_tail, result.stderr_tail)
        assert budget.error_code == "completed_response_usage_missing"
        assert budget.usage.total == 0
        return
    assert result.termination_reason == "token_limit", (result.stdout_tail, result.stderr_tail)
    assert 3 <= len(requests) <= 4
    assert budget.usage.total >= 300
    assert budget.usage.cache_read_tokens >= 240
    assert not result.timed_out
    assert all("<rollout_budget>" not in json.dumps(request) for request in requests)
