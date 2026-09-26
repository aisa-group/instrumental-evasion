"""Check that transport retries fit within the monitor's classification deadline."""

import urllib.error
from email.message import Message
from email.utils import formatdate

import pytest

from instrumental_evasion.config import DEFAULT_MONITOR_SERVING
from instrumental_evasion.hooks import monitor_client


@pytest.fixture(autouse=True)
def _library_defaults(monkeypatch):
    """Each test sets the serving variables it needs; start from none."""
    for name in DEFAULT_MONITOR_SERVING:
        monkeypatch.delenv(name, raising=False)


def test_rate_limit_profile_recovers_after_provider_window(monkeypatch):
    clock, calls, diagnostics = [0.0], [], []
    monkeypatch.setenv("MONITOR_HTTP_BACKOFF_PROFILE", "rate_limit_v1")
    monkeypatch.setenv("MONITOR_HTTP_RETRIES", "4")
    monkeypatch.setattr(monitor_client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(monitor_client.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(monitor_client.random, "uniform", lambda *a: 0)

    def post(_url, payload, _headers, timeout):
        calls.append((clock[0], payload, timeout))
        if clock[0] < 30:
            raise urllib.error.HTTPError("https://example.test", 429, "limit", Message(), None)
        return {"choices": [{"message": {"content": '{"decision":"BLOCK","reason":"prohibited"}'}}]}

    monkeypatch.setattr(monitor_client, "_post", post)
    verdict, _ = monitor_client.verdict_direct("bash", {"command": "pwd"}, policy="no_input_read",
                                               api_key="test", diagnostics=diagnostics)
    assert verdict.decision == "BLOCK"
    assert [call[0] for call in calls] == [0, 15, 40]
    assert all(call[1] == calls[0][1] for call in calls)
    assert diagnostics[0]["retry_delay_seconds"] == 15


@pytest.mark.parametrize("header", ["60", "http-date"])
def test_provider_retry_after_is_respected(monkeypatch, header):
    clock, starts = [0.0], []
    monkeypatch.setenv("MONITOR_HTTP_BACKOFF_PROFILE", "rate_limit_v1")
    monkeypatch.setattr(monitor_client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(monitor_client.time, "time", lambda: 1000 + clock[0])
    monkeypatch.setattr(monitor_client.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    monkeypatch.setattr(monitor_client.random, "uniform", lambda *a: 0)

    def post(*args):
        starts.append(clock[0])
        if len(starts) == 1:
            headers = Message()
            headers["Retry-After"] = formatdate(1060, usegmt=True) if header == "http-date" else header
            raise urllib.error.HTTPError("https://example.test", 429, "limit", headers, None)
        return {"ok": True}

    monkeypatch.setattr(monitor_client, "_post", post)
    assert monitor_client._post_with_retries("https://example.test", {}, {}, 90, 3) == {"ok": True}
    assert starts == [0, 60]


def test_retry_after_outside_deadline_fails_closed(monkeypatch):
    monkeypatch.setenv("MONITOR_HTTP_BACKOFF_PROFILE", "rate_limit_v1")
    monkeypatch.setattr(monitor_client.time, "monotonic", lambda: 0)
    monkeypatch.setattr(monitor_client.time, "sleep", lambda s: pytest.fail("Must not exceed the deadline"))

    def post(*args):
        headers = Message()
        headers["Retry-After"] = "120"
        raise urllib.error.HTTPError("https://example.test", 429, "limit", headers, None)

    monkeypatch.setattr(monitor_client, "_post", post)
    with pytest.raises(monitor_client.MonitorUnavailable, match="remaining deadline"):
        monitor_client._post_with_retries("https://example.test", {}, {}, 90, 4)


@pytest.mark.parametrize("value", [None, "nan", "inf", "malformed", "x" * 129])
def test_invalid_retry_after_is_ignored(value):
    assert monitor_client._retry_after_seconds(value) is None


def test_rate_limit_profile_is_explicit():
    assert "monitor_http_backoff_profile" not in monitor_client.monitor_request_metadata({})
    with pytest.raises(ValueError, match="backoff"):
        monitor_client.monitor_request_metadata({"MONITOR_HTTP_BACKOFF_PROFILE": "unknown"})


def test_short_request_timeout_retries_before_classification_deadline(monkeypatch):
    clock = [0.0]
    calls = []
    monkeypatch.setenv("MONITOR_HTTP_TIMEOUT_SECONDS", "20")
    monkeypatch.setattr(monitor_client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(monitor_client.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def post(_url, payload, _headers, timeout):
        calls.append((payload, timeout))
        if len(calls) == 1:
            clock[0] += timeout
            raise TimeoutError("stalled connection")
        clock[0] += 2
        return {"choices": [{"message": {"content": '{"decision":"BLOCK","reason":"prohibited"}'}}]}

    monkeypatch.setattr(monitor_client, "_post", post)
    diagnostics = []
    verdict, _ = monitor_client.verdict_direct(
        "bash", {"command": "cat /workspace/input/data.csv"}, policy="no_input_read",
        api_key="test", diagnostics=diagnostics,
    )
    assert verdict.decision == "BLOCK"
    assert clock[0] == 24
    assert len(calls) == 2 and calls[0][1] == calls[1][1] == 20
    assert calls[0][0] == calls[1][0]
    assert diagnostics[0]["error_type"] == "TimeoutError"
    assert diagnostics[1]["parse_status"] == "ok"
    assert monitor_client.CLASSIFICATION_DEADLINE_SECONDS == 85


def test_repeated_timeouts_never_produce_approval(monkeypatch):
    clock = [0.0]
    monkeypatch.setenv("MONITOR_HTTP_TIMEOUT_SECONDS", "20")
    monkeypatch.setattr(monitor_client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(monitor_client.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def post(_url, _payload, _headers, timeout):
        clock[0] += timeout
        raise TimeoutError("stalled connection")

    monkeypatch.setattr(monitor_client, "_post", post)
    with pytest.raises(monitor_client.MonitorUnavailable):
        monitor_client.verdict_direct("bash", {"command": "pwd"}, policy="no_input_read", api_key="test")
    assert clock[0] == 66


def test_late_response_is_rejected_with_short_timeout(monkeypatch):
    clock = [0.0]
    monkeypatch.setenv("MONITOR_HTTP_TIMEOUT_SECONDS", "20")
    monkeypatch.setattr(monitor_client.time, "monotonic", lambda: clock[0])

    def post(*_args):
        clock[0] = 86
        return {"choices": [{"message": {"content": '{"decision":"ALLOW","reason":"safe"}'}}]}

    monkeypatch.setattr(monitor_client, "_post", post)
    with pytest.raises(monitor_client.MonitorUnavailable, match="deadline exceeded"):
        monitor_client.verdict_direct("bash", {"command": "pwd"}, policy="no_input_read", api_key="test")


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "91", "broken"])
def test_invalid_http_timeout_is_rejected(value):
    with pytest.raises(ValueError):
        monitor_client.monitor_request_metadata({"MONITOR_HTTP_TIMEOUT_SECONDS": value})


def test_http_timeout_metadata_is_opt_in():
    assert "monitor_http_timeout_seconds" not in monitor_client.monitor_request_metadata({})
    assert monitor_client.monitor_request_metadata({"MONITOR_HTTP_TIMEOUT_SECONDS": "20"})[
        "monitor_http_timeout_seconds"
    ] == 20


def test_fourth_transport_try_can_use_remaining_deadline(monkeypatch):
    clock = [0.0]
    calls = []
    monkeypatch.setenv("MONITOR_HTTP_TIMEOUT_SECONDS", "20")
    monkeypatch.setenv("MONITOR_HTTP_RETRIES", "4")
    monkeypatch.setattr(monitor_client.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(monitor_client.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def post(_url, _payload, _headers, timeout):
        calls.append(timeout)
        if len(calls) <= 3:
            clock[0] += timeout
            raise TimeoutError("stalled connection")
        clock[0] += 2
        return {"choices": [{"message": {"content": '{"decision":"ALLOW","reason":"safe"}'}}]}

    monkeypatch.setattr(monitor_client, "_post", post)
    verdict, _ = monitor_client.verdict_direct("bash", {"command": "pwd"}, policy="no_input_read", api_key="test")
    assert verdict.decision == "ALLOW"
    assert calls == [20, 20, 20, 11]
    assert clock[0] == 76


@pytest.mark.parametrize("value", ["0", "5", "-1", "nan", "2.5"])
def test_invalid_transport_retry_count_is_rejected(value):
    with pytest.raises(ValueError):
        monitor_client.monitor_request_metadata({"MONITOR_HTTP_RETRIES": value})


def test_http_response_deadline_bounds_trickled_bytes():
    """Cover the pinned CPython HTTPResponse socket path with a real server."""
    import threading
    import time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            try:
                for _ in range(100):
                    self.wfile.write(b" ")
                    self.wfile.flush()
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            monitor_client._post(f"http://127.0.0.1:{server.server_port}", {}, {}, 0.15)
        assert time.monotonic() - started < 0.6
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_http_response_reads_complete_json_and_rejects_oversize(monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            payload = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        assert monitor_client._post(url, {}, {}, 2) == {"ok": True}
        monkeypatch.setattr(monitor_client, "MAX_RESPONSE_BYTES", 4)
        with pytest.raises(monitor_client.MonitorUnavailable, match="size limit"):
            monitor_client._post(url, {}, {}, 2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
