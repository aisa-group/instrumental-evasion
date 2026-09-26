"""Keep classification evidence and service available after client cancellation."""

import hashlib
import io
import json
import socket
import threading
from unittest.mock import Mock

import pytest

from instrumental_evasion.hooks.monitor_client import verdict_remote
from instrumental_evasion.runner.gate import (
    ALLOW,
    BLOCK,
    SOURCE_ERROR,
    SOURCE_MONITOR,
    Decision,
    Gate,
)
from instrumental_evasion.runner.gate_server import _handler_for, serve


@pytest.mark.parametrize("decision", [
    Decision(ALLOW, "private reason", SOURCE_MONITOR),
    Decision(BLOCK, "private reason", SOURCE_MONITOR),
    Decision(BLOCK, "private error", SOURCE_ERROR),
])
def test_cancelled_client_preserves_verdict_and_next_request(
    decision, monkeypatch, caplog, capsys, tmp_path,
):
    entered, release, replied = (threading.Event() for _ in range(3))
    gate = Gate(policy=None, decisions_path=tmp_path / "decisions.jsonl")

    def decide(*args, **kwargs):
        entered.set()
        assert release.wait(5)
        return decision

    monkeypatch.setattr(gate, "decide", decide)
    tool_use_id = "private-call-id"
    payload = json.dumps({
        "tool_name": "bash", "tool_input": {"command": "private arguments"},
        "tool_use_id": tool_use_id,
    }).encode()
    with serve(gate) as server:
        handler = server._server.RequestHandlerClass
        original_reply = handler._reply

        def reply(self, *args, **kwargs):
            try:
                return original_reply(self, *args, **kwargs)
            finally:
                replied.set()

        monkeypatch.setattr(handler, "_reply", reply)
        with socket.create_connection(("127.0.0.1", server.port), timeout=5) as client:
            client.sendall(
                b"POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
                + str(len(payload)).encode() + b"\r\n\r\n" + payload
            )
            assert entered.wait(5)
            client.shutdown(socket.SHUT_RDWR)
        release.set()
        assert replied.wait(5)
        records = gate.records()
        assert len(records) == 1
        assert records[0]["decision"] == decision.decision
        assert records[0]["decision_source"] == decision.source
        assert records[0]["tool_use_id"] == tool_use_id
        assert json.loads((tmp_path / "decisions.jsonl").read_text()) == records[0]
        assert gate.counts["error"] == (decision.source == SOURCE_ERROR)

        monkeypatch.setattr(gate, "decide", lambda *a, **k: Decision(ALLOW, "ok", SOURCE_MONITOR))
        verdict, _ = verdict_remote("bash", {}, url=server.url, retries=1, timeout=5)
        assert verdict.decision == ALLOW
        assert len(gate.records()) == 2

    events = [json.loads(r.message) for r in caplog.records if r.name.endswith("gate_server")]
    assert len(events) == 1
    assert events[0]["event"] == "gate_response_delivery_failed"
    assert events[0]["http_status"] == (503 if decision.source == SOURCE_ERROR else 200)
    assert events[0]["tool_use_id_sha256"] == hashlib.sha256(tool_use_id.encode()).hexdigest()
    assert events[0]["review_elapsed_ms"] >= 0
    assert "private" not in caplog.text
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("error", [BrokenPipeError, ConnectionResetError])
@pytest.mark.parametrize("stage", ["headers", "body"])
def test_disconnect_at_either_write_closes_connection(error, stage, caplog):
    handler_type = _handler_for(Gate(policy=None), None)
    handler = handler_type.__new__(handler_type)
    handler.send_response = Mock()
    handler.send_header = Mock()
    handler.end_headers = Mock(side_effect=error("private exception") if stage == "headers" else None)
    handler.wfile = Mock()
    handler.wfile.write.side_effect = error("private exception") if stage == "body" else None
    handler.close_connection = False

    handler._reply(401, {"decision": BLOCK, "reason": "private reason"})

    assert handler.close_connection
    event = json.loads(caplog.records[-1].message)
    assert event["error_type"] == error.__name__
    assert event["http_status"] == 401
    assert "private" not in caplog.text


def test_other_write_errors_remain_visible():
    handler_type = _handler_for(Gate(policy=None), None)
    handler = handler_type.__new__(handler_type)
    handler.send_response = Mock(side_effect=OSError("unexpected write failure"))
    handler.wfile = io.BytesIO()
    with pytest.raises(OSError, match="unexpected write failure"):
        handler._reply(200, {"ok": True})
