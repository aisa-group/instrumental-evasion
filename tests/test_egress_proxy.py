"""Check long-lived CONNECT tunnels and bounded proxy cleanup."""

import socket
import threading

import pytest

from instrumental_evasion.hooks.egress_proxy import _Handler, _Server, serve


def test_tunnel_survives_inherited_socket_timeouts():
    server = _Server(("127.0.0.1", 0), _Handler)
    handler = object.__new__(_Handler)
    handler.server = server
    client, client_relay = socket.socketpair()
    upstream_relay, upstream = socket.socketpair()
    for relay in (client_relay, upstream_relay):
        relay.settimeout(0.03)
    for peer in (client, upstream):
        peer.settimeout(1)
    thread = threading.Thread(target=handler._pump, args=(client_relay, upstream_relay), daemon=True)
    thread.start()
    try:
        # Both directions must survive a pause longer than the inherited limit.
        threading.Event().wait(0.15)
        client.sendall(b"request after pause")
        assert upstream.recv(100) == b"request after pause"
        upstream.sendall(b"stream after pause")
        assert client.recv(100) == b"stream after pause"
    finally:
        server.server_close()
        for connection in (client, client_relay, upstream_relay, upstream):
            connection.close()
        thread.join(timeout=2)


@pytest.mark.parametrize("host,status,connect_calls", [
    ("openrouter.ai", b"502", 1),
    ("openrouter.ai.evil.test", b"403", 0),
    ("unapproved.example", b"403", 0),
])
def test_allowlist_and_bounded_connect_remain_unchanged(monkeypatch, host, status, connect_calls):
    create_connection = socket.create_connection
    attempted = []

    def unavailable(address, *, timeout):
        attempted.append((address, timeout))
        assert timeout == 30
        raise TimeoutError("Upstream connection timed out")

    monkeypatch.setattr(socket, "create_connection", unavailable)
    server = serve(0, ("openrouter.ai",), None)
    try:
        with create_connection(server.server_address, timeout=1) as client:
            client.sendall(f"CONNECT {host}:443 HTTP/1.1\r\nHost: {host}:443\r\n\r\n".encode())
            response = client.recv(1024)
            assert response.split()[1] == status
        assert len(attempted) == connect_calls
    finally:
        server.shutdown()
        server.server_close()


def test_proxy_close_interrupts_idle_tunnels():
    server = _Server(("127.0.0.1", 0), _Handler)
    handler = object.__new__(_Handler)
    handler.server = server
    client, client_relay = socket.socketpair()
    upstream_relay, upstream = socket.socketpair()
    for peer in (client, upstream):
        peer.settimeout(1)
    thread = threading.Thread(target=handler._pump, args=(client_relay, upstream_relay), daemon=True)
    thread.start()
    try:
        # Receipt proves that the tunnel entered its relay loop before cleanup.
        client.sendall(b"ready")
        assert upstream.recv(100) == b"ready"
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()
        assert client.recv(100) == b""
        assert upstream.recv(100) == b""
    finally:
        server.server_close()
        for connection in (client, client_relay, upstream_relay, upstream):
            connection.close()
        thread.join(timeout=2)
