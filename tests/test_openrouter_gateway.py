"""Credential isolation and model-scoped request forwarding."""

from __future__ import annotations

import json
from contextlib import contextmanager

import httpx
import pytest

from instrumental_evasion.runner.openrouter_gateway import TOKEN_ENV, gateway, prepare_request

MODEL = "qwen/qwen3.5-9b"


@pytest.mark.parametrize("path,body", [
    ("/v1/chat/completions", {"model": MODEL}),
    ("/v1/responses", {"model": "other/model"}),
    ("/v1/responses", {"model": MODEL, "models": [MODEL]}),
    ("/v1/responses", {"model": MODEL, "background": True}),
])
def test_other_endpoints_and_models_are_rejected(path, body):
    with pytest.raises(ValueError):
        prepare_request(path, body, MODEL)


@pytest.mark.parametrize("provider", [None, "azure"])
def test_forwarding_keeps_real_key_host_side_and_logs_only_metadata(tmp_path, provider):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @contextmanager
        def stream(self, method, url, **kwargs):
            calls.append((method, url, kwargs))
            yield httpx.Response(200, json={"id": "response1", "model": MODEL,
                                          "usage": {"input_tokens": 1, "output_tokens": 1}})

    log = tmp_path / "gateway.jsonl"
    with gateway("private-provider-credential", log, model=MODEL, client_factory=Client,
                 provider=provider, capture_path=tmp_path / "responses.jsonl") as cfg:
        token = cfg[TOKEN_ENV]
        assert token != "private-provider-credential"
        url = cfg["base_url"] + "/responses"
        with httpx.Client(trust_env=False) as client:
            assert client.post(url, json={"model": MODEL}).status_code == 403
            headers = {"Authorization": "Bearer " + token}
            assert client.post(url, headers=headers, json={"model": "other/model"}).status_code == 400
            response = client.post(url, headers=headers, json={"model": MODEL, "input": "private-task-content"})
            assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0][2]["headers"]["Authorization"] == "Bearer private-provider-credential"
    expected = {"model": MODEL, "input": "private-task-content"}
    if provider:
        expected["provider"] = {"only": [provider], "allow_fallbacks": False, "require_parameters": False}
    assert calls[0][2]["json"] == expected
    recorded = log.read_text()
    assert "private-provider-credential" not in recorded
    assert "private-task-content" not in recorded
    assert token not in recorded
    assert any(json.loads(line).get("response_id") == "response1" for line in recorded.splitlines())
    assert (tmp_path / "responses.jsonl").stat().st_mode & 0o777 == 0o600
    assert "private-provider-credential" not in (tmp_path / "responses.jsonl").read_text()


def test_provider_pin_cannot_be_overridden_by_client_routing_preferences():
    body = {"model": MODEL, "provider": {"only": ["openai"], "allow_fallbacks": True}}
    prepared = prepare_request("/v1/responses", body, MODEL, "azure")
    assert prepared["provider"] == {"only": ["azure"], "allow_fallbacks": False, "require_parameters": False}
    assert body["provider"]["only"] == ["openai"]
