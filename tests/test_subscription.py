"""Credential resolution for subscription-billed models.

These tests never touch the real credential stores: each one points the module
at a temporary file, so running the suite cannot read, refresh, or overwrite a
developer's actual tokens.
"""

from __future__ import annotations

import json
import time

import pytest

from instrumental_evasion import subscription
from instrumental_evasion.subscription import SubscriptionError


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Point the module at empty temp files and clear inherited env vars."""
    monkeypatch.setattr(subscription, "CLAUDE_CREDENTIALS", tmp_path / "claude.json")
    monkeypatch.setattr(subscription, "CODEX_AUTH", tmp_path / "codex.json")
    for var in (
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CODEX_ACCESS_TOKEN",
        "CHATGPT_ACCOUNT_ID",
    ):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def write_claude(path, token="sk-ant-oat01-token", expires_in=3600):
    path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": token,
                    "refreshToken": "sk-ant-ort01-refresh",
                    "expiresAt": int((time.time() + expires_in) * 1000),
                    "subscriptionType": "max",
                }
            }
        )
    )


def test_env_token_wins_over_credential_file(isolated, monkeypatch):
    write_claude(subscription.CLAUDE_CREDENTIALS, token="from-file")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "from-env")
    credential = subscription.anthropic_credential()
    assert credential.token == "from-env"
    assert credential.source == "$CLAUDE_CODE_OAUTH_TOKEN"


def test_reads_claude_credential_file(isolated):
    write_claude(subscription.CLAUDE_CREDENTIALS)
    credential = subscription.anthropic_credential()
    assert credential.token == "sk-ant-oat01-token"
    assert "max plan" in credential.source
    assert not credential.expired


def test_missing_claude_credentials_names_the_fix(isolated):
    with pytest.raises(SubscriptionError, match="claude setup-token"):
        subscription.anthropic_credential()


def test_expiring_token_is_refreshed(isolated, monkeypatch):
    write_claude(subscription.CLAUDE_CREDENTIALS, expires_in=60)
    monkeypatch.setattr(
        subscription,
        "_post_json",
        lambda url, payload, timeout=30.0: {
            "access_token": "refreshed-token",
            "expires_in": 28800,
        },
    )
    credential = subscription.anthropic_credential()
    assert credential.token == "refreshed-token"
    # The new token is written back so the next job does not refresh again.
    stored = json.loads(subscription.CLAUDE_CREDENTIALS.read_text())
    assert stored["claudeAiOauth"]["accessToken"] == "refreshed-token"


def test_refresh_failure_falls_back_to_the_stored_token(isolated, monkeypatch):
    write_claude(subscription.CLAUDE_CREDENTIALS, expires_in=60)

    def boom(url, payload, timeout=30.0):
        raise OSError("network down")

    monkeypatch.setattr(subscription, "_post_json", boom)
    # Still usable for the next few seconds; better to try than to abort.
    assert subscription.anthropic_credential().token == "sk-ant-oat01-token"


def test_check_never_refreshes(isolated, monkeypatch):
    """A status query must not spend a refresh token or rewrite the file."""
    write_claude(subscription.CLAUDE_CREDENTIALS, expires_in=60)
    monkeypatch.setattr(
        subscription,
        "_post_json",
        lambda *a, **k: pytest.fail("check must not perform a refresh"),
    )
    before = subscription.CLAUDE_CREDENTIALS.read_text()
    assert subscription.main(["subscription", "check"]) == 1  # openai unavailable
    assert subscription.CLAUDE_CREDENTIALS.read_text() == before


def test_submission_preflight_refreshes_without_printing_credentials(isolated, monkeypatch, capsys):
    write_claude(subscription.CLAUDE_CREDENTIALS, expires_in=60)
    monkeypatch.setattr(subscription, "_post_json", lambda *a, **k: {
        "access_token": "private-refreshed-token", "expires_in": 28800,
    })
    assert subscription.main(["subscription", "preflight-anthropic"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "Claude subscription preflight passed.\n"
    assert captured.err == ""


@pytest.mark.parametrize("expires_in", [-60, 60])
def test_submission_preflight_rejects_failed_refresh(isolated, monkeypatch, capsys, expires_in):
    write_claude(subscription.CLAUDE_CREDENTIALS, expires_in=expires_in)

    def fail(*args, **kwargs):
        raise OSError("private-provider-error-and-token")

    monkeypatch.setattr(subscription, "_post_json", fail)
    before = subscription.CLAUDE_CREDENTIALS.read_bytes()
    assert subscription.main(["subscription", "preflight-anthropic"]) == 2
    assert subscription.CLAUDE_CREDENTIALS.read_bytes() == before
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "refresh the CLI login" in captured.err
    assert "private-provider" not in captured.err


def test_submission_preflight_does_not_refresh_healthy_token(isolated, monkeypatch):
    write_claude(subscription.CLAUDE_CREDENTIALS)
    monkeypatch.setattr(subscription, "_post_json", lambda *a, **k: pytest.fail("unneeded refresh"))
    assert subscription.main(["subscription", "preflight-anthropic"]) == 0


def test_metered_codex_key_is_not_a_subscription(isolated):
    subscription.CODEX_AUTH.write_text(json.dumps({"OPENAI_API_KEY": "sk-proj-abc"}))
    with pytest.raises(SubscriptionError, match="codex login"):
        subscription.openai_credential()


def test_codex_plan_token_and_account_id(isolated):
    subscription.CODEX_AUTH.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "plan-token",
                    "refresh_token": "r",
                    "account_id": "acct-123",
                    "expires_at": time.time() + 3600,
                }
            }
        )
    )
    credential = subscription.openai_credential()
    assert credential.token == "plan-token"
    assert credential.account_id == "acct-123"


def test_account_id_recovered_from_id_token(isolated):
    import base64

    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "acct-from-jwt"}}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    subscription.CODEX_AUTH.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "plan-token",
                    "id_token": f"header.{payload}.signature",
                    "expires_at": time.time() + 3600,
                }
            }
        )
    )
    assert subscription.openai_credential().account_id == "acct-from-jwt"


def test_environment_survives_one_provider_missing(isolated):
    write_claude(subscription.CLAUDE_CREDENTIALS)
    env = subscription.environment()
    assert env["ANTHROPIC_AUTH_TOKEN"] == "sk-ant-oat01-token"
    assert "CHATGPT_API_KEY" not in env


def test_environment_reports_every_failure(isolated):
    with pytest.raises(SubscriptionError) as exc:
        subscription.environment()
    assert "claude setup-token" in str(exc.value)
    assert "codex login" in str(exc.value)


def test_describe_never_leaks_the_token(isolated):
    write_claude(subscription.CLAUDE_CREDENTIALS, token="super-secret-token")
    assert "super-secret-token" not in subscription.anthropic_credential().describe()


def test_dotenv_token_is_visible(tmp_path, monkeypatch):
    """A token in .env is found, since the runner loads its keys from there."""
    from dotenv import load_dotenv

    env_file = tmp_path / ".env"
    env_file.write_text("CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-from-dotenv\n")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    load_dotenv(env_file, override=False)
    assert subscription.anthropic_credential().token == "sk-ant-oat01-from-dotenv"


def test_metered_key_is_not_shadowed_by_the_plan_token(isolated, monkeypatch):
    """Clients that prefer ANTHROPIC_AUTH_TOKEN over ANTHROPIC_API_KEY would
    silently replace an exported metered key with the plan token."""
    write_claude(subscription.CLAUDE_CREDENTIALS)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-metered")
    with pytest.raises(SubscriptionError, match="metered key is used"):
        subscription.environment(providers=("anthropic",))


def test_subscription_reach_is_limited_to_haiku():
    assert subscription.subscription_can_reach("anthropic/claude-haiku-4-5-20251001")
    for blocked in ("anthropic/claude-sonnet-5", "anthropic/claude-opus-5"):
        assert not subscription.subscription_can_reach(blocked)
