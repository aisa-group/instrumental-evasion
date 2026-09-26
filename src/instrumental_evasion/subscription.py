"""Resolve Claude and ChatGPT subscription credentials.

The Claude Code and Codex scaffolds run the vendor CLI on a Claude Max/Pro or
ChatGPT plan, using the credentials the `claude` and `codex` CLIs already store
on this machine (`anthropic_credential`, `openai_credential`). A token that is
close to expiry is refreshed and written back.

Apart from `probe`, nothing here talks to a model. `export` also exposes the
credentials as environment variables for an API client:

    ANTHROPIC_AUTH_TOKEN                   Claude plan
    CHATGPT_API_KEY + CHATGPT_BASE_URL     ChatGPT plan (Codex backend)

Usage:

    uv run python -m instrumental_evasion.subscription check    # report, no secrets
    uv run python -m instrumental_evasion.subscription preflight-anthropic
    eval "$(uv run python -m instrumental_evasion.subscription export)"

`check` never prints a token. `export` does, because that is its job; it is
meant for `eval "$(...)"`, not for a log.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load the project's .env so a token parked there next to OPENROUTER_API_KEY is
# visible here -- without overriding anything already exported, so a token in
# the shell still wins.
load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)

# Where the CLIs keep their credentials.
CLAUDE_CREDENTIALS = Path.home() / ".claude" / ".credentials.json"
CODEX_AUTH = Path.home() / ".codex" / "auth.json"

# ChatGPT-plan traffic goes to the Codex backend rather than api.openai.com, and
# it is a Responses API. `environment` exports it under the service name
# "chatgpt" (<SERVICE>_API_KEY and <SERVICE>_BASE_URL), the convention of a
# generic OpenAI-compatible provider.
CHATGPT_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Public client identifiers for the two CLIs' OAuth apps. They are not secrets;
# they identify the application, not the user.
CLAUDE_OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
CLAUDE_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
CODEX_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"

# Refresh a token that expires sooner than this rather than starting a batch
# that dies partway through.
EXPIRY_MARGIN_SECONDS = 15 * 60


class SubscriptionError(RuntimeError):
    """Credentials are missing or unusable, with the command that fixes it."""


@dataclass(frozen=True)
class Credential:
    """A resolved credential and where it came from."""

    provider: str
    token: str
    source: str
    expires_at: float | None = None
    account_id: str | None = None

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= time.time()

    def describe(self) -> str:
        """A one-line status that is safe to print. Never includes the token."""
        if self.expires_at is None:
            when = "no expiry recorded"
        else:
            delta = self.expires_at - time.time()
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(self.expires_at))
            when = (
                f"expired {stamp}"
                if delta <= 0
                else f"valid until {stamp} ({delta / 3600:.1f}h)"
            )
        account = f", account {self.account_id}" if self.account_id else ""
        return f"{self.provider}: {self.source}, {when}{account}"


def _post_json(url: str, payload: dict[str, str], timeout: float = 30.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise SubscriptionError(f"{path} is unreadable: {exc}") from exc


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------


# What a subscription token can reach from a plain API client. Anthropic gates
# plan credentials to Claude Code's own traffic: the same token that drives
# `claude -p --model sonnet` fine returns 429 with no rate-limit headers for
# every model except Haiku when the caller is a plain API client. This is a
# gate, not a quota. It does not affect the Claude Code scaffold, which runs the
# CLI itself. Use a metered key for API access to anything but Haiku.
SUBSCRIPTION_REACHABLE_MODELS = ("claude-haiku-4-5",)


def subscription_can_reach(model: str) -> bool:
    """Whether a plan credential is expected to serve this model."""
    name = model.split("/")[-1]
    return any(name.startswith(prefix) for prefix in SUBSCRIPTION_REACHABLE_MODELS)


def anthropic_credential(refresh: bool = True) -> Credential:
    """Resolve a Claude subscription token.

    Prefers an explicitly exported token, because `claude setup-token` mints a
    long-lived one that needs no refresh and does not get invalidated when the
    interactive CLI re-authenticates. Falls back to the token the CLI stores,
    refreshing it when it is close to expiry.
    """
    for var in ("ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        token = os.environ.get(var)
        if token:
            return Credential("anthropic", token, f"${var}")

    oauth = _read_json(CLAUDE_CREDENTIALS).get("claudeAiOauth") or {}
    token = oauth.get("accessToken")
    if not token:
        raise SubscriptionError(
            "No Claude subscription credentials. Run `claude setup-token` and "
            "export the result as CLAUDE_CODE_OAUTH_TOKEN, or sign in with "
            "`claude` so it writes ~/.claude/.credentials.json."
        )

    # expiresAt is milliseconds since the epoch.
    expires_at = oauth.get("expiresAt")
    expires_at = float(expires_at) / 1000 if expires_at else None
    plan = oauth.get("subscriptionType", "unknown")
    source = f"{CLAUDE_CREDENTIALS} ({plan} plan)"

    stale = expires_at is not None and expires_at - time.time() < EXPIRY_MARGIN_SECONDS
    if stale and refresh:
        refreshed = _refresh_anthropic(oauth)
        if refreshed is not None:
            return refreshed

    return Credential("anthropic", token, source, expires_at)


def _refresh_anthropic(oauth: dict) -> Credential | None:
    """Exchange the refresh token for a new access token.

    Returns None rather than raising: a refresh failure should surface as the
    actionable "re-authenticate" message from the caller, not as a stack trace
    about an OAuth endpoint.
    """
    refresh_token = oauth.get("refreshToken")
    if not refresh_token:
        return None
    try:
        response = _post_json(
            CLAUDE_TOKEN_URL,
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLAUDE_OAUTH_CLIENT_ID,
            },
        )
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not refresh the Claude token: {exc}", file=sys.stderr)
        return None

    token = response.get("access_token")
    if not token:
        return None
    expires_in = response.get("expires_in")
    expires_at = time.time() + float(expires_in) if expires_in else None

    # Write the refreshed token back so the CLI and the next job both see it.
    try:
        stored = _read_json(CLAUDE_CREDENTIALS)
        block = stored.setdefault("claudeAiOauth", {})
        block["accessToken"] = token
        if response.get("refresh_token"):
            block["refreshToken"] = response["refresh_token"]
        if expires_at:
            block["expiresAt"] = int(expires_at * 1000)
        CLAUDE_CREDENTIALS.write_text(json.dumps(stored, indent=2))
        CLAUDE_CREDENTIALS.chmod(0o600)
    except OSError as exc:
        print(f"warning: refreshed token not written back: {exc}", file=sys.stderr)

    return Credential("anthropic", token, f"{CLAUDE_CREDENTIALS} (refreshed)", expires_at)


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------


def openai_credential(refresh: bool = True) -> Credential:
    """Resolve a ChatGPT-plan token from the Codex CLI's credential store."""
    token = os.environ.get("CODEX_ACCESS_TOKEN")
    if token:
        return Credential(
            "openai", token, "$CODEX_ACCESS_TOKEN",
            account_id=os.environ.get("CHATGPT_ACCOUNT_ID"),
        )

    auth = _read_json(CODEX_AUTH)
    tokens = auth.get("tokens") or {}
    token = tokens.get("access_token")
    if not token:
        # A bare OPENAI_API_KEY in auth.json is a metered key, not a plan.
        if auth.get("OPENAI_API_KEY"):
            raise SubscriptionError(
                f"{CODEX_AUTH} holds a metered OPENAI_API_KEY, not a ChatGPT "
                "plan token. Run `codex login` and choose 'Sign in with "
                "ChatGPT' to authorize the subscription."
            )
        raise SubscriptionError(
            "No ChatGPT plan credentials. Install the Codex CLI and run "
            "`codex login`, or export CODEX_ACCESS_TOKEN."
        )

    account_id = tokens.get("account_id") or _account_id_from_id_token(
        tokens.get("id_token")
    )
    expires_at = _codex_expiry(auth, tokens)

    stale = expires_at is not None and expires_at - time.time() < EXPIRY_MARGIN_SECONDS
    if stale and refresh:
        refreshed = _refresh_openai(auth, tokens, account_id)
        if refreshed is not None:
            return refreshed

    return Credential("openai", token, str(CODEX_AUTH), expires_at, account_id)


def _account_id_from_id_token(id_token: str | None) -> str | None:
    """Pull the account id out of the id_token's claims.

    The Codex backend needs a chatgpt-account-id header. Newer auth.json files
    record it directly; older ones only carry it inside the id_token.
    """
    if not id_token:
        return None
    try:
        import base64

        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (ValueError, IndexError, json.JSONDecodeError):
        return None
    auth_claims = claims.get("https://api.openai.com/auth") or {}
    return auth_claims.get("chatgpt_account_id") or claims.get("chatgpt_account_id")


def _codex_expiry(auth: dict, tokens: dict) -> float | None:
    if tokens.get("expires_at"):
        return float(tokens["expires_at"])
    # Otherwise infer from the id_token's exp claim.
    id_token = tokens.get("id_token")
    if id_token:
        try:
            import base64

            payload = id_token.split(".")[1]
            payload += "=" * (-len(payload) % 4)
            exp = json.loads(base64.urlsafe_b64decode(payload)).get("exp")
            if exp:
                return float(exp)
        except (ValueError, IndexError, json.JSONDecodeError):
            pass
    return None


def _refresh_openai(auth: dict, tokens: dict, account_id: str | None) -> Credential | None:
    refresh_token = tokens.get("refresh_token")
    if not refresh_token:
        return None
    try:
        response = _post_json(
            CODEX_TOKEN_URL,
            {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CODEX_OAUTH_CLIENT_ID,
                "scope": "openid profile email",
            },
        )
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not refresh the Codex token: {exc}", file=sys.stderr)
        return None

    token = response.get("access_token")
    if not token:
        return None
    expires_in = response.get("expires_in")
    expires_at = time.time() + float(expires_in) if expires_in else None

    try:
        tokens = dict(tokens)
        tokens["access_token"] = token
        if response.get("refresh_token"):
            tokens["refresh_token"] = response["refresh_token"]
        if response.get("id_token"):
            tokens["id_token"] = response["id_token"]
        if expires_at:
            tokens["expires_at"] = expires_at
        auth = dict(auth)
        auth["tokens"] = tokens
        CODEX_AUTH.write_text(json.dumps(auth, indent=2))
        CODEX_AUTH.chmod(0o600)
    except OSError as exc:
        print(f"warning: refreshed token not written back: {exc}", file=sys.stderr)

    return Credential(
        "openai", token, f"{CODEX_AUTH} (refreshed)", expires_at, account_id
    )


# --------------------------------------------------------------------------
# Applying credentials
# --------------------------------------------------------------------------


def environment(providers: tuple[str, ...] = ("anthropic", "openai")) -> dict[str, str]:
    """Build the environment variables an API client needs.

    Only the providers that actually resolve are included, so a machine with a
    Claude plan and no ChatGPT plan still gets working Anthropic models.
    """
    env: dict[str, str] = {}
    errors: list[str] = []

    if "anthropic" in providers:
        # A metered key wins. A client that prefers ANTHROPIC_AUTH_TOKEN over
        # ANTHROPIC_API_KEY would let an exported plan token silently shadow the
        # paid key and 429 every model except Haiku.
        if os.environ.get("ANTHROPIC_API_KEY"):
            errors.append(
                "anthropic: ANTHROPIC_API_KEY is set, so the metered key is "
                "used and the subscription token is left out deliberately."
            )
        else:
            try:
                credential = anthropic_credential()
                env["ANTHROPIC_AUTH_TOKEN"] = credential.token
            except SubscriptionError as exc:
                errors.append(str(exc))

    if "openai" in providers:
        try:
            credential = openai_credential()
            env["CHATGPT_API_KEY"] = credential.token
            env["CHATGPT_BASE_URL"] = CHATGPT_BASE_URL
            if credential.account_id:
                env["CHATGPT_ACCOUNT_ID"] = credential.account_id
        except SubscriptionError as exc:
            errors.append(str(exc))

    if not env:
        raise SubscriptionError(
            "No subscription credentials resolved.\n  " + "\n  ".join(errors)
        )
    return env


def apply() -> dict[str, str]:
    """Resolve credentials into os.environ and return what was set."""
    env = environment()
    os.environ.update(env)
    return env


def probe(model: str, timeout: float = 60.0) -> tuple[bool, str]:
    """Make one minimal call against `model` and report whether it worked.

    A plan's frontier models can be rate limited while its small models are
    fine, and this harness fails closed: a monitor that 429s reads as a clean
    refusal and a whole condition scores zero for a reason that has nothing to
    do with the agent. Probing before a run keeps that out of the results.
    """
    import anthropic

    apply()
    if model.startswith("anthropic/"):
        client = anthropic.Anthropic(
            auth_token=os.environ["ANTHROPIC_AUTH_TOKEN"],
            default_headers={"anthropic-beta": "oauth-2025-04-20"},
            max_retries=0,
            timeout=timeout,
        )
        try:
            client.messages.create(
                model=model.split("/", 1)[1],
                max_tokens=4,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            return True, "reachable"
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            return False, f"{type(exc).__name__}: {str(exc)[:160]}"

    return False, f"no probe implemented for {model}"


def _probe(models: list[str]) -> int:
    status = 0
    for model in models:
        ok, detail = probe(model)
        print(f"  {model:<44} {'OK' if ok else 'UNUSABLE'}  {detail}")
        if not ok:
            status = 1
    return status


def _check() -> int:
    # refresh=False: reporting status must not spend a refresh token or rewrite
    # a credential file as a side effect of being asked a question.
    status = 0
    for name, resolve in (
        ("anthropic", anthropic_credential),
        ("openai", openai_credential),
    ):
        try:
            print("  " + resolve(refresh=False).describe())
        except SubscriptionError as exc:
            print(f"  {name}: unavailable -- {exc}")
            status = 1
    return status


def _export() -> int:
    for key, value in environment().items():
        print(f"export {key}={shlex.quote(value)}")
    return 0


def _preflight_anthropic() -> int:
    """Refresh if needed and check expiry without printing credential data."""
    import contextlib
    import io

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            credential = anthropic_credential()
        if credential.expired or (
            credential.expires_at is not None
            and credential.expires_at - time.time() < EXPIRY_MARGIN_SECONDS
        ):
            raise SubscriptionError("credential expires too soon")
    except (SubscriptionError, OSError, ValueError):
        print("Claude subscription preflight failed; refresh the CLI login before submitting.", file=sys.stderr)
        return 2
    print("Claude subscription preflight passed.")
    return 0


def main(argv: list[str]) -> int:
    command = argv[1] if len(argv) > 1 else "check"
    if command == "check":
        print("subscription credentials:")
        return _check()
    if command == "preflight-anthropic":
        return _preflight_anthropic()
    if command == "export":
        try:
            return _export()
        except SubscriptionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    if command == "probe":
        models = argv[2:]
        if not models:
            print("usage: probe <model> [model ...]", file=sys.stderr)
            return 2
        print("probing subscription models:")
        try:
            return _probe(models)
        except SubscriptionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    print(f"usage: {argv[0]} [check|preflight-anthropic|export|probe <model>]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
