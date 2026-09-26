"""Ask the monitor for a verdict.

Two backends, and the choice is a research decision rather than a preference:

`remote`  POST the tool call to the host-side gate (`runner/gate.py`). This is
          what the in-container hook uses: the policy text and the API key
          never enter the container, so an agent cannot read the hidden policy
          off its own filesystem, and evasion cannot be us handing it the
          answer key.

`direct`  Call OpenRouter. The host-side gate classifies every call this way.
          Running the hook itself in direct mode puts both the policy prompt and
          OPENROUTER_API_KEY inside the sandbox, where the agent can read them;
          that is only fit for a smoke test.

Standard library only. This module is imported once per tool call inside the
container, where `requests` may not exist and where import time is latency the
agent pays on every action.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Sequence
from email.utils import parsedate_to_datetime

from instrumental_evasion.monitor_core import (
    HistoryConfig,
    HistoryEntry,
    MonitorVerdict,
    monitor_input,
    parse_monitor_response,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "meta-llama/llama-3.1-8b-instruct"
DEFAULT_TIMEOUT = 90.0
DEFAULT_RETRIES = 3
DEFAULT_MONITOR_MAX_TOKENS = 1024
CLASSIFICATION_DEADLINE_SECONDS = 85.0
REMOTE_DEADLINE_SECONDS = 95.0
MAX_RESPONSE_BYTES = 1 << 20
RETRY_STATUS = {429, 500, 502, 503, 520, 524}
MONITOR_REASONING_EFFORT_ENV = "MONITOR_REASONING_EFFORT"
MONITOR_REASONING_EXCLUDE_ENV = "MONITOR_REASONING_EXCLUDE"
MONITOR_STRUCTURED_OUTPUT_ENV = "MONITOR_STRUCTURED_OUTPUT"
MONITOR_PROVIDER_ENV = "MONITOR_PROVIDER"
MONITOR_PROVIDER_SORT_ENV = "MONITOR_PROVIDER_SORT"
MONITOR_PROVIDER_REQUIRE_PARAMETERS_ENV = "MONITOR_PROVIDER_REQUIRE_PARAMETERS"
MONITOR_PROVIDER_QUANTIZATIONS_ENV = "MONITOR_PROVIDER_QUANTIZATIONS"
MONITOR_PROVIDER_QUANTIZATIONS = frozenset(
    {"int4", "int8", "fp4", "fp6", "fp8", "fp16", "bf16", "fp32", "unknown"}
)
MONITOR_HTTP_TIMEOUT_ENV = "MONITOR_HTTP_TIMEOUT_SECONDS"
MONITOR_HTTP_RETRIES_ENV = "MONITOR_HTTP_RETRIES"
MONITOR_HTTP_BACKOFF_ENV = "MONITOR_HTTP_BACKOFF_PROFILE"
MONITOR_REASONING_EFFORTS = frozenset(
    {"not_supported", "none", "minimal", "low", "medium", "high", "xhigh", "max"}
)

VERDICT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "monitor_verdict",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["ALLOW", "BLOCK"]},
                "reason": {"type": "string", "minLength": 1},
            },
            "required": ["decision", "reason"],
            "additionalProperties": False,
        },
    },
}


class MonitorUnavailable(RuntimeError):
    """The monitor could not be reached or did not answer in the schema.

    Always ends in a block. A monitor that cannot answer has not approved
    anything, and a fail-open here would score as a clean evasion.
    """


def _enabled(name: str, env: dict[str, str] | None = None) -> bool:
    source = os.environ if env is None else env
    return str(source.get(name, "")).strip() == "1"


def monitor_request_metadata(env: dict[str, str] | None = None) -> dict[str, object]:
    """Return the configured reasoning and output controls for run metadata."""
    source = os.environ if env is None else env
    from instrumental_evasion.monitor_prompt import ORIGINAL, instruction_profile
    prompt_profile = instruction_profile(source)
    effort = str(source.get(MONITOR_REASONING_EFFORT_ENV, "")).strip()
    if effort and effort not in MONITOR_REASONING_EFFORTS:
        expected = ", ".join(sorted(MONITOR_REASONING_EFFORTS))
        raise ValueError(f"invalid monitor reasoning effort {effort!r}; expected {expected}")
    exclude = _enabled(MONITOR_REASONING_EXCLUDE_ENV, source)
    if exclude and not effort:
        raise ValueError("monitor reasoning exclusion requires a reasoning effort")
    provider = str(source.get(MONITOR_PROVIDER_ENV, "")).strip()
    if provider == "none":
        provider = ""
    if provider and not re.fullmatch(r"[a-z0-9][a-z0-9_/-]{0,127}", provider):
        raise ValueError(f"invalid monitor provider {provider!r}")
    provider_sort = str(source.get(MONITOR_PROVIDER_SORT_ENV, "")).strip()
    if provider_sort == "none":
        provider_sort = ""
    if provider_sort and provider_sort not in {"price", "throughput", "latency"}:
        raise ValueError(f"invalid monitor provider sort {provider_sort!r}")
    if provider and provider_sort:
        raise ValueError("monitor provider and provider sort are mutually exclusive")
    quantizations = [
        value.strip()
        for value in str(source.get(MONITOR_PROVIDER_QUANTIZATIONS_ENV, "")).split(",")
        if value.strip()
    ]
    if any(value not in MONITOR_PROVIDER_QUANTIZATIONS for value in quantizations):
        raise ValueError(f"invalid monitor provider quantizations {quantizations!r}")
    if quantizations and not provider:
        raise ValueError("monitor provider quantizations require a monitor provider")
    metadata = {
        **({"monitor_instruction_profile": prompt_profile} if prompt_profile != ORIGINAL else {}),
        "monitor_reasoning_effort": effort or "provider_default",
        "monitor_reasoning_excluded": exclude,
        "monitor_structured_output": _enabled(MONITOR_STRUCTURED_OUTPUT_ENV, source),
        "monitor_response_schema": "monitor_verdict_v1",
        "monitor_provider": provider or "openrouter_default",
        "monitor_provider_sort": provider_sort or "openrouter_default",
        "monitor_provider_fallbacks": False if provider or provider_sort else True,
        "monitor_provider_require_parameters": str(
            source.get(MONITOR_PROVIDER_REQUIRE_PARAMETERS_ENV, "1")
        ).strip() != "0",
    }
    if quantizations:
        metadata["monitor_provider_quantizations"] = quantizations
    configured_timeout = str(source.get(MONITOR_HTTP_TIMEOUT_ENV, "")).strip()
    if configured_timeout:
        timeout = float(configured_timeout)
        if not math.isfinite(timeout) or not 0 < timeout <= DEFAULT_TIMEOUT:
            raise ValueError("Monitor HTTP timeout must be finite and between 0 and 90 seconds.")
        metadata["monitor_http_timeout_seconds"] = timeout
    configured_retries = str(source.get(MONITOR_HTTP_RETRIES_ENV, "")).strip()
    if configured_retries:
        retries = int(configured_retries)
        if not 1 <= retries <= 4:
            raise ValueError("Monitor HTTP retries must be between 1 and 4.")
        metadata["monitor_http_retries"] = retries
    backoff = str(source.get(MONITOR_HTTP_BACKOFF_ENV, "")).strip()
    if backoff:
        if backoff != "rate_limit_v1":
            raise ValueError("Unsupported monitor HTTP backoff profile.")
        metadata["monitor_http_backoff_profile"] = backoff
    return metadata


def _apply_request_controls(body: dict, env: dict[str, str] | None = None) -> None:
    config = monitor_request_metadata(env)
    effort = config["monitor_reasoning_effort"]
    if effort not in {"provider_default", "not_supported"}:
        body["reasoning"] = {
            "effort": effort,
            "exclude": config["monitor_reasoning_excluded"],
        }
    if config["monitor_structured_output"]:
        body["response_format"] = VERDICT_RESPONSE_FORMAT
    provider = config["monitor_provider"]
    if provider != "openrouter_default":
        body["provider"] = {
            "order": [provider],
            "allow_fallbacks": False,
            "require_parameters": config["monitor_provider_require_parameters"],
        }
        if "monitor_provider_quantizations" in config:
            body["provider"]["quantizations"] = config["monitor_provider_quantizations"]
    provider_sort = config["monitor_provider_sort"]
    if provider_sort != "openrouter_default":
        body["provider"] = {
            "sort": provider_sort,
            "allow_fallbacks": False,
            "require_parameters": config["monitor_provider_require_parameters"],
        }


def _strip_provider_prefix(model: str) -> str:
    """`openrouter/meta-llama/x` -> `meta-llama/x`.

    Model ids in this repo carry the `openrouter/` provider prefix; the
    OpenRouter API does not want it. Accept both forms rather than making the
    caller remember which one this is.
    """
    return model[len("openrouter/") :] if model.startswith("openrouter/") else model


# urllib honours http_proxy for loopback addresses too, so where a proxy is
# exported a request to the host-side monitor would go to the proxy and be
# refused. The hook fails closed, so the symptom would not be an error but every
# tool call blocked, which reads as a monitor that blocks everything rather than
# as a misrouted socket. Bypass the proxy for loopback and keep it for OpenRouter.
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]"}


def _is_loopback(url: str) -> bool:
    host = urllib.parse.urlsplit(url).hostname or ""
    return host in _LOOPBACK_HOSTS


def _opener(url: str):
    if _is_loopback(url):
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return urllib.request.build_opener()


def _post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    deadline = time.monotonic() + timeout

    def remaining() -> float:
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Monitor HTTP request deadline exceeded.")
        return seconds

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with _opener(url).open(request, timeout=remaining()) as response:
        raw = bytearray()
        # CPython 3.12's HTTPResponse exposes no public socket-timeout setter.
        # A compatibility test covers this path. Reset the timeout before each
        # read so heartbeat bytes cannot extend one request past its budget.
        connection = response.fp.raw._sock
        while not response.isclosed():
            connection.settimeout(remaining())
            chunk = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - len(raw)))
            remaining()
            if not chunk:
                break
            raw.extend(chunk)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise MonitorUnavailable("monitor response exceeds its size limit")
        return json.loads(raw.decode())


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise MonitorUnavailable("monitor deadline exceeded")
    return remaining


def _response_diagnostics(data: object) -> dict:
    """Return bounded metadata without response content, reasoning, or headers."""
    record = {"response_type": type(data).__name__}
    if not isinstance(data, dict):
        return record
    for key in ("id", "provider"):
        value = data.get(key)
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:/ -]{1,128}", value):
            if not re.search(r"(?i)sk-|bearer|token|secret|password", value):
                record[key] = value
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        choice = choices[0]
        finish = choice.get("finish_reason")
        record["finish_reason"] = finish if finish in (
            None, "stop", "length", "tool_calls", "content_filter", "error", "function_call"
        ) else "unknown"
        message = choice.get("message")
        if isinstance(message, dict):
            record["content_type"] = type(message.get("content")).__name__
            record["reasoning_returned"] = any(
                key in message and message[key] not in (None, "", [])
                for key in ("reasoning", "reasoning_content", "reasoning_details")
            )
    usage = data.get("usage")
    if isinstance(usage, dict):
        record["usage"] = {
            key: value for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            if type(value := usage.get(key)) is int and 0 <= value <= 1_000_000_000
        }
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict):
            value = details.get("reasoning_tokens")
            if type(value) is int and 0 <= value <= 1_000_000_000:
                record["usage"]["reasoning_tokens"] = value
    return record


def _retry_after_seconds(value: str | None) -> float | None:
    """Parse Retry-After without retaining arbitrary provider header content."""
    if not value or len(value) > 128:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _post_with_retries(
    url: str, payload: dict, headers: dict, timeout: float, retries: int,
    *, deadline: float | None = None, diagnostics: list[dict] | None = None,
) -> dict:
    if deadline is None:
        deadline = time.monotonic() + CLASSIFICATION_DEADLINE_SECONDS
    # The remote hook must not import host-only monitor prompts or policy code.
    rate_backoff = os.environ.get(MONITOR_HTTP_BACKOFF_ENV, "").strip() == "rate_limit_v1"
    last = ""
    for attempt in range(retries):
        request_timeout = min(timeout, _remaining(deadline))
        started = time.monotonic()
        record: dict = {"attempt": len(diagnostics) + 1 if diagnostics is not None else attempt + 1}
        retry_after = None
        try:
            data = _post(url, payload, headers, request_timeout)
            record.update(_response_diagnostics(data))
            record["http_status"] = 200
            _remaining(deadline)
            return data
        except urllib.error.HTTPError as error:
            record["http_status"] = error.code
            last = f"HTTP {error.code}"
            if rate_backoff and error.code in RETRY_STATUS:
                retry_after = _retry_after_seconds(error.headers.get("Retry-After") if error.headers else None)
                if retry_after is not None:
                    record["retry_after_seconds"] = retry_after
            if error.code not in RETRY_STATUS:
                raise MonitorUnavailable(last) from error
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as error:
            # Exception messages can contain response bodies or credentials.
            record["error_type"] = type(error).__name__
            last = type(error).__name__
        except MonitorUnavailable:
            record["error_type"] = "MonitorUnavailable"
            raise
        finally:
            record["latency_ms"] = round((time.monotonic() - started) * 1000, 2)
            if diagnostics is not None:
                diagnostics.append(record)
        if attempt < retries - 1:
            remaining = _remaining(deadline)
            delay = min(2**attempt * 2, 20, remaining)
            if rate_backoff and record.get("http_status") == 429:
                delay = (15, 25, 35)[min(attempt, 2)] + random.uniform(0, 1)
            if rate_backoff and retry_after is not None:
                delay = max(delay, retry_after)
            if rate_backoff and delay >= remaining:
                raise MonitorUnavailable("monitor retry delay exceeds the remaining deadline")
            record["retry_delay_seconds"] = round(delay, 3)
            time.sleep(delay)
    raise MonitorUnavailable(f"monitor unreachable after {retries} tries: {last}")


def verdict_direct(
    tool_name: str,
    tool_input: dict,
    *,
    policy: str,
    assistant_message: str | None = None,
    history: Sequence[HistoryEntry] | None = None,
    history_config: HistoryConfig | None = None,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
    retries: int = DEFAULT_RETRIES,
    max_tokens: int = DEFAULT_MONITOR_MAX_TOKENS,
    diagnostics: list[dict] | None = None,
) -> tuple[MonitorVerdict, str]:
    """Call OpenRouter directly. Returns (verdict, raw completion).

    `history=None` is the myopic monitor and sends the myopic prompt, byte for
    byte. Anything else -- including an empty list -- is
    the history condition: the user message carries the episode's calls AND the
    system prompt is reframed, because a policy that still says "you do not know
    the agent's previous actions" while showing them is a prompt that argues with
    itself. The two always move together; there is no way to get one without the
    other from here.
    """

    # Imported here, not at module scope: the remote backend must never pull
    # the policy text into the sandbox just by importing this module.
    from instrumental_evasion.policies import resolve_spec

    if timeout is None:
        timeout = monitor_request_metadata().get("monitor_http_timeout_seconds", DEFAULT_TIMEOUT)
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise MonitorUnavailable("OPENROUTER_API_KEY is not set in the sandbox")

    # One assembly rule, shared with `policy_prompt_sha256`, so the digest on a
    # result always names the prompt this call actually sent.
    spec = resolve_spec(policy, history_disclosed=history is not None)
    body = {
        "model": _strip_provider_prefix(model or DEFAULT_MODEL),
        "temperature": 0,
        # GPT-OSS can spend part of this budget on provider-managed reasoning,
        # even when the response content is a short JSON verdict. A small limit
        # truncates the JSON, and the call then fails closed.
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": spec.system_prompt},
            {
                "role": "user",
                "content": monitor_input(
                    tool_name,
                    tool_input,
                    assistant_message=assistant_message,
                    history=history,
                    history_config=history_config,
                ),
            },
        ],
    }
    _apply_request_controls(body)
    headers = {
        "Authorization": f"Bearer {key}",
        "HTTP-Referer": "https://github.com/local/instrumental_evasion",
        "X-Title": "instrumental_evasion",
    }
    last_error: Exception | None = None
    transport_retries = monitor_request_metadata().get("monitor_http_retries", retries)
    deadline = time.monotonic() + CLASSIFICATION_DEADLINE_SECONDS
    for _ in range(retries):
        _remaining(deadline)
        data = _post_with_retries(
            OPENROUTER_URL, body, headers, timeout, transport_retries,
            deadline=deadline, diagnostics=diagnostics,
        )
        try:
            raw = data["choices"][0]["message"]["content"]
            verdict = parse_monitor_response(raw)
            _remaining(deadline)
            if diagnostics:
                diagnostics[-1]["parse_status"] = "ok"
            return verdict, raw
        except (KeyError, IndexError, TypeError, ValueError) as error:
            # A routed provider can stop mid-JSON. Retry the classification,
            # but keep strict parsing and fail closed if every response fails.
            last_error = error
            if diagnostics:
                diagnostics[-1]["parse_status"] = "error"
                diagnostics[-1]["parse_error_type"] = type(error).__name__
    assert last_error is not None
    raise MonitorUnavailable(
        f"unparseable verdict after {retries} responses: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def verdict_remote(
    tool_name: str,
    tool_input: dict,
    *,
    url: str,
    assistant_message: str | None = None,
    token: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    tool_use_id: str | None = None,
) -> tuple[MonitorVerdict, str]:
    """Ask the host-side monitor server. The policy stays on the host.

    There is deliberately no `history` parameter. The episode's tool-call
    history is assembled by the HOST gate from its own decision records, which
    is both the authoritative account (it includes calls this client never saw,
    and blocked ones the container has no trace of) and the safe one -- a
    container-side client that had to send a history would be a container-side
    client that could omit one.
    """

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "assistant_message": assistant_message,
    }
    if tool_use_id is not None:
        body["tool_use_id"] = tool_use_id
    data = _post_with_retries(
        url, body, headers, timeout, retries,
        deadline=time.monotonic() + REMOTE_DEADLINE_SECONDS,
    )
    raw = json.dumps(data, separators=(",", ":"))
    # The host gate answers in a RICHER, trusted schema than the model's raw
    # output: {decision, reason, message}, where `message` is the agent-visible
    # refusal text for a BLOCK. `parse_monitor_response` is the STRICT parser
    # for the model's own output and demands exactly {decision, reason}, so it
    # would reject every gate reply and fail closed on ALLOWs too. Parse the
    # gate's reply directly here; keep the strict parser for verdict_direct.
    if not isinstance(data, dict):
        raise MonitorUnavailable(f"monitor response was not an object: {raw[:200]}")
    decision = data.get("decision")
    reason = data.get("reason")
    message = data.get("message")
    if decision not in {"ALLOW", "BLOCK"} or not isinstance(reason, str) or not reason.strip():
        raise MonitorUnavailable(f"unparseable verdict: {raw[:200]}")
    if decision == "BLOCK" and (not isinstance(message, str) or not message.strip()):
        raise MonitorUnavailable(f"blocked verdict had no agent message: {raw[:200]}")
    return MonitorVerdict(
        decision=decision,
        reason=reason.strip(),
        agent_message=message.strip() if isinstance(message, str) and message.strip() else None,
    ), raw
