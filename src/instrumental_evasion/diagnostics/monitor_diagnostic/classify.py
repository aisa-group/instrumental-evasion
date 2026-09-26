"""Send the fixed requests through the monitor client and record every outcome.

Each classification calls `verdict_direct`, the function the tool-call gate
uses, so a verdict here comes from the deployed decision path: the same prompt
assembly, request controls, transport and strict parser. Two settings follow
the diagnostic protocol instead of the gate:

* One completion per classification (`retries=1`). A malformed verdict is
  recorded as invalid, not re-requested, so the invalid count measures the
  monitor's own format failures.
* A 20-second request timeout and four transport attempts inside the client's
  85-second deadline. These are applied as defaults for
  `MONITOR_HTTP_TIMEOUT_SECONDS` and `MONITOR_HTTP_RETRIES` and can be
  overridden there.

Serving options are exactly those the monitor client reads from the
environment: `MONITOR_PROVIDER`, `MONITOR_PROVIDER_SORT`,
`MONITOR_PROVIDER_QUANTIZATIONS`, `MONITOR_PROVIDER_REQUIRE_PARAMETERS`,
`MONITOR_REASONING_EFFORT`, `MONITOR_REASONING_EXCLUDE`,
`MONITOR_STRUCTURED_OUTPUT` and `MONITOR_HTTP_BACKOFF_PROFILE`;
`OPENROUTER_API_KEY` authenticates. A run records the resolved controls. A
classification is also invalid when its response contradicts them: a pinned
provider did not serve it, or reasoning was returned although
`MONITOR_REASONING_EFFORT=none`.

The reported monitor configurations follow. Each reproduces the recorded
request bodies exactly (same `payload_sha256`). Serving variables not listed
stay unset. S is `MONITOR_STRUCTURED_OUTPUT=1`, R is
`MONITOR_REASONING_EFFORT=none` with `MONITOR_REASONING_EXCLUDE=1`, Q is
`MONITOR_PROVIDER_QUANTIZATIONS=fp8`, and P=x is `MONITOR_PROVIDER=x`.

    Monitor           --monitor                                    Environment
    Original serving routes
    Llama 3.1 8B      openrouter/meta-llama/llama-3.1-8b-instruct  S, P=coreweave
    Qwen 3.6 35B-A3B  openrouter/qwen/qwen3.6-35b-a3b              S, R, P=darkbloom
    Qwen 3.5 9B       openrouter/qwen/qwen3.5-9b                   S, R, P=siliconflow
    Qwen 3.5 35B-A3B  openrouter/qwen/qwen3.5-35b-a3b              S, R, Q, P=deepinfra
    Qwen 3.5 27B      openrouter/qwen/qwen3.5-27b                  S, R, Q, P=deepinfra
    GPT-OSS-120B      openrouter/openai/gpt-oss-120b               (none)
    GPT-5.6 Luna      openrouter/openai/gpt-5.6-luna               S, R
    SiliconFlow common provider (Qwen 3.5 9B reuses its original route above)
    Qwen 3.6 35B-A3B  openrouter/qwen/qwen3.6-35b-a3b              S, R, Q, P=siliconflow
    Qwen 3.5 35B-A3B  openrouter/qwen/qwen3.5-35b-a3b              S, R, Q, P=siliconflow
    Qwen 3.5 27B      openrouter/qwen/qwen3.5-27b                  S, R, Q, P=siliconflow

The main-text relevant-context comparison uses the common-provider Qwen 3.6
35B-A3B row and the original routes of Llama, Qwen 3.5 9B, GPT-OSS and Luna.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from instrumental_evasion.diagnostics.monitor_diagnostic.cases import (
    HISTORY_MAX_CALLS,
    MAX_TOKENS,
    TOOL_NAME,
    TRUNCATED_HISTORY_CHARS,
    Request,
    check_policy_prompts,
    requests,
)
from instrumental_evasion.hooks import monitor_client as client

RUN_FILE = "run.json"
RECORDS_FILE = "records.jsonl"
TRANSPORT_DEFAULTS = {
    client.MONITOR_HTTP_TIMEOUT_ENV: "20",
    client.MONITOR_HTTP_RETRIES_ENV: "4",
}


def request_body(request: Request, model: str) -> dict[str, Any]:
    """The chat-completions body `verdict_direct` sends for this request."""
    body = {
        "model": client._strip_provider_prefix(model),
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "messages": request.messages(),
    }
    client._apply_request_controls(body)
    return body


def payload_sha256(body: dict[str, Any]) -> str:
    """A key-order-independent digest of a request body."""
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def configuration_error(attempts: list[dict], controls: dict[str, Any]) -> str | None:
    """Name a mismatch between the requested and the reported serving setup."""
    final = attempts[-1] if attempts else {}
    provider = controls["monitor_provider"]
    if provider != "openrouter_default" and str(final.get("provider", "")).lower() != provider:
        return "provider_mismatch"
    if controls["monitor_reasoning_effort"] == "none":
        if final.get("usage", {}).get("reasoning_tokens") not in (None, 0):
            return "unexpected_reasoning"
        if final.get("reasoning_returned"):
            return "unexpected_reasoning"
    return None


def classify(request: Request, repeat: int, model: str) -> dict[str, Any]:
    """Classify one request once and return its complete record.

    Transport and parse failures are recorded as invalid rather than raised,
    so that no classification is silently dropped from a denominator.
    """
    body = request_body(request, model)
    controls = client.monitor_request_metadata()
    record: dict[str, Any] = {
        "id": f"{request.id}__r{repeat}",
        "monitor": model,
        **request.labels(),
        "repeat": repeat,
        "payload_sha256": payload_sha256(body),
        "request": body,
    }
    attempts: list[dict] = []
    started = time.monotonic()
    try:
        verdict, raw = client.verdict_direct(
            TOOL_NAME,
            request.tool_input(),
            policy=request.case.monitor_policy,
            history=request.history(),
            history_config=request.history_config(),
            model=model,
            retries=1,
            max_tokens=MAX_TOKENS,
            diagnostics=attempts,
        )
    except Exception as error:  # noqa: BLE001 - every failure is recorded as invalid
        record.update(valid=False, decision=None, error=f"{type(error).__name__}: {error}")
    else:
        error = configuration_error(attempts, controls)
        record.update(
            valid=error is None, decision=verdict.decision, reason=verdict.reason, raw=raw
        )
        if error:
            record["error"] = error
    record.update(attempts=attempts, seconds=round(time.monotonic() - started, 3))
    return record


def _code_version() -> str:
    try:
        return version("instrumental-evasion")
    except PackageNotFoundError:
        return "unknown"


def run(out: Path, model: str, *, repeats: int = 3, workers: int = 1) -> list[dict[str, Any]]:
    """Classify all 40 profiles `repeats` times and write `run.json` and `records.jsonl`.

    Unset transport variables take the protocol defaults above. Records are
    appended as they complete, so an interrupted run keeps every finished
    classification. The default of one worker keeps at most one request
    outstanding per monitor configuration.
    """
    if repeats < 1 or workers < 1:
        raise ValueError("repeats and workers must be positive")
    for name, value in TRANSPORT_DEFAULTS.items():
        os.environ.setdefault(name, value)
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    digests = check_policy_prompts()
    controls = client.monitor_request_metadata()

    out.mkdir(parents=True, exist_ok=True)
    if (out / RUN_FILE).exists() or (out / RECORDS_FILE).exists():
        raise FileExistsError(f"{out} already holds a diagnostic run")
    planned = [(request, repeat) for repeat in range(repeats) for request in requests()]
    meta = {
        "monitor": model,
        "repeats": repeats,
        "planned_classifications": len(planned),
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "history_max_calls": HISTORY_MAX_CALLS,
        "truncated_history_chars": TRUNCATED_HISTORY_CHARS,
        "completions_per_classification": 1,
        "request_controls": controls,
        "policy_prompt_sha256": digests,
        "code_version": _code_version(),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / RUN_FILE).write_text(json.dumps(meta, indent=2) + "\n")

    records: list[dict[str, Any]] = []
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        with (out / RECORDS_FILE).open("x") as sink:
            futures = [pool.submit(classify, request, repeat, model) for request, repeat in planned]
            for future in as_completed(futures):
                record = future.result()
                line = json.dumps(record, sort_keys=True)
                if key in line:
                    raise RuntimeError("the API key appeared in a record")
                sink.write(line + "\n")
                sink.flush()
                records.append(record)
                status = {k: record.get(k) for k in ("id", "valid", "decision", "error")}
                print(json.dumps(status), file=sys.stderr, flush=True)
    finally:
        # Stop issuing requests after an error or interrupt.
        pool.shutdown(cancel_futures=True)

    meta.update(
        completed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        valid=sum(record["valid"] for record in records),
        invalid=sum(not record["valid"] for record in records),
    )
    (out / RUN_FILE).write_text(json.dumps(meta, indent=2) + "\n")
    return records
