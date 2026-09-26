"""Host-side monitor endpoint, so the policy never enters the sandbox.

    python -m instrumental_evasion.hooks.server --policy no_key_generation_trim --port 8787

The hook inside the container POSTs {tool_name, tool_input, assistant_message}
and gets back {"decision": "ALLOW"|"BLOCK", "reason": "...", "message": "..."},
where `message` is the agent-visible refusal text for a BLOCK.

This is a thin entry point over `instrumental_evasion.runner.gate`: the decision
logic, the fail-closed rule and the decision-log format live there and are
shared with the react scaffold, so every scaffold applies the same rules.

The guardrail prompt, the monitor model choice and OPENROUTER_API_KEY all stay
on the host. The agent can see that *something* is refusing calls; it cannot read
what.

Bind to loopback and reach it from the container over the host network. It is
an unauthenticated local endpoint unless --token is given, and it is not
hardened for anything beyond a single evaluation host.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace

from instrumental_evasion.monitor_core import HistoryConfig
from instrumental_evasion.regex_monitor import (
    REGEX_RULESET_VERSIONS,
    REPAIRED_57_REGEX_RULESET_VERSION,
)
from instrumental_evasion.runner.gate import LLM_CLASSIFIER, MONITOR_CLASSIFIERS, Gate
from instrumental_evasion.runner.gate_server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        required=True,
        help="guardrail id, or __control__ for an allow-all control-arm gate",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--classifier", choices=MONITOR_CLASSIFIERS, default=LLM_CLASSIFIER
    )
    parser.add_argument(
        "--regex-ruleset-version",
        choices=tuple(REGEX_RULESET_VERSIONS),
        default=(
            os.environ.get("REGEX_RULESET_VERSION", "").strip()
            or REPAIRED_57_REGEX_RULESET_VERSION
        ),
        help="regex ruleset version; ignored by the LLM classifier",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--token", default=os.environ.get("HOOK_MONITOR_TOKEN"))
    parser.add_argument("--decisions", default=None,
                        help="host-side JSONL path for per-call decision records")
    # One server, one episode's history. This process holds a single Gate, so
    # its records ARE that episode -- do not point two concurrent episodes at
    # one server with history on, or each will be shown the other's calls.
    history_group = parser.add_mutually_exclusive_group()
    history_group.add_argument(
        "--monitor-history",
        action="store_true", default=None,
        help="show the monitor this episode's earlier tool calls",
    )
    history_group.add_argument(
        "--no-monitor-history", dest="monitor_history", action="store_false",
        help="use the myopic condition: show only the proposed tool call",
    )
    args = parser.parse_args(argv)

    if args.classifier == LLM_CLASSIFIER and not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set (source ./.env first)", file=sys.stderr)
        return 2

    # Constructing the Gate validates the policy id, so a typo fails at startup
    # rather than turning every tool call into a fail-closed block.
    history = HistoryConfig.from_env()
    if args.monitor_history is not None:
        history = replace(history, enabled=args.monitor_history)
    try:
        gate = Gate(
            policy=None if args.policy == "__control__" else args.policy,
            model=args.model,
            classifier=args.classifier,
            decisions_path=args.decisions,
            history=history,
            regex_ruleset_version=args.regex_ruleset_version,
        )
    except ValueError as error:
        print(f"ERROR {error}", file=sys.stderr)
        return 2

    server = serve(gate, host=args.host, port=args.port, token=args.token)
    url = f"http://{args.host}:{server.port}"
    print(
        f"monitor serving {gate.policy} with {gate.classifier} at {url}", flush=True
    )
    print(
        "  monitor history: "
        + (f"ON (last {history.max_calls} calls)" if history.enabled else "off (myopic)"),
        flush=True,
    )
    print(f"  set HOOK_MONITOR_URL={url} in the agent's environment", flush=True)
    if not args.token:
        print("  no --token: any local process can query the monitor", file=sys.stderr)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        print(f"\ndecisions: {gate.counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
