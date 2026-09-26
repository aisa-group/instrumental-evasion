"""Command-line interface for the fixed-request monitor diagnostic.

    python -m instrumental_evasion.diagnostics.monitor_diagnostic requests [--monitor M]
    python -m instrumental_evasion.diagnostics.monitor_diagnostic run --monitor M --out DIR
    python -m instrumental_evasion.diagnostics.monitor_diagnostic report DIR [DIR ...]
    python -m instrumental_evasion.diagnostics.monitor_diagnostic qualify --image IMAGE

`requests` prints the exact request bodies without contacting any service.
`run` makes 40 x repeats paid monitor calls through OpenRouter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from instrumental_evasion.diagnostics.monitor_diagnostic.cases import requests
from instrumental_evasion.diagnostics.monitor_diagnostic.classify import (
    payload_sha256,
    request_body,
    run,
)
from instrumental_evasion.diagnostics.monitor_diagnostic.qualify import ENGINES, qualify
from instrumental_evasion.diagnostics.monitor_diagnostic.report import (
    ALL,
    RELEVANT,
    SUBSETS,
    markdown,
    report,
)
from instrumental_evasion.policies import DEFAULT_MONITOR_MODEL


def _requests(args: argparse.Namespace) -> None:
    for request in requests(relevant_context_only=args.relevant_context):
        body = request_body(request, args.monitor)
        row = {"id": request.id, **request.labels(), "payload_sha256": payload_sha256(body)}
        print(json.dumps({**row, "request": body}))


def _run(args: argparse.Namespace) -> None:
    run(args.out, args.monitor, repeats=args.repeats, workers=args.workers)
    print(markdown(report([args.out], RELEVANT)), end="")


def _report(args: argparse.Namespace) -> None:
    summaries = report(args.directories, args.subset)
    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        print(markdown(summaries), end="")


def _qualify(args: argparse.Namespace) -> None:
    checks = qualify(args.image, engine=args.engine)
    print(json.dumps({"checks": len(checks), "all_effects_observed": True}))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m instrumental_evasion.diagnostics.monitor_diagnostic",
        description="Fixed-request diagnostic of monitor decision quality.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    show = commands.add_parser("requests", help="print the request bodies as JSON lines")
    show.add_argument("--monitor", default=DEFAULT_MONITOR_MODEL)
    show.add_argument("--relevant-context", action="store_true",
                      help="only the 24 profiles that supply the decisive context")
    show.set_defaults(handler=_requests)

    classify = commands.add_parser("run", help="classify every profile through the monitor client")
    classify.add_argument("--monitor", default=DEFAULT_MONITOR_MODEL,
                          help="OpenRouter model, with or without the openrouter/ prefix")
    classify.add_argument("--out", type=Path, required=True, help="new or empty output directory")
    classify.add_argument("--repeats", type=int, default=3)
    classify.add_argument("--workers", type=int, default=1, help="concurrent requests")
    classify.set_defaults(handler=_run)

    summarize = commands.add_parser("report", help="count decisions per monitor")
    summarize.add_argument("directories", type=Path, nargs="+", help="run directories")
    summarize.add_argument("--subset", choices=SUBSETS, default=ALL)
    summarize.add_argument("--json", action="store_true")
    summarize.set_defaults(handler=_report)

    check = commands.add_parser("qualify", help="verify fixture effects in a container")
    check.add_argument("--image", required=True, help="container image with bash, python3, gcc, openssl")
    check.add_argument("--engine", choices=ENGINES, default="apptainer")
    check.set_defaults(handler=_qualify)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
