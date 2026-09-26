"""Command line: `python -m instrumental_evasion.diagnostics.forged_denial {run,report}`."""

from __future__ import annotations

import argparse

from instrumental_evasion.diagnostics.forged_denial import report, run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m instrumental_evasion.diagnostics.forged_denial",
        description="Forged-denial diagnostic: run the paired tasks, or count completions.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run.add_arguments(commands.add_parser(
        "run", help="run clean/injected x repeats for one model",
        description=run.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    ))
    report.add_arguments(commands.add_parser(
        "report", help="count completions per model and condition",
        description=report.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    ))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run.main(args) if args.command == "run" else report.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
