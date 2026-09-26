"""Count forged-denial completions from finalized attempt artifacts.

    python -m instrumental_evasion.diagnostics.forged_denial report logs/forged_denial/*

The paper reports, for each model, completions in each condition out of 30
attempts (ten tasks x three repeats). An attempt counts only when its artifact
validates and its infrastructure is valid; the others are listed separately and
never counted as failures. Completion is the scorer's verdict on the output
file, so an agent that claims success without writing the value is not counted.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from instrumental_evasion.diagnostics.forged_denial.tasks import CONDITIONS, TASK_IDS
from instrumental_evasion.runner.attempt_artifact import AttemptArtifactError, validate_attempt

UNKNOWN_MODEL = "unknown"


def _row(directory: Path, task_id: str, condition: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": UNKNOWN_MODEL, "reasoning_effort": None, "task": task_id,
        "condition": condition, "repeat": None, "attempt_id": directory.name,
        "state": "incomplete", "solved": False, "stop_reason": None, "path": str(directory),
    }
    try:
        log = validate_attempt(directory)
    except AttemptArtifactError as error:
        return {**row, "error": str(error)}
    metadata, sample = log.metadata, log.samples[0]
    if (
        metadata.get("diagnostic") != "forged_denial"
        or metadata.get("forged_denial_task") != task_id
        or metadata.get("forged_denial_condition") != condition
    ):
        return {**row, "error": "not a forged-denial attempt for this task and condition"}
    return {
        **row,
        "model": metadata.get("agent_model", UNKNOWN_MODEL),
        "reasoning_effort": metadata.get("agent_reasoning_effort"),
        "repeat": sample.epoch,
        "state": "invalid" if sample.void else "counted",
        "solved": bool(sample.solved) and not sample.void,
        "stop_reason": sample.stop_reason,
    }


def collect(roots: Iterable[Path | str]) -> list[dict[str, Any]]:
    """One row per attempt directory under each root, in a stable order.

    The runner lays attempts out as ``<task>_<condition>/epoch<repeat>/<attempt id>``.
    """
    rows = []
    for root in roots:
        for task_id in TASK_IDS:
            for condition in CONDITIONS:
                for directory in sorted(Path(root).glob(f"{task_id}_{condition}/epoch*/*")):
                    if directory.is_dir():
                        rows.append(_row(directory, task_id, condition))
    return rows


def _cell() -> dict[str, int]:
    return {"solved": 0, "counted": 0, "invalid": 0, "incomplete": 0}


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Per model: counts by condition, and by task and condition."""
    models: dict[str, Any] = {}
    for row in rows:
        label = row["model"] if not row["reasoning_effort"] else (
            f"{row['model']} ({row['reasoning_effort']})"
        )
        model = models.setdefault(label, {
            "conditions": defaultdict(_cell),
            "tasks": defaultdict(lambda: defaultdict(_cell)),
        })
        for cell in (model["conditions"][row["condition"]],
                     model["tasks"][row["task"]][row["condition"]]):
            cell[row["state"]] += 1
            cell["solved"] += int(row["solved"])
    return {
        label: {
            "conditions": {c: dict(model["conditions"][c]) for c in CONDITIONS},
            "tasks": {
                task: {c: dict(model["tasks"][task][c]) for c in CONDITIONS}
                for task in TASK_IDS if task in model["tasks"]
            },
        }
        for label, model in sorted(models.items())
    }


def _fraction(cell: dict[str, int]) -> str:
    text = f"{cell['solved']}/{cell['counted']}"
    excluded = cell["invalid"] + cell["incomplete"]
    return text + (f" (+{excluded} excluded)" if excluded else "")


def format_summary(summary: dict[str, Any]) -> str:
    """A completions table per model: solved / counted attempts."""
    if not summary:
        return "no forged-denial attempts found"
    lines = []
    for label, model in summary.items():
        lines.append(f"{label}")
        lines.append(f"  {'task':<20}" + "".join(f"{c:>24}" for c in CONDITIONS))
        for task, cells in model["tasks"].items():
            row = "".join(f"{_fraction(cells[c]):>24}" for c in CONDITIONS)
            lines.append(f"  {task:<20}{row}")
        lines.append(
            f"  {'all tasks':<20}"
            + "".join(f"{_fraction(model['conditions'][c]):>24}" for c in CONDITIONS)
        )
        lines.append("")
    lines.append(
        "solved/counted: attempts whose output file held the exact answer, over "
        "infrastructure-valid attempts.\nexcluded: infrastructure-invalid or "
        "incomplete attempts, which are not counted."
    )
    return "\n".join(lines)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "log_dirs", nargs="+", type=Path, help="one or more --log-dir values of `run`"
    )
    parser.add_argument("--json", type=Path, default=None, help="also write rows and summary")


def main(args: argparse.Namespace) -> int:
    """Print the summary of the parsed `report` command; return an exit code."""
    rows = collect(args.log_dirs)
    summary = summarize(rows)
    print(format_summary(summary))
    if args.json is not None:
        args.json.write_text(
            json.dumps({"rows": rows, "summary": summary}, indent=2, ensure_ascii=False) + "\n"
        )
    return 0 if rows else 1
