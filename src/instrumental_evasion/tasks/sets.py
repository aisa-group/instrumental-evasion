"""Saved evaluation rosters and their admission evidence."""

from __future__ import annotations

import json
from pathlib import Path

TASK_SET_IDS = ("task_set_1",)
DATA_DIR = Path(__file__).with_name("task_sets")


def load_task_set(name: str) -> dict:
    """Read a named roster without loading images or experimental artifacts."""
    identifier = name.lower().replace(" ", "_").replace("-", "_")
    if identifier not in TASK_SET_IDS:
        raise ValueError(f"unknown task set {name!r}; choose from {TASK_SET_IDS}")
    manifest = json.loads((DATA_DIR / f"{identifier}.json").read_text())
    tasks = manifest["task_ids"]
    if manifest["id"] != identifier or not tasks or len(tasks) != len(set(tasks)):
        raise ValueError(f"invalid roster for {identifier}")
    return manifest
