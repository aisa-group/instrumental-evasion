"""Saved rosters preserve the paper task set."""

from __future__ import annotations


import pytest

from instrumental_evasion.tasks import registry
from instrumental_evasion.tasks.sets import load_task_set


def test_task_set_1_is_the_existing_default_fifty_in_the_same_order():
    manifest = load_task_set("task set 1")
    assert tuple(manifest["task_ids"]) == registry.TASK_IDS
    assert len(manifest["task_ids"]) == 50
    pairs = registry.selected(task_set="task_set_1")
    assert [(t.task_id, t.arm) for t in pairs] == [
        (t.task_id, t.arm) for t in registry.selected()
    ]
    assert pairs[0].spec.extra_metadata["evaluation_task_set_name"] == "task set 1"


@pytest.mark.parametrize("name", ["task_set_3", "../../.env"])
def test_unknown_task_set_is_rejected(name):
    with pytest.raises(ValueError, match="unknown task set"):
        load_task_set(name)
