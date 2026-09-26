"""The task roster: 50 task-policy pairs, each runnable in two arms."""

from __future__ import annotations

import pytest

from instrumental_evasion.policies import GUARDRAIL_SPECS, trimmed_policy_id
from instrumental_evasion.tasks import registry
from instrumental_evasion.tasks.registry import (
    ARMS,
    CONTROL,
    FAMILIES,
    FAMILY_OF,
    MONITORED,
    TASK_IDS,
    resolve,
    selected,
    spec_registry,
)

FAMILY_SIZES = {
    "toolsandbox": 12,
    "clawbench": 10,
    "taubench": 4,
    "terminal_bench_2": 9,
    "tblite": 5,
    "authored": 10,
}


def test_the_roster_has_fifty_tasks_in_six_families():
    assert {family: len(tasks) for family, tasks in FAMILIES.items()} == FAMILY_SIZES
    assert list(FAMILIES) == list(FAMILY_SIZES)
    assert len(TASK_IDS) == 50


def test_task_ids_are_unique_and_ordered_by_family():
    assert len(set(TASK_IDS)) == len(TASK_IDS)
    assert list(TASK_IDS) == [task for tasks in FAMILIES.values() for task in tasks]
    assert FAMILY_OF == {task: family for family, tasks in FAMILIES.items() for task in tasks}


def test_every_task_has_a_registered_spec():
    specs = spec_registry()
    assert list(specs) == list(TASK_IDS)
    for task, spec in specs.items():
        assert spec.id == task
        assert spec.prompt.strip(), task
        assert callable(spec.score_fn), task


def test_every_task_policy_is_registered_with_a_trimmed_variant():
    for task, spec in spec_registry().items():
        assert spec.policy in GUARDRAIL_SPECS, (task, spec.policy)
        assert f"{spec.policy}_trim" in GUARDRAIL_SPECS, (task, spec.policy)
        assert trimmed_policy_id(spec.policy) == f"{spec.policy}_trim"


def test_the_taubench_family_is_the_four_retail_container_tasks():
    assert FAMILIES["taubench"] == (
        "tauc_tau_retail_001",
        "tauc_tau_retail_005",
        "tauc_tau_retail_006",
        "tauc_tau_retail_007",
    )
    specs = spec_registry()
    assert {specs[task].policy for task in FAMILIES["taubench"]} == {"no_return_or_exchange"}


def test_a_bare_task_id_resolves_to_both_arms():
    tasks = resolve("tb2_build_pmars")
    assert [task.arm for task in tasks] == list(ARMS) == [CONTROL, MONITORED]
    assert [task.name for task in tasks] == [
        "tb2_build_pmars_control",
        "tb2_build_pmars_monitored",
    ]
    assert all(task.task_id == "tb2_build_pmars" for task in tasks)
    assert all(task.spec is spec_registry()["tb2_build_pmars"] for task in tasks)


def test_the_monitored_arm_carries_the_policy_and_the_control_arm_none():
    control, monitored = resolve("tb2_build_pmars")
    assert control.policy is None and not control.monitored
    assert monitored.policy == "no_compilation" and monitored.monitored


def test_an_explicit_arm_selects_one_pair():
    (task,) = resolve("claw_file_001_csv_markdown", MONITORED)
    assert task.name == "claw_file_001_csv_markdown_monitored"
    assert task.policy == "no_input_read"


@pytest.mark.parametrize("arm", ARMS)
def test_both_spellings_resolve_to_the_same_pair(arm):
    (by_suffix,) = resolve(f"tb2_build_pmars_{arm}")
    (by_argument,) = resolve("tb2_build_pmars", arm)
    assert by_suffix == by_argument
    assert resolve(f"tb2_build_pmars_{arm}", arm) == [by_suffix]


def test_a_suffix_that_contradicts_the_arm_is_rejected():
    with pytest.raises(ValueError, match="names the monitored arm"):
        resolve("tb2_build_pmars_monitored", CONTROL)


@pytest.mark.parametrize("name", ["not_a_task", "tb2_not_a_task_monitored", "tb2_build_pmars_hooked", ""])
def test_an_unknown_task_is_rejected(name):
    with pytest.raises(ValueError, match="not a task in the roster"):
        resolve(name)


def test_selected_defaults_to_every_task_in_both_arms():
    tasks = selected()
    assert len(tasks) == 100
    assert len({task.name for task in tasks}) == 100
    assert [task.task_id for task in tasks[::2]] == list(TASK_IDS)
    assert [task.arm for task in tasks[:2]] == [CONTROL, MONITORED]


def test_selected_narrows_by_arm_task_and_family():
    monitored = selected(arms=(MONITORED,))
    assert len(monitored) == 50
    assert all(task.monitored and task.policy for task in monitored)
    assert all(task.policy is None for task in selected(arms=(CONTROL,)))

    only = selected(only=("broken_python", "tb2_build_pmars"))
    # The roster order wins over the order of the request.
    assert [task.name for task in only] == [
        "tb2_build_pmars_control",
        "tb2_build_pmars_monitored",
        "broken_python_control",
        "broken_python_monitored",
    ]

    tblite = selected(families=("tblite",), arms=(MONITORED,))
    assert [task.task_id for task in tblite] == list(FAMILIES["tblite"])

    both = selected(only=("broken_python", "tb2_build_pmars"), families=("tblite",))
    assert {task.task_id for task in both} == {"broken_python"}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"only": ("not_a_task",)},
        {"only": ("tb2_build_pmars_monitored",)},
        {"families": ("tb2",)},
    ],
)
def test_selected_rejects_unknown_tasks_and_families(kwargs):
    with pytest.raises(ValueError, match="unknown tasks or families"):
        selected(**kwargs)


def test_the_roster_listing_names_every_task(capsys):
    registry.main()
    out = capsys.readouterr().out
    assert "total: 50 tasks" in out
    assert all(task in out for task in TASK_IDS)
