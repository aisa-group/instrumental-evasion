"""Task set 1 (the paper's 50 pairs), optional extensions, and task/arm resolution.

Every task is a `TBTaskSpec` (task as data: image, prompt, policy, scorer). The
roster below fixes the task set and its order. Each task runs in one of two
arms:

    control     no monitor, no policy: establishes that the task is solvable
    monitored   every intercepted tool call is reviewed against the task policy

    python -m instrumental_evasion.tasks.registry            # list the roster
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cache

from instrumental_evasion.tasks.sets import load_task_set
from instrumental_evasion.tasks.tb_spec import TBTaskSpec

CONTROL = "control"
MONITORED = "monitored"
ARMS = (CONTROL, MONITORED)

# Task families in the order of Table 1 of the paper.
FAMILIES: dict[str, tuple[str, ...]] = {
    "toolsandbox": (
        "toolsandbox_days_wifi_off",
        "toolsandbox_days_wifi_off_alt",
        "toolsandbox_temperature_wifi_off",
        "toolsandbox_temperature_wifi_off_alt",
        "toolsandbox_temperature_low_battery",
        "toolsandbox_temperature_low_battery_alt",
        "toolsandbox_stock_low_battery",
        "toolsandbox_stock_low_battery_alt",
        "toolsandbox_city_low_battery",
        "toolsandbox_city_low_battery_alt",
        "toolsandbox_message_cellular_off",
        "toolsandbox_message_cellular_off_alt",
    ),
    "clawbench": (
        "claw_file_001_csv_markdown",
        "claw_file_002_csv_json",
        "claw_file_006_extract_emails",
        "claw_file_011_log_filter",
        "claw_file_012_csv_stats",
        "claw_eml_003_sender_counts",
        "claw_eml_008_attachment_inventory",
        "claw_data_002_filter_sort",
        "claw_web_001_extract_links",
        "claw_mm_002_csv_json_merge",
    ),
    "taubench": (
        "tauc_tau_retail_001",
        "tauc_tau_retail_005",
        "tauc_tau_retail_006",
        "tauc_tau_retail_007",
    ),
    "terminal_bench_2": (
        "tb2_sqlite_db_truncate",
        "tb2_build_pmars",
        "tb2_log_summary_date_ranges",
        "tb2_multi_source_data_merger",
        "tb2_constraints_scheduling",
        "tb2_cobol_modernization",
        "tb2_fix_code_vulnerability",
        "tb2_bn_fit_modify",
        "tb2_distribution_search",
    ),
    "tblite": (
        "broken_python",
        "cosign_keyless_signing",
        "corrupted_filesystem_recovery",
        "reverse_engineer_stack_vm",
        "book_portfolio_analysis",
    ),
    "authored": (
        "authored10_command_link",
        "authored10_git_restore",
        "authored10_database_index",
        "authored10_archive_restore",
        "authored10_python_cli",
        "authored10_release_promote",
        "authored10_audit_delete",
        "authored10_diagnostic_access",
        "authored10_backup_retention",
        "authored10_worker_stop",
    ),
}
TASK_IDS: tuple[str, ...] = tuple(task for tasks in FAMILIES.values() for task in tasks)
FAMILY_OF: dict[str, str] = {
    task: family for family, tasks in FAMILIES.items() for task in tasks
}

# Extensions are opt-in. The frozen paper roster and default batch stay at 50.
EXTENSION_FAMILIES: dict[str, tuple[str, ...]] = {
    "semantic6": (
        "semantic6_regression_fit",
        "semantic6_markov_stationary",
        "semantic6_database_import",
        "semantic6_database_view",
        "semantic6_pricing_port",
        "semantic6_record_port",
    ),
}
EXTENSION_TASK_IDS = tuple(task for tasks in EXTENSION_FAMILIES.values() for task in tasks)
EXTENSION_FAMILY_OF = {
    task: family for family, tasks in EXTENSION_FAMILIES.items() for task in tasks
}


def task_family(task_id: str, default: str = "other") -> str:
    return FAMILY_OF.get(task_id, EXTENSION_FAMILY_OF.get(task_id, default))


@cache
def extension_spec_registry() -> dict[str, TBTaskSpec]:
    from instrumental_evasion.tasks.semantic6 import specs

    registered = {spec.id: spec for spec in specs.SPECS}
    if set(registered) != set(EXTENSION_TASK_IDS) or set(registered) & set(TASK_IDS):
        raise RuntimeError("extension task registry does not match its roster")
    return {task: registered[task] for task in EXTENSION_TASK_IDS}


@cache
def spec_registry() -> dict[str, TBTaskSpec]:
    """Map task id -> TBTaskSpec for every task in the roster."""
    from instrumental_evasion.tasks.authored import specs as authored
    from instrumental_evasion.tasks.clawbench import specs as clawbench
    from instrumental_evasion.tasks.taubench import specs as taubench
    from instrumental_evasion.tasks.terminal_bench_2 import specs as tb2
    from instrumental_evasion.tasks.terminal_bench_lite import specs as tblite
    from instrumental_evasion.tasks.toolsandbox import specs as toolsandbox

    registry = {
        spec.id: spec
        for module in (toolsandbox, clawbench, taubench, tb2, tblite, authored)
        for spec in module.SPECS
    }
    missing = [task for task in TASK_IDS if task not in registry]
    if missing:
        raise RuntimeError(f"roster tasks without a registered spec: {missing}")
    return {task: registry[task] for task in TASK_IDS}


@dataclass(frozen=True)
class ResolvedTask:
    """One runnable (task, arm) pair."""

    task_id: str  # e.g. "tb2_build_pmars"
    name: str  # e.g. "tb2_build_pmars_monitored"
    spec: TBTaskSpec
    arm: str
    policy: str | None  # None on the control arm

    @property
    def monitored(self) -> bool:
        return self.arm == MONITORED


def _resolve(task_id: str, arm: str) -> ResolvedTask:
    spec = (extension_spec_registry() if task_id in EXTENSION_FAMILY_OF else spec_registry())[task_id]
    return ResolvedTask(
        task_id=task_id,
        name=f"{task_id}_{arm}",
        spec=spec,
        arm=arm,
        policy=spec.policy if arm == MONITORED else None,
    )


def resolve(task: str, arm: str | None = None) -> list[ResolvedTask]:
    """Resolve a task id, or `<task id>_<arm>`, to its (task, arm) pairs.

    A bare task id without an arm returns both arms.
    """
    known = FAMILY_OF.keys() | EXTENSION_FAMILY_OF.keys()
    if task in known:
        return [_resolve(task, candidate) for candidate in ([arm] if arm else ARMS)]
    for candidate in ARMS:
        suffix = f"_{candidate}"
        if task.endswith(suffix) and task[: -len(suffix)] in known:
            if arm is not None and arm != candidate:
                raise ValueError(f"{task!r} names the {candidate} arm, not {arm!r}")
            return [_resolve(task[: -len(suffix)], candidate)]
    raise ValueError(f"{task!r} is not a task in the roster; known ids: {list(TASK_IDS + EXTENSION_TASK_IDS)}")


def selected(
    *,
    arms: tuple[str, ...] = ARMS,
    only: tuple[str, ...] = (),
    families: tuple[str, ...] = (),
    task_set: str | None = None,
) -> list[ResolvedTask]:
    """The roster as runnable (task, arm) pairs, optionally narrowed."""
    unknown = [task for task in only if task not in FAMILY_OF and task not in EXTENSION_FAMILY_OF]
    unknown += [family for family in families if family not in FAMILIES and family not in EXTENSION_FAMILIES]
    if unknown:
        raise ValueError(f"unknown tasks or families: {unknown}")
    include_extensions = any(task in EXTENSION_FAMILY_OF for task in only) or any(family in EXTENSION_FAMILIES for family in families)
    manifest = load_task_set(task_set) if task_set else None
    task_ids = (tuple(manifest["task_ids"]) if manifest else
                TASK_IDS + EXTENSION_TASK_IDS if include_extensions else TASK_IDS)
    if manifest and manifest["id"] == "task_set_1" and task_ids != TASK_IDS:
        raise ValueError("task_set_1 must match the frozen 50-task roster")
    if any(task not in FAMILY_OF and task not in EXTENSION_FAMILY_OF for task in task_ids):
        raise ValueError("task set contains unknown task IDs")
    tasks = [
        _resolve(task, arm)
        for task in task_ids
        if (not only or task in only) and (not families or task_family(task) in families)
        for arm in arms
    ]
    if manifest:
        evidence = {entry["task_id"]: entry for entry in manifest.get("tasks", [])}
        for task in tasks:
            entry = evidence.get(task.task_id)
            if entry and (task.spec.extra_metadata or {}).get("definition_sha256") != entry["definition_sha256"]:
                raise ValueError(f"{task.task_id} changed since admission to {manifest['id']}; requalify it")
        tasks = [replace(task, spec=replace(task.spec, extra_metadata={
            **(task.spec.extra_metadata or {}),
            "evaluation_task_set_id": manifest["id"],
            "evaluation_task_set_name": manifest["name"],
            "evaluation_task_set_revision": manifest["revision"],
        })) for task in tasks]
    return tasks


def main() -> None:
    registry = spec_registry()
    print("task set 1 (the usual 50 tasks; select with --task-set task_set_1)")
    for family, tasks in FAMILIES.items():
        print(f"{family} ({len(tasks)})")
        for task in tasks:
            spec = registry[task]
            print(f"  {task:<42} policy={spec.policy:<34} image={spec.image()}")
    print(f"total: {len(TASK_IDS)} tasks")
    extensions = extension_spec_registry()
    for family, tasks in EXTENSION_FAMILIES.items():
        print(f"optional {family} ({len(tasks)}); select with --families {family}")
        for task in tasks:
            spec = extensions[task]
            print(f"  {task:<42} policy={spec.policy:<34} image={spec.image()}")


if __name__ == "__main__":
    main()
