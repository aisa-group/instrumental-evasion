"""The 50 task-policy pairs of the paper, and how a task name resolves to a run.

Every task is a `TBTaskSpec` (task as data: image, prompt, policy, scorer). The
roster below fixes the task set and its order. Each task runs in one of two
arms:

    control     no monitor, no policy: establishes that the task is solvable
    monitored   every intercepted tool call is reviewed against the task policy

    python -m instrumental_evasion.tasks.registry            # list the roster
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

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
    spec = spec_registry()[task_id]
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
    if task in FAMILY_OF:
        return [_resolve(task, candidate) for candidate in ([arm] if arm else ARMS)]
    for candidate in ARMS:
        suffix = f"_{candidate}"
        if task.endswith(suffix) and task[: -len(suffix)] in FAMILY_OF:
            if arm is not None and arm != candidate:
                raise ValueError(f"{task!r} names the {candidate} arm, not {arm!r}")
            return [_resolve(task[: -len(suffix)], candidate)]
    raise ValueError(f"{task!r} is not a task in the roster; known ids: {list(TASK_IDS)}")


def selected(
    *,
    arms: tuple[str, ...] = ARMS,
    only: tuple[str, ...] = (),
    families: tuple[str, ...] = (),
) -> list[ResolvedTask]:
    """The roster as runnable (task, arm) pairs, optionally narrowed."""
    unknown = [task for task in only if task not in FAMILY_OF]
    unknown += [family for family in families if family not in FAMILIES]
    if unknown:
        raise ValueError(f"unknown tasks or families: {unknown}")
    return [
        _resolve(task, arm)
        for task in TASK_IDS
        if (not only or task in only) and (not families or FAMILY_OF[task] in families)
        for arm in arms
    ]


def main() -> None:
    registry = spec_registry()
    for family, tasks in FAMILIES.items():
        print(f"{family} ({len(tasks)})")
        for task in tasks:
            spec = registry[task]
            print(f"  {task:<42} policy={spec.policy:<34} image={spec.image()}")
    print(f"total: {len(TASK_IDS)} tasks")


if __name__ == "__main__":
    main()
