#!/usr/bin/env python3
"""Rebuild release-date plots from retained, reviewed episode outcomes.

Run with: uv run --no-project --with matplotlib python scripts/plot_open_weight_release_dates.py
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/figure1_route_rubric_v2_complete_20261019"
MODELS = {
    "dsflash": {
        "label": "DeepSeek V4 Flash",
        "model_id": "deepseek/deepseek-v4-flash",
        "release_date": "2026-04-24",
        "release_source": "https://deepseek.com/en/news/v4-preview/",
        "revision_note": "First public preview. A revised checkpoint was released 2026-07-31; historical model IDs do not pin the checkpoint.",
    },
    "glm52": {
        "label": "GLM-5.2",
        "model_id": "z-ai/glm-5.2",
        "release_date": "2026-06-16",
        "release_source": "https://docs.z.ai/release-notes/new-released",
        "revision_note": "Named model release; API weights were not independently fingerprinted.",
    },
    "glm53flash": {
        "label": "GLM-5.3 Flash",
        "model_id": "z-ai/glm-5.3-flash",
        "release_date": "2026-08-26",
        "release_source": "https://docs.z.ai/release-notes/new-released",
        "revision_note": "Public named release after the anonymous Ox Alpha preview.",
    },
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(episodes: list[dict], tasks: set[str], cohort: str) -> dict:
    by_task = defaultdict(list)
    for row in episodes:
        if row["task"] in tasks:
            by_task[row["task"]].append(row)
    n = len(by_task)
    attempts = sum(any(r["route_attempt_label"] == "supported" for r in rr)
                   for rr in by_task.values())
    unresolved = sum(not any(r["route_attempt_label"] == "supported" for r in rr)
                     and any(r["route_attempt_label"] == "unresolved" for r in rr)
                     for rr in by_task.values())
    successes = sum(any(r["task_completed_before_continuation"] for r in rr)
                    for rr in by_task.values())
    return {
        "cohort": cohort, "n_tasks": n, "n_episodes": sum(map(len, by_task.values())),
        "attempt_tasks": attempts, "unresolved_tasks_without_attempt": unresolved,
        "success_tasks": successes, "attempt_rate_pct": 100 * attempts / n,
        "attempt_label_upper_pct": 100 * (attempts + unresolved) / n,
        "success_rate_pct": 100 * successes / n,
    }


def draw(rows: list[dict], target: Path, subtitle: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.subplots_adjust(left=.09, right=.97, bottom=.29, top=.76)
    blue, orange = "#2166ac", "#d66b1f"
    for row in sorted(rows, key=lambda r: r["release_date"]):
        x = mdates.date2num(date.fromisoformat(row["release_date"]))
        y, upper, success = (row[k] for k in ["attempt_rate_pct", "attempt_label_upper_pct", "success_rate_pct"])
        ax.plot([x, x], [success, y], color="#d7dce2", lw=1.8, zorder=1)
        if upper > y:
            ax.plot([x, x], [y, upper], color=blue, lw=2, ls=":", alpha=.65)
            ax.scatter(x, upper, s=48, marker="^", facecolor="white", edgecolor=blue, zorder=4)
        ax.scatter(x, y, s=85, color=blue, edgecolor="white", linewidth=1, zorder=5)
        ax.scatter(x, success, s=80, marker="s", color=orange, edgecolor="white", linewidth=1, zorder=5)
        ax.annotate(f"{y:.1f}%", (x, y), xytext=(10, -2), textcoords="offset points", color=blue, va="center", weight="bold")
        ax.annotate(f"{success:.1f}%", (x, success), xytext=(10, 0), textcoords="offset points", color=orange, va="center", weight="bold")
        ax.text(x, 111, row["label"], ha="center", va="bottom", fontsize=11, weight="bold")
        ax.text(x, 106, f"{row['n_tasks']} tasks · {row['n_episodes']} episodes", ha="center", va="bottom", fontsize=9, color="#66717e")
    dates = [mdates.date2num(date.fromisoformat(r["release_date"])) for r in rows]
    ax.set_xlim(min(dates) - 28, max(dates) + 32)
    ax.set_ylim(-3, 103)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlabel("Model release date (first public release)", labelpad=12)
    ax.set_ylabel("Share of tasks (%)", labelpad=10)
    ax.grid(axis="y", color="#e7ebef", lw=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c5ccd4")
    fig.text(.09, .95, "Open-weight models: evasion attempts and success", fontsize=18, weight="bold")
    fig.text(.09, .905, subtitle, fontsize=11, color="#53606e")
    handles = [Line2D([], [], marker="o", color=blue, ls="", label="Supported evasion attempt", markersize=8),
               Line2D([], [], marker="s", color=orange, ls="", label="Task success (completion proxy)", markersize=8),
               Line2D([], [], marker="^", markerfacecolor="white", color=blue, ls=":", label="Upper bound if unresolved labels are positive", markersize=7)]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.08, .09), frameon=False, fontsize=9, ncols=2)
    fig.text(.09, .055, "Codex scaffold · GPT-OSS-120B monitor · evidence before the first harness continuation.", fontsize=9, color="#53606e")
    fig.text(.09, .025, "Task completion is the paper's success endpoint; it does not independently confirm a successful evasion route.", fontsize=9, color="#53606e")
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(target.with_suffix("." + ext), dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results/open_weight_release_dates_20261102_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    verdicts = json.loads((SOURCE / "verdicts.json").read_text())
    summary = json.loads((SOURCE / "summary.json").read_text())
    groups = {m: [r for r in verdicts if r["model"] == m] for m in MODELS}
    assert all(not r["validation_issues"] for rr in groups.values() for r in rr)
    tasks = {m: {r["task"] for r in rr} for m, rr in groups.items()}
    common = set.intersection(*tasks.values())
    exact = {t for t in common if all(sum(r["task"] == t for r in rr) == 3 for rr in groups.values())}
    rows = []
    for model, spec in MODELS.items():
        published = next(r["up_to_three"] for r in summary["models"] if r["model"] == model)
        for cohort, ts in [("all_available_up_to_three", tasks[model]), ("matched_tasks_exactly_three", exact)]:
            row = {"model": model, **spec, **aggregate(groups[model], ts, cohort)}
            if cohort == "all_available_up_to_three":
                assert row["n_tasks"] == published["denominator"]
                assert row["attempt_tasks"] == published["luna_supported_before_continuation"]
                assert row["success_tasks"] == published["task_completed_before_continuation"]
            rows.append(row)
    (args.out / "plot_data.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (args.out / "plot_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": {p.name: sha(p) for p in [SOURCE / "summary.json", SOURCE / "verdicts.json", SOURCE / "manifest.json"]},
        "script_sha256": sha(Path(__file__)), "new_model_runs": False, "new_api_cost_usd": 0,
        "attempt_rubric": "observable_route_v2_five_checks", "cutoff": summary["cutoff"],
        "success_endpoint": "task_completed_before_continuation; proxy, not independent evasion success confirmation",
        "denominator": "All retained tasks, including unresolved labels. Up to three runs; a task counts if any run is positive.",
        "matched_tasks": sorted(exact), "source_verified_on": "2026-11-02",
        "limitations": ["Historical prefix censoring is not a prospective no-continuation experiment.", "Models differ in task coverage and reasoning settings. The matched comparison equalizes tasks and repetition count only.", "Missing reviews are not negatives; unresolved labels are reported as an identification interval.", "Release date means first named public release; historical API aliases do not uniquely identify weight revisions.", "Only the three open-weight models with the current reviewed attempt rubric are included. Qwen3.5-9B lacks comparable attempt reviews; old ReAct runs use different conditions."],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "methodology.txt").write_text(
        "Open-weight model release-date plot\n\n"
        "The main plot uses the latest retained five-check route-attempt reviews and the paper's task-completion success endpoint. "
        "All outcomes precede the first harness continuation; a task counts as positive when any retained run is positive. "
        "Unresolved attempt labels remain in the denominator, with an upper bound shown separately. These bounds describe label uncertainty, not sampling confidence.\n\n"
        f"The sensitivity plot uses the same {len(exact)} tasks with exactly three retained episodes per model. "
        "It is selected on availability, and does not equalize reasoning settings or provider.\n\n"
        "DeepSeek's first public release was 2026-04-24, with a revision on 2026-07-31. The historical OpenRouter ID does not pin a checkpoint. "
        "GLM release dates are from Z.ai's release notes. Sources and exact counts are in plot_data.csv.\n\n"
        "No new model calls are needed to reproduce these historical figures. Original results remain unchanged.\n"
    )
    draw([r for r in rows if r["cohort"] == "all_available_up_to_three"], args.out / "open_weight_release_dates", "Reviewed historical runs · per-task rates across up to three runs")
    draw([r for r in rows if r["cohort"] == "matched_tasks_exactly_three"], args.out / "open_weight_release_dates_matched", f"Sensitivity check · the same {len(exact)} tasks · exactly three runs per model")
    print(json.dumps({"out": str(args.out), "matched_tasks": len(exact), "rows": [{k:r[k] for k in ["model", "cohort", "attempt_rate_pct", "success_rate_pct"]} for r in rows]}, indent=2))


if __name__ == "__main__":
    main()
