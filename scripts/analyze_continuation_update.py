#!/usr/bin/env python3
"""Extend the frozen continuation audit with the October 16 GPT campaigns.

Reads saved artifacts only. Both endpoints use the same reviewed-effect set.
"""

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path

from analyze_figure1_no_continuation import analyze_episode, digest, verified_bytes, write_json


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/continuation_update_20261019"


def aggregate(label, model, cohort, denominator, episodes, *, group_tasks=False):
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode["task"] if group_tasks else episode["attempt_id"]].append(episode)
    if len(groups) != denominator:
        raise ValueError(f"Coverage mismatch: {model}/{cohort}")
    counts = {field: sum(any(e[field] for e in group) for group in groups.values())
              for field in ("original_task_completed", "original_success",
                            "task_completed_before_continuation", "success_before_continuation")}
    original = counts["original_success"]
    cutoff = counts["success_before_continuation"]
    if cutoff > original:
        raise ValueError("Cutoff success is not a subset of original success")
    return {"model": model, "label": label, "cohort": cohort, "denominator": denominator,
            **counts, "original_asr_pct": 100 * original / denominator,
            "cutoff_asr_pct": 100 * cutoff / denominator,
            "drop_pp": 100 * (original - cutoff) / denominator,
            "relative_loss_pct": 100 * (original - cutoff) / original if original else None}


def main():
    inputs = json.loads((OUT / "gpt_inputs.json").read_text())
    for item in inputs["provenance"]:
        verified_bytes(ROOT / item["path"], item["sha256"])
    reviews = defaultdict(list)
    for review in inputs["effect_reviews"]:
        reviews[review["attempt_id"]].append(review)
    selections = inputs["selections"]
    if len(selections) != 300 or len({e["attempt_id"] for e in selections}) != 300:
        raise ValueError("Expected 300 unique selected attempts")
    with ThreadPoolExecutor(max_workers=8) as pool:
        episodes = list(pool.map(
            lambda row: analyze_episode(row, reviews[row["attempt_id"]],
                                        {"archive": ROOT, "source": ROOT}), selections))
    for row in episodes:
        source = json.loads((ROOT / row["result_path"]).read_text())
        sample = source["samples"][0]
        if source["task_id"] != row["task"] or source["metadata"]["agent_model"] != row["model"]:
            raise ValueError("Model/task identity mismatch")
        valid = (sample["infrastructure_status"] == "valid"
                 and not sample["monitor_errors"] and not sample["error"])
        if valid != row["native_valid"]:
            raise ValueError("Validity changed")
        row["recorded_native_success"] = bool(sample["solved"])

    # Preserve the original eight cohorts and their already-audited effects.
    historical = json.loads((ROOT / "results/figure1_no_continuation_20261013/episodes.json").read_text())
    prior = json.loads((ROOT / "results/figure1_no_continuation_20261013/summary.json").read_text())
    summaries = []
    for model in prior["models"]:
        rows = [e for e in historical if e["model"] == model["model"]]
        summaries.append(aggregate(model["label"], model["model"], "single_run",
                                   model["single_run_denominator"],
                                   [e for e in rows if "single_run" in e["cohorts"]]))
        summaries.append(aggregate(model["label"], model["model"], "three_runs", 50,
                                   [e for e in rows if "three_runs" in e["cohorts"]], group_tasks=True))

    # Fable's memory-database witness belongs in the uncensored numerator too.
    fable = json.loads((ROOT / "results/fable_no_continuation_20261013/recheck_20261019/episodes.json").read_text())
    latest = [e for e in fable if e.get("latest_category")]
    if len(latest) != 39:
        raise ValueError("Fable selection changed")
    for episode in latest:
        episode["original_success"] |= bool(episode["effect_reviews"])
    summaries.append(aggregate("Fable 5.1", "fable51", "single_run", 39, latest))

    for model, label in (("gpt-6.1-sol", "GPT-6.1 Sol"), ("gpt-6-luna", "GPT-6 Luna")):
        rows = [e for e in episodes if e["model"] == model and e["native_valid"]]
        singles = [e for e in rows if e["epoch"] == 0]
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["task"]].append(row)
        complete = {task for task, attempts in grouped.items()
                    if sorted(e["epoch"] for e in attempts) == [0, 1, 2]}
        summaries.append(aggregate(label, model, "single_run", len(singles), singles))
        summaries.append(aggregate(label, model, "three_runs", len(complete),
                                   [e for e in rows if e["task"] in complete], group_tasks=True))

    write_json(OUT / "gpt_episodes.json", episodes)
    write_json(OUT / "summary.json", {"inputs_sha256": digest((OUT / "gpt_inputs.json").read_bytes()),
                                      "analysis_sha256": digest(Path(__file__).read_bytes()),
                                      "models": summaries, "limitations": inputs["limitations"]})
    with (OUT / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    for row in summaries:
        print(f"{row['label']} {row['cohort']}: "
              f"{row['original_success']}/{row['denominator']} -> "
              f"{row['success_before_continuation']}/{row['denominator']}; "
              f"drop {row['drop_pp']:.2f} pp")


if __name__ == "__main__":
    main()
