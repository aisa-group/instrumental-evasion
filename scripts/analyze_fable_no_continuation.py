#!/usr/bin/env python3
"""Audit Fable continuation cutoffs without pooling campaigns into a task rate."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path

from analyze_figure1_no_continuation import analyze_episode, digest, verified_bytes, write_json


def tally(label: str, episodes: list[dict]) -> dict:
    def retained_effect(episode: dict, *, include_memory: bool) -> bool:
        return any(review["retained"] and (include_memory or not review.get("in_memory_database"))
                   for review in episode["effect_reviews"])

    return {
        "cohort": label, "episodes": len(episodes),
        "distinct_tasks": len({e["task"] for e in episodes}),
        "recorded_task_completed": sum(e["original_task_completed"] for e in episodes),
        "recorded_task_or_effect_success": sum(e["original_success"] for e in episodes),
        "full_before_continuation": sum(e["task_completed_before_continuation"] for e in episodes),
        "partial_before_continuation_excluding_memory_database": sum(
            e["task_completed_before_continuation"] or retained_effect(e, include_memory=False)
            for e in episodes),
        "partial_before_continuation_including_memory_database": sum(
            e["success_before_continuation"] for e in episodes),
        "early_success_task_ids": sorted({e["task"] for e in episodes if e["success_before_continuation"]}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    # analyze_episode's archive key is the artifact root, which here holds
    # original native logs. Publication provenance is checked separately.
    roots = {"archive": args.source, "source": args.source, "publication": args.publication}
    for item in inputs["provenance"]:
        verified_bytes(roots[item["root"]] / item["path"], item["sha256"])
    reviews = {r["attempt_id"]: [r] for r in inputs["effect_reviews"]}
    selections = inputs["selections"]
    if len(selections) != len({r["attempt_id"] for r in selections}):
        raise ValueError("Duplicate selection")
    if len(selections) != 161:
        raise ValueError("Fable archive coverage changed")
    with ThreadPoolExecutor(max_workers=12) as pool:
        episodes = list(pool.map(lambda r: analyze_episode(r, reviews.get(r["attempt_id"], []), roots), selections))
    valid = [e for e in episodes if e["native_valid"]]
    if len(valid) != 90 or sum(e["original_success"] for e in valid) != 9:
        raise ValueError("Native Fable baseline changed")
    latest = [e for e in episodes if e.get("latest_category")]
    categories = Counter(e["latest_category"] for e in latest)
    expected = {"native_valid": 32, "cyber_blocked": 4, "manual_stop_censored": 3}
    if dict(categories) != expected or len({e["task"] for e in latest}) != 39:
        raise ValueError("Latest adjudicated selection changed")
    if sum(e["original_success"] for e in latest) != 3:
        raise ValueError("Latest adjudicated baseline changed")
    summaries = [tally("latest_adjudicated_39", latest), tally("all_native_valid_attempts", valid)]
    campaigns = []
    for campaign in sorted({e["campaign"] for e in episodes}):
        all_rows = [e for e in episodes if e["campaign"] == campaign]
        included = [e for e in all_rows if e["native_valid"]]
        campaigns.append({**tally(campaign, included), "all_artifacts": len(all_rows),
                          "native_invalid": len(all_rows) - len(included)})
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "episodes.json", episodes)
    write_json(args.out / "summary.json", {
        "version": "fable_first_continuation_cutoff_20261013_v1",
        "inputs_sha256": digest(args.inputs.read_bytes()),
        "analysis_sha256": digest(Path(__file__).read_bytes()),
        "cutoff_analysis_sha256": digest(Path(__file__).with_name("analyze_figure1_no_continuation.py").read_bytes()),
        "cohorts": summaries, "campaigns": campaigns,
        "latest_categories": dict(categories), "limitations": inputs["limitations"],
    })
    fields = [k for k in summaries[0] if k != "early_success_task_ids"]
    with (args.out / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in summaries)
    with (args.out / "campaigns.csv").open("w", newline="") as stream:
        campaign_fields = [*fields, "all_artifacts", "native_invalid"]
        writer = csv.DictWriter(stream, campaign_fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in campaign_fields} for r in campaigns)
    for r in summaries:
        print(f"{r['cohort']}: {r['recorded_task_completed']}/{r['episodes']} recorded completions -> "
              f"{r['full_before_continuation']}/{r['episodes']} before continuation; "
              f"partial {r['partial_before_continuation_including_memory_database']}/{r['episodes']} "
              f"({r['partial_before_continuation_excluding_memory_database']} excluding memory databases)")


if __name__ == "__main__":
    main()
