"""Drop invalid auxiliary citations from negative reviews, preserving their labels."""

import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("NO_CONTINUATION_WORK_DIR", str(ROOT / "results/no_continuation_main_figure_20261019")))
spec = importlib.util.spec_from_file_location("judge", ROOT.parent / "agent_breakout/src/agent_breakout/analysis/luna_marker_agreement.py")
judge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judge)


def main():
    destination = OUT / "evidence_filtered"
    destination.mkdir(exist_ok=True)
    changed = 0
    for row in json.loads((OUT / "selection.json").read_text()):
        identifier = row["attempt_id"]
        for phase in ["evidence_rechecks", "final_rechecks", "audit_rechecks", "rechecks", "judgments"]:
            source = OUT / phase / f"{identifier}.json"
            if source.exists():
                review = json.loads(source.read_text())
                break
        else:
            continue
        if review["evidence_valid"]:
            continue
        verdict = review["judgment"]
        assert verdict["route_attempt"] == "no_supported", identifier
        case = json.loads((OUT / "cases" / source.name).read_text())
        events = {e["event_id"]: e["text"] for e in case["events"]}
        valid, invalid = [], []
        for evidence in verdict["evidence"]:
            target = valid if evidence["event_id"] in events and judge.quote_matches(events[evidence["event_id"]], evidence["quote"]) else invalid
            target.append(evidence)
        repaired = {**verdict, "evidence": valid}
        assert valid and invalid and judge.valid_evidence(case, repaired), identifier
        result = {
            **review,
            "judgment": repaired,
            "phase": "evidence_filtered",
            "evidence_valid": True,
            "usage": {"cost": 0},
            "source_review": str(source),
            "source_usage": review.get("usage"),
            "evidence_filter": {
                "rule": "Only remove nonmatching auxiliary citations from a negative verdict that retains exact-quote evidence. Preserve the label, category, rationale, and all valid citations. Positive and unresolved verdicts cannot use this filter.",
                "removed_citations": invalid,
            },
        }
        (destination / source.name).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        changed += 1
    print("Negative reviews with invalid auxiliary citations removed:", changed)


if __name__ == "__main__":
    main()
