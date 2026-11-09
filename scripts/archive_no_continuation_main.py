"""Archive the exact prefix cases, reviews, and source hashes used by Figure 1."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    rows = json.loads((args.work / "selection.json").read_text())
    phases = ["superseded_inherited", "inherited", "judgments", "rechecks", "audit_rechecks", "final_rechecks", "evidence_rechecks", "evidence_filtered"]
    artifact = args.destination / "review_evidence.jsonl.gz"
    with artifact.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
        for row in rows:
            identifier = row["attempt_id"]
            case_raw = (args.work / "cases" / f"{identifier}.json").read_text()
            assert hashlib.sha256(case_raw.encode()).hexdigest() == row["case_sha256"]
            reviews = {}
            for phase in phases:
                path = args.work / phase / f"{identifier}.json"
                if path.exists():
                    reviews[phase] = json.loads(path.read_text())
            assert reviews, identifier
            value = {"selection": row, "case_json": case_raw, "reviews": reviews}
            inherited = reviews.get("inherited", reviews.get("superseded_inherited", {}))
            if inherited.get("source_full_review"):
                source_raw = Path(inherited["source_full_review"]).read_text()
                assert hashlib.sha256(source_raw.encode()).hexdigest() == inherited["source_full_review_sha256"]
                value["source_full_review_json"] = source_raw
            encoded = json.dumps(value, ensure_ascii=False)
            assert not re.search(r"\b(?:sk-proj-|sk-or-v1-)[A-Za-z0-9_-]{12,}", encoded), identifier
            stream.write((encoded + "\n").encode())
    for name in ["selection.json", "summary.json", "summary.csv", "verdicts.json", "rubric.txt", "schema.json"]:
        shutil.copy2(args.work / name, args.destination / name)
    manifest = {
        "selected_attempts": len(rows),
        "review_evidence_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "format": "One authenticated selection, exact case JSON, all local review phases, and inherited source review per gzip JSONL record.",
    }
    (args.destination / "archive.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
