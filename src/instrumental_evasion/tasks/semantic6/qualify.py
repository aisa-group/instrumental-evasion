"""Offline author qualification: no model, monitor, container, or network calls."""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
from pathlib import Path

from . import reference, specs


class LocalSandbox:
    """A fresh temporary workdir for trusted fixture/reference code only."""

    def __init__(self, root: Path):
        self.root = root

    def host_path(self, path: str):
        return self.root / Path(path).relative_to(specs.WORKDIR)

    async def write_file(self, path: str, value: str | bytes):
        target = self.host_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(value.encode() if isinstance(value, str) else value)


async def qualify(seeds):
    old_seed = os.environ.get("SEMANTIC6_FIXTURE_SEED")
    try:
        for seed in seeds:
            os.environ["SEMANTIC6_FIXTURE_SEED"] = seed
            for task in specs.PROMPTS:
                with tempfile.TemporaryDirectory(prefix="semantic6-qualification-") as directory:
                    sb = LocalSandbox(Path(directory))
                    await specs.setup(sb, task)
                    if (await specs.score(sb, task))[0]:
                        raise AssertionError(f"untouched fixture passes: {task}")
                    reference.solve(task, sb.root)
                    before = {p.relative_to(sb.root): p.read_bytes() for p in sb.root.rglob("*") if p.is_file()}
                    completed, _, _ = await specs.score(sb, task)
                    after = {p.relative_to(sb.root): p.read_bytes() for p in sb.root.rglob("*") if p.is_file()}
                    if not completed or before != after:
                        raise AssertionError(f"reference failed or grading changed task files: {task}")
                    print(f"PASS {task} seed={seed}")
    finally:
        if old_seed is None:
            os.environ.pop("SEMANTIC6_FIXTURE_SEED", None)
        else:
            os.environ["SEMANTIC6_FIXTURE_SEED"] = old_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", default=["qualification-0", "qualification-1", "qualification-2"])
    args = parser.parse_args()
    asyncio.run(qualify(args.seeds))


if __name__ == "__main__":
    main()
