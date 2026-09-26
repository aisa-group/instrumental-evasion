"""Generate the sanctioned ``tau`` CLI staged into a task sandbox."""

from __future__ import annotations

CLI_SRC = r'''#!/usr/bin/env python3
"""tau -- act on the store database."""
import json
import sys
from importlib import import_module

sys.path.insert(0, "/app/.tau")
DATA = "/app/data/db.json"
DOMAIN = "TAU_DOMAIN_SLOT"
_tools_mod = import_module("tau_bench.envs." + DOMAIN + ".tools")
TOOLS = {c.get_info()["function"]["name"]: c for c in _tools_mod.ALL_TOOLS}

def _load():
    with open(DATA) as fh:
        return json.load(fh)

def _save(data):
    with open(DATA, "w") as fh:
        json.dump(data, fh)

def _list():
    for name in sorted(TOOLS):
        info = TOOLS[name].get_info()["function"]
        props = info.get("parameters", {}).get("properties", {})
        required = set(info.get("parameters", {}).get("required", []))
        cols = ", ".join((p + "*" if p in required else p) for p in props) or "(none)"
        print(name)
        print("    " + cols)
    print("\n* = required. Pass args as one JSON object. Writes commit to the database.")

def main(argv):
    if not argv or argv[0] in ("--list", "-l", "--help", "-h"):
        _list()
        return 0
    name = argv[0]
    if name not in TOOLS:
        print("Error: unknown tool " + repr(name) + ". Run ./tau --list.", file=sys.stderr)
        return 2
    kwargs = {}
    if len(argv) > 1 and argv[1].strip():
        try:
            kwargs = json.loads(argv[1])
        except json.JSONDecodeError as exc:
            print("Error: args must be one JSON object: " + str(exc), file=sys.stderr)
            return 2
    data = _load()
    try:
        result = TOOLS[name].invoke(data, **kwargs)
    except TypeError as exc:
        print("Error: bad arguments for " + name + ": " + str(exc), file=sys.stderr)
        return 2
    _save(data)
    print(result)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
'''


def cli_for(domain: str) -> str:
    return CLI_SRC.replace("TAU_DOMAIN_SLOT", domain)
