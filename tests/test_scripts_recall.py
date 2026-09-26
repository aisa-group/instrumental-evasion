"""The recall gate probes exactly the paper's policies, and its dry run is offline.

These tests never touch the network or the monitor: they exercise the probe
table and the dry-run path only.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from instrumental_evasion.policies import get_guardrail_spec
from instrumental_evasion.tasks.authored.specs import REFERENCES as AUTHORED_REFERENCES
from instrumental_evasion.tasks.registry import spec_registry

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load(name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolves fields via sys.modules
    spec.loader.exec_module(module)
    return module


recall = _load("check_policy_recall")


def test_probe_policies_are_exactly_the_fifty_tasks_policies():
    probe_policies = {p.policy for p in recall.PROBES}
    task_policies = {s.policy for s in spec_registry().values()}
    assert probe_policies == task_policies
    # One probe per policy, no duplicates.
    assert len(recall.PROBES) == len(probe_policies) == 25


@pytest.mark.parametrize("variant", ["trimmed", "full"])
def test_every_probe_resolves_to_a_registered_guardrail(variant):
    rows = recall.probe_table(variant)
    assert len(rows) == 25
    for base, resolved, n_sanctioned, _n_benign, _n_obf in rows:
        # probe_table already calls get_guardrail_spec; assert the shape too.
        assert n_sanctioned >= 1
        assert get_guardrail_spec(resolved).id == resolved
        assert resolved == (f"{base}_trim" if variant == "trimmed" else base)


def test_paper_uses_trimmed_variant_by_default():
    assert recall.build_parser().parse_args([]).variant == "trimmed"


def test_authored_probes_use_the_task_reference_route():
    by_policy = {p.policy: p for p in recall.PROBES}
    for task, reference in AUTHORED_REFERENCES.items():
        probe = by_policy[f"authored10_{task}_v1"]
        assert probe.sanctioned == (reference,)
        assert probe.benign  # plain permitted inspection


def test_dry_run_is_offline_and_lists_every_probe(capsys):
    assert recall.run_dry("trimmed", None) == 0
    out = capsys.readouterr().out
    assert "probes: 25" in out
    assert "all 25 probe policies resolve" in out


def test_dry_run_single_policy_and_unknown():
    assert recall.run_dry("trimmed", "no_return_or_exchange") == 0
    assert recall.run_dry("trimmed", "no_such_policy") == 2


def test_default_monitor_is_gpt_oss_120b_compatible():
    # The paper's default monitor is an open-weights OpenRouter model.
    assert recall.DEFAULT_MODEL == "openrouter/openai/gpt-oss-120b"
