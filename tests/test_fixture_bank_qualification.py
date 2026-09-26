"""Reject broken paired fixtures before any sandbox or agent operation."""

import asyncio
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from instrumental_evasion.runner import run
from instrumental_evasion.tasks.clawbench import specs
from instrumental_evasion.tasks.clawbench.fixture_bank import (
    MAX_BANK_BYTES,
    FixtureBankError,
    read_fixture_bank,
)

TASK = "claw_file_001_csv_markdown"
TOKEN = "a1b2c3d4"


@pytest.mark.parametrize("payload", [
    '{"schema_version":1,"assignments":{"claw_file_001_csv_markdown":"a1b2c3d4"}}',
    '[]', '{}', '{',
    '{"claw_file_001_csv_markdown":"a1b2c3d4","claw_file_001_csv_markdown":"11111111"}',
    '{"claw_file_001_csv_markdown":"A1B2C3D4"}',
    '{"another_task":"a1b2c3d4"}',
    ' ' * (MAX_BANK_BYTES + 1),
])
def test_setup_and_preflight_reject_the_same_invalid_bank(tmp_path, monkeypatch, payload):
    path = tmp_path / "bank.json"
    path.write_text(payload)
    monkeypatch.setenv(specs.FIXTURE_BANK_ENV, str(path))
    for operation in (
        lambda: specs._fixture_token(TASK),
        lambda: specs.validate_configured_fixture_bank([TASK]),
    ):
        with pytest.raises(FixtureBankError) as caught:
            operation()
        assert TOKEN not in str(caught.value)
        assert str(path) not in str(caught.value)


def test_bank_digest_binds_preflight_to_the_setup_input_without_disclosing_tokens(
    tmp_path, monkeypatch,
):
    path = tmp_path / "bank.json"
    payload = json.dumps({TASK: TOKEN}).encode()
    path.write_bytes(payload)
    monkeypatch.setenv(specs.FIXTURE_BANK_ENV, str(path))
    metadata = specs.validate_configured_fixture_bank([TASK, TASK])
    assert metadata["sha256"] == hashlib.sha256(payload).hexdigest()
    assert metadata["required_task_ids"] == [TASK]
    assert specs._fixture_token(TASK) == (TOKEN, "paired_bank")
    assert TOKEN not in json.dumps(metadata)
    assert TOKEN not in repr(read_fixture_bank(path, [TASK]))


def test_random_configuration_check_does_not_generate_an_assignment(monkeypatch):
    monkeypatch.delenv(specs.FIXTURE_BANK_ENV, raising=False)
    monkeypatch.setattr(specs, "_token", lambda: pytest.fail("preflight drew a token"))
    assert specs.validate_configured_fixture_bank([TASK])["mode"] == "random"


def test_unrelated_task_does_not_read_the_configured_bank(monkeypatch):
    monkeypatch.setenv(specs.FIXTURE_BANK_ENV, "/missing/unused-bank.json")
    assert specs.validate_configured_fixture_bank(["tb2_build_pmars"])["mode"] == "not_applicable"


def test_nonregular_bank_is_rejected_without_reading(tmp_path):
    path = tmp_path / "bank.fifo"
    os.mkfifo(path)
    with pytest.raises(FixtureBankError, match="regular file"):
        read_fixture_bank(path, [TASK])


def test_invalid_bank_stops_native_runner_before_scaffold_construction(
    monkeypatch, capsys,
):
    monkeypatch.setattr(run, "load_env_file", lambda: None)
    monkeypatch.setattr(run, "get_scaffold", lambda *a, **k: pytest.fail("scaffold started"))
    monkeypatch.setenv(specs.FIXTURE_BANK_ENV, "/missing/bank.json")
    assert run.main([TASK, "--arm", "monitored", "--dry-run"]) == 2
    assert "fixture configuration" in capsys.readouterr().err


def test_invalid_bank_stops_fixture_setup_before_sandbox_commands(monkeypatch):
    monkeypatch.setenv(specs.FIXTURE_BANK_ENV, "/missing/bank.json")

    async def forbidden(*args, **kwargs):
        pytest.fail("sandbox command ran")

    with pytest.raises(FixtureBankError):
        asyncio.run(specs._setup(SimpleNamespace(exec=forbidden), specs.SCENARIOS[0]))
