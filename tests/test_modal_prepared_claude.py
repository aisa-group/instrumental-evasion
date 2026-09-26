"""Verify prepared CLI identity before any agent invocation."""

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from instrumental_evasion.hooks import deploy
from instrumental_evasion.runner.sandbox import ExecResult
from instrumental_evasion.runner.scaffolds.claude_code import ClaudeCodeScaffold


@pytest.mark.parametrize("existing", ["matching", "missing", "different", "unprotected"])
def test_prepared_cli_is_reused_only_after_identity_check(tmp_path, monkeypatch, existing):
    source = tmp_path / "claude"
    source.write_bytes(b"qualified executable")
    monkeypatch.setattr(deploy, "CLAUDE_BIN", source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    outputs = [digest if existing == "matching" else existing, digest]
    staged = []

    async def execute(command, **kwargs):
        output = outputs.pop(0)
        return ExecResult(output != "missing", int(output == "missing"), output, "")

    async def stage(*args, **kwargs):
        staged.append((args, kwargs))

    sandbox = SimpleNamespace(exec=execute, stage_path=stage)
    asyncio.run(ClaudeCodeScaffold._stage_modal_cli(sandbox))
    assert len(staged) == int(existing != "matching")
    if staged:
        assert staged[0] == ((source, "/usr/local/bin/claude"), {"read_only": True})


def test_bad_staged_identity_stops_startup(tmp_path, monkeypatch):
    source = tmp_path / "claude"
    source.write_bytes(b"qualified executable")
    monkeypatch.setattr(deploy, "CLAUDE_BIN", source)

    async def execute(*args, **kwargs):
        return ExecResult(True, 0, "wrong hash", "")

    async def stage(*args, **kwargs):
        return None

    with pytest.raises(RuntimeError, match="does not match"):
        asyncio.run(ClaudeCodeScaffold._stage_modal_cli(SimpleNamespace(exec=execute, stage_path=stage)))
