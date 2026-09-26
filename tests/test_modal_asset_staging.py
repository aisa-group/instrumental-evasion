"""Keep file contents, task paths, and permissions during Modal staging."""

import asyncio
from types import SimpleNamespace

import pytest

from instrumental_evasion.runner.modal import ModalSandbox
from instrumental_evasion.runner.sandbox import ExecResult


@pytest.mark.parametrize("task_root", [None, "/opt/task-rootfs"])
def test_binary_staging_streams_to_physical_task_path(tmp_path, task_root):
    source = tmp_path / "cli"
    source.write_bytes(b"\x00\xffbinary\n")
    source.chmod(0o755)
    copied = []
    commands = []
    remote = SimpleNamespace(filesystem=SimpleNamespace(
        copy_from_local=lambda local, target: copied.append((local.read_bytes(), target)),
    ))
    sandbox = ModalSandbox(remote, "/app", "blackhole", object(), task_root=task_root)

    async def execute(command, **kwargs):
        commands.append((command, kwargs))
        return ExecResult(True, 0, "", "")

    sandbox.exec = execute
    asyncio.run(sandbox.stage_path(source, "/usr/local/bin/cli", read_only=True))
    target = (task_root or "") + "/usr/local/bin/cli"
    assert copied == [(b"\x00\xffbinary\n", target)]
    assert [row[0] for row in commands] == [["chmod", "755", target], ["chmod", "-R", "a-w", target]]
    assert all(row[1]["inside_task"] is False for row in commands)


def test_failed_transfer_does_not_claim_protected_asset(tmp_path):
    source = tmp_path / "cli"
    source.write_bytes(b"binary")
    remote = SimpleNamespace(filesystem=SimpleNamespace(
        copy_from_local=lambda *args: (_ for _ in ()).throw(OSError("transfer failed")),
    ))
    sandbox = ModalSandbox(remote, "/app", "blackhole", object())
    with pytest.raises(OSError, match="transfer failed"):
        asyncio.run(sandbox.stage_path(source, "/usr/local/bin/cli", read_only=True))
