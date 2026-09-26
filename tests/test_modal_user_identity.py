"""Keep the requested OS identity outside PRoot identity emulation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from instrumental_evasion.runner.modal import ModalSandbox


def remote_process():
    return SimpleNamespace(
        stdin=SimpleNamespace(write_eof=Mock(), drain=Mock()),
        stdout=SimpleNamespace(read=Mock(return_value="")),
        stderr=SimpleNamespace(read=Mock(return_value="")),
        wait=Mock(return_value=0),
    )


@pytest.mark.parametrize("task_root", [None, "/opt/task-rootfs"])
@pytest.mark.parametrize("inside_task", [False, True])
def test_numeric_user_drops_os_identity_before_task_execution(task_root, inside_task):
    remote = SimpleNamespace(exec=Mock(return_value=remote_process()))
    sandbox = ModalSandbox(remote, "/app", "none", None, task_root=task_root)
    result = asyncio.run(sandbox.exec(["true"], user="1000", inside_task=inside_task))
    assert result.success
    command = remote.exec.call_args.args
    if task_root and inside_task:
        assert command[:5] == (
            "setpriv", "--reuid=1000", "--regid=1000", "--clear-groups", "proot",
        )
        assert "--change-id" not in command
    else:
        assert command[:3] == ("/opt/pyrt/bin/python3", "-I", "-c")
        launcher = command[3]
        assert launcher.index("os.setgroups([])") < launcher.index("os.setgid(uid)")
        assert launcher.index("os.setgid(uid)") < launcher.index("os.setuid(uid)")
        assert launcher.index("os.setuid(uid)") < launcher.index("os.execvpe")
        assert command[4:] == ("1000", "true")
    remote.exec.assert_called_once()


def test_failed_identity_launcher_has_no_root_fallback():
    process = remote_process()
    process.wait.return_value = 1
    remote = SimpleNamespace(exec=Mock(return_value=process))
    sandbox = ModalSandbox(remote, "/app", "none", None, task_root="/opt/task-rootfs")
    result = asyncio.run(sandbox.exec(["true"], user="1000"))
    assert not result.success
    assert result.returncode == 1
    remote.exec.assert_called_once()


def test_trusted_setup_without_requested_user_keeps_its_identity():
    remote = SimpleNamespace(exec=Mock(return_value=remote_process()))
    sandbox = ModalSandbox(remote, "/app", "none", None, task_root="/opt/task-rootfs")
    asyncio.run(sandbox.exec(["true"], inside_task=False))
    assert remote.exec.call_args.args == ("true",)
