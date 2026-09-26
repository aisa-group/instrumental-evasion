"""Prepared Modal images: manifest resolution and runtime packaging."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


def test_prepared_manifest_resolves_an_immutable_image(monkeypatch, tmp_path):
    from instrumental_evasion.runner import modal_images

    path = tmp_path / "images.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "images": {
                    "task:example:scaffold:codex": {
                        "image_id": "im-prepared123",
                        "task_root": "/opt/task-rootfs",
                        "published_name": "ie-example",
                    }
                },
            }
        )
    )
    seen = []
    monkeypatch.setattr(
        modal_images.modal.Image,
        "from_id",
        lambda image_id: seen.append(image_id) or object(),
    )

    manifest = modal_images.PreparedImageManifest.from_path(path)
    _, task_root = manifest.resolve("task:example:scaffold:codex")

    assert seen == ["im-prepared123"]
    assert task_root == "/opt/task-rootfs"


def test_prepared_manifest_fails_closed_for_a_missing_condition(tmp_path):
    from instrumental_evasion.runner.modal_images import PreparedImageManifest

    path = tmp_path / "images.json"
    path.write_text('{"version": 1, "images": {}}')
    manifest = PreparedImageManifest.from_path(path)

    with pytest.raises(RuntimeError, match="no entry"):
        manifest.resolve("task:example:scaffold:codex")


def test_undefined_manifest_value_disables_prepared_images(monkeypatch):
    from instrumental_evasion.runner.modal_images import MANIFEST_ENV, PreparedImageManifest

    monkeypatch.setenv(MANIFEST_ENV, "UNDEFINED")
    assert PreparedImageManifest.from_env() is None


def test_modal_runtime_uses_prepared_agent_image_without_lazy_build(
    monkeypatch, tmp_path
):
    from instrumental_evasion.runner import modal as modal_runtime
    from instrumental_evasion.runner.modal_images import MANIFEST_ENV

    path = tmp_path / "images.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "images": {
                    "task:example:scaffold:codex": {
                        "image_id": "im-prepared123",
                        "task_root": None,
                    }
                },
            }
        )
    )
    monkeypatch.setenv(MANIFEST_ENV, str(path))
    prepared = object()
    monkeypatch.setattr(modal_runtime.modal.Image, "from_id", lambda _: prepared)
    monkeypatch.setattr(
        modal_runtime,
        "_image_for",
        lambda _: pytest.fail("the lazy image path must not run"),
    )

    scaffold = type("Scaffold", (), {"name": "codex"})()
    spec = type("Spec", (), {"id": "example"})()
    runtime = modal_runtime.ModalRuntime(scaffold=scaffold)
    resolved = asyncio.run(runtime._agent_image(spec))

    assert resolved.image is prepared
    assert runtime.describe()["runtime_prepared_images"] is True


def test_modal_runtime_rejects_a_manifest_for_another_scaffold(
    monkeypatch, tmp_path
):
    from instrumental_evasion.runner import modal as modal_runtime
    from instrumental_evasion.runner.modal_images import MANIFEST_ENV

    path = tmp_path / "images.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "images": {
                    "task:example:scaffold:react": {
                        "image_id": "im-react123",
                        "task_root": None,
                    }
                },
            }
        )
    )
    monkeypatch.setenv(MANIFEST_ENV, str(path))
    runtime = modal_runtime.ModalRuntime(
        scaffold=type("Scaffold", (), {"name": "codex"})()
    )

    with pytest.raises(RuntimeError, match="no entry"):
        asyncio.run(runtime._agent_image(type("Spec", (), {"id": "example"})()))


def test_codex_prepared_runtime_uses_remote_artifacts_not_local_runtime_trees():
    import inspect

    from instrumental_evasion.runner.scaffolds.codex import CodexScaffold

    source = inspect.getsource(CodexScaffold.modal_runtime_image)
    assert "python:3.12.13-slim-bookworm" in source
    assert "codex-package-{target}.tar.gz" in source
    assert 'add_local_dir(hook_src, "/envtools"' in source
    assert "HOOK_PYTHON" not in source
    assert "add_local_dir(package" not in source


@pytest.mark.parametrize("flag", [False, True])
def test_prepared_image_preserves_runtime_packaging(flag):
    from instrumental_evasion.runner.modal_images import PreparedImage

    image = PreparedImage("im-example", None, runtime_embedded=flag)
    assert PreparedImage.from_dict(image.as_dict(), key="example") == image
    assert PreparedImage.from_dict({"image_id": "im-old"}, key="old").runtime_embedded is False


@pytest.mark.parametrize("flag", ["true", 1, None])
def test_prepared_image_rejects_ambiguous_runtime_packaging(flag):
    from instrumental_evasion.runner.modal_images import PreparedImage

    with pytest.raises(ValueError, match="runtime_embedded"):
        PreparedImage.from_dict({"image_id": "im-example", "runtime_embedded": flag}, key="example")


@pytest.mark.parametrize("embedded", [False, True])
def test_startup_mounts_only_a_separate_runtime(monkeypatch, tmp_path, embedded):
    from instrumental_evasion.runner import modal as runtime_module
    from instrumental_evasion.runner.modal_images import PreparedImage, PreparedImageManifest
    from instrumental_evasion.runner.sandbox import ExecResult

    monkeypatch.delenv("MODAL_IMAGE_MANIFEST", raising=False)
    remote = SimpleNamespace(mount_image=Mock())
    monkeypatch.setattr(runtime_module.modal.Sandbox, "create", lambda *a, **kw: remote)
    monkeypatch.setattr(runtime_module.modal.Image, "from_id", lambda value: value)
    monkeypatch.setattr(runtime_module.ModalSandbox, "exec", AsyncMock(return_value=ExecResult(True, 0, "", "")))
    runtime = runtime_module.ModalRuntime(scaffold=SimpleNamespace(name="codex", modal_runtime_image=True))
    runtime._app = object()
    images = {"task:example:scaffold:codex": PreparedImage("im-task", None, runtime_embedded=embedded)}
    if not embedded:
        images["scaffold-runtime:codex"] = PreparedImage("im-runtime", None)
    runtime._prepared_images = PreparedImageManifest(tmp_path / "images.json", images)
    spec = SimpleNamespace(id="example", network_for=lambda **kw: "blackhole", container_env={}, workdir="/app")
    asyncio.run(runtime.open_sandbox(spec, monitored=False))
    if embedded:
        remote.mount_image.assert_not_called()
    else:
        remote.mount_image.assert_called_once_with("/opt", "im-runtime")
