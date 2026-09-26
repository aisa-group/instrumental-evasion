"""Keep published image names within the provider's complete-name limit."""

import hashlib
import re
from types import SimpleNamespace

import pytest

from instrumental_evasion.runner.prepare_modal_images import _published_name


def test_explicit_sif_source_records_the_actual_fixture(tmp_path):
    from instrumental_evasion.runner.prepare_modal_images import _source_provenance

    image = tmp_path / "fixture.sif"
    image.write_bytes(b"fixture with installed dependencies")
    spec = SimpleNamespace(id="tb2_build_pmars", docker_image="upstream:tag", image=lambda: image)
    provenance = _source_provenance(spec, prefer_sif=True)
    assert provenance["kind"] == "sif"
    assert provenance["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()


def test_explicit_sif_source_never_falls_back_to_registry(monkeypatch, tmp_path):
    from instrumental_evasion.runner import modal

    def unexpected_registry(*args, **kwargs):
        pytest.fail("The configured SIF was replaced with its upstream image.")

    monkeypatch.setattr(modal.modal.Image, "from_registry", unexpected_registry)
    spec = SimpleNamespace(
        id="tb2_build_pmars", docker_image="upstream:tag", image=lambda: tmp_path / "missing.sif"
    )
    with pytest.raises(FileNotFoundError):
        modal._image_for(spec, prefer_sif=True)


def test_long_toolsandbox_image_name_is_accepted_by_pinned_sdk():
    from modal._image import _validate_image_name

    name = _published_name("task:toolsandbox_temperature_low_battery:scaffold:claude_code", "im-example")
    _validate_image_name(name)
    assert len(name) <= 63
    assert re.fullmatch(r"[a-z0-9-]+", name)


def test_truncated_readable_names_keep_image_identity():
    key = "task:" + "long-task-" * 20
    assert _published_name(key, "im-one") != _published_name(key, "im-two")
