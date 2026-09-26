"""Prepared Modal image records for fast, reproducible Sandbox startup.

The preparation command builds local task and scaffold layers once. Evaluation
workers then load the immutable Modal image ID from a shared manifest. They do
not scan a SIF or upload a CLI package during ``Sandbox.create``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import modal

MANIFEST_VERSION = 1
MANIFEST_ENV = "MODAL_IMAGE_MANIFEST"


def task_image_key(spec: Any, scaffold_name: str) -> str:
    """Return the manifest key for an agent task image."""
    task_id = getattr(spec, "id", None) or getattr(spec, "tb_name", None)
    if not task_id:
        raise ValueError("a prepared Modal task image needs a stable task id")
    return f"task:{task_id}:scaffold:{scaffold_name}"


def sidecar_image_key(sidecar: Any) -> str:
    """Return the manifest key for a target-service image."""
    identity = getattr(sidecar, "tb_name", None) or getattr(sidecar, "id", None)
    if not identity:
        registry = getattr(sidecar, "docker_image", None)
        identity = f"registry:{registry}" if registry else None
    if not identity:
        raise ValueError("a prepared Modal sidecar image needs a stable identity")
    return f"sidecar:{identity}"


def scaffold_runtime_key(scaffold_name: str) -> str:
    """Return the manifest key for a shared CLI runtime mounted at /opt."""
    return f"scaffold-runtime:{scaffold_name}"


@dataclass(frozen=True)
class PreparedImage:
    """One immutable image and its task-root and runtime packaging conventions.

    An embedded runtime already contains the CLI, interpreter, and hook code.
    The runtime must not overlay a separate scaffold image on its /opt path.
    """

    image_id: str
    task_root: str | None
    published_name: str = ""
    runtime_embedded: bool = False

    @classmethod
    def from_dict(cls, value: Any, *, key: str) -> "PreparedImage":
        if not isinstance(value, dict):
            raise ValueError(f"prepared Modal image {key!r} must be an object")
        image_id = value.get("image_id")
        task_root = value.get("task_root")
        published_name = value.get("published_name", "")
        runtime_embedded = value.get("runtime_embedded", False)
        if not isinstance(image_id, str) or not image_id.startswith("im-"):
            raise ValueError(f"prepared Modal image {key!r} has an invalid image_id")
        if task_root is not None and not isinstance(task_root, str):
            raise ValueError(f"prepared Modal image {key!r} has an invalid task_root")
        if not isinstance(published_name, str):
            raise ValueError(
                f"prepared Modal image {key!r} has an invalid published_name"
            )
        if type(runtime_embedded) is not bool:
            raise ValueError(f"prepared Modal image {key!r} has an invalid runtime_embedded flag")
        return cls(image_id, task_root, published_name, runtime_embedded)

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "task_root": self.task_root,
            "published_name": self.published_name,
            **({"runtime_embedded": True} if self.runtime_embedded else {}),
        }


class PreparedImageManifest:
    """Strict lookup table for images built before an evaluation starts."""

    def __init__(self, path: Path, images: dict[str, PreparedImage]) -> None:
        self.path = path
        self.images = images

    @classmethod
    def from_path(cls, path: Path) -> "PreparedImageManifest":
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read Modal image manifest {path}: {error}") from error
        if not isinstance(raw, dict) or raw.get("version") != MANIFEST_VERSION:
            raise ValueError(
                f"Modal image manifest {path} must have version {MANIFEST_VERSION}"
            )
        values = raw.get("images")
        if not isinstance(values, dict):
            raise ValueError(f"Modal image manifest {path} has no images object")
        images = {
            key: PreparedImage.from_dict(value, key=key)
            for key, value in values.items()
            if isinstance(key, str)
        }
        if len(images) != len(values):
            raise ValueError(f"Modal image manifest {path} has a non-string key")
        return cls(path, images)

    @classmethod
    def from_env(cls) -> "PreparedImageManifest | None":
        value = os.environ.get(MANIFEST_ENV, "").strip()
        if value.upper() == "UNDEFINED":
            value = ""
        return cls.from_path(Path(value)) if value else None

    def resolve(self, key: str) -> tuple[modal.Image, str | None]:
        try:
            prepared = self.images[key]
        except KeyError as error:
            raise RuntimeError(
                f"prepared Modal image manifest {self.path} has no entry for {key!r}; "
                "run the image preparation command for this exact condition"
            ) from error
        return modal.Image.from_id(prepared.image_id), prepared.task_root


def write_manifest(
    path: Path, images: dict[str, PreparedImage], provenance: dict[str, Any]
) -> None:
    """Atomically write trusted preparation output outside agent sandboxes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    payload = {
        "version": MANIFEST_VERSION,
        "images": {key: value.as_dict() for key, value in sorted(images.items())},
        "provenance": provenance,
    }
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
