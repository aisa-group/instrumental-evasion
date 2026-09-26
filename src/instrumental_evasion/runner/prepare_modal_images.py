"""Build and publish immutable Modal images before an evaluation batch.

Example:

    uv run python -m instrumental_evasion.runner.prepare_modal_images \
      --scaffold codex --manifest runs/modal-images-codex.json

Then point the runner at the result with MODAL_IMAGE_MANIFEST=<manifest>.

The command runs outside every agent sandbox. It can read task images and
scaffold binaries, but it never copies provider credentials into an image.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from pathlib import Path
from typing import Any

import modal

from instrumental_evasion.runner.modal import (
    _APP_NAME,
    _CLOSED_DOMAIN_SENTINEL,
    ModalImage,
    _image_for,
)
from instrumental_evasion.runner.modal_images import (
    PreparedImage,
    scaffold_runtime_key,
    sidecar_image_key,
    task_image_key,
    write_manifest,
)
from instrumental_evasion.runner.run import load_env_file
from instrumental_evasion.runner.scaffolds import available, get_scaffold
from instrumental_evasion.tasks.registry import selected


def _published_name(key: str, image_id: str) -> str:
    """Return a short versioned name whose suffix identifies the built image."""
    prefix = "ie-"
    suffix = hashlib.sha256(image_id.encode()).hexdigest()[:12]
    # Modal requires the complete name to contain fewer than 64 characters.
    readable_limit = 63 - len(prefix) - len(suffix) - 1
    readable = re.sub(r"[^a-z0-9-]+", "-", key.lower()).strip("-")[:readable_limit]
    return f"{prefix}{readable}-{suffix}"


def _source_provenance(obj: Any, *, prefer_sif: bool = False) -> dict[str, Any]:
    registry = getattr(obj, "docker_image", None)
    if registry and not prefer_sif:
        return {"kind": "registry", "reference": str(registry)}
    task_id = str(getattr(obj, "id", ""))
    if not prefer_sif and task_id.startswith("tauc_tau_"):
        return {"kind": "native_recipe", "id": task_id}
    image_fn = getattr(obj, "image", None)
    if not callable(image_fn):
        return {"kind": "native_recipe", "id": str(getattr(obj, "id", ""))}
    path = Path(image_fn())
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(chunk)
    return {
        "kind": "sif",
        "sha256": digest.hexdigest(),
        "size": path.stat().st_size,
        "source_name": path.name,
    }


def _build_one(
    *,
    key: str,
    resolved: Any,
    app: modal.App,
    provenance: dict[str, Any],
    verify: bool,
    verify_command: tuple[str, ...] = ("true",),
) -> PreparedImage:
    print(f"building remote image for {key}", flush=True)
    built = resolved.image.build(app=app)
    image_id = built.object_id
    if not isinstance(image_id, str) or not image_id.startswith("im-"):
        raise RuntimeError(f"Modal did not return an immutable image ID for {key!r}")
    name = _published_name(key, image_id)
    built.publish(name)
    print(f"published {key} as {image_id}", flush=True)
    provenance[key]["image_id"] = image_id
    provenance[key]["published_name"] = name
    if verify:
        print(f"verifying sandbox startup for {key}", flush=True)
        started = time.monotonic()
        sandbox = modal.Sandbox.create(
            *verify_command,
            app=app,
            image=modal.Image.from_id(image_id),
            timeout=300,
            outbound_domain_allowlist=[_CLOSED_DOMAIN_SENTINEL],
        )
        try:
            sandbox.wait()
            code = sandbox.poll()
        finally:
            sandbox.terminate(wait=False)
        if code != 0:
            raise RuntimeError(f"prepared Modal image {key!r} exited with {code}")
        provenance[key]["verification_startup_seconds"] = round(
            time.monotonic() - started, 3
        )
        print(
            f"verified {key} in "
            f"{provenance[key]['verification_startup_seconds']} seconds",
            flush=True,
        )
    return PreparedImage(image_id, resolved.task_root, name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaffold", choices=available(), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--image-source", choices=("configured", "sif"), default="configured",
        help="use sif to preserve packages added to the configured task and sidecar images",
    )
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="publish images without starting a closed-network verification sandbox",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file()
    scaffold = get_scaffold(args.scaffold)
    scaffold_description = scaffold.describe()
    if scaffold.name == "codex" and not scaffold_description.get("codex_package"):
        raise SystemExit(
            "Codex package is unavailable; set CODEX_BIN to a packaged installation"
        )
    tasks = selected(
        arms=("monitored",),
        only=tuple(args.tasks),
    )
    if not tasks:
        raise SystemExit("no tasks matched")

    app = modal.App.lookup(_APP_NAME, create_if_missing=True)
    images: dict[str, PreparedImage] = {}
    entries: dict[str, Any] = {}
    with modal.enable_output():
        if hasattr(scaffold, "modal_runtime_image"):
            runtime_key = scaffold_runtime_key(scaffold.name)
            entries[runtime_key] = {
                "source": {
                    "kind": "modal_native_scaffold_recipe",
                    "scaffold": scaffold.name,
                    "version": scaffold_description.get("scaffold_version", ""),
                }
            }
            images[runtime_key] = _build_one(
                key=runtime_key,
                resolved=ModalImage(scaffold.modal_runtime_image()),
                app=app,
                provenance=entries,
                verify=not args.no_verify,
                verify_command=(
                    "sh",
                    "-c",
                    "test -x /codex/bin/codex && "
                    "test -x /pyrt/bin/python3 && "
                    "/codex/bin/codex --version && /pyrt/bin/python3 --version",
                ),
            )
        for task in tasks:
            key = task_image_key(task.spec, scaffold.name)
            if key not in images:
                source = _source_provenance(task.spec, prefer_sif=args.image_source == "sif")
                entries[key] = {"source": source, "scaffold": scaffold_description}
                resolved = _image_for(task.spec, prefer_sif=args.image_source == "sif")
                image = resolved.image
                if hasattr(scaffold, "modal_image") and not hasattr(
                    scaffold, "modal_runtime_image"
                ):
                    image = scaffold.modal_image(
                        image, root=resolved.task_root or ""
                    )
                combined = ModalImage(image, resolved.task_root)
                print(f"building {key}", flush=True)
                images[key] = _build_one(
                    key=key,
                    resolved=combined,
                    app=app,
                    provenance=entries,
                    verify=not args.no_verify,
                )

            sidecar = task.spec.sidecar
            if sidecar is not None:
                sidecar_key = sidecar_image_key(sidecar)
                if sidecar_key not in images:
                    entries[sidecar_key] = {"source": _source_provenance(sidecar, prefer_sif=args.image_source == "sif")}
                    print(f"building {sidecar_key}", flush=True)
                    images[sidecar_key] = _build_one(
                        key=sidecar_key,
                        resolved=_image_for(sidecar, prefer_sif=args.image_source == "sif"),
                        app=app,
                        provenance=entries,
                        verify=not args.no_verify,
                    )

    provenance = {
        "scaffold": scaffold.name,
        "image_source": args.image_source,
        "task_count": len(tasks),
        "entries": entries,
    }
    write_manifest(args.manifest, images, provenance)
    print(
        f"wrote {len(images)} immutable image records to {args.manifest}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
