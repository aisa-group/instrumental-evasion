"""Install the pinned model catalog for an explicitly requested direct interface."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def install_direct_catalog(home: Path, container_home: str, model: str) -> str:
    """Validate the prepared catalog, copy it into disposable storage, and return its setting."""
    path = os.environ.get("CODEX_DIRECT_MODEL_CATALOG")
    expected = os.environ.get("CODEX_DIRECT_MODEL_CATALOG_SHA256")
    if not path or not expected:
        raise ValueError("The direct interface requires a pinned model catalog and checksum.")
    source = Path(path)
    if source.stat().st_size > 2_000_000:
        raise ValueError("The direct model catalog exceeds its size limit.")
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("The direct model catalog checksum changed.")
    catalog = json.loads(raw)
    models = catalog.get("models", [])
    if len(models) != 1 or models[0].get("slug") != model or models[0].get("tool_mode") != "direct":
        raise ValueError("The catalog must select direct tools for exactly the requested model.")
    (home / "models-direct.json").write_bytes(raw)
    return f'model_catalog_json = "{container_home}/models-direct.json"'
