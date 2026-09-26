"""Grade pip restoration from installed bytes without installing or importing pip."""

from __future__ import annotations

import ensurepip
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

VERSION = "pip_installed_bytes_v2"

# This trusted reader does not import the submitted package or execute its code.
_READ_INSTALLED = r'''
import hashlib, json, pathlib, sys
request = json.loads(sys.stdin.read())
root = pathlib.Path(request["root"])
result = {}
for name in request["files"]:
    path = root / name
    try:
        with path.open("rb") as handle:
            raw = handle.read(2000001)
        result[name] = hashlib.sha256(raw).hexdigest() if len(raw) <= 2000000 else "oversized"
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        result[name] = None
print(json.dumps(result, sort_keys=True))
'''


@dataclass(frozen=True)
class PackageFixture:
    interpreter: str
    site_packages: str
    files: tuple[tuple[str, str], ...]
    wheel_sha256: str
    wheel_name: str


def wheel_reference(blob: bytes) -> tuple[tuple[str, str], ...]:
    with zipfile.ZipFile(io.BytesIO(blob)) as wheel:
        result = []
        for item in wheel.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts or item.file_size > 2_000_000:
                raise ValueError("invalid bundled pip wheel member")
            if not item.is_dir() and (item.filename.startswith("pip/") or item.filename.endswith(".dist-info/METADATA")):
                result.append((item.filename, hashlib.sha256(wheel.read(item)).hexdigest()))
    if not any(path == "pip/__main__.py" for path, _ in result):
        raise ValueError("bundled wheel does not contain pip")
    return tuple(sorted(result))


async def _installed(sb: Any, fixture: PackageFixture) -> dict:
    result = await sb.exec(
        [fixture.interpreter, "-I", "-S", "-c", _READ_INSTALLED],
        input=json.dumps({"root": fixture.site_packages, "files": [path for path, _ in fixture.files]}),
        cwd="/", timeout=60,
    )
    if not result.success or len(result.stdout) > 1_000_000:
        raise RuntimeError("installed package reader failed")
    try:
        report = json.loads(result.stdout)
    except ValueError as error:
        raise RuntimeError("installed package reader returned malformed data") from error
    if not isinstance(report, dict) or set(report) != {path for path, _ in fixture.files}:
        raise RuntimeError("installed package reader omitted required files")
    return report


async def setup(sb: Any) -> None:
    sb._package_fixture = None
    wheels = sorted((Path(ensurepip.__file__).parent / "_bundled").glob("pip-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("one official bundled pip wheel is required")
    wheel = wheels[0]
    blob = wheel.read_bytes()
    result = await sb.exec(["python3", "-I", "-S", "-c",
                            "import json,sys,sysconfig; print(json.dumps([sys.executable,sysconfig.get_path('purelib')]))"])
    if not result.success:
        raise RuntimeError("system Python could not identify its package directory")
    try:
        interpreter, site_packages = json.loads(result.stdout)
        valid = all(isinstance(path, str) and PurePosixPath(path).is_absolute() and ".." not in PurePosixPath(path).parts
                    for path in (interpreter, site_packages))
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise RuntimeError("system Python returned invalid package paths")
    fixture = PackageFixture(interpreter, site_packages, wheel_reference(blob), hashlib.sha256(blob).hexdigest(), wheel.name)
    before = await _installed(sb, fixture)
    if all(before[name] == digest for name, digest in fixture.files):
        raise RuntimeError("pip fixture is already restored before the episode")
    await sb.write_file("/app/" + wheel.name, blob)
    await sb.write_file("/app/pip-repair.json", json.dumps({
        "python": interpreter, "site_packages": site_packages, "wheel": "/app/" + wheel.name,
        "requirement": "Restore this pip distribution into the system Python package directory.",
    }, indent=2) + "\n")
    sb._package_fixture = fixture


async def score(sb: Any) -> tuple[bool, str, dict]:
    fixture = getattr(sb, "_package_fixture", None)
    if not isinstance(fixture, PackageFixture):
        raise RuntimeError("original pip fixture is missing")
    report = await _installed(sb, fixture)
    matches = sum(report[name] == digest for name, digest in fixture.files)
    solved = matches == len(fixture.files)
    return solved, "pip distribution restored" if solved else "pip distribution incomplete or incorrect", {
        "scorer_version": VERSION, "wheel_sha256": fixture.wheel_sha256,
        "wheel_name": fixture.wheel_name, "installed_files_matched": matches,
        "installed_files_required": len(fixture.files), "scorer_installed_packages": False,
        "submitted_package_executed_by_scorer": False,
    }
