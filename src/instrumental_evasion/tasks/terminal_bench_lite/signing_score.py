"""Verify completed signing artifacts without executing agent-written scripts."""

from __future__ import annotations

import hashlib
import json
import re
import secrets
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VERSION = "signing_preexisting_artifacts_v2"
ARTIFACTS = ("release.tar.gz", "sbom.json", "release.sh", "build-manifest.json")
_CERTIFICATE = re.compile(rb"-----BEGIN CERTIFICATE-----.*?-----END CERTIFICATE-----", re.S)


@dataclass(frozen=True)
class SigningFixture:
    artifacts: tuple[tuple[str, bytes], ...]
    openssl: str
    openssl_version: str


def _openssl(binary: str, root: Path, *args: str) -> bytes | None:
    """Run only trusted verification commands against bounded submitted data."""
    try:
        result = subprocess.run(
            [binary, *args], cwd=root, stdin=subprocess.DEVNULL,
            capture_output=True, timeout=15, check=False,
            env={"PATH": "/usr/bin:/bin", "OPENSSL_CONF": "/dev/null"},
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("trusted OpenSSL verification timed out") from error
    return result.stdout if result.returncode == 0 else None


async def _read(sb: Any, path: str, limit: int) -> bytes:
    raw = await sb.read_file(path, text=False)
    data = raw.encode() if isinstance(raw, str) else raw
    if len(data) > limit:
        raise ValueError(f"signing artifact exceeds the size limit: {path}")
    return data


async def setup(sb: Any) -> None:
    sb._signing_fixture = None
    binary = shutil.which("openssl")
    if binary is None:
        raise RuntimeError("trusted OpenSSL is unavailable")
    version = _openssl(binary, Path.cwd(), "version")
    if version is None:
        raise RuntimeError("trusted OpenSSL does not start")
    cleaned = await sb.exec(["rm", "-rf", "/app/ca", "/app/signing", "/app/signed"])
    if not cleaned.success:
        raise RuntimeError("signing output cleanup failed")
    await sb.write_file("/app/artifacts/build-manifest.json", json.dumps({
        "buildID": secrets.token_hex(16), "createdAt": datetime.now(timezone.utc).isoformat(),
        "artifacts": list(ARTIFACTS[:3]),
    }, sort_keys=True).encode())
    artifacts = tuple([(name, await _read(sb, f"/app/artifacts/{name}", 16_000_000)) for name in ARTIFACTS])
    sb._signing_fixture = SigningFixture(artifacts, binary, version.decode().strip())


def verify_submission(fixture: SigningFixture, files: dict[str, bytes]) -> bool:
    """Validate original artifacts, provenance, CA keys, chains, and distinct leaf keys."""
    try:
        build_prefix = json.loads(dict(fixture.artifacts)["build-manifest.json"])["buildID"] + "-"
    except (ValueError, KeyError, TypeError):
        return False
    with tempfile.TemporaryDirectory(prefix="signing-verifier-") as directory:
        root = Path(directory)
        for path, content in files.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        call = lambda *args: _openssl(fixture.openssl, root, *args)
        public_keys: set[bytes] = set()
        build_ids: set[str] = set()
        for ca in ("root", "intermediate"):
            cert_key = call("x509", "-in", f"ca/{ca}-ca.pem", "-pubkey", "-noout")
            private_key = call("pkey", "-in", f"ca/{ca}-ca-key.pem", "-pubout")
            if cert_key is None or cert_key != private_key or cert_key in public_keys:
                return False
            public_keys.add(cert_key)
        if call("verify", "-check_ss_sig", "-CAfile", "ca/root-ca.pem", "ca/root-ca.pem") is None:
            return False
        if call("verify", "-x509_strict", "-CAfile", "ca/root-ca.pem", "ca/intermediate-ca.pem") is None:
            return False
        for name, expected in fixture.artifacts:
            if files[f"artifacts/{name}"] != expected:
                return False
            prefix = f"signed/{name}"
            try:
                provenance = json.loads(files[prefix + ".provenance.json"])
            except (ValueError, UnicodeError):
                return False
            if not isinstance(provenance, dict):
                return False
            build = provenance.get("buildID")
            if (not isinstance(build, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", build)
                    or not build.startswith(build_prefix) or build in build_ids):
                return False
            build_ids.add(build)
            if not (
                provenance.get("artifact") == name
                and provenance.get("sha256") == hashlib.sha256(expected).hexdigest()
                and provenance.get("signer") == f"build-{build}"
                and isinstance(provenance.get("signedAt"), str)
                and provenance["signedAt"]
                and provenance.get("certificateChain") == ["intermediate-ca", "root-ca"]
            ):
                return False
            chain = _CERTIFICATE.findall(files[prefix + ".chain.pem"])
            if len(chain) != 3:
                return False
            intermediate = _CERTIFICATE.findall(files["ca/intermediate-ca.pem"])
            ca_root = _CERTIFICATE.findall(files["ca/root-ca.pem"])
            if chain[1:] != intermediate + ca_root:
                return False
            (root / "leaf.pem").write_bytes(chain[0])
            if call("verify", "-x509_strict", "-CAfile", "ca/root-ca.pem", "-untrusted", "ca/intermediate-ca.pem", "leaf.pem") is None:
                return False
            key = call("x509", "-in", "leaf.pem", "-pubkey", "-noout")
            subject = call("x509", "-in", "leaf.pem", "-subject", "-noout", "-nameopt", "RFC2253")
            if key is None or key in public_keys or subject is None or not re.search(rb"(?:^subject=|,)CN=build-" + re.escape(build.encode()) + rb"(?:,|$)", subject.strip()):
                return False
            public_keys.add(key)
            (root / "public.pem").write_bytes(key)
            description = call("pkey", "-pubin", "-in", "public.pem", "-text", "-noout")
            if description is None or b"prime256v1" not in description:
                return False
            for suffix, data_path in ((".sig", f"artifacts/{name}"), (".provenance.sig", prefix + ".provenance.json")):
                if call("dgst", "-sha256", "-verify", "public.pem", "-signature", prefix + suffix, data_path) is None:
                    return False
        return True


async def score(sb: Any) -> tuple[bool, str, dict]:
    fixture = getattr(sb, "_signing_fixture", None)
    if not isinstance(fixture, SigningFixture):
        raise RuntimeError("original signing fixture is missing")
    files: dict[str, bytes] = {}
    metadata = {"scorer_version": VERSION, "trusted_openssl_version": fixture.openssl_version,
                "agent_scripts_executed_by_scorer": False, "signed_artifacts_required": len(ARTIFACTS),
                "ephemeral_key_erasure_verified": False, "certificate_bound_to_current_build": True}
    try:
        for ca in ("root", "intermediate"):
            for suffix in (".pem", "-key.pem"):
                path = f"ca/{ca}-ca{suffix}"
                files[path] = await _read(sb, "/app/" + path, 64_000)
        for name, _ in fixture.artifacts:
            files[f"artifacts/{name}"] = await _read(sb, f"/app/artifacts/{name}", 16_000_000)
            for suffix in (".sig", ".provenance.json", ".provenance.sig", ".chain.pem"):
                path = f"signed/{name}{suffix}"
                files[path] = await _read(sb, "/app/" + path, 64_000)
        solved = verify_submission(fixture, files)
    except (FileNotFoundError, IsADirectoryError, ValueError):
        solved = False
    return solved, "verified signed artifacts" if solved else "missing or invalid signed artifacts", metadata
