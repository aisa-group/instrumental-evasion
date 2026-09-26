"""Read paired fixture assignments for task setup and the startup check."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

MAX_BANK_BYTES = 65_536
BANK_SCHEMA = "clawterminal_fixture_bank_v1"
_TOKEN = re.compile(r"[0-9a-f]{8}")


class FixtureBankError(RuntimeError):
    """The configured bank cannot supply the required fixture assignments."""


@dataclass(frozen=True)
class FixtureBank:
    sha256: str
    assignments: tuple[tuple[str, str], ...] = field(repr=False)

    def token(self, scenario_id: str) -> str:
        for key, value in self.assignments:
            if key == scenario_id:
                return value
        raise FixtureBankError(f"paired fixture assignment is missing for {scenario_id}")

    def metadata(self, required_ids: Iterable[str]) -> dict[str, object]:
        """Describe the checked input without exposing assignment values or paths."""
        return {
            "schema": BANK_SCHEMA,
            "mode": "paired_bank",
            "sha256": self.sha256,
            "required_task_ids": sorted(set(required_ids)),
        }


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureBankError("paired fixture bank contains a duplicate key")
        result[key] = value
    return result


def read_fixture_bank(path: Path | str, required_ids: Iterable[str]) -> FixtureBank:
    """Read a bounded bank and reject malformed or missing required assignments.

    The format is a JSON object from scenario IDs to eight lowercase hex
    characters. File errors and parse errors expose no bank content or token.
    """
    try:
        # Nonblocking open prevents a replaced FIFO from hanging the check.
        descriptor = os.open(Path(path), os.O_RDONLY | os.O_NONBLOCK)
        try:
            with os.fdopen(descriptor, "rb") as stream:
                descriptor = -1
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise FixtureBankError("paired fixture bank must be a regular file")
                payload = stream.read(MAX_BANK_BYTES + 1)
        finally:
            if descriptor != -1:
                os.close(descriptor)
    except OSError as error:
        raise FixtureBankError("paired fixture bank is not readable") from error
    if len(payload) > MAX_BANK_BYTES:
        raise FixtureBankError("paired fixture bank exceeds 65536 bytes")
    try:
        data = json.loads(payload, object_pairs_hook=_unique_object)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise FixtureBankError("paired fixture bank is not valid JSON") from error
    if not isinstance(data, dict) or not data:
        raise FixtureBankError("paired fixture bank must be a nonempty assignment object")
    if any(
        not key or not isinstance(value, str) or _TOKEN.fullmatch(value) is None
        for key, value in data.items()
    ):
        raise FixtureBankError("paired fixture bank has an invalid assignment")
    bank = FixtureBank(hashlib.sha256(payload).hexdigest(), tuple(sorted(data.items())))
    for scenario_id in sorted(set(required_ids)):
        bank.token(scenario_id)
    return bank
