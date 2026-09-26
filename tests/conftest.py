"""Shared test isolation."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _restore_environment():
    """Undo direct `os.environ` writes, e.g. the paper defaults `runner.run.main` applies."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)
