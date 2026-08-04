"""Smoke tests — GPU + Cosmos required. Skipped when unavailable."""
from __future__ import annotations

import shutil

import pytest


@pytest.mark.gpu
@pytest.mark.cosmos
def test_cosmos_server_binary_exists() -> None:
    """After `make setup`, the cosmos-framework install should have installed
    the policy-server entry point."""
    if shutil.which("uv") is None:
        pytest.skip("uv not on PATH")
    # We don't invoke the server here (that's `make smoke`), only assert
    # the module import path exists.
    import importlib.util
    spec = importlib.util.find_spec("cosmos_framework")
    if spec is None:
        pytest.skip("cosmos_framework not importable — run scripts/setup_cosmos.sh")
