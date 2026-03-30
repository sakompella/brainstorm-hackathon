"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import urllib.request

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "e2e: end-to-end browser tests (need live server)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip e2e tests unless --base-url is provided and the server is reachable."""
    base_url = config.getoption("--base-url", default=None)

    if base_url and _server_reachable(f"{base_url}/health"):
        return

    skip = pytest.mark.skip(
        reason="e2e tests require --base-url pointing to a running server"
    )
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip)


def _server_reachable(url: str) -> bool:
    """Quick check if the health endpoint responds."""
    try:
        with urllib.request.urlopen(url, timeout=2):
            return True
    except Exception:
        return False
