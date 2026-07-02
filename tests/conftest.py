"""Fixtures communes.

Marqueur @pytest.mark.requires_ollama (§19.4) : skip auto si Ollama down.
"""

from __future__ import annotations

import httpx
import pytest

from mnemos.clock import FixedClock

OLLAMA_HOST = "http://localhost:11434"


def _ollama_up() -> bool:
    try:
        return httpx.get(f"{OLLAMA_HOST}/api/version", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _ollama_up():
        return
    skip = pytest.mark.skip(reason="Ollama down — test requires_ollama skippé")
    for item in items:
        if "requires_ollama" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fixed_clock() -> FixedClock:
    # 2026-07-02T10:00:00Z
    return FixedClock(start_ms=1_782_727_200_000)
