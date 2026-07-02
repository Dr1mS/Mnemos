"""Tests ModelManager (§7.2) — exclusion mutuelle + stress anti-deadlock.

Tests obligatoires Phase 1 : jamais deux tiers actives simultanément,
terminaison garantie (pas de deadlock), limites de concurrence respectées.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mnemos.config import Settings
from mnemos.llm.model_manager import ModelManager, Tier


def make_manager(**overrides: Any) -> ModelManager:
    settings = Settings(_env_file=None, **overrides)
    # client jamais utilisé par acquire/release/use → stub None assumé
    return ModelManager(settings, client=None)  # type: ignore[arg-type]


class TierTracker:
    """Instrumente l'occupation des tiers pour détecter les violations."""

    def __init__(self, limits: dict[Tier, int]) -> None:
        self.active: dict[Tier, int] = {Tier.SMALL: 0, Tier.MEDIUM: 0}
        self.limits = limits
        self.violations: list[str] = []
        self.completed = 0

    def enter(self, tier: Tier) -> None:
        self.active[tier] += 1
        other = Tier.MEDIUM if tier is Tier.SMALL else Tier.SMALL
        if self.active[other] > 0:
            self.violations.append(f"{tier} et {other} actives simultanément")
        if self.active[tier] > self.limits[tier]:
            self.violations.append(f"{tier} dépasse sa limite ({self.active[tier]})")

    def leave(self, tier: Tier) -> None:
        self.active[tier] -= 1
        self.completed += 1


async def run_task(
    manager: ModelManager, tracker: TierTracker, tier: Tier, duration_s: float
) -> None:
    async with manager.use(tier):
        tracker.enter(tier)
        await asyncio.sleep(duration_s)
        tracker.leave(tier)


async def test_exclusion_mutuelle_small_medium() -> None:
    manager = make_manager()
    tracker = TierTracker(manager._limits)
    tasks = [run_task(manager, tracker, Tier.SMALL, 0.01) for _ in range(8)]
    tasks += [run_task(manager, tracker, Tier.MEDIUM, 0.01) for _ in range(3)]
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=10)
    assert tracker.violations == []
    assert tracker.completed == 11


async def test_stress_pas_de_deadlock() -> None:
    """N SMALL + M MEDIUM entrelacées, durées variables → terminaison + invariants."""
    manager = make_manager()
    tracker = TierTracker(manager._limits)
    tasks = []
    for i in range(40):
        tasks.append(run_task(manager, tracker, Tier.SMALL, 0.001 * (i % 5 + 1)))
        if i % 5 == 0:
            tasks.append(run_task(manager, tracker, Tier.MEDIUM, 0.003))
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=30)
    assert tracker.violations == []
    assert tracker.completed == 48


async def test_limite_concurrence_small() -> None:
    """Jamais plus de LLM_TIER_SMALL_CONCURRENCY appels SMALL en vol."""
    manager = make_manager(LLM_TIER_SMALL_CONCURRENCY=2)
    tracker = TierTracker(manager._limits)
    await asyncio.wait_for(
        asyncio.gather(*(run_task(manager, tracker, Tier.SMALL, 0.01) for _ in range(10))),
        timeout=10,
    )
    assert tracker.violations == []


async def test_medium_finit_par_passer() -> None:
    """Après un burst SMALL fini, MEDIUM obtient la tier (pas de famine ici)."""
    manager = make_manager()
    order: list[str] = []

    async def small() -> None:
        async with manager.use(Tier.SMALL):
            order.append("small")
            await asyncio.sleep(0.005)

    async def medium() -> None:
        async with manager.use(Tier.MEDIUM):
            order.append("medium")

    await asyncio.wait_for(
        asyncio.gather(small(), small(), medium()), timeout=5
    )
    assert order.count("medium") == 1
    assert order[-1] == "medium"  # medium passe après le burst small


def test_tier_for_profil_dev() -> None:
    """EXTRACTION_MODEL == SALIENCE_MODEL → tout en SMALL, exclusion inerte."""
    m = make_manager()  # défauts : qwen3:4b partout
    assert m.tier_for("bge-m3") is Tier.SMALL
    assert m.tier_for("qwen3:4b") is Tier.SMALL


def test_tier_for_profil_gpu() -> None:
    """EXTRACTION_MODEL distinct (qwen3:8b) → MEDIUM, exclusion active."""
    m = make_manager(EXTRACTION_MODEL="qwen3:8b")
    assert m.tier_for("qwen3:4b") is Tier.SMALL
    assert m.tier_for("qwen3:8b") is Tier.MEDIUM


async def test_release_remet_la_tier_a_none() -> None:
    manager = make_manager()
    async with manager.use(Tier.MEDIUM):
        assert manager._active_tier is Tier.MEDIUM
    assert manager._active_tier is None
    assert manager._active_count == 0


async def test_exception_dans_use_libere_la_tier() -> None:
    """Un crash dans le bloc use() ne doit pas laisser la tier verrouillée."""
    manager = make_manager()
    with pytest.raises(RuntimeError, match="boom"):
        async with manager.use(Tier.MEDIUM):
            raise RuntimeError("boom")
    # La tier doit être libre : un autre acquire ne doit pas bloquer.
    await asyncio.wait_for(manager.acquire(Tier.SMALL), timeout=1)
    await manager.release(Tier.SMALL)
