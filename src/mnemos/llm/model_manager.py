"""ModelManager (§7.2) — tiered semaphore routing.

Implémentation de référence du spec, à ne pas "améliorer" sans mesure :
c'est le composant le plus piégeux du système (deadlocks Oracle Engine
sur PulseWorld).

Invariants :
- Une seule tier "active" à la fois (contrainte VRAM/RAM).
- _active_count = nombre d'appels en vol sur la tier active.
- Un appel d'une autre tier attend que _active_count retombe à 0.
- INTERDIT d'attendre un changement de tier en tenant un lock ou un
  sémaphore — c'est exactement le deadlock PulseWorld. D'où une unique
  Condition, pas de sémaphores séparés.

Famine assumée (§7.2) : sous trafic SMALL continu, MEDIUM attend. C'est le
bon trade-off MVP — pas de mécanisme de fairness sans mesure.

Routage par MODÈLE, pas par usage : en profil dev, EXTRACTION_MODEL ==
SALIENCE_MODEL → l'extraction tourne en tier SMALL et l'exclusion ne se
déclenche jamais. Elle protège le profil GPU (qwen3:8b en MEDIUM).

Tous les appels Ollama passent par ce manager (anti-pattern 1, §20).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from enum import StrEnum
from typing import Any, Literal

from mnemos.config import Settings
from mnemos.llm.ollama_client import OllamaClient


class Tier(StrEnum):
    SMALL = "small"  # bge-m3, qwen3:4b (salience + extraction en profil dev)
    MEDIUM = "medium"  # qwen3:8b (profil GPU uniquement, optionnel)
    # LARGE réservé pour future extension


class ModelManager:
    def __init__(self, settings: Settings, client: OllamaClient) -> None:
        self._client = client
        self._think = settings.LLM_THINK
        self._limits = {
            Tier.SMALL: settings.LLM_TIER_SMALL_CONCURRENCY,
            Tier.MEDIUM: settings.LLM_TIER_MEDIUM_CONCURRENCY,
        }
        # Modèles routés SMALL ; tout le reste (qwen3:8b…) part en MEDIUM.
        self._small_models = {settings.EMBED_MODEL, settings.SALIENCE_MODEL}
        if settings.EXTRACTION_MODEL == settings.SALIENCE_MODEL:
            self._small_models.add(settings.EXTRACTION_MODEL)
        self._state = asyncio.Condition()
        self._active_tier: Tier | None = None
        self._active_count = 0

    def tier_for(self, model: str) -> Tier:
        return Tier.SMALL if model in self._small_models else Tier.MEDIUM

    async def acquire(self, tier: Tier) -> None:
        async with self._state:
            await self._state.wait_for(
                lambda: self._active_tier in (None, tier)
                and self._active_count < self._limits[tier]
            )
            self._active_tier = tier
            self._active_count += 1

    async def release(self, tier: Tier) -> None:
        async with self._state:
            self._active_count -= 1
            if self._active_count == 0:
                self._active_tier = None
            self._state.notify_all()

    @asynccontextmanager
    async def use(self, tier: Tier) -> AsyncIterator[None]:
        await self.acquire(tier)
        try:
            yield
        finally:
            await self.release(tier)

    # ── Point de sortie unique vers Ollama (anti-pattern 1) ──────────────────

    async def embed(self, text: str, model: str) -> list[float]:
        async with self.use(self.tier_for(model)):
            return await self._client.embed(text, model)

    async def embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
        async with self.use(self.tier_for(model)):
            return await self._client.embed_batch(texts, model)

    async def generate(
        self,
        prompt: str,
        model: str,
        format: Literal["json"] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        async with self.use(self.tier_for(model)):
            return await self._client.generate(
                prompt, model, format=format, options=options, think=self._think
            )

    async def health_check(self) -> bool:
        # Pas de tier : simple GET /api/version, aucune inférence.
        return await self._client.health_check()
