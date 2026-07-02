"""Intégration Phase 1 (requires_ollama) — round-trip bge-m3 + generate qwen3.

Skip auto si Ollama down (conftest §19.4).
"""

from __future__ import annotations

import json

import pytest

from mnemos.config import Settings
from mnemos.embeddings.dense import DenseEmbedder
from mnemos.llm.model_manager import ModelManager
from mnemos.llm.ollama_client import OllamaClient

pytestmark = pytest.mark.requires_ollama


@pytest.fixture
async def manager() -> ModelManager:
    settings = Settings(_env_file=None)
    return ModelManager(settings, OllamaClient(settings))


async def test_embedding_roundtrip_bge_m3(manager: ModelManager) -> None:
    settings = Settings(_env_file=None)
    embedder = DenseEmbedder(manager, settings)
    vec = await embedder.embed("Alice habite à Lyon.")
    assert len(vec) == 1024  # dim bge-m3 (§5.1)
    assert all(isinstance(x, float) for x in vec)
    # Déterminisme : même texte → même vecteur (via cache ET via Ollama)
    assert await embedder.embed("Alice habite à Lyon.") == vec


async def test_embed_batch_coherent_avec_unitaire(manager: ModelManager) -> None:
    settings = Settings(_env_file=None)
    embedder = DenseEmbedder(manager, settings)
    single = await embedder.embed("le thé vert")
    [batched] = await manager.embed_batch(["le thé vert"], settings.EMBED_MODEL)
    # Même contenu → vecteurs identiques (même endpoint /api/embed)
    assert batched == pytest.approx(single, abs=1e-6)


async def test_generate_json_think_false(manager: ModelManager) -> None:
    """Vérifie que think=false passe bien sur Ollama réel et que le JSON est propre."""
    raw = await manager.generate(
        'Return ONLY this JSON object: {"ok": true}',
        Settings(_env_file=None).SALIENCE_MODEL,
        format="json",
        options={"temperature": 0.0, "num_predict": 64},
    )
    data = json.loads(raw)  # pas de préambule de thinking → parse direct
    assert data.get("ok") is True


async def test_health_check(manager: ModelManager) -> None:
    assert await manager.health_check() is True
