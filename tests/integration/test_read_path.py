"""Read path avec Ollama réel (§19.2) — précision de la recherche hybride.

Write 50 épisodes → les requêtes retrouvent les bons épisodes en top-k.
+ Pattern separation : même contenu, buckets de 4h différents (§8.2).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text

from mnemos.clock import FixedClock
from mnemos.config import Settings
from mnemos.embeddings.dense import DenseEmbedder
from mnemos.embeddings.sparse import hamming_distance
from mnemos.llm.model_manager import ModelManager
from mnemos.llm.ollama_client import OllamaClient
from mnemos.models.base import make_async_engine
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
from mnemos.stores.episodic import EpisodicStore

pytestmark = pytest.mark.requires_ollama

TOPICS = [
    ("cuisine", "J'ai testé une recette de ramen au miso hier soir, un délice."),
    ("cuisine", "Le secret d'un bon risotto c'est le bouillon ajouté louche par louche."),
    ("code", "Le bug venait d'une race condition dans le worker asyncio."),
    ("code", "J'ai migré le projet vers SQLAlchemy 2.0 en mode async."),
    ("sport", "Séance de grimpe en salle, j'ai enfin passé mon premier 6b."),
    ("sport", "Le vélo au bord du Rhône dimanche matin, 40 km sans douleur."),
    ("musique", "Le dernier album de jazz fusion tourne en boucle chez moi."),
    ("voyage", "On planifie deux semaines au Japon pour voir Kyoto et Osaka."),
    ("santé", "Le médecin recommande plus de sommeil et moins d'écrans le soir."),
    ("travail", "La réunion de sprint planning a duré deux heures, épuisant."),
]


@pytest.fixture
async def store(tmp_path: Path, fixed_clock: FixedClock) -> AsyncIterator[EpisodicStore]:
    settings = Settings(_env_file=None)
    engine = make_async_engine(tmp_path / "episodic.db")
    async with engine.begin() as conn:
        for stmt in EPISODIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    manager = ModelManager(settings, OllamaClient(settings))
    yield EpisodicStore(engine, DenseEmbedder(manager, settings), fixed_clock, settings)
    await engine.dispose()


async def test_write_50_search_precis(store: EpisodicStore, fixed_clock: FixedClock) -> None:
    for i in range(5):
        for _topic, content in TOPICS:
            await store.write(f"{content} (jour {i})", role="user")
            fixed_clock.advance(3_600_000)  # 1h entre épisodes

    queries = [
        ("comment réussir un risotto ?", "risotto"),
        ("problème de concurrence asyncio", "race condition"),
        ("escalade en salle", "grimpe"),
        ("itinéraire pour le Japon", "Japon"),
    ]
    for query, expected_fragment in queries:
        results = await store.search(query, k=5)
        assert results, f"aucun résultat pour {query!r}"
        top_contents = " | ".join(r.episode.content for r in results[:3])
        assert expected_fragment.lower() in top_contents.lower(), (
            f"{query!r} → top-3 sans {expected_fragment!r} : {top_contents}"
        )


async def test_pattern_separation_buckets_4h(
    store: EpisodicStore, fixed_clock: FixedClock
) -> None:
    """Deux épisodes de même contenu dans des buckets de 4h différents →
    codes sparse distincts en DB (§8.2, test obligatoire Phase 2)."""
    ep1 = await store.write("rendez-vous dentiste à confirmer", role="user")
    fixed_clock.advance(5 * 3_600_000)  # 5h → bucket suivant garanti
    ep2 = await store.write("rendez-vous dentiste à confirmer", role="user")

    async with store._sessions() as session:
        rows = await session.execute(
            text("SELECT episode_id, sparse_bits FROM episodes_sparse "
                 "WHERE episode_id IN (:a, :b)"),
            {"a": ep1.id, "b": ep2.id},
        )
        bits = {row[0]: row[1] for row in rows}
    assert hamming_distance(bits[ep1.id], bits[ep2.id]) > 0
    # et les deux restent retrouvables
    results = await store.search("rendez-vous dentiste", k=5)
    found = {r.episode.id for r in results}
    assert {ep1.id, ep2.id} <= found
