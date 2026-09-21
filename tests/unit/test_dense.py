"""Tests DenseEmbedder (§8.1) — cache LRU, pas d'appel Ollama réel."""

from __future__ import annotations

from typing import Any

from mnemos.config import Settings
from mnemos.embeddings.dense import DenseEmbedder


class StubManager:
    """Double du ModelManager : vecteur déterministe, compte les appels."""

    def __init__(self) -> None:
        self.embed_calls = 0
        self.batch_calls = 0

    async def embed(self, text: str, model: str) -> list[float]:
        self.embed_calls += 1
        return [float(len(text)), 0.0]

    async def embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
        self.batch_calls += 1
        return [[float(len(t)), 0.0] for t in texts]


def make_embedder(cache_size: int = 1000) -> tuple[DenseEmbedder, StubManager]:
    stub = StubManager()
    settings = Settings(_env_file=None)
    return DenseEmbedder(stub, settings, cache_size=cache_size), stub  # type: ignore[arg-type]


async def test_cache_hit_evite_le_recompute() -> None:
    embedder, stub = make_embedder()
    v1 = await embedder.embed("bonjour")
    v2 = await embedder.embed("bonjour")
    assert v1 == v2
    assert stub.embed_calls == 1


async def test_textes_differents_recomputent() -> None:
    embedder, stub = make_embedder()
    await embedder.embed("a")
    await embedder.embed("bb")
    assert stub.embed_calls == 2


async def test_eviction_lru() -> None:
    embedder, stub = make_embedder(cache_size=2)
    await embedder.embed("un")
    await embedder.embed("deux")
    await embedder.embed("trois")  # évince "un"
    assert embedder.cache_len == 2
    await embedder.embed("un")  # recompute
    assert stub.embed_calls == 4


async def test_batch_ne_refetch_que_les_manquants() -> None:
    embedder, stub = make_embedder()
    await embedder.embed("connu")
    vecs = await embedder.embed_batch(["connu", "nouveau"])
    assert len(vecs) == 2
    assert stub.embed_calls == 1
    assert stub.batch_calls == 1  # un seul batch, avec uniquement "nouveau"


async def test_batch_tout_en_cache_zero_appel(monkeypatch: Any) -> None:
    embedder, stub = make_embedder()
    await embedder.embed_batch(["a", "b"])
    calls_before = stub.batch_calls
    await embedder.embed_batch(["a", "b"])
    assert stub.batch_calls == calls_before


async def test_embed_batch_sous_concurrence_ne_perd_pas_le_cache() -> None:
    """Reproduit la panne de production du 21/09 (concurrence 16).

    `embed_batch` testait la présence des clés AVANT l'await puis relisait le
    cache APRÈS. Entre les deux, d'autres coroutines remplissent le cache et
    évincent l'entrée qu'on croyait acquise : `move_to_end` levait alors
    KeyError et tout le lot d'Add échouait en HTTP 500.

    Cache minuscule + lots concurrents = éviction garantie pendant l'attente.
    """
    import asyncio

    class SlowManager(StubManager):
        async def embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
            await asyncio.sleep(0)  # rend la main : les autres coroutines s'intercalent
            return await super().embed_batch(texts, model)

    stub = SlowManager()
    settings = Settings(_env_file=None)
    embedder = DenseEmbedder(stub, settings, cache_size=4)  # type: ignore[arg-type]

    await embedder.embed_batch(["commun-a", "commun-b"])  # amorce le cache

    async def lot(n: int) -> list[list[float]]:
        return await embedder.embed_batch([f"texte-{n}-{j}" for j in range(3)] + ["commun-a"])

    results = await asyncio.gather(*(lot(n) for n in range(16)))

    for n, vectors in enumerate(results):
        assert len(vectors) == 4
        for j in range(3):
            assert vectors[j] == [float(len(f"texte-{n}-{j}")), 0.0]
        assert vectors[3] == [float(len("commun-a")), 0.0]
