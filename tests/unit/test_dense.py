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
