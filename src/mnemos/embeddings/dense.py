"""Embeddings denses (§8.1).

Wrapper autour du ModelManager (jamais OllamaClient direct — anti-pattern 1)
avec cache LRU en mémoire (1000 entrées) sur le hash du contenu, pour éviter
de recomputer les embeddings d'un même texte.
"""

from __future__ import annotations

from collections import OrderedDict
from hashlib import blake2b

from mnemos.config import Settings
from mnemos.llm.model_manager import ModelManager

DEFAULT_CACHE_SIZE = 1000


def _content_key(text: str) -> bytes:
    return blake2b(text.encode("utf-8"), digest_size=16).digest()


class DenseEmbedder:
    def __init__(
        self,
        manager: ModelManager,
        settings: Settings,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self._manager = manager
        self._model = settings.EMBED_MODEL
        self._cache_size = cache_size
        self._cache: OrderedDict[bytes, list[float]] = OrderedDict()

    async def embed(self, text: str) -> list[float]:
        key = _content_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        vector = await self._manager.embed(text, self._model)
        self._cache[key] = vector
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Batch avec cache : seuls les textes non cachés partent vers Ollama."""
        keys = [_content_key(t) for t in texts]
        missing = [(i, t) for i, (k, t) in enumerate(zip(keys, texts, strict=True))
                   if k not in self._cache]
        if missing:
            vectors = await self._manager.embed_batch([t for _, t in missing], self._model)
            for (i, _), vec in zip(missing, vectors, strict=True):
                self._cache[keys[i]] = vec
                if len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        result = []
        for k in keys:
            self._cache.move_to_end(k)
            result.append(self._cache[k])
        return result

    @property
    def cache_len(self) -> int:
        return len(self._cache)
