"""Client HTTP Ollama (§7.1).

httpx.AsyncClient, timeouts explicites (60s embed, 300s generate — profil
CPU oblige), retry exponentiel (3 tentatives) sur erreurs réseau/5xx,
PAS de retry sur erreurs HTTP 4xx.

⚠ Ne jamais appeler ce client directement depuis le code applicatif :
tout passe par le ModelManager (anti-pattern 1, §20).
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

import httpx

from mnemos.config import Settings
from mnemos.logging import get_logger

logger = get_logger(__name__)

EMBED_TIMEOUT_S = 60.0
GENERATE_TIMEOUT_S = 300.0
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_S = 0.5


class OllamaError(Exception):
    """Erreur Ollama non récupérable (4xx, réponse malformée)."""


class OllamaClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._host = settings.OLLAMA_HOST.rstrip("/")
        self._client = client or httpx.AsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        """POST avec retry exponentiel sur erreurs transitoires uniquement."""
        last_exc: Exception | None = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                resp = await self._client.post(
                    f"{self._host}{path}", json=payload, timeout=timeout_s
                )
                if 400 <= resp.status_code < 500:
                    # 4xx : erreur de requête, retry inutile (§7.1)
                    raise OllamaError(f"HTTP {resp.status_code} sur {path} : {resp.text[:200]}")
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                return data
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                delay = RETRY_BASE_DELAY_S * (2**attempt)
                logger.warning(
                    "ollama_retry", path=path, attempt=attempt + 1, delay_s=delay, error=str(exc)
                )
                await asyncio.sleep(delay)
        raise OllamaError(f"échec après {RETRY_ATTEMPTS} tentatives sur {path}") from last_exc

    async def embed(self, text: str, model: str) -> list[float]:
        vectors = await self.embed_batch([text], model)
        return vectors[0]

    async def embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
        data = await self._post(
            "/api/embed", {"model": model, "input": texts}, timeout_s=EMBED_TIMEOUT_S
        )
        embeddings: list[list[float]] = data.get("embeddings", [])
        if len(embeddings) != len(texts):
            raise OllamaError(
                f"embed_batch : {len(embeddings)} vecteurs pour {len(texts)} textes"
            )
        return embeddings

    async def generate(
        self,
        prompt: str,
        model: str,
        format: Literal["json"] | None = None,
        options: dict[str, Any] | None = None,
        think: bool = False,  # TOUJOURS False pour qwen3 (§2 : JSON cassé + latence ×5-10)
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": think,
        }
        if format is not None:
            payload["format"] = format
        if options is not None:
            payload["options"] = options
        data = await self._post("/api/generate", payload, timeout_s=GENERATE_TIMEOUT_S)
        response = data.get("response")
        if not isinstance(response, str):
            raise OllamaError("generate : champ 'response' absent ou invalide")
        return response

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self._host}/api/version", timeout=5)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False
