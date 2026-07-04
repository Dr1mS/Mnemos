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
HEALTH_TIMEOUT_S = 2.0  # sonde /health : court, appelé à chaque tick Atelios


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

    async def embed_probe(self, model: str) -> str | None:
        """Sonde réelle de /api/embed (§Santé) : la panne qui a rendu query ET
        write inutilisables était sur cet endpoint, pas sur /api/version — un
        simple health_check() ne l'aurait pas vue.

        Retourne None si l'embedding répond, sinon une chaîne décrivant la
        panne (à afficher dans /health). Timeout court (2 s)."""
        try:
            resp = await self._client.post(
                f"{self._host}/api/embed",
                json={"model": model, "input": ["ping"]},
                timeout=HEALTH_TIMEOUT_S,
            )
        except httpx.TimeoutException:
            # >2s : soit /api/embed est vraiment en peine, soit le modèle
            # d'embedding est en cours de chargement (cold start après un
            # restart Ollama ou l'expiration du keep_alive). Dans les deux cas
            # l'embedding n'est pas prêt à servir CE tick — on le signale.
            return (
                f"timeout > {HEALTH_TIMEOUT_S:g}s sur /api/embed ({self._host}) "
                f"— modèle {model} en chargement (cold start) ou endpoint en peine"
            )
        except httpx.HTTPError as exc:
            return f"/api/embed injoignable ({self._host}) : {exc}"
        if resp.status_code != 200:
            # Modèle non pullé → 404 ; c'est le cas typique d'une panne embed.
            return f"/api/embed HTTP {resp.status_code} (modèle {model} ?) : {resp.text[:120]}"
        embeddings = resp.json().get("embeddings")
        if not embeddings:
            return f"/api/embed a répondu sans embeddings (modèle {model})"
        return None
