"""Client d'embeddings parlant directement à `llama-server` (llama.cpp).

Pourquoi doubler `OllamaClient` (mesures du 21/09, §11 de AML_COMPETITION_PREP) :

Ollama force `-np 1` pour les modèles d'embedding et interpose un saut
parent → runner, avec un aller-retour `/tokenize` par appel. Sous charge, son
ordonnanceur démarre des runners supplémentaires et répond HTTP 400 pendant
leur montée, ce qui produisait 2 % d'échecs sur /add. Le surcoût mesuré est de
~146 ms par embedding, indépendamment de la version : 162 ms via Ollama contre
**15,8 ms** en direct, pour le même fichier de modèle.

Les vecteurs sont identiques (cosinus 0,999999650) : aucun bench n'est invalidé.

`llama-server` expose une API compatible OpenAI. On n'utilise que
`/v1/embeddings` et `/health` — la génération (saillance, extraction) reste sur
Ollama, qui gère le cycle de vie de plusieurs modèles.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from mnemos.config import Settings
from mnemos.llm.ollama_client import (
    _PROBE_TIMEOUT_PREFIX,
    EMBED_TIMEOUT_S,
    HEALTH_TIMEOUT_S,
    RETRY_ATTEMPTS,
    RETRY_BASE_DELAY_S,
    OllamaError,
)
from mnemos.logging import get_logger

logger = get_logger(__name__)

# Pour un embedding, llama.cpp exige que toute l'entrée tienne dans un seul lot
# physique. Au-delà, il répond 500 « input (N tokens) is too large to process ».
# Rejouer la même requête ne peut pas aboutir : le smoke test du 22/09 est resté
# bloqué un quart d'heure là-dessus, la plateforme rejouant en boucle.
_TOO_LARGE_MARKERS = ("too large to process", "exceeds the available context", "n_ubatch")

# Budget de repli, en caractères. Volontairement prudent : en CJK un caractère
# vaut environ un token, donc 8 000 caractères restent sous les 8 192 tokens de
# bge-m3 quelle que soit la langue. On ne tronque QUE si le serveur a refusé —
# un texte latin de 30 000 caractères passe donc intact.
SAFE_INPUT_CHARS = 8000
MIN_INPUT_CHARS = 1000


def is_input_too_large(body: str) -> bool:
    """Vrai si llama-server refuse l'entrée pour cause de taille."""
    lowered = body.lower()
    return any(marker in lowered for marker in _TOO_LARGE_MARKERS)


class LlamaCppClient:
    """Embeddings via `llama-server`. `model` est accepté puis ignoré : le
    serveur sert le modèle qu'on lui a passé en ligne de commande."""

    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._host = settings.LLAMACPP_HOST.rstrip("/")
        self._client = client or httpx.AsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def embed(self, text: str, model: str) -> list[float]:
        vectors = await self.embed_batch([text], model)
        return vectors[0]

    async def embed_batch(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        data = await self._post_embeddings(texts, model)
        rows = data.get("data")
        if not isinstance(rows, list) or len(rows) != len(texts):
            raise OllamaError(
                f"embed_batch : {len(rows) if isinstance(rows, list) else '?'} "
                f"vecteurs pour {len(texts)} textes"
            )
        # L'ordre n'est pas garanti par le contrat OpenAI : on trie sur `index`.
        ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
        vectors = [row["embedding"] for row in ordered]
        if any(not isinstance(v, list) or not v for v in vectors):
            raise OllamaError("embed_batch : vecteur vide ou malformé")
        return vectors

    async def _post_embeddings(self, texts: list[str], model: str) -> dict[str, Any]:
        """POST avec repli exponentiel sur erreurs transitoires (transport, 5xx).

        Pas d'équivalent du 400 passager d'Ollama : sans saut parent → runner,
        un 4xx de llama-server désigne une vraie erreur de requête.

        Cas particulier, l'entrée trop longue : on réessaie en tronquant au lieu
        de rejouer la même requête indéfiniment (voir SAFE_INPUT_CHARS)."""
        last_error: str | None = None
        last_exc: Exception | None = None
        sent = texts
        budget: int | None = None
        for attempt in range(RETRY_ATTEMPTS):
            try:
                resp = await self._client.post(
                    f"{self._host}/v1/embeddings",
                    json={"input": sent, "model": model},
                    timeout=EMBED_TIMEOUT_S,
                )
                if resp.status_code >= 400 and is_input_too_large(resp.text[:400]):
                    budget = SAFE_INPUT_CHARS if budget is None else budget // 2
                    if budget < MIN_INPUT_CHARS:
                        raise OllamaError(
                            f"entrée toujours refusée à {MIN_INPUT_CHARS} caractères : "
                            f"{resp.text[:200]}"
                        )
                    trop_longs = sum(1 for t in sent if len(t) > budget)
                    sent = [t[:budget] for t in sent]
                    # Journalisé : une troncature perd de l'information, il faut
                    # pouvoir le constater après coup plutôt que le découvrir
                    # dans des scores inexpliqués.
                    logger.warning(
                        "llamacpp_entree_tronquee",
                        budget_caracteres=budget,
                        textes_tronques=trop_longs,
                        sur=len(sent),
                    )
                    last_error = resp.text[:200]
                    continue  # réessai immédiat : inutile d'attendre, la cause est connue
                if 400 <= resp.status_code < 500:
                    raise OllamaError(
                        f"HTTP {resp.status_code} sur /v1/embeddings : {resp.text[:200]}"
                    )
                resp.raise_for_status()
                data: dict[str, Any] = resp.json()
                return data
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                last_error = str(exc)
            if attempt == RETRY_ATTEMPTS - 1:
                break
            delay = RETRY_BASE_DELAY_S * (2**attempt)
            logger.warning(
                "llamacpp_retry", attempt=attempt + 1, delay_s=delay, error=last_error
            )
            await asyncio.sleep(delay)
        raise OllamaError(
            f"échec après {RETRY_ATTEMPTS} tentatives sur /v1/embeddings : {last_error}"
        ) from last_exc

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self._host}/health", timeout=5)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def version_probe(self) -> str | None:
        """`GET /health` renvoie `{"status": "ok"}` quand le modèle est chargé,
        et 503 tant qu'il charge. Retourne None si OK, sinon le message."""
        try:
            resp = await self._client.get(f"{self._host}/health", timeout=5)
        except httpx.TimeoutException:
            return f"{_PROBE_TIMEOUT_PREFIX}5s sur /health ({self._host}) — serveur saturé"
        except httpx.HTTPError as exc:
            return f"/health injoignable ({self._host}) : {exc}"
        if resp.status_code == 503:
            # Chargement du modèle : passager, pas une panne.
            return f"{_PROBE_TIMEOUT_PREFIX}0s — modèle en chargement ({self._host})"
        if resp.status_code != 200:
            return f"/health HTTP {resp.status_code} ({self._host})"
        return None

    async def embed_probe(self, model: str) -> str | None:
        """Sonde réelle de /v1/embeddings (§Santé). None si l'embedding répond."""
        try:
            resp = await self._client.post(
                f"{self._host}/v1/embeddings",
                json={"input": ["ping"], "model": model},
                timeout=HEALTH_TIMEOUT_S,
            )
        except httpx.TimeoutException:
            return (
                f"{_PROBE_TIMEOUT_PREFIX}{HEALTH_TIMEOUT_S:g}s sur /v1/embeddings "
                f"({self._host}) — serveur en peine"
            )
        except httpx.HTTPError as exc:
            return f"/v1/embeddings injoignable ({self._host}) : {exc}"
        if resp.status_code != 200:
            return f"/v1/embeddings HTTP {resp.status_code} : {resp.text[:120]}"
        if not resp.json().get("data"):
            return f"/v1/embeddings a répondu sans vecteur ({self._host})"
        return None
