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
RETRY_ATTEMPTS = 4
RETRY_BASE_DELAY_S = 0.5  # transport : 0,5 + 1 + 2 = 3,5 s
# Démarrage d'un runner Ollama supplémentaire : ~7 s mesurées le 21/09. Le
# backoff amont doit couvrir cette fenêtre, sinon les réessais échouent aussi.
UPSTREAM_RETRY_BASE_DELAY_S = 1.5  # 1,5 + 3 + 6 = 10,5 s
HEALTH_TIMEOUT_S = 2.0  # sonde /health : court, appelé à chaque tick Atelios
_PROBE_TIMEOUT_PREFIX = "timeout > "

# Ollama force `-np 1` pour les modèles d'embedding : un seul /api/embed à la
# fois par runner. Sous concurrence il démarre des runners supplémentaires, et
# pendant leur montée (~7 s) il répond **HTTP 400** dont le corps porte l'erreur
# réseau interne vers le runner pas encore joignable. C'est transitoire, pas une
# requête invalide. Mesuré le 21/09 : 30 échecs sur 120 embeddings au palier de
# montée en charge, puis 0 erreur à 49 et 72 req/s une fois les runners chauds.
# Un vrai 4xx (modèle absent, corps malformé) ne porte aucun de ces marqueurs et
# doit continuer d'échouer immédiatement.
_TRANSIENT_UPSTREAM_MARKERS = (
    "dial tcp",
    "connection refused",
    "health resp",
    "connectex",  # message Winsock localisé, côté Windows
)


def is_probe_timeout(probe_error: str) -> bool:
    """Vrai si embed_probe a échoué par dépassement de délai (charge ou cold
    start) plutôt que par une panne franche (process mort, modèle absent)."""
    return probe_error.startswith(_PROBE_TIMEOUT_PREFIX)


def is_transient_upstream_error(body: str) -> bool:
    """Vrai si un 4xx d'Ollama décrit un runner interne pas encore joignable
    (voir _TRANSIENT_UPSTREAM_MARKERS) plutôt qu'une requête invalide."""
    lowered = body.lower()
    return any(marker in lowered for marker in _TRANSIENT_UPSTREAM_MARKERS)


def is_probe_transient(probe_error: str) -> bool:
    """Vrai si l'échec de sonde est passager — délai dépassé ou runner amont en
    cours de démarrage. /health doit alors répondre « degraded » (2xx) et non
    « unhealthy » : la plateforme AML lit tout non-2xx comme un service tombé."""
    return is_probe_timeout(probe_error) or is_transient_upstream_error(probe_error)


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
        last_error: str | None = None
        for attempt in range(RETRY_ATTEMPTS):
            base_delay = RETRY_BASE_DELAY_S
            try:
                resp = await self._client.post(
                    f"{self._host}{path}", json=payload, timeout=timeout_s
                )
                if 400 <= resp.status_code < 500:
                    body = resp.text[:200]
                    if not is_transient_upstream_error(body):
                        # 4xx : erreur de requête, retry inutile (§7.1)
                        raise OllamaError(f"HTTP {resp.status_code} sur {path} : {body}")
                    # Runner amont en cours de démarrage : réessai à cadence lente.
                    last_error = f"HTTP {resp.status_code} sur {path} : {body}"
                    base_delay = UPSTREAM_RETRY_BASE_DELAY_S
                else:
                    resp.raise_for_status()
                    data: dict[str, Any] = resp.json()
                    return data
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                last_error = str(exc)
            if attempt == RETRY_ATTEMPTS - 1:
                break
            delay = base_delay * (2**attempt)
            logger.warning(
                "ollama_retry", path=path, attempt=attempt + 1, delay_s=delay, error=last_error
            )
            await asyncio.sleep(delay)
        raise OllamaError(
            f"échec après {RETRY_ATTEMPTS} tentatives sur {path} : {last_error}"
        ) from last_exc

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

    async def version_probe(self) -> str | None:
        """Comme health_check, mais dit *pourquoi* c'est tombé.

        Un délai dépassé signifie un Ollama saturé qui répond encore au reste :
        c'est passager. Une connexion refusée signifie un Ollama absent. Les
        confondre faisait répondre 503 à /health pendant une simple pointe de
        charge (mesuré le 21/09), ce que la plateforme AML lit comme un service
        tombé. Retourne None si OK, sinon le message."""
        try:
            resp = await self._client.get(f"{self._host}/api/version", timeout=5)
        except httpx.TimeoutException:
            return f"{_PROBE_TIMEOUT_PREFIX}5s sur /api/version ({self._host}) — endpoint saturé"
        except httpx.HTTPError as exc:
            return f"/api/version injoignable ({self._host}) : {exc}"
        if resp.status_code != 200:
            return f"/api/version HTTP {resp.status_code} ({self._host})"
        return None

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
                f"{_PROBE_TIMEOUT_PREFIX}{HEALTH_TIMEOUT_S:g}s sur /api/embed ({self._host}) "
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
