"""Client d'embeddings llama.cpp — contrat, ordre des vecteurs, erreurs.

Le point délicat : l'API compatible OpenAI ne garantit pas l'ordre de `data`.
Rendre les vecteurs mélangés associerait chaque message au mauvais embedding —
une corruption silencieuse, invisible des tests de bout en bout.
"""

from __future__ import annotations

import httpx
import pytest

from mnemos.config import Settings
from mnemos.llm.llamacpp_client import LlamaCppClient
from mnemos.llm.ollama_client import OllamaError


def make_client(handler: object) -> LlamaCppClient:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return LlamaCppClient(settings, httpx.AsyncClient(transport=transport))


def embeddings_response(vectors: list[list[float]], *, shuffled: bool = False) -> httpx.Response:
    rows = [{"index": i, "embedding": v} for i, v in enumerate(vectors)]
    if shuffled:
        rows = list(reversed(rows))
    return httpx.Response(200, json={"data": rows, "model": "bge-m3"})


async def test_embed_batch_respecte_l_ordre_demande() -> None:
    """`data` revient dans le désordre : on doit trier sur `index`."""
    wanted = [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]

    def handler(request: httpx.Request) -> httpx.Response:
        return embeddings_response(wanted, shuffled=True)

    client = make_client(handler)
    got = await client.embed_batch(["a", "b", "c"], "bge-m3")
    assert got == wanted, "un vecteur associé au mauvais texte = corruption silencieuse"
    await client.aclose()


async def test_embed_unitaire() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return embeddings_response([[0.5] * 1024])

    client = make_client(handler)
    assert len(await client.embed("bonjour", "bge-m3")) == 1024
    await client.aclose()


async def test_liste_vide_sans_appel_reseau() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("aucun appel ne doit partir pour une liste vide")

    client = make_client(handler)
    assert await client.embed_batch([], "bge-m3") == []
    await client.aclose()


async def test_nombre_de_vecteurs_incoherent_echoue() -> None:
    """Deux textes, un seul vecteur : refuser plutôt que décaler les données."""

    def handler(request: httpx.Request) -> httpx.Response:
        return embeddings_response([[1.0, 0.0]])

    client = make_client(handler)
    with pytest.raises(OllamaError, match="1 vecteurs pour 2 textes"):
        await client.embed_batch(["a", "b"], "bge-m3")
    await client.aclose()


async def test_4xx_echoue_sans_reessai() -> None:
    """Sans saut parent → runner, un 4xx est une vraie erreur de requête."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text='{"error":"invalid input"}')

    client = make_client(handler)
    with pytest.raises(OllamaError, match="400"):
        await client.embed_batch(["a"], "bge-m3")
    assert calls["n"] == 1
    await client.aclose()


async def test_health_503_est_passager(monkeypatch: pytest.MonkeyPatch) -> None:
    """503 = modèle en chargement : /health doit le lire comme « degraded »."""
    from mnemos.llm.ollama_client import is_probe_transient

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"status": "loading model"})

    client = make_client(handler)
    error = await client.version_probe()
    assert error is not None and is_probe_transient(error)
    await client.aclose()


async def test_health_ok() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    client = make_client(handler)
    assert await client.version_probe() is None
    assert await client.health_check() is True
    await client.aclose()
