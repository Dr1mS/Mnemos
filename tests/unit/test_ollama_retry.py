"""Classement des erreurs Ollama : transitoire (réessai) vs définitive (échec).

Contexte (21/09) : sous concurrence, Ollama démarre des runners supplémentaires
pour un modèle d'embedding et répond HTTP 400 pendant leur montée. Traité comme
un 4xx définitif, cela produisait 9 % de HTTP 500 sur /add et des 503 sur
/health. Ces tests figent la distinction.
"""

from __future__ import annotations

import httpx
import pytest

from mnemos.config import Settings
from mnemos.llm.ollama_client import (
    OllamaClient,
    OllamaError,
    is_probe_transient,
    is_transient_upstream_error,
)

RUNNER_DOWN = (
    '{"error":"Post \\"http://127.0.0.1:58768/tokenize\\": dial tcp 127.0.0.1:58768: '
    'connectex: Une tentative de connexion a échoué."}'
)
HEALTH_RESP = (
    '{"error":"health resp: Get \\"http://127.0.0.1:22010/health\\": '
    'dial tcp 127.0.0.1:22010: connection refused"}'
)
MODEL_ABSENT = '{"error":"model \\"bge-m3\\" not found, try pulling it first"}'


def client_with(handler: object, monkeypatch: pytest.MonkeyPatch) -> OllamaClient:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    transport = httpx.MockTransport(handler)  # type: ignore[arg-type]
    return OllamaClient(settings, httpx.AsyncClient(transport=transport))


@pytest.mark.parametrize("body", [RUNNER_DOWN, HEALTH_RESP])
def test_marqueurs_transitoires_reconnus(body: str) -> None:
    assert is_transient_upstream_error(body)
    assert is_probe_transient(body)


def test_modele_absent_reste_definitif() -> None:
    """Un vrai 4xx ne doit pas être réessayé : échec immédiat et lisible."""
    assert not is_transient_upstream_error(MODEL_ABSENT)
    assert not is_probe_transient(MODEL_ABSENT)


async def test_400_transitoire_est_reessaye(monkeypatch: pytest.MonkeyPatch) -> None:
    """Le runner amont monte : le 1er appel échoue, le 2e réussit."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(400, text=RUNNER_DOWN)
        return httpx.Response(200, json={"embeddings": [[0.5] * 1024]})

    monkeypatch.setattr("mnemos.llm.ollama_client.UPSTREAM_RETRY_BASE_DELAY_S", 0.0)
    client = client_with(handler, monkeypatch)
    vectors = await client.embed_batch(["bonjour"], "bge-m3")
    assert len(vectors) == 1
    assert calls["n"] == 2
    await client.aclose()


async def test_400_definitif_echoue_sans_reessai(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, text=MODEL_ABSENT)

    client = client_with(handler, monkeypatch)
    with pytest.raises(OllamaError, match="not found"):
        await client.embed_batch(["bonjour"], "bge-m3")
    assert calls["n"] == 1  # aucun réessai
    await client.aclose()
