"""Oubli sur le chemin AML (catégorie D3) : la consigne et son accusé ne sont
jamais mémorisés, et rien d'autre n'est touché.

Un `/search` à top_k=100 sur quelques épisodes les renvoie tous : on lit donc
directement ce qui a été mémorisé.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from tests.conftest import make_stub_app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    app, engines = await make_stub_app(tmp_path)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            yield c
    for engine in engines:
        await engine.dispose()


def _msg(role: str, content: str, ts: int) -> dict[str, object]:
    return {"role": role, "content": content, "timestamp": 1_700_000_000_000 + ts * 1000}


async def _add(client: httpx.AsyncClient, rid: str, messages: list[dict[str, object]],
               user: str = "u1", session: str = "s1") -> None:
    r = await client.post("/add", json={
        "request_id": rid, "user_id": user, "session_id": session, "messages": messages,
    })
    assert r.status_code == 200 and r.json()["success"] is True


async def _memorise(client: httpx.AsyncClient, user: str = "u1") -> list[str]:
    r = await client.post("/search", json={"query": "jazz", "user_id": user, "top_k": 100})
    return [it["content"] for it in r.json()["data"]]


CONSIGNE = "Please forget that I love jazz."
ACCUSE = "Got it — I'll forget that you love jazz."


async def test_consigne_et_accuse_jamais_memorises(client: httpx.AsyncClient) -> None:
    await _add(client, "r1", [
        _msg("user", "I love jazz, especially Coltrane.", 1),
        _msg("assistant", "Great taste!", 2),
        _msg("user", CONSIGNE, 3),
        _msg("assistant", ACCUSE, 4),
        _msg("user", "What should I do tonight?", 5),
    ])
    contenus = await _memorise(client)
    assert CONSIGNE not in contenus
    assert ACCUSE not in contenus
    # Règle étroite : aucun souvenir EXISTANT n'est touché, pas même l'énoncé d'origine.
    assert "I love jazz, especially Coltrane." in contenus
    assert "What should I do tonight?" in contenus
    assert "Great taste!" in contenus


async def test_accuse_dans_le_lot_suivant(client: httpx.AsyncClient) -> None:
    """La plateforme découpe par lots de 20 : la consigne peut clore un lot et
    son accusé ouvrir le suivant. Il doit disparaître quand même."""
    await _add(client, "r1", [_msg("user", "Hi there.", 1), _msg("user", CONSIGNE, 2)])
    await _add(client, "r2", [_msg("assistant", ACCUSE, 3), _msg("user", "Thanks.", 4)])
    contenus = await _memorise(client)
    assert CONSIGNE not in contenus and ACCUSE not in contenus
    assert "Thanks." in contenus


async def test_attente_d_accuse_isolee_par_session_et_par_user(client: httpx.AsyncClient) -> None:
    """Une consigne en fin de lot dans une session ne doit pas faire disparaître
    le premier message assistant d'une autre session, ni d'un autre utilisateur."""
    await _add(client, "r1", [_msg("user", CONSIGNE, 1)], user="u1", session="s1")
    await _add(client, "r2", [_msg("assistant", "Autre session.", 2)], user="u1", session="s2")
    await _add(client, "r3", [_msg("assistant", "Autre user.", 3)], user="u2", session="s1")
    assert "Autre session." in await _memorise(client, "u1")
    assert "Autre user." in await _memorise(client, "u2")


async def test_lot_suivant_qui_commence_par_l_utilisateur(client: httpx.AsyncClient) -> None:
    """Pas d'accusé : l'attente s'éteint, rien d'autre n'est écarté."""
    await _add(client, "r1", [_msg("user", CONSIGNE, 1)])
    await _add(client, "r2", [_msg("user", "Anyway.", 2), _msg("assistant", "Sure!", 3)])
    contenus = await _memorise(client)
    assert "Anyway." in contenus and "Sure!" in contenus


async def test_lot_entierement_oublie_est_idempotent(client: httpx.AsyncClient) -> None:
    """Un lot qui ne contient qu'une consigne et son accusé n'écrit rien, mais
    sa requête est inscrite : un rejeu ne doit pas réarmer l'attente d'accusé,
    sinon le prochain message assistant disparaîtrait à tort."""
    # Le lot se termine sur une consigne SANS accusé : l'attente est armée, puis
    # consommée par le lot r2. Un rejeu tardif de r1 ne doit pas la réarmer.
    await _add(client, "r1", [_msg("user", CONSIGNE, 1)])
    await _add(client, "r2", [_msg("assistant", ACCUSE, 2)])
    await _add(client, "r1", [_msg("user", CONSIGNE, 1)])
    await _add(client, "r3", [_msg("assistant", "Useful advice about jazz clubs.", 3)])
    contenus = await _memorise(client)
    assert ACCUSE not in contenus
    assert "Useful advice about jazz clubs." in contenus


async def test_piege_n_est_pas_oublie(client: httpx.AsyncClient) -> None:
    """« Don't forget » est un rappel : il doit être mémorisé, et le message
    assistant qui suit aussi."""
    await _add(client, "r1", [
        _msg("user", "Don't forget that I have a jazz concert on Friday.", 1),
        _msg("assistant", "Noted, the concert is on Friday.", 2),
    ])
    contenus = await _memorise(client)
    assert "Don't forget that I have a jazz concert on Friday." in contenus
    assert "Noted, the concert is on Friday." in contenus
