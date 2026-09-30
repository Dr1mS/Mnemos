"""Oubli sur le chemin AML (catégorie D3), variante retenue le 30/09/2026.

La consigne d'oubli et l'accusé de réception de l'assistant sont MÉMORISÉS : pour
le répondeur, ce sont eux qui disent quoi éviter. Seuls les échos ANTÉRIEURS de
l'assistant, qui affirmaient la préférence, sont effacés. Supprimer aussi la
consigne et l'accusé (première version, f412fda) faisait tomber le score d'oubli
de Nemotron 3 Ultra de 47,4 % à 15,5 % (bench/bench_forget_variants.py).

L'embedder de cette route hache le texte : seul un texte IDENTIQUE à la cible
garantit un cosinus de 1. Ce sont des tests de câblage ; le ciblage lui-même est
testé avec un embedder thématique dans test_forget_echoes.py.

On lit la BASE, pas /search : la route déduplique les contenus identiques, si
bien qu'un écho de l'assistant identique au message de l'utilisateur y serait
invisible — et un test fondé sur /search passerait même sans aucune suppression
(constaté par mutation le 30/09).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text
from tests.conftest import make_stub_app


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    app, engines = await make_stub_app(tmp_path)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            c.app_state = app.state  # type: ignore[attr-defined]
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


async def _memorise(client: httpx.AsyncClient, user: str = "u1") -> list[tuple[str, str]]:
    """(rôle, contenu) de chaque épisode stocké pour cet utilisateur."""
    store = client.app_state.store  # type: ignore[attr-defined]
    async with store._sessions() as s:
        rows = await s.execute(
            text("SELECT role, content FROM episodes WHERE tenant = :t ORDER BY created_at"),
            {"t": user})
        return [(r[0], r[1]) for r in rows]


CONSIGNE = "Please forget that I love jazz."
ACCUSE = "Got it — I'll forget that you love jazz."
ECHO = "I love jazz"  # texte identique à la cible : cosinus 1 avec l'embedder haché


async def test_consigne_et_accuse_sont_memorises(client: httpx.AsyncClient) -> None:
    """Le cœur de la variante retenue : garder ce qui NIE la préférence."""
    await _add(client, "r1", [_msg("user", CONSIGNE, 1), _msg("assistant", ACCUSE, 2)])
    stockes = await _memorise(client)
    assert ("user", CONSIGNE) in stockes
    assert ("assistant", ACCUSE) in stockes


async def test_echo_anterieur_de_l_assistant_efface(client: httpx.AsyncClient) -> None:
    await _add(client, "r1", [
        _msg("user", ECHO, 1),            # même texte, mais UTILISATEUR : gardé
        _msg("assistant", ECHO, 2),       # écho de l'assistant : effacé
    ])
    await _add(client, "r2", [_msg("user", CONSIGNE, 3), _msg("assistant", ACCUSE, 4)])
    stockes = await _memorise(client)
    assert ("user", ECHO) in stockes      # la parole de l'utilisateur est gardée
    assert ("assistant", ECHO) not in stockes
    assert ("user", CONSIGNE) in stockes and ("assistant", ACCUSE) in stockes


async def test_echo_du_meme_lot_avant_la_consigne_non_insere(client: httpx.AsyncClient) -> None:
    """La position de la consigne dans le lot borne ce qui la précède."""
    await _add(client, "r1", [
        _msg("assistant", ECHO, 1),       # avant la consigne : non inséré
        _msg("user", CONSIGNE, 2),
        _msg("assistant", ACCUSE, 3),
        _msg("assistant", ECHO, 4),       # après : histoire nouvelle, gardé
    ])
    stockes = await _memorise(client)
    assert stockes.count(("assistant", ECHO)) == 1  # seul celui d'après la consigne
    assert stockes[-1] == ("assistant", ECHO)
    assert ("user", CONSIGNE) in stockes and ("assistant", ACCUSE) in stockes


async def test_rien_n_est_efface_sans_consigne(client: httpx.AsyncClient) -> None:
    """« Don't forget » est un rappel : aucun écho ne doit disparaître."""
    await _add(client, "r1", [_msg("assistant", ECHO, 1)])
    await _add(client, "r2", [_msg("user", "Don't forget that I love jazz.", 2)])
    assert ("assistant", ECHO) in await _memorise(client)


async def test_rejeu_d_un_lot_avec_consigne_idempotent(client: httpx.AsyncClient) -> None:
    await _add(client, "r1", [_msg("assistant", ECHO, 1)])
    lot = [_msg("user", CONSIGNE, 2), _msg("assistant", ACCUSE, 3)]
    await _add(client, "r2", lot)
    await _add(client, "r2", lot)         # rejeu de la plateforme : aucun doublon
    stockes = await _memorise(client)
    assert stockes.count(("user", CONSIGNE)) == 1 and stockes.count(("assistant", ACCUSE)) == 1
    assert ("assistant", ECHO) not in stockes


async def test_isolation_entre_utilisateurs(client: httpx.AsyncClient) -> None:
    await _add(client, "r1", [_msg("assistant", ECHO, 1)], user="u2")
    await _add(client, "r2", [_msg("user", CONSIGNE, 2)], user="u1")
    assert ("assistant", ECHO) in await _memorise(client, "u2")
