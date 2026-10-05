"""Dates relatives sur la route AML : stockées résolues, indexées brutes.

Le souvenir stocké porte la date résolue (« yesterday (27 August 2023) ») ; la
recherche le rend tel quel, sans rien y ajouter (décision de l'utilisateur du
04/10). L'index — embedding dense et bits lexicaux — reste celui du message brut,
pour que le classement mesuré et le seuil de l'oubli restent valables.
"""

from __future__ import annotations

import struct
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from sqlalchemy import text
from tests.conftest import make_stub_app

from mnemos.config import Settings
from mnemos.embeddings.sparse import sparse_encode

LUNDI = int(datetime(2023, 8, 28, 15, 19, tzinfo=UTC).timestamp() * 1000)
BRUT = "I took my kids to a park yesterday."
ANNOTE = "I took my kids to a park yesterday (27 August 2023)."


def _settings(tmp_path: Path, annoter: bool) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        DATA_DIR=tmp_path, EPISODIC_DB=tmp_path / "episodic.db",
        SEMANTIC_DB=tmp_path / "semantic.db", PROCEDURAL_DIR=tmp_path / "procedural",
        RELATIVE_DATES_ANNOTATION=annoter,
    )


async def _client(tmp_path: Path, annoter: bool) -> AsyncIterator[httpx.AsyncClient]:
    app, engines = await make_stub_app(tmp_path, settings=_settings(tmp_path, annoter))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as c:
            c.app_state = app.state  # type: ignore[attr-defined]
            yield c
    for engine in engines:
        await engine.dispose()


@pytest.fixture
async def annote(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    async for c in _client(tmp_path, annoter=True):
        yield c


@pytest.fixture
async def brut(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    async for c in _client(tmp_path, annoter=False):
        yield c


async def _add(client: httpx.AsyncClient, rid: str, messages: list[dict[str, object]]) -> None:
    r = await client.post("/add", json={"request_id": rid, "user_id": "u1",
                                        "session_id": "s1", "messages": messages})
    assert r.status_code == 200 and r.json()["success"] is True


async def _stocke(client: httpx.AsyncClient) -> list[tuple[str, str, int]]:
    store = client.app_state.store  # type: ignore[attr-defined]
    async with store._sessions() as s:
        rows = await s.execute(text(
            "SELECT id, content, created_at FROM episodes WHERE tenant = 'u1' ORDER BY created_at"))
        return [(r[0], r[1], r[2]) for r in rows]


async def test_stocke_resolu_et_rendu_tel_quel(annote: httpx.AsyncClient) -> None:
    await _add(annote, "r1", [{"role": "user", "content": BRUT, "timestamp": LUNDI}])
    assert [c for _, c, _ in await _stocke(annote)] == [ANNOTE]
    r = await annote.post("/search", json={"query": "When did I go to the park?",
                                           "user_id": "u1", "top_k": 5})
    assert r.json()["data"][0]["content"] == ANNOTE


async def test_sans_horodatage_de_la_source_rien_n_est_ajoute(annote: httpx.AsyncClient) -> None:
    """Sans horodatage, l'ancre serait l'heure de réception : date fausse."""
    await _add(annote, "r1", [{"role": "user", "content": BRUT}])
    assert [c for _, c, _ in await _stocke(annote)] == [BRUT]


async def test_reglage_desactive_par_defaut(brut: httpx.AsyncClient) -> None:
    assert Settings(_env_file=None).RELATIVE_DATES_ANNOTATION is False  # type: ignore[call-arg]
    await _add(brut, "r1", [{"role": "user", "content": BRUT, "timestamp": LUNDI}])
    assert [c for _, c, _ in await _stocke(brut)] == [BRUT]


async def test_index_construit_sur_le_texte_brut(annote: httpx.AsyncClient) -> None:
    await _add(annote, "r1", [{"role": "user", "content": BRUT, "timestamp": LUNDI}])
    (ep_id, contenu, cree), = await _stocke(annote)
    assert contenu == ANNOTE
    store = annote.app_state.store  # type: ignore[attr-defined]
    async with store._sessions() as s:
        bits = (await s.execute(text("SELECT sparse_bits FROM episodes_sparse WHERE episode_id = :i"),
                                {"i": ep_id})).scalar_one()
        blob = (await s.execute(text("SELECT embedding FROM episodes_vec WHERE episode_id = :i"),
                                {"i": ep_id})).scalar_one()
    assert bytes(bits) == sparse_encode(BRUT, cree)
    assert bytes(bits) != sparse_encode(ANNOTE, cree)
    stocke = struct.unpack(f"{len(blob) // 4}f", blob)
    attendu = (await store._embedder.embed_batch([BRUT]))[0]
    assert stocke == pytest.approx(attendu, abs=1e-6)


async def test_oubli_cible_toujours_le_texte_brut(annote: httpx.AsyncClient) -> None:
    """L'écho de l'assistant contient « yesterday » : s'il était indexé annoté, son
    cosinus avec la cible tomberait et l'oubli ne l'effacerait plus."""
    echo = "I went to the park yesterday"
    await _add(annote, "r1", [{"role": "assistant", "content": echo, "timestamp": LUNDI}])
    await _add(annote, "r2", [{"role": "user", "content": f"Please forget that {echo}.",
                               "timestamp": LUNDI + 60_000}])
    contenus = [c for _, c, _ in await _stocke(annote)]
    assert not any(c.startswith(echo) for c in contenus)
    assert any(c.startswith("Please forget that") for c in contenus)
