"""Effacement des échos antérieurs de l'assistant (axe 3, règle validée le 29/09).

L'embedder de test habituel hache le texte : deux phrases du même sujet y ont
une similarité quelconque, ce qui rend tout test de ciblage vide de sens.
`TopicEmbedder` projette chaque texte sur des directions thématiques : même
sujet → vecteurs quasi colinéaires, sujets distincts → quasi orthogonaux, et un
mélange de trois sujets donne un cosinus de 1/√3 ≈ 0,58 avec chacun — sous le
seuil de 0,65, ce qui permet de tester la frontière exactement.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from sqlalchemy import text

from mnemos.clock import FixedClock
from mnemos.config import Settings
from mnemos.models.base import make_async_engine
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
from mnemos.stores.episodic import (
    FORGET_ECHO_MAX,
    BatchEpisodeItem,
    EpisodicStore,
    ForgetRequest,
)

SUJETS = ["jazz", "aquarium", "garden", "cycling"]
DIM = 1024


class TopicEmbedder:
    """Une direction par sujet, plus une composante de fond commune et faible
    pour qu'aucun texte n'ait un vecteur nul."""

    async def embed(self, content: str) -> list[float]:
        v = [0.0] * DIM
        bas = content.lower()
        for i, sujet in enumerate(SUJETS):
            if sujet in bas:
                v[i] = 1.0
        v[DIM - 1] = 0.05
        return v

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


TENANT = "u1"


@pytest.fixture
async def store(tmp_path: Path, fixed_clock: FixedClock) -> AsyncIterator[EpisodicStore]:
    engine = make_async_engine(tmp_path / "episodic.db")
    async with engine.begin() as conn:
        for stmt in EPISODIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    yield EpisodicStore(engine, TopicEmbedder(),  # type: ignore[arg-type]
                        fixed_clock, Settings(_env_file=None, DATA_DIR=tmp_path))
    await engine.dispose()


def _item(role: str, content: str, tenant: str = TENANT) -> BatchEpisodeItem:
    return BatchEpisodeItem(content=content, role=role, session_id="s", tenant=tenant)


async def _contenus(store: EpisodicStore, tenant: str = TENANT) -> set[str]:
    async with store._sessions() as s:
        return {r[0] for r in (await s.execute(
            text("SELECT content FROM episodes WHERE tenant = :t"), {"t": tenant}))}


async def _indexes(store: EpisodicStore) -> int:
    async with store._sessions() as s:
        return int((await s.execute(text("SELECT count(*) FROM episodes_vec"))).scalar_one())


JAZZ = ForgetRequest(target="I love jazz", position=0)


async def test_echo_assistant_existant_efface(store: EpisodicStore) -> None:
    await store.write_batch([
        _item("user", "I love jazz, especially late at night."),
        _item("assistant", "Since you love jazz, try the Blue Note club."),
        _item("assistant", "For your garden, plant tomatoes in May."),
    ])
    assert await _indexes(store) == 3

    await store.write_batch([_item("user", "Thanks.")], forget=[JAZZ])

    restants = await _contenus(store)
    # L'écho de l'assistant disparaît, vecteur compris : un épisode effacé mais
    # encore indexé occuperait une place dans le KNN.
    assert "Since you love jazz, try the Blue Note club." not in restants
    assert await _indexes(store) == 3  # 3 - 1 effacé + 1 « Thanks. »
    # Le message de l'UTILISATEUR survit : au même seuil, ce sont ses questions.
    assert "I love jazz, especially late at night." in restants
    # Un autre sujet n'est pas touché.
    assert "For your garden, plant tomatoes in May." in restants


async def test_seuil_de_similarite(store: EpisodicStore) -> None:
    """Trois sujets mêlés : cosinus 1/√3 ≈ 0,58 avec « jazz », sous le seuil."""
    melange = "Mixing jazz, aquarium visits and garden work makes a nice weekend."
    await store.write_batch([_item("assistant", melange)])
    await store.write_batch([_item("user", "Ok.")], forget=[JAZZ])
    assert melange in await _contenus(store)
    assert 1 / math.sqrt(3) < 0.65  # la frontière testée est bien sous le seuil


async def test_plafond_par_consigne(store: EpisodicStore) -> None:
    await store.write_batch([_item("assistant", f"Jazz tip number {i}.") for i in range(7)])
    await store.write_batch([_item("user", "Ok.")], forget=[JAZZ])
    survivants = [c for c in await _contenus(store) if c.startswith("Jazz tip")]
    assert len(survivants) == 7 - FORGET_ECHO_MAX


async def test_echo_du_lot_courant_avant_la_consigne_non_insere(store: EpisodicStore) -> None:
    """Dans le lot, seul ce qui PRÉCÈDE la consigne est visé : ce qui la suit
    — une re-déclaration, une nouvelle réponse — survit."""
    lot = [
        _item("assistant", "Since you love jazz, here is a playlist."),  # avant : effacé
        _item("user", "Anything else?"),
        _item("assistant", "More jazz: try Coltrane's Blue Train."),     # après : gardé
    ]
    ecrits = await store.write_batch(lot, forget=[ForgetRequest("I love jazz", position=2)])
    contenus = await _contenus(store)
    assert "Since you love jazz, here is a playlist." not in contenus
    assert "More jazz: try Coltrane's Blue Train." in contenus
    assert "Anything else?" in contenus
    assert len(ecrits) == 2


async def test_lot_vide_efface_quand_meme_et_inscrit_la_requete(store: EpisodicStore) -> None:
    """Un lot réduit à une consigne et son accusé n'a rien à insérer, mais ses
    échos existants doivent disparaître, et la requête doit être inscrite."""
    await store.write_batch([_item("assistant", "Since you love jazz, see this gig.")])
    ecrits = await store.write_batch([], request_id="r9", forget=[JAZZ], tenant=TENANT)
    assert ecrits == []
    assert "Since you love jazz, see this gig." not in await _contenus(store)
    assert await store.has_request(TENANT, "r9")


async def test_isolation_par_tenant(store: EpisodicStore) -> None:
    await store.write_batch([_item("assistant", "Since you love jazz, enjoy!", tenant="u2")])
    await store.write_batch([_item("user", "Ok.")], forget=[JAZZ])
    assert "Since you love jazz, enjoy!" in await _contenus(store, "u2")


async def test_sans_consigne_rien_n_est_efface(store: EpisodicStore) -> None:
    await store.write_batch([_item("assistant", "Since you love jazz, try this.")])
    await store.write_batch([_item("user", "I love jazz too.")])
    assert "Since you love jazz, try this." in await _contenus(store)
