"""Tests EpisodicStore sans Ollama — stub embedder déterministe, DB tmp.

Couvre : write/get, filtres de search, decay sans double-comptage (Clock
mocké, §19 Phase 2), règles d'archivage §9.2, hooks de consolidation.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from hashlib import blake2b
from pathlib import Path

import pytest
from sqlalchemy import text

from mnemos.clock import FixedClock
from mnemos.config import Settings
from mnemos.models.base import make_async_engine
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
from mnemos.stores.episodic import (
    DAY_MS,
    DENSE_WEIGHT,
    RECENCY_WEIGHT,
    SPARSE_WEIGHT,
    BatchEpisodeItem,
    EpisodicStore,
)
from mnemos.tagger.salience import SalienceScores


class StubEmbedder:
    """Vecteur 1024-dim déterministe dérivé du hash du texte — pas d'Ollama."""

    async def embed(self, content: str) -> list[float]:
        seed = blake2b(content.encode(), digest_size=8).digest()
        base = [(b / 255.0) - 0.5 for b in seed]
        return (base * 128)[:1024]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


def scores(combined: float, self_ref: float = 0.5) -> SalienceScores:
    return SalienceScores(
        surprise=0.5, arousal=0.3, self_ref=self_ref, recurrence=0.1, combined=combined
    )


@pytest.fixture
async def store(tmp_path: Path, fixed_clock: FixedClock) -> AsyncIterator[EpisodicStore]:
    engine = make_async_engine(tmp_path / "episodic.db")
    async with engine.begin() as conn:
        for stmt in EPISODIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    yield EpisodicStore(
        engine,
        StubEmbedder(),  # type: ignore[arg-type]
        fixed_clock,
        Settings(_env_file=None, DATA_DIR=tmp_path),
    )
    await engine.dispose()


async def test_write_puis_get(store: EpisodicStore, fixed_clock: FixedClock) -> None:
    ep = await store.write("Alice bosse chez Datalyse", role="user", session_id="s1")
    got = await store.get_by_id(ep.id)
    assert got is not None
    assert got.content == "Alice bosse chez Datalyse"
    assert got.created_at == fixed_clock.now_ms()
    assert got.salience == 0.5  # défaut avant scoring async (§13.3)
    assert got.decay_state == 1.0


async def test_write_avec_salience(store: EpisodicStore) -> None:
    ep = await store.write("big news", role="user", salience_scores=scores(0.9, self_ref=1.0))
    got = await store.get_by_id(ep.id)
    assert got is not None and got.salience == 0.9 and got.self_ref == 1.0


async def test_search_retrouve_et_filtre(store: EpisodicStore) -> None:
    await store.write("le thé vert japonais", role="user", session_id="s1")
    await store.write("réunion projet demain", role="user", session_id="s2")
    # même contenu → même vecteur stub → dense_sim max
    results = await store.search("le thé vert japonais", k=5)
    assert results[0].episode.content == "le thé vert japonais"
    # filtre session
    only_s2 = await store.search("le thé vert japonais", k=5, session_id="s2")
    assert all(r.episode.session_id == "s2" for r in only_s2)


async def test_search_exclut_archives(store: EpisodicStore) -> None:
    ep = await store.write("souvenir périmé", role="user")
    await store.mark_consolidated(ep.id)
    # force decay_state sous 0.1 puis archive (règle 1)
    async with store._sessions() as session, session.begin():
        await session.execute(
            text("UPDATE episodes SET decay_state = 0.05 WHERE id = :id"), {"id": ep.id}
        )
    await store.archive_old()
    assert await store.search("souvenir périmé", k=5) == []


async def test_decay_pas_de_double_comptage(
    store: EpisodicStore, fixed_clock: FixedClock
) -> None:
    """Deux runs rapprochés : le second ne re-soustrait pas l'âge complet (§9.2)."""
    ep = await store.write("épisode neutre", role="user")  # salience 0.5
    fixed_clock.advance(10 * DAY_MS)
    await store.apply_decay()
    got1 = await store.get_by_id(ep.id)
    assert got1 is not None
    # 1.0 - 0.05 * 10 * (2 - 0.5) = 0.25
    assert got1.decay_state == pytest.approx(0.25)
    # second run immédiat : elapsed = 0 → aucun changement
    await store.apply_decay()
    got2 = await store.get_by_id(ep.id)
    assert got2 is not None
    assert got2.decay_state == pytest.approx(0.25)


async def test_decay_module_par_salience(store: EpisodicStore, fixed_clock: FixedClock) -> None:
    """Plus saillant → décroît plus lentement (§9.2)."""
    low = await store.write("bruit", role="user", salience_scores=scores(0.1))
    high = await store.write("important", role="user", salience_scores=scores(0.9))
    fixed_clock.advance(5 * DAY_MS)
    await store.apply_decay()
    got_low, got_high = await store.get_by_id(low.id), await store.get_by_id(high.id)
    assert got_low is not None and got_high is not None
    assert got_low.decay_state < got_high.decay_state


async def test_decay_clamp_zero(store: EpisodicStore, fixed_clock: FixedClock) -> None:
    await store.write("très vieux", role="user", salience_scores=scores(0.0))
    fixed_clock.advance(400 * DAY_MS)
    await store.apply_decay()
    results = await store.search("très vieux", k=5)
    assert all(r.episode.decay_state >= 0.0 for r in results)


async def test_archive_regle_1_decru_et_consolide(
    store: EpisodicStore, fixed_clock: FixedClock
) -> None:
    ep = await store.write("consolidé puis décru", role="user", salience_scores=scores(0.5))
    await store.mark_consolidated(ep.id)
    fixed_clock.advance(30 * DAY_MS)  # 1 - 0.05*30*1.5 = -1.25 → clamp 0 < 0.1
    await store.apply_decay()
    report = await store.archive_old()
    assert report.decayed_consolidated == 1
    assert report.expired_unconsolidated == 0


async def test_archive_regle_2_expire_jamais_candidat(
    store: EpisodicStore, fixed_clock: FixedClock
) -> None:
    """Épisode ancien, salience < seuil, jamais consolidé → archivé (règle 2)."""
    await store.write("bruit ancien", role="user", salience_scores=scores(0.3))
    fixed_clock.advance(91 * DAY_MS)  # > EPISODIC_RETENTION_DAYS=90
    report = await store.archive_old()
    assert report.expired_unconsolidated == 1


async def test_archive_ne_touche_pas_les_candidats_consolidation(
    store: EpisodicStore, fixed_clock: FixedClock
) -> None:
    """Salience haute non consolidée : jamais archivée par la règle 2."""
    await store.write("important non consolidé", role="user", salience_scores=scores(0.9))
    fixed_clock.advance(91 * DAY_MS)
    report = await store.archive_old()
    assert report.expired_unconsolidated == 0


async def test_pending_consolidation(store: EpisodicStore, fixed_clock: FixedClock) -> None:
    old = await store.write("saillant et mûr", role="user", salience_scores=scores(0.8))
    flat = await store.write("pas assez saillant", role="user", salience_scores=scores(0.4))
    fixed_clock.advance(2 * 3_600_000)  # les deux premiers ont maintenant 2h
    recent = await store.write("saillant mais trop récent", role="user",
                               salience_scores=scores(0.9))
    pending = await store.list_pending_consolidation(
        min_salience=0.6, min_age_hours=1, limit=10
    )
    ids = {e.id for e in pending}
    assert old.id in ids  # mûr ET saillant
    assert flat.id not in ids  # sous le seuil de salience
    assert recent.id not in ids  # sous le délai


async def test_mark_consolidated_extraction_failed(store: EpisodicStore) -> None:
    ep = await store.write("échec extraction", role="user")
    await store.mark_consolidated(ep.id, extraction_failed=True)
    got = await store.get_by_id(ep.id)
    assert got is not None
    assert got.consolidated_at is not None
    assert got.extraction_failed == 1


async def test_set_entity_refs(store: EpisodicStore) -> None:
    ep = await store.write("Tom bosse chez Airbus", role="user")
    await store.set_entity_refs(ep.id, ["Tom", "Airbus"])
    got = await store.get_by_id(ep.id)
    assert got is not None
    assert got.entity_refs == '["Tom", "Airbus"]'


async def test_dump_archived_jsonl_puis_delete(
    store: EpisodicStore, fixed_clock: FixedClock, tmp_path: Path
) -> None:
    """§9.2 : dump JSONL des archivés puis DELETE des 3 tables."""
    import json as _json

    from sqlalchemy import text as _sql

    ep = await store.write("souvenir à archiver", role="user", salience_scores=scores(0.3))
    keep = await store.write("souvenir gardé", role="user", salience_scores=scores(0.9))
    fixed_clock.advance(91 * DAY_MS)
    await store.archive_old()  # règle 2 archive le premier

    report = await store.dump_archived()
    assert report.dumped == 1
    assert report.path is not None
    dump_text = Path(report.path).read_text()  # noqa: ASYNC240 — test, I/O local ok
    lines = [_json.loads(line) for line in dump_text.splitlines()]
    assert lines[0]["id"] == ep.id
    assert lines[0]["content"] == "souvenir à archiver"

    # Purge complète des 3 tables pour l'archivé, l'autre intact
    assert await store.get_by_id(ep.id) is None
    assert await store.get_by_id(keep.id) is not None
    async with store._sessions() as session:
        n_sparse = (
            await session.execute(_sql("SELECT COUNT(*) FROM episodes_sparse"))
        ).scalar_one()
        n_vec = (
            await session.execute(_sql("SELECT COUNT(*) FROM episodes_vec"))
        ).scalar_one()
    assert n_sparse == 1 and n_vec == 1

    # Re-dump : plus rien
    assert (await store.dump_archived()).dumped == 0


async def test_update_salience_async(store: EpisodicStore) -> None:
    ep = await store.write("scoring différé", role="user")
    await store.update_salience(ep.id, scores(0.85, self_ref=0.9))
    got = await store.get_by_id(ep.id)
    assert got is not None
    assert got.salience == 0.85
    assert got.self_ref == 0.9


async def test_write_batch(store: EpisodicStore, fixed_clock: FixedClock) -> None:
    items = [
        BatchEpisodeItem(
            content=f"Message {i} de la session de test",
            role="user" if i % 2 == 0 else "assistant",
            session_id="s_batch",
            created_at=fixed_clock.now_ms() + (i * 1000),
        )
        for i in range(5)
    ]
    episodes = await store.write_batch(items)
    assert len(episodes) == 5

    for i, ep in enumerate(episodes):
        assert ep.content == f"Message {i} de la session de test"
        assert ep.role == ("user" if i % 2 == 0 else "assistant")
        assert ep.created_at == fixed_clock.now_ms() + (i * 1000)
        got = await store.get_by_id(ep.id)
        assert got is not None
        assert got.content == ep.content

    # Recherche vectorielle et hybride sur un élément du lot
    results = await store.search("Message 3", k=5, session_id="s_batch")
    assert any("Message 3" in r.episode.content for r in results)


async def test_write_batch_empty(store: EpisodicStore) -> None:
    episodes = await store.write_batch([])
    assert episodes == []


# ── Récence : référence = le plus récent des candidats, pas l'heure au mur ──

# Deux ans avant l'horloge de test (2026-07-02). Les données d'évaluation AML
# portent leurs dates d'origine : c'est le cas nominal, pas un cas limite.
_ANCIEN_MS = 1_688_299_200_000  # 2023-07-02
_RECENT_MS = _ANCIEN_MS + 60 * DAY_MS  # 2023-08-31


async def test_recence_active_sur_corpus_rejoue(store: EpisodicStore) -> None:
    """Sur un corpus daté d'il y a trois ans, la récence doit encore trancher.

    Contre l'heure au mur, `2 ** (-1096/30)` vaut ~1e-11 pour les deux épisodes :
    le terme n'était pas faiblement pondéré, il était **nul**. Mesuré le
    29/09/2026 sur le bench D1 dur, l'énoncé le plus ancien sortait premier dans
    11 cas sur 13. Rapporté au plus récent des candidats, le terme redevient un
    vrai gradient.

    Contenu identique dans les deux épisodes : les composantes dense et sparse
    sont donc à égalité stricte et seule la récence peut les départager.
    """
    contenu = "je vis à Limoges depuis peu"
    await store.write_batch([
        BatchEpisodeItem(content=contenu, role="user", created_at=_ANCIEN_MS),
        BatchEpisodeItem(content=contenu, role="user", created_at=_RECENT_MS),
    ])

    resultats = await store.search(contenu, k=10)
    assert len(resultats) == 2

    par_date = {r.episode.created_at: r for r in resultats}
    recent, ancien = par_date[_RECENT_MS], par_date[_ANCIEN_MS]

    # Le plus récent des candidats est la référence : récence exactement 1.
    assert recent.recency == 1.0
    # L'ancien décroît sans s'effondrer — 60 jours, demi-vie 30 → 0,25.
    assert ancien.recency == pytest.approx(0.25, abs=0.01)

    # Dense et sparse à égalité, donc l'ordre vient de la seule récence.
    assert recent.dense_sim == pytest.approx(ancien.dense_sim)
    assert recent.sparse_sim == pytest.approx(ancien.sparse_sim)
    assert resultats[0].episode.created_at == _RECENT_MS


async def test_score_borne_a_un(store: EpisodicStore) -> None:
    """Les poids sommaient à 1,1, donc le score dépassait 1 dès que la récence
    cessait d'être nulle — et devenait incomparable au score des faits
    (`1 - cosine` ∈ [0,1]) au moment de la fusion dans /search.

    L'assertion sur la somme des poids est celle qui compte : borner le score
    observé ne suffisait pas à détecter le défaut, puisque la récence morte le
    maintenait sous 1 par accident."""
    assert abs(DENSE_WEIGHT + SPARSE_WEIGHT + RECENCY_WEIGHT - 1.0) < 1e-9

    await store.write_batch([
        BatchEpisodeItem(content="exactement la même phrase", role="user",
                         created_at=_RECENT_MS),
    ])
    resultats = await store.search("exactement la même phrase", k=5)
    # Candidat unique donc référence de récence = lui-même, récence = 1 : c'est
    # le cas qui faisait déborder l'ancienne pondération.
    assert resultats[0].recency == 1.0
    assert resultats[0].score <= 1.0
