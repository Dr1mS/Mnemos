"""Episodic store (§9) — write, recherche hybride, decay, archivage.

Recherche (§9.2) : KNN dense top-50 (vec0, cosine) → filtres Python
(session, fenêtre, archived, salience) → re-rank hybride
`0.7*dense + 0.3*sparse + 0.1*récence` → top-k. Pondérations configurables.

Décroissance (§9.2) : elapsed depuis COALESCE(last_decayed_at, created_at),
JAMAIS depuis created_at seul (double-comptage → décroissance quadratique).
Temps via Clock injectable (§6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import sqlite_vec  # type: ignore[import-untyped]
from sqlalchemy import CursorResult, select, text, update
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker
from ulid import ULID

from mnemos.clock import Clock
from mnemos.config import Settings
from mnemos.embeddings.dense import DenseEmbedder
from mnemos.embeddings.sparse import sparse_encode, sparse_similarity
from mnemos.logging import get_logger
from mnemos.models.episodic import Episode, EpisodeSparse
from mnemos.tagger.salience import SalienceScores

logger = get_logger(__name__)

KNN_CANDIDATES = 50
DAY_MS = 86_400_000

# Pondérations du score hybride (§8.2, documentées au README)
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3
RECENCY_WEIGHT = 0.1
RECENCY_HALF_LIFE_DAYS = 30.0


@dataclass(frozen=True)
class ScoredEpisode:
    episode: Episode
    score: float
    dense_sim: float
    sparse_sim: float
    recency: float


@dataclass(frozen=True)
class DecayReport:
    scanned: int
    dry_run: bool
    now_ms: int


@dataclass(frozen=True)
class ArchiveReport:
    decayed_consolidated: int  # règle 1 §9.2
    expired_unconsolidated: int  # règle 2 §9.2
    dry_run: bool


@dataclass(frozen=True)
class ArchiveDumpReport:
    dumped: int
    path: str | None


class EpisodicStore:
    def __init__(
        self,
        engine: AsyncEngine,
        embedder: DenseEmbedder,
        clock: Clock,
        settings: Settings,
    ) -> None:
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)
        self._embedder = embedder
        self._clock = clock
        self._settings = settings

    # ── Write path (partie synchrone du §13.3) ───────────────────────────────

    async def write(
        self,
        content: str,
        role: str,
        session_id: str | None = None,
        salience_scores: SalienceScores | None = None,
    ) -> Episode:
        now = self._clock.now_ms()
        dense = await self._embedder.embed(content)
        sparse = sparse_encode(content, now)
        episode = Episode(
            id=str(ULID()),
            created_at=now,
            session_id=session_id,
            role=role,
            content=content,
            **(
                {
                    "salience": salience_scores["combined"],
                    "surprise": salience_scores["surprise"],
                    "arousal": salience_scores["arousal"],
                    "self_ref": salience_scores["self_ref"],
                    "recurrence": salience_scores["recurrence"],
                }
                if salience_scores is not None
                else {}
            ),
        )
        async with self._sessions() as session, session.begin():
            session.add(episode)
            session.add(EpisodeSparse(episode_id=episode.id, sparse_bits=sparse))
            await session.execute(
                text("INSERT INTO episodes_vec(episode_id, embedding) VALUES (:id, :emb)"),
                {"id": episode.id, "emb": sqlite_vec.serialize_float32(dense)},
            )
        logger.info("episode_written", episode_id=episode.id, session_id=session_id, role=role)
        return episode

    async def update_salience(self, episode_id: str, scores: SalienceScores) -> None:
        """Mise à jour asynchrone post-scoring (§13.3) — hors write path."""
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(Episode)
                .where(Episode.id == episode_id)
                .values(
                    salience=scores["combined"],
                    surprise=scores["surprise"],
                    arousal=scores["arousal"],
                    self_ref=scores["self_ref"],
                    recurrence=scores["recurrence"],
                )
            )

    # ── Read path ─────────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        k: int = 10,
        session_id: str | None = None,
        time_window: tuple[datetime, datetime] | None = None,
        min_salience: float = 0.0,
    ) -> list[ScoredEpisode]:
        now = self._clock.now_ms()
        dense = await self._embedder.embed(query)
        query_sparse = sparse_encode(query, now)

        async with self._sessions() as session:
            knn = await session.execute(
                text(
                    "SELECT episode_id, distance FROM episodes_vec "
                    "WHERE embedding MATCH :emb AND k = :k"
                ),
                {"emb": sqlite_vec.serialize_float32(dense), "k": KNN_CANDIDATES},
            )
            distances = {row[0]: float(row[1]) for row in knn}
            if not distances:
                return []
            episodes = (
                (
                    await session.execute(
                        select(Episode, EpisodeSparse.sparse_bits)
                        .join(EpisodeSparse, EpisodeSparse.episode_id == Episode.id)
                        .where(Episode.id.in_(distances))
                    )
                )
                .tuples()
                .all()
            )

        # Filtres Python (§9.2 étape 3)
        scored: list[ScoredEpisode] = []
        for episode, sparse_bits in episodes:
            if episode.archived:
                continue
            if episode.salience < min_salience:
                continue
            if session_id is not None and episode.session_id != session_id:
                continue
            if time_window is not None:
                start_ms = int(time_window[0].timestamp() * 1000)
                end_ms = int(time_window[1].timestamp() * 1000)
                if not start_ms <= episode.created_at <= end_ms:
                    continue
            dense_sim = 1.0 - distances[episode.id]  # distance cosine → similarité
            sparse_sim = sparse_similarity(query_sparse, sparse_bits)
            age_days = max(0.0, (now - episode.created_at) / DAY_MS)
            recency = 2.0 ** (-age_days / RECENCY_HALF_LIFE_DAYS)
            score = (
                DENSE_WEIGHT * dense_sim
                + SPARSE_WEIGHT * sparse_sim
                + RECENCY_WEIGHT * recency
            )
            scored.append(ScoredEpisode(episode, score, dense_sim, sparse_sim, recency))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:k]

    async def get_by_id(self, episode_id: str) -> Episode | None:
        async with self._sessions() as session:
            return await session.get(Episode, episode_id)

    async def list_recent(self, session_id: str | None = None, n: int = 5) -> list[Episode]:
        """Derniers épisodes (ordre chronologique) — historique du scoring §13.2."""
        stmt = (
            select(Episode)
            .where(Episode.archived == 0)
            .order_by(Episode.created_at.desc())
            .limit(n)
        )
        if session_id is not None:
            stmt = stmt.where(Episode.session_id == session_id)
        async with self._sessions() as session:
            rows = list((await session.execute(stmt)).scalars())
        return list(reversed(rows))

    # ── Consolidation hooks (§15) ─────────────────────────────────────────────

    async def pending_counts(self) -> dict[str, int]:
        """Ce qui attend l'IA locale (observabilité) : épisodes jamais scorés,
        candidats mûrs pour consolidation, saillants mais trop récents."""
        now = self._clock.now_ms()
        cutoff = now - int(self._settings.CONSOLIDATION_DELAY_HOURS * 3_600_000)
        threshold = self._settings.SALIENCE_THRESHOLD_CONSOLIDATE
        async with self._sessions() as session:
            row = await session.execute(
                text(
                    """
                    SELECT
                      SUM(CASE WHEN surprise IS NULL THEN 1 ELSE 0 END),
                      SUM(CASE WHEN surprise IS NOT NULL AND consolidated_at IS NULL
                               AND salience > :thr AND created_at <= :cutoff
                               THEN 1 ELSE 0 END),
                      SUM(CASE WHEN surprise IS NOT NULL AND consolidated_at IS NULL
                               AND salience > :thr AND created_at > :cutoff
                               THEN 1 ELSE 0 END)
                    FROM episodes WHERE archived = 0
                    """
                ),
                {"thr": threshold, "cutoff": cutoff},
            )
            unscored, ready, too_recent = row.one()
        return {
            "unscored": int(unscored or 0),
            "consolidation_ready": int(ready or 0),
            "consolidation_waiting": int(too_recent or 0),
        }

    async def list_unscored(self, limit: int = 50) -> list[Episode]:
        """Épisodes jamais passés au scoring de salience (surprise IS NULL,
        §5.1) — jobs perdus quand le process meurt avant que la queue draine.
        Rattrapés par le worker (§15.1)."""
        async with self._sessions() as session:
            rows = await session.execute(
                select(Episode)
                .where(Episode.surprise.is_(None), Episode.archived == 0)
                .order_by(Episode.created_at)
                .limit(limit)
            )
            return list(rows.scalars())

    async def list_pending_consolidation(
        self, min_salience: float, min_age_hours: float, limit: int
    ) -> list[Episode]:
        cutoff = self._clock.now_ms() - int(min_age_hours * 3_600_000)
        async with self._sessions() as session:
            rows = await session.execute(
                select(Episode)
                .where(
                    Episode.consolidated_at.is_(None),
                    Episode.archived == 0,
                    Episode.salience > min_salience,
                    Episode.created_at <= cutoff,
                )
                .order_by(Episode.salience.desc())
                .limit(limit)
            )
            return list(rows.scalars())

    async def mark_consolidated(self, episode_id: str, extraction_failed: bool = False) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(Episode)
                .where(Episode.id == episode_id)
                .values(
                    consolidated_at=self._clock.now_ms(),
                    extraction_failed=1 if extraction_failed else 0,
                )
            )

    async def set_entity_refs(self, episode_id: str, entity_names: list[str]) -> None:
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(Episode)
                .where(Episode.id == episode_id)
                .values(entity_refs=json.dumps(entity_names, ensure_ascii=False))
            )

    # ── Lifecycle (§9.2) ──────────────────────────────────────────────────────

    async def apply_decay(self, dry_run: bool = False) -> DecayReport:
        now = self._clock.now_ms()
        rate = self._settings.DECAY_RATE_DAILY
        async with self._sessions() as session, session.begin():
            if dry_run:
                count = len(
                    (
                        await session.execute(select(Episode.id).where(Episode.archived == 0))
                    ).all()
                )
                return DecayReport(scanned=count, dry_run=True, now_ms=now)
            result = await session.execute(
                text(
                    """
                    UPDATE episodes
                    SET decay_state = MAX(
                          0.0,
                          decay_state
                          - :rate
                            * ((:now - COALESCE(last_decayed_at, created_at)) / 86400000.0)
                            * (2 - salience)
                        ),
                        last_decayed_at = :now
                    WHERE archived = 0
                    """
                ),
                {"rate": rate, "now": now},
            )
            scanned = cast("CursorResult[Any]", result).rowcount or 0
        logger.info("decay_applied", scanned=scanned, now_ms=now)
        return DecayReport(scanned=scanned, dry_run=False, now_ms=now)

    async def archive_old(self, dry_run: bool = False) -> ArchiveReport:
        now = self._clock.now_ms()
        retention_cutoff = now - self._settings.EPISODIC_RETENTION_DAYS * DAY_MS
        rule1 = (
            Episode.archived == 0,
            Episode.decay_state < 0.1,
            Episode.consolidated_at.is_not(None),
        )
        rule2 = (
            Episode.archived == 0,
            Episode.created_at < retention_cutoff,
            Episode.salience < self._settings.SALIENCE_THRESHOLD_CONSOLIDATE,
            Episode.consolidated_at.is_(None),
        )
        async with self._sessions() as session, session.begin():
            if dry_run:
                n1 = len((await session.execute(select(Episode.id).where(*rule1))).all())
                n2 = len((await session.execute(select(Episode.id).where(*rule2))).all())
                return ArchiveReport(n1, n2, dry_run=True)
            r1 = await session.execute(update(Episode).where(*rule1).values(archived=1))
            r2 = await session.execute(update(Episode).where(*rule2).values(archived=1))
        report = ArchiveReport(
            cast("CursorResult[Any]", r1).rowcount or 0,
            cast("CursorResult[Any]", r2).rowcount or 0,
            dry_run=False,
        )
        logger.info(
            "archive_applied",
            decayed_consolidated=report.decayed_consolidated,
            expired_unconsolidated=report.expired_unconsolidated,
        )
        return report

    async def dump_archived(self) -> ArchiveDumpReport:
        """Dump JSONL des épisodes archivés vers data/archive/YYYY-MM.jsonl
        puis DELETE (§9.2, worker mensuel). Deletes explicites sur les trois
        tables : le FK CASCADE de episodes_sparse ne fire pas (PRAGMA
        foreign_keys OFF par défaut) et episodes_vec (vec0) n'a pas de FK.
        """
        now_dt = self._clock.now_dt()
        archive_dir = self._settings.DATA_DIR / "archive"
        async with self._sessions() as session, session.begin():
            rows = list(
                (await session.execute(select(Episode).where(Episode.archived == 1))).scalars()
            )
            if not rows:
                return ArchiveDumpReport(dumped=0, path=None)
            archive_dir.mkdir(parents=True, exist_ok=True)
            path = archive_dir / f"{now_dt.year}-{now_dt.month:02d}.jsonl"
            with path.open("a") as fh:
                for ep in rows:
                    record = {
                        k: v for k, v in vars(ep).items() if not k.startswith("_")
                    }
                    fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            ids = [ep.id for ep in rows]
            for stmt in (
                "DELETE FROM episodes_sparse WHERE episode_id IN (SELECT id FROM episodes WHERE archived = 1)",  # noqa: E501
                "DELETE FROM episodes_vec WHERE episode_id IN (SELECT id FROM episodes WHERE archived = 1)",  # noqa: E501
                "DELETE FROM episodes WHERE archived = 1",
            ):
                await session.execute(text(stmt))
        logger.info("archive_dumped", count=len(ids), path=str(path))
        return ArchiveDumpReport(dumped=len(ids), path=str(path))
