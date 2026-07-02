"""Schemas Pydantic (§16.2) — validation stricte, rejet 422 sur input invalide."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mnemos.models.episodic import Episode
from mnemos.models.semantic import Fact
from mnemos.router.orchestrator import QueryResult
from mnemos.stores.episodic import ScoredEpisode
from mnemos.stores.semantic import ScoredFact
from mnemos.stores.working import WMItem


class EpisodeCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1, max_length=32_000)
    role: Literal["user", "assistant", "system"]
    session_id: str | None = Field(default=None, max_length=256)


class EpisodeOut(BaseModel):
    id: str
    created_at: int
    session_id: str | None
    role: str
    content: str
    # None tant que le scoring asynchrone n'est pas passé (§16.1) —
    # détecté via surprise IS NULL (§5.1).
    salience: float | None
    surprise: float | None
    arousal: float | None
    self_ref: float | None
    recurrence: float | None
    decay_state: float
    consolidated_at: int | None
    archived: bool

    @classmethod
    def from_episode(cls, ep: Episode) -> EpisodeOut:
        scored = ep.surprise is not None
        return cls(
            id=ep.id,
            created_at=ep.created_at,
            session_id=ep.session_id,
            role=ep.role,
            content=ep.content,
            salience=ep.salience if scored else None,
            surprise=ep.surprise,
            arousal=ep.arousal,
            self_ref=ep.self_ref,
            recurrence=ep.recurrence,
            decay_state=ep.decay_state,
            consolidated_at=ep.consolidated_at,
            archived=bool(ep.archived),
        )


class ScoredEpisodeOut(BaseModel):
    episode: EpisodeOut
    score: float
    dense_sim: float
    sparse_sim: float
    recency: float

    @classmethod
    def from_scored(cls, s: ScoredEpisode) -> ScoredEpisodeOut:
        return cls(
            episode=EpisodeOut.from_episode(s.episode),
            score=s.score,
            dense_sim=s.dense_sim,
            sparse_sim=s.sparse_sim,
            recency=s.recency,
        )


class QueryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: str = Field(min_length=1, max_length=4_000)
    session_id: str | None = Field(default=None, max_length=256)
    k: int = Field(default=10, ge=1, le=100)


class FactOut(BaseModel):
    id: str
    subject: str
    predicate: str
    object: str
    valid_from: int
    valid_until: int | None
    confidence: float
    superseded_by: str | None

    @classmethod
    def from_fact(cls, f: Fact) -> FactOut:
        return cls(
            id=f.id,
            subject=f.subject,
            predicate=f.predicate,
            object=f.object,
            valid_from=f.valid_from,
            valid_until=f.valid_until,
            confidence=f.confidence,
            superseded_by=f.superseded_by,
        )


class ScoredFactOut(BaseModel):
    fact: FactOut
    score: float

    @classmethod
    def from_scored(cls, s: ScoredFact) -> ScoredFactOut:
        return cls(fact=FactOut.from_fact(s.fact), score=s.score)


class WMItemOut(BaseModel):
    content: str
    role: str
    timestamp_ms: int

    @classmethod
    def from_item(cls, item: WMItem) -> WMItemOut:
        return cls(content=item.content, role=item.role, timestamp_ms=item.timestamp_ms)


class QueryResultOut(BaseModel):
    type: str
    episodes: list[ScoredEpisodeOut]
    facts: list[ScoredFactOut]
    history: list[FactOut]
    working: list[WMItemOut]
    procedural: list[str]

    @classmethod
    def from_result(cls, r: QueryResult) -> QueryResultOut:
        return cls(
            type=r.type.value,
            episodes=[ScoredEpisodeOut.from_scored(e) for e in r.episodes],
            facts=[ScoredFactOut.from_scored(f) for f in r.facts],
            history=[FactOut.from_fact(f) for f in r.history],
            working=[WMItemOut.from_item(w) for w in r.working],
            procedural=r.procedural,
        )


class HealthOut(BaseModel):
    ok: bool
    ollama: bool
    dbs: dict[str, bool]
    salience_queue_depth: int
    worker_last_run: str | None = None


class ConsolidationReportOut(BaseModel):
    candidates: int
    consolidated: int
    extraction_failures: int
    facts_inserted: int
    facts_superseded: int
    facts_duplicate: int
    entities_upserted: int


class DecayReportOut(BaseModel):
    scanned: int
    dry_run: bool
    now_ms: int
