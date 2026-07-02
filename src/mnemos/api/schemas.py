"""Schemas Pydantic (§16.2) — validation stricte, rejet 422 sur input invalide."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mnemos.models.episodic import Episode
from mnemos.stores.episodic import ScoredEpisode


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


class HealthOut(BaseModel):
    ok: bool
    ollama: bool
    dbs: dict[str, bool]
    salience_queue_depth: int
    worker_last_run: str | None = None  # renseigné à partir de la Phase 4
