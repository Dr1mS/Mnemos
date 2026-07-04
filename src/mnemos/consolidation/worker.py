"""Worker de consolidation (§15.1) — épisodique → sémantique.

Acquisition de la tier PAR ÉPISODE, pas autour du batch : chaque appel
extractor.extract → manager.generate acquiert/relâche la tier pour UN
épisode. Entre deux épisodes, les appels SMALL en attente s'intercalent —
c'est le prix de la non-famine du write path (§15.1).

Anti-pattern 8 : extraction échouée 2× → mark consolidated avec
extraction_failed=1, on passe au suivant. Pas de boucle infinie.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from mnemos.clock import Clock
from mnemos.config import Settings
from mnemos.consolidation.extractor import Extraction, FactExtractor
from mnemos.logging import get_logger
from mnemos.stores.episodic import ArchiveReport, DecayReport, EpisodicStore
from mnemos.stores.semantic import SemanticStore
from mnemos.tagger.salience import SalienceTagger

logger = get_logger(__name__)

EXTRACTION_ATTEMPTS = 2


@dataclass
class ConsolidationReport:
    candidates: int = 0
    consolidated: int = 0
    extraction_failures: int = 0
    facts_inserted: int = 0
    facts_superseded: int = 0
    facts_duplicate: int = 0
    entities_upserted: int = 0
    rescored: int = 0  # épisodes surprise IS NULL rattrapés (jobs perdus)
    decay: DecayReport | None = None
    archive: ArchiveReport | None = None
    actions: dict[str, int] = field(default_factory=dict)


class ConsolidationWorker:
    def __init__(
        self,
        episodic: EpisodicStore,
        semantic: SemanticStore,
        extractor: FactExtractor,
        settings: Settings,
        clock: Clock,
        tagger: SalienceTagger | None = None,
    ) -> None:
        self._episodic = episodic
        self._semantic = semantic
        self._extractor = extractor
        self._settings = settings
        self._clock = clock
        self._tagger = tagger

    async def run_once(self) -> ConsolidationReport:
        report = ConsolidationReport()
        # Rattrapage : épisodes jamais scorés (queue morte avant drain, §13.3).
        # Sans ça, ils gardent salience=0.5 < seuil et ne consolident jamais.
        if self._tagger is not None:
            unscored = await self._episodic.list_unscored()
            for i, episode in enumerate(unscored, 1):
                self._write_status(phase="scoring", done=i - 1, total=len(unscored),
                                   current=episode.id)
                scores = await self._tagger.score(episode.content, [])
                await self._episodic.update_salience(episode.id, scores)
                report.rescored += 1
            if report.rescored:
                logger.info("salience_rescored", count=report.rescored)
        candidates = await self._episodic.list_pending_consolidation(
            min_salience=self._settings.SALIENCE_THRESHOLD_CONSOLIDATE,
            min_age_hours=self._settings.CONSOLIDATION_DELAY_HOURS,
            limit=self._settings.CONSOLIDATION_BATCH_SIZE,
        )
        report.candidates = len(candidates)
        durations: list[float] = []
        for i, episode in enumerate(candidates, 1):
            avg = sum(durations) / len(durations) if durations else None
            self._write_status(
                phase="extracting", done=i - 1, total=len(candidates),
                current=episode.id,
                eta_s=round(avg * (len(candidates) - i + 1)) if avg else None,
            )
            t0 = time.monotonic()
            extraction = await self._extract_with_retry(
                episode.id, episode.content, episode.role, episode.created_at,
                episode.tenant,
            )
            durations.append(time.monotonic() - t0)
            if extraction is None:
                await self._episodic.mark_consolidated(episode.id, extraction_failed=True)
                report.extraction_failures += 1
                continue
            # DB uniquement à partir d'ici — hors tier (§15.1). Tout est écrit
            # dans le tenant de l'épisode source : isolation stricte, un épisode
            # d'un tenant ne peut jamais produire de faits dans un autre.
            for entity in extraction.entities:
                await self._semantic.upsert_entity(
                    entity.name, entity.entity_type, entity.aliases, tenant=episode.tenant
                )
                report.entities_upserted += 1
            await self._episodic.set_entity_refs(
                episode.id, [e.name for e in extraction.entities]
            )
            for fact in extraction.facts:
                result = await self._semantic.add_fact(
                    subject=fact.subject,
                    predicate=fact.predicate,
                    object_=fact.object,
                    source_episode_ids=[episode.id],
                    confidence=fact.confidence,
                    tenant=episode.tenant,
                )
                if result.action == "inserted":
                    report.facts_inserted += 1
                elif result.action == "superseded":
                    report.facts_superseded += 1
                else:
                    report.facts_duplicate += 1
            await self._episodic.mark_consolidated(episode.id)
            report.consolidated += 1
            logger.info(
                "episode_consolidated",
                episode_id=episode.id,
                facts=len(extraction.facts),
                entities=len(extraction.entities),
            )
        report.decay = await self._episodic.apply_decay()
        report.archive = await self._episodic.archive_old()
        self._write_last_run_marker()
        self._write_status(
            phase="idle",
            last_run={
                "at": self._clock.now_dt().isoformat(),
                "candidates": report.candidates,
                "consolidated": report.consolidated,
                "failures": report.extraction_failures,
                "facts_inserted": report.facts_inserted,
                "facts_superseded": report.facts_superseded,
                "rescored": report.rescored,
            },
            next_run_in_minutes=self._settings.CONSOLIDATION_INTERVAL_MINUTES,
        )
        logger.info(
            "consolidation_run_done",
            candidates=report.candidates,
            consolidated=report.consolidated,
            failures=report.extraction_failures,
            inserted=report.facts_inserted,
            superseded=report.facts_superseded,
            duplicate=report.facts_duplicate,
        )
        return report

    async def _extract_with_retry(
        self, episode_id: str, content: str, role: str, created_at: int, tenant: str
    ) -> Extraction | None:
        for attempt in range(1, EXTRACTION_ATTEMPTS + 1):
            try:
                return await self._extractor.extract(content, role, created_at, tenant)
            except Exception as exc:  # noqa: BLE001 — anti-pattern 8
                logger.warning(
                    "extraction_attempt_failed",
                    episode_id=episode_id,
                    attempt=attempt,
                    error=str(exc),
                )
        logger.error("extraction_gave_up", episode_id=episode_id)
        return None

    def _write_last_run_marker(self) -> None:
        """Marker lu par `mnemos doctor` et GET /v1/health."""
        marker = self._settings.DATA_DIR / "worker_last_run"
        try:
            marker.write_text(self._clock.now_dt().isoformat())
        except OSError as exc:
            logger.warning("worker_marker_write_failed", error=str(exc))

    def _write_status(self, phase: str, **fields: object) -> None:
        """Statut temps réel dans data/worker_status.json — observabilité
        cross-process (mnemos status, /v1/health, viz). Best-effort : une
        erreur d'écriture ne doit jamais interrompre la consolidation."""
        status = {
            "phase": phase,  # scoring | extracting | idle
            "updated_at": self._clock.now_dt().isoformat(),
            **fields,
        }
        try:
            (self._settings.DATA_DIR / "worker_status.json").write_text(
                json.dumps(status, ensure_ascii=False)
            )
        except OSError as exc:
            logger.warning("worker_status_write_failed", error=str(exc))
