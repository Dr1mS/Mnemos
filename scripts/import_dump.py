#!/usr/bin/env python3
"""Import d'un dump de mémoire externe dans Mnemos + rapport d'extraction.

Rejoue des épisodes JSONL {content, role, session_id, created_at} dans le
VRAI pipeline : embedding + salience (LLM) + consolidation (LLM), avec les
timestamps d'origine (FixedClock par épisode).

Sémantique d'import : `last_decayed_at = maintenant` — la mémoire "naît" à
l'import ; sans ça, le premier apply_decay soustrait tout l'âge réel et les
épisodes anciens partent en archive avant d'avoir servi.

Si --facts est fourni (JSONL {subject, predicate, object, confidence}),
compare les faits extraits par Mnemos aux faits de référence (rappel fuzzy).
Les faits de référence ne sont PAS importés — voir --seed-facts pour ça.

Usage :
  python scripts/import_dump.py --episodes dump/episodes.jsonl \
      --facts dump/facts.jsonl --data-dir data/import
  python scripts/import_dump.py --episodes ... --seed-facts   # importe aussi
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import text  # noqa: E402

from mnemos.clock import FixedClock  # noqa: E402
from mnemos.config import Settings  # noqa: E402
from mnemos.consolidation.extractor import FactExtractor  # noqa: E402
from mnemos.consolidation.worker import ConsolidationWorker  # noqa: E402
from mnemos.embeddings.dense import DenseEmbedder  # noqa: E402
from mnemos.llm.model_manager import ModelManager  # noqa: E402
from mnemos.llm.ollama_client import OllamaClient  # noqa: E402
from mnemos.models.base import make_async_engine  # noqa: E402
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL  # noqa: E402
from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL  # noqa: E402
from mnemos.ontology import PREDICATES  # noqa: E402
from mnemos.stores.episodic import EpisodicStore  # noqa: E402
from mnemos.stores.semantic import SemanticStore  # noqa: E402
from mnemos.tagger.salience import SalienceTagger  # noqa: E402

# Prédicats interchangeables pour le rappel fuzzy (mêmes familles que le POC)
PREDICATE_EQUIV: dict[str, set[str]] = {
    "owns": {"owns", "has_attribute"},
    "is_a": {"is_a", "has_attribute", "has_skill", "works_at"},
    "has_goal": {"has_goal", "knows_about", "has_attribute"},
    "has_attribute": {"has_attribute", "prefers", "has_goal", "knows_about", "owns", "is_a"},
    "knows_about": {"knows_about", "has_skill", "has_goal", "has_attribute"},
    "has_skill": {"has_skill", "has_attribute", "knows_about", "is_a"},
    "prefers": {"prefers", "has_attribute"},
    "lives_in": {"lives_in"},
    "works_at": {"works_at", "is_a"},
    "dislikes": {"dislikes", "has_attribute"},
}


def tokens(s: str) -> set[str]:
    import re

    return {t for t in re.findall(r"\w+", s.lower()) if len(t) > 2}


def object_overlap(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def fuzzy_match(ref: dict[str, str], extracted: list[dict[str, str]]) -> dict[str, str] | None:
    """Un fait extrait correspond-il au fait de référence ? Même famille de
    predicate + même subject + recouvrement d'objects ≥ 0.4."""
    preds_ok = PREDICATE_EQUIV.get(ref["predicate"], {ref["predicate"]})
    best, best_score = None, 0.0
    for f in extracted:
        if f["predicate"] not in preds_ok:
            continue
        if ref["subject"].lower() != f["subject"].lower() and "user" not in f["subject"].lower():
            continue
        score = object_overlap(ref["object"], f["object"])
        if score > best_score:
            best, best_score = f, score
    return best if best_score >= 0.4 else None


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--facts", type=Path, default=None,
                        help="faits de référence pour le rapport de rappel")
    parser.add_argument("--seed-facts", action="store_true",
                        help="importe aussi les faits de référence via add_fact")
    parser.add_argument("--data-dir", type=Path, default=Path("data/import"))
    args = parser.parse_args()

    t_start = time.perf_counter()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    settings = Settings(
        _env_file=None,
        DATA_DIR=args.data_dir,
        EPISODIC_DB=args.data_dir / "episodic.db",
        SEMANTIC_DB=args.data_dir / "semantic.db",
        PROCEDURAL_DIR=args.data_dir / "procedural",
        CONSOLIDATION_BATCH_SIZE=200,
    )
    episodes = sorted(
        (json.loads(line) for line in args.episodes.read_text().splitlines() if line.strip()),
        key=lambda r: r["created_at"],
    )
    now_real = int(datetime.now(tz=UTC).timestamp() * 1000)
    clock = FixedClock(start_ms=episodes[0]["created_at"])

    epi_engine = make_async_engine(settings.EPISODIC_DB)
    sem_engine = make_async_engine(settings.SEMANTIC_DB)
    async with epi_engine.begin() as conn:
        for stmt in EPISODIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    async with sem_engine.begin() as conn:
        for stmt in SEMANTIC_SCHEMA_SQL:
            await conn.execute(text(stmt))
    client = OllamaClient(settings)
    manager = ModelManager(settings, client)
    embedder = DenseEmbedder(manager, settings)
    episodic = EpisodicStore(epi_engine, embedder, clock, settings)
    semantic = SemanticStore(sem_engine, embedder, clock, settings)
    tagger = SalienceTagger(manager, settings)
    worker = ConsolidationWorker(
        episodic, semantic, FactExtractor(manager, settings), settings, clock
    )

    # ── Ingestion : timestamps d'origine + salience réelle ───────────────────
    print(f"ingestion de {len(episodes)} épisodes…", flush=True)
    history: list[str] = []
    for i, record in enumerate(episodes, 1):
        clock.advance(max(0, record["created_at"] - clock.now_ms()))
        episode = await episodic.write(
            record["content"], role=record.get("role", "user"),
            session_id=record.get("session_id"),
        )
        scores = await tagger.score(record["content"], history[-5:])
        await episodic.update_salience(episode.id, scores)
        history.append(record["content"])
        day = datetime.fromtimestamp(record["created_at"] / 1000, tz=UTC).date()
        print(f"  {i:2d}/{len(episodes)}  [{day}]  salience={scores['combined']:.2f}  "
              f"{record['content'][:60]!r}", flush=True)
    # Sémantique d'import : le decay démarre à l'import, pas à created_at.
    async with episodic._sessions() as session, session.begin():
        await session.execute(
            text("UPDATE episodes SET last_decayed_at = :now"), {"now": now_real}
        )

    # ── Consolidation (extraction réelle) ────────────────────────────────────
    clock.advance(max(0, now_real - clock.now_ms()))
    print("consolidation…", flush=True)
    report = await worker.run_once()
    print(f"  candidats={report.candidates} consolidés={report.consolidated} "
          f"échecs={report.extraction_failures} faits: +{report.facts_inserted} "
          f"~{report.facts_superseded} ={report.facts_duplicate} "
          f"entités={report.entities_upserted}", flush=True)

    # ── Rapport : faits extraits ─────────────────────────────────────────────
    current = await semantic.get_current_facts()
    print(f"\n{len(current)} faits courants extraits par Mnemos :", flush=True)
    for f in sorted(current, key=lambda f: (f.predicate, f.object)):
        print(f"  {f.subject}  {f.predicate:14s}  {f.object[:80]}", flush=True)

    # ── Comparaison avec la référence ────────────────────────────────────────
    if args.facts:
        refs = [json.loads(line) for line in args.facts.read_text().splitlines()
                if line.strip()]
        extracted = [
            {"subject": f.subject, "predicate": f.predicate, "object": f.object}
            for f in current
        ]
        found, missed = [], []
        for ref in refs:
            if ref["predicate"] not in PREDICATES:
                continue
            match = fuzzy_match(ref, extracted)
            (found if match else missed).append((ref, match))
        print(f"\nrappel vs référence : {len(found)}/{len(refs)} faits retrouvés", flush=True)
        for ref, match in found:
            assert match is not None
            print(f"  ✓ {ref['predicate']:14s} {ref['object'][:52]!r}\n"
                  f"      ↪ extrait : {match['predicate']} {match['object'][:52]!r}", flush=True)
        for ref, _ in missed:
            print(f"  ✗ {ref['predicate']:14s} {ref['object'][:70]!r}", flush=True)

        if args.seed_facts:
            print("\nimport des faits de référence (add_fact)…", flush=True)
            actions = {"inserted": 0, "superseded": 0, "duplicate": 0}
            for ref in refs:
                result = await semantic.add_fact(
                    subject=ref["subject"], predicate=ref["predicate"],
                    object_=ref["object"], source_episode_ids=["seed_import"],
                    confidence=float(ref.get("confidence", 1.0)),
                )
                actions[result.action] += 1
            print(f"  {actions}", flush=True)

    duplicates = await semantic.count_duplicate_current()
    minutes = (time.perf_counter() - t_start) / 60
    print(f"\ndoublons courants : {duplicates} — terminé en {minutes:.1f} min", flush=True)

    await epi_engine.dispose()
    await sem_engine.dispose()
    await client.aclose()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
