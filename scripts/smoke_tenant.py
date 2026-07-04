"""Smoke test multi-tenant (Lot 1) — critère de fin.

Scénario end-to-end sur des DB temporaires (jamais la prod), avec Ollama réel :
  1. crée un tenant applicatif `atelios` (3 écritures) + une écriture perso ;
  2. force une consolidation → vérifie que les faits d'atelios portent
     subject='atelios' et que RIEN ne fuit dans le tenant personnel ;
  3. étanchéité croisée : query/facts d'un tenant ne voit jamais l'autre ;
  4. /health : Ollama up (embedding OK) puis simulation down (panne nommée).

Usage :
    .venv/Scripts/python.exe scripts/smoke_tenant.py

Sort en code 0 si tout passe, 1 sinon. N'écrit que dans un dossier temporaire.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

from sqlalchemy import text

from mnemos.clock import Clock
from mnemos.config import Settings
from mnemos.consolidation.extractor import FactExtractor
from mnemos.consolidation.worker import ConsolidationWorker
from mnemos.embeddings.dense import DenseEmbedder
from mnemos.llm.model_manager import ModelManager
from mnemos.llm.ollama_client import OllamaClient
from mnemos.models.base import make_async_engine
from mnemos.models.episodic import EPISODIC_SCHEMA_SQL
from mnemos.models.semantic import SEMANTIC_SCHEMA_SQL
from mnemos.stores.episodic import EpisodicStore
from mnemos.stores.semantic import SemanticStore
from mnemos.tagger.salience import SalienceTagger

PERSONAL = "user"
APP = "atelios"

_checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool) -> None:
    _checks.append((label, ok))
    mark = "[OK]" if ok else "[XX]"
    print(f"  {mark} {label}")


async def main() -> int:
    d = Path(tempfile.mkdtemp(prefix="mnemos_smoke_"))
    # CONSOLIDATION_DELAY_HOURS=0 : dans un smoke test les épisodes viennent
    # d'être écrits ; on ne veut pas attendre 1 h pour qu'ils deviennent
    # candidats. La salience réelle (qwen3:4b) est calculée par le rescore du
    # worker (list_unscored) avant la sélection des candidats.
    settings = Settings(
        _env_file=None, DATA_DIR=d, EPISODIC_DB=d / "e.db", SEMANTIC_DB=d / "s.db",
        CONSOLIDATION_DELAY_HOURS=0,
    )
    epi = make_async_engine(settings.EPISODIC_DB)
    sem = make_async_engine(settings.SEMANTIC_DB)
    async with epi.begin() as c:
        for stmt in EPISODIC_SCHEMA_SQL:
            await c.execute(text(stmt))
    async with sem.begin() as c:
        for stmt in SEMANTIC_SCHEMA_SQL:
            await c.execute(text(stmt))

    client = OllamaClient(settings)
    manager = ModelManager(settings, client)
    embedder = DenseEmbedder(manager, settings)
    clock = Clock()
    episodic = EpisodicStore(epi, embedder, clock, settings)
    semantic = SemanticStore(sem, embedder, clock, settings)
    tagger = SalienceTagger(manager, settings)
    worker = ConsolidationWorker(
        episodic, semantic, FactExtractor(manager, settings), settings, clock, tagger=tagger
    )

    try:
        # ── 1. Écritures : 3 dans atelios (faits saillants) + 1 perso ────────
        # On force une salience haute sur les épisodes atelios : le tagger de
        # salience est encore user-centric (self_ref bas pour un énoncé projet)
        # → c'est un point Lot 2 (verrouillé). Ici on veut exercer le CHEMIN
        # d'extraction atelios, pas tester le scoring. Salience explicite =
        # candidats garantis.
        high = {"surprise": 0.9, "arousal": 0.5, "self_ref": 0.9,
                "recurrence": 0.0, "combined": 0.95}
        print("\n[1] écritures tenant atelios (3) + perso (1)")
        for content in (
            "Le projet Atelios vise à livrer un moteur de simulation en Rust.",
            "L'équipe Atelios travaille depuis Grenoble.",
            "Atelios a pour objectif une démo publique en septembre.",
        ):
            ep = await episodic.write(content, role="user", tenant=APP)
            await episodic.update_salience(ep.id, high)  # type: ignore[arg-type]
        perso = await episodic.write(
            "Moi, j'habite à Annecy et je préfère le thé.", role="user", tenant=PERSONAL
        )
        await episodic.update_salience(perso.id, high)  # type: ignore[arg-type]
        check("4 épisodes écrits (3 atelios + 1 perso)", True)

        # ── 2. Consolidation → faits ─────────────────────────────────────────
        print("\n[2] consolidation (extraction LLM réelle — peut prendre ~1 min sur CPU)")
        report = await worker.run_once()
        print(
            f"    candidats={report.candidates} consolidés={report.consolidated} "
            f"faits +{report.facts_inserted} ~{report.facts_superseded}"
        )
        atelios_facts = await semantic.get_current_facts(tenant=APP)
        personal_facts = await semantic.get_current_facts(tenant=PERSONAL)
        print(f"    faits atelios : {[(f.subject, f.predicate, f.object) for f in atelios_facts]}")
        print(f"    faits perso   : {[(f.subject, f.predicate, f.object) for f in personal_facts]}")

        check("des faits atelios ont été produits", len(atelios_facts) > 0)
        check(
            "tous les faits atelios portent subject='atelios'",
            all(f.subject == APP for f in atelios_facts),
        )
        check(
            "aucun fait atelios n'a subject='user'",
            not any(f.subject == PERSONAL for f in atelios_facts),
        )
        check(
            "tous les faits atelios portent tenant='atelios'",
            all(f.tenant == APP for f in atelios_facts),
        )

        # ── 3. Étanchéité croisée ────────────────────────────────────────────
        print("\n[3] étanchéité croisée")
        # facts : le tenant perso ne voit aucun objet d'atelios
        atelios_objects = {f.object for f in atelios_facts}
        personal_objects = {f.object for f in personal_facts}
        check(
            "les faits perso et atelios sont disjoints",
            atelios_objects.isdisjoint(personal_objects) or not atelios_objects,
        )
        # query épisodique : chaque tenant ne récupère que ses épisodes
        a_hits = await episodic.search("Atelios simulation Rust", k=10, tenant=APP)
        p_hits = await episodic.search("Atelios simulation Rust", k=10, tenant=PERSONAL)
        check("query atelios ne renvoie que des épisodes atelios",
              all(e.episode.tenant == APP for e in a_hits) and len(a_hits) > 0)
        check("query perso ne voit aucun épisode atelios",
              all(e.episode.tenant == PERSONAL for e in p_hits))
        # search_facts isolé
        af = await semantic.search_facts("simulation Rust Grenoble", k=10, tenant=APP)
        pf = await semantic.search_facts("simulation Rust Grenoble", k=10, tenant=PERSONAL)
        check("search_facts atelios ne renvoie que des faits atelios",
              all(s.fact.tenant == APP for s in af))
        check("search_facts perso ne renvoie aucun fait atelios",
              all(s.fact.tenant == PERSONAL for s in pf))
        # rétraction isolée : rétracter dans un tenant ne touche pas l'autre
        if atelios_facts:
            f0 = atelios_facts[0]
            await semantic.retract_fact(f0.subject, f0.predicate, f0.object, tenant=PERSONAL)
            still = await semantic.get_current_facts(tenant=APP)
            check("retract sur tenant perso ne touche pas atelios",
                  any(f.id == f0.id for f in still))

        # ── 4. /health up puis down ──────────────────────────────────────────
        print("\n[4] santé")
        db_ok = (await episodic.ping()) is None and (await semantic.ping()) is None
        check("DB accessibles (ping)", db_ok)
        await manager.embed_probe()  # warm-up
        embed_err = await manager.embed_probe()
        check("embedding Ollama joignable (up)", embed_err is None)

        # down simulé : client pointant sur un port mort
        dead_settings = Settings(_env_file=None, OLLAMA_HOST="http://localhost:1")
        dead_client = OllamaClient(dead_settings)
        dead_err = await dead_client.embed_probe(settings.EMBED_MODEL)
        await dead_client.aclose()
        check("panne embedding détectée et nommée (down)",
              dead_err is not None and "/api/embed" in dead_err)
        print(f"    message de panne : {dead_err}")

    finally:
        await epi.dispose()
        await sem.dispose()
        await client.aclose()

    passed = sum(1 for _, ok in _checks if ok)
    total = len(_checks)
    print(f"\n{'='*50}\nResultat : {passed}/{total} checks")
    if passed == total:
        print("SMOKE TEST OK")
        return 0
    print("SMOKE TEST ECHOUE - checks en echec :")
    for label, ok in _checks:
        if not ok:
            print(f"  [XX] {label}")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
