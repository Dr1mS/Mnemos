#!/usr/bin/env python3
"""Critère "done" du MVP (§23).

1. Reset les DBs (répertoire dédié data/demo — ne touche pas aux DBs de dev).
2. Écrit 50 messages simulés (Alice : changement de job, déménagement,
   préférences, bruit) avec salience RÉELLE (qwen3:4b) et horloge simulée
   sur 5 jours.
3. Lance le worker une fois (extraction réelle).
4. Vérifie 10 checks et affiche un rapport pass/fail.

Si ce script passe en vert sur la machine cible, le MVP est livré.

Usage : python scripts/demo.py    (~15-20 min sur le profil dev CPU)
"""

from __future__ import annotations

import asyncio
import shutil
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
from mnemos.router.orchestrator import RouterOrchestrator  # noqa: E402
from mnemos.stores.episodic import EpisodicStore  # noqa: E402
from mnemos.stores.semantic import SemanticStore  # noqa: E402
from mnemos.stores.working import WorkingMemoryRegistry  # noqa: E402
from mnemos.tagger.salience import SalienceTagger  # noqa: E402

DEMO_DIR = Path("data/demo")
STEP_MS = 2 * 3_600_000 + 24 * 60_000  # ~2h24 entre messages → 50 msgs ≈ 5 jours

# (contenu, kind) — kind ∈ {"fact": révélation perso, "noise": bruit}
# Les checks 9-10 s'appuient sur ce marquage.
MESSAGES: list[tuple[str, str]] = [
    # ── Jour 1 : Alice se présente ──
    ("Salut ! Moi c'est Alice, je suis data engineer.", "fact"),
    ("Je bosse chez Datalyse depuis trois ans, une boîte d'analytics lyonnaise.", "fact"),
    ("J'habite à Lyon, dans le quartier de la Croix-Rousse.", "fact"),
    ("Tu peux me convertir 25 miles en kilomètres ?", "noise"),
    ("Je préfère le thé au café, surtout le thé vert japonais.", "fact"),
    ("ok merci", "noise"),
    ("Mon chat s'appelle Miso, une siamoise de deux ans que j'adore.", "fact"),
    ("Quelle heure est-il à Tokyo ?", "noise"),
    ("J'adore l'escalade, j'y vais deux fois par semaine.", "fact"),
    ("Il pleut encore aujourd'hui...", "noise"),
    # ── Jour 2 ──
    ("Mon frère Tom travaille chez Airbus à Toulouse.", "fact"),
    ("J'aimerais apprendre Rust cette année, c'est mon objectif principal.", "fact"),
    ("C'est quoi la capitale de l'Australie ?", "noise"),
    ("Je déteste les open spaces, c'est impossible de me concentrer.", "fact"),
    ("lol d'accord", "noise"),
    ("Je parle couramment anglais et je me débrouille en japonais.", "fact"),
    ("Tu peux m'expliquer les décorateurs Python ?", "noise"),
    ("Ma meilleure amie Sarah vient de déménager à Berlin.", "fact"),
    ("super, merci pour l'explication", "noise"),
    ("Je fais du vélo le long du Rhône tous les dimanches matin.", "fact"),
    # ── Jour 3 : le pivot ──
    ("Grosse nouvelle : j'ai reçu une offre de Nexora, une startup IA parisienne !", "fact"),
    ("Si j'accepte, il faudra déménager à Paris...", "noise"),
    ("Ça y est, j'ai signé ! Je quitte Datalyse, je travaille maintenant chez Nexora.", "fact"),
    ("haha oui carrément", "noise"),
    ("Je suis senior ML engineer chez Nexora maintenant, une belle évolution.", "fact"),
    ("Tu connais des bons podcasts sur le machine learning ?", "noise"),
    ("Mes collègues de Datalyse m'ont organisé un pot de départ super émouvant.", "fact"),
    ("Le TGV Lyon-Paris c'est 2h, ça va.", "noise"),
    ("J'ai trouvé un appart dans le 11e arrondissement de Paris !", "fact"),
    ("quelle est la hauteur de la tour eiffel ?", "noise"),
    # ── Jour 4 : nouvelle vie ──
    ("Ça y est, j'ai déménagé ! J'habite maintenant à Paris, dans le 11e.", "fact"),
    ("Miso a un peu de mal avec le nouvel appart, elle miaule la nuit.", "fact"),
    ("Premier jour chez Nexora aujourd'hui, l'équipe a l'air top.", "fact"),
    ("merci !", "noise"),
    ("Finalement je ne bois plus de thé, je suis passée au maté.", "fact"),
    ("Comment on fait un backup SQLite proprement ?", "noise"),
    ("J'ai commencé les cours de japonais du soir, mon sensei est génial.", "fact"),
    ("d'accord, je vois", "noise"),
    ("Mon nouveau manager s'appelle Karim, il vient de chez DeepMind.", "fact"),
    ("Il fait beau à Paris cette semaine.", "noise"),
    # ── Jour 5 ──
    ("Je me suis inscrite à une salle d'escalade près de République.", "fact"),
    ("Nexora vient de lever 20 millions, ambiance électrique au bureau !", "fact"),
    ("c'est noté", "noise"),
    ("Mon objectif Rust avance bien : j'ai fini le Rust Book ce week-end.", "fact"),
    ("Tu peux me rappeler ce que j'ai dit sur mon frère ?", "noise"),
    ("J'ai adopté un deuxième chat, il s'appelle Yuzu !", "fact"),
    ("ok super", "noise"),
    ("La cantine de Nexora est incroyable, je vais prendre trois kilos.", "noise"),
    ("Sarah me manque, on se faisait des soirées jeux chaque semaine à Lyon.", "fact"),
    ("Bref, la vie parisienne me plaît bien pour l'instant.", "noise"),
]

GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"


def report_check(num: int, label: str, ok: bool, detail: str = "") -> bool:
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    print(f"  [{num:2d}] {mark}  {label}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


async def main() -> int:
    t_start = time.perf_counter()

    # ── 1. Reset DBs (répertoire dédié) ──────────────────────────────────────
    if DEMO_DIR.exists():  # noqa: ASYNC240 — setup one-shot, I/O local assumé
        shutil.rmtree(DEMO_DIR)
    DEMO_DIR.mkdir(parents=True)  # noqa: ASYNC240
    settings = Settings(
        _env_file=None,
        DATA_DIR=DEMO_DIR,
        EPISODIC_DB=DEMO_DIR / "episodic.db",
        SEMANTIC_DB=DEMO_DIR / "semantic.db",
        PROCEDURAL_DIR=DEMO_DIR / "procedural",
        CONSOLIDATION_BATCH_SIZE=60,  # un seul run pour les ~28 candidats
    )
    # Horloge simulée : démarre il y a 5 jours, finit ~maintenant.
    now_real = int(datetime.now(tz=UTC).timestamp() * 1000)
    clock = FixedClock(start_ms=now_real - len(MESSAGES) * STEP_MS - 2 * 3_600_000)

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
    orchestrator = RouterOrchestrator(episodic, semantic, WorkingMemoryRegistry())
    print(f"DBs reset dans {DEMO_DIR}/ — horloge simulée sur 5 jours", flush=True)

    # ── 2. 50 messages, salience réelle ──────────────────────────────────────
    print(f"écriture + scoring de {len(MESSAGES)} messages (qwen3:4b réel)…", flush=True)
    history: list[str] = []
    timestamps: list[int] = []
    episode_ids: list[str] = []
    for i, (content, _kind) in enumerate(MESSAGES, 1):
        episode = await episodic.write(content, role="user", session_id="demo")
        scores = await tagger.score(content, history[-5:])
        await episodic.update_salience(episode.id, scores)
        timestamps.append(episode.created_at)
        episode_ids.append(episode.id)
        history.append(content)
        print(
            f"  {i:2d}/{len(MESSAGES)}  salience={scores['combined']:.2f}  {content[:56]!r}",
            flush=True,
        )
        clock.advance(STEP_MS)

    # ── 3. Worker (consolidation réelle) ─────────────────────────────────────
    clock.advance(2 * 3_600_000)  # le dernier message dépasse le délai d'1h
    print("run du worker de consolidation…", flush=True)
    report = await worker.run_once()
    print(
        f"  candidats={report.candidates} consolidés={report.consolidated} "
        f"échecs={report.extraction_failures} faits: +{report.facts_inserted} "
        f"~{report.facts_superseded} ={report.facts_duplicate} "
        f"entités={report.entities_upserted}",
        flush=True,
    )

    # ── 4. Les 10 checks ─────────────────────────────────────────────────────
    print("\nchecks §23 :", flush=True)
    results: list[bool] = []

    # 1-2 : les faits actuels reflètent le DERNIER état, pas un mélange.
    works = await semantic.get_current_facts("user", "works_at")
    results.append(report_check(
        1, "works_at courant unique == Nexora",
        len(works) == 1 and "nexora" in works[0].object.lower(),
        f"courants : {[f.object for f in works]}",
    ))
    lives = await semantic.get_current_facts("user", "lives_in")
    results.append(report_check(
        2, "lives_in courant unique == Paris",
        len(lives) == 1 and "paris" in lives[0].object.lower(),
        f"courants : {[f.object for f in lives]}",
    ))

    # 3-4 : l'historique montre la chaîne de versioning.
    works_hist = await semantic.get_history("user", "works_at")
    datalyse_superseded = any(
        "datalyse" in f.object.lower() and f.valid_until is not None and f.superseded_by
        for f in works_hist
    )
    results.append(report_check(
        3, "history works_at : Datalyse invalidé + superseded_by",
        len(works_hist) >= 2 and datalyse_superseded,
        f"{len(works_hist)} versions",
    ))
    lives_hist = await semantic.get_history("user", "lives_in")
    lyon_superseded = any(
        "lyon" in f.object.lower() and f.valid_until is not None for f in lives_hist
    )
    results.append(report_check(
        4, "history lives_in : Lyon invalidé",
        len(lives_hist) >= 2 and lyon_superseded,
        f"{len(lives_hist)} versions",
    ))

    # 5 : les multi coexistent — deux chats, pas de supersession indue.
    owns = await semantic.get_current_facts("user", "owns")
    owned = " ".join(f.object.lower() for f in owns)
    results.append(report_check(
        5, "multi préservé : Miso ET Yuzu en owns courants",
        "miso" in owned and "yuzu" in owned,
        f"owns : {[f.object for f in owns]}",
    ))

    # 6 : aucun fait courant en doublon (quality gate §21).
    duplicates = await semantic.count_duplicate_current()
    results.append(report_check(6, "zéro doublon courant (gate §21)", duplicates == 0,
                                f"{duplicates} doublon(s)"))

    # 7 : récupération épisodique par fenêtre temporelle (jours 1-2).
    win_start = datetime.fromtimestamp(timestamps[0] / 1000, tz=UTC)
    win_end = datetime.fromtimestamp(timestamps[19] / 1000, tz=UTC)
    early = await episodic.search("Datalyse analytics Lyon", k=10,
                                  time_window=(win_start, win_end))
    in_window = all(
        timestamps[0] <= s.episode.created_at <= timestamps[19] for s in early
    )
    results.append(report_check(
        7, "recherche fenêtre temporelle (jours 1-2) précise",
        len(early) >= 1 and in_window,
        f"{len(early)} résultats, tous dans la fenêtre : {in_window}",
    ))

    # 8 : query ambiguë → épisodique + sémantique mergés cohéremment.
    ambiguous = await orchestrator.query("Nexora", k=10)
    results.append(report_check(
        8, "query ambiguë 'Nexora' → épisodes ET faits",
        ambiguous.type.value == "unknown"
        and len(ambiguous.episodes) > 0
        and any("nexora" in f.fact.object.lower() or "nexora" in f.fact.subject.lower()
                for f in ambiguous.facts),
        f"{len(ambiguous.episodes)} épisodes, {len(ambiguous.facts)} faits",
    ))

    # 9 : le bruit n'est pas consolidé.
    noise_consolidated = 0
    n_noise = 0
    for (_content, kind), ep_id in zip(MESSAGES, episode_ids, strict=True):
        if kind != "noise":
            continue
        n_noise += 1
        ep = await episodic.get_by_id(ep_id)
        if ep is not None and ep.consolidated_at is not None and not ep.extraction_failed:
            noise_consolidated += 1
    results.append(report_check(
        9, "bruit non consolidé (≤ 20% de faux positifs)",
        noise_consolidated <= n_noise * 0.2,
        f"{noise_consolidated}/{n_noise} messages de bruit consolidés",
    ))

    # 10 : la salience discrimine faits perso vs bruit.
    fact_high, n_fact, noise_low = 0, 0, 0
    for (_content, kind), ep_id in zip(MESSAGES, episode_ids, strict=True):
        ep = await episodic.get_by_id(ep_id)
        assert ep is not None
        if kind == "fact":
            n_fact += 1
            fact_high += ep.salience >= settings.SALIENCE_THRESHOLD_CONSOLIDATE
        else:
            noise_low += ep.salience < settings.SALIENCE_THRESHOLD_CONSOLIDATE
    results.append(report_check(
        10, "salience discriminante (≥70% faits hauts, ≥70% bruit bas)",
        fact_high >= n_fact * 0.7 and noise_low >= n_noise * 0.7,
        f"faits ≥ seuil : {fact_high}/{n_fact}, bruit < seuil : {noise_low}/{n_noise}",
    ))

    # ── Rapport final ─────────────────────────────────────────────────────────
    passed = sum(results)
    minutes = (time.perf_counter() - t_start) / 60
    verdict = f"{GREEN}MVP LIVRÉ{RESET}" if passed == len(results) else f"{RED}ÉCHEC{RESET}"
    print(f"\n{passed}/{len(results)} checks verts en {minutes:.1f} min → {verdict}", flush=True)

    await epi_engine.dispose()
    await sem_engine.dispose()
    await client.aclose()
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
