"""CLI Typer (§17).

Phase 0 : `mnemos doctor` + `mnemos version`. Les autres commandes
(serve, write, search, query, facts, consolidate, decay, export, stats)
arrivent avec leurs phases respectives.
"""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Callable, Coroutine
from pathlib import Path

import typer

from mnemos import __version__
from mnemos.config import Settings, get_settings

app = typer.Typer(name="mnemos", help="Mnemos — multi-system memory for LLM agents.")

OK = typer.style("✓", fg=typer.colors.GREEN, bold=True)
KO = typer.style("✗", fg=typer.colors.RED, bold=True)
WARN = typer.style("~", fg=typer.colors.YELLOW, bold=True)


@app.command()
def version() -> None:
    """Affiche la version."""
    typer.echo(f"mnemos {__version__}")


@app.command()
def serve() -> None:
    """Lance le serveur API (§17)."""
    import uvicorn

    from mnemos.server import create_app

    settings = get_settings()
    uvicorn.run(create_app(settings), host=settings.API_HOST, port=settings.API_PORT)


# ── Composants partagés par les commandes directes (hors serveur) ────────────


class _Components:
    """Assemblage complet des composants Mnemos pour le CLI."""

    def __init__(self) -> None:
        from mnemos.clock import Clock
        from mnemos.consolidation.extractor import FactExtractor
        from mnemos.consolidation.worker import ConsolidationWorker
        from mnemos.embeddings.dense import DenseEmbedder
        from mnemos.llm.model_manager import ModelManager
        from mnemos.llm.ollama_client import OllamaClient
        from mnemos.models.base import make_async_engine
        from mnemos.router.orchestrator import RouterOrchestrator
        from mnemos.stores.episodic import EpisodicStore
        from mnemos.stores.procedural import ProceduralStore
        from mnemos.stores.semantic import SemanticStore
        from mnemos.stores.working import WorkingMemoryRegistry

        self.settings = get_settings()
        self.clock = Clock()
        self.client = OllamaClient(self.settings)
        manager = ModelManager(self.settings, self.client)
        embedder = DenseEmbedder(manager, self.settings)
        self.episodic_engine = make_async_engine(self.settings.EPISODIC_DB)
        self.semantic_engine = make_async_engine(self.settings.SEMANTIC_DB)
        self.episodic = EpisodicStore(self.episodic_engine, embedder, self.clock, self.settings)
        self.semantic = SemanticStore(self.semantic_engine, embedder, self.clock, self.settings)
        self.procedural = ProceduralStore(self.settings.PROCEDURAL_DIR, self.clock)
        self.orchestrator = RouterOrchestrator(
            self.episodic, self.semantic, WorkingMemoryRegistry(), self.procedural
        )
        self.worker = ConsolidationWorker(
            self.episodic, self.semantic, FactExtractor(manager, self.settings),
            self.settings, self.clock,
        )

    async def aclose(self) -> None:
        await self.episodic_engine.dispose()
        await self.semantic_engine.dispose()
        await self.client.aclose()


def _run_with_components(fn: Callable[[_Components], Coroutine[None, None, None]]) -> None:
    import asyncio

    async def _wrapped() -> None:
        components = _Components()
        try:
            await fn(components)
        finally:
            await components.aclose()

    asyncio.run(_wrapped())


@app.command()
def write(
    content: str,
    role: str = typer.Option("user", "--role"),
    session: str | None = typer.Option(None, "--session"),
) -> None:
    """Écrit un épisode (§17). Salience scorée de façon synchrone ici (CLI)."""

    async def _write(c: _Components) -> None:
        episode = await c.episodic.write(content, role=role, session_id=session)
        typer.echo(f"écrit : {episode.id} (salience par défaut 0.5, scoring via serveur)")

    _run_with_components(_write)


@app.command()
def search(query: str, k: int = typer.Option(10, "-k")) -> None:
    """Recherche épisodique hybride (§17)."""

    async def _search(c: _Components) -> None:
        results = await c.episodic.search(query, k=k)
        if not results:
            typer.echo("aucun résultat")
            return
        for s in results:
            typer.echo(f"{s.score:.3f}  [{s.episode.id}]  {s.episode.content[:80]}")

    _run_with_components(_search)


@app.command()
def query(q: str, session: str | None = typer.Option(None, "--session")) -> None:
    """Query routée multi-store (§17)."""

    async def _query(c: _Components) -> None:
        result = await c.orchestrator.query(q, session_id=session)
        typer.echo(f"type : {result.type.value}")
        for s in result.episodes:
            typer.echo(f"  épisode {s.score:.3f}  {s.episode.content[:70]}")
        for f in result.facts:
            typer.echo(f"  fait {f.score:.3f}  {f.fact.subject} {f.fact.predicate} {f.fact.object}")
        for fact in result.history:
            status = "courant" if fact.valid_until is None else "invalidé"
            typer.echo(f"  history [{status}]  {fact.subject} {fact.predicate} {fact.object}")
        for name in result.procedural:
            typer.echo(f"  skill  {name}")

    _run_with_components(_query)


@app.command()
def facts(
    subject: str | None = typer.Option(None, "--subject"),
    predicate: str | None = typer.Option(None, "--predicate"),
    history: bool = typer.Option(False, "--history"),
) -> None:
    """Faits courants, ou historique complet avec --history (§17)."""

    async def _facts(c: _Components) -> None:
        if history:
            if not subject or not predicate:
                typer.echo("--history exige --subject et --predicate", err=True)
                raise typer.Exit(code=2)
            rows = await c.semantic.get_history(subject, predicate)
        else:
            rows = await c.semantic.get_current_facts(subject, predicate)
        for f in rows:
            status = "courant " if f.valid_until is None else "invalidé"
            typer.echo(
                f"[{status}]  {f.subject}  {f.predicate}  {f.object}  "
                f"(conf {f.confidence:.2f})"
            )
        if not rows:
            typer.echo("aucun fait")

    _run_with_components(_facts)


@app.command()
def consolidate() -> None:
    """Force un run du worker de consolidation (§17)."""

    async def _consolidate(c: _Components) -> None:
        report = await c.worker.run_once()
        typer.echo(
            f"candidats={report.candidates} consolidés={report.consolidated} "
            f"échecs={report.extraction_failures} faits: +{report.facts_inserted} "
            f"~{report.facts_superseded} ={report.facts_duplicate} "
            f"entités={report.entities_upserted}"
        )

    _run_with_components(_consolidate)


@app.command()
def decay(dry_run: bool = typer.Option(False, "--dry-run")) -> None:
    """Force apply_decay (§17)."""

    async def _decay(c: _Components) -> None:
        report = await c.episodic.apply_decay(dry_run=dry_run)
        typer.echo(f"épisodes traités : {report.scanned} (dry_run={report.dry_run})")

    _run_with_components(_decay)


@app.command()
def worker() -> None:
    """Worker standalone (§17) — consolidation périodique + dump mensuel des
    archivés. Instance unique via lock fichier dans data/ (§15.1)."""
    import os

    settings = get_settings()
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = settings.DATA_DIR / "worker.lock"
    try:
        # O_EXCL : création atomique — échoue si le lock existe déjà.
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        typer.echo(
            f"{KO} un worker tourne déjà (ou lock orphelin) : {lock_path}\n"
            f"   Si aucun worker n'est actif, supprimer le fichier.",
            err=True,
        )
        raise typer.Exit(code=1) from None

    async def _worker(c: _Components) -> None:
        import asyncio as aio

        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger

        scheduler = AsyncIOScheduler()
        scheduler.add_job(
            c.worker.run_once,
            IntervalTrigger(minutes=c.settings.CONSOLIDATION_INTERVAL_MINUTES),
        )
        # Dump mensuel des archivés (§9.2) — le 1er du mois à 03:00.
        scheduler.add_job(c.episodic.dump_archived, CronTrigger(day=1, hour=3))
        scheduler.start()
        typer.echo(
            f"worker démarré (consolidation toutes les "
            f"{c.settings.CONSOLIDATION_INTERVAL_MINUTES} min, Ctrl-C pour arrêter)"
        )
        await c.worker.run_once()  # premier tick immédiat
        try:
            await aio.Event().wait()
        finally:
            scheduler.shutdown(wait=False)

    try:
        _run_with_components(_worker)
    except KeyboardInterrupt:
        typer.echo("worker arrêté")
    finally:
        lock_path.unlink(missing_ok=True)


@app.command()
def backup(out_dir: Path = typer.Option(Path("./backups"), "--out")) -> None:
    """Backup atomique des DBs via VACUUM INTO (anti-pattern 9 : jamais de
    copie brute, le -wal resterait incohérent)."""
    from datetime import UTC, datetime

    settings = get_settings()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    for name, db_path in (("episodic", settings.EPISODIC_DB),
                          ("semantic", settings.SEMANTIC_DB)):
        if not db_path.exists():
            typer.echo(f"{WARN} {name} : {db_path} absent, skip")
            continue
        target = out_dir / f"mnemos_{name}_{stamp}.db"
        conn = sqlite3.connect(db_path)
        try:
            conn.execute("VACUUM INTO ?", (str(target),))
        finally:
            conn.close()
        typer.echo(f"{OK} {name} → {target}")


@app.command()
def export(
    out: Path = typer.Option(..., "--out"),
    format: str = typer.Option("jsonl", "--format"),
) -> None:
    """Export JSONL des épisodes + faits (§17). Une ligne = {type, data}."""
    if format != "jsonl":
        typer.echo("seul --format jsonl est supporté", err=True)
        raise typer.Exit(code=2)

    async def _export(c: _Components) -> None:
        import json

        from sqlalchemy import select

        from mnemos.models.episodic import Episode
        from mnemos.models.semantic import Fact

        n = 0
        with out.open("w") as fh:
            async with c.episodic._sessions() as session:
                for ep in (await session.execute(select(Episode))).scalars():
                    row = {k: v for k, v in vars(ep).items() if not k.startswith("_")}
                    fh.write(json.dumps({"type": "episode", "data": row},
                                        ensure_ascii=False) + "\n")
                    n += 1
            async with c.semantic._sessions() as session:
                for fact in (await session.execute(select(Fact))).scalars():
                    row = {k: v for k, v in vars(fact).items() if not k.startswith("_")}
                    fh.write(json.dumps({"type": "fact", "data": row},
                                        ensure_ascii=False) + "\n")
                    n += 1
        typer.echo(f"{n} enregistrements → {out}")

    _run_with_components(_export)


@app.command()
def stats() -> None:
    """Stats globales (§17)."""

    async def _stats(c: _Components) -> None:
        from sqlalchemy import text as sql

        async with c.episodic._sessions() as session:
            total = (await session.execute(sql("SELECT COUNT(*) FROM episodes"))).scalar_one()
            consolidated = (
                await session.execute(
                    sql("SELECT COUNT(*) FROM episodes WHERE consolidated_at IS NOT NULL")
                )
            ).scalar_one()
            archived = (
                await session.execute(sql("SELECT COUNT(*) FROM episodes WHERE archived = 1"))
            ).scalar_one()
        async with c.semantic._sessions() as session:
            current = (
                await session.execute(
                    sql("SELECT COUNT(*) FROM facts WHERE valid_until IS NULL")
                )
            ).scalar_one()
            superseded = (
                await session.execute(
                    sql("SELECT COUNT(*) FROM facts WHERE valid_until IS NOT NULL")
                )
            ).scalar_one()
            entities = (await session.execute(sql("SELECT COUNT(*) FROM entities"))).scalar_one()
        duplicates = await c.semantic.count_duplicate_current()
        skills = len(c.procedural.list_skills())
        typer.echo(f"épisodes        : {total} (consolidés {consolidated}, archivés {archived})")
        typer.echo(f"faits courants  : {current} (invalidés {superseded}, doublons {duplicates})")
        typer.echo(f"entités         : {entities}")
        typer.echo(f"skills          : {skills}")

    _run_with_components(_stats)


def _check_ollama(settings: Settings) -> tuple[bool, list[str]]:
    """Ollama joignable + modèles requis pullés."""
    import httpx

    lines: list[str] = []
    try:
        r = httpx.get(f"{settings.OLLAMA_HOST}/api/version", timeout=5)
        r.raise_for_status()
        lines.append(f"{OK} Ollama up ({settings.OLLAMA_HOST}, v{r.json().get('version', '?')})")
    except Exception as exc:  # noqa: BLE001 — doctor : diagnostic, pas crash
        lines.append(f"{KO} Ollama injoignable sur {settings.OLLAMA_HOST} : {exc}")
        return False, lines

    try:
        tags = httpx.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5).json()
        installed = {m["name"] for m in tags.get("models", [])}
        # "qwen3:4b" doit matcher "qwen3:4b" exact ou préfixe de tag installé
        ok = True
        for role, model in (
            ("embed", settings.EMBED_MODEL),
            ("salience", settings.SALIENCE_MODEL),
            ("extraction", settings.EXTRACTION_MODEL),
        ):
            found = any(name == model or name.startswith(f"{model}") for name in installed) or any(
                name.split(":")[0] == model for name in installed
            )
            if found:
                lines.append(f"{OK} modèle {role} : {model}")
            else:
                lines.append(f"{KO} modèle {role} manquant : {model} (ollama pull {model})")
                ok = False
        return ok, lines
    except Exception as exc:  # noqa: BLE001
        lines.append(f"{KO} listing des modèles impossible : {exc}")
        return False, lines


def _check_dbs(settings: Settings) -> tuple[bool, list[str]]:
    """Version tables Alembic au head pour chaque base."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    lines: list[str] = []
    ini = Path("alembic.ini")
    if not ini.exists():
        return False, [f"{KO} alembic.ini introuvable (lancer depuis la racine du projet)"]

    script = ScriptDirectory.from_config(Config(str(ini)))
    heads = set(script.get_heads())
    if not heads:
        return True, [f"{OK} aucune migration définie (Phase 0) — DBs à créer en Phase 2/4"]

    ok = True
    dbs = (("episodic", settings.EPISODIC_DB), ("semantic", settings.SEMANTIC_DB))
    for db_name, db_path in dbs:
        if not db_path.exists():
            lines.append(f"{KO} {db_name} : {db_path} absent (alembic upgrade head)")
            ok = False
            continue
        with sqlite3.connect(db_path) as conn:
            try:
                rows = conn.execute(
                    f"SELECT version_num FROM alembic_version_{db_name}"  # noqa: S608
                ).fetchall()
                current = {r[0] for r in rows}
            except sqlite3.OperationalError:
                current = set()
        if current & heads:
            lines.append(f"{OK} {db_name} migré (head {next(iter(current))[:12]})")
        else:
            lines.append(f"{KO} {db_name} pas au head (alembic upgrade head)")
            ok = False
    return ok, lines


def _check_sqlite_vec() -> tuple[bool, list[str]]:
    """Extension sqlite-vec chargeable dans ce runtime."""
    try:
        import sqlite_vec  # type: ignore[import-untyped]

        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        (vec_version,) = conn.execute("SELECT vec_version()").fetchone()
        conn.close()
        return True, [f"{OK} sqlite-vec chargeable ({vec_version})"]
    except Exception as exc:  # noqa: BLE001
        return False, [f"{KO} sqlite-vec non chargeable : {exc}"]


def _check_disk(settings: Settings) -> tuple[bool, list[str]]:
    settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(settings.DATA_DIR)
    free_gb = usage.free / 1e9
    if free_gb < 2:
        return False, [f"{KO} espace disque faible : {free_gb:.1f} GB libres"]
    return True, [f"{OK} espace disque : {free_gb:.0f} GB libres"]


def _check_worker(settings: Settings) -> tuple[bool, list[str]]:
    """Dernier run du worker — informatif tant que le worker n'existe pas (Phase 4)."""
    marker = settings.DATA_DIR / "worker_last_run"
    if not marker.exists():
        return True, [f"{WARN} worker : jamais lancé (normal avant Phase 4)"]
    return True, [f"{OK} worker : dernier run enregistré ({marker.read_text().strip()})"]


@app.command()
def doctor() -> None:
    """Health check + diagnostic (§17)."""
    settings = get_settings()
    all_ok = True
    checks: list[Callable[[], tuple[bool, list[str]]]] = [
        lambda: _check_ollama(settings),
        lambda: _check_dbs(settings),
        _check_sqlite_vec,
        lambda: _check_disk(settings),
        lambda: _check_worker(settings),
    ]
    for check in checks:
        ok, lines = check()
        all_ok &= ok
        for line in lines:
            typer.echo(line)
    if not all_ok:
        typer.echo(typer.style("\ndoctor : échec — corriger les points ✗", fg=typer.colors.RED))
        raise typer.Exit(code=1)
    typer.echo(typer.style("\ndoctor : tout est vert", fg=typer.colors.GREEN, bold=True))
