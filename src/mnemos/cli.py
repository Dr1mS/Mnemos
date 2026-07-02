"""CLI Typer (§17).

Phase 0 : `mnemos doctor` + `mnemos version`. Les autres commandes
(serve, write, search, query, facts, consolidate, decay, export, stats)
arrivent avec leurs phases respectives.
"""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Callable
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
