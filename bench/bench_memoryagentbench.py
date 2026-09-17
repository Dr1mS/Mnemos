"""CLI d'évaluation inspirée de la taxonomie MemoryAgentBench (ICLR 2026)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from bench.backends.mnemos_backend import MnemosBackend
from bench.config import BenchConfig
from bench.memoryagentbench.runner import run_memoryagentbench

app = typer.Typer(help="Évaluation synthétique inspirée de MemoryAgentBench (ICLR 2026).")


@app.command()
def run(
    dry_run: bool = typer.Option(False, "--dry-run", "--stub", help="Exécution déterministe rapide sans appel LLM"),
    model: str = typer.Option("qwen3:4b", "--model", help="Modèle Ollama pour les inférences"),
    embed_model: str = typer.Option("bge-m3:latest", "--embed-model", help="Modèle d'embeddings"),
    bench_dir: Path = typer.Option(Path("data/bench_mabench"), "--bench-dir", help="Dossier isolé de données"),
    results_dir: Path = typer.Option(Path("bench/results"), "--results-dir", help="Dossier de sortie des rapports"),
) -> None:
    config = BenchConfig(
        bench_dir=bench_dir,
        embed_model=embed_model,
        llm_model=model,
        dry_run=dry_run,
        results_dir=results_dir,
    )
    backend = MnemosBackend(config)

    typer.echo("=" * 70)
    typer.echo("Lancement de l'évaluation inspirée de MemoryAgentBench (ICLR 2026)")
    typer.echo(f"Mode : {'DRY-RUN / STUB' if dry_run else 'RÉEL OLLAMA'}")
    typer.echo(f"Modèle : {model} | Embeddings : {embed_model}")
    typer.echo("=" * 70)

    report = asyncio.run(run_memoryagentbench(backend, config))

    typer.echo("\n" + "=" * 70)
    typer.echo(f"Score Global : {report.overall_score * 100:.1f}% ({report.passed_samples}/{report.total_samples})")
    typer.echo(f"  • Accurate Retrieval (AR)       : {report.ar_score * 100:.1f}%")
    typer.echo(f"  • Test-Time Learning (TTL)      : {report.ttl_score * 100:.1f}%")
    typer.echo(f"  • Long-Range Understanding (LRU): {report.lru_score * 100:.1f}%")
    typer.echo(f"  • Conflict Resolution (CR)      : {report.cr_score * 100:.1f}%")
    typer.echo(f"Rapport markdown sauvegardé dans : {config.results_dir / 'memoryagentbench_report.md'}")
    typer.echo("=" * 70)


if __name__ == "__main__":
    app()
