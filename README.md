# Mnemos

Serveur de mémoire local pour agents LLM — quatre stores parallèles
(working / épisodique / sémantique / procédural), tagger de saillance,
router de lecture et consolidation asynchrone avec versioning des faits.

**Spec complète : [MNEMOS_SPEC.md](MNEMOS_SPEC.md)** (rev 1.2).
Choix des modèles : [poc/RESULTS.md](poc/RESULTS.md).

## Stack

Python 3.12 · FastAPI · SQLAlchemy 2 (async) · SQLite + sqlite-vec ·
Ollama (`bge-m3` + `qwen3:4b`, `think=false`).

## Setup rapide (Linux)

```sh
scripts/setup_ollama_models.sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
alembic upgrade head
mnemos doctor
```

## Recherche hybride épisodique

Score combiné = `0.7 * cosine(dense) + 0.3 * (1 - hamming/256)` —
dense = bge-m3 (1024-dim), sparse = hashing 256-bit avec bits temporels
(bucket 4h). Pondération configurable, cf. spec §8.2.
