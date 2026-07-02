# Mnemos

Serveur de mémoire local pour agents LLM — quatre stores parallèles avec des
dynamiques différentes, inspirés des systèmes de mémoire biologiques :

| Store | Dynamique | Persistance |
|---|---|---|
| **Working** | 5 derniers tours par session, éviction LRU | RAM |
| **Épisodique** | souvenirs bruts, décroissance modulée par saillance | SQLite + vec |
| **Sémantique** | faits versionnés (jamais écrasés : supersédés) | SQLite + vec |
| **Procédural** | skills enregistrés manuellement | filesystem |

Un **tagger de saillance** (LLM) filtre ce qui mérite d'être retenu, un
**router** classifie les questions (FR/EN) et interroge les bons stores, et
une **boucle de consolidation** asynchrone extrait les faits des épisodes
saillants — avec résolution de conflit : un nouveau job *remplace* l'ancien
(`works_at` est functional), une nouvelle préférence *coexiste* avec les
autres (`prefers` est multi).

**Spec complète : [MNEMOS_SPEC.md](MNEMOS_SPEC.md)** (rev 1.2).
Choix des modèles : [poc/RESULTS.md](poc/RESULTS.md).

## Stack

Python 3.12 · FastAPI · SQLAlchemy 2 async · SQLite + sqlite-vec · Alembic
multi-DB · structlog · APScheduler · Ollama (`bge-m3` + `qwen3:4b`,
**`think=false` obligatoire** — le mode thinking casse le JSON structuré et
multiplie la latence CPU par 5-10×).

Deux profils matériels : dev Linux CPU-only (16 GB RAM) et déploiement
Windows RTX 3070 Ti. Un seul LLM résident (~3.7 GB avec l'embedder).

## Setup (Linux)

```sh
scripts/setup_ollama_models.sh          # bge-m3 + qwen3:4b
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
alembic upgrade head                    # migre les deux DBs
mnemos doctor                           # tout doit être vert
```

## Utilisation

```sh
mnemos serve                            # API sur 127.0.0.1:8765
mnemos worker                           # consolidation périodique (lock fichier)

mnemos write "Je préfère le maté au thé." --session s1
mnemos search "maté" -k 5
mnemos query "où j'habite ?"            # routage auto multi-store
mnemos facts --subject user --predicate works_at --history
mnemos consolidate                      # run manuel du worker
mnemos stats                            # + compteur de doublons (gate qualité)
mnemos export --out backup.jsonl
mnemos backup --out ./backups           # VACUUM INTO atomique
```

API : `POST /v1/episodes`, `GET /v1/episodes/search`, `POST /v1/query`,
`GET /v1/facts…` *(à venir)*, `POST /v1/sessions/{id}/reset`,
`GET /v1/health`, `POST /v1/admin/{consolidate,decay}`.
Auth optionnelle par header `X-API-Key` (si `API_KEY` est défini).

## Architecture du write path

```
POST /v1/episodes ──sync──▶ embedding (bge-m3, ~230 ms) + sparse ─▶ SQLite
                └──async──▶ queue salience ─▶ qwen3:4b ─▶ update (jamais bloquant)
```

L'épisode est cherchable immédiatement ; `salience: null` tant que le
scoring n'est pas passé. Cible : write path p50 < 500 ms (validée par test).

## Recherche hybride épisodique

Score = `0.7·cosine(dense) + 0.3·(1 − hamming/256) + 0.1·récence` —
dense = bge-m3 (1024-dim), sparse = hashing 256-bit dont 32 bits temporels
(bucket 4h, pattern separation pragmatique). Cf. spec §8.2.

## Critère "done"

```sh
python scripts/demo.py
```

Rejoue 50 messages d'une utilisatrice fictive (changement de job,
déménagement, préférences, bruit) sur 5 jours simulés, consolide, puis
vérifie 10 checks : versioning des faits, coexistence des multi, fenêtres
temporelles, discrimination de la salience… Vert = MVP livré.

## Tests

```sh
pytest -m "not requires_ollama"   # rapide, sans LLM (~3 s)
pytest                            # complet avec Ollama réel (~3 min)
ruff check src tests && mypy      # lint + typage strict
```
