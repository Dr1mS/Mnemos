# 🧠 Mnemos

**[English](#english) · [Français](#français)**

<a name="english"></a>

**Long-term memory for LLM agents that works like yours — and runs entirely on your machine.**

Mnemos gives Claude (or any agent) a persistent local memory: it remembers what matters, forgets the noise, updates what changes without erasing history, and answers "where do I live?" six months later. No data ever leaves your machine — the models (Ollama), the databases (SQLite) and the memory itself all run locally, **even on a GPU-less PC**.

```
you : "It's official! I'm leaving Datalyse, I now work at Nexora."
                     │
                     ▼           works_at ─ Datalyse   [invalidated  2026-02→2026-07]
   [salience 0.95 → consolidation]  ─────▶ works_at ─ Nexora     [current]
                                            prefers  ─ tea        [current, untouched]

six months later : "where did I work before?"  →  "Datalyse, until July."
```

## Why four memory systems?

Because the brain doesn't have just one. A single "RAG + vector DB" store blends everything together: stale facts pollute current ones, precise memories merge into semantic mush, and nothing is ever forgotten. Mnemos mirrors the biological architecture:

| In your brain | In Mnemos | What it does |
|---|---|---|
| **Working memory** (prefrontal cortex) | `WorkingMemory` | Last 5 conversation turns, volatile, per session |
| **Hippocampus** (episodic memory) | `EpisodicStore` | Raw, timestamped, precise memories — "what happened on Tuesday" |
| **Dentate gyrus** (pattern separation) | 256-bit sparse coding | Similar memories stay distinct — orthogonal codes with temporal bits (4h buckets) |
| **Amygdala** (emotional tagging) | `SalienceTagger` | An LLM scores every memory: surprise, intensity, self-revelation. Bland content never gets consolidated |
| **Sleep / dreaming** (hippocampo-cortical consolidation) | `ConsolidationWorker` | Periodically, salient episodes are *replayed* and their facts extracted into semantic memory |
| **Cortex** (semantic memory) | `SemanticStore` | Durable facts — versioned: a new job **replaces** the old one (`works_at` is functional), a new preference **coexists** (`prefers` is multi) |
| **Active forgetting** | Salience-modulated decay | Bland memories fade then get archived; striking ones persist |
| **Basal ganglia** (skills) | `ProceduralStore` | Know-how (skills), consulted best-effort |

The golden rule, borrowed from neuroscience: **memories are never overwritten, they are superseded**. "I don't like coffee anymore" doesn't destroy the fact — it invalidates it with a date, and the full history stays queryable (`--history`).

## ✨ Features

- 🔒 **100% local** — Ollama (`bge-m3` + `qwen3:4b`) + SQLite/sqlite-vec. Validated on a CPU-only i7-6700 with 16 GB RAM
- ⚡ **Write path < 500 ms** — synchronous embedding, asynchronous LLM scoring (never blocking)
- 🔍 **Hybrid search** — `0.7·dense + 0.3·sparse + 0.1·recency`, with time-window filters
- 🗂️ **Versioned facts** — supersession on functional predicates, coexistence on multi, explicit retraction, full audit chain
- 🧭 **FR/EN router** — lexical classification ("yesterday" → episodic, "what do you know about" → semantic, "how did my preference change" → history)
- 🔌 **Native MCP** — 5 tools (`memory_query`, `memory_write`, `memory_forget`, `memory_facts`, `memory_consolidate`) for Claude Code & Claude Desktop
- 🛡️ **Measured defense in depth** — salience filters emotional-but-impersonal content, the extractor rejects hypotheticals/past-tense/third-party statements (bench: 0 traps end-to-end on an adversarial corpus)

## 🚀 Quickstart (Linux)

```sh
# 1. Local models (~3.7 GB)
scripts/setup_ollama_models.sh

# 2. Environment
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
alembic upgrade head
mnemos doctor          # everything should be green ✓

# 3. Try it
mnemos write "I prefer maté over tea."
mnemos search "maté"
mnemos query "what do you know about me?"
```

### Connecting Claude

**Claude Code**: the project's `.mcp.json` is enough — open a session in the repo and approve the `mnemos` server.

**Claude Desktop** (Linux beta ≥ June 2026) — in `~/.config/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "/path/to/Mnemos/.venv/bin/mnemos-mcp",
      "env": { "DATA_DIR": "/path/to/Mnemos/data/memory", "...": "..." }
    }
  }
}
```

**Automatic consolidation** — user systemd service running `mnemos worker` (hourly tick + monthly archive dump, single-instance lock): see `scripts/`.

## 🏗️ Architecture

```
                          ┌─────────────────────────────────────────┐
 POST /v1/episodes ──────▶│ embed bge-m3 + 256-bit sparse  (~230 ms)│──▶ episodic.db
 (or memory_write)   sync └─────────────────────────────────────────┘
                     async ┌────────────────┐   ┌──────────────────┐
                      └───▶│ salience queue │──▶│ qwen3:4b (amygd.)│──▶ UPDATE salience
                           └────────────────┘   └──────────────────┘
 ┌─ worker (sleep) ── every 60 min ────────────────────────────────────────────┐
 │  episodes with salience > 0.6 and age > 1h ──▶ fact+entity extraction       │
 │  ──▶ add_fact : inserted │ superseded │ duplicate  ──▶ semantic.db          │
 │  then decay + archiving of faded memories                                   │
 └──────────────────────────────────────────────────────────────────────────────┘
 POST /v1/query ──▶ FR/EN classifier ──▶ fan-out {episodic, semantic,
 (or memory_query)                       history, working, procedural}
```

Full specification: **[MNEMOS_SPEC.md](MNEMOS_SPEC.md)** (rev 1.2, French) · model benchmarks: **[poc/RESULTS.md](poc/RESULTS.md)**

> ⚠️ qwen3 family: `think=false` is **mandatory** — thinking mode breaks structured JSON under Ollama and multiplies CPU latency by 5-10×.

## 🧰 CLI & API

| CLI | HTTP API (`mnemos serve`, port 8765) |
|---|---|
| `mnemos write / search / query` | `POST /v1/episodes` · `GET /v1/episodes/search` · `POST /v1/query` |
| `mnemos facts --history` | `GET /v1/episodes/{id}` |
| `mnemos consolidate / decay / worker` | `POST /v1/admin/consolidate` · `POST /v1/admin/decay` |
| `mnemos stats / doctor / export / backup` | `GET /v1/health` · `POST /v1/sessions/{id}/reset` |

Optional auth via `X-API-Key` header. Atomic backups via `VACUUM INTO` (never raw-copy a WAL database).

## ✅ Done criterion & tests

```sh
python scripts/demo.py            # the acid test: 50 simulated messages over
                                  # 5 days, real salience + extraction,
                                  # 10 checks (versioning, multi, time
                                  # windows, noise…) — 10/10 on the CPU profile
pytest -m "not requires_ollama"   # ~150 fast tests without LLM (~4 s)
pytest                            # full suite with real Ollama (~3 min)
ruff check src tests && mypy      # lint + strict typing
```

Importing an existing memory (JSONL episodes + distilled facts): `scripts/import_dump.py --episodes … --seed-facts`.

## 🗺️ Roadmap

- [x] MVP: 4 stores + consolidation + router (spec §18, 7 phases)
- [x] MCP server (Claude Code, Claude Desktop)
- [x] Fact retraction — negation detection delegated to the consuming LLM via `memory_forget`
- [x] Recovery of lost salience scorings on worker restart
- [ ] Episodic fallback when semantic scores are low
- [ ] Extraction mode for non-conversational content (summaries)
- [ ] Semantic forgetting (confidence decay for unreinforced facts)
- [ ] claude.ai web/mobile connector (remote MCP + OAuth 2.1)

---
---

<a name="français"></a>

# 🧠 Mnemos — Français

**Une mémoire à long terme pour agents LLM, qui fonctionne comme la vôtre — et qui tourne entièrement chez vous.**

Mnemos donne à Claude (ou n'importe quel agent) une mémoire persistante locale : il retient ce qui compte, oublie le bruit, met à jour ce qui change sans écraser l'historique, et répond « où j'habite ? » six mois plus tard. Aucune donnée ne quitte votre machine — les modèles (Ollama), les bases (SQLite) et la mémoire vivent en local, y compris sur un PC **sans GPU**.

```
vous : "Ça y est, j'ai signé ! Je quitte Datalyse, je bosse chez Nexora."
                     │
                     ▼           works_at ─ Datalyse   [invalidé  2026-02→2026-07]
   [salience 0.95 → consolidation]  ─────▶ works_at ─ Nexora     [courant]
                                            prefers  ─ thé        [courant, intact]

six mois plus tard : "où est-ce que je bossais avant ?"  →  "Datalyse, jusqu'en juillet."
```

## Pourquoi quatre mémoires ?

Parce que le cerveau n'en a pas qu'une. Un store unique type "RAG + vector DB" mélange tout : les faits périmés polluent les faits courants, les souvenirs précis fusionnent en bouillie sémantique, et rien n'est jamais oublié. Mnemos reprend l'architecture biologique :

| Dans votre cerveau | Dans Mnemos | Ce que ça fait |
|---|---|---|
| **Mémoire de travail** (cortex préfrontal) | `WorkingMemory` | Les 5 derniers tours de conversation, volatile, par session |
| **Hippocampe** (mémoire épisodique) | `EpisodicStore` | Les souvenirs bruts, datés, précis — « ce qui s'est passé mardi » |
| **Gyrus denté** (pattern separation) | Sparse coding 256-bit | Deux souvenirs similaires restent distincts — codes orthogonaux avec bits temporels (bucket 4h) |
| **Amygdale** (marquage émotionnel) | `SalienceTagger` | Un LLM score chaque souvenir : surprise, intensité, révélation personnelle. Ce qui est fade ne sera jamais consolidé |
| **Sommeil / rêve** (consolidation hippocampo-corticale) | `ConsolidationWorker` | Périodiquement, les épisodes saillants sont *rejoués* et leurs faits extraits vers la mémoire sémantique |
| **Cortex** (mémoire sémantique) | `SemanticStore` | Les faits durables — versionnés : un nouveau job **remplace** l'ancien (`works_at` est functional), une nouvelle préférence **coexiste** (`prefers` est multi) |
| **Oubli actif** | Decay modulé par salience | Les souvenirs fades s'estompent puis s'archivent ; les marquants persistent |
| **Ganglions de la base** (habiletés) | `ProceduralStore` | Les savoir-faire (skills), consultés en best-effort |

La règle d'or héritée de la neuro : **on n'écrase jamais un souvenir, on le supersède**. « Je n'aime plus le café » ne détruit pas le fait — il l'invalide avec la date, et l'historique complet reste interrogeable (`--history`).

## ✨ Fonctionnalités

- 🔒 **100 % local** — Ollama (`bge-m3` + `qwen3:4b`) + SQLite/sqlite-vec. Validé sur un i7-6700 CPU-only, 16 GB RAM
- ⚡ **Write path < 500 ms** — embedding synchrone, scoring LLM asynchrone (jamais bloquant)
- 🔍 **Recherche hybride** — `0.7·dense + 0.3·sparse + 0.1·récence`, avec fenêtres temporelles
- 🗂️ **Faits versionnés** — supersession sur les prédicats fonctionnels, coexistence sur les multi, rétractation explicite, chaîne d'audit complète
- 🧭 **Router FR/EN** — classification lexicale (« hier » → épisodique, « qu'est-ce que tu sais sur » → sémantique, « comment ma préférence a évolué » → historique)
- 🔌 **MCP natif** — 5 tools (`memory_query`, `memory_write`, `memory_forget`, `memory_facts`, `memory_consolidate`) pour Claude Code & Claude Desktop
- 🛡️ **Défense en profondeur mesurée** — la salience filtre l'émotionnel-non-personnel, l'extracteur rejette hypothétiques/temps passé/tiers (bench : 0 piège end-to-end sur corpus adversarial)

## 🚀 Démarrage rapide (Linux)

```sh
# 1. Modèles locaux (~3.7 GB)
scripts/setup_ollama_models.sh

# 2. Environnement
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"
alembic upgrade head
mnemos doctor          # tout doit être vert ✓

# 3. Essayer
mnemos write "Je préfère le maté au thé."
mnemos search "maté"
mnemos query "qu'est-ce que tu sais sur moi ?"
```

### Brancher Claude

**Claude Code** : le `.mcp.json` du projet suffit — ouvrez une session dans le repo et approuvez le serveur `mnemos`.

**Claude Desktop** (Linux beta ≥ juin 2026) — dans `~/.config/Claude/claude_desktop_config.json` : voir l'exemple de la section anglaise.

**Consolidation automatique** — service systemd user (`mnemos worker` : tick horaire + dump mensuel des archives, verrou d'instance unique) : voir `scripts/`.

## 🧰 CLI & API, critère "done", feuille de route

Identiques à la section anglaise ci-dessus — `mnemos --help` pour le détail des commandes, `python scripts/demo.py` pour le juge de paix (10/10 checks sur le profil CPU), et la roadmap est tenue à jour dans la version anglaise.

Spécification complète : **[MNEMOS_SPEC.md](MNEMOS_SPEC.md)** (rev 1.2) · benchs des modèles : **[poc/RESULTS.md](poc/RESULTS.md)**

---

*Mnemos — the Titaness of memory, mother of the Muses. A memory worth keeping is a memory worth versioning.*
