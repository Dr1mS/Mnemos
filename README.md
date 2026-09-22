# 🧠 Mnemos

**[English](#english) · [Français](#français)**

---

## 📋 For Agent Memory Leaderboard reviewers

> You are on branch `feat/aml-adapter`, which carries the code evaluated in the
> **Agent Memory Challenge, Cycle 2**. Everything below in this section is written for
> you; the rest of the README is the general project documentation.

**Declared commit:** `2f6539cd478c3d42afc952765b89353bb7b81016` — the evaluated code.
Any later commit on this branch touches documentation only; you can verify that the
evaluated code has not moved with `git diff 2f6539c..HEAD -- src/`, which is empty.
**Track:** Industry · **Endpoints:** `POST /add`, `POST /search`, `GET /health` (unauthenticated)
**Auth:** Bearer, Token or X-Api-Key, same key · **Declared concurrency:** Add 16, Search 16, top_k 100

### What the evaluated configuration actually runs

| | |
|---|---|
| **Models** | `bge-m3` embeddings only. **No LLM is called during Add or Search**, and no external API. |
| **Embedding server** | `llama-server` (llama.cpp) in direct mode, 16 parallel slots, 8192 context and physical batch per slot. |
| **Storage** | SQLite + `sqlite-vec`, one transaction per Add; HTTP 200 is returned only after commit, so a memory is searchable immediately. |
| **Isolation** | `user_id` is the only boundary. `session_id` groups episodes but never filters Search. |
| **Retrieval** | Hybrid: `0.7·dense + 0.3·sparse + 0.1·recency`, deduplicated, ranked, at most `top_k`. Search **never generates answers**; `options` are accepted and ignored. |
| **Disabled for this evaluation** | The LLM consolidation layer (salience scoring, fact extraction). Measured as no gain in retrieval or answer accuracy while halving throughput. See `AML_DEPLOYMENT_WINDOWS.md`. |

### Verify the contract yourself

```bash
python scripts/aml_selftest.py --url <deployment-url> --key <api-key>
```

Checks, in order: `/health` answers 2xx **without** authentication; `/add` is synchronous
and echoes `request_id`, `user_id` and `session_id` unchanged with `success: true`;
`/search` returns a `data` array of at most `top_k` items, each with a non-empty `id` and
`content`; a different `user_id` sees nothing; an invalid key is refused with 401.

### Two properties worth checking explicitly

- **Add is idempotent on `request_id`, per `user_id`.** Your contract replays the same
  logical write up to 32 times; the identifier is recorded *inside the same transaction*
  as the memories, so a replay returns success without duplicating anything.
  Test: `tests/integration/test_aml_adapter.py::test_aml_add_rejeu_ne_duplique_pas`.
- **Oversized inputs degrade instead of failing.** A message too long for the embedding
  model is truncated *for embedding only* and logged; the stored content stays whole.
  Test: `tests/unit/test_llamacpp_client.py`.

### Measured capacity

Through the public endpoint, at the declared concurrency: 400 Add and 800 Search
requests, **zero errors**. Add 12.0 messages/s, Search 20.3 requests/s (top_k=100).
Both are bounded by the embedding model on the host GPU, not by our code: with
embeddings stubbed out the storage path sustains 615 messages/s.

### Reproduce

`AML_DEPLOYMENT_WINDOWS.md` is the full deployment guide (environment, scheduled tasks,
reverse proxy). `MNEMOS_SPEC.md` is the design spec. Benchmarks and their raw results are
in `bench/` and `bench/results/gpu/`. Tests: `pytest` — 252 collected, 242 passing with no
network access at all; the 10 marked `requires_ollama` are skipped unless a local Ollama is
running, and they cover the consolidation path that this evaluation does not use.

### Third-party work

`bge-m3` (Chen et al., 2024, arXiv:2402.03216, MIT) · `llama.cpp` / `llama-server`
(ggml-org, MIT, used unmodified — command-line options only) · `sqlite-vec` (Alex Garcia).
Everything else is original work by a single author. Public datasets used for our own
benchmarking (LoCoMo, PersonaMem) are cited in `bench/`; **no AML evaluation data was used
for development or tuning**. Developed with AI coding assistants (Claude Code, Antigravity).

---

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
- 🔌 **Native MCP** — 5 tools (`memory_query`, `memory_write`, `memory_forget`, `memory_facts`, `memory_consolidate`) for Antigravity CLI (`agy`), Claude Code & Claude Desktop
- 🏛️ **Multi-tenant** — a `tenant` dimension isolates parallel memories (personal, an app, an NPC…) with strict end-to-end sealing. Optional everywhere, defaults to `user` — existing clients are untouched. Contract: **[MNEMOS_API.md](MNEMOS_API.md)**
- 🌌 **3D visualizer** — your memory as a living constellation: entities as stars, facts as glowing links, superseded facts as tethered ghosts, memories as dust that literally fades with decay
- 🛡️ **Measured defense in depth** — salience filters emotional-but-impersonal content, the extractor rejects hypotheticals/past-tense/third-party statements (bench: 0 traps end-to-end on an adversarial corpus)
- ❤️‍🩹 **Operational health** — `GET /v1/health` probes both DBs *and* the Ollama embedding endpoint (the outage that breaks read *and* write), naming the failing dependency — 2 s timeout, meant to be polled every tick

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

### Connecting Antigravity CLI, Claude & AI Agents

**Antigravity CLI (`agy`)**:
- **Workspace plugin** (automatic): the repository contains `.agents/plugins/mnemos/` — opening `agy` in this repository automatically discovers and connects the `mnemos` MCP server.
- **Global config**: add `mnemos` to `~/.gemini/config/mcp_config.json`:
  ```json
  {
    "mcpServers": {
      "mnemos": {
        "command": "/path/to/Mnemos/.venv/bin/mnemos-mcp",
        "env": {
          "DATA_DIR": "/path/to/Mnemos/data/memory"
        }
      }
    }
  }
  ```

**Claude Code**: the project's `.mcp.json` is enough — open a session in the repo and approve the `mnemos` server.

**Claude Desktop** (Linux beta ≥ June 2026 / Windows) — in `~/.config/Claude/claude_desktop_config.json` (or `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

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

**Automatic background service**:
- **Linux**: user systemd service running `mnemos worker` (hourly tick + monthly archive dump, single-instance lock).
- **Windows**: scheduled task via PowerShell (`scripts/register_task.ps1`) or background scripts (`scripts/serve.ps1` / `scripts/serve.bat`).

## 🌌 Memory Constellation — the 3D visualizer

Watch your memory live: a weightless force-graph where **entities are stars** (sized by how often they recur), **facts are glowing links** colored by family, **superseded facts drift behind their successor as tethered ghosts**, and **raw memories are orbiting dust** — opacity is their decay state, halo is their salience. Faded memories literally go dark.

```sh
mnemos serve
# then open  →  http://127.0.0.1:8765/viz
```

Hover for tooltips, click for the inspector (including a fact's full version history), search to highlight, filter by family or minimum salience. Single-file page (three.js + UnrealBloom + 3d-force-graph via pinned CDNs), fed by `GET /v1/graph`, refreshed every 30 s — new memories pulse into existence. Works standalone with demo data if the API is unreachable.

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

 every store scopes by tenant (default 'user') ──▶ strict isolation, no cross-tenant read/write
```

Full specification: **[MNEMOS_SPEC.md](MNEMOS_SPEC.md)** (rev 1.2, French) · model benchmarks: **[poc/RESULTS.md](poc/RESULTS.md)**

> ⚠️ qwen3 family: `think=false` is **mandatory** — thinking mode breaks structured JSON under Ollama and multiplies CPU latency by 5-10×.

## 🧰 CLI & API

| CLI | HTTP API (`mnemos serve`, port 8765) |
|---|---|
| `mnemos write / search / query` | `POST /v1/episodes` · `GET /v1/episodes/search` · `POST /v1/query` |
| `mnemos facts --history` | `GET /v1/facts` · `GET /v1/facts/history` · `GET /v1/episodes/{id}` |
| `mnemos consolidate / decay / worker` | `POST /v1/admin/consolidate` · `POST /v1/admin/decay` |
| `mnemos stats / doctor / export / backup` | `GET /v1/health` · `POST /v1/sessions/{id}/reset` |

Every endpoint takes an optional `tenant` (defaults to `user`). Optional auth via `X-API-Key` header. Atomic backups via `VACUUM INTO` (never raw-copy a WAL database). **Full HTTP contract: [MNEMOS_API.md](MNEMOS_API.md).**

## ✅ Done criterion & tests

```sh
python scripts/demo.py            # the acid test: 50 simulated messages over
                                  # 5 days, real salience + extraction,
                                  # 10 checks (versioning, multi, time
                                  # windows, noise…) — 10/10 on the CPU profile
pytest -m "not requires_ollama"   # ~175 fast tests without LLM (incl. tenant isolation)
pytest                            # full suite with real Ollama (~3 min)
ruff check src tests && mypy      # lint + strict typing
```

Importing an existing memory (JSONL episodes + distilled facts): `scripts/import_dump.py --episodes … --seed-facts`.

## 🗺️ Roadmap

- [x] MVP: 4 stores + consolidation + router (spec §18, 7 phases)
- [x] MCP server (Claude Code, Claude Desktop)
- [x] Fact retraction — negation detection delegated to the consuming LLM via `memory_forget`
- [x] Recovery of lost salience scorings on worker restart
- [x] 3D visualizer — Memory Constellation (`/viz`)
- [x] Multi-tenant — isolated parallel memories, canonical subject per tenant, tenant-scoped `/v1/health` (contract in `MNEMOS_API.md`)
- [ ] Tenant-aware salience (the tagger is still user-centric — a non-personal tenant under-scores)
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
- 🔌 **MCP natif** — 5 tools (`memory_query`, `memory_write`, `memory_forget`, `memory_facts`, `memory_consolidate`) pour Antigravity CLI (`agy`), Claude Code & Claude Desktop
- 🏛️ **Multi-tenant** — une dimension `tenant` isole des mémoires parallèles (perso, une app, un NPC…) avec étanchéité stricte end-to-end. Optionnel partout, défaut `user` — les clients existants ne changent pas. Contrat : **[MNEMOS_API.md](MNEMOS_API.md)**
- 🌌 **Visualiseur 3D** — votre mémoire en constellation vivante : entités-étoiles, faits-liens lumineux, faits supersédés en fantômes rattachés, souvenirs en poussière qui s'éteint littéralement avec le decay
- 🛡️ **Défense en profondeur mesurée** — la salience filtre l'émotionnel-non-personnel, l'extracteur rejette hypothétiques/temps passé/tiers (bench : 0 piège end-to-end sur corpus adversarial)
- ❤️‍🩹 **Santé opérationnelle** — `GET /v1/health` sonde les deux DB *et* l'endpoint d'embedding Ollama (la panne qui casse lecture *et* écriture), en nommant la dépendance fautive — timeout 2 s, pensé pour être appelé à chaque tick

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

### Brancher Antigravity CLI, Claude & les agents

**Antigravity CLI (`agy`)** :
- **Plugin de workspace** (automatique) : le repo contient `.agents/plugins/mnemos/` — lancer `agy` dans ce projet détecte et active automatiquement le serveur MCP `mnemos`.
- **Config globale** : déclarer `mnemos` dans `~/.gemini/config/mcp_config.json` (voir l'exemple JSON de la section anglaise).

**Claude Code** : le `.mcp.json` du projet suffit — ouvrez une session dans le repo et approuvez le serveur `mnemos`.

**Claude Desktop** (Linux beta ≥ juin 2026 / Windows) — dans `~/.config/Claude/claude_desktop_config.json` (ou `%APPDATA%\Claude\claude_desktop_config.json` sous Windows) : voir l'exemple de la section anglaise.

**Service d'arrière-plan & consolidation automatique** :
- **Linux** : service systemd user (`mnemos worker` : tick horaire + dump mensuel des archives, verrou d'instance unique).
- **Windows** : tâche planifiée via PowerShell (`scripts/register_task.ps1`) ou scripts d'arrière-plan (`scripts/serve.ps1` / `scripts/serve.bat`).

## 🌌 Memory Constellation — le visualiseur 3D

Regardez votre mémoire vivre : un graphe en apesanteur où **les entités sont des étoiles** (taille selon leur récurrence), **les faits des liens lumineux** colorés par famille, **les faits supersédés des fantômes** qui flottent derrière leur successeur, et **les souvenirs bruts une poussière en orbite** — opacité = decay, halo = salience. Les souvenirs oubliés s'éteignent littéralement.

```sh
mnemos serve
# puis ouvrir  →  http://127.0.0.1:8765/viz
```

Survol pour les tooltips, clic pour l'inspecteur (avec l'historique complet des versions d'un fait), recherche, filtres par famille et salience minimum. Page single-file (three.js + UnrealBloom + 3d-force-graph via CDN épinglés), nourrie par `GET /v1/graph`, rafraîchie toutes les 30 s — les nouveaux souvenirs apparaissent en pulsant. Fonctionne en démo autonome si l'API est injoignable.

## 🧰 CLI & API, critère "done", feuille de route

Identiques à la section anglaise ci-dessus — `mnemos --help` pour le détail des commandes, `python scripts/demo.py` pour le juge de paix (10/10 checks sur le profil CPU), et la roadmap est tenue à jour dans la version anglaise.

**Multi-tenant** : chaque endpoint accepte un `tenant` optionnel (défaut `user`), avec isolation stricte. Contrat HTTP complet consommé par les intégrations : **[MNEMOS_API.md](MNEMOS_API.md)**. Smoke test end-to-end : `python scripts/smoke_tenant.py` (Ollama réel).

Spécification complète : **[MNEMOS_SPEC.md](MNEMOS_SPEC.md)** (rev 1.2) · benchs des modèles : **[poc/RESULTS.md](poc/RESULTS.md)**


## License / Licence

Apache License 2.0 — see [LICENSE](LICENSE). Copyright 2026 Adrien.M (Dr1mS).

---

*Mnemos — the Titaness of memory, mother of the Muses. A memory worth keeping is a memory worth versioning.*
