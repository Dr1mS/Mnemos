# 🧠 Mnemos

**[English](#english) · [Français](#français)**

---

## 📋 Agent Memory Leaderboard (Cycle 2)

> Mnemos took part in the **Agent Memory Challenge, Cycle 2** (Textual track,
> Industry board). The evaluated code lives on branch `feat/aml-adapter`, which stays
> frozen at the last declared commit during the official review. `main` continues from
> that commit.

| Version | Fixed commit | Full task |
|---|---|---|
| `0.1.0-aml` | `2f6539cd478c3d42afc952765b89353bb7b81016` | `teval_a880b13143f5d169` (23 September 2026) |
| `0.2.0-aml` | `3fed00c002aa07b035d2186959be9b431e896200` | `teval_f5766d6ba6ad2914` (28 October 2026) |

`git checkout 3fed00c` gives exactly the code served during the second Full; `GET /health`
reports `"version": "0.2.0"`. The official leaderboard is expected in mid-November 2026.

**Track:** Industry · **Endpoints:** `POST /add`, `POST /search`, `GET /health` (unauthenticated)
**Auth:** Bearer, Token or X-Api-Key, same key · **Declared concurrency:** Add 16, Search 16, top_k 100

### What the evaluated configuration actually runs (0.2.0-aml)

| | |
|---|---|
| **Models** | `bge-m3` embeddings only. **No LLM is called during Add or Search**, and no external API. |
| **Embedding server** | `llama-server` (llama.cpp) in direct mode, 16 parallel slots, 8192 context and physical batch per slot. |
| **Storage** | SQLite + `sqlite-vec`, one transaction per Add; HTTP 200 is returned only after commit, so a memory is searchable immediately. |
| **Isolation** | `user_id` is the only boundary. `session_id` groups episodes but never filters Search. |
| **Retrieval** | Hybrid: dense similarity, lexical token overlap and recency (weights 0.7 / 0.3 / 0.1, normalised), deduplicated, ranked, at most `top_k`. Search **never generates answers**; `options` are accepted and ignored. |
| **Write-time rules** | Forget requests and relative dates, see below. Both are deterministic and rule-based. |
| **Disabled for this evaluation** | The LLM consolidation layer (salience scoring, fact extraction). Measured as no gain in retrieval or answer accuracy while halving throughput. See `docs/AML_DEPLOYMENT_WINDOWS.md`. |

### Changes since 0.1.0-aml

- **Lexical ranking.** The 0.1.0 lexical term was a Hamming similarity on 256-bit hashed
  codes: it favoured short messages, and 32 of its bits encoded the time of the query, so
  the same query could rank differently depending on the hour. It now measures the share
  of the query's tokens present in the memory.
- **Forget requests.** The contract has no delete operation, so "please forget that I…"
  arrives as an ordinary message. On Add, such a request (English or French) removes at
  most 5 earlier **assistant** messages of the same `user_id` that match its target
  (cosine ≥ 0.65). User messages, the request itself and the acknowledgement are kept.
- **Relative dates resolved at write time.** When a message carries its timestamp,
  relative days, months and years get the absolute date appended: "yesterday
  (27 August 2023)". Indexing uses the raw message; nothing is added at Search time.
  `RELATIVE_DATES_ANNOTATION=false` turns it off.

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

Through the public endpoint, at the declared concurrency, on the fixed commit of 0.2.0-aml
and an empty database (22 October 2026): 400 Add and 800 Search requests, **zero errors**.
Add 11.5 messages/s (22.3 s p50 / 25.8 s p95 per request of ~16 messages), Search
28.6 requests/s (543 ms p50 / 963 ms p95, top_k=100). For 0.1.0-aml: Add 12.0 messages/s,
Search 20.3 requests/s.
Both are bounded by the embedding model on the host GPU, not by our code: with
embeddings stubbed out the storage path sustains 615 messages/s.

### Reproduce

`docs/AML_DEPLOYMENT_WINDOWS.md` is the full deployment guide (environment, scheduled tasks,
reverse proxy). `MNEMOS_SPEC.md` is the design spec. Benchmarks and their raw results are
in `bench/` and `bench/results/` (one folder per topic, indexed in `bench/results/README.md`).
Tests: `pytest` — 388 collected. Without network access, the 10 marked `requires_ollama`
are skipped: they need a local Ollama and cover the consolidation path that this evaluation
does not use.

### Third-party work

`bge-m3` (Chen et al., 2024, arXiv:2402.03216, MIT) · `llama.cpp` / `llama-server`
(ggml-org, MIT, used unmodified — command-line options only) · `sqlite-vec` (Alex Garcia).
Everything else is original work by a single author. Development used public data only:
LoCoMo (including LoCoMo-Refined), PersonaMem (v1 and v2), LongMemEval, BEAM, CL-bench and
MemoryAgentBench, cited in `bench/`, with the answer and judge prompts published in the AML
public repository. **No AML evaluation data and no private result from the platform was
used for development or tuning**. Developed with AI coding assistants (Claude Code, Antigravity).

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
| **Dentate gyrus** (pattern separation) | 256-bit sparse coding | A hashed lexical code per memory; search scores the share of the query's tokens it contains. Temporal bits (4h buckets) are still encoded but no longer scored |
| **Amygdala** (emotional tagging) | `SalienceTagger` | An LLM scores every memory: surprise, intensity, self-revelation. Bland content never gets consolidated |
| **Sleep / dreaming** (hippocampo-cortical consolidation) | `ConsolidationWorker` | Periodically, salient episodes are *replayed* and their facts extracted into semantic memory |
| **Cortex** (semantic memory) | `SemanticStore` | Durable facts — versioned: a new job **replaces** the old one (`works_at` is functional), a new preference **coexists** (`prefers` is multi) |
| **Active forgetting** | Salience-modulated decay | Bland memories fade then get archived; striking ones persist |
| **Basal ganglia** (skills) | `ProceduralStore` | Know-how (skills), consulted best-effort |

The golden rule, borrowed from neuroscience: **memories are never overwritten, they are superseded**. "I don't like coffee anymore" doesn't destroy the fact — it invalidates it with a date, and the full history stays queryable (`--history`).

> ⚠️ **Status of consolidation (October 2026).** The layers above the episodic store
> (salience, sleep-time fact extraction, versioned facts) are the least proven part of
> Mnemos. In our autumn 2026 measurements they brought no gain in answer accuracy:
> on LoCoMo the facts were inert, and on a hard knowledge-update set the answer model
> copied stale or misattributed facts (4/13 correct with facts, 13/13 with episodes
> alone). Extraction also ran at under 1 episode/s, against ~5/s for plain ingestion.
> They were therefore disabled for both AML evaluations and are being re-evaluated;
> the episodic path (`/add`, `/search`) is the one validated at scale.

## ✨ Features

- 🔒 **100% local** — Ollama (`bge-m3` + `qwen3:4b`) or llama.cpp's `llama-server` for embeddings + SQLite/sqlite-vec. Validated on a CPU-only i7-6700 with 16 GB RAM
- ⚡ **Write path < 500 ms** — synchronous embedding, asynchronous LLM scoring (never blocking)
- 🔍 **Hybrid search** — dense similarity, lexical token overlap and recency (weights 0.7 / 0.3 / 0.1, normalised), with time-window filters
- 🗂️ **Versioned facts** — supersession on functional predicates, coexistence on multi, explicit retraction, full audit chain
- 🧭 **FR/EN router** — lexical classification ("yesterday" → episodic, "what do you know about" → semantic, "how did my preference change" → history)
- 🔌 **Native MCP** — 5 tools (`memory_query`, `memory_write`, `memory_forget`, `memory_facts`, `memory_consolidate`) for Antigravity CLI (`agy`), Claude Code & Claude Desktop
- 🏛️ **Multi-tenant** — a `tenant` dimension isolates parallel memories (personal, an app, an NPC…) with strict end-to-end sealing. Optional everywhere, defaults to `user` — existing clients are untouched. Contract: **[docs/MNEMOS_API.md](docs/MNEMOS_API.md)**
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

### Faster embeddings with llama.cpp

With a GPU, `llama-server` serves `bge-m3` with parallel slots and is the backend that
sustained both AML evaluations. Start it, then point Mnemos at it in `.env`:

```sh
llama-server --model <bge-m3 GGUF> --host 127.0.0.1 --port 8899 \
  -c 131072 -np 16 --embedding -b 8192 -ub 8192 --no-webui
```

```ini
EMBED_BACKEND=llamacpp
LLAMACPP_HOST=http://127.0.0.1:8899
```

Ollama stays the default (`EMBED_BACKEND=ollama`). Supervision under Windows:
`scripts/register_llamacpp_task.ps1`, details in [docs/AML_DEPLOYMENT_WINDOWS.md](docs/AML_DEPLOYMENT_WINDOWS.md).

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

Full specification: **[MNEMOS_SPEC.md](MNEMOS_SPEC.md)** (rev 1.2, French) · model benchmarks: **[archive/poc/RESULTS.md](archive/poc/RESULTS.md)**

> ⚠️ qwen3 family: `think=false` is **mandatory** — thinking mode breaks structured JSON under Ollama and multiplies CPU latency by 5-10×.

## 🧰 CLI & API

| CLI | HTTP API (`mnemos serve`, port 8765) |
|---|---|
| `mnemos write / search / query` | `POST /v1/episodes` · `GET /v1/episodes/search` · `POST /v1/query` |
| `mnemos facts --history` | `GET /v1/facts` · `GET /v1/facts/history` · `GET /v1/episodes/{id}` |
| `mnemos consolidate / decay / worker` | `POST /v1/admin/consolidate` · `POST /v1/admin/decay` |
| `mnemos stats / doctor / export / backup` | `GET /v1/health` · `POST /v1/sessions/{id}/reset` |
| — | `POST /add` · `POST /search` · `GET /health` (AML contract) |

Every endpoint takes an optional `tenant` (defaults to `user`). Optional auth via `X-API-Key` header. Atomic backups via `VACUUM INTO` (never raw-copy a WAL database). **Full HTTP contract: [docs/MNEMOS_API.md](docs/MNEMOS_API.md).**

`POST /add` and `POST /search` follow the AML contract: `user_id` is the boundary, Add is
idempotent on `request_id`, and Search returns stored memories, never generated answers.
The forget-request rule and the resolution of relative dates apply **only on `/add`**,
not on `/v1/episodes` nor on the MCP tools, which still use the earlier write path.

## ✅ Done criterion & tests

```sh
python scripts/demo.py            # the acid test: 50 simulated messages over
                                  # 5 days, real salience + extraction,
                                  # 10 checks (versioning, multi, time
                                  # windows, noise…) — 10/10 on the CPU profile
pytest -m "not requires_ollama"   # 378 tests without LLM (incl. tenant isolation)
pytest                            # full suite, 388 tests, with real Ollama (~3 min)
ruff check src tests && mypy      # lint + strict typing
```

Importing an existing memory (JSONL episodes + distilled facts): `scripts/import_dump.py --episodes … --seed-facts`.

## 🗺️ Roadmap

- [x] MVP: 4 stores + consolidation + router (spec §18, 7 phases)
- [x] MCP server (Claude Code, Claude Desktop)
- [x] Fact retraction — negation detection delegated to the consuming LLM via `memory_forget`
- [x] Recovery of lost salience scorings on worker restart
- [x] 3D visualizer — Memory Constellation (`/viz`)
- [x] Multi-tenant — isolated parallel memories, canonical subject per tenant, tenant-scoped `/v1/health` (contract in `docs/MNEMOS_API.md`)
- [x] AML contract (`/add`, `/search`), evaluated twice in the Agent Memory Challenge, Cycle 2
- [ ] Partition the vector index by `tenant` (today every kNN scans all tenants' vectors)
- [ ] Size the database connection pool to the declared concurrency (fixed at 5 + 10 today)
- [ ] MCP tools on the validated path: `memory_add` / `memory_search` over `/add` and `/search`
- [ ] Re-evaluate consolidation: keep, rework or drop (see the status note above)
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
| **Gyrus denté** (pattern separation) | Sparse coding 256-bit | Un code lexical haché par souvenir ; la recherche note la part des mots de la requête qu'il contient. Les bits temporels (tranches de 4 h) sont encore encodés mais ne comptent plus dans le score |
| **Amygdale** (marquage émotionnel) | `SalienceTagger` | Un LLM score chaque souvenir : surprise, intensité, révélation personnelle. Ce qui est fade ne sera jamais consolidé |
| **Sommeil / rêve** (consolidation hippocampo-corticale) | `ConsolidationWorker` | Périodiquement, les épisodes saillants sont *rejoués* et leurs faits extraits vers la mémoire sémantique |
| **Cortex** (mémoire sémantique) | `SemanticStore` | Les faits durables — versionnés : un nouveau job **remplace** l'ancien (`works_at` est functional), une nouvelle préférence **coexiste** (`prefers` est multi) |
| **Oubli actif** | Decay modulé par salience | Les souvenirs fades s'estompent puis s'archivent ; les marquants persistent |
| **Ganglions de la base** (habiletés) | `ProceduralStore` | Les savoir-faire (skills), consultés en best-effort |

La règle d'or héritée de la neuro : **on n'écrase jamais un souvenir, on le supersède**. « Je n'aime plus le café » ne détruit pas le fait — il l'invalide avec la date, et l'historique complet reste interrogeable (`--history`).

> ⚠️ **Statut de la consolidation (octobre 2026).** Les couches au-dessus de la mémoire
> épisodique (salience, extraction de faits pendant le « sommeil », faits versionnés) sont
> la partie la moins éprouvée de Mnemos. Nos mesures de l'automne 2026 n'y ont trouvé aucun
> gain sur la justesse des réponses : sur LoCoMo, les faits étaient inertes, et sur un jeu
> difficile de mises à jour, le modèle répondeur recopiait des faits périmés ou mal
> attribués (4/13 justes avec les faits, 13/13 avec les seuls épisodes). L'extraction
> tournait aussi à moins d'un épisode par seconde, contre ~5/s pour l'ingestion simple.
> Elles ont donc été éteintes pour les deux évaluations AML et sont en cours de
> réévaluation ; le chemin épisodique (`/add`, `/search`) est celui validé à l'échelle.

## ✨ Fonctionnalités

- 🔒 **100 % local** — Ollama (`bge-m3` + `qwen3:4b`) ou `llama-server` (llama.cpp) pour les embeddings + SQLite/sqlite-vec. Validé sur un i7-6700 CPU-only, 16 GB RAM
- ⚡ **Write path < 500 ms** — embedding synchrone, scoring LLM asynchrone (jamais bloquant)
- 🔍 **Recherche hybride** — similarité dense, recouvrement lexical des mots et récence (poids 0,7 / 0,3 / 0,1, normalisés), avec fenêtres temporelles
- 🗂️ **Faits versionnés** — supersession sur les prédicats fonctionnels, coexistence sur les multi, rétractation explicite, chaîne d'audit complète
- 🧭 **Router FR/EN** — classification lexicale (« hier » → épisodique, « qu'est-ce que tu sais sur » → sémantique, « comment ma préférence a évolué » → historique)
- 🔌 **MCP natif** — 5 tools (`memory_query`, `memory_write`, `memory_forget`, `memory_facts`, `memory_consolidate`) pour Antigravity CLI (`agy`), Claude Code & Claude Desktop
- 🏛️ **Multi-tenant** — une dimension `tenant` isole des mémoires parallèles (perso, une app, un NPC…) avec étanchéité stricte end-to-end. Optionnel partout, défaut `user` — les clients existants ne changent pas. Contrat : **[docs/MNEMOS_API.md](docs/MNEMOS_API.md)**
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

### Des embeddings plus rapides avec llama.cpp

Avec un GPU, `llama-server` sert `bge-m3` avec des slots parallèles : c'est le moteur qui a
tenu les deux évaluations AML. Lancez-le, puis indiquez-le à Mnemos dans `.env` :

```sh
llama-server --model <GGUF de bge-m3> --host 127.0.0.1 --port 8899 \
  -c 131072 -np 16 --embedding -b 8192 -ub 8192 --no-webui
```

```ini
EMBED_BACKEND=llamacpp
LLAMACPP_HOST=http://127.0.0.1:8899
```

Ollama reste le moteur par défaut (`EMBED_BACKEND=ollama`). Supervision sous Windows :
`scripts/register_llamacpp_task.ps1`, détails dans [docs/AML_DEPLOYMENT_WINDOWS.md](docs/AML_DEPLOYMENT_WINDOWS.md).

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

**Multi-tenant** : chaque endpoint accepte un `tenant` optionnel (défaut `user`), avec isolation stricte. Contrat HTTP complet consommé par les intégrations : **[docs/MNEMOS_API.md](docs/MNEMOS_API.md)**. Smoke test end-to-end : `python scripts/smoke_tenant.py` (Ollama réel).

**`POST /add` et `POST /search`** suivent le contrat AML : `user_id` est la frontière, l'Add
est idempotent sur `request_id`, et le Search rend les souvenirs stockés, jamais une réponse
générée. La règle d'oubli et la résolution des dates relatives ne s'appliquent **qu'à
`/add`**, ni à `/v1/episodes` ni aux outils MCP, qui passent encore par l'ancien chemin
d'écriture.

Spécification complète : **[MNEMOS_SPEC.md](MNEMOS_SPEC.md)** (rev 1.2) · benchs des modèles : **[archive/poc/RESULTS.md](archive/poc/RESULTS.md)**


## License / Licence

Apache License 2.0 — see [LICENSE](LICENSE). Copyright 2026 Adrien.M (Dr1mS).

---

*Mnemos — the Titaness of memory, mother of the Muses. A memory worth keeping is a memory worth versioning.*
