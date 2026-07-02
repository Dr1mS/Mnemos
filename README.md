# 🧠 Mnemos

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

---

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

---

## ✨ Fonctionnalités

- 🔒 **100 % local** — Ollama (`bge-m3` + `qwen3:4b`) + SQLite/sqlite-vec. Validé sur un i7-6700 CPU-only, 16 GB RAM
- ⚡ **Write path < 500 ms** — embedding synchrone, scoring LLM asynchrone (jamais bloquant)
- 🔍 **Recherche hybride** — `0.7·dense + 0.3·sparse + 0.1·récence`, avec fenêtres temporelles
- 🗂️ **Faits versionnés** — supersession sur les prédicats fonctionnels, coexistence sur les multi, rétractation explicite, chaîne d'audit complète
- 🧭 **Router FR/EN** — classification lexicale (« hier » → épisodique, « qu'est-ce que tu sais sur » → sémantique, « comment ma préférence a évolué » → historique)
- 🔌 **MCP natif** — 5 tools (`memory_query`, `memory_write`, `memory_forget`, `memory_facts`, `memory_consolidate`) pour Claude Code & Claude Desktop
- 🛡️ **Défense en profondeur mesurée** — la salience filtre l'émotionnel-non-personnel, l'extracteur rejette hypothétiques/temps passé/tiers (bench : 0 piège end-to-end sur corpus adversarial)

---

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

**Claude Desktop** (Linux beta ≥ juin 2026) — dans `~/.config/Claude/claude_desktop_config.json` :

```json
{
  "mcpServers": {
    "mnemos": {
      "command": "/chemin/vers/Mnemos/.venv/bin/mnemos-mcp",
      "env": { "DATA_DIR": "/chemin/vers/Mnemos/data/memoire", "...": "..." }
    }
  }
}
```

**Consolidation automatique** — service systemd user (`mnemos worker` : tick horaire + dump mensuel des archives, verrou d'instance unique) : voir `scripts/` et le [service d'exemple](MNEMOS_SPEC.md).

---

## 🏗️ Architecture

```
                          ┌─────────────────────────────────────────┐
 POST /v1/episodes ──────▶│ embed bge-m3 + sparse 256-bit  (~230 ms)│──▶ episodic.db
 (ou memory_write)   sync └─────────────────────────────────────────┘
                     async ┌────────────────┐   ┌──────────────────┐
                      └───▶│ queue salience │──▶│ qwen3:4b (amygd.)│──▶ UPDATE salience
                           └────────────────┘   └──────────────────┘
 ┌─ worker (sommeil) ── toutes les 60 min ────────────────────────────────────┐
 │  épisodes salience > 0.6 et âge > 1h ──▶ extraction faits+entités (qwen3)  │
 │  ──▶ add_fact : inserted │ superseded │ duplicate  ──▶ semantic.db         │
 │  puis decay + archivage des souvenirs fades                                │
 └─────────────────────────────────────────────────────────────────────────────┘
 POST /v1/query ──▶ classifier FR/EN ──▶ fan-out {épisodique, sémantique,
 (ou memory_query)                       historique, working, procédural}
```

Spécification complète : **[MNEMOS_SPEC.md](MNEMOS_SPEC.md)** (rev 1.2) · choix des modèles benchés : **[poc/RESULTS.md](poc/RESULTS.md)**

> ⚠️ Famille qwen3 : `think=false` **obligatoire** — le mode thinking casse le JSON structuré sous Ollama et multiplie la latence CPU par 5-10.

---

## 🧰 CLI & API

| CLI | API HTTP (`mnemos serve`, port 8765) |
|---|---|
| `mnemos write / search / query` | `POST /v1/episodes` · `GET /v1/episodes/search` · `POST /v1/query` |
| `mnemos facts --history` | `GET /v1/episodes/{id}` |
| `mnemos consolidate / decay / worker` | `POST /v1/admin/consolidate` · `POST /v1/admin/decay` |
| `mnemos stats / doctor / export / backup` | `GET /v1/health` · `POST /v1/sessions/{id}/reset` |

Auth optionnelle par header `X-API-Key`. Backup atomique par `VACUUM INTO` (jamais de copie brute d'un WAL).

---

## ✅ Critère "done" & tests

```sh
python scripts/demo.py            # le juge de paix : 50 messages simulés sur
                                  # 5 jours, salience + extraction réelles,
                                  # 10 checks (versioning, multi, fenêtres,
                                  # bruit…) — 10/10 sur le profil CPU
pytest -m "not requires_ollama"   # ~150 tests rapides sans LLM (~4 s)
pytest                            # complet avec Ollama réel (~3 min)
ruff check src tests && mypy      # lint + typage strict
```

Import d'une mémoire existante (épisodes JSONL + faits distillés) : `scripts/import_dump.py --episodes … --seed-facts`.

---

## 🗺️ Feuille de route

- [x] MVP 4 stores + consolidation + router (spec §18, 7 phases)
- [x] Serveur MCP (Claude Code, Claude Desktop)
- [x] Rétractation de faits — la détection de négation est déléguée au LLM consommateur via `memory_forget`
- [x] Rattrapage des scorings perdus au redémarrage du worker
- [ ] Fallback épisodique quand les scores sémantiques sont faibles
- [ ] Mode extraction pour contenu non-conversationnel (résumés)
- [ ] Oubli sémantique (decay de confidence des faits non renforcés)
- [ ] Connecteur claude.ai web/mobile (MCP distant + OAuth 2.1)

---

*Mnemos — la titanide de la mémoire, mère des Muses. Un souvenir qui mérite d'être gardé mérite d'être versionné.*
