# Mnemos — Multi-System Memory for LLM Agents

**Spec implémentation pour Claude Code.**
Cibles matérielles (deux profils) :
- **Profil dev (Linux)** : Ubuntu 24.04, i7-6700 4c/8t, 16 GB RAM, CPU-only, bash.
- **Profil déploiement (Windows)** : Windows 11, RTX 3070 Ti (8 GB VRAM), 64 GB RAM, PowerShell.

> **Rev 1.2 (2026-07-02)** — conclusions du POC modèles (`poc/RESULTS.md`) : passage à `qwen3:4b` avec `think=false` pour salience ET extraction (§2, §4, §7), prompt d'extraction v4 avec exemples de bascule (§15.2), double profil matériel dev CPU / déploiement GPU (§2, §4, §21), scripts de setup bash + PowerShell (§3, §4).
>
> **Rev 1.1 (2026-07-02)** — corrections après revue : cardinalité des prédicats (§10.2), decay sans double-comptage (§5.1, §9.2), règle d'archivage unifiée (§9.2), implémentation de référence du ModelManager + acquisition MEDIUM par épisode (§7.2, §15.1), salience asynchrone hors write path (§13.3, §16.1), flux entités branché à la consolidation (§15), saturation du sparse coding (§8.2), renommage `valence` → `arousal`, horloge injectable (§6).

---

## 1. Résumé exécutif

Mnemos est un serveur de mémoire local pour agents LLM, exposé via une API HTTP. Il implémente quatre stores parallèles avec des dynamiques différentes (working / épisodique / sémantique / procédural), un tagger de saillance qui filtre les écritures, un router qui orchestre les lectures, et une boucle de consolidation asynchrone qui transfère l'épisodique vers le sémantique avec versioning des faits.

L'architecture évite les pièges classiques des systèmes à store unique : pas de fusion sémantique des épisodes, conflits de faits résolus par versioning (pas par accumulation), oubli épisodique modulé par saillance. (L'oubli côté sémantique — décroissance de confiance des faits non renforcés — est volontairement différé, cf. §22.)

**Non-goals (à ne pas faire)** :
- Pas de Test-Time Training / fine-tuning des poids du modèle (trop fragile, hors scope MVP).
- Pas de pattern separation neural sophistiqué (DG-style sparse autoencoders) — on utilise une approche pragmatique par hashing.
- Pas de UI graphique. CLI + API uniquement. Une UI peut venir après.
- Pas de multi-utilisateur. Single tenant.

---

## 2. Stack technique

| Composant | Choix | Version | Justification |
|---|---|---|---|
| Langage | Python | 3.12 | Convention projet, perf asyncio |
| Package manager | uv | latest | Convention projet |
| API | FastAPI | ^0.115 | Standard |
| ORM | SQLAlchemy | ^2.0 | Async support |
| Migrations | Alembic | ^1.13 | Versionning schéma |
| DB | SQLite + sqlite-vec | ≥0.1.6 | Simplicité, perf single-tenant, embedding natif |
| Logs | structlog | ^25 | Convention projet |
| Scheduler | APScheduler | ^3.10 | Worker consolidation |
| Tests | pytest, pytest-asyncio | latest | Standard |
| LLM runtime | Ollama (local) | ≥0.5 | Déjà installé |
| Embedding model | `bge-m3` | via Ollama | Multilingue FR/EN, 1024-dim, ~1.2 GB — validé POC : p50 231 ms sur CPU |
| Salience model | `qwen3:4b` (`think=false`) | via Ollama | ~2.5 GB, plausibilité 17/17 au POC |
| Consolidation model | `qwen3:4b` (`think=false`) | via Ollama | même modèle que salience → zéro swap ; `qwen3:8b` en option sur profil GPU |

**⚠ Famille qwen3 = thinking models : `think=false` obligatoire sur tout appel generate.** Le mode thinking casse le JSON structuré sous Ollama (issue ollama#10929) et multiplie la latence CPU par 5-10×.

**Budget mémoire** :
- Profil dev (16 GB RAM, CPU-only) : bge-m3 (~1.2 GB) + qwen3:4b (~2.5 GB) résidents ≈ 3.7 GB. Un seul LLM pour salience et extraction → jamais de swap de modèle. La contrainte est la latence CPU (~8 tok/s en génération), pas la mémoire.
- Profil déploiement (8 GB VRAM) : mêmes modèles ≈ 3.7 GB + KV cache → confortable. Si `qwen3:8b` est activé pour l'extraction, l'exclusion small/medium du ModelManager redevient active (cf. §7).
- Le ModelManager (cf. §7) reste l'unique point de sortie vers Ollama dans les deux profils.

---

## 3. Structure projet

```
mnemos/
├── pyproject.toml
├── README.md
├── CLAUDE.md                       # Pointeur vers ce spec
├── .env.example
├── .gitignore
├── alembic.ini
├── alembic/
│   ├── env.py
│   └── versions/
├── data/                           # SQLite files (gitignored)
│   ├── episodic.db
│   ├── semantic.db
│   └── procedural/
├── scripts/
│   ├── setup_ollama_models.sh      # profil Linux
│   ├── setup_ollama_models.ps1     # profil Windows
│   ├── reset_dbs.sh
│   ├── reset_dbs.ps1
│   └── benchmark.py
├── src/mnemos/
│   ├── __init__.py
│   ├── config.py
│   ├── logging.py
│   ├── clock.py                    # Horloge injectable (cf §6)
│   ├── server.py                   # FastAPI app entry
│   ├── cli.py                      # Typer CLI
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── ollama_client.py
│   │   └── model_manager.py        # Tiered semaphore routing
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── dense.py
│   │   └── sparse.py               # Pattern-separation hashing
│   ├── stores/
│   │   ├── __init__.py
│   │   ├── working.py
│   │   ├── episodic.py
│   │   ├── semantic.py
│   │   └── procedural.py
│   ├── tagger/
│   │   ├── __init__.py
│   │   └── salience.py
│   ├── router/
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   └── orchestrator.py
│   ├── consolidation/
│   │   ├── __init__.py
│   │   ├── extractor.py
│   │   ├── conflict_resolver.py
│   │   └── worker.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py
│   │   ├── schemas.py
│   │   └── deps.py
│   └── models/                     # SQLAlchemy ORM
│       ├── __init__.py
│       ├── episodic.py
│       └── semantic.py
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── test_sparse.py
    │   ├── test_salience.py
    │   ├── test_router.py
    │   └── test_conflict_resolver.py
    ├── integration/
    │   ├── test_write_path.py
    │   ├── test_read_path.py
    │   └── test_consolidation_loop.py
    └── e2e/
        └── test_full_session.py
```

---

## 4. Setup initial (Windows PowerShell)

### 4.1 Modèles Ollama

`scripts/setup_ollama_models.sh` (Linux) / `scripts/setup_ollama_models.ps1` (Windows) :

```sh
ollama pull bge-m3
ollama pull qwen3:4b
# ollama pull qwen3:8b   # optionnel — profil GPU uniquement (extraction)
ollama list
```

### 4.2 Environnement Python

Linux :

```sh
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
alembic upgrade head
```

Windows :

```powershell
uv venv --python 3.12
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
alembic upgrade head
```

### 4.3 `.env.example`

```env
# Ollama
OLLAMA_HOST=http://localhost:11434
EMBED_MODEL=bge-m3
SALIENCE_MODEL=qwen3:4b
EXTRACTION_MODEL=qwen3:4b
# Profil GPU (3070 Ti) : EXTRACTION_MODEL=qwen3:8b possible → tier MEDIUM active
LLM_THINK=false   # famille qwen3 : jamais de mode thinking (JSON cassé + latence ×5-10)

# Storage
DATA_DIR=./data
EPISODIC_DB=./data/episodic.db
SEMANTIC_DB=./data/semantic.db
PROCEDURAL_DIR=./data/procedural

# Server
API_HOST=127.0.0.1
API_PORT=8765
LOG_LEVEL=INFO

# Memory dynamics
SALIENCE_THRESHOLD_CONSOLIDATE=0.6   # Au-dessus : consolide
SALIENCE_THRESHOLD_DECAY_FAST=0.2    # En-dessous : décroissance rapide
DECAY_RATE_DAILY=0.05                 # Décroissance par jour si non saillant
CONSOLIDATION_DELAY_HOURS=1           # Attente avant consolidation
EPISODIC_RETENTION_DAYS=90            # Au-delà : archivage même sans consolidation (règle complète §9.2)

# Consolidation worker
CONSOLIDATION_INTERVAL_MINUTES=60
CONSOLIDATION_BATCH_SIZE=20

# Concurrency (ModelManager)
LLM_TIER_SMALL_CONCURRENCY=4   # Embedding, salience
LLM_TIER_MEDIUM_CONCURRENCY=1  # Extraction (7B exclusif)
```

---

## 5. Schémas de base de données

### 5.1 Episodic (`episodic.db`)

```sql
-- sqlite-vec doit être chargé en runtime
-- ALL TIMESTAMPS in UTC, stored as INTEGER (unix epoch ms)

CREATE TABLE episodes (
  id              TEXT PRIMARY KEY,             -- ULID
  created_at      INTEGER NOT NULL,
  session_id      TEXT,
  role            TEXT NOT NULL,                -- 'user' | 'assistant' | 'system'
  content         TEXT NOT NULL,
  -- Salience scoring (NULL = scoring asynchrone pas encore passé, cf. §13.3)
  salience        REAL NOT NULL DEFAULT 0.5,    -- [0..1]
  surprise        REAL,
  arousal         REAL,                          -- intensité émotionnelle [0..1], pos. ou nég.
  self_ref        REAL,
  recurrence      REAL,
  -- Lifecycle
  decay_state       REAL NOT NULL DEFAULT 1.0,  -- [0..1], décroît avec temps
  last_decayed_at   INTEGER,                     -- dernier passage d'apply_decay (NULL = jamais)
  consolidated_at   INTEGER,                     -- NULL = pas encore consolidé
  extraction_failed INTEGER NOT NULL DEFAULT 0, -- 0/1, cf. anti-pattern 8
  archived          INTEGER NOT NULL DEFAULT 0, -- 0/1
  -- Refs
  entity_refs     TEXT NOT NULL DEFAULT '[]'    -- JSON array of entity names
) STRICT;

CREATE INDEX idx_episodes_created_at ON episodes(created_at DESC);
CREATE INDEX idx_episodes_session ON episodes(session_id, created_at DESC);
CREATE INDEX idx_episodes_consolidation ON episodes(consolidated_at, salience DESC);
CREATE INDEX idx_episodes_archived ON episodes(archived);

-- Dense embeddings via sqlite-vec virtual table
CREATE VIRTUAL TABLE episodes_vec USING vec0(
  episode_id      TEXT PRIMARY KEY,
  embedding       FLOAT[1024]                   -- bge-m3 dim
);

-- Sparse pattern-separation codes (custom format, cf §8.2)
CREATE TABLE episodes_sparse (
  episode_id      TEXT PRIMARY KEY,
  sparse_bits     BLOB NOT NULL,                -- 256-bit packed
  FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
) STRICT;
```

### 5.2 Semantic (`semantic.db`)

```sql
-- Faits versionnés, à la Memstate.
-- Subject + predicate + valid_until=NULL = "fait courant"

CREATE TABLE facts (
  id               TEXT PRIMARY KEY,            -- ULID
  subject          TEXT NOT NULL,
  predicate        TEXT NOT NULL,
  object           TEXT NOT NULL,
  -- Temporal validity
  valid_from       INTEGER NOT NULL,
  valid_until      INTEGER,                     -- NULL = encore valide
  -- Provenance
  confidence       REAL NOT NULL DEFAULT 1.0,
  source_episodes  TEXT NOT NULL DEFAULT '[]',  -- JSON array of episode IDs
  -- Versioning
  superseded_by    TEXT,                        -- FK → facts.id
  created_at       INTEGER NOT NULL,
  FOREIGN KEY (superseded_by) REFERENCES facts(id)
) STRICT;

CREATE INDEX idx_facts_subject ON facts(subject, predicate, valid_until);
CREATE INDEX idx_facts_current ON facts(valid_until) WHERE valid_until IS NULL;
CREATE INDEX idx_facts_superseded ON facts(superseded_by);

-- Entités (résolution des co-références minimales)
CREATE TABLE entities (
  canonical_name   TEXT PRIMARY KEY,
  aliases          TEXT NOT NULL DEFAULT '[]',  -- JSON array
  entity_type      TEXT,                        -- 'person'|'org'|'place'|'concept'|'product'|null
  first_seen       INTEGER NOT NULL,
  last_seen        INTEGER NOT NULL,
  episode_count    INTEGER NOT NULL DEFAULT 0
) STRICT;

CREATE INDEX idx_entities_last_seen ON entities(last_seen DESC);

-- Embeddings sémantiques pour recherche fuzzy sur sujets
CREATE VIRTUAL TABLE facts_vec USING vec0(
  fact_id          TEXT PRIMARY KEY,
  embedding        FLOAT[1024]
);
```

### 5.3 Procedural

Pas de DB. Filesystem :

```
data/procedural/
├── send_email_with_attachment/
│   ├── skill.py
│   └── meta.json     # {desc, signature, success_rate, last_used_at, version}
├── search_company_records/
│   └── ...
└── _registry.json     # Index global des skills disponibles
```

---

## 6. Configuration applicative

`src/mnemos/config.py` — pydantic-settings, charge depuis `.env`. Tous les paramètres de §4.3 typés. Une seule instance Settings injectée via DI FastAPI.

`src/mnemos/logging.py` — structlog avec processeurs : timestamp ISO, level, logger name, JSON renderer en prod, console renderer en dev (détecté via `LOG_LEVEL=DEBUG`).

`src/mnemos/clock.py` — **horloge injectable** : `class Clock: def now_ms(self) -> int`. Tout le code applicatif (decay, archivage, valid_from/valid_until, sparse temporal bits) obtient le temps via cette instance, jamais via `datetime.now()` direct. Indispensable pour tester la décroissance et le versioning temporel sans time-travel réel — à mettre en place dès la Phase 0, pénible à retrofitter.

---

## 7. Composant : LLM client + ModelManager

**Critique** — c'est ici que les Oracle Engine deadlocks ont été créés sur PulseWorld. Le ModelManager est obligatoire.

### 7.1 `ollama_client.py`

```python
class OllamaClient:
    async def embed(self, text: str, model: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str], model: str) -> list[list[float]]: ...
    async def generate(
        self,
        prompt: str,
        model: str,
        format: Literal["json"] | None = None,
        options: dict | None = None,
        think: bool = False,   # TOUJOURS False pour qwen3 (cf. §2)
    ) -> str: ...
    async def health_check(self) -> bool: ...
```

Implémentation : `httpx.AsyncClient`, timeouts explicites (60s embed, 300s generate — profil CPU oblige), retry exponentiel (3 tentatives) sur erreurs réseau, **pas** de retry sur erreurs HTTP 4xx. Le paramètre `think` est transmis tel quel à l'API Ollama ; il vient de `Settings.LLM_THINK`, jamais hardcodé à True.

### 7.2 `model_manager.py`

Le ModelManager route chaque appel vers une tier (small/medium/large) avec sémaphore par tier. Exclusion mutuelle entre tiers (cf. budget VRAM §2).

Implémentation de référence — c'est le composant le plus piégeux du système, ne pas improviser :

```python
class Tier(str, Enum):
    SMALL = "small"      # bge-m3, qwen3:4b (salience + extraction en profil dev)
    MEDIUM = "medium"    # qwen3:8b (profil GPU uniquement, optionnel)
    # LARGE réservé pour future extension

class ModelManager:
    """
    Invariants :
    - Une seule tier "active" à la fois (contrainte VRAM).
    - _active_count = nombre d'appels en vol sur la tier active.
    - Un appel d'une autre tier attend que _active_count retombe à 0.
    - INTERDIT d'attendre un changement de tier en tenant un lock ou un
      sémaphore — c'est exactement le deadlock PulseWorld. D'où une unique
      Condition, pas de sémaphores séparés.
    """
    def __init__(self, settings: Settings, client: OllamaClient):
        self._limits = {
            Tier.SMALL: settings.LLM_TIER_SMALL_CONCURRENCY,
            Tier.MEDIUM: settings.LLM_TIER_MEDIUM_CONCURRENCY,
        }
        self._state = asyncio.Condition()
        self._active_tier: Tier | None = None
        self._active_count = 0

    async def acquire(self, tier: Tier) -> None:
        async with self._state:
            await self._state.wait_for(
                lambda: self._active_tier in (None, tier)
                and self._active_count < self._limits[tier]
            )
            self._active_tier = tier
            self._active_count += 1

    async def release(self, tier: Tier) -> None:
        async with self._state:
            self._active_count -= 1
            if self._active_count == 0:
                self._active_tier = None
            self._state.notify_all()

    @asynccontextmanager
    async def use(self, tier: Tier):
        await self.acquire(tier)
        try:
            yield
        finally:
            await self.release(tier)
```

**Règle d'exclusion** : si une tier est active avec count > 0, l'autre tier doit attendre. Cela évite que la consolidation (medium) tourne pendant que l'API sert (small) et fasse OOM.

**Cas profil dev (un seul LLM)** : quand `EXTRACTION_MODEL == SALIENCE_MODEL`, l'extraction est routée en tier SMALL — l'exclusion ne se déclenche jamais, c'est attendu. Le ModelManager route par **modèle**, pas par usage. Le stress test d'exclusion reste obligatoire : il protège le profil GPU (qwen3:8b en MEDIUM).

**Famine assumée** : sous trafic SMALL continu, `_active_count` peut ne jamais retomber à 0 et MEDIUM attend. C'est le bon trade-off pour un MVP : la latence du write path prime sur la ponctualité de la consolidation (qui retentera au prochain tick). Ne PAS ajouter de mécanisme de fairness sans mesure démontrant le besoin. Le pendant côté worker : acquisition MEDIUM **par épisode**, pas autour du batch (cf. §15.1), sinon c'est la consolidation qui affame le write path pendant des minutes.

**Test obligatoire (Phase 1)** : stress test asyncio — N tâches SMALL + M tâches MEDIUM concurrentes, assert jamais deux tiers actives simultanément, assert terminaison (pas de deadlock).

Tous les appels Ollama passent par ce manager. **Aucun call direct à OllamaClient depuis le code applicatif**.

---

## 8. Composants : embeddings

### 8.1 Dense (`embeddings/dense.py`)

Wrapper autour de OllamaClient.embed avec cache LRU en mémoire (taille 1000) sur le hash du contenu, pour éviter de recomputer les embeddings d'un même texte.

### 8.2 Sparse (`embeddings/sparse.py`)

Pattern separation pragmatique. **Pas de modèle neural** — hashing déterministe.

Algorithme :
1. Tokenizer simple (split + lowercase, FR + EN). **Cap : 64 premiers tokens uniques** — au-delà, le OR sature les 224 bits de contenu (à ~100 tokens uniques on dépasse 35 % de remplissage et la distance de Hamming perd son pouvoir discriminant sur les longs épisodes). Log en DEBUG si tronqué.
2. Pour chaque token, hash BLAKE2b-128 → mod 224 → bit position (bits 0–223).
3. Allouer aussi 32 bits "temporels" (bits 224–255) : hash de `(year, week_of_year, day_of_week, hour_of_day // 4)`.
4. OR de tous les bits → vecteur 256-bit.
5. Stockage : `BLOB` 32 bytes.
6. Similarité : Hamming distance (popcount XOR).

**Résolution temporelle = bucket de 4h** : deux épisodes de même contenu dans le même bucket (même année/semaine/jour/tranche de 4h) ont des codes sparse identiques — c'est voulu, la séparation temporelle est grossière par design. Conséquence pour les tests de pattern separation : choisir des timestamps dans des buckets différents, sinon le test échoue à tort.

```python
def sparse_encode(content: str, timestamp_ms: int) -> bytes:
    bits = bytearray(32)  # 256 bits
    # Content bits
    for token in tokenize(content):
        h = blake2b(token.encode(), digest_size=2).digest()
        pos = int.from_bytes(h, 'little') % 224  # 224 bits content
        bits[pos // 8] |= (1 << (pos % 8))
    # Temporal bits (32 dernières positions)
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    temporal_seed = f"{dt.year}-W{dt.isocalendar().week}-{dt.weekday()}-{dt.hour // 4}"
    th = blake2b(temporal_seed.encode(), digest_size=4).digest()
    for i in range(32):
        if th[i // 8] & (1 << (i % 8)):
            bits[28 + i // 8] |= (1 << (i % 8))
    return bytes(bits)

def hamming_distance(a: bytes, b: bytes) -> int:
    return (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).bit_count()
```

**Recherche hybride** : score combiné = `0.7 * dense_cosine + 0.3 * (1 - hamming_dist / 256)`.

C'est imparfait mais empiriquement améliore la précision épisodique (épisodes orthogonaux dans le temps deviennent distinguables). Document la métrique dans le README et garde la pondération configurable.

---

## 9. Composant : Episodic Store

### 9.1 Interface

```python
class EpisodicStore:
    async def write(
        self,
        content: str,
        role: str,
        session_id: str | None = None,
        salience_scores: SalienceScores | None = None,
    ) -> Episode: ...

    async def search(
        self,
        query: str,
        k: int = 10,
        session_id: str | None = None,
        time_window: tuple[datetime, datetime] | None = None,
        min_salience: float = 0.0,
    ) -> list[ScoredEpisode]: ...

    async def get_by_id(self, episode_id: str) -> Episode | None: ...

    async def mark_consolidated(self, episode_id: str) -> None: ...

    async def set_entity_refs(self, episode_id: str, entity_names: list[str]) -> None: ...

    async def apply_decay(self, dry_run: bool = False) -> DecayReport: ...

    async def archive_old(self, dry_run: bool = False) -> ArchiveReport: ...
```

### 9.2 Notes implémentation

- **Décroissance** : `apply_decay` est appelé par le worker. Formule : `new_decay = current_decay - DECAY_RATE_DAILY * elapsed_days * (2 - salience)`, où `elapsed_days` = temps écoulé depuis `COALESCE(last_decayed_at, created_at)` — **jamais** depuis `created_at` seul, sinon chaque run du worker re-soustrait la totalité de l'âge (double-comptage → décroissance quadratique). Chaque passage met à jour `last_decayed_at = now()`. Plus l'épisode est saillant, plus il décroît lentement. Le temps vient du `Clock` injectable (§6).
- **Archivage** : `archived=1` quand l'une de ces deux conditions est vraie :
  1. `decay_state < 0.1` ET `consolidated_at IS NOT NULL` (épisode décru dont les faits ont été extraits) ;
  2. `age > EPISODIC_RETENTION_DAYS` ET `salience < SALIENCE_THRESHOLD_CONSOLIDATE` ET `consolidated_at IS NULL` (épisode ancien qui ne sera jamais candidat à consolidation — sans cette règle, les épisodes à faible salience s'accumulent indéfiniment).

  Les épisodes archivés ne sont plus retournés par `search` mais restent en DB pour audit.
- **Dump JSONL périodique** des archivés vers `data/archive/YYYY-MM.jsonl` puis `DELETE` de la DB. Worker mensuel.
- **WAL mode** activé sur SQLite : `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`.
- **Recherche** :
  1. Encode query (dense + sparse).
  2. KNN top-50 via vec_distance_cosine sur `episodes_vec`.
  3. Filtre Python : session_id, time_window, archived=0, salience >= min.
  4. Re-rank par score hybride dense + sparse + récence.
  5. Retourne top-k.

---

## 10. Composant : Semantic Store

### 10.1 Interface

```python
class SemanticStore:
    async def add_fact(
        self,
        subject: str,
        predicate: str,
        object_: str,
        source_episode_ids: list[str],
        confidence: float = 1.0,
    ) -> FactWriteResult:
        """
        Insert fact, detecting and resolving conflicts.
        Returns FactWriteResult with action: 'inserted'|'superseded'|'duplicate'.
        """

    async def get_current_facts(
        self,
        subject: str | None = None,
        predicate: str | None = None,
    ) -> list[Fact]: ...

    async def get_history(self, subject: str, predicate: str) -> list[Fact]:
        """Tous les faits (incl. invalidés) pour cette paire subject/predicate."""

    async def search_facts(self, query: str, k: int = 10) -> list[ScoredFact]: ...

    async def upsert_entity(self, name: str, entity_type: str | None = None) -> Entity: ...
```

**Embedding des faits (`facts_vec`)** : le texte embeddé est la phrase `"{subject} {predicate} {object}"`. **Recherche (`search_facts`)** : KNN top-`4*k` sur `facts_vec`, puis JOIN sur `facts` + filtre `valid_until IS NULL` en SQL, re-rank, top-k. Le sur-fetch est obligatoire : les tables virtuelles vec0 ne filtrent pas par métadonnée, et sans lui les faits invalidés consomment les slots du KNN et on renvoie moins de k résultats valides.

### 10.2 Résolution de conflit (`conflict_resolver.py`)

Chaque predicate du vocabulaire contrôlé `mnemos.ontology.PREDICATES` porte une **cardinalité** — c'est ce qui décide si un object différent est une contradiction ou une valeur supplémentaire :

| Cardinalité | Prédicats | Sémantique |
|---|---|---|
| `functional` (une seule valeur courante) | `works_at`, `lives_in` | Un nouvel object contredit et remplace l'ancien |
| `multi` (plusieurs valeurs courantes coexistent) | `prefers`, `dislikes`, `owns`, `is_a`, `has_attribute`, `knows_about`, `has_goal`, `has_skill` | Les valeurs s'additionnent |

Quand `add_fact(s, p, o, ...)` est appelé :

0. **Normalisation entités** : `subject` et `object` passent par la résolution d'alias (table `entities` : lookup casse-insensible sur `canonical_name` + `aliases`) avant toute comparaison. Sans ça, "Google" / "google" / "Google Inc" créent des faits parallèles jamais dédupliqués.
1. Cherche les faits courants (`valid_until IS NULL`) avec même `subject` + `predicate`.
2. Quatre cas :
   - **Aucun match** → INSERT direct, retourne `action='inserted'`.
   - **Match avec même object** (après normalisation) → mise à jour `confidence` (max), append `source_episodes`, retourne `action='duplicate'`.
   - **Match avec object différent, predicate `functional`** → ancien fait : `valid_until=now()`, `superseded_by=new_id`. Nouveau fait inséré. Retourne `action='superseded'`.
   - **Match avec object différent, predicate `multi`** → INSERT additionnel, retourne `action='inserted'`. **Jamais de supersession** : "je préfère le thé" n'invalide pas "je préfère le café" ; l'utilisateur possède une voiture ET un vélo. Superseder ici détruirait activement de la mémoire correcte — pire qu'accumuler.

La **rétractation explicite** d'un fait multi-valué ("je n'aime plus le café", "j'ai vendu ma voiture") est hors scope MVP (cf. §22) : elle exige que l'extractor détecte la négation, pas seulement le fait.

Si le predicate proposé n'existe pas dans le vocabulaire, mappe au plus proche (similarité fuzzy) ou range-le sous `has_attribute` avec le predicate brut comme partie de l'object. **Ne jamais** créer un nouveau predicate à la volée — la fragmentation est le principal échec mode.

---

## 11. Composant : Working Memory

In-memory uniquement. Pas de persistance.

```python
class WorkingMemory:
    def __init__(self, max_items: int = 5):
        self._items: deque[WMItem] = deque(maxlen=max_items)
        self._active_entities: set[str] = set()
        self._current_session_id: str | None = None

    def push(self, content: str, role: str) -> None: ...
    def get_context(self) -> list[WMItem]: ...
    def get_active_entities(self) -> set[str]: ...
    def reset(self) -> None: ...
```

Une instance par session. Le serveur maintient un `dict[session_id, WorkingMemory]` avec eviction LRU sur 100 sessions max.

---

## 12. Composant : Procedural Store

```python
class ProceduralStore:
    def list_skills(self) -> list[SkillMeta]: ...
    def get_skill(self, name: str) -> Skill | None: ...
    def register_skill(self, name: str, code: str, meta: SkillMeta) -> None: ...
    def update_stats(self, name: str, success: bool) -> None: ...
```

Hors scope MVP pour l'auto-amélioration (Voyager-style). Le store accepte uniquement enregistrement manuel et lecture. Auto-amélioration en Phase 4.

---

## 13. Composant : Salience Tagger

### 13.1 Interface

```python
class SalienceScores(TypedDict):
    surprise: float      # [0..1]
    arousal: float       # [0..1] — intensité émotionnelle, positive OU négative.
                         # Nommé "arousal" et pas "valence" : une valence serait
                         # signée ; ici le signe est volontairement perdu.
    self_ref: float      # [0..1]
    recurrence: float    # [0..1]
    combined: float      # [0..1]

class SalienceTagger:
    async def score(self, content: str, recent_history: list[str]) -> SalienceScores: ...
```

### 13.2 Implémentation

Un seul appel LLM (qwen2.5:3b) en JSON mode. Prompt :

```
You score the salience of a single message for memory consolidation.
Return JSON with four floats in [0,1]:

- surprise: how unexpected/novel is this content vs typical conversation
- arousal: emotional intensity (positive or negative, both score high)
- self_ref: how much the user reveals about themselves (preferences, identity, life facts)
- recurrence: 0 if this topic is new in the recent history, higher if it repeats

Recent history (last 5 turns):
{recent_history}

Current message:
{content}

Output ONLY JSON: {"surprise": 0.X, "arousal": 0.X, "self_ref": 0.X, "recurrence": 0.X}
```

Calcul `combined` :
```python
combined = max(
    0.4 * surprise + 0.3 * self_ref + 0.2 * arousal + 0.1 * recurrence,
    self_ref,  # Self-reference seul suffit à passer le seuil
)
```

`self_ref` boost-floor parce qu'un fait sur le user est toujours intéressant à consolider, même si pas surprenant.

Robustesse : si parse JSON échoue, retourne `combined=0.5` (neutre) et log une erreur. Ne jamais bloquer le write path à cause d'une erreur de salience.

### 13.3 Position dans le write path : asynchrone

Le scoring de salience est un appel LLM 3B (~1–2 s) — il ne doit **jamais** être dans le chemin critique de `POST /v1/episodes`. Déroulé du write :

1. **Synchrone** : écriture de l'épisode en DB avec `salience=0.5` (défaut) + calcul de l'embedding dense/sparse (tier SMALL, rapide) → l'épisode est immédiatement cherchable, la réponse API part.
2. **Asynchrone** : le scoring salience est enqueué (queue asyncio bornée, workers en tâche de fond) et met à jour les colonnes `salience`/`surprise`/`arousal`/`self_ref`/`recurrence` quand il aboutit — bien avant `CONSOLIDATION_DELAY_HOURS`, donc sans impact sur la consolidation.

La réponse de `POST /v1/episodes` retourne `salience: null` tant que le score n'est pas calculé (cf. §16.1). Si la queue est pleine, drop le scoring (l'épisode garde 0.5) et log — jamais de backpressure sur le write.

---

## 14. Composant : Router

### 14.1 Classification de requête (`classifier.py`)

Classification rule-based en première ligne, LLM fallback uniquement si ambigu.

```python
class QueryType(str, Enum):
    EPISODIC_TEMPORAL = "episodic_temporal"  # "hier", "la dernière fois", "quand"
    EPISODIC_FUZZY = "episodic_fuzzy"        # "j'ai dit quoi sur X"
    SEMANTIC_FACT = "semantic_fact"          # "qu'est-ce que tu sais sur Y"
    SEMANTIC_HISTORY = "semantic_history"    # "comment ma préférence X a évolué"
    PROCEDURAL = "procedural"                # "comment je fais Z"
    WORKING = "working"                      # "où on en est"
    UNKNOWN = "unknown"

def classify(query: str) -> QueryType:
    # Patterns lexicaux d'abord
    # FR + EN
    ...
```

Mots-clés FR : `hier|avant-hier|la dernière fois|quand|aujourd'hui|cette semaine` → EPISODIC_TEMPORAL ; `je préfère|tu sais que|c'est qui|qu'est-ce que` → SEMANTIC_FACT ; etc.

### 14.2 Orchestration (`orchestrator.py`)

```python
class RouterOrchestrator:
    async def query(self, q: str, session_id: str, k: int = 10) -> QueryResult:
        qtype = classify(q)
        # Fan-out parallèle aux stores pertinents
        tasks = []
        if qtype in (EPISODIC_TEMPORAL, EPISODIC_FUZZY, UNKNOWN):
            tasks.append(self.episodic.search(q, k=k))
        if qtype in (SEMANTIC_FACT, SEMANTIC_HISTORY, UNKNOWN):
            tasks.append(self.semantic.search_facts(q, k=k))
        if qtype == WORKING:
            tasks.append(self.working.get_context_for(session_id))
        # Procedural toujours en best-effort
        results = await asyncio.gather(*tasks)
        # Re-rank cross-store par score normalisé
        return rerank_and_merge(results, k=k)
```

UNKNOWN consulte tout (épisodique + sémantique). C'est le fallback safe.

---

## 15. Composant : Worker de consolidation

### 15.1 Boucle principale

```python
class ConsolidationWorker:
    async def run_once(self) -> ConsolidationReport:
        # 1. Sélectionne candidats : salience > seuil ET age > delay ET non-consolidé
        candidates = await self.episodic.list_pending_consolidation(
            min_salience=settings.SALIENCE_THRESHOLD_CONSOLIDATE,
            min_age_hours=settings.CONSOLIDATION_DELAY_HOURS,
            limit=settings.CONSOLIDATION_BATCH_SIZE,
        )
        # 2. Pour chaque épisode : extraction de faits + entités.
        #    Acquisition MEDIUM PAR ÉPISODE, pas autour du batch : tenir la tier
        #    pendant 20 extractions bloquerait tous les writes API (tier SMALL)
        #    pendant plusieurs minutes. Entre deux épisodes, les appels SMALL en
        #    attente peuvent s'intercaler. Coût : un éventuel swap de modèle
        #    (~5-10s) quand ça arrive — c'est le prix de la non-famine du write
        #    path. Si aucun appel SMALL ne s'intercale, Ollama garde le 7B
        #    chargé (keep_alive) et il n'y a pas de swap.
        for episode in candidates:
            async with self.model_manager.use(Tier.MEDIUM):
                extraction = await self.extractor.extract(episode)
            # DB uniquement à partir d'ici — hors tier.
            for e in extraction.entities:
                await self.semantic.upsert_entity(e.name, e.entity_type)
            await self.episodic.set_entity_refs(
                episode.id, [e.name for e in extraction.entities]
            )
            for f in extraction.facts:
                await self.semantic.add_fact(**f)
            await self.episodic.mark_consolidated(episode.id)
        # 3. Décroissance + archivage
        await self.episodic.apply_decay()
        await self.episodic.archive_old()
        return ConsolidationReport(...)
```

Scheduling : APScheduler `interval` trigger toutes les `CONSOLIDATION_INTERVAL_MINUTES` minutes. Une instance unique (verrou via fichier lock dans `data/`).

### 15.2 Extracteur de faits + entités (`extractor.py`)

Prompt v4 (`think=false`, JSON mode) — issu du POC (`poc/RESULTS.md`) : les règles seules sur-suppriment avec un 4B (passé composé, has_goal) ; les exemples de bascule sont la partie qui porte. Ne pas les retirer pour "raccourcir le prompt".

```
Extract structured facts and named entities from this conversation episode.
Output JSON object: {"facts": [...], "entities": [...]}.
Each fact: {subject, predicate, object, confidence}.
Each entity: {name, entity_type, aliases} where entity_type is one of
             person|org|place|concept|product, and aliases lists other
             surface forms used in the episode (may be empty).

Allowed predicates: works_at, lives_in, prefers, dislikes, owns,
                    is_a, has_attribute, knows_about, has_goal, has_skill

Rules:
- subject is EXACTLY "user" when the fact is about the user speaking; when
  the sentence explicitly names another person/entity as the actor, that
  entity is the subject
- if the actor is a pronoun whose referent is NOT named in this episode,
  extract NOTHING about it
- extract facts that are CURRENTLY true. A past event that established a
  current state IS a current fact. A state explicitly ended is NOT.
- personal goals and desires to learn/do something ARE facts (has_goal)
- IGNORE questions, unrealistic hypotheticals/conditionals, jokes, sarcasm,
  and statements the speaker is unsure about
- use canonical English for predicates, but keep the object in the original
  language of the episode (do not translate it)
- confidence is a number between 0.0 and 1.0
- entities: only entities actually mentioned; use the most complete surface
  form as name
- if nothing extractable, return {"facts": [], "entities": []}

Examples:
- "Avant je bossais chez TechCorp." → facts: []  (state ended, no longer true)
- "J'ai adopté un chat, Yuzu." → {"subject": "user", "predicate": "owns", "object": "Yuzu", "confidence": 0.9}  (past event, current state)
- "J'aimerais apprendre Rust." → {"subject": "user", "predicate": "has_goal", "object": "Rust", "confidence": 0.9}
- "Mon frère Tom travaille chez Airbus." → {"subject": "Tom", "predicate": "works_at", "object": "Airbus", "confidence": 0.9}
- "Je ne bois plus de thé, je suis passée au maté." → {"subject": "user", "predicate": "prefers", "object": "maté", "confidence": 0.9}  (only the NEW preference)
- "Si je gagnais au loto, j'achèterais une villa." → facts: []  (hypothetical)

Episode (role={role}, timestamp={ts}):
{content}

Output ONLY JSON.
```

C'est ce prompt qui alimente le flux entités : le worker upsert les entités extraites (`upsert_entity`), remplit `episodes.entity_refs`, et le conflict resolver s'appuie ensuite sur `entities.aliases` pour normaliser subject/object (§10.2, étape 0). Personne d'autre ne peuple ces tables.

Validation post-parsing : predicate ∈ vocabulaire, subject/object non vides, confidence ∈ [0,1], entity_type ∈ vocabulaire ou null. Toute extraction invalide est skippée et loggée.

---

## 16. API HTTP (`api/routes.py`)

Tous les endpoints en `/v1/`. Auth : optionnelle pour MVP (header `X-API-Key` si défini dans .env, sinon ouvert sur localhost).

### 16.1 Endpoints

```
POST   /v1/episodes
       body: {content, role, session_id?}
       returns: {id, salience, ...}
       # salience: null tant que le scoring asynchrone (§13.3) n'est pas passé ;
       # l'épisode est néanmoins déjà écrit et cherchable (embedding synchrone).

GET    /v1/episodes/search?q=...&k=10&session_id=...&min_salience=...
       returns: [{episode, score}, ...]

GET    /v1/facts?subject=...&predicate=...
       returns: [Fact, ...]   # uniquement courants

GET    /v1/facts/history?subject=...&predicate=...
       returns: [Fact, ...]   # historique complet

POST   /v1/query
       body: {q, session_id?, k?}
       returns: {type, episodes, facts, working, procedural}

POST   /v1/sessions/{session_id}/reset
       reset working memory pour cette session

GET    /v1/health
       returns: {ok, ollama, dbs, worker_last_run, vram_estimate}

POST   /v1/admin/consolidate
       force run du worker (auth requise)

POST   /v1/admin/decay
       force apply_decay (auth requise)
```

### 16.2 Schemas Pydantic dans `api/schemas.py`. Validation stricte. Rejet 422 sur input invalide.

---

## 17. CLI (`cli.py`)

Typer-based.

```
mnemos serve                     # Lance le serveur
mnemos worker                    # Lance le worker en standalone
mnemos write "<content>" --role user --session foo
mnemos search "<query>" -k 10
mnemos query "<query>"
mnemos facts --subject user
mnemos consolidate               # Force consolidation
mnemos decay                     # Force decay
mnemos export --format jsonl --out backup.jsonl
mnemos stats                     # Stats globales
mnemos doctor                    # Health check + diagnostic
```

`mnemos doctor` vérifie : Ollama up, modèles pullés, DBs migrées, sqlite-vec chargé, espace disque, dernier run du worker.

---

## 18. Plan d'implémentation par phases

**Phase 0 — Setup (0.5j)**
- pyproject.toml + uv install
- structure de dossiers
- alembic init — attention : setup **multi-database** (deux engines episodic/semantic, deux version tables) ; les tables virtuelles vec0 sont invisibles à l'autogenerate → migrations manuelles pour elles
- config.py + logging.py + clock.py (horloge injectable, §6)
- Test : `mnemos doctor` passe

**Phase 1 — LLM infrastructure (1j)**
- ollama_client + model_manager
- embeddings/dense + tests
- Test : embedding bge-m3 round-trip
- Test : ModelManager exclusion mutuelle small/medium
- Test : ModelManager stress test concurrent — pas de deadlock, jamais deux tiers actives (§7.2)

**Phase 2 — Episodic store (1.5j)**
- Schéma DB + migrations
- sparse.py + tests Hamming distance
- EpisodicStore.write + search + apply_decay
- Tests integration : write 100 episodes → search retrieves correctly
- Test : pattern separation (deux épisodes même contenu, timestamps dans des **buckets de 4h différents** → distinguables, cf. §8.2)
- Test : apply_decay idempotent sur runs rapprochés (pas de double-comptage, via Clock mocké)

**Phase 3 — Salience + write path (1j)**
- SalienceTagger + queue de scoring asynchrone (§13.3)
- API POST /v1/episodes complet (write + embed synchrones, tag asynchrone)
- Test : self_ref boost-floor fonctionne
- Test : write path total (API → DB, embedding inclus) en < 500ms ; salience mise à jour après coup

**Phase 4 — Semantic store + consolidation (2j)**
- Schéma facts + entities
- SemanticStore.add_fact avec conflict resolution
- Tests unitaires conflict resolver (5 cas : new, duplicate, superseded [functional], insert additionnel [multi], history ; + normalisation d'alias)
- Extractor + worker
- Test : consolidation loop end-to-end (write 10 episodes → consolidate → query semantic)

**Phase 5 — Router + query API (1j)**
- Classifier
- Orchestrator
- POST /v1/query
- Tests : queries FR + EN, bonne classification

**Phase 6 — Procedural + admin + CLI (1j)**
- ProceduralStore lecture seule
- Endpoints admin
- CLI complet

**Phase 7 — Hardening (1j)**
- Logs structurés partout
- Métriques basiques (Prometheus optionnel, ou juste compteurs en log)
- Worker lock filesystem
- Backup script PowerShell : copie atomique des .db
- README final

**Total estimé : ~9 jours-dev.**

---

## 19. Stratégie de test

### 19.1 Unit
- `test_sparse.py` : Hamming distance, encoding déterministe, séparation temporelle.
- `test_salience.py` : mock LLM, vérifie combined formula, fallback parse error.
- `test_router.py` : classification de 30+ queries FR/EN canoniques.
- `test_conflict_resolver.py` : 5 cas (insert / duplicate / supersede functional / insert additionnel multi / history) + normalisation d'alias.

### 19.2 Integration (avec Ollama réel)
- `test_write_path.py` : POST /v1/episodes → DB cohérent.
- `test_read_path.py` : write 50 → search retrieves top-k corrects.
- `test_consolidation_loop.py` : write 10 episodes contenant des faits → run worker → semantic store cohérent.

### 19.3 E2E
- `test_full_session.py` : simulation d'une session de 50 tours, vérifie que :
  - Working memory taille bornée
  - Episodic décroît si salience faible
  - Facts versionnés correctement (e.g. user dit "je bosse chez X" puis "j'ai changé, je bosse chez Y" → ancien superseded)
  - Query retourne le fait courant

### 19.4 Marqueur `@pytest.mark.requires_ollama`
Skip auto si Ollama down. CI peut runner sans.

---

## 20. Anti-patterns à éviter (debrief)

1. **Pas d'appel direct à Ollama hors ModelManager.** Le ModelManager est l'unique point de sortie. Toute violation = risque de deadlock VRAM.
2. **Pas de write au sémantique sans passer par add_fact.** La résolution de conflit doit toujours s'exécuter.
3. **Pas de création de predicate à la volée.** Vocabulaire fermé. Si un predicate manque, on l'ajoute consciemment dans `ontology.py` et on migre.
4. **Pas de logique métier dans les routes API.** Routes = parsing/validation/délégation. Tout le métier dans les composants.
5. **Pas de mock du sparse coding pour "passer les tests".** Les tests vérifient la pattern separation réelle.
6. **Pas de cosine similarity sur les embeddings sémantiques de faits sans filtrer `valid_until IS NULL`.** Sinon les faits invalidés polluent les résultats — c'est le bug de Mem0 qu'on essaie d'éviter.
7. **Pas de seuil dur "salience < 0.5 = drop".** Toujours `< SALIENCE_THRESHOLD_DECAY_FAST` configurable. Empirique, pas magique.
8. **Pas de retry naïf sur extractor.** Si l'extraction échoue 2× sur un épisode, mark `consolidated_at=now()` avec `extraction_failed=1` (colonne prévue au schéma §5.1) et passe au suivant. Pas de boucle infinie.
9. **Pas de WAL.db laissé traîner.** Backup script doit faire `VACUUM INTO` pour atomicité, pas un copy brut.
10. **Pas de log des contenus utilisateur en production.** Log les IDs, salience scores, timing — jamais le contenu brut sauf en `LOG_LEVEL=DEBUG`.

---

## 21. Quality gates avant merge

Avant tout merge sur `main` :

- [ ] `pytest -m "not requires_ollama"` passe à 100%
- [ ] `pytest -m requires_ollama` passe à 100% sur la machine cible
- [ ] `mnemos doctor` retourne tout ✓
- [ ] `ruff check src tests` clean
- [ ] `mypy src --strict` clean
- [ ] `mnemos stats` après le run e2e montre :
  - Episodes écrits = ceux attendus
  - Facts consolidés cohérents avec extraction
  - Aucun fait courant en doublon (même subject+predicate, valid_until=NULL)
- [ ] Profil GPU : pic VRAM < 7 GB pendant `test_full_session`. Profil dev CPU : aucun swap (RSS Ollama + serveur < 12 GB) et write path p50 < 500 ms pendant `test_full_session`
- [ ] Aucune erreur `OOM` ou `deadlock` dans les logs

---

## 22. Décisions volontairement laissées ouvertes (à arbitrer en Phase 7+)

- **Oubli sémantique** : les faits ont une `confidence` mais aucun mécanisme de décroissance/oubli en MVP — seul l'épisodique oublie. Une décroissance de confiance des faits non renforcés (non re-confirmés par de nouveaux épisodes) est la suite logique, à concevoir quand on aura des données réelles sur la distribution des faits.
- **Rétractation de faits multi-valués** : "je n'aime plus le café" devrait invalider le fait `prefers(user, café)` existant, mais exige que l'extractor détecte les négations de façon fiable. Hors MVP (cf. §10.2) ; en attendant, correction manuelle via l'API admin.
- **Prometheus / OpenTelemetry** : pas en MVP, à ajouter quand le système tournera en continu.
- **Multi-langue prédicats** : pour l'instant prédicats anglais hardcodés. Localisation de l'ontologie = post-v1.
- **Compaction d'épisodes en résumé** : actuellement, archivage = dump JSONL. Une étape "compaction LLM" qui résume des épisodes par session avant archivage est intéressante mais hors scope.
- **TTT / fine-tuning d'un adapter LoRA sur les faits consolidés** : explicitement non-goal. À reconsidérer si les benchmarks montrent un plafond clair sur la sémantique.

---

## 23. Critère "done" du MVP

Un script `scripts/demo.py` qui :
1. Reset les DBs.
2. Écrit 50 messages simulés (utilisateur fictif "Alice" qui change de boulot, déménage, exprime des préférences).
3. Lance le worker une fois.
4. Émet 10 queries vérifiant :
   - Récupération épisodique précise par fenêtre temporelle.
   - Faits actuels reflètent le dernier état (pas un mélange).
   - Historique d'un fait montre la chaîne de versioning.
   - Query ambiguë retourne épisodique + sémantique mergés cohéremment.
5. Affiche un rapport pass/fail par check.

Si ce script passe en vert sur la machine 3070 Ti, le MVP est livré.

---

**Fin du spec. Pour Claude Code : commencer par Phase 0. Lire ce document avant chaque phase. Les "anti-patterns à éviter" sont à relire avant chaque commit.**
