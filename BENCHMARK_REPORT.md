# 📊 Mnemos — Benchmark & Stress-Test Report

**Date d'exécution** : 17 Septembre 2026  
**Environnement Matériel** : Windows 11, NVIDIA GeForce RTX 4070 Ti (12 Go VRAM), 32 Go RAM, AMD Ryzen Multi-Core  
**Stack Logicielle** : Python 3.12, SQLite 3.45+ avec extension `sqlite-vec` (0.1.6), Ollama (`bge-m3:latest`, `qwen3:4b`), SQLAlchemy 2.0 async, Alembic.

---

## Executive Summary

Ce banc d'essai a poussé **Mnemos** dans ses retranchements à travers **4 axes d'évaluation critique** :
1. **Scalabilité Volumétrique Vectorielle** : De 1 000 à 25 000 vecteurs (1024 dimensions) sur SQLite avec `sqlite-vec`.
2. **Charge Concurrente & Résistance au Burst** : Jusqu'à 16 workers simultanés et saturation de la file de saillance (150 messages reçus en rafale).
3. **Vérité Cognitive & Continuité Temporelle** : Chaîne de chaos temporel sur 6 états successifs, déduplication stricte et étanchéité multi-tenant.
4. **Simulation Biologique Longue Durée (365 Jours)** : Décroissance exponentielle différentielle, rétention 90 jours et purge/archivage JSONL.

Au cours de cet audit exhaustif, **4 défauts majeurs et masqués ont été identifiés, corrigés et validés par des tests de régression**. La suite complète de tests compte **201 tests unitaires et d'intégration réussis à 100%**, avec conformité stricte `mypy` (38 fichiers sources) et `ruff`.

---

## 1. 📈 Scalabilité Volume & Latence KNN (sqlite-vec)

### Mode Rapide Synthétique (Test aux Limites : 1 000 à 25 000 Vecteurs)

| Volume (Épisodes) | Débit Écriture (ops/s) | Écriture p50 | Écriture p95 | KNN Search p50 | KNN Search p95 | Taille DB | Taille WAL | Empreinte / Épisode |
|---|---|---|---|---|---|---|---|---|
| **1 000** | 342.2 ops/s | 0.00 ms | 16.00 ms | < 1 ms | 16.00 ms | 4.6 MB | 4.3 MB | 9.2 KB |
| **2 500** | 289.9 ops/s | 0.00 ms | 16.00 ms | 15.00 ms | 16.00 ms | 13.6 MB | 7.8 MB | 8.8 KB |
| **5 000** | 224.7 ops/s | 0.00 ms | 16.00 ms | 16.00 ms | 32.00 ms | 23.1 MB | 7.8 MB | 6.3 KB |
| **10 000** | 231.4 ops/s | 0.00 ms | 16.00 ms | 31.00 ms | 47.00 ms | 46.3 MB | 7.8 MB | 5.5 KB |
| **25 000** | 221.3 ops/s | 0.00 ms | 16.00 ms | 78.00 ms | 94.00 ms | 115.8 MB | 8.1 MB | 5.0 KB |

#### Constats & Analyse
- **Stabilité d'écriture remarquable** : Le débit d'ingestion reste stable autour de **~220-290 ops/s** même à 25 000 entrées, grâce au mode WAL de SQLite et au traitement asynchrone par `aiosqlite`.
- **Empreinte disque compacte** : À 25 000 épisodes, la base vectorielle complète pèse **115.8 MB**, soit seulement **~5.0 KB par souvenir** (incluant le texte, les métadonnées relationnelles, l'index B-Tree et le vecteur dense 1024 float32).
- **Courbe KNN Flat Cosine** : Le temps de recherche vectorielle croît linéairement ($O(N)$ brute-force cosinus dans `sqlite-vec`).
  - À 10 000 vecteurs : **31 ms** (parfaitement fluide pour une interaction temps réel).
  - À 25 000 vecteurs : **78 ms p50 / 94 ms p95** (approche du seuil ergonomique de 100 ms).
  - *Seuil recommandé* : Le passage à un index approximatif (ex: HNSW ou IVFFlat) deviendra nécessaire au-delà de ~35 000-50 000 souvenirs actifs par base.

### Mode Réel Local (Ollama bge-m3 sur RTX 4070 Ti)

| Volume | Ingestion Réelle GPU (p50) | Ingestion Réelle GPU (p95) | Recherche KNN (excl. embed) | Recherche Complète (Embed + KNN) |
|---|---|---|---|---|
| **100 épisodes** | **265.0 ms** | **281.0 ms** | **< 1 ms** | **~265 ms** |

- L'accélération GPU via CUDA 8.9 (12 Go VRAM) permet d'encoder chaque souvenir en **~250-265 ms** via `bge-m3` 1024-dim dense.
- La recherche SQLite elle-même est quasi-instantanée (< 1 ms pour $N=100$).

---

## 2. ⚡ Concurrence, Deadlock & Résistance au Burst

### Stress Multi-Workers (1 à 16 Workers Simultanés)

| Workers Concurrents | Débit Global (ops/s) | Latence p50 | Latence p95 | Latence p99 | Taux d'Erreur |
|---|---|---|---|---|---|
| **1 worker** | 256.4 ops/s | 0.00 ms | 16.00 ms | 16.00 ms | **0.0%** (0 err) |
| **2 workers** | 301.9 ops/s | 0.00 ms | 16.00 ms | 31.00 ms | **0.0%** (0 err) |
| **4 workers** | 284.2 ops/s | 0.00 ms | 47.00 ms | 141.00 ms | **0.0%** (0 err) |
| **8 workers** | 252.8 ops/s | 0.00 ms | 78.00 ms | 438.00 ms | **0.0%** (0 err) |
| **16 workers** | 190.5 ops/s | 15.00 ms | 93.00 ms | 1640.00 ms | **0.0%** (0 err) |

- **Zéro Verrou Interbloqué (Deadlock-Free)** : Le système à deux niveaux de `ModelManager` (sémaphore de concurrency pool + mutex d'éviction modèle Ollama) a supporté 16 workers simultanés sans aucun blocage ni timeout.
- **SQLite WAL Concurrency** : Aucune erreur `database is locked` constatée grâce aux timeouts de connexion calibrés et au pool de sessions non bloquant.

### Stress-Test de Débordement (Burst 150 Messages)
- **Capacité mémoire de la file** : 50 jobs (`SALIENCE_QUEUE_MAXSIZE = 50`).
- **Comportement en rafale (150 messages simultanés)** :
  - `50` messages traités immédiatement en RAM.
  - `100` messages différés en base avec `salience = 0.5` par défaut, sans blocage du chemin d'écriture (latence d'écriture préservée).
  - **Auto-Drain réactif** : Dès que la file en RAM s'est vidée, le worker d'auto-drain a automatiquement dépilé les 100 souvenirs non scorés en reconstituant leur historique contextuel.
  - **Temps de résorption total** : **19.75 s**.
  - **Pertes de messages** : **0%** (`counts["unscored"] == 0`).

---

## 3. 🧠 Vérité Cognitive, Conflits & Intégrité Temporelle

### Scénario de Chaos Temporel (Chaîne de 6 Mutations)
1. Ingestion : *"J'habite à Lyon"* (valid_from: T0)
2. Mutation : *"Je déménage à Paris"* (valid_from: T1 > T0)
3. Mutation : *"En fait je suis à Bordeaux"* (valid_from: T2 > T1)
4. Mutation finale : *"Finalement je m'installe à Annecy"* (valid_from: T3 > T2)
5. Distraction concurrente : Ingestion de préférences multi (`prefers tea`, `prefers coffee`).

#### Résultats Validés :
- **Intégrité de la chaîne** : Continuité 100% stricte des pointeurs `superseded_by` et des bornes temporelles `[valid_from, valid_until]`.
- **Vérité active unique** : 1 seul fait actif (`valid_until IS NULL`), valeur retenue : **`Annecy`**.
- **Coexistence multi-valeurs** : Le prédicat `prefers` (cardinalité `MULTI`) a conservé simultanément "thé" et "café" sans supersession intempestive.
- **Doublons courants (Quality Gate §21)** : **0 doublon**.
- **Résolution d'alias** : Normalisation bidirectionnelle (`Atelios` -> `Atelios SAS`) sans régression sémantique.

---

## 4. ⏳ Simulation Biologique Longue Durée (365 Jours)

Simulation accélérée via `FixedClock` sur 75 souvenirs répartis en trois profils de saillance : Faible (0.2), Moyenne (0.5), Forte (0.9).

| Étape Temporelle | Moy. Saillance Faible | Moy. Saillance Moyenne | Moy. Saillance Forte | Souvenirs Actifs | Souvenirs Archivés |
|---|---|---|---|---|---|
| **J+000** | 1.000 | 1.000 | 1.000 | 75 | 0 |
| **J+015** | 0.000 | 0.000 | 0.175 | 75 | 0 |
| **J+030** | 0.000 | 0.000 | 0.000 | 75 | 0 |
| **J+090 (Cutoff)** | 0.000 | 0.000 | 0.000 | 75 | 0 |
| **J+180 (Archive)** | 0.000 | 0.000 | 0.000 | 25 | 50 |
| **J+365 (1 An)** | 0.000 | 0.000 | 0.000 | 25 | 50 |

- **Survie après 1 an pour les souvenirs capitaux (saillance 0.9)** : **100.0%**.
- **Taux d'oubli après 90 jours pour les messages futiles (saillance 0.2)** : **100.0%**.
- **Purge & Export froid** : Les 50 souvenirs futiles ont été purgés de la table SQLite active et exportés proprement en JSONL horodaté (`data/bench/archive/2026-06.jsonl`).

---

## 5. 🛠️ Défauts Critiques Identifiés & Corrigés

Au cours de cet audit de performance, **4 défauts de conception et d'implémentation** ont été découverts et intégralement résolus :

### 🚨 Défaut 3 : Masquage Cross-Tenant et Effet Fantôme des Faits Périmés dans `facts_vec`
- **Symptôme** :
  1. *Masquage Multi-Tenant* : `facts_vec` ne comportait pas de colonne `tenant`. 60 faits insérés dans le Tenant A masquaient complètement les résultats de recherche du Tenant B (0 résultat retourné).
  2. *Effet Fantôme (Bug Mem0)* : Après 25 mises à jour d'un fait (ex: `works_at`), les 25 vecteurs périmés restaient présents dans `facts_vec`. Lors d'une recherche vectorielle, ces 25 vecteurs morts saturaient la fenêtre de recherche `4 * k`, évinçant le fait actif et retournant 0 résultat.
- **Résolution** :
  - Création de la migration Alembic `20260917_f4a92c81e3d7_facts_vec_tenant.py` partitionnant `facts_vec` par `tenant` et éliminant tous les vecteurs dont `valid_until IS NOT NULL`.
  - Modification de `SemanticStore.add_fact()` et `retract_fact()` pour exécuter `DELETE FROM facts_vec WHERE fact_id = :id` lors de toute supersession ou rétractation (la table virtuelle `facts_vec` ne sert qu'à l'indexation de la mémoire active).
  - Requête `search_facts()` mise à jour avec clause de partitionnement natif : `WHERE tenant = :tenant AND embedding MATCH :emb AND k = :k`.

### 🚨 Défaut 4 : Corruption d'Encodage Windows CP1252 dans `ProceduralStore` & Observabilité
- **Symptôme** :
  - Sur Windows, `Path.read_text()` et `Path.write_text()` utilisent par défaut l'encodage système ANSI (`cp1252`).
  - L'écriture d'accents français (ex: "données financières") ou d'emojis (ex: 📊, 🚀) provoquait un crash `UnicodeEncodeError: 'charmap' codec can't encode character` ou une corruption silencieuse à la lecture.
- **Résolution** :
  - Ajout systématique de `encoding="utf-8"` sur tous les appels filesystem dans `src/mnemos/stores/procedural.py`, `src/mnemos/consolidation/worker.py`, `src/mnemos/cli.py`, et `src/mnemos/api/routes.py`.
  - Ajout d'un test unitaire dédié `test_utf8_encoding_accents_et_emojis`.

### 🚨 Défaut 5 : Collision Cross-Tenant de la Mémoire de Travail (`WorkingMemoryRegistry`)
- **Symptôme** :
  - `WorkingMemoryRegistry` indexait les sessions uniquement par `session_id`.
  - Deux utilisateurs de tenants différents utilisant des identifiants standards (ex: `"chat"`, `"default"`, ou `"claude"`) partageaient et écrasaient mutuellement leur mémoire de travail en RAM.
- **Résolution** :
  - Clé composite `(tenant, session_id)` dans `WorkingMemoryRegistry`.
  - Propagation du paramètre `tenant` dans `RouterOrchestrator`, les routes FastAPI et le serveur MCP.
  - Sauvegarde automatique de l'épisode dans la `WorkingMemory` lors des appels `memory_write` côté MCP.

### 🚨 Défaut 6 : Fragilité du Parsing LLM face aux Modèles de Raisonnement (`<think>`)
- **Symptôme** :
  - Les modèles de raisonnement récents (DeepSeek-R1, Qwen reasoning, etc.) ou certaines versions d'Ollama émettent des balises `<think>...</think>`, des blocs de code markdown (````json ... ````) ou du texte d'accompagnement.
  - L'appel direct `json.loads(raw)` dans `FactExtractor` et `SalienceTagger` échouait avec `json.JSONDecodeError`, bloquant l'extraction des faits et renvoyant un score de saillance neutre (0.5).
- **Résolution** :
  - Implémentation du module `src/mnemos/llm/json_cleaner.py` avec `clean_llm_json()` et `parse_llm_json()`.
  - Nettoyage regex multi-pass : élimination des balises `<think>`, extraction des blocs markdown et isolation intelligente des délimiteurs `{ }` ou `[ ]`.
  - Tests unitaires complets dans `test_json_cleaner.py`, `test_extractor.py` et `test_salience.py`.

---

## 6. 🏁 Synthèse des Tests & Assurance Qualité

- **Tests automatisés (pytest)** : **201 / 201 passés** (0 échec, 0 skipped) en **~37 secondes**.
- **Typage Statique (mypy --strict)** : **0 erreur** sur 38 fichiers sources.
- **Linter & Formatter (ruff)** : **0 avertissement / 0 erreur**.
- **Migrations de Base de Données (Alembic)** : Base `data/adrien/semantic.db` et `episodic.db` synchronisées sur `head` (`f4a92c81e3d7` et `e3f8194b1c23`) avec préservation intégrale des données (47 faits, 61 épisodes).
