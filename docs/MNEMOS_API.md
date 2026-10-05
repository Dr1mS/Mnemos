# MNEMOS_API.md — contrat HTTP (v1)

Contrat de l'API HTTP de Mnemos après le Lot 1 (multi-tenant). Ce document est
la référence consommée par le repo **Atelios** et tout autre client. Il décrit
les routes, leurs champs, les défauts, et la sémantique du tenant.

- Base URL par défaut : `http://127.0.0.1:8765`
- Toutes les routes sont préfixées `/v1`.
- Auth : optionnelle. Si `API_KEY` est défini côté serveur (`.env`), toute
  requête doit porter le header `X-API-Key: <clé>` (401 sinon). Sinon, ouvert
  (localhost).
- Corps et réponses en JSON. Validation stricte : entrée invalide → `422`.
- Horodatages : entiers, epoch **millisecondes UTC**.

---

## Multi-tenant

Chaque donnée (épisode, fait, entité) appartient à un **tenant** — la cloison
d'isolation. L'isolation est **stricte** : aucune requête ne lit ou n'écrit
hors du tenant demandé.

- Le champ `tenant` est **optionnel partout**. Son défaut est **`user`** (la
  mémoire personnelle historique). Un client qui n'envoie jamais `tenant`
  travaille donc exclusivement sur `user` — comportement identique à l'avant
  Lot 1.
- Sur les `POST`, `tenant` est un champ du corps JSON. Sur les `GET`, c'est un
  query param (`?tenant=...`).
- Contraintes : `tenant` est une chaîne non vide, longueur ≤ 128.
- **Sujet canonique** : les faits extraits par consolidation portent un
  `subject` dérivé du tenant. Pour `user`, `subject = "user"`. Pour un tenant
  applicatif comme `atelios`, `subject = "atelios"` (le nom du tenant, sauf
  mapping explicite côté serveur). Un acteur tiers explicitement nommé dans un
  épisode devient le `subject` de son fait, quel que soit le tenant.

Créer un tenant ne demande aucune opération d'administration : écrire avec un
`tenant` nouveau le fait exister. Il n'y a pas d'étape de provisioning.

---

## Endpoints

### `POST /v1/episodes` — écrire un épisode

Écrit un épisode. L'embedding est calculé de façon synchrone (épisode
immédiatement cherchable) ; le scoring de salience et la consolidation en faits
sont asynchrones.

Corps :

| champ | type | requis | défaut | notes |
|---|---|---|---|---|
| `content` | string | oui | — | 1..32000 caractères |
| `role` | string | oui | — | `user` \| `assistant` \| `system` |
| `session_id` | string \| null | non | `null` | ≤ 256 caractères |
| `tenant` | string | non | `user` | cloison d'isolation |

Réponse `201` — objet épisode :

```json
{
  "id": "01K...",
  "tenant": "user",
  "created_at": 1782727200000,
  "session_id": null,
  "role": "user",
  "content": "...",
  "salience": null,
  "surprise": null, "arousal": null, "self_ref": null, "recurrence": null,
  "decay_state": 1.0,
  "consolidated_at": null,
  "archived": false
}
```

`salience` (et `surprise`/`arousal`/`self_ref`/`recurrence`) valent `null` tant
que le scoring asynchrone n'est pas passé. L'épisode est néanmoins déjà écrit
et cherchable.

### `GET /v1/episodes/search` — recherche épisodique hybride

Query params :

| param | type | défaut | notes |
|---|---|---|---|
| `q` | string | — (requis) | ≥ 1 caractère |
| `k` | int | `10` | 1..100 |
| `session_id` | string \| null | `null` | |
| `min_salience` | float | `0.0` | 0..1 |
| `tenant` | string | `user` | |

Réponse `200` — liste de `{episode, score, dense_sim, sparse_sim, recency}`,
où `episode` est un objet épisode (cf. ci-dessus).

### `GET /v1/episodes/{episode_id}` — un épisode par id

Réponse `200` objet épisode, ou `404` si inconnu. (Note : la lecture par id
n'est pas filtrée par tenant — l'id ULID est unique tous tenants confondus ;
l'objet retourné porte son `tenant`.)

### `GET /v1/facts` — faits courants

Faits actuellement valides (`valid_until` nul) du tenant.

Query params :

| param | type | défaut | notes |
|---|---|---|---|
| `subject` | string \| null | `null` | filtre optionnel (résolu via alias) |
| `predicate` | string \| null | `null` | filtre optionnel |
| `tenant` | string | `user` | |

Réponse `200` — liste d'objets fait :

```json
{
  "id": "01K...",
  "tenant": "user",
  "subject": "user",
  "predicate": "lives_in",
  "object": "Annecy",
  "valid_from": 1782727200000,
  "valid_until": null,
  "confidence": 0.9,
  "superseded_by": null
}
```

### `GET /v1/facts/history` — historique d'un fait

Chaîne de versioning complète (faits invalidés inclus) d'une paire
subject/predicate dans le tenant, du plus ancien au plus récent.

Query params :

| param | type | défaut | notes |
|---|---|---|---|
| `subject` | string | — (requis) | ≥ 1 caractère |
| `predicate` | string | — (requis) | ≥ 1 caractère |
| `tenant` | string | `user` | |

Réponse `200` — liste d'objets fait (mêmes champs que `/v1/facts`). Les faits
supersédés portent `valid_until` et `superseded_by`.

### `POST /v1/query` — requête routée multi-store

Route automatiquement (règles lexicales FR/EN, cf. classifier) vers
l'épisodique, le sémantique, l'historique ou le contexte de session.

Corps :

| champ | type | requis | défaut | notes |
|---|---|---|---|---|
| `q` | string | oui | — | 1..4000 caractères |
| `session_id` | string \| null | non | `null` | ≤ 256 |
| `k` | int | non | `10` | 1..100 |
| `tenant` | string | non | `user` | |

Réponse `200` :

```json
{
  "type": "semantic_fact",
  "episodes": [ {episode, score, ...} ],
  "facts":    [ {fact, score} ],
  "history":  [ {fait} ],
  "working":  [ {content, role, timestamp_ms} ],
  "procedural": ["skill_name", ...]
}
```

### `GET /v1/health` — santé opérationnelle

À appeler à chaque tick. Vérifie ce qui rend Mnemos **utilisable** : les deux
DB répondent à une vraie requête ET l'endpoint d'embedding Ollama (`/api/embed`)
répond. Une panne de `/api/embed` casse `query` ET `write` — d'où une sonde
dédiée, distincte de `/api/version`. Timeout court (2 s) sur la sonde embedding.

Réponse `200` (toujours 200, même en panne — lire les booléens) :

```json
{
  "ok": true,
  "ollama": true,
  "embedding": true,
  "dbs": { "episodic": true, "semantic": true },
  "failures": {},
  "salience_queue_depth": 0,
  "worker_last_run": "2026-07-04T20:00:00+00:00",
  "worker": { "phase": "idle", ... },
  "pending": { "unscored": 0, "consolidation_ready": 0, "consolidation_waiting": 0 }
}
```

- `ok` = `ollama && embedding && toutes les DB`. **C'est le booléen à tester.**
- `failures` : map `{dépendance: message}` nommant précisément ce qui est en
  panne. Clés possibles : `ollama`, `embedding`, `episodic_db`, `semantic_db`.
  Vide si tout va bien. Exemple en panne :
  ```json
  { "ok": false, "ollama": true, "embedding": false,
    "failures": { "embedding": "/api/embed HTTP 404 (modèle bge-m3 ?)" } }
  ```
- Un timeout embedding (> 2 s) est signalé comme panne — un modèle en cold
  start n'est pas prêt à servir ce tick.
- `pending` est `null` si la DB épisodique est en panne (la santé ne dépend pas
  de ce compteur).

Query param optionnel : aucun tenant — `/health` est une vue globale du système.

### `GET /v1/graph` — graphe mémoire (visualiseur)

Query param `tenant` (défaut `user`). Réponse `{entities, facts, memories,
generated_at, tenant}` — filtrée par tenant. Le filtre `subject` utilise le
sujet canonique du tenant. Contrat détaillé dans `api/graph.py`.

### `POST /v1/sessions/{session_id}/reset`

Réinitialise la working memory de la session. `204`. Idempotent.

### `POST /v1/admin/consolidate` — forcer une consolidation

Auth requise (si `API_KEY`). Lance un run du worker. Le worker balaie **tous
les tenants** et écrit les faits de chaque épisode dans **son** tenant. Réponse
`200` : `{candidates, consolidated, extraction_failures, facts_inserted,
facts_superseded, facts_duplicate, entities_upserted}`.

### `POST /v1/admin/decay` — forcer la décroissance

Auth requise. Query param `dry_run` (défaut `false`). Réponse `{scanned,
dry_run, now_ms}`. La décroissance/l'archivage sont des balayages de cycle de
vie appliqués uniformément (indépendants du tenant).

---

## Notes d'implémentation utiles au client

- **Recherche vectorielle et tenant** : les tables vec0 (`episodes_vec`,
  `facts_vec`) n'ont pas de colonne de métadonnée ; le tenant est appliqué au
  JOIN post-KNN. Conséquence : sur une base partagée par de nombreux tenants,
  un `k` donné peut renvoyer moins de `k` résultats du tenant demandé si les
  voisins vectoriels appartiennent à d'autres tenants. En déploiement
  mono-tenant (une paire de DB par tenant, cf. serveur MCP), le cas ne se
  présente pas.
- **Cardinalité des prédicats** : `works_at` et `lives_in` sont *functional*
  (un nouvel object supersède l'ancien) ; les autres (`prefers`, `dislikes`,
  `owns`, `is_a`, `has_attribute`, `knows_about`, `has_goal`, `has_skill`) sont
  *multi* (les valeurs coexistent). Vocabulaire fermé.
- **Non-régression** : un client qui n'envoie jamais `tenant` obtient
  exactement le comportement d'avant le Lot 1 (tenant `user`).
