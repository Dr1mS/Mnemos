# TODO — Lot 2 (qualité d'extraction) — VERROUILLÉ

> Ce lot ne démarre **pas** sans accord explicite. Ce fichier consigne les
> observations d'audit du Lot 1 pour préparer le travail. Les items ci-dessous
> sont **repérés, pas corrigés**.

## Contexte

Le smoke test multi-tenant (`scripts/smoke_tenant.py`, Ollama réel) a produit,
pour le tenant `atelios`, des faits corrects sur le plan de l'isolation et du
subject, mais **de qualité d'extraction inégale** — exactement les symptômes
que le Lot 2 doit traiter :

```
atelios | has_goal  | Rust                                  (object tronqué : perte de "moteur de simulation en")
atelios | works_at  | Grenoble                              (works_at pour une localisation d'équipe → devrait être lives_in/localisation)
atelios | has_goal  | démone publique en septembre          (typo du 4B : "démone" pour "démo")
```

Ces faits sont utiles comme **cas de test de non-régression** du futur fix.

## Items du prompt (à couvrir)

### 1. Couverture des prédicats dans le few-shot (~4/10 couverts)

Le prompt v4 (`consolidation/extractor.py`, `EXTRACTION_PROMPT`) ne montre en
exemple que `owns`, `has_goal`, `works_at`, `prefers`. Les 6 autres
(`lives_in`, `dislikes`, `is_a`, `has_attribute`, `knows_about`, `has_skill`)
ne sont jamais illustrés → le 4B les sous-utilise ou les mappe mal (cf.
`works_at Grenoble` ci-dessus qui aurait dû être `lives_in`).
**À faire** : un exemple de bascule par prédicat, sans casser la longueur qui
fait tenir le v4.

### 2. `resolve_object` force les concepts vers le node type `projet`

`api/graph.py`, `resolve_object()` : tout object non-entité devient une entité
synthétique `type="projet"` par défaut (via `FACT_TYPE`). Un object comme
"Rust" (concept/skill) ou "Annecy" (lieu non enregistré comme entité) atterrit
en `projet`. **À faire** : mapper le type de concept depuis la cardinalité /
le prédicat, et enregistrer les objects récurrents comme entités typées.

### 3. Le plancher de salience dans `combine()` amplifie les erreurs

`tagger/salience.py`, `combine()` : `combined = max(pondération, self_ref)`. Le
boost-floor `self_ref` fait passer le seuil à tout énoncé « sur le user », y
compris des extractions bruitées, ce qui les envoie en consolidation. Sur un
tenant non-personnel, le problème est **inverse** : le tagger est user-centric
(`self_ref` bas pour un énoncé projet), donc les épisodes atelios légitimes
**n'atteignent pas** le seuil 0.6 (constaté au smoke test : sans salience
forcée, `candidats=0` pour atelios). **À faire** :
   - rendre le prompt de salience conscient du sujet canonique du tenant (comme
     l'extracteur l'est désormais, cf. P2) ;
   - réévaluer le plancher `self_ref` (ou le remplacer par un plancher fondé
     sur le sujet canonique du tenant).

### 4. Faits aberrants réels (mémoire actuelle) → tests de non-régression

Cas tirés de la mémoire perso à transformer en tests du futur fix :
- `user — prefers — 1m81` (une taille n'est pas une préférence → `has_attribute`)
- `user — prefers — zéro revenu`
- `user — prefers — 66 à 98% du revenu total`
- noms de projets éparpillés arbitrairement entre `has_goal` / `prefers` /
  `owns`.

**À faire** : figer ces épisodes-sources en fixtures, asserter le fix.

## Observations additionnelles (audit Lot 1)

- **Salience non tenant-aware** (lié à l'item 3) : `SalienceTagger.score` prend
  `content` + `recent_history` mais pas de tenant/sujet. Le prompt parle de
  « the user ». À aligner sur l'extracteur (injection du sujet canonique).
- **`map_predicate` fuzzy cutoff 0.75** : `work_at` → `works_at` fonctionne,
  mais un prédicat FR mal traduit par le 4B tombe en `has_attribute` (fallback)
  sans trace exploitable côté produit. Envisager un log métier / compteur.
