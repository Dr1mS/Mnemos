# TODO — Lot 2 (qualité d'extraction) — TERMINÉ & VALIDÉ

> **Statut au 18/09/2026** : Lot 2 traité avec succès sur la branche `feat/aml-adapter`.
> Tous les items ont été corrigés, testés et vérifiés par non-régression.

## Synthèse des résolutions

### 1. Couverture des 10 prédicats dans le few-shot (✓ TERMINÉ)
- Le prompt d'extraction v5 (`consolidation/extractor.py`) illustre désormais les 10 prédicats autorisés :
  `works_at`, `lives_in`, `prefers`, `dislikes`, `owns`, `is_a`, `has_attribute`, `knows_about`, `has_goal`, `has_skill`.
- Ajout de règles de démarcation strictes :
  - `works_at` (employeur/organisation) vs `lives_in` (ville/région/pays).
  - `has_skill` (compétence active maîtrisée) vs `has_goal` (aspiration future).
  - `prefers`/`dislikes` (goûts) vs `has_attribute` (traits physiques, métriques, revenus).

### 2. Typage sémantique dans `resolve_object` (✓ TERMINÉ)
- `api/graph.py`, `resolve_object()` : dérivation dynamique du type de nœud selon la sémantique du prédicat :
  - `lives_in` → `lieu`
  - `works_at` → `organisation`
  - `is_a`, `has_attribute` → `personne`
  - `has_goal`, `has_skill`, `owns` → `projet`
  - Évite que des concepts comme "Annecy" ou "Nexora" soient arbitrairement typés en "projet".

### 3. Salience tenant-aware (✓ TERMINÉ)
- `tagger/salience.py` :
  - Le prompt de saillance reçoit désormais `{subject}` (dérivé de `canonical_subject(tenant)`).
  - `SalienceTagger.score` accepte `tenant`.
  - `ScoringJob` transporte le champ `tenant`.
  - Les workers de la file et l'auto-drain passent le tenant lors de l'évaluation du score de saillance.
  - Résout le problème des tenants projets/agents (ex: `atelios`) dont les annonces étaient sous-notées par le prisme user-centrique.

### 4. Faits aberrants réels & Tests de non-régression (✓ TERMINÉ)
- Les cas aberrants documentés (`user — prefers — 1m81`, `user — prefers — zéro revenu`, `user — prefers — 66 à 98% du revenu total`) sont désormais strictement discriminés vers `has_attribute`.
- Fixtures et tests de non-régression ajoutés dans `tests/unit/test_extractor.py` et `tests/unit/test_salience.py`.

### 5. Traçabilité des fallbacks de prédicats (✓ TERMINÉ)
- `map_predicate()` émet un log `warning` explicite avec le prédicat brut, l'objet et le fallback `FALLBACK_PREDICATE`.
