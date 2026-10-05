# POC — Sélection des modèles (2026-07-02)

Bench des charges réelles de Mnemos (embedding / salience §13.2 / extraction §15.2)
sur la machine de dev : **i7-6700 (4c/8t), 16 GB RAM, CPU-only, Ubuntu 24.04, Ollama 0.24**.
Corpus : 22 épisodes FR/EN (persona Alice), 15 gold facts, 9 pièges
(négation, hypothétique, coréférence, tiers, incertitude, sarcasme, temps passé).

Script : `bench_models.py` (stdlib only). Sorties brutes : `raw_extractions.jsonl`.

## Modèles retenus

| Rôle | Spec v1.1 | **Retenu (dev CPU)** | Mesure |
|---|---|---|---|
| Embedding | bge-m3 | **bge-m3** (inchangé) | p50 231 ms — sous la cible write path < 500 ms |
| Salience | qwen2.5:3b | **qwen3:4b** `think=false` | plausibilité **17/17** (2.5:3b : 15/17), p50 11.2 s (async) |
| Extraction | qwen2.5:7b | **qwen3:4b** `think=false` | qualité ≥ 7B, **2.2× plus rapide** (p50 10.8 s vs 23.7 s), 0 prédicat hors vocab |

Un seul LLM résident (~2.5 GB) + bge-m3 (1.2 GB) ≈ 3.7 GB : plus de swap de
modèle entre tiers, batch de consolidation ~3× plus court.

**⚠ qwen3 = famille thinking : `think=false` obligatoire** (le mode thinking
casse le JSON structuré sous Ollama — issue ollama#10929 — et multiplie la
latence CPU par 5-10×).

## Prompt d'extraction : v4 remplace §15.2

Itérations : spec → v2 (règles langue/passé : régression subject) → v3
(sur-suppression du passé composé et de has_goal) → **v4** (règles
rééquilibrées + 6 exemples de bascule — un 4B suit mal les règles abstraites,
il suit bien les exemples).

| Prompt (qwen3:4b) | Rappel | Pièges | Pièges après porte salience | Silence sur bruit |
|---|---|---|---|---|
| spec §15.2 | 9/15 | 2/9 | 1/9 | 4/7 |
| **v4** | **12/15** | 1/9 | **0/9** | **7/7** |

Apports v4 : temps passé (« avant je bossais chez TechCorp » n'écrase plus le
job courant — c'était le pire piège : `works_at` est functional, le fait
périmé aurait *supersédé* le fait vrai), hypothétiques, sujets canoniques
(`user`/entité nommée), objects non traduits, confidence float.

## Défense en profondeur validée (end-to-end)

La porte de salience (combined ≥ 0.6) et l'extracteur se complètent :
- salience trappe : émotionnel non-perso (France 5-0 → 0.58), coréférences
  orphelines (moto → 0.57), sarcasme, small talk ;
- extracteur v4 trappe : temps passé, hypothétiques, questions, tiers.

Aucun épisode porteur de gold n'est filtré à tort par la porte.

## À reporter dans le spec au moment de l'implémentation

1. §2/§4 : modèles → `qwen3:4b` (+ `think=false` dans OllamaClient), profil
   matériel CPU (budget RAM remplace budget VRAM).
2. §15.2 : prompt v4 (cf. `EXTRACTION_PROMPT_V4` dans `bench_models.py`).
3. §7 : ModelManager — un seul modèle LLM ⇒ l'exclusion SMALL/MEDIUM devient
   une simple limite de concurrence (à conserver quand même : garde-fou si on
   ré-introduit un modèle MEDIUM).

## Limites connues (acceptées pour le MVP)

- « Je déteste les open spaces » : sur-supprimé par v4 (un exemple `dislikes`
  dans le prompt réglerait probablement — rendements décroissants).
- Doubles faits dans une phrase : souvent 1/2 extrait.
- Corpus de 22 épisodes : smoke test, pas une éval statistique.
