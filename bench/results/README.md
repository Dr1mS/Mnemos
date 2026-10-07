# Résultats des benchs

Un dossier par sujet. Chaque fichier `.json` est le rapport complet d'un run, et le `.log`
de même nom sa sortie console. Les `*.cache.jsonl` sont des caches de reprise : ils restent
en local et ne sont jamais commités. Le contexte et les conclusions de chaque mesure sont dans
`docs/ameliorations-2e-full.md`.

| Dossier | Ce qu'on y mesure | Scripts |
|---|---|---|
| `classement/` | formule de score de la recherche (recouvrement contre Hamming, récence) et non-régression LoCoMo / D1 dur | `bench_rerank_variants.py`, `bench_locomo_qa.py`, `bench_update_hard.py` |
| `oubli/` | détection des demandes d'oubli, ciblage des échos, variantes A-D, vérification sur le vrai code ; rang de la consigne dans le top 100, simulation de son retour dans le contexte, lien consigne ↔ accusé (validation sur 240 personas neufs) | `bench_forget_detect.py`, `bench_forget_targeting.py`, `bench_forget_variants.py`, `bench_personamem_qa.py`, `bench_oubli_consigne_rang.py`, `bench_oubli_injection.py`, `bench_oubli_placement.py`, `bench_oubli_rattachement.py`, `bench_oubli_precision.py` |
| `consolidation/` | faits extraits par un modèle de langage : débit et effet sur les réponses (conclusion : éteinte) | `bench_update_hard.py`, `bench_locomo_qa.py`, `bench_knowledge_update.py` |
| `dates/` | rappel des preuves temporelles, consignes officielles, normaliseur de dates (LoCoMo brut et Refined) | `bench_temporal_recall.py`, `bench_locomo_qa.py --consignes officielles`, `bench_dates_reponse.py`, `bench_dates_precision.py` |
| `multisauts/` | preuves multi-sauts manquées et leviers de sélection (conclusion : aucun levier simple) | `bench_multisauts_diag.py`, `bench_multisauts_leviers.py` |
| `carte/` | carte des faiblesses : preuve dans le top 100, réponse A et plafond O, consignes officielles | `bench_carte_personamem.py`, `bench_carte_longmemeval.py`, `bench_clbench_couverture.py` (CL-bench, sans modèle) |
| `charge/` | tests de charge sur l'API publique | `scripts/aml_loadtest.py` |
| `categories/` | catégories G (règles) et H (abstention) | `bench_rules.py`, `bench_abstention.py` |
| `cycle1/` | résultats du premier cycle (septembre), dont le banc « superiority » invalidé (`invalid/`) | anciens scripts de `bench/` |
