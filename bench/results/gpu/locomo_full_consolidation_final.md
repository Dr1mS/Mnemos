# Rapport de Benchmark LoCoMo — Mnemos

**Date** : 19/09/2026 18:45:02  
**Mode d'évaluation** : `OLLAMA`  
**Dataset** : `locomo_sample.json` (Caroline & Melanie)  

## 1. Performance d'Ingestion (Écriture synchrone)

| Métrique | Valeur |
|---|---|
| Messages stockés | **419** (19 sessions) |
| Temps total d'ingestion | **13.52 s** |
| Latence moyenne / session | **711.6 ms** |
| Débit d'ingestion | **31.0 messages/sec** |

## 2. Précision de Récupération (Recall@K sur 199 questions)

| Rang de Récupération | Taux de Rappel |
|---|---|
| **MRR (Mean Reciprocal Rank)** | **0.257** |
| **Recall@1** | **15.6%** |
| **Recall@3** | **31.2%** |
| **Recall@5** | **39.2%** |
| **Recall@10** | **52.3%** |
| Latence moyenne de recherche | **187.60 ms** |

## 3. Décomposition par Catégorie Cognitive

| Catégorie | Questions | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|
| Single-hop Factuel | 32 | 3.1% | 31.2% | 37.5% | 50.0% |
| Raisonnement Temporel | 37 | 29.7% | 48.6% | 54.1% | 73.0% |
| Multi-hop & Inférence | 13 | 7.7% | 23.1% | 23.1% | 23.1% |
| Domaine Ouvert / Contexte | 70 | 15.7% | 30.0% | 42.9% | 60.0% |
| Gouvernance & Préférences | 47 | 14.9% | 21.3% | 27.7% | 34.0% |
