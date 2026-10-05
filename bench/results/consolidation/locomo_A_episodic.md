# Rapport de Benchmark LoCoMo — Mnemos

**Date** : 19/09/2026 16:36:22  
**Mode d'évaluation** : `OLLAMA`  
**Dataset** : `locomo_sample.json` (Caroline & Melanie)  

## 1. Performance d'Ingestion (Écriture synchrone)

| Métrique | Valeur |
|---|---|
| Messages stockés | **419** (19 sessions) |
| Temps total d'ingestion | **6.75 s** |
| Latence moyenne / session | **355.3 ms** |
| Débit d'ingestion | **62.0 messages/sec** |

## 2. Précision de Récupération (Recall@K sur 199 questions)

| Rang de Récupération | Taux de Rappel |
|---|---|
| **MRR (Mean Reciprocal Rank)** | **0.291** |
| **Recall@1** | **19.6%** |
| **Recall@3** | **33.2%** |
| **Recall@5** | **40.7%** |
| **Recall@10** | **55.3%** |
| Latence moyenne de recherche | **193.69 ms** |

## 3. Décomposition par Catégorie Cognitive

| Catégorie | Questions | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|
| Single-hop Factuel | 32 | 12.5% | 25.0% | 40.6% | 50.0% |
| Raisonnement Temporel | 37 | 32.4% | 48.6% | 59.5% | 67.6% |
| Multi-hop & Inférence | 13 | 15.4% | 23.1% | 23.1% | 38.5% |
| Domaine Ouvert / Contexte | 70 | 21.4% | 34.3% | 41.4% | 64.3% |
| Gouvernance & Préférences | 47 | 12.8% | 27.7% | 29.8% | 40.4% |
