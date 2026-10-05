# Rapport de Benchmark LoCoMo — Mnemos

**Date** : 19/09/2026 16:37:20  
**Mode d'évaluation** : `OLLAMA`  
**Dataset** : `locomo_sample.json` (Caroline & Melanie)  

## 1. Performance d'Ingestion (Écriture synchrone)

| Métrique | Valeur |
|---|---|
| Messages stockés | **35** (2 sessions) |
| Temps total d'ingestion | **0.98 s** |
| Latence moyenne / session | **491.9 ms** |
| Débit d'ingestion | **35.6 messages/sec** |

## 2. Précision de Récupération (Recall@K sur 27 questions)

| Rang de Récupération | Taux de Rappel |
|---|---|
| **MRR (Mean Reciprocal Rank)** | **0.328** |
| **Recall@1** | **22.2%** |
| **Recall@3** | **37.0%** |
| **Recall@5** | **55.6%** |
| **Recall@10** | **63.0%** |
| Latence moyenne de recherche | **153.63 ms** |

## 3. Décomposition par Catégorie Cognitive

| Catégorie | Questions | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|
| Single-hop Factuel | 9 | 0.0% | 33.3% | 44.4% | 44.4% |
| Raisonnement Temporel | 4 | 75.0% | 75.0% | 75.0% | 75.0% |
| Multi-hop & Inférence | 1 | 100.0% | 100.0% | 100.0% | 100.0% |
| Domaine Ouvert / Contexte | 8 | 25.0% | 37.5% | 75.0% | 87.5% |
| Gouvernance & Préférences | 5 | 0.0% | 0.0% | 20.0% | 40.0% |
