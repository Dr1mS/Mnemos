# Rapport de Benchmark LoCoMo — Mnemos

**Date** : 19/09/2026 18:14:51  
**Mode d'évaluation** : `OLLAMA`  
**Dataset** : `locomo_sample.json` (Caroline & Melanie)  

## 1. Performance d'Ingestion (Écriture synchrone)

| Métrique | Valeur |
|---|---|
| Messages stockés | **419** (19 sessions) |
| Temps total d'ingestion | **12.37 s** |
| Latence moyenne / session | **650.9 ms** |
| Débit d'ingestion | **33.9 messages/sec** |

## 2. Précision de Récupération (Recall@K sur 199 questions)

| Rang de Récupération | Taux de Rappel |
|---|---|
| **MRR (Mean Reciprocal Rank)** | **0.251** |
| **Recall@1** | **14.6%** |
| **Recall@3** | **30.2%** |
| **Recall@5** | **39.2%** |
| **Recall@10** | **52.3%** |
| Latence moyenne de recherche | **187.62 ms** |

## 3. Décomposition par Catégorie Cognitive

| Catégorie | Questions | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|
| Single-hop Factuel | 32 | 0.0% | 21.9% | 34.4% | 50.0% |
| Raisonnement Temporel | 37 | 29.7% | 48.6% | 54.1% | 73.0% |
| Multi-hop & Inférence | 13 | 7.7% | 23.1% | 23.1% | 30.8% |
| Domaine Ouvert / Contexte | 70 | 14.3% | 31.4% | 44.3% | 58.6% |
| Gouvernance & Préférences | 47 | 14.9% | 21.3% | 27.7% | 34.0% |
