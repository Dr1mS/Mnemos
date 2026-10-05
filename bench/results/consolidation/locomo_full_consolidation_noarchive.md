# Rapport de Benchmark LoCoMo — Mnemos

**Date** : 19/09/2026 19:46:32  
**Mode d'évaluation** : `OLLAMA`  
**Dataset** : `locomo_sample.json` (Caroline & Melanie)  

## 1. Performance d'Ingestion (Écriture synchrone)

| Métrique | Valeur |
|---|---|
| Messages stockés | **419** (19 sessions) |
| Temps total d'ingestion | **14.02 s** |
| Latence moyenne / session | **737.6 ms** |
| Débit d'ingestion | **29.9 messages/sec** |

## 2. Précision de Récupération (Recall@K sur 199 questions)

| Rang de Récupération | Taux de Rappel |
|---|---|
| **MRR (Mean Reciprocal Rank)** | **0.256** |
| **Recall@1** | **14.6%** |
| **Recall@3** | **32.7%** |
| **Recall@5** | **40.2%** |
| **Recall@10** | **53.8%** |
| Latence moyenne de recherche | **181.55 ms** |

## 3. Décomposition par Catégorie Cognitive

| Catégorie | Questions | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|
| Single-hop Factuel | 32 | 3.1% | 31.2% | 37.5% | 50.0% |
| Raisonnement Temporel | 37 | 24.3% | 48.6% | 51.4% | 73.0% |
| Multi-hop & Inférence | 13 | 7.7% | 23.1% | 23.1% | 30.8% |
| Domaine Ouvert / Contexte | 70 | 17.1% | 32.9% | 45.7% | 61.4% |
| Gouvernance & Préférences | 47 | 12.8% | 23.4% | 29.8% | 36.2% |
