# Rapport de Benchmark LoCoMo — Mnemos

**Date** : 18/09/2026 15:13:34  
**Mode d'évaluation** : `OLLAMA`  
**Dataset** : `locomo_sample.json` (Caroline & Melanie)  

## 1. Performance d'Ingestion (Écriture synchrone)

| Métrique | Valeur |
|---|---|
| Messages stockés | **35** (2 sessions) |
| Temps total d'ingestion | **10.84 s** |
| Latence moyenne / session | **5420.9 ms** |
| Débit d'ingestion | **3.2 messages/sec** |

## 2. Précision de Récupération (Recall@K sur 27 questions)

| Rang de Récupération | Taux de Rappel |
|---|---|
| **MRR (Mean Reciprocal Rank)** | **0.268** |
| **Recall@1** | **14.8%** |
| **Recall@3** | **37.0%** |
| **Recall@5** | **40.7%** |
| **Recall@10** | **51.9%** |
| Latence moyenne de recherche | **125.04 ms** |

## 3. Décomposition par Catégorie Cognitive

| Catégorie | Questions | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|---|
| Single-hop Factuel | 9 | 0.0% | 11.1% | 11.1% | 22.2% |
| Raisonnement Temporel | 4 | 50.0% | 50.0% | 50.0% | 50.0% |
| Multi-hop & Inférence | 1 | 100.0% | 100.0% | 100.0% | 100.0% |
| Domaine Ouvert / Contexte | 8 | 12.5% | 62.5% | 62.5% | 75.0% |
| Gouvernance & Préférences | 5 | 0.0% | 20.0% | 40.0% | 60.0% |
