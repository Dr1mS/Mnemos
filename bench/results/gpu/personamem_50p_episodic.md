# Rapport PersonaMem-v2 — Mnemos

- **Date** : 2026-09-19T16:33:44 — commit `4b83840-dirty`
- **Mode** : épisodique seul (sans LLM)
- **GPU** : NVIDIA GeForce RTX 4070 Ti — modèles Ollama chargés : `bge-m3:latest` (100% VRAM)
- **Embedding** : `bge-m3` — LLM : `aucun`
- **Protocole** : endpoints AML `/add` + `/search`, top_k=100, Add découpés à 20 messages / 2000 mots, message system exclu
- **Échantillon** : 50 personas, 148 questions (16 écartées faute de preuve dans l'historique ingéré)

## 1. Récupération de la preuve

| | Questions | MRR | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---|---|---|---|---|---|---|---|---|
| **Global** | 148 | 0.283 | 23.0% | 26.4% | 29.7% | 38.5% | 53.4% | 73.6% | 92.6% |

Erreurs de contrat : Add **0** / Search **0**.

## 2. Performance

| Métrique | Valeur |
|---|---|
| Messages ingérés | 11470 (724 requêtes Add) |
| Temps d'ingestion | 225.5 s |
| Débit d'ingestion | 50.9 msg/s |
| Latence Add p50 / p95 | 321 / 372 ms |
| Latence Search p50 / p95 | 205.5 / 218.8 ms |

## 3. Par type de préférence

| | Questions | MRR | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---|---|---|---|---|---|---|---|---|
| anti_stereotypical_pref | 30 | 0.065 | 0.0% | 3.3% | 6.7% | 20.0% | 43.3% | 66.7% | 93.3% |
| ask_to_forget | 32 | 0.969 | 93.8% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| health_and_medical_conditions | 19 | 0.157 | 10.5% | 10.5% | 15.8% | 26.3% | 52.6% | 68.4% | 89.5% |
| neutral_preferences | 28 | 0.089 | 3.6% | 3.6% | 10.7% | 21.4% | 35.7% | 67.9% | 89.3% |
| stereotypical_pref | 19 | 0.076 | 0.0% | 5.3% | 5.3% | 21.1% | 47.4% | 68.4% | 94.7% |
| therapy_background | 20 | 0.101 | 5.0% | 10.0% | 15.0% | 20.0% | 25.0% | 60.0% | 85.0% |

## 4. Par titulaire de la préférence (`who`)

| | Questions | MRR | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---|---|---|---|---|---|---|---|---|
| others | 11 | 0.105 | 0.0% | 9.1% | 27.3% | 36.4% | 54.5% | 72.7% | 100.0% |
| self | 137 | 0.297 | 24.8% | 27.7% | 29.9% | 38.7% | 53.3% | 73.7% | 92.0% |

## 5. Par préférence mise à jour (`updated`)

| | Questions | MRR | R@1 | R@3 | R@5 | R@10 | R@20 | R@50 | R@100 |
|---|---|---|---|---|---|---|---|---|---|
| False | 116 | 0.094 | 3.4% | 6.0% | 10.3% | 21.6% | 40.5% | 66.4% | 90.5% |
| True | 32 | 0.969 | 93.8% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
