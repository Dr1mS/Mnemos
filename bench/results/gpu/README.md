# Benchs sur GPU — 19/09/2026

Machine : Windows 11, AMD Ryzen 9 7950X, 32 Go, NVIDIA RTX 4070 Ti 12 Go, Ollama 0.32.6.
Modèles : `bge-m3` (embeddings), `qwen2.5:3b` (saillance, extraction, réponses du comparatif).
Les baselines CPU (i7-6700) restent dans `bench/results/` ; les tests de stress dans
`benchmarks/results/gpu/`.

| Fichier | Contenu |
|---|---|
| `locomo_A_episodic.md` | LoCoMo, 419 messages / 199 questions, sans LLM, via `/add` + `/search` (top_k=10) |
| `locomo_B_consolidation_2sessions.md` | LoCoMo sessions 1-2, avec saillance et consolidation (27 questions) |
| `locomo_full_consolidation_before.md` | LoCoMo complet avec consolidation, avant les correctifs du 19/09 |
| `locomo_full_consolidation_final.md` | Idem avec le prompt d'extraction final (contraintes de santé en `has_attribute`) |
| `personamem_50p_episodic.md` / `.json` | PersonaMem-v2, 50 personas / 148 questions, épisodique seul, top_k=100 |
| `report.md` / `metrics.json` | Comparatif des 10 architectures, **avant** le correctif du routeur (rétention Mnemos 39 %) |
| `memoryagentbench_report.md` | Jeu inspiré de MemoryAgentBench, 8 épreuves |
| `loadtest_public.json` | Test de charge via l'URL publique (`scripts/aml_loadtest.py`) |

Le comparatif Mnemos **après** correctif du routeur et du typage est dans
`../gpu_mnemos_final/` (rétention 96 %, vérité active 100 %).

Limite de mesure : LoCoMo et PersonaMem ne reconnaissent que des épisodes-preuves. Un fait
consolidé ne peut pas être compté, même s'il aide à répondre : ces benchs ne mesurent pas
l'apport de la consolidation à la qualité des réponses.
