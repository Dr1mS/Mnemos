# Rapport d'Évaluation Comparatif des Architectures de Mémoire

**Date d'évaluation** : 2026-09-17 18:43:23  
**Environnement Matériel** : Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz (8 vCPUs), 15.49 Go RAM  
**Moteur LLM / Embeddings** : `qwen2.5:3b` (dry_run: False) | `bge-m3:latest`  
**Méthode Statistique** : Évaluation déterministe (température 0.0, seed 42) sur jeu Dev (70%, 14 instances/épreuve). Intervalles de Wilson à 95% ($z=1.96$), normalisation au plafond Oracle et tests de McNemar par paires vs Mnemos avec correction de Holm-Bonferroni sur l'ensemble des comparaisons multiples.

---

## Évolution de Mnemos : Avant vs Après Correctif (Commit 4f9fc9e + Loop Guard)

| Épreuve / Sonde | Mnemos Avant Correctif | Mnemos Après Correctif | Évolution Absolue |
|---|---|---|---|
| **Sonde 1.1 (Vérité active)** | 100.0% | 100.0% | +0.0% |
| **Sonde 1.2 (Chronologie)** | 57.1% | 50.0% | -7.1% |
| **Sonde 1b.1 (Vérité hors ontologie)** | 78.6% | 78.6% | +0.0% |
| **Sonde 1b.2 (Chronologie hors ontologie)** | 14.3% | 42.9% | +28.6% |
| **Épreuve 2 (Rétention 90 jours)** | 3.6% | 85.7% | +82.1% |
| **Épreuve 3 (Conventions)** | 50.0% | 57.1% | +7.1% |

---

## 1. Tableau Synthétique des Mesures (Jeu Dev)

| Architecture | Famille | Sonde 1.1 [% Oracle] | Sonde 1.2 [% Oracle] | Sonde 1b.1 [% Oracle] | Sonde 1b.2 [% Oracle] | Rétention (90j) [% Oracle] | Conventions [% Oracle] | Lignes Actives / Archivées |
|---|---|---|---|---|---|---|---|---|
| **EmptyControlMemory** | Contrôle Négatif (Mémoire Vide) | 0% [0%] | 0% [0%] | 0% [0%] | 0% [0%] | 0% [0%] | 0% [0%] | 0 / 0 |
| **OracleControlMemory** | Contrôle Positif (Oracle Parfait) | 100% [100%] | 100% [100%] | 64% [100%] | 14% [100%] | 100% [100%] | 71% [100%] | 100 / 0 |
| **SlidingWindowMemory** | Baseline R0 (FIFO Recency Window) | 100% [100%] | 0% [0%] | 100% [156%] | 0% [0%] | 0% [0%] | 0% [0%] | 10 / 0 |
| **ObsidianMarkdownMemory** | Baseline R1 (Append-only & Full-Text) | 0% [0%] | 0% [0%] | 79% [122%] | 7% [50%] | 96% [96%] | 29% [40%] | 100 / 0 |
| **SummaryBufferMemory** | Baseline R2 (Rolling Auto-Summarizer & Compactor) | 100% [100%] | 0% [0%] | 100% [156%] | 0% [0%] | 0% [0%] | 0% [0%] | 5 / 0 |
| **NaiveVectorMemory** | Baseline R3 (RAG Vectoriel Dense Standard) | 36% [36%] | 57% [57%] | 71% [111%] | 43% [300%] | 96% [96%] | 43% [60%] | 100 / 0 |
| **Mem0LikeMemory** | Référence (Vector-CRUD type Mem0) | 64% [64%] | 64% [64%] | 71% [111%] | 50% [350%] | 96% [96%] | 43% [60%] | 100 / 0 |
| **GenerativeAgentsLikeMemory** | Référence (Flux de mémoire type Stanford Generative Agents) | 14% [14%] | 71% [71%] | 71% [111%] | 29% [200%] | 93% [93%] | 79% [110%] | 100 / 0 |
| **GraphitiLikeMemory** | Référence (Graphe de connaissances bi-temporel type Graphiti) | 100% [100%] | 0% [0%] | 64% [100%] | 29% [200%] | 93% [93%] | 21% [30%] | 100 / 0 |
| **Mnemos** | Architecture Multi-Stores (Mnemos) | 100% [100%] | 50% [50%] | 79% [122%] | 43% [300%] | 86% [86%] | 57% [80%] | 2 / 98 |

---

## 2. Analyse Détaillée par Épreuve

### Épreuve 1 : Mutation Temporelle (Ontologie Fermée)

| Système | Sonde 1.1 (Wilson 95%) | % Oracle | McNemar (Holm) | Sonde 1.2 (Wilson 95%) | % Oracle | McNemar (Holm) | Pollution Contexte |
|---|---|---|---|---|---|---|---|
| **EmptyControlMemory** | 0% [0%-22%] | 0% | p_corr=0.005 | 0% [0%-22%] | 0% | non significatif (p_corr=0.671) | 0.0% |
| **OracleControlMemory** | 100% [78%-100%] | 100% | non significatif (p_corr=1.000) | 100% [78%-100%] | 100% | non significatif (p_corr=0.671) | 0.0% |
| **SlidingWindowMemory** | 100% [78%-100%] | 100% | non significatif (p_corr=1.000) | 0% [0%-22%] | 0% | non significatif (p_corr=0.671) | 0.0% |
| **ObsidianMarkdownMemory** | 0% [0%-22%] | 0% | p_corr=0.005 | 0% [0%-22%] | 0% | non significatif (p_corr=0.671) | 100.0% |
| **SummaryBufferMemory** | 100% [78%-100%] | 100% | non significatif (p_corr=1.000) | 0% [0%-22%] | 0% | non significatif (p_corr=0.671) | 0.0% |
| **NaiveVectorMemory** | 36% [16%-61%] | 36% | non significatif (p_corr=0.183) | 57% [33%-79%] | 57% | non significatif (p_corr=1.000) | 73.2% |
| **Mem0LikeMemory** | 64% [39%-84%] | 64% | non significatif (p_corr=1.000) | 64% [39%-84%] | 64% | non significatif (p_corr=1.000) | 50.0% |
| **GenerativeAgentsLikeMemory** | 14% [4%-40%] | 14% | p_corr=0.025 | 71% [45%-88%] | 71% | non significatif (p_corr=1.000) | 64.3% |
| **GraphitiLikeMemory** | 100% [78%-100%] | 100% | non significatif (p_corr=1.000) | 0% [0%-22%] | 0% | non significatif (p_corr=0.671) | 0.0% |
| **Mnemos** | 100% [78%-100%] | 100% | Réf (Mnemos) | 50% [27%-73%] | 50% | Réf (Mnemos) | 0.0% |

### Épreuve 1b : Mutation Temporelle (Hors Ontologie Fermée)

| Système | Sonde 1b.1 (Wilson 95%) | % Oracle | McNemar (Holm) | Sonde 1b.2 (Wilson 95%) | % Oracle | McNemar (Holm) | Pollution Contexte |
|---|---|---|---|---|---|---|---|
| **EmptyControlMemory** | 0% [0%-22%] | 0% | p_corr=0.048 | 0% [0%-22%] | 0% | non significatif (p_corr=1.000) | 0.0% |
| **OracleControlMemory** | 64% [39%-84%] | 100% | non significatif (p_corr=1.000) | 14% [4%-40%] | 100% | non significatif (p_corr=1.000) | 50.0% |
| **SlidingWindowMemory** | 100% [78%-100%] | 156% | non significatif (p_corr=1.000) | 0% [0%-22%] | 0% | non significatif (p_corr=1.000) | 0.0% |
| **ObsidianMarkdownMemory** | 79% [52%-92%] | 122% | non significatif (p_corr=1.000) | 7% [1%-32%] | 50% | non significatif (p_corr=1.000) | 26.8% |
| **SummaryBufferMemory** | 100% [78%-100%] | 156% | non significatif (p_corr=1.000) | 0% [0%-22%] | 0% | non significatif (p_corr=1.000) | 0.0% |
| **NaiveVectorMemory** | 71% [45%-88%] | 111% | non significatif (p_corr=1.000) | 43% [21%-67%] | 300% | non significatif (p_corr=1.000) | 37.5% |
| **Mem0LikeMemory** | 71% [45%-88%] | 111% | non significatif (p_corr=1.000) | 50% [27%-73%] | 350% | non significatif (p_corr=1.000) | 32.1% |
| **GenerativeAgentsLikeMemory** | 71% [45%-88%] | 111% | non significatif (p_corr=1.000) | 29% [12%-55%] | 200% | non significatif (p_corr=1.000) | 37.5% |
| **GraphitiLikeMemory** | 64% [39%-84%] | 100% | non significatif (p_corr=1.000) | 29% [12%-55%] | 200% | non significatif (p_corr=1.000) | 49.4% |
| **Mnemos** | 79% [52%-92%] | 122% | Réf (Mnemos) | 43% [21%-67%] | 300% | Réf (Mnemos) | 37.5% |

### Épreuve 2 : Rétention des Faits Critiques (90 Jours)

| Système | Réponse Exacte (Wilson 95%) | % Oracle | McNemar (Holm) | Rappel Contexte | Bruit Rappel | Lignes Actives / Archivées |
|---|---|---|---|---|---|---|
| **EmptyControlMemory** | 0% [0%-12%] | 0% | p_corr=0.000 | 0% | 0.0% | 0 / 0 |
| **OracleControlMemory** | 100% [88%-100%] | 100% | non significatif (p_corr=1.000) | 100% | 0.0% | 100 / 0 |
| **SlidingWindowMemory** | 0% [0%-12%] | 0% | p_corr=0.000 | 0% | 100.0% | 10 / 0 |
| **ObsidianMarkdownMemory** | 96% [82%-99%] | 96% | non significatif (p_corr=1.000) | 100% | 0.0% | 100 / 0 |
| **SummaryBufferMemory** | 0% [0%-12%] | 0% | p_corr=0.000 | 0% | 100.0% | 5 / 0 |
| **NaiveVectorMemory** | 96% [82%-99%] | 96% | non significatif (p_corr=1.000) | 100% | 75.0% | 100 / 0 |
| **Mem0LikeMemory** | 96% [82%-99%] | 96% | non significatif (p_corr=1.000) | 100% | 75.0% | 100 / 0 |
| **GenerativeAgentsLikeMemory** | 93% [77%-98%] | 93% | non significatif (p_corr=1.000) | 100% | 50.0% | 100 / 0 |
| **GraphitiLikeMemory** | 93% [77%-98%] | 93% | non significatif (p_corr=1.000) | 100% | 75.0% | 100 / 0 |
| **Mnemos** | 86% [68%-94%] | 86% | Réf (Mnemos) | 86% | 0.0% | 2 / 98 |

### Épreuve 3 : Respect des Conventions de Projet Arbitraires

| Système | Succès Conventions (Wilson 95%) | % Oracle | McNemar (Holm) | Règle Cible Rappelée | Règles en Contexte |
|---|---|---|---|---|---|
| **EmptyControlMemory** | 0% [0%-22%] | 0% | non significatif (p_corr=0.359) | 0% | 0.0/5 |
| **OracleControlMemory** | 71% [45%-88%] | 100% | non significatif (p_corr=1.000) | 100% | 2.4/5 |
| **SlidingWindowMemory** | 0% [0%-22%] | 0% | non significatif (p_corr=0.359) | 0% | 0.0/5 |
| **ObsidianMarkdownMemory** | 29% [12%-55%] | 40% | non significatif (p_corr=1.000) | 43% | 0.8/5 |
| **SummaryBufferMemory** | 0% [0%-22%] | 0% | non significatif (p_corr=0.359) | 0% | 0.0/5 |
| **NaiveVectorMemory** | 43% [21%-67%] | 60% | non significatif (p_corr=1.000) | 79% | 1.0/5 |
| **Mem0LikeMemory** | 43% [21%-67%] | 60% | non significatif (p_corr=1.000) | 57% | 0.9/5 |
| **GenerativeAgentsLikeMemory** | 79% [52%-92%] | 110% | non significatif (p_corr=1.000) | 79% | 2.7/5 |
| **GraphitiLikeMemory** | 21% [8%-48%] | 30% | non significatif (p_corr=1.000) | 21% | 0.3/5 |
| **Mnemos** | 57% [33%-79%] | 80% | Réf (Mnemos) | 79% | 0.9/5 |

---

## 3. Notes Méthodologiques & Précisions Techniques
- **Intervalles de confiance de Wilson (95%)** : Calculés avec $z = 1.96$ pour chaque proportion de succès empirique.
- **Normalisation au plafond Oracle (% Oracle)** : Défini par $\text{Score}_{\text{système}} / \text{Score}_{\text{oracle}}$. Permet d'isoler la contribution propre de la mémoire par rapport aux limites d'instruction du modèle.
- **Test de McNemar avec correction de Holm-Bonferroni** : Test exact binomial bilatéral apparié sur chaque item identique entre Mnemos et les autres architectures. La correction séquentielle de Holm est appliquée à l'ensemble des tests pour contrôler rigoureusement le FWER (Family-Wise Error Rate). Toute comparaison affichant $p_{\text{corr}} > 0.05$ est explicitement marquée *non significatif*.
- **Condition négative stricte (Sondes 1.1 et 1b.1)** : Rejet formel si une valeur périmée est présentée comme actuelle.
- **Mesure physique du stockage** : Comptage exact des `Lignes actives / Lignes archivées` pour quantifier la purge du bruit et la persistance des faits critiques.