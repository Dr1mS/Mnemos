# Rapport d'Évaluation Comparatif des Architectures de Mémoire

**Date d'évaluation** : 2026-09-19 18:40:29  
**Environnement Matériel** : AMD64 Family 25 Model 97 Stepping 2, AuthenticAMD (32 vCPUs), 0.0 Go RAM  
**Moteur LLM / Embeddings** : `qwen2.5:3b` (dry_run: False) | `bge-m3:latest`  
**Méthode Statistique** : Évaluation déterministe (température 0.0, seed 42) sur jeu Dev (70%, 14 instances/épreuve). Intervalles de Wilson à 95% ($z=1.96$), normalisation au plafond Oracle et tests de McNemar par paires vs Mnemos avec correction de Holm-Bonferroni sur l'ensemble des comparaisons multiples.

---

## Évolution de Mnemos : Avant vs Après Correctif (Commit 4f9fc9e + Loop Guard)

| Épreuve / Sonde | Mnemos Avant Correctif | Mnemos Après Correctif | Évolution Absolue |
|---|---|---|---|
| **Sonde 1.1 (Vérité active)** | 100.0% | 100.0% | +0.0% |
| **Sonde 1.2 (Chronologie)** | 57.1% | 71.4% | +14.3% |
| **Sonde 1b.1 (Vérité hors ontologie)** | 78.6% | 78.6% | +0.0% |
| **Sonde 1b.2 (Chronologie hors ontologie)** | 14.3% | 42.9% | +28.6% |
| **Épreuve 2 (Rétention 90 jours)** | 3.6% | 96.4% | +92.8% |
| **Épreuve 3 (Conventions)** | 50.0% | 64.3% | +14.3% |

---

## 1. Tableau Synthétique des Mesures (Jeu Dev)

| Architecture | Famille | Sonde 1.1 [% Oracle] | Sonde 1.2 [% Oracle] | Sonde 1b.1 [% Oracle] | Sonde 1b.2 [% Oracle] | Rétention (90j) [% Oracle] | Conventions [% Oracle] | Lignes Actives / Archivées |
|---|---|---|---|---|---|---|---|---|
| **Mnemos** | Architecture Multi-Stores (Mnemos) | 100% [N/A] | 71% [N/A] | 79% [N/A] | 43% [N/A] | 96% [N/A] | 64% [N/A] | 3 / 98 |

---

## 2. Analyse Détaillée par Épreuve

### Épreuve 1 : Mutation Temporelle (Ontologie Fermée)

| Système | Sonde 1.1 (Wilson 95%) | % Oracle | McNemar (Holm) | Sonde 1.2 (Wilson 95%) | % Oracle | McNemar (Holm) | Pollution Contexte |
|---|---|---|---|---|---|---|---|
| **Mnemos** | 100% [78%-100%] | N/A | Réf (Mnemos) | 71% [45%-88%] | N/A | Réf (Mnemos) | 0.0% |

### Épreuve 1b : Mutation Temporelle (Hors Ontologie Fermée)

| Système | Sonde 1b.1 (Wilson 95%) | % Oracle | McNemar (Holm) | Sonde 1b.2 (Wilson 95%) | % Oracle | McNemar (Holm) | Pollution Contexte |
|---|---|---|---|---|---|---|---|
| **Mnemos** | 79% [52%-92%] | N/A | Réf (Mnemos) | 43% [21%-67%] | N/A | Réf (Mnemos) | 37.5% |

### Épreuve 2 : Rétention des Faits Critiques (90 Jours)

| Système | Réponse Exacte (Wilson 95%) | % Oracle | McNemar (Holm) | Rappel Contexte | Bruit Rappel | Lignes Actives / Archivées |
|---|---|---|---|---|---|---|
| **Mnemos** | 96% [82%-99%] | N/A | Réf (Mnemos) | 100% | 0.0% | 3 / 98 |

### Épreuve 3 : Respect des Conventions de Projet Arbitraires

| Système | Succès Conventions (Wilson 95%) | % Oracle | McNemar (Holm) | Règle Cible Rappelée | Règles en Contexte |
|---|---|---|---|---|---|
| **Mnemos** | 64% [39%-84%] | N/A | Réf (Mnemos) | 79% | 0.9/5 |

---

## 3. Notes Méthodologiques & Précisions Techniques
- **Intervalles de confiance de Wilson (95%)** : Calculés avec $z = 1.96$ pour chaque proportion de succès empirique.
- **Normalisation au plafond Oracle (% Oracle)** : Défini par $\text{Score}_{\text{système}} / \text{Score}_{\text{oracle}}$. Permet d'isoler la contribution propre de la mémoire par rapport aux limites d'instruction du modèle.
- **Test de McNemar avec correction de Holm-Bonferroni** : Test exact binomial bilatéral apparié sur chaque item identique entre Mnemos et les autres architectures. La correction séquentielle de Holm est appliquée à l'ensemble des tests pour contrôler rigoureusement le FWER (Family-Wise Error Rate). Toute comparaison affichant $p_{\text{corr}} > 0.05$ est explicitement marquée *non significatif*.
- **Condition négative stricte (Sondes 1.1 et 1b.1)** : Rejet formel si une valeur périmée est présentée comme actuelle.
- **Mesure physique du stockage** : Comptage exact des `Lignes actives / Lignes archivées` pour quantifier la purge du bruit et la persistance des faits critiques.