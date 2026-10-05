# 📊 Rapport de Benchmark & Stress-Test — Mnemos

*Date d'exécution : 2026-09-19 16:35:11*
*Durée totale du banc de test : 86.37 s*

---

## 1. 📈 Scalabilité Volume & Latence KNN (sqlite-vec)

**Mode** : `fast`

| Volume | Débit Écrit. (ops/s) | Écrit. p50 | Écrit. p95 | KNN p50 | KNN p95 | Base | WAL | Bytes/Ép. |
|--------|----------------------|------------|------------|---------|---------|------|-----|-----------|
| 2,500 | 464.6 | 1.56 ms | 3.27 ms | 9.13 ms | 9.79 ms | 13688.0 KB | 7829.6 KB | 8814 B |
| 5,000 | 427.2 | 2.37 ms | 3.28 ms | 14.30 ms | 15.27 ms | 23188.0 KB | 7829.6 KB | 6352 B |
| 10,000 | 445.7 | 2.26 ms | 3.28 ms | 27.39 ms | 28.22 ms | 46356.0 KB | 7829.6 KB | 5549 B |
| 25,000 | 446.9 | 2.25 ms | 3.23 ms | 65.95 ms | 67.26 ms | 115892.0 KB | 8143.5 KB | 5080 B |

- **Rappel multi-tenant asymétrique** : `100.0%` (Tenant B avec 5% du volume face à 95% Tenant A).
- **Latence de recherche partitionnée Tenant B** : `50.65 ms`.

---

## 2. ⚡ Concurrence, Deadlock & Salience Queue

| Workers Simultanés | Débit Global (ops/s) | Latence p50 | Latence p95 | Latence p99 | Erreurs |
|--------------------|----------------------|-------------|-------------|-------------|---------|
| 1 workers | 372.5 ops/s | 1.34 ms | 4.28 ms | 24.88 ms | 0 |
| 2 workers | 498.0 ops/s | 2.09 ms | 10.31 ms | 31.34 ms | 0 |
| 4 workers | 491.5 ops/s | 2.19 ms | 25.58 ms | 72.51 ms | 0 |
| 8 workers | 362.4 ops/s | 2.73 ms | 43.99 ms | 259.96 ms | 0 |
| 16 workers | 302.5 ops/s | 4.05 ms | 64.74 ms | 873.34 ms | 0 |

- **Intégrité Tiers ModelManager (Pas de Deadlock)** : `✅ CONFORME`
- **Stress-Test Burst Saillance (150 messages)** :
  - Acceptés immédiatement en mémoire : `50`
  - Différés en base sans perte : `100`
  - Temps de résorption (Auto-Drain) : `20.53 s`
  - Épisodes non scorés restants : `0` `✅ Zéro perte`

---

## 3. 🧠 Vérité Cognitive & Intégrité Temporelle

- **Chaos Temporel (Prédicat `lives_in`)** :
  - Total faits générés dans la chaîne : `4`
  - Faits restés actifs (`valid_until IS NULL`) : `1`
  - Continuité stricte `superseded_by` / `valid_from` : `✅ OUI`
  - Dernière vérité retenue : **`Annecy`**
- **Respect cardinalité (ONE vs MANY)** : `✅ CONFORME`
- **Isolation cognitive cross-tenant** : `✅ STRICTE`
- **Déduplication par alias** : `✅ CONFORME`
- **Doublons de faits courants (Quality Gate §21)** : `0`

---

## 4. ⏳ Simulation Longue Durée (365 Jours — Decay & Oubli)

| Jour Virtuel | Moy. Decay Low (0.2) | Moy. Decay Med (0.5) | Moy. Decay High (0.9) | Actifs | Archivés |
|--------------|----------------------|----------------------|-----------------------|--------|----------|
| J+000 | 1.000 | 1.000 | 1.000 | 75 | 0 |
| J+015 | 0.000 | 0.000 | 0.175 | 75 | 0 |
| J+030 | 0.000 | 0.000 | 0.000 | 75 | 0 |
| J+060 | 0.000 | 0.000 | 0.000 | 75 | 0 |
| J+090 | 0.000 | 0.000 | 0.000 | 75 | 0 |
| J+180 | 0.000 | 0.000 | 0.000 | 25 | 50 |
| J+365 | 0.000 | 0.000 | 0.000 | 25 | 50 |

- **Survie à J+365 (Forte saillance)** : `100.0%`
- **Survie à J+365 (Faible saillance)** : `0.0%`
- **Souvenirs purgés & exportés JSONL** : `50`
