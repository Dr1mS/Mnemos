# 📊 Rapport de Benchmark & Stress-Test — Mnemos

*Date d'exécution : 2026-09-17 00:28:33*
*Durée totale du banc de test : 32.61 s*

---

## 1. 📈 Scalabilité Volume & Latence KNN (sqlite-vec)

**Mode** : `real`

| Volume | Débit Écrit. (ops/s) | Écrit. p50 | Écrit. p95 | KNN p50 | KNN p95 | Base | WAL | Bytes/Ép. |
|--------|----------------------|------------|------------|---------|---------|------|-----|-----------|
| 20 | 5.1 | 250.00 ms | 281.00 ms | 15.00 ms | 266.00 ms | 4212.0 KB | 4337.3 KB | 437724 B |
| 50 | 4.1 | 265.00 ms | 281.00 ms | 0.00 ms | 16.00 ms | 4212.0 KB | 4337.3 KB | 175090 B |
| 100 | 3.8 | 265.00 ms | 281.00 ms | 0.00 ms | 16.00 ms | 4224.0 KB | 4337.3 KB | 87668 B |

- **Rappel multi-tenant asymétrique** : `100.0%` (Tenant B avec 5% du volume face à 95% Tenant A).
- **Latence de recherche partitionnée Tenant B** : `265.00 ms`.

---
