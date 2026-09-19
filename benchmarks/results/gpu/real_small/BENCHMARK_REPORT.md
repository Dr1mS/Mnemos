# 📊 Rapport de Benchmark & Stress-Test — Mnemos

*Date d'exécution : 2026-09-19 16:35:36*
*Durée totale du banc de test : 24.84 s*

---

## 1. 📈 Scalabilité Volume & Latence KNN (sqlite-vec)

**Mode** : `real`

| Volume | Débit Écrit. (ops/s) | Écrit. p50 | Écrit. p95 | KNN p50 | KNN p95 | Base | WAL | Bytes/Ép. |
|--------|----------------------|------------|------------|---------|---------|------|-----|-----------|
| 20 | 5.8 | 212.86 ms | 282.25 ms | 3.85 ms | 172.98 ms | 4212.0 KB | 4337.3 KB | 437724 B |
| 50 | 5.6 | 205.67 ms | 222.34 ms | 3.72 ms | 4.01 ms | 4212.0 KB | 4337.3 KB | 175090 B |
| 100 | 5.1 | 205.64 ms | 225.19 ms | 3.68 ms | 4.29 ms | 4224.0 KB | 4337.3 KB | 87668 B |

- **Rappel multi-tenant asymétrique** : `100.0%` (Tenant B avec 5% du volume face à 95% Tenant A).
- **Latence de recherche partitionnée Tenant B** : `204.79 ms`.

---
