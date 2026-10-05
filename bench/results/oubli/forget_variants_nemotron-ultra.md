# Oubli : quelle variante aide le répondeur ?

Généré le 30/09/2026 10:53 — 1540 réponses (385 questions × 4 variantes).
Répondeur `nvidia/nemotron-3-ultra-550b-a55b` (nvidia), num_ctx 131072, top_k 100. Appels en échec : 4.

A rien · B tout supprimer (f412fda) · C garder la consigne · D garder consigne + accusé

## validation

| variante | ask_to_forget | autres | global | fuite brute | fuite affirmative | tronqués |
|---|---|---|---|---|---|---|
| A rien | 49.3 % (75) | 39.4 % (142) | 42.9 % (217) | 68 % | 32 % | 0 |
| B tout supprimer (f412fda) | 17.3 % (75) | 42.3 % (142) | 33.6 % (217) | 16 % | 16 % | 0 |
| C garder la consigne | 28.0 % (75) | 43.7 % (142) | 38.2 % (217) | 28 % | 16 % | 0 |
| D garder consigne + accusé | 56.0 % (75) | 39.4 % (142) | 45.2 % (217) | 65 % | 15 % | 0 |

Appariement contre A (perdues / gagnées, McNemar exact) :

- B tout supprimer (f412fda), ask_to_forget : perdues 24, gagnées 0, p = 0.00
- B tout supprimer (f412fda), autres : perdues 8, gagnées 12, p = 0.50
- C garder la consigne, ask_to_forget : perdues 17, gagnées 1, p = 0.00
- C garder la consigne, autres : perdues 8, gagnées 14, p = 0.29
- D garder consigne + accusé, ask_to_forget : perdues 0, gagnées 5, p = 0.06
- D garder consigne + accusé, autres : perdues 5, gagnées 5, p = 1.00

## calibration

| variante | ask_to_forget | autres | global | fuite brute | fuite affirmative | tronqués |
|---|---|---|---|---|---|---|
| A rien | 42.0 % (69) | 41.4 % (99) | 41.7 % (168) | 83 % | 48 % | 0 |
| B tout supprimer (f412fda) | 10.1 % (69) | 47.5 % (99) | 32.1 % (168) | 23 % | 23 % | 0 |
| C garder la consigne | 30.4 % (69) | 47.5 % (99) | 40.5 % (168) | 45 % | 23 % | 0 |
| D garder consigne + accusé | 50.7 % (69) | 45.5 % (99) | 47.6 % (168) | 77 % | 22 % | 0 |

Appariement contre A (perdues / gagnées, McNemar exact) :

- B tout supprimer (f412fda), ask_to_forget : perdues 22, gagnées 0, p = 0.00
- B tout supprimer (f412fda), autres : perdues 5, gagnées 11, p = 0.21
- C garder la consigne, ask_to_forget : perdues 14, gagnées 6, p = 0.12
- C garder la consigne, autres : perdues 3, gagnées 9, p = 0.15
- D garder consigne + accusé, ask_to_forget : perdues 2, gagnées 8, p = 0.11
- D garder consigne + accusé, autres : perdues 4, gagnées 8, p = 0.39

## total

| variante | ask_to_forget | autres | global | fuite brute | fuite affirmative | tronqués |
|---|---|---|---|---|---|---|
| A rien | 45.8 % (144) | 40.2 % (241) | 42.3 % (385) | 75 % | 40 % | 0 |
| B tout supprimer (f412fda) | 13.9 % (144) | 44.4 % (241) | 33.0 % (385) | 19 % | 19 % | 0 |
| C garder la consigne | 29.2 % (144) | 45.2 % (241) | 39.2 % (385) | 36 % | 19 % | 0 |
| D garder consigne + accusé | 53.5 % (144) | 41.9 % (241) | 46.2 % (385) | 71 % | 18 % | 0 |

Appariement contre A (perdues / gagnées, McNemar exact) :

- B tout supprimer (f412fda), ask_to_forget : perdues 46, gagnées 0, p = 0.00
- B tout supprimer (f412fda), autres : perdues 13, gagnées 23, p = 0.13
- C garder la consigne, ask_to_forget : perdues 31, gagnées 7, p = 0.00
- C garder la consigne, autres : perdues 11, gagnées 23, p = 0.06
- D garder consigne + accusé, ask_to_forget : perdues 2, gagnées 13, p = 0.01
- D garder consigne + accusé, autres : perdues 9, gagnées 13, p = 0.52

## Contrôle de fidélité

Sans objet pour ce répondeur : les chiffres de référence du vrai code (26,7 % / 18,7 %) ont été obtenus avec qwen2.5:7b. La fidélité de la simulation elle-même est établie par le run 7B de la même nuit (même cache, mêmes contextes).
