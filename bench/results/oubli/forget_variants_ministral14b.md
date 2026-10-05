# Oubli : quelle variante aide le répondeur ?

Généré le 30/09/2026 01:50 — 1540 réponses (385 questions × 4 variantes).
Répondeur `ministral-14b-2512` (mistral), num_ctx 131072, top_k 100. Appels en échec : 0.

A rien · B tout supprimer (f412fda) · C garder la consigne · D garder consigne + accusé

## validation

| variante | ask_to_forget | autres | global | fuite brute | fuite affirmative | tronqués |
|---|---|---|---|---|---|---|
| A rien | 20.0 % (75) | 39.4 % (142) | 32.7 % (217) | 68 % | 32 % | 0 |
| B tout supprimer (f412fda) | 13.3 % (75) | 43.7 % (142) | 33.2 % (217) | 16 % | 16 % | 0 |
| C garder la consigne | 13.3 % (75) | 44.4 % (142) | 33.6 % (217) | 28 % | 16 % | 0 |
| D garder consigne + accusé | 29.3 % (75) | 38.7 % (142) | 35.5 % (217) | 65 % | 15 % | 0 |

Appariement contre A (perdues / gagnées, McNemar exact) :

- B tout supprimer (f412fda), ask_to_forget : perdues 10, gagnées 5, p = 0.30
- B tout supprimer (f412fda), autres : perdues 11, gagnées 17, p = 0.34
- C garder la consigne, ask_to_forget : perdues 9, gagnées 4, p = 0.27
- C garder la consigne, autres : perdues 10, gagnées 17, p = 0.25
- D garder consigne + accusé, ask_to_forget : perdues 3, gagnées 10, p = 0.09
- D garder consigne + accusé, autres : perdues 9, gagnées 8, p = 1.00

## calibration

| variante | ask_to_forget | autres | global | fuite brute | fuite affirmative | tronqués |
|---|---|---|---|---|---|---|
| A rien | 24.6 % (69) | 38.4 % (99) | 32.7 % (168) | 83 % | 48 % | 0 |
| B tout supprimer (f412fda) | 10.1 % (69) | 49.5 % (99) | 33.3 % (168) | 23 % | 23 % | 0 |
| C garder la consigne | 14.5 % (69) | 51.5 % (99) | 36.3 % (168) | 45 % | 23 % | 0 |
| D garder consigne + accusé | 30.4 % (69) | 39.4 % (99) | 35.7 % (168) | 77 % | 22 % | 0 |

Appariement contre A (perdues / gagnées, McNemar exact) :

- B tout supprimer (f412fda), ask_to_forget : perdues 12, gagnées 2, p = 0.01
- B tout supprimer (f412fda), autres : perdues 5, gagnées 16, p = 0.03
- C garder la consigne, ask_to_forget : perdues 10, gagnées 3, p = 0.09
- C garder la consigne, autres : perdues 5, gagnées 18, p = 0.01
- D garder consigne + accusé, ask_to_forget : perdues 6, gagnées 10, p = 0.45
- D garder consigne + accusé, autres : perdues 6, gagnées 7, p = 1.00

## total

| variante | ask_to_forget | autres | global | fuite brute | fuite affirmative | tronqués |
|---|---|---|---|---|---|---|
| A rien | 22.2 % (144) | 39.0 % (241) | 32.7 % (385) | 75 % | 40 % | 0 |
| B tout supprimer (f412fda) | 11.8 % (144) | 46.1 % (241) | 33.2 % (385) | 19 % | 19 % | 0 |
| C garder la consigne | 13.9 % (144) | 47.3 % (241) | 34.8 % (385) | 36 % | 19 % | 0 |
| D garder consigne + accusé | 29.9 % (144) | 39.0 % (241) | 35.6 % (385) | 71 % | 18 % | 0 |

Appariement contre A (perdues / gagnées, McNemar exact) :

- B tout supprimer (f412fda), ask_to_forget : perdues 22, gagnées 7, p = 0.01
- B tout supprimer (f412fda), autres : perdues 16, gagnées 33, p = 0.02
- C garder la consigne, ask_to_forget : perdues 19, gagnées 7, p = 0.03
- C garder la consigne, autres : perdues 15, gagnées 35, p = 0.01
- D garder consigne + accusé, ask_to_forget : perdues 9, gagnées 20, p = 0.06
- D garder consigne + accusé, autres : perdues 15, gagnées 15, p = 1.00

## Contrôle de fidélité

Sans objet pour ce répondeur : les chiffres de référence du vrai code (26,7 % / 18,7 %) ont été obtenus avec qwen2.5:7b. La fidélité de la simulation elle-même est établie par le run 7B de la même nuit (même cache, mêmes contextes).
