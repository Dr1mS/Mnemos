# Oubli : quelle variante aide le répondeur ?

Généré le 30/09/2026 02:33 — 1540 réponses (385 questions × 4 variantes).
Répondeur `qwen2.5:7b-instruct-q4_K_M` (ollama), num_ctx 24576, top_k 100. Appels en échec : 6.

A rien · B tout supprimer (f412fda) · C garder la consigne · D garder consigne + accusé

## validation

| variante | ask_to_forget | autres | global | fuite brute | fuite affirmative | tronqués |
|---|---|---|---|---|---|---|
| A rien | 24.0 % (75) | 39.4 % (142) | 34.1 % (217) | 68 % | 32 % | 34 |
| B tout supprimer (f412fda) | 18.7 % (75) | 39.4 % (142) | 32.3 % (217) | 16 % | 16 % | 40 |
| C garder la consigne | 18.7 % (75) | 35.9 % (142) | 30.0 % (217) | 28 % | 16 % | 36 |
| D garder consigne + accusé | 29.3 % (75) | 40.1 % (142) | 36.4 % (217) | 65 % | 15 % | 39 |

Appariement contre A (perdues / gagnées, McNemar exact) :

- B tout supprimer (f412fda), ask_to_forget : perdues 8, gagnées 4, p = 0.39
- B tout supprimer (f412fda), autres : perdues 9, gagnées 9, p = 1.00
- C garder la consigne, ask_to_forget : perdues 8, gagnées 4, p = 0.39
- C garder la consigne, autres : perdues 12, gagnées 7, p = 0.36
- D garder consigne + accusé, ask_to_forget : perdues 1, gagnées 5, p = 0.22
- D garder consigne + accusé, autres : perdues 5, gagnées 6, p = 1.00

## calibration

| variante | ask_to_forget | autres | global | fuite brute | fuite affirmative | tronqués |
|---|---|---|---|---|---|---|
| A rien | 20.3 % (69) | 35.4 % (99) | 29.2 % (168) | 83 % | 48 % | 20 |
| B tout supprimer (f412fda) | 26.1 % (69) | 39.4 % (99) | 33.9 % (168) | 23 % | 23 % | 29 |
| C garder la consigne | 24.6 % (69) | 38.4 % (99) | 32.7 % (168) | 45 % | 23 % | 25 |
| D garder consigne + accusé | 26.1 % (69) | 33.3 % (99) | 30.4 % (168) | 77 % | 22 % | 25 |

Appariement contre A (perdues / gagnées, McNemar exact) :

- B tout supprimer (f412fda), ask_to_forget : perdues 5, gagnées 9, p = 0.42
- B tout supprimer (f412fda), autres : perdues 3, gagnées 7, p = 0.34
- C garder la consigne, ask_to_forget : perdues 5, gagnées 8, p = 0.58
- C garder la consigne, autres : perdues 4, gagnées 7, p = 0.55
- D garder consigne + accusé, ask_to_forget : perdues 1, gagnées 5, p = 0.22
- D garder consigne + accusé, autres : perdues 6, gagnées 4, p = 0.75

## total

| variante | ask_to_forget | autres | global | fuite brute | fuite affirmative | tronqués |
|---|---|---|---|---|---|---|
| A rien | 22.2 % (144) | 37.8 % (241) | 31.9 % (385) | 75 % | 40 % | 54 |
| B tout supprimer (f412fda) | 22.2 % (144) | 39.4 % (241) | 33.0 % (385) | 19 % | 19 % | 69 |
| C garder la consigne | 21.5 % (144) | 36.9 % (241) | 31.2 % (385) | 36 % | 19 % | 61 |
| D garder consigne + accusé | 27.8 % (144) | 37.3 % (241) | 33.8 % (385) | 71 % | 18 % | 64 |

Appariement contre A (perdues / gagnées, McNemar exact) :

- B tout supprimer (f412fda), ask_to_forget : perdues 13, gagnées 13, p = 1.00
- B tout supprimer (f412fda), autres : perdues 12, gagnées 16, p = 0.57
- C garder la consigne, ask_to_forget : perdues 13, gagnées 12, p = 1.00
- C garder la consigne, autres : perdues 16, gagnées 14, p = 0.86
- D garder consigne + accusé, ask_to_forget : perdues 2, gagnées 10, p = 0.04
- D garder consigne + accusé, autres : perdues 11, gagnées 10, p = 1.00

## Contrôle de fidélité (validation)

Vrai code, 29/09 : sans oubli 26,7 %, avec l'oubli f412fda 18,7 % sur les 75 questions ask_to_forget. Simulation : A 24.0 %, B 18.7 %. Un écart notable signifierait que la simulation ne reproduit pas la production et que les variantes C et D ne peuvent pas être lues.
