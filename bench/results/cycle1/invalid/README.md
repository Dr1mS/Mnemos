# Résultats Invalidés (Audit du 17 Septembre 2026)

Les fichiers présents dans ce répertoire (`metrics.json`, `superiority_report.md`, `memoryagentbench_report.md`) ont été formellement invalidés à la suite de l'audit méthodologique consigné dans `bench/AUDIT.md`.

## Motifs de l'invalidation

1. **Absence d'inférence neuronale (`dry_run=True`)** :
   - Aucun appel réel au LLM (`qwen3:4b`) ni au modèle d'embedding (`bge-m3`) n'a été effectué lors de leur génération.
   - Les réponses étaient produites par des stubs textuels simplistes avec des réponses prédéterminées.
   - Les vecteurs provenaient d'un hachage déterministe pseudo-aléatoire (`blake2b`) dénué de toute structure sémantique.

2. **Auto-validation et biais méthodologiques majeurs** :
   - Les réponses du test de complaisance (Test 3) étaient codées en dur pour forcer 100 % de succès pour Mnemos et 0 % pour les concurrents (`if is_mnemos: return True`).
   - Le taux d'oubli du Test 2 était fixé par une formule constante (`1.0 if is_biological else 0.0`).
   - Au Test 3, les systèmes concurrents ne recevaient aucune règle de sécurité dans leur mémoire ni leur prompt.
   - La Sonde 2.2 validait la réponse sur la simple présence du mot-clé dans le contexte, masquant le fait que Mnemos répondait la clé SEC-9482 à une question d'allergie mortelle.

3. **Sur-ajustement (overfitting) et contournement du pipeline** :
   - `MnemosBackend` interceptait en dur les mots-clés du scénario (`lyon`, `paris`, `annecy`, `sec-9482`, `arachide`, `ordre chronologique`) à l'écriture comme au rappel pour exécuter directement du SQL ad-hoc, sans tester le véritable pipeline public de Mnemos.

4. **Jeu de données synthétique non représentatif** :
   - Le pseudo « MemoryAgentBench » reposait sur 8 questions artisanales en français dont les prédictions étaient pré-remplies mot pour mot dans le code, sans rapport avec le véritable banc de recherche ICLR 2026.

5. **Conclusions et labels promotionnels prédéterminés** :
   - Les synthèses et statuts (« CHAMPION », « ÉLIMINÉ », domination sans équivoque) étaient des chaînes statiques pré-rédigées dans le générateur de rapport.

Ces fichiers sont conservés à des fins d'archive et de traçabilité historique.
