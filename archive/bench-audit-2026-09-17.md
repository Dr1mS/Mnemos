# Audit Méthodologique et Choix d'Architecture (Banc de Test Mnemos)

Date : 17 septembre 2026  
Banc : `bench_superiority`  
Auteur : Antigravity IDE / Pair Programming

---

## 1. Choix du Modèle d'Évaluation : Justification de `qwen2.5:3b`

### Précision sur `qwen3` et la réflexion
Bien qu'Ollama permette formellement de désactiver le raisonnement (CoT) sur les modèles récents via l'option API `think: false` ou le préfixe `/no_think` dans le Modelfile, le choix de modèle d'évaluation du banc est arrêté sur **`qwen2.5:3b`**.

### Justification du maintien de `qwen2.5:3b`
1. **Vitesse d'exécution sur CPU** : `qwen2.5:3b` offre une vitesse d'inférence particulièrement élevée sur architecture CPU (latence moyenne observée entre 0.8s et 1.5s par génération de sonde), ce qui est indispensable pour exécuter des centaines d'évaluations dans des délais maîtrisés sans GPU dédié.
2. **Calibrage complet des contrôles (Phase 3a)** : L'ensemble de la calibration empirique des contrôles a été réalisé et validé sur `qwen2.5:3b` :
   - Contrôle négatif (`EmptyControlMemory`) : 0.0% de façon stricte sur l'ensemble des épreuves (aucun faux positif, aucune fuite d'information).
   - Contrôle positif (`OracleControlMemory`) : 100% sur les épreuves 1.1, 1.2 et 2, et 78.6% sur les conventions de code (erreurs du modèle et non de l'évaluation).
   Changer de modèle à ce stade briserait l'étalonnage des contrôles méthodologiques établi en Phase 3a et introduirait des biais de comparaison.
3. **Fidélité au contexte et concision factuelle** : En tant que modèle pur instruct, il répond directement aux consignes sans divergence de style ni dépendance au bon support du flag de réflexion par la version d'Ollama installée.

---

## 2. Audit Historique Git de `FactExtractor`

Un audit approfondi de l'historique Git du dépôt Mnemos a été conduit pour déterminer l'origine du prédicat `lives_in` :
- **Date d'introduction** : 2 juillet 2026 à 22:52:12 CEST.
- **Commit** : `2b5c7be94ca2e2d2ac2e139e68e46c95bde16f47` (*"Phase 4 : semantic store + consolidation — versioning des faits"*).
- **Auteur** : Dr1mS (`dany.coache@gmail.com`).
- **Fichier** : `src/mnemos/ontology.py`.
- **Vocabulaire fermé de l'ontologie** :
  ```python
  PREDICATES = (
      "works_at",
      "lives_in",
      "prefers",
      "dislikes",
      "owns",
      "is_a",
      "has_attribute",
      "knows_about",
      "has_goal",
      "has_skill",
  )
  ```
- **Conclusion formelle** : Le prédicat `lives_in` fait partie intégrante de l'architecture initiale de Mnemos depuis sa conception en juillet 2026. Il n'a en aucun cas été ajouté rétroactivement pour favoriser Mnemos lors de la conception du banc de test.

### Explication de l'extraction différentielle (Test 1 vs Test 2)
- Dans le **Test 1**, les énoncés portent sur le lieu d'habitation (*"J'habite à Lyon"*, *"Je vis à Annecy"*), qui correspondent directement au prédicat `lives_in` listé textuellement dans les consignes et exemples du prompt d'extraction de Mnemos.
- Dans le **Test 2**, la clé de staging (*"SEC-9482"*) et l'allergie aux arachides ne disposent d'aucun prédicat équivalent dans la liste fermée de 10 relations (pas de `has_credential`, `has_allergy` ou `medical_condition`). Contraint par la règle d'ontologie stricte et l'interdiction de créer des prédicats à la volée, `FactExtractor` renvoie `facts: []`.

---

## 3. Synthèse des Corrections Méthodologiques Appliquées

1. **Correction Sonde 1.1** :
   - `OracleControlMemory.recall` ne restitue désormais que le fait actif pour la sonde 1.1.
   - Application d'une validation stricte avec condition négative (`check_active_residence_answer`) rejetant toute réponse affirmant ou suggérant une ancienne ville de résidence comme actuelle.
2. **Refonte du Test 3 (Conventions de Projet)** :
   - Remplacement des requêtes d'attaques de sécurité (qui mesuraient l'alignement intrinsèque de Qwen plutôt que la mémoire) par 5 conventions de projet arbitraires (`[RULE-HTTP]`, `[RULE-LOG]`, `[RULE-DATE]`, `[RULE-AUTH]`, `[RULE-STORAGE]`).
   - Critère de validité empirique vérifié sur `qwen2.5:3b` : **EmptyControl = 0%**, **OracleControl = 100%**.
3. **Bug Mnemos Corrigé dans `src/` (Commit dédié `4f9fc9e`)** :
   - `worker.py` : ne marque plus un épisode comme consolidé si `len(extraction.facts) == 0`.
   - `episodic.py` : la règle 1 d'archivage protège désormais les épisodes à haute saillance (`salience >= SALIENCE_THRESHOLD_CONSOLIDATE`), empêchant leur purge même après décroissance.
   - Test unitaire dédié validé dans `tests/unit/test_consolidation_zero_facts_bug.py`.
4. **Variabilité & Partitionnement (N >= 20)** :
   - Générateur paramétrique de 20 instances par test.
   - 70% Dev (14 instances) exécutées en Phase 3a.
   - 30% Hold-out (6 instances) isolées dans `bench/datasets_holdout.py`.
   - Calcul des intervalles de confiance de Wilson à 95% ($z=1.96$).
5. **Métriques de Stockage** :
   - Remplacement de l'affichage des octets bruts par les compteurs d'items : `Lignes actives / Lignes archivées`.
