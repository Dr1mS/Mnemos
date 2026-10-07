# Améliorations de Mnemos pour le 2e Full AML (fin septembre – octobre 2026)

Ce document explique ce qui a changé dans Mnemos depuis le 1er Full, pourquoi, et ce que
chaque changement rapporte, mesuré sur des jeux publics. Il dit aussi ce qui a été essayé et
abandonné, pour ne pas le refaire.

Tous les chiffres viennent de nos propres benchs (`bench/`), sur des données publiques ou
fabriquées. Aucun score privé de la plateforme n'y figure.

**État du déploiement** : rien de ce qui suit ne tourne encore en production. La prod sert le
code du 1er Full. Le nouveau code sera déployé avec la remise à zéro de la base, avant le 2e
Full.

---

## En un coup d'œil

| Amélioration | Ce que ça change | Effet mesuré | Réglage |
|---|---|---|---|
| **Classement de la recherche** | le score lexical compte les mots de la question présents dans le souvenir | D1 dur : bonne valeur dans le top 3, 5-8/13 → **13/13** ; PersonaMem : preuve dans le top 10, 35 % → **51 %** ; résultats reproductibles | toujours actif |
| **Oubli** (« forget that I… ») | les anciennes affirmations de l'assistant sont effacées ; la demande et l'accusé sont gardés | questions d'oubli PersonaMem : 26,7 % → **30,7 %** (7B), 45,8 % → **53,5 %** (Nemotron) | toujours actif |
| **Dates relatives résolues à l'écriture** | « yesterday » devient « yesterday (27 August 2023) » dans le souvenir stocké | questions de dates LoCoMo : 108 → **166** sur 282 (Nemotron), 139 → **178** sur 280 (7B) | `RELATIVE_DATES_ANNOTATION` (activé) |

---

## 1. Classement de la recherche

**Le problème.** Le score d'un souvenir mélange une similarité de sens (embedding, 70 %) et une
similarité lexicale (30 %). La partie lexicale utilisait une distance de Hamming sur des
empreintes de 256 bits. Elle avait deux défauts mesurés :
- **elle favorisait les messages courts** : à mots communs égaux, un message long était jugé
  plus loin ;
- **elle dépendait de l'heure de la requête** : 32 des 256 bits encodaient une tranche de 4 h.
  Le même code, sur les mêmes données, ne donnait pas le même classement selon l'heure (14
  réponses sur 50 différentes d'un run à l'autre).

**Ce qui a changé** (`src/mnemos/embeddings/sparse.py`, `query_coverage`). La partie lexicale
mesure désormais **la part des mots de la question présents dans le souvenir**, sur les seuls
bits de contenu. Les poids sont normalisés (somme 1).

**Effet** (`bench/bench_rerank_variants.py`, mêmes candidats, seule la formule change) :

| | avant | après |
|---|---|---|
| D1 dur : valeur actuelle dans le top 3 | 5 à 8 / 13 | **13 / 13** |
| PersonaMem : preuve dans le top 10 (rang médian) | 35 % (16) | **51 % (6)** |
| LoCoMo : preuve dans le top 10 (rang médian) | 59-63 % (6) | **67 % (3)** |
| même mesure relancée à une autre heure | 14 réponses / 50 différentes | **0** |

Réponses LoCoMo (150 questions) : 71 → 74, sans régression dans aucune catégorie.

**Essayé et rejeté** : rendre la récence « vivante » en mesurant l'âge par rapport au souvenir
le plus récent. Ça corrigeait le cas visé, mais la non-régression LoCoMo a montré la bonne
preuve reculer sur 104 questions sur 150. Annulé.

---

## 2. Oubli

**Le problème.** Le contrat AML n'a pas d'opération de suppression. « Please forget that I love
jazz » arrive comme un message ordinaire, et c'est à la mémoire de l'exécuter.

**Ce qui a changé** (`src/mnemos/router/forget.py`, `src/mnemos/stores/episodic.py`) :
1. **Détection** des demandes d'oubli, en anglais et en français. La précision passe d'abord :
   « don't forget » est un rappel, pas une demande d'oubli.
2. **Ciblage** des anciennes affirmations **de l'assistant** qui reprennent la préférence
   (cosinus ≥ 0,65 avec la cible, 5 au plus). Ce que l'utilisateur a dit n'est jamais effacé,
   et rien de ce qui suit la demande ne l'est non plus.
3. **Effacement physique**, dans la même transaction que l'écriture.
4. **La demande et l'accusé de réception sont gardés** : ce sont eux qui disent au répondeur quoi
   éviter.

**Effet** (questions d'oubli de PersonaMem, mêmes questions avant et après) :

| répondeur | rien | tout effacer (1re version) | **version retenue** |
|---|---|---|---|
| qwen2.5 7B, vrai code | 26,7 % | 18,7 % | **30,7 %** |
| Nemotron 3 Ultra (550B) | 45,8 % | 13,9 % | **53,5 %** (p = 0,01) |

**La leçon** : la première version effaçait aussi la demande et l'accusé. Un indicateur de
« fuite » s'effondrait (68 % → 16 %), mais les réponses empiraient : il comptait comme une
fuite la phrase même qui dit quoi oublier. On juge désormais toute modification à la réponse
finale, jamais à un indicateur intermédiaire.

---

## 3. Dates relatives résolues à l'écriture

**Le problème.** Les répondeurs ne convertissent pas « yesterday » en date, même avec
l'horodatage du souvenir sous les yeux. C'est pourtant ce que leur demande la consigne
officielle de la plateforme (règle 7 : *convert relative times like "yesterday", "last month",
and "last year" into dates*). Sur LoCoMo, 30 des 32 échecs aux questions de dates venaient de
là, alors que la bonne réplique était bien retrouvée.

**Ce qui a changé** (`src/mnemos/router/relative_dates.py`, route `/add`). Quand un message
porte l'horodatage de sa source, la date visée est ajoutée entre parenthèses après
l'expression :

> I took my kids to a park yesterday **(27 August 2023)**.

- **Jours, mois, années seulement** : yesterday, last night, two days ago, tomorrow, last /
  this / next month, last / this / next year.
- **Jamais les semaines** (« last Friday », « last week ») : le juge officiel exige une forme
  relative quand le corrigé en a une, et compte faux une date calculée.
- **Jamais les durées ni les formes vagues** (« ten years ago », « recently »).
- **Les mots d'origine restent intacts**, la date est seulement ajoutée.
- **Rien n'est ajouté au moment de la recherche** : `/search` renvoie le souvenir tel qu'il a été
  stocké.
- **L'index reste celui du message brut** (embedding et score lexical) : le classement mesuré
  et le seuil de l'oubli restent valables.
- **Sans horodatage de la source, rien n'est annoté** : l'ancre serait l'heure de réception, et
  la date serait fausse.

**Effet.**
- Précision du calcul de date, sans modèle de langage (`bench/bench_dates_precision.py`) :
  13/13 en développement, **75/77** en validation. Les deux écarts sont une erreur d'étiquette
  du jeu et une expression portant sur un autre événement.
- Réponses, avec les consignes officielles du répondeur et du juge, sur 9 conversations LoCoMo
  non vues pendant la conception (`bench/bench_dates_reponse.py`) :

| répondeur = juge | questions de dates | autres catégories |
|---|---|---|
| Nemotron 3 Ultra | 108 → **166** / 282 (9 perdues, 67 gagnées) | 41 → 41 / 60 |
| qwen2.5 7B | 139 → **178** / 280 (14 perdues, 53 gagnées) | 40 → 38 / 60 (bruit du juge) |
| qwen2.5 7B, **LoCoMo-Refined** (corrigés de la plateforme) | 126 → **166** / 252 (14 perdues, 54 gagnées) | 48 → 48 / 60 |

LoCoMo-Refined est la version que passe la plateforme : 337 questions revues, et des corrigés en
listes de réponses acceptables.

**Le désactiver** : `RELATIVE_DATES_ANNOTATION=false` dans `.env`.

---

## Essayé et abandonné

| Piste | Pourquoi c'est abandonné |
|---|---|
| **Consolidation** (faits extraits par un modèle de langage) | trop lente : 0,56 à 0,86 souvenir/s, alors qu'un Full en ingère ~4,9/s. Et nuisible : D1 dur 4/13 avec les faits contre 13/13 sans, même avec un répondeur de 550B, qui recopie le fait mal extrait |
| **Récence relative** | voir plus haut : 104 questions LoCoMo sur 150 reculaient |
| **Annoter les dates au moment de la recherche** | trop proche de la règle « Search must not disguise answers as memory records » ; remplacé par l'annotation à l'écriture |
| **Multi-sauts par reclassement** (vivier élargi, noms propres, voisinage, diversité) | au mieux +2 points, payés par des reculs ailleurs. Les preuves ratées sont indirectes (« I can't have dairy » pour une allergie) : il faudrait un modèle de langage dans le chemin critique |
| **Rendre la consigne d'oubli avec son accusé** (lien « consigne ↔ message suivant ») | quand `/search` omet « Please forget that I… », les réponses s'effondrent ; la remettre en répare 10 sur 18, et l'ajouter quand c'est la bonne fait 12 gagnées, 0 perdue. Mais sans étiquette, la règle ajoute aussi des consignes sans rapport à ~60 % des autres questions : sur 240 personas neufs, total −3 sur 664 questions. Une variante plus précise ferait ~+0,5 point, sous la résolution des mesures (le répondeur change 22 % de ses verdicts à contexte identique). Non livré pour l'instant |
| **Épingler les consignes durables** (« Always … when I ask about … ») | sur BEAM, `/search` ne rend toutes les preuves que pour 13 questions de consignes sur 40, et un détecteur strict les trouve toutes sans se déclencher ailleurs (LoCoMo 0, PersonaMem 0). Mais le répondeur les respecte déjà sans les voir : épinglées, la note baisse légèrement (−0,04), et même la seule bonne consigne n'apporte que +0,075 |
| **Exploiter CL-bench (catégorie G)** | `/search` y rend déjà 100 % de l'historique, quelle que soit la façon dont la plateforme découpe les tâches ; seul l'ordre resterait, et le contrat l'impose (par pertinence). G dépend du répondeur, pas de la mémoire |

---

## Comment ces décisions ont été prises

- **Mesurer la réponse finale**, avec les consignes officielles du répondeur et du juge, publiées
  dans le dépôt des organisateurs (`github.com/AML-memory/agent-memory-leaderboard`). `bench/aml_officiel.py`
  les lit depuis une copie locale.
- **Plusieurs répondeurs**, du 7B local au 550B distant : une conclusion qui ne tient qu'avec un
  petit modèle n'en est pas une.
- **Règle de décision écrite avant la mesure**, pour ne pas ajuster jusqu'à ce que ça ait l'air bon.
- **Développement et validation séparés** : la conception se fait sur une partie des données, la
  décision sur une autre, non vue.
- **Tests par mutation** : chaque test est vérifié en cassant volontairement le code qu'il doit
  protéger.
