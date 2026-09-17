# Rapport d'Évaluation — Jeu de tests inspiré de la taxonomie MemoryAgentBench (ICLR 2026)

> ⚠️ **DRY RUN — RÉSULTATS NON SIGNIFICATIFS (MODE STUB SANS MODÈLE NEURONAL)**

- **Système évalué** : `EmptyControlMemory`
- **Modèle LLM** : `qwen2.5:3b` (dry_run: True)
- **Modèle d'embedding** : `bge-m3:latest`
- **Score Global** : **0.0%** (0/8)

> *Avertissement méthodologique : Ce jeu d'épreuve est un banc synthétique ciblé inspiré des 4 compétences de MemoryAgentBench, et non l'exécution du corpus officiel intégral multi-mille instances.*

---

## Résultats par Compétence

| Compétence | Échantillons | Score |
|---|---|---|
| **Accurate Retrieval (AR)** | 2 | 0.0% |
| **Test-Time Learning (TTL)** | 2 | 0.0% |
| **Long-Range Understanding (LRU)** | 2 | 0.0% |
| **Conflict Resolution (CR)** | 2 | 0.0% |

---

## Détail des Épreuves Individuelles

### AR_001 (Accurate_Retrieval) : Échec
- **Question** : *Quel est le port assigné pour le service d'authentification interne ?*
- **Vérité attendue** : `8443`
- **Réponse produite** : `[DRY_RUN]`

### AR_002 (Accurate_Retrieval) : Échec
- **Question** : *Dans quelle région sont hébergés les serveurs de build CI/CD ?*
- **Vérité attendue** : `Francfort eu-central-1`
- **Réponse produite** : `[DRY_RUN]`

### TTL_001 (Test_Time_Learning) : Échec
- **Question** : *Quel est l'identifiant pour le ticket n°402 et quel mot de clôture dois-tu inclure ?*
- **Vérité attendue** : `TIK-402 [STATUT_OK]`
- **Réponse produite** : `[DRY_RUN]`

### TTL_002 (Test_Time_Learning) : Échec
- **Question** : *Quel est le montant de la prestation en appliquant la règle de devise ?*
- **Vérité attendue** : `1000 EUR [USD/EUR]`
- **Réponse produite** : `[DRY_RUN]`

### LRU_001 (Long_Range_Understanding) : Échec
- **Question** : *Quelle technologie et quel modèle spécifique David utilise-t-il pour son pipeline de traitement ?*
- **Vérité attendue** : `PyTorch et le module SegMed`
- **Réponse produite** : `[DRY_RUN]`

### LRU_002 (Long_Range_Understanding) : Échec
- **Question** : *Quelle est la référence du capteur acoustique et quelle précision a été mesurée aux essais ?*
- **Vérité attendue** : `SONAR-Rail-4 avec 98.4% de précision`
- **Réponse produite** : `[DRY_RUN]`

### CR_001 (Conflict_Resolution) : Échec
- **Question** : *Où Alice travaille-t-elle actuellement et quel est son poste ?*
- **Vérité attendue** : `Directrice IA chez Wayne Enterprises`
- **Réponse produite** : `[DRY_RUN]`

### CR_002 (Conflict_Resolution) : Échec
- **Question** : *Quelle est l'adresse IP active actuelle du serveur primaire ?*
- **Vérité attendue** : `192.168.1.99`
- **Réponse produite** : `[DRY_RUN]`
