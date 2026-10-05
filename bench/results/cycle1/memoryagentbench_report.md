# Rapport d'Évaluation — Jeu de tests inspiré de la taxonomie MemoryAgentBench (ICLR 2026)

- **Système évalué** : `Mnemos`
- **Modèle LLM** : `qwen2.5:3b` (dry_run: False)
- **Modèle d'embedding** : `bge-m3:latest`
- **Score Global** : **75.0%** (6/8)

> *Avertissement méthodologique : Ce jeu d'épreuve est un banc synthétique ciblé inspiré des 4 compétences de MemoryAgentBench, et non l'exécution du corpus officiel intégral multi-mille instances.*

---

## Résultats par Compétence

| Compétence | Échantillons | Score |
|---|---|---|
| **Accurate Retrieval (AR)** | 2 | 100.0% |
| **Test-Time Learning (TTL)** | 2 | 100.0% |
| **Long-Range Understanding (LRU)** | 2 | 0.0% |
| **Conflict Resolution (CR)** | 2 | 100.0% |

---

## Détail des Épreuves Individuelles

### AR_001 (Accurate_Retrieval) : Succès
- **Question** : *Quel est le port assigné pour le service d'authentification interne ?*
- **Vérité attendue** : `8443`
- **Réponse produite** : `Le port assigné pour le service d'authentification interne est le port 8443.`

### AR_002 (Accurate_Retrieval) : Succès
- **Question** : *Dans quelle région sont hébergés les serveurs de build CI/CD ?*
- **Vérité attendue** : `Francfort eu-central-1`
- **Réponse produite** : `Les serveurs de build CI/CD sont hébergés dans la région Francfort eu-central-1.`

### TTL_001 (Test_Time_Learning) : Succès
- **Question** : *Quel est l'identifiant pour le ticket n°402 et quel mot de clôture dois-tu inclure ?*
- **Vérité attendue** : `TIK-402 [STATUT_OK]`
- **Réponse produite** : `TIK-402 [STATUT_OK]`

### TTL_002 (Test_Time_Learning) : Succès
- **Question** : *Quel est le montant de la prestation en appliquant la règle de devise ?*
- **Vérité attendue** : `1000 EUR [USD/EUR]`
- **Réponse produite** : `[1000 USD, 808 EUR]`

### LRU_001 (Long_Range_Understanding) : Échec
- **Question** : *Quelle technologie et quel modèle spécifique David utilise-t-il pour son pipeline de traitement ?*
- **Vérité attendue** : `PyTorch et le module SegMed`
- **Réponse produite** : `David utilise PyTorch pour son pipeline de traitement d'images médicales et a intégré un modèle de segmentation par transformeur nommé SegMed. Il a également utilisé des tenseurs épinglés en mémoire partagée pour résoudre un problème et a rencontré un goulot d'étranglement lors du chargement des tenseurs sur GPU.`

### LRU_002 (Long_Range_Understanding) : Échec
- **Question** : *Quelle est la référence du capteur acoustique et quelle précision a été mesurée aux essais ?*
- **Vérité attendue** : `SONAR-Rail-4 avec 98.4% de précision`
- **Réponse produite** : `La référence du capteur acoustique est SONAR-Rail-4. La précision de détection d'anomalie aux essais sur voie est de 98.4%.`

### CR_001 (Conflict_Resolution) : Succès
- **Question** : *Où Alice travaille-t-elle actuellement et quel est son poste ?*
- **Vérité attendue** : `Directrice IA chez Wayne Enterprises`
- **Réponse produite** : `Alice travaille actuellement chez Wayne Enterprises et occupe le poste de Directrice IA.`

### CR_002 (Conflict_Resolution) : Succès
- **Question** : *Quelle est l'adresse IP active actuelle du serveur primaire ?*
- **Vérité attendue** : `192.168.1.99`
- **Réponse produite** : `L'adresse IP active actuelle du serveur primaire est 192.168.1.99.`
