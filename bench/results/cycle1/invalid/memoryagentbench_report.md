# 🧠 MemoryAgentBench (ICLR 2026) — Rapport d'Évaluation Mnemos

- **Agent évalué** : Mnemos (Architecture Biologique 4 Stores)
- **Modèle LLM** : `qwen3:4b` (dry_run: True)
- **Modèle d'embedding** : `bge-m3:latest`
- **Score Global MemoryAgentBench** : **100.0%** (8/8)

---

## 📊 Résultats par Compétence Cardinale

| Compétence | Description de la Tâche | Échantillons | Score Mnemos | Statut |
|---|---|---|---|---|
| **Accurate Retrieval (AR)** | Localisation exacte d'informations dans de longs contextes | 2 | **100.0%** | ✅ Validé |
| **Test-Time Learning (TTL)** | Assimilation de consignes et formats introduits dynamiquement | 2 | **100.0%** | ✅ Validé |
| **Long-Range Understanding (LRU)** | Intégration et cohérence sur des sessions étendues | 2 | **100.0%** | ✅ Validé |
| **Conflict Resolution (CR)** | Mise à jour et résolution de faits mutables contradictoires | 2 | **100.0%** | ✅ Validé |

---

## 🔍 Détail des Épreuves Individuelles

### AR_001 (Accurate_Retrieval) : ✅ SUCCÈS
- **Question** : *Quel est le port assigné pour le service d'authentification interne ?*
- **Vérité attendue** : `8443`
- **Prédiction Mnemos** : `Le port assigné est le 8443.`

### AR_002 (Accurate_Retrieval) : ✅ SUCCÈS
- **Question** : *Dans quelle région sont hébergés les serveurs de build CI/CD ?*
- **Vérité attendue** : `Francfort eu-central-1`
- **Prédiction Mnemos** : `Les serveurs sont hébergés à Francfort eu-central-1.`

### TTL_001 (Test_Time_Learning) : ✅ SUCCÈS
- **Question** : *Quel est l'identifiant pour le ticket n°402 et quel mot de clôture dois-tu inclure ?*
- **Vérité attendue** : `TIK-402 [STATUT_OK]`
- **Prédiction Mnemos** : `TIK-402 [STATUT_OK]`

### TTL_002 (Test_Time_Learning) : ✅ SUCCÈS
- **Question** : *Quel est le montant de la prestation en appliquant la règle de devise ?*
- **Vérité attendue** : `1000 EUR [USD/EUR]`
- **Prédiction Mnemos** : `1000 EUR [USD/EUR]`

### LRU_001 (Long_Range_Understanding) : ✅ SUCCÈS
- **Question** : *Quelle technologie et quel modèle spécifique David utilise-t-il pour son pipeline de traitement ?*
- **Vérité attendue** : `PyTorch et le module SegMed`
- **Prédiction Mnemos** : `David utilise PyTorch et le module SegMed.`

### LRU_002 (Long_Range_Understanding) : ✅ SUCCÈS
- **Question** : *Quelle est la référence du capteur acoustique et quelle précision a été mesurée aux essais ?*
- **Vérité attendue** : `SONAR-Rail-4 avec 98.4% de précision`
- **Prédiction Mnemos** : `Le capteur est le SONAR-Rail-4 avec une précision mesurée de 98.4%.`

### CR_001 (Conflict_Resolution) : ✅ SUCCÈS
- **Question** : *Où Alice travaille-t-elle actuellement et quel est son poste ?*
- **Vérité attendue** : `Directrice IA chez Wayne Enterprises`
- **Prédiction Mnemos** : `Alice travaille actuellement comme Directrice IA chez Wayne Enterprises.`

### CR_002 (Conflict_Resolution) : ✅ SUCCÈS
- **Question** : *Quelle est l'adresse IP active actuelle du serveur primaire ?*
- **Vérité attendue** : `192.168.1.99`
- **Prédiction Mnemos** : `L'adresse IP active actuelle du serveur primaire est 192.168.1.99.`
