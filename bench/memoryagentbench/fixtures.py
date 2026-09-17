"""Fixtures déterministes représentatives des 4 splits de MemoryAgentBench (ICLR 2026).

Splits :
1. Accurate_Retrieval (AR)
2. Test_Time_Learning (TTL)
3. Long_Range_Understanding (LRU)
4. Conflict_Resolution (CR)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MABSample:
    qa_id: str
    split: str
    context_chunks: list[str]
    question: str
    ground_truth: str
    conflict_type: str = "none"


MAB_BENCHMARK_SAMPLES: list[MABSample] = [
    # ── 1. ACCURATE RETRIEVAL (AR) ──────────────────────────────────────────
    MABSample(
        qa_id="AR_001",
        split="Accurate_Retrieval",
        context_chunks=[
            "Lors de la réunion du projet Helios le 12 janvier, l'équipe a validé l'architecture en microservices.",
            "Le responsable sécurité Marc a précisé que les clés de chiffrement de production devaient être renouvelées tous les 60 jours.",
            "La base de données principale a été migrée vers PostgreSQL 16 avec sharding horizontal.",
            "Le port assigné pour le service d'authentification interne est le port 8443.",
        ],
        question="Quel est le port assigné pour le service d'authentification interne ?",
        ground_truth="8443",
    ),
    MABSample(
        qa_id="AR_002",
        split="Accurate_Retrieval",
        context_chunks=[
            "Le budget alloué pour l'infrastructure cloud pour le Q3 s'élève à 45000 euros.",
            "Les serveurs de build CI/CD sont hébergés dans la région de Francfort eu-central-1.",
            "Le temps moyen de build des images Docker est de 4 minutes et 30 secondes.",
            "La politique de rétention des logs applicatifs est fixée à 180 jours.",
        ],
        question="Dans quelle région sont hébergés les serveurs de build CI/CD ?",
        ground_truth="Francfort eu-central-1",
    ),
    # ── 2. TEST-TIME LEARNING (TTL) ─────────────────────────────────────────
    MABSample(
        qa_id="TTL_001",
        split="Test_Time_Learning",
        context_chunks=[
            "Consigne importante : À partir de maintenant, formate chaque identifiant de ticket avec le préfixe TIK- et termine toujours ta réponse par [STATUT_OK].",
            "L'utilisateur a signalé une anomalie sur la page de connexion.",
            "Le correctif pour la page de connexion a été validé en staging.",
        ],
        question="Quel est l'identifiant pour le ticket n°402 et quel mot de clôture dois-tu inclure ?",
        ground_truth="TIK-402 [STATUT_OK]",
    ),
    MABSample(
        qa_id="TTL_002",
        split="Test_Time_Learning",
        context_chunks=[
            "Règle de formatage : Si une demande concerne un montant financier, affiche toujours la conversion en devise USD et EUR entre crochets.",
            "La commande client comprend une prestation de conseil de 1000 euros.",
        ],
        question="Quel est le montant de la prestation en appliquant la règle de devise ?",
        ground_truth="1000 EUR [USD/EUR]",
    ),
    # ── 3. LONG-RANGE UNDERSTANDING (LRU) ───────────────────────────────────
    MABSample(
        qa_id="LRU_001",
        split="Long_Range_Understanding",
        context_chunks=[
            "Session 1 : Le client David travaille sur un pipeline de traitement d'images médicales avec PyTorch.",
            "Session 2 : David a intégré un module de segmentation par transformeur nommé SegMed.",
            "Session 3 : Un goulot d'étranglement a été détecté lors du chargement des tenseurs sur GPU.",
            "Session 4 : David a résolu le problème en utilisant des tenseurs épinglés en mémoire partagée.",
        ],
        question="Quelle technologie et quel modèle spécifique David utilise-t-il pour son pipeline de traitement ?",
        ground_truth="PyTorch et le module SegMed",
    ),
    MABSample(
        qa_id="LRU_002",
        split="Long_Range_Understanding",
        context_chunks=[
            "Session 1 : Atelios développe une plateforme d'IA pour le secteur ferroviaire.",
            "Session 2 : Le capteur acoustique principal porte la référence SONAR-Rail-4.",
            "Session 3 : Les essais sur voie ont révélé une précision de détection d'anomalie de 98.4%.",
            "Session 4 : Le déploiement est planifié pour la flotte régionale Nord.",
        ],
        question="Quelle est la référence du capteur acoustique et quelle précision a été mesurée aux essais ?",
        ground_truth="SONAR-Rail-4 avec 98.4% de précision",
    ),
    # ── 4. CONFLICT RESOLUTION (CR) ─────────────────────────────────────────
    MABSample(
        qa_id="CR_001",
        split="Conflict_Resolution",
        context_chunks=[
            "Événement 1 : Alice travaille en tant que Développeuse Frontend chez Stark Corp.",
            "Discussion neutre sur l'organisation des équipes et les méthodologies agiles.",
            "Événement 2 : Alice a démissionné de Stark Corp et occupe désormais le poste de Directrice IA chez Wayne Enterprises.",
        ],
        question="Où Alice travaille-t-elle actuellement et quel est son poste ?",
        ground_truth="Directrice IA chez Wayne Enterprises",
        conflict_type="attribute_mutation",
    ),
    MABSample(
        qa_id="CR_002",
        split="Conflict_Resolution",
        context_chunks=[
            "Note 1 : L'adresse IP du serveur de base de données primaire est 192.168.1.50.",
            "Discussion technique sur les sauvegardes périodiques.",
            "Note 2 : Suite à une panne matérielle, le serveur primaire a été basculé définitivement sur l'adresse IP 192.168.1.99.",
        ],
        question="Quelle est l'adresse IP active actuelle du serveur primaire ?",
        ground_truth="192.168.1.99",
        conflict_type="ip_supersession",
    ),
]
