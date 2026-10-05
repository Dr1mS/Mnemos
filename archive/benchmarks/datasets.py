"""Jeux de données et générateurs synthétiques pour les benchmarks de Mnemos."""

from __future__ import annotations

import random

# Sujets et thèmes pour la génération synthétique à grande échelle
TOPICS = [
    (
        "tech",
        [
            "Je viens de migrer la base de données PostgreSQL vers un cluster distribué.",
            "On a corrigé une race condition dans le worker de tâches asynchrones.",
            "L'index vectoriel HNSW consomme beaucoup trop de RAM sur l'instance c6i.",
            "J'utilise Python 3.12 avec Pydantic v2 pour valider les payloads entrants.",
            "La latence réseau p99 est passée de 45ms à 12ms après le passage en HTTP/2.",
            "On utilise Ruff pour le linting et mypy en mode strict sur tout le repo.",
            "Le pipeline CI/CD sur GitHub Actions met 4 minutes pour les tests unitaires.",
            "J'ai configuré un reverse proxy Caddy avec renouvellement automatique des certificats Let's Encrypt.",
        ],
    ),
    (
        "personal",
        [
            "J'ai commencé la course à pied, j'ai fait 8km autour du lac ce matin.",
            "Je prépare un risotto aux champignons pour le dîner ce soir.",
            "Je dois penser à renouveler mon passeport avant le voyage en mai.",
            "Mon chat s'appelle Miso, c'est un chartreux de trois ans très joueur.",
            "J'habite à Annecy depuis deux ans près du canal du Thiou.",
            "Je suis allergique aux arachides et aux noix de cajou, c'est très strict.",
            "Mon café préféré est un éthiopien lavé préparé en V60 le matin.",
            "J'ai fini de lire le livre sur les architectures cognitives hier soir.",
        ],
    ),
    (
        "work",
        [
            "La réunion de synchronisation produit aura lieu demain à 14h.",
            "Le client demande une estimation du coût infrastructure pour 50 000 utilisateurs.",
            "Le rapport d'audit de sécurité a identifié deux vulnérabilités moyennes.",
            "On va recruter un ingénieur machine learning senior pour le trimestre prochain.",
            "Le contrat de maintenance annuel a été validé par la direction financière.",
        ],
    ),
    (
        "noise",
        [
            "Bonjour, comment ça va aujourd'hui ?",
            "Tu peux me rappeler la commande curl pour afficher les headers HTTP ?",
            "Il fait quel temps à Paris demain après-midi ?",
            "Merci beaucoup pour ton aide sur ce problème !",
            "ls -la /var/log/syslog | grep error",
            "OK je note, on verra ça plus tard.",
            "Peux-tu reformuler cette phrase pour un e-mail professionnel ?",
            "Je fais une pause de dix minutes et je reviens.",
        ],
    ),
]


def generate_synthetic_episodes(
    count: int, tenant: str = "user", seed: int = 42
) -> list[dict[str, str]]:
    """Génère un flux d'épisodes variés et réalistes."""
    rng = random.Random(seed)
    episodes: list[dict[str, str]] = []

    all_templates = []
    for topic_name, phrases in TOPICS:
        for p in phrases:
            all_templates.append((topic_name, p))

    roles = ["user", "user", "assistant"]  # 2/3 user, 1/3 assistant

    for i in range(count):
        topic, phrase = rng.choice(all_templates)
        # Légère variation pour éviter les doublons stricts dans les tests volumétriques
        suffix = f" (ref #{i})" if (i >= len(all_templates)) else ""
        content = f"{phrase}{suffix}"
        role = rng.choice(roles)
        episodes.append({
            "content": content,
            "role": role,
            "tenant": tenant,
            "session_id": f"session_{i // 20}",
            "topic": topic,
        })

    return episodes


# Scénario du Chaos Temporel : 6 étapes de vie avec révisions et rétractions
TEMPORAL_CHAOS_EVENTS = [
    {
        "step": 1,
        "content": "J'habite à Paris dans le 11e arrondissement depuis 3 ans.",
        "expected_fact": ("user", "lives_in", "Paris"),
    },
    {
        "step": 2,
        "content": "Mon chat s'appelle Miso et il a 2 ans.",
        "expected_fact": ("user", "owns_pet", "Miso"),
    },
    {
        "step": 3,
        "content": "Je pars vivre à Tokyo pour un contrat d'expatriation de 6 mois.",
        "expected_fact": ("user", "lives_in", "Tokyo"),
    },
    {
        "step": 4,
        "content": "Je suis rentré en France, j'ai trouvé un appartement à Lyon dans le quartier de la Croix-Rousse.",
        "expected_fact": ("user", "lives_in", "Lyon"),
    },
    {
        "step": 5,
        "content": "Mon contrat à Lyon a été annulé avant de commencer, en fait je me suis installé définitivement à Annecy.",
        "expected_fact": ("user", "lives_in", "Annecy"),
    },
]


# Scénario de Désambiguïsation d'homonymes
HOMONYM_EVENTS = [
    {
        "content": "Mon collègue Alex travaille chez Google sur les modèles de langage depuis 2 ans.",
        "entity": "Alex (collègue)",
        "expected_fact": ("Alex", "works_at", "Google"),
    },
    {
        "content": "Mon cousin Alex habite à Bordeaux et il est menuisier ébéniste.",
        "entity": "Alex (cousin)",
        "expected_fact": ("Alex", "lives_in", "Bordeaux"),
    },
    {
        "content": "Alex le collègue de chez Google vient d'être promu Staff Engineer.",
        "entity": "Alex (collègue)",
        "expected_fact": ("Alex", "role", "Staff Engineer"),
    },
]
