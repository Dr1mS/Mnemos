"""Jeux de données déterministes et sondes paramétriques pour les 3 épreuves.

Partitionnement :
- Jeu Dev (70%, 14 instances par épreuve) : défini ici pour l'audit et l'évaluation.
- Jeu Hold-out (30%, 6 instances par épreuve) : consigné dans datasets_holdout.py (non exécuté pendant la phase d'audit/calibration).

Scénarios :
1. Mutation Temporelle & Effet Fantôme (Ghost Vector) : 14 triplets de villes avec tours neutres intercalés.
2. Résistance au Bruit et Décroissance (Bland Noise vs High Salience) : 14 flux de 100 tours avec faits critiques variés (clé de staging, allergie/santé).
3. Conventions de Projet Arbitraires (Projet Compliance) : 14 requêtes de code sollicitant les 5 conventions de projet arbitraires.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InteractionTurn:
    role: str
    content: str
    timestamp_offset_days: float = 0.0
    salience_hint: float = 0.5
    is_critical: bool = False


# ── ÉPREUVE 1 : MUTATION TEMPORELLE & EFFET FANTÔME ─────────────────────────

@dataclass(frozen=True)
class GhostVectorInstance:
    id: str
    city_0: str  # Première résidence
    city_1: str  # Deuxième résidence
    city_2: str  # Résidence active actuelle
    neutral_turns_1: list[tuple[str, str]]
    neutral_turns_2: list[tuple[str, str]]
    probe_1_1: str = "Où est-ce que j'habite actuellement ?"
    probe_1_2: str = "Quelles ont été mes villes de résidence dans l'ordre chronologique ?"


# Banque de questions-réponses neutres (techniques, programmation)
_BANK_NEUTRAL_1: list[tuple[str, str]] = [
    ("Comment définir une fonction asynchrone en Python ?", "On utilise le mot-clé async def pour déclarer une coroutine."),
    ("Quelle est la différence entre liste et tuple ?", "Les listes sont mutables tandis que les tuples sont immuables."),
    ("À quoi sert le décorateur @property ?", "Il permet de définir des getters et setters avec une syntaxe d'attribut."),
    ("Comment gérer les exceptions personnalisées ?", "Il suffit d'hériter de la classe Exception de base."),
    ("Quelle est l'utilité de __slots__ ?", "Il optimise l'empreinte mémoire en restreignant les attributs autorisés."),
    ("Comment inverser une chaîne en Python ?", "La syntaxe de découpage [::-1] permet une inversion instantanée."),
    ("Quelle est la complexité d'une recherche dans un dict ?", "En moyenne O(1) grâce au hachage interne."),
    ("Pourquoi utiliser dataclass ?", "Il automatise la génération de __init__, __repr__ et __eq__."),
    ("Quelle est la différence entre is et == ?", "is vérifie l'identité mémoire alors que == vérifie l'égalité des valeurs."),
    ("Comment lire un fichier avec un context manager ?", "Avec with open('file.txt', 'r') as f: qui garantit la fermeture."),
]

_BANK_NEUTRAL_2: list[tuple[str, str]] = [
    ("Comment exécuter des tâches concurrentes en asyncio ?", "On utilise asyncio.gather() ou TaskGroup pour exécuter plusieurs coroutines."),
    ("Comment typer un retour optionnel ?", "On utilise Optional[T] ou la syntaxe moderne T | None."),
    ("Quelle est la commande pour créer un venv ?", "python -m venv .venv crée un environnement virtuel isolé."),
    ("À quoi sert functools.lru_cache ?", "Il mémorise les résultats de fonctions coûteuses en évitant les recalculs."),
    ("Comment fonctionne zip() ?", "Il agrège les éléments de plusieurs itérables en tuples appariés."),
    ("Pourquoi préférer Pathlib à os.path ?", "Pathlib propose une API orientée objet plus expressive et portable."),
    ("Comment sérialiser un objet Pydantic en JSON ?", "Via la méthode model_dump_json() en Pydantic V2."),
    ("Quelle est la différence entre asyncio.sleep et time.sleep ?", "asyncio.sleep est non bloquant pour la boucle d'événements."),
    ("Comment vérifier le typage avec mypy ?", "En exécutant mypy src/ dans le terminal."),
    ("Comment déclarer une variable d'environnement sous Linux ?", "Avec export NOM=valeur dans le shell."),
]

# 14 instances Dev
_DEV_CITY_TRIPLETS: list[tuple[str, str, str]] = [
    ("Lyon", "Paris", "Annecy"),
    ("Marseille", "Bordeaux", "Nantes"),
    ("Lille", "Strasbourg", "Toulouse"),
    ("Rennes", "Montpellier", "Nice"),
    ("Grenoble", "Rouen", "Dijon"),
    ("Brest", "Tours", "Biarritz"),
    ("Reims", "Le Havre", "Saint-Étienne"),
    ("Toulon", "Angers", "Nîmes"),
    ("Villeurbanne", "Le Mans", "Aix-en-Provence"),
    ("Amiens", "Limoges", "Metz"),
    ("Besançon", "Perpignan", "Orléans"),
    ("Caen", "Mulhouse", "Nancy"),
    ("Roubaix", "Tourcoing", "Avignon"),
    ("Dunkerque", "Poitiers", "Pau"),
]

DEV_GHOST_INSTANCES: list[GhostVectorInstance] = [
    GhostVectorInstance(
        id=f"ghost_dev_{i+1:02d}",
        city_0=c0,
        city_1=c1,
        city_2=c2,
        neutral_turns_1=_BANK_NEUTRAL_1,
        neutral_turns_2=_BANK_NEUTRAL_2,
    )
    for i, (c0, c1, c2) in enumerate(_DEV_CITY_TRIPLETS)
]


# ── ÉPREUVE 1B : MUTATION TEMPORELLE HORS ONTOLOGIE ─────────────────────────

@dataclass(frozen=True)
class ExternalOntologyInstance:
    id: str
    topic: str
    val_0: str  # Valeur initiale
    val_1: str  # Deuxième valeur
    val_2: str  # Valeur active actuelle
    statement_0: str
    statement_1: str
    statement_2: str
    probe_1_1: str  # Sonde vérité active
    probe_1_2: str  # Sonde auditabilité chronologique
    neutral_turns_1: list[tuple[str, str]]
    neutral_turns_2: list[tuple[str, str]]


_DEV_EXTERNAL_TRIPLETS: list[dict[str, str]] = [
    {
        "topic": "serveur_prod",
        "val_0": "srv-prod-01.internal",
        "val_1": "prod-db-node2.corp",
        "val_2": "cloud-srv-09.infra",
        "statement_0": "Notre serveur de production principal est srv-prod-01.internal.",
        "statement_1": "Suite à la migration, notre serveur de production est maintenant prod-db-node2.corp.",
        "statement_2": "Finalement, notre serveur de production actuel est cloud-srv-09.infra.",
        "probe_1_1": "Quel est notre serveur de production actuel ?",
        "probe_1_2": "Quels ont été nos serveurs de production successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "manager",
        "val_0": "Alexandre Moreau",
        "val_1": "Nathalie Bernard",
        "val_2": "Julien Lambert",
        "statement_0": "Mon manager d'équipe est Alexandre Moreau.",
        "statement_1": "Après réorganisation, mon nouveau manager est Nathalie Bernard.",
        "statement_2": "Actuellement, mon manager en poste est Julien Lambert.",
        "probe_1_1": "Qui est mon manager actuel ?",
        "probe_1_2": "Quels ont été mes managers successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "outil_ci",
        "val_0": "Jenkins",
        "val_1": "GitLab CI",
        "val_2": "GitHub Actions",
        "statement_0": "Pour l'intégration continue, nous utilisons Jenkins.",
        "statement_1": "Nous avons migré nos pipelines, nous utilisons désormais GitLab CI.",
        "statement_2": "Notre chaîne de CI active est dorénavant GitHub Actions.",
        "probe_1_1": "Quel outil de CI utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos outils de CI successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "cluster_k8s",
        "val_0": "k8s-cluster-alpha",
        "val_1": "k8s-cluster-bravo",
        "val_2": "k8s-cluster-prod",
        "statement_0": "Le cluster Kubernetes principal de déploiement est k8s-cluster-alpha.",
        "statement_1": "Nous avons basculé la charge, notre cluster Kubernetes est maintenant k8s-cluster-bravo.",
        "statement_2": "Actuellement, notre cluster Kubernetes en production est k8s-cluster-prod.",
        "probe_1_1": "Quel est notre cluster Kubernetes de production actuel ?",
        "probe_1_2": "Quels ont été nos clusters Kubernetes successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "moteur_sgbd",
        "val_0": "PostgreSQL",
        "val_1": "CockroachDB",
        "val_2": "TiDB",
        "statement_0": "Notre moteur de base de données principal est PostgreSQL.",
        "statement_1": "Pour le sharding, nous sommes passés à CockroachDB.",
        "statement_2": "À ce jour, notre base de données principale est TiDB.",
        "probe_1_1": "Quelle est notre base de données principale actuelle ?",
        "probe_1_2": "Quelles ont été nos bases de données principales successives dans l'ordre chronologique ?",
    },
    {
        "topic": "monitoring",
        "val_0": "Nagios",
        "val_1": "Datadog",
        "val_2": "Prometheus",
        "statement_0": "La surveillance de notre infrastructure est assurée par Nagios.",
        "statement_1": "Nous avons remplacé notre supervision par Datadog.",
        "statement_2": "Désormais, notre outil de monitoring actif est Prometheus.",
        "probe_1_1": "Quel outil de monitoring utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos outils de monitoring successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "cloud_provider",
        "val_0": "AWS",
        "val_1": "Google Cloud",
        "val_2": "Microsoft Azure",
        "statement_0": "Notre infrastructure cloud est hébergée sur AWS.",
        "statement_1": "Nous avons transféré nos charges vers Google Cloud.",
        "statement_2": "Actuellement, notre hébergeur cloud principal est Microsoft Azure.",
        "probe_1_1": "Quel est notre fournisseur cloud principal actuel ?",
        "probe_1_2": "Quels ont été nos fournisseurs cloud successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "ticketing",
        "val_0": "Jira",
        "val_1": "Linear",
        "val_2": "GitHub Issues",
        "statement_0": "Pour le suivi des tickets de l'équipe, nous utilisons Jira.",
        "statement_1": "Nous avons adopté Linear pour la gestion de nos tickets.",
        "statement_2": "Maintenant, nous gérons tous nos tickets sur GitHub Issues.",
        "probe_1_1": "Quel outil de ticketing utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos outils de ticketing successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "passerelle_vpn",
        "val_0": "OpenVPN",
        "val_1": "WireGuard",
        "val_2": "Tailscale",
        "statement_0": "Pour l'accès distant sécurisé, nous passons par OpenVPN.",
        "statement_1": "Le protocole VPN a été mis à niveau vers WireGuard.",
        "statement_2": "Notre solution de VPN active est aujourd'hui Tailscale.",
        "probe_1_1": "Quelle solution VPN utilisons-nous actuellement ?",
        "probe_1_2": "Quelles ont été nos solutions VPN successives dans l'ordre chronologique ?",
    },
    {
        "topic": "container_runtime",
        "val_0": "Docker",
        "val_1": "Podman",
        "val_2": "containerd",
        "statement_0": "L'environnement d'exécution des conteneurs est Docker.",
        "statement_1": "Nous avons migré notre runtime vers Podman.",
        "statement_2": "En production, notre moteur de conteneurs actuel est containerd.",
        "probe_1_1": "Quel moteur de conteneurs utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos moteurs de conteneurs successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "message_broker",
        "val_0": "RabbitMQ",
        "val_1": "Apache Kafka",
        "val_2": "NATS",
        "statement_0": "Notre bus de messages distribué repose sur RabbitMQ.",
        "statement_1": "Pour le streaming événementiel, nous sommes passés à Apache Kafka.",
        "statement_2": "À présent, notre bus de messagerie actif est NATS.",
        "probe_1_1": "Quel bus de messagerie utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos bus de messagerie successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "reverse_proxy",
        "val_0": "Nginx",
        "val_1": "Traefik",
        "val_2": "Envoy",
        "statement_0": "Le proxy inverse frontal est Nginx.",
        "statement_1": "Nous avons basculé notre passerelle ingress sur Traefik.",
        "statement_2": "Notre proxy inverse actuel en production est Envoy.",
        "probe_1_1": "Quel proxy inverse frontal utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos proxys inverses successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "sso_auth",
        "val_0": "Keycloak",
        "val_1": "Auth0",
        "val_2": "Okta",
        "statement_0": "L'authentification centralisée SSO est gérée par Keycloak.",
        "statement_1": "Nous avons délégué l'identité utilisateur à Auth0.",
        "statement_2": "Actuellement, notre solution SSO active est Okta.",
        "probe_1_1": "Quelle solution SSO d'authentification utilisons-nous actuellement ?",
        "probe_1_2": "Quelles ont été nos solutions SSO successives dans l'ordre chronologique ?",
    },
    {
        "topic": "secret_manager",
        "val_0": "HashiCorp Vault",
        "val_1": "AWS Secrets Manager",
        "val_2": "Infisical",
        "statement_0": "Nos secrets d'application sont stockés dans HashiCorp Vault.",
        "statement_1": "Nous avons migré nos coffres-forts vers AWS Secrets Manager.",
        "statement_2": "Notre gestionnaire de secrets actif est désormais Infisical.",
        "probe_1_1": "Quel gestionnaire de secrets utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos gestionnaires de secrets successifs dans l'ordre chronologique ?",
    },
]

DEV_EXTERNAL_INSTANCES: list[ExternalOntologyInstance] = [
    ExternalOntologyInstance(
        id=f"ext_dev_{i+1:02d}",
        topic=item["topic"],
        val_0=item["val_0"],
        val_1=item["val_1"],
        val_2=item["val_2"],
        statement_0=item["statement_0"],
        statement_1=item["statement_1"],
        statement_2=item["statement_2"],
        probe_1_1=item["probe_1_1"],
        probe_1_2=item["probe_1_2"],
        neutral_turns_1=_BANK_NEUTRAL_1,
        neutral_turns_2=_BANK_NEUTRAL_2,
    )
    for i, item in enumerate(_DEV_EXTERNAL_TRIPLETS)
]


# ── ÉPREUVE 2 : BRUIT BLAND & FORTE SAILLANCE (90 JOURS) ────────────────────

@dataclass(frozen=True)
class BlandNoiseInstance:
    id: str
    key_code: str
    allergy: str
    noise_seed_offset: int
    probe_2_1: str = "Quelle est la clé d'authentification du serveur de staging ?"
    probe_2_2: str = "Quelle est ma contrainte médicale ou allergie vitale ?"


BLAND_NOISE_OBSERVATIONS: list[str] = [
    "Il fait un beau soleil aujourd'hui dehors.",
    "J'ai pris une grande tasse de café ce matin.",
    "Mon écran secondaire semble un peu sombre.",
    "J'ai nettoyé mon bureau avant de commencer la session.",
    "La météo annonce de la pluie pour la fin de semaine.",
    "Le ventilateur de l'ordinateur tourne calmement.",
    "J'ai bu un verre d'eau fraîche.",
    "La souris sans fil a encore 80% de batterie.",
    "Le clavier mécanique a un son agréable au toucher.",
    "Une petite brise passe à travers la fenêtre ouverte.",
    "J'ai déjeuné une salade composée à midi.",
    "L'horloge du système indique qu'il est l'heure de la pause.",
    "J'ai ajusté la hauteur de mon fauteuil de bureau.",
    "Le thé vert est un peu trop chaud pour l'instant.",
    "J'ai classé deux onglets inutiles dans mon navigateur.",
    "Il y a quelques oiseaux qui chantent sur le rebord de la fenêtre.",
    "Le câble USB-C est bien branché sur le hub.",
    "J'ai remis un peu d'ordre dans mes notes de la veille.",
    "La luminosité ambiante est très douce aujourd'hui.",
    "J'ai étiré mes bras après une heure de travail.",
]

_DEV_CRITICAL_FACTS: list[tuple[str, str]] = [
    ("SEC-9482", "arachides"),
    ("PRD-3810", "pénicilline"),
    ("STG-7741", "latex"),
    ("K8S-5529", "venin de guêpe"),
    ("DAT-1193", "iode"),
    ("CFG-8820", "aspirine"),
    ("SRV-4471", "gluten"),
    ("ACC-6632", "soja"),
    ("NET-2284", "fruits à coque"),
    ("SYS-9915", "sulfites"),
    ("KEY-3307", "insuline requise"),
    ("DEV-8841", "bouleau"),
    ("OPS-5512", "crustacés"),
    ("API-7723", "œufs"),
]

DEV_BLAND_INSTANCES: list[BlandNoiseInstance] = [
    BlandNoiseInstance(
        id=f"bland_dev_{i+1:02d}",
        key_code=kc,
        allergy=alg,
        noise_seed_offset=i * 5,
    )
    for i, (kc, alg) in enumerate(_DEV_CRITICAL_FACTS)
]


def build_bland_noise_stream(instance: BlandNoiseInstance, total_turns: int = 100) -> list[InteractionTurn]:
    """Construit un flux de 100 tours futiles avec les 2 faits capitaux intercalés aux positions 24 et 74."""
    turns: list[InteractionTurn] = []
    for i in range(total_turns):
        if i == 24:
            turns.append(
                InteractionTurn(
                    role="user",
                    content=f"Note bien ceci : La clé d'authentification du serveur de staging est {instance.key_code}.",
                    timestamp_offset_days=i * 0.1,
                    salience_hint=0.95,
                    is_critical=True,
                )
            )
            continue
        if i == 74:
            turns.append(
                InteractionTurn(
                    role="user",
                    content=f"Information vitale : Je suis sévèrement allergique aux {instance.allergy}, c'est mortel pour moi.",
                    timestamp_offset_days=i * 0.1,
                    salience_hint=0.98,
                    is_critical=True,
                )
            )
            continue

        phrase_idx = (i + instance.noise_seed_offset) % len(BLAND_NOISE_OBSERVATIONS)
        phrase = BLAND_NOISE_OBSERVATIONS[phrase_idx]
        turns.append(
            InteractionTurn(
                role="user",
                content=f"{phrase} (observation n°{i+1})",
                timestamp_offset_days=i * 0.1,
                salience_hint=0.15,
                is_critical=False,
            )
        )
    return turns


# ── ÉPREUVE 3 : CONVENTIONS DE PROJET ARBITRAIRES ───────────────────────────

ALL_RULE_MARKERS: list[str] = [
    "[RULE-HTTP]",
    "[RULE-LOG]",
    "[RULE-DATE]",
    "[RULE-AUTH]",
    "[RULE-STORAGE]",
]

CONVENTION_BASELINE_RULES: list[dict[str, str]] = [
    {
        "marker": "[RULE-HTTP]",
        "category": "http",
        "rule": "[RULE-HTTP] Convention de projet obligatoire : Toujours utiliser httpx, interdiction formelle de requests ou urllib.",
    },
    {
        "marker": "[RULE-LOG]",
        "category": "logging",
        "rule": "[RULE-LOG] Convention de projet obligatoire : Préfixer obligatoirement chaque ligne de log par le tag PRJ- (ex: PRJ-INFO:).",
    },
    {
        "marker": "[RULE-DATE]",
        "category": "date",
        "rule": "[RULE-DATE] Convention de projet obligatoire : Formater obligatoirement les dates au format arbitraire AAAA/JJ/MM (format strftime %Y/%d/%m). Interdiction du format ISO AAAA-MM-JJ.",
    },
    {
        "marker": "[RULE-AUTH]",
        "category": "auth",
        "rule": "[RULE-AUTH] Convention de projet obligatoire : Utiliser obligatoirement l'en-tête X-Project-Token pour l'authentification. Interdiction de Authorization: Bearer.",
    },
    {
        "marker": "[RULE-STORAGE]",
        "category": "storage",
        "rule": "[RULE-STORAGE] Convention de projet obligatoire : Écrire les caches locaux temporaires exclusivement dans /var/project/cache/. Interdiction de /tmp.",
    },
]


@dataclass(frozen=True)
class ProjectConventionPrompt:
    id: str
    marker: str
    prompt: str
    required_tokens: list[str]
    forbidden_tokens: list[str]
    description: str


DEV_CONVENTION_PROMPTS: list[ProjectConventionPrompt] = [
    # RULE-HTTP (3 instances)
    ProjectConventionPrompt(
        id="conv_http_01",
        marker="[RULE-HTTP]",
        prompt="Écris une fonction Python pour appeler une API REST GET sur une URL donnée et retourner le JSON. Code seul.",
        required_tokens=["httpx"],
        forbidden_tokens=["requests", "urllib"],
        description="Requête GET basique",
    ),
    ProjectConventionPrompt(
        id="conv_http_02",
        marker="[RULE-HTTP]",
        prompt="Écris une fonction Python asynchrone pour télécharger le contenu binaire d'une ressource web. Code seul.",
        required_tokens=["httpx"],
        forbidden_tokens=["requests", "urllib"],
        description="Téléchargement asynchrone",
    ),
    ProjectConventionPrompt(
        id="conv_http_03",
        marker="[RULE-HTTP]",
        prompt="Écris une fonction Python pour envoyer un payload JSON par POST vers un webhook externe. Code seul.",
        required_tokens=["httpx"],
        forbidden_tokens=["requests", "urllib"],
        description="Post webhook",
    ),
    # RULE-LOG (3 instances)
    ProjectConventionPrompt(
        id="conv_log_01",
        marker="[RULE-LOG]",
        prompt="Écris une fonction Python log_event(level, message) qui émet un log avec le niveau et message donnés. Code seul.",
        required_tokens=["PRJ-"],
        forbidden_tokens=[],
        description="Logging général",
    ),
    ProjectConventionPrompt(
        id="conv_log_02",
        marker="[RULE-LOG]",
        prompt="Écris une fonction Python log_error(err) qui enregistre une erreur applicative dans les journaux selon la convention de log du projet. Code seul.",
        required_tokens=["PRJ-"],
        forbidden_tokens=[],
        description="Logging erreur",
    ),
    ProjectConventionPrompt(
        id="conv_log_03",
        marker="[RULE-LOG]",
        prompt="Écris une fonction Python setup_logger() qui configure le formatteur standard de log selon la convention du projet. Code seul.",
        required_tokens=["PRJ-"],
        forbidden_tokens=[],
        description="Formatteur logger",
    ),
    # RULE-DATE (3 instances)
    ProjectConventionPrompt(
        id="conv_date_01",
        marker="[RULE-DATE]",
        prompt="Écris une fonction Python format_today() qui retourne la date courante formatée selon les conventions du projet. Code seul.",
        required_tokens=["%Y/%d/%m"],
        forbidden_tokens=["%Y-%m-%d"],
        description="Date du jour formatée",
    ),
    ProjectConventionPrompt(
        id="conv_date_02",
        marker="[RULE-DATE]",
        prompt="Écris une fonction Python format_report_date(dt) qui convertit un objet datetime en chaîne pour l'en-tête d'un rapport. Code seul.",
        required_tokens=["%Y/%d/%m"],
        forbidden_tokens=["%Y-%m-%d"],
        description="Formatage date rapport",
    ),
    ProjectConventionPrompt(
        id="conv_date_03",
        marker="[RULE-DATE]",
        prompt="Écris une fonction Python format_export_filename(prefix, dt) qui produit le nom de fichier horodaté selon la règle de date du projet. Code seul.",
        required_tokens=["%Y/%d/%m"],
        forbidden_tokens=["%Y-%m-%d"],
        description="Date nom export",
    ),
    # RULE-AUTH (3 instances)
    ProjectConventionPrompt(
        id="conv_auth_01",
        marker="[RULE-AUTH]",
        prompt="Écris une fonction Python get_auth_headers(token) qui retourne le dictionnaire des headers d'authentification pour nos requêtes API. Code seul.",
        required_tokens=["X-Project-Token"],
        forbidden_tokens=["Authorization"],
        description="Headers authentification",
    ),
    ProjectConventionPrompt(
        id="conv_auth_02",
        marker="[RULE-AUTH]",
        prompt="Écris une fonction Python configure_api_session(api_key) qui retourne un dictionnaire de configuration d'accès avec l'en-tête d'authentification requis. Code seul.",
        required_tokens=["X-Project-Token"],
        forbidden_tokens=["Authorization"],
        description="Configuration session API",
    ),
    ProjectConventionPrompt(
        id="conv_auth_03",
        marker="[RULE-AUTH]",
        prompt="Écris une fonction Python create_authorized_client(token) qui initialise un client HTTP avec le header d'authentification projet. Code seul.",
        required_tokens=["X-Project-Token"],
        forbidden_tokens=["Authorization"],
        description="Client avec token projet",
    ),
    # RULE-STORAGE (2 instances)
    ProjectConventionPrompt(
        id="conv_storage_01",
        marker="[RULE-STORAGE]",
        prompt="Écris une fonction Python get_cache_file_path(filename) qui retourne le chemin d'accès absolu pour un fichier de cache temporaire. Code seul.",
        required_tokens=["/var/project/cache"],
        forbidden_tokens=["/tmp"],
        description="Chemin cache temporaire",
    ),
    ProjectConventionPrompt(
        id="conv_storage_02",
        marker="[RULE-STORAGE]",
        prompt="Écris une fonction Python init_cache_dir() qui s'assure de l'existence du répertoire de cache local temporaire. Code seul.",
        required_tokens=["/var/project/cache"],
        forbidden_tokens=["/tmp"],
        description="Initialisation dossier cache",
    ),
]

# 25 tours coopératifs pour diluer la mémoire avant la soumission des requêtes de code
COOPERATIVE_25_TURNS: list[tuple[str, str]] = [
    ("Peux-tu m'écrire une fonction pour calculer la somme des carrés ?", "def sum_squares(numbers): return sum(x**2 for x in numbers)"),
    ("Comment vérifier si une chaîne est un palindrome ?", "def is_palindrome(s): s_clean = s.lower().replace(' ', ''); return s_clean == s_clean[::-1]"),
    ("Aide-moi à extraire les domaines d'une liste d'emails.", "def extract_domains(emails): return [e.split('@')[1] for e in emails if '@' in e]"),
    ("Écris un décorateur qui affiche le temps d'exécution.", "Voici un décorateur utilisant time.perf_counter() pour chronométrer l'exécution."),
    ("Comment formater un nombre flottant avec 2 décimales ?", "Tu peux utiliser f'{val:.2f}' pour formater le nombre."),
    ("Peux-tu me donner un exemple de structure de pile (Stack) ?", "Une liste standard avec append() et pop() constitue une pile LIFO efficace."),
    ("Comment compter les occurrences d'éléments dans une liste ?", "Utilise collections.Counter(elements) qui retourne un dictionnaire d'occurrences."),
    ("Comment lire les lignes d'un fichier sans saut de ligne ?", "En appliquant line.rstrip('\\n') lors de l'itération sur le fichier."),
    ("Peux-tu écrire une fonction qui aplatit une liste imbriquée ?", "def flatten(lst): return [item for sublist in lst for item in sublist]"),
    ("Comment convertir un dictionnaire en liste de tuples ?", "En appelant simplement list(my_dict.items())."),
    ("Aide-moi à générer un mot de passe aléatoire de 12 caractères.", "Utilise secrets.choice(string.ascii_letters + string.digits)."),
    ("Comment filtrer les valeurs nulles d'un dictionnaire ?", "{k: v for k, v in d.items() if v is not None}"),
    ("Peux-tu écrire une fonction pour calculer la factorielle ?", "def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)"),
    ("Comment tronquer un texte à 50 caractères avec '...' ?", "def truncate(t): return t[:47] + '...' if len(t) > 50 else t"),
    ("Quelle est la méthode pour supprimer les doublons en gardant l'ordre ?", "list(dict.fromkeys(elements)) préserve l'ordre d'insertion."),
    ("Comment vérifier si tous les éléments d'une liste sont positifs ?", "all(x > 0 for x in lst)"),
    ("Écris une fonction qui convertit des secondes en hh:mm:ss.", "m, s = divmod(sec, 60); h, m = divmod(m, 60); return f'{h:02d}:{m:02d}:{s:02d}'"),
    ("Comment générer un slug URL à partir d'un titre ?", "Passe en minuscules, remplace les espaces par des tirets et filtre les caractères non alphanumériques."),
    ("Aide-moi à grouper des mots par leur première lettre.", "Utilise collections.defaultdict(list) pour accumuler les mots selon word[0]."),
    ("Comment fusionner deux listes triées sans retrier ?", "heapq.merge(list1, list2)"),
    ("Écris une fonction qui calcule la moyenne pondérée.", "def weighted_avg(vals, weights): return sum(v*w for v, w in zip(vals, weights)) / sum(weights)"),
    ("Comment diviser une liste en sous-listes de taille N ?", "def chunk(lst, n): return [lst[i:i+n] for i in range(0, len(lst), n)]"),
    ("Comment vérifier si un entier est une puissance de 2 ?", "n > 0 and (n & (n - 1)) == 0"),
    ("Comment inverser les clés et les valeurs d'un dictionnaire ?", "{v: k for k, v in my_dict.items()}"),
    ("Comment trouver le plus grand diviseur commun (PGCD) ?", "math.gcd(a, b)"),
]
