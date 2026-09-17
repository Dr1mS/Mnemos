"""Jeux de données de Test Hold-out (strictement isolés, >= 20 instances par épreuve).

ATTENTION MÉTHODOLOGIQUE :
- Ce fichier est le jeu de test final en aveugle (Hold-out).
- Il utilise des FORMULATIONS ET TOURNURES DIVERSES et distinctes du jeu Dev.
- Il comporte 20 instances indépendantes pour chaque épreuve (60 instances au total).
- NE DOIT PAS être exécuté pendant la Phase 3b sur le jeu Dev. Il ne sera exécuté qu'une seule fois, à la toute fin.
"""

from __future__ import annotations

from bench.datasets import (
    BlandNoiseInstance,
    ExternalOntologyInstance,
    GhostVectorInstance,
    InteractionTurn,
    ProjectConventionPrompt,
    _BANK_NEUTRAL_1,
    _BANK_NEUTRAL_2,
)

# ── ÉPREUVE 1 : HOLDOUT MUTATION TEMPORELLE (20 instances, formulations variées) ──

_HOLDOUT_CITIES_DATA: list[tuple[str, str, str, str, str, str]] = [
    # (c0, c1, c2, phrase_t0, phrase_t1, phrase_t2)
    (
        "Poitiers", "Nanterre", "Créteil",
        "Mon adresse initiale était située à Poitiers.",
        "Par la suite, j'ai pris un appartement à Nanterre.",
        "Désormais, j'ai posé mes valises pour de bon à Créteil.",
    ),
    (
        "Versailles", "Courbevoie", "Pau",
        "Au départ, j'étais domicilié à Versailles.",
        "Changement de situation : j'ai transféré ma résidence principale à Courbevoie.",
        "À ce jour, ma vie se déroule entièrement à Pau.",
    ),
    (
        "Vitry-sur-Seine", "Asnières-sur-Seine", "Colombes",
        "J'ai commencé par m'installer à Vitry-sur-Seine.",
        "Plus tard, j'ai emménagé du côté d'Asnières-sur-Seine.",
        "Finalement, je réside dorénavant à Colombes.",
    ),
    (
        "Aulnay-sous-Bois", "La Rochelle", "Rueil-Malmaison",
        "Mon foyer initial se trouvait à Aulnay-sous-Bois.",
        "J'ai ensuite quitté la ville pour élire domicile à La Rochelle.",
        "Actuellement, mon point de chute définitif est à Rueil-Malmaison.",
    ),
    (
        "Antibes", "Saint-Maur-des-Fossés", "Calais",
        "Je vivais d'abord du côté d'Antibes.",
        "J'ai changé de logement pour habiter à Saint-Maur-des-Fossés.",
        "Pour info, c'est désormais à Calais que je demeure.",
    ),
    (
        "Champigny-sur-Marne", "Aubervilliers", "Béziers",
        "Ma première attache géographique était Champigny-sur-Marne.",
        "J'ai déménagé par la suite à Aubervilliers.",
        "Maintenant, je suis durablement logé à Béziers.",
    ),
    (
        "Cannes", "Bourges", "Saint-Nazaire",
        "À la base, j'avais mon chez-moi à Cannes.",
        "J'ai pris un nouveau bail pour habiter à Bourges.",
        "À l'heure actuelle, je réside à Saint-Nazaire.",
    ),
    (
        "Colmar", "Quimper", "Valence",
        "Initialement, mon domicile légal était à Colmar.",
        "J'ai ensuite posé mes meubles à Quimper.",
        "Dernière nouvelle : j'ai emménagé pour de bon à Valence.",
    ),
    (
        "Ajaccio", "Vénissieux", "Troyes",
        "J'ai passé mes premières années à Ajaccio.",
        "Puis je suis venu m'installer à Vénissieux.",
        "Présentement, c'est à Troyes que je suis établi.",
    ),
    (
        "Neuilly-sur-Seine", "Antony", "Pessac",
        "Au début, j'habitais à Neuilly-sur-Seine.",
        "J'ai basculé de logement vers Antony.",
        "En définitive, je vis maintenant à Pessac.",
    ),
    (
        "Ivry-sur-Seine", "Chambéry", "Lorient",
        "Mon point de départ était Ivry-sur-Seine.",
        "Quelque temps après, j'ai emménagé à Chambéry.",
        "Mon adresse courante est fixée à Lorient.",
    ),
    (
        "Sarcelles", "Cergy", "Niort",
        "J'étais localisé à Sarcelles au commencement.",
        "J'ai ensuite déplacé ma résidence vers Cergy.",
        "Mon foyer se situe à présent à Niort.",
    ),
    (
        "Montauban", "Beauvais", "Hyères",
        "J'ai d'abord résidé dans la ville de Montauban.",
        "Puis j'ai migré vers Beauvais.",
        "Désormais, je me suis fixé à Hyères.",
    ),
    (
        "Cholet", "Vannes", "Évry-Courcouronnes",
        "Ma précédente maison était à Cholet.",
        "J'ai fait mes cartons pour m'établir à Vannes.",
        "Aujourd'hui, j'ai mon adresse principale à Évry-Courcouronnes.",
    ),
    (
        "Pantin", "Fontenay-sous-Bois", "Arles",
        "J'avais pris mes quartiers à Pantin.",
        "J'ai changé de commune pour habiter à Fontenay-sous-Bois.",
        "C'est décidé, je suis dorénavant domicilié à Arles.",
    ),
    (
        "Sartrouville", "Clichy", "Narbonne",
        "Au tout départ, je logeais à Sartrouville.",
        "Une opportunité m'a fait emménager à Clichy.",
        "Depuis peu, c'est à Narbonne que j'habite.",
    ),
    (
        "Fréjus", "Belfort", "Albi",
        "J'avais élu domicile à Fréjus.",
        "Par la suite, je me suis transporté à Belfort.",
        "C'est officiel : j'habite désormais à Albi.",
    ),
    (
        "Grasse", "Laval", "Meaux",
        "Mon premier pied-à-terre était à Grasse.",
        "J'ai déménagé pour vivre à Laval.",
        "À ce jour, mon lieu de vie exclusif est Meaux.",
    ),
    (
        "Bobigny", "Châteauroux", "Carcassonne",
        "J'ai débuté en résidant à Bobigny.",
        "Puis j'ai loué une maison à Châteauroux.",
        "Désormais, ma résidence active est à Carcassonne.",
    ),
    (
        "Blois", "Saint-Brieuc", "Bayonne",
        "J'étais établi initialement à Blois.",
        "J'ai ensuite changé d'horizon pour résider à Saint-Brieuc.",
        "Finalement, je suis installé à Bayonne pour y vivre.",
    ),
]

HOLDOUT_GHOST_INSTANCES: list[GhostVectorInstance] = [
    GhostVectorInstance(
        id=f"ghost_hld_{i+1:02d}",
        city_0=c0,
        city_1=c1,
        city_2=c2,
        neutral_turns_1=_BANK_NEUTRAL_1,
        neutral_turns_2=_BANK_NEUTRAL_2,
    )
    for i, (c0, c1, c2, _, _, _) in enumerate(_HOLDOUT_CITIES_DATA)
]


# ── ÉPREUVE 1B : HOLDOUT MUTATION TEMPORELLE HORS ONTOLOGIE (20 instances) ──

_HOLDOUT_EXTERNAL_DATA: list[dict[str, str]] = [
    {
        "topic": "registre_conteneur",
        "val_0": "Docker Hub",
        "val_1": "Quay.io",
        "val_2": "Harbor",
        "statement_0": "Nos images de conteneurs sont hébergées initialement sur Docker Hub.",
        "statement_1": "Nous avons opéré un transfert de nos dépôts d'images vers Quay.io.",
        "statement_2": "À présent, notre registre de conteneurs privé exclusif est Harbor.",
        "probe_1_1": "Quel est notre registre de conteneurs actuel ?",
        "probe_1_2": "Quels ont été nos registres de conteneurs successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "cache_distribue",
        "val_0": "Memcached",
        "val_1": "Redis",
        "val_2": "Valkey",
        "statement_0": "Le cache applicatif distribué est pris en charge par Memcached.",
        "statement_1": "Nous avons modernisé la couche de mise en cache avec Redis.",
        "statement_2": "Par mesure de pérennité, notre moteur de cache actif est désormais Valkey.",
        "probe_1_1": "Quel moteur de cache distribué utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos moteurs de cache successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "apm_observabilite",
        "val_0": "New Relic",
        "val_1": "Dynatrace",
        "val_2": "AppDynamics",
        "statement_0": "Notre plateforme d'APM de suivi des performances était New Relic.",
        "statement_1": "Nous avons ensuite déployé Dynatrace pour monitorer nos applications.",
        "statement_2": "Désormais, notre APM principal en production est AppDynamics.",
        "probe_1_1": "Quelle plateforme d'APM utilisons-nous actuellement ?",
        "probe_1_2": "Quelles ont été nos plateformes d'APM successives dans l'ordre chronologique ?",
    },
    {
        "topic": "collecteur_logs",
        "val_0": "Logstash",
        "val_1": "Fluentbit",
        "val_2": "Vector",
        "statement_0": "Le pipeline d'ingestion de logs utilisait au départ Logstash.",
        "statement_1": "Pour réduire la charge mémoire, nous avons basculé sur Fluentbit.",
        "statement_2": "Actuellement, notre collecteur et routeur de logs unique est Vector.",
        "probe_1_1": "Quel collecteur de logs utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos collecteurs de logs successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "fournisseur_dns",
        "val_0": "Cloudflare",
        "val_1": "Route 53",
        "val_2": "NS1",
        "statement_0": "Notre zone DNS d'entreprise était initialement gérée par Cloudflare.",
        "statement_1": "Nous avons réorienté la gestion de nos enregistrements DNS vers Route 53.",
        "statement_2": "Aujourd'hui, l'ensemble de notre routage DNS est confié à NS1.",
        "probe_1_1": "Quel est notre fournisseur DNS d'entreprise actuel ?",
        "probe_1_2": "Quels ont été nos fournisseurs DNS successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "load_balancer",
        "val_0": "HAProxy",
        "val_1": "F5 BIG-IP",
        "val_2": "MetalLB",
        "statement_0": "L'équilibrage de charge réseau a d'abord reposé sur HAProxy.",
        "statement_1": "Nous avons ensuite migré nos répartiteurs vers F5 BIG-IP.",
        "statement_2": "En cluster bare-metal, notre load balancer opérationnel est MetalLB.",
        "probe_1_1": "Quel équilibreur de charge réseau utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos équilibreurs de charge successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "reseau_cdn",
        "val_0": "Akamai",
        "val_1": "Fastly",
        "val_2": "Cloudflare CDN",
        "statement_0": "La distribution de contenu statique mondial était assurée par Akamai.",
        "statement_1": "Nous sommes ensuite passés par le réseau CDN de Fastly.",
        "statement_2": "Désormais, notre réseau de diffusion de contenu actif est Cloudflare CDN.",
        "probe_1_1": "Quel réseau CDN utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos réseaux CDN successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "forge_logicielle",
        "val_0": "Bitbucket",
        "val_1": "GitLab self-hosted",
        "val_2": "GitHub Enterprise",
        "statement_0": "Le dépôt de nos sources a commencé sur Bitbucket.",
        "statement_1": "Nous avons par la suite monté une instance interne de GitLab self-hosted.",
        "statement_2": "À ce jour, notre forge logicielle officielle est GitHub Enterprise.",
        "probe_1_1": "Quelle est notre forge logicielle officielle actuelle ?",
        "probe_1_2": "Quelles ont été nos forges logicielles successives dans l'ordre chronologique ?",
    },
    {
        "topic": "messagerie_pro",
        "val_0": "Mattermost",
        "val_1": "Slack",
        "val_2": "Discord",
        "statement_0": "Notre équipe communiquait au départ via Mattermost.",
        "statement_1": "Nous avons ensuite transité sur l'espace Slack de l'entreprise.",
        "statement_2": "Finalement, nos canaux d'échange internes sont regroupés sur Discord.",
        "probe_1_1": "Quelle messagerie professionnelle utilisons-nous actuellement ?",
        "probe_1_2": "Quelles ont été nos messageries professionnelles successives dans l'ordre chronologique ?",
    },
    {
        "topic": "documentation_wiki",
        "val_0": "Confluence",
        "val_1": "BookStack",
        "val_2": "Obsidian Vault",
        "statement_0": "Le wiki d'équipe et la base documentaire résidaient dans Confluence.",
        "statement_1": "Nous avons exporté l'ensemble de notre documentation vers BookStack.",
        "statement_2": "Actuellement, notre base de connaissances partagée est un Obsidian Vault.",
        "probe_1_1": "Quel outil de documentation d'équipe utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos outils de documentation successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "feature_flags",
        "val_0": "LaunchDarkly",
        "val_1": "Flagsmith",
        "val_2": "Unleash",
        "statement_0": "L'activation progressive des fonctionnalités était pilotée par LaunchDarkly.",
        "statement_1": "Nous avons basculé notre gestion de drapeaux de fonctionnalités sur Flagsmith.",
        "statement_2": "Désormais, notre serveur de feature flags en production est Unleash.",
        "probe_1_1": "Quel outil de feature flags utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos outils de feature flags successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "orchestration_taches",
        "val_0": "Apache Airflow",
        "val_1": "Prefect",
        "val_2": "Dagster",
        "statement_0": "L'orchestration des flux de données était configurée sous Apache Airflow.",
        "statement_1": "Nous avons réécrit nos pipelines de données avec Prefect.",
        "statement_2": "En production, notre orchestrateur de données opérationnel est Dagster.",
        "probe_1_1": "Quel orchestrateur de tâches utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos orchestrateurs de tâches successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "depot_artefacts",
        "val_0": "Nexus Repository",
        "val_1": "JFrog Artifactory",
        "val_2": "Cloudsmith",
        "statement_0": "Nos paquets binaires internes étaient hébergés dans Nexus Repository.",
        "statement_1": "Nous avons ensuite déployé nos librairies sur JFrog Artifactory.",
        "statement_2": "Dorénavant, notre registre d'artefacts applicatifs centralisé est Cloudsmith.",
        "probe_1_1": "Quel registre d'artefacts utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos registres d'artefacts successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "stockage_objet",
        "val_0": "MinIO",
        "val_1": "AWS S3",
        "val_2": "Cloudflare R2",
        "statement_0": "Le stockage d'objets non structurés a démarré sur une instance MinIO.",
        "statement_1": "Puis nous avons migré nos compartiments sur AWS S3.",
        "statement_2": "Pour supprimer les frais de sortie, notre stockage objet principal est Cloudflare R2.",
        "probe_1_1": "Quel stockage objet principal utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos stockages objets successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "suivi_erreurs",
        "val_0": "Rollbar",
        "val_1": "Bugsnag",
        "val_2": "Sentry",
        "statement_0": "La remontée des exceptions non capturées passait par Rollbar.",
        "statement_1": "Nous avons expérimenté la capture des crashs avec Bugsnag.",
        "statement_2": "À ce jour, notre outil de tracking d'erreurs en production est Sentry.",
        "probe_1_1": "Quel outil de suivi d'erreurs utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos outils de suivi d'erreurs successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "moteur_recherche",
        "val_0": "Elasticsearch",
        "val_1": "OpenSearch",
        "val_2": "Meilisearch",
        "statement_0": "L'indexation de texte intégral s'appuyait sur Elasticsearch.",
        "statement_1": "Nous avons fork notre moteur vers OpenSearch.",
        "statement_2": "Actuellement, notre moteur de recherche rapide en production est Meilisearch.",
        "probe_1_1": "Quel moteur de recherche utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos moteurs de recherche successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "base_series_temporelles",
        "val_0": "InfluxDB",
        "val_1": "TimescaleDB",
        "val_2": "VictoriaMetrics",
        "statement_0": "La métrologie fine et les séries temporelles étaient stockées dans InfluxDB.",
        "statement_1": "Nous avons unifié le stockage avec une extension TimescaleDB.",
        "statement_2": "En production, notre stockage de séries temporelles actif est VictoriaMetrics.",
        "probe_1_1": "Quelle base de séries temporelles utilisons-nous actuellement ?",
        "probe_1_2": "Quelles ont été nos bases de séries temporelles successives dans l'ordre chronologique ?",
    },
    {
        "topic": "passerelle_api",
        "val_0": "Kong Gateway",
        "val_1": "Tyk",
        "val_2": "KrakenD",
        "statement_0": "L'exposition publique de nos microservices était assurée par Kong Gateway.",
        "statement_1": "Nous avons changé de passerelle API pour adopter Tyk.",
        "statement_2": "Désormais, notre API gateway ultra-légère en place est KrakenD.",
        "probe_1_1": "Quelle passerelle API utilisons-nous actuellement ?",
        "probe_1_2": "Quelles ont été nos passerelles API successives dans l'ordre chronologique ?",
    },
    {
        "topic": "agent_telemetrie",
        "val_0": "Telegraf",
        "val_1": "OpenTelemetry Collector",
        "val_2": "Grafana Agent",
        "statement_0": "La collecte des métriques machine reposait sur Telegraf.",
        "statement_1": "Nous avons uniformisé nos traces avec OpenTelemetry Collector.",
        "statement_2": "À présent, l'agent de télémétrie déployé sur chaque nœud est Grafana Agent.",
        "probe_1_1": "Quel agent de télémétrie utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos agents de télémétrie successifs dans l'ordre chronologique ?",
    },
    {
        "topic": "hebergement_statique",
        "val_0": "Vercel",
        "val_1": "Netlify",
        "val_2": "Cloudflare Pages",
        "statement_0": "Le frontend et nos pages vitrines étaient déployés chez Vercel.",
        "statement_1": "Nous avons testé et migré notre site statique sur Netlify.",
        "statement_2": "Actuellement, notre hébergeur statique de production est Cloudflare Pages.",
        "probe_1_1": "Quel hébergeur statique utilisons-nous actuellement ?",
        "probe_1_2": "Quels ont été nos hébergeurs statiques successifs dans l'ordre chronologique ?",
    },
]

HOLDOUT_EXTERNAL_INSTANCES: list[ExternalOntologyInstance] = [
    ExternalOntologyInstance(
        id=f"ext_hld_{i+1:02d}",
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
    for i, item in enumerate(_HOLDOUT_EXTERNAL_DATA)
]


# ── ÉPREUVE 2 : HOLDOUT BRUIT & DÉCROISSANCE (20 flux, formulations variées) ──

_HOLDOUT_FACTS_DATA: list[tuple[str, str, str, str]] = [
    # (key_code, allergy, phrasing_key, phrasing_allergy)
    (
        "HLD-1001", "sésame",
        "Garde impérativement en mémoire : Le secret d'accès pour l'infrastructure de recette est HLD-1001.",
        "Alerte sécurité vitale : Je présente une intolérance aiguë et létale au sésame.",
    ),
    (
        "HLD-1002", "nickel",
        "Consigne capitale d'infrastructure : Le jeton confidentiel du serveur de pré-production est HLD-1002.",
        "Point médical critique : Toute exposition ou contact avec le nickel provoque une réaction anaphylactique d'urgence.",
    ),
    (
        "HLD-1003", "morphine",
        "Mémorise cette clé technique hautement prioritaire : Le code maître du cluster QA est HLD-1003.",
        "Avertissement sanitaire d'urgence : L'administration de morphine m'est formellement mortelle.",
    ),
    (
        "HLD-1004", "amoxicilline",
        "Paramètre critique de déploiement : La clé d'authentification du bastion de test est HLD-1004.",
        "Information santé vitale : Je fais un choc allergique fatal en cas de prise d'amoxicilline.",
    ),
    (
        "HLD-1005", "fructose",
        "Donnée essentielle de sécurité réseau : La clé secrète de la passerelle intermédiaire est HLD-1005.",
        "Contrainte médicale impérative : Mon organisme ne tolère aucun fructose, risque d'œdème mortel.",
    ),
    (
        "HLD-1006", "céphalosporines",
        "Secret d'administration système : L'identifiant sécurisé de l'environnement miroir est HLD-1006.",
        "Urgence médicale absolue : Présence d'une allergie sévère et létale aux céphalosporines.",
    ),
    (
        "HLD-1007", "avocat",
        "Information d'infrastructure confidentielle : Le mot de passe applicatif du noeud de staging est HLD-1007.",
        "Alerte médicale personnelle : L'ingestion d'avocat déclenche chez moi un arrêt respiratoire immédiat.",
    ),
    (
        "HLD-1008", "kiwi",
        "Clé d'authentification vitale pour l'infra : Le jeton de validation du pipeline est HLD-1008.",
        "Condition de santé critique : Je souffre d'une allergie mortelle au kiwi.",
    ),
    (
        "HLD-1009", "mollusques",
        "Clé maître de l'environnement de validation : Le token d'accès interne est HLD-1009.",
        "Avertissement vital : La moindre trace de mollusques dans mon alimentation entraîne la mort.",
    ),
    (
        "HLD-1010", "moutarde",
        "Identifiant d'accès technique très sensible : La clé du serveur d'intégration est HLD-1010.",
        "Urgence de sécurité physique : Je suis gravement allergique à la moutarde, risque mortel avéré.",
    ),
    (
        "HLD-1011", "céleri",
        "Consigne stricte pour les serveurs de tests : La clé de sécurité autorisée est HLD-1011.",
        "Alerte de santé majeure : Le céleri est pour moi un poison violent provoquant un choc fatal.",
    ),
    (
        "HLD-1012", "lupin",
        "Code secret pour les services de staging : L'API key du nœud de recette est HLD-1012.",
        "Avis médical critique : Mon dossier signale une allergie sévère mortelle à la farine de lupin.",
    ),
    (
        "HLD-1013", "poisson",
        "Token de connexion à retenir absolument : La clé du serveur intermédiaire est HLD-1013.",
        "Information vitale urgente : Consommation de poisson strictement proscrite sous peine de mort.",
    ),
    (
        "HLD-1014", "latex",
        "Clé confidentielle pour le cluster de validation : Le credential maître est HLD-1014.",
        "Précision médicale vitale : Risque anaphylactique majeur et mortel en présence de latex.",
    ),
    (
        "HLD-1015", "vancomycine",
        "Identifiant confidentiel du service de recette : La clé de session est HLD-1015.",
        "Alerte médicale absolue : La vancomycine m'est totalement mortelle.",
    ),
    (
        "HLD-1016", "crevettes",
        "Jeton d'autorisation pour les tests end-to-end : La clé de build est HLD-1016.",
        "Contrainte de survie élémentaire : Ingestion de crevettes formellement mortelle pour moi.",
    ),
    (
        "HLD-1017", "noisettes",
        "Jeton d'API exclusif pour l'infrastructure QA : Le secret configuré est HLD-1017.",
        "Fait médical prioritaire : Choc anaphylactique mortel instantané en cas de noisettes.",
    ),
    (
        "HLD-1018", "procaïne",
        "Secret d'accès réseau pour le staging : La clé du reverse-proxy est HLD-1018.",
        "Notice médicale capitale : Allergie mortelle confirmée à la procaïne.",
    ),
    (
        "HLD-1019", "châtaignes",
        "Clé d'autorisation système pour les déploiements de test : Le token est HLD-1019.",
        "Consigne sanitaire absolue : Je suis sévèrement allergique aux châtaignes, pronostic vital engagé.",
    ),
    (
        "HLD-1020", "anti-inflammatoires",
        "Jeton de sécurité de secours pour la pré-prod : La clé enregistrée est HLD-1020.",
        "Mise en garde vitale : Tout traitement aux anti-inflammatoires déclenche un œdème de Quincke mortel.",
    ),
]

HOLDOUT_BLAND_INSTANCES: list[BlandNoiseInstance] = [
    BlandNoiseInstance(
        id=f"bland_hld_{i+1:02d}",
        key_code=kc,
        allergy=alg,
        noise_seed_offset=200 + i * 11,
    )
    for i, (kc, alg, _, _) in enumerate(_HOLDOUT_FACTS_DATA)
]


def build_holdout_bland_noise_stream(instance: BlandNoiseInstance, total_turns: int = 100) -> list[InteractionTurn]:
    """Construit le flux hold-out avec formulations alternatives pour faits critiques."""
    idx = int(instance.id.split("_")[-1]) - 1
    _, _, phrase_key, phrase_allergy = _HOLDOUT_FACTS_DATA[idx]
    turns: list[InteractionTurn] = []
    from bench.datasets import BLAND_NOISE_OBSERVATIONS

    for i in range(total_turns):
        if i == 24:
            turns.append(
                InteractionTurn(
                    role="user",
                    content=phrase_key,
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
                    content=phrase_allergy,
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
                content=f"{phrase} (remarque hold-out n°{i+1})",
                timestamp_offset_days=i * 0.1,
                salience_hint=0.15,
                is_critical=False,
            )
        )
    return turns


# ── ÉPREUVE 3 : HOLDOUT CONVENTIONS DE PROJET (20 requêtes variées) ─────────

HOLDOUT_CONVENTION_PROMPTS: list[ProjectConventionPrompt] = [
    # HTTP (4 instances)
    ProjectConventionPrompt(
        id="conv_hld_http_01",
        marker="[RULE-HTTP]",
        prompt="Rédige une fonction Python pour streamer une réponse HTTP en continu depuis une URL cible. Code seul.",
        required_tokens=["httpx"],
        forbidden_tokens=["requests", "urllib"],
        description="Stream HTTP streaming",
    ),
    ProjectConventionPrompt(
        id="conv_hld_http_02",
        marker="[RULE-HTTP]",
        prompt="Développe une fonction Python asynchrone effectuant des requêtes de monitoring sur une liste de serveurs. Code seul.",
        required_tokens=["httpx"],
        forbidden_tokens=["requests", "urllib"],
        description="Monitoring HTTP asynchrone",
    ),
    ProjectConventionPrompt(
        id="conv_hld_http_03",
        marker="[RULE-HTTP]",
        prompt="Écris un wrapper de client HTTP avec gestion des timeouts et retries automatiques. Code seul.",
        required_tokens=["httpx"],
        forbidden_tokens=["requests", "urllib"],
        description="Wrapper client HTTP",
    ),
    ProjectConventionPrompt(
        id="conv_hld_http_04",
        marker="[RULE-HTTP]",
        prompt="Conçois une fonction Python pour téléverser un fichier multipart/form-data vers une API distante. Code seul.",
        required_tokens=["httpx"],
        forbidden_tokens=["requests", "urllib"],
        description="Upload multipart HTTP",
    ),
    # LOG (4 instances)
    ProjectConventionPrompt(
        id="conv_hld_log_01",
        marker="[RULE-LOG]",
        prompt="Écris une fonction Python log_metric(name, value) qui enregistre une métrique dans les logs selon le standard de log du projet. Code seul.",
        required_tokens=["PRJ-"],
        forbidden_tokens=[],
        description="Log métrique système",
    ),
    ProjectConventionPrompt(
        id="conv_hld_log_02",
        marker="[RULE-LOG]",
        prompt="Rédige une fonction Python log_audit_trail(user_id, action) consignant une action d'audit en respectant la convention de log du projet. Code seul.",
        required_tokens=["PRJ-"],
        forbidden_tokens=[],
        description="Log piste d'audit",
    ),
    ProjectConventionPrompt(
        id="conv_hld_log_03",
        marker="[RULE-LOG]",
        prompt="Écris un hook de capture d'exceptions sys.excepthook formatant l'erreur fatale selon la convention de log du projet. Code seul.",
        required_tokens=["PRJ-"],
        forbidden_tokens=[],
        description="Hook d'erreur fatale",
    ),
    ProjectConventionPrompt(
        id="conv_hld_log_04",
        marker="[RULE-LOG]",
        prompt="Développe une fonction Python create_rotating_logger() qui initialise un logger rotatif conforme au préfixe projet. Code seul.",
        required_tokens=["PRJ-"],
        forbidden_tokens=[],
        description="Logger rotatif conforme",
    ),
    # DATE (4 instances)
    ProjectConventionPrompt(
        id="conv_hld_date_01",
        marker="[RULE-DATE]",
        prompt="Écris une fonction Python parse_deadline(str_date) qui parse une chaîne de date formatée selon la convention de date du projet. Code seul.",
        required_tokens=["%Y/%d/%m"],
        forbidden_tokens=["%Y-%m-%d"],
        description="Parsing date deadline",
    ),
    ProjectConventionPrompt(
        id="conv_hld_date_02",
        marker="[RULE-DATE]",
        prompt="Rédige une fonction Python get_billing_period_string(start_dt, end_dt) générant l'intervalle de facturation au format de date projet. Code seul.",
        required_tokens=["%Y/%d/%m"],
        forbidden_tokens=["%Y-%m-%d"],
        description="Intervalle facturation date",
    ),
    ProjectConventionPrompt(
        id="conv_hld_date_03",
        marker="[RULE-DATE]",
        prompt="Développe une fonction Python format_release_tag(dt) qui génère un tag de version horodaté selon la règle de date du projet. Code seul.",
        required_tokens=["%Y/%d/%m"],
        forbidden_tokens=["%Y-%m-%d"],
        description="Tag release horodaté",
    ),
    ProjectConventionPrompt(
        id="conv_hld_date_04",
        marker="[RULE-DATE]",
        prompt="Écris une fonction Python format_log_timestamp(dt) retournant la portion date d'un timestamp respectant scrupuleusement la convention du projet. Code seul.",
        required_tokens=["%Y/%d/%m"],
        forbidden_tokens=["%Y-%m-%d"],
        description="Horodatage log date",
    ),
    # AUTH (4 instances)
    ProjectConventionPrompt(
        id="conv_hld_auth_01",
        marker="[RULE-AUTH]",
        prompt="Écris un middleware FastAPI extrayant le jeton d'authentification client conformément à l'en-tête exigé par le projet. Code seul.",
        required_tokens=["X-Project-Token"],
        forbidden_tokens=["Authorization"],
        description="Middleware FastAPI auth",
    ),
    ProjectConventionPrompt(
        id="conv_hld_auth_02",
        marker="[RULE-AUTH]",
        prompt="Développe un décorateur require_valid_token() qui vérifie la présence du header d'authentification officiel du projet. Code seul.",
        required_tokens=["X-Project-Token"],
        forbidden_tokens=["Authorization"],
        description="Décorateur validation token",
    ),
    ProjectConventionPrompt(
        id="conv_hld_auth_03",
        marker="[RULE-AUTH]",
        prompt="Rédige une fonction Python inject_auth_credentials(headers_dict, secret) qui ajoute la clé d'authentification projet au dictionnaire d'en-têtes. Code seul.",
        required_tokens=["X-Project-Token"],
        forbidden_tokens=["Authorization"],
        description="Injection credentials auth",
    ),
    ProjectConventionPrompt(
        id="conv_hld_auth_04",
        marker="[RULE-AUTH]",
        prompt="Écris une fonction Python validate_incoming_auth(request) inspectant les headers d'une requête HTTP pour extraire le token projet. Code seul.",
        required_tokens=["X-Project-Token"],
        forbidden_tokens=["Authorization"],
        description="Validation header auth",
    ),
    # STORAGE (4 instances)
    ProjectConventionPrompt(
        id="conv_hld_storage_01",
        marker="[RULE-STORAGE]",
        prompt="Écris une fonction Python get_lock_file_path(job_id) qui retourne le chemin du verrou temporaire local selon le dossier imposé par le projet. Code seul.",
        required_tokens=["/var/project/cache"],
        forbidden_tokens=["/tmp"],
        description="Verrou temporaire local",
    ),
    ProjectConventionPrompt(
        id="conv_hld_storage_02",
        marker="[RULE-STORAGE]",
        prompt="Développe une fonction Python get_session_store_path(session_id) retournant le chemin d'accès au fichier de session temporaire local. Code seul.",
        required_tokens=["/var/project/cache"],
        forbidden_tokens=["/tmp"],
        description="Fichier session temporaire",
    ),
    ProjectConventionPrompt(
        id="conv_hld_storage_03",
        marker="[RULE-STORAGE]",
        prompt="Rédige une fonction Python rotate_temp_artifacts(max_age_h) qui purge les artéfacts temporaires dans le répertoire de cache officiel du projet. Code seul.",
        required_tokens=["/var/project/cache"],
        forbidden_tokens=["/tmp"],
        description="Purge artéfacts temporaires",
    ),
    ProjectConventionPrompt(
        id="conv_hld_storage_04",
        marker="[RULE-STORAGE]",
        prompt="Écris une fonction Python prepare_temp_buffer_file(name) créant un fichier tampon temporaire dans l'emplacement de cache réservé au projet. Code seul.",
        required_tokens=["/var/project/cache"],
        forbidden_tokens=["/tmp"],
        description="Fichier tampon temporaire",
    ),
]
