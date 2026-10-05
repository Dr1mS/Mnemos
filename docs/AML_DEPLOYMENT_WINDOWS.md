# 🚀 Guide de Déploiement Windows (RTX 4070 Ti) — VPS Coolify + Tailscale
## Agent Memory Challenge (Cycle 2) — Agent Memory Leaderboard (AML)

Ce guide explique comment faire tourner **Mnemos** sur votre PC Windows (GPU RTX 4070 Ti)
et l'exposer en HTTPS sur une **URL fixe** (`https://mnemos.dr1ms.fr`) via le VPS Coolify.

```
Plateforme AML ──HTTPS──▶ mnemos.dr1ms.fr (VPS : Traefik/Coolify, certificat Let's Encrypt)
                              └── Tailscale (VPN chiffré) ──▶ PC Windows :8765 (Mnemos + Ollama GPU)
```

> **Pourquoi pas un Quick Tunnel Cloudflare (`cloudflared tunnel --url`) ?** L'URL
> `trycloudflare.com` est aléatoire et change à chaque relance, Cloudflare ne garantit
> « aucun SLA ni uptime » et limite à 200 requêtes simultanées. Or l'URL fait partie de
> la version déclarée à AML, et le règlement exige que l'endpoint reste joignable et
> stable **30 jours après la soumission** d'un run Full. Le Quick Tunnel
> (`scripts/start_aml_tunnel.ps1`) reste utile pour un test ponctuel, pas pour concourir.

---

## 📋 Table des Matières
1. [Pré-requis](#1-pré-requis)
2. [Étape 1 : Ollama & VRAM GPU](#étape-1--ollama--vram-gpu)
3. [Étape 2 : Projet Mnemos & `.env`](#étape-2--projet-mnemos--env)
4. [Étape 3 : Tailscale (PC + VPS)](#étape-3--tailscale-pc--vps)
5. [Étape 4 : Lancement du serveur Mnemos](#étape-4--lancement-du-serveur-mnemos)
6. [Étape 5 : Route Traefik dans Coolify](#étape-5--route-traefik-dans-coolify)
7. [Étape 6 : Auto-test du contrat](#étape-6--auto-test-du-contrat)
8. [Étape 7 : Tenir 30 jours](#étape-7--tenir-30-jours)
9. [Étape 8 : Demande d'accès AML](#étape-8--demande-daccès-aml)
10. [Dépannage](#dépannage)

---

## 1. Pré-requis

* **PC** : Windows 10/11, NVIDIA RTX 4070 Ti, pilotes récents.
* **VPS** : Coolify avec son proxy Traefik ; le DNS `*.dr1ms.fr` pointe déjà dessus
  (wildcard OVH → rien à ajouter pour `mnemos.dr1ms.fr`).
* **Outils PC** : [Ollama](https://ollama.com/download/windows), [uv](https://github.com/astral-sh/uv),
  [Git](https://gitforwindows.org/), [Tailscale](https://tailscale.com/download/windows).
* **Compte** Tailscale gratuit.

---

## Étape 1 : Ollama & VRAM GPU

```powershell
ollama pull bge-m3
ollama pull qwen2.5:3b
```

**Empêcher le déchargement de la VRAM** (sinon rechargement à froid de 2 à 10 s après
5 min d'inactivité) : `Win + R` → `sysdm.cpl` → **Paramètres système avancés** →
**Variables d'environnement** → *Variables système* → **Nouvelle** :
`OLLAMA_KEEP_ALIVE` = `-1`, puis redémarrer Ollama depuis la barre des tâches.

Vérification (les deux modèles doivent afficher `100% GPU` une fois utilisés) :
```powershell
ollama ps
```

---

## Étape 2 : Projet Mnemos & `.env`

```powershell
git checkout feat/aml-adapter
git pull origin feat/aml-adapter
uv sync
```

Générer la clé secrète que la plateforme utilisera (*Memory System Key*) :
```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
```

L'adresse Tailscale du PC (`100.x.y.z`) s'obtient à l'étape 3. Créer ensuite `.env` à la
racine (fichier ignoré par git — ne jamais commiter la clé) :
```env
# Clé déclarée à AML (Memory System Key). Obligatoire : sans elle l'API est ouverte à tous.
# Seule la valeur de la clé est comparée ; le header peut être Bearer, Token ou X-API-Key.
API_KEY=<clé générée ci-dessus>

# Écoute UNIQUEMENT sur l'interface Tailscale : invisible du réseau local,
# joignable seulement depuis le VPS.
API_HOST=100.x.y.z
API_PORT=8765
LOG_LEVEL=INFO

# Modèles Ollama
OLLAMA_HOST=http://localhost:11434
EMBED_MODEL=bge-m3
SALIENCE_MODEL=qwen2.5:3b
EXTRACTION_MODEL=qwen2.5:3b
LLM_THINK=false

# Mode épisodique pour la compétition : aucun LLM pendant l'évaluation, seul bge-m3.
# Mesuré : le LLM de fond divise le débit par ~2 sans gain en récupération ni en
# réponse (cf. bench/results/gpu/). Pour réactiver la consolidation :
# CONSOLIDATION_AUTO=true et retirer SALIENCE_QUEUE_WORKERS.
CONSOLIDATION_AUTO=false
SALIENCE_QUEUE_WORKERS=0
CONSOLIDATION_DELAY_HOURS=0.0

# Pas d'oubli temporel : les messages AML portent leurs dates d'origine (souvent
# anciennes) ; avec la rétention par défaut, ils seraient archivés comme « vieux ».
EPISODIC_RETENTION_DAYS=36500
DECAY_RATE_DAILY=0.0
```
`qwen2.5:3b` n'est alors plus utilisé en production : seul `bge-m3` doit être chargé.

Créer les bases SQLite (une seule fois, sur un clone neuf — lit les chemins depuis `.env`) :
```powershell
.\.venv\Scripts\python.exe -m alembic upgrade head
```

---

## Étape 3 : Tailscale (PC + VPS)

### 3.1 Sur le PC Windows
```powershell
winget install --id Tailscale.Tailscale
```
Se connecter via l'icône Tailscale de la barre des tâches, puis relever l'adresse du PC :
```powershell
tailscale ip -4
```
→ reporter cette adresse dans `API_HOST` du `.env`.

### 3.2 Sur le VPS (SSH, sur l'hôte — pas dans un conteneur)
```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```
Ouvrir le lien affiché pour rattacher le VPS au même compte.

### 3.3 Dans la console Tailscale (https://login.tailscale.com/admin/machines)
Pour le PC **et** le VPS : menu `…` → **Disable key expiry**, pour qu'aucune clé
n'expire pendant la compétition.

---

## Étape 4 : Lancement du serveur Mnemos

### 4.1 Pare-feu Windows (PowerShell **administrateur**, une seule fois)
N'autorise le port 8765 qu'aux adresses Tailscale :
```powershell
New-NetFirewallRule -DisplayName "Mnemos AML (Tailscale)" -Direction Inbound `
  -Protocol TCP -LocalPort 8765 -RemoteAddress 100.64.0.0/10 -Action Allow
```
Si Windows affiche une alerte de pare-feu pour `python.exe` / `mnemos.exe` au premier
lancement, cliquer **Autoriser** : le serveur n'écoute que sur l'interface Tailscale.

### 4.2 Test manuel
```powershell
.\.venv\Scripts\mnemos.exe serve
```
Attendu : `Uvicorn running on http://100.x.y.z:8765`.

### 4.3 Démarrage automatique (à l'ouverture de session, relance si le process meurt)
Arrêter d'abord le serveur lancé à la main (`Ctrl+C`), sinon le port 8765 est occupé.
```powershell
.\scripts\register_task.ps1 -TaskName "Mnemos AML" -LogFile "data\aml\serve.log"
Start-ScheduledTask -TaskName "Mnemos AML"
```
Suivre les logs : `Get-Content data\aml\serve.log -Wait -Tail 20`.

Redémarrer (après une mise à jour du code, par exemple). `Stop-ScheduledTask` ne tue que
`cmd.exe` : le processus Python garde le port 8765 et le redémarrage échouerait, il faut
donc l'arrêter explicitement :
```powershell
Stop-ScheduledTask -TaskName "Mnemos AML"
Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue |
  ForEach-Object { Stop-Process -Id $_.OwningProcess }
Start-ScheduledTask -TaskName "Mnemos AML"
```

---

## Étape 5 : Route Traefik dans Coolify

### 5.1 Vérifier que le proxy Coolify joint le PC (SSH sur le VPS)
```bash
curl -s http://100.x.y.z:8765/health
docker exec coolify-proxy wget -qO- http://100.x.y.z:8765/health
```
Les deux doivent renvoyer `{"status":"healthy",...}`. La seconde commande prouve que le
conteneur Traefik atteint bien le réseau Tailscale de l'hôte.

### 5.2 Ajouter la configuration dynamique
Coolify → **Servers** → *(le serveur)* → **Proxy** → **Dynamic Configurations** →
ajouter un fichier `mnemos.yaml` (remplacer `100.x.y.z`) :
```yaml
http:
  routers:
    mnemos:
      rule: Host(`mnemos.dr1ms.fr`)
      entryPoints:
        - https
      service: mnemos
      priority: 1000
      tls:
        certResolver: letsencrypt
    mnemos-http:
      rule: Host(`mnemos.dr1ms.fr`)
      entryPoints:
        - http
      middlewares:
        - mnemos-redirect
      service: mnemos
      priority: 1000
  middlewares:
    mnemos-redirect:
      redirectScheme:
        scheme: https
  services:
    mnemos:
      loadBalancer:
        servers:
          - url: 'http://100.x.y.z:8765'
```
> Les noms `http`, `https` et `letsencrypt` sont ceux de la configuration Traefik par
> défaut de Coolify. En cas de doute, les vérifier dans **Proxy → Configuration**
> (`--entrypoints.<nom>.address` et `--certificatesresolvers.<nom>`).
> `priority: 1000` garantit que cette route passe avant un éventuel routeur générique.

Traefik recharge le fichier automatiquement ; le certificat Let's Encrypt est émis à la
première requête HTTPS (quelques secondes).

---

## Étape 6 : Auto-test du contrat

Depuis n'importe quelle machine :
```powershell
.\.venv\Scripts\python.exe scripts\aml_selftest.py --url https://mnemos.dr1ms.fr --key <API_KEY>
```
Le script vérifie `/health` sans auth, l'écho exact des identifiants sur `/add`, la forme
de `/search` (`data`, ≤ `top_k`, `id` et `content` non vides), l'isolation par `user_id`
et le refus d'une clé invalide. Ses écritures vont dans un `user_id` jetable et unique.
Options : `--auth bearer|token|x-api-key`.

### Test de charge (capacité à déclarer)
```powershell
.\.venv\Scripts\python.exe scripts\aml_loadtest.py --url https://mnemos.dr1ms.fr --key <API_KEY>
```
Mesure débit, latences p50/p95 et erreurs pour des concurrences croissantes (Add : 1, 2, 4,
8 ; Search : 1, 4, 8, 16), avec des historiques PersonaMem découpés comme la plateforme
et des requêtes toutes distinctes (pas de cache). Il écrit ~2 000 messages dans des
`user_id` `loadtest:*`, que la consolidation traitera ensuite : repartir d'une base vide
avant l'inscription (arrêter le serveur, supprimer `data\aml\*.db`,
`alembic upgrade head`, relancer).

---

## Étape 7 : Tenir 30 jours

La checklist du run Full engage l'endpoint à rester **joignable et stable 30 jours après
la soumission**. Sur un PC de bureau, les coupures viennent surtout de la veille et des
redémarrages :

* **Désactiver la veille** (secteur) :
  ```powershell
  powercfg /change standby-timeout-ac 0
  powercfg /change hibernate-timeout-ac 0
  ```
* **Suspendre Windows Update** (Paramètres → Windows Update → *Suspendre les mises à
  jour*, jusqu'à 5 semaines) pendant la fenêtre d'évaluation.
* **Rester connecté** : la tâche « Mnemos Serve » et Ollama démarrent à l'ouverture de
  session. Verrouiller l'écran (`Win + L`) ne coupe rien ; se déconnecter, si.
* **Surveillance** : une sonde externe sur `https://mnemos.dr1ms.fr/health` (UptimeRobot).
  `/health` vérifie réellement Ollama, l'embedding et les deux bases (résultat mis en
  cache 30 s) :
  * `200 {"status":"healthy"}` : tout répond ;
  * `200 {"status":"degraded"}` : l'embedding répond lentement (charge, chargement du
    modèle), ce n'est pas une panne ;
  * `503 {"status":"unhealthy","failures":[...]}` : composant en panne (`ollama`,
    `embedding`, `episodic_db`, `semantic_db`), détails dans `data\aml\serve.log`.

---

## Étape 8 : Demande d'accès AML

* **Track** : `Textual Memory` — **Division** : `Open-source Methods`
* **Endpoints** : `https://mnemos.dr1ms.fr/add` et `https://mnemos.dr1ms.fr/search`
  (health : `https://mnemos.dr1ms.fr/health`)
* **Authentification** : `Bearer` + la valeur d'`API_KEY` (Token et X-Api-Key sont aussi acceptés)
* **Dépôt public + commit figé** : le commit exact déployé sur le PC

---

## Dépannage

| Problème | Cause probable | Solution |
|---|---|---|
| `mnemos serve` : « error while attempting to bind » | Tailscale pas encore connecté, ou `API_HOST` ≠ IP Tailscale | `tailscale ip -4`, corriger `.env`, relancer |
| `curl` depuis le VPS échoue, `tailscale ping <PC>` OK | Pare-feu Windows | Règle de l'étape 4.1 ; vérifier aucune règle *Bloquer* sur `python.exe` |
| `docker exec coolify-proxy wget …` échoue alors que `curl` sur l'hôte marche | Routage conteneur → Tailscale | Vérifier que Tailscale tourne sur l'hôte (`tailscale status`) |
| `https://mnemos.dr1ms.fr` → 404 | Route Traefik absente ou mal nommée | Vérifier le fichier dynamique et les noms d'entrypoints (étape 5.2) |
| `https://mnemos.dr1ms.fr` → 502 / 504 | PC éteint, en veille ou serveur arrêté | `Get-ScheduledTask "Mnemos Serve"`, étape 7 |
| 401 Unauthorized | Clé ou schéma d'auth différent de celui déclaré à AML | Vérifier `API_KEY` et le header |
| Latence élevée au premier appel | Modèle déchargé de la VRAM | `OLLAMA_KEEP_ALIVE=-1` (étape 1) |
