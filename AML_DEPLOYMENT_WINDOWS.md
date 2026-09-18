# 🚀 Guide de Déploiement Windows (RTX 4070 Ti) & Cloudflare Tunnel
## Agent Memory Challenge (Cycle 2) — Agent Memory Leaderboard (AML)

Ce guide pas-à-pas explique comment lancer **Mnemos** sur votre machine Windows avec votre GPU (RTX 4070 Ti) et l'exposer publiquement en HTTPS pour la compétition AML via **Cloudflare Tunnel**.

---

## 📋 Table des Matières
1. [Pré-requis](#1-pré-requis)
2. [Étape 1 : Préparation d'Ollama & VRAM GPU](#étape-1--préparation-dollama--vram-gpu)
3. [Étape 2 : Préparation du projet Mnemos](#étape-2--préparation-du-projet-mnemos)
4. [Étape 3 : Installation de Cloudflare Tunnel (cloudflared)](#étape-3--installation-de-cloudflare-tunnel-cloudflared)
5. [Étape 4 : Lancement du serveur Mnemos](#étape-4--lancement-du-serveur-mnemos)
6. [Étape 5 : Ouverture du Tunnel HTTPS Cloudflare](#étape-5--ouverture-du-tunnel-https-cloudflare)
7. [Étape 6 : Test de validation (Self-Test)](#étape-6--test-de-validation-self-test)
8. [Étape 7 : Soumission à l'Arena AML](#étape-7--soumission-à-larena-aml)
9. [Dépannage & FAQ](#dépannage--faq)

---

## 1. Pré-requis

* **OS** : Windows 10/11 (64-bit).
* **GPU** : NVIDIA GeForce RTX 4070 Ti avec pilotes NVIDIA récents.
* **Outils** :
  * [Ollama pour Windows](https://ollama.com/download/windows)
  * [Python 3.12](https://www.python.org/downloads/) ou [uv](https://github.com/astral-sh/uv)
  * [Git pour Windows](https://gitforwindows.org/)
  * [Cloudflare Tunnel CLI (`cloudflared`)](https://github.com/cloudflare/cloudflared/releases)

---

## Étape 1 : Préparation d'Ollama & VRAM GPU

### 1.1 Télécharger les modèles nécessaires
Ouvrez un terminal **PowerShell** et téléchargez les deux modèles requis :
```powershell
ollama pull bge-m3
ollama pull qwen2.5:3b
```

### 1.2 ⚠️ Configuration critique : Empêcher le déchargement de VRAM
Par défaut sous Windows, Ollama décharge les modèles de la VRAM au bout de 5 minutes d'inactivité. Pour éviter un temps de rechargement à froid (2 à 10 secondes) lors de l'évaluation de l'Arena :

#### Option A — Via les variables d'environnement système Windows (Recommandé) :
1. Tapez `Touches Win + R`, saisissez `sysdm.cpl` puis Entrée.
2. Onglet **Paramètres système avancés** > **Variables d'environnement...**
3. Sous *Variables système*, cliquez sur **Nouvelle...** :
   * Nom : `OLLAMA_KEEP_ALIVE`
   * Valeur : `-1`
4. Redémarrez l'application Ollama depuis la barre des tâches.

#### Option B — En PowerShell avant de lancer Ollama :
```powershell
$env:OLLAMA_KEEP_ALIVE = "-1"
ollama serve
```

---

## Étape 2 : Préparation du projet Mnemos

1. Ouvrez un terminal **PowerShell** dans le dossier de Mnemos.
2. Basculez sur la branche de compétition :
   ```powershell
   git checkout feat/aml-adapter
   git pull origin feat/aml-adapter
   ```
3. Activez votre environnement virtuel Python :
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   *(Si un message bloque l'exécution de scripts : `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

4. Créez ou éditez le fichier `.env` à la racine du projet :
   ```env
   # Clé d'authentification secrète fournie à l'Arena AML
   API_KEY=votre_cle_secrete_aml_2026

   # Configuration Réseau
   API_HOST=127.0.0.1
   API_PORT=8765
   LOG_LEVEL=INFO

   # Modèles Ollama
   OLLAMA_HOST=http://localhost:11434
   EMBED_MODEL=bge-m3
   SALIENCE_MODEL=qwen2.5:3b
   EXTRACTION_MODEL=qwen2.5:3b
   LLM_THINK=false

   # Consolidation Cognitive en continu (Essentiel pour l'Arena AML)
   CONSOLIDATION_AUTO=true
   CONSOLIDATION_INTERVAL_SECONDS=5.0
   CONSOLIDATION_DELAY_HOURS=0.0
   ```

---

## Étape 3 : Installation de Cloudflare Tunnel (`cloudflared`)

### Méthode 1 : Via WinGet (Le plus simple)
Dans PowerShell :
```powershell
winget install --id Cloudflare.cloudflared
```
*Fermez et rouvrez PowerShell après l'installation pour que la commande soit dans votre PATH.*

### Méthode 2 : Téléchargement direct de l'exécutable
1. Téléchargez [`cloudflared-windows-amd64.exe`](https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe).
2. Renommez-le en `cloudflared.exe`.
3. Placez-le dans un dossier de votre PATH (ex: `C:\Windows\System32` ou dans le dossier de Mnemos).

Vérifiez l'installation :
```powershell
cloudflared --version
```

---

## Étape 4 : Lancement du serveur Mnemos

Dans votre premier terminal PowerShell (avec le venv activé) :
```powershell
.\.venv\Scripts\mnemos.exe serve
```
Ou via Uvicorn directement :
```powershell
uvicorn mnemos.server:create_app --factory --host 127.0.0.1 --port 8765
```

Vous devez voir :
```text
INFO:     Started server process [...]
INFO:     Waiting for application startup.
INFO:     server_started host='127.0.0.1' port=8765
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8765 (Press CTRL+C to quit)
```

---

## Étape 5 : Ouverture du Tunnel HTTPS Cloudflare

Ouvrez un **deuxième terminal PowerShell** et lancez :
```powershell
cloudflared tunnel --url http://127.0.0.1:8765
```

Au bout de quelques secondes, repérez la ligne encadrée dans la console :
```text
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
|  https://nom-aleatoire-genere.trycloudflare.com                                            |
+--------------------------------------------------------------------------------------------+
```

> ⚠️ **IMPORTANT** : Laissez ce terminal **ouvert**. Tant que `cloudflared` tourne, votre tunnel reste actif avec son URL HTTPS sécurisée.

---

## Étape 6 : Test de validation (Self-Test)

Ouvrez un **troisième terminal** (ou utilisez curl.exe) pour vérifier le bon fonctionnement à travers l'URL publique Cloudflare.

Remplacez `https://VOTRE-URL.trycloudflare.com` et `votre_cle_secrete_aml_2026` par vos valeurs réelles :

### 6.1 Test Sonde de santé (`GET /health`)
```powershell
curl.exe -s https://VOTRE-URL.trycloudflare.com/health
```
**Réponse attendue (sans authentification) :**
```json
{"status":"healthy","version":"0.1.0"}
```

### 6.2 Test Écriture mémoire (`POST /add`)
```powershell
curl.exe -s -X POST https://VOTRE-URL.trycloudflare.com/add `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer votre_cle_secrete_aml_2026" `
  -d '{
    "request_id": "test_win_01",
    "user_id": "user_win_test",
    "session_id": "sess_win_test",
    "messages": [{
      "role": "user",
      "content": "J aime la programmation et j utilise une RTX 4070 Ti.",
      "timestamp": 1704067200000
    }]
  }'
```
**Réponse attendue :**
```json
{"success":true,"request_id":"test_win_01","user_id":"user_win_test","session_id":"sess_win_test"}
```

### 6.3 Test Recherche hybride (`POST /search`)
```powershell
curl.exe -s -X POST https://VOTRE-URL.trycloudflare.com/search `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer votre_cle_secrete_aml_2026" `
  -d '{
    "query": "Quelle carte graphique est utilisee ?",
    "user_id": "user_win_test",
    "top_k": 5
  }'
```
**Réponse attendue :**
```json
{
  "data": [
    {
      "id": "ep_...",
      "content": "J aime la programmation et j utilise une RTX 4070 Ti.",
      "score": 0.65,
      "created_at": "2024-01-01T00:00:00Z"
    }
  ]
}
```

---

## Étape 7 : Soumission à l'Arena AML

Dans l'interface de soumission du concours **Agent Memory Leaderboard (Cycle 2)** :
* **Track** : `Textual Memory`
* **Division** : `Open-source Methods`
* **API Endpoint URL** : `https://VOTRE-URL.trycloudflare.com` *(l'URL Cloudflare)*
* **API Key / Token** : `votre_cle_secrete_aml_2026` *(la valeur définie dans votre `.env`)*
* **Architecture / Framework** : `Mnemos (Hybrid Episodic-Semantic Memory)`

---

## Dépannage & FAQ

| Problème | Cause | Solution |
|---|---|---|
| `curl: (7) Failed to connect` en local | Le serveur Mnemos n'est pas démarré | Vérifiez le terminal 1 (`mnemos serve`). |
| `cloudflared` affiche une erreur de port | Mauvais port renseigné | Vérifiez que vous avez bien mis `http://127.0.0.1:8765`. |
| Erreur 401 Unauthorized | Mauvaise clé d'API | Vérifiez le header `Authorization: Bearer <votre_clé>` et le fichier `.env`. |
| Latence élevée au début | Modèle froid / rechargement VRAM | Vérifiez que `OLLAMA_KEEP_ALIVE=-1` est bien configuré. |
| Crash GPU / OOM | Mémoire GPU saturée par d'autres logiciels | Fermez les jeux ou applications consommatrices de VRAM sur la RTX 4070 Ti. |
