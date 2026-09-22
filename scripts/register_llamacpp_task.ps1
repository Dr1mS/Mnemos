# Enregistre "Mnemos llama.cpp" dans le Planificateur de tâches Windows : lance
# `llama-server` (embeddings bge-m3) à l'ouverture de session, relance auto si
# le process meurt. À exécuter une fois, sans droits admin.
#
#   .\scripts\register_llamacpp_task.ps1
#   Start-ScheduledTask -TaskName "Mnemos llama.cpp"
#
# Pourquoi ce serveur plutôt qu'Ollama pour les embeddings (mesuré le 21/09) :
# Ollama force -np 1 sur les modèles d'embedding et interpose un saut
# parent -> runner ; sous concurrence il répond HTTP 400 pendant la montée d'un
# runner. La plateforme AML impose Add et Search a une concurrence >= 16, ou
# cette pile echouait deja a 2 % des la concurrence 1. En direct : 16 slots,
# 51 msg/s en Add et 48 req/s en Search, zero erreur, vecteurs identiques
# (cosinus 0,999999650) donc aucun bench invalide.
param(
    [string]$TaskName = "Mnemos llama.cpp",
    # 16 slots = la concurrence minimale imposée par AML. Chaque slot reçoit
    # ContextSize / Parallel ; 131072/16 = 8192, le contexte maximal de bge-m3.
    [int]$Parallel = 16,
    [int]$ContextSize = 131072,
    # Taille du lot PHYSIQUE. Pour un embedding, llama.cpp exige que toute
    # l'entrée y tienne : a 2048, un message de 2 737 tokens des donnees
    # d'evaluation faisait echouer /v1/embeddings en 500 (« input is too large
    # to process, increase the physical batch size »), et le smoke du 22/09 a
    # bloque a 90 %. A 8192, toute entree jusqu'au maximum de bge-m3 passe.
    [int]$BatchSize = 8192,
    [int]$Port = 8899,
    [string]$LogFile = "data\aml\llamacpp.log"
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

# llama-server est livré avec Ollama : rien à installer.
$exe = Join-Path $env:LOCALAPPDATA "Programs\Ollama\lib\ollama\llama-server.exe"
if (-not (Test-Path $exe)) {
    throw "llama-server introuvable : $exe. Installer Ollama, ou adapter ce chemin."
}

# Résout le blob GGUF de bge-m3 via le manifeste Ollama : pas de sha256 codé en
# dur, qui casserait au prochain `ollama pull`.
$manifest = Join-Path $env:USERPROFILE ".ollama\models\manifests\registry.ollama.ai\library\bge-m3\latest"
if (-not (Test-Path $manifest)) {
    throw "Manifeste bge-m3 introuvable : $manifest. Lancer d'abord : ollama pull bge-m3"
}
$digest = (Get-Content $manifest -Raw | ConvertFrom-Json).layers |
    Where-Object { $_.mediaType -like "*.model" } |
    Select-Object -First 1 -ExpandProperty digest
if (-not $digest) { throw "Aucune couche de modèle dans le manifeste bge-m3." }
$blob = Join-Path $env:USERPROFILE ".ollama\models\blobs\$($digest -replace ':','-')"
if (-not (Test-Path $blob)) { throw "Blob du modèle introuvable : $blob" }

# Appel direct de l'exe, sans cmd.exe : `cmd /c` ampute la première et la
# dernière paire de guillemets quand la ligne commence par un guillemet, ce qui
# casse le chemin du modèle. llama-server sait écrire son log lui-même
# (--log-file), donc aucune redirection n'est nécessaire.
$logPath = Join-Path $repoRoot $LogFile
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $logPath) | Out-Null

$serverArgs = @(
    "--model `"$blob`"",
    "--host 127.0.0.1", "--port $Port",
    "-c $ContextSize", "-np $Parallel",
    "--embedding",              # sert /v1/embeddings, pas de génération
    "-b $BatchSize", "-ub $BatchSize",
    "--no-webui", "--offline",
    "--log-file `"$logPath`"",
    "--no-log-prefix", "--no-log-timestamps"
) -join " "

$action = New-ScheduledTaskAction -Execute $exe -Argument $serverArgs -WorkingDirectory $repoRoot

$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0)  # illimité

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "llama-server : embeddings bge-m3 pour Mnemos ($Parallel slots, port $Port)" `
    -Force -ErrorAction Stop | Out-Null

Write-Host "Tache '$TaskName' enregistree ($Parallel slots, $($ContextSize / $Parallel) de contexte par slot)."
Write-Host "Demarrage : Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Verification : (Invoke-WebRequest http://127.0.0.1:$Port/health -UseBasicParsing).Content"
