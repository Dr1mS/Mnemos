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
    # ContextSize / Parallel ; 65536/16 = 4096, soit le contexte plein de bge-m3.
    [int]$Parallel = 16,
    [int]$ContextSize = 65536,
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

$serverArgs = @(
    "--model `"$blob`"",
    "--host 127.0.0.1", "--port $Port",
    "-c $ContextSize", "-np $Parallel",
    "--embedding",              # sert /v1/embeddings, pas de génération
    "-b 2048", "-ub 2048",
    "--no-webui", "--offline",
    "--no-log-prefix", "--no-log-timestamps"
) -join " "

# cmd.exe pour la redirection du log, comme register_task.ps1.
$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c `"$exe`" $serverArgs >> `"$LogFile`" 2>&1" `
    -WorkingDirectory $repoRoot

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
