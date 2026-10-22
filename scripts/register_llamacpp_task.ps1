# Enregistre "Mnemos llama.cpp" dans le Planificateur de tâches Windows : lance
# `llama-server` (embeddings bge-m3) à l'ouverture de session et le relance s'il
# meurt. À exécuter une fois, sans droits admin.
#
#   .\scripts\register_llamacpp_task.ps1
#   Start-ScheduledTask -TaskName "Mnemos llama.cpp"
#
# Deux niveaux de relance, comme pour "Mnemos AML" (register_task.ps1) : avant le
# 22/10/2026, la tâche lançait l'exe en direct, et RestartCount ne relance pas un
# processus mort en cours de route. llama-server tombé, chaque /add et /search échoue.
# - scripts\llamacpp_loop.cmd relance llama-server 15 s après sa mort ;
# - un déclencheur répété toutes les $RelanceMinutes min relance la boucle
#   elle-même si tout l'arbre a été tué (IgnoreNew : sans effet tant qu'elle tourne).
# Conséquence : pour arrêter llama-server, Disable-ScheduledTask d'abord
# (docs\AML_DEPLOYMENT_WINDOWS.md, étape 4.3).
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
    # Chemins relatifs au dépôt, ou absolus. llama-server écrase LogFile à chaque
    # démarrage ; la boucle garde le précédent dans <LogFile>.prev.
    [string]$LogFile = "data\aml\llamacpp.log",
    [string]$LoopLogFile = "data\aml\llamacpp_loop.log",
    [int]$RelanceMinutes = 5
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

function Resolve-RepoPath([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { $Path } else { Join-Path $repoRoot $Path }
}
$logPath = Resolve-RepoPath $LogFile
$loopLogPath = Resolve-RepoPath $LoopLogFile
foreach ($p in $logPath, $loopLogPath) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $p) | Out-Null
}

# Les options de llama-server sont écrites dans llamacpp_loop.cmd ; seules les
# valeurs passent ici. /s + paire de guillemets englobante : sans elle, `cmd /c`
# ampute la première et la dernière paire quand la ligne commence par un
# guillemet, ce qui cassait le chemin du modèle (AML_COMPETITION_PREP.md §12).
$loop = Join-Path $repoRoot "scripts\llamacpp_loop.cmd"
$loopArgs = @(
    "`"$loop`"", "`"$loopLogPath`"", "`"$exe`"", "`"$blob`"",
    $Parallel, $ContextSize, $BatchSize, $Port, "`"$logPath`""
) -join " "
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/s /c `"$loopArgs`"" `
    -WorkingDirectory $repoRoot

$atLogOn = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
# Répétition sur un déclencheur -Once qui part maintenant : posée sur AtLogOn,
# elle n'agirait qu'à la prochaine ouverture de session (cf. register_task.ps1).
$filet = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $RelanceMinutes)
$filet.Repetition.StopAtDurationEnd = $false  # ne jamais arrêter la boucle en cours

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew `
    -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0)  # illimité

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger @($atLogOn, $filet) -Settings $settings `
    -Description "llama-server : embeddings bge-m3 pour Mnemos ($Parallel slots, port $Port)" `
    -Force -ErrorAction Stop | Out-Null

Write-Host "Tache '$TaskName' enregistree ($Parallel slots, $($ContextSize / $Parallel) de contexte par slot, filet toutes les $RelanceMinutes min)."
Write-Host "Demarrage : Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "Verification : (Invoke-WebRequest http://127.0.0.1:$Port/health -UseBasicParsing).Content"
