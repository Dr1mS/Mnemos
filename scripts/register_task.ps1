# Enregistre "Mnemos Serve" dans le Planificateur de tâches Windows :
# lance `mnemos serve` à l'ouverture de session et le relance s'il meurt.
# À exécuter une fois (pas besoin de droits admin, tâche scope utilisateur courant).
#
# Déploiement AML : .\scripts\register_task.ps1 -TaskName "Mnemos AML" -LogFile "data\aml\serve.log"
#
# Deux niveaux de relance, car RestartCount ne couvre que l'échec au démarrage
# de la tâche, pas la mort du processus en cours de route (14/10/2026 : un
# Stop-Process sur le port 8765 a coupé la prod 30 min, tâche restée « Prête ») :
# - scripts\serve_loop.cmd relance `mnemos serve` 15 s après sa mort ;
# - un déclencheur répété toutes les $RelanceMinutes min relance la boucle
#   elle-même si tout l'arbre a été tué (IgnoreNew : sans effet tant qu'elle tourne).
# Conséquence : pour arrêter le serveur, Disable-ScheduledTask d'abord
# (docs\AML_DEPLOYMENT_WINDOWS.md, étape 4.3).
param(
    [string]$TaskName = "Mnemos Serve",
    # Chemin relatif au dépôt. Vide = sortie du serveur non conservée.
    [string]$LogFile = "",
    [int]$RelanceMinutes = 5
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$loop = Join-Path $repoRoot "scripts\serve_loop.cmd"

# Adresse d'écoute exacte, lue dans .env comme le fait le serveur (défauts de
# config.py sinon). La boucle s'en sert pour ne jamais lancer une seconde
# instance contre la prod : l'application ouvre ses bases avant le bind.
$apiHost = "127.0.0.1"; $apiPort = "8765"
$envFile = Join-Path $repoRoot ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^API_HOST=(.+)$') { $apiHost = $Matches[1].Trim() }
        if ($line -match '^API_PORT=(.+)$') { $apiPort = $Matches[1].Trim() }
    }
}
$listenAddr = "${apiHost}:${apiPort}"

# cmd.exe plutôt qu'un wrapper .ps1 : aucune policy d'exécution PowerShell à
# satisfaire. /s + paire de guillemets englobante : sans elle, `cmd /c` ampute
# la première et la dernière paire quand la ligne commence par un guillemet
# (piège déjà rencontré avec llama-server, AML_COMPETITION_PREP.md §12).
$action = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/s /c `"`"$loop`" `"$LogFile`" `"$listenAddr`"`"" `
    -WorkingDirectory $repoRoot

# -User : sans lui, le déclencheur vaut pour toute ouverture de session et exige les droits admin.
$atLogOn = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
# Répétition sur un déclencheur -Once qui part maintenant : posée sur AtLogOn,
# elle n'agirait qu'à la prochaine ouverture de session. Sans -RepetitionDuration,
# elle est indéfinie.
$filet = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $RelanceMinutes)
$filet.Repetition.StopAtDurationEnd = $false  # ne jamais arrêter la boucle en cours

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew `
    -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0)  # illimité

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger @($atLogOn, $filet) -Settings $settings `
    -Description "Serveur Mnemos (API + MCP + constellation /viz), configuré par .env, relancé s'il meurt" `
    -Force -ErrorAction Stop | Out-Null

Write-Host "Tâche '$TaskName' enregistrée (écoute $listenAddr, filet toutes les $RelanceMinutes min)."
Write-Host "Démarrage : Start-ScheduledTask -TaskName '$TaskName'"
