# Enregistre "Mnemos Serve" dans le Planificateur de tâches Windows :
# lance `mnemos serve` à l'ouverture de session, relance auto si le process meurt.
# À exécuter une fois (pas besoin de droits admin, tâche scope utilisateur courant).
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $repoRoot "scripts\serve.ps1"

# Appel direct de l'exe mnemos plutôt qu'un wrapper .ps1 : évite tout besoin
# de policy d'exécution PowerShell sur une tâche planifiée persistante.
$action = New-ScheduledTaskAction -Execute (Join-Path $repoRoot ".venv\Scripts\mnemos.exe") `
    -Argument "serve" -WorkingDirectory $repoRoot

$trigger = New-ScheduledTaskTrigger -AtLogOn

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0)  # illimité

Register-ScheduledTask -TaskName "Mnemos Serve" `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Serveur Mnemos (API + MCP + constellation /viz) sur le port 8765" `
    -Force

Write-Host "Tâche 'Mnemos Serve' enregistrée. Démarrage : Start-ScheduledTask -TaskName 'Mnemos Serve'"
