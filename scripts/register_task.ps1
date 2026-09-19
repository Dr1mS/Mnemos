# Enregistre "Mnemos Serve" dans le Planificateur de tâches Windows :
# lance `mnemos serve` à l'ouverture de session, relance auto si le process meurt.
# À exécuter une fois (pas besoin de droits admin, tâche scope utilisateur courant).
#
# Déploiement AML : .\scripts\register_task.ps1 -TaskName "Mnemos AML" -LogFile "data\aml\serve.log"
param(
    [string]$TaskName = "Mnemos Serve",
    # Chemin relatif au dépôt. Vide = sortie du serveur non conservée.
    [string]$LogFile = ""
)
$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$mnemosExe = Join-Path $repoRoot ".venv\Scripts\mnemos.exe"

# Appel direct de l'exe mnemos plutôt qu'un wrapper .ps1 : évite tout besoin
# de policy d'exécution PowerShell sur une tâche planifiée persistante.
if ($LogFile) {
    # cmd.exe pour la redirection ; PYTHONIOENCODING évite l'échec d'écriture des
    # logs accentués dans un fichier sous la page de code ANSI de Windows.
    $action = New-ScheduledTaskAction -Execute "cmd.exe" `
        -Argument "/c set `"PYTHONIOENCODING=utf-8`" && `"$mnemosExe`" serve >> `"$LogFile`" 2>&1" `
        -WorkingDirectory $repoRoot
} else {
    $action = New-ScheduledTaskAction -Execute $mnemosExe -Argument "serve" -WorkingDirectory $repoRoot
}

# -User : sans lui, le déclencheur vaut pour toute ouverture de session et exige les droits admin.
$trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0)  # illimité

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger $trigger -Settings $settings `
    -Description "Serveur Mnemos (API + MCP + constellation /viz), configuré par .env" `
    -Force -ErrorAction Stop | Out-Null

Write-Host "Tâche '$TaskName' enregistrée. Démarrage : Start-ScheduledTask -TaskName '$TaskName'"
