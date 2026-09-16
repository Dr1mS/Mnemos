# Lance `mnemos serve` en tâche de fond, log vers data/adrien/serve.log.
# Utilisé par la tâche planifiée Windows "Mnemos Serve" (voir scripts/register_task.ps1).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$env:PYTHONIOENCODING = "utf-8"
$logDir = ".\data\adrien"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

& ".\.venv\Scripts\mnemos.exe" serve *>> "$logDir\serve.log"
