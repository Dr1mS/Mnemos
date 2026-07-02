# Supprime les DBs (episodic + semantic) et rejoue les migrations.
Set-Location (Join-Path $PSScriptRoot "..")

Remove-Item -Force -ErrorAction SilentlyContinue data/episodic.db*, data/semantic.db*
alembic -x db=episodic upgrade head
alembic -x db=semantic upgrade head
Write-Host "DBs reset."
