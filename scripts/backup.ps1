# Backup atomique via VACUUM INTO (anti-pattern 9).
param([string]$OutDir = "./backups")
Set-Location (Join-Path $PSScriptRoot "..")
mnemos backup --out $OutDir
