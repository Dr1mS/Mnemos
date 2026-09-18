# Lance Cloudflare Tunnel vers Mnemos (port 8765).
# Affiche l'URL publique HTTPS pour l'Agent Memory Challenge (Cycle 2).

$ErrorActionPreference = "Continue"

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "   Mnemos - Ouverture du Tunnel HTTPS Cloudflare     " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$cloudflaredCmd = Get-Command cloudflared -ErrorAction SilentlyContinue

if (-not $cloudflaredCmd) {
    Write-Host "[ERREUR] 'cloudflared' n'est pas installe ou pas dans le PATH." -ForegroundColor Red
    Write-Host "Pour l'installer rapidement :" -ForegroundColor Yellow
    Write-Host "  winget install --id Cloudflare.cloudflared" -ForegroundColor White
    Write-Host "Puis relancez ce script." -ForegroundColor Yellow
    Write-Host ""
    Pause
    exit 1
}

Write-Host "[OK] cloudflared detecte : $($cloudflaredCmd.Source)" -ForegroundColor Green
Write-Host "[INFO] Connexion au serveur local http://127.0.0.1:8765..." -ForegroundColor Yellow
Write-Host "[INFO] L'URL publique HTTPS apparaitra ci-dessous dans quelques secondes :" -ForegroundColor Yellow
Write-Host ""

& cloudflared tunnel --url http://127.0.0.1:8765
