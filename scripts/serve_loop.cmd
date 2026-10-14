@echo off
REM Boucle de service de la tache planifiee (voir scripts\register_task.ps1) :
REM relance `mnemos serve` des qu'il meurt. RestartCount du Planificateur ne
REM couvre que l'echec au demarrage de la tache, pas la mort du processus en
REM cours de route : le 14/10/2026, un Stop-Process sur le port 8765 a coupe
REM la prod 30 min sans aucune relance.
REM
REM   serve_loop.cmd [fichier_log] [adresse_ecoute]
REM
REM fichier_log    : relatif a la racine du depot ; vide = sortie jetee.
REM adresse_ecoute : IP:port de .env, par ex. 100.76.193.75:8765. Si un
REM   processus ecoute deja a cette adresse PRECISE, on attend sans rien lancer :
REM   l'application ouvre ses bases AVANT le bind, une seconde instance
REM   toucherait donc la prod a chaque tour avant d'echouer. L'adresse precise,
REM   pas le port seul : un serveur de test sur 127.0.0.1:8765 ne doit pas
REM   passer pour Mnemos.
REM
REM Arret volontaire : Disable-ScheduledTask PUIS Stop-ScheduledTask, puis
REM arreter Python (docs\AML_DEPLOYMENT_WINDOWS.md). Sinon il revient en 15 s.
REM
REM Pause par ping et non par `timeout` : sans console, `timeout` echoue
REM aussitot et la boucle tournerait a vide.

setlocal
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0.."

set "LOG=%~1"
if "%LOG%"=="" set "LOG=nul"
set "ADDR=%~2"
set "ATTENTE="

echo [serve_loop] %date% %time% boucle demarree (adresse %ADDR%) >> "%LOG%"

:boucle
if "%ADDR%"=="" goto lancer
netstat -ano -p tcp | findstr /c:" %ADDR% " | findstr /c:"LISTENING" >nul
if not errorlevel 1 goto attente

:lancer
set "ATTENTE="
echo [serve_loop] %date% %time% lancement de mnemos serve >> "%LOG%"
".venv\Scripts\mnemos.exe" serve >> "%LOG%" 2>&1
set "CODE=%errorlevel%"
echo [serve_loop] %date% %time% mnemos serve sorti, code %CODE%, relance dans 15 s >> "%LOG%"
ping -n 16 127.0.0.1 >nul
goto boucle

:attente
if not defined ATTENTE echo [serve_loop] %date% %time% %ADDR% deja en ecoute, attente >> "%LOG%"
set "ATTENTE=1"
ping -n 16 127.0.0.1 >nul
goto boucle
