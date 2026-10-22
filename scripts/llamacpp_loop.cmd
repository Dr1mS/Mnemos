@echo off
REM Boucle de service de la tache "Mnemos llama.cpp" (voir
REM scripts\register_llamacpp_task.ps1) : relance llama-server des qu'il meurt.
REM Meme defaut que serve_loop.cmd avant le 14/10/2026 : la tache lancait
REM l'exe en direct, et RestartCount ne couvre que l'echec au demarrage. Un
REM llama-server mort en cours de run faisait echouer chaque /add et /search.
REM
REM   llamacpp_loop.cmd log_boucle exe modele parallele contexte lot port log_llama
REM
REM log_boucle : journal de la boucle. Distinct du journal de llama-server, que
REM   le serveur garde ouvert.
REM log_llama  : --log-file de llama-server, qui l'ECRASE a chaque demarrage.
REM   Avant chaque lancement, l'ancien est copie dans <log_llama>.prev : sans
REM   ca, la relance effacerait la trace du plantage.
REM Chemins relatifs a la racine du depot, ou absolus.
REM
REM Si un processus ecoute deja sur 127.0.0.1:<port>, on attend sans rien
REM lancer : un second llama-server chargerait le modele en VRAM pour rien
REM avant d'echouer au bind.
REM
REM Arret volontaire : Disable-ScheduledTask PUIS Stop-ScheduledTask, puis
REM arreter llama-server (docs\AML_DEPLOYMENT_WINDOWS.md). Sinon il revient en 15 s.
REM
REM Pause par ping et non par `timeout` : sans console, `timeout` echoue
REM aussitot et la boucle tournerait a vide.

setlocal
cd /d "%~dp0.."

set "LOG=%~1"
if "%LOG%"=="" set "LOG=nul"
set "EXE=%~2"
set "MODELE=%~3"
set "PARALLELE=%~4"
set "CONTEXTE=%~5"
set "LOT=%~6"
set "PORT=%~7"
set "LLOG=%~8"
set "ADDR=127.0.0.1:%PORT%"
set "ATTENTE="

echo [llamacpp_loop] %date% %time% boucle demarree (adresse %ADDR%) >> "%LOG%"

:boucle
netstat -ano -p tcp | findstr /c:" %ADDR% " | findstr /c:"LISTENING" >nul
if not errorlevel 1 goto attente

set "ATTENTE="
if exist "%LLOG%" copy /y "%LLOG%" "%LLOG%.prev" >nul
echo [llamacpp_loop] %date% %time% lancement de llama-server >> "%LOG%"
"%EXE%" --model "%MODELE%" --host 127.0.0.1 --port %PORT% -c %CONTEXTE% -np %PARALLELE% --embedding -b %LOT% -ub %LOT% --no-webui --offline --log-file "%LLOG%" --no-log-prefix --no-log-timestamps >nul 2>&1
set "CODE=%errorlevel%"
echo [llamacpp_loop] %date% %time% llama-server sorti, code %CODE%, relance dans 15 s >> "%LOG%"
ping -n 16 127.0.0.1 >nul
goto boucle

:attente
if not defined ATTENTE echo [llamacpp_loop] %date% %time% %ADDR% deja en ecoute, attente >> "%LOG%"
set "ATTENTE=1"
ping -n 16 127.0.0.1 >nul
goto boucle
