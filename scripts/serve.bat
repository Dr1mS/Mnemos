@echo off
REM Lance le serveur Mnemos (mnemos serve) au premier plan, sortie visible.
REM Ctrl+C pour arreter proprement. Pour un lancement en tache de fond avec log,
REM utiliser scripts\serve.ps1 a la place.

setlocal
set "PYTHONIOENCODING=utf-8"

REM Racine du repo = dossier parent de ce script (scripts\..).
cd /d "%~dp0.."

if not exist ".venv\Scripts\mnemos.exe" (
  echo [ERREUR] .venv\Scripts\mnemos.exe introuvable.
  echo Cree l'environnement virtuel et installe mnemos avant de lancer.
  exit /b 1
)

echo Demarrage de Mnemos serve ^(API sur http://127.0.0.1:8765^)...
echo Ctrl+C pour arreter.
echo.

".venv\Scripts\mnemos.exe" serve
