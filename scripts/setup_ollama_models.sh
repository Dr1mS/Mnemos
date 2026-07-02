#!/usr/bin/env bash
# Profil Linux — modèles validés par le POC (poc/RESULTS.md)
set -euo pipefail

ollama pull bge-m3
ollama pull qwen3:4b
# ollama pull qwen3:8b   # optionnel — profil GPU uniquement (extraction)
ollama list
