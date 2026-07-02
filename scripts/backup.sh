#!/usr/bin/env bash
# Backup atomique via VACUUM INTO (anti-pattern 9).
set -euo pipefail
cd "$(dirname "$0")/.."
mnemos backup --out "${1:-./backups}"
