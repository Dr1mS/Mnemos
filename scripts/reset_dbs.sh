#!/usr/bin/env bash
# Supprime les DBs (episodic + semantic) et rejoue les migrations.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -f data/episodic.db* data/semantic.db*
alembic -x db=episodic upgrade head
alembic -x db=semantic upgrade head
echo "DBs reset."
