"""Répondeur distant pour les benchs (API compatibles OpenAI) — jamais pour `src/`.

Le répondeur local (`qwen2.5:7b-instruct`) est faible : proche du hasard sur les
questions d'oubli de PersonaMem, fort biais de position, contexte tronqué au-delà
de 24 576 jetons. Ce client permet de rejouer les mêmes comparaisons avec des
modèles bien plus forts, offerts gratuitement (relevé du 30/09/2026) :

* ``nvidia`` — build.nvidia.com, 40 req/min. `nemotron-3-ultra-550b-a55b` et
  `nemotron-3-super-120b-a12b` répondent directement par une lettre avec
  ``enable_thinking=False`` ; sans ce drapeau, ils déversent leur raisonnement
  dans la réponse. Beaucoup de modèles du catalogue renvoient 404, 410 ou
  dépassent 2 minutes : seul le catalogue vivant (`/v1/models`) fait foi.
* ``mistral`` — La Plateforme, plan gratuit « Experiment ». Seuls les petits
  modèles y ont un quota (`ministral-14b-2512` : 30 req/min) ; Small, Medium et
  Magistral sont à quota zéro.

Uniquement pour des données publiques ou fabriquées : le gratuit de Mistral
entraîne ses modèles sur les requêtes, et NVIDIA ne publie pas sa politique.
Aucune donnée d'évaluation AML ne doit jamais passer par ici.

La clé est lue dans l'environnement ou dans `.env`, et n'est jamais affichée.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

PROVIDERS: dict[str, dict[str, Any]] = {
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "key_env": "NVIDIA_NIM_API_KEY",
        "min_interval_s": 60 / 36,   # 40 req/min annoncés, marge de 10 %
        "extra": {"chat_template_kwargs": {"enable_thinking": False}},
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "key_env": "MISTRAL_API_KEY",
        "min_interval_s": 60 / 27,   # 30 req/min annoncés, marge de 10 %
        "extra": {},
    },
}

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_REPRISES = 10          # une surcharge nocturne ne doit pas tuer un run de plusieurs heures
_ATTENTE_MAX_S = 180.0


def _lire_cle(nom: str) -> str:
    if os.environ.get(nom):
        return os.environ[nom].strip()
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for ligne in env.read_text(encoding="utf-8").splitlines():
            if ligne.startswith(f"{nom}="):
                return ligne.split("=", 1)[1].strip().strip("\"'")
    raise SystemExit(f"Clé {nom} introuvable dans l'environnement ni dans .env")


class RemoteChat:
    """Même interface que `OllamaClient.generate` pour ce dont les benchs ont besoin."""

    def __init__(self, provider: str) -> None:
        conf = PROVIDERS[provider]
        self.provider = provider
        self._conf = conf
        self._client = httpx.AsyncClient(
            base_url=conf["base_url"], timeout=180,
            headers={"Authorization": f"Bearer {_lire_cle(conf['key_env'])}"},
        )
        self._dernier = 0.0
        self.reprises = 0
        self.echecs = 0

    async def generate(self, prompt: str, model: str, options: dict[str, Any] | None = None,
                       **_: Any) -> str:
        options = options or {}
        corps = {
            "model": model,
            "temperature": options.get("temperature", 0.0),
            "max_tokens": options.get("num_predict", 16),
            "messages": [{"role": "user", "content": prompt}],
            **self._conf["extra"],
        }
        derniere_erreur = ""
        for essai in range(_REPRISES):
            attente = self._conf["min_interval_s"] - (time.monotonic() - self._dernier)
            if attente > 0:
                await asyncio.sleep(attente)
            self._dernier = time.monotonic()
            try:
                r = await self._client.post("/chat/completions", json=corps)
                if r.status_code == 200:
                    contenu = r.json()["choices"][0]["message"].get("content") or ""
                    return _THINK_RE.sub("", contenu).strip()
                derniere_erreur = f"HTTP {r.status_code} {r.text[:120]}"
                if r.status_code not in (408, 409, 425, 429, 500, 502, 503, 504):
                    break  # erreur définitive : inutile d'insister
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                derniere_erreur = type(exc).__name__
                # Nuit du 30/09 : après une coupure de NVIDIA, le client ouvert
                # depuis des heures n'a plus jamais réussi à se connecter, alors
                # qu'une connexion neuve passait. On repart donc d'un client neuf.
                await self._client.aclose()
                self._client = httpx.AsyncClient(
                    base_url=self._conf["base_url"], timeout=180,
                    headers={"Authorization": f"Bearer {_lire_cle(self._conf['key_env'])}"},
                )
            self.reprises += 1
            await asyncio.sleep(min(_ATTENTE_MAX_S, 5.0 * 2 ** essai))
        self.echecs += 1
        raise RuntimeError(f"{self.provider}/{model} : échec après {_REPRISES} essais ({derniere_erreur})")

    async def aclose(self) -> None:
        await self._client.aclose()
