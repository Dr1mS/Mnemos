"""Auto-test du contrat AML contre un déploiement Mnemos (local ou public).

Usage :
    .venv\\Scripts\\python.exe scripts\\aml_selftest.py `
        --url https://mnemos.dr1ms.fr --key <API_KEY>

Vérifie /health (sans auth), /add (success=true, echo exact des identifiants),
/search (data en liste, <= top_k, id et content non vides, souvenir retrouvé)
et le refus d'une clé invalide (401). Les écritures vont dans un user_id
dédié et unique par exécution : elles n'interfèrent avec aucune autre mémoire.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from typing import Any

import httpx

AUTH_SCHEMES = ("bearer", "token", "x-api-key")


def auth_headers(scheme: str, key: str) -> dict[str, str]:
    if scheme == "x-api-key":
        return {"X-API-Key": key}
    return {"Authorization": f"{scheme.capitalize()} {key}"}


class SelfTest:
    def __init__(self) -> None:
        self.failures = 0

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        print(f"  {'✓' if ok else '✗'} {label}{f' — {detail}' if detail else ''}")
        if not ok:
            self.failures += 1


def post(
    client: httpx.Client, path: str, payload: dict[str, Any], headers: dict[str, str]
) -> tuple[httpx.Response, float]:
    t0 = time.perf_counter()
    resp = client.post(path, json=payload, headers=headers)
    return resp, (time.perf_counter() - t0) * 1000


def run(url: str, key: str | None, scheme: str) -> int:
    t = SelfTest()
    run_id = uuid.uuid4().hex[:12]
    user_id = f"selftest:{run_id}"
    headers = auth_headers(scheme, key) if key else {}

    with httpx.Client(base_url=url.rstrip("/"), timeout=60.0) as client:
        print(f"Cible : {url} (auth : {scheme if key else 'aucune'})\n")

        print("[1] GET /health (sans authentification)")
        resp = client.get("/health")
        t.check(200 <= resp.status_code < 300, "réponse 2xx", f"HTTP {resp.status_code}")

        print("[2] POST /add")
        add = {
            "request_id": f"{user_id}:chunk-0",
            "user_id": user_id,
            "session_id": f"{user_id}:session-0",
            "messages": [
                {"role": "user", "timestamp": 1704067200000,
                 "content": "Ma couleur préférée est le vert émeraude."},
                {"role": "assistant", "timestamp": 1704067215000,
                 "content": "C'est noté, le vert émeraude."},
            ],
        }
        resp, lat = post(client, "/add", add, headers)
        body = resp.json() if resp.status_code == 200 else {}
        t.check(resp.status_code == 200, "HTTP 200", f"HTTP {resp.status_code} en {lat:.0f} ms")
        t.check(body.get("success") is True, "success == true")
        for field in ("request_id", "user_id", "session_id"):
            t.check(body.get(field) == add[field], f"{field} renvoyé à l'identique")

        top_k = 5
        print(f"[3] POST /search (top_k={top_k})")
        search = {"query": "Quelle est ma couleur préférée ?", "user_id": user_id, "top_k": top_k}
        resp, lat = post(client, "/search", search, headers)
        data = resp.json().get("data") if resp.status_code == 200 else None
        t.check(resp.status_code == 200, "HTTP 200", f"HTTP {resp.status_code} en {lat:.0f} ms")
        t.check(isinstance(data, list), "champ data présent et de type liste")
        items = data if isinstance(data, list) else []
        t.check(len(items) <= top_k, "au plus top_k résultats", f"{len(items)} résultat(s)")
        t.check(
            all(it.get("id") and str(it.get("content", "")).strip() for it in items),
            "chaque item a un id et un content non vides",
        )
        t.check(any("émeraude" in str(it.get("content", "")) for it in items), "souvenir retrouvé")

        print("[4] Isolation et authentification")
        resp, _ = post(client, "/search", {**search, "user_id": f"{user_id}:autre"}, headers)
        other = resp.json().get("data", []) if resp.status_code == 200 else None
        t.check(other == [], "aucun résultat pour un autre user_id")
        if key:
            resp, _ = post(client, "/search", search, auth_headers(scheme, key + "-invalide"))
            t.check(
                resp.status_code == 401, "clé invalide refusée (401)", f"HTTP {resp.status_code}"
            )

    if t.failures:
        print(f"\n❌ {t.failures} vérification(s) en échec")
    else:
        print("\n✅ Contrat respecté")
    return 1 if t.failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-test du contrat AML d'un déploiement")
    parser.add_argument("--url", required=True, help="Base URL, ex. https://mnemos.dr1ms.fr")
    parser.add_argument("--key", default=None, help="API_KEY du serveur (Memory System Key)")
    parser.add_argument(
        "--auth", choices=AUTH_SCHEMES, default="bearer", help="Schéma d'authentification"
    )
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # console Windows cp1252
    sys.exit(run(args.url, args.key, args.auth))


if __name__ == "__main__":
    main()
