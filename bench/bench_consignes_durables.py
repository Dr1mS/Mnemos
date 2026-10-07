"""Consignes durables (« Always … when I ask about … ») : détecteur, et sélection.

Carte BEAM 100K (07/10/2026, `bench_beam_preuves.py`) : pour les questions « instruction
following », toutes les preuves ne sont dans le top 100 de `/search` que 13 fois sur 40. Les
messages manqués sont des consignes durables de l'utilisateur (« Always format all code
snippets with syntax highlighting when I ask about implementation details. ») que la
question suivante (« show me how to implement a login feature ») ne rappelle pas.

Phase `detecteur`, sans modèle de langage :
  * rappel sur BEAM 100K : les 40 consignes testées sont-elles détectées ?
  * déclenchements ailleurs : LoCoMo, LongMemEval-S, PersonaMem (historiques de la carte).
    La leçon de l'oubli (`bench_oubli_*`) : un ajout sans rapport coûte, et le net total
    peut s'annuler. Il faut donc un détecteur qui ne se déclenche presque jamais à tort.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.bench_beam_preuves import charger as charger_beam
from bench.bench_beam_preuves import ids_preuves

MAX_MOTS = 40
_DEBUT = re.compile(
    r"^\s*(?:please\s+)?(?:(?P<toujours>always|never)\b|(?P<desormais>from now on|going forward)\b)",
    re.IGNORECASE)
_CONDITION = re.compile(r"\b(?:when(?:ever)?|if|each time|every time)\s+(?:i|we)\b", re.IGNORECASE)


def est_consigne_durable(contenu: str, role: str) -> bool:
    """Message utilisateur court qui fixe une règle pour la suite : « Always/Never … when I … »,
    ou « From now on / Going forward … »."""
    if role != "user":
        return False
    texte = contenu.strip()
    if not texte or len(texte.split()) > MAX_MOTS:
        return False
    m = _DEBUT.match(texte)
    if m is None:
        return False
    return bool(m.group("desormais")) or bool(_CONDITION.search(texte))


def beam() -> dict[str, Any]:
    testees: set[str] = set()
    detectees: set[str] = set()
    par_conv = []
    for n in range(1, 21):
        sessions, qs = charger_beam(n)
        par_id = {int(m["id"]): m for s in sessions for m in s}
        for q in qs["instruction_following"]:
            for i in ids_preuves(q["source_chat_ids"]):
                m = par_id[i]
                if m["role"] == "user" and re.match(r"\s*always\b", m["content"], re.IGNORECASE):
                    testees.add(f"{n}:{i}")
        d = [f"{n}:{int(m['id'])}" for s in sessions for m in s if est_consigne_durable(str(m["content"]), m["role"])]
        detectees.update(d)
        par_conv.append(len(d))
    return {"consignes_testees": len(testees), "detectees_parmi_testees": len(testees & detectees),
            "detections_totales": len(detectees), "detections_hors_testees": len(detectees - testees),
            "detections_par_conversation": par_conv,
            "manquees": sorted(testees - detectees)}


def ailleurs() -> dict[str, Any]:
    """Messages détectés sur les autres jeux : chaque détection est un ajout potentiel."""
    out: dict[str, Any] = {}
    exemples: dict[str, list[str]] = {}
    # LoCoMo : tous les tours des deux locuteurs, pris comme « user »
    locomo = json.loads(Path("bench/data/locomo10.json").read_text(encoding="utf-8"))
    tours = [str(t.get("text", "")) for c in locomo for k, v in c["conversation"].items()
             if k.startswith("session_") and isinstance(v, list) for t in v]
    hits = [t for t in tours if est_consigne_durable(t, "user")]
    out["locomo"] = {"messages": len(tours), "detections": len(hits)}
    exemples["locomo"] = hits[:5]
    # LongMemEval-S : messages utilisateur des historiques
    lme = json.loads(Path("bench/data/longmemeval/longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
    vus: set[str] = set()
    hits = []
    n_user = 0
    for q in lme:
        for s in q["haystack_sessions"]:
            for t in s:
                c = str(t.get("content", ""))
                if t.get("role") != "user" or c in vus:
                    continue
                vus.add(c)
                n_user += 1
                if est_consigne_durable(c, "user"):
                    hits.append(c)
    out["longmemeval_s"] = {"messages_user_uniques": n_user, "detections": len(hits)}
    exemples["longmemeval_s"] = hits[:8]
    # PersonaMem : les 70 historiques de la carte
    from bench.bench_personamem import DATA_DIR, load_chat, load_personas
    lignes = load_personas(DATA_DIR / "val.csv")
    dispo = sorted(p for p, rs in lignes.items()
                   if (DATA_DIR / "chats" / Path(rs[0]["chat_history_32k_link"]).name).exists())
    hits = []
    n_user = 0
    par_persona: Counter[str] = Counter()
    for pid in random.Random(7).sample(dispo, 70):
        for m in load_chat(DATA_DIR / "chats" / Path(lignes[pid][0]["chat_history_32k_link"]).name):
            if m["role"] != "user":
                continue
            n_user += 1
            if est_consigne_durable(str(m["content"]), "user"):
                hits.append(str(m["content"]))
                par_persona[pid] += 1
    out["personamem"] = {"messages_user": n_user, "detections": len(hits),
                         "personas_touches": len(par_persona), "max_par_persona": max(par_persona.values(), default=0)}
    exemples["personamem"] = hits[:8]
    out["exemples"] = {k: [e[:160] for e in v] for k, v in exemples.items()}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Consignes durables : détecteur")
    parser.add_argument("--output", type=Path, default=Path("bench/results/consignes/detecteur.json"))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    res = {"beam_100k": beam(), "ailleurs": ailleurs()}
    b = res["beam_100k"]
    print(f"BEAM 100K : {b['detectees_parmi_testees']}/{b['consignes_testees']} consignes testées détectées ;"
          f" {b['detections_totales']} détections dont {b['detections_hors_testees']} hors des testées ;"
          f" par conversation {b['detections_par_conversation']}")
    for k, v in res["ailleurs"].items():
        if k != "exemples":
            print(f"{k} : {v}")
    for k, v in res["ailleurs"]["exemples"].items():
        for e in v:
            print(f"  [{k}] {e!r}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Rapport : {args.output}")


if __name__ == "__main__":
    main()
