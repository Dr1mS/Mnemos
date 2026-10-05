"""Précision du normaliseur de dates sur LoCoMo, sans aucun modèle de langage.

Pour chaque question de dates (catégorie 2), on annote les répliques-preuves avec
la date de leur session (`mnemos.router.relative_dates`), puis on compare la date
ajoutée au corrigé, quand le corrigé est une date seule, à la même granularité.
Développement : conv-26, d'où le périmètre a été tiré (05/10/2026). Validation :
les neuf autres conversations, sans retouche des motifs.

Mesuré le 05/10 : 13/13 en développement, 75/77 en validation. Les deux écarts
sont une erreur d'étiquette (« January 9, 2023 » pour une session de janvier 2024)
et une expression portant sur un autre événement de la même réplique.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.bench_locomo import parse_locomo_datetime
from mnemos.router.relative_dates import annotate_relative_dates

LOCOMO = Path("bench/data/locomo10.json")
DEV = "conv-26"
MOIS = ["january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december"]
_AJOUT = re.compile(
    r" \(((?:\d{1,2} )?(?:January|February|March|April|May|June|July|August|September"
    r"|October|November|December)? ?\d{4})\)")

Date = tuple[str, tuple[int, ...]]


def lire_date(texte: str) -> Date | None:
    """(granularité, valeur) d'une date écrite seule ; None sinon."""
    s = re.sub(r"[,\.]", " ", texte.lower()).replace("januarty", "january")
    s = " ".join(re.sub(r"\b(on|in|the)\b", " ", s).split())
    m = re.fullmatch(r"(\d{1,2}) ([a-z]+) (\d{4})", s)
    if m and m.group(2) in MOIS:
        return ("jour", (int(m.group(3)), MOIS.index(m.group(2)) + 1, int(m.group(1))))
    m = re.fullmatch(r"([a-z]+) (\d{1,2}) (\d{4})", s)
    if m and m.group(1) in MOIS:
        return ("jour", (int(m.group(3)), MOIS.index(m.group(1)) + 1, int(m.group(2))))
    m = re.fullmatch(r"([a-z]+) (\d{4})", s)
    if m and m.group(1) in MOIS:
        return ("mois", (int(m.group(2)), MOIS.index(m.group(1)) + 1))
    m = re.fullmatch(r"(\d{4})", s)
    return ("annee", (int(m.group(1)),)) if m else None


def _tours(conv: dict[str, Any]) -> dict[str, tuple[int, str, str]]:
    tours: dict[str, tuple[int, str, str]] = {}
    for k, v in conv.items():
        if k.startswith("session_") and not k.endswith("date_time"):
            quand = conv.get(f"{k}_date_time") or ""
            base = parse_locomo_datetime(quand)
            for t in v:
                tours[t["dia_id"]] = (base, t["text"], quand)
    return tours


def _preuves(q: dict[str, Any]) -> list[str]:
    return [e.strip() for ev in q.get("evidence") or []
            for e in str(ev).replace(",", ";").split(";") if e.strip()]


def mesurer() -> tuple[dict[str, Counter[str]], list[tuple[str, ...]]]:
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    ecarts: list[tuple[str, ...]] = []
    for c in json.loads(LOCOMO.read_text(encoding="utf-8")):
        moitie = "dev" if c["sample_id"] == DEV else "val"
        tours = _tours(c["conversation"])
        for q in c["qa"]:
            if q.get("category") != 2 or q.get("answer") is None:
                continue
            corrige = lire_date(str(q["answer"]))
            ajouts = [(lire_date(a), a, quand, texte)
                      for base, texte, quand in (tours[e] for e in _preuves(q) if e in tours)
                      for a in _AJOUT.findall(annotate_relative_dates(texte, base))]
            if not ajouts:
                stats[moitie]["rien d'annoté" + ("" if corrige else " (corrigé non daté)")] += 1
            elif corrige is None:
                stats[moitie]["annoté, corrigé non daté"] += 1
            else:
                memes = [a for a in ajouts if a[0] and a[0][0] == corrige[0]]
                if any(a[0] == corrige for a in memes):
                    stats[moitie]["annoté = corrigé"] += 1
                elif memes:
                    stats[moitie]["annoté ≠ corrigé"] += 1
                    ecarts.append((c["sample_id"], str(q["answer"]), memes[0][1], memes[0][2],
                                   memes[0][3][:110]))
                else:
                    stats[moitie]["annoté, autre granularité"] += 1
    return stats, ecarts


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    stats, ecarts = mesurer()
    for moitie in ("dev", "val"):
        s = stats[moitie]
        ok, ko = s["annoté = corrigé"], s["annoté ≠ corrigé"]
        print(f"{moitie} : {dict(sorted(s.items()))}\n   précision arithmétique : {ok}/{ok + ko}")
    print("\nannoté ≠ corrigé, même granularité (à lire à la main) :")
    for e in ecarts:
        print("  ", e)


if __name__ == "__main__":
    main()
