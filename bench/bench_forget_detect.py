"""Précision et rappel du détecteur de demandes d'oubli (axe 3).

Aucun LLM, aucun GPU : `detect_forget` est une fonction pure, on la passe sur :

* **PersonaMem-v2**, les 84 861 messages utilisateur des 735 historiques 32k.
  Étiquette : un message est une consigne s'il commence par « (please) forget »
  — la grammaire des 12 588 demandes du jeu, vérifiée à la main sur échantillon.
  Découpage par persona : identifiant pair → calibration, impair → validation.
  La validation n'est lue qu'une fois le détecteur figé.
* **LoCoMo**, 419 tours d'une conversation naturelle : que des négatifs.
* **FORGET_CASES** (`bench/datasets.py`), cas fabriqués : français, formes
  polies, et surtout les pièges.

Chaque faux positif et chaque faux négatif est listé. Un taux ne dit pas si
l'erreur est tolérable ; le message, lui, le dit.

    python bench/bench_forget_detect.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.bench_personamem import DATA_DIR, load_chat, load_personas
from bench.datasets import FORGET_CASES
from mnemos.router.forget import detect_forget

# Étiquette PersonaMem : la grammaire des consignes du jeu. La validation a
# révélé une consigne hors de ce moule (« Please remove from your memory that
# I… », persona 273) : l'erreur était dans l'étiquette, pas dans le détecteur.
_PM_LABEL = re.compile(
    r"^\s*(?:please\s+)?(?:forget\b|(?:remove|delete|erase)\s+from\s+your\s+memory\b)",
    re.IGNORECASE,
)


def _mesure(exemples: list[tuple[str, bool, str]]) -> dict[str, Any]:
    vp = fp = fn = vn = 0
    faux_pos: list[tuple[str, str]] = []
    faux_neg: list[tuple[str, str]] = []
    for texte, attendu, source in exemples:
        predit = detect_forget(texte, "user") is not None
        if predit and attendu:
            vp += 1
        elif predit:
            fp += 1
            faux_pos.append((source, texte))
        elif attendu:
            fn += 1
            faux_neg.append((source, texte))
        else:
            vn += 1
    return {
        "positifs": vp + fn, "negatifs": fp + vn,
        "precision": round(vp / (vp + fp), 4) if vp + fp else None,
        "rappel": round(vp / (vp + fn), 4) if vp + fn else None,
        "faux_positifs": faux_pos, "faux_negatifs": faux_neg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Détecteur d'oubli : précision et rappel")
    parser.add_argument("--output", type=Path, default=Path("bench/results/forget_detect.json"))
    parser.add_argument("--validation", action="store_true",
                        help="mesurer aussi la moitié de validation — seulement détecteur figé")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    rows = load_personas(DATA_DIR / "val.csv")
    calib: list[tuple[str, bool, str]] = []
    valid: list[tuple[str, bool, str]] = []
    for pid, rs in rows.items():
        chemin = DATA_DIR / "chats" / Path(rs[0]["chat_history_32k_link"]).name
        if not chemin.exists():
            continue
        cible = calib if int(pid) % 2 == 0 else valid
        for m in load_chat(chemin):
            if m["role"] == "user":
                cible.append((m["content"], bool(_PM_LABEL.search(m["content"])), f"pm:{pid}"))

    conv = json.loads(Path("bench/data/locomo_sample.json").read_text(encoding="utf-8"))["conversation"]
    locomo = [(t.get("text", ""), False, "locomo") for k, v in conv.items()
              if k.startswith("session_") and not k.endswith("_date_time") for t in v]
    fabriques = [(t, a, f"fabriqué: {note}") for t, a, note in FORGET_CASES]

    resultats = {
        "PersonaMem calibration": _mesure(calib),
        **({"PersonaMem validation": _mesure(valid)} if args.validation else {}),
        "LoCoMo (négatifs)": _mesure(locomo),
        "cas fabriqués": _mesure(fabriques),
    }
    for nom, r in resultats.items():
        p = "—" if r["precision"] is None else f"{r['precision']:.4f}"
        rp = "—" if r["rappel"] is None else f"{r['rappel']:.4f}"
        print(f"{nom:24s} positifs {r['positifs']:6d} | négatifs {r['negatifs']:6d}"
              f" | précision {p} | rappel {rp}"
              f" | FP {len(r['faux_positifs'])} | FN {len(r['faux_negatifs'])}")
    for nom, r in resultats.items():
        for genre in ("faux_positifs", "faux_negatifs"):
            if r[genre]:
                print(f"\n{nom} — {genre.replace('_', ' ')} ({len(r[genre])}) :")
                for source, texte in r[genre][:15]:
                    print(f"  [{source}] {texte[:150]!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(resultats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nRapport : {args.output}")


if __name__ == "__main__":
    main()
