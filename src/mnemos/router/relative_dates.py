"""Dates relatives résolues à l'écriture : « yesterday » → « yesterday (27 August 2023) ».

Le répondeur officiel d'AML a pour consigne (règle 7, dépôt public des
organisateurs) de convertir « yesterday », « last month », « last year » en
dates « quand l'horodatage du souvenir le permet », et de garder relatives les
expressions en semaines. Mesuré le 05/10/2026 avec ces consignes (LoCoMo conv-26,
Nemotron 3 Ultra) : 12 des 22 échecs aux questions de dates viennent d'une
expression en jours, mois ou années restée telle quelle (« yesterday » pour
« 27 August 2023 »). Le juge, lui, accepte les semaines sous forme relative et
compte faux une date calculée quand le corrigé est relatif.

D'où le périmètre, figé avant validation (AML_COMPETITION_PREP.md §27) :
- jour, mois, année : on ajoute la date visée entre parenthèses, après
  l'expression, sans toucher aux mots d'origine ;
- semaines : jamais (« last Friday », « last week », « next week »…) ;
- durées : jamais (« ten years ago » répond tel quel à « il y a combien de temps ») ;
- vague : jamais (« recently », « a few days ago »).

L'appelant n'annote que si le message porte un horodatage de la SOURCE : sans
lui, l'ancre serait l'heure de réception et chaque date serait fausse, puis
recopiée telle quelle. Ancre = date UTC de l'horodatage.

Décision de l'utilisateur (04/10) : rien n'est ajouté au texte au moment de la
recherche. Cette annotation fait partie du souvenir, construit une fois à
l'écriture, sans connaître aucune question.
"""

from __future__ import annotations

import re
from datetime import UTC, date, datetime, timedelta

_MOIS = ("January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December")

_NOMBRES = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}

# « the last year » est une durée (les douze derniers mois), « the next year »
# l'année d'après un autre événement : seul « last year » nu désigne une année.
_PAS_THE = r"(?<!the )"
_POSSESSIF = r"(?:'s|’s)?"

# Ordre important : les formes longues d'abord (« the day before yesterday »
# avant « yesterday »). Chaque alternative nomme son décalage.
_MOTIF = re.compile(
    r"\b(?:"
    r"(?P<avant_hier>the day before yesterday)"
    r"|(?P<apres_demain>the day after tomorrow)"
    rf"|(?P<hier>yesterday{_POSSESSIF}|{_PAS_THE}last night)"
    rf"|(?P<aujourdhui>today{_POSSESSIF}|tonight{_POSSESSIF}|this (?:morning|afternoon|evening))"
    rf"|(?P<demain>tomorrow{_POSSESSIF})"
    r"|(?P<jours_n>(?P<n>\d{1,2}|" + "|".join(_NOMBRES) + r") days? ago)"
    rf"|{_PAS_THE}(?P<mois>(?P<mois_qui>last|this|next) month{_POSSESSIF})"
    rf"|{_PAS_THE}(?P<annee>(?P<annee_qui>last|this|next) year{_POSSESSIF})"
    r")\b"
    # Déjà annoté : ne pas annoter deux fois.
    r"(?! \()",
    re.IGNORECASE,
)


def _jour(d: date) -> str:
    return f"{d.day} {_MOIS[d.month - 1]} {d.year}"


def _mois_decale(d: date, decalage: int) -> str:
    index = d.year * 12 + (d.month - 1) + decalage
    return f"{_MOIS[index % 12]} {index // 12}"


def _resoudre(m: re.Match[str], ancre: date) -> str | None:
    if m.group("avant_hier"):
        return _jour(ancre - timedelta(days=2))
    if m.group("apres_demain"):
        return _jour(ancre + timedelta(days=2))
    if m.group("hier"):
        return _jour(ancre - timedelta(days=1))
    if m.group("aujourdhui"):
        return _jour(ancre)
    if m.group("demain"):
        return _jour(ancre + timedelta(days=1))
    if m.group("jours_n"):
        n = m.group("n").lower()
        jours = int(n) if n.isdigit() else _NOMBRES[n]
        return _jour(ancre - timedelta(days=jours)) if 1 <= jours <= 31 else None
    decalages = {"last": -1, "this": 0, "next": 1}
    if m.group("mois"):
        return _mois_decale(ancre, decalages[m.group("mois_qui").lower()])
    if m.group("annee"):
        return str(ancre.year + decalages[m.group("annee_qui").lower()])
    return None


def annotate_relative_dates(text: str, timestamp_ms: int) -> str:
    """Ajoute la date résolue après chaque expression du périmètre.

    `timestamp_ms` doit être l'horodatage de la SOURCE (voir le module)."""
    ancre = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC).date()

    def remplacer(m: re.Match[str]) -> str:
        resolu = _resoudre(m, ancre)
        return f"{m.group(0)} ({resolu})" if resolu else m.group(0)

    return _MOTIF.sub(remplacer, text)
