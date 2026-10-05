"""Dates relatives résolues à l'écriture : périmètre figé le 05/10/2026.

Jour, mois, année : annotés. Semaines, durées, formes vagues : jamais — le juge
officiel accepte les semaines sous forme relative et compte faux une date
calculée quand le corrigé est relatif (voir le module)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from mnemos.router.relative_dates import annotate_relative_dates


def _ts(*args: int) -> int:
    return int(datetime(*args, tzinfo=UTC).timestamp() * 1000)


LUNDI_28_AOUT = _ts(2023, 8, 28, 15, 19)

ANNOTES = [
    ("I took my kids to a park yesterday.", "yesterday (27 August 2023)"),
    ("Yesterday was great", "Yesterday (27 August 2023)"),
    ("yesterday's game", "yesterday's (27 August 2023)"),
    ("Last night was fun", "Last night (27 August 2023)"),
    ("the day before yesterday", "the day before yesterday (26 August 2023)"),
    ("Today is my birthday", "Today (28 August 2023)"),
    ("tonight we celebrate", "tonight (28 August 2023)"),
    ("this morning I ran", "this morning (28 August 2023)"),
    ("this evening", "this evening (28 August 2023)"),
    ("tomorrow I fly", "tomorrow (29 August 2023)"),
    ("the day after tomorrow", "the day after tomorrow (30 August 2023)"),
    ("two days ago", "two days ago (26 August 2023)"),
    ("3 days ago", "3 days ago (25 August 2023)"),
    ("a day ago", "a day ago (27 August 2023)"),
    ("last month", "last month (July 2023)"),
    ("this month", "this month (August 2023)"),
    ("next month we go camping", "next month (September 2023)"),
    ("last year", "last year (2022)"),
    ("Last year's festival", "Last year's (2022) festival"),
    ("this year", "this year (2023)"),
    ("next year", "next year (2024)"),
]


@pytest.mark.parametrize(("texte", "attendu"), ANNOTES)
def test_jour_mois_annee_annotes(texte: str, attendu: str) -> None:
    assert attendu in annotate_relative_dates(texte, LUNDI_28_AOUT)


JAMAIS = [
    # Semaines : le juge les veut relatives.
    "last Friday I went", "on Friday", "next Friday", "this Friday", "last week",
    "last weekend", "this past weekend", "this weekend", "next week", "two weeks ago",
    # Durées : « ten years ago » répond tel quel à « il y a combien de temps ».
    "ten years ago", "three months ago", "for two years",
    # Vague.
    "a few days ago", "a couple of days ago", "recently", "the other day",
    # Durée et non année : « the last year » = les douze derniers mois.
    "the last year was hard", "the next year after that", "the last month of school",
    # « last » / « next » hors du temps.
    "At last!", "the last time", "next to me", "the last night of the trip",
    # Au-delà d'un mois : pas une date de calendrier fiable.
    "45 days ago",
]


@pytest.mark.parametrize("texte", JAMAIS)
def test_semaines_durees_et_vague_jamais_annotes(texte: str) -> None:
    assert annotate_relative_dates(texte, LUNDI_28_AOUT) == texte


def test_mots_d_origine_intacts() -> None:
    texte = "Hey! Since we last spoke, I took my kids to a park yesterday. They loved it."
    annote = annotate_relative_dates(texte, LUNDI_28_AOUT)
    assert annote.replace(" (27 August 2023)", "") == texte


def test_pas_deux_annotations() -> None:
    une_fois = annotate_relative_dates("yesterday", LUNDI_28_AOUT)
    assert annotate_relative_dates(une_fois, LUNDI_28_AOUT) == une_fois


def test_plusieurs_expressions() -> None:
    annote = annotate_relative_dates("We met 3 days ago and the day before yesterday",
                                     LUNDI_28_AOUT)
    assert annote == ("We met 3 days ago (25 August 2023) and "
                      "the day before yesterday (26 August 2023)")


def test_passage_d_annee() -> None:
    nouvel_an = _ts(2024, 1, 1, 0, 30)
    assert annotate_relative_dates("yesterday, last month", nouvel_an) == (
        "yesterday (31 December 2023), last month (December 2023)")
    decembre = _ts(2023, 12, 3)
    assert annotate_relative_dates("next month", decembre) == "next month (January 2024)"


def test_ancre_en_utc() -> None:
    """Un message à 23 h 30 UTC reste daté de son jour UTC."""
    tard = _ts(2023, 8, 28, 23, 30)
    assert annotate_relative_dates("yesterday", tard) == "yesterday (27 August 2023)"


def test_texte_sans_expression_inchange() -> None:
    texte = "I love hiking with my dog in the mountains."
    assert annotate_relative_dates(texte, LUNDI_28_AOUT) == texte
