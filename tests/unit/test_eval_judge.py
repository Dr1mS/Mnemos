"""Le juge des benchs est un instrument : il doit être testé comme tel.

Un juge qui recale une bonne réponse gonfle la difficulté apparente et envoie
corriger des problèmes qui n'existent pas. C'est exactement ce qui s'est produit
le 29/09 : `check_active_fact_answer` rejetait « Avant chez Datalyse, maintenant
chez Nexora » — la réponse idéale — parce qu'il cherchait un marqueur d'actualité
n'importe où après la valeur périmée, sans regarder ce qu'il qualifiait.

Ces cas gèlent les deux erreurs possibles : le faux rejet (une réponse juste,
correctement située dans le passé) et le faux blanchiment (une valeur périmée ou
un leurre présenté comme actuel).
"""

from __future__ import annotations

import pytest
from bench.eval_utils import check_active_fact_answer, mentions_term, normalize_text

ACTIVE = "Nexora"
PERIMEES = ["Datalyse", "Kyndra", "Velora", "Mirestone"]

ACCEPTES = [
    "Nexora",
    "Vous travaillez chez Nexora.",
    # La bascule correctement formulée : le marqueur d'actualité qualifie la
    # valeur active, pas la périmée qui le précède.
    "Avant chez Datalyse, maintenant chez Nexora.",
    "Anciennement Datalyse, puis Kyndra, aujourd hui Nexora.",
    "Vous travailliez chez Datalyse, vous etes chez Nexora.",
    "Vous etiez chez Kyndra, desormais chez Nexora.",
    "Vous avez quitte Datalyse pour Nexora.",
    "Vous etiez a Kyndra et vous etes a Nexora.",
]

REJETES = [
    "Kyndra",                                                    # valeur périmée seule
    "Velora",                                                    # leurre seul
    "Nexora et Velora",                                          # leurre juxtaposé
    "Datalyse est actuellement votre employeur, et Nexora aussi.",  # périmée donnée actuelle
    "Votre nouveau employeur Kyndra, sinon Nexora.",              # périmée donnée actuelle
    "Vous ne travaillez plus chez Nexora.",                       # active niée
    "Vous travaillez chez Nexora et chez Kyndra.",                # périmée non située
]


@pytest.mark.parametrize("reponse", ACCEPTES)
def test_reponses_correctes_acceptees(reponse: str) -> None:
    ok, raison = check_active_fact_answer(reponse, ACTIVE, PERIMEES)
    assert ok, f"faux rejet : {reponse!r} — {raison}"


@pytest.mark.parametrize("reponse", REJETES)
def test_reponses_fautives_rejetees(reponse: str) -> None:
    ok, _ = check_active_fact_answer(reponse, ACTIVE, PERIMEES)
    assert not ok, f"faux blanchiment : {reponse!r}"


def test_appartenance_au_mot_pres() -> None:
    """`in` nu faisait de `Gap` un sous-mot de `gaspillage`."""
    assert not mentions_term(normalize_text("un gaspillage total"), normalize_text("Gap"))
    assert mentions_term(normalize_text("je vis a Gap depuis un an"), normalize_text("Gap"))
    assert mentions_term(normalize_text("le semi-marathon de Nantes"),
                         normalize_text("semi-marathon de Nantes"))


def test_conjonction_et_ne_blanchit_pas() -> None:
    """Le radical d'imparfait `et` ne doit jamais attraper la conjonction.

    Sans terminaison obligatoire, « Nexora et Kyndra » contiendrait un
    « marqueur de passé » et toute valeur périmée passerait."""
    ok, _ = check_active_fact_answer("Nexora et Kyndra", ACTIVE, PERIMEES)
    assert not ok
