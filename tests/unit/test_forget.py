"""Détecteur de demandes d'oubli (axe 3, catégorie AML D3).

Un faux positif efface la mémoire de l'utilisateur : les pièges sont la moitié
la plus importante de ces tests. Mesure à grande échelle dans
`bench/bench_forget_detect.py` (PersonaMem-v2, 84 861 messages réels).
"""

from __future__ import annotations

import pytest

from mnemos.router.forget import detect_forget, forgotten_in_batch

# (message, cible attendue après nettoyage)
CONSIGNES = [
    # La grammaire de PersonaMem-v2, 12 588 occurrences réelles
    ("Please forget that I use cloud storage to back up files.",
     "I use cloud storage to back up files"),
    ("Please forget my preference for hearty, rustic home cooking from your memory.",
     "hearty, rustic home cooking"),
    ("Please forget the detail about me feeling inadequate after comparing my career.",
     "me feeling inadequate after comparing my career"),
    ("Forget that I prefer contemporary educational tools over traditional materials.",
     "I prefer contemporary educational tools over traditional materials"),
    # Auto-référence implicite : « the preference » adressé à l'assistant
    ("Please forget the preference about avoiding celebrity news from your memory.",
     "avoiding celebrity news"),
    # Verbe générique + ancre mémoire (trouvé en validation, persona 273)
    ("Please remove from your memory that I endured cyber harassment.",
     "I endured cyber harassment"),
    ("Please delete from your memory that I live in Lyon.", "I live in Lyon"),
    # Formes polies et interjections
    ("Can you forget my home address?", "my home address"),
    ("Hey, please forget that I'm vegetarian.", "I'm vegetarian"),
    # Français
    ("Oublie que je suis allergique aux arachides.", "je suis allergique aux arachides"),
    ("Peux-tu oublier mon adresse ?", "mon adresse"),
    ("S'il te plaît, oublie que j'ai déménagé à Nantes.", "j'ai déménagé à Nantes"),
    ("Efface de ta mémoire que je fume.", "je fume"),
]

PIEGES = [
    "Don't forget that I have a meeting tomorrow.",      # rappel, l'inverse d'un oubli
    "N'oublie pas que je pars lundi.",
    "Oublie pas de m'appeler demain.",                   # le français parlé élide le « ne »
    "I'll never forget that trip to Japan with my sister.",
    "Je n'oublierai jamais ce voyage.",
    "I forgot my keys at home again.",
    "J'ai oublié mon parapluie.",
    "Forget it.",                                        # laisse tomber
    "Forget about it, not important.",
    "Oublie ça.",                                        # anaphorique : cible devinée, écarté
    "What did I ask you to forget?",                     # question sur l'oubli : sonde D3
    "Qu'est-ce que je t'ai demandé d'oublier ?",
    "Did you forget what I told you?",
    "Can you forget things?",                            # rien de personnel à effacer
    "She wants to forget her ex.",                       # 3e personne
    "Please delete this file from the repository.",      # piste Coding : pas d'ancre mémoire
    "Remove that function, it's unused.",
    "Supprime ce fichier, il ne sert plus.",
    "Forget the details about pricing and focus on the features.",  # pluriel : consigne de travail
    "Help me tidy this email: Hi Tom, if the date changed, please disregard this note entirely.",
    "Remember that I prefer window seats.",              # mémorisation, l'inverse
]


@pytest.mark.parametrize(("message", "cible"), CONSIGNES)
def test_consigne_detectee_et_cible_nettoyee(message: str, cible: str) -> None:
    d = detect_forget(message, "user")
    assert d is not None, f"consigne manquée : {message!r}"
    assert d.target == cible


@pytest.mark.parametrize("message", PIEGES)
def test_piege_rejete(message: str) -> None:
    d = detect_forget(message, "user")
    assert d is None, f"faux positif — effacerait la mémoire : {message!r} → {d}"


def test_seuls_les_messages_utilisateur_comptent() -> None:
    """L'accusé de réception de l'assistant (« Got it — I'll forget that you… »)
    n'est pas une consigne : c'est l'utilisateur qui décide de ce qu'on oublie."""
    assert detect_forget("Please forget that I use Notion.", "assistant") is None
    assert detect_forget("Got it — I'll forget that you use Notion.", "assistant") is None


def test_message_long_rejete() -> None:
    """Au-delà de 60 mots, le verbe appartient presque toujours à un récit ou à
    un texte cité (PersonaMem : 31 mots au plus pour une vraie consigne)."""
    long = "Please forget that I " + "really " * 70 + "like jazz."
    assert detect_forget(long, "user") is None


def test_famille_rapportee() -> None:
    explicite = detect_forget("Please forget that I like jazz.", "user")
    ancree = detect_forget("Please delete from your memory that I like jazz.", "user")
    assert explicite is not None and explicite.family == "explicit"
    assert ancree is not None and ancree.family == "anchored"


# ── forgotten_in_batch : ce qui ne sera jamais mémorisé dans un lot /add ──

C = "Please forget that I love jazz."
A = "Got it — I'll forget that you love jazz."


def test_lot_consigne_et_accuse_oublies_rien_d_autre() -> None:
    lot = [("user", "I love jazz."), ("assistant", "Nice!"), ("user", C),
           ("assistant", A), ("user", "What now?")]
    oublier, attend = forgotten_in_batch(lot)
    assert oublier == [False, False, True, True, False]
    assert attend is False


def test_lot_qui_finit_sur_une_consigne_attend_l_accuse() -> None:
    oublier, attend = forgotten_in_batch([("user", "Hi."), ("user", C)])
    assert oublier == [False, True] and attend is True
    # Le lot suivant s'ouvre sur l'accusé : il disparaît, l'attente s'éteint.
    oublier, attend = forgotten_in_batch([("assistant", A), ("user", "Ok.")], ack_pending=True)
    assert oublier == [True, False] and attend is False


def test_attente_eteinte_par_un_message_utilisateur() -> None:
    oublier, attend = forgotten_in_batch([("user", "Anyway."), ("assistant", "Sure.")],
                                         ack_pending=True)
    assert oublier == [False, False] and attend is False


def test_consigne_sans_accuse_puis_nouvelle_consigne() -> None:
    oublier, attend = forgotten_in_batch([("user", C), ("user", "Please forget that I own a cat."),
                                          ("assistant", "Done.")])
    assert oublier == [True, True, True] and attend is False


def test_texte_d_assistant_qui_parle_d_oubli_n_est_pas_une_consigne() -> None:
    oublier, _ = forgotten_in_batch([("assistant", "Please forget that I said that."),
                                     ("assistant", "Anyway, here is the plan.")])
    assert oublier == [False, False]
