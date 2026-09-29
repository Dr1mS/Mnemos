"""Détection des demandes d'oubli (axe 3, catégorie AML D3) — règles FR + EN.

Le contrat AML n'a ni endpoint ni champ de suppression : « oublie mon adresse »
arrive comme un message ordinaire, et c'est à la mémoire de l'interpréter.
Ce module dit seulement **si** un message est une consigne d'oubli et **ce
qu'il vise**. Il ne supprime rien : le ciblage des épisodes est ailleurs.

**Précision d'abord.** Un faux positif efface la mémoire de l'utilisateur et
dégrade toutes les catégories d'un coup ; un faux négatif nous laisse au score
actuel. D'où quatre exigences cumulatives :

1. un message **utilisateur** ;
2. la consigne **en tête** du message (après une interjection au plus) — sinon
   « help me tidy this email: … please disregard this note … » supprimerait
   quelque chose sur la foi d'un texte cité ;
3. une forme **impérative** ou de demande polie (« can you forget… ») — jamais
   « don't forget », « oublie pas » (le français parlé élide le *ne*),
   « I forgot », « what did I ask you to forget? » ;
4. un complément qui **parle de l'utilisateur** (« that I… », « my… », « que
   je… », « mon… ») — « forget it » (laisse tomber) ou « can you forget
   things? » ne visent rien d'effaçable.

Les verbes génériques (delete, remove, efface, supprime…) exigent en plus une
**ancre mémoire** (« from your memory », « de ta mémoire ») : sans elle, la
piste Coding (« delete this file », « remove that function ») serait un
générateur industriel de faux positifs.

Écarté volontairement : la forme anaphorique seule (« forget that. », « oublie
ça ») — sans complément, la cible serait devinée, et « forget it » veut le plus
souvent dire « laisse tomber ». Et la similarité d'embedding comme détecteur :
elle encode mal la polarité, « je n'oublierai jamais ce voyage » et « oublie
mon voyage » sont voisins.

Mesuré sur PersonaMem-v2 (84 861 messages utilisateur, 735 personas), les
demandes réelles suivent une grammaire étroite — « (Please) forget that I… /
my preference for… / the detail about me… », une phrase, 11 mots en médiane —
et représentent 14,8 % des messages. Voir `bench/bench_forget_detect.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Une consigne d'oubli est brève (PersonaMem : 31 mots au plus). Au-delà, le
# verbe a toutes les chances d'appartenir à un texte cité ou à un récit.
MAX_WORDS = 60
MIN_TARGET_WORDS = 2


@dataclass(frozen=True)
class ForgetDirective:
    """Une consigne d'oubli détectée.

    `target` est ce qu'il faut oublier, dans les mots de l'utilisateur, débarrassé
    de la formule de consigne (« Please forget that I use cloud storage » →
    « I use cloud storage »). C'est lui que le ciblage compare aux épisodes."""

    target: str
    family: str  # "explicit" | "anchored" — pour le diagnostic


_APOS = "['’]"

# Interjection tolérée en tête : « Hey, please forget… », « Ok, oublie que… ».
_LEAD = (
    r"^\s*(?:(?:hey|hi|ok(?:ay)?|also|and|actually|oh|alright|by the way|btw|one more thing"
    r"|au fait|d" + _APOS + r"accord|et|aussi|bon|alors|tiens|ah)\s*[,.!:;—–-]*\s+)?"
)

_POLITE_EN = r"(?:please\s+)?(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?)?"
_POLITE_FR = (
    r"(?:s" + _APOS + r"il (?:te|vous) pla[iî]t\s*,?\s*|stp\s*,?\s*|merci d" + _APOS + r"\s*)?"
    r"(?:(?:peux|pourrais|pourriez|pouvez)-(?:tu|vous)\s+|(?:tu peux|vous pouvez)\s+)?"
)

# Verbes d'oubli explicites : aucune ancre requise.
_EXPLICIT = re.compile(
    _LEAD + r"(?:"
    + _POLITE_EN + r"(?:forget|stop remembering|no longer remember|disregard)"
    + r"|(?:please\s+)?(?:don" + _APOS + r"?t|do not)\s+remember"
    + r"|" + _POLITE_FR + r"(?:oublie[rz]?|ne (?:retiens|retenez|retenir) plus"
    + r"|ne (?:te|vous) souviens plus|ne (?:te|vous) souvenez plus|ne plus (?:te|vous) souvenir)"
    + r")\b\s*[,:]?\s*(?P<reste>.*)$",
    re.IGNORECASE | re.DOTALL,
)

# Verbes génériques : l'ancre mémoire est obligatoire.
_ANCHORED = re.compile(
    _LEAD + r"(?:"
    + _POLITE_EN + r"(?:delete|remove|erase|wipe|drop|clear)"
    + r"|" + _POLITE_FR + r"(?:efface[rz]?|supprime[rz]?|retire[rz]?)"
    + r")\b\s*(?P<reste>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_MEMORY_ANCHOR = re.compile(
    r"\b(?:from|out of) (?:your|the) memory\b|\bwhat i (?:told|said to) you\b"
    r"|\bde (?:ta|votre) m[ée]moire\b|\bce que je (?:t|vous)" + _APOS + r"ai dit\b",
    re.IGNORECASE,
)

# Gardes : formes qui ressemblent à une consigne sans en être une.
_GUARDS = re.compile(
    _LEAD + r"(?:"
    r"(?:don" + _APOS + r"?t|do not|never|n" + _APOS + r"|ne\s+)\s*(?:forget|oublie)"
    r"|oublie[rz]?\s+pas\b"                     # « oublie pas de… » = n'oublie pas
    r"|forget (?:it|about it|that)\s*[.!]*$"    # « forget it » = laisse tomber
    r"|oublie (?:ça|cela|le|la)\s*[.!]*$"
    r")",
    re.IGNORECASE,
)

# Le complément doit parler de l'utilisateur.
_SELF_REF = re.compile(
    r"\b(?:i|i" + _APOS + r"(?:m|ve|d|ll)|my|me|mine|myself"
    r"|je|j" + _APOS + r"|mon|ma|mes|moi|m" + _APOS + r")\b|\bj" + _APOS,
    re.IGNORECASE,
)

# « Please forget the preference about encountering prejudice in academia » ne
# contient ni I, ni my, ni me : « the preference », adressé à l'assistant,
# désigne pourtant celle de l'utilisateur. Singulier seulement — « forget the
# details about pricing, focus on features » est une consigne de travail.
_IMPLICIT_SELF = re.compile(
    r"^the (?:preference|detail|fact|information|memory)\s+(?:about|that|for|of|on)\b",
    re.IGNORECASE,
)

# Formule de consigne à retirer du complément.
_TARGET_PREFIX = re.compile(
    # L'ancre mémoire en tête (« delete from your memory that I… ») d'abord :
    # sinon « de » serait retiré seul et « ta mémoire que je… » resterait.
    r"^(?:(?:from|out of) (?:your|the) memory(?: that| about)?"
    r"|de (?:ta|votre) m[ée]moire(?: que| qu" + _APOS + r")?"
    r"|that|about|the (?:fact|detail|preference|information|info)(?: that| about| on| for)?"
    r"|my preference (?:for|about|of|that|to)?|everything about|all about"
    r"|que|qu" + _APOS + r"|le fait que|de|d" + _APOS + r"|ce que)\s*",
    re.IGNORECASE,
)
_TARGET_SUFFIX = re.compile(
    r"\s*(?:,?\s*(?:please|s" + _APOS + r"il (?:te|vous) pla[iî]t|stp))?"
    r"\s*(?:(?:from|out of|in) (?:your|the) memory|de (?:ta|votre) m[ée]moire)?"
    r"\s*[.!?…]*\s*$",
    re.IGNORECASE,
)


def _clean_target(raw: str) -> str:
    t = raw.strip()
    t = _TARGET_PREFIX.sub("", t, count=1)
    t = _TARGET_SUFFIX.sub("", t, count=1)
    return t.strip(" ,;:—–-\"'«»")


def detect_forget(content: str, role: str) -> ForgetDirective | None:
    """La consigne d'oubli portée par ce message, ou None.

    Pure et sans I/O : appelée pour chaque message utilisateur du chemin /add."""
    if role != "user":
        return None
    text = content.strip()
    if not text or len(text.split()) > MAX_WORDS:
        return None
    if _GUARDS.match(text):
        return None
    # Une question n'est une consigne que sous la forme polie « can you forget… ? ».
    if text.rstrip().endswith("?") and not re.search(
        r"\b(?:can|could|would|will) you\b|-(?:tu|vous)\b|\b(?:tu peux|vous pouvez)\b",
        text[:40], re.IGNORECASE,
    ):
        return None

    for regex, family in ((_EXPLICIT, "explicit"), (_ANCHORED, "anchored")):
        m = regex.match(text)
        if not m:
            continue
        if family == "anchored" and not _MEMORY_ANCHOR.search(text):
            continue
        brut = m.group("reste")
        target = _clean_target(brut)
        # L'auto-référence se vérifie sur le complément BRUT : le nettoyage retire
        # justement « my preference for », qui la portait.
        if len(target.split()) < MIN_TARGET_WORDS:
            continue
        if not (_SELF_REF.search(brut) or _IMPLICIT_SELF.match(brut)
                or _MEMORY_ANCHOR.search(text)):
            continue
        return ForgetDirective(target=target, family=family)
    return None


def forgotten_in_batch(
    messages: list[tuple[str, str]], ack_pending: bool = False
) -> tuple[list[bool], bool]:
    """Quels messages d'un lot `/add` ne doivent jamais être mémorisés.

    `messages` : (rôle, contenu) dans l'ordre du lot. `ack_pending` : le lot
    précédent de la même session s'est terminé sur une consigne, dont l'accusé
    de réception ouvre donc ce lot.

    Deux messages disparaissent pour chaque consigne : **la consigne elle-même**
    et **l'accusé de réception de l'assistant qui la suit**. Tous deux répètent
    ce qu'il faut oublier — relevé sur PersonaMem-v2, l'accusé répète la
    préférence dans 77 cas sur 119 (« Got it — I'll forget that you watch
    historical documentaries »). Les garder, c'est la faire fuiter dans le
    contexte du répondeur, qui n'applique pas l'instruction (13 fois sur 13).

    Mesuré sur 47 personas et 874 consignes (bench/bench_forget_targeting.py) :
    la préférence oubliée remonte dans le top 10 pour 49 % des questions au lieu
    de 83 %, sans qu'aucun message-preuve d'une autre question soit touché — ce
    ne sont que des messages nouveaux, jamais des souvenirs existants.

    Rend (oublier[i], accusé encore attendu après ce lot)."""
    oublier = [False] * len(messages)
    attend_accuse = ack_pending
    for i, (role, content) in enumerate(messages):
        if attend_accuse and role == "assistant":
            oublier[i] = True
            attend_accuse = False
            continue
        attend_accuse = False
        if detect_forget(content, role) is not None:
            oublier[i] = True
            attend_accuse = True
    return oublier, attend_accuse
