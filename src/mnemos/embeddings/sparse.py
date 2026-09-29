"""Pattern separation pragmatique (§8.2) — hashing déterministe, pas de neural.

Vecteur 256-bit : 224 bits de contenu (hash des tokens) + 32 bits temporels
(bucket de 4h). Résolution temporelle grossière PAR DESIGN : deux épisodes
de même contenu dans le même bucket ont des codes identiques.

Cap : 64 premiers tokens uniques — au-delà, le OR sature les 224 bits et la
distance de Hamming perd son pouvoir discriminant (anti-pattern 5 : ne pas
mocker ça pour "passer les tests").
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from hashlib import blake2b

from mnemos.logging import get_logger

logger = get_logger(__name__)

CONTENT_BITS = 224
TOTAL_BITS = 256
TOKEN_CAP = 64

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(content: str) -> list[str]:
    """Tokenizer simple (split + lowercase, FR + EN), tokens uniques ordonnés,
    cappés à TOKEN_CAP."""
    seen: dict[str, None] = {}
    for tok in _TOKEN_RE.findall(content.lower()):
        if tok not in seen:
            seen[tok] = None
    tokens = list(seen)
    if len(tokens) > TOKEN_CAP:
        logger.debug("sparse_token_cap", total=len(tokens), kept=TOKEN_CAP)
        tokens = tokens[:TOKEN_CAP]
    return tokens


def sparse_encode(content: str, timestamp_ms: int) -> bytes:
    bits = bytearray(32)  # 256 bits
    # Content bits (0–223)
    for token in tokenize(content):
        h = blake2b(token.encode(), digest_size=2).digest()
        pos = int.from_bytes(h, "little") % CONTENT_BITS
        bits[pos // 8] |= 1 << (pos % 8)
    # Temporal bits (224–255) — bucket de 4h
    dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)
    temporal_seed = f"{dt.year}-W{dt.isocalendar().week}-{dt.weekday()}-{dt.hour // 4}"
    th = blake2b(temporal_seed.encode(), digest_size=4).digest()
    for i in range(32):
        if th[i // 8] & (1 << (i % 8)):
            bits[28 + i // 8] |= 1 << (i % 8)
    return bytes(bits)


def hamming_distance(a: bytes, b: bytes) -> int:
    return (int.from_bytes(a, "little") ^ int.from_bytes(b, "little")).bit_count()


def sparse_similarity(a: bytes, b: bytes) -> float:
    """Similarité normalisée [0..1] sur les 256 bits, temporels compris.

    À réserver à la comparaison de deux ÉPISODES, où les bits temporels sont
    précisément le mécanisme de séparation voulu. Pour classer des épisodes
    face à une requête, utiliser `query_coverage` — voir pourquoi ci-dessous."""
    return 1.0 - hamming_distance(a, b) / TOTAL_BITS


# Masque des 224 bits de contenu : les 4 derniers octets (bits 224–255) portent
# le bucket temporel et sont mis à zéro.
_CONTENT_MASK = int.from_bytes(b"\xff" * 28 + b"\x00" * 4, "little")


def query_coverage(query: bytes, episode: bytes) -> float:
    """Part [0..1] des jetons de la requête présents dans l'épisode.

    C'est la composante lexicale du score de recherche. Elle remplace la
    similarité de Hamming pour deux défauts mesurés le 29/09/2026 :

    **Hamming pénalise la longueur.** `distance = |Q| + |E| − 2·|Q∩E|` : à
    recouvrement égal, un épisode long est plus « loin » qu'un court. Sur 30 000
    candidats LoCoMo, Hamming donnait 0,919 au quart des épisodes les plus
    courts et 0,831 au quart le plus long, quand la similarité dense était plate
    (0,515 contre 0,510). Le terme récompensait les répliques creuses.

    **Les bits temporels sont du bruit face à une requête.** Une requête est
    encodée avec l'heure courante et ne porte jamais la date d'un épisode
    rejoué. Sur deux épisodes de contenu identique, la moitié de la distance de
    Hamming venait de là, et c'est elle seule qui les départageait. Le
    classement changeait selon la tranche de 4 h où l'on interrogeait (hit@10
    LoCoMo de 0,593 à 0,627 sur six tranches).

    Mesuré par `bench/bench_rerank_variants.py`, mêmes candidats, seule la
    formule change, poids 0,7/0,3 inchangés :

        PersonaMem, 164 questions, 50 conversations
            Hamming       hit@10 0,348   rang médian 16   rappel@100 0,835
            recouvrement  hit@10 0,506   rang médian  6   rappel@100 0,860
        LoCoMo, 150 questions
            Hamming       hit@10 0,607   rang médian  6   rappel@100 0,833
            recouvrement  hit@10 0,673   rang médian  3   rappel@100 0,877

    Seuls les bits de contenu entrent en compte. Les bits temporels restent
    écrits en base, où ils séparent deux épisodes de même contenu à des dates
    différentes : c'est leur rôle."""
    q = int.from_bytes(query, "little") & _CONTENT_MASK
    e = int.from_bytes(episode, "little") & _CONTENT_MASK
    return (q & e).bit_count() / max(1, q.bit_count())
