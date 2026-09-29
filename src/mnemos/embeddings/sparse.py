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
    précisément le mécanisme de séparation voulu. Pour comparer une requête à
    un épisode, utiliser `content_similarity` — voir pourquoi ci-dessous."""
    return 1.0 - hamming_distance(a, b) / TOTAL_BITS


# Masque des 224 bits de contenu : les 4 derniers octets (bits 224–255) portent
# le bucket temporel et sont mis à zéro.
_CONTENT_MASK = int.from_bytes(b"\xff" * 28 + b"\x00" * 4, "little")


def content_similarity(a: bytes, b: bytes) -> float:
    """Similarité [0..1] sur les seuls bits de contenu.

    Une requête est encodée avec l'heure courante, un épisode avec l'horodatage
    du message. Les deux buckets ne coïncident jamais sur un corpus rejoué :
    les 32 bits temporels sont alors **du bruit, pas un signal**.

    Mesuré le 29/09/2026 sur deux épisodes de contenu identique (donc 14 bits de
    contenu différents dans les deux cas), requête 2026 contre épisodes 2023 :

        hamming total 28 = 14 de contenu + 14 de temporel
        hamming total 29 = 14 de contenu + 15 de temporel

    La moitié de la distance venait du temporel, et c'est **lui seul** qui
    départageait les deux épisodes (0,8906 contre 0,8867) alors que leur contenu
    était à égalité stricte. Le classement se jouait sur le hasard d'un hash.

    Les bits temporels restent écrits en base : ils séparent deux épisodes de
    même contenu à des dates différentes, ce qui est leur rôle. On cesse
    seulement de les interroger depuis une requête qui ne peut pas les porter."""
    ai = int.from_bytes(a, "little") & _CONTENT_MASK
    bi = int.from_bytes(b, "little") & _CONTENT_MASK
    return 1.0 - (ai ^ bi).bit_count() / CONTENT_BITS
