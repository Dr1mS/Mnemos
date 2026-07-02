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
    """Similarité normalisée [0..1] pour le score hybride (§8.2)."""
    return 1.0 - hamming_distance(a, b) / TOTAL_BITS
