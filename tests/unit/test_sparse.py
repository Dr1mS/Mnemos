"""Tests sparse coding (§8.2, §19.1) — encoding déterministe, Hamming,
séparation temporelle, saturation.

⚠ Buckets de 4h : les tests de séparation temporelle DOIVENT choisir des
timestamps dans des buckets différents (§8.2), sinon échec à tort.
"""

from __future__ import annotations

from datetime import UTC, datetime

from mnemos.embeddings.sparse import (
    CONTENT_BITS,
    TOKEN_CAP,
    content_similarity,
    hamming_distance,
    sparse_encode,
    sparse_similarity,
    tokenize,
)


def ts(hour: int, day: int = 1) -> int:
    return int(datetime(2026, 7, day, hour, 0, tzinfo=UTC).timestamp() * 1000)


def content_popcount(code: bytes) -> int:
    """Popcount de la seule région contenu (bits 0–223 = 28 premiers octets)."""
    return int.from_bytes(code[:28], "little").bit_count()


def content_region(code: bytes) -> bytes:
    return code[:28]


def temporal_region(code: bytes) -> bytes:
    return code[28:]


def test_encoding_deterministe() -> None:
    a = sparse_encode("Alice bosse chez Datalyse", ts(10))
    b = sparse_encode("Alice bosse chez Datalyse", ts(10))
    assert a == b
    assert len(a) == 32


def test_contenus_differents_se_separent() -> None:
    a = sparse_encode("Alice adore le thé vert japonais", ts(10))
    b = sparse_encode("Le worker consolide les épisodes chaque heure", ts(10))
    assert hamming_distance(a, b) > 10


def test_separation_temporelle_buckets_differents() -> None:
    """Même contenu, buckets 4h différents (10h → bucket 2, 15h → bucket 3) :
    la région contenu est identique, seule la région temporelle bouge."""
    a = sparse_encode("réunion projet mnemos", ts(10))
    b = sparse_encode("réunion projet mnemos", ts(15))
    assert content_region(a) == content_region(b)
    assert temporal_region(a) != temporal_region(b)
    assert 0 < hamming_distance(a, b) <= 32


def test_contenu_different_ne_touche_que_la_region_contenu() -> None:
    """Contenus différents, même bucket : seule la région contenu bouge."""
    a = sparse_encode("réunion projet mnemos", ts(10))
    b = sparse_encode("le chat mange des croquettes au saumon", ts(10))
    assert temporal_region(a) == temporal_region(b)
    assert content_region(a) != content_region(b)


def test_meme_bucket_meme_code() -> None:
    """Voulu par design (§8.2) : même contenu, même bucket de 4h → identiques."""
    assert sparse_encode("hello", ts(10)) == sparse_encode("hello", ts(11))


def test_jours_differents_buckets_differents() -> None:
    a = sparse_encode("hello", ts(10, day=1))
    b = sparse_encode("hello", ts(10, day=2))
    assert hamming_distance(a, b) > 0


def test_cap_tokens_limite_la_saturation() -> None:
    long_content = " ".join(f"token{i}" for i in range(300))
    code = sparse_encode(long_content, ts(10))
    assert content_popcount(code) <= TOKEN_CAP  # ≤ 64 bits contenu (collisions déduites)
    assert content_popcount(code) < CONTENT_BITS * 0.35  # pas de saturation (§8.2)


def test_tokenize_unique_lowercase_cap() -> None:
    toks = tokenize("Thé THÉ thé café")
    assert toks == ["thé", "café"]
    assert len(tokenize(" ".join(f"t{i}" for i in range(100)))) == TOKEN_CAP


def test_hamming_proprietes() -> None:
    a = sparse_encode("aaa bbb", ts(10))
    b = sparse_encode("ccc ddd", ts(10))
    assert hamming_distance(a, a) == 0
    assert hamming_distance(a, b) == hamming_distance(b, a)
    assert sparse_similarity(a, a) == 1.0
    assert 0.0 <= sparse_similarity(a, b) < 1.0


# ── content_similarity : les bits temporels sont du bruit face à une requête ──


def test_content_similarity_ignore_la_date() -> None:
    """Deux épisodes de contenu identique à des dates différentes doivent être
    à égalité stricte face à une même requête.

    `sparse_similarity` les départageait sur le seul hasard du hash temporel :
    mesuré le 29/09/2026, contenu à égalité (14 bits d'écart dans les deux cas)
    mais similarités 0,8906 contre 0,8867. Le classement se jouait sur du bruit.
    """
    requete = sparse_encode("dans quelle ville est-ce que je vis", ts(10, day=1))
    ancien = sparse_encode("je vis à Limoges", ts(10, day=2))
    recent = sparse_encode("je vis à Limoges", ts(10, day=20))

    assert content_similarity(requete, ancien) == content_similarity(requete, recent)
    # Le défaut corrigé : l'ancienne mesure, elle, les sépare.
    assert sparse_similarity(requete, ancien) != sparse_similarity(requete, recent)


def test_content_similarity_reste_discriminante() -> None:
    requete = sparse_encode("le chat mange des croquettes", ts(10))
    proche = sparse_encode("le chat mange des croquettes", ts(10, day=5))
    loin = sparse_encode("le worker consolide les épisodes", ts(10, day=5))
    assert content_similarity(requete, proche) == 1.0
    assert content_similarity(requete, loin) < content_similarity(requete, proche)


def test_content_similarity_bornee() -> None:
    a = sparse_encode("aaa bbb", ts(10))
    b = sparse_encode("ccc ddd", ts(10))
    for x, y in ((a, a), (a, b), (b, a)):
        assert 0.0 <= content_similarity(x, y) <= 1.0
