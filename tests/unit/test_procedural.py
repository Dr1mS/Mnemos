"""Tests ProceduralStore (§12) — registre filesystem, stats, recherche naïve."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemos.clock import FixedClock
from mnemos.stores.procedural import ProceduralStore, SkillMeta


@pytest.fixture
def store(tmp_path: Path, fixed_clock: FixedClock) -> ProceduralStore:
    return ProceduralStore(tmp_path / "procedural", fixed_clock)


def make_meta(name: str, desc: str = "envoie un mail avec pièce jointe") -> SkillMeta:
    return SkillMeta(name=name, desc=desc, signature="send(to: str, body: str) -> bool")


def test_vide_au_depart(store: ProceduralStore) -> None:
    assert store.list_skills() == []
    assert store.get_skill("inconnu") is None


def test_register_puis_get(store: ProceduralStore) -> None:
    store.register_skill("send_email", "def send(): ...", make_meta("send_email"))
    skill = store.get_skill("send_email")
    assert skill is not None
    assert skill.code == "def send(): ..."
    assert skill.meta.desc.startswith("envoie un mail")
    assert [m.name for m in store.list_skills()] == ["send_email"]


def test_nom_invalide_rejete(store: ProceduralStore) -> None:
    with pytest.raises(ValueError, match="invalide"):
        store.register_skill("../evil", "x", make_meta("../evil"))


def test_update_stats_moyenne_incrementale(
    store: ProceduralStore, fixed_clock: FixedClock
) -> None:
    store.register_skill("s", "x", make_meta("s"))
    store.update_stats("s", success=True)
    store.update_stats("s", success=False)
    [meta] = store.list_skills()
    assert meta.uses == 2
    assert meta.success_rate == 0.5
    assert meta.last_used_at == fixed_clock.now_ms()


def test_update_stats_skill_inconnu_ne_crash_pas(store: ProceduralStore) -> None:
    store.update_stats("fantome", success=True)  # log + no-op


def test_search_par_mots_cles(store: ProceduralStore) -> None:
    store.register_skill("send_email", "x", make_meta("send_email"))
    store.register_skill(
        "backup_db", "x", make_meta("backup_db", desc="sauvegarde la base de données")
    )
    results = store.search("comment envoyer un mail ?")
    assert [m.name for m in results] == ["send_email"]
    assert store.search("zzz aucun match") == []


def test_registry_persiste_entre_instances(
    tmp_path: Path, fixed_clock: FixedClock
) -> None:
    root = tmp_path / "procedural"
    ProceduralStore(root, fixed_clock).register_skill("s", "code", make_meta("s"))
    reloaded = ProceduralStore(root, fixed_clock)
    assert [m.name for m in reloaded.list_skills()] == ["s"]
