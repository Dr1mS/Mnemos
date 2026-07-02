"""Tests WorkingMemory (§11) — taille bornée, LRU registry, reset."""

from __future__ import annotations

from mnemos.stores.working import WorkingMemory, WorkingMemoryRegistry


def test_taille_bornee_a_5() -> None:
    wm = WorkingMemory()
    for i in range(8):
        wm.push(f"m{i}", "user", i)
    context = wm.get_context()
    assert len(context) == 5
    assert [item.content for item in context] == ["m3", "m4", "m5", "m6", "m7"]


def test_reset() -> None:
    wm = WorkingMemory()
    wm.push("x", "user", 0)
    wm.add_entities({"Tom"})
    wm.reset()
    assert wm.get_context() == []
    assert wm.get_active_entities() == set()


def test_registry_get_or_create_reutilise() -> None:
    reg = WorkingMemoryRegistry()
    a = reg.get_or_create("s1")
    assert reg.get_or_create("s1") is a
    assert len(reg) == 1


def test_registry_eviction_lru() -> None:
    reg = WorkingMemoryRegistry(max_sessions=2)
    reg.get_or_create("s1")
    reg.get_or_create("s2")
    reg.get_or_create("s1")  # s1 redevient récent
    reg.get_or_create("s3")  # évince s2 (LRU)
    assert reg.peek("s2") is None
    assert reg.peek("s1") is not None
    assert len(reg) == 2


def test_registry_reset_idempotent() -> None:
    reg = WorkingMemoryRegistry()
    assert reg.reset("inconnue") is False
    wm = reg.get_or_create("s1")
    wm.push("x", "user", 0)
    assert reg.reset("s1") is True
    assert wm.get_context() == []


def test_peek_ne_cree_pas() -> None:
    reg = WorkingMemoryRegistry()
    assert reg.peek("jamais-vue") is None
    assert len(reg) == 0
