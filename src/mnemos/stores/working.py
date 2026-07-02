"""Working memory (§11) — in-memory uniquement, pas de persistance.

Une instance par session ; le serveur maintient un registre avec éviction
LRU sur 100 sessions max.
"""

from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass

WM_MAX_ITEMS = 5
REGISTRY_MAX_SESSIONS = 100


@dataclass(frozen=True)
class WMItem:
    content: str
    role: str
    timestamp_ms: int


class WorkingMemory:
    def __init__(self, max_items: int = WM_MAX_ITEMS) -> None:
        self._items: deque[WMItem] = deque(maxlen=max_items)
        self._active_entities: set[str] = set()

    def push(self, content: str, role: str, timestamp_ms: int) -> None:
        self._items.append(WMItem(content=content, role=role, timestamp_ms=timestamp_ms))

    def add_entities(self, names: set[str]) -> None:
        self._active_entities |= names

    def get_context(self) -> list[WMItem]:
        return list(self._items)

    def get_active_entities(self) -> set[str]:
        return set(self._active_entities)

    def reset(self) -> None:
        self._items.clear()
        self._active_entities.clear()


class WorkingMemoryRegistry:
    """dict[session_id, WorkingMemory] avec éviction LRU (§11)."""

    def __init__(self, max_sessions: int = REGISTRY_MAX_SESSIONS) -> None:
        self._sessions: OrderedDict[str, WorkingMemory] = OrderedDict()
        self._max_sessions = max_sessions

    def get_or_create(self, session_id: str) -> WorkingMemory:
        if session_id in self._sessions:
            self._sessions.move_to_end(session_id)
            return self._sessions[session_id]
        wm = WorkingMemory()
        self._sessions[session_id] = wm
        if len(self._sessions) > self._max_sessions:
            evicted, _ = self._sessions.popitem(last=False)
        return wm

    def peek(self, session_id: str) -> WorkingMemory | None:
        """Lecture sans création (queries sur session inconnue)."""
        wm = self._sessions.get(session_id)
        if wm is not None:
            self._sessions.move_to_end(session_id)
        return wm

    def reset(self, session_id: str) -> bool:
        wm = self._sessions.get(session_id)
        if wm is None:
            return False
        wm.reset()
        return True

    def __len__(self) -> int:
        return len(self._sessions)
