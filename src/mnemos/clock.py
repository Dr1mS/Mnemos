"""Horloge injectable (§6).

Tout le code applicatif (decay, archivage, valid_from/valid_until, bits
temporels du sparse coding) obtient le temps via une instance de Clock,
jamais via datetime.now() direct. Indispensable pour tester la décroissance
et le versioning temporel sans time-travel réel.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime


class Clock:
    """Horloge système. Source de vérité : epoch ms UTC (convention DB §5.1)."""

    def now_ms(self) -> int:
        return time.time_ns() // 1_000_000

    def now_dt(self) -> datetime:
        return datetime.fromtimestamp(self.now_ms() / 1000, tz=UTC)


class FixedClock(Clock):
    """Horloge de test : temps contrôlé manuellement."""

    def __init__(self, start_ms: int) -> None:
        self._now_ms = start_ms

    def now_ms(self) -> int:
        return self._now_ms

    def advance(self, ms: int) -> None:
        if ms < 0:
            raise ValueError("le temps ne recule pas")
        self._now_ms += ms
