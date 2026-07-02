from __future__ import annotations

import pytest

from mnemos.clock import Clock, FixedClock


def test_clock_now_ms_monotonic_enough() -> None:
    c = Clock()
    a = c.now_ms()
    b = c.now_ms()
    assert b >= a
    assert a > 1_700_000_000_000  # epoch ms plausible (post-2023)


def test_fixed_clock_is_frozen(fixed_clock: FixedClock) -> None:
    assert fixed_clock.now_ms() == fixed_clock.now_ms()


def test_fixed_clock_advance(fixed_clock: FixedClock) -> None:
    t0 = fixed_clock.now_ms()
    fixed_clock.advance(86_400_000)  # +1 jour
    assert fixed_clock.now_ms() == t0 + 86_400_000


def test_fixed_clock_refuses_backwards(fixed_clock: FixedClock) -> None:
    with pytest.raises(ValueError, match="recule"):
        fixed_clock.advance(-1)


def test_now_dt_utc(fixed_clock: FixedClock) -> None:
    dt = fixed_clock.now_dt()
    assert dt.tzinfo is not None
    assert dt.utcoffset().total_seconds() == 0  # type: ignore[union-attr]
