"""Injectable clock: the service reads wall time only through this seam.

The planner and dispatch modules take time as an argument and never read a clock;
the resident service is the one place that must, so it does so through a
:class:`Clock` that tests replace with :class:`FakeClock`.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Protocol

__all__ = ["Clock", "FakeClock", "SystemClock"]


class Clock(Protocol):
    def now(self) -> datetime:
        """Current aware-UTC instant."""
        ...

    async def sleep(self, seconds: float) -> None:
        """Suspend for *seconds* (non-negative)."""
        ...


class SystemClock:
    """Real wall clock and real ``asyncio.sleep``."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(max(0.0, seconds))


class FakeClock:
    """Manually advanced clock for tests.

    ``sleep`` advances the fake time by the requested amount and yields once to
    the event loop, so a loop that sleeps progresses deterministically without
    real waiting.
    """

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock start must be timezone-aware")
        self._now = start.astimezone(UTC)
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)

    def set(self, at: datetime) -> None:
        self._now = at.astimezone(UTC)

    async def sleep(self, seconds: float) -> None:
        seconds = max(0.0, seconds)
        self.sleeps.append(seconds)
        self._now += timedelta(seconds=seconds)
        await asyncio.sleep(0)
