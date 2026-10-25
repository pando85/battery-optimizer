"""Slot arithmetic for the resident service — UTC indexing, local labels, DST.

Every slot is a half-open UTC interval ``[start, start + interval)`` on a grid
anchored at the Unix epoch.  Local time (Europe/Madrid) is only ever a *label*
derived from a UTC instant, never the index: that is what keeps a 23-hour
spring-forward day at 92 quarter-hours and a 25-hour fall-back day at 100, with
no ambiguous or nonexistent slot (the same rule
:func:`battery_optimizer.build.make_utc_grid` applies offline).

Local wall-clock *inputs* (a PVPC ``price_02h`` attribute, a local day) are
resolved through :func:`local_day_bounds_utc`, which builds the day's UTC span
from the fold=0 local midnights, so a nonexistent local time is never
constructed and an ambiguous one is never guessed.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

__all__ = [
    "DEFAULT_INTERVAL_SECONDS",
    "floor_slot",
    "local_date",
    "local_day_bounds_utc",
    "local_day_slots",
    "next_slot_boundary",
    "require_aware",
    "slot_range",
]

DEFAULT_INTERVAL_SECONDS = 900


def require_aware(value: datetime, name: str = "timestamp") -> datetime:
    """Return *value* converted to UTC, refusing a naive datetime."""
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError(f"{name} must be timezone-aware, got naive {value!r}")
    return value.astimezone(UTC)


def floor_slot(at: datetime, interval_seconds: int = DEFAULT_INTERVAL_SECONDS) -> datetime:
    """Start of the slot containing *at* (UTC grid anchored at the epoch)."""
    at = require_aware(at, "at")
    epoch_s = int(at.timestamp())
    floored = epoch_s - (epoch_s % interval_seconds)
    return datetime.fromtimestamp(floored, tz=UTC)


def next_slot_boundary(at: datetime, interval_seconds: int = DEFAULT_INTERVAL_SECONDS) -> datetime:
    """First slot start strictly after *at*."""
    return floor_slot(at, interval_seconds) + timedelta(seconds=interval_seconds)


def slot_range(
    start: datetime, end: datetime, interval_seconds: int = DEFAULT_INTERVAL_SECONDS
) -> list[datetime]:
    """Slot starts covering ``[start, end)``; *start* is floored onto the grid."""
    start = floor_slot(start, interval_seconds)
    end = require_aware(end, "end")
    step = timedelta(seconds=interval_seconds)
    out: list[datetime] = []
    current = start
    while current < end:
        out.append(current)
        current += step
    return out


def local_date(at: datetime, tz: ZoneInfo) -> date:
    """Local calendar date of a UTC instant."""
    return require_aware(at, "at").astimezone(tz).date()


def local_day_bounds_utc(day: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """UTC ``[start, end)`` of a local calendar day.

    Local midnight is never ambiguous or nonexistent in Europe/Madrid (DST moves
    at 02:00/03:00), and ``datetime.combine(..., tzinfo=tz)`` resolves fold=0,
    so the result is the first instant whose local date is *day*.
    """
    start = datetime.combine(day, time.min, tzinfo=tz).astimezone(UTC)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=tz).astimezone(UTC)
    return start, end


def local_day_slots(
    day: date, tz: ZoneInfo, interval_seconds: int = DEFAULT_INTERVAL_SECONDS
) -> list[datetime]:
    """Every slot start of a local day: 92, 96 or 100 quarter-hours."""
    start, end = local_day_bounds_utc(day, tz)
    return slot_range(start, end, interval_seconds)
