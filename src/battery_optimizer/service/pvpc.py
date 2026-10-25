"""Publication-aware PVPC price curve from the live ``sensor.esios_pvpc`` state.

The ``pvpc_updated`` integration (a fork of HA's ``pvpc_hourly_pricing``, vendoring
``aiopvpc``) publishes hourly prices as state attributes:

* ``price_HHh`` for every local hour of *today*, and — once the next day is
  published (the integration fetches it from 20:00 local, after REE's ~20:15
  release) — ``price_next_day_HHh`` for *tomorrow*;
* on a fall-back day the repeated local hour gets a ``_d`` suffix on its
  **second** occurrence (``price_02h`` = 02:00 CEST, ``price_02h_d`` = 02:00
  CET); on a spring-forward day ``price_02h`` simply does not exist.

"Today" is the local date of the state's ``last_updated`` (the integration
recomputes the attributes every hour on the hour).  This module reproduces the
integration's key rule exactly — walk the day's UTC hours in order, label each
with its local hour, suffix a repeat with ``_d`` — so no local wall-clock time is
ever converted back to UTC and DST cannot shift a price by an hour.

Prices are EUR/kWh, pretax (ADR-0003), may be zero or negative, and are mapped
onto the quarter-hour slot grid by repeating each hourly price over its four
slots (the same hourly-to-15-minute convention the offline build uses).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .entity_store import EntityState
from .timegrid import local_date, local_day_bounds_utc

__all__ = ["PriceCurve", "PriceParseError", "parse_pvpc_state", "pvpc_hour_keys"]


class PriceParseError(ValueError):
    """The price state cannot be turned into a usable curve."""


@dataclass(frozen=True)
class PriceCurve:
    """Quarter-hour prices keyed by UTC slot start."""

    interval_seconds: int
    prices: dict[datetime, float]
    source_updated_at: datetime
    reference_day: date
    tomorrow_published: bool
    expected_hours: dict[str, int]
    parsed_hours: dict[str, int]

    @property
    def start(self) -> datetime | None:
        return min(self.prices) if self.prices else None

    @property
    def end(self) -> datetime | None:
        """End of the last priced slot (exclusive)."""
        if not self.prices:
            return None
        return max(self.prices) + timedelta(seconds=self.interval_seconds)

    def price_at(self, slot_start: datetime) -> float | None:
        return self.prices.get(slot_start)

    def contiguous_from(self, start: datetime, max_slots: int | None = None) -> list[float]:
        """Prices for consecutive slots from *start* until the first gap."""
        out: list[float] = []
        step = timedelta(seconds=self.interval_seconds)
        current = start
        while current in self.prices and (max_slots is None or len(out) < max_slots):
            out.append(self.prices[current])
            current += step
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "reference_day": self.reference_day.isoformat(),
            "source_updated_at": self.source_updated_at.isoformat(),
            "tomorrow_published": self.tomorrow_published,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "slots": len(self.prices),
            "expected_hours": dict(self.expected_hours),
            "parsed_hours": dict(self.parsed_hours),
        }


def pvpc_hour_keys(day: date, tz: ZoneInfo, *, next_day: bool) -> list[tuple[str, datetime]]:
    """``(attribute_key, utc_hour_start)`` for every hour of a local day, in order."""
    prefix = "price_next_day_" if next_day else "price_"
    start, end = local_day_bounds_utc(day, tz)
    keys: list[tuple[str, datetime]] = []
    used: set[str] = set()
    current = start
    while current < end:
        key = f"{prefix}{current.astimezone(tz).hour:02d}h"
        if key in used:  # the repeated fall-back hour, exactly as aiopvpc names it
            key += "_d"
        used.add(key)
        keys.append((key, current))
        current += timedelta(hours=1)
    return keys


def _price(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except TypeError, ValueError:
        return None
    return number if math.isfinite(number) else None


def parse_pvpc_state(
    state: EntityState, tz: ZoneInfo, *, interval_seconds: int = 900
) -> PriceCurve:
    """Turn the PVPC sensor state into a :class:`PriceCurve`.

    Raises:
        PriceParseError: the state is unavailable or carries no hourly price.
    """
    if not state.available:
        raise PriceParseError(f"{state.entity_id} is {state.state!r}")
    if 3600 % interval_seconds:
        raise PriceParseError(f"interval_seconds={interval_seconds} does not divide an hour")
    today = local_date(state.last_updated, tz)
    attrs = state.attributes
    slots_per_hour = 3600 // interval_seconds
    step = timedelta(seconds=interval_seconds)
    prices: dict[datetime, float] = {}
    expected: dict[str, int] = {}
    parsed: dict[str, int] = {}
    tomorrow_published = any(str(k).startswith("price_next_day_") for k in attrs)
    for label, day, next_day in (
        ("today", today, False),
        ("tomorrow", today + timedelta(days=1), True),
    ):
        keys = pvpc_hour_keys(day, tz, next_day=next_day)
        expected[label] = len(keys)
        count = 0
        for key, hour_start in keys:
            value = _price(attrs.get(key))
            if value is None:
                continue
            count += 1
            for k in range(slots_per_hour):
                prices[hour_start + k * step] = value
        parsed[label] = count
    if not prices:
        raise PriceParseError(f"{state.entity_id} has no price_HHh attributes")
    return PriceCurve(
        interval_seconds=interval_seconds,
        prices=dict(sorted(prices.items())),
        source_updated_at=state.last_updated,
        reference_day=today,
        tomorrow_published=tomorrow_published and parsed["tomorrow"] > 0,
        expected_hours=expected,
        parsed_hours=parsed,
    )
