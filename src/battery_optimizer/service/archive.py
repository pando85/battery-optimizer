"""Passive data collection: record builders and the weather-fetch schedule.

Two append-only private archives, written through :class:`~.sink.ShadowSink` next to
the slot and event records.  They are inputs for *later* offline work, never inputs of
the planner: a failure to collect or to write is counted and logged, and never changes
readiness or a plan.

* **PVPC publication snapshots** (Price Outlook PO-B): one record per state change of
  the price entity (plus the first state seen after a start), with HA's raw attributes
  verbatim.  They measure when the next day's prices really appear.
* **Weather forecast archive** (Load Forecaster P4): the hourly forecast HA holds for
  the configured weather entity, fetched with ``weather.get_forecasts`` on a fixed
  UTC grid, so a real forecast (not a persistence proxy) can be evaluated later.

This module is pure: no I/O, no clock.  The runtime owns scheduling and writing.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from typing import Any

from .entity_store import EntityState
from .timegrid import local_date

__all__ = [
    "NEXT_DAY_PREFIX",
    "PVPC_SNAPSHOT_KIND",
    "WEATHER_FORECAST_KIND",
    "next_fetch_time",
    "pvpc_snapshot_record",
    "retry_delay_s",
    "weather_forecast_record",
]

PVPC_SNAPSHOT_KIND = "pvpc_snapshot"
WEATHER_FORECAST_KIND = "weather_forecast"
NEXT_DAY_PREFIX = "price_next_day_"

_RETRY_INITIAL_S = 30.0
_RETRY_MAX_S = 600.0


def pvpc_snapshot_record(
    state: EntityState,
    *,
    observed_at: datetime,
    tz: Any,
    trigger: str,
    code_version: str,
) -> dict[str, Any]:
    """One snapshot of the price entity.

    ``observed_at`` is the service clock when the change was applied; ``last_changed``
    / ``last_updated`` / ``last_reported`` are Home Assistant's own timestamps.
    ``attributes`` is the raw attribute dict, untouched.  ``next_day_present`` is true
    when any ``price_next_day_*`` attribute exists; ``reference_day`` is the local date
    of ``last_updated`` (the day the ``price_HHh`` keys refer to).
    """
    attributes = dict(state.attributes)
    return {
        "kind": PVPC_SNAPSHOT_KIND,
        "entity_id": state.entity_id,
        "trigger": trigger,
        "observed_at": observed_at.isoformat(),
        "last_changed": state.last_changed.isoformat(),
        "last_updated": state.last_updated.isoformat(),
        "last_reported": state.last_reported.isoformat(),
        "state": state.state,
        "attributes": attributes,
        "next_day_present": any(str(key).startswith(NEXT_DAY_PREFIX) for key in attributes),
        "reference_day": local_date(state.last_updated, tz).isoformat(),
        "code_version": code_version,
    }


def weather_forecast_record(
    *,
    requested_at: datetime,
    received_at: datetime,
    entity_id: str,
    forecast_type: str,
    forecast: list[Any],
    code_version: str,
) -> dict[str, Any]:
    """One fetched forecast; ``forecast`` is HA's raw list, entries untouched."""
    return {
        "kind": WEATHER_FORECAST_KIND,
        "requested_at": requested_at.isoformat(),
        "received_at": received_at.isoformat(),
        "entity_id": entity_id,
        "type": forecast_type,
        "forecast": forecast,
        "code_version": code_version,
    }


def next_fetch_time(after: datetime, interval_minutes: int, offset_minutes: int) -> datetime:
    """First grid instant strictly after *after*.

    The grid is ``offset_minutes`` past every ``interval_minutes`` of the UTC day
    (``interval_minutes`` divides 1440, so the grid repeats identically every day).
    """
    after = after.astimezone(UTC)
    day_start = after.replace(hour=0, minute=0, second=0, microsecond=0)
    step = interval_minutes * 60
    elapsed = (after - day_start).total_seconds() - offset_minutes * 60
    k = math.floor(elapsed / step) + 1
    return day_start + timedelta(seconds=offset_minutes * 60 + k * step)


def retry_delay_s(consecutive_failures: int) -> float:
    """Capped exponential backoff after the n-th consecutive failed fetch (n >= 1)."""
    return min(_RETRY_MAX_S, _RETRY_INITIAL_S * 2 ** min(10, max(0, consecutive_failures - 1)))
