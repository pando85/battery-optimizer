"""OMIE-based next-day price estimators (PO-C of the price-outlook improvement plan).

At the 14:00 decision tomorrow's PVPC is not published (REE releases it at about 20:15), but
OMIE's day-ahead marginal price for tomorrow is public from about 13:00.  PVPC is, to a good
approximation, an affine function of the OMIE price whose level depends on the 2.0TD tariff
period (the regulated tolls and charges differ between P1, P2 and P3, and the energy term is a
blend of the day-ahead price and slowly moving futures prices).  The estimators here fit that
relation on the **published** days just before the decision and apply it to tomorrow's OMIE.

* ``omie_period``: ``PVPC_hour = beta * mean(OMIE ES over the hour) + intercept(period)``, one
  slope and one intercept per tariff period, fitted by least squares on the hourly prices of the
  last ``window_days`` fully published local days (walk-forward: nothing after the decision is
  read, the fit window ends at the last published day).
* ``omie_period_month``: the same with one extra level offset per calendar month present in the
  window (a longer window), predicted at the level of the **latest** month of the window.  It
  tracks drift of the regulated and futures components over the window instead of averaging it.

Both are non-oracle outlooks registered with ``needs_omie=True`` and
``inputs_usable_from=OMIE_NEXT_DAY_USABLE_FROM`` (13:30 local): the seam does not run them
before that time of day and hands them only the OMIE prices usable at the decision (see
:mod:`battery_optimizer.price_outlook`).  They answer only the slots of whole local days they
can price: a target hour whose OMIE price is missing in any quarter, a missing side input or a
fit window that is too short leaves the slot ``unavailable`` (the ``published_only`` fallback),
and the reason is returned in the diagnostics.

``omie_quantile`` (P10/P50/P90 residual quantiles) from the plan is **not** built: it exists to
feed PO-D (scenario prices), which stays deferred after the failed gate GA.

**Tariff periods.**  2.0TD: P3 (valle) 00-08 every day and all day on weekends and on the
national fixed-date holidays; P1 (punta) 10-14 and 18-22 on other days; P2 (llano) otherwise.
The national holidays are the nine fixed dates of the CNMC circular (1 Jan, 6 Jan, 1 May,
15 Aug, 12 Oct, 1 Nov, 6 Dec, 8 Dec, 25 Dec); regional and movable holidays are not part of the
period definition.

**Choice of the fit window.**  ``window_days`` of both estimators was chosen on **2024 data
only** and recorded before the 2025 and 2026 evaluation was run: see
``docs/planning/price-outlook-improvement-plan.md`` (PO-C).
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from .price_outlook import (
    OMIE_NEXT_DAY_USABLE_FROM,
    OutlookRequest,
    OutlookSide,
    OutlookSlots,
    register_price_outlook,
)

__all__ = [
    "NATIONAL_HOLIDAYS",
    "OMIE_PERIOD_WINDOW_DAYS",
    "OMIE_PERIOD_MONTH_WINDOW_DAYS",
    "load_omie_side",
    "tariff_period",
]

#: Fit window (published days) of ``omie_period``; chosen on 2024 data only.
OMIE_PERIOD_WINDOW_DAYS = 28
#: Fit window of ``omie_period_month``; chosen on 2024 data only.
OMIE_PERIOD_MONTH_WINDOW_DAYS = 120
#: Fewer fully published days than this in the window: no fit.
MIN_FIT_DAYS = 7
#: A tariff period with fewer fit rows than this makes the fit unusable (an intercept would
#: be undetermined).
MIN_PERIOD_ROWS = 6

#: ``(month, day)`` of the national fixed-date holidays that count as P3 all day (2.0TD).
NATIONAL_HOLIDAYS: frozenset[tuple[int, int]] = frozenset(
    {(1, 1), (1, 6), (5, 1), (8, 15), (10, 12), (11, 1), (12, 6), (12, 8), (12, 25)}
)

_PERIODS = (1, 2, 3)


def tariff_period(day: date, local_hour: int) -> int:
    """The 2.0TD period (1 punta, 2 llano, 3 valle) of a local clock hour of *day*."""
    if day.weekday() >= 5 or (day.month, day.day) in NATIONAL_HOLIDAYS or local_hour < 8:
        return 3
    if 10 <= local_hour < 14 or 18 <= local_hour < 22:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Calendar of the grid (calendar knowledge only, cached for the last grid used)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Calendar:
    starts: Sequence[datetime]
    tz: ZoneInfo
    hour_id: np.ndarray
    """UTC epoch hour of every interval; consecutive on a uniform grid."""
    period: np.ndarray
    """2.0TD period (1..3) of every interval."""
    month: np.ndarray
    """Local ``year * 12 + month - 1`` of every interval."""


_CALENDAR: _Calendar | None = None


def _as_utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _calendar(starts: Sequence[datetime], tz: ZoneInfo) -> _Calendar:
    global _CALENDAR  # noqa: PLW0603 - one-entry cache keyed on the identity of the grid
    cached = _CALENDAR
    if cached is not None and cached.starts is starts and cached.tz == tz:
        return cached
    seconds = np.array([_as_utc(s).timestamp() for s in starts], dtype=np.float64)
    hour_id = np.floor(seconds / 3600.0).astype(np.int64)
    unique, inverse = np.unique(hour_id, return_inverse=True)
    period = np.empty(unique.size, dtype=np.int8)
    month = np.empty(unique.size, dtype=np.int64)
    for k, hour in enumerate(unique.tolist()):
        local = datetime.fromtimestamp(hour * 3600, tz=UTC).astimezone(tz)
        period[k] = tariff_period(local.date(), local.hour)
        month[k] = local.year * 12 + local.month - 1
    built = _Calendar(starts, tz, hour_id, period[inverse], month[inverse])
    _CALENDAR = built
    return built


# ---------------------------------------------------------------------------
# Hourly aggregation and the fit
# ---------------------------------------------------------------------------


def _hourly(values: np.ndarray, hour_rel: np.ndarray, n_hours: int, per_hour: int) -> np.ndarray:
    """Mean of the *per_hour* quarter values of each hour; ``NaN`` unless all are known."""
    finite = np.isfinite(values)
    total = np.bincount(hour_rel, weights=np.where(finite, values, 0.0), minlength=n_hours)
    count = np.bincount(hour_rel, weights=finite.astype(float), minlength=n_hours)
    return np.where(count >= per_hour, total / per_hour, np.nan)


def _fit_design(
    omie: np.ndarray, period: np.ndarray, month_dummies: np.ndarray | None
) -> np.ndarray:
    cols = [omie, *((period == p).astype(float) for p in _PERIODS)]
    matrix = np.column_stack(cols)
    if month_dummies is not None and month_dummies.size:
        matrix = np.column_stack([matrix, month_dummies])
    return matrix


def _estimate(request: OutlookRequest, *, window_days: int, by_month: bool) -> OutlookSlots:
    n = request.target_idx.size
    none = np.full(n, np.nan), np.zeros(n, dtype=bool)

    def fallback(reason: str, **extra: Any) -> OutlookSlots:
        return (*none, {"fallback": reason, **extra})

    if int(window_days) < MIN_FIT_DAYS:
        raise ValueError(f"window_days must be >= {MIN_FIT_DAYS}, got {window_days}")
    omie = request.omie_eur_per_kwh
    if omie is None:
        return fallback("omie_side_input_missing")
    cal = _calendar(request.interval_starts_utc, request.tz)
    known = int(request.published_end_idx)
    target = np.asarray(request.target_idx, dtype=np.int64)
    if int(target[-1]) >= int(request.omie_end_idx):
        # The seam only runs the outlook once the next day's OMIE is usable, but a plan that
        # reaches a day further out has no OMIE price for it.
        usable = target < int(request.omie_end_idx)
    else:
        usable = np.ones(n, dtype=bool)

    # Fit window: the last ``window_days`` local days that are fully published.
    his = [int(d[2]) for d in request.days]
    j_end = bisect.bisect_right(his, known)  # days [0, j_end) end at or before `known`
    j_start = max(0, j_end - int(window_days))
    if j_end - j_start < MIN_FIT_DAYS:
        return fallback("fit_window_short", fit_days=int(j_end - j_start))
    lo = int(request.days[j_start][1])
    hi = int(max(target[usable].max() + 1 if usable.any() else known, known))
    hi = min(hi, cal.hour_id.size)
    hour_rel = cal.hour_id[lo:hi] - cal.hour_id[lo]
    n_hours = int(hour_rel[-1]) + 1
    per_hour = int(np.bincount(hour_rel).max())

    pvpc = np.full(hi - lo, np.nan)
    pvpc[: known - lo] = request.published_prices[lo:known]
    omie_window = np.full(hi - lo, np.nan)
    m = min(int(omie.size), hi) - lo
    if m > 0:
        omie_window[:m] = omie[lo : lo + m]
    pvpc_h = _hourly(pvpc, hour_rel, n_hours, per_hour)
    omie_h = _hourly(omie_window, hour_rel, n_hours, per_hour)
    first_of_hour = np.searchsorted(hour_rel, np.arange(n_hours))
    period_h = cal.period[lo:hi][first_of_hour]
    month_h = cal.month[lo:hi][first_of_hour]
    hour_slot0 = lo + first_of_hour

    rows = np.isfinite(pvpc_h) & np.isfinite(omie_h) & (hour_slot0 < known)
    if int(rows.sum()) < MIN_FIT_DAYS * 12:
        return fallback("fit_rows_short", fit_rows=int(rows.sum()))
    for p in _PERIODS:
        if int((rows & (period_h == p)).sum()) < MIN_PERIOD_ROWS:
            return fallback("period_not_in_fit_window", period=p)

    month_dummies = None
    months: list[int] = []
    if by_month:
        months = sorted({int(x) for x in month_h[rows]})
        reference = months[-1]
        others = [x for x in months if x != reference]
        if others:
            month_dummies = np.column_stack([(month_h == x).astype(float) for x in others])
    x = _fit_design(
        omie_h[rows], period_h[rows], None if month_dummies is None else month_dummies[rows]
    )
    coef, *_ = np.linalg.lstsq(x, pvpc_h[rows], rcond=None)
    if not np.isfinite(coef).all():
        return fallback("fit_failed")
    beta, intercepts = float(coef[0]), coef[1:4]

    # Predict the target hours at the reference level (the latest month of the window).
    t_hour = np.clip(cal.hour_id[target] - cal.hour_id[lo], 0, n_hours - 1)
    omie_t = omie_h[t_hour]
    period_t = cal.period[target]
    ok = usable & np.isfinite(omie_t)
    est = beta * omie_t + intercepts[period_t - 1]
    est = np.where(ok, est, np.nan)
    diagnostics: dict[str, Any] = {
        "fit_days": int(j_end - j_start),
        "fit_rows": int(rows.sum()),
        "beta": round(beta, 4),
        "intercepts": [round(float(v), 4) for v in intercepts],
    }
    if by_month:
        diagnostics["fit_months"] = len(months)
    if not ok.all():
        diagnostics["fallback"] = (
            "omie_missing_for_target_hours"
            if usable.all()
            else "plan_reaches_beyond_usable_omie_day"
        )
        diagnostics["slots_without_omie"] = int((~ok).sum())
    return est, ok, diagnostics


@register_price_outlook(
    "omie_period",
    inputs_usable_from=OMIE_NEXT_DAY_USABLE_FROM,
    needs_omie=True,
    description=(
        "PVPC = beta * OMIE + intercept per 2.0TD period, fitted on the trailing published "
        "days; OMIE for D+1 from 13:30 local on D"
    ),
)
def _omie_period(
    request: OutlookRequest, *, window_days: int = OMIE_PERIOD_WINDOW_DAYS
) -> OutlookSlots:
    """Tomorrow's PVPC from tomorrow's OMIE and the relation fitted on recent published days."""
    return _estimate(request, window_days=int(window_days), by_month=False)


@register_price_outlook(
    "omie_period_month",
    inputs_usable_from=OMIE_NEXT_DAY_USABLE_FROM,
    needs_omie=True,
    description=(
        "omie_period with a level offset per calendar month of a longer window, predicted at "
        "the latest month's level (tracks drift of the regulated and futures components)"
    ),
)
def _omie_period_month(
    request: OutlookRequest, *, window_days: int = OMIE_PERIOD_MONTH_WINDOW_DAYS
) -> OutlookSlots:
    """As ``omie_period`` with month terms, to follow drift inside a longer fit window."""
    return _estimate(request, window_days=int(window_days), by_month=True)


# ---------------------------------------------------------------------------
# Side input
# ---------------------------------------------------------------------------


def load_omie_side(
    interval_starts_utc: Sequence[datetime],
    *,
    data_dir: Path | str | None = None,
    tz: ZoneInfo | None = None,
) -> OutlookSide:
    """The OMIE ES series (EUR/kWh) on the dataset grid, from the verified OMIE store.

    Reads through the successful ``omie`` manifests with full hash verification
    (:func:`battery_optimizer.omie_prices.load_omie_es_prices`), covering every local delivery
    day of the grid.  A slot without a stored price is ``NaN``; the as-of rule is **not**
    applied here, the price-outlook seam applies it per decision.

    Raises:
        ValueError: the grid is not a uniform 15-minute UTC grid.
        battery_optimizer.omie_prices.OmieReadError: a stored part is missing or altered.
    """
    from .extract_omie import MARKET_TZ
    from .omie_prices import SLOT, load_omie_es_prices

    if not interval_starts_utc:
        raise ValueError("the dataset grid is empty")
    zone = tz or MARKET_TZ
    starts = [_as_utc(s) for s in interval_starts_utc]
    steps = {(b - a) for a, b in zip(starts, starts[1:], strict=False)}
    if steps and steps != {SLOT}:
        raise ValueError("the OMIE side input needs a uniform 15-minute dataset grid")
    first = starts[0].astimezone(zone).date()
    last = (starts[-1] + SLOT - timedelta(seconds=1)).astimezone(zone).date()
    omie = load_omie_es_prices(data_dir, start_date=first, end_date=last)
    index = {s: i for i, s in enumerate(omie.slot_start)}
    values = np.full(len(starts), np.nan)
    for k, s in enumerate(starts):
        i = index.get(s)
        if i is not None:
            values[k] = omie.price_es_eur_mwh[i] / 1000.0
    return OutlookSide(
        values,
        source={
            "kind": "omie_es_day_ahead",
            "unit": "EUR/kWh",
            "delivery_dates": [first.isoformat(), last.isoformat()],
            "missing_dates": [d.isoformat() for d in omie.missing_dates],
            "slots_without_price": int(np.isnan(values).sum()),
        },
    )
