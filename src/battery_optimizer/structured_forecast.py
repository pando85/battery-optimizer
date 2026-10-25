"""Structured component load forecaster (LF-1, P4 of the load-forecaster improvement plan).

``forecast = heat pump (climate + hot water) + residual``.  The heat pump carries most of
LF-0's error and, unlike the rest of the household, its schedule is a **rule**: AppDaemon's
``climate_control`` switches the unit on in the hours priced below two thirds of the
trailing mean price (or the cheapest ``min_hours_per_day`` hours when fewer qualify), and
``dhw_control`` forces hot water in the cheapest hour of each 12-hour block.  So *when* the
heat pump runs is known from prices, and the forecaster only has to say *how much* it draws
in those hours.

Components
----------
* **P4a, schedule mask** (:func:`replay_climate_rule`): an exact replay of ``climate_control``
  for each target day.  The rule is versioned (:data:`CLIMATE_RULE_VERSIONS`), the heat or
  cool mode is inferred from the last 48 h of measured heat against cool energy (the mode
  select has no history), and the enable flag and ``min_hours_per_day`` are read as of the
  decision.  Prices are strictly as-of: a day whose PVPC is public at the decision uses the
  published prices; any other day uses the ``price_outlook`` param (default ``persistence``,
  the last published day) and is recorded as ``estimated``.
* **P4b, energy per selected hour**: one ridge regression per mode on the as-of outdoor
  temperature (persistence: the same clock hour of the last day it was measured, usually
  yesterday), the position inside the block of selected hours, the number of selected hours
  and a start-up term.  It is refit once per monthly walk-forward fold on history before the
  fold (the seam's ``fit`` step).  Outside the mask the climate energy is a small constant.
* **P4c, hot water**: the ``dhw_control`` replay places the forced hours; the energy is the
  trailing mean by (hours since the last forced hour, quarter of the hour) over the last
  ``window_days`` days.
* **P4d, residual**: the **residual forecaster** (param, default ``trailing_mean``) forecasts
  ``main - heat pump`` through the same seam, so nothing is counted twice.
* **P4e, fallbacks**: a disabled rule, an unknown mode or flag, prices or temperature missing,
  or no fitted model make the climate part fall back to LF-0 (the trailing mean of the climate
  series).  The hot-water part falls back the same way.  Every fallback and its reason is
  recorded per target day.  Without side inputs the forecast is LF-0 of the main load.
* **Second look (P4 second pass, all off by default so the first P4 results reproduce)**:
  ``price_outlook="omie_period"`` replays the schedule of tomorrow on the OMIE-based price
  estimate (PO-C) instead of on ``persistence``; ``disabled_fallback="disabled_history"``
  forecasts the climate part of a day whose rule is disabled from the recent *disabled-state*
  days instead of from the trailing mean of every day (which still carries the climate energy of
  the enabled days before it); ``climate_lf0_season="heating"`` is the seasonal hybrid: on the
  days of that season the climate part is the LF-0 trailing mean even when the rule is enabled.

All information is as-of the cutoff of the seam: this forecaster declares an intraday cutoff,
so the 14:00 and 20:30 events see the slots of their own day before the event.  Prices of the
next local day count as public from 20:15 local, so the 14:00 event plans tomorrow on
estimated prices and the 20:30 event on published ones.

The AppDaemon source of truth is the ``homelab`` repository (``apps/home-assistant/appdaemon``);
the commits named in the version tables below are the ones whose behaviour each version mirrors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from .load_forecast import (
    DEFAULT_LOAD_FORECASTER,
    DayForecast,
    FitRequest,
    ForecastRequest,
    SideInputs,
    SideView,
    fold_state,
    get_forecaster,
    register_forecaster,
    resolve_forecaster_params,
    run_forecaster,
)
from .policies import DEFAULT_FORECAST_WINDOW_DAYS, trailing_slot_mean_for_day
from .price_outlook import (
    SLOT_ESTIMATED,
    OutlookSide,
    get_price_outlook,
    run_price_outlook,
)

__all__ = [
    "CLIMATE_RULE_VERSIONS",
    "DHW_RULE_VERSIONS",
    "RULE_MIRROR",
    "ClimateRuleVersion",
    "COOLING_MONTHS",
    "HEATING_MONTHS",
    "ClimateSchedule",
    "DhwRuleVersion",
    "block_positions",
    "infer_mode",
    "replay_climate_rule",
    "replay_dhw_rule",
    "replay_mask_for_day",
]

# ---------------------------------------------------------------------------
# The AppDaemon rules, versioned
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClimateRuleVersion:
    """One behaviour of ``climate_control``, effective from a local day."""

    label: str
    effective_from: date
    history_days: int
    """Days of PVPC state history whose mean sets the cheap limit (``get_history(days=...)``)."""
    accepts_negative_prices: bool
    """Before this was true a negative price raised and no schedule was registered."""
    commit: str
    """The homelab commit whose ``climate_control.py`` this version mirrors."""


@dataclass(frozen=True)
class DhwRuleVersion:
    """One behaviour of ``dhw_control``, effective from a local day."""

    label: str
    effective_from: date
    interval_hours: int
    accepts_negative_prices: bool
    commit: str


#: ``climate_control`` since its first release.  The 10-day window became 7 days when the
#: history call gained retries; negative prices were accepted from the next change.
CLIMATE_RULE_VERSIONS: tuple[ClimateRuleVersion, ...] = (
    ClimateRuleVersion("climate_control@fda18eb2", date(2023, 11, 25), 10, False, "fda18eb2"),
    ClimateRuleVersion("climate_control@e734f5e9", date(2026, 4, 28), 7, False, "e734f5e9"),
    ClimateRuleVersion("climate_control@5af36d49", date(2026, 9, 5), 7, True, "5af36d49"),
)

#: ``dhw_control``: absent before 2025-03-16, one forced hour per day until the interval
#: became configurable (12 h from 2025-12-29), negative prices accepted from 2026-09-05.
DHW_RULE_VERSIONS: tuple[DhwRuleVersion, ...] = (
    DhwRuleVersion("dhw_control@363047d3", date(2025, 3, 16), 24, False, "363047d3"),
    DhwRuleVersion("dhw_control@153d7f88", date(2025, 12, 29), 12, False, "153d7f88"),
    DhwRuleVersion("dhw_control@5af36d49", date(2026, 9, 5), 12, True, "5af36d49"),
)

#: Recorded in the pre-registration and the forecaster provenance.
RULE_MIRROR = {
    "source": "homelab repository, apps/home-assistant/appdaemon/apps",
    "climate_control": [
        {
            "label": v.label,
            "from": v.effective_from.isoformat(),
            "history_days": v.history_days,
            "accepts_negative_prices": v.accepts_negative_prices,
        }
        for v in CLIMATE_RULE_VERSIONS
    ],
    "dhw_control": [
        {
            "label": v.label,
            "from": v.effective_from.isoformat(),
            "interval_hours": v.interval_hours,
            "accepts_negative_prices": v.accepts_negative_prices,
        }
        for v in DHW_RULE_VERSIONS
    ],
    "cheap_limit_fraction_of_trailing_mean": "2/3",
    "decision_time_local": "00:00:40",
}

#: Local months of the pre-registered seasons (gate G4; the shoulder months Apr, May and Oct
#: belong to neither).  ``forecast_eval.G4_SEASONS`` repeats them and a test pins that.
HEATING_MONTHS = (11, 12, 1, 2, 3)
COOLING_MONTHS = (6, 7, 8, 9)
_SEASON_MONTHS = {"": (), "heating": HEATING_MONTHS, "cooling": COOLING_MONTHS}
DISABLED_FALLBACKS = ("lf0", "disabled_history")
#: A prior day counts as a disabled-state sample only with at least this share of its slots usable.
DISABLED_MIN_KNOWN_FRACTION = 0.9

SLOTS_PER_HOUR = 4
#: The mode is read from the last 48 h of measured heat and cool energy.
MODE_WINDOW_SLOTS = 192
MODE_MIN_KNOWN_FRACTION = 0.75
MODE_MIN_KWH = 0.2
#: A trailing mean needs at least a day of hourly prices behind it.
MIN_TRAILING_HOURS = 24
#: Fewer fitted rows than ten selected days of twelve hours: no model.
MIN_FIT_ROWS = 480
#: Fewer eligible history days than this: no hot-water profile.
MIN_DHW_DAYS = 7
DHW_MAX_K = 12
#: Look back this many slots (two days) for the last known control state.
FLAG_LOOKBACK_SLOTS = 192
#: Outdoor temperature persistence looks at most this many days back.
TEMPERATURE_MAX_LAG_DAYS = 3
RIDGE = 1e-3
HEAT_KNOTS = (6.0, 11.0, 16.0)
COOL_KNOTS = (26.0, 30.0, 34.0)

#: Side series names the structured forecaster reads.
SERIES = (
    "hp_heat_kwh",
    "hp_cool_kwh",
    "hp_dhw_kwh",
    "outdoor_temp_c",
    "climate_enable",
    "climate_min_hours",
    "dhw_enable",
)
FEATURE_COLUMNS = {
    "hp_heat_kwh": "hp_heat_kwh",
    "hp_cool_kwh": "hp_cool_kwh",
    "hp_dhw_kwh": "hp_dhw_kwh",
    "outdoor_temp_c": "outdoor_temp_mean_c",
    "climate_enable": "climate_enable_at_start",
    "climate_min_hours": "climate_min_hours_at_start",
    "dhw_enable": "dhw_enable_at_start",
}


def _version_for(versions: Sequence[Any], day: date) -> Any | None:
    chosen = None
    for version in versions:
        if version.effective_from <= day:
            chosen = version
    return chosen


def _as_date(label: Any) -> date:
    return label if isinstance(label, date) else date.fromisoformat(str(label)[:10])


# ---------------------------------------------------------------------------
# Rule replay (pure functions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClimateSchedule:
    """The hours ``climate_control`` selects for one local day."""

    selected: np.ndarray
    """``bool[24]`` by local clock hour."""

    cheap_branch: bool
    """``True`` when at least ``min_hours`` hours were cheaper than the limit."""

    limit: float
    """The cheap-price limit: two thirds of the trailing mean."""

    @property
    def hours(self) -> int:
        return int(self.selected.sum())


def replay_climate_rule(
    hour_prices: np.ndarray,
    present: np.ndarray,
    *,
    trailing_mean: float,
    min_hours: int,
) -> ClimateSchedule | None:
    """Replay ``climate_control._register_schedulers`` for one local day.

    Args:
        hour_prices: PVPC price of each local clock hour (``price_{h:02d}h``), ``NaN`` where
            the hour does not exist or is unknown.
        present: ``bool[24]``, the clock hours the day has (23 on a spring-forward day).
        trailing_mean: the mean of the PVPC state history over the version's window.
        min_hours: ``input_number.appdaemon_climate_min_hours_per_day``.

    Returns:
        The selection, or ``None`` when the day has fewer priced hours than *min_hours*
        (the app then starts the unit and retries, which is not a schedule).  Hours priced
        strictly below ``2/3 * trailing_mean`` are selected when there are at least
        *min_hours* of them; otherwise the *min_hours* cheapest, ties by clock hour.
    """
    prices = np.asarray(hour_prices, dtype=float)
    valid = np.asarray(present, dtype=bool) & np.isfinite(prices)
    wanted = max(int(min_hours), 0)
    if int(valid.sum()) < wanted:
        return None
    limit = float(trailing_mean) * 2.0 / 3.0
    cheap = valid & (prices < limit)
    selected = np.zeros(24, dtype=bool)
    if int(cheap.sum()) >= wanted:
        selected = cheap
        branch = True
    else:
        hours = np.flatnonzero(valid)
        order = np.lexsort((hours, prices[hours]))
        selected[hours[order[:wanted]]] = True
        branch = False
    return ClimateSchedule(selected=selected, cheap_branch=branch, limit=limit)


def replay_dhw_rule(
    hour_prices: np.ndarray, present: np.ndarray, *, interval_hours: int
) -> tuple[int, ...]:
    """Replay ``dhw_control``: the cheapest hour of each *interval_hours* block of the day.

    The day's priced hours are cut into consecutive chunks of *interval_hours* (the last one
    shorter on a 23-hour day) and the first cheapest hour of each chunk is forced.
    """
    prices = np.asarray(hour_prices, dtype=float)
    hours = [h for h in range(24) if present[h] and np.isfinite(prices[h])]
    chunks = [hours[i : i + interval_hours] for i in range(0, len(hours), interval_hours)]
    return tuple(min(chunk, key=lambda h: float(prices[h])) for chunk in chunks if chunk)


def block_positions(selected: np.ndarray) -> np.ndarray:
    """Position of each selected clock hour inside its block (0, 1, 2 = third or later).

    ``-1`` for an hour that is not selected.  A block is a run of consecutive clock hours,
    which is how ``climate_control`` groups the hours it schedules.
    """
    positions = np.full(24, -1, dtype=np.int64)
    run = -1
    for hour in range(24):
        if selected[hour]:
            run = run + 1 if hour > 0 and selected[hour - 1] else 0
            positions[hour] = min(run, 2)
        else:
            run = -1
    return positions


def infer_mode(heat: np.ndarray, cool: np.ndarray, cutoff: int) -> tuple[str | None, str]:
    """``("heat" | "cool", "")`` from the last 48 h before *cutoff*, else ``(None, reason)``.

    The heat or cool select of ``climate_control`` has no history, so the mode is the one
    whose measured energy dominates the last 48 h.  Too many unknown slots, or hardly any
    climate energy at all, leave it unknown.
    """
    start = max(0, int(cutoff) - MODE_WINDOW_SLOTS)
    h = np.asarray(heat[start:cutoff], dtype=float)
    c = np.asarray(cool[start:cutoff], dtype=float)
    if h.size < MODE_WINDOW_SLOTS:
        return None, "mode_history_short"
    known = np.isfinite(h) & np.isfinite(c)
    if known.mean() < MODE_MIN_KNOWN_FRACTION:
        return None, "mode_data_missing"
    heat_kwh = float(h[known].sum())
    cool_kwh = float(c[known].sum())
    if heat_kwh + cool_kwh < MODE_MIN_KWH:
        return None, "mode_no_recent_climate_energy"
    return ("cool" if cool_kwh > heat_kwh else "heat"), ""


# ---------------------------------------------------------------------------
# As-of inputs
# ---------------------------------------------------------------------------


def _asof_flag(series: np.ndarray, lo: int, cut: int) -> float:
    """The control state at the day's first slot if known by *cut*, else the last known one."""
    if cut <= 0:
        return float("nan")
    i = min(int(lo), cut - 1)
    window = series[max(0, i - FLAG_LOOKBACK_SLOTS + 1) : i + 1]
    finite = np.flatnonzero(np.isfinite(window))
    return float(window[finite[-1]]) if finite.size else float("nan")


def _hourly_mean(series: np.ndarray, real_hours: int) -> np.ndarray:
    """Mean of the known quarters of each real hour (at least two), ``NaN`` otherwise."""
    padded = np.full(SLOTS_PER_HOUR * real_hours, np.nan)
    m = min(series.size, padded.size)
    padded[:m] = series[:m]
    rows = padded.reshape(real_hours, SLOTS_PER_HOUR)
    finite = np.isfinite(rows)
    count = finite.sum(axis=1)
    total = np.where(finite, rows, 0.0).sum(axis=1)
    return np.where(count >= 2, total / np.maximum(count, 1), np.nan)


def _persisted_temperature(
    view: SideView, hourly_temp: np.ndarray, slots: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Outdoor temperature persistence for *slots*: the same local clock hour, as measured.

    Looks one, two, then three local days back for the most recent day whose hour was
    measured by the cutoff.  Returns ``(temperature, lag_days)``; ``NaN`` and ``0`` when
    nothing was measured.
    """
    real = slots // SLOTS_PER_HOUR
    day = view.day_of_real_hour[real]
    clock = view.local_hour[real * SLOTS_PER_HOUR]
    out = np.full(slots.size, np.nan)
    lag = np.zeros(slots.size, dtype=np.int64)
    for back in range(1, TEMPERATURE_MAX_LAG_DAYS + 1):
        source_day = day - back
        ok = source_day >= 0
        source = view.first_real_hour[np.clip(source_day, 0, None), clock]
        ok &= (source >= 0) & (source < hourly_temp.size)
        value = np.where(ok, hourly_temp[np.clip(source, 0, max(hourly_temp.size - 1, 0))], np.nan)
        take = np.isnan(out) & np.isfinite(value)
        out[take] = value[take]
        lag[take] = back
    return out, lag


class _PriceHours:
    """Hourly price access over a working price array (published, then estimated)."""

    def __init__(self, view: SideView, days: Sequence[tuple[Any, int, int]], prices: np.ndarray):
        self.view = view
        self.days = days
        real = prices.size // SLOTS_PER_HOUR
        rows = prices[: SLOTS_PER_HOUR * real].reshape(real, SLOTS_PER_HOUR)
        finite = np.isfinite(rows)
        count = finite.sum(axis=1)
        hourly = np.where(
            count > 0, np.where(finite, rows, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan
        )
        valid = np.isfinite(hourly)
        self.hourly = hourly
        self._sum = np.concatenate(([0.0], np.cumsum(np.where(valid, hourly, 0.0))))
        self._count = np.concatenate(([0], np.cumsum(valid)))

    def clock_prices(self, j: int) -> tuple[np.ndarray, np.ndarray]:
        """``(price[24], present[24])`` of local day *j* by clock hour; first pass of a repeat."""
        first = self.view.first_real_hour[j]
        present = first >= 0
        inside = present & (first < self.hourly.size)
        prices = np.where(
            inside, self.hourly[np.clip(first, 0, max(self.hourly.size - 1, 0))], np.nan
        )
        return prices, present

    def trailing_mean(self, j: int, history_days: int) -> float | None:
        """Mean hourly price from *history_days* days before day *j* to day *j*'s first hour."""
        first = self.days[max(0, j - history_days)][1] // SLOTS_PER_HOUR
        last = self.days[j][1] // SLOTS_PER_HOUR
        if last >= self.hourly.size:
            return None
        count = int(self._count[last + 1] - self._count[first])
        if count < MIN_TRAILING_HOURS:
            return None
        return float((self._sum[last + 1] - self._sum[first]) / count)


# ---------------------------------------------------------------------------
# Climate energy model (P4b)
# ---------------------------------------------------------------------------


def _design(
    mode: str, temp: np.ndarray, position: np.ndarray, quarter: np.ndarray, hours: int
) -> np.ndarray:
    """Design matrix of the per-slot climate regression (without the intercept column)."""
    knots = HEAT_KNOTS if mode == "heat" else COOL_KNOTS
    first = position == 0
    cols = [temp, *(np.maximum(0.0, temp - k) for k in knots)]
    cols += [(first & (quarter == q)).astype(float) for q in range(4)]
    cols += [(position == 1).astype(float), first * temp, np.full(temp.size, hours / 12.0)]
    return np.column_stack(cols)


@dataclass
class _ClimateModel:
    mode: str
    intercept: float
    beta: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    out_of_mask_kwh: float
    cap_kwh: float
    rows: int
    days: int

    def predict(self, features: np.ndarray) -> np.ndarray:
        z = (features - self.mean) / self.scale
        return np.clip(self.intercept + z @ self.beta, 0.0, self.cap_kwh)


def _fit_ridge(
    mode: str, x: np.ndarray, y: np.ndarray, out_of_mask: np.ndarray, days: int
) -> _ClimateModel:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = (x - mean) / scale
    design = np.column_stack([np.ones(z.shape[0]), z])
    penalty = np.eye(design.shape[1]) * RIDGE * z.shape[0]
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return _ClimateModel(
        mode=mode,
        intercept=float(coef[0]),
        beta=coef[1:],
        mean=mean,
        scale=scale,
        out_of_mask_kwh=float(out_of_mask.mean()) if out_of_mask.size else 0.0,
        cap_kwh=float(np.quantile(y, 0.995)),
        rows=int(y.size),
        days=int(days),
    )


def _climate_schedule_for_day(
    *,
    j: int,
    view: SideView,
    days: Sequence[tuple[Any, int, int]],
    prices: _PriceHours,
    min_hours: float,
) -> tuple[ClimateSchedule | None, str, str]:
    """``(schedule, reason, version label)`` of the replayed rule for local day *j*."""
    label, _lo, _hi = days[j]
    version = _version_for(CLIMATE_RULE_VERSIONS, _as_date(label))
    if version is None:
        return None, "rule_not_deployed", ""
    hour_prices, present = prices.clock_prices(j)
    if not np.isfinite(hour_prices[present]).all():
        return None, "prices_incomplete", version.label
    if not version.accepts_negative_prices and (hour_prices[present] < 0).any():
        return None, "rule_error_negative_price", version.label
    mean = prices.trailing_mean(j, version.history_days)
    if mean is None:
        return None, "price_history_short", version.label
    schedule = replay_climate_rule(
        hour_prices, present, trailing_mean=mean, min_hours=int(min_hours)
    )
    if schedule is None:
        return None, "prices_incomplete", version.label
    return schedule, "", version.label


def disabled_state_profile(
    climate_series: np.ndarray,
    keep: np.ndarray,
    enable: np.ndarray,
    days: Sequence[tuple[Any, int, int]],
    j: int,
    *,
    history_days: int,
    lookback_days: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Climate energy of day *j* from the recent days whose rule was **disabled**.

    The slot-wise mean over the last *history_days* complete local days within *lookback_days*
    before day *j* whose enable flag, read at the day's first slot, is 0 (a day with fewer than
    :data:`DISABLED_MIN_KNOWN_FRACTION` of its slots usable is skipped).  With no such day the
    profile is zero: a disabled rule that nobody has been running by hand adds no climate
    energy, and the trailing mean of *every* day would carry the enabled days' energy into it.

    *climate_series* and *keep* are the history the decision knows (cut at the local midnight
    of the cutoff's day), so a day past that cut is never complete and never a sample.
    """
    known = int(keep.size)
    chosen: list[int] = []
    for k in range(j - 1, max(-1, j - 1 - int(lookback_days)), -1):
        _label, lo, hi = days[k]
        if hi > known or not (np.isfinite(enable[lo]) and enable[lo] == 0.0):
            continue
        if keep[lo:hi].mean() < DISABLED_MIN_KNOWN_FRACTION:
            continue
        chosen.append(k)
        if len(chosen) >= int(history_days):
            break
    chosen_keep = np.zeros(known, dtype=bool)
    for k in chosen:
        lo, hi = days[k][1], days[k][2]
        chosen_keep[lo:hi] = keep[lo:hi]
    window = (j - min(chosen)) if chosen else 1
    values, _record = trailing_slot_mean_for_day(climate_series, chosen_keep, days, j, window)
    return values, {
        "samples": len(chosen),
        "history_days": int(history_days),
        "lookback_days": int(lookback_days),
        "source": "disabled_state_mean" if chosen else "zero_no_disabled_history",
    }


def replay_mask_for_day(
    view: SideView,
    days: Sequence[tuple[Any, int, int]],
    day_prices: np.ndarray,
    j: int,
    cut: int,
) -> tuple[np.ndarray | None, str]:
    """The ``bool[24]`` hours ``climate_control`` selects for local day *j* on *day_prices*.

    *day_prices* is a working price array (one value per interval, ``NaN`` unknown, at least
    up to the end of day *j*): the published prices plus whatever estimate the caller wants
    to test.  The enable flag and ``min_hours_per_day`` are read as of *cut*, like the
    forecaster does.  Returns ``(None, reason)`` when the rule has no schedule for the day
    (disabled, flags unknown, prices incomplete, ...).  Evaluation diagnostics only.
    """
    _label, lo, _hi = days[j]
    enable_flag = _asof_flag(view.series["climate_enable"], lo, cut)
    hours_flag = _asof_flag(view.series["climate_min_hours"], lo, cut)
    if not np.isfinite(enable_flag):
        return None, "enable_unknown"
    if enable_flag != 1.0:
        return None, "rule_disabled"
    if not np.isfinite(hours_flag):
        return None, "min_hours_unknown"
    prices = _PriceHours(view, days, np.asarray(day_prices, dtype=float))
    schedule, reason, _version = _climate_schedule_for_day(
        j=j, view=view, days=days, prices=prices, min_hours=hours_flag
    )
    if schedule is None:
        return None, reason
    return schedule.selected, ""


def _fit(request: FitRequest, **_params: Any) -> dict[str, Any]:
    """Fit one climate regression per mode on the replayed history before the fold."""
    view = request.side
    if view is None:
        return {"models": {}, "reason": "no_side_inputs"}
    cut = int(request.fit_cutoff_idx)
    days = request.days
    heat = view.series["hp_heat_kwh"]
    cool = view.series["hp_cool_kwh"]
    temp_series = view.series["outdoor_temp_c"]
    enable = view.series["climate_enable"]
    min_hours = view.series["climate_min_hours"]
    real_hours = cut // SLOTS_PER_HOUR
    hourly_temp = _hourly_mean(temp_series, real_hours)
    prices = _PriceHours(view, days, np.asarray(view.published_prices, dtype=float))
    series = {"heat": heat, "cool": cool}
    rows: dict[str, list[np.ndarray]] = {"heat": [], "cool": []}
    targets: dict[str, list[np.ndarray]] = {"heat": [], "cool": []}
    outside: dict[str, list[np.ndarray]] = {"heat": [], "cool": []}
    used_days = {"heat": 0, "cool": 0}
    for j, (_label, lo, hi) in enumerate(days):
        if hi > cut:
            break
        if lo >= cut or not (np.isfinite(enable[lo]) and enable[lo] == 1.0):
            continue
        if not np.isfinite(min_hours[lo]):
            continue
        mode, _why = infer_mode(heat, cool, lo)
        if mode is None:
            continue
        schedule, _reason, _label_v = _climate_schedule_for_day(
            j=j, view=view, days=days, prices=prices, min_hours=float(min_hours[lo])
        )
        if schedule is None:
            continue
        slots = np.arange(lo, hi)
        temp, _lag = _persisted_temperature(view, hourly_temp, slots)
        y = series[mode][slots]
        in_mask = schedule.selected[view.local_hour[slots]]
        known = np.isfinite(y)
        position = block_positions(schedule.selected)[view.local_hour[slots]]
        use = in_mask & known & np.isfinite(temp)
        if use.any():
            rows[mode].append(
                _design(mode, temp[use], position[use], view.quarter[slots][use], schedule.hours)
            )
            targets[mode].append(y[use])
            used_days[mode] += 1
        rest = ~in_mask & known
        if rest.any():
            outside[mode].append(y[rest])
    models: dict[str, _ClimateModel] = {}
    for mode in ("heat", "cool"):
        if not rows[mode]:
            continue
        x = np.concatenate(rows[mode])
        y = np.concatenate(targets[mode])
        if y.size < MIN_FIT_ROWS:
            continue
        rest = np.concatenate(outside[mode]) if outside[mode] else np.zeros(0)
        models[mode] = _fit_ridge(mode, x, y, rest, used_days[mode])
    return {
        "models": models,
        "fit_cutoff_idx": cut,
        "rows": {m: int(sum(a.shape[0] for a in rows[m])) for m in rows},
        "days": used_days,
    }


# ---------------------------------------------------------------------------
# One decision
# ---------------------------------------------------------------------------


@dataclass
class _DayParts:
    """The components of one target day's forecast, plus its diagnostics."""

    climate: np.ndarray
    dhw: np.ndarray
    resid: np.ndarray
    climate_lf0: np.ndarray
    dhw_lf0: np.ndarray
    mask: np.ndarray | None
    diag: dict[str, Any]
    price_provenance: str = "published"
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> np.ndarray:
        return np.maximum(self.climate + self.dhw + self.resid, 0.0)


def _recent_scale(
    *,
    model: _ClimateModel,
    mode: str,
    view: SideView,
    days: Sequence[tuple[Any, int, int]],
    prices: _PriceHours,
    hourly_temp: np.ndarray,
    cut: int,
    window: int,
) -> tuple[float, int]:
    """Observed over predicted climate energy in the mask of the last *window* complete days.

    The fitted regression is a monthly snapshot; the heat pump's behaviour drifts inside the
    month (set points, season).  This is the ratio, clipped to ``[0.5, 2]``, of what the
    model would have predicted for the enabled days just before the cutoff and what was
    measured, applied to the in-mask prediction.  ``(1.0, 0)`` without at least two such days.
    """
    if window < 1:
        return 1.0, 0
    series = view.series["hp_heat_kwh" if mode == "heat" else "hp_cool_kwh"]
    enable = view.series["climate_enable"]
    min_hours = view.series["climate_min_hours"]
    last = _day_of_slot(days, min(max(cut - 1, 0), int(days[-1][2]) - 1))
    predicted = observed = 0.0
    used = 0
    for j in range(last, max(-1, last - window - 1), -1):
        _label, lo, hi = days[j]
        if hi > cut or used >= window:
            continue
        if not (np.isfinite(enable[lo]) and enable[lo] == 1.0 and np.isfinite(min_hours[lo])):
            continue
        schedule, _why, _version = _climate_schedule_for_day(
            j=j, view=view, days=days, prices=prices, min_hours=float(min_hours[lo])
        )
        if schedule is None:
            continue
        slots = np.arange(lo, hi)
        temp, _lag = _persisted_temperature(view, hourly_temp, slots)
        in_mask = schedule.selected[view.local_hour[slots]]
        ok = in_mask & np.isfinite(temp) & np.isfinite(series[slots])
        if not ok.any():
            continue
        position = block_positions(schedule.selected)[view.local_hour[slots]]
        features = _design(mode, temp[ok], position[ok], view.quarter[slots][ok], schedule.hours)
        predicted += float(model.predict(features).sum())
        observed += float(series[slots][ok].sum())
        used += 1
    if used < 2 or predicted <= 0.0 or observed <= 0.0:
        return 1.0, used
    return float(np.clip(observed / predicted, 0.5, 2.0)), used


def _day_of_slot(days: Sequence[tuple[Any, int, int]], idx: int) -> int:
    for j, (_label, lo, hi) in enumerate(days):
        if lo <= idx < hi:
            return j
    return len(days) - 1


def _working_prices(
    view: SideView,
    days: Sequence[tuple[Any, int, int]],
    cut: int,
    targets: Sequence[int],
    outlook: str,
) -> tuple[np.ndarray, dict[int, str]]:
    """Published prices, then the outlook's estimate for the unpublished target slots."""
    plan_end = max(int(days[j][2]) for j in targets)
    published = view.published_end_idx
    prices = np.full(plan_end, np.nan)
    m = min(published, plan_end)
    prices[:m] = view.published_prices[:m]
    provenance: dict[int, str] = {}
    estimated = np.zeros(plan_end, dtype=bool)
    if plan_end > max(cut, published) and cut < plan_end:
        result = run_price_outlook(
            outlook,
            prices=view.published_prices,
            published_end_idx=published,
            decision_idx=cut,
            plan_end_idx=plan_end,
            interval_starts_utc=view.interval_starts_utc,
            days=days,
            tz=view.tz,
            # The view already holds only the OMIE days usable at the cutoff; the seam cuts it
            # again at the decision instant, which is the same instant.
            side=(
                None
                if view.omie_eur_per_kwh is None
                else OutlookSide(view.omie_eur_per_kwh, fingerprint="as-of view")
            ),
        )
        start = max(cut, published)
        values = result.values[start - cut :]
        flags = result.provenance[start - cut :] == SLOT_ESTIMATED
        prices[start:plan_end] = np.where(flags, values, np.nan)
        estimated[start:plan_end] = flags
    for j in targets:
        _label, lo, hi = days[j]
        if hi <= published:
            provenance[j] = "published"
            continue
        tail = np.arange(max(lo, published, 0), hi)
        provenance[j] = "estimated" if estimated[tail].all() else "unavailable"
    return prices, provenance


def _parts(
    *,
    view: SideView,
    history_load: np.ndarray,
    history_usable: np.ndarray,
    days: Sequence[tuple[Any, int, int]],
    cut: int,
    targets: Sequence[int],
    fitted: Any,
    window_days: int,
    residual_forecaster: str,
    price_outlook: str,
    calibration_days: int,
    disabled_fallback: str = "lf0",
    disabled_history_days: int = 28,
    disabled_lookback_days: int = 60,
    climate_lf0_season: str = "",
) -> list[_DayParts]:
    """The structured forecast of *targets* by component, knowing only what *view* shows."""
    heat = view.series["hp_heat_kwh"]
    cool = view.series["hp_cool_kwh"]
    dhw = view.series["hp_dhw_kwh"]
    temp_series = view.series["outdoor_temp_c"]
    enable = view.series["climate_enable"]
    min_hours = view.series["climate_min_hours"]
    dhw_enable = view.series["dhw_enable"]

    # LF-0 basis: the local midnight of the day holding the cutoff.
    day_of_cut = _day_of_slot(days, min(cut, int(days[-1][2]) - 1))
    midnight = int(min(days[day_of_cut][1], cut))
    resid_spec = get_forecaster(residual_forecaster)
    resid_cut = int(cut) if resid_spec.intraday_cutoff else midnight

    def _known(upto: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h, c, d = heat[:upto], cool[:upto], dhw[:upto]
        keep = history_usable[:upto] & np.isfinite(h) & np.isfinite(c) & np.isfinite(d)
        return h, c, d, keep

    h_r, c_r, d_r, keep_r = _known(resid_cut)
    resid_series = np.where(keep_r, history_load[:resid_cut] - (h_r + c_r + d_r), 0.0)
    resid_params = resolve_forecaster_params(residual_forecaster, window_days=window_days)
    resid_result = run_forecaster(
        residual_forecaster,
        load_kwh=resid_series,
        usable=keep_r,
        days=days,
        cutoff_idx=resid_cut,
        target_days=targets,
        params=resid_params,
    )
    h_m, c_m, d_m, keep_m = _known(midnight)
    climate_series = np.where(keep_m, h_m + c_m, 0.0)
    dhw_series = np.where(keep_m, d_m, 0.0)
    hybrid_months = _SEASON_MONTHS[climate_lf0_season]

    mode, mode_reason = infer_mode(heat, cool, cut)
    prices_arr, provenance = _working_prices(view, days, cut, targets, price_outlook)
    prices = _PriceHours(view, days, prices_arr)
    hourly_temp = _hourly_mean(temp_series, cut // SLOTS_PER_HOUR)
    models = (fitted or {}).get("models", {}) if isinstance(fitted, dict) else {}
    dhw_profile = _DhwProfile.build(
        view=view,
        days=days,
        cut=cut,
        targets=targets,
        prices=prices,
        dhw=dhw,
        dhw_enable=dhw_enable,
        window_days=window_days,
    )

    out: list[_DayParts] = []
    scales: dict[str, tuple[float, int]] = {}
    for index, j in enumerate(targets):
        label, lo, hi = days[j]
        slots = np.arange(lo, hi)
        climate_lf0, lf0_record = trailing_slot_mean_for_day(
            climate_series, keep_m, days, j, window_days
        )
        dhw_lf0, _ = trailing_slot_mean_for_day(dhw_series, keep_m, days, j, window_days)
        record: dict[str, Any] = {
            "local_day": str(label),
            "intervals": int(hi - lo),
            "price_provenance": provenance[j],
        }
        climate = climate_lf0
        mask = None
        climate_source, climate_reason = "fallback_lf0", ""
        enable_flag = _asof_flag(enable, lo, cut)
        hours_flag = _asof_flag(min_hours, lo, cut)
        schedule: ClimateSchedule | None = None
        rule_label = ""
        if not np.isfinite(enable_flag):
            climate_reason = "enable_unknown"
        elif enable_flag != 1.0:
            climate_reason = "rule_disabled"
        elif not np.isfinite(hours_flag):
            climate_reason = "min_hours_unknown"
        elif mode is None:
            climate_reason = mode_reason
        elif _as_date(label).month in hybrid_months:
            climate_reason = "season_hybrid"
        elif provenance[j] == "unavailable":
            climate_reason = "prices_unavailable"
        else:
            schedule, climate_reason, rule_label = _climate_schedule_for_day(
                j=j, view=view, days=days, prices=prices, min_hours=hours_flag
            )
        if climate_reason == "rule_disabled" and disabled_fallback == "disabled_history":
            climate, history_record = disabled_state_profile(
                climate_series,
                keep_m,
                enable,
                days,
                j,
                history_days=int(disabled_history_days),
                lookback_days=int(disabled_lookback_days),
            )
            climate_source = "disabled_history"
            record["disabled_history"] = history_record
        if schedule is not None:
            model = models.get(mode)
            if model is None:
                climate_reason = "model_not_fitted"
            else:
                temp, lag = _persisted_temperature(view, hourly_temp, slots)
                in_mask = schedule.selected[view.local_hour[slots]]
                if not np.isfinite(temp[in_mask]).all():
                    climate_reason = "temperature_missing"
                else:
                    position = block_positions(schedule.selected)[view.local_hour[slots]]
                    climate = np.full(slots.size, model.out_of_mask_kwh)
                    if in_mask.any():
                        features = _design(
                            mode,
                            temp[in_mask],
                            position[in_mask],
                            view.quarter[slots][in_mask],
                            schedule.hours,
                        )
                        scale, scale_days = scales.setdefault(
                            mode,
                            _recent_scale(
                                model=model,
                                mode=mode,
                                view=view,
                                days=days,
                                prices=prices,
                                hourly_temp=hourly_temp,
                                cut=cut,
                                window=int(calibration_days),
                            ),
                        )
                        climate[in_mask] = model.predict(features) * scale
                    mask = in_mask
                    climate_source = "structured"
                    record["climate"] = {
                        "mode": mode,
                        "rule_version": rule_label,
                        "selected_hours": [int(h) for h in np.flatnonzero(schedule.selected)],
                        "cheap_branch": bool(schedule.cheap_branch),
                        "temperature_lag_days_max": int(lag[in_mask].max()) if in_mask.any() else 0,
                        "recent_scale": round(float(scales.get(mode, (1.0, 0))[0]), 4),
                    }
        dhw_values, dhw_source, dhw_reason = dhw_profile.day(
            j, dhw_lf0, provenance[j], _asof_flag(dhw_enable, lo, cut)
        )
        resid, resid_diag = resid_result.values[lo:hi], resid_result.per_day[index]
        record.update(
            {
                "climate_source": climate_source,
                "climate_fallback_reason": climate_reason if climate_source != "structured" else "",
                "dhw_source": dhw_source,
                "dhw_fallback_reason": dhw_reason if dhw_source != "structured" else "",
                "residual_forecaster": residual_forecaster,
                "labels": {
                    "prices": provenance[j],
                    "climate": (
                        "structured"
                        if climate_source == "structured"
                        else f"fallback:{climate_reason}"
                    ),
                    "dhw": "structured" if dhw_source == "structured" else f"fallback:{dhw_reason}",
                },
            }
        )
        for key in (
            "slots_pooled_fallback",
            "slots_zero_no_history",
            "window_days_with_usable_data",
        ):
            record[key] = int(resid_diag.get(key) or 0) + int(lf0_record.get(key) or 0)
        record["fallback"] = "none"
        out.append(
            _DayParts(
                climate=np.asarray(climate, dtype=float),
                dhw=np.asarray(dhw_values, dtype=float),
                resid=np.asarray(resid, dtype=float),
                climate_lf0=np.asarray(climate_lf0, dtype=float),
                dhw_lf0=np.asarray(dhw_lf0, dtype=float),
                mask=mask,
                diag=record,
                price_provenance=provenance[j],
            )
        )
    return out


@dataclass
class _DhwProfile:
    """Hot-water energy by hours since the last forced hour and quarter of the hour."""

    view: SideView
    days: Sequence[tuple[Any, int, int]]
    prices: _PriceHours
    rule_real: np.ndarray
    cells: np.ndarray
    pooled: float
    eligible_days: int
    targets_rules: dict[int, tuple[int, ...] | None]

    @classmethod
    def build(
        cls,
        *,
        view: SideView,
        days: Sequence[tuple[Any, int, int]],
        cut: int,
        targets: Sequence[int],
        prices: _PriceHours,
        dhw: np.ndarray,
        dhw_enable: np.ndarray,
        window_days: int,
    ) -> _DhwProfile:
        last_known = _day_of_slot(days, min(max(cut - 1, 0), int(days[-1][2]) - 1))
        first_day = max(0, last_known - int(window_days))
        rule_hours: dict[int, tuple[int, ...]] = {}
        eligible: list[int] = []

        def replay(j: int) -> tuple[int, ...] | None:
            version = _version_for(DHW_RULE_VERSIONS, _as_date(days[j][0]))
            if version is None:
                return None
            hour_prices, present = prices.clock_prices(j)
            if not np.isfinite(hour_prices[present]).all():
                return None
            if not version.accepts_negative_prices and (hour_prices[present] < 0).any():
                return None
            return replay_dhw_rule(hour_prices, present, interval_hours=version.interval_hours)

        for j in range(first_day, last_known + 1):
            hours = replay(j)
            if hours is None:
                continue
            lo = days[j][1]
            flag = _asof_flag(dhw_enable, lo, cut)
            if flag == 1.0:
                rule_hours[j] = hours
                if j > first_day:
                    eligible.append(j)
        targets_rules: dict[int, tuple[int, ...] | None] = {}
        for j in targets:
            hours = rule_hours.get(j) or replay(j)
            targets_rules[j] = hours
            if hours is not None and _asof_flag(dhw_enable, days[j][1], cut) == 1.0:
                rule_hours[j] = hours
        real: list[int] = []
        for j, hours in rule_hours.items():
            for h in hours:
                first = int(view.first_real_hour[j, h])
                if first >= 0:
                    real.append(first)
        rule_real = np.array(sorted(set(real)), dtype=np.int64)
        cells = np.zeros((DHW_MAX_K + 1, 4, 2))
        for j in eligible:
            lo, hi = days[j][1], min(days[j][2], cut)
            slots = np.arange(lo, hi)
            y = dhw[slots]
            ok = np.isfinite(y)
            k = cls._k(rule_real, slots // SLOTS_PER_HOUR)
            q = view.quarter[slots]
            np.add.at(cells[:, :, 0], (k[ok], q[ok]), y[ok])
            np.add.at(cells[:, :, 1], (k[ok], q[ok]), 1.0)
        total = float(cells[:, :, 0].sum())
        count = float(cells[:, :, 1].sum())
        return cls(
            view=view,
            days=days,
            prices=prices,
            rule_real=rule_real,
            cells=cells,
            pooled=total / count if count else 0.0,
            eligible_days=len(eligible),
            targets_rules=targets_rules,
        )

    @staticmethod
    def _k(rule_real: np.ndarray, real_hours: np.ndarray) -> np.ndarray:
        if rule_real.size == 0:
            return np.full(real_hours.size, DHW_MAX_K, dtype=np.int64)
        pos = np.searchsorted(rule_real, real_hours, side="right") - 1
        since = np.where(pos >= 0, real_hours - rule_real[np.clip(pos, 0, None)], DHW_MAX_K)
        return np.minimum(since, DHW_MAX_K).astype(np.int64)

    def day(
        self, j: int, fallback: np.ndarray, provenance: str, flag: float
    ) -> tuple[np.ndarray, str, str]:
        """``(values, source, reason)`` for target day *j*."""
        if not np.isfinite(flag):
            return fallback, "fallback_lf0", "enable_unknown"
        if flag != 1.0:
            return fallback, "fallback_lf0", "rule_disabled"
        if provenance == "unavailable":
            return fallback, "fallback_lf0", "prices_unavailable"
        if self.targets_rules.get(j) is None:
            return fallback, "fallback_lf0", "rule_not_applicable"
        if self.eligible_days < MIN_DHW_DAYS:
            return fallback, "fallback_lf0", "profile_history_short"
        _label, lo, hi = self.days[j]
        slots = np.arange(lo, hi)
        k = self._k(self.rule_real, slots // SLOTS_PER_HOUR)
        q = self.view.quarter[slots]
        total = self.cells[k, q, 0]
        count = self.cells[k, q, 1]
        values = np.where(count >= 3, total / np.maximum(count, 1.0), self.pooled)
        return values, "structured", ""


# ---------------------------------------------------------------------------
# Evaluation diagnostics (hindsight: reported by the harness, never part of a forecast)
# ---------------------------------------------------------------------------


def _diagnose(
    *,
    side: SideInputs,
    load: np.ndarray,
    usable: np.ndarray,
    days: Sequence[tuple[Any, int, int]],
    decisions: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    """Component accuracy, mask hit-rate and fallback counts over the given decisions.

    Each decision is ``{"kind", "cutoff_idx", "target_days", "start_idx", "stop_idx"}``: the
    forecast made at ``cutoff_idx`` for ``target_days``, scored on the slots of
    ``[start_idx, stop_idx)`` (the plan horizon).  Only slots where the main load, the three
    heat-pump series and the temperature are known are scored.  The same decisions are
    forecast through the seam's fit step, so the numbers are the planner's own.
    """
    resolved = resolve_forecaster_params("structured", overrides=params)
    heat_o, cool_o, dhw_o = (side.series[k] for k in ("hp_heat_kwh", "hp_cool_kwh", "hp_dhw_kwh"))
    known_o = np.isfinite(heat_o) & np.isfinite(cool_o) & np.isfinite(dhw_o) & usable[: side.n]
    climate_o = np.where(known_o, heat_o + cool_o, 0.0)
    dhw_obs = np.where(known_o, dhw_o, 0.0)
    acc: dict[str, dict[str, float]] = {}

    def add(group: str, **values: float) -> None:
        slot = acc.setdefault(group, {})
        for key, value in values.items():
            slot[key] = slot.get(key, 0.0) + float(value)

    counts: dict[str, dict[str, int]] = {}

    def count(group: str, label: str) -> None:
        counts.setdefault(group, {})
        counts[group][label] = counts[group].get(label, 0) + 1

    for decision in decisions:
        cut = int(decision["cutoff_idx"])
        targets = [int(j) for j in decision["target_days"]]
        view = side.view(cut, days)
        fitted = fold_state("structured", resolved, load, usable, days, cut, side)
        parts = _parts(
            view=view,
            history_load=np.asarray(load[:cut], dtype=float),
            history_usable=np.asarray(usable[:cut], dtype=bool),
            days=days,
            cut=cut,
            targets=targets,
            fitted=fitted,
            window_days=int(resolved["window_days"]),
            residual_forecaster=str(resolved["residual_forecaster"]),
            price_outlook=str(resolved["price_outlook"]),
            calibration_days=int(resolved["calibration_days"]),
            disabled_fallback=str(resolved["disabled_fallback"]),
            disabled_history_days=int(resolved["disabled_history_days"]),
            disabled_lookback_days=int(resolved["disabled_lookback_days"]),
            climate_lf0_season=str(resolved["climate_lf0_season"]),
        )
        kind = str(decision["kind"])
        lo_h, hi_h = int(decision["start_idx"]), int(decision["stop_idx"])
        for j, part in zip(targets, parts, strict=True):
            _label, lo, hi = days[j]
            a, b = max(lo, lo_h), min(hi, hi_h)
            if a >= b:
                continue
            sl = slice(a - lo, b - lo)
            idx = np.arange(a, b)
            ok = known_o[idx]
            labels = part.diag["labels"]
            for field_name in ("prices", "climate", "dhw"):
                count(f"{kind}/{field_name}", labels[field_name])
            if not ok.any():
                continue
            i = idx[ok]
            structured_climate = part.diag["climate_source"] == "structured"
            c_pred, c_lf0, c_obs = part.climate[sl][ok], part.climate_lf0[sl][ok], climate_o[i]
            d_pred, d_lf0, d_obs = part.dhw[sl][ok], part.dhw_lf0[sl][ok], dhw_obs[i]
            for group in ("all", f"kind:{kind}", f"prices:{part.price_provenance}"):
                add(
                    group,
                    slots=ok.sum(),
                    climate_abs=np.abs(c_pred - c_obs).sum(),
                    climate_lf0_abs=np.abs(c_lf0 - c_obs).sum(),
                    climate_obs=c_obs.sum(),
                    dhw_abs=np.abs(d_pred - d_obs).sum(),
                    dhw_lf0_abs=np.abs(d_lf0 - d_obs).sum(),
                    dhw_obs=d_obs.sum(),
                    hp_abs=np.abs((c_pred + d_pred) - (c_obs + d_obs)).sum(),
                    hp_lf0_abs=np.abs((c_lf0 + d_lf0) - (c_obs + d_obs)).sum(),
                    hp_obs=(c_obs + d_obs).sum(),
                )
            if structured_climate and part.mask is not None:
                for group in (
                    "mask:all",
                    f"mask:kind:{kind}",
                    f"mask:prices:{part.price_provenance}",
                ):
                    in_mask = part.mask[sl][ok]
                    add(
                        group,
                        slots=ok.sum(),
                        mask_slots=in_mask.sum(),
                        climate_obs=c_obs.sum(),
                        climate_in_mask=c_obs[in_mask].sum(),
                        decisions_days=1,
                    )
    raw = {
        "decisions": len(decisions),
        "sums": acc,
        "counts": counts,
        "note": (
            "slots where the main load, the three heat-pump series are known; WAPE = sum of "
            "absolute error / sum of observed; *_lf0 is the trailing mean of the same component "
            "at the local midnight; mask hit-rate = observed climate kWh inside the replayed "
            "hours / observed climate kWh, over structured climate days"
        ),
    }
    return {**summarise_diagnostics(raw), "sums": acc}


def summarise_diagnostics(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Ratios of :func:`_diagnose` sums: component WAPE (against LF-0), mask hit-rate, counts."""
    out: dict[str, Any] = {"decisions": raw["decisions"], "counts": raw["counts"], "groups": {}}

    def ratio(a: float, b: float) -> float | None:
        return a / b if b > 0 else None

    for group, s in raw["sums"].items():
        if group.startswith("mask:"):
            out["groups"][group] = {
                "hit_rate": ratio(s["climate_in_mask"], s["climate_obs"]),
                "mask_fraction": ratio(s["mask_slots"], s["slots"]),
                "slots": int(s["slots"]),
            }
        else:
            out["groups"][group] = {
                "slots": int(s["slots"]),
                "climate_wape": ratio(s["climate_abs"], s["climate_obs"]),
                "climate_wape_lf0": ratio(s["climate_lf0_abs"], s["climate_obs"]),
                "dhw_wape": ratio(s["dhw_abs"], s["dhw_obs"]),
                "dhw_wape_lf0": ratio(s["dhw_lf0_abs"], s["dhw_obs"]),
                "heat_pump_wape": ratio(s["hp_abs"], s["hp_obs"]),
                "heat_pump_wape_lf0": ratio(s["hp_lf0_abs"], s["hp_obs"]),
            }
    out["note"] = raw["note"]
    return out


# ---------------------------------------------------------------------------
# Registered forecaster
# ---------------------------------------------------------------------------


def _check_params(
    residual_forecaster: str,
    price_outlook: str,
    *,
    disabled_fallback: str = "lf0",
    disabled_history_days: int = 28,
    disabled_lookback_days: int = 60,
    climate_lf0_season: str = "",
) -> None:
    if disabled_fallback not in DISABLED_FALLBACKS:
        raise ValueError(
            f"disabled_fallback={disabled_fallback!r} must be one of {list(DISABLED_FALLBACKS)}"
        )
    if int(disabled_history_days) < 1 or int(disabled_lookback_days) < int(disabled_history_days):
        raise ValueError(
            "disabled_history_days must be >= 1 and disabled_lookback_days >= "
            f"disabled_history_days, got {disabled_history_days} and {disabled_lookback_days}"
        )
    if climate_lf0_season not in _SEASON_MONTHS:
        raise ValueError(
            f"climate_lf0_season={climate_lf0_season!r} must be one of "
            f"{[k for k in _SEASON_MONTHS if k]} or empty"
        )
    resid = get_forecaster(residual_forecaster)
    if resid.oracle or resid.side_loader is not None:
        raise ValueError(
            f"residual_forecaster={residual_forecaster!r} must be a non-oracle forecaster "
            "without side inputs"
        )
    if get_price_outlook(price_outlook).oracle:
        raise ValueError(f"price_outlook={price_outlook!r} is an oracle and cannot be deployed")


def _load_side(
    params: Mapping[str, Any], *, table: Any, data_dir: Path | str | None = None
) -> SideInputs:
    """Build the structured forecaster's side inputs from the feature set it names."""
    from zoneinfo import ZoneInfo

    import pyarrow as pa

    from .features import FEATURES_DIRNAME, column_array, load_features
    from .paths import resolve_data_dir

    _check_params(
        str(params["residual_forecaster"]),
        str(params["price_outlook"]),
        disabled_fallback=str(params["disabled_fallback"]),
        disabled_history_days=int(params["disabled_history_days"]),
        disabled_lookback_days=int(params["disabled_lookback_days"]),
        climate_lf0_season=str(params["climate_lf0_season"]),
    )
    feature_id = str(params.get("features") or "")
    if not feature_id:
        raise ValueError("the structured forecaster needs features=<feature set id>")
    root = (
        resolve_data_dir(str(data_dir) if data_dir else None) / "analysis" / "v1" / FEATURES_DIRNAME
    )
    matches = sorted(p for p in root.glob(f"{feature_id}*") if p.is_dir())
    if len(matches) != 1:
        raise ValueError(
            f"features={feature_id!r} matches {len(matches)} feature sets under {root}"
        )
    features, manifest = load_features(matches[0])
    f_ts = features.column("interval_start").cast(pa.int64()).to_numpy()
    d_ts = table.column("interval_start").cast(pa.int64()).to_numpy()
    order = np.searchsorted(f_ts, d_ts)
    order_clipped = np.clip(order, 0, f_ts.size - 1)
    found = (order < f_ts.size) & (f_ts[order_clipped] == d_ts)
    series: dict[str, np.ndarray] = {}
    for name, column in FEATURE_COLUMNS.items():
        values = column_array(features, column)[order_clipped]
        series[name] = np.where(found, values, np.nan)
    price_column = table.column("price_eur_per_kwh").to_pylist()
    prices = np.array([np.nan if v is None else float(v) for v in price_column])
    tz = (
        ZoneInfo(str(table.column("tz_local")[0].as_py()))
        if "tz_local" in table.column_names
        else ZoneInfo("Europe/Madrid")
    )
    starts = table.column("interval_start").to_pylist()
    source: dict[str, Any] = {
        "kind": "structured_side_inputs",
        "feature_set_id": str(manifest.get("feature_set_id") or matches[0].name),
        "feature_manifest_sha256": _manifest_sha(matches[0]),
        "slots_without_features": int((~found).sum()),
    }
    omie = None
    if get_price_outlook(str(params["price_outlook"])).needs_omie:
        from .omie_outlook import load_omie_side

        omie_side = load_omie_side(starts, data_dir=data_dir, tz=tz)
        omie = omie_side.omie_eur_per_kwh
        source["omie"] = omie_side.describe()
    return SideInputs(
        interval_starts_utc=starts,
        tz=tz,
        prices=prices,
        series=series,
        source=source,
        omie_eur_per_kwh=omie,
    )


def _manifest_sha(directory: Path) -> str:
    import hashlib

    from .features import MANIFEST_NAME

    return hashlib.sha256((directory / MANIFEST_NAME).read_bytes()).hexdigest()


def _lf0_total(request: ForecastRequest, window_days: int) -> list[DayForecast]:
    """No side data: the trailing mean of the main load, labelled as the reason."""
    out: list[DayForecast] = []
    load, keep = request.history_load_kwh, request.history_usable
    for j in request.target_days:
        values, record = trailing_slot_mean_for_day(load, keep, request.days, j, window_days)
        record = {
            **record,
            "climate_source": "fallback_lf0",
            "climate_fallback_reason": "no_side_inputs",
            "dhw_source": "fallback_lf0",
            "dhw_fallback_reason": "no_side_inputs",
            "labels": {
                "prices": "none",
                "climate": "fallback:no_side_inputs",
                "dhw": "fallback:no_side_inputs",
            },
        }
        out.append((values, record))
    return out


@register_forecaster(
    "structured",
    method="structured_components",
    fallback_rule=(
        "climate or hot-water part falls back to the trailing mean (LF-0) of that component when "
        "the rule is disabled, the mode, a flag, prices or temperature are unknown, or no model "
        "is fitted; the reason is recorded per target day; without side inputs the forecast is "
        "LF-0 of the main load"
    ),
    fit=_fit,
    intraday_cutoff=True,
    side_loader=_load_side,
    diagnose=_diagnose,
)
def _structured(
    request: ForecastRequest,
    *,
    features: str = "",
    window_days: int = DEFAULT_FORECAST_WINDOW_DAYS,
    residual_forecaster: str = DEFAULT_LOAD_FORECASTER,
    price_outlook: str = "persistence",
    calibration_days: int = 7,
    disabled_fallback: str = "lf0",
    disabled_history_days: int = 28,
    disabled_lookback_days: int = 60,
    climate_lf0_season: str = "",
) -> list[DayForecast]:
    """LF-1: replayed heat-pump schedule x fitted energy per hour, plus a residual forecaster."""
    if int(window_days) < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")
    _check_params(
        residual_forecaster,
        price_outlook,
        disabled_fallback=disabled_fallback,
        disabled_history_days=disabled_history_days,
        disabled_lookback_days=disabled_lookback_days,
        climate_lf0_season=climate_lf0_season,
    )
    if request.side is None:
        return _lf0_total(request, int(window_days))
    parts = _parts(
        view=request.side,
        history_load=request.history_load_kwh,
        history_usable=request.history_usable,
        days=request.days,
        cut=request.cutoff_idx,
        targets=request.target_days,
        fitted=request.fitted,
        window_days=int(window_days),
        residual_forecaster=residual_forecaster,
        price_outlook=price_outlook,
        calibration_days=int(calibration_days),
        disabled_fallback=disabled_fallback,
        disabled_history_days=int(disabled_history_days),
        disabled_lookback_days=int(disabled_lookback_days),
        climate_lf0_season=climate_lf0_season,
    )
    return [(p.total, p.diag) for p in parts]
