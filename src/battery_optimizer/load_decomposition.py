"""Load decomposition and LF-0 error attribution (P3 of the load-forecaster plan).

Splits the canonical gross load into the observable sub-loads of the feature table and finds out
where the error of the production forecaster (LF-0, ``policies.forecast_load_trailing_slot_mean``)
comes from:

* ``main``: the canonical gross load of the slot;
* ``servers``: the servers circuit;
* ``heat_pump``: electrical input of the heat pump, ``heat + dhw + cool``;
* ``residual = main - servers - heat_pump``.

Method
------
LF-0 is applied to ``main`` and to every component series separately, **with one common
usability mask** (the *joint set*: the canonical benchmark mask, all sub-loads known, and the
slots where ``servers + heat_pump`` exceeds ``main`` by more than a tolerance removed, because
they are measurement faults and not load).  The forecast is linear in the series for a fixed
mask, so ``e_main = e_servers + e_heat_pump + e_residual`` holds exactly per slot (the report
states the largest deviation).  That makes the error variance of the main forecast split
exactly: ``share_c = Cov(e_c, e_main) / Var(e_main)`` and the shares sum to one.

Error is ``forecast - measured`` in kWh per 15-minute slot.  Evaluation slots are the joint-set
slots of days that have at least ``min_history_days`` days of usable history.  The
price-weighted absolute error ``sum(price * |error|)`` is a EUR proxy and is computed with the
canonical dataset's price.  "Headroom" is the price-weighted error that would remain if one
component were forecast perfectly (``e_main - e_c``).

The report is private: it contains energy and money figures of one household and is written
under the data directory only.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa

from battery_optimizer import features
from battery_optimizer.paths import resolve_data_dir

log = logging.getLogger(__name__)

SEASONS = {
    12: "winter",
    1: "winter",
    2: "winter",
    3: "spring",
    4: "spring",
    5: "spring",
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}
SEASON_ORDER = ("winter", "spring", "summer", "autumn")

DEFAULT_WINDOW_DAYS = 28
MIN_HISTORY_DAYS = 14
VIOLATION_TOLERANCE_KWH = 0.02
#: A component whose share of the LF-0 error variance reaches this is called material.  The plan
#: does not fix a number, so this is a stated heuristic and the report prints the shares.
MATERIAL_SHARE = 0.20
DHW_INTERVAL_HOURS = (12, 24)
CHEAP_PRICE_FACTOR = 2.0 / 3.0
PRICE_HISTORY_DAYS = 7
#: Hours after a DHW rule hour that still count as part of its heating cycle.
DHW_WINDOW_HOURS = 3

COMPONENTS = ("servers", "heat_pump", "residual")
DETAIL_COMPONENTS = ("servers", "hp_heat", "hp_dhw", "hp_cool", "heat_pump", "residual")


def season_label(day: date) -> str:
    """Meteorological season of a local day, e.g. ``2025-winter`` (December joins next winter)."""
    season = SEASONS[day.month]
    year = day.year + 1 if day.month == 12 else day.year
    return f"{year}-{season}"


def season_of_year(day: date) -> str:
    return SEASONS[day.month]


# ---------------------------------------------------------------------------
# Aligned frame
# ---------------------------------------------------------------------------


@dataclass
class Frame:
    """The canonical grid with the feature columns aligned to it (NaN where unknown)."""

    start_us: np.ndarray
    local_day: list[date]
    local_hour: np.ndarray
    price: np.ndarray
    usable: np.ndarray
    main: np.ndarray
    servers: np.ndarray
    hp_heat: np.ndarray
    hp_dhw: np.ndarray
    hp_cool: np.ndarray
    climate_min_hours: np.ndarray
    climate_enable: np.ndarray
    dhw_enable: np.ndarray
    dhw_force_mean: np.ndarray
    timezone: str = "Europe/Madrid"
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.start_us.size)

    @property
    def heat_pump(self) -> np.ndarray:
        return self.hp_heat + self.hp_dhw + self.hp_cool

    @property
    def residual(self) -> np.ndarray:
        return self.main - self.servers - self.heat_pump


def build_frame(dataset: Any, feats: pa.Table) -> Frame:
    """Align the feature table to a loaded canonical dataset on ``interval_start``."""
    from battery_optimizer.benchmark import usable_mask

    table = dataset.table
    d_ts = table.column("interval_start").cast(pa.int64()).to_numpy()
    f_ts = features.column_array_ts(feats)
    pos = np.searchsorted(f_ts, d_ts)
    pos_c = np.clip(pos, 0, max(f_ts.size - 1, 0))
    present = (pos < f_ts.size) & (f_ts[pos_c] == d_ts)

    def aligned(name: str) -> np.ndarray:
        col = features.column_array(feats, name)
        return np.where(present, col[pos_c], np.nan)

    tz_name = table.column("tz_local")[0].as_py()
    tz = ZoneInfo(tz_name)
    starts = table.column("interval_start").to_pylist()
    local_hour = np.array([s.astimezone(tz).hour for s in starts], dtype=np.int64)
    usable, _reasons = usable_mask(table)
    price = table.column("price_eur_per_kwh").to_numpy(zero_copy_only=False).astype(np.float64)
    gross = table.column("gross_kwh").to_numpy(zero_copy_only=False).astype(np.float64)
    return Frame(
        start_us=d_ts,
        local_day=table.column("local_day").to_pylist(),
        local_hour=local_hour,
        price=price,
        usable=usable,
        main=gross,
        servers=aligned("servers_kwh"),
        hp_heat=aligned("hp_heat_kwh"),
        hp_dhw=aligned("hp_dhw_kwh"),
        hp_cool=aligned("hp_cool_kwh"),
        climate_min_hours=aligned("climate_min_hours_at_start"),
        climate_enable=aligned("climate_enable_at_start"),
        dhw_enable=aligned("dhw_enable_at_start"),
        dhw_force_mean=aligned("dhw_force_mean"),
        timezone=tz_name,
    )


def joint_mask(frame: Frame, *, tolerance_kwh: float = VIOLATION_TOLERANCE_KWH) -> np.ndarray:
    """Slots where main, servers and all heat-pump inputs are known and plausible."""
    known = frame.usable & np.isfinite(frame.servers) & np.isfinite(frame.heat_pump)
    known &= np.isfinite(frame.main)
    excess = np.where(known, frame.servers + frame.heat_pump - frame.main, 0.0)
    return known & (excess <= tolerance_kwh)


# ---------------------------------------------------------------------------
# LF-0 on every series
# ---------------------------------------------------------------------------


def lf0_errors(
    frame: Frame, joint: np.ndarray, *, window_days: int = DEFAULT_WINDOW_DAYS
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    """LF-0 error (forecast - measured) of every series, and the evaluation mask."""
    from battery_optimizer.policies import (
        contiguous_day_bounds,
        forecast_load_trailing_slot_mean,
    )

    days = contiguous_day_bounds(frame.local_day)
    series = {
        "main": frame.main,
        "servers": frame.servers,
        "hp_heat": frame.hp_heat,
        "hp_dhw": frame.hp_dhw,
        "hp_cool": frame.hp_cool,
        "heat_pump": frame.heat_pump,
        "residual": frame.residual,
    }
    errors: dict[str, np.ndarray] = {}
    per_day_main: list[dict[str, Any]] = []
    for name, values in series.items():
        load = np.where(joint, values, 0.0)
        forecast, per_day = forecast_load_trailing_slot_mean(
            load, days, window_days=window_days, usable=joint
        )
        errors[name] = np.where(joint, forecast - values, np.nan)
        if name == "main":
            per_day_main = per_day
    # Production LF-0 mask for reference: its own usable mask, scored on the same slots.
    load = np.where(frame.usable, frame.main, 0.0)
    forecast, _ = forecast_load_trailing_slot_mean(
        load, days, window_days=window_days, usable=frame.usable
    )
    errors["main_production_mask"] = np.where(joint, forecast - frame.main, np.nan)

    history_ok = np.zeros(frame.n, dtype=bool)
    for (_label, start, stop), info in zip(days, per_day_main, strict=True):
        if info["window_days_with_usable_data"] >= min(MIN_HISTORY_DAYS, window_days):
            history_ok[start:stop] = True
    evaluation = joint & history_ok
    additive = np.nanmax(
        np.abs(
            errors["main"][evaluation]
            - errors["servers"][evaluation]
            - errors["heat_pump"][evaluation]
            - errors["residual"][evaluation]
        )
    )
    meta = {
        "window_days": window_days,
        "min_history_days": min(MIN_HISTORY_DAYS, window_days),
        "max_additivity_deviation_kwh": float(additive) if evaluation.any() else None,
    }
    return errors, evaluation, meta


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _safe_div(a: float, b: float) -> float | None:
    return float(a / b) if b else None


def energy_shares(frame: Frame, sel: np.ndarray) -> dict[str, Any]:
    """kWh of each component and its share of the main load over the selected slots."""
    main = float(frame.main[sel].sum())
    parts = {
        "servers": frame.servers,
        "hp_heat": frame.hp_heat,
        "hp_dhw": frame.hp_dhw,
        "hp_cool": frame.hp_cool,
        "heat_pump": frame.heat_pump,
        "residual": frame.residual,
    }
    out: dict[str, Any] = {"slots": int(sel.sum()), "main_kwh": main}
    for name, values in parts.items():
        kwh = float(values[sel].sum())
        out[f"{name}_kwh"] = kwh
        out[f"{name}_share"] = _safe_div(kwh, main)
    return out


def error_stats(
    errors: dict[str, np.ndarray], price: np.ndarray, sel: np.ndarray
) -> dict[str, Any]:
    """MAE, variance shares and price-weighted error of each series over the selected slots."""
    n = int(sel.sum())
    out: dict[str, Any] = {"slots": n}
    if n < 2:
        return out
    e_main = errors["main"][sel]
    p = price[sel]
    var_main = float(np.var(e_main))
    mae_main = float(np.mean(np.abs(e_main)))
    pw_main = float(np.sum(p * np.abs(e_main)))
    out["price_weighted_total_eur_proxy"] = pw_main
    for name in ("main", "main_production_mask", *DETAIL_COMPONENTS):
        e = errors[name][sel]
        entry: dict[str, Any] = {
            "mae_kwh": float(np.mean(np.abs(e))),
            "bias_kwh": float(np.mean(e)),
            "rmse_kwh": float(np.sqrt(np.mean(e**2))),
            "price_weighted_abs_error": float(np.sum(p * np.abs(e))),
        }
        if name != "main_production_mask":
            cov = float(np.mean((e - e.mean()) * (e_main - e_main.mean())))
            entry["var_share"] = _safe_div(cov, var_main)
            entry["standalone_var_ratio"] = _safe_div(float(np.var(e)), var_main)
            entry["mae_ratio_to_main"] = _safe_div(entry["mae_kwh"], mae_main)
            entry["price_weighted_share_standalone"] = _safe_div(
                entry["price_weighted_abs_error"], pw_main
            )
            if name != "main":
                remaining = float(np.sum(p * np.abs(e_main - e)))
                entry["headroom_price_weighted_abs_error"] = pw_main - remaining
                entry["headroom_share"] = _safe_div(pw_main - remaining, pw_main)
        out[name] = entry
    return out


def group_indices(frame: Frame, keys: list[str]) -> dict[str, np.ndarray]:
    """Boolean masks per distinct key (the key list is per slot)."""
    arr = np.array(keys, dtype=object)
    return {k: arr == k for k in sorted(set(keys))}


def decompose(
    frame: Frame,
    errors: dict[str, np.ndarray],
    evaluation: np.ndarray,
    joint: np.ndarray,
) -> dict[str, Any]:
    """Overall, per-season, per-month and top-price-tercile tables."""
    price_ok = evaluation & np.isfinite(frame.price)
    threshold = float(np.percentile(frame.price[price_ok], 100 * 2 / 3)) if price_ok.any() else None
    top = price_ok & (frame.price >= threshold) if threshold is not None else price_ok & False

    season_keys = [season_label(d) for d in frame.local_day]
    year_season_keys = [season_of_year(d) for d in frame.local_day]
    month_keys = [f"{d.year}-{d.month:02d}" for d in frame.local_day]

    def table(keys: list[str]) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for key, mask in group_indices(frame, keys).items():
            joint_sel = joint & mask
            eval_sel = evaluation & mask
            if not joint_sel.any():
                continue
            rows[key] = {
                "energy": energy_shares(frame, joint_sel),
                "error": error_stats(errors, frame.price, eval_sel),
                "error_top_price_tercile": error_stats(errors, frame.price, top & mask),
            }
        return rows

    return {
        "price_tercile_threshold_eur_per_kwh": threshold,
        "overall": {
            "energy": energy_shares(frame, joint),
            "error": error_stats(errors, frame.price, price_ok),
            "error_top_price_tercile": error_stats(errors, frame.price, top),
        },
        "by_season_of_year": table(year_season_keys),
        "by_season": table(season_keys),
        "by_month": table(month_keys),
    }


# ---------------------------------------------------------------------------
# P4 evidence: do the AppDaemon rules explain when the heat pump runs?
# ---------------------------------------------------------------------------


def local_day_prices(frame: Frame) -> dict[date, dict[int, float]]:
    """Hourly price per local day: ``{day: {local_hour: price}}``, mean of the hour's slots.

    An hour whose slots disagree or lack a price is left out (the caller skips the day).
    """
    out: dict[date, dict[int, list[float]]] = {}
    for day, hour, price in zip(frame.local_day, frame.local_hour.tolist(), frame.price.tolist()):
        if np.isfinite(price):
            out.setdefault(day, {}).setdefault(hour, []).append(price)
    hourly: dict[date, dict[int, float]] = {}
    for day, hours in out.items():
        hourly[day] = {}
        for hour, values in hours.items():
            if max(values) - min(values) < 1e-9:
                hourly[day][hour] = values[0]
    return hourly


def select_climate_hours(prices: dict[int, float], limit: float, min_hours: int) -> list[int]:
    """The hours the ``climate_control`` rule switches the heat mode on for one day.

    Hours priced below *limit* are all selected when there are at least *min_hours* of them,
    otherwise the *min_hours* cheapest hours (ties keep the earlier hour), as in the AppDaemon
    app.
    """
    hours = sorted(prices)
    cheap = [h for h in hours if prices[h] < limit]
    if len(cheap) >= min_hours:
        return cheap
    ordered = sorted(hours, key=lambda h: prices[h])
    return sorted(ordered[:min_hours])


def dhw_rule_hours(prices: dict[int, float], interval_hours: int) -> list[int]:
    """The hour of the cheapest price in each ``interval_hours`` block, as ``dhw_control`` does."""
    hours = sorted(prices)
    out = []
    for i in range(0, len(hours), interval_hours):
        block = hours[i : i + interval_hours]
        if block:
            out.append(min(block, key=lambda h: prices[h]))
    return out


def climate_evidence(frame: Frame) -> dict[str, Any]:
    """Share of heat-pump heating energy inside the hours the climate rule would select."""
    prices = local_day_prices(frame)
    days = sorted(prices)
    slot_lists: dict[date, list[int]] = {}
    for i, day in enumerate(frame.local_day):
        slot_lists.setdefault(day, []).append(i)
    slot_idx = {day: np.array(items) for day, items in slot_lists.items()}

    rows: dict[str, dict[str, float]] = {}
    skipped: dict[str, int] = {
        "no_full_price_day": 0,
        "no_price_history": 0,
        "min_hours_unknown": 0,
        "climate_disabled": 0,
        "heat_unknown": 0,
    }
    used_days = 0
    disabled_heat = 0.0
    for day in days:
        hourly = prices[day]
        if len(hourly) < 23:
            skipped["no_full_price_day"] += 1
            continue
        history = [
            v
            for k in range(1, PRICE_HISTORY_DAYS + 1)
            for v in prices.get(day - timedelta(days=k), {}).values()
        ]
        if len(history) < 3 * 24:
            skipped["no_price_history"] += 1
            continue
        idx = slot_idx[day]
        midnight = idx[0]
        min_hours = frame.climate_min_hours[midnight]
        enable = frame.climate_enable[midnight]
        if not np.isfinite(min_hours) or not np.isfinite(enable):
            skipped["min_hours_unknown"] += 1
            continue
        if enable < 0.5:
            skipped["climate_disabled"] += 1
            disabled_heat += float(np.nansum(frame.hp_heat[idx]))
            continue
        heat = frame.hp_heat[idx]
        known = np.isfinite(heat)
        if not known.any():
            skipped["heat_unknown"] += 1
            continue
        limit = CHEAP_PRICE_FACTOR * float(np.mean(history))
        chosen = set(select_climate_hours(hourly, limit, int(float(min_hours))))
        run_on = chosen | {h + 1 for h in chosen if h + 1 < 24}
        hours = frame.local_hour[idx]
        in_chosen = np.isin(hours, list(chosen))
        in_run_on = np.isin(hours, list(run_on))
        total = float(heat[known].sum())
        used_days += 1
        month = f"{day.year}-{day.month:02d}"
        for key in (month, "all", f"season:{season_of_year(day)}"):
            row = rows.setdefault(
                key,
                {
                    "days": 0,
                    "heat_kwh": 0.0,
                    "heat_in_selected_kwh": 0.0,
                    "heat_in_selected_or_next_hour_kwh": 0.0,
                    "selected_hours": 0.0,
                    "hours_in_day": 0.0,
                    "cheap_rule_days": 0,
                },
            )
            row["days"] += 1
            row["heat_kwh"] += total
            row["heat_in_selected_kwh"] += float(heat[known & in_chosen].sum())
            row["heat_in_selected_or_next_hour_kwh"] += float(heat[known & in_run_on].sum())
            row["selected_hours"] += len(chosen)
            row["hours_in_day"] += len(hourly)
            row["cheap_rule_days"] += int(
                sum(1 for h in hourly if hourly[h] < limit) >= int(float(min_hours))
            )
    for row in rows.values():
        row["share_in_selected"] = _safe_div(row["heat_in_selected_kwh"], row["heat_kwh"])
        row["share_in_selected_or_next_hour"] = _safe_div(
            row["heat_in_selected_or_next_hour_kwh"], row["heat_kwh"]
        )
        row["selected_time_share"] = _safe_div(row["selected_hours"], row["hours_in_day"])
        if row["share_in_selected"] is not None and row["selected_time_share"]:
            row["enrichment"] = row["share_in_selected"] / row["selected_time_share"]
    return {
        "rule": (
            "00:00:40 local, enabled flag and min_hours as of local midnight; hours priced below "
            "2/3 of the mean hourly price of the previous 7 days are all selected when there are "
            "at least min_hours of them, else the min_hours cheapest hours; heat mode on in "
            "those hours."
        ),
        "days_used": used_days,
        "days_skipped": skipped,
        "heat_kwh_on_disabled_days": disabled_heat,
        "by_group": {k: rows[k] for k in sorted(rows)},
    }


def after_force_on_mask(frame: Frame, hours: int = DHW_WINDOW_HOURS) -> np.ndarray:
    """Slots from any observed force-DHW-on slot up to *hours* hours later."""
    force_on = np.isfinite(frame.dhw_force_mean) & (frame.dhw_force_mean > 0)
    width = hours * 4
    spread = np.convolve(force_on.astype(np.int64), np.ones(width, dtype=np.int64))[: frame.n]
    return spread > 0


def dhw_evidence(frame: Frame) -> dict[str, Any]:
    """DHW energy and the force-DHW switch against the 12 h cheapest-hour rule.

    Only days whose ``dhw_enable`` flag is on at local midnight take part, since the rule does
    nothing otherwise.  The flag series starts in March 2025.
    """
    prices = local_day_prices(frame)
    slot_lists: dict[date, list[int]] = {}
    for i, day in enumerate(frame.local_day):
        slot_lists.setdefault(day, []).append(i)
    after_force = after_force_on_mask(frame)
    skipped = {"no_full_price_day": 0, "dhw_disabled_or_unknown": 0}
    result: dict[str, Any] = {
        "rule": (
            "00:00:30 local, one force-DHW switch-on per block of interval_hours at the cheapest "
            "hour of the block; the current configuration is 12 h"
        ),
        "window_hours_after_rule_hour": DHW_WINDOW_HOURS,
    }
    for interval in DHW_INTERVAL_HOURS:
        rows: dict[str, dict[str, float]] = {}
        skipped = {"no_full_price_day": 0, "dhw_disabled_or_unknown": 0}
        for day in sorted(prices):
            hourly = prices[day]
            if len(hourly) < 23:
                skipped["no_full_price_day"] += 1
                continue
            idx = np.array(slot_lists[day])
            enable = frame.dhw_enable[idx[0]]
            if not np.isfinite(enable) or enable < 0.5:
                skipped["dhw_disabled_or_unknown"] += 1
                continue
            hours = frame.local_hour[idx]
            dhw = frame.hp_dhw[idx]
            force = frame.dhw_force_mean[idx]
            chosen = dhw_rule_hours(hourly, interval)
            window = {h + k for h in chosen for k in range(DHW_WINDOW_HOURS)}
            in_hour = np.isin(hours, chosen)
            in_window = np.isin(hours, list(window))
            known = np.isfinite(dhw)
            force_on = np.isfinite(force) & (force > 0)
            for key in (f"{day.year}-{day.month:02d}", "all"):
                row = rows.setdefault(
                    key,
                    {
                        "days": 0,
                        "dhw_kwh": 0.0,
                        "dhw_in_rule_hour_kwh": 0.0,
                        "dhw_in_rule_window_kwh": 0.0,
                        "dhw_after_observed_force_on_kwh": 0.0,
                        "rule_hours": 0,
                        "rule_hours_with_force_on": 0,
                        "force_on_slots": 0,
                        "force_on_slots_in_rule_hour": 0,
                    },
                )
                row["days"] += 1
                row["dhw_kwh"] += float(dhw[known].sum())
                row["dhw_in_rule_hour_kwh"] += float(dhw[known & in_hour].sum())
                row["dhw_in_rule_window_kwh"] += float(dhw[known & in_window].sum())
                row["dhw_after_observed_force_on_kwh"] += float(dhw[known & after_force[idx]].sum())
                row["rule_hours"] += len(chosen)
                row["rule_hours_with_force_on"] += int(
                    sum(1 for h in chosen if (force_on & (hours == h)).any())
                )
                row["force_on_slots"] += int(force_on.sum())
                row["force_on_slots_in_rule_hour"] += int((force_on & in_hour).sum())
        for row in rows.values():
            row["share_dhw_in_rule_hour"] = _safe_div(row["dhw_in_rule_hour_kwh"], row["dhw_kwh"])
            row["share_dhw_in_rule_window"] = _safe_div(
                row["dhw_in_rule_window_kwh"], row["dhw_kwh"]
            )
            row["share_dhw_after_observed_force_on"] = _safe_div(
                row["dhw_after_observed_force_on_kwh"], row["dhw_kwh"]
            )
            row["share_rule_hours_with_force_on"] = _safe_div(
                row["rule_hours_with_force_on"], row["rule_hours"]
            )
            row["share_force_on_slots_in_rule_hour"] = _safe_div(
                row["force_on_slots_in_rule_hour"], row["force_on_slots"]
            )
        result[f"interval_{interval}h"] = {"days_skipped": skipped, "by_group": rows}
    return result


# ---------------------------------------------------------------------------
# G3 verdict, markdown and writing
# ---------------------------------------------------------------------------


def g3_indicator(decomp: dict[str, Any]) -> dict[str, Any]:
    """Which observable components carry a material share of the LF-0 error variance."""
    out: dict[str, Any] = {"material_share_threshold": MATERIAL_SHARE}
    for scope in ("error", "error_top_price_tercile"):
        stats = decomp["overall"][scope]
        shares = {c: (stats.get(c) or {}).get("var_share") for c in COMPONENTS}
        out[scope] = {
            "var_share": shares,
            "material_observable": [
                c
                for c in ("servers", "heat_pump")
                if shares.get(c) is not None and shares[c] >= MATERIAL_SHARE
            ],
            "headroom_share": {
                c: (stats.get(c) or {}).get("headroom_share")
                for c in ("servers", "heat_pump", "residual")
            },
        }
    return out


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _num(x: float | None, digits: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    d = summary["decomposition"]
    lines = [
        "# P3 sub-load decomposition (private)",
        "",
        f"- feature set: `{summary['inputs']['feature_set_id']}`",
        f"- canonical dataset: `{summary['inputs']['dataset_id']}` "
        f"(window {summary['inputs']['dataset_window']})",
        f"- LF-0 window: {summary['lf0']['window_days']} local days, causal; common mask for every "
        "series, so the component errors add up to the main error "
        f"(max deviation {summary['lf0']['max_additivity_deviation_kwh']:.2e} kWh)",
        f"- joint slots: {summary['joint']['slots']} of {summary['joint']['grid_slots']}; "
        f"excluded as faults (servers + heat pump > main by more than "
        f"{summary['joint']['violation_tolerance_kwh']} kWh): "
        f"{summary['joint']['fault_slots_excluded']}",
        f"- evaluation slots (>= {summary['lf0']['min_history_days']} days of history): "
        f"{summary['joint']['evaluation_slots']}",
        "- top price tercile: price >= "
        f"{_num(d['price_tercile_threshold_eur_per_kwh'], 4)} EUR/kWh",
        "",
        "Error = forecast - measured, kWh per slot. `var share` = Cov(e_c, e_main) / Var(e_main); "
        "the three components sum to 100%. `pw` = sum(price x |error|) in EUR (proxy). "
        "`headroom` = reduction of pw error if the component were forecast perfectly.",
        "",
    ]

    def err_table(title: str, scope: str) -> None:
        lines.extend([f"## {title}", ""])
        lines.append(
            "| component | MAE kWh | bias kWh | var share | standalone var ratio | pw EUR "
            "| pw share | headroom |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        stats = d["overall"][scope]
        for name in ("main", "main_production_mask", *DETAIL_COMPONENTS):
            e = stats.get(name)
            if not e:
                continue
            lines.append(
                f"| {name} | {_num(e['mae_kwh'], 4)} | {_num(e['bias_kwh'], 4)} "
                f"| {_pct(e.get('var_share'))} | {_num(e.get('standalone_var_ratio'), 2)} "
                f"| {_num(e['price_weighted_abs_error'], 1)} "
                f"| {_pct(e.get('price_weighted_share_standalone'))} "
                f"| {_pct(e.get('headroom_share'))} |"
            )
        lines.append("")

    err_table("LF-0 error, all evaluation slots", "error")
    err_table("LF-0 error, top price tercile", "error_top_price_tercile")

    def energy_table(title: str, key: str) -> None:
        lines.extend([f"## {title}", ""])
        lines.append(
            "| group | slots | main kWh | servers | heat pump | heat | dhw | cool | residual "
            "| main MAE kWh | var share srv / hp / res | var share hp, top tercile |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name, row in d[key].items():
            en, er, top = row["energy"], row["error"], row["error_top_price_tercile"]

            def vs(stats: dict[str, Any], comp: str) -> float | None:
                return (stats.get(comp) or {}).get("var_share")

            lines.append(
                f"| {name} | {en['slots']} | {en['main_kwh']:.0f} | {_pct(en['servers_share'])} "
                f"| {_pct(en['heat_pump_share'])} | {_pct(en['hp_heat_share'])} "
                f"| {_pct(en['hp_dhw_share'])} | {_pct(en['hp_cool_share'])} "
                f"| {_pct(en['residual_share'])} "
                f"| {_num((er.get('main') or {}).get('mae_kwh'), 4)} "
                f"| {_pct(vs(er, 'servers'), 0)} / {_pct(vs(er, 'heat_pump'), 0)} / "
                f"{_pct(vs(er, 'residual'), 0)} | {_pct(vs(top, 'heat_pump'), 0)} |"
            )
        lines.append("")

    energy_table("By season of year", "by_season_of_year")
    energy_table("By season instance", "by_season")
    energy_table("By month", "by_month")

    p4 = summary["p4_evidence"]
    cl = p4["climate"]
    lines.extend(
        [
            "## P4 evidence: climate rule vs heat-pump heating energy",
            "",
            f"Days used: {cl['days_used']}; skipped: {cl['days_skipped']}; heating energy on the "
            f"disabled days: {cl['heat_kwh_on_disabled_days']:.1f} kWh",
            "",
            "| group | days | heat kWh | share in selected hours | with next hour | "
            "selected time share | enrichment |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for name, row in cl["by_group"].items():
        if name.startswith("20") and len(name) == 7 and name[5:] not in ("01", "04", "07", "10"):
            continue
        lines.append(
            f"| {name} | {row['days']} | {row['heat_kwh']:.0f} | {_pct(row['share_in_selected'])} "
            f"| {_pct(row['share_in_selected_or_next_hour'])} "
            f"| {_pct(row['selected_time_share'])} | {_num(row.get('enrichment'), 2)} |"
        )
    lines.extend(["", "## P4 evidence: DHW against the cheapest-slot rule", ""])
    for interval in DHW_INTERVAL_HOURS:
        block = p4["dhw"][f"interval_{interval}h"]
        allrow = block["by_group"].get("all")
        if not allrow:
            continue
        lines.append(
            f"- interval {interval} h ({allrow['days']} enabled days, skipped "
            f"{block['days_skipped']}): DHW energy in the rule hour "
            f"{_pct(allrow['share_dhw_in_rule_hour'])}, within {DHW_WINDOW_HOURS} h of it "
            f"{_pct(allrow['share_dhw_in_rule_window'])}, within {DHW_WINDOW_HOURS} h after any "
            f"observed force-on {_pct(allrow['share_dhw_after_observed_force_on'])}; rule hours "
            f"with the force switch on {_pct(allrow['share_rule_hours_with_force_on'])}; "
            f"force-on slots inside rule hours "
            f"{_pct(allrow['share_force_on_slots_in_rule_hour'])}"
        )
    g3 = summary["g3"]
    lines.extend(["", "## G3 indicator", ""])
    for scope in ("error", "error_top_price_tercile"):
        item = g3[scope]
        lines.append(
            f"- {scope}: var share "
            + ", ".join(f"{k} {_pct(v)}" for k, v in item["var_share"].items())
            + f"; material observable components (>= {_pct(g3['material_share_threshold'], 0)}): "
            f"{item['material_observable'] or 'none'}"
        )
    lines.append("")
    return "\n".join(lines)


def _replace_atomic(path: Path, payload: bytes) -> None:
    """Write a derived report file atomically (a report is regenerated, not content-addressed)."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(payload)
    tmp.chmod(0o600)
    os.replace(tmp, path)


def write_report(
    *,
    features_dir: str | Path,
    dataset_dir: str | Path,
    data_dir: str | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> Path:
    """Compute the decomposition and write ``p3-report/summary.{json,md}`` in the feature set."""
    from battery_optimizer.benchmark import load_dataset

    root = resolve_data_dir(data_dir)
    feats, manifest = features.load_features(features_dir)
    dataset = load_dataset(dataset_dir, data_dir=root)
    frame = build_frame(dataset, feats)
    joint = joint_mask(frame)
    base_known = frame.usable & np.isfinite(frame.servers) & np.isfinite(frame.heat_pump)
    errors, evaluation, lf0_meta = lf0_errors(frame, joint, window_days=window_days)
    decomp = decompose(frame, errors, evaluation, joint)
    summary: dict[str, Any] = {
        "inputs": {
            "feature_set_id": manifest["feature_set_id"],
            "dataset_id": dataset.dataset_id,
            "dataset_window": (
                f"{dataset.manifest.get('requested_window', {}).get('start')} .. "
                f"{dataset.manifest.get('requested_window', {}).get('end')}"
            ),
        },
        "lf0": lf0_meta,
        "joint": {
            "grid_slots": frame.n,
            "slots": int(joint.sum()),
            "fault_slots_excluded": int((base_known & ~joint).sum()),
            "violation_tolerance_kwh": VIOLATION_TOLERANCE_KWH,
            "evaluation_slots": int(evaluation.sum()),
        },
        "decomposition": decomp,
        "p4_evidence": {"climate": climate_evidence(frame), "dhw": dhw_evidence(frame)},
        "g3": g3_indicator(decomp),
    }
    out_dir = Path(features_dir).expanduser() / "p3-report"
    _replace_atomic(out_dir / "summary.json", features.json_bytes(summary))
    _replace_atomic(out_dir / "summary.md", render_markdown(summary).encode())
    log.info("wrote %s", out_dir)
    return out_dir
