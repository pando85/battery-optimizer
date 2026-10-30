#!/usr/bin/env python3
"""Read-only data audit for the C2 climate models (PR 1 of the C2 design).

Answers, on the real climate feature set and the dataset prices, the questions the thermal models
and the gate depend on: how coarse the thermostat readings are, where the data has gaps, how
quantized the energy counters are, whether the heat-pump return temperature can serve as a second
measurement of the floor slab, the heating curve, how well the legacy-rule replay predicts when
heat is delivered, how the Shelly channel 2 compares with HeishaMon per season, whether the
outdoor sensor sees the sun, how many slots survive the training filters per season and which
``min_hours`` regimes season S3 contains.

Usage:
    uv run python hack/climate_audit.py \\
        --feature-set <climate feature set id or prefix> \\
        --dataset ~/data/<store>/analysis/v1/<dataset_id> \\
        [--config config/climate.yaml] [--data-dir ~/data/<store>] [--name <suffix>]

The report is written to ``<data_dir>/analysis/v1/climate-audit/<feature set>-<config>[-name]/``
as ``report.json`` and ``report.md``; nothing else is written and the repository is never
touched.  The site coordinates come from ``<data_dir>/climate/site.yaml`` (private, never
committed).  The report holds private figures: keep it in the data dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "src"))

from battery_optimizer.climate.config import load_climate_config  # noqa: E402
from battery_optimizer.climate.data import (  # noqa: E402
    FILTERS,
    ROOMS,
    ClimateData,
    load_climate_data,
)
from battery_optimizer.extract_prom import ENTITY_SPECS  # noqa: E402
from battery_optimizer.features import (  # noqa: E402
    CLIMATE_SIGNALS,
    FEATURES_DIRNAME,
    FeatureError,
    RawSeries,
    Signal,
    json_bytes,
    load_features,
    load_signal_samples,
)
from battery_optimizer.paths import resolve_data_dir, write_part_atomic  # noqa: E402

SEASONS = ("S1", "S2", "S3")
SLOT_H = 0.25
#: A heating slot counts as "heat delivered" at or above this thermal energy (kWh per slot).
HEAT_ON_KWH = 0.05

THERMOSTATS = {room: f"climate.thermostat_{room}" for room in ROOMS}
#: ``load_signal_samples`` keys a pinned-metric series ``entity#<metric name>``.
TEMP_KEY = {r: f"{e}#{ENTITY_SPECS[e].metric}" for r, e in THERMOSTATS.items()}
SETPOINT_KEY = {r: f"{e}#{ENTITY_SPECS[f'{e}#setpoint'].metric}" for r, e in THERMOSTATS.items()}
RAW_TEMPERATURES = {
    "outdoor": "sensor.aquarea_outside_ambient_temperature",
    "flow": "sensor.aquarea_outlet_temperature",
    "return": "sensor.aquarea_inlet_temperature",
    "tank": "sensor.aquarea_tank_temperature",
    "flow_target": "sensor.aquarea_outlet_target_temperature",
}
RAW_COUNTERS = {
    "heat_produced": "sensor.aquarea_heat_energy_produced",
    "heat_consumed": "sensor.aquarea_heat_energy_consumed",
    "channel2": "sensor.shellyem_main_channel_2_energy",
    "starts": "sensor.aquarea_compressor_start_stop_counter",
}


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _pct(values: np.ndarray, qs=(5, 25, 50, 75, 95)) -> dict[str, float | None]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"p{q}": None for q in qs} | {"n": 0}
    return {f"p{q}": float(np.percentile(v, q)) for q in qs} | {"n": int(v.size)}


def _rate(num: float, den: float) -> float | None:
    return float(num / den) if den > 0 else None


def _runs(flag: np.ndarray) -> list[tuple[int, int]]:
    """``(start, length)`` of every maximal run of True."""
    f = np.concatenate(([0], np.asarray(flag, dtype=np.int8), [0]))
    edges = np.flatnonzero(np.diff(f))
    return [(int(a), int(b - a)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _iso(t_us: int) -> str:
    return datetime.fromtimestamp(t_us / 1e6, tz=UTC).strftime("%Y-%m-%dT%H:%MZ")


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
        return None
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def _ols(x: np.ndarray, y: np.ndarray) -> dict[str, float | None]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) < 1e-9:
        return {"slope": None, "intercept": None, "r": None, "n": int(ok.sum())}
    slope, intercept = np.polyfit(x[ok], y[ok], 1)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r": _corr(x, y),
        "n": int(ok.sum()),
    }


def _season_slots(d: ClimateData, season: str, months: str = "gate") -> np.ndarray:
    return d.season_slots(season, months=months)


# ---------------------------------------------------------------------------
# Raw-sample sections
# ---------------------------------------------------------------------------


def resolution(raw: RawSeries) -> dict[str, Any]:
    """Reading resolution of one raw series: grid step, value grids and how often it changes."""
    t = raw.t_us.astype(np.int64)
    v = raw.values.astype(float)
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    if v.size < 2:
        return {"samples": int(v.size)}
    distinct = np.unique(np.round(v, 4))
    gaps = np.diff(distinct)
    changes = np.flatnonzero(np.diff(v) != 0)
    steps = np.round(np.abs(np.diff(v)[changes]), 3)
    top = Counter(steps.tolist()).most_common(3)
    span_days = (t[-1] - t[0]) / 86_400e6
    change_t = t[changes + 1]
    between = np.diff(change_t) / 60e6
    out: dict[str, Any] = {
        "samples": int(v.size),
        "distinct_values": int(distinct.size),
        "min_distinct_gap": float(gaps.min()) if gaps.size else None,
        "most_common_steps": [{"step": s, "share": c / max(steps.size, 1)} for s, c in top],
        "share_on_half_grid": float(np.mean(np.abs(v * 2 - np.round(v * 2)) < 1e-6)),
        "share_on_integer_grid": float(np.mean(np.abs(v - np.round(v)) < 1e-6)),
        "share_on_tenth_grid": float(np.mean(np.abs(v * 10 - np.round(v * 10)) < 1e-6)),
        "value_changes_per_day": float(changes.size / span_days) if span_days > 0 else None,
        "minutes_between_changes": _pct(between),
    }
    return out


def counter_quantum(raw: RawSeries) -> dict[str, Any]:
    """Increment structure of a monotonic counter (smallest step, how often it ticks)."""
    t, v = raw.t_us.astype(np.int64), raw.values.astype(float)
    up = np.diff(v) > 0
    inc = np.diff(v)[up]
    if inc.size == 0:
        return {"samples": int(v.size), "increments": 0}
    dt = np.diff(t)[up] / 60e6
    rounded = np.round(inc, 4)
    top = Counter(rounded.tolist()).most_common(3)
    return {
        "samples": int(v.size),
        "increments": int(inc.size),
        "smallest_increment": float(rounded.min()),
        "most_common_increments": [{"step": s, "share": c / inc.size} for s, c in top],
        "increment": _pct(inc),
        "minutes_between_increments": _pct(dt),
        "share_of_samples_that_tick": float(up.mean()),
        "rollbacks": int((np.diff(v) < 0).sum()),
    }


# ---------------------------------------------------------------------------
# Slot-level sections (pure functions of ClimateData)
# ---------------------------------------------------------------------------


def gaps_per_signal(d: ClimateData, coverage: dict[str, np.ndarray]) -> dict[str, Any]:
    """Unknown-slot runs per feature signal (coverage below the builder's minimum)."""
    out: dict[str, Any] = {}
    for name, cov in sorted(coverage.items()):
        unknown = ~(cov >= 0.8)
        runs = _runs(unknown)
        long_runs = sorted(runs, key=lambda r: -r[1])[:5]
        out[name] = {
            "unknown_share": float(unknown.mean()),
            "runs": len(runs),
            "runs_ge_1h": sum(1 for _s, n in runs if n >= 4),
            "runs_ge_6h": sum(1 for _s, n in runs if n >= 24),
            "runs_ge_24h": sum(1 for _s, n in runs if n >= 96),
            "longest": [
                {"start": _iso(int(d.starts_us[s])), "hours": n * SLOT_H} for s, n in long_runs
            ],
        }
    return out


def stale_runs(d: ClimateData, min_hours: float = 24.0) -> dict[str, Any]:
    """Runs where a comfort or outdoor slot mean does not change at all (held values)."""
    out: dict[str, Any] = {}
    series = {f"room_{r}": d.room_temp[r] for r in ROOMS} | {"outdoor": d.t_out}
    for name, x in series.items():
        same = np.zeros(x.size, dtype=bool)
        same[1:] = (x[1:] == x[:-1]) & np.isfinite(x[1:])
        runs = [(s, n) for s, n in _runs(same) if n * SLOT_H >= min_hours]
        out[name] = [{"start": _iso(int(d.starts_us[s])), "hours": n * SLOT_H} for s, n in runs[:8]]
    return out


def counter_slot_quantization(d: ClimateData) -> dict[str, Any]:
    """The heat-produced counter at slot level: zero deltas while running, daily agreement."""
    q = d.q_heat_kwh
    delta = d.columns["hp_heat_energy_out_delta"]
    both = np.isfinite(q) & np.isfinite(delta)
    run = both & (q >= HEAT_ON_KWH)
    out: dict[str, Any] = {
        "slots_compared": int(both.sum()),
        "slots_running": int(run.sum()),
        "running_with_zero_counter_delta": _rate(((delta == 0) & run).sum(), run.sum()),
        "idle_with_counter_delta": _rate(
            ((delta > 0) & both & (q < HEAT_ON_KWH)).sum(), (both & (q < HEAT_ON_KWH)).sum()
        ),
        "slot_corr": _corr(q[both], delta[both]),
        "running_delta_over_power_integral": _pct(delta[run] / q[run]),
    }
    nz = delta[both & (delta > 0)]
    out["nonzero_delta_kwh"] = _pct(nz)
    if nz.size:
        out["smallest_nonzero_delta_kwh"] = float(nz.min())
    return out


def daily_counter_agreement(d: ClimateData, min_daily_kwh: float) -> dict[str, Any]:
    check = d.counter_check(min_daily_kwh)
    out: dict[str, Any] = {
        "days": int(check["ratio"].size),
        "ratio_power_over_counter": _pct(check["ratio"]),
    }
    if check["ratio"].size:
        season_of = {day: d.day_season[j] for j, (day, _l, _h) in enumerate(d.days)}
        for season in SEASONS:
            sel = np.array([season_of[x] == season for x in check["days"]], dtype=bool)
            out[season] = _pct(check["ratio"][sel])
    return out


def return_temperature_proxy(d: ClimateData, min_flow: float) -> dict[str, Any]:
    """Return temperature in steady circulation as a slab proxy (S1/S2/S3, gate months)."""
    c = d.columns
    flow_rate = c["pump_flow_mean"]
    ret, supply = c["return_temp_mean_c"], c["flow_temp_mean_c"]
    with np.errstate(invalid="ignore"):
        circulating = (flow_rate >= min_flow) & ~d.dhw & ~d.cooling
    out: dict[str, Any] = {"min_flow_l_per_min": min_flow}
    prev = np.zeros(d.n, dtype=bool)
    prev[1:] = circulating[:-1]
    steady = circulating & prev  # at least one full slot (15 min) into a circulation run
    for season in SEASONS:
        sel = _season_slots(d, season) & d.train
        s = sel & steady & np.isfinite(ret) & np.isfinite(d.t_air) & np.isfinite(supply)
        gate = sel & np.isfinite(d.t_air)
        entry: dict[str, Any] = {
            "slots": int(s.sum()),
            "share_of_gate_slots": _rate(s.sum(), gate.sum()),
        }
        if s.sum() > 50:
            r, ta, sup, tout = ret[s], d.t_air[s], supply[s], d.t_out[s]
            # within-run noise: slot-to-slot change of the return temperature
            both = s[1:] & s[:-1]
            step = (ret[1:] - ret[:-1])[both]
            entry.update(
                {
                    "return_minus_t_air_k": _pct(r - ta),
                    "flow_minus_return_k": _pct(sup - r),
                    "return_step_std_k": float(np.std(step) / np.sqrt(2)) if step.size else None,
                    "return_std_k": float(np.std(r)),
                    "corr_return_t_air": _corr(r, ta),
                    "corr_return_flow_temp": _corr(r, sup),
                    "corr_return_t_out": _corr(r, tout),
                }
            )
            flow_only = np.column_stack([np.ones(r.size), sup])
            beta0, *_ = np.linalg.lstsq(flow_only, r, rcond=None)
            r2_flow = 1 - (r - flow_only @ beta0).var() / r.var() if r.var() > 0 else None
            design = np.column_stack([np.ones(r.size), sup, ta, tout])
            beta, *_ = np.linalg.lstsq(design, r, rcond=None)
            resid = r - design @ beta
            entry["regression_return_on_flow_t_air_t_out"] = {
                "coef_flow": float(beta[1]),
                "coef_t_air": float(beta[2]),
                "coef_t_out": float(beta[3]),
                "r2": float(1 - resid.var() / r.var()) if r.var() > 0 else None,
                "r2_flow_temp_only": None if r2_flow is None else float(r2_flow),
                "resid_std_k": float(resid.std()),
            }
            # thermal power implied by flow x dT, against the integrated output (diagnostic only)
            flow_kw = flow_rate[s] * (sup - r) * 4.186 / 60.0
            q_kw = d.q_heat_kw[s]
            ratio = (
                flow_kw[np.isfinite(q_kw) & (q_kw > 0.2)] / q_kw[np.isfinite(q_kw) & (q_kw > 0.2)]
            )
            entry["flow_times_dt_over_q_heat"] = _pct(ratio)
            # how much of the slow signal is left once the daily mean is removed
            days = d.day_of_slot[s]
            slow = np.array([r[days == k].mean() for k in np.unique(days)])
            entry["daily_mean_return_std_k"] = float(slow.std()) if slow.size > 2 else None
            entry["daily_means_n"] = int(slow.size)
        out[season] = entry
    return out


def heating_curve(d: ClimateData, min_flow: float) -> dict[str, Any]:
    """Flow target (and measured flow) against outdoor temperature in heating slots."""
    c = d.columns
    target, supply = c["flow_target_temp_mean_c"], c["flow_temp_mean_c"]
    heat_on = (d.q_heat_kwh >= HEAT_ON_KWH) & ~d.dhw & ~d.cooling
    ok = heat_on & np.isfinite(target) & np.isfinite(d.t_out) & d.train
    edges = np.arange(-6.0, 22.0, 2.0)
    out: dict[str, Any] = {"bin_k": 2.0, "slots": int(ok.sum()), "bins": []}
    for lo in edges:
        sel = ok & (d.t_out >= lo) & (d.t_out < lo + 2.0)
        if sel.sum() < 20:
            continue
        row: dict[str, Any] = {
            "t_out_from": float(lo),
            "n": int(sel.sum()),
            "flow_target": _pct(target[sel], (25, 50, 75)),
            "flow_measured_median": float(np.nanmedian(supply[sel])),
        }
        for season in SEASONS:
            s = sel & _season_slots(d, season, "fit")
            row[season] = float(np.median(target[s])) if s.sum() >= 20 else None
        out["bins"].append(row)
    xs = np.array([b["t_out_from"] + 1.0 for b in out["bins"]])
    ys = np.array([b["flow_target"]["p50"] for b in out["bins"]])
    if xs.size >= 3:
        slope, intercept = np.polyfit(xs, ys, 1)
        out["linear_fit_on_bin_medians"] = {
            "slope_k_per_k": float(slope),
            "intercept": float(intercept),
        }
    out["flow_target_spread_within_bin_k"] = _pct(
        np.array([b["flow_target"]["p75"] - b["flow_target"]["p25"] for b in out["bins"]])
    )
    out["tracking_error_flow_minus_target_k"] = _pct((supply - target)[ok])
    months = np.array([d.days[j][0].strftime("%Y-%m") for j in d.day_of_slot], dtype=object)
    by_month: dict[str, Any] = {}
    for month in sorted(set(months[ok])):
        sel = ok & (months == month)
        values, counts = np.unique(np.round(target[sel]), return_counts=True)
        keep = counts / counts.sum() >= 0.05
        by_month[month] = {
            "slots": int(sel.sum()),
            "values_with_5pct_share": {
                str(int(v)): float(c / counts.sum())
                for v, c in zip(values[keep], counts[keep], strict=True)
            },
            "corr_with_t_out": _corr(target[sel], d.t_out[sel]),
        }
    out["flow_target_by_month"] = by_month
    return out


def legacy_mask_hit_rate(d: ClimateData) -> dict[str, Any]:
    """How often heat is delivered when the replayed rule permits it, and when it does not."""
    heat_on = (d.q_heat_kwh >= HEAT_ON_KWH) & ~d.dhw & ~d.cooling
    known = np.isfinite(d.q_heat_kwh) & ~d.dhw & ~d.cooling & np.isfinite(d.u)
    out: dict[str, Any] = {"heat_on_threshold_kwh": HEAT_ON_KWH}

    def block(sel: np.ndarray) -> dict[str, Any]:
        s = sel & known
        u1, u0 = s & (d.u == 1.0), s & (d.u == 0.0)
        energy = np.where(np.isfinite(d.q_heat_kwh), d.q_heat_kwh, 0.0)
        return {
            "slots_u1": int(u1.sum()),
            "slots_u0": int(u0.sum()),
            "heat_on_given_u1": _rate((heat_on & u1).sum(), u1.sum()),
            "heat_on_given_u0": _rate((heat_on & u0).sum(), u0.sum()),
            "heat_energy_share_in_u1": _rate(energy[u1].sum(), energy[s].sum()),
            "u1_share_of_slots": _rate(u1.sum(), s.sum()),
            "mean_q_kwh_per_slot_u1": float(energy[u1].mean()) if u1.any() else None,
            "mean_q_kwh_per_slot_u0": float(energy[u0].mean()) if u0.any() else None,
        }

    for season in SEASONS:
        out[season] = block(_season_slots(d, season))
    # per day: share of the day's heat inside the permitted hours
    shares: list[float] = []
    heat_days = 0
    for j, (_label, lo, hi) in enumerate(d.days):
        if not d.day_complete[j] or not (d.day_gate_month[j] and d.day_season[j] == "S3"):
            continue
        sl = slice(lo, hi)
        if not np.isfinite(d.u[sl]).all():
            continue
        e = np.where(np.isfinite(d.q_heat_kwh[sl]) & ~d.dhw[sl], d.q_heat_kwh[sl], 0.0)
        if e.sum() > 1.0:
            heat_days += 1
            shares.append(float(e[d.u[sl] == 1.0].sum() / e.sum()))
    out["S3_per_day_heat_share_inside_mask"] = _pct(np.array(shares)) | {
        "days_with_heat": heat_days
    }
    # timing check: hit rate against the mask shifted by whole hours (a shift peaks off 0)
    shifts: dict[str, Any] = {}
    sel = _season_slots(d, "S3") & known
    for k in range(-3, 4):
        u = np.roll(d.u, 4 * k)
        s = sel & np.isfinite(u)
        shifts[str(k)] = {
            "heat_on_given_u1": _rate((heat_on & s & (u == 1.0)).sum(), (s & (u == 1.0)).sum()),
            "heat_on_given_u0": _rate((heat_on & s & (u == 0.0)).sum(), (s & (u == 0.0)).sum()),
        }
    out["S3_shifted_mask_hours"] = shifts
    reasons = Counter()
    for j, (_label, _lo, _hi) in enumerate(d.days):
        if d.day_season[j] and d.day_complete[j] and d.day_fit_month[j]:
            reasons[(d.day_season[j], d.u_reason[j] or "replayed")] += 1
    out["day_status"] = {f"{s}:{r}": n for (s, r), n in sorted(reasons.items())}
    return out


def channel2_ratio(d: ClimateData) -> dict[str, Any]:
    ch2 = d.columns["branch_aero_kwh"]
    hp_total = d.e_heat_kwh + d.columns["hp_dhw_kwh"] + d.columns["hp_cool_kwh"]
    out: dict[str, Any] = {}
    for season in SEASONS:
        sel = _season_slots(d, season)
        daily_ratio: list[float] = []
        for j, (_label, lo, hi) in enumerate(d.days):
            if not (d.day_complete[j] and d.day_season[j] == season and d.day_gate_month[j]):
                continue
            a, b = ch2[lo:hi], hp_total[lo:hi]
            if np.isfinite(a).all() and np.isfinite(b).all() and b.sum() > 1.0:
                daily_ratio.append(float(a.sum() / b.sum()))
        both = sel & np.isfinite(ch2) & np.isfinite(hp_total)
        idle = both & (hp_total == 0.0)
        ols = _ols(hp_total[both], ch2[both])
        out[season] = {
            "daily_ratio": _pct(np.array(daily_ratio)),
            "idle_channel2_watts": _pct(ch2[idle] / SLOT_H * 1000.0),
            "slot_regression_ch2_on_heishamon": ols,
            "running_excess_watts_at_slot_level": _pct(
                (ch2[both & (hp_total > 0.05)] - hp_total[both & (hp_total > 0.05)])
                / SLOT_H
                * 1000.0
            ),
        }
    return out


def outdoor_sensor_sun(d: ClimateData) -> dict[str, Any]:
    """Does the outdoor sensor see the sun?  Diurnal anomaly against the clear-sky proxy."""
    anomaly = np.full(d.n, np.nan)
    day_amp, day_sun = [], []
    for j, (_label, lo, hi) in enumerate(d.days):
        if not (d.day_complete[j] and d.day_gate_month[j] and d.day_season[j]):
            continue
        x = d.t_out[lo:hi]
        if np.isfinite(x).mean() < 0.9:
            continue
        anomaly[lo:hi] = x - np.nanmean(x)
        day_amp.append(float(np.nanmax(x) - np.nanmin(x)))
        day_sun.append(float(d.solar[lo:hi].sum() * SLOT_H))
    out: dict[str, Any] = {"days": len(day_amp)}
    ok = np.isfinite(anomaly)
    out["anomaly_vs_solar"] = _ols(d.solar[ok], anomaly[ok])
    profile = []
    for hour in range(24):
        s = ok & (d.local_hour == hour)
        profile.append(float(np.median(anomaly[s])) if s.any() else None)
    out["median_anomaly_by_local_hour_k"] = profile
    peak_hour = int(np.nanargmax(np.array([np.nan if p is None else p for p in profile])))
    out["peak_hour_local"] = peak_hour
    # when does the sun peak?  the mean solar proxy by local hour in the same days
    sun_by_hour = [float(d.solar[ok & (d.local_hour == h)].mean()) for h in range(24)]
    out["solar_peak_hour_local"] = int(np.argmax(sun_by_hour))
    out["midday_minus_night_k"] = float(
        np.nanmedian(anomaly[ok & (d.local_hour >= 12) & (d.local_hour <= 15)])
        - np.nanmedian(anomaly[ok & (d.local_hour >= 2) & (d.local_hour <= 6)])
    )
    from scipy.stats import spearmanr

    if len(day_amp) > 5 and np.std(day_sun) > 0 and np.std(day_amp) > 0:
        out["daily_amplitude_vs_clear_sky_sum_spearman"] = float(
            spearmanr(day_sun, day_amp).statistic
        )
        out["daily_amplitude_k"] = _pct(np.array(day_amp))
    # self-heating / plume: compressor on against off in the same night, sun below horizon
    freq = d.columns["compressor_freq_mean"]
    diffs = []
    for j, (_label, lo, hi) in enumerate(d.days):
        if not (d.day_complete[j] and d.day_gate_month[j] and d.day_season[j]):
            continue
        sl = slice(lo, hi)
        night = (d.solar[sl] == 0.0) & np.isfinite(d.t_out[sl]) & np.isfinite(freq[sl])
        on, off = night & (freq[sl] > 0), night & (freq[sl] == 0)
        if on.sum() >= 4 and off.sum() >= 4:
            diffs.append(float(d.t_out[sl][on].mean() - d.t_out[sl][off].mean()))
    out["night_compressor_on_minus_off_k"] = _pct(np.array(diffs)) | {"days": len(diffs)}
    # a running unit that disturbs the sensor steps it at start and the other way at stop
    dark = np.zeros(d.n, dtype=bool)
    dark[1:-1] = (d.solar[:-2] == 0) & (d.solar[1:-1] == 0) & (d.solar[2:] == 0)
    known = np.isfinite(freq) & np.isfinite(d.t_out)
    change = np.full(d.n, np.nan)
    change[1:-1] = d.t_out[2:] - d.t_out[:-2]
    prev_on = np.zeros(d.n, dtype=bool)
    prev_on[1:] = freq[:-1] > 0
    with np.errstate(invalid="ignore"):
        start = dark & known & ~prev_on & (freq > 0)
        stop = dark & known & prev_on & (freq == 0)
        any_night = dark & known
    out["night_t_out_change_over_two_slots_k"] = {
        "at_compressor_start": _pct(change[start], (25, 50, 75)),
        "at_compressor_stop": _pct(change[stop], (25, 50, 75)),
        "all_dark_slots": _pct(change[any_night], (25, 50, 75)),
    }
    return out


def thermostat_signal(d: ClimateData) -> dict[str, Any]:
    """Slot-level behaviour of T_air: step sizes, stalls and the daily swing (gate months)."""
    out: dict[str, Any] = {}
    for season in SEASONS:
        sel = _season_slots(d, season) & d.train
        step = np.full(d.n, np.nan)
        step[1:] = d.t_air[1:] - d.t_air[:-1]
        both = sel.copy()
        both[0] = False
        both &= np.roll(sel, 1) & np.isfinite(step)
        swing = []
        for j, (_label, lo, hi) in enumerate(d.days):
            if d.day_complete[j] and d.day_season[j] == season and d.day_gate_month[j]:
                x = d.t_air[lo:hi]
                if np.isfinite(x).mean() > 0.9:
                    swing.append(float(np.nanmax(x) - np.nanmin(x)))
        out[season] = {
            "abs_step_per_slot_k": _pct(np.abs(step[both])),
            "share_of_slots_unchanged": _rate((step[both] == 0).sum(), both.sum()),
            "daily_range_k": _pct(np.array(swing)),
        }
    return out


def comfort_realized(d: ClimateData, low: float) -> dict[str, Any]:
    """What legacy delivers: hours and degree-hours below the comfort floor, per room and season."""
    out: dict[str, Any] = {"low_c": low}
    for season in SEASONS:
        sel = _season_slots(d, season)
        days = max(int(sel.sum() / 96), 1)
        entry: dict[str, Any] = {}
        for room in d.comfort_rooms:
            x = d.room_temp[room]
            ok = sel & np.isfinite(x)
            below = np.maximum(0.0, low - x[ok])
            entry[room] = {
                "share_slots_below": _rate((x[ok] < low).sum(), ok.sum()),
                "kh_below_per_day": float(below.sum() * SLOT_H / days),
                "p5_temp_c": float(np.percentile(x[ok], 5)) if ok.any() else None,
            }
        ok = sel & np.isfinite(d.t_air)
        entry["t_air"] = {
            "share_slots_below": _rate((d.t_air[ok] < low).sum(), ok.sum()),
            "p5_temp_c": float(np.percentile(d.t_air[ok], 5)) if ok.any() else None,
            "p50_temp_c": float(np.percentile(d.t_air[ok], 50)) if ok.any() else None,
        }
        out[season] = entry
    return out


def filter_survival(d: ClimateData) -> dict[str, Any]:
    """Share of slots surviving the training filters, per season (fit and gate months)."""
    out: dict[str, Any] = {}
    for season in SEASONS:
        for months in ("fit", "gate"):
            sel = _season_slots(d, season, months)
            total = int(sel.sum())
            if not total:
                continue
            entry: dict[str, Any] = {"slots": total}
            for name in FILTERS:
                entry[f"drops_{name}"] = float(d.drop[name][sel].mean())
            survive = sel & d.train
            entry["survive"] = float(survive.sum() / total)
            entry["survive_with_u"] = float((survive & np.isfinite(d.u)).sum() / total)
            entry["survive_not_dhw"] = float((survive & ~d.dhw).sum() / total)
            entry["surviving_complete_days_ge_50pct"] = int(
                sum(
                    1
                    for j, (_l, lo, hi) in enumerate(d.days)
                    if d.day_complete[j] and sel[lo] and survive[lo:hi].mean() >= 0.5
                )
            )
            out[f"{season}_{months}"] = entry
    return out


def min_hours_regimes(d: ClimateData) -> dict[str, Any]:
    """Runs of days with the same ``min_hours`` setting, per season (days of complete data)."""
    hours = d.columns["climate_min_hours_at_start"]
    out: dict[str, Any] = {}
    for season in SEASONS:
        rows = []
        run: dict[str, Any] | None = None
        for j, (label, lo, _hi) in enumerate(d.days):
            if d.day_season[j] != season or not d.day_complete[j]:
                continue
            value = hours[lo]
            value = None if not np.isfinite(value) else float(value)
            gate = bool(d.day_gate_month[j])
            replayed = d.u_reason[j] == ""
            heating = bool(np.nansum(np.where(d.dhw[lo:_hi], 0.0, d.q_heat_kwh[lo:_hi])) > 1.0)
            if run is None or run["min_hours"] != value:
                run = {
                    "min_hours": value,
                    "from": label.isoformat(),
                    "to": label.isoformat(),
                    "days": 0,
                    "gate_month_days": 0,
                    "replayed_days": 0,
                    "heating_days": 0,
                }
                rows.append(run)
            run["to"] = label.isoformat()
            run["days"] += 1
            run["gate_month_days"] += int(gate)
            run["replayed_days"] += int(replayed)
            run["heating_days"] += int(heating)
        out[season] = rows
    out["S3_distinct_values_in_gate_months"] = sorted(
        {
            float(hours[lo])
            for j, (_l, lo, _h) in enumerate(d.days)
            if d.day_season[j] == "S3" and d.day_gate_month[j] and np.isfinite(hours[lo])
        }
    )
    return out


def coverage_summary(d: ClimateData) -> dict[str, Any]:
    out: dict[str, Any] = {
        "slots": d.n,
        "first": _iso(int(d.starts_us[0])),
        "last": _iso(int(d.starts_us[-1])),
    }
    for season in SEASONS:
        sel = _season_slots(d, season, "fit")
        out[season] = {
            "slots": int(sel.sum()),
            "t_air_known": float(np.isfinite(d.t_air[sel]).mean()) if sel.any() else None,
            "q_heat_known": float(np.isfinite(d.q_heat_kwh[sel]).mean()) if sel.any() else None,
            "price_known": float(np.isfinite(d.price[sel]).mean()) if sel.any() else None,
            "dhw_share": float(d.dhw[sel].mean()) if sel.any() else None,
            "cooling_share": float(d.cooling[sel].mean()) if sel.any() else None,
        }
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_audit(d: ClimateData, raw: dict[str, RawSeries], coverage, cfg) -> dict[str, Any]:
    report: dict[str, Any] = {
        "identity": {
            "feature_set_id": d.feature_set_id,
            "dataset_id": d.dataset_id,
            "config_sha256": d.config_sha256,
            "site_sha256": d.site_sha256,
        },
        "coverage": coverage_summary(d),
        "gaps_per_signal": gaps_per_signal(d, coverage),
        "stale_runs_ge_24h": stale_runs(d),
        "resolution": {
            "thermostat_temperature": {
                r: resolution(raw[TEMP_KEY[r]]) for r in ROOMS if TEMP_KEY[r] in raw
            },
            "thermostat_setpoint": {
                r: resolution(raw[SETPOINT_KEY[r]]) for r in ROOMS if SETPOINT_KEY[r] in raw
            },
            "heat_pump_temperatures": {
                k: resolution(raw[e]) for k, e in RAW_TEMPERATURES.items() if e in raw
            },
        },
        "counter_quantization": {
            "raw": {k: counter_quantum(raw[e]) for k, e in RAW_COUNTERS.items() if e in raw},
            "slot_level": counter_slot_quantization(d),
            "daily_power_over_counter": daily_counter_agreement(
                d, cfg.signals.counter_check.min_daily_kwh
            ),
        },
        "return_temperature_slab_proxy": return_temperature_proxy(
            d, cfg.model.slab_proxy.min_flow_l_per_min
        ),
        "heating_curve": heating_curve(d, cfg.model.slab_proxy.min_flow_l_per_min),
        "thermostat_signal": thermostat_signal(d),
        "comfort_realized_by_legacy": comfort_realized(d, cfg.comfort.low_c),
        "legacy_mask_hit_rate": legacy_mask_hit_rate(d),
        "channel2_over_heishamon": channel2_ratio(d),
        "outdoor_sensor_sun": outdoor_sensor_sun(d),
        "filter_survival": filter_survival(d),
        "min_hours_regimes": min_hours_regimes(d),
    }
    return report


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.3g}"
    return str(x)


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Climate data audit (private: numbers from the real installation)", ""]
    ident = report["identity"]
    lines += [
        f"- feature set `{ident['feature_set_id']}`",
        f"- dataset `{ident['dataset_id']}`",
        "",
    ]

    def dump(title: str, obj: Any) -> None:
        lines.append(f"## {title}")
        lines.append("```json")
        lines.append(json.dumps(obj, indent=1, sort_keys=False, default=str))
        lines.append("```")
        lines.append("")

    for key, value in report.items():
        if key != "identity":
            dump(key.replace("_", " "), value)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--feature-set", required=True, help="climate feature set id (or prefix)")
    parser.add_argument("--dataset", required=True, help="canonical dataset directory (prices)")
    parser.add_argument("--config", default="config/climate.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--name", default="", help="suffix of the report directory")
    args = parser.parse_args(argv)

    root = resolve_data_dir(args.data_dir)
    cfg = load_climate_config(args.config)
    data = load_climate_data(
        cfg, feature_set=args.feature_set, dataset=args.dataset, data_dir=str(root)
    )
    base = root / "analysis" / "v1" / FEATURES_DIRNAME
    feature_dir = next(p for p in base.glob(f"{data.feature_set_id}*") if p.is_dir())
    table, manifest = load_features(feature_dir)

    coverage = {
        c.removesuffix("_coverage"): table.column(c).to_numpy(zero_copy_only=False).astype(float)
        for c in table.column_names
        if c.endswith("_coverage")
    }
    wanted: list[Signal] = [
        s
        for s in CLIMATE_SIGNALS
        if s.entity
        in set(THERMOSTATS.values()) | set(RAW_TEMPERATURES.values()) | set(RAW_COUNTERS.values())
    ]
    window = manifest["window"]
    try:
        series, _parts, _ids = load_signal_samples(
            root,
            tuple(wanted),
            window_start=datetime.fromisoformat(window["start"].replace("Z", "+00:00")),
            window_end=datetime.fromisoformat(window["end"].replace("Z", "+00:00")),
        )
    except FeatureError as exc:
        print(
            f"climate_audit: raw samples unavailable ({exc}); raw sections skipped", file=sys.stderr
        )
        series = {}

    report = run_audit(data, series, coverage, cfg)
    name = f"{data.feature_set_id[:12]}-{cfg.sha256[:12]}" + (f"-{args.name}" if args.name else "")
    out_dir = root / "analysis" / "v1" / "climate-audit" / name
    write_part_atomic(out_dir / "report.json", json_bytes(report))
    write_part_atomic(out_dir / "report.md", render_markdown(report).encode("utf-8"))
    print(f"climate_audit: report written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
