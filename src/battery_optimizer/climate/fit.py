"""``climate-fit``: fit the thermal models and their heat response per fold, validate, decide.

For every fold of ``config/climate.yaml`` (fit on the training seasons, test on the next one):

1. Room offsets (``room_offsets``), the thermal fits (A and B, Q-driven), their observers, the heat
   responses (u-driven) and the starts model use the fold's training slots only: complete local
   days of the training seasons in the fit months.  Nothing of the test season enters a fit.
2. Validation on the test season's complete days in the gate months: forecasts from 00, 06, 12
   and 18 local with the observer state, 24 h open loop on the measured outdoor temperature, in
   two modes: Q-driven (measured heat) and u-driven (replayed ``u`` through the response).  RMSE
   and bias at 1, 6, 12 and 24 h against persistence and diurnal persistence, scored on the slots
   the training filters keep.  u-driven daily thermal energy (MAPE, season and ``min_hours``
   regime bias) from the 00 origin of each day, and the legacy replay's degree-hours below the
   comfort floor over episodes of consecutive simulable days.
3. Acceptance (fold whose test season is S3) per criterion, model selection A vs B and the
   verdict: ACCEPTED, or UNDETERMINED when no model passes.  The legacy-replay electricity and
   cost criteria need the COP model (C2 PR 2) and the simulator (PR 4): they are ``pending``.

A forecast stops at the first slot whose inputs are unknown (for example the outdoor temperature
during a compressor run longer than the de-biasing cap); later horizons of that origin are not
scored.  Output: ``<data_dir>/analysis/v1/climate-models/<model_id>/`` with ``model.json``,
``validation.json`` and ``validation.md`` (private figures: the data dir only).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from battery_optimizer.climate import plant, thermal
from battery_optimizer.climate.config import ClimateConfig, load_climate_config
from battery_optimizer.climate.data import (
    SITE_FILE,
    ClimateData,
    Site,
    load_climate_data,
    load_site,
    room_offsets,
    steady_circulation,
)
from battery_optimizer.features import json_bytes
from battery_optimizer.manifest import code_version
from battery_optimizer.paths import resolve_data_dir, write_part_atomic

__all__ = [
    "MODELS_DIRNAME",
    "FoldResult",
    "compute_model_id",
    "fit_and_validate",
    "fit_fold",
    "run_climate_fit",
    "validate",
    "write_model",
]

log = logging.getLogger(__name__)

MODELS_DIRNAME = "climate-models"
MODEL_SCHEMA_VERSION = 1
#: The observer must have run this long on known inputs before an origin (B's slab converges).
WARMUP_SLOTS = 96
HORIZON_SLOTS = 96
SLOT_H = thermal.STEP_H
#: Parameters compared between the folds (``b_ret`` is a measurement offset, not physics).
STABILITY_EXCLUDED = frozenset({"b_ret"})


# ---------------------------------------------------------------------------
# Masks and inputs
# ---------------------------------------------------------------------------


def train_slots(d: ClimateData, seasons) -> np.ndarray:
    flag = np.isin(d.day_season, list(seasons)) & d.day_complete & d.day_fit_month
    return d.day_mask(flag)


def test_days(d: ClimateData, season: str) -> np.ndarray:
    return (d.day_season == season) & d.day_complete & d.day_gate_month


def heating_days(d: ClimateData, min_kwh: float) -> np.ndarray:
    """Per local day: heat (outside DHW slots) above *min_kwh*, as the audit counts it."""
    q = np.where(d.dhw, 0.0, d.q_heat_kwh)
    return np.array([np.nansum(q[lo:hi]) > min_kwh for _l, lo, hi in d.days], dtype=bool)


def q_inputs(d: ClimateData) -> np.ndarray:
    w = thermal.input_matrix(d.q_heat_kw, d.t_out_debiased, d.solar)
    w[d.cooling] = np.nan  # a cooling slot is not a heating input
    return w


def comfort_matrix(d: ClimateData, field: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.vstack([field[r] for r in d.comfort_rooms])


def plant_inputs(d: ClimateData) -> plant.PlantInputs:
    return plant.PlantInputs(
        u=d.u,
        dhw=d.dhw,
        t_out=d.t_out_debiased,
        solar=d.solar,
        t_flow_target=d.columns["flow_target_temp_mean_c"],
        first_hour=plant.first_hour_of_block(d.u),
        local_hour=d.local_hour,
        setpoints=comfort_matrix(d, d.room_setpoint),
    )


def slab_measurement(d: ClimateData, cfg: ClimateConfig) -> np.ndarray:
    """Return temperature in steady circulation (flow, minutes into the run, no DHW/cooling)."""
    sp = cfg.model.slab_proxy
    c = d.columns
    with np.errstate(invalid="ignore"):
        circulating = (c["pump_flow_mean"] >= sp.min_flow_l_per_min) & ~d.dhw & ~d.cooling
    steady = steady_circulation(circulating, sp.min_minutes_into_run)
    return np.where(steady, c["return_temp_mean_c"], np.nan)


def fit_inputs(d: ClimateData, cfg: ClimateConfig, train: np.ndarray) -> thermal.FitInputs:
    """Q-driven fit inputs restricted to *train*: 48 h windows from local midnight (pairs of
    local days), scored on the slots no training filter drops."""
    w = q_inputs(d)
    usable = np.isfinite(w).all(axis=1) & train
    days_per_window = cfg.model.window_hours // 24
    window_id = np.where(train, d.day_of_slot // days_per_window, -1)
    y_init = np.where(train, d.t_air, np.nan)
    y_air = np.where(train & d.train, d.t_air, np.nan)
    y_ret = np.where(train & d.train, slab_measurement(d, cfg), np.nan)
    segments = thermal.segments_in_windows(window_id, usable, y_init)
    return thermal.FitInputs(w=w, y_air=y_air, y_init=y_init, segments=segments, y_ret=y_ret)


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------


class FittedModel:
    """A fitted thermal model with its observer and heat response."""

    def __init__(self, kind: str, fit: thermal.ThermalFit, gain, process_var, response, stats):
        self.kind = kind
        self.fit = fit
        self.ad, self.bd = thermal.discretize(*thermal.continuous(kind, fit.params))
        self.gain = gain
        self.process_var = process_var
        self.response = response
        self.response_stats = stats
        self.states: np.ndarray | None = None
        self.age: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "thermal": self.fit.to_dict(),
            "observer": {
                "gain": [float(g) for g in self.gain],
                "process_var_k2": self.process_var,
                "measurement_sigma_k": thermal.MEASUREMENT_SIGMA_K,
                "warmup_slots": WARMUP_SLOTS,
            },
            "response": self.response.to_dict(),
            "response_fit": self.response_stats,
        }


class FoldResult:
    def __init__(self, index: int, train: tuple[str, ...], test: str):
        self.index, self.train, self.test = index, train, test
        self.offsets: np.ndarray | None = None
        self.cap: plant.ValveCap | None = None
        self.models: dict[str, FittedModel] = {}
        self.starts: plant.StartsModel | None = None
        self.starts_fit: dict[str, Any] = {}
        self.validation: dict[str, Any] = {}


def _response_selection(d: ClimateData, train: np.ndarray, cap_meas: np.ndarray) -> np.ndarray:
    """Permitted, non-DHW training slots with known heat, outdoor temperature and valve state
    (the valve-shut and window filters stay: the cap models the valves)."""
    drop = d.drop
    return (
        train
        & d.u_known
        & (d.u == 1.0)
        & ~d.dhw
        & ~d.cooling
        & np.isfinite(d.q_heat_kw)
        & np.isfinite(d.t_out_debiased)
        & np.isfinite(cap_meas)
        & ~drop["thermostat_outage"]
        & ~drop["unknown_quality"]
    )


def _day_features(d: ClimateData, j: int) -> tuple[int, float, float, float] | None:
    """``(blocks, hours_on, mean T_out, starts)`` of local day *j*, or ``None``."""
    _l, lo, hi = d.days[j]
    starts = d.columns["compressor_starts_delta"][lo:hi]
    q = d.q_heat_kwh[lo:hi]
    t_out = d.t_out_debiased[lo:hi]
    if not (np.isfinite(starts).all() and np.isfinite(q).all()) or np.isfinite(t_out).mean() < 0.5:
        return None
    hours_on = float(((q >= plant.HEAT_ON_KWH) & ~d.dhw[lo:hi]).sum() * SLOT_H)
    return (
        plant.permission_blocks(d.u[lo:hi]),
        hours_on,
        float(np.nanmean(t_out)),
        float(starts.sum()),
    )


def _starts_rows(d: ClimateData, day_flags: np.ndarray) -> np.ndarray:
    rows = []
    for j in np.flatnonzero(day_flags & d.day_u_known & d.day_complete):
        row = _day_features(d, int(j))
        if row is not None:
            rows.append(row)
    return np.array(rows, dtype=float).reshape(-1, 4)


def fit_fold(d: ClimateData, cfg: ClimateConfig, index: int, phi: float) -> FoldResult:
    """Fit offsets, thermal models, observers, responses and starts on fold *index*."""
    fold = cfg.folds[index]
    res = FoldResult(index, tuple(fold.train), fold.test)
    train = train_slots(d, fold.train)
    res.offsets = room_offsets(d.room_temp, d.comfort_rooms, d.t_air, d.local_hour, train & d.train)
    res.cap = plant.ValveCap(phi, cfg.model.response.plant_valve_cap_margin_k, res.offsets)
    cap_meas = res.cap.measured(comfort_matrix(d, d.room_temp), comfort_matrix(d, d.room_setpoint))
    fi = fit_inputs(d, cfg, train)
    w = fi.w
    y_obs = d.t_air
    meas_var = thermal.MEASUREMENT_SIGMA_K**2
    sel = _response_selection(d, train, cap_meas)
    first = plant.first_hour_of_block(d.u)
    t_ft = d.columns["flow_target_temp_mean_c"]
    t_ref = cfg.comfort.low_c
    for kind in thermal.KINDS:
        fit = thermal.fit_thermal(kind, fi, cfg.model.envelope)
        ad, bd = thermal.discretize(*thermal.continuous(kind, fit.params))
        proc = thermal.process_variance(
            ad, bd, np.where(train[:, None], w, np.nan), fi.y_air, meas_var
        )
        gain = thermal.observer_gain(ad, proc, meas_var)
        slab = None
        if kind == "B":
            p = fit.params
            slab = (float(np.exp(-p["k"] * thermal.STEP_H / p["c_s"])), p["k"])
        states, age = thermal.observe(ad, bd, w, y_obs, gain, fit.slab_offset_k, slab)
        if kind == "A":
            s = sel
            response, stats = plant.fit_response_a(
                d.q_heat_kw[s], cap_meas[s], d.t_out_debiased[s], first[s], t_ref
            )
        else:
            s = sel & np.isfinite(t_ft) & np.isfinite(states[:, 1])
            response, stats = plant.fit_response_b(
                d.q_heat_kw[s], cap_meas[s], d.t_out_debiased[s], t_ft[s], states[s, 1], t_ref
            )
        model = FittedModel(kind, fit, gain, proc, response, stats)
        model.states, model.age = states, age
        res.models[kind] = model
    rows = _starts_rows(d, np.isin(d.day_season, list(fold.train)) & d.day_fit_month)
    if rows.shape[0] >= 3:
        res.starts = plant.fit_starts(rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3])
        pred = res.starts.predict(rows[:, 0], rows[:, 1], rows[:, 2])
        res.starts_fit = {"days": int(rows.shape[0]), **_error_stats(pred, rows[:, 3])}
    return res


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _error_stats(pred: np.ndarray, obs: np.ndarray) -> dict[str, Any]:
    if pred.size == 0:
        return {
            "n": 0,
            "rmse": None,
            "mae": None,
            "bias": None,
            "mean_obs": None,
            "mean_pred": None,
        }
    e = pred - obs
    return {
        "n": int(pred.size),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "mean_obs": float(np.mean(obs)),
        "mean_pred": float(np.mean(pred)),
    }


def _origins(d: ClimateData, cfg: ClimateConfig, days: np.ndarray, models) -> np.ndarray:
    hours = set(cfg.model.forecast_origins_local_hours)
    slots = d.day_mask(days)
    first_of_hour = np.ones(d.n, dtype=bool)
    first_of_hour[1:] = d.local_hour[1:] != d.local_hour[:-1]
    ok = slots & first_of_hour & np.isin(d.local_hour, list(hours))
    for m in models:
        ok &= m.age >= WARMUP_SLOTS
    return np.flatnonzero(ok)


def _run_length(known: np.ndarray, start: int, limit: int) -> int:
    seg = known[start : start + limit]
    bad = np.flatnonzero(~seg)
    return int(bad[0]) if bad.size else int(seg.size)


def forecasts(
    d: ClimateData,
    model: FittedModel,
    origins: np.ndarray,
    mode: str,
    cap: plant.ValveCap,
    inp: plant.PlantInputs,
    known_u: np.ndarray,
) -> dict[int, list[tuple[int, float]]]:
    """``{origin: [(steps ahead j, predicted air temperature of slot origin + j)]}``."""
    w = q_inputs(d)
    known_q = np.isfinite(w).all(axis=1)
    drive = np.where(known_q[:, None], w, 0.0) @ model.bd.T
    out: dict[int, list[tuple[int, float]]] = {}
    for o in origins:
        x0 = model.states[o]
        if mode == "q":
            length = _run_length(known_q, o, HORIZON_SLOTS)
            xs = thermal.simulate(model.ad, drive[o : o + length], x0)
        else:
            length = _run_length(known_u, o, HORIZON_SLOTS)
            xs, _q = plant.simulate_u(
                model.kind, model.ad, model.bd, x0, o, o + length, inp, model.response, cap
            )
        out[int(o)] = [(j, float(xs[j + 1, 0])) for j in range(length)]
    return out


def horizon_metrics(
    d: ClimateData,
    cfg: ClimateConfig,
    preds: Mapping[str, dict[int, list[tuple[int, float]]]],
    scored: np.ndarray,
) -> dict[str, Any]:
    """RMSE and bias per horizon on the (origin, horizon) pairs every model and both baselines
    have, so all numbers of a horizon compare the same pairs."""
    y = d.t_air
    out: dict[str, Any] = {}
    kinds = list(preds)
    for h in cfg.model.forecast_horizons_hours:
        j = 4 * h - 1
        rows: dict[str, list[float]] = {k: [] for k in [*kinds, "persistence", "diurnal", "obs"]}
        for o in preds[kinds[0]]:
            t = o + j
            if t >= d.n or not scored[t] or not np.isfinite(y[t]):
                continue
            pers, diur = y[o - 1], y[t - 96] if t >= 96 else np.nan
            if not (np.isfinite(pers) and np.isfinite(diur)):
                continue
            vals = {}
            for k in kinds:
                p = preds[k].get(o, [])
                if len(p) <= j:
                    break
                vals[k] = p[j][1]
            else:
                for k in kinds:
                    rows[k].append(vals[k])
                rows["persistence"].append(pers)
                rows["diurnal"].append(diur)
                rows["obs"].append(y[t])
        obs = np.array(rows["obs"])
        entry = {
            k: _error_stats(np.array(rows[k]), obs) for k in [*kinds, "persistence", "diurnal"]
        }
        best = min(
            (entry[b]["rmse"] for b in ("persistence", "diurnal") if entry[b]["rmse"] is not None),
            default=None,
        )
        for k in kinds:
            r = entry[k]["rmse"]
            entry[k]["gain_vs_better_baseline"] = (
                None if r is None or not best else float(1.0 - r / best)
            )
        out[str(h)] = entry
    return out


def _simulable_days(d, days, inp_known, kind_known_q=None) -> np.ndarray:
    ok = days & d.day_u_known & d.day_complete
    out = np.zeros(len(d.days), dtype=bool)
    for j in np.flatnonzero(ok):
        _l, lo, hi = d.days[j]
        good = inp_known[lo:hi].all()
        if kind_known_q is not None:
            good &= kind_known_q[lo:hi].all()
        out[j] = good
    return out


def daily_energy(
    d: ClimateData,
    cfg: ClimateConfig,
    models: Mapping[str, FittedModel],
    days: np.ndarray,
    cap: plant.ValveCap,
    inp: plant.PlantInputs,
    known_u: np.ndarray,
) -> dict[str, Any]:
    """u-driven daily thermal energy from each day's 00 origin, and comfort per regime."""
    min_kwh = cfg.signals.counter_check.min_daily_kwh
    acc = cfg.acceptance.u_driven
    low = cfg.comfort.low_c
    sim_days = _simulable_days(d, days, known_u, np.isfinite(d.q_heat_kwh))
    heat_day = heating_days(d, min_kwh)
    min_hours = d.columns["climate_min_hours_at_start"]
    regime_of = np.array([min_hours[lo] for _l, lo, _h in d.days])
    rows = []
    for j in np.flatnonzero(sim_days):
        _l, lo, hi = d.days[j]
        if any(m.age[lo] < WARMUP_SLOTS for m in models.values()):
            continue
        row = {"day": j, "meas_kwh": float(np.sum(d.q_heat_kwh[lo:hi])), "regime": regime_of[j]}
        row["meas_below"] = float(np.mean(d.t_air[lo:hi][np.isfinite(d.t_air[lo:hi])] < low))
        for k, m in models.items():
            xs, q = plant.simulate_u(k, m.ad, m.bd, m.states[lo], lo, hi, inp, m.response, cap)
            row[k] = float(q.sum() * SLOT_H)
            row[f"{k}_below"] = float(np.mean(xs[1:, 0] < low))
        rows.append(row)
    out: dict[str, Any] = {"days": len(rows), "min_daily_kwh": min_kwh}
    meas = np.array([r["meas_kwh"] for r in rows])
    regimes = np.array([r["regime"] for r in rows], dtype=float)
    for k in models:
        sim = np.array([r[k] for r in rows])
        heat = meas >= min_kwh
        ape = np.abs(sim[heat] - meas[heat]) / meas[heat]
        entry: dict[str, Any] = {
            "heating_days": int(heat.sum()),
            "mape": float(np.mean(ape)) if ape.size else None,
            "season_bias": float(sim.sum() / meas.sum() - 1.0) if meas.sum() > 0 else None,
            "sim_kwh": float(sim.sum()),
            "meas_kwh": float(meas.sum()),
            "regimes": {},
        }
        for value in sorted({v for v in regime_of[days] if np.isfinite(v)}):
            in_regime = days & (regime_of == value)
            n_heat = int((in_regime & heat_day).sum())
            sel = regimes == value
            ms, ss = meas[sel], sim[sel]
            below_m = np.array([r["meas_below"] for r, s in zip(rows, sel, strict=True) if s])
            below_s = np.array([r[f"{k}_below"] for r, s in zip(rows, sel, strict=True) if s])
            entry["regimes"][f"{value:g}"] = {
                "heating_days_in_season": n_heat,
                "tested": n_heat >= acc.regime_min_heating_days,
                "simulated_days": int(sel.sum()),
                "bias": float(ss.sum() / ms.sum() - 1.0) if ms.sum() > 0 else None,
                "meas_kwh_per_day": float(ms.mean()) if ms.size else None,
                "sim_kwh_per_day": float(ss.mean()) if ss.size else None,
                "share_t_air_below_low_meas": float(below_m.mean()) if below_m.size else None,
                "share_t_air_below_low_sim": float(below_s.mean()) if below_s.size else None,
            }
        out[k] = entry
    return out


def legacy_replay_kh(
    d: ClimateData,
    cfg: ClimateConfig,
    models: Mapping[str, FittedModel],
    days: np.ndarray,
    cap: plant.ValveCap,
    inp: plant.PlantInputs,
    known_u: np.ndarray,
) -> dict[str, Any]:
    """Degree-hours below the comfort floor per comfort room: the legacy rule replayed through
    each model (episodes of consecutive simulable days, started from the observer) against the
    measured room temperatures on the same slots."""
    low = cfg.comfort.low_c
    sim_days = _simulable_days(d, days, known_u)
    episodes: list[tuple[int, int]] = []
    n_days = 0
    j = 0
    while j < len(d.days):
        if not sim_days[j]:
            j += 1
            continue
        k = j
        while k + 1 < len(d.days) and sim_days[k + 1]:
            k += 1
        lo, hi = d.days[j][1], d.days[k][2]
        if all(m.age[lo] >= WARMUP_SLOTS for m in models.values()):
            episodes.append((lo, hi))
            n_days += k - j + 1
        j = k + 1
    rooms = comfort_matrix(d, d.room_temp)
    out: dict[str, Any] = {"episodes": len(episodes), "days": n_days}
    meas_kh = np.zeros(len(d.comfort_rooms))
    sim_kh = {k: np.zeros(len(d.comfort_rooms)) for k in models}
    for lo, hi in episodes:
        hours = d.local_hour[lo:hi]
        known = np.isfinite(rooms[:, lo:hi])
        meas_kh += np.where(known, np.maximum(0.0, low - rooms[:, lo:hi]), 0.0).sum(1) * SLOT_H
        for k, m in models.items():
            xs, _q = plant.simulate_u(k, m.ad, m.bd, m.states[lo], lo, hi, inp, m.response, cap)
            room_sim = xs[1:, 0][None, :] + np.nan_to_num(cap.offsets[:, hours])
            sim_kh[k] += np.where(known, np.maximum(0.0, low - room_sim), 0.0).sum(1) * SLOT_H
    out["meas_kh"] = {r: float(v) for r, v in zip(d.comfort_rooms, meas_kh, strict=True)}
    out["meas_kh_total"] = float(meas_kh.sum())
    for k in models:
        total = float(sim_kh[k].sum())
        out[k] = {
            "sim_kh": {r: float(v) for r, v in zip(d.comfort_rooms, sim_kh[k], strict=True)},
            "sim_kh_total": total,
            "abs_error_kh": abs(total - out["meas_kh_total"]),
            "rel_error": (
                abs(total - out["meas_kh_total"]) / out["meas_kh_total"]
                if out["meas_kh_total"] > 0
                else None
            ),
        }
    return out


def starts_validation(d: ClimateData, starts: plant.StartsModel | None, days) -> dict[str, Any]:
    rows = _starts_rows(d, days)
    if starts is None or rows.shape[0] == 0:
        return {"days": int(rows.shape[0])}
    pred = starts.predict(rows[:, 0], rows[:, 1], rows[:, 2])
    stats = _error_stats(pred, rows[:, 3])
    stats["ratio_mean_pred_to_obs"] = (
        float(pred.mean() / rows[:, 3].mean()) if rows[:, 3].mean() > 0 else None
    )
    return {"days": int(rows.shape[0]), **stats}


def validate(d: ClimateData, cfg: ClimateConfig, fold: FoldResult) -> dict[str, Any]:
    """Validation of a fitted fold on its test season (gate months)."""
    days = test_days(d, fold.test)
    models = fold.models
    origins = _origins(d, cfg, days, models.values())
    inp = plant_inputs(d)
    known_u = inp.known("B") & np.isfinite(q_inputs(d)[:, 1:]).all(axis=1)
    cap = fold.cap
    scored = d.train & d.day_mask(days)
    q_preds = {k: forecasts(d, m, origins, "q", cap, inp, known_u) for k, m in models.items()}
    u_origins = origins[d.u_known[origins]]
    u_preds = {k: forecasts(d, m, u_origins, "u", cap, inp, known_u) for k, m in models.items()}
    return {
        "test_season": fold.test,
        "test_days": int(days.sum()),
        "origins": int(origins.size),
        "u_origins": int(u_origins.size),
        "q_driven": horizon_metrics(d, cfg, q_preds, scored),
        "u_driven": horizon_metrics(d, cfg, u_preds, scored),
        "daily_energy": daily_energy(d, cfg, models, days, cap, inp, known_u),
        "legacy_replay_kh": legacy_replay_kh(d, cfg, models, days, cap, inp, known_u),
        "starts": starts_validation(d, fold.starts, days),
    }


# ---------------------------------------------------------------------------
# Acceptance and selection
# ---------------------------------------------------------------------------


def _criterion(cid: str, value, threshold, passed, note: str = "") -> dict[str, Any]:
    status = "pending" if passed is None else ("pass" if passed else "fail")
    return {"id": cid, "value": value, "threshold": threshold, "status": status, "note": note}


def _le(value, limit) -> bool:
    return value is not None and np.isfinite(value) and value <= limit


def stability(folds: list[FoldResult], kind: str, limit: float) -> dict[str, Any]:
    a, b = folds[0].models[kind].fit.params, folds[-1].models[kind].fit.params
    out = {}
    for name in a:
        if name in STABILITY_EXCLUDED or name not in b:
            continue
        lo = min(abs(a[name]), abs(b[name]))
        rel = abs(a[name] - b[name]) / lo if lo > 0 else float("inf")
        out[name] = {"first": a[name], "last": b[name], "rel_diff": rel, "pass": rel <= limit}
    return out


def acceptance(cfg: ClimateConfig, folds: list[FoldResult], kind: str) -> dict[str, Any]:
    """Every criterion of ``acceptance`` for model *kind* on the fold that tests S3."""
    acc = cfg.acceptance
    fold = next(f for f in folds if f.test == "S3")
    v = fold.validation
    crit = []
    qd = v["q_driven"]
    for h, limit in sorted(acc.q_driven.rmse_k.items()):
        r = qd[str(h)][kind]["rmse"]
        crit.append(_criterion(f"q_rmse_{h}h", r, limit, _le(r, limit)))
    bias = qd["24"][kind]["bias"]
    crit.append(
        _criterion(
            "q_abs_bias_24h",
            None if bias is None else abs(bias),
            acc.q_driven.abs_bias_24h_k,
            _le(None if bias is None else abs(bias), acc.q_driven.abs_bias_24h_k),
        )
    )
    for h in acc.q_driven.gain_horizons_hours:
        g = qd[str(h)][kind]["gain_vs_better_baseline"]
        limit = acc.q_driven.min_gain_vs_baseline
        crit.append(_criterion(f"q_gain_vs_baseline_{h}h", g, limit, g is not None and g >= limit))
    ur = v["u_driven"]["24"][kind]["rmse"]
    crit.append(
        _criterion("u_rmse_24h", ur, acc.u_driven.rmse_24h_k, _le(ur, acc.u_driven.rmse_24h_k))
    )
    en = v["daily_energy"][kind]
    crit.append(
        _criterion(
            "u_daily_energy_mape",
            en["mape"],
            acc.u_driven.daily_energy_mape,
            _le(en["mape"], acc.u_driven.daily_energy_mape),
        )
    )
    sb = None if en["season_bias"] is None else abs(en["season_bias"])
    crit.append(
        _criterion(
            "u_season_energy_bias",
            sb,
            acc.u_driven.season_energy_bias,
            _le(sb, acc.u_driven.season_energy_bias),
        )
    )
    for name, reg in en["regimes"].items():
        if not reg["tested"]:
            continue
        rb = None if reg["bias"] is None else abs(reg["bias"])
        crit.append(
            _criterion(
                f"u_regime_energy_bias_min_hours_{name}",
                rb,
                acc.u_driven.regime_energy_bias,
                _le(rb, acc.u_driven.regime_energy_bias),
            )
        )
    lr = acc.legacy_replay
    crit.append(
        _criterion(
            "legacy_replay_electricity",
            None,
            lr.electricity_rel_error,
            None,
            "pending COP (C2 PR 2) and the simulator (PR 4)",
        )
    )
    crit.append(
        _criterion(
            "legacy_replay_cost",
            None,
            lr.cost_rel_error,
            None,
            "pending COP (C2 PR 2) and the simulator (PR 4)",
        )
    )
    kh = v["legacy_replay_kh"][kind]
    kh_pass = (kh["rel_error"] is not None and kh["rel_error"] <= lr.kh_below_low_rel_error) or (
        kh["abs_error_kh"] <= lr.kh_below_low_abs_error
    )
    crit.append(
        _criterion(
            "legacy_replay_kh_below_low",
            {"rel_error": kh["rel_error"], "abs_error_kh": kh["abs_error_kh"]},
            {"rel": lr.kh_below_low_rel_error, "abs_kh": lr.kh_below_low_abs_error},
            kh_pass,
            "summed over the comfort rooms; passes within the relative or the absolute bound",
        )
    )
    failures = {f"fold{f.index + 1}": f.models[kind].fit.failures() for f in folds}
    crit.append(
        _criterion(
            "parameters_inside_envelope",
            failures,
            "no parameter on a bound or outside the envelope; identifiable",
            all(not x for x in failures.values()),
        )
    )
    stab = stability(folds, kind, acc.parameters.fold_rel_diff)
    crit.append(
        _criterion(
            "parameters_fold_stability",
            {n: s["rel_diff"] for n, s in stab.items()},
            acc.parameters.fold_rel_diff,
            all(s["pass"] for s in stab.values()) if len(folds) > 1 else None,
            "|a - b| / min(|a|, |b|) between the first and the last fold",
        )
    )
    failed = [c["id"] for c in crit if c["status"] == "fail"]
    pending = [c["id"] for c in crit if c["status"] == "pending"]
    return {
        "criteria": crit,
        "failed": failed,
        "pending": pending,
        "accepted": not failed,
    }


def select(cfg: ClimateConfig, folds: list[FoldResult], verdicts: Mapping[str, Any]):
    """A unless B beats it by ``b_min_rmse_gain`` (Q-driven, S3) at the selection horizons."""
    sel = cfg.model.selection
    fold = next(f for f in folds if f.test == "S3")
    qd = fold.validation["q_driven"]
    gains = {}
    for h in sel.horizons_hours:
        ra, rb = qd[str(h)]["A"]["rmse"], qd[str(h)]["B"]["rmse"]
        gains[str(h)] = None if not ra or rb is None else float(1.0 - rb / ra)
    b_better = all(g is not None and g >= sel.b_min_rmse_gain for g in gains.values())
    accepted = [k for k in ("A", "B") if verdicts[k]["accepted"]]
    if not accepted:
        choice = None
    elif accepted == ["A", "B"]:
        choice = "B" if b_better else "A"
    else:
        choice = accepted[0]
    return {
        "b_rmse_gain_vs_a": gains,
        "b_beats_a": b_better,
        "accepted_models": accepted,
        "selected": choice,
        "verdict": "ACCEPTED" if choice else "UNDETERMINED",
        "pending": verdicts[choice]["pending"] if choice else [],
    }


# ---------------------------------------------------------------------------
# Identity and output
# ---------------------------------------------------------------------------


def fold_windows(cfg: ClimateConfig) -> list[dict[str, Any]]:
    def span(name: str) -> list[str]:
        s = cfg.seasons[name]
        return [s.start.isoformat(), s.end.isoformat()]

    return [
        {"train": {s: span(s) for s in f.train}, "test": {f.test: span(f.test)}} for f in cfg.folds
    ]


def compute_model_id(
    *,
    feature_set_id: str,
    dataset_id: str | None,
    windows: list[dict[str, Any]],
    code_ver: str,
    config_sha256: str,
    site_sha256: str,
) -> str:
    payload = {
        "feature_set_id": feature_set_id,
        "dataset_id": dataset_id,
        "windows": windows,
        "code_version": code_ver,
        "config_sha256": config_sha256,
        "site_sha256": site_sha256,
        "schema_version": MODEL_SCHEMA_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def _fmt(x, nd=3) -> str:
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def render_markdown(model: dict[str, Any], validation: dict[str, Any]) -> str:
    lines = [
        f"# Climate model `{model['model_id'][:12]}`",
        "",
        f"- Feature set `{model['lineage']['feature_set_id']}`, dataset "
        f"`{model['lineage']['dataset_id']}`, code `{model['lineage']['code_version']}`",
        f"- Config sha256 `{model['lineage']['config_sha256'][:12]}`, site sha256 "
        f"`{model['lineage']['site_sha256'][:12]}`, phi {model['phi']}",
        "",
        f"**Verdict: {validation['selection']['verdict']}** (selected: "
        f"{validation['selection']['selected'] or 'none'}; pending: "
        f"{', '.join(validation['selection']['pending']) or 'none'})",
        "",
    ]
    for fold in model["folds"]:
        lines += [
            f"## Fold {fold['fold']}: train {'+'.join(fold['train'])}, test {fold['test']}",
            "",
        ]
        lines += [
            "| model | parameters | tau h | C kWh/K | balance °C | fit failures |",
            "|---|---|---|---|---|---|",
        ]
        for kind, m in fold["models"].items():
            th = m["thermal"]
            params = ", ".join(f"{k}={v:.3g}" for k, v in th["params"].items())
            der = th["derived"]
            lines.append(
                f"| {kind} | {params} | {der['tau_h']:.1f} | {der['c_kwh_per_k']:.2f} | "
                f"{der['balance_temp_c']:.1f} | {'; '.join(th['failures']) or 'none'} |"
            )
        lines.append("")
        for kind, m in fold["models"].items():
            resp = ", ".join(f"{k}={v:.3g}" for k, v in m["response"].items())
            lines.append(f"- Response {kind}: {resp} (fit {m['response_fit']})")
        lines += [f"- Starts model: {fold['starts']} (fit {fold['starts_fit']})", ""]
        v = validation["folds"][fold["fold"] - 1]
        for mode in ("q_driven", "u_driven"):
            lines += [
                f"### {mode} (test {v['test_season']}, gate months)",
                "",
                "| h | n | A rmse | A bias | B rmse | B bias | persistence | diurnal "
                "| A gain | B gain |",
                "|---|---|---|---|---|---|---|---|---|---|",
            ]
            for h, e in v[mode].items():
                lines.append(
                    f"| {h} | {e['A']['n']} | {_fmt(e['A']['rmse'])} | {_fmt(e['A']['bias'])} | "
                    f"{_fmt(e['B']['rmse'])} | {_fmt(e['B']['bias'])} | "
                    f"{_fmt(e['persistence']['rmse'])} | {_fmt(e['diurnal']['rmse'])} | "
                    f"{_fmt(e['A']['gain_vs_better_baseline'])} | "
                    f"{_fmt(e['B']['gain_vs_better_baseline'])} |"
                )
            lines.append("")
        de = v["daily_energy"]
        lines += [f"### u-driven daily energy ({de['days']} days)", ""]
        for kind in fold["models"]:
            e = de[kind]
            lines.append(f"- {kind}: MAPE {_fmt(e['mape'])}, season bias {_fmt(e['season_bias'])}")
            for name, r in e["regimes"].items():
                lines.append(
                    f"  - min_hours {name}: tested {r['tested']}, bias {_fmt(r['bias'])}, "
                    f"share T_air < low measured {_fmt(r['share_t_air_below_low_meas'])}, "
                    f"simulated {_fmt(r['share_t_air_below_low_sim'])} "
                    f"({r['simulated_days']} days)"
                )
        kh = v["legacy_replay_kh"]
        lines += ["", f"### Legacy replay, Kh below the floor ({kh['days']} days)", ""]
        for kind in fold["models"]:
            lines.append(
                f"- {kind}: rel error {_fmt(kh[kind]['rel_error'])}, abs error "
                f"{_fmt(kh[kind]['abs_error_kh'], 1)} Kh"
            )
        lines += ["", f"- Starts validation: {v['starts']}", ""]
    lines += ["## Acceptance (fold testing S3)", ""]
    for kind, verdict in validation["acceptance"].items():
        lines += [f"### Model {kind}: {'accepted' if verdict['accepted'] else 'not accepted'}", ""]
        lines += ["| criterion | value | threshold | status |", "|---|---|---|---|"]
        for c in verdict["criteria"]:
            lines.append(
                f"| {c['id']} | {_fmt(c['value'])} | {_fmt(c['threshold'])} | {c['status']} |"
            )
        lines.append("")
    sel = validation["selection"]
    lines += [
        "## Selection",
        "",
        f"- B RMSE gain over A: {sel['b_rmse_gain_vs_a']}; B beats A: {sel['b_beats_a']}",
        f"- Accepted: {sel['accepted_models']}; selected: {sel['selected']}; verdict "
        f"{sel['verdict']}",
        "",
    ]
    return "\n".join(lines)


def fit_and_validate(
    d: ClimateData, cfg: ClimateConfig, site: Site
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit and validate every fold; return the ``model.json`` and ``validation.json`` payloads."""
    phi = site.phi if site.phi is not None else cfg.model.response.phi_default
    folds = []
    for i in range(len(cfg.folds)):
        log.info("climate-fit: fold %d", i + 1)
        fold = fit_fold(d, cfg, i, phi)
        fold.validation = validate(d, cfg, fold)
        folds.append(fold)
    verdicts = {k: acceptance(cfg, folds, k) for k in thermal.KINDS}
    selection = select(cfg, folds, verdicts)
    windows = fold_windows(cfg)
    ver = code_version()
    model_id = compute_model_id(
        feature_set_id=d.feature_set_id,
        dataset_id=d.dataset_id,
        windows=windows,
        code_ver=ver,
        config_sha256=cfg.sha256,
        site_sha256=site.sha256,
    )
    model = {
        "model_id": model_id,
        "schema_version": MODEL_SCHEMA_VERSION,
        "lineage": {
            "feature_set_id": d.feature_set_id,
            "dataset_id": d.dataset_id,
            "code_version": ver,
            "config_sha256": cfg.sha256,
            "config_version": cfg.config_version,
            "site_sha256": site.sha256,
            "windows": windows,
        },
        "phi": phi,
        "comfort_rooms": list(d.comfort_rooms),
        "selected": selection["selected"],
        "verdict": selection["verdict"],
        "folds": [
            {
                "fold": f.index + 1,
                "train": list(f.train),
                "test": f.test,
                "room_offsets_k": [
                    [None if np.isnan(v) else float(v) for v in row] for row in f.offsets
                ],
                "models": {k: m.to_dict() for k, m in f.models.items()},
                "starts": f.starts.to_dict() if f.starts else None,
                "starts_fit": f.starts_fit,
            }
            for f in folds
        ],
    }
    validation = {
        "model_id": model_id,
        "folds": [f.validation for f in folds],
        "acceptance": verdicts,
        "selection": selection,
    }
    return model, validation


def write_model(root: Path, model: dict[str, Any], validation: dict[str, Any]) -> Path:
    """Write the three files under ``<root>/analysis/v1/climate-models/<model_id>/``.

    Content-addressed and no-clobber: the same id with the same bytes is a no-op, with different
    bytes an error (``FileExistsError``)."""
    out = root / "analysis" / "v1" / MODELS_DIRNAME / model["model_id"]
    write_part_atomic(out / "model.json", json_bytes(model))
    write_part_atomic(out / "validation.json", json_bytes(validation))
    write_part_atomic(out / "validation.md", render_markdown(model, validation).encode())
    return out


def run_climate_fit(
    *,
    feature_set: str,
    dataset: str | Path,
    config_path: str | Path = "config/climate.yaml",
    data_dir: str | None = None,
) -> Path:
    """Fit, validate and write one model directory; return it."""
    root = resolve_data_dir(data_dir)  # refuses a data dir inside a git tree
    cfg = load_climate_config(config_path)
    site = load_site(root / SITE_FILE)
    d = load_climate_data(
        cfg, feature_set=feature_set, dataset=dataset, data_dir=str(root), site=site
    )
    model, validation = fit_and_validate(d, cfg, site)
    out = write_model(root, model, validation)
    log.info("wrote climate model %s", out)
    return out
