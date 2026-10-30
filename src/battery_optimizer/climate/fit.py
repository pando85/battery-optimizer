"""``climate-fit``: fit the thermal models and their heat response, refit them weekly, validate,
decide.

For the fold of ``config/climate.yaml`` (fit on the training season S2, test on S3; S1 is a
different plant regime and is excluded from training, ``seasons.training_excluded``):

1. Room offsets (``room_offsets``), the *prior* thermal fits (A and B, Q-driven), the heat
   responses (u-driven) and the starts model use the fold's training slots only: complete local
   days of the training season in the fit months.
2. Causal weekly refits (``model.adaptive``): from the first gate-month day of the test season,
   every ``refit_every_days`` local days the thermal parameters are refitted on the fit-month days
   of the ``window_days`` before that origin, shrunk toward the prior (``thermal.fit_thermal``).
   Each week's observer runs from the start of its window, so every state, parameter and forecast
   at an origin uses only data before it.  Response, valve cap, offsets and starts stay the prior's.
3. Validation on the test season's complete days in the gate months: forecasts from 00, 06, 12
   and 18 local with the week's observer state and parameters, 24 h open loop on the measured
   outdoor temperature (``ClimateData.t_out_model``), in two modes: Q-driven (measured heat) and
   u-driven (replayed ``u`` through the response).  RMSE and bias at 1, 6, 12 and 24 h against
   persistence and diurnal persistence, scored on the slots the training filters keep.  u-driven
   daily thermal energy (MAPE, season and ``min_hours`` regime bias) from the 00 origin of each
   day; the legacy replay over episodes of consecutive simulable days (parameters switch at the
   week boundaries): degree-hours below the comfort floor, and heating electricity and its cost
   through the COP model (``cop.model``, trained on the training season) against the realized.
4. Acceptance per criterion (COP limits included), model selection A vs B and the verdict:
   ACCEPTED, or UNDETERMINED when no model passes.

The thermal fit scores every slot whose air temperature is a valid measurement
(:func:`valid_air`): the valves-shut and rule-disabled filters describe the plant, not bad data, and
their heat still drives the simulation, so leaving them unscored biased the fit.  The validation
scoring keeps the predeclared training-filter mask.

A forecast stops at the first slot whose inputs are unknown; later horizons of that origin are not
scored.  :func:`select_shrinkage` is the S2-only choice of the shrinkage strengths
(``hack/climate_shrinkage.py``).  Output: ``<data_dir>/analysis/v1/climate-models/<model_id>/`` with
``model.json``, ``validation.json`` and ``validation.md`` (private figures: the data dir only).
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from battery_optimizer.climate import cop, plant, thermal
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
    "Adaptive",
    "ClimateFitError",
    "FoldResult",
    "adaptive_refits",
    "compute_model_id",
    "fit_and_validate",
    "fit_fold",
    "run_climate_fit",
    "select_shrinkage",
    "valid_air",
    "validate",
    "write_model",
]

log = logging.getLogger(__name__)

MODELS_DIRNAME = "climate-models"
MODEL_SCHEMA_VERSION = 2
#: The observer must have run this long on known inputs before an origin (B's slab converges).
WARMUP_SLOTS = 96
HORIZON_SLOTS = 96
SLOT_H = thermal.STEP_H
SLOTS_PER_DAY = 96
#: Training filters that mark a bad air-temperature measurement; the thermal fit does not score
#: those slots.  ``valves_shut`` and ``rule_disabled_day`` are plant states, ``outdoor_unknown``
#: an input gap (the slot cannot be simulated anyway): their slots are scored.
BAD_AIR_FILTERS = ("unknown_quality", "window_open", "thermostat_outage", "cooling")


class ClimateFitError(ValueError):
    """``climate-fit`` cannot run on this configuration or data."""


# ---------------------------------------------------------------------------
# Masks and inputs
# ---------------------------------------------------------------------------


def _dates(d: ClimateData) -> np.ndarray:
    return np.array([label for label, _lo, _hi in d.days], dtype=object)


def train_days(d: ClimateData, seasons: Iterable[str]) -> np.ndarray:
    return np.isin(d.day_season, list(seasons)) & d.day_complete & d.day_fit_month


def train_slots(d: ClimateData, seasons) -> np.ndarray:
    return d.day_mask(train_days(d, seasons))


def test_days(d: ClimateData, season: str) -> np.ndarray:
    return (d.day_season == season) & d.day_complete & d.day_gate_month


def heating_days(d: ClimateData, min_kwh: float) -> np.ndarray:
    """Per local day: heat (outside DHW slots) above *min_kwh*, as the audit counts it."""
    q = np.where(d.dhw, 0.0, d.q_heat_kwh)
    return np.array([np.nansum(q[lo:hi]) > min_kwh for _l, lo, hi in d.days], dtype=bool)


def valid_air(d: ClimateData) -> np.ndarray:
    """Slots whose air temperature is a valid measurement (known, and no ``BAD_AIR_FILTERS``)."""
    bad = np.logical_or.reduce([d.drop[name] for name in BAD_AIR_FILTERS])
    return np.isfinite(d.t_air) & ~bad


def q_inputs(d: ClimateData) -> np.ndarray:
    w = thermal.input_matrix(d.q_heat_kw, d.t_out_model, d.solar)
    w[d.cooling] = np.nan  # a cooling slot is not a heating input
    return w


def comfort_matrix(d: ClimateData, field: Mapping[str, np.ndarray]) -> np.ndarray:
    return np.vstack([field[r] for r in d.comfort_rooms])


def plant_inputs(d: ClimateData) -> plant.PlantInputs:
    return plant.PlantInputs(
        u=d.u,
        dhw=d.dhw,
        t_out=d.t_out_model,
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
    local days), scored on every training slot with a valid air temperature (:func:`valid_air`)."""
    w = q_inputs(d)
    usable = np.isfinite(w).all(axis=1) & train
    days_per_window = cfg.model.window_hours // 24
    window_id = np.where(train, d.day_of_slot // days_per_window, -1)
    y_init = np.where(train, d.t_air, np.nan)
    scored = train & valid_air(d)
    y_air = np.where(scored, d.t_air, np.nan)
    y_ret = np.where(scored, slab_measurement(d, cfg), np.nan)
    segments = thermal.segments_in_windows(window_id, usable, y_init)
    return thermal.FitInputs(w=w, y_air=y_air, y_init=y_init, segments=segments, y_ret=y_ret)


# ---------------------------------------------------------------------------
# Fitted models
# ---------------------------------------------------------------------------


class FittedModel:
    """A fitted thermal model with its observer and heat response."""

    def __init__(
        self, kind: str, fit: thermal.ThermalFit, gain, process_var, response=None, stats=None
    ):
        self.kind = kind
        self.fit = fit
        self.ad, self.bd = thermal.discretize(*thermal.continuous(kind, fit.params))
        self.gain = gain
        self.process_var = process_var
        self.response = response
        self.response_stats = stats or {}
        self.states: np.ndarray | None = None
        self.age: np.ndarray | None = None

    def at(self, t: int) -> FittedModel:
        """The parameters that hold at slot *t* (a static model: itself)."""
        return self

    def pieces(self, lo: int, hi: int) -> list[tuple[int, int, FittedModel]]:
        return [(lo, hi, self)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "thermal": self.fit.to_dict(),
            "observer": {
                "gain": [float(g) for g in self.gain],
                "process_var_k2": self.process_var,
                "measurement_sigma_k": thermal.MEASUREMENT_SIGMA_K,
                "warmup_slots": WARMUP_SLOTS,
            },
            "response": self.response.to_dict() if self.response is not None else None,
            "response_fit": self.response_stats,
        }


def fit_thermal_model(
    d: ClimateData,
    cfg: ClimateConfig,
    kind: str,
    train: np.ndarray,
    *,
    prior: Mapping[str, float] | None = None,
    shrinkage: float = 0.0,
) -> FittedModel:
    """Thermal fit on the *train* slots and its observer gain (no states, no response)."""
    fi = fit_inputs(d, cfg, train)
    fit = thermal.fit_thermal(kind, fi, cfg.model.envelope, prior=prior, shrinkage=shrinkage)
    ad, bd = thermal.discretize(*thermal.continuous(kind, fit.params))
    meas_var = thermal.MEASUREMENT_SIGMA_K**2
    proc = thermal.process_variance(
        ad, bd, np.where(train[:, None], fi.w, np.nan), fi.y_air, meas_var
    )
    return FittedModel(kind, fit, thermal.observer_gain(ad, proc, meas_var), proc)


def observe(
    d: ClimateData, m: FittedModel, w: np.ndarray, lo: int, hi: int
) -> tuple[np.ndarray, np.ndarray]:
    """Observer states and ages of slots ``lo..hi-1``, the filter started at *lo*."""
    slab = None
    if m.kind == "B":
        p = m.fit.params
        slab = (float(np.exp(-p["k"] * thermal.STEP_H / p["c_s"])), p["k"])
    return thermal.observe(m.ad, m.bd, w[lo:hi], d.t_air[lo:hi], m.gain, m.fit.slab_offset_k, slab)


@dataclass
class Week:
    """One causal refit: parameters fitted on the window before *start*, used from *start*."""

    start: date
    lo: int
    hi: int
    window: tuple[date, date]  # [first day, origin): the days the fit may read
    model: FittedModel
    #: Why the prior was kept instead of a refit (``""`` for a refit).
    fallback: str = ""

    @property
    def ok(self) -> bool:
        """A refit with every parameter inside the envelope and none on a bound (``g_s`` at its
        0 bound is legitimate: ``thermal.FAIL_ON_BOUND`` does not list it)."""
        f = self.model.fit
        return not self.fallback and not f.on_bound and not f.outside_envelope

    def to_dict(self) -> dict[str, Any]:
        f = self.model.fit
        return {
            "start": self.start.isoformat(),
            "window": [self.window[0].isoformat(), self.window[1].isoformat()],
            "params": f.params,
            "derived": thermal.derived(self.model.kind, f.params),
            "ok": self.ok,
            "fallback": self.fallback,
            "on_bound": f.on_bound,
            "outside_envelope": f.outside_envelope,
            "identifiable": f.identifiable,
            "converged": f.converged,
            "rmse_air_k": f.rmse_air_k,
            "n_air": f.n_air,
            "n_segments": f.n_segments,
        }


class Adaptive:
    """Weekly causal refits of one model kind: the parameters and observer state per slot."""

    def __init__(self, kind: str, n: int, nx: int, shrinkage: float):
        self.kind = kind
        self.shrinkage = shrinkage
        self.weeks: list[Week] = []
        self.states = np.full((n, nx), np.nan)
        self.age = np.full(n, -1, dtype=np.int64)
        self.week_of_slot = np.full(n, -1, dtype=np.int64)

    def at(self, t: int) -> FittedModel:
        return self.weeks[int(self.week_of_slot[t])].model

    def pieces(self, lo: int, hi: int) -> list[tuple[int, int, FittedModel]]:
        """Runs of slots of one week in ``lo..hi-1``; a slot outside every week keeps the
        parameters of the slot before it (or, at the start, the next week's)."""
        idx = self.week_of_slot[lo:hi].copy()
        known = np.flatnonzero(idx >= 0)
        if known.size == 0:
            raise ClimateFitError(f"no weekly refit covers slots {lo}..{hi}")
        fill = np.maximum.accumulate(np.where(idx >= 0, np.arange(idx.size), -1))
        idx = np.where(fill >= 0, idx[np.maximum(fill, 0)], idx[known[0]])
        cuts = [0, *(np.flatnonzero(np.diff(idx)) + 1).tolist(), idx.size]
        return [
            (lo + a, lo + b, self.weeks[int(idx[a])].model)
            for a, b in zip(cuts[:-1], cuts[1:], strict=True)
        ]

    def summary(self) -> dict[str, Any]:
        weeks = [w.to_dict() for w in self.weeks]
        names = list(self.weeks[0].model.fit.params) if self.weeks else []
        refits = [w for w in self.weeks if not w.fallback]
        ranges = {
            n: [
                float(min(w.model.fit.params[n] for w in refits)),
                float(max(w.model.fit.params[n] for w in refits)),
            ]
            for n in names
            if refits and all(n in w.model.fit.params for w in refits)
        }
        return {
            "shrinkage": self.shrinkage,
            "weeks": len(self.weeks),
            "ok": sum(w.ok for w in self.weeks),
            "ok_share": (sum(w.ok for w in self.weeks) / len(self.weeks)) if self.weeks else None,
            "fallbacks": sum(bool(w.fallback) for w in self.weeks),
            "not_identifiable": sum(not w.model.fit.identifiable for w in refits),
            "not_converged": sum(not w.model.fit.converged for w in refits),
            "param_ranges": ranges,
            "refits": weeks,
        }


def week_starts(first: date, last: date, every_days: int) -> list[date]:
    out, day = [], first
    while day <= last:
        out.append(day)
        day += timedelta(days=every_days)
    return out


def adaptive_refits(
    d: ClimateData,
    cfg: ClimateConfig,
    prior: FittedModel,
    first: date,
    last: date,
    shrinkage: float,
    *,
    w: np.ndarray | None = None,
) -> Adaptive:
    """Causal weekly refits of *prior*'s kind from local day *first* through *last*.

    The refit at origin ``D`` reads only the fit-month complete days in ``[D - window_days, D)``
    of seasons outside ``seasons.training_excluded`` (and before ``reserved_from``); its observer
    starts at ``D - window_days`` and its parameters hold for ``[D, D + refit_every_days)``.  A
    window without a fit segment keeps the prior (a fallback, not an ok refit).
    """
    a = cfg.model.adaptive
    w = q_inputs(d) if w is None else w
    labels = _dates(d)
    index = {label: j for j, label in enumerate(labels)}
    allowed = (
        (d.day_season != "")
        & ~np.isin(d.day_season, list(cfg.seasons.training_excluded))
        & d.day_complete
        & d.day_fit_month
        & np.array([lab < cfg.seasons.reserved_from for lab in labels], dtype=bool)
    )
    out = Adaptive(prior.kind, d.n, prior.ad.shape[0], float(shrinkage))
    for start in week_starts(first, last, a.refit_every_days):
        j = index.get(start)
        if j is None:
            continue
        end = start + timedelta(days=a.refit_every_days)
        lo = d.days[j][1]
        hi = d.days[index[end]][1] if end in index else d.n
        window = (start - timedelta(days=a.window_days), start)
        in_window = np.array([window[0] <= lab < window[1] for lab in labels], dtype=bool)
        train = d.day_mask(allowed & in_window)
        fallback = ""
        try:
            m = fit_thermal_model(
                d, cfg, prior.kind, train, prior=prior.fit.params, shrinkage=shrinkage
            )
        except ValueError as exc:
            m = FittedModel(prior.kind, prior.fit, prior.gain, prior.process_var)
            fallback = str(exc) or "no fit"
        m.response, m.response_stats = prior.response, prior.response_stats
        obs_lo = d.days[index[window[0]]][1] if window[0] in index else 0
        states, age = observe(d, m, w, obs_lo, hi)
        out.states[lo:hi] = states[lo - obs_lo :]
        out.age[lo:hi] = age[lo - obs_lo :]
        out.week_of_slot[lo:hi] = len(out.weeks)
        out.weeks.append(Week(start, lo, hi, window, m, fallback))
    return out


# ---------------------------------------------------------------------------
# One fold
# ---------------------------------------------------------------------------


class FoldResult:
    def __init__(self, index: int, train: tuple[str, ...], test: str):
        self.index, self.train, self.test = index, train, test
        self.offsets: np.ndarray | None = None
        self.cap: plant.ValveCap | None = None
        self.models: dict[str, FittedModel] = {}
        self.adaptive: dict[str, Adaptive] = {}
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
        & np.isfinite(d.t_out_model)
        & np.isfinite(cap_meas)
        & ~drop["thermostat_outage"]
        & ~drop["unknown_quality"]
    )


def _day_features(d: ClimateData, j: int) -> tuple[int, float, float, float] | None:
    """``(blocks, hours_on, mean T_out, starts)`` of local day *j*, or ``None``."""
    _l, lo, hi = d.days[j]
    starts = d.columns["compressor_starts_delta"][lo:hi]
    q = d.q_heat_kwh[lo:hi]
    t_out = d.t_out_model[lo:hi]
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


def _shrinkage(cfg: ClimateConfig) -> dict[str, float]:
    values = dict(cfg.model.adaptive.shrinkage)
    missing = [k for k in thermal.KINDS if values.get(k) is None]
    if missing:
        raise ClimateFitError(
            f"model.adaptive.shrinkage is not selected for {missing}: choose it on S2 "
            "(hack/climate_shrinkage.py) and commit it before any held-out run"
        )
    return {k: float(values[k]) for k in thermal.KINDS}


def _span(d: ClimateData, days: np.ndarray) -> tuple[date, date]:
    idx = np.flatnonzero(days)
    if idx.size == 0:
        raise ClimateFitError("no test day")
    return d.days[int(idx[0])][0], d.days[int(idx[-1])][0]


def fit_fold(d: ClimateData, cfg: ClimateConfig, index: int, phi: float) -> FoldResult:
    """Fit offsets, prior thermal models, observers, responses and starts on fold *index*, then
    the causal weekly refits over its test season."""
    shrinkage = _shrinkage(cfg)
    fold = cfg.folds[index]
    res = FoldResult(index, tuple(fold.train), fold.test)
    train = train_slots(d, fold.train)
    res.offsets = room_offsets(d.room_temp, d.comfort_rooms, d.t_air, d.local_hour, train & d.train)
    res.cap = plant.ValveCap(phi, cfg.model.response.plant_valve_cap_margin_k, res.offsets)
    cap_meas = res.cap.measured(comfort_matrix(d, d.room_temp), comfort_matrix(d, d.room_setpoint))
    w = q_inputs(d)
    t_out = d.t_out_model
    sel = _response_selection(d, train, cap_meas)
    first = plant.first_hour_of_block(d.u)
    t_ft = d.columns["flow_target_temp_mean_c"]
    t_ref = cfg.comfort.low_c
    span = _span(d, test_days(d, fold.test))
    for kind in thermal.KINDS:
        model = fit_thermal_model(d, cfg, kind, train)
        states, age = observe(d, model, w, 0, d.n)
        if kind == "A":
            s = sel
            response, stats = plant.fit_response_a(
                d.q_heat_kw[s], cap_meas[s], t_out[s], first[s], t_ref
            )
        else:
            s = sel & np.isfinite(t_ft) & np.isfinite(states[:, 1])
            response, stats = plant.fit_response_b(
                d.q_heat_kw[s], cap_meas[s], t_out[s], t_ft[s], states[s, 1], t_ref
            )
        model.response, model.response_stats = response, stats
        model.states, model.age = states, age
        res.models[kind] = model
        log.info("climate-fit: weekly refits of %s", kind)
        res.adaptive[kind] = adaptive_refits(d, cfg, model, *span, shrinkage[kind], w=w)
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
    model: FittedModel | Adaptive,
    origins: np.ndarray,
    mode: str,
    cap: plant.ValveCap | None,
    inp: plant.PlantInputs | None,
    known_u: np.ndarray | None,
    *,
    w: np.ndarray | None = None,
) -> dict[int, list[tuple[int, float]]]:
    """``{origin: [(steps ahead j, predicted air temperature of slot origin + j)]}``, with the
    parameters that hold at the origin for the whole horizon."""
    w = q_inputs(d) if w is None else w
    known_q = np.isfinite(w).all(axis=1)
    out: dict[int, list[tuple[int, float]]] = {}
    for o in origins:
        x0 = model.states[o]
        m = model.at(int(o))
        if mode == "q":
            length = _run_length(known_q, o, HORIZON_SLOTS)
            xs = thermal.simulate(m.ad, w[o : o + length] @ m.bd.T, x0)
        else:
            length = _run_length(known_u, o, HORIZON_SLOTS)
            xs, _q = plant.simulate_u(m.kind, m.ad, m.bd, x0, o, o + length, inp, m.response, cap)
        out[int(o)] = [(j, float(xs[j + 1, 0])) for j in range(length)]
    return out


def simulate_u(
    model: FittedModel | Adaptive,
    lo: int,
    hi: int,
    inp: plant.PlantInputs,
    cap: plant.ValveCap,
) -> tuple[np.ndarray, np.ndarray]:
    """u-driven states ``x[lo..hi]`` and heat (kW) from the observer state at *lo*; the
    parameters switch at the week boundaries (each was fitted before its week began)."""
    x = model.states[lo]
    xs, qs = [np.asarray(x, dtype=float)[None, :]], []
    for a, b, m in model.pieces(lo, hi):
        part, q = plant.simulate_u(m.kind, m.ad, m.bd, x, a, b, inp, m.response, cap)
        xs.append(part[1:])
        qs.append(q)
        x = part[-1]
    return np.vstack(xs), np.concatenate(qs)


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
    models: Mapping[str, FittedModel | Adaptive],
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
            xs, q = simulate_u(m, lo, hi, inp, cap)
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


def legacy_replay(
    d: ClimateData,
    cfg: ClimateConfig,
    models: Mapping[str, FittedModel | Adaptive],
    days: np.ndarray,
    cap: plant.ValveCap,
    inp: plant.PlantInputs,
    known_u: np.ndarray,
    cop_model: cop.CopModel | None,
) -> dict[str, Any]:
    """The legacy rule replayed through each model over episodes of consecutive simulable days,
    started from the observer, against the measured values on the same slots.

    Degree-hours below the comfort floor per comfort room; heating electricity of the simulated
    heat through *cop_model* (the lift read from the flow target, the input the response and any
    policy simulation have) and its cost at the PVPC price, against the measured heat-mode
    electricity and its cost, on the episode slots whose electricity and price are known."""
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
    e_meas = d.e_heat_kwh
    price = d.price
    t_ft = d.columns["flow_target_temp_mean_c"]
    meas_e = meas_cost = 0.0
    sim_e = dict.fromkeys(models, 0.0)
    sim_cost = dict.fromkeys(models, 0.0)
    for lo, hi in episodes:
        hours = d.local_hour[lo:hi]
        known = np.isfinite(rooms[:, lo:hi])
        meas_kh += np.where(known, np.maximum(0.0, low - rooms[:, lo:hi]), 0.0).sum(1) * SLOT_H
        billed = np.isfinite(e_meas[lo:hi]) & np.isfinite(price[lo:hi])
        meas_e += float(e_meas[lo:hi][billed].sum())
        meas_cost += float((e_meas[lo:hi] * price[lo:hi])[billed].sum())
        for k, m in models.items():
            xs, q = simulate_u(m, lo, hi, inp, cap)
            room_sim = xs[1:, 0][None, :] + np.nan_to_num(cap.offsets[:, hours])
            sim_kh[k] += np.where(known, np.maximum(0.0, low - room_sim), 0.0).sum(1) * SLOT_H
            if cop_model is not None:
                q_kwh = q * SLOT_H
                heat = q_kwh > 0.0
                e = np.zeros(q.size)
                e[heat] = cop.predict_electricity(
                    cop_model, q_kwh[heat], t_ft[lo:hi][heat], inp.t_out[lo:hi][heat]
                )
                sim_e[k] += float(e[billed].sum())
                sim_cost[k] += float((e * np.where(billed, price[lo:hi], 0.0))[billed].sum())
    out["meas_kh"] = {r: float(v) for r, v in zip(d.comfort_rooms, meas_kh, strict=True)}
    out["meas_kh_total"] = float(meas_kh.sum())
    out["meas_electricity_kwh"] = meas_e
    out["meas_cost"] = meas_cost
    out["cop_model"] = cop_model.selected if cop_model is not None else None

    def rel(sim: float, meas: float) -> float | None:
        return abs(sim - meas) / meas if meas > 0 else None

    for k in models:
        total = float(sim_kh[k].sum())
        out[k] = {
            "sim_kh": {r: float(v) for r, v in zip(d.comfort_rooms, sim_kh[k], strict=True)},
            "sim_kh_total": total,
            "abs_error_kh": abs(total - out["meas_kh_total"]),
            "rel_error": rel(total, out["meas_kh_total"]),
            "sim_electricity_kwh": sim_e[k] if cop_model is not None else None,
            "electricity_rel_error": rel(sim_e[k], meas_e) if cop_model is not None else None,
            "electricity_signed_error": (sim_e[k] / meas_e - 1.0)
            if cop_model is not None and meas_e > 0
            else None,
            "sim_cost": sim_cost[k] if cop_model is not None else None,
            "cost_rel_error": rel(sim_cost[k], meas_cost) if cop_model is not None else None,
            "cost_signed_error": (sim_cost[k] / meas_cost - 1.0)
            if cop_model is not None and meas_cost > 0
            else None,
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


def t_out_fill_summary(d: ClimateData, slots: np.ndarray) -> dict[str, Any]:
    """How much of *slots* the model's outdoor series filled from the fused one, and how much
    stays unknown."""
    filled = d.t_out_filled if d.t_out_filled is not None else np.zeros(d.n, dtype=bool)
    n = int(slots.sum())
    return {
        "slots": n,
        "filled_slots": int((filled & slots).sum()),
        "filled_share": float((filled & slots).sum() / n) if n else None,
        "debiased_unknown_share": float((~np.isfinite(d.t_out_debiased) & slots).sum() / n)
        if n
        else None,
        "model_unknown_share": float((~np.isfinite(d.t_out_model) & slots).sum() / n)
        if n
        else None,
    }


def validate(
    d: ClimateData, cfg: ClimateConfig, fold: FoldResult, cop_model: cop.CopModel | None = None
) -> dict[str, Any]:
    """Validation of the weekly refits of a fold on its test season (gate months)."""
    days = test_days(d, fold.test)
    models = fold.adaptive
    origins = _origins(d, cfg, days, models.values())
    inp = plant_inputs(d)
    w = q_inputs(d)
    known_u = inp.known("B") & np.isfinite(w[:, 1:]).all(axis=1)
    cap = fold.cap
    scored = d.train & d.day_mask(days)
    q_preds = {k: forecasts(d, m, origins, "q", cap, inp, known_u, w=w) for k, m in models.items()}
    u_origins = origins[d.u_known[origins]]
    u_preds = {
        k: forecasts(d, m, u_origins, "u", cap, inp, known_u, w=w) for k, m in models.items()
    }
    return {
        "test_season": fold.test,
        "test_days": int(days.sum()),
        "origins": int(origins.size),
        "u_origins": int(u_origins.size),
        "t_out_fill": {
            "train": t_out_fill_summary(d, train_slots(d, fold.train)),
            "test": t_out_fill_summary(d, d.day_mask(days)),
        },
        "q_driven": horizon_metrics(d, cfg, q_preds, scored),
        "u_driven": horizon_metrics(d, cfg, u_preds, scored),
        "daily_energy": daily_energy(d, cfg, models, days, cap, inp, known_u),
        "legacy_replay": legacy_replay(d, cfg, models, days, cap, inp, known_u, cop_model),
        "starts": starts_validation(d, fold.starts, days),
        "adaptive": {k: m.summary() for k, m in models.items()},
    }


# ---------------------------------------------------------------------------
# Shrinkage selection (S2 only)
# ---------------------------------------------------------------------------


def _choose(rows: Mapping[str, Mapping[str, Any]], min_ok: float) -> str:
    """The grid value with the lowest score among those meeting *min_ok* (none: the largest);
    a tie goes to the larger value."""
    names = list(rows)
    eligible = [n for n in names if (rows[n]["ok_share"] or 0.0) >= min_ok]
    if not eligible:
        return max(names, key=lambda n: rows[n]["shrinkage"])

    def key(n: str) -> tuple[float, float]:
        score = rows[n]["score"]
        return (score if score is not None else float("inf"), -rows[n]["shrinkage"])

    return min(eligible, key=key)


def select_shrinkage(d: ClimateData, cfg: ClimateConfig) -> dict[str, Any]:
    """Choose ``model.adaptive.shrinkage`` per model on the selection season only.

    The prior is fitted on the season's fit-month days up to ``selection.prior_until``; weekly
    refits run from the next day through the season's last gate-month day for every grid value.
    Each is scored like the acceptance (Q-driven RMSE on the training-filter mask, origins at
    00/06/12/18, common pairs) at ``acceptance.q_driven.gain_horizons_hours``; the choice is
    :func:`_choose` on the mean of those RMSEs.  Nothing outside the season is fitted or scored
    (the forecasts start in the season's gate months and end within a day).
    """
    a = cfg.model.adaptive
    sel = a.selection
    season = sel.season
    if season == "S3" or season in cfg.seasons.training_excluded:
        raise ClimateFitError("the shrinkage selection must run on a training season")
    labels = _dates(d)
    before = np.array([lab <= sel.prior_until for lab in labels], dtype=bool)
    prior_days = train_days(d, [season]) & before
    eval_days = (d.day_season == season) & d.day_complete & d.day_gate_month & ~before
    if not prior_days.any() or not eval_days.any():
        raise ClimateFitError(f"no prior or evaluation day in {season}")
    span = _span(d, eval_days)
    w = q_inputs(d)
    last_hi = d.days[int(np.flatnonzero(eval_days)[-1])][2]
    obs_hi = min(d.n, last_hi + HORIZON_SLOTS)
    scored = d.train & d.day_mask(eval_days)
    horizons = [str(h) for h in cfg.acceptance.q_driven.gain_horizons_hours]
    min_ok = cfg.acceptance.parameters.weekly_refit_ok_share
    out: dict[str, Any] = {
        "season": season,
        "prior_until": sel.prior_until.isoformat(),
        "evaluation": [span[0].isoformat(), span[1].isoformat()],
        "grid": list(sel.grid),
        "horizons_hours": horizons,
        "min_ok_share": min_ok,
        "models": {},
    }
    for kind in thermal.KINDS:
        log.info("shrinkage selection: prior %s", kind)
        prior = fit_thermal_model(d, cfg, kind, d.day_mask(prior_days))
        prior.states = np.full((d.n, prior.ad.shape[0]), np.nan)
        prior.age = np.full(d.n, -1, dtype=np.int64)
        prior.states[:obs_hi], prior.age[:obs_hi] = observe(d, prior, w, 0, obs_hi)
        variants: dict[str, FittedModel | Adaptive] = {"prior_only": prior}
        for lam in sel.grid:
            log.info("shrinkage selection: %s, shrinkage %g", kind, lam)
            variants[f"{lam:g}"] = adaptive_refits(d, cfg, prior, *span, float(lam), w=w)
        origins = _origins(d, cfg, eval_days, variants.values())
        preds = {
            n: forecasts(d, m, origins, "q", None, None, None, w=w) for n, m in variants.items()
        }
        metrics = horizon_metrics(d, cfg, preds, scored)
        rows: dict[str, dict[str, Any]] = {}
        for name, m in variants.items():
            rmse = {h: metrics[h][name]["rmse"] for h in horizons}
            score = (
                None
                if any(v is None for v in rmse.values())
                else float(np.mean(list(rmse.values())))
            )
            row: dict[str, Any] = {
                "rmse": {h: metrics[h][name]["rmse"] for h in metrics},
                "bias": {h: metrics[h][name]["bias"] for h in metrics},
                "score": score,
            }
            if isinstance(m, Adaptive):
                summary = m.summary()
                row.update(
                    shrinkage=m.shrinkage,
                    ok_share=summary["ok_share"],
                    weeks=summary["weeks"],
                    fallbacks=summary["fallbacks"],
                    param_ranges=summary["param_ranges"],
                )
            rows[name] = row
        grid_rows = {n: r for n, r in rows.items() if n != "prior_only"}
        chosen = _choose(grid_rows, min_ok)
        out["models"][kind] = {
            "prior": prior.fit.to_dict(),
            "origins": int(origins.size),
            "pairs": {h: metrics[h]["persistence"]["n"] for h in metrics},
            "baselines": {
                h: {b: metrics[h][b]["rmse"] for b in ("persistence", "diurnal")} for h in metrics
            },
            "variants": rows,
            "chosen": grid_rows[chosen]["shrinkage"],
        }
    return out


def render_selection(report: Mapping[str, Any]) -> str:
    lines = [
        "# Shrinkage selection (S2 only; private figures)",
        "",
        f"- season {report['season']}, prior up to {report['prior_until']}, weekly refits "
        f"{report['evaluation'][0]}..{report['evaluation'][1]}",
        f"- score: mean Q-driven RMSE at {', '.join(report['horizons_hours'])} h among grid "
        f"values with ok share >= {report['min_ok_share']}",
        "",
    ]
    for kind, m in report["models"].items():
        lines += [
            f"## Model {kind}: chosen shrinkage {m['chosen']:g} ({m['origins']} origins)",
            "",
            "| variant | score | rmse 6 h | rmse 12 h | rmse 24 h | bias 24 h | ok share | "
            "fallbacks |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, r in m["variants"].items():
            lines.append(
                f"| {name} | {_fmt(r['score'])} | {_fmt(r['rmse'].get('6'))} | "
                f"{_fmt(r['rmse'].get('12'))} | {_fmt(r['rmse'].get('24'))} | "
                f"{_fmt(r['bias'].get('24'))} | {_fmt(r.get('ok_share'))} | "
                f"{r.get('fallbacks', '–')} |"
            )
        lines += ["", f"- baselines: {m['baselines']}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Acceptance and selection
# ---------------------------------------------------------------------------


def _criterion(cid: str, value, threshold, passed, note: str = "") -> dict[str, Any]:
    status = "pending" if passed is None else ("pass" if passed else "fail")
    return {"id": cid, "value": value, "threshold": threshold, "status": status, "note": note}


def _le(value, limit) -> bool:
    return value is not None and np.isfinite(value) and value <= limit


def acceptance(
    cfg: ClimateConfig,
    folds: list[FoldResult],
    kind: str,
    cop_fold: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Every criterion of ``acceptance`` (and the COP limits) for model *kind* on the fold that
    tests S3."""
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
    rep = v["legacy_replay"][kind]
    no_cop = "no COP model (see cop)"
    for cid, key, limit in (
        ("legacy_replay_electricity", "electricity_rel_error", lr.electricity_rel_error),
        ("legacy_replay_cost", "cost_rel_error", lr.cost_rel_error),
    ):
        value = rep[key]
        crit.append(
            _criterion(
                cid,
                value,
                limit,
                _le(value, limit),
                "simulated legacy against realized, through the COP model"
                if v["legacy_replay"]["cop_model"]
                else no_cop,
            )
        )
    kh_pass = (rep["rel_error"] is not None and rep["rel_error"] <= lr.kh_below_low_rel_error) or (
        rep["abs_error_kh"] <= lr.kh_below_low_abs_error
    )
    crit.append(
        _criterion(
            "legacy_replay_kh_below_low",
            {"rel_error": rep["rel_error"], "abs_error_kh": rep["abs_error_kh"]},
            {"rel": lr.kh_below_low_rel_error, "abs_kh": lr.kh_below_low_abs_error},
            kh_pass,
            "summed over the comfort rooms; passes within the relative or the absolute bound",
        )
    )
    failures = fold.models[kind].fit.failures()
    crit.append(
        _criterion(
            "parameters_inside_envelope",
            {"prior": failures},
            "no parameter on a bound or outside the envelope; identifiable",
            not failures,
            "the prior fit on the training season",
        )
    )
    summary = v["adaptive"][kind]
    crit.append(
        _criterion(
            "parameters_weekly_refits",
            summary["ok_share"],
            acc.parameters.weekly_refit_ok_share,
            summary["ok_share"] is not None
            and summary["ok_share"] >= acc.parameters.weekly_refit_ok_share,
            f"{summary['ok']} of {summary['weeks']} weekly refits with every parameter inside "
            "the envelope and none on a bound (g_s at 0 allowed; a fallback to the prior fails)",
        )
    )
    limits = cfg.cop
    if cop_fold is None or "validation" not in cop_fold:
        note = (cop_fold or {}).get("error", "no COP model")
        crit.append(_criterion("cop_daily_mape", None, limits.daily_mape, False, note))
        crit.append(_criterion("cop_season_error", None, limits.season_error, False, note))
    else:
        cv = cop_fold["validation"][cop_fold["selected"]]
        note = f"{cop_fold['selected']} given measured Q, {cv['days']} days"
        crit.append(
            _criterion(
                "cop_daily_mape",
                cv["daily_mape"],
                limits.daily_mape,
                _le(cv["daily_mape"], limits.daily_mape),
                note,
            )
        )
        se = abs(cv["season_error"])
        crit.append(
            _criterion(
                "cop_season_error", se, limits.season_error, _le(se, limits.season_error), note
            )
        )
    failed = [c["id"] for c in crit if c["status"] == "fail"]
    pending = [c["id"] for c in crit if c["status"] == "pending"]
    return {
        "criteria": crit,
        "failed": failed,
        "pending": pending,
        "accepted": not failed and not pending,
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
    input_sha256: str,
) -> str:
    """SHA-256 over every input: *input_sha256* (``ClimateData.input_sha256``) adds the station
    manifest and the outdoor source, so a different T_out series is a different model."""
    payload = {
        "feature_set_id": feature_set_id,
        "dataset_id": dataset_id,
        "windows": windows,
        "code_version": code_ver,
        "config_sha256": config_sha256,
        "site_sha256": site_sha256,
        "input_sha256": input_sha256,
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


def _range(v: list[float]) -> str:
    return f"{v[0]:.3g}..{v[1]:.3g}"


def render_markdown(model: dict[str, Any], validation: dict[str, Any]) -> str:
    lines = [
        f"# Climate model `{model['model_id'][:12]}`",
        "",
        f"- Feature set `{model['lineage']['feature_set_id']}`, dataset "
        f"`{model['lineage']['dataset_id']}`, code `{model['lineage']['code_version']}`",
        f"- Config sha256 `{model['lineage']['config_sha256'][:12]}`, site sha256 "
        f"`{model['lineage']['site_sha256'][:12]}`, inputs sha256 "
        f"`{model['lineage']['input_sha256'][:12]}`, phi {model['phi']}",
        "",
        f"**Verdict: {validation['selection']['verdict']}** (selected: "
        f"{validation['selection']['selected'] or 'none'}; pending: "
        f"{', '.join(validation['selection']['pending']) or 'none'})",
        "",
    ]
    for fold in model["folds"]:
        v = validation["folds"][fold["fold"] - 1]
        lines += [
            f"## Fold {fold['fold']}: train {'+'.join(fold['train'])}, test {fold['test']}",
            "",
            f"- Outdoor series: {v['t_out_fill']}",
            "",
            "| prior | parameters | tau h | C kWh/K | balance °C | fit failures |",
            "|---|---|---|---|---|---|",
        ]
        for kind, m in fold["models"].items():
            th = m["thermal"]
            params = ", ".join(f"{k}={x:.3g}" for k, x in th["params"].items())
            der = th["derived"]
            lines.append(
                f"| {kind} | {params} | {der['tau_h']:.1f} | {der['c_kwh_per_k']:.2f} | "
                f"{der['balance_temp_c']:.1f} | {'; '.join(th['failures']) or 'none'} |"
            )
        lines.append("")
        for kind, m in fold["models"].items():
            resp = ", ".join(f"{k}={x:.3g}" for k, x in m["response"].items())
            lines.append(f"- Response {kind}: {resp} (fit {m['response_fit']})")
        lines += [f"- Starts model: {fold['starts']} (fit {fold['starts_fit']})", ""]
        for kind, a in v["adaptive"].items():
            ranges = ", ".join(f"{n} {_range(r)}" for n, r in a["param_ranges"].items())
            lines += [
                f"### Weekly refits {kind} (shrinkage {a['shrinkage']:g})",
                "",
                f"- {a['ok']} of {a['weeks']} ok, {a['fallbacks']} fallbacks, "
                f"{a['not_identifiable']} not identifiable, {a['not_converged']} not converged",
                f"- parameter ranges: {ranges}",
                "",
            ]
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
        lr = v["legacy_replay"]
        lines += [
            "",
            f"### Legacy replay ({lr['days']} days, {lr['episodes']} episodes, COP "
            f"{lr['cop_model']})",
            "",
        ]
        for kind in fold["models"]:
            r = lr[kind]
            lines.append(
                f"- {kind}: Kh rel error {_fmt(r['rel_error'])}, abs error "
                f"{_fmt(r['abs_error_kh'], 1)} Kh; electricity signed error "
                f"{_fmt(r['electricity_signed_error'])}, cost signed error "
                f"{_fmt(r['cost_signed_error'])}"
            )
        lines += ["", f"- Starts validation: {v['starts']}", ""]
    cv = validation.get("cop", {})
    if "validation" in cv:
        lines += [f"## COP ({cv['selected']}, trained on {'+'.join(cv['train'])})", ""]
        for name, r in cv["validation"].items():
            lines.append(
                f"- {name}: daily MAPE {_fmt(r['daily_mape'])}, season error "
                f"{_fmt(r['season_error'])}, passes {r['passes']}"
            )
        lines.append("")
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


def _cop_fold(d: ClimateData, cfg: ClimateConfig) -> tuple[dict[str, Any], cop.CopModel | None]:
    """The COP model of the fold that tests S3 (trained on its training seasons)."""
    try:
        report = cop.fit_and_validate(d, cfg)
    except cop.CopError as exc:
        return {"error": f"COP model: {exc}"}, None
    fold = next(f for f in report["folds"] if f["test"] == "S3")
    keep = ("train", "test", "train_hours", "test_hours", "model", "model_id", "selected")
    out = {k: fold[k] for k in keep}
    out["held_out_best"] = fold["held_out_best"]
    out["validation"] = {
        name: {k: x for k, x in r.items() if k != "residuals_by_t_out"}
        for name, r in fold["validation"].items()
    }
    return out, cop.CopModel.from_dict(fold["model"])


def fit_and_validate(
    d: ClimateData, cfg: ClimateConfig, site: Site
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit and validate every fold; return the ``model.json`` and ``validation.json`` payloads."""
    _shrinkage(cfg)  # refuse before any fit when the shrinkage was not predeclared
    phi = site.phi if site.phi is not None else cfg.model.response.phi_default
    cop_fold, cop_model = _cop_fold(d, cfg)
    folds = []
    for i in range(len(cfg.folds)):
        log.info("climate-fit: fold %d", i + 1)
        fold = fit_fold(d, cfg, i, phi)
        fold.validation = validate(d, cfg, fold, cop_model)
        folds.append(fold)
    verdicts = {k: acceptance(cfg, folds, k, cop_fold) for k in thermal.KINDS}
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
        input_sha256=d.input_sha256,
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
            "input_sha256": d.input_sha256,
            "station_sha256": d.station_sha256,
            "t_out_source": d.t_out_source,
            "t_out_fill_unknown_from_fused": cfg.signals.t_out_fill_unknown_from_fused,
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
                    [None if np.isnan(x) else float(x) for x in row] for row in f.offsets
                ],
                "models": {k: m.to_dict() for k, m in f.models.items()},
                "weekly_refits": {k: [w.to_dict() for w in a.weeks] for k, a in f.adaptive.items()},
                "starts": f.starts.to_dict() if f.starts else None,
                "starts_fit": f.starts_fit,
            }
            for f in folds
        ],
    }
    validation = {
        "model_id": model_id,
        "folds": [f.validation for f in folds],
        "cop": cop_fold,
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
    _shrinkage(cfg)
    site = load_site(root / SITE_FILE)
    if cfg.signals.t_out_fill_unknown_from_fused and site.reference_station is None:
        raise ClimateFitError(
            "signals.t_out_fill_unknown_from_fused needs reference_station in the site file"
        )
    d = load_climate_data(
        cfg, feature_set=feature_set, dataset=dataset, data_dir=str(root), site=site
    )
    model, validation = fit_and_validate(d, cfg, site)
    out = write_model(root, model, validation)
    log.info("wrote climate model %s", out)
    return out
