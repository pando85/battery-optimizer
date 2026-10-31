"""B0's room-air two-state model with a diurnal gain ("B2+D"), and the MPC model builders.

Structure (B0 report, R2; exploratory, not an accepted model): C2's air + slab model B,

    C_a dT_a/dt = k (T_s - T_a) + UA (T_out - T_a) + q0 + g_s S + sum_i g_i D_i
    C_s dT_s/dt = Q - k (T_s - T_a)

with three changes against C2:

* **fitted on room air only.**  The return-water temperature is a flow-dependent proxy of the
  water loop, not a slab thermometer; with it the slab capacity sits on its bound, without it the
  slab is identifiable (about 10 to 13 kWh/K on both seasons);
* **a diurnal gain** ``D``: 24 h and 12 h harmonics (``diurnal_harmonics``), separate for weekdays
  and weekends.  It stands for occupancy and internal gains and for the diurnal error of the
  outdoor sensor, which have no measured driver; it is the largest single gain of B0;
* **causal re-estimation** of ``ua`` and ``q0`` (``model.reestimate``) on a trailing window, every
  few days, the other parameters held at the season prior: the season-level shift of apparent UA
  drives the 24 h bias.

The prior is fitted on the *other* heating season (S2 for an S3 replay, S3 for an S2 replay), so
neither season is scored in sample; S3 was inspected in C2 and B0, so every result is exploratory.

The heat law is C2's response B refitted on the observer slab states of this model:
``Q = valve(T_air) clip(H_e (T_flow_target - T_s), 0, qm0 + qm1 (T_ref - T_out))`` on the
permitted, non-DHW training slots (:func:`battery_optimizer.climate.fit._response_selection`).

:func:`from_c2_record` rebuilds C2's model B from its stored record for comparison.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from battery_optimizer.climate import fit, plant, thermal
from battery_optimizer.climate.config import ClimateConfig
from battery_optimizer.climate.data import ClimateData, room_offsets
from battery_optimizer.climate.mpc_model import HeatLaw, RcComfortModel, rc_matrices

__all__ = [
    "B2D_NAME",
    "C2_NAME",
    "PARAM_BOX",
    "SeasonModel",
    "diurnal_matrix",
    "fit_b2d",
    "fit_heat_law",
    "from_c2_record",
    "make_model",
    "reestimate",
    "slot_calendar",
]

B2D_NAME = "b2d"
C2_NAME = "c2"
#: Search box of B0 (r2 challengers): q0 may be negative once the diurnal term carries the mean.
PARAM_BOX: dict[str, tuple[float, float]] = {
    "ua": (0.03, 0.5),
    "q0": (-1.0, 3.0),
    "g_s": (0.0, 10.0),
    "c_a": (0.05, 60.0),
    "c_s": (0.5, 200.0),
    "k": (0.05, 20.0),
    "g": (-2.0, 2.0),
}
_BASE = ("ua", "c_a", "c_s", "k", "q0", "g_s")
_STARTS = (
    {"ua": 0.1, "q0": 0.3, "g_s": 0.7, "c_a": 3.0, "c_s": 10.0, "k": 1.5},
    {"ua": 0.1, "q0": 0.3, "g_s": 0.7, "c_a": 10.0, "c_s": 3.0, "k": 1.0},
    {"ua": 0.1, "q0": 0.3, "g_s": 0.7, "c_a": 1.0, "c_s": 20.0, "k": 3.0},
)
MAX_NFEV = 300


# ---------------------------------------------------------------------------
# Calendar and inputs
# ---------------------------------------------------------------------------


def slot_calendar(d: ClimateData) -> tuple[np.ndarray, np.ndarray]:
    """``(local hour as a fraction, weekend flag)`` of every slot (local zones are whole hours
    from UTC, so the quarter of the UTC hour is the local quarter)."""
    quarter = ((d.starts_us // 900_000_000) % 4).astype(float)
    weekend = np.array([label.weekday() >= 5 for label, _lo, _hi in d.days], dtype=bool)
    return d.local_hour + quarter / 4.0, weekend[d.day_of_slot]


def diurnal_matrix(hour_frac: np.ndarray, weekend: np.ndarray, harmonics: int) -> np.ndarray:
    """``(n, 4 * harmonics)``: cos/sin of each harmonic of the day, weekday columns then weekend."""
    cols = []
    h = np.asarray(hour_frac, dtype=float)
    for mask in (~np.asarray(weekend, bool), np.asarray(weekend, bool)):
        m = mask.astype(float)
        for k in range(1, harmonics + 1):
            cols += [m * np.cos(2 * np.pi * k * h / 24.0), m * np.sin(2 * np.pi * k * h / 24.0)]
    return np.column_stack(cols) if cols else np.zeros((h.size, 0))


def full_inputs(d: ClimateData, diurnal: np.ndarray) -> np.ndarray:
    """``w = [Q, T_out, S, 1, D...]`` of the measured history (cooling slots unknown)."""
    return np.column_stack([fit.q_inputs(d), diurnal])


def _names(n_diurnal: int) -> tuple[str, ...]:
    return (*_BASE, *(f"g{i}" for i in range(n_diurnal)))


def _box(name: str) -> tuple[float, float]:
    return PARAM_BOX["g"] if name.startswith("g") and name != "g_s" else PARAM_BOX[name]


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


def _segments(d: ClimateData, w: np.ndarray, y: np.ndarray, train: np.ndarray):
    usable = np.isfinite(w).all(axis=1) & train
    window_id = np.where(train, d.day_of_slot // 2, -1)  # 48 h windows from local midnight
    y_init = np.where(train, y, np.nan)
    return thermal.segments_in_windows(window_id, usable, y_init), y_init


class _Residuals:
    """Air-only simulation error with the initial slab temperature projected out per segment."""

    def __init__(self, w, y_init, y_scored, segments, n_diurnal):
        self.w, self.y_init, self.y, self.segments = w, y_init, y_scored, segments
        self.n_diurnal = n_diurnal
        self.idx = [np.flatnonzero(np.isfinite(y_scored[lo:hi])) for lo, hi in segments]
        self.max_len = max(hi - lo for lo, hi in segments)

    def __call__(self, p: Mapping[str, float]) -> np.ndarray:
        ad, bd = rc_matrices(dict(p), self.n_diurnal)
        drive = self.w @ bd.T
        homog = thermal.simulate(ad, np.zeros((self.max_len, 2)), np.array([0.0, 1.0]))[1:, 0]
        out = []
        for (lo, hi), ia in zip(self.segments, self.idx, strict=True):
            y0 = self.y_init[lo - 1]
            xs = thermal.simulate(ad, drive[lo:hi], np.array([y0, 0.0]))[1:]
            r = xs[ia, 0] - self.y[lo:hi][ia]
            g = homog[ia]
            gg = float(g @ g)
            s = -float(r @ g) / gg if gg > 1e-12 else y0
            out.append(r + g * s)
        return np.concatenate(out) if out else np.zeros(0)


def fit_b2d(
    d: ClimateData, w: np.ndarray, train: np.ndarray, n_diurnal: int
) -> tuple[dict[str, float], dict[str, Any]]:
    """The season prior: every parameter, air-only, on the *train* slots (B0's ``fitm``)."""
    y = d.t_air
    segments, y_init = _segments(d, w, y, train)
    if not segments:
        raise ValueError("no fit segment: not enough slots with known inputs")
    scored = np.where(train & fit.valid_air(d), y, np.nan)
    res_fn = _Residuals(w, y_init, scored, segments, n_diurnal)
    names = _names(n_diurnal)
    lo = np.array([_box(n)[0] for n in names])
    hi = np.array([_box(n)[1] for n in names])
    best = None
    for start in _STARTS:
        x0 = np.clip(
            [start.get(n, 0.0) for n in names], lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo)
        )
        r = least_squares(
            lambda th: res_fn(dict(zip(names, th, strict=True))),
            x0,
            bounds=(lo, hi),
            x_scale="jac",
            max_nfev=MAX_NFEV,
        )
        if best is None or r.cost < best.cost:
            best = r
    assert best is not None
    params = {n: float(v) for n, v in zip(names, best.x, strict=True)}
    span = hi - lo
    on_bound = [
        n
        for n, v, a, b, s in zip(names, best.x, lo, hi, span, strict=True)
        if (v - a) <= 1e-3 * s or (b - v) <= 1e-3 * s
    ]
    info = {
        "rmse_air_k": float(np.sqrt(np.mean(best.fun**2))),
        "on_bound": on_bound,
        "segments": len(segments),
        "nfev": int(best.nfev),
    }
    return params, info


def reestimate(
    d: ClimateData,
    w: np.ndarray,
    prior: Mapping[str, float],
    window: np.ndarray,
    names: Sequence[str],
    n_diurnal: int,
) -> tuple[dict[str, float], bool]:
    """Refit *names* (e.g. ``ua``, ``q0``) on the *window* slots, the rest held at *prior*.

    Returns ``(params, refitted)``; a window without a fit segment keeps the prior."""
    if not names:
        return dict(prior), False
    y = d.t_air
    segments, y_init = _segments(d, w, y, window)
    if not segments:
        return dict(prior), False
    scored = np.where(window & fit.valid_air(d), y, np.nan)
    res_fn = _Residuals(w, y_init, scored, segments, n_diurnal)
    names = list(names)
    lo = np.array([_box(n)[0] for n in names])
    hi = np.array([_box(n)[1] for n in names])

    def resid(th):
        return res_fn({**prior, **dict(zip(names, th, strict=True))})

    x0 = np.clip([prior[n] for n in names], lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))
    r = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=100)
    return {**prior, **{n: float(v) for n, v in zip(names, r.x, strict=True)}}, True


def fit_heat_law(
    d: ClimateData, states: np.ndarray, cap: plant.ValveCap, train: np.ndarray, t_ref: float
) -> tuple[HeatLaw, dict[str, Any]]:
    """C2's response B on this model's observer slab states (permitted, non-DHW training slots)."""
    cap_meas = cap.measured(
        fit.comfort_matrix(d, d.room_temp), fit.comfort_matrix(d, d.room_setpoint)
    )
    t_ft = d.columns["flow_target_temp_mean_c"]
    sel = (
        fit._response_selection(d, train, cap_meas) & np.isfinite(t_ft) & np.isfinite(states[:, 1])
    )
    if sel.sum() < 50:
        raise ValueError("too few permitted slots to fit the heat law")
    response, stats = plant.fit_response_b(
        d.q_heat_kw[sel], cap_meas[sel], d.t_out_model[sel], t_ft[sel], states[sel, 1], t_ref
    )
    return HeatLaw.from_response(response), stats


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def make_model(
    name: str,
    params: Mapping[str, float],
    n_diurnal: int,
    law: HeatLaw,
    cap: plant.ValveCap,
    rooms: Sequence[str],
    *,
    gain: np.ndarray | None = None,
    process_var: float | None = None,
    slab_offset_k: float = 0.0,
) -> RcComfortModel:
    """An :class:`RcComfortModel` from parameters (observer gain from *process_var* if no
    *gain*)."""
    ad, bd = rc_matrices(dict(params), n_diurnal)
    if gain is None:
        meas = thermal.MEASUREMENT_SIGMA_K**2
        gain = thermal.observer_gain(ad, process_var if process_var else 1e-3, meas)
    return RcComfortModel(
        name=name,
        ad=ad,
        bd=bd,
        heat_capacities=np.array([params["c_a"], params["c_s"]], dtype=float),
        law=law,
        cap=cap,
        rooms=tuple(rooms),
        gain=np.asarray(gain, float),
        slab_offset_k=float(slab_offset_k),
        params=tuple(sorted((k, float(v)) for k, v in params.items())),
    )


@dataclass
class SeasonModel:
    """The model schedule of one replayed season: ``(first slot, model)`` pieces, ascending, and
    the causal observer states of the whole grid (``NaN`` outside the pieces)."""

    season: str
    prior_season: str
    pieces: list[tuple[int, RcComfortModel]]
    states: np.ndarray
    age: np.ndarray
    info: dict[str, Any]

    def at(self, t: int) -> RcComfortModel:
        model = self.pieces[0][1]
        for lo, m in self.pieces:
            if lo <= t:
                model = m
        return model


def process_var(d: ClimateData, ad, bd, w, train) -> float:
    meas = thermal.MEASUREMENT_SIGMA_K**2
    y = np.where(train & fit.valid_air(d), d.t_air, np.nan)
    return thermal.process_variance(ad, bd, np.where(train[:, None], w, np.nan), y, meas)


def from_c2_record(
    record: Mapping[str, Any] | str | Path, rooms: Sequence[str]
) -> tuple[RcComfortModel, dict[str, Any]]:
    """C2's model B (static S2 prior) from ``climate-models/<id>/model.json``."""
    if not isinstance(record, Mapping):
        record = json.loads(Path(record).read_text())
    fold = record["folds"][0]
    b = fold["models"]["B"]
    params = {k: float(v) for k, v in b["thermal"]["params"].items() if k != "b_ret"}
    r = b["response"]
    law = HeatLaw(float(r["h_e"]), float(r["qm0"]), float(r["qm1"]), float(r["t_ref"]))
    offsets = np.array(fold["room_offsets_k"], dtype=float)
    cap = plant.ValveCap(float(record.get("phi") or 0.3), 0.25, offsets)
    model = make_model(
        C2_NAME,
        params,
        0,
        law,
        cap,
        rooms,
        gain=np.array(b["observer"]["gain"], dtype=float),
        slab_offset_k=float(b["thermal"].get("slab_offset_k", 0.0)),
    )
    info = {
        "model_id": record.get("model_id"),
        "verdict": record.get("verdict"),
        "train": fold.get("train"),
    }
    return model, info


def cap_for(d: ClimateData, cfg: ClimateConfig, train: np.ndarray, phi: float) -> plant.ValveCap:
    """Room offsets and the valve cap of the training slots (as ``climate-fit`` builds them)."""
    offsets = room_offsets(d.room_temp, d.comfort_rooms, d.t_air, d.local_hour, train & d.train)
    return plant.ValveCap(phi, cfg.model.response.plant_valve_cap_margin_k, offsets)
