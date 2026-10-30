"""Thermal models of the house (C2): the first-order model A and the slab + air model B.

Units: temperature in °C, power in kW, capacity in kWh/K, time in hours.  One step is a 15-minute
slot (``STEP_H``).  The input of a slot is ``w = [Q, T_out, S, 1]``: the heat delivered (mean kW
over the slot), the outdoor temperature, the clear-sky solar proxy and a constant for the internal
gain ``q0``.

* **A**, first order: ``C dT/dt = Q + UA (T_out - T) + q0 + g_s S`` with ``C = tau * UA``.
* **B**, slab + air: ``C_a dT_a/dt = k (T_s - T_a) + UA (T_out - T_a) + q0 + g_s S`` and
  ``C_s dT_s/dt = Q - k (T_s - T_a)``.  The return temperature in steady circulation is a second,
  noisy slab measurement ``y2 = T_s + b_ret``; it is used only in the fit.

Both are discretised exactly under a zero-order hold (``scipy.linalg.expm`` of the augmented
matrix).  A measurement of slot ``t`` (a slot mean) is compared with the state at the end of the
slot, ``x[t + 1]``; the observer and the forecasts use the same convention.

The fit is bounded simulation-error least squares (``scipy.optimize.least_squares``) over
segments: maximal runs of slots with known inputs inside 48 h windows from local midnight, each
started from the measured air temperature of the slot before it.  B's initial slab temperature is
a nuisance parameter per segment, eliminated in closed form (variable projection: the state is
linear in it).  A parameter on a bound of the search box, or a derived one outside the envelope of
``config/climate.yaml``, makes the fit a failure, and so does a parameter that the data cannot pin
down (relative standard error above ``IDENT_MAX_REL_SE`` from the Jacobian at the optimum).

With a *prior* (a parameter set fitted elsewhere) and a *shrinkage* strength ``lambda``, the fit
adds ``lambda * sum(((p - p_prior) / width)^2)`` to the cost, ``width`` being each parameter's
search-box width, and starts from the prior: the causal weekly refits of ``climate-fit`` stay close
to the training-season fit unless their window's data say otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.linalg import expm, solve_discrete_are
from scipy.optimize import least_squares
from scipy.signal import lfilter

__all__ = [
    "KINDS",
    "STEP_H",
    "FitInputs",
    "ThermalFit",
    "bounds_for",
    "continuous",
    "derived",
    "discretize",
    "fit_thermal",
    "input_matrix",
    "observe",
    "observer_gain",
    "param_names",
    "segments_in_windows",
    "simulate",
]

STEP_H = 0.25
KINDS = ("A", "B")

#: Search box of the parameters the envelope does not bound.  ``g_s`` at its lower bound means
#: "no solar gain detectable" and is not a failure; the others are (see ``FAIL_ON_BOUND``).
G_S_BOUNDS = (0.0, 10.0)  # kW per unit of the clear-sky proxy
C_A_BOUNDS = (0.05, 60.0)  # kWh/K, air node of B
B_RET_BOUNDS = (-15.0, 15.0)  # K, return temperature minus slab
FAIL_ON_BOUND = frozenset({"ua", "tau", "q0", "c_a", "c_s", "k", "b_ret"})
#: Parameters whose relative standard error decides identifiability.
IDENT_PARAMS = frozenset({"ua", "tau", "c_a", "c_s", "k"})
IDENT_MAX_REL_SE = 0.5
#: Initial residual scales; B with a return measurement re-weights from the first pass.
AIR_SIGMA_K = 0.1
RETURN_SIGMA_K = 0.5
#: Thermostat measurement noise for the observer: the 0.1 K reading steps (about 0.03 K) and the
#: ~30 min update hold; the lag variogram of the readings shows little white noise beyond that.
MEASUREMENT_SIGMA_K = 0.05
MIN_SEGMENT_SLOTS = 32
BOUND_REL_TOL = 1e-3

_STARTS = {
    "A": [{"ua": 0.15, "tau": 60.0, "q0": 0.5, "g_s": 0.5}],
    "B": [
        {"ua": 0.15, "c_a": 2.0, "c_s": 8.0, "k": 1.0, "q0": 0.5, "g_s": 0.5, "b_ret": 1.0},
        {"ua": 0.2, "c_a": 6.0, "c_s": 15.0, "k": 0.5, "q0": 0.5, "g_s": 0.5, "b_ret": 0.0},
        {"ua": 0.1, "c_a": 0.5, "c_s": 4.0, "k": 2.0, "q0": 0.5, "g_s": 0.5, "b_ret": 2.0},
    ],
}


def param_names(kind: str, with_return: bool = True) -> tuple[str, ...]:
    if kind == "A":
        return ("ua", "tau", "q0", "g_s")
    if kind == "B":
        base = ("ua", "c_a", "c_s", "k", "q0", "g_s")
        return (*base, "b_ret") if with_return else base
    raise ValueError(f"unknown thermal model {kind!r}")


def bounds_for(kind: str, envelope: Mapping[str, Sequence[float]], with_return: bool = True):
    """Search box per parameter, from the envelope where it has one."""
    box = {
        "ua": tuple(envelope["ua_kw_per_k"]),
        "tau": tuple(envelope["tau_h"]),
        "q0": tuple(envelope["q0_kw"]),
        "g_s": G_S_BOUNDS,
        "c_a": C_A_BOUNDS,
        "c_s": tuple(envelope["c_slab_kwh_per_k"]),
        "k": tuple(envelope["k_kw_per_k"]),
        "b_ret": B_RET_BOUNDS,
    }
    return {name: box[name] for name in param_names(kind, with_return)}


def derived(kind: str, p: Mapping[str, float], comfort_c: float = 22.0) -> dict[str, float]:
    """Physical reading: total capacity, time constant and balance temperature (no sun)."""
    if kind == "A":
        c = p["tau"] * p["ua"]
        tau = p["tau"]
    else:
        c = p["c_a"] + p["c_s"]
        tau = c / p["ua"]
    out = {"c_kwh_per_k": c, "tau_h": tau, "balance_temp_c": comfort_c - p["q0"] / p["ua"]}
    if kind == "B":
        out["slab_tau_h"] = p["c_s"] / p["k"]
    return out


def continuous(kind: str, p: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """``(A_c, B_c)`` of ``dx/dt = A_c x + B_c w`` with ``w = [Q, T_out, S, 1]``."""
    ua, q0, g_s = p["ua"], p["q0"], p["g_s"]
    if kind == "A":
        c = p["tau"] * ua
        return np.array([[-ua / c]]), np.array([[1.0 / c, ua / c, g_s / c, q0 / c]])
    ca, cs, k = p["c_a"], p["c_s"], p["k"]
    ac = np.array([[-(k + ua) / ca, k / ca], [k / cs, -k / cs]])
    bc = np.array([[0.0, ua / ca, g_s / ca, q0 / ca], [1.0 / cs, 0.0, 0.0, 0.0]])
    return ac, bc


def discretize(ac: np.ndarray, bc: np.ndarray, step_h: float = STEP_H):
    """Exact zero-order-hold discretisation: ``x[t+1] = A_d x[t] + B_d w[t]``."""
    n, m = bc.shape
    aug = np.zeros((n + m, n + m))
    aug[:n, :n] = ac
    aug[:n, n:] = bc
    e = expm(aug * step_h)
    return e[:n, :n], e[:n, n:]


def input_matrix(q_kw: np.ndarray, t_out: np.ndarray, solar: np.ndarray) -> np.ndarray:
    q = np.asarray(q_kw, dtype=float)
    return np.column_stack([q, np.asarray(t_out, float), np.asarray(solar, float), np.ones(q.size)])


def _modes(ad: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Real eigen-decomposition of ``A_d`` (an RC network has real, distinct poles)."""
    lam, vec = np.linalg.eig(ad)
    if np.abs(lam.imag).max() > 1e-12 or (lam.size > 1 and abs(lam[0] - lam[-1]) < 1e-12):
        return None
    vec = vec.real
    return lam.real, vec, np.linalg.inv(vec)


def simulate(ad: np.ndarray, drive: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """States ``x[0..L]`` of ``x[t+1] = A_d x[t] + drive[t]`` (``drive = w @ B_d.T``)."""
    drive = np.asarray(drive, dtype=float)
    x0 = np.asarray(x0, dtype=float)
    length, n = drive.shape
    out = np.empty((length + 1, n))
    out[0] = x0
    modes = _modes(ad)
    if modes is None:  # pragma: no cover - not reachable for the RC models
        x = x0.copy()
        for t in range(length):
            x = ad @ x + drive[t]
            out[t + 1] = x
        return out
    lam, vec, inv = modes
    z0 = inv @ x0
    dz = drive @ inv.T
    z = np.empty((length, n))
    for i in range(n):
        z[:, i], _ = lfilter([1.0], [1.0, -lam[i]], dz[:, i], zi=[lam[i] * z0[i]])
    out[1:] = z @ vec.T
    return out


# ---------------------------------------------------------------------------
# Segments
# ---------------------------------------------------------------------------


def segments_in_windows(
    window_id: np.ndarray, usable: np.ndarray, y_air: np.ndarray, min_slots: int = MIN_SEGMENT_SLOTS
) -> list[tuple[int, int]]:
    """Maximal runs of *usable* slots inside one window (``window_id >= 0``), each starting where
    the slot before it has a measured air temperature (the initial state); shorter than
    *min_slots* is dropped."""
    wid = np.asarray(window_id)
    ok = np.asarray(usable, dtype=bool) & (wid >= 0)
    out: list[tuple[int, int]] = []
    n = ok.size
    t = 0
    while t < n:
        if not ok[t]:
            t += 1
            continue
        end = t
        while end < n and ok[end] and wid[end] == wid[t]:
            end += 1
        lo = t
        while lo < end and not (lo > 0 and np.isfinite(y_air[lo - 1])):
            lo += 1
        if end - lo >= min_slots:
            out.append((lo, end))
        t = end
    return out


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------


@dataclass
class FitInputs:
    """Inputs of a simulation-error fit on the whole grid.

    ``y_air`` is the measured air temperature where it is scored (``NaN`` elsewhere, e.g. slots a
    training filter drops); ``y_ret`` the return temperature in steady circulation (``NaN``
    elsewhere), or ``None``.  ``y_init`` is the air temperature used as initial state."""

    w: np.ndarray
    y_air: np.ndarray
    y_init: np.ndarray
    segments: list[tuple[int, int]]
    y_ret: np.ndarray | None = None

    @property
    def has_return(self) -> bool:
        if self.y_ret is None:
            return False
        return any(np.isfinite(self.y_ret[lo:hi]).any() for lo, hi in self.segments)


@dataclass
class ThermalFit:
    kind: str
    params: dict[str, float]
    bounds: dict[str, tuple[float, float]]
    on_bound: list[str]
    outside_envelope: list[str]
    rel_se: dict[str, float]
    identifiable: bool
    converged: bool
    rmse_air_k: float
    rmse_return_k: float | None
    n_air: int
    n_return: int
    n_segments: int
    slab_offset_k: float
    sigma: tuple[float, float]
    nfev: int
    message: str
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return (
            self.converged and not self.on_bound and not self.outside_envelope and self.identifiable
        )

    def failures(self) -> list[str]:
        out = [f"on bound: {name}" for name in self.on_bound]
        out += [f"outside envelope: {name}" for name in self.outside_envelope]
        if not self.identifiable:
            out.append("not identifiable")
        if not self.converged:
            out.append(f"not converged: {self.message}")
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "params": self.params,
            "derived": derived(self.kind, self.params),
            "bounds": {k: list(v) for k, v in self.bounds.items()},
            "on_bound": self.on_bound,
            "outside_envelope": self.outside_envelope,
            "rel_se": self.rel_se,
            "identifiable": self.identifiable,
            "converged": self.converged,
            "ok": self.ok,
            "failures": self.failures(),
            "rmse_air_k": self.rmse_air_k,
            "rmse_return_k": self.rmse_return_k,
            "n_air": self.n_air,
            "n_return": self.n_return,
            "n_segments": self.n_segments,
            "slab_offset_k": self.slab_offset_k,
            "sigma": list(self.sigma),
            "nfev": self.nfev,
            "message": self.message,
        }


class _Problem:
    """Residuals of one fit; B's initial slab temperatures are projected out per segment."""

    def __init__(
        self, kind: str, names: tuple[str, ...], fi: FitInputs, sigma: tuple[float, float]
    ):
        self.kind, self.names, self.fi = kind, names, fi
        self.sig_a, self.sig_r = sigma
        self.use_ret = "b_ret" in names
        self.max_len = max(hi - lo for lo, hi in fi.segments)
        self.idx = []
        for lo, hi in fi.segments:
            a = np.flatnonzero(np.isfinite(fi.y_air[lo:hi]))
            r = (
                np.flatnonzero(np.isfinite(fi.y_ret[lo:hi]))
                if self.use_ret and fi.y_ret is not None
                else np.zeros(0, dtype=np.int64)
            )
            self.idx.append((a, r))

    def params(self, theta: np.ndarray) -> dict[str, float]:
        p = dict(zip(self.names, (float(v) for v in theta), strict=True))
        if self.kind == "B" and not self.use_ret:
            p["b_ret"] = 0.0
        return p

    def run(self, theta: np.ndarray) -> tuple[np.ndarray, list[float]]:
        p = self.params(theta)
        ad, bd = discretize(*continuous(self.kind, p))
        drive = self.fi.w @ bd.T
        b_ret = p.get("b_ret", 0.0)
        homog = None
        if self.kind == "B":  # response of the state to a unit initial slab temperature
            homog = simulate(ad, np.zeros((self.max_len, 2)), np.array([0.0, 1.0]))[1:]
        res: list[np.ndarray] = []
        ts0: list[float] = []
        for (lo, hi), (ia, ir) in zip(self.fi.segments, self.idx, strict=True):
            y0 = self.fi.y_init[lo - 1]
            x0 = np.array([y0]) if self.kind == "A" else np.array([y0, 0.0])
            xs = simulate(ad, drive[lo:hi], x0)[1:]
            ra = (xs[ia, 0] - self.fi.y_air[lo:hi][ia]) / self.sig_a
            if self.kind == "A":
                res.append(ra)
                continue
            ga = homog[ia, 0] / self.sig_a
            parts_r, parts_g = [ra], [ga]
            if ir.size:
                rr = (xs[ir, 1] + b_ret - self.fi.y_ret[lo:hi][ir]) / self.sig_r
                parts_r.append(rr)
                parts_g.append(homog[ir, 1] / self.sig_r)
            r, g = np.concatenate(parts_r), np.concatenate(parts_g)
            gg = float(g @ g)
            s = -float(r @ g) / gg if gg > 1e-12 else y0
            res.append(r + g * s)
            ts0.append(s - y0)
        return np.concatenate(res), ts0

    def split(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Unweighted air and return residuals (K) at *theta*."""
        p = self.params(theta)
        ad, bd = discretize(*continuous(self.kind, p))
        drive = self.fi.w @ bd.T
        _r, ts0 = self.run(theta)
        air, ret = [], []
        for j, ((lo, hi), (ia, ir)) in enumerate(zip(self.fi.segments, self.idx, strict=True)):
            y0 = self.fi.y_init[lo - 1]
            x0 = np.array([y0]) if self.kind == "A" else np.array([y0, y0 + ts0[j]])
            xs = simulate(ad, drive[lo:hi], x0)[1:]
            air.append(xs[ia, 0] - self.fi.y_air[lo:hi][ia])
            if self.kind == "B" and ir.size:
                ret.append(xs[ir, 1] + p.get("b_ret", 0.0) - self.fi.y_ret[lo:hi][ir])
        return np.concatenate(air), (np.concatenate(ret) if ret else np.zeros(0))


def _on_bound(theta, lo, hi, active) -> list[bool]:
    span = hi - lo
    return [
        bool(a != 0) or (v - low) <= BOUND_REL_TOL * s or (high - v) <= BOUND_REL_TOL * s
        for v, low, high, s, a in zip(theta, lo, hi, span, active, strict=True)
    ]


def _rel_se(jac: np.ndarray, theta: np.ndarray, cost: float) -> np.ndarray:
    m, p = jac.shape
    js = jac * np.abs(theta)[None, :]
    s2 = 2.0 * cost / max(m - p, 1)
    _u, sv, vt = np.linalg.svd(js, full_matrices=False)
    keep = sv > sv[0] * 1e-8
    cov = (vt[keep].T / sv[keep] ** 2) @ vt[keep] * s2
    out = np.sqrt(np.maximum(np.diag(cov), 0.0))
    # a direction the residuals do not see at all: every parameter in it is undetermined
    if (~keep).any():
        out[np.abs(vt[~keep]).max(axis=0) > 0.1] = np.inf
    return out


def fit_thermal(
    kind: str,
    fi: FitInputs,
    envelope: Mapping[str, Sequence[float]],
    *,
    starts: Sequence[Mapping[str, float]] | None = None,
    prior: Mapping[str, float] | None = None,
    shrinkage: float = 0.0,
) -> ThermalFit:
    """Fit model *kind* on *fi*; the best of the start points wins.

    With a return measurement the fit runs twice: the second pass re-weights the air and return
    residuals by their first-pass RMS and restarts from the first-pass optimum.  A *prior* is the
    single start point (unless *starts* is given) and, with *shrinkage* > 0, the centre of the
    penalty described in the module docstring; the reported RMSE is the data part only."""
    if not fi.segments:
        raise ValueError("no fit segment: not enough slots with known inputs")
    with_return = kind == "B" and fi.has_return
    names = param_names(kind, with_return)
    box = bounds_for(kind, envelope, with_return)
    lo = np.array([box[n][0] for n in names])
    hi = np.array([box[n][1] for n in names])
    sigma = (AIR_SIGMA_K, RETURN_SIGMA_K)
    passes = 2 if with_return else 1
    if starts is None and prior is not None:
        starts = [{n: prior.get(n, _STARTS[kind][0][n]) for n in names}]
    points = list(starts or _STARTS[kind])
    penalised = [i for i, n in enumerate(names) if prior is not None and n in prior]
    centre = np.array([prior[names[i]] for i in penalised]) if penalised else np.zeros(0)
    scale = np.sqrt(max(shrinkage, 0.0)) / (hi - lo)[penalised]

    def penalty(th: np.ndarray) -> np.ndarray:
        return scale * (th[penalised] - centre)

    best = None
    for _ in range(passes):
        problem = _Problem(kind, names, fi, sigma)
        best = None
        for start in points:
            x0 = np.clip([start[n] for n in names], lo + 1e-6 * (hi - lo), hi - 1e-6 * (hi - lo))
            res = least_squares(
                lambda th: np.concatenate([problem.run(th)[0], penalty(th)]),
                x0,
                bounds=(lo, hi),
                x_scale="jac",
                method="trf",
                max_nfev=400,
            )
            if best is None or res.cost < best.cost:
                best = res
        points = [dict(zip(names, best.x, strict=True))]
        air, ret = problem.split(best.x)
        sigma = (
            max(float(np.sqrt(np.mean(air**2))), 1e-3),
            max(float(np.sqrt(np.mean(ret**2))), 1e-3) if ret.size else RETURN_SIGMA_K,
        )
    assert best is not None
    theta = best.x
    params = problem.params(theta)
    on = _on_bound(theta, lo, hi, best.active_mask)
    on_bound = [n for n, flag in zip(names, on, strict=True) if flag and n in FAIL_ON_BOUND]
    tau = derived(kind, params)["tau_h"]
    outside = []
    if kind == "B" and not envelope["tau_h"][0] <= tau <= envelope["tau_h"][1]:
        outside.append("tau")
    rel = _rel_se(best.jac, theta, best.cost)
    rel_se = {n: float(v) for n, v in zip(names, rel, strict=True)}
    ident_names = [n for n in names if n in IDENT_PARAMS]
    identifiable = all(
        np.isfinite(rel_se[n]) and rel_se[n] <= IDENT_MAX_REL_SE for n in ident_names
    )
    air, ret = problem.split(theta)
    _r, ts0 = problem.run(theta)
    return ThermalFit(
        kind=kind,
        params=params,
        bounds={n: (float(box[n][0]), float(box[n][1])) for n in names},
        on_bound=on_bound,
        outside_envelope=outside,
        rel_se=rel_se,
        identifiable=identifiable,
        converged=bool(best.status > 0),
        rmse_air_k=float(np.sqrt(np.mean(air**2))),
        rmse_return_k=float(np.sqrt(np.mean(ret**2))) if ret.size else None,
        n_air=int(air.size),
        n_return=int(ret.size),
        n_segments=len(fi.segments),
        slab_offset_k=float(np.median(ts0)) if ts0 else 0.0,
        sigma=sigma,
        nfev=int(best.nfev),
        message=str(best.message),
        extra={"shrinkage": float(shrinkage)} if prior is not None else {},
    )


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------


def observer_gain(ad: np.ndarray, process_var: float, meas_var: float) -> np.ndarray:
    """Steady-state Kalman gain for an air-temperature measurement of ``x[t + 1]``."""
    n = ad.shape[0]
    h = np.zeros((1, n))
    h[0, 0] = 1.0
    w = np.eye(n) * process_var
    p = solve_discrete_are(ad.T, h.T, w, np.array([[meas_var]]))
    return (p @ h.T / (h @ p @ h.T + meas_var)).ravel()


def process_variance(
    ad: np.ndarray, bd: np.ndarray, w: np.ndarray, y: np.ndarray, meas_var: float
) -> float:
    """One-step air-temperature process noise of a model, from consecutive measurements.

    The air row is predicted from the measured air temperature (and, for B, the same value for
    the slab, which only affects the coupling term); the innovation variance minus twice the
    measurement variance is the process part (floored)."""
    prev, cur = y[:-1], y[1:]
    ok = np.isfinite(prev) & np.isfinite(cur) & np.isfinite(w[1:]).all(axis=1)
    if ok.sum() < 10:
        return 0.01
    x_prev = np.repeat(prev[ok][:, None], ad.shape[0], axis=1)
    pred = x_prev @ ad[0] + w[1:][ok] @ bd[0]
    e = cur[ok] - pred
    return float(max(np.var(e) - 2.0 * meas_var, 1e-4))


def observe(
    ad: np.ndarray,
    bd: np.ndarray,
    w: np.ndarray,
    y: np.ndarray,
    gain: np.ndarray,
    slab_offset: float = 0.0,
    slab: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Filtered state at the start of every slot and the slots since the filter (re)started.

    ``x[t]`` uses the measurements of slots ``< t``.  A slot whose heat is known but whose
    outdoor temperature is not (a compressor run longer than the de-biasing cap) is bridged with
    the measured air temperature: it becomes the air state and, for B, drives the slab equation,
    which needs only ``Q`` and ``T_a`` (*slab* = ``(exp(-k dt / C_s), k)``).  Any other unknown
    input, or no measurement to bridge with, stops the filter; it restarts from the next measured
    air temperature (B's slab at ``slab_offset`` above it).  ``NaN`` and age ``-1`` where there
    is no estimate."""
    n_slots, nx = w.shape[0], ad.shape[0]
    xs = np.full((n_slots, nx), np.nan)
    age = np.full(n_slots, -1, dtype=np.int64)
    known = np.isfinite(w).all(axis=1)
    q_known = np.isfinite(w[:, 0]) & np.isfinite(w[:, 3])
    drive = np.where(known[:, None], w, 0.0) @ bd.T
    x: np.ndarray | None = None
    a = 0
    for t in range(n_slots):
        if x is None and t > 0 and np.isfinite(y[t - 1]):
            x = np.array([y[t - 1]] if nx == 1 else [y[t - 1], y[t - 1] + slab_offset])
            a = 0
        if x is None:
            continue
        xs[t] = x
        age[t] = a
        if known[t]:
            x = ad @ x + drive[t]
            if np.isfinite(y[t]):
                x = x + gain * (y[t] - x[0])
        elif q_known[t] and np.isfinite(y[t]) and (nx == 1 or slab is not None):
            if nx == 1:
                x = np.array([y[t]])
            else:
                decay, k = slab
                target = y[t] + w[t, 0] / k
                x = np.array([y[t], decay * x[1] + (1.0 - decay) * target])
        else:
            x = None
            continue
        a += 1
    return xs, age
