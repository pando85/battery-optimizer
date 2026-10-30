"""Fused outdoor temperature: a station reference plus a Kalman-smoothed local offset.

The heat pump's own outdoor sensor is biased (cold while the compressor runs, warm in the sun) and
only trustworthy on part of the slots; a nearby station is unbiased but not at the house.  The two
are fused through their difference, the *offset*::

    o(t) = house(t) - r(t)         observed only on trusted dark slots
    o(t) = mu + x(t),  x(t+1) = phi x(t) + w,  w ~ N(0, q);   y = mu + x + v,  v ~ N(0, r)

``r(t)`` is the station temperature interpolated to the slot (see
:func:`battery_optimizer.climate.station.reference_at_slots`).  A trusted dark reading is one with
the compressor off long enough and no sun, so the offset stays the true microclimate difference
and not the sensor's bias.  The state is a slot-by-slot AR(1) (one step is one 15-minute slot);
``mu, phi, q, r`` are fitted by maximum likelihood on the Kalman filter's innovations, on training
data only.  The fused series is ``r + mu + x_hat`` with the smoother's per-slot standard
deviation, and falls back to the caller's debiased series where ``r`` is unknown.

* :func:`fit_offset_model`: the maximum-likelihood fit (training slots only).
* :func:`kalman_filter`: causal estimate, the mode C3 will use with a forecast as ``r``.
* :func:`rts_smoother`: two-sided estimate for offline history.
* :func:`fuse`: the fused series and its standard deviation from either.
* :func:`leave_block_out` and :func:`compressor_start_step`: the switch-rule test and the
  residual-step diagnostic.

Pure numpy and scipy; nothing here reads a file or knows a station.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "FilterResult",
    "OffsetFit",
    "OffsetParams",
    "SmootherResult",
    "compressor_start_step",
    "fit_offset_model",
    "fuse",
    "interpolate_trusted",
    "kalman_filter",
    "leave_block_out",
    "offset_observations",
    "rts_smoother",
]

_LOG_2PI = math.log(2.0 * math.pi)
_PHI_MAX = 1.0 - 1e-6
_LOG_BOUND = (-18.0, 6.0)


@dataclass(frozen=True)
class OffsetParams:
    """The offset model: ``mu`` (K), AR(1) coefficient ``phi`` per slot, state variance ``q``
    and observation variance ``r`` (K^2)."""

    mu: float
    phi: float
    q: float
    r: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.phi < 1.0 and self.q > 0.0 and self.r > 0.0):
            raise ValueError(f"invalid offset parameters {self}")

    @property
    def stationary_variance(self) -> float:
        return self.q / (1.0 - self.phi * self.phi)

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class OffsetFit:
    params: OffsetParams
    loglik: float
    n_obs: int
    converged: bool


@dataclass(frozen=True)
class FilterResult:
    """Causal estimate per slot: ``x`` and ``p`` use the observations up to and including the
    slot; ``x_pred`` and ``p_pred`` only those before it.  ``loglik`` is the innovation sum."""

    x: np.ndarray
    p: np.ndarray
    x_pred: np.ndarray
    p_pred: np.ndarray
    loglik: float


@dataclass(frozen=True)
class SmootherResult:
    x: np.ndarray
    p: np.ndarray
    filtered: FilterResult


def offset_observations(
    house: np.ndarray, reference: np.ndarray, observed: np.ndarray
) -> np.ndarray:
    """``house - reference`` where *observed* (and both are known), ``NaN`` elsewhere."""
    house = np.asarray(house, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    ok = np.asarray(observed, dtype=bool) & np.isfinite(house) & np.isfinite(reference)
    y = np.full(house.size, np.nan)
    y[ok] = house[ok] - reference[ok]
    return y


# ---------------------------------------------------------------------------
# Likelihood
# ---------------------------------------------------------------------------


def _loglik(
    idx: Sequence[int], y: Sequence[float], mu: float, phi: float, q: float, r: float
) -> float:
    """Innovation log-likelihood at the observed slots only (exact across the gaps between)."""
    phi2 = phi * phi
    p_inf = q / (1.0 - phi2)
    x = 0.0
    p = p_inf
    prev = None
    total = 0.0
    for i, obs in zip(idx, y, strict=True):
        if prev is None:
            xp, pp = 0.0, p_inf
        else:
            a = phi ** (i - prev)
            a2 = a * a
            xp = a * x
            pp = a2 * p + p_inf * (1.0 - a2)
        f = pp + r
        v = obs - mu - xp
        total -= 0.5 * (_LOG_2PI + math.log(f) + v * v / f)
        k = pp / f
        x = xp + k * v
        p = (1.0 - k) * pp
        prev = i
    return total


def _unpack(theta: np.ndarray) -> tuple[float, float, float, float]:
    mu, a, lq, lr = (float(t) for t in theta)
    phi = min(1.0 / (1.0 + math.exp(-a)), _PHI_MAX)
    return mu, phi, math.exp(lq), math.exp(lr)


def fit_offset_model(
    y: np.ndarray, *, start: OffsetParams | None = None, min_obs: int = 200
) -> OffsetFit:
    """Maximum-likelihood ``mu, phi, q, r`` from the offset observations *y* (``NaN`` = none).

    Pass only training slots: whatever is finite in *y* tunes the parameters.  The likelihood is
    the Kalman filter's innovation likelihood, maximised with L-BFGS-B over
    ``(mu, logit phi, log q, log r)`` from a few starting points.
    """
    y = np.asarray(y, dtype=np.float64)
    idx = np.flatnonzero(np.isfinite(y))
    if idx.size < min_obs:
        raise ValueError(f"need at least {min_obs} offset observations, got {idx.size}")
    obs = y[idx].tolist()
    ind = idx.tolist()
    var = float(np.var(y[idx]))
    mean = float(np.mean(y[idx]))

    def nll(theta: np.ndarray) -> float:
        mu, phi, q, r = _unpack(theta)
        return -_loglik(ind, obs, mu, phi, q, r)

    starts: list[np.ndarray] = []
    for phi0 in (0.9, 0.99, 0.999):
        for share in (0.3, 0.7):  # share of the variance that is observation noise
            r0 = max(var * share, 1e-4)
            q0 = max(var * (1.0 - share) * (1.0 - phi0 * phi0), 1e-8)
            starts.append(
                np.array([mean, math.log(phi0 / (1.0 - phi0)), math.log(q0), math.log(r0)])
            )
    if start is not None:
        starts.insert(
            0,
            np.array(
                [
                    start.mu,
                    math.log(start.phi / (1.0 - start.phi)),
                    math.log(start.q),
                    math.log(start.r),
                ]
            ),
        )
    bounds = [(-50.0, 50.0), (-6.0, 14.0), _LOG_BOUND, _LOG_BOUND]
    best = None
    for theta0 in starts:
        res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    mu, phi, q, r = _unpack(best.x)
    return OffsetFit(
        OffsetParams(mu, phi, q, r), float(-best.fun), int(idx.size), bool(best.success)
    )


# ---------------------------------------------------------------------------
# Filter and smoother
# ---------------------------------------------------------------------------


def kalman_filter(y: np.ndarray, params: OffsetParams) -> FilterResult:
    """Causal Kalman filter of the offset state ``x`` (``mu`` is removed from *y*).

    Every slot gets an estimate; a slot without an observation is a prediction whose variance
    grows back towards the stationary one.  This is the live mode: use only the past.
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    phi, q, r, mu = params.phi, params.q, params.r, params.mu
    phi2 = phi * phi
    x_f = np.empty(n)
    p_f = np.empty(n)
    x_p = np.empty(n)
    p_p = np.empty(n)
    x, p = 0.0, params.stationary_variance
    total = 0.0
    obs_list = y.tolist()
    for t in range(n):
        if t:
            x, p = phi * x, phi2 * p + q
        x_p[t], p_p[t] = x, p
        obs = obs_list[t]
        if obs == obs:  # not NaN
            f = p + r
            v = obs - mu - x
            total -= 0.5 * (_LOG_2PI + math.log(f) + v * v / f)
            k = p / f
            x += k * v
            p *= 1.0 - k
        x_f[t], p_f[t] = x, p
    return FilterResult(x_f, p_f, x_p, p_p, total)


def rts_smoother(y: np.ndarray, params: OffsetParams) -> SmootherResult:
    """Kalman filter plus the Rauch-Tung-Striebel backward pass: both sides of every gap."""
    f = kalman_filter(y, params)
    n = f.x.size
    x_s = f.x.copy()
    p_s = f.p.copy()
    phi = params.phi
    xs_next, ps_next = float(x_s[-1]), float(p_s[-1])
    xf, pf, xp, pp = f.x.tolist(), f.p.tolist(), f.x_pred.tolist(), f.p_pred.tolist()
    for t in range(n - 2, -1, -1):
        c = pf[t] * phi / pp[t + 1]
        xs_next = xf[t] + c * (xs_next - xp[t + 1])
        ps_next = pf[t] + c * c * (ps_next - pp[t + 1])
        x_s[t], p_s[t] = xs_next, ps_next
    return SmootherResult(x_s, p_s, f)


def fuse(
    reference: np.ndarray, y: np.ndarray, params: OffsetParams, *, mode: str = "smoother"
) -> tuple[np.ndarray, np.ndarray]:
    """``(t_out_fused, sd)``: ``reference + mu + x_hat`` and the state's standard deviation.

    Where *reference* is ``NaN`` the result is ``NaN`` (the caller falls back to its own
    series).  *mode* is ``"smoother"`` (offline) or ``"filter"`` (live, causal).
    """
    if mode == "smoother":
        est = rts_smoother(y, params)
        x, p = est.x, est.p
    elif mode == "filter":
        est_f = kalman_filter(y, params)
        x, p = est_f.x, est_f.p
    else:
        raise ValueError(f"mode must be 'smoother' or 'filter', not {mode!r}")
    reference = np.asarray(reference, dtype=np.float64)
    fused = reference + params.mu + x
    sd = np.sqrt(p)
    sd[~np.isfinite(reference)] = np.nan
    return fused, sd


# ---------------------------------------------------------------------------
# Switch-rule test and diagnostics
# ---------------------------------------------------------------------------


def interpolate_trusted(values: np.ndarray, trusted: np.ndarray, max_gap_slots: int) -> np.ndarray:
    """Linear interpolation in time between the *trusted* slots, as ``t_out_debiased`` does.

    A slot is filled when at most *max_gap_slots* slots lie between its trusted neighbours; a
    longer run and anything outside the first and last trusted slot is ``NaN``.  Trusted slots
    keep their value.
    """
    values = np.asarray(values, dtype=np.float64)
    trusted = np.asarray(trusted, dtype=bool) & np.isfinite(values)
    n = values.size
    out = np.full(n, np.nan)
    out[trusted] = values[trusted]
    idx = np.flatnonzero(trusted)
    if idx.size < 2:
        return out
    slots = np.arange(n)
    nxt = np.searchsorted(idx, slots)
    inside = (nxt > 0) & (nxt < idx.size) & ~trusted
    gap = np.zeros(n, dtype=np.int64)
    gap[inside] = idx[nxt[inside]] - idx[nxt[inside] - 1] - 1
    fill = inside & (gap <= max_gap_slots)
    out[fill] = np.interp(slots[fill], idx, values[idx])
    return out


def leave_block_out(
    *,
    house: np.ndarray,
    reference: np.ndarray,
    trusted: np.ndarray,
    observed: np.ndarray,
    params: OffsetParams,
    region: np.ndarray,
    block_slots: int,
    stride_slots: int,
    phase_slots: int,
    max_gap_slots: int,
    block_ok: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Hide blocks of trusted readings and predict them two ways.

    Blocks of *block_slots* consecutive slots start every *stride_slots* from *phase_slots*
    and count when every slot of the block is in *region*, trusted and (when given) in
    *block_ok*, for instance the dark slots, to test only night blocks.  Their readings are
    removed from both predictors: (a) the interpolation of the remaining trusted readings, as
    ``t_out_debiased`` does it, and (b) the fused series, with the hidden readings also removed
    from the smoother's input (the parameters stay as fitted).  Returns per hidden slot the
    ``slot`` index, ``truth`` (the hidden reading), ``debiased``, ``fused`` (``NaN`` where a
    predictor has no value), ``dark`` (the slot was a smoother observation) and ``block``.
    """
    house = np.asarray(house, dtype=np.float64)
    trusted = np.asarray(trusted, dtype=bool) & np.isfinite(house)
    region = np.asarray(region, dtype=bool)
    eligible = trusted if block_ok is None else trusted & np.asarray(block_ok, dtype=bool)
    n = house.size
    starts = [
        s
        for s in range(phase_slots, n - block_slots + 1, stride_slots)
        if region[s : s + block_slots].all() and eligible[s : s + block_slots].all()
    ]
    empty = {k: np.zeros(0) for k in ("slot", "truth", "debiased", "fused", "block")}
    if not starts:
        return empty | {"dark": np.zeros(0, dtype=bool)}
    hidden = np.zeros(n, dtype=bool)
    block = np.full(n, -1, dtype=np.int64)
    for b, s in enumerate(starts):
        hidden[s : s + block_slots] = True
        block[s : s + block_slots] = b
    debiased = interpolate_trusted(house, trusted & ~hidden, max_gap_slots)
    y = offset_observations(house, reference, np.asarray(observed, dtype=bool) & ~hidden)
    fused, _sd = fuse(reference, y, params)
    sel = np.flatnonzero(hidden)
    return {
        "slot": sel,
        "truth": house[sel],
        "debiased": debiased[sel],
        "fused": fused[sel],
        "dark": np.asarray(observed, dtype=bool)[sel],
        "block": block[sel],
    }


def compressor_start_step(
    series: np.ndarray,
    reference: np.ndarray,
    compressor_freq: np.ndarray,
    *,
    min_off_slots: int,
    window_slots: int = 2,
    lag_slots: int = 0,
) -> np.ndarray:
    """Step of *series* across each compressor start, net of the reference's own change.

    A start is a slot with the compressor running after at least *min_off_slots* slots off.
    The step is the mean of *series* over *window_slots* slots starting *lag_slots* after the
    start minus its mean over the same number just before, minus the same
    for *reference*, so weather moving during the start does not count; the raw sensor shows
    about -2 K, a series free of the compressor's bias about 0.  One value per usable start.
    """
    series = np.asarray(series, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    freq = np.asarray(compressor_freq, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        off = np.isfinite(freq) & (freq == 0.0)
        on = np.isfinite(freq) & (freq > 0.0)
    n = freq.size
    steps: list[float] = []
    w = window_slots
    for s in range(max(min_off_slots, w), n - w - lag_slots + 1):
        if not on[s] or not off[s - min_off_slots : s].all():
            continue
        before = slice(s - w, s)
        after = slice(s + lag_slots, s + lag_slots + w)
        a = series[after].mean() - series[before].mean()
        b = reference[after].mean() - reference[before].mean()
        if np.isfinite(a) and np.isfinite(b):
            steps.append(float(a - b))
    return np.array(steps)
