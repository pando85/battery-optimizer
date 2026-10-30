"""Heat response of the plant to the permission ``u`` (C2), the valve cap and the starts model.

Within a permitted slot the pump's own control sets the heat, so the delivered heat is modelled
as a response to ``u`` (kW, mean over the slot):

* **A:** ``Q = u (1 - dhw) max(0, c0 + c1 (T_ref - T_out) + c2 first_hour_of_block)``.
* **B:** ``Q = u (1 - dhw) clip(H_e (T_flow_target - T_s), 0, Q_max(T_out))`` with
  ``Q_max = qm0 + qm1 (T_ref - T_out)``.  The site has no weather-compensated heating curve (the
  flow target is a constant setting per season), so the measured flow target is the input.

Both are capped by the room valves: ``Q x (phi + (1 - phi) open)``, where ``open`` is the share of
comfort rooms whose temperature (``T_air`` plus the room's offset for the hour) is below its
setpoint minus ``model.response.plant_valve_cap_margin_k`` and ``phi`` the always-open share.

The starts model is ``starts/day = alpha blocks + beta hours_on + gamma hours_on T_out`` (no
intercept, as the plan writes it), with ``blocks`` the permission blocks of the day, ``hours_on``
the hours with heat delivered and ``T_out`` the day's mean outdoor temperature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

__all__ = [
    "HEAT_ON_KWH",
    "PlantInputs",
    "ResponseA",
    "ResponseB",
    "StartsModel",
    "ValveCap",
    "first_hour_of_block",
    "fit_response_a",
    "fit_response_b",
    "fit_starts",
    "permission_blocks",
    "simulate_u",
]

#: A slot delivers heat at or above this thermal energy (kWh), as in the audit.
HEAT_ON_KWH = 0.05
SLOTS_PER_HOUR = 4


def first_hour_of_block(u: np.ndarray) -> np.ndarray:
    """Slots in the first hour of a run of ``u == 1`` (runs continue across midnight)."""
    on = np.asarray(u) == 1.0
    out = np.zeros(on.size, dtype=bool)
    run = 0
    for t in range(on.size):
        run = run + 1 if on[t] else 0
        out[t] = 0 < run <= SLOTS_PER_HOUR
    return out


def permission_blocks(u_day: np.ndarray) -> int:
    """Runs of ``u == 1`` within one day."""
    on = (np.asarray(u_day) == 1.0).astype(np.int8)
    return int((np.diff(np.concatenate(([0], on))) == 1).sum())


@dataclass(frozen=True)
class ValveCap:
    """``phi + (1 - phi) open``; *offsets* is ``room_offsets`` of the fold's training slots."""

    phi: float
    margin_k: float
    offsets: np.ndarray  # (comfort rooms, 24); NaN hours count as offset 0

    def factor(self, t_air: float, hour: int, setpoints: np.ndarray) -> float:
        temps = t_air + np.nan_to_num(self.offsets[:, hour])
        open_share = float(np.mean(temps < setpoints - self.margin_k))
        return self.phi + (1.0 - self.phi) * open_share

    def measured(self, room_temps: np.ndarray, setpoints: np.ndarray) -> np.ndarray:
        """The factor from measured room temperatures (``NaN`` where a room is unknown)."""
        with np.errstate(invalid="ignore"):
            open_share = (room_temps < setpoints - self.margin_k).mean(axis=0)
        known = np.isfinite(room_temps).all(axis=0) & np.isfinite(setpoints).all(axis=0)
        return np.where(known, self.phi + (1.0 - self.phi) * open_share, np.nan)


@dataclass(frozen=True)
class ResponseA:
    c0: float
    c1: float
    c2: float
    t_ref: float

    def power(self, t_out: np.ndarray | float, first: np.ndarray | bool) -> np.ndarray | float:
        """Heat (kW) of a permitted, non-DHW slot before the valve cap."""
        return np.maximum(0.0, self.c0 + self.c1 * (self.t_ref - t_out) + self.c2 * first)

    def to_dict(self) -> dict[str, float]:
        return {"c0": self.c0, "c1": self.c1, "c2": self.c2, "t_ref": self.t_ref}


@dataclass(frozen=True)
class ResponseB:
    h_e: float
    qm0: float
    qm1: float
    t_ref: float

    def q_max(self, t_out: np.ndarray | float) -> np.ndarray | float:
        return np.maximum(0.0, self.qm0 + self.qm1 * (self.t_ref - t_out))

    def power(self, t_out, t_flow_target, t_slab):
        """Heat (kW) of a permitted, non-DHW slot before the valve cap."""
        return np.clip(self.h_e * (t_flow_target - t_slab), 0.0, self.q_max(t_out))

    def to_dict(self) -> dict[str, float]:
        return {"h_e": self.h_e, "qm0": self.qm0, "qm1": self.qm1, "t_ref": self.t_ref}


def _fit_stats(q: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    return {
        "slots": int(q.size),
        "rmse_kw": float(np.sqrt(np.mean((pred - q) ** 2))) if q.size else None,
        "bias": float(pred.sum() / q.sum() - 1.0) if q.size and q.sum() > 0 else None,
    }


def fit_response_a(
    q_kw: np.ndarray, cap: np.ndarray, t_out: np.ndarray, first: np.ndarray, t_ref: float
) -> tuple[ResponseA, dict[str, Any]]:
    """Least squares of ``Q = cap (c0 + c1 (T_ref - T_out) + c2 first)`` on permitted slots."""
    x = np.column_stack([cap, cap * (t_ref - t_out), cap * first.astype(float)])
    coef, *_ = np.linalg.lstsq(x, q_kw, rcond=None)
    model = ResponseA(float(coef[0]), float(coef[1]), float(coef[2]), float(t_ref))
    return model, _fit_stats(q_kw, cap * model.power(t_out, first))


def fit_response_b(
    q_kw: np.ndarray,
    cap: np.ndarray,
    t_out: np.ndarray,
    t_flow_target: np.ndarray,
    t_slab: np.ndarray,
    t_ref: float,
) -> tuple[ResponseB, dict[str, Any]]:
    """Bounded least squares of ``Q = cap clip(H_e (T_ft - T_s), 0, Q_max(T_out))``."""

    def resid(th: np.ndarray) -> np.ndarray:
        m = ResponseB(th[0], th[1], th[2], t_ref)
        return cap * m.power(t_out, t_flow_target, t_slab) - q_kw

    res = least_squares(
        resid,
        np.array([0.5, 4.0, 0.0]),
        bounds=([1e-3, 0.0, -2.0], [50.0, 30.0, 2.0]),
        x_scale="jac",
    )
    model = ResponseB(float(res.x[0]), float(res.x[1]), float(res.x[2]), float(t_ref))
    stats = _fit_stats(q_kw, cap * model.power(t_out, t_flow_target, t_slab))
    stats["at_search_bound"] = bool((res.active_mask != 0).any())
    return model, stats


@dataclass
class PlantInputs:
    """Per-slot inputs of a u-driven simulation on the whole grid."""

    u: np.ndarray
    dhw: np.ndarray
    t_out: np.ndarray
    solar: np.ndarray
    t_flow_target: np.ndarray
    first_hour: np.ndarray
    local_hour: np.ndarray
    setpoints: np.ndarray  # (comfort rooms, slots)

    def known(self, kind: str) -> np.ndarray:
        ok = (
            np.isfinite(self.u)
            & np.isfinite(self.t_out)
            & np.isfinite(self.solar)
            & np.isfinite(self.setpoints).all(axis=0)
        )
        if kind == "B":  # the flow target matters only when heat can be delivered
            needs = (self.u == 1.0) & ~self.dhw
            ok &= ~needs | np.isfinite(self.t_flow_target)
        return ok


def simulate_u(
    kind: str,
    ad: np.ndarray,
    bd: np.ndarray,
    x0: np.ndarray,
    lo: int,
    hi: int,
    inp: PlantInputs,
    response: ResponseA | ResponseB,
    cap: ValveCap,
) -> tuple[np.ndarray, np.ndarray]:
    """States ``x[lo..hi]`` and heat (kW) of slots ``lo..hi-1`` driven by ``u`` through the
    response and the valve cap (closed loop: the cap and B's response read the state)."""
    length = hi - lo
    xs = np.empty((length + 1, ad.shape[0]))
    q = np.zeros(length)
    x = np.asarray(x0, dtype=float).copy()
    xs[0] = x
    for j in range(length):
        t = lo + j
        if inp.u[t] == 1.0 and not inp.dhw[t]:
            if kind == "A":
                base = float(response.power(inp.t_out[t], inp.first_hour[t]))
            else:
                base = float(response.power(inp.t_out[t], inp.t_flow_target[t], x[1]))
            q[j] = base * cap.factor(x[0], int(inp.local_hour[t]), inp.setpoints[:, t])
        x = ad @ x + bd @ np.array([q[j], inp.t_out[t], inp.solar[t], 1.0])
        xs[j + 1] = x
    return xs, q


# ---------------------------------------------------------------------------
# Starts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StartsModel:
    alpha: float
    beta: float
    gamma: float

    def predict(self, blocks, hours_on, t_out):
        return self.alpha * blocks + self.beta * hours_on + self.gamma * hours_on * t_out

    def to_dict(self) -> dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}


def fit_starts(
    blocks: np.ndarray, hours_on: np.ndarray, t_out: np.ndarray, starts: np.ndarray
) -> StartsModel:
    x = np.column_stack([blocks, hours_on, hours_on * t_out]).astype(float)
    coef, *_ = np.linalg.lstsq(x, np.asarray(starts, dtype=float), rcond=None)
    return StartsModel(float(coef[0]), float(coef[1]), float(coef[2]))
