"""The model interface of the comfort MPC, and its linear RC implementation.

The optimiser (:mod:`battery_optimizer.climate.mpc`) never reads a thermal-model parameter.  It
reads only what :class:`MpcModel` gives it:

* a linear state equation on the 15-minute grid, ``x[t+1] = A x[t] + b_q Q[t] + drive[t]``, where
  ``Q`` is the space heat (mean kW over the slot) and ``drive`` the exogenous part (outdoor
  temperature, sun, internal and diurnal gains), :meth:`MpcModel.drive`;
* room outputs, ``T_room[r, t] = c . x[t] + offset[r, t]`` (:meth:`MpcModel.room_outputs`), compared
  with the comfort minimums at the end of each slot (the convention of ``thermal.py``);
* the heat delivered at each candidate water temperature along a reference trajectory,
  :meth:`MpcModel.heat_table` (the plant law may be nonlinear in the state; the optimiser
  re-linearises along its own simulated plan);
* a nonlinear closed-loop simulation of a chosen plan, :meth:`MpcModel.simulate`, which is what the
  planned comfort is reported from;
* the heat capacity of each state (kWh/K), for the terminal value of stored heat.

A model with another structure (more states, a per-room model, a valve model, another heat law)
plugs in by implementing these six members; the optimiser does not change.

:class:`RcComfortModel` is the implementation for the RC models of this package: model B of C2
(:func:`battery_optimizer.climate.b2d.from_c2_record`) and B0's room-air two-state model with a
diurnal gain (:mod:`battery_optimizer.climate.b2d`).  Its heat law is C2's response B,
``Q = valve(T_air) * clip(H_e (T_water - T_slab), 0, qm0 + qm1 (T_ref - T_out))``, with the valve
cap of :class:`battery_optimizer.climate.plant.ValveCap` evaluated against the setpoints the plan
assumes.  The state is ``[T_air, T_slab]``; ``T_air`` is the mean of the comfort rooms and a room
reads ``T_air`` plus its median offset by local hour.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from battery_optimizer.climate import plant, thermal

__all__ = [
    "Exogenous",
    "HeatLaw",
    "MpcModel",
    "RcComfortModel",
    "curve_water",
    "water_levels",
]


@dataclass(frozen=True)
class Exogenous:
    """Per-slot exogenous inputs over a horizon (arrays of length ``T``, ``diurnal`` ``(T, m)``).

    ``setpoints`` is ``(rooms, T)``: the thermostat setpoints the valve cap reads (the plan's own
    schedule setpoints in the MPC, the measured ones when a legacy action is replayed)."""

    t_out: np.ndarray
    solar: np.ndarray
    diurnal: np.ndarray
    local_hour: np.ndarray
    setpoints: np.ndarray

    @property
    def length(self) -> int:
        return int(self.t_out.size)


@runtime_checkable
class MpcModel(Protocol):
    """What the comfort MPC needs from a thermal and plant model (see the module docstring)."""

    name: str
    rooms: tuple[str, ...]

    @property
    def n_states(self) -> int: ...

    @property
    def capacities(self) -> np.ndarray: ...

    def dynamics(self) -> tuple[np.ndarray, np.ndarray]:
        """``(A, b_q)``: the state matrix and the response of the state to 1 kW of heat."""
        ...

    def drive(self, ex: Exogenous) -> np.ndarray:
        """``(T, n)``: the exogenous part of the state equation, slot by slot."""
        ...

    def room_outputs(self, local_hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(C, offsets)``: ``C`` is ``(rooms, n)``, ``offsets`` ``(rooms, T)``."""
        ...

    def heat_table(self, x_ref: np.ndarray, ex: Exogenous, water: np.ndarray) -> np.ndarray:
        """``(T, K)`` kW delivered when heating at each candidate water temperature ``(T, K)``,
        with the state of slot ``t`` taken as ``x_ref[t]`` (``(T, n)``)."""
        ...

    def simulate(
        self, x0: np.ndarray, on: np.ndarray, water: np.ndarray, ex: Exogenous
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(xs, q)``: states ``(T + 1, n)`` and heat ``(T,)`` kW of a plan (``on`` 0/1, ``water``
        degC per slot), with the plant law closed on the simulated state."""
        ...


# ---------------------------------------------------------------------------
# Water temperature
# ---------------------------------------------------------------------------


def curve_water(t_out: np.ndarray, curve: Sequence[tuple[float, float]]) -> np.ndarray:
    """The software heating curve: linear between the points, clamped outside them."""
    xs = np.array([p[0] for p in curve], dtype=float)
    ys = np.array([p[1] for p in curve], dtype=float)
    return np.interp(np.asarray(t_out, dtype=float), xs, ys)


def water_levels(
    t_out: np.ndarray,
    curve: Sequence[tuple[float, float]],
    offsets: Sequence[int],
    lo: int,
    hi: int,
) -> tuple[np.ndarray, np.ndarray]:
    """``(water, distinct)``, both ``(T, K)``: the integer water target of each offset
    (``clamp(round(curve + offset), lo, hi)``, halves rounded up) and whether the level is the
    first occurrence of its value in the slot (clamping makes duplicates; only the first is a
    choice, so the MILP has no symmetric twins)."""
    base = curve_water(t_out, curve)
    off = np.asarray(offsets, dtype=float)
    water = np.clip(np.floor(base[:, None] + off[None, :] + 0.5), lo, hi)
    distinct = np.ones(water.shape, dtype=bool)
    for k in range(1, water.shape[1]):
        distinct[:, k] = ~(water[:, :k] == water[:, k : k + 1]).any(axis=1)
    return water, distinct


# ---------------------------------------------------------------------------
# RC implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HeatLaw:
    """C2's response B: ``clip(H_e (T_water - T_slab), 0, qm0 + qm1 (T_ref - T_out))`` kW."""

    h_e: float
    qm0: float
    qm1: float
    t_ref: float

    @classmethod
    def from_response(cls, r: plant.ResponseB) -> HeatLaw:
        return cls(float(r.h_e), float(r.qm0), float(r.qm1), float(r.t_ref))

    def power(self, t_out, water, t_slab) -> np.ndarray:
        q_max = np.maximum(0.0, self.qm0 + self.qm1 * (self.t_ref - np.asarray(t_out, float)))
        return np.clip(self.h_e * (np.asarray(water, float) - t_slab), 0.0, q_max)

    def to_dict(self) -> dict[str, float]:
        return {"h_e": self.h_e, "qm0": self.qm0, "qm1": self.qm1, "t_ref": self.t_ref}


@dataclass(frozen=True)
class RcComfortModel:
    """A discrete RC model ``x[t+1] = A x[t] + B w[t]`` with ``w = [Q, T_out, S, 1, D...]`` and
    the heat law of :class:`HeatLaw` (state ``[T_air, T_slab]``)."""

    name: str
    ad: np.ndarray
    bd: np.ndarray
    heat_capacities: np.ndarray
    law: HeatLaw
    cap: plant.ValveCap
    rooms: tuple[str, ...]
    gain: np.ndarray
    slab_offset_k: float = 0.0
    params: tuple[tuple[str, float], ...] = ()

    @property
    def n_states(self) -> int:
        return int(self.ad.shape[0])

    @property
    def n_diurnal(self) -> int:
        return int(self.bd.shape[1] - 4)

    @property
    def capacities(self) -> np.ndarray:
        return self.heat_capacities

    def dynamics(self) -> tuple[np.ndarray, np.ndarray]:
        return self.ad, self.bd[:, 0]

    def inputs(self, q: np.ndarray, ex: Exogenous) -> np.ndarray:
        """``w`` of the slots: heat, outdoor, sun, constant and the diurnal columns."""
        n = ex.length
        cols = [np.asarray(q, float), ex.t_out, ex.solar, np.ones(n)]
        d = np.asarray(ex.diurnal, float).reshape(n, -1)[:, : self.n_diurnal]
        return np.column_stack([*cols, d]) if self.n_diurnal else np.column_stack(cols)

    def drive(self, ex: Exogenous) -> np.ndarray:
        return self.inputs(np.zeros(ex.length), ex) @ self.bd.T

    def room_outputs(self, local_hour: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = np.zeros((len(self.rooms), self.n_states))
        c[:, 0] = 1.0
        hours = np.asarray(local_hour, dtype=np.int64)
        return c, np.nan_to_num(self.cap.offsets[:, hours])

    def _valve(self, t_air: np.ndarray, ex: Exogenous) -> np.ndarray:
        temps = t_air[None, :] + np.nan_to_num(self.cap.offsets[:, ex.local_hour.astype(np.int64)])
        open_share = (temps < ex.setpoints - self.cap.margin_k).mean(axis=0)
        return self.cap.phi + (1.0 - self.cap.phi) * open_share

    def heat_table(self, x_ref: np.ndarray, ex: Exogenous, water: np.ndarray) -> np.ndarray:
        x_ref = np.asarray(x_ref, float)
        base = self.law.power(ex.t_out[:, None], water, x_ref[:, 1][:, None])
        return base * self._valve(x_ref[:, 0], ex)[:, None]

    def simulate(
        self, x0: np.ndarray, on: np.ndarray, water: np.ndarray, ex: Exogenous
    ) -> tuple[np.ndarray, np.ndarray]:
        n = ex.length
        drive = self.drive(ex)
        bq = self.bd[:, 0]
        xs = np.empty((n + 1, self.n_states))
        q = np.zeros(n)
        x = np.asarray(x0, dtype=float).copy()
        xs[0] = x
        for t in range(n):
            if on[t] > 0.5:
                base = float(self.law.power(ex.t_out[t], water[t], x[1]))
                valve = self._valve(np.array([x[0]]), _slot(ex, t))[0]
                q[t] = base * valve
            x = self.ad @ x + bq * q[t] + drive[t]
            xs[t + 1] = x
        return xs, q

    def observe(self, w: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Causal observer states ``x[t]`` (slots ``< t`` only) and ages (``thermal.observe``)."""
        return thermal.observe(
            self.ad, self.bd, w, y, self.gain, self.slab_offset_k, self._slab_terms()
        )

    def _slab_terms(self) -> tuple[float, float]:
        p = dict(self.params)
        return float(np.exp(-p["k"] * thermal.STEP_H / p["c_s"])), float(p["k"])

    def describe(self) -> dict[str, object]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "heat_law": self.law.to_dict(),
            "valve_phi": self.cap.phi,
            "valve_margin_k": self.cap.margin_k,
            "capacities_kwh_per_k": [float(c) for c in self.heat_capacities],
            "rooms": list(self.rooms),
        }


def _slot(ex: Exogenous, t: int) -> Exogenous:
    return Exogenous(
        t_out=ex.t_out[t : t + 1],
        solar=ex.solar[t : t + 1],
        diurnal=ex.diurnal[t : t + 1],
        local_hour=ex.local_hour[t : t + 1],
        setpoints=ex.setpoints[:, t : t + 1],
    )


def rc_matrices(params: dict[str, float], n_diurnal: int) -> tuple[np.ndarray, np.ndarray]:
    """Discrete ``(A, B)`` of the air + slab model with optional diurnal gains ``g0..``.

    ``C_a dT_a = k (T_s - T_a) + UA (T_out - T_a) + q0 + g_s S + sum g_i D_i`` and
    ``C_s dT_s = Q - k (T_s - T_a)`` (C2's model B; with ``n_diurnal = 0`` exactly
    ``thermal.continuous("B", ...)``)."""
    ca, cs, k, ua = params["c_a"], params["c_s"], params["k"], params["ua"]
    ac = np.array([[-(k + ua) / ca, k / ca], [k / cs, -k / cs]])
    bc = np.zeros((2, 4 + n_diurnal))
    bc[0, 1:4] = [ua / ca, params["g_s"] / ca, params["q0"] / ca]
    bc[0, 4:] = np.array([params[f"g{i}"] for i in range(n_diurnal)]) / ca
    bc[1, 0] = 1.0 / cs
    return thermal.discretize(ac, bc)
