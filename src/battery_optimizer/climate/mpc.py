"""Price-blind comfort MPC (track B, step B1): a small MILP on the 15-minute grid.

**No price enters it.**  It minimises the predicted heat-pump electricity of space heating plus a
small penalty per compressor start, subject to the comfort schedule as soft constraints, the
compressor's minimum on and off times, the water clamp and exogenous hot-water (DHW) blocks.  It
plans; it never actuates anything (shadow phase, ``AGENTS.md``).

Formulation (``T`` slots, ``K`` water levels, ``n`` states, ``R`` rooms):

* ``z[t, k]`` binary: heat in slot ``t`` at water level ``k``.  The levels are the software curve
  plus a small offset set (``water.offsets_k``), rounded and clamped to the machine range, so the
  water decision is ``K`` (default 4) binaries per slot instead of 16 integer temperatures; the
  curve already carries the weather compensation and the offset is what the optimiser trades
  (more heat per slot against a lower COP).  ``on[t] = sum_k z[t, k] <= 1``; ``z = 0`` in DHW slots
  and for a level that duplicates a lower one after clamping.
* Heat and electricity of a level are parameters of the slot: ``Q[t, k]`` from the model's heat
  table along a reference trajectory and ``E[t, k]`` from the COP model at that water temperature
  and the slot's outdoor temperature.  The plan is solved, simulated with the nonlinear plant law,
  the table re-evaluated along that simulated plan and solved again
  (``solver.relinearize_iterations``).  This keeps the MILP linear and small: the state-dependence
  of the heat (slab temperature, valve cap) enters through the reference, not through big-M rows.
* State: ``x[t + 1] = A x[t] + b_q sum_k Q[t, k] z[t, k] + drive[t]``, ``x[0]`` the observer's
  estimate at the decision.
* Comfort (soft): ``c_r . x[t + 1] + offset[r, t] + s[t, r] >= min[r, t]``, ``s >= 0``, costing
  ``penalty * s * 0.25`` (kWh-equivalent per Kh).  Slots without a minimum have no row.
* Compressor: ``c[t] = on[t] + dhw[t]`` (hot water runs the compressor too); start ``u[t] >= c[t] -
  c[t-1]``, stop ``v[t] >= c[t-1] - c[t]`` (continuous in [0, 1], integral at the optimum); minimum
  on ``sum_{j < U} u[t - j] <= c[t]``, minimum off ``sum_{j < D} v[t - j] <= 1 - c[t]``, with the
  run already in progress at the decision carried in.
* Terminal (optional): heat content ``sum_i C_i (x_i[T] - x_i[0]) + s_T >= 0``, the shortfall
  ``s_T`` (kWh of heat) charged at the horizon's mean electricity per heat unit at offset 0.

Solved with ``scipy.optimize.milp`` (HiGHS) under a wall-clock limit and a relative gap, with the
same status handling as the battery oracle: an incumbent at the time limit is reported as such,
never as optimal, and an infeasible or empty result is counted, not hidden.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from battery_optimizer.climate import cop as cop_mod
from battery_optimizer.climate.mpc_model import Exogenous, MpcModel, water_levels

__all__ = [
    "MpcError",
    "MpcProblem",
    "MpcResult",
    "PlanSettings",
    "build_problem",
    "plan",
    "solve",
]

STEP_H = 0.25
_STATUS_OPTIMAL = 0
_STATUS_LIMIT = 1
_STATUS_INFEASIBLE = 2
X_BOUND = 200.0  # K: a loose box on the states (keeps HiGHS away from free columns)


class MpcError(RuntimeError):
    """The MPC inputs are inconsistent."""


@dataclass(frozen=True)
class PlanSettings:
    """The configuration of one plan, read from ``config/climate-mpc.yaml``."""

    curve: tuple[tuple[float, float], ...]
    offsets: tuple[int, ...]
    water_min: int
    water_max: int
    min_on_slots: int
    min_off_slots: int
    start_penalty_kwh: float
    penalty_kwh_per_kh: float
    terminal: bool
    time_limit_s: float
    mip_rel_gap: float
    relinearize_iterations: int

    @classmethod
    def from_config(cls, cfg) -> PlanSettings:
        m = cfg.machine
        return cls(
            curve=tuple((float(a), float(b)) for a, b in cfg.water.curve),
            offsets=tuple(int(o) for o in cfg.water.offsets_k),
            water_min=int(m.water_min_c),
            water_max=int(m.water_max_c),
            min_on_slots=-(-int(m.min_on_minutes) // 15),
            min_off_slots=-(-int(m.min_off_minutes) // 15),
            start_penalty_kwh=float(m.start_penalty_kwh),
            penalty_kwh_per_kh=float(cfg.comfort.penalty_kwh_per_kh),
            terminal=bool(cfg.terminal.value_stored_heat),
            time_limit_s=float(cfg.solver.time_limit_s),
            mip_rel_gap=float(cfg.solver.mip_rel_gap),
            relinearize_iterations=int(cfg.solver.relinearize_iterations),
        )


@dataclass
class MpcProblem:
    """Everything one MILP solve reads, as plain arrays (no model object)."""

    a: np.ndarray  # (n, n)
    b_q: np.ndarray  # (n,)
    drive: np.ndarray  # (T, n)
    x0: np.ndarray  # (n,)
    out_c: np.ndarray  # (R, n)
    out_offset: np.ndarray  # (R, T)
    minimum: np.ndarray  # (R, T), NaN = no constraint
    heat: np.ndarray  # (T, K) kW
    elec: np.ndarray  # (T, K) kWh per slot
    allowed: np.ndarray  # (T, K) bool
    dhw: np.ndarray  # (T,) bool
    capacities: np.ndarray  # (n,) kWh/K
    terminal_value: float  # kWh_el per kWh_th of stored-heat shortfall (0 = off)
    prev_on: bool = False
    prev_run_slots: int = 10**6  # slots the compressor has been in its current state
    min_on_slots: int = 2
    min_off_slots: int = 2
    start_penalty_kwh: float = 0.0
    penalty_kwh_per_kh: float = 20.0

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.heat.shape[0], self.heat.shape[1], self.a.shape[0], self.out_c.shape[0]


@dataclass
class MpcResult:
    """A solved plan (or the reason there is none) and its solver telemetry."""

    status: str  # optimal | gap | time_limit | infeasible | error
    on: np.ndarray
    level: np.ndarray  # index into the levels, -1 when off
    objective: float = float("nan")
    gap: float = float("nan")
    solve_s: float = 0.0
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status in ("optimal", "gap", "time_limit") and self.on.size > 0


def _forced(p: MpcProblem) -> tuple[np.ndarray, np.ndarray]:
    """Slots forced on / forced off by the run in progress at the decision."""
    t = p.heat.shape[0]
    force_on = np.zeros(t, dtype=bool)
    force_off = np.zeros(t, dtype=bool)
    if p.prev_on and p.prev_run_slots < p.min_on_slots:
        force_on[: p.min_on_slots - p.prev_run_slots] = True
    if not p.prev_on and p.prev_run_slots < p.min_off_slots:
        force_off[: p.min_off_slots - p.prev_run_slots] = True
    return force_on, force_off


def solve(p: MpcProblem, *, time_limit_s: float = 20.0, mip_rel_gap: float = 1e-3) -> MpcResult:
    """Solve one MILP (see the module docstring for the formulation)."""
    t_n, k_n, n, r_n = p.shape
    dhw = np.asarray(p.dhw, dtype=bool)
    allowed = np.asarray(p.allowed, dtype=bool) & ~dhw[:, None]
    has_min = np.isfinite(p.minimum)
    rows_r = np.argwhere(has_min.T)  # (t, r) pairs with a minimum
    force_on, force_off = _forced(p)
    # DHW runs the compressor: it already satisfies a minimum on time, and it overrides a minimum
    # off time (hot water is exogenous, the plan cannot move it)
    force_on &= ~dhw
    force_off &= ~dhw

    # variable layout
    iz = 0
    ix = iz + t_n * k_n  # x[1..T]
    iu = ix + t_n * n
    iv = iu + t_n
    is_ = iv + t_n
    it = is_ + rows_r.shape[0]
    nvar = it + 1

    def zi(t, k):
        return iz + t * k_n + k

    def xi(t, i):  # state at the END of slot t (x[t + 1])
        return ix + t * n + i

    cost = np.zeros(nvar)
    cost[iz:ix] = np.where(allowed, p.elec, 0.0).ravel()
    cost[iu:iv] = p.start_penalty_kwh
    cost[is_:it] = p.penalty_kwh_per_kh * STEP_H
    cost[it] = p.terminal_value

    lb = np.zeros(nvar)
    ub = np.ones(nvar)
    ub[iz:ix] = allowed.ravel().astype(float)
    lb[ix:iu] = -X_BOUND
    ub[ix:iu] = X_BOUND
    ub[is_:it] = np.inf
    ub[it] = np.inf if p.terminal_value > 0 else 0.0
    for t in np.flatnonzero(force_off):
        ub[zi(t, 0) : zi(t, 0) + k_n] = 0.0
    integrality = np.zeros(nvar)
    integrality[iz:ix] = 1

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    r = 0

    def add(entries, low, high):
        nonlocal r
        for c, v in entries:
            rows.append(r)
            cols.append(c)
            vals.append(v)
        lo.append(low)
        hi.append(high)
        r += 1

    heat = np.where(allowed, p.heat, 0.0)
    # dynamics: x[t+1] - A x[t] - b_q sum_k Q z = drive[t]  (x[0] is the constant x0)
    for t in range(t_n):
        for i in range(n):
            ent = [(xi(t, i), 1.0)]
            rhs = float(p.drive[t, i])
            if t == 0:
                rhs += float(p.a[i] @ p.x0)
            else:
                ent += [(xi(t - 1, j), -float(p.a[i, j])) for j in range(n) if p.a[i, j] != 0.0]
            if p.b_q[i] != 0.0:
                ent += [(zi(t, k), -float(p.b_q[i] * heat[t, k])) for k in range(k_n)]
            add(ent, rhs, rhs)
    # one level per slot, forced-on slots
    for t in range(t_n):
        low = 1.0 if force_on[t] else 0.0
        add([(zi(t, k), 1.0) for k in range(k_n)], low, 1.0)
    # comfort rows
    for j, (t, rr) in enumerate(rows_r):
        ent = [(xi(t, i), float(p.out_c[rr, i])) for i in range(n) if p.out_c[rr, i] != 0.0]
        ent.append((is_ + j, 1.0))
        add(ent, float(p.minimum[rr, t] - p.out_offset[rr, t]), np.inf)

    # starts and stops: c[t] = sum_k z[t, k] + dhw[t]
    def c_terms(t, sign):
        return [(zi(t, k), sign) for k in range(k_n)]

    prev = 1.0 if p.prev_on else 0.0
    for t in range(t_n):
        d_now = float(dhw[t])
        if t == 0:
            add([(iu, 1.0), *c_terms(0, -1.0)], d_now - prev, np.inf)
            add([(iv, 1.0), *c_terms(0, 1.0)], prev - d_now, np.inf)
        else:
            d_prev = float(dhw[t - 1])
            add([(iu + t, 1.0), *c_terms(t, -1.0), *c_terms(t - 1, 1.0)], d_now - d_prev, np.inf)
            add([(iv + t, 1.0), *c_terms(t - 1, -1.0), *c_terms(t, 1.0)], d_prev - d_now, np.inf)
    for t in range(t_n):
        if p.min_on_slots > 1:
            window = range(max(0, t - p.min_on_slots + 1), t + 1)
            add([*((iu + j, 1.0) for j in window), *c_terms(t, -1.0)], -np.inf, float(dhw[t]))
        if p.min_off_slots > 1 and not dhw[t]:
            window = range(max(0, t - p.min_off_slots + 1), t + 1)
            add([*((iv + j, 1.0) for j in window), *c_terms(t, 1.0)], -np.inf, 1.0 - float(dhw[t]))
    # terminal heat content
    if p.terminal_value > 0:
        ent = [(xi(t_n - 1, i), float(p.capacities[i])) for i in range(n)]
        ent.append((it, 1.0))
        add(ent, float(p.capacities @ p.x0), np.inf)

    a_mat = coo_matrix((vals, (rows, cols)), shape=(r, nvar)).tocsr()
    options = {"time_limit": float(time_limit_s), "mip_rel_gap": float(mip_rel_gap)}
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Unrecognized options", category=RuntimeWarning)
        res = milp(
            cost,
            constraints=LinearConstraint(a_mat, np.array(lo), np.array(hi)),
            integrality=integrality,
            bounds=Bounds(lb, ub),
            options=options,
        )
    elapsed = time.perf_counter() - started
    status = int(res.status)
    message = str(res.message)
    empty = np.zeros(0)
    if res.x is None:
        kind = "infeasible" if status == _STATUS_INFEASIBLE else "error"
        return MpcResult(kind, empty, empty.astype(int), solve_s=elapsed, message=message)
    z = np.round(res.x[iz:ix].reshape(t_n, k_n)).astype(int)
    on = z.sum(axis=1).astype(float)
    level = np.where(on > 0, z.argmax(axis=1), -1)
    gap = getattr(res, "mip_gap", None)
    gap = float(gap) if gap is not None and np.isfinite(gap) else float("nan")
    if status == _STATUS_OPTIMAL:
        kind = "optimal" if not (gap > mip_rel_gap * 1.0001) else "gap"
    elif status == _STATUS_LIMIT and "time limit" in message.lower():
        kind = "time_limit"
    else:
        kind = "gap"
    slack = res.x[is_:it]
    return MpcResult(
        kind,
        on,
        level,
        objective=float(res.fun),
        gap=gap,
        solve_s=elapsed,
        message=message,
        extra={
            "milp_kh_below": float(slack.sum() * STEP_H),
            "terminal_shortfall_kwh_th": float(res.x[it]),
            "n_binary": int(t_n * k_n),
            "n_rows": int(r),
        },
    )


# ---------------------------------------------------------------------------
# One plan: levels, heat table, solve, re-linearise
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    """One decision's plan: the chosen schedule and its nonlinear simulation."""

    result: MpcResult
    on: np.ndarray
    water: np.ndarray  # degC per slot, NaN when off
    offset_index: np.ndarray
    xs: np.ndarray  # (T + 1, n) simulated states of the plan
    q_kw: np.ndarray  # simulated heat
    elec_kwh: np.ndarray  # electricity of the simulated heat
    starts: int
    solves: list[MpcResult]
    lift_outside_table: float  # share of planned heating slots outside the COP lift table


def build_problem(
    model: MpcModel,
    x0: np.ndarray,
    ex: Exogenous,
    minimum: np.ndarray,
    dhw: np.ndarray,
    cop_model: cop_mod.CopModel,
    settings: PlanSettings,
    x_ref: np.ndarray,
    *,
    prev_on: bool = False,
    prev_run_slots: int = 10**6,
) -> tuple[MpcProblem, np.ndarray]:
    """The MILP of one decision with the heat table along *x_ref*; also returns the levels."""
    water, distinct = water_levels(
        ex.t_out, settings.curve, settings.offsets, settings.water_min, settings.water_max
    )
    heat = model.heat_table(x_ref, ex, water)
    elec = _electricity(cop_model, heat, water, ex.t_out)
    allowed = distinct & np.isfinite(heat) & np.isfinite(elec)
    a, b_q = model.dynamics()
    c, off = model.room_outputs(ex.local_hour)
    terminal_value = 0.0
    if settings.terminal:
        h0, e0 = heat[:, 0], elec[:, 0]
        ok = h0 > 0
        terminal_value = float(e0[ok].sum() / (h0[ok].sum() * STEP_H)) if ok.any() else 0.25
    problem = MpcProblem(
        a=a,
        b_q=b_q,
        drive=model.drive(ex),
        x0=np.asarray(x0, float),
        out_c=c,
        out_offset=off,
        minimum=minimum,
        heat=heat,
        elec=elec,
        allowed=allowed,
        dhw=np.asarray(dhw, bool),
        capacities=np.asarray(model.capacities, float),
        terminal_value=terminal_value,
        prev_on=prev_on,
        prev_run_slots=prev_run_slots,
        min_on_slots=settings.min_on_slots,
        min_off_slots=settings.min_off_slots,
        start_penalty_kwh=settings.start_penalty_kwh + _start_term(cop_model),
        penalty_kwh_per_kh=settings.penalty_kwh_per_kh,
    )
    return problem, water


def _start_term(cop_model: cop_mod.CopModel) -> float:
    """The regression's electricity per start, when it is the selected COP model."""
    return float(max(cop_model.coefficients[3], 0.0)) if cop_model.selected == "regression" else 0.0


def _electricity(cop_model, heat_kw, water, t_out) -> np.ndarray:
    """kWh per slot of heating at each level (hours-on term included for the regression)."""
    q = np.asarray(heat_kw, float) * STEP_H
    to = np.broadcast_to(np.asarray(t_out, float)[:, None], q.shape)
    return np.asarray(
        cop_mod.predict_electricity(cop_model, q, water, to, starts=0.0, hours_on=STEP_H)
    )


def plan(
    model: MpcModel,
    x0: np.ndarray,
    ex: Exogenous,
    minimum: np.ndarray,
    dhw: np.ndarray,
    cop_model: cop_mod.CopModel,
    settings: PlanSettings,
    *,
    prev_on: bool = False,
    prev_run_slots: int = 10**6,
) -> Plan:
    """Solve, simulate the plan with the nonlinear plant law, re-linearise and solve again."""
    t_n = ex.length
    x_ref = np.repeat(np.asarray(x0, float)[None, :], t_n, axis=0)
    solves: list[MpcResult] = []
    best: tuple[MpcResult, np.ndarray] | None = None
    for _ in range(settings.relinearize_iterations + 1):
        problem, water = build_problem(
            model,
            x0,
            ex,
            minimum,
            dhw,
            cop_model,
            settings,
            x_ref,
            prev_on=prev_on,
            prev_run_slots=prev_run_slots,
        )
        res = solve(problem, time_limit_s=settings.time_limit_s, mip_rel_gap=settings.mip_rel_gap)
        solves.append(res)
        if not res.ok:
            break
        best = (res, water)
        w_plan = np.where(res.level >= 0, water[np.arange(t_n), np.maximum(res.level, 0)], np.nan)
        xs, _q = model.simulate(x0, res.on, np.nan_to_num(w_plan), ex)
        x_ref = xs[:-1]
    if best is None:
        nan = np.full(t_n, np.nan)
        return Plan(
            result=solves[-1],
            on=np.zeros(t_n),
            water=nan,
            offset_index=np.full(t_n, -1),
            xs=np.full((t_n + 1, len(x0)), np.nan),
            q_kw=nan,
            elec_kwh=nan,
            starts=0,
            solves=solves,
            lift_outside_table=float("nan"),
        )
    res, water = best
    w_plan = np.where(res.level >= 0, water[np.arange(t_n), np.maximum(res.level, 0)], np.nan)
    xs, q = model.simulate(x0, res.on, np.nan_to_num(w_plan), ex)
    elec = np.where(
        res.on > 0,
        cop_mod.predict_electricity(
            cop_model, q * STEP_H, np.nan_to_num(w_plan), ex.t_out, starts=0.0, hours_on=STEP_H
        ),
        0.0,
    )
    c = res.on + np.asarray(dhw, float)
    prev = 1.0 if prev_on else 0.0
    starts = int((np.diff(np.concatenate(([prev], c))) > 0.5).sum())
    if cop_model.selected == "regression":
        elec = elec + _start_term(cop_model) * np.diff(np.concatenate(([prev], c))).clip(0)
    return Plan(
        result=res,
        on=res.on,
        water=w_plan,
        offset_index=res.level,
        xs=xs,
        q_kw=q,
        elec_kwh=elec,
        starts=starts,
        solves=solves,
        lift_outside_table=_lift_outside(cop_model, w_plan, ex.t_out, res.on),
    )


def _lift_outside(cop_model, water, t_out, on) -> float:
    if not cop_model.lift_table or not (on > 0).any():
        return 0.0
    lift = (water - t_out)[on > 0]
    lo = cop_model.lift_table[0][0]
    hi = cop_model.lift_table[-1][0] + cop_model.lift_bin_k
    return float(((lift < lo) | (lift >= hi)).mean())
