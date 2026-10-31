"""The comfort MILP (``climate/mpc.py``): feasibility, machine limits, soft comfort, DHW, levels."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from battery_optimizer.climate.cop import COP_MODEL_VERSION, CopModel
from battery_optimizer.climate.mpc import MpcProblem, PlanSettings, plan, solve
from battery_optimizer.climate.mpc_model import Exogenous, MpcModel, water_levels

T = 24
A = 0.98  # 1-state house, per slot
B_Q = 0.06  # K per kW-slot
T_EQ = 18.0  # temperature the house drifts to with no heat


def _problem(**kw) -> MpcProblem:
    n_levels = kw.pop("levels", 2)
    heat = kw.pop("heat", np.full((T, n_levels), 3.0))
    elec = kw.pop("elec", np.tile(np.arange(1, n_levels + 1, dtype=float) * 0.2, (T, 1)))
    minimum = kw.pop("minimum", np.full((1, T), np.nan))
    base = MpcProblem(
        a=np.array([[A]]),
        b_q=np.array([B_Q]),
        drive=np.full((T, 1), (1 - A) * T_EQ),
        x0=np.array([kw.pop("x0", 21.0)]),
        out_c=np.array([[1.0]]),
        out_offset=np.zeros((1, T)),
        minimum=minimum,
        heat=heat,
        elec=elec,
        allowed=np.ones(heat.shape, dtype=bool),
        dhw=np.zeros(T, dtype=bool),
        capacities=np.array([10.0]),
        terminal_value=0.0,
        min_on_slots=2,
        min_off_slots=2,
        start_penalty_kwh=0.0,
        penalty_kwh_per_kh=20.0,
    )
    return replace(base, **kw)


def _simulate(p: MpcProblem, on: np.ndarray, level: np.ndarray) -> np.ndarray:
    x = float(p.x0[0])
    out = []
    for t in range(T):
        q = p.heat[t, level[t]] if on[t] else 0.0
        x = A * x + B_Q * q + p.drive[t, 0]
        out.append(x)
    return np.array(out)


def _runs(on: np.ndarray, value: int) -> list[tuple[int, int]]:
    """Runs of *value* as ``(start, length)``."""
    out, t = [], 0
    while t < on.size:
        if on[t] == value:
            s = t
            while t < on.size and on[t] == value:
                t += 1
            out.append((s, t - s))
        else:
            t += 1
    return out


def test_a_warm_house_with_no_binding_minimum_does_not_heat():
    res = solve(_problem(x0=23.0, minimum=np.full((1, T), 19.0)))
    assert res.status == "optimal"
    assert res.on.sum() == 0
    assert res.objective == pytest.approx(0.0, abs=1e-9)


def test_the_minimum_is_met_when_heat_is_available_and_the_cheaper_level_wins():
    minimum = np.full((1, T), np.nan)
    minimum[0, 8:] = 21.0
    p = _problem(x0=20.0, minimum=minimum, penalty_kwh_per_kh=1000.0)
    res = solve(p)
    assert res.ok and res.on.sum() > 0
    temps = _simulate(p, res.on, np.maximum(res.level, 0))
    assert (temps[8:] >= 21.0 - 1e-6).all()
    assert set(res.level[res.on > 0]) == {0}  # same heat, lower electricity
    assert res.extra["milp_kh_below"] == pytest.approx(0.0, abs=1e-6)


def test_minimum_on_and_off_times_hold():
    minimum = np.full((1, T), 21.0)
    p = _problem(x0=21.0, minimum=minimum, heat=np.full((T, 2), 1.0))
    res = solve(p)
    on = res.on.astype(int)
    assert on.sum() > 0
    for s, n in _runs(on, 1):
        if s + n < T:  # a run cut by the horizon end may be shorter
            assert n >= 2
    for s, n in _runs(on, 0):
        if s > 0 and s + n < T:  # a gap between two runs
            assert n >= 2


def test_the_run_in_progress_is_carried_in():
    # just started: must stay on for the rest of the minimum on time, though nothing needs heat
    res = solve(_problem(x0=23.0, prev_on=True, prev_run_slots=1))
    assert res.on[0] == 1 and res.on[1:].sum() == 0
    # just stopped: must stay off for the rest of the minimum off time, though the house is cold
    res = solve(_problem(x0=19.0, minimum=np.full((1, T), 21.0), prev_on=False, prev_run_slots=1))
    assert res.on[0] == 0 and res.on[1] == 1


def test_an_unreachable_minimum_stays_feasible_through_the_soft_penalty():
    res = solve(_problem(x0=19.0, minimum=np.full((1, T), 30.0)))
    assert res.status == "optimal"
    assert res.on.sum() == T  # heats every slot it can
    assert res.extra["milp_kh_below"] > 0


def test_dhw_slots_carry_no_space_heat_and_count_as_compressor_on():
    dhw = np.zeros(T, dtype=bool)
    dhw[4:8] = True
    res = solve(_problem(x0=19.0, minimum=np.full((1, T), 30.0), dhw=dhw))
    assert res.on[4:8].sum() == 0
    assert res.on[:4].sum() == 4 and res.on[8:].sum() == T - 8
    # an off slot right before DHW would violate nothing: DHW is exogenous; a start before it is
    # one start, not two (the compressor runs through the hot-water block)
    p = _problem(x0=19.0, minimum=np.full((1, T), 30.0), dhw=dhw, start_penalty_kwh=1.0)
    assert solve(p).objective == pytest.approx(solve(replace(p, start_penalty_kwh=0)).objective + 1)


def test_the_start_penalty_merges_runs():
    minimum = np.full((1, T), np.nan)
    minimum[0, [6, 18]] = 21.3
    p = _problem(x0=21.0, minimum=minimum, heat=np.full((T, 1), 2.0), elec=np.full((T, 1), 0.1))
    free = solve(p)
    costly = solve(replace(p, start_penalty_kwh=5.0))
    starts = lambda r: len(_runs(r.on.astype(int), 1))  # noqa: E731
    assert starts(costly) <= starts(free)
    assert starts(costly) == 1


def test_the_terminal_value_keeps_the_stored_heat():
    p = _problem(x0=22.0)
    assert solve(p).on.sum() == 0  # without it the plan coasts
    kept = solve(replace(p, terminal_value=0.5))
    assert kept.on.sum() > 0
    assert kept.extra["terminal_shortfall_kwh_th"] < 2.0


def test_disallowed_levels_are_never_chosen():
    allowed = np.ones((T, 2), dtype=bool)
    allowed[:, 0] = False
    res = solve(_problem(x0=19.0, minimum=np.full((1, T), 21.0), allowed=allowed))
    assert (res.level[res.on > 0] == 1).all()


def test_water_levels_are_integers_clamped_and_deduplicated():
    curve = ((-5.0, 32.0), (15.0, 25.0))
    water, distinct = water_levels(np.array([-5.0, 15.0, 5.0]), curve, (0, 1, 2, 3), 26, 33)
    assert water[0].tolist() == [32, 33, 33, 33]
    assert distinct[0].tolist() == [True, True, False, False]
    assert water[1].tolist() == [26, 26, 27, 28]  # 25 is below the clamp
    assert distinct[1].tolist() == [True, False, True, True]
    assert water[2].tolist() == [29, 30, 31, 32]  # 28.5 rounds up
    assert (water == np.round(water)).all()
    full, _ = water_levels(np.linspace(-30, 40, 50), curve, (0, 1, 2, 3), 20, 35)
    assert full.min() >= 20 and full.max() <= 35


# ---------------------------------------------------------------------------
# plan(): the loop with any model implementing the interface
# ---------------------------------------------------------------------------


class OneStateModel:
    """A 1-state house with a water-dependent heat law, implementing :class:`MpcModel`."""

    name = "one-state"
    rooms = ("hall_kitchen",)
    n_states = 1
    capacities = np.array([10.0])

    def dynamics(self):
        return np.array([[A]]), np.array([B_Q])

    def drive(self, ex):
        return ((1 - A) * ex.t_out)[:, None]

    def room_outputs(self, local_hour):
        return np.array([[1.0]]), np.zeros((1, np.asarray(local_hour).size))

    def heat_table(self, x_ref, ex, water):
        return np.clip(0.5 * (water - x_ref[:, :1]), 0.0, 6.0)

    def simulate(self, x0, on, water, ex):
        x = float(x0[0])
        xs, q = [x], np.zeros(ex.length)
        for t in range(ex.length):
            if on[t] > 0.5:
                q[t] = float(np.clip(0.5 * (water[t] - x), 0.0, 6.0))
            x = A * x + B_Q * q[t] + (1 - A) * ex.t_out[t]
            xs.append(x)
        return np.array(xs)[:, None], q


def _cop() -> CopModel:
    return CopModel(
        version=COP_MODEL_VERSION,
        selected="lift_table",
        coefficients=(0.2, 0.01, 0.0, 0.05, 0.05),
        knee_c=5.0,
        table_bin_k=2.0,
        table=((0.0, 4.0, 100),),
        cop_range=(1.0, 9.0),
        train_seasons=("S2",),
        n_train_rows=100,
        feature_set_id="synthetic",
        config_sha256="synthetic",
        lift_bin_k=2.0,
        lift_table=tuple((float(lo), 6.0 - lo / 10.0, 100) for lo in range(10, 40, 2)),
    )


def _settings(**kw) -> PlanSettings:
    base = PlanSettings(
        curve=((-5.0, 32.0), (15.0, 25.0)),
        offsets=(0, 1, 2, 3),
        water_min=20,
        water_max=35,
        min_on_slots=2,
        min_off_slots=2,
        start_penalty_kwh=0.05,
        penalty_kwh_per_kh=20.0,
        terminal=True,
        time_limit_s=10.0,
        mip_rel_gap=1e-4,
        relinearize_iterations=1,
    )
    return replace(base, **kw)


def _exogenous(t_out: float = 5.0) -> Exogenous:
    return Exogenous(
        t_out=np.full(T, t_out),
        solar=np.zeros(T),
        diurnal=np.zeros((T, 0)),
        local_hour=np.arange(T) % 24,
        setpoints=np.full((1, T), 24.0),
    )


def test_any_model_implementing_the_interface_plugs_into_plan():
    model = OneStateModel()
    assert isinstance(model, MpcModel)
    minimum = np.full((1, T), np.nan)
    minimum[0, 12:] = 21.5
    ex = _exogenous(15.0)
    p = plan(model, np.array([21.0]), ex, minimum, np.zeros(T, bool), _cop(), _settings())
    assert p.result.ok and len(p.solves) == 2
    assert np.isfinite(p.water[p.on > 0]).all() and np.isnan(p.water[p.on == 0]).all()
    xs, q = model.simulate(np.array([21.0]), p.on, np.nan_to_num(p.water), ex)
    assert np.allclose(xs, p.xs) and np.allclose(q, p.q_kw)
    assert (p.xs[13:, 0] >= 21.5 - 0.05).all()
    assert p.elec_kwh[p.on == 0].sum() == 0
    assert p.starts >= 1


def test_a_failed_solve_returns_an_empty_plan_not_an_exception():
    model = OneStateModel()
    p = plan(
        model,
        np.array([21.0]),
        _exogenous(),
        np.full((1, T), 22.0),
        np.zeros(T, bool),
        _cop(),
        _settings(time_limit_s=1e-9),
    )
    assert p.result.status in {"time_limit", "error", "infeasible", "optimal", "gap"}
    if not p.result.ok:
        assert p.on.sum() == 0 and np.isnan(p.elec_kwh).all()
