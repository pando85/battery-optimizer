"""The MPC model interface (``climate/mpc_model.py``) and B0's B2+D model (``climate/b2d.py``)."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
from climate_fakes import build, grid, make_table

from battery_optimizer.climate import b2d, fit, plant, thermal
from battery_optimizer.climate.mpc_model import (
    Exogenous,
    HeatLaw,
    MpcModel,
    RcComfortModel,
    curve_water,
    rc_matrices,
)

ROOMS = ("bedroom", "hall_kitchen", "office_t")
TRUE = {"ua": 0.1, "c_a": 2.0, "c_s": 12.0, "k": 2.0, "q0": 0.3, "g_s": 0.8}
DIURNAL = {"g0": 0.25, "g1": -0.15, "g2": 0.2, "g3": 0.1}  # one harmonic, weekday then weekend
LAW = HeatLaw(h_e=0.5, qm0=4.0, qm1=0.1, t_ref=22.0)


def _cap(offsets: np.ndarray | None = None, phi: float = 0.3) -> plant.ValveCap:
    return plant.ValveCap(phi, 0.25, np.zeros((3, 24)) if offsets is None else offsets)


def _model(params=None, n_diurnal=0, **kw) -> RcComfortModel:
    p = {**TRUE, **(DIURNAL if n_diurnal else {}), **(params or {})}
    return b2d.make_model("test", p, n_diurnal, LAW, kw.pop("cap", _cap()), ROOMS, **kw)


def _ex(n: int, t_out: float = 5.0, setpoint: float = 24.0, m: int = 0) -> Exogenous:
    return Exogenous(
        t_out=np.full(n, t_out),
        solar=np.zeros(n),
        diurnal=np.zeros((n, m)),
        local_hour=np.arange(n) // 4 % 24,
        setpoints=np.full((3, n), setpoint),
    )


def test_the_rc_model_implements_the_interface():
    assert isinstance(_model(), MpcModel)


def test_without_diurnal_terms_the_matrices_are_c2_model_b():
    ad, bd = rc_matrices(TRUE, 0)
    ad_b, bd_b = thermal.discretize(*thermal.continuous("B", {**TRUE, "b_ret": 0.0}))
    assert np.allclose(ad, ad_b) and np.allclose(bd, bd_b)


def test_heat_table_is_the_plant_law_along_the_reference_and_simulate_agrees():
    m = _model()
    x_ref = np.array([[21.0, 23.0], [21.0, 30.0]])
    water = np.array([[25.0, 35.0], [25.0, 35.0]])
    table = m.heat_table(x_ref, _ex(2), water)
    # every room below 24 - 0.25: the valve factor is 1
    assert table[0].tolist() == pytest.approx([1.0, LAW.power(5.0, 35.0, 23.0)])
    assert table[1, 0] == 0.0  # the slab is warmer than the water: no heat
    xs, q = m.simulate(x_ref[0], np.array([1.0]), np.array([35.0]), _ex(1))
    assert q[0] == pytest.approx(table[0, 1])
    assert xs.shape == (2, 2)
    # the cap: every room at its setpoint leaves only the always-open share
    shut = m.heat_table(x_ref[:1], _ex(1, setpoint=20.0), water[:1])
    assert shut[0, 1] == pytest.approx(0.3 * table[0, 1])


def test_the_run_logic_switches_heat_off_when_every_valve_is_shut():
    m = _model()
    x0 = np.array([21.0, 23.0])
    shut = _ex(1, setpoint=20.0)  # every room above its setpoint
    assert m.valves(x0[None, :], shut).tolist() == [0.0]
    assert m.valves(x0[None, :], _ex(1)).tolist() == [1.0]
    _xs, q = m.simulate(x0, np.array([1.0]), np.array([35.0]), shut)
    assert q[0] == 0.0
    _xs, q_legacy = m.simulate(x0, np.array([1.0]), np.array([35.0]), shut, run_logic=False)
    assert q_legacy[0] == pytest.approx(0.3 * LAW.power(5.0, 35.0, 23.0))
    assert m.valve_factor(np.array([0.0, 1.0])).tolist() == pytest.approx([0.3, 1.0])


def test_room_outputs_read_the_hourly_offsets():
    offsets = np.zeros((3, 24))
    offsets[1, 7] = 0.4
    m = _model(cap=_cap(offsets))
    c, off = m.room_outputs(np.array([6, 7]))
    assert c.tolist() == [[1.0, 0.0]] * 3
    assert off[1].tolist() == [0.0, 0.4]


def test_curve_is_linear_and_clamped_outside_its_points():
    curve = ((-5.0, 32.0), (15.0, 25.0))
    assert curve_water(np.array([-20.0, -5.0, 5.0, 15.0, 30.0]), curve).tolist() == [
        32.0,
        32.0,
        28.5,
        25.0,
        25.0,
    ]


def test_diurnal_matrix_separates_weekdays_and_weekends():
    hours = np.array([0.0, 6.0, 0.0, 6.0])
    weekend = np.array([False, False, True, True])
    d = b2d.diurnal_matrix(hours, weekend, 2)
    assert d.shape == (4, 8)
    assert (d[2:, :4] == 0).all() and (d[:2, 4:] == 0).all()
    assert d[0, 0] == pytest.approx(1.0) and d[1, 1] == pytest.approx(1.0)
    assert b2d.diurnal_matrix(hours, weekend, 0).shape == (4, 0)


def test_from_c2_record_rebuilds_model_b():
    record = {
        "model_id": "abc",
        "verdict": "UNDETERMINED",
        "phi": 0.3,
        "folds": [
            {
                "train": ["S2"],
                "room_offsets_k": np.zeros((3, 24)).tolist(),
                "models": {
                    "B": {
                        "thermal": {"params": {**TRUE, "b_ret": 0.4}, "slab_offset_k": 1.5},
                        "response": LAW.to_dict(),
                        "observer": {"gain": [0.5, 0.1]},
                    }
                },
            }
        ],
    }
    m, info = b2d.from_c2_record(record, ROOMS)
    assert info["model_id"] == "abc"
    assert m.n_states == 2 and m.n_diurnal == 0 and m.slab_offset_k == 1.5
    assert m.capacities.tolist() == [TRUE["c_a"], TRUE["c_s"]]
    assert m.law == LAW


# ---------------------------------------------------------------------------
# B2+D identification on a synthetic house
# ---------------------------------------------------------------------------


def _house(params: dict[str, float], days: int = 12, seed: int = 0):
    """A climate data set whose comfort rooms follow the B2+D model with *params*."""
    starts = grid(date(2025, 1, 6), days)  # a Monday
    n = starts.size
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    t_out = 6.0 + 4.0 * np.sin(2 * np.pi * (t / 96.0 - 0.3)) + rng.normal(0, 0.3, n).cumsum() * 0.05
    q_kw = np.where((t // 8) % 3 == 0, 3.0 + rng.uniform(0, 1.5, n), 0.0)
    base = build(make_table(starts, outdoor_temp_mean_c=t_out))
    hour_frac, weekend = b2d.slot_calendar(base)
    diurnal = b2d.diurnal_matrix(hour_frac, weekend, 1)
    ad, bd = rc_matrices(params, 4)
    w = np.column_stack([q_kw, t_out, base.solar, np.ones(n), diurnal])
    xs = thermal.simulate(ad, w @ bd.T, np.array([21.0, 23.0]))[1:]
    air = np.round(xs[:, 0], 2)
    overrides = {f"room_{r}_temp_mean_c": air for r in ROOMS}
    overrides.update({f"room_{r}_setpoint_mean_c": 30.0 for r in (*ROOMS, "office_a")})
    table = make_table(
        starts,
        outdoor_temp_mean_c=t_out,
        hp_heat_out_kwh=q_kw / 4.0,
        hp_heat_energy_out_delta=q_kw / 4.0,
        **overrides,
    )
    d = build(table)
    return d, b2d.full_inputs(d, diurnal)


@pytest.fixture(scope="module")
def house():
    return _house({**TRUE, **DIURNAL})


def test_b2d_recovers_the_slab_and_the_diurnal_gain_from_air_alone(house):
    d, w = house
    train = np.ones(d.n, dtype=bool)
    params, info = b2d.fit_b2d(d, w, train, 4)
    assert info["on_bound"] == []
    for name in ("ua", "c_s", "k", "q0"):
        assert params[name] == pytest.approx(TRUE[name], rel=0.15), name
    assert params["g0"] == pytest.approx(DIURNAL["g0"], abs=0.1)
    assert info["rmse_air_k"] < 0.02


def test_reestimation_moves_only_the_named_parameters_and_only_on_the_window(house):
    d, w = house
    shifted, w2 = _house({**TRUE, **DIURNAL, "ua": 0.13}, seed=1)
    prior = {**TRUE, **DIURNAL}
    window = np.zeros(shifted.n, dtype=bool)
    window[: 8 * 96] = True
    p, ok = b2d.reestimate(shifted, w2, prior, window, ["ua", "q0"], 4)
    assert ok
    assert p["ua"] == pytest.approx(0.13, rel=0.1)
    assert {k: v for k, v in p.items() if k not in ("ua", "q0")} == {
        k: v for k, v in prior.items() if k not in ("ua", "q0")
    }
    same, refitted = b2d.reestimate(shifted, w2, prior, np.zeros(shifted.n, bool), ["ua"], 4)
    assert not refitted and same == prior


def test_the_observer_is_causal(house):
    d, w = house
    m = _model(n_diurnal=4, gain=np.array([0.5, 0.05]))
    xs, _age = m.observe(w, d.t_air)
    y2 = d.t_air.copy()
    y2[500:] += 3.0
    w2 = w.copy()
    w2[500:, 0] *= 2
    xs2, _ = m.observe(w2, y2)
    assert np.array_equal(xs[:501], xs2[:501], equal_nan=True)  # x[t]: slots before t only
    assert not np.allclose(xs[502:], xs2[502:])


def test_fit_heat_law_needs_permitted_slots(house):
    d, _w = house
    with pytest.raises(ValueError, match="too few permitted slots"):
        b2d.fit_heat_law(d, np.zeros((d.n, 2)), _cap(), fit.train_slots(d, ["S2"]), 22.0)
