"""Thermal models and heat response on synthetic houses (no real data, no network)."""

from __future__ import annotations

import numpy as np
import pytest
from climate_fakes import CONFIG

from battery_optimizer.climate import plant, thermal

ENVELOPE = CONFIG.model.envelope
TRUE_A = {"ua": 0.2, "tau": 80.0, "q0": 0.6, "g_s": 0.8}
TRUE_B = {"ua": 0.2, "c_a": 3.0, "c_s": 12.0, "k": 1.2, "q0": 0.6, "g_s": 0.8, "b_ret": 1.5}


def _drivers(days: int = 20, constant_outdoor: bool = False):
    t = np.arange(days * 96) * thermal.STEP_H
    if constant_outdoor:
        t_out, solar = np.full(t.size, 5.0), np.zeros(t.size)
    else:
        t_out = 5 + 4 * np.sin(2 * np.pi * (t - 14) / 24) + 3 * np.sin(2 * np.pi * t / 168)
        solar = np.maximum(0.0, np.sin(2 * np.pi * (t - 6) / 24))
    hour = t % 24
    q = np.where((hour > 2) & (hour < 9), 4.0, 0.0) + np.where((hour > 14) & (hour < 16), 3.0, 0)
    return thermal.input_matrix(q, t_out, solar)


def _house(kind, params, w, x0, seed=0):
    """Measured air (noise + 0.1 K steps) and, for B, a noisy return in circulation."""
    rng = np.random.default_rng(seed)
    ad, bd = thermal.discretize(*thermal.continuous(kind, params))
    xs = thermal.simulate(ad, w @ bd.T, np.array(x0))
    y = np.round(xs[1:, 0] + rng.normal(0, 0.05, w.shape[0]), 1)
    y_ret = None
    if kind == "B":
        noise = rng.normal(0, 0.3, w.shape[0])
        y_ret = np.where(w[:, 0] > 0, xs[1:, 1] + params["b_ret"] + noise, np.nan)
    return xs, y, y_ret


def _fit_inputs(w, y, y_ret=None):
    window = np.arange(w.shape[0]) // 192
    segments = thermal.segments_in_windows(window, np.ones(w.shape[0], bool), y)
    return thermal.FitInputs(w=w, y_air=y, y_init=y, segments=segments, y_ret=y_ret)


# ---------------------------------------------------------------------------
# Discretisation and simulation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("kind", "params"), [("A", TRUE_A), ("B", TRUE_B)])
def test_zoh_is_exact_against_a_fine_euler_step(kind, params):
    ac, bc = thermal.continuous(kind, params)
    ad, bd = thermal.discretize(ac, bc)
    x0 = np.array([20.0] if kind == "A" else [20.0, 24.0])
    w = np.array([3.0, 4.0, 0.5, 1.0])
    x = x0.copy()
    steps = 9000  # 0.1 s
    h = thermal.STEP_H / steps
    for _ in range(steps):
        x = x + h * (ac @ x + bc @ w)
    assert np.allclose(ad @ x0 + bd @ w, x, atol=1e-6)


@pytest.mark.parametrize(("kind", "params"), [("A", TRUE_A), ("B", TRUE_B)])
def test_simulate_matches_the_step_recursion(kind, params):
    ad, bd = thermal.discretize(*thermal.continuous(kind, params))
    w = _drivers(2)
    x0 = np.array([21.0] if kind == "A" else [21.0, 25.0])
    xs = thermal.simulate(ad, w @ bd.T, x0)
    x = x0.copy()
    for t in range(w.shape[0]):
        x = ad @ x + bd @ w[t]
    assert np.allclose(xs[-1], x) and np.allclose(xs[0], x0)


def test_segments_start_after_a_measurement_and_stay_inside_a_window():
    n = 400
    window = np.arange(n) // 192
    usable = np.ones(n, bool)
    usable[100:110] = False
    y = np.full(n, 21.0)
    y[109:115] = np.nan  # the slot before a start must be measured
    segs = thermal.segments_in_windows(window, usable, y, min_slots=20)
    # slot 0 has no measured slot before it; the last 16 slots are too short a run
    assert segs == [(1, 100), (116, 192), (192, 384)]


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


def test_recovers_model_a_from_noisy_quantised_readings():
    w = _drivers()
    _xs, y, _ = _house("A", TRUE_A, w, [21.0])
    fit = thermal.fit_thermal("A", _fit_inputs(w, y), ENVELOPE)
    assert fit.ok, fit.failures()
    for name in ("ua", "tau", "q0", "g_s"):
        assert fit.params[name] == pytest.approx(TRUE_A[name], rel=0.1), name
    assert fit.rmse_air_k < 0.1


def test_recovers_model_b_with_the_slab_measurement():
    w = _drivers()
    _xs, y, y_ret = _house("B", TRUE_B, w, [21.0, 23.0])
    fit = thermal.fit_thermal("B", _fit_inputs(w, y, y_ret), ENVELOPE)
    assert fit.ok, fit.failures()
    for name in ("ua", "c_a", "c_s", "k", "b_ret"):
        assert fit.params[name] == pytest.approx(TRUE_B[name], rel=0.1), name
    assert fit.n_return > 0


def test_b_is_not_identifiable_without_the_slab_measurement_and_it_is_detected():
    # constant outdoor temperature and no sun: the air temperature sees the heat only through a
    # second-order transfer function with three coefficients, so (C_a, C_s, k) is a family
    w = _drivers(constant_outdoor=True)
    _xs, y, y_ret = _house("B", TRUE_B, w, [21.0, 23.0])
    blind = thermal.fit_thermal("B", _fit_inputs(w, y), ENVELOPE)
    assert not blind.identifiable and not blind.ok
    assert "not identifiable" in blind.failures()
    assert max(blind.rel_se[n] for n in ("c_a", "c_s", "k")) > thermal.IDENT_MAX_REL_SE
    seen = thermal.fit_thermal("B", _fit_inputs(w, y, y_ret), ENVELOPE)
    assert seen.identifiable, seen.rel_se
    assert seen.params["c_s"] == pytest.approx(TRUE_B["c_s"], rel=0.1)


def test_a_parameter_on_an_envelope_bound_is_a_failed_fit():
    leaky = dict(TRUE_A, ua=0.8)  # above the envelope's 0.5 kW/K
    w = _drivers()
    _xs, y, _ = _house("A", leaky, w, [21.0])
    fit = thermal.fit_thermal("A", _fit_inputs(w, y), ENVELOPE)
    assert "ua" in fit.on_bound and not fit.ok
    assert fit.params["ua"] == pytest.approx(ENVELOPE["ua_kw_per_k"][1], rel=1e-3)
    assert "on bound: ua" in fit.to_dict()["failures"]


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------


def test_observer_converges_to_the_true_slab_state_from_a_wrong_start():
    w = _drivers(6)
    xs, y, _ = _house("B", TRUE_B, w, [21.0, 26.0])
    ad, bd = thermal.discretize(*thermal.continuous("B", TRUE_B))
    gain = thermal.observer_gain(ad, 0.002, thermal.MEASUREMENT_SIGMA_K**2)
    est, age = thermal.observe(ad, bd, w, y, gain, slab_offset=-4.0)  # starts 9 K too cold
    assert abs(est[1, 1] - xs[1, 1]) > 5.0
    truth = xs[:-1]  # est[t] is the state at the start of slot t
    late = slice(48 * 4, None)
    assert np.abs(est[late, 1] - truth[late, 1]).max() < 0.5
    assert np.abs(est[late, 0] - truth[late, 0]).max() < 0.2
    assert age[-1] == w.shape[0] - 2


def test_observer_bridges_an_unknown_outdoor_temperature_and_stops_without_heat():
    w = _drivers(3)
    xs, y, _ = _house("B", TRUE_B, w, [21.0, 23.0])
    ad, bd = thermal.discretize(*thermal.continuous("B", TRUE_B))
    gain = thermal.observer_gain(ad, 0.002, thermal.MEASUREMENT_SIGMA_K**2)
    slab = (float(np.exp(-TRUE_B["k"] * thermal.STEP_H / TRUE_B["c_s"])), TRUE_B["k"])
    gap = w.copy()
    gap[100:130, 1] = np.nan  # outdoor unknown, heat known: bridged
    est, age = thermal.observe(ad, bd, gap, y, gain, 0.0, slab)
    assert (age[100:200] >= 0).all() and age[200] > age[99]
    assert abs(est[130, 1] - xs[130, 1]) < 0.5
    gap[150, 0] = np.nan  # heat unknown too: the filter stops and restarts
    _est, age = thermal.observe(ad, bd, gap, y, gain, 0.0, slab)
    assert age[150] > 0 and age[151] == 0  # restarted from the reading of slot 150


# ---------------------------------------------------------------------------
# Plant: response, valve cap, starts
# ---------------------------------------------------------------------------


def test_first_hour_of_block_and_blocks():
    u = np.array([0, 1, 1, 1, 1, 1, 0, 1, 1, np.nan, 1], dtype=float)
    assert list(np.flatnonzero(plant.first_hour_of_block(u))) == [1, 2, 3, 4, 7, 8, 10]
    assert plant.permission_blocks(u) == 3


def test_valve_cap_counts_the_rooms_below_setpoint_with_the_plant_margin():
    offsets = np.zeros((3, 24))
    offsets[0, 5] = 1.0
    cap = plant.ValveCap(phi=0.3, margin_k=0.25, offsets=offsets)
    sp = np.array([22.0, 22.0, 22.0])
    assert cap.factor(21.5, 0, sp) == pytest.approx(1.0)
    # at hour 5 the first room reads 22.5: one of three valves shut
    assert cap.factor(21.5, 5, sp) == pytest.approx(0.3 + 0.7 * 2 / 3)
    assert cap.factor(21.8, 0, sp) == pytest.approx(0.3)  # within the margin: shut
    rooms = np.array([[21.0, 23.0, np.nan], [21.0, 21.0, 21.0], [21.0, 21.0, 21.0]])
    meas = cap.measured(rooms, np.full((3, 3), 22.0))
    assert meas[0] == pytest.approx(1.0) and meas[1] == pytest.approx(0.3 + 0.7 * 2 / 3)
    assert np.isnan(meas[2])


def test_response_b_is_clipped_between_zero_and_the_capacity():
    r = plant.ResponseB(h_e=1.0, qm0=3.0, qm1=0.1, t_ref=22.0)
    assert r.power(2.0, 26.0, 25.0) == pytest.approx(1.0)
    assert r.power(2.0, 26.0, 27.0) == 0.0
    assert r.power(2.0, 40.0, 25.0) == pytest.approx(5.0)  # Q_max(2 °C) = 3 + 0.1 * 20


def test_response_fits_recover_known_coefficients():
    rng = np.random.default_rng(1)
    n = 2000
    t_out = rng.uniform(-2, 15, n)
    first = rng.random(n) < 0.25
    cap = np.where(rng.random(n) < 0.2, 0.3, 1.0)
    true_a = plant.ResponseA(1.0, 0.15, 1.2, 22.0)
    q = cap * true_a.power(t_out, first) + rng.normal(0, 0.05, n)
    fit_a, stats = plant.fit_response_a(q, cap, t_out, first, 22.0)
    assert (fit_a.c0, fit_a.c1, fit_a.c2) == pytest.approx((1.0, 0.15, 1.2), rel=0.05)
    assert abs(stats["bias"]) < 0.01
    t_slab = rng.uniform(22, 27, n)
    true_b = plant.ResponseB(0.8, 3.0, 0.1, 22.0)
    q = cap * true_b.power(t_out, 28.0, t_slab) + rng.normal(0, 0.05, n)
    fit_b, _ = plant.fit_response_b(q, cap, t_out, np.full(n, 28.0), t_slab, 22.0)
    assert (fit_b.h_e, fit_b.qm0, fit_b.qm1) == pytest.approx((0.8, 3.0, 0.1), rel=0.05)


def _plant_inputs(n, u, dhw=None):
    return plant.PlantInputs(
        u=u,
        dhw=np.zeros(n, bool) if dhw is None else dhw,
        t_out=np.full(n, 5.0),
        solar=np.zeros(n),
        t_flow_target=np.full(n, 28.0),
        first_hour=plant.first_hour_of_block(u),
        local_hour=np.arange(n) // 4 % 24,
        setpoints=np.full((3, n), 30.0),
    )


def test_u_driven_simulation_heats_only_when_permitted_and_not_in_dhw():
    n = 16
    u = np.zeros(n)
    u[4:12] = 1.0
    dhw = np.zeros(n, bool)
    dhw[8] = True
    inp = _plant_inputs(n, u, dhw)
    cap = plant.ValveCap(0.3, 0.25, np.zeros((3, 24)))
    ad, bd = thermal.discretize(*thermal.continuous("A", TRUE_A))
    _xs, q = plant.simulate_u(
        "A", ad, bd, np.array([21.0]), 0, n, inp, plant.ResponseA(2.0, 0.0, 1.0, 22.0), cap
    )
    assert list(np.flatnonzero(q)) == [4, 5, 6, 7, 9, 10, 11]
    assert q[4] == pytest.approx(3.0) and q[9] == pytest.approx(2.0)
    ad, bd = thermal.discretize(*thermal.continuous("B", TRUE_B))
    xs, q = plant.simulate_u(
        "B", ad, bd, np.array([21.0, 27.5]), 0, n, inp, plant.ResponseB(1.0, 5.0, 0.0, 22.0), cap
    )
    assert q[4] == pytest.approx(28.0 - xs[4, 1])  # reads the simulated slab
    assert inp.known("B").all()
    inp.t_flow_target[5] = np.nan
    assert not inp.known("B")[5] and inp.known("A")[5]


def test_starts_model_recovers_its_coefficients():
    rng = np.random.default_rng(2)
    blocks = rng.integers(1, 5, 60).astype(float)
    hours = rng.uniform(2, 15, 60)
    t_out = rng.uniform(-2, 14, 60)
    true = plant.StartsModel(1.5, 0.8, -0.03)
    starts = true.predict(blocks, hours, t_out) + rng.normal(0, 0.1, 60)
    fit = plant.fit_starts(blocks, hours, t_out, starts)
    assert (fit.alpha, fit.beta, fit.gamma) == pytest.approx((1.5, 0.8, -0.03), rel=0.1)
