"""The offline MPC replay (``climate/mpc_replay.py``): causality, schedule, summary."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from climate_fakes import build, grid, make_table

from battery_optimizer.climate import b2d, mpc_replay, plant
from battery_optimizer.climate.cop import COP_MODEL_VERSION, CopModel
from battery_optimizer.climate.mpc import PlanSettings
from battery_optimizer.climate.mpc_config import load_mpc_config
from battery_optimizer.climate.mpc_model import HeatLaw

CFG = load_mpc_config(Path(__file__).resolve().parents[1] / "config/climate-mpc.yaml")
ROOMS = ("bedroom", "hall_kitchen", "office_t")
PARAMS = {"ua": 0.1, "c_a": 2.0, "c_s": 12.0, "k": 2.0, "q0": 0.3, "g_s": 0.8}
PARAMS.update({f"g{i}": 0.05 * (i % 3) for i in range(8)})
T0 = 3 * 96 + 40  # day 4, 10:00 local


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


def _data(perturb_after: int | None = None, outdoor_too: bool = True):
    """Six synthetic days; with *perturb_after*, every measurement from that slot on differs."""
    starts = grid(date(2025, 1, 6), 6)
    n = starts.size
    t = np.arange(n)
    rng = np.random.default_rng(0)
    t_out = 5.0 + 3.0 * np.sin(2 * np.pi * t / 96.0)
    room = 21.5 + 0.5 * np.sin(2 * np.pi * t / 96.0) + rng.normal(0, 0.05, n)
    q = np.where((t // 6) % 4 == 0, 1.0, 0.0)
    dhw = np.where(t % 96 == 20, 0.3, 0.0)
    flow = np.full(n, 27.0)
    if perturb_after is not None:
        room[perturb_after:] += 2.0
        q[perturb_after:] = 0.7
        flow[perturb_after:] = 30.0
        if outdoor_too:
            t_out[perturb_after:] -= 4.0
    rooms = {f"room_{r}_temp_mean_c": room for r in (*ROOMS, "office_a")}
    table = make_table(
        starts,
        outdoor_temp_mean_c=t_out,
        hp_heat_out_kwh=q,
        hp_heat_energy_out_delta=q,
        hp_dhw_kwh=dhw,
        flow_target_temp_mean_c=flow,
        **rooms,
    )
    return build(table)


def _context(d, weather: str) -> mpc_replay.Context:
    cfg = CFG
    if weather != cfg.replay.weather:
        raw = cfg.to_dict()
        raw["replay"]["weather"] = weather
        cfg = _section_cfg(raw, CFG)
    hour_frac, weekend = b2d.slot_calendar(d)
    diurnal = b2d.diurnal_matrix(hour_frac, weekend, 2)
    causal = mpc_replay.causal_outdoor(d)
    w = b2d.full_inputs(d, diurnal)
    w[:, 1] = d.t_out_model if weather == "perfect" else causal
    cap = plant.ValveCap(0.3, 0.25, np.zeros((3, 24)))
    model = b2d.make_model(
        "test", PARAMS, 8, HeatLaw(0.5, 4.0, 0.1, 22.0), cap, ROOMS, gain=np.array([0.5, 0.05])
    )
    states, age = model.observe(w, d.t_air)
    sm = b2d.SeasonModel("S2", "S3", [(0, model)], states, age, {})
    return mpc_replay.Context(
        d=d,
        cfg=cfg,
        settings=replace(PlanSettings.from_config(cfg), relinearize_iterations=1),
        season=sm,
        cop_model=_cop(),
        diurnal=diurnal,
        hour_frac=hour_frac,
        weekend=weekend,
        t_out_model=w[:, 1],
        t_out_causal=causal,
        compressor=mpc_replay.compressor_state(d),
    )


def _section_cfg(raw, like):
    from battery_optimizer.climate.config import Section

    def wrap(x):
        return Section({k: wrap(v) for k, v in x.items()}) if isinstance(x, dict) else x

    out = type(like)(wrap(raw))
    object.__setattr__(out, "sha256", like.sha256)
    return out


def _plan_part(rec):
    return {k: v for k, v in rec.items() if k not in ("accuracy", "solves")}


def test_persistence_decisions_read_nothing_after_the_decision():
    base = mpc_replay.run_decision(_context(_data(), "persistence"), T0, accuracy=False)
    later = mpc_replay.run_decision(_context(_data(T0), "persistence"), T0, accuracy=False)
    assert base is not None and base["status"] == "optimal"
    assert _plan_part(base) == _plan_part(later)
    inp_a = mpc_replay.decision_inputs(_context(_data(), "persistence"), T0)
    inp_b = mpc_replay.decision_inputs(_context(_data(T0), "persistence"), T0)
    for key in ("x0", "minimum", "dhw"):
        assert np.array_equal(inp_a[key], inp_b[key])
    assert np.array_equal(inp_a["ex"].t_out, inp_b["ex"].t_out)
    assert (inp_a["prev_on"], inp_a["prev_run_slots"]) == (
        inp_b["prev_on"],
        inp_b["prev_run_slots"],
    )


def test_perfect_weather_is_the_only_declared_foresight():
    # rooms, heat and water after the decision do not matter ...
    base = mpc_replay.run_decision(_context(_data(), "perfect"), T0, accuracy=False)
    rooms_only = _data(T0, outdoor_too=False)
    later = mpc_replay.run_decision(_context(rooms_only, "perfect"), T0, accuracy=False)
    assert _plan_part(base) == _plan_part(later)
    # ... the outdoor temperature over the horizon does (perfect foresight, declared)
    inp = mpc_replay.decision_inputs(_context(_data(T0), "perfect"), T0)
    ref = mpc_replay.decision_inputs(_context(_data(), "perfect"), T0)
    assert not np.allclose(inp["ex"].t_out, ref["ex"].t_out)
    assert np.allclose(inp["x0"], ref["x0"])


def test_no_state_or_no_horizon_means_no_decision():
    ctx = _context(_data(), "perfect")
    assert mpc_replay.decision_inputs(ctx, 10) is None  # observer still warming up
    assert mpc_replay.decision_inputs(ctx, ctx.d.n - 10) is None  # horizon past the data


def test_the_run_in_progress_comes_from_measured_slots():
    c = np.array([0, 0, 1, 1, 1, np.nan, 0], dtype=float)
    assert mpc_replay._run_in_progress(c, 5) == (True, 3)
    assert mpc_replay._run_in_progress(c, 2) == (False, 2)
    assert mpc_replay._run_in_progress(c, 6) == (False, 10**6)  # unknown: no carry-in


def test_comfort_minimum_follows_the_schedule():
    hours = np.array([8.75, 9.0, 11.75, 12.0, 21.75, 22.0])
    weekday = mpc_replay.comfort_minimum(ROOMS, hours, np.zeros(6, bool), CFG)
    weekend = mpc_replay.comfort_minimum(ROOMS, hours, np.ones(6, bool), CFG)
    hall = ROOMS.index("hall_kitchen")
    assert weekday[hall].tolist() == [21.0, 21.0, 21.0, 23.5, 23.5, 21.0]
    assert weekend[hall].tolist() == [21.0, 23.5, 23.5, 23.5, 23.5, 21.0]
    assert (weekday[[0, 2]] == 21.0).all()
    with pytest.raises(ValueError, match="day room"):
        mpc_replay.comfort_minimum(("bedroom",), hours, np.zeros(6, bool), CFG)


def test_causal_outdoor_holds_the_last_valid_reading():
    d = _data()
    valid = d.t_out_valid.copy()
    c = mpc_replay.causal_outdoor(d)
    assert np.allclose(c[valid], d.t_out[valid])


def test_a_day_replays_and_summarises():
    ctx = _context(_data(), "perfect")
    days = mpc_replay.replay_days(ctx, [3])
    assert days[0]["decisions"] and "plan_on" in days[0]["decisions"][0]
    summary = mpc_replay.summarise(days, ctx.cfg)
    assert summary["solver"]["infeasible_or_error"] == 0
    assert summary["solver"]["decisions"] == len(days[0]["decisions"])
    assert summary["comparison"]["days"] == 1
    assert set(summary["accuracy"]) >= {"1h", "6h", "12h", "24h", "daily_heat"}


def test_material_difference_rules():
    rows = [
        {"plan_kwh": 10.0, "legacy_kwh": 10.0, "overlap": 0.9, "hdd": 5.0, "legacy_model_kwh": 9},
        {"plan_kwh": 13.0, "legacy_kwh": 10.0, "overlap": 0.9, "hdd": 6.0, "legacy_model_kwh": 9},
        {"plan_kwh": 10.0, "legacy_kwh": 10.0, "overlap": 0.2, "hdd": 7.0, "legacy_model_kwh": 9},
    ]
    for r in rows:
        r.update(plan_day_kh=0.0, plan_floor_kh=0.0, legacy_day_kh=1.0, legacy_floor_kh=0.0)
    out = mpc_replay._compare(rows, CFG)
    assert out["material"]["kwh_share"] == pytest.approx(1 / 3)
    assert out["material"]["schedule_share"] == pytest.approx(1 / 3)
    assert out["material"]["either_share"] == pytest.approx(2 / 3)
    assert out["plan_over_legacy_kwh"] == pytest.approx(33 / 30)
