"""``climate-fit`` on a synthetic house: fold, weekly refits, validation, no leak, causality,
identity and output hygiene.

The seasons of ``config/climate.yaml`` are moved into three weeks of January 2025 (a copy in
``tmp_path``, with a 4-day refit window every 2 days) so a whole run stays small.  No real data,
no network.
"""

from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from climate_fakes import CONFIG, SITE, build, grid, make_table

from battery_optimizer import cli
from battery_optimizer.climate import fit, thermal
from battery_optimizer.climate.config import load_climate_config
from battery_optimizer.features import json_bytes

TRUE_A = {"ua": 0.2, "tau": 80.0, "q0": 2.0, "g_s": 0.5}
FIRST, DAYS = date(2025, 1, 1), 23
OFFSETS = (0.3, -0.2, -0.1)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    raw = CONFIG.path.read_text()
    for old, new in (
        ("S1: {start: 2023-11-08, end: 2024-04-30}", "S1: {start: 2025-01-02, end: 2025-01-08}"),
        ("S2: {start: 2024-10-01, end: 2025-04-30}", "S2: {start: 2025-01-09, end: 2025-01-15}"),
        ("S3: {start: 2025-10-01, end: 2026-04-30}", "S3: {start: 2025-01-16, end: 2025-01-22}"),
        ("reserved_from: 2026-10-01", "reserved_from: 2025-02-01"),
        ("window_days: 28", "window_days: 4"),
        ("max_gap_hours: 6.0", "max_gap_hours: 12.0"),  # the synthetic runs last up to 8 h
        ("refit_every_days: 7", "refit_every_days: 2"),
        ("shrinkage: {A: 1000000.0, B: 10000.0}", "shrinkage: {A: 100.0, B: 100.0}"),
        ("prior_until: 2024-12-31", "prior_until: 2025-01-12"),
        ("grid: [0, 1, 10, 100, 1000, 10000, 100000, 1000000]", "grid: [0, 100, 1000000]"),
    ):
        assert old in raw
        raw = raw.replace(old, new)
    path = tmp_path_factory.mktemp("cfg") / "climate.yaml"
    path.write_text(raw)
    return load_climate_config(path)


def _cop(lift):
    """The synthetic heat pump: COP falls with the lift."""
    return 6.0 - 0.08 * lift


def _prices(starts: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(3)
    hours = starts.size // 4
    hourly = 0.1 + 0.05 * np.sin(np.arange(hours) * 2 * np.pi / 24) + rng.uniform(0, 0.02, hours)
    return np.repeat(hourly, 4)


def _house(cfg, *, test_shift_k: float = 0.0, test_heat_factor: float = 1.0):
    """A first-order house heated at 3 kW whenever the replayed legacy rule permits it."""
    starts = grid(FIRST, DAYS)
    prices = _prices(starts)
    base = build(make_table(starts), prices, config=cfg)
    u = np.nan_to_num(base.u)
    t = np.arange(starts.size) * thermal.STEP_H
    t_out = 5 + 4 * np.sin(2 * np.pi * (t - 14) / 24) + 2 * np.sin(2 * np.pi * t / 120)
    q_kw = 3.0 * u * (1.0 + 0.1 * np.sin(2 * np.pi * t / 7.0))
    ad, bd = thermal.discretize(*thermal.continuous("A", TRUE_A))
    drive = thermal.input_matrix(q_kw, t_out, base.solar) @ bd.T
    t_air = thermal.simulate(ad, drive, [21.0])[1:, 0]
    test = base.season_slots("S3")
    q_kw = np.where(test, q_kw * test_heat_factor, q_kw)
    rng = np.random.default_rng(7)
    rooms = {}
    for room, off in zip(("bedroom", "hall_kitchen", "office_t"), OFFSETS, strict=True):
        reading = t_air + off + rng.normal(0, 0.03, t.size) + np.where(test, test_shift_k, 0.0)
        rooms[f"room_{room}_temp_mean_c"] = np.round(reading, 1)
    on = q_kw > 0
    run_start = on & ~np.concatenate(([False], on[:-1]))
    flow = t_air + 4.0
    table = make_table(
        starts,
        outdoor_temp_mean_c=t_out,
        compressor_freq_mean=np.where(on, 40.0, 0.0),
        hp_heat_out_kwh=q_kw * thermal.STEP_H,
        hp_heat_kwh=q_kw * thermal.STEP_H / _cop(flow - t_out),
        hp_heat_energy_out_delta=q_kw * thermal.STEP_H,
        pump_flow_mean=np.where(on, 12.0, 0.0),
        return_temp_mean_c=t_air + 3.0,
        flow_temp_mean_c=flow,
        flow_target_temp_mean_c=flow,
        compressor_starts_delta=run_start.astype(float),
        room_office_a_temp_mean_c=t_air,
        **rooms,
        **{f"room_{r}_setpoint_mean_c": 30.0 for r in ("bedroom", "hall_kitchen", "office_t")},
    )
    return build(table, prices, config=cfg)


@pytest.fixture(scope="module")
def house(cfg):
    return _house(cfg)


@pytest.fixture(scope="module")
def result(house, cfg):
    return fit.fit_and_validate(house, cfg, SITE)


def test_fit_recovers_the_house_and_validates_the_fold(result, cfg):
    model, validation = result
    assert [(f["train"], f["test"]) for f in model["folds"]] == [(["S2"], "S3")]
    fold = model["folds"][0]
    params = fold["models"]["A"]["thermal"]["params"]
    assert params["ua"] == pytest.approx(TRUE_A["ua"], rel=0.15)
    assert params["tau"] == pytest.approx(TRUE_A["tau"], rel=0.15)
    response = fold["models"]["A"]["response"]
    # about 3 kW when permitted, at the mean outdoor temperature (5 C)
    at_mean = response["c0"] + response["c1"] * (response["t_ref"] - 5.0)
    assert at_mean == pytest.approx(3.0, abs=0.3)
    # the per-room offsets are the planted ones (hours with data)
    offsets = np.array(fold["room_offsets_k"], dtype=float)
    assert np.nanmedian(offsets, axis=1) == pytest.approx(OFFSETS, abs=0.06)
    # weekly refits from the first test day, every 2 days, each near the truth
    weeks = fold["weekly_refits"]["A"]
    assert [w["start"] for w in weeks] == ["2025-01-16", "2025-01-18", "2025-01-20", "2025-01-22"]
    assert weeks[0]["window"] == ["2025-01-12", "2025-01-16"]
    for w in weeks:
        assert w["params"]["ua"] == pytest.approx(TRUE_A["ua"], rel=0.2) and not w["fallback"]
    v = validation["folds"][0]
    assert v["test_season"] == "S3" and v["origins"] > 0
    assert v["q_driven"]["24"]["A"]["rmse"] < 0.2
    assert v["adaptive"]["A"]["weeks"] == 4 and v["adaptive"]["A"]["shrinkage"] == 100.0
    assert v["t_out_fill"]["test"]["filled_slots"] == 0  # no station: nothing to fill from
    assert set(validation["acceptance"]) == {"A", "B"}
    crit = {c["id"]: c for c in validation["acceptance"]["A"]["criteria"]}
    assert {
        "q_rmse_6h",
        "u_rmse_24h",
        "legacy_replay_kh_below_low",
        "legacy_replay_electricity",
        "legacy_replay_cost",
        "parameters_weekly_refits",
        "cop_daily_mape",
        "cop_season_error",
    } <= set(crit)
    assert "parameters_fold_stability" not in crit and not hasattr(fit, "stability")
    assert crit["q_gain_vs_baseline_12h"]["threshold"] == 0.0
    assert crit["parameters_weekly_refits"]["threshold"] == 0.95
    # the COP (lift table, trained on S2) prices the legacy replay: nothing is pending
    assert "error" not in validation["cop"], validation["cop"]
    assert validation["cop"]["selected"] == "lift_table" and validation["cop"]["train"] == ["S2"]
    assert crit["cop_daily_mape"]["status"] == "pass"
    assert validation["acceptance"]["A"]["pending"] == []
    replay = v["legacy_replay"]["A"]
    assert replay["electricity_rel_error"] is not None and replay["cost_rel_error"] is not None
    assert validation["selection"]["verdict"] in {"ACCEPTED", "UNDETERMINED"}
    json.loads(json_bytes(model))
    json.loads(json_bytes(validation))
    text = fit.render_markdown(model, validation)
    assert "## Acceptance" in text and "Weekly refits A" in text


def test_held_out_season_does_not_leak_into_offsets_or_prior_fits(cfg, house):
    leaky = _house(cfg, test_shift_k=3.0, test_heat_factor=2.0)
    assert not np.allclose(leaky.t_air, house.t_air, equal_nan=True)
    a, b = fit.fit_fold(house, cfg, 0, 0.3), fit.fit_fold(leaky, cfg, 0, 0.3)
    assert np.array_equal(a.offsets, b.offsets, equal_nan=True)
    for kind in thermal.KINDS:
        ma, mb = a.models[kind], b.models[kind]
        assert ma.fit.params == mb.fit.params
        assert np.array_equal(ma.gain, mb.gain) and ma.process_var == mb.process_var
        assert ma.response == mb.response
        # the first weekly refit reads only the days before the test season
        assert (
            a.adaptive[kind].weeks[0].model.fit.params == b.adaptive[kind].weeks[0].model.fit.params
        )
    assert a.starts == b.starts
    # the held-out change does reach the validation
    assert fit.validate(house, cfg, a)["q_driven"] != fit.validate(leaky, cfg, b)["q_driven"]


def test_weekly_refits_are_causal(cfg, house):
    """Changing everything from one origin on leaves every earlier refit and state untouched."""
    cut_day = date(2025, 1, 20)
    perturbed = _house(cfg, test_shift_k=0.0)
    labels = [label for label, _lo, _hi in house.days]
    cut = house.days[labels.index(cut_day)][1]
    rng = np.random.default_rng(11)
    for room in house.comfort_rooms:
        perturbed.room_temp[room][cut:] += rng.normal(1.5, 0.5, house.n - cut)
    perturbed.t_air[cut:] = np.mean([perturbed.room_temp[r][cut:] for r in house.comfort_rooms], 0)
    perturbed.q_heat_kwh[cut:] *= 1.7
    prior = fit.fit_fold(house, cfg, 0, 0.3).models["A"]
    first, last = date(2025, 1, 16), date(2025, 1, 22)
    a = fit.adaptive_refits(house, cfg, prior, first, last, 100.0)
    b = fit.adaptive_refits(perturbed, cfg, prior, first, last, 100.0)
    for wa, wb in zip(a.weeks, b.weeks, strict=True):
        if wa.start <= cut_day:  # fitted on days before cut_day only
            assert wa.model.fit.params == wb.model.fit.params, wa.start
        else:
            assert wa.model.fit.params != wb.model.fit.params, wa.start
    # observer states up to the cut use measurements before it only
    assert np.array_equal(a.states[: cut + 1], b.states[: cut + 1], equal_nan=True)
    assert not np.allclose(a.states[cut + 2 :], b.states[cut + 2 :], equal_nan=True)


def test_fit_scores_valve_shut_and_rule_disabled_slots_but_not_bad_air(cfg, house):
    d = copy.copy(house)
    d.drop = {k: v.copy() for k, v in house.drop.items()}
    n = d.n
    marks = {
        "valves_shut": slice(100, 110),
        "rule_disabled_day": slice(200, 210),
        "window_open": slice(300, 310),
        "thermostat_outage": slice(400, 410),
        "unknown_quality": slice(500, 510),
    }
    for name, sl in marks.items():
        d.drop[name][sl] = True
    d.train = ~np.logical_or.reduce(list(d.drop.values()))
    fi = fit.fit_inputs(d, cfg, np.ones(n, dtype=bool))
    scored = np.isfinite(fi.y_air)
    for name in ("valves_shut", "rule_disabled_day"):
        assert scored[marks[name]].all(), name  # plant states: their heat drives and is scored
    for name in ("window_open", "thermostat_outage", "unknown_quality"):
        assert not scored[marks[name]].any(), name
    assert np.array_equal(scored, fit.valid_air(d))
    # the slab measurement follows the same mask
    assert not np.isfinite(fi.y_ret[marks["window_open"]]).any()


def test_weekly_criterion_counts_bounds_and_fallbacks(cfg, result):
    model, validation = result
    summary = validation["folds"][0]["adaptive"]["A"]
    assert summary["ok"] == sum(w["ok"] for w in summary["refits"])
    assert summary["ok_share"] == pytest.approx(summary["ok"] / summary["weeks"])
    good = thermal.ThermalFit(
        "A",
        {"ua": 0.1, "tau": 50.0, "q0": 0.5, "g_s": 0.0},
        {},
        [],
        [],
        {},
        False,
        False,
        0.1,
        None,
        1,
        0,
        1,
        0.0,
        (0.1, 0.5),
        1,
        "",
    )
    week = fit.Week(date(2025, 1, 1), 0, 1, (date(2024, 12, 4), date(2025, 1, 1)), None)
    week.model = type("M", (), {"fit": good, "kind": "A"})()
    assert week.ok  # g_s at 0 is legitimate; identifiability is reported, not gated
    good.on_bound = ["ua"]
    assert not week.ok
    good.on_bound = []
    week.fallback = "no fit segment"
    assert not week.ok


def test_climate_fit_refuses_an_unselected_shrinkage(house, cfg, tmp_path):
    text = cfg.path.read_text().replace(
        "shrinkage: {A: 100.0, B: 100.0}", "shrinkage: {A: 100.0, B: null}"
    )
    path = tmp_path / "climate.yaml"
    path.write_text(text)
    with pytest.raises(fit.ClimateFitError, match="not selected"):
        fit.fit_and_validate(house, load_climate_config(path), SITE)


def test_shrinkage_selection_reads_s2_only(cfg, house):
    report = fit.select_shrinkage(house, cfg)
    assert report["season"] == "S2" and report["evaluation"] == ["2025-01-13", "2025-01-15"]
    for kind in thermal.KINDS:
        m = report["models"][kind]
        assert m["chosen"] in (0.0, 100.0, 1e6)
        assert set(m["variants"]) == {"prior_only", "0", "100", "1e+06"}
        assert m["origins"] > 0
    leaky = _house(cfg, test_shift_k=3.0, test_heat_factor=2.0)
    again = fit.select_shrinkage(leaky, cfg)
    assert again == report
    assert "chosen shrinkage" in fit.render_selection(report)


def test_model_id_is_deterministic_and_covers_every_input():
    base = {
        "feature_set_id": "f" * 64,
        "dataset_id": "d" * 64,
        "windows": fit.fold_windows(CONFIG),
        "code_ver": "a" * 40,
        "config_sha256": CONFIG.sha256,
        "site_sha256": "s" * 64,
        "input_sha256": "i" * 64,
    }
    first = fit.compute_model_id(**base)
    assert first == fit.compute_model_id(**dict(base))
    assert len(first) == 64 and int(first, 16) >= 0
    for key in base:
        changed = dict(base)
        changed[key] = [] if key == "windows" else "x"
        assert fit.compute_model_id(**changed) != first, key


def test_model_id_follows_the_outdoor_inputs(cfg, house):
    """A different station manifest (the fused T_out the fill reads) is a different model."""
    a = copy.copy(house)
    b = copy.copy(house)
    a.fusion = type("F", (), {"station_sha256": "station-a"})()
    b.fusion = type("F", (), {"station_sha256": "station-b"})()
    assert a.input_sha256 != b.input_sha256 != house.input_sha256
    ids = {
        fit.compute_model_id(
            feature_set_id="f",
            dataset_id="d",
            windows=fit.fold_windows(cfg),
            code_ver="c",
            config_sha256=cfg.sha256,
            site_sha256="s",
            input_sha256=d.input_sha256,
        )
        for d in (house, a, b)
    }
    assert len(ids) == 3


def test_write_model_is_atomic_and_no_clobber(tmp_path, result):
    model, validation = result
    out = fit.write_model(tmp_path, model, validation)
    assert out == tmp_path / "analysis" / "v1" / "climate-models" / model["model_id"]
    assert sorted(p.name for p in out.iterdir()) == [
        "model.json",
        "validation.json",
        "validation.md",
    ]
    assert (out / "model.json").stat().st_mode & 0o777 == 0o600
    assert fit.write_model(tmp_path, model, validation) == out  # same bytes: a no-op
    with pytest.raises(FileExistsError):
        fit.write_model(tmp_path, dict(model, phi=0.5), validation)


def test_run_refuses_a_data_dir_inside_a_git_tree():
    repo = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="inside git worktree"):
        fit.run_climate_fit(feature_set="x", dataset="y", data_dir=str(repo / "data"))


def test_cli_climate_fit_prints_the_directory_and_verdict(tmp_path, monkeypatch, capsys):
    calls = {}

    def fake_run(**kwargs):
        calls.update(kwargs)
        (tmp_path / "validation.json").write_text(
            json.dumps({"selection": {"verdict": "UNDETERMINED", "selected": None}})
        )
        return tmp_path

    monkeypatch.setattr(fit, "run_climate_fit", fake_run)
    code = cli.main(["climate-fit", "--feature-set", "abc", "--dataset", "ds", "--data-dir", "/x"])
    assert code == 0
    assert calls == {
        "feature_set": "abc",
        "dataset": "ds",
        "config_path": "config/climate.yaml",
        "data_dir": "/x",
    }
    out = capsys.readouterr().out
    assert f"model directory: {tmp_path}" in out and "verdict UNDETERMINED" in out
