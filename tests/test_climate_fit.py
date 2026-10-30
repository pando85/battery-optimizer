"""``climate-fit`` on a synthetic house: folds, validation, no leak, identity and output hygiene.

The seasons of ``config/climate.yaml`` are moved into three weeks of January 2025 (a copy in
``tmp_path``) so a whole run stays small.  No real data, no network.
"""

from __future__ import annotations

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
    ):
        assert old in raw
        raw = raw.replace(old, new)
    path = tmp_path_factory.mktemp("cfg") / "climate.yaml"
    path.write_text(raw)
    return load_climate_config(path)


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
    q_kw = 3.0 * u
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
    table = make_table(
        starts,
        outdoor_temp_mean_c=t_out,
        hp_heat_out_kwh=q_kw * thermal.STEP_H,
        hp_heat_kwh=q_kw * thermal.STEP_H / 3.0,
        hp_heat_energy_out_delta=q_kw * thermal.STEP_H,
        pump_flow_mean=np.where(on, 12.0, 0.0),
        return_temp_mean_c=t_air + 3.0,
        flow_temp_mean_c=t_air + 4.0,
        flow_target_temp_mean_c=30.0,
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


def test_fit_recovers_the_house_and_validates_every_fold(result, cfg):
    model, validation = result
    fold2 = model["folds"][1]
    assert (fold2["train"], fold2["test"]) == (["S1", "S2"], "S3")
    params = fold2["models"]["A"]["thermal"]["params"]
    assert params["ua"] == pytest.approx(TRUE_A["ua"], rel=0.15)
    assert params["tau"] == pytest.approx(TRUE_A["tau"], rel=0.15)
    response = fold2["models"]["A"]["response"]
    assert response["c0"] + response["c2"] == pytest.approx(3.0, abs=0.3)  # 3 kW when permitted
    # the per-room offsets are the planted ones (hours with data)
    offsets = np.array(fold2["room_offsets_k"], dtype=float)
    assert np.nanmedian(offsets, axis=1) == pytest.approx(OFFSETS, abs=0.06)
    v = validation["folds"][1]
    assert v["test_season"] == "S3" and v["origins"] > 0
    assert v["q_driven"]["24"]["A"]["rmse"] < 0.2
    assert set(validation["acceptance"]) == {"A", "B"}
    ids = {c["id"] for c in validation["acceptance"]["A"]["criteria"]}
    assert {"q_rmse_6h", "u_rmse_24h", "legacy_replay_kh_below_low"} <= ids
    pending = validation["acceptance"]["A"]["pending"]
    assert pending == ["legacy_replay_electricity", "legacy_replay_cost"]
    assert validation["selection"]["verdict"] in {"ACCEPTED", "UNDETERMINED"}
    json.loads(json_bytes(model))
    json.loads(json_bytes(validation))
    assert "## Acceptance" in fit.render_markdown(model, validation)


def test_held_out_season_does_not_leak_into_offsets_or_fits(cfg, house):
    leaky = _house(cfg, test_shift_k=3.0, test_heat_factor=2.0)
    assert not np.allclose(leaky.t_air, house.t_air, equal_nan=True)
    a, b = fit.fit_fold(house, cfg, 1, 0.3), fit.fit_fold(leaky, cfg, 1, 0.3)
    assert np.array_equal(a.offsets, b.offsets, equal_nan=True)
    for kind in thermal.KINDS:
        ma, mb = a.models[kind], b.models[kind]
        assert ma.fit.params == mb.fit.params
        assert np.array_equal(ma.gain, mb.gain) and ma.process_var == mb.process_var
        assert ma.response == mb.response
    assert a.starts == b.starts
    # the held-out change does reach the validation
    assert fit.validate(house, cfg, a)["q_driven"] != fit.validate(leaky, cfg, b)["q_driven"]


def test_model_id_is_deterministic_and_covers_every_input():
    base = {
        "feature_set_id": "f" * 64,
        "dataset_id": "d" * 64,
        "windows": fit.fold_windows(CONFIG),
        "code_ver": "a" * 40,
        "config_sha256": CONFIG.sha256,
        "site_sha256": "s" * 64,
    }
    first = fit.compute_model_id(**base)
    assert first == fit.compute_model_id(**dict(base))
    assert len(first) == 64 and int(first, 16) >= 0
    for key in base:
        changed = dict(base)
        changed[key] = [] if key == "windows" else "x"
        assert fit.compute_model_id(**changed) != first, key


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
