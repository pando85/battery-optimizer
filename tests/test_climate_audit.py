"""``hack/climate_audit.py`` on synthetic data: the pure analysis functions and a smoke run."""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
from climate_fakes import CONFIG, SLOTS_DAY, build, grid, make_table

from battery_optimizer.features import RawSeries, json_bytes

SCRIPT = Path(__file__).resolve().parents[1] / "hack" / "climate_audit.py"
spec = importlib.util.spec_from_file_location("climate_audit", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

DAY_US = 86_400 * 1_000_000


def _price_pattern(n_days: int, first_day: date) -> np.ndarray:
    rng = np.random.default_rng(3)
    hours = grid(first_day, n_days).size // 4
    hourly = 0.1 + 0.05 * np.sin(np.arange(hours) * 2 * np.pi / 24) + rng.uniform(0, 0.02, hours)
    return np.repeat(hourly, 4)


def _raw(values, step_minutes=1):
    v = np.asarray(values, dtype=float)
    t = np.arange(v.size, dtype=np.int64) * step_minutes * 60 * 1_000_000
    return RawSeries(entity="x", metric="m", unit="u", t_us=t, values=v)


def test_runs_finds_maximal_true_runs():
    flag = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=bool)
    assert audit._runs(flag) == [(1, 2), (5, 1), (7, 3)]
    assert audit._runs(np.zeros(4, dtype=bool)) == []


def test_resolution_reports_the_reading_grid_and_update_rate():
    # a thermostat on a 0.1 K grid that changes every 30 minutes, ten samples apart (3 min)
    values = np.repeat(np.round(20 + 0.1 * np.arange(40), 1), 10)
    r = audit.resolution(_raw(values, step_minutes=3))
    assert abs(r["min_distinct_gap"] - 0.1) < 1e-6
    assert r["share_on_tenth_grid"] == 1.0
    assert r["minutes_between_changes"]["p50"] == 30.0
    assert r["value_changes_per_day"] > 40
    # a whole-degree sensor
    whole = audit.resolution(_raw(np.repeat([5.0, 6.0, 7.0, 6.0], 5)))
    assert whole["min_distinct_gap"] == 1.0 and whole["share_on_integer_grid"] == 1.0
    assert audit.resolution(_raw([1.0]))["samples"] == 1


def test_counter_quantum_finds_the_step_and_rollbacks():
    q = audit.counter_quantum(_raw([0, 0.01, 0.01, 0.02, 0.0195, 0.0295, 0.0395]))
    assert q["increments"] == 4 and q["smallest_increment"] == 0.01
    assert q["rollbacks"] == 1
    assert audit.counter_quantum(_raw([1.0, 1.0, 1.0]))["increments"] == 0


def test_min_hours_regimes_group_days_by_setting():
    starts = grid(date(2025, 11, 1), 12)
    hours = np.full(starts.size, 10.0)
    hours[5 * SLOTS_DAY : 9 * SLOTS_DAY] = 15.0
    hours[9 * SLOTS_DAY :] = 10.0
    data = build(make_table(starts, climate_min_hours_at_start=hours))
    regimes = audit.min_hours_regimes(data)
    rows = regimes["S3"]
    assert [(r["min_hours"], r["days"]) for r in rows] == [(10.0, 5), (15.0, 4), (10.0, 3)]
    assert rows[1]["from"] == "2025-11-06" and rows[1]["to"] == "2025-11-09"
    assert regimes["S3_distinct_values_in_gate_months"] == [10.0, 15.0]


def test_legacy_mask_hit_rate_counts_heat_inside_and_outside_the_mask():
    first, n_days = date(2025, 12, 1), 20
    prices = _price_pattern(n_days, first)
    starts = grid(first, n_days)
    u = build(make_table(starts), prices).u
    # heat exactly where the replayed rule permits it, plus a little leakage outside
    q = np.where(np.nan_to_num(u) == 1.0, 0.5, 0.0)
    q[np.flatnonzero(u == 0.0)[::50]] = 0.5
    data = build(make_table(starts, hp_heat_out_kwh=q), prices)
    hit = audit.legacy_mask_hit_rate(data)["S3"]
    assert hit["heat_on_given_u1"] == 1.0
    assert 0.0 < hit["heat_on_given_u0"] < 0.05
    assert hit["heat_energy_share_in_u1"] > 0.9
    shifted = audit.legacy_mask_hit_rate(data)["S3_shifted_mask_hours"]
    assert shifted["0"]["heat_on_given_u0"] < shifted["2"]["heat_on_given_u0"]


def test_outdoor_sensor_sun_recovers_a_planted_solar_bump():
    first, n_days = date(2025, 12, 1), 10
    starts = grid(first, n_days)
    base = build(make_table(starts))
    data = build(make_table(starts, outdoor_temp_mean_c=5.0 + 6.0 * base.solar))
    sun = audit.outdoor_sensor_sun(data)
    assert 5.0 < sun["anomaly_vs_solar"]["slope"] < 7.0 and sun["anomaly_vs_solar"]["r"] > 0.99
    assert sun["peak_hour_local"] == sun["solar_peak_hour_local"]
    assert sun["midday_minus_night_k"] > 2.0


def test_outdoor_sensor_compressor_step_is_detected():
    first, n_days = date(2025, 12, 1), 10
    starts = grid(first, n_days)
    freq = np.zeros(starts.size)
    for k in range(n_days):  # a night run from 02:00 to 06:00 local on every day
        freq[k * SLOTS_DAY + 2 * 4 : k * SLOTS_DAY + 6 * 4] = 40.0
    data = build(
        make_table(starts, outdoor_temp_mean_c=5.0 - 2.0 * (freq > 0), compressor_freq_mean=freq)
    )
    sun = audit.outdoor_sensor_sun(data)
    assert sun["night_compressor_on_minus_off_k"]["p50"] < -1.5
    edges = sun["night_t_out_change_over_two_slots_k"]
    assert edges["at_compressor_start"]["p50"] < -1.0
    assert edges["at_compressor_stop"]["p50"] > 1.0


def test_filter_survival_and_coverage_share_the_season_slots():
    first, n_days = date(2025, 12, 1), 6
    starts = grid(first, n_days)
    cool = np.zeros(starts.size)
    cool[:SLOTS_DAY] = 1.0  # one cooling day
    data = build(make_table(starts, hp_cool_kwh=cool), _price_pattern(n_days, first))
    surv = audit.filter_survival(data)["S3_gate"]
    assert surv["slots"] == n_days * SLOTS_DAY
    assert abs(surv["drops_cooling"] - 1 / n_days) < 1e-9
    assert surv["survive"] <= 1 - surv["drops_cooling"]


def test_run_audit_smoke_and_report_is_json(tmp_path):
    first, n_days = date(2025, 12, 1), 20
    starts = grid(first, n_days)
    data = build(make_table(starts), _price_pattern(n_days, first))
    coverage = {"room_bedroom_temp": np.ones(starts.size), "outdoor_temp": np.ones(starts.size)}
    report = audit.run_audit(data, {}, coverage, CONFIG)
    for key in (
        "coverage",
        "gaps_per_signal",
        "counter_quantization",
        "return_temperature_slab_proxy",
        "heating_curve",
        "legacy_mask_hit_rate",
        "channel2_over_heishamon",
        "outdoor_sensor_sun",
        "filter_survival",
        "min_hours_regimes",
    ):
        assert key in report
    parsed = json.loads(json_bytes(report))
    assert parsed["gaps_per_signal"]["outdoor_temp"]["unknown_share"] == 0.0
    assert "# Climate data audit" in audit.render_markdown(report)


def test_steady_circulation_honours_the_minutes_into_the_run():
    circ = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1], dtype=bool)
    # 10 min: at least one full slot of circulation before the slot (15 min)
    expected = [False, False, True, True, False, False, False, False, True]
    assert list(audit.steady_circulation(circ, 10)) == expected
    # 30 min: two full slots before it
    assert list(np.flatnonzero(audit.steady_circulation(circ, 30))) == [3]
    # 0 min: every circulating slot
    assert np.array_equal(audit.steady_circulation(circ, 0), circ)


def test_report_name_is_sanitised():
    import argparse

    import pytest

    for good in ("", "v2", "run-1.final_x"):
        assert audit._report_name(good) == good
    for bad in ("../x", "a/b", "a..b", "x y", "..", "a\\b", "é"):
        with pytest.raises(argparse.ArgumentTypeError):
            audit._report_name(bad)


def test_sun_section_without_a_qualifying_day_does_not_crash():
    starts = grid(date(2025, 7, 1), 3)  # outside every season: no day qualifies
    data = build(make_table(starts))
    sun = audit.outdoor_sensor_sun(data)
    assert sun["days"] == 0 and sun["peak_hour_local"] is None


def test_outdoor_debias_section_reports_difference_shares_and_the_residual_step():
    first, n_days = date(2025, 12, 1), 12
    starts = grid(first, n_days)
    n = starts.size
    truth = 5.0 + 0.0 * np.arange(n)
    freq = np.zeros(n)
    raw = truth.copy()
    for k in range(n_days):  # a 3 h night run every day at 02:00, cooling the sensor by 2 K
        s = k * SLOTS_DAY + 2 * 4
        freq[s : s + 12] = 40.0
        raw[s : s + 12] -= 2.0
        raw[s + 12 : s + 14] -= 1.0
    data = build(make_table(starts, outdoor_temp_mean_c=raw, compressor_freq_mean=freq))
    rep = audit.outdoor_debias(data, CONFIG)["all"]
    assert rep["debiased_minus_raw_k_mean_compressor_running"] > 1.9
    assert abs(rep["debiased_minus_raw_k_mean_compressor_off"]) < 0.5
    assert 0.0 < rep["share_interpolated"] < 0.2 and 0.0 < rep["share_nan"] < 0.01
    steps = audit.outdoor_debias(data, CONFIG)["night_change_over_two_slots_k"]
    assert steps["raw"]["at_compressor_start"]["p50"] < -1.5
    assert abs(steps["debiased"]["at_compressor_start"]["p50"]) < 0.1
    assert abs(steps["debiased"]["at_compressor_stop"]["p50"]) < 0.1


def test_main_writes_only_under_the_data_dir(tmp_path, monkeypatch):
    import hashlib

    import pyarrow.parquet as pq

    root = tmp_path / "store"
    directory = root / "analysis" / "v1" / "features" / "abc123"
    (directory / "parts").mkdir(parents=True)
    (root / "manifests").mkdir()
    table = make_table(grid(date(2025, 12, 1), 6))
    part = directory / "parts" / "part.parquet"
    pq.write_table(table, part)
    entry = {
        "path": "parts/part.parquet",
        "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
        "rows": table.num_rows,
    }
    window = {"start": "2025-11-30T23:00:00Z", "end": "2025-12-06T23:00:00Z"}
    (directory / "features_manifest.json").write_text(
        json.dumps({"feature_set_id": "abc123", "part_files": [entry], "window": window})
    )
    (root / "climate").mkdir()
    (root / "climate" / "site.yaml").write_text("latitude: 40\nlongitude: 0\n")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    real_load = audit.load_climate_data
    monkeypatch.setattr(
        audit, "load_climate_data", lambda cfg, **kw: real_load(cfg, **{**kw, "dataset": None})
    )
    before = {p for p in tmp_path.rglob("*")}
    code = audit.main(
        [
            "--feature-set", "abc",
            "--dataset", str(tmp_path / "unused"),
            "--config", str(CONFIG.path),
            "--data-dir", str(root),
            "--name", "t",
        ]
    )  # fmt: skip
    assert code == 0
    created = {p for p in tmp_path.rglob("*")} - before
    report_dir = root / "analysis" / "v1" / "climate-audit"
    assert created and all(p == report_dir or report_dir in p.parents for p in created)
    assert {p.name for p in created if p.is_file()} == {"report.json", "report.md"}
    assert list(cwd.iterdir()) == []
    assert not (CONFIG.path.parent / "report.json").exists()
