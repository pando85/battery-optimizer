"""Heat-pump electricity (COP) model on synthetic data: no network, no real data directory."""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pytest
import yaml
from climate_fakes import REPO_ROOT, SITE, grid, make_table

from battery_optimizer.climate import cop
from battery_optimizer.climate.config import load_climate_config
from battery_optimizer.climate.data import build_climate_data

TRUE = {"a": 0.10, "b": 0.006, "c": 0.004, "d": 0.07, "e": 0.05}
KNEE = 5.0
#: Three short January seasons so a synthetic grid of a few weeks covers the fold (fit S2, test S3).
SEASON_DAYS = {"S1": date(2024, 1, 1), "S2": date(2024, 1, 11), "S3": date(2024, 1, 21)}
DAYS_PER_SEASON = 8


@pytest.fixture(scope="module")
def config(tmp_path_factory: pytest.TempPathFactory):
    raw = yaml.safe_load((REPO_ROOT / "config" / "climate.yaml").read_text())
    for name, start in SEASON_DAYS.items():
        raw["seasons"][name] = {"start": start, "end": start + timedelta(days=9)}
    raw["seasons"]["fit_months"] = [1]
    raw["seasons"]["gate_months"] = [1]
    raw["seasons"]["reserved_from"] = date(2024, 3, 1)
    path = tmp_path_factory.mktemp("cfg") / "climate.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return load_climate_config(path)


def make_data(
    config,
    *,
    law,
    seed: int = 1,
    noise: float = 0.0,
    t_out_of=None,
    knock_out=None,
    dhw_hours=(),
    cooling_hours=(),
    scale_q=None,
):
    """ClimateData of 28 days from the first season day; ``law(q, t_flow, ...)`` gives E.

    Per hour: heat ``q``, ``k`` compressor-on slots (the first ``k``), flow temperature, starts.
    ``knock_out`` is a set of hour indexes whose de-biased outdoor temperature is unknown.
    Returns ``(data, per-hour truth)``.
    """
    rng = np.random.default_rng(seed)
    first = SEASON_DAYS["S1"]
    n_days = (SEASON_DAYS["S3"] - first).days + DAYS_PER_SEASON
    starts_us = grid(first, n_days)
    n = starts_us.size
    n_hours = n // 4
    q = rng.uniform(0.3, 3.0, n_hours)
    if scale_q is not None:
        q = scale_q(q)
    on_slots = rng.integers(1, 5, n_hours)
    t_flow = rng.uniform(25.0, 40.0, n_hours)
    starts = rng.integers(0, 3, n_hours).astype(float)
    hour_of_day = (np.arange(n_hours) % 24).astype(float)
    t_out = (
        t_out_of(hour_of_day, rng)
        if t_out_of
        else 6.0 + 7.0 * np.sin(np.arange(n_hours) / 11.0) + rng.normal(0, 1, n_hours)
    )
    hours_on = on_slots * 0.25
    e = law(q, t_flow, t_out, starts, hours_on)
    e = e + rng.normal(0.0, noise, n_hours)

    slot = np.arange(4)[None, :]
    on = slot < on_slots[:, None]
    q_slot = np.where(on, (q / on_slots)[:, None], 0.0)
    e_slot = np.where(on, (e / on_slots)[:, None], 0.0)
    freq = np.where(on, 40.0, 0.0)
    cols = {
        "hp_heat_out_kwh": q_slot.reshape(-1),
        "hp_heat_kwh": e_slot.reshape(-1),
        "flow_temp_mean_c": np.repeat(t_flow, 4),
        "compressor_freq_mean": freq.reshape(-1),
        "compressor_starts_delta": np.repeat(starts / 4.0, 4),
        "outdoor_temp_mean_c": np.repeat(t_out, 4),
    }
    for hour in dhw_hours:
        cols["hp_dhw_kwh"] = cols.get("hp_dhw_kwh", np.zeros(n))
        cols["hp_dhw_out_kwh"] = cols.get("hp_dhw_out_kwh", np.zeros(n))
        cols["hp_dhw_kwh"][hour * 4 : hour * 4 + 4] = 0.3
        cols["hp_dhw_out_kwh"][hour * 4 : hour * 4 + 4] = 0.9
    for hour in cooling_hours:
        cols["hp_cool_kwh"] = cols.get("hp_cool_kwh", np.zeros(n))
        cols["hp_cool_kwh"][hour * 4 : hour * 4 + 4] = 0.2
    data = build_climate_data(make_table(starts_us, **cols), config=config, site=SITE)
    debiased = np.repeat(t_out, 4)
    for hour in knock_out or ():
        debiased[hour * 4 : hour * 4 + 4] = np.nan
    data.t_out_debiased = debiased
    truth = {"q": q, "e": e, "t_flow": t_flow, "t_out": t_out, "starts": starts, "on": hours_on}
    return data, truth


def true_law(q, t_flow, t_out, starts, on):
    a, b, c, d, e = (TRUE[k] for k in "abcde")
    return q * (a + b * (t_flow - t_out) + c * np.maximum(0.0, KNEE - t_out)) + d * starts + e * on


def step_law(q, t_flow, t_out, starts, on):
    """COP is a step in T_out: a table follows it, the linear-lift regression cannot."""
    return q / np.where(t_out > 6.0, 5.5, 2.5)


# ---------------------------------------------------------------------------
# Recovery and selection
# ---------------------------------------------------------------------------


def test_regression_recovers_known_parameters_and_beats_the_table(config):
    # T_out is held clear of the knee region's edge so each term is identifiable
    data, _ = make_data(config, law=true_law, noise=0.002)
    report = cop.fit_and_validate(data, config)
    assert [(f["train"], f["test"]) for f in report["folds"]] == [(["S2"], "S3")]
    for fold in report["folds"]:
        coef = fold["model"]["coefficients"]
        for name, value in TRUE.items():
            assert coef[name] == pytest.approx(value, rel=0.15, abs=0.01), name
        # the model is predeclared (cop.model); the held-out best is reported beside it
        assert fold["selected"] == "lift_table" == fold["model"]["selected"]
        assert "predeclared" in fold["selection_reason"]
        assert fold["held_out_best"] == "regression"
        assert fold["validation"]["regression"]["passes"] is True
        assert fold["selected_passes"] == fold["validation"]["lift_table"]["passes"]
        assert fold["validation"]["regression"]["daily_mape"] < 0.02
        assert (
            fold["validation"]["table"]["daily_mape"]
            > fold["validation"]["regression"]["daily_mape"]
        )
        assert fold["sanity"]["all_ok"] is True
        assert all(t["ok"] for t in fold["sanity"]["terms"].values())


def test_d_and_e_sanity_check_flags_an_implausible_term(config):
    # a negative running overhead is not physical: the check must say so
    bad = {**TRUE, "e": -0.3}

    def law(q, t_flow, t_out, starts, on):
        return q * (bad["a"] + bad["b"] * (t_flow - t_out)) + bad["d"] * starts + bad["e"] * on

    data, _ = make_data(config, law=lambda *x: np.maximum(law(*x), 0.01), noise=0.0)
    terms = cop.fit_and_validate(data, config)["folds"][0]["sanity"]["terms"]
    assert terms["e"]["ok"] is False and terms["d"]["ok"] is True


def test_the_table_is_reported_best_when_the_regression_loses(config):
    data, _ = make_data(config, law=step_law, noise=0.0)
    report = cop.fit_and_validate(data, config)
    for fold in report["folds"]:
        assert fold["held_out_best"] == "table" and fold["selected"] == "lift_table"
        reg, tab = fold["validation"]["regression"], fold["validation"]["table"]
        assert tab["daily_mape"] < reg["daily_mape"]
        model = cop.CopModel.from_dict(fold["model"]).with_selected("table")
        # the simulator API answers with the table, whatever flow or starts it is given
        got = cop.predict_electricity(model, 2.0, 30.0, np.array([3.0, 10.0]), 9, 9)
        np.testing.assert_allclose(got, [2.0 / 2.5, 2.0 / 5.5])
        np.testing.assert_allclose(cop.predict_electricity(model, 2.0, 99.0, 12.0, 0, 0), 2.0 / 5.5)


def test_held_out_best_is_by_daily_mape_with_a_tie_to_the_earlier_candidate():
    reg = {"daily_mape": 0.05}
    assert cop.select_model({"regression": reg, "table": {"daily_mape": 0.05}})[0] == "regression"
    assert cop.select_model({"regression": reg, "table": {"daily_mape": 0.049}})[0] == "table"
    three = {"regression": reg, "table": {"daily_mape": 0.04}, "lift_table": {"daily_mape": 0.04}}
    assert cop.select_model(three)[0] == "table"
    three["lift_table"] = {"daily_mape": 0.03}
    assert cop.select_model(three)[0] == "lift_table"


def test_validation_limits_come_from_the_config(config):
    data, _ = make_data(config, law=true_law, noise=0.002)
    fold = cop.fit_and_validate(data, config)["folds"][0]
    assert fold["limits"] == {
        "daily_mape": config.cop.daily_mape,
        "season_error": config.cop.season_error,
    }
    rows = cop.hourly_rows(data, config)
    model = cop.CopModel.from_dict(fold["model"])
    test = rows.select((rows.season == "S2") & rows.gate_month)
    good = cop.validate(model, test, config)["regression"]
    assert good["passes"] and abs(good["season_error"]) < 0.01
    # a model that predicts 10 % too much fails the 3 % season limit and the 8 % daily limit
    inflated = cop.CopModel.from_dict(
        {
            **fold["model"],
            "coefficients": {k: 1.1 * v for k, v in fold["model"]["coefficients"].items()},
        }
    )
    bad = cop.validate(inflated, test, config)["regression"]
    assert bad["season_error"] == pytest.approx(0.10, abs=0.01)
    assert bad["daily_mape"] == pytest.approx(0.10, abs=0.01)
    assert bad["passes"] is False


def test_residuals_by_t_out_bin_cover_every_validation_hour(config):
    data, _ = make_data(config, law=true_law, noise=0.002)
    fold = cop.fit_and_validate(data, config)["folds"][0]
    for name in ("regression", "table", "lift_table"):
        bins = fold["validation"][name]["residuals_by_t_out"]
        assert sum(b["hours"] for b in bins) == fold["validation"][name]["hours"]
        edges = [b["t_out_from_c"] for b in bins]
        assert edges == sorted(edges) and all(e % config.cop.table_bin_k == 0 for e in edges)
        total_pred = sum(b["predicted_kwh"] for b in bins)
        assert total_pred > 0


# ---------------------------------------------------------------------------
# Exclusions
# ---------------------------------------------------------------------------


def test_exclusion_filters_and_their_accounting(config):
    n_hours = 28 * 24

    def scale_q(q):
        q = q.copy()
        q[10:20] = 0.05  # below min_q_kwh
        return q

    def law(q, t_flow, t_out, starts, on):
        e = true_law(q, t_flow, t_out, starts, on)
        e[30:35] = q[30:35] / 12.0  # implied COP above the range
        e[40:45] = q[40:45] / 0.5  # implied COP below the range
        return e

    data, _ = make_data(
        config,
        law=law,
        scale_q=scale_q,
        knock_out={50, 51, 52},
        dhw_hours={60, 61},
        cooling_hours={70},
    )
    rows = cop.hourly_rows(data, config)
    ex = rows.exclusions
    assert ex["hours"] == n_hours
    assert ex["dhw_or_cooling"] == 3
    assert ex["outdoor_unknown"] == 3
    assert ex["q_below_min"] == 10
    assert ex["cop_out_of_range"] == 10
    assert ex["eligible"] == rows.n == n_hours - 3 - 3 - 10 - 10
    # every kept row is inside the thresholds
    assert (rows.q >= config.cop.min_q_kwh).all()
    lo, hi = config.cop.cop_range
    assert ((rows.q / rows.e >= lo) & (rows.q / rows.e <= hi)).all()
    kept_hours = set((rows.hour_start_us - data.starts_us[0]) // 3_600_000_000)
    assert kept_hours.isdisjoint({10, 15, 19, 30, 34, 40, 44, 50, 51, 52, 60, 61, 70})


def test_dhw_hours_do_not_leak_into_the_fit(config):
    dhw = set(range(100, 140))
    data, _ = make_data(config, law=true_law, noise=0.001, seed=5, dhw_hours=dhw)
    for hour in dhw:  # electricity of a hot-water hour is not heat-mode electricity
        data.e_heat_kwh[hour * 4 : hour * 4 + 4] *= 5.0
    rows = cop.hourly_rows(data, config)
    kept = set(((rows.hour_start_us - data.starts_us[0]) // 3_600_000_000).astype(int).tolist())
    assert kept.isdisjoint(dhw)
    assert rows.exclusions["dhw_or_cooling"] == len(dhw)
    fit = cop.fit_regression(rows, KNEE)["coefficients"]
    for got, name in zip(fit, "abcde", strict=True):
        assert got == pytest.approx(TRUE[name], rel=0.15, abs=0.01), name


def test_unknown_inputs_drop_the_hour_instead_of_zero_filling(config):
    data, _ = make_data(config, law=true_law, noise=0.0)
    before = cop.hourly_rows(data, config).n
    data.columns["compressor_starts_delta"][5 * 4 + 1] = np.nan
    data.columns["flow_temp_mean_c"][8 * 4] = np.nan  # a heat-flowing slot without a flow reading
    rows = cop.hourly_rows(data, config)
    assert rows.exclusions["unknown_input"] >= 1
    assert rows.n <= before - 1


def test_hourly_aggregation_sums_energy_and_counts_on_time(config):
    data, truth = make_data(config, law=true_law, noise=0.0, seed=9)
    rows = cop.hourly_rows(data, config)
    first = ((rows.hour_start_us - data.starts_us[0]) // 3_600_000_000).astype(int)
    np.testing.assert_allclose(rows.q, truth["q"][first])
    np.testing.assert_allclose(rows.e, truth["e"][first])
    np.testing.assert_allclose(rows.hours_on, truth["on"][first])
    np.testing.assert_allclose(rows.starts, truth["starts"][first], atol=1e-12)
    np.testing.assert_allclose(rows.t_flow, truth["t_flow"][first])
    np.testing.assert_allclose(rows.t_out, truth["t_out"][first])


def test_hourly_flow_temperature_is_weighted_by_heat(config):
    data, _ = make_data(config, law=true_law, noise=0.0, seed=9)
    j = 30  # one hour: heat 1 and 3 kWh in the first two slots at 20 and 40 C, nothing after
    sl = slice(j * 4, j * 4 + 4)
    data.q_heat_kwh[sl] = [1.0, 3.0, 0.0, 0.0]
    data.e_heat_kwh[sl] = [0.25, 0.75, 0.0, 0.0]
    data.columns["flow_temp_mean_c"][sl] = [20.0, 40.0, 33.0, np.nan]  # idle slots do not count
    rows = cop.hourly_rows(data, config)
    at = int(np.flatnonzero(rows.hour_start_us == data.starts_us[j * 4])[0])
    assert rows.t_flow[at] == pytest.approx((1.0 * 20.0 + 3.0 * 40.0) / 4.0)
    assert rows.q[at] == pytest.approx(4.0)


def test_season_and_month_membership_follows_the_config(config):
    data, _ = make_data(config, law=true_law)
    rows = cop.hourly_rows(data, config)
    assert set(np.unique(rows.season)) == {"S1", "S2", "S3"}
    # the days after each season's 8 synthetic days and before the next season are not in a season
    assert rows.fit_month.all() and rows.gate_month.all()


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def _rows(t_out, q, e):
    n = len(t_out)
    z = np.zeros(n)
    return cop.HourlyRows(
        hour_start_us=np.arange(n),
        day=np.zeros(n, dtype=int),
        season=np.array(["S1"] * n, dtype=object),
        fit_month=np.ones(n, bool),
        gate_month=np.ones(n, bool),
        q=np.asarray(q, float),
        e=np.asarray(e, float),
        t_flow=z + 30.0,
        t_out=np.asarray(t_out, float),
        starts=z,
        hours_on=z,
        exclusions={},
    )


def test_table_bins_are_two_kelvin_steps_with_heat_over_electricity_ratios():
    t = np.repeat([0.5, 1.5, 3.1], cop.MIN_BIN_ROWS)
    q = np.concatenate([np.full(10, 2.0), np.full(10, 4.0), np.full(10, 3.0)])
    e = np.concatenate([np.full(10, 1.0), np.full(10, 1.0), np.full(10, 1.0)])
    table = cop.fit_table(_rows(t, q, e), 2.0)
    assert table == ((0.0, pytest.approx(3.0), 20), (2.0, pytest.approx(3.0), 10))


def test_sparse_bins_are_left_out_and_read_the_nearest_populated_bin():
    t = np.concatenate([np.full(cop.MIN_BIN_ROWS, 1.0), np.full(cop.MIN_BIN_ROWS, 9.0), [5.0]])
    q = np.concatenate([np.full(cop.MIN_BIN_ROWS, 2.0), np.full(cop.MIN_BIN_ROWS, 4.0), [1.0]])
    e = np.ones(t.size)
    table = cop.fit_table(_rows(t, q, e), 2.0)
    assert [row[0] for row in table] == [0.0, 8.0]
    model = _model(table=table)
    # 5 C is in the empty bin [4, 6): nearest centre is 1.0 (distance 4) vs 9.0 (distance 4):
    # a tie resolves to the first (colder) bin; far outside the range reads the end bins
    np.testing.assert_allclose(model.cop_table([-20.0, 1.0, 9.0, 40.0]), [2.0, 2.0, 4.0, 4.0])
    assert np.isnan(model.cop_table(np.nan))


def test_lift_table_bins_flow_minus_outdoor_with_the_same_rules():
    # lift 30 - t_out: 21 K and 23 K fill two bins, one hour at 27 K is too sparse
    t_out = np.concatenate([np.full(10, 9.0), np.full(10, 7.0), [3.0]])
    q = np.concatenate([np.full(10, 4.0), np.full(10, 3.0), [1.0]])
    rows = _rows(t_out, q, np.ones(t_out.size))
    table = cop.fit_lift_table(rows, 2.0)
    assert table == ((20.0, pytest.approx(4.0), 10), (22.0, pytest.approx(3.0), 10))
    model = _model(lift_table=table, selected="lift_table")
    # 27 K reads the nearest populated bin (centre 23); far below reads the first bin
    np.testing.assert_allclose(model.cop_lift([21.0, 23.5, 27.0, 5.0]), [4.0, 3.0, 3.0, 4.0])
    assert np.isnan(model.cop_lift(np.nan))
    # the simulator API reads q and the lift; starts and on-time do not enter a table
    got = cop.predict_electricity(model, 6.0, np.array([30.0, 35.0]), 9.0, 9, 9)
    np.testing.assert_allclose(got, [6.0 / 4.0, 6.0 / 3.0])
    with pytest.raises(cop.CopError, match="lift"):
        cop.fit_lift_table(_rows([1.0], [1.0], [1.0]), 2.0)


def test_lift_table_follows_a_flow_change_the_t_out_table_cannot(config):
    # COP falls with the lift only; the test season runs the water at the top of the training range
    def law(q, t_flow, t_out, starts, on):
        return q / (8.0 - 0.15 * (t_flow - t_out))

    data, _ = make_data(
        config, law=law, noise=0.0, t_out_of=lambda h, rng: rng.uniform(2, 12, h.size)
    )
    s3 = data.day_mask(data.day_season == "S3")
    flow = data.columns["flow_temp_mean_c"]
    flow[s3] = 38.0
    data.e_heat_kwh[:] = data.q_heat_kwh / (8.0 - 0.15 * (flow - data.t_out_debiased))
    fold = cop.fit_and_validate(data, config)["folds"][0]
    v = fold["validation"]
    assert v["lift_table"]["daily_mape"] < 0.03 and v["lift_table"]["passes"]
    assert v["table"]["season_error"] < -0.10 and not v["table"]["passes"]


def test_no_populated_bin_is_an_error():
    with pytest.raises(cop.CopError):
        cop.fit_table(_rows([1.0, 2.0], [1.0, 1.0], [1.0, 1.0]), 2.0)


# ---------------------------------------------------------------------------
# Parameter record and prediction API
# ---------------------------------------------------------------------------


def _model(**over):
    base = {
        "version": cop.COP_MODEL_VERSION,
        "selected": "regression",
        "coefficients": (0.1, 0.006, 0.004, 0.07, 0.05),
        "knee_c": 5.0,
        "table_bin_k": 2.0,
        "table": ((0.0, 3.0, 20), (2.0, 4.0, 20)),
        "cop_range": (1.0, 9.0),
        "train_seasons": ("S1",),
        "n_train_rows": 40,
        "feature_set_id": "fs",
        "config_sha256": "cfg",
        "lift_bin_k": 2.0,
        "lift_table": ((20.0, 4.0, 20), (22.0, 3.5, 20), (30.0, 2.5, 20)),
    }
    return cop.CopModel(**{**base, **over})


def test_model_record_is_versioned_hashable_and_round_trips_through_json():
    m = _model()
    assert m.version == cop.COP_MODEL_VERSION == 2
    assert hash(m) == hash(_model()) and {m, _model()} == {m}
    again = cop.CopModel.from_dict(json.loads(json.dumps(m.to_dict())))
    assert again == m and again.model_id == m.model_id
    assert len(m.model_id) == 64
    # any change of a fitted value, of the choice or of the lineage is a different model
    ids = {
        m.model_id,
        _model(selected="table").model_id,
        _model(coefficients=(0.1, 0.006, 0.004, 0.07, 0.051)).model_id,
        _model(config_sha256="other").model_id,
        _model(table=((0.0, 3.1, 20), (2.0, 4.0, 20))).model_id,
        _model(lift_table=((20.0, 3.0, 20), (22.0, 2.9, 20))).model_id,
    }
    assert len(ids) == 6
    with pytest.raises(cop.CopError):
        cop.CopModel.from_dict({**m.to_dict(), "version": 99})
    with pytest.raises(cop.CopError):
        _model(selected="neither")
    with pytest.raises(cop.CopError, match="lift table is empty"):
        _model(selected="lift_table", lift_table=())


def test_predict_electricity_matches_the_formula_broadcasts_and_never_goes_negative():
    m = _model()
    q, tf, to, st, on = 2.0, 35.0, 2.0, 1.0, 0.75
    expect = q * (0.1 + 0.006 * (tf - to) + 0.004 * (5.0 - to)) + 0.07 * st + 0.05 * on
    assert cop.predict_electricity(m, q, tf, to, st, on) == pytest.approx(expect)
    # above the knee the defrost term vanishes
    hot = cop.predict_electricity(m, q, tf, 12.0, 0.0, 0.0)
    assert hot == pytest.approx(q * (0.1 + 0.006 * (tf - 12.0)))
    many = cop.predict_electricity(
        m, np.array([1.0, 2.0]), 30.0, np.array([[0.0], [10.0]]), 0.0, 0.0
    )
    assert many.shape == (2, 2)
    neg = _model(coefficients=(0.1, 0.0, 0.0, -5.0, 0.0))
    assert cop.predict_electricity(neg, 1.0, 30.0, 5.0, 1.0, 0.0) == 0.0
    table = _model(selected="table")
    assert cop.predict_electricity(table, 6.0, 99.0, 1.0) == pytest.approx(2.0)  # 6 kWh / COP 3


def test_with_selected_changes_only_the_choice():
    m = _model()
    t = m.with_selected("table")
    assert t.selected == "table" and t.coefficients == m.coefficients and t.table == m.table


# ---------------------------------------------------------------------------
# DHW COP and report
# ---------------------------------------------------------------------------


def test_dhw_cop_is_reported_per_season_for_information(config):
    data, _ = make_data(config, law=true_law, dhw_hours=set(range(0, 48)))
    out = cop.dhw_cop(data, ("S1", "S2", "S3"))
    assert out["S1"]["cop"] == pytest.approx(3.0)  # 0.9 / 0.3 in each DHW slot
    assert out["S1"]["slots"] == 48 * 4
    assert out["S3"]["cop"] is None and out["S3"]["slots"] == 0


def test_report_is_json_ready_and_renders(config):
    data, _ = make_data(config, law=true_law, noise=0.002)
    report = cop.fit_and_validate(data, config)
    json.dumps(report, allow_nan=False)
    text = cop.render_markdown(report)
    assert "regression" in text and "table" in text and "Hot water COP" in text
    assert "selected" in text and "Residuals by outdoor temperature bin" in text
    assert (
        report["folds"][0]["model_id"]
        == cop.CopModel.from_dict(report["folds"][0]["model"]).model_id
    )
