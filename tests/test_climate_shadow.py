"""Live climate shadow (track B2): inputs, decisions, scoring, persistence and the job."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from climate_shadow_fakes import (
    CCFG,
    MPC,
    SETTINGS,
    SITE,
    FakeSamples,
    forecast_record,
    house,
    prior,
    utc,
)

from battery_optimizer.climate import b2d, mpc_replay, shadow, thermal
from battery_optimizer.climate.mpc_model import Exogenous
from battery_optimizer.service.climate_shadow import (
    ClimateShadowJob,
    latest_forecast,
    next_run_time,
)
from battery_optimizer.service.config import ClimateShadowSettings
from battery_optimizer.service.sink import ShadowSink, iter_records

TZ = ZoneInfo("Europe/Madrid")
# Monday 2026-11-16, local midnight is 23:00 UTC the day before
DAY = date(2026, 11, 16)
MIDNIGHT = utc(2026, 11, 15, 23)


def live_data(t0: datetime, days: int = 4, samples=None):
    start = t0 - timedelta(days=days)
    # samples up to and including t0, as the job's cut keeps them
    series = samples if samples is not None else house(start, t0 + timedelta(minutes=5))
    return shadow.build_live_data(
        series, start=start, end=t0 + timedelta(hours=24), config=CCFG, site=SITE
    )


def model_rec(sp):
    return {
        "params": dict(sp.prior.params),
        "model_hash": shadow.model_hash(sp.prior_id, sp.prior.params),
    }


# ---------------------------------------------------------------------------
# Regime and Home Assistant inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("state", "exists", "synced", "expected"),
    [
        (None, False, True, "legacy"),
        ("on", True, True, "comfort"),
        ("off", True, True, "comfort_off"),
        ("unavailable", True, True, "unknown"),
        ("on", True, False, "unknown"),
        (None, False, False, "unknown"),
    ],
)
def test_regime_detection(state, exists, synced, expected):
    assert shadow.detect_regime(state, exists=exists, synced=synced) == expected


def _slots(start: datetime, n: int) -> np.ndarray:
    return np.array([int((start + timedelta(minutes=15 * k)).timestamp() * 1e6) for k in range(n)])


@pytest.mark.parametrize("text", [None, "", "unknown", "unavailable", "garbage", "25:00-26:00"])
def test_a_missing_or_unreadable_dhw_plan_is_none(text):
    assert shadow.parse_dhw_plan(text, starts_us=_slots(MIDNIGHT, 96), tz=TZ) is None


def test_dhw_plan_formats():
    slots = _slots(MIDNIGHT, 96)  # local 00:00 .. 23:45
    iso = "2026-11-16T02:00:00+01:00/2026-11-16T03:00:00+01:00"
    mask = shadow.parse_dhw_plan(iso, starts_us=slots, tz=TZ)
    assert np.flatnonzero(mask).tolist() == [8, 9, 10, 11]
    two = iso + ", 2026-11-16T14:00:00+01:00/2026-11-16T14:30:00+01:00"
    assert shadow.parse_dhw_plan(two, starts_us=slots, tz=TZ).sum() == 6
    js = json.dumps({"start": "2026-11-16T02:00:00+01:00", "end": "2026-11-16T03:00:00+01:00"})
    assert np.array_equal(shadow.parse_dhw_plan(js, starts_us=slots, tz=TZ), mask)
    daily = shadow.parse_dhw_plan("02:00-03:00", starts_us=slots, tz=TZ)
    assert np.array_equal(daily, mask)
    wrap = shadow.parse_dhw_plan("23:30-00:30", starts_us=slots, tz=TZ)
    assert np.flatnonzero(wrap).tolist() == [0, 1, 94, 95]
    naive = "2026-11-16T02:00/2026-11-16T03:00"  # no offset: refused
    assert shadow.parse_dhw_plan(naive, starts_us=slots, tz=TZ) is None


def test_forecast_series_interpolates_and_holds_the_ends():
    rec = {
        "forecast": [
            {"datetime": "2026-11-16T00:00:00+00:00", "temperature": 4.0},
            {"datetime": "2026-11-16T01:00:00+00:00", "temperature": 8.0},
            {"datetime": "bad", "temperature": 1.0},
        ]
    }
    out = shadow.forecast_series(rec, _slots(utc(2026, 11, 15, 23, 30), 8))
    # midpoints 23:37.5 (held first), ..., 00:07.5 -> 4.5, 00:52.5 -> 7.5, 01:22.5 (held last)
    assert out[0] == 4.0 and out[2] == pytest.approx(4.5) and out[5] == pytest.approx(7.5)
    assert out[-1] == 8.0
    assert np.isnan(shadow.forecast_series(None, _slots(MIDNIGHT, 2))).all()


def test_the_latest_archived_forecast_is_read(tmp_path):
    sink = ShadowSink(tmp_path)
    old, new = utc(2026, 11, 15, 20, 5), utc(2026, 11, 15, 22, 5)
    sink.append_weather_forecast(old, forecast_record(old))
    sink.append_weather_forecast(new, forecast_record(new))
    later = utc(2026, 11, 15, 23, 5)
    sink.append_weather_forecast(later, forecast_record(later))  # after the run: never read
    got = latest_forecast(
        tmp_path, "weather.forecast_home", utc(2026, 11, 15, 23), timedelta(hours=3)
    )
    assert got["requested_at"] == new.isoformat()
    too_old = latest_forecast(
        tmp_path, "weather.forecast_home", utc(2026, 11, 16, 6), timedelta(hours=3)
    )
    assert too_old is None


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def test_a_decision_plans_from_the_forecast_and_reads_nothing_after_t0():
    sp = prior()
    t0 = MIDNIGHT + timedelta(hours=10)
    d = live_data(t0)
    fc = forecast_record(t0 - timedelta(minutes=55))
    inputs = shadow.LiveInputs(regime="legacy", forecast=fc)
    rec = shadow.decide(sp, model_rec(sp), d, MPC, t0, inputs, settings=SETTINGS)
    assert rec["status"] == "planned", rec.get("status")
    assert rec["inputs"]["weather"] == "forecast"
    assert rec["inputs"]["dhw_source"] == "history"  # no plan helper: yesterday's DHW slots
    # 03:00 UTC is local 04:00: 16 slots after t0 = 10:00 local is the next day's 04:00
    assert rec["inputs"]["dhw_slots"] == [72]
    plan = rec["plan"]
    assert len(plan["plan_on"]) == 96 and len(plan["plan_offset_k"]) == 96
    assert set(plan["plan_offset_k"]) <= set(MPC.water.offsets_k)
    assert len(rec["origin"]["x0"]) == 2
    # measured data after t0 changes nothing
    later = live_data(t0, samples=_perturbed_after(t0))
    rec2 = shadow.decide(sp, model_rec(sp), later, MPC, t0, inputs, settings=SETTINGS)
    assert rec2["origin"] == rec["origin"] and rec2["inputs"] == rec["inputs"]


def _perturbed_after(t0):
    data = house(t0 - timedelta(days=4), t0 + timedelta(hours=6))
    cut = int(t0.timestamp() * 1e6)
    for s in data.values():
        s.values[s.t_us >= cut] += 5.0
    return data


def test_the_plan_helper_and_away_reach_the_plan():
    sp = prior()
    t0 = MIDNIGHT + timedelta(hours=10)
    d = live_data(t0)
    inputs = shadow.LiveInputs(regime="comfort", dhw_plan="12:00-13:00", away=True)
    rec = shadow.decide(sp, model_rec(sp), d, MPC, t0, inputs, settings=SETTINGS)
    assert rec["inputs"]["weather"] == "persistence"  # no forecast archived
    assert rec["inputs"]["dhw_source"] == "plan"
    assert rec["inputs"]["dhw_slots"] == [8, 9, 10, 11]
    floor = MPC.comfort.floor_c + MPC.comfort.setpoint_margin_k
    assert all(v == floor for row in rec["inputs"]["setpoints"] for v in row)


def test_without_away_the_day_room_gets_its_optimum_start_setpoint():
    sp = prior()
    t0 = MIDNIGHT  # Monday 00:00: the 12:00 window, leads of hours before it
    d = live_data(t0)
    rec = shadow.decide(
        sp, model_rec(sp), d, MPC, t0, shadow.LiveInputs(regime="legacy"), settings=SETTINGS
    )
    hall = list(sp.prior.rooms).index("hall_kitchen")
    day_sp = MPC.comfort.day_min_c + MPC.comfort.setpoint_margin_k
    row = rec["inputs"]["setpoints"][hall]
    assert row[47] == day_sp and row[-1] < day_sp  # 11:45 in the early start; 23:45 not
    assert rec["midnight"] is True


# ---------------------------------------------------------------------------
# Scoring: hand-computed values
# ---------------------------------------------------------------------------


def test_scoring_due_waits_for_the_day_of_the_last_horizon_slot():
    t0 = MIDNIGHT + timedelta(hours=10)  # last slot 09:45 local the next day
    assert not shadow.scoring_due(t0, 24, MIDNIGHT + timedelta(days=1, hours=23), TZ)
    assert shadow.scoring_due(t0, 24, MIDNIGHT + timedelta(days=2), TZ)
    assert shadow.scoring_due(MIDNIGHT, 24, MIDNIGHT + timedelta(days=1), TZ)


def test_the_room_mean_error_is_the_heat_driven_prediction_plus_the_room_offsets():
    sp = prior({"office_a": 1.2, "bedroom": -0.4})
    t0 = MIDNIGHT + timedelta(hours=10)
    d0 = live_data(t0)
    dec = shadow.decide(
        sp, model_rec(sp), d0, MPC, t0, shadow.LiveInputs(regime="legacy"), settings=SETTINGS
    )
    d = live_data(t0 + timedelta(days=2), days=6)
    rec = shadow.score(sp, dec, d, MPC, shadow.DESIGN_ROOMS)
    assert rec is not None and rec["action"] == "permission"
    # by hand: simulate the model from x0 with the measured heat and outdoor temperature
    i = int(np.searchsorted(d.starts_us, int(t0.timestamp() * 1e6)))
    model = sp.prior.model()
    t_out = np.where(np.isfinite(d.t_out_model), d.t_out_model, mpc_replay.causal_outdoor(d))
    w = model.inputs(
        np.where(d.dhw, 0.0, d.q_heat_kw)[i : i + 96],
        Exogenous(
            t_out[i : i + 96],
            d.solar[i : i + 96],
            b2d.diurnal_matrix(*b2d.slot_calendar(d), 2)[i : i + 96],
            d.local_hour[i : i + 96],
            np.zeros((3, 96)),
        ),
    )
    xs = thermal.simulate(model.ad, w @ model.bd.T, np.array(dec["origin"]["x0"]))
    rooms = np.vstack([d.room_temp[r] for r in shadow.DESIGN_ROOMS]).mean(axis=0)
    offset = (1.2 - 0.4) / 4
    for hh in (1, 6, 12, 24):
        j = 4 * hh - 1
        cell = rec["room_mean"][f"h{hh}"]
        assert cell["heat"] == pytest.approx(xs[j + 1, 0] + offset - rooms[i + j], abs=1e-9)
        assert cell["persistence"] == pytest.approx(rooms[i - 1] - rooms[i + j], abs=1e-9)
    # the same, with the weather the plan used (no forecast archived: the reading held)
    assert dec["inputs"]["weather"] == "persistence"
    assert rec["room_mean"]["h6"]["heat_plan_weather"] is not None


def test_a_decision_without_an_origin_is_not_scored():
    assert shadow.score(prior(), {"t0": MIDNIGHT.isoformat()}, None, MPC, ["bedroom"]) is None


def test_day_validity_follows_the_preregistration():
    d = live_data(MIDNIGHT + timedelta(days=2), days=4)
    j = next(k for k, (label, _l, _h) in enumerate(d.days) if label == DAY)
    rec = shadow.day_record(d, j, shadow.DESIGN_ROOMS)
    assert rec["valid_heating_day"] and rec["space_heat_slots"] > 0
    assert rec["coverage"] == {"rooms": 1.0, "energy": 1.0, "outdoor": 1.0}
    # without the office-A thermostat the design room-mean is unknown: not a valid day
    series = house(MIDNIGHT - timedelta(days=2), MIDNIGHT + timedelta(days=2))
    key = next(s.source for s in shadow.LIVE_SIGNALS if s.alias == "room_office_a_temp")
    s = series[key]
    keep = s.t_us < int((MIDNIGHT + timedelta(hours=20)).timestamp() * 1e6)
    series[key] = type(s)(s.entity, s.metric, s.unit, s.t_us[keep], s.values[keep])
    d2 = live_data(MIDNIGHT + timedelta(days=2), samples=series)
    j2 = next(k for k, (label, _l, _h) in enumerate(d2.days) if label == DAY)
    bad = shadow.day_record(d2, j2, shadow.DESIGN_ROOMS)
    assert bad["coverage"]["rooms"] < 0.9 and not bad["valid_heating_day"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_the_prior_round_trips_and_is_content_addressed(tmp_path):
    sp = prior({"office_a": 1.5})
    path = shadow.save_prior(tmp_path, sp)
    assert path.parent.name == sp.prior_id
    back = shadow.load_prior(tmp_path, sp.prior_id[:6])
    assert back.to_dict() == sp.to_dict() and back.prior_id == sp.prior_id
    assert back.gate_offsets(["office_a", "bedroom"])[0] == pytest.approx(0.75)
    tampered = json.loads(path.read_text())
    tampered["params"]["ua"] = 0.2
    path.chmod(0o600)
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="hashes to"):
        shadow.load_prior(tmp_path, sp.prior_id)


def test_climate_records_round_trip_through_the_sink(tmp_path):
    sink = ShadowSink(tmp_path)
    at = utc(2026, 11, 16, 0, 10)
    rec = {"kind": "climate_score", "t0": at.isoformat(), "room_mean": {"h6": {"heat": 0.1}}}
    sink.append_climate("climate_scores", at, rec)
    with pytest.raises(ValueError):
        sink.append_climate("plans", at, rec)
    got = list(iter_records(tmp_path, "climate_scores", at, at + timedelta(hours=1)))
    assert got == [{"schema_version": 1, **rec}]


# ---------------------------------------------------------------------------
# The job with fake data sources
# ---------------------------------------------------------------------------


def make_job(tmp_path, samples, **settings):
    return ClimateShadowJob(
        settings=ClimateShadowSettings(enabled=True, prior="aaaaaa", **settings),
        climate_config=CCFG,
        mpc_config=MPC,
        prior=prior(),
        site=SITE,
        sink=ShadowSink(tmp_path),
        data_dir=tmp_path,
        samples=samples,
        code_version="test",
        plan_settings=SETTINGS,
    )


def read(tmp_path, kind):
    return list(iter_records(tmp_path, kind, utc(2026, 10, 1), utc(2027, 1, 1)))


def test_next_run_is_the_boundary_plus_the_offset():
    assert next_run_time(utc(2026, 11, 16, 9, 59), timedelta(hours=1), 10) == utc(
        2026, 11, 16, 10, 10
    )
    assert next_run_time(utc(2026, 11, 16, 10, 10), timedelta(hours=1), 10) == utc(
        2026, 11, 16, 11, 10
    )


def test_the_job_decides_refits_scores_and_resumes(tmp_path):
    samples = FakeSamples(MIDNIGHT - timedelta(days=12), MIDNIGHT + timedelta(days=3))
    job = make_job(tmp_path, samples)
    legacy = shadow.LiveInputs(regime="legacy")
    runs = [MIDNIGHT, MIDNIGHT + timedelta(hours=10), MIDNIGHT + timedelta(days=1)]
    for at in runs:
        summary = job.run_once(at + timedelta(minutes=10), legacy)
        assert summary["status"] == "planned", summary
    assert job.run_once(runs[-1] + timedelta(minutes=20), legacy)["status"] == "skipped"
    decisions = read(tmp_path, "climate_decisions")
    assert [d["t0"] for d in decisions] == [r.isoformat() for r in runs]
    assert all(d["regime"] == "legacy" and d["code_version"] == "test" for d in decisions)
    models = read(tmp_path, "climate_models")
    assert len(models) == 1 and models[0]["window"]["days"] == 7
    assert decisions[0]["model_hash"] == models[0]["model_hash"]
    # the midnight origin of DAY is scored at the next midnight, the 10:00 one not yet
    scores = read(tmp_path, "climate_scores")
    assert [s["t0"] for s in scores] == [MIDNIGHT.isoformat()]
    assert "day" in scores[0]["accuracy"] and scores[0]["plan"]["kwh"] is not None
    days = read(tmp_path, "climate_days")
    assert [d["day"] for d in days] == [DAY.isoformat()] and days[0]["regimes"] == ["legacy"]
    # a restarted job resumes the model and the queue, and scores the rest when due
    again = make_job(tmp_path, samples)
    again.run_once(MIDNIGHT + timedelta(days=2, minutes=10), legacy)
    assert len(read(tmp_path, "climate_models")) == 1  # the weekly refit is not due yet
    scored = {s["t0"] for s in read(tmp_path, "climate_scores")}
    assert scored == {r.isoformat() for r in runs}
    snap = again.snapshot()
    assert snap["model_hash"] == models[0]["model_hash"] and snap["queue"] == 1
    assert snap["report"]["windows"]["legacy"]["verdict"] == "UNDETERMINED"


def test_a_data_failure_is_recorded_not_raised(tmp_path):
    def broken(_now):
        raise OSError("prometheus down")

    job = make_job(tmp_path, broken)
    summary = job.run_once(MIDNIGHT + timedelta(minutes=10), shadow.LiveInputs(regime="unknown"))
    assert summary["status"] == "error"
    (rec,) = read(tmp_path, "climate_decisions")
    assert rec["status"] == "error" and "prometheus down" in rec["error"]
