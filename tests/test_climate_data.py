"""Climate loader: derived signals, filters, solar proxy and replay parity on synthetic data.

Everything here is synthetic: no network, no credentials, no real data directory.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from climate_fakes import CONFIG, MADRID, SITE, SLOTS_DAY, make_table
from climate_fakes import build as _build
from climate_fakes import grid as _grid

from battery_optimizer.climate.config import load_climate_config
from battery_optimizer.climate.data import (
    SLOT_US,
    ClimateDataError,
    Site,
    build_climate_data,
    dataset_prices,
    debias_outdoor,
    load_climate_data,
    load_site,
    room_offsets,
    solar_proxy,
)
from battery_optimizer.load_forecast import SideInputs
from battery_optimizer.policies import contiguous_day_bounds
from battery_optimizer.structured_forecast import replay_mask_for_day

# ---------------------------------------------------------------------------
# Calendar: DST days
# ---------------------------------------------------------------------------


def test_dst_days_have_92_and_100_slots_and_are_complete():
    spring = _build(make_table(_grid(date(2025, 3, 29), 4)))
    assert [(d, hi - lo) for d, lo, hi in spring.days] == [
        (date(2025, 3, 29), 96),
        (date(2025, 3, 30), 92),
        (date(2025, 3, 31), 96),
        (date(2025, 4, 1), 96),
    ]
    assert spring.day_complete.all()
    autumn = _build(make_table(_grid(date(2025, 10, 25), 3)))
    assert [hi - lo for _d, lo, hi in autumn.days] == [96, 100, 96]
    assert autumn.day_complete.all()
    # the hour of day follows the wall clock: 02 is skipped in spring, 02 repeats in autumn
    spring_day = spring.local_hour[spring.days[1][1] : spring.days[1][2]]
    assert 2 not in spring_day and np.bincount(spring_day, minlength=24).max() == 4
    autumn_day = autumn.local_hour[autumn.days[1][1] : autumn.days[1][2]]
    assert np.bincount(autumn_day, minlength=24)[2] == 8


def test_partial_edge_days_are_not_complete():
    # a window that starts and ends at UTC midnight cuts the first and last local day
    start = int(datetime(2025, 11, 5, tzinfo=UTC).timestamp()) * 1_000_000
    starts = np.arange(start, start + 3 * 96 * SLOT_US, SLOT_US)
    data = _build(make_table(starts))
    assert list(data.day_complete) == [False, True, True, False]
    assert data.season_slots("S3").sum() == 2 * 96
    assert data.season_slots("S3", months="gate").sum() == 2 * 96


def test_seasons_and_months_are_per_local_day():
    data = _build(make_table(_grid(date(2025, 9, 29), 4)))
    assert list(data.day_season) == ["", "", "S3", "S3"]
    assert list(data.day_fit_month) == [False, False, True, True]
    assert list(data.day_gate_month) == [False, False, False, False]


def test_non_uniform_grid_and_missing_columns_are_refused():
    starts = _grid(date(2025, 12, 1), 1)
    with pytest.raises(ClimateDataError, match="uniform 15-minute grid"):
        _build(make_table(np.delete(starts, 10)))
    with pytest.raises(ClimateDataError, match="lacks the column"):
        _build(make_table(starts).drop_columns(["hp_heat_out_kwh"]))


# ---------------------------------------------------------------------------
# T_air, offsets, Q_heat, DHW, cooling, valves
# ---------------------------------------------------------------------------


def test_t_air_is_the_mean_of_the_comfort_rooms_only():
    starts = _grid(date(2025, 12, 1), 1)
    table = make_table(
        starts,
        room_bedroom_temp_mean_c=20.0,
        room_hall_kitchen_temp_mean_c=22.0,
        room_office_t_temp_mean_c=24.0,
        room_office_a_temp_mean_c=35.0,
    )
    assert np.allclose(_build(table).t_air, 22.0)


def test_t_air_is_unknown_unless_every_comfort_room_is_known():
    starts = _grid(date(2025, 12, 1), 1)
    bedroom = np.full(starts.size, 21.0)
    bedroom[5] = np.nan
    data = _build(make_table(starts, room_bedroom_temp_mean_c=bedroom))
    assert np.isnan(data.t_air[5]) and np.isfinite(data.t_air[4])
    assert data.drop["unknown_quality"][5] and not data.drop["unknown_quality"][4]


def test_room_offsets_are_hourly_medians_over_the_chosen_slots():
    starts = _grid(date(2025, 12, 1), 6)
    n = starts.size
    hour = np.array([datetime.fromtimestamp(s / 1e6, UTC).astimezone(MADRID).hour for s in starts])
    bedroom = 21.0 + np.where(hour < 12, -0.5, 0.25)
    hall = 21.0 + np.where(hour < 12, 0.5, 0.0)
    office = np.full(n, 21.0)
    table = make_table(
        starts,
        room_bedroom_temp_mean_c=bedroom,
        room_hall_kitchen_temp_mean_c=hall,
        room_office_t_temp_mean_c=office,
    )
    data = _build(table)
    t_air = (bedroom + hall + office) / 3.0
    assert np.allclose(data.t_air, t_air)
    rooms = ("bedroom", "hall_kitchen", "office_t")
    temps = {r: data.room_temp[r] for r in rooms}
    offsets = room_offsets(temps, rooms, data.t_air, data.local_hour, data.train)
    assert offsets.shape == (3, 24)
    # t_air - room offsets sum to zero at every hour
    assert np.allclose(offsets.sum(axis=0), 0.0)
    assert np.allclose(offsets[0, :12], -0.5 - (-0.5 + 0.5) / 3)
    # an outlier minority does not move the median
    spiked = bedroom.copy()
    spiked[hour == 3] += 5.0 * (np.arange(n)[hour == 3] % 4 == 0)
    out = room_offsets(
        {"bedroom": spiked, "hall_kitchen": hall, "office_t": office},
        rooms,
        (spiked + hall + office) / 3,
        data.local_hour,
        np.ones(n, dtype=bool),
    )
    assert np.isclose(out[0, 3], offsets[0, 3], atol=1e-6)
    none = room_offsets(temps, rooms, t_air, data.local_hour, np.zeros(n, dtype=bool))
    assert np.isnan(none).all()


def test_offsets_are_not_a_loader_output_and_only_see_the_mask():
    """A fold fits offsets on its own mask: nothing held out may reach them."""
    starts = _grid(date(2025, 12, 1), 4)
    n = starts.size
    data = _build(make_table(starts))
    assert not hasattr(data, "room_offsets")
    rooms = ("bedroom", "hall_kitchen", "office_t")
    temps = {r: np.full(n, 21.0) for r in rooms}
    temps["bedroom"][2 * SLOTS_DAY :] += 3.0  # the "held-out" half is different
    t_air = np.mean([temps[r] for r in rooms], axis=0)
    fit = np.zeros(n, dtype=bool)
    fit[: 2 * SLOTS_DAY] = True
    offsets = room_offsets(temps, rooms, t_air, data.local_hour, fit)
    assert np.allclose(offsets, 0.0)  # unchanged by the later, hotter bedroom


def test_q_heat_and_the_daily_counter_cross_check():
    starts = _grid(date(2025, 12, 1), 3)
    n = starts.size
    q = np.full(n, 0.5)
    counter = np.full(n, 0.5)
    counter[:SLOTS_DAY] *= 1.1  # day one: the counter reads 10 % above the power integral
    q[SLOTS_DAY + 3] = np.nan  # day two has an unknown slot: it does not take part
    counter[2 * SLOTS_DAY :] = 0.001  # day three: a quiet day below the minimum
    data = _build(make_table(starts, hp_heat_out_kwh=q, hp_heat_energy_out_delta=counter))
    assert np.allclose(data.q_heat_kw, q * 4, equal_nan=True)
    check = data.counter_check(CONFIG.signals.counter_check.min_daily_kwh)
    assert list(check["days"]) == [date(2025, 12, 1)]
    assert np.isclose(check["ratio"][0], 1 / 1.1)


def test_dhw_cooling_masks_and_valve_state():
    starts = _grid(date(2025, 12, 1), 1)
    n = starts.size
    dhw_e = np.zeros(n)
    dhw_e[10] = 0.2  # electric draw
    dhw_q = np.zeros(n)
    dhw_q[20] = 0.6  # thermal output only
    force = np.zeros(n)
    force[30] = 1.0  # force-DHW switch on
    cool = np.zeros(n)
    cool[40] = 0.3
    temp = np.full(n, 21.0)
    temp[50] = 24.5  # above its setpoint of 24 -> valve closed
    table = make_table(
        starts,
        hp_dhw_kwh=dhw_e,
        hp_dhw_out_kwh=dhw_q,
        dhw_force_mean=force,
        hp_cool_kwh=cool,
        room_bedroom_temp_mean_c=temp,
    )
    data = _build(table)
    assert list(np.flatnonzero(data.dhw)) == [10, 20, 30]
    assert list(np.flatnonzero(data.cooling)) == [40]
    assert list(np.flatnonzero(data.drop["cooling"])) == [40]
    assert data.valve_open_inferred["bedroom"][49] and not data.valve_open_inferred["bedroom"][50]
    assert data.valve_open_inferred["office_a"].all()


def test_unknown_dhw_is_not_silently_zero():
    starts = _grid(date(2025, 12, 1), 1)
    dhw_e = np.zeros(starts.size)
    dhw_e[7] = np.nan
    data = _build(make_table(starts, hp_dhw_kwh=dhw_e))
    assert not data.dhw[7]
    assert data.drop["unknown_quality"][7]


# ---------------------------------------------------------------------------
# Training filters
# ---------------------------------------------------------------------------


def test_window_open_needs_a_fall_of_more_than_half_a_kelvin_in_one_slot():
    starts = _grid(date(2025, 12, 1), 1)
    temp = np.full(starts.size, 21.0)
    temp[10:] -= 0.5  # exactly 0.5: not a window-open event
    temp[20:] -= 0.6  # more than 0.5: event at slot 20
    temp[30:] += 2.0  # a rise is not an event
    temp[40:] -= 0.3
    temp[41:] -= 0.3  # two small steps: neither is an event
    data = _build(make_table(starts, room_hall_kitchen_temp_mean_c=temp))
    assert list(np.flatnonzero(data.drop["window_open"])) == [20]


def test_window_open_ignores_the_excluded_room():
    starts = _grid(date(2025, 12, 1), 1)
    office_a = np.full(starts.size, 21.0)
    office_a[10:] -= 3.0
    data = _build(make_table(starts, room_office_a_temp_mean_c=office_a))
    assert not data.drop["window_open"].any()


def test_valves_shut_when_two_comfort_rooms_are_at_or_above_setpoint():
    starts = _grid(date(2025, 12, 1), 1)
    n = starts.size
    bedroom = np.full(n, 21.0)
    hall = np.full(n, 21.0)
    office = np.full(n, 21.0)
    bedroom[5:] = 24.0  # at setpoint: one room
    hall[8:] = 24.5  # a second room: shut
    office[12:] = 24.0  # three rooms: still shut
    # the excluded room never counts
    data = _build(
        make_table(
            starts,
            room_bedroom_temp_mean_c=bedroom,
            room_hall_kitchen_temp_mean_c=hall,
            room_office_t_temp_mean_c=office,
            room_office_a_temp_mean_c=30.0,
        )
    )
    assert list(np.flatnonzero(~data.drop["valves_shut"])) == list(range(8))
    assert data.drop["valves_shut"][8:].all()


def test_thermostat_outage_window_drops_whole_local_days():
    starts = _grid(date(2026, 2, 7), 8)
    data = _build(make_table(starts))
    dropped = [d for d, lo, _hi in data.days if data.drop["thermostat_outage"][lo]]
    assert dropped == [date(2026, 2, 9), date(2026, 2, 10), date(2026, 2, 11), date(2026, 2, 12)]
    assert data.drop["thermostat_outage"].sum() == 4 * SLOTS_DAY


def test_train_is_the_slots_no_filter_drops():
    starts = _grid(date(2025, 12, 1), 1)
    cool = np.zeros(starts.size)
    cool[3] = 1.0
    freq = np.zeros(starts.size)
    freq[20:80] = 40.0  # a 15 h run: longer than the interpolation cap, so T_out is unknown
    data = _build(make_table(starts, hp_cool_kwh=cool, compressor_freq_mean=freq))
    assert not data.train[3] and data.train[5]
    unknown = ~np.isfinite(data.t_out_debiased)
    assert unknown[20:80].all() and not unknown[2:20].any()
    assert not data.train[unknown].any()
    assert data.drop["outdoor_unknown"][unknown].all()
    assert not data.drop["unknown_quality"][20:80].any()  # a separate reason, not "unknown"


def test_filters_can_be_switched_off_by_configuration(tmp_path):
    raw = CONFIG.path.read_text()
    raw = raw.replace("drop_cooling_slots: true", "drop_cooling_slots: false")
    raw = raw.replace("drop_unknown_quality: true", "drop_unknown_quality: false")
    path = tmp_path / "climate.yaml"
    path.write_text(raw)
    cfg = load_climate_config(path)
    starts = _grid(date(2025, 12, 1), 1)
    cool = np.zeros(starts.size)
    cool[3] = 1.0
    heat = np.full(starts.size, 0.4)
    heat[9] = np.nan
    data = build_climate_data(
        make_table(starts, hp_cool_kwh=cool, hp_heat_out_kwh=heat), config=cfg, site=SITE
    )
    assert data.cooling[3] and data.train[3] and data.train[9]


# ---------------------------------------------------------------------------
# Solar proxy
# ---------------------------------------------------------------------------


def _utc_us(year, month, day, hour=0, minute=0) -> np.ndarray:
    return np.array([int(datetime(year, month, day, hour, minute, tzinfo=UTC).timestamp()) * 10**6])


def test_solar_proxy_at_the_solstices_and_equinox():
    lat = 40.0
    june = solar_proxy(_utc_us(2025, 6, 21, 12, 0), lat, 0.0)[0]
    assert june == pytest.approx(np.sin(np.radians(90 - lat + 23.44)), abs=0.004)
    december = solar_proxy(_utc_us(2025, 12, 21, 12, 0), lat, 0.0)[0]
    assert december == pytest.approx(np.sin(np.radians(90 - lat - 23.44)), abs=0.006)
    equinox = solar_proxy(_utc_us(2025, 3, 20, 12, 0), lat, 0.0)[0]
    assert equinox == pytest.approx(np.cos(np.radians(lat)), abs=0.01)
    assert june > equinox > december > 0.0


def test_solar_proxy_is_zero_at_night_and_follows_longitude():
    t = np.concatenate([_utc_us(2025, 6, 21, h) for h in range(24)])
    s = solar_proxy(t, 40.0, 0.0)
    assert s[0] == 0.0 and s[23] == 0.0 and s[12] > 0.9
    assert (s >= 0).all() and (s <= 1).all()
    # 15 degrees east brings solar noon one hour earlier
    east = solar_proxy(t, 40.0, 15.0)
    assert int(np.argmax(east)) == int(np.argmax(s)) - 1
    # polar night: no sun at all at 80 N in December
    assert (solar_proxy(t + 183 * 86400 * 10**6, 80.0, 0.0) == 0.0).all()


def test_solar_noon_follows_the_equation_of_time_and_the_data_carries_the_proxy():
    # on 23 September the sun runs about 7 minutes ahead of the clock: solar noon is before 12:00
    t = np.concatenate([_utc_us(2025, 9, 23, 11, m) for m in range(0, 60)])
    noon = int(np.argmax(solar_proxy(t, 40.0, 0.0)))
    assert 45 <= noon <= 57
    starts = _grid(date(2025, 6, 21), 1)
    data = _build(make_table(starts))
    assert data.solar.max() > 0.9 and (data.solar[data.local_hour < 4] == 0).all()


# ---------------------------------------------------------------------------
# Replayed u
# ---------------------------------------------------------------------------


def _price_pattern(n_days: int, first_day: date) -> np.ndarray:
    """Varying hourly prices (flat within the hour) on the loader's local-day grid."""
    starts = _grid(first_day, n_days)
    rng = np.random.default_rng(7)
    hours = starts.size // 4
    hourly = 0.10 + 0.05 * np.sin(np.arange(hours) * 2 * np.pi / 24) + rng.uniform(0, 0.02, hours)
    return np.repeat(hourly, 4)


def _direct_replay(table: pa.Table, prices: np.ndarray):
    """The structured forecaster's own replay for each day, with a view cut at the decision."""
    starts = [
        datetime.fromtimestamp(int(u) / 1e6, UTC)
        for u in table["interval_start"].cast(pa.int64()).to_numpy()
    ]
    local = [s.astimezone(MADRID).date() for s in starts]
    days = contiguous_day_bounds(local)
    side = SideInputs(
        interval_starts_utc=starts,
        tz=MADRID,
        prices=prices,
        series={
            "climate_enable": table["climate_enable_at_start"].to_numpy(),
            "climate_min_hours": table["climate_min_hours_at_start"].to_numpy(),
        },
    )
    out = {}
    for j, (label, lo, hi) in enumerate(days):
        view = side.view(lo + 1, days)  # what is known 40 s into the day
        out[label] = replay_mask_for_day(view, days, prices[:hi], j, lo + 1)
    return out


def test_replayed_u_matches_structured_forecast_over_dst_days():
    first = date(2025, 10, 14)
    n_days = 20  # spans the 100-slot day of 2025-10-26
    starts = _grid(first, n_days)
    prices = _price_pattern(n_days, first)
    enable = np.ones(starts.size)
    min_hours = np.full(starts.size, 6.0)
    min_hours[10 * SLOTS_DAY :] = 9.0  # a later change of the lever
    table = make_table(starts, climate_enable_at_start=enable, climate_min_hours_at_start=min_hours)
    data = _build(table, prices)
    expected = _direct_replay(table, prices)
    checked = 0
    for label, lo, hi in data.days:
        mask, reason = expected[label]
        assert data.u_reason[[d for d, _l, _h in data.days].index(label)] == reason
        if mask is None:
            assert np.isnan(data.u[lo:hi]).all()
            continue
        checked += 1
        assert np.array_equal(data.u[lo:hi], mask[data.local_hour[lo:hi]].astype(float))
    assert checked >= n_days - 11  # the first ten days have no trailing mean yet
    # the 100-slot day: the repeated hour carries the same permission twice
    autumn = [i for i, (d, _l, _h) in enumerate(data.days) if d == date(2025, 10, 26)][0]
    _d, lo, hi = data.days[autumn]
    assert hi - lo == 100
    hours = data.local_hour[lo:hi]
    assert np.array_equal(data.u[lo:hi][hours == 2], np.full(8, data.u[lo:hi][hours == 2][0]))


def test_replayed_u_selects_the_cheapest_hours_below_the_limit():
    first = date(2025, 12, 1)
    n_days = 12
    starts = _grid(first, n_days)
    hourly = np.full(n_days * 24, 0.20)
    hourly[-24 + 2 : -24 + 5] = 0.05  # three cheap hours on the last day: 02, 03, 04
    prices = np.repeat(hourly, 4)
    data = _build(make_table(starts, climate_min_hours_at_start=2.0), prices)
    lo, hi = data.days[-1][1], data.days[-1][2]
    selected = sorted(set(data.local_hour[lo:hi][data.u[lo:hi] == 1.0]))
    assert selected == [2, 3, 4]
    # fewer cheap hours than min_hours: the min_hours cheapest are used instead
    hourly[-24 + 3 : -24 + 5] = 0.20
    data = _build(make_table(starts, climate_min_hours_at_start=2.0), np.repeat(hourly, 4))
    selected = sorted(set(data.local_hour[lo:hi][data.u[lo:hi] == 1.0]))
    assert len(selected) == 2 and 2 in selected


def test_days_with_the_rule_disabled_are_excluded():
    first = date(2025, 12, 1)
    n_days = 14
    starts = _grid(first, n_days)
    prices = _price_pattern(n_days, first)
    enable = np.ones(starts.size)
    enable[12 * SLOTS_DAY : 13 * SLOTS_DAY] = 0.0
    enable[13 * SLOTS_DAY :] = np.nan
    data = _build(make_table(starts, climate_enable_at_start=enable), prices)
    reasons = dict(zip([d for d, _l, _h in data.days], data.u_reason, strict=True))
    assert reasons[date(2025, 12, 13)] == "rule_disabled"
    # an unknown flag keeps its last known state for up to two days (structured_forecast rule)
    assert reasons[date(2025, 12, 14)] == "rule_disabled"
    lo, hi = data.days[12][1], data.days[12][2]
    assert np.isnan(data.u[lo:hi]).all()
    assert data.drop["rule_disabled_day"][lo:hi].all()
    assert not data.drop["rule_disabled_day"][:lo].any()
    assert not data.train[lo:hi].any()


def test_without_prices_there_is_no_u_and_no_disabled_filter():
    starts = _grid(date(2025, 12, 1), 3)
    data = _build(make_table(starts))
    assert np.isnan(data.u).all()
    assert set(data.u_reason) <= {"prices_incomplete", "price_history_short"}
    assert data.train[2:].all()  # the first two slots have no 30 minutes of compressor history


# ---------------------------------------------------------------------------
# Prices, site file and the on-disk loader
# ---------------------------------------------------------------------------


def test_dataset_prices_align_by_time_and_drop_unusable_quality():
    starts = _grid(date(2025, 12, 1), 1)
    ds_starts = starts[2:10]
    table = pa.table(
        {
            "interval_start": pa.array(ds_starts, pa.timestamp("us", tz="UTC")),
            "price_eur_per_kwh": pa.array([0.1, 0.2, None, 0.4, 0.5, 0.6, 0.7, 0.8], pa.float64()),
            "price_quality": ["measured"] * 4
            + ["ambiguous", "missing", "forward_filled", "measured"],
        }
    )
    aligned = dataset_prices(table, starts)
    assert np.isnan(aligned[:2]).all() and np.isnan(aligned[10:]).all()
    assert aligned[2] == 0.1 and aligned[3] == 0.2 and np.isnan(aligned[4])
    assert aligned[5] == 0.4 and np.isnan(aligned[6]) and np.isnan(aligned[7])
    assert aligned[8] == 0.7 and aligned[9] == 0.8


def test_site_file_is_validated_and_hashed(tmp_path):
    good = tmp_path / "site.yaml"
    good.write_text("latitude: 41.5\nlongitude: -2.0\nphi: 0.25\n")
    site = load_site(good)
    assert (site.latitude, site.longitude, site.phi) == (41.5, -2.0, 0.25)
    assert site.sha256 == hashlib.sha256(good.read_bytes()).hexdigest()
    for body, match in [
        ("latitude: 41.5\n", "needs latitude and longitude"),
        ("latitude: 95\nlongitude: 0\n", "latitude must be"),
        ("latitude: x\nlongitude: 0\n", "latitude must be"),
        ("latitude: 1\nlongitude: 0\nsurprise: 1\n", "unknown key"),
        ("latitude: 1\nlongitude: 0\nphi: 2\n", "phi must be"),
        ("- 1\n", "needs latitude and longitude"),
    ]:
        bad = tmp_path / "bad.yaml"
        bad.write_text(body)
        with pytest.raises(ClimateDataError, match=match):
            load_site(bad)
    with pytest.raises(ClimateDataError, match="cannot read"):
        load_site(tmp_path / "absent.yaml")


def _write_feature_set(root: Path, table: pa.Table, name: str) -> None:
    directory = root / "analysis" / "v1" / "features" / name
    (directory / "parts").mkdir(parents=True)
    part = directory / "parts" / "part.parquet"
    pq.write_table(table, part)
    entry = {
        "path": "parts/part.parquet",
        "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
        "rows": table.num_rows,
    }
    (directory / "features_manifest.json").write_text(
        json.dumps({"feature_set_id": name, "part_files": [entry]})
    )


def test_load_climate_data_reads_a_verified_feature_set(tmp_path):
    root = tmp_path / "store"
    table = make_table(_grid(date(2025, 12, 1), 2))
    _write_feature_set(root, table, "abc123")
    (root / "climate").mkdir()
    (root / "climate" / "site.yaml").write_text("latitude: 40\nlongitude: 0\n")
    data = load_climate_data(CONFIG, feature_set="abc", data_dir=str(root))
    assert data.feature_set_id == "abc123" and data.dataset_id is None and data.n == 2 * SLOTS_DAY
    assert data.site_sha256 == load_site(root / "climate" / "site.yaml").sha256
    assert data.config_sha256 == CONFIG.sha256
    with pytest.raises(ClimateDataError, match="matches 0"):
        load_climate_data(CONFIG, feature_set="zzz", data_dir=str(root))
    # a tampered part is refused through the manifest hash
    (root / "analysis" / "v1" / "features" / "abc123" / "parts" / "part.parquet").write_bytes(b"x")
    with pytest.raises(Exception, match="sha256"):
        load_climate_data(CONFIG, feature_set="abc", data_dir=str(root))


def test_load_climate_data_refuses_a_default_feature_set(tmp_path):
    root = tmp_path / "store"
    table = pa.table({"interval_start": pa.array([0], pa.timestamp("us", tz="UTC"))})
    _write_feature_set(root, table, "def456")
    with pytest.raises(ClimateDataError, match="not a climate feature set"):
        load_climate_data(
            CONFIG,
            feature_set="def",
            data_dir=str(root),
            site=Site(40.0, 0.0, None, "x"),
        )


# ---------------------------------------------------------------------------
# Outdoor temperature de-biasing
# ---------------------------------------------------------------------------


def _biased_series(n=120):
    truth = 5.0 + 0.01 * np.arange(n)
    freq = np.zeros(n)
    freq[10:14] = 40.0  # four running slots
    raw = truth.copy()
    raw[10:14] -= 2.0  # the unit cools its own sensor ...
    raw[14:16] -= 1.0  # ... and the reading recovers over the next half hour
    return truth, raw, freq


def test_debias_keeps_slots_off_for_the_minimum_and_interpolates_the_rest():
    truth, raw, freq = _biased_series()
    out, valid, interpolated = debias_outdoor(raw, freq, min_off_slots=2, max_gap_slots=8)
    # 14 and 15 have been off for less than 30 min; 0 and 1 have no history yet
    assert list(np.flatnonzero(~valid)) == [0, 1, 10, 11, 12, 13, 14, 15]
    assert list(np.flatnonzero(interpolated)) == [10, 11, 12, 13, 14, 15]
    assert np.allclose(out[2:], truth[2:])  # a linear truth is recovered exactly
    assert np.isnan(out[:2]).all() and valid[16]


def test_debias_leaves_long_runs_unknown_and_never_extrapolates():
    truth, raw, freq = _biased_series(n=60)
    freq[10:30] = 40.0  # 20 slots (5 h)
    capped, valid, interpolated = debias_outdoor(raw, freq, min_off_slots=2, max_gap_slots=8)
    assert np.isnan(capped[10:32]).all() and not interpolated.any()
    filled, _v, interp = debias_outdoor(raw, freq, min_off_slots=2, max_gap_slots=24)
    assert np.isfinite(filled[10:32]).all() and interp[10:32].all()
    # nothing before the first valid slot or after the last one
    freq = np.zeros(60)
    freq[:5] = 40.0
    freq[-5:] = 40.0
    edge, valid, _i = debias_outdoor(np.full(60, 5.0), freq, min_off_slots=2, max_gap_slots=100)
    assert np.isnan(edge[:7]).all() and np.isnan(edge[-5:]).all() and np.isfinite(edge[7:-5]).all()
    assert not valid[:7].any()


def test_debias_treats_unknown_frequency_and_unknown_readings_as_not_valid():
    raw = np.full(30, 5.0)
    freq = np.zeros(30)
    freq[8] = np.nan
    raw[20] = np.nan
    out, valid, interpolated = debias_outdoor(raw, freq, min_off_slots=1, max_gap_slots=4)
    assert not valid[8] and not valid[9] and valid[10]  # a known off period is needed first
    assert not valid[20] and interpolated[20] and out[20] == 5.0


def test_loader_exposes_raw_and_debiased_outdoor_temperature():
    starts = _grid(date(2025, 12, 1), 2)
    n = starts.size
    truth = 5.0 + 0.005 * np.arange(n)
    freq = np.zeros(n)
    freq[40:44] = 40.0
    raw = truth.copy()
    raw[40:46] -= 2.0
    data = _build(make_table(starts, outdoor_temp_mean_c=raw, compressor_freq_mean=freq))
    assert np.array_equal(data.t_out, raw)  # raw stays for diagnostics
    assert np.allclose(data.t_out_debiased[30:60], truth[30:60])
    assert data.t_out_interpolated[40:46].all() and data.t_out_valid[46]
    assert data.drop["outdoor_unknown"].sum() == 2 and data.train[44]  # no history at the start
    # the configured thresholds drive the function: 30 min is two slots, 6 h is 24 slots
    assert CONFIG.signals.t_out_debias.min_off_minutes == 30
    assert CONFIG.signals.t_out_debias.max_gap_hours == 6.0


# ---------------------------------------------------------------------------
# Causality of the replayed u
# ---------------------------------------------------------------------------


def test_replayed_u_of_a_day_does_not_depend_on_later_prices_or_flags():
    first, n_days = date(2025, 12, 1), 16
    starts = _grid(first, n_days)
    prices = _price_pattern(n_days, first)
    enable = np.ones(starts.size)
    hours = np.full(starts.size, 6.0)
    base = _build(
        make_table(starts, climate_enable_at_start=enable, climate_min_hours_at_start=hours), prices
    )
    rng = np.random.default_rng(11)
    for j in (11, 12, 14):
        _d, lo, hi = base.days[j]
        later_prices = prices.copy()
        later_prices[hi:] = rng.uniform(0.01, 0.5, later_prices.size - hi)
        later_enable = enable.copy()
        later_enable[lo + 1 :] = 0.0  # the flag flips right after the decision
        later_hours = hours.copy()
        later_hours[lo + 1 :] = 20.0
        changed = _build(
            make_table(
                starts, climate_enable_at_start=later_enable, climate_min_hours_at_start=later_hours
            ),
            later_prices,
        )
        assert np.array_equal(changed.u[:hi], base.u[:hi], equal_nan=True)
    # the other direction: a change before the day does change it
    earlier = prices.copy()
    earlier[: 10 * SLOTS_DAY] = 0.9
    assert not np.array_equal(
        _build(make_table(starts), earlier).u, _build(make_table(starts), prices).u, equal_nan=True
    )


def test_u_known_marks_exactly_the_replayed_slots():
    first, n_days = date(2025, 12, 1), 14
    starts = _grid(first, n_days)
    enable = np.ones(starts.size)
    enable[12 * SLOTS_DAY : 13 * SLOTS_DAY] = 0.0
    data = _build(make_table(starts, climate_enable_at_start=enable), _price_pattern(n_days, first))
    assert np.array_equal(data.u_known, np.isfinite(data.u))
    for j, (_label, lo, hi) in enumerate(data.days):
        assert data.u_known[lo:hi].all() == bool(data.day_u_known[j])
        assert data.u_known[lo:hi].any() == bool(data.day_u_known[j])
    assert not data.day_u_known[12] and data.day_u_known[11]
    assert data.u_reason[12] == "rule_disabled"
