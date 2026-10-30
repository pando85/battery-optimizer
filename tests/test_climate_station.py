"""SYNOP decoding, the Ogimet fetcher, the immutable store and the reference interpolation.

Synthetic reports only: no network (an ``httpx.MockTransport``), no real station, no real data.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime

import httpx
import numpy as np
import pytest
from station_fakes import STATION, hourly, ogimet_transport, synop_csv, synop_line

from battery_optimizer.climate import station as st
from battery_optimizer.climate.station import (
    StationError,
    decode_synop_temperature,
    fetch_station,
    load_station,
    month_range,
    parse_getsynop,
    reference_at_slots,
)

# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


def test_decodes_positive_and_negative_tenths():
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2401 10084 20083 39394=") == 8.4
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2401 11012 20083 39394=") == -1.2
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2401 10000 20083=") == 0.0
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2401 11000 20083=") == 0.0
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2401 10356 20083=") == 35.6


def test_group_one_is_read_by_position_not_by_first_digit():
    # iRiXhVV starts with 1 (iR = 1) and N = 1 in Nddff: neither is the temperature
    assert decode_synop_temperature("AAXX 15001 12345 15/// 11012 10084 20083=") == 8.4
    assert decode_synop_temperature("AAXX 15001 12345 12456 11012 11055 20083=") == -5.5
    # a later group that also starts with 1 (section 3, 10099) must not be picked up
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2401 1//// 20075 333 10099=") is None


def test_wind_speed_99_adds_a_00fff_group_before_the_temperature():
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2499 00105 11012 20083=") == -1.2
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2499 00105 10084 20083=") == 8.4
    # without the extra group, ff = 99 would leave 00105 where the temperature should be
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2499 10084 20083=") is None


def test_missing_nil_and_malformed_reports_are_none():
    assert decode_synop_temperature("AAXX 15001 12345 NIL==") is None
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2401 1//// 20083=") is None
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2401 19999 20083=") is None  # s = 9
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2401 1008x 20083=") is None
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2401=") is None
    assert decode_synop_temperature("AAXX 15001 12345 05///=") is None
    assert decode_synop_temperature("BBXX 15001 12345 05/// /2401 10084=") is None
    assert decode_synop_temperature("") is None
    # another station's report is not this station's
    assert decode_synop_temperature("AAXX 15001 12346 05/// /2401 10084=", "12345") is None
    assert decode_synop_temperature("AAXX 15001 12345 05/// /2401 10084=", "12345") == 8.4


def test_parse_getsynop_skips_nil_keeps_the_first_duplicate_and_sorts():
    t0 = datetime(2024, 1, 15, tzinfo=UTC)
    lines = [
        synop_line(STATION, t0.replace(hour=2), 3.0),
        synop_line(STATION, t0.replace(hour=0), -1.2),
        f"{STATION},2024,01,15,01,00,AAXX 15011 {STATION} NIL==",
        synop_line(STATION, t0.replace(hour=3), None),
        synop_line(STATION, t0.replace(hour=2), 9.9),
        "",
    ]
    times, temps = parse_getsynop("\n".join(lines), STATION)
    assert times.tolist() == [int(t0.timestamp()) * 10**6 + h * 3600 * 10**6 for h in (0, 2)]
    assert temps.tolist() == [-1.2, 3.0]


def test_parse_getsynop_refuses_a_non_report_line_and_another_station():
    with pytest.raises(StationError, match="unexpected response line"):
        parse_getsynop("# Quota exceeded, try later\n", STATION)
    with pytest.raises(StationError, match="holds station"):
        parse_getsynop(synop_line("88888", datetime(2024, 1, 1, tzinfo=UTC), 1.0), STATION)
    assert parse_getsynop("", STATION)[0].size == 0


# ---------------------------------------------------------------------------
# Reference interpolation and the gap bridge
# ---------------------------------------------------------------------------


def _us(hours: float) -> int:
    return int(hours * 3600 * 10**6)


def test_reference_interpolates_linearly_to_slot_midpoints():
    times = np.array([_us(h) for h in (0, 1, 2)])
    temp = np.array([10.0, 12.0, 11.0])
    mid = np.array([_us(0.125), _us(0.5), _us(1.0), _us(1.5)])
    out = reference_at_slots(times, temp, mid, 3.0)
    assert np.allclose(out, [10.25, 11.0, 12.0, 11.5])


def test_gaps_up_to_the_bridge_are_filled_and_longer_ones_are_nan():
    hours = [0, 1, 2, 5, 6, 10, 11]  # gaps: 3 h (2 -> 5, bridged), 4 h (6 -> 10, not)
    times = np.array([_us(h) for h in hours])
    temp = np.array([float(h) for h in hours])
    mid = np.array([_us(h) for h in (3.5, 8.0, 12.0, -0.5)])
    out = reference_at_slots(times, temp, mid, 3.0)
    assert out[0] == pytest.approx(3.5)  # linear in time inside the 3 h gap
    assert np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3])
    wider = reference_at_slots(times, temp, mid, 4.0)
    assert wider[1] == pytest.approx(8.0)  # the bridge is the configured length
    assert reference_at_slots(np.zeros(0, np.int64), np.zeros(0), mid, 3.0).size == mid.size


def test_a_three_hourly_station_is_bridged_hour_by_hour():
    times = np.array([_us(h) for h in range(0, 24, 3)])
    temp = np.arange(8, dtype=float)
    mid = np.array([_us(h + 0.5) for h in range(21)])
    assert np.isfinite(reference_at_slots(times, temp, mid, 3.0)).all()
    assert np.isnan(reference_at_slots(times, temp, mid, 2.5)).all()


# ---------------------------------------------------------------------------
# Fetcher and store
# ---------------------------------------------------------------------------


class Clock:
    """Fake time: ``sleep`` advances it, each request takes a second."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _temp(t: datetime) -> float:
    return 5.0 + 0.1 * (t.day % 10) - 0.01 * t.hour


def test_month_range_and_windows():
    assert month_range(date(2023, 11, 1), date(2024, 2, 1)) == [
        (2023, 11),
        (2023, 12),
        (2024, 1),
        (2024, 2),
    ]
    assert st._month_window(2024, 2) == ("202402010000", "202402292359")
    assert st._month_window(2023, 12) == ("202312010000", "202312312359")


def test_fetch_stores_each_month_once_with_a_manifest_and_spaces_requests(tmp_path):
    calls: list[dict] = []
    clock = Clock()
    client = httpx.Client(transport=ogimet_transport(_temp, calls))
    months = [(2024, 1), (2024, 2), (2024, 3)]
    fetched = fetch_station(
        tmp_path,
        STATION,
        months,
        client=client,
        clock=clock.clock,
        sleep=clock.sleep,
        now=lambda: datetime(2026, 10, 30, 12, 0, tzinfo=UTC),
    )
    assert fetched == ["2024-01", "2024-02", "2024-03"]
    assert [c["begin"] for c in calls] == ["202401010000", "202402010000", "202403010000"]
    assert all(c["block"] == STATION for c in calls)
    assert len(clock.sleeps) == 2 and all(s >= 5.0 for s in clock.sleeps)

    raw = tmp_path / "climate" / "station" / STATION / "raw"
    manifest = json.loads((raw / "2024-02.manifest.json").read_text())
    payload = (raw / "2024-02.csv").read_bytes()
    assert manifest["sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["source"] == "ogimet-getsynop" and manifest["station"] == STATION
    assert manifest["request"] == {"begin": "202402010000", "end": "202402292359"}
    assert manifest["fetched_at"] == "2026-10-30T12:00:00Z"
    assert manifest["parser_version"] == st.PARSER_VERSION

    # a second run fetches nothing: each month is fetched once and kept immutable
    calls.clear()
    assert fetch_station(tmp_path, STATION, months, client=client, sleep=clock.sleep) == []
    assert calls == []
    assert (raw / "2024-02.csv").read_bytes() == payload


def test_fetch_retries_server_errors_and_timeouts_with_backoff(tmp_path):
    clock = Clock()
    attempts = {"n": 0}
    inner = ogimet_transport(_temp)

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(503)
        if attempts["n"] == 2:
            raise httpx.ReadTimeout("slow", request=request)
        return inner.handle_request(request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    got = fetch_station(
        tmp_path, STATION, [(2024, 1)], client=client, sleep=clock.sleep, clock=clock.clock,
        backoff_s=10.0,
    )  # fmt: skip
    assert got == ["2024-01"] and attempts["n"] == 3
    assert clock.sleeps == [10.0, 30.0]  # exponential backoff


def test_fetch_gives_up_and_stores_nothing_for_a_bad_response(tmp_path):
    clock = Clock()

    def always_500(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = httpx.Client(transport=httpx.MockTransport(always_500))
    with pytest.raises(StationError, match="giving up"):
        fetch_station(tmp_path, STATION, [(2024, 1)], client=client, sleep=clock.sleep, retries=2)
    assert len(clock.sleeps) == 2

    def quota(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="Quota exceeded\n")

    with pytest.raises(StationError, match="unexpected response line"):
        fetch_station(
            tmp_path,
            STATION,
            [(2024, 1)],
            client=httpx.Client(transport=httpx.MockTransport(quota)),
        )

    def forbidden(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    with pytest.raises(StationError, match="HTTP 403"):
        fetch_station(
            tmp_path,
            STATION,
            [(2024, 1)],
            client=httpx.Client(transport=httpx.MockTransport(forbidden)),
        )
    assert not (tmp_path / "climate" / "station" / STATION / "raw").exists()


def test_load_station_verifies_months_and_hashes_the_manifest(tmp_path):
    client = httpx.Client(transport=ogimet_transport(_temp))
    months = [(2024, 1), (2024, 2)]
    fetch_station(tmp_path, STATION, months, client=client, sleep=lambda s: None)
    one = load_station(tmp_path, STATION, months)
    assert one.station == STATION and one.months == ("2024-01", "2024-02")
    assert np.all(np.diff(one.times_us) > 0)
    first = datetime(2024, 1, 1, tzinfo=UTC)
    assert one.times_us[0] == int(first.timestamp()) * 10**6
    assert one.temp_c[0] == pytest.approx(_temp(first))
    assert load_station(tmp_path, STATION, months).manifest_sha256 == one.manifest_sha256
    # a different set of months is a different input
    fetch_station(tmp_path, STATION, [(2024, 3)], client=client, sleep=lambda s: None)
    assert load_station(tmp_path, STATION, [(2024, 1)]).manifest_sha256 != one.manifest_sha256
    # a month that was never fetched, and a raw file that no longer matches its manifest
    with pytest.raises(StationError, match="lacks the month 2024-05"):
        load_station(tmp_path, STATION, [(2024, 5)])
    raw = tmp_path / "climate" / "station" / STATION / "raw" / "2024-01.csv"
    raw.chmod(0o600)
    raw.write_bytes(raw.read_bytes().replace(b"AAXX", b"AAXX ", 1))
    with pytest.raises(StationError, match="manifest sha256"):
        load_station(tmp_path, STATION, months)


def test_the_manifest_hash_follows_the_parser_version(tmp_path, monkeypatch):
    client = httpx.Client(transport=ogimet_transport(_temp))
    fetch_station(tmp_path, STATION, [(2024, 1)], client=client, sleep=lambda s: None)
    before = load_station(tmp_path, STATION, [(2024, 1)]).manifest_sha256
    monkeypatch.setattr(st, "PARSER_VERSION", "synop-next")
    assert load_station(tmp_path, STATION, [(2024, 1)]).manifest_sha256 != before


def test_station_ids_are_five_digit_wmo_numbers(tmp_path):
    for bad in ("8222", "abcde", "../1234", ""):
        with pytest.raises(StationError, match="5-digit"):
            st.station_dir(tmp_path, bad)


def test_csv_helper_round_trip():
    times = hourly(datetime(2024, 1, 1, tzinfo=UTC), 3)
    t, v = parse_getsynop(synop_csv(STATION, times, [1.5, -0.1, None]), STATION)
    assert v.tolist() == [1.5, -0.1] and t.size == 2
