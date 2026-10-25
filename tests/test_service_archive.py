"""Passive archives: PVPC publication snapshots and the weather-forecast archive."""

from __future__ import annotations

import asyncio
import json
import stat
from datetime import datetime, timedelta

import pytest
from service_fakes import (
    NOW,
    TARIFF,
    TZ,
    WEATHER_FORECAST,
    FakeHAServer,
    entity_state,
    feed,
    pvpc_attributes,
    pvpc_state,
    service_config,
    utc,
    wait_until,
)

from battery_optimizer.service.archive import next_fetch_time, retry_delay_s
from battery_optimizer.service.clock import FakeClock
from battery_optimizer.service.config import load_service_config
from battery_optimizer.service.ha_client import HACommandError
from battery_optimizer.service.runtime import ShadowService
from battery_optimizer.service.sink import iter_records
from battery_optimizer.service.timegrid import local_date

PVPC = "sensor.esios_pvpc"
ARCHIVE_ON = {
    "archive.pvpc_snapshots.enabled": True,
    "archive.weather_forecast.enabled": True,
}


def make_service(tmp_path, clock=NOW, server=None, **overrides):
    clock = FakeClock(clock) if isinstance(clock, datetime) else clock
    service = ShadowService(
        config=service_config(tmp_path, **{**ARCHIVE_ON, **overrides}),
        tariff=TARIFF,
        data_dir=tmp_path / "data",
        clock=clock,
        code_version="test-sha",
        ws_transport=server or FakeHAServer(),
    )
    return service


@pytest.fixture
def cleanup():
    services = []
    yield services.append
    for service in services:
        service._executor.shutdown(wait=True)
        service._io.shutdown(wait=True)
        service._history_pool.shutdown(wait=True)


def counter(service, name, **labels):
    value = service.metrics.registry.get_sample_value(f"battery_optimizer_{name}_total", labels)
    assert value is not None, (name, labels)
    return value


def records(tmp_path, kind):
    root = tmp_path / "data" / "service" / "v1" / kind
    return [json.loads(line) for path in sorted(root.rglob("*.jsonl")) for line in open(path)]


async def drain(service):
    """Let spawned archive tasks and their executor writes finish."""
    for _ in range(5):
        await asyncio.sleep(0.01)
    await asyncio.get_running_loop().run_in_executor(service._io, lambda: None)


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("after", "interval", "offset", "expected"),
    [
        (utc(2026, 10, 20, 10, 50), 60, 5, utc(2026, 10, 20, 11, 5)),
        (utc(2026, 10, 20, 11, 4, 59), 60, 5, utc(2026, 10, 20, 11, 5)),
        (utc(2026, 10, 20, 11, 5), 60, 5, utc(2026, 10, 20, 12, 5)),  # strictly after
        (utc(2026, 10, 20, 11, 5, 0, 1), 60, 5, utc(2026, 10, 20, 12, 5)),
        (utc(2026, 10, 20, 23, 30), 60, 5, utc(2026, 10, 21, 0, 5)),
        (utc(2026, 10, 20, 0, 0), 60, 5, utc(2026, 10, 20, 0, 5)),
        (utc(2026, 10, 20, 10, 50), 30, 5, utc(2026, 10, 20, 11, 5)),
        (utc(2026, 10, 20, 11, 6), 30, 5, utc(2026, 10, 20, 11, 35)),
        (utc(2026, 10, 20, 10, 50), 60, 0, utc(2026, 10, 20, 11, 0)),
        (utc(2026, 10, 20, 10, 50), 1440, 5, utc(2026, 10, 21, 0, 5)),
    ],
)
def test_next_fetch_time_is_a_fixed_utc_grid(after, interval, offset, expected):
    assert next_fetch_time(after, interval, offset) == expected


def test_next_fetch_time_ignores_the_local_zone():
    local = datetime(2026, 10, 20, 12, 50, tzinfo=TZ)  # 10:50 UTC
    assert next_fetch_time(local, 60, 5) == utc(2026, 10, 20, 11, 5)


def test_retry_backoff_is_capped():
    assert [retry_delay_s(n) for n in (1, 2, 3, 4, 5, 6)] == [30, 60, 120, 240, 480, 600]
    assert retry_delay_s(10_000) == 600


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_committed_config_enables_both_archives():
    from service_fakes import REPO

    config = load_service_config(REPO / "config" / "service.yaml")
    assert config.archive.pvpc_snapshots.enabled
    weather = config.archive.weather_forecast
    assert weather.enabled
    assert (weather.entity_id, weather.type) == ("weather.forecast_home", "hourly")
    assert (weather.interval_minutes, weather.offset_minutes) == (60, 5)


def test_archives_are_off_when_the_section_is_absent(tmp_path):
    import yaml
    from service_fakes import REPO

    raw = yaml.safe_load((REPO / "config" / "service.yaml").read_text())
    raw.pop("archive")
    path = tmp_path / "service.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = load_service_config(path)
    assert not config.archive.pvpc_snapshots.enabled
    assert not config.archive.weather_forecast.enabled


def test_archive_settings_do_not_change_the_planning_config_hash(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    on = service_config(tmp_path / "a", **ARCHIVE_ON)
    off = service_config(
        tmp_path / "b",
        **{"archive.pvpc_snapshots.enabled": False, "archive.weather_forecast.enabled": False},
    )
    assert on.config_hash() == off.config_hash()
    assert on.public_dict()["archive"]["weather_forecast"]["enabled"] is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"archive.weather_forecast.entity_id": "switch.heater"}, "weather"),
        ({"archive.weather_forecast.entity_id": "weather.a,switch.b"}, "weather"),
        ({"archive.weather_forecast.entity_id": None}, "weather"),
        ({"archive.weather_forecast.type": "minutely"}, "type"),
        ({"archive.weather_forecast.interval_minutes": 0}, "interval_minutes"),
        ({"archive.weather_forecast.interval_minutes": 7}, "divide 1440"),
        ({"archive.weather_forecast.interval_minutes": 60.5}, "interval_minutes"),
        ({"archive.weather_forecast.offset_minutes": 60}, "below interval_minutes"),
        ({"archive.weather_forecast.offset_minutes": -1}, "offset_minutes"),
        ({"archive.weather_forecast.enabled": "yes"}, "true or false"),
        ({"archive.pvpc_snapshots.enabled": 1}, "true or false"),
        ({"archive.weather_forecast.interval": 60}, "unknown archive.weather_forecast"),
        ({"archive.pvpc_snapshots.path": "/tmp/x"}, "unknown archive.pvpc_snapshots"),
        ({"archive.other": {}}, "unknown archive keys"),
    ],
)
def test_invalid_archive_config_is_refused(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        service_config(tmp_path, **override)


# ---------------------------------------------------------------------------
# PVPC publication snapshots
# ---------------------------------------------------------------------------


def test_every_price_state_change_is_snapshotted_verbatim(tmp_path, cleanup):
    service = make_service(tmp_path)
    cleanup(service)
    day = local_date(NOW, TZ)
    first = NOW.replace(minute=0)  # 10:00Z hourly update
    published = utc(2026, 10, 20, 18, 17, 3)  # 20:17 CEST: next day appears

    async def go():
        service._loop = asyncio.get_running_loop()
        service.store.apply(pvpc_state(day, first))
        service.store.apply(entity_state(PVPC, 0.15, first, reported=first + timedelta(seconds=60)))
        service.clock.set(published + timedelta(seconds=2))
        attrs = pvpc_attributes(day, tomorrow=True)
        attrs["odd"] = {"nested": [1, 2.5, None, "x"]}
        service.store.apply(entity_state(PVPC, 0.31, published, attributes=attrs))
        await drain(service)
        return attrs

    attrs = asyncio.run(go())
    rows = records(tmp_path, "pvpc_snapshots")
    # the REFRESHED observation (same state re-confirmed) is not a change
    assert [r["trigger"] for r in rows] == ["initial", "change"]
    initial, change = rows
    assert initial["next_day_present"] is False
    assert initial["state"] == "0.15"
    assert initial["last_updated"] == first.isoformat()
    assert change["next_day_present"] is True
    assert change["state"] == "0.31"
    assert change["attributes"] == attrs  # raw, untouched
    assert change["reference_day"] == day.isoformat()
    assert change["last_updated"] == change["last_changed"] == published.isoformat()
    assert change["last_reported"] == published.isoformat()
    assert change["observed_at"] == (published + timedelta(seconds=2)).isoformat()
    assert change["entity_id"] == PVPC
    assert change["kind"] == "pvpc_snapshot"
    assert change["schema_version"] == 1
    assert change["code_version"] == "test-sha"
    assert counter(service, "pvpc_snapshots", result="written") == 2
    assert counter(service, "pvpc_snapshots", result="failed") == 0
    gauge = service.metrics.registry.get_sample_value(
        "battery_optimizer_archive_last_success_timestamp_seconds", {"archive": "pvpc_snapshots"}
    )
    assert gauge == pytest.approx(service.clock.now().timestamp(), abs=5)
    assert service.status()["archives"]["pvpc_snapshots"]["enabled"] is True


def test_reference_day_follows_the_local_date_of_last_updated(tmp_path, cleanup):
    service = make_service(tmp_path)
    cleanup(service)
    late = utc(2026, 10, 20, 22, 30)  # 00:30 CEST on the 21st

    async def go():
        service._loop = asyncio.get_running_loop()
        service.store.apply(pvpc_state(local_date(late, TZ), late))
        await drain(service)

    asyncio.run(go())
    (row,) = records(tmp_path, "pvpc_snapshots")
    assert row["reference_day"] == "2026-10-21"


def test_unavailable_price_states_are_archived_too(tmp_path, cleanup):
    service = make_service(tmp_path)
    cleanup(service)

    async def go():
        service._loop = asyncio.get_running_loop()
        service.store.apply(entity_state(PVPC, "unavailable", NOW, attributes={}))
        await drain(service)

    asyncio.run(go())
    (row,) = records(tmp_path, "pvpc_snapshots")
    assert row["state"] == "unavailable"
    assert row["attributes"] == {} and row["next_day_present"] is False
    assert service.price_curve is None  # planning still saw (and refused) the state


def test_snapshots_are_partitioned_by_utc_day_and_private(tmp_path, cleanup):
    service = make_service(tmp_path)
    cleanup(service)
    before = utc(2026, 10, 20, 23, 59, 30)
    after = utc(2026, 10, 21, 0, 0, 30)

    async def go():
        service._loop = asyncio.get_running_loop()
        for at in (before, after):
            service.clock.set(at)
            service.store.apply(pvpc_state(local_date(at, TZ), at))
            await drain(service)

    asyncio.run(go())
    root = tmp_path / "data" / "service" / "v1" / "pvpc_snapshots"
    files = sorted(root.rglob("*.jsonl"))
    assert [f.parent.name for f in files] == ["date=2026-10-20", "date=2026-10-21"]
    assert files[0].name == "pvpc_snapshots.jsonl"
    for file in files:
        assert stat.S_IMODE(file.stat().st_mode) == 0o600
        assert stat.S_IMODE(file.parent.stat().st_mode) == 0o700
        assert file.read_text().endswith("\n")
    rows = list(iter_records(tmp_path / "data", "pvpc_snapshots", before, after))
    assert len(rows) == 2


def test_snapshot_write_failures_are_counted_and_never_reach_planning(
    tmp_path, cleanup, monkeypatch
):
    service = make_service(tmp_path)
    cleanup(service)
    feed(service, NOW)  # price curve parsed, listener ran with no loop: nothing archived yet
    assert service.price_curve is not None

    def boom(at, record):
        raise OSError("disk full")

    monkeypatch.setattr(service.sink, "append_pvpc_snapshot", boom)
    day = local_date(NOW, TZ)
    when = NOW + timedelta(minutes=5)

    async def go():
        service._loop = asyncio.get_running_loop()
        service.store.apply(pvpc_state(day, when, tomorrow=True))
        await drain(service)
        # a value JSON cannot carry (the sink refuses NaN) fails the same way
        monkeypatch.undo()
        attrs = pvpc_attributes(day, tomorrow=True)
        attrs["bad"] = float("nan")
        service.store.apply(entity_state(PVPC, 0.2, when + timedelta(minutes=1), attributes=attrs))
        await drain(service)

    asyncio.run(go())
    assert counter(service, "pvpc_snapshots", result="failed") == 2
    assert counter(service, "pvpc_snapshots", result="written") == 0
    assert records(tmp_path, "pvpc_snapshots") == []
    assert service.price_curve is not None and service.price_curve.tomorrow_published
    assert service.price_error is None
    assert "prices_missing" not in [r for r, _ in service.evaluate_inputs(NOW)]


def test_disabled_snapshots_write_nothing(tmp_path, cleanup):
    service = make_service(tmp_path, **{"archive.pvpc_snapshots.enabled": False})
    cleanup(service)

    async def go():
        service._loop = asyncio.get_running_loop()
        service.store.apply(pvpc_state(local_date(NOW, TZ), NOW))
        await drain(service)

    asyncio.run(go())
    assert records(tmp_path, "pvpc_snapshots") == []
    assert counter(service, "pvpc_snapshots", result="written") == 0


def test_only_the_price_entity_is_snapshotted(tmp_path, cleanup):
    service = make_service(tmp_path)
    cleanup(service)

    async def go():
        service._loop = asyncio.get_running_loop()
        feed(service, NOW, history=False)  # power + price
        await drain(service)

    asyncio.run(go())
    assert {r["entity_id"] for r in records(tmp_path, "pvpc_snapshots")} == {PVPC}


# ---------------------------------------------------------------------------
# Weather forecast archive
# ---------------------------------------------------------------------------


async def synced_service(service, server):
    task = asyncio.create_task(service.ha.run())
    await wait_until(lambda: service.ha.stats.synced)
    return task


async def stop(service, task):
    await service.ha.stop()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def test_weather_forecast_is_fetched_with_the_guarded_call_and_archived(tmp_path, cleanup):
    server = FakeHAServer()
    service = make_service(tmp_path, server=server)
    cleanup(service)

    async def go():
        task = await synced_service(service, server)
        ok = await service.archive_weather_forecast()
        await stop(service, task)
        return ok

    assert asyncio.run(go()) is True
    (call,) = server.service_calls
    assert call == {
        "id": call["id"],
        "type": "call_service",
        "domain": "weather",
        "service": "get_forecasts",
        "service_data": {"type": "hourly"},
        "target": {"entity_id": "weather.forecast_home"},
        "return_response": True,
    }
    (row,) = records(tmp_path, "weather_forecasts")
    assert row["kind"] == "weather_forecast"
    assert row["entity_id"] == "weather.forecast_home"
    assert row["type"] == "hourly"
    assert row["requested_at"] == NOW.isoformat()
    assert row["received_at"] == NOW.isoformat()
    assert row["forecast"] == WEATHER_FORECAST  # the raw list, entries untouched
    assert row["schema_version"] == 1 and row["code_version"] == "test-sha"
    path = next((tmp_path / "data" / "service" / "v1" / "weather_forecasts").rglob("*.jsonl"))
    assert path.parent.name == "date=2026-10-20" and path.name == "weather_forecasts.jsonl"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert counter(service, "weather_forecast_fetches", result="ok") == 1
    assert counter(service, "weather_forecast_fetches", result="failed") == 0
    gauge = service.metrics.registry.get_sample_value(
        "battery_optimizer_archive_last_success_timestamp_seconds", {"archive": "weather_forecast"}
    )
    assert gauge == pytest.approx(NOW.timestamp(), abs=1)
    status = service.status()["archives"]["weather_forecast"]
    assert status["last_error"] is None and status["last_success_at"]


def test_configured_entity_and_type_reach_the_call(tmp_path, cleanup):
    server = FakeHAServer()
    service = make_service(
        tmp_path,
        server=server,
        **{
            "archive.weather_forecast.entity_id": "weather.home_2",
            "archive.weather_forecast.type": "daily",
        },
    )
    cleanup(service)

    async def go():
        task = await synced_service(service, server)
        await service.archive_weather_forecast()
        await stop(service, task)

    asyncio.run(go())
    (call,) = server.service_calls
    assert call["target"] == {"entity_id": "weather.home_2"}
    assert call["service_data"] == {"type": "daily"}
    (row,) = records(tmp_path, "weather_forecasts")
    assert (row["entity_id"], row["type"]) == ("weather.home_2", "daily")


@pytest.mark.parametrize(
    "behaviour",
    [
        {"weather_error": True},  # token not allowed / HA refuses
        {"weather_silent": True},  # no answer: request timeout
        {"weather_forecast": []},  # empty forecast
        {"weather_response": {}},  # entity missing from the response
        {"weather_response": {"weather.forecast_home": {"forecast": "x"}}},
        {"weather_response": {"weather.forecast_home": {"forecast": [{"t": float("nan")}]}}},
    ],
)
def test_failed_fetches_are_counted_and_write_nothing(tmp_path, cleanup, behaviour):
    server = FakeHAServer()
    server.behaviours = [behaviour]
    service = make_service(tmp_path, server=server)
    cleanup(service)

    async def go():
        task = await synced_service(service, server)
        ok = await service.archive_weather_forecast()
        await stop(service, task)
        return ok

    assert asyncio.run(go()) is False
    assert records(tmp_path, "weather_forecasts") == []
    assert counter(service, "weather_forecast_fetches", result="failed") == 1
    assert counter(service, "weather_forecast_fetches", result="ok") == 0
    assert service.status()["archives"]["weather_forecast"]["last_error"]


def test_a_failed_fetch_never_changes_readiness_or_the_session(tmp_path, cleanup):
    server = FakeHAServer()
    server.behaviours = [{"weather_error": True}]
    service = make_service(tmp_path, server=server)
    cleanup(service)
    feed(service, NOW)
    before = service.readiness(NOW)

    async def go():
        task = await synced_service(service, server)
        ok = await service.archive_weather_forecast()
        still_synced = service.ha.stats.synced
        await stop(service, task)
        return ok, still_synced

    ok, still_synced = asyncio.run(go())
    assert ok is False and still_synced
    assert service.ha.stats.failures == {}  # a refused call is not a session failure
    after = service.readiness(NOW)
    assert (after.reasons, after.inputs_ready) == (before.reasons, before.inputs_ready)


def test_fetch_without_a_session_fails_cleanly(tmp_path, cleanup):
    service = make_service(tmp_path)
    cleanup(service)
    assert asyncio.run(service.archive_weather_forecast()) is False
    assert counter(service, "weather_forecast_fetches", result="failed") == 1
    with pytest.raises(HACommandError):
        asyncio.run(service.ha.get_weather_forecast("weather.forecast_home"))


def test_a_session_that_ends_mid_request_is_a_failure_not_a_crash(tmp_path, cleanup):
    server = FakeHAServer()
    server.behaviours = [{"weather_silent": True}]
    service = make_service(tmp_path, server=server)
    cleanup(service)

    async def go():
        task = await synced_service(service, server)
        fetch = asyncio.create_task(service.archive_weather_forecast())
        await wait_until(lambda: server.service_calls)
        server.connections[0].drop()  # the socket dies while the call is pending
        ok = await asyncio.wait_for(fetch, timeout=3)
        await stop(service, task)
        return ok

    assert asyncio.run(go()) is False
    assert counter(service, "weather_forecast_fetches", result="failed") == 1


def test_the_guard_still_blocks_other_calls_through_the_client(tmp_path, cleanup):
    server = FakeHAServer()
    service = make_service(tmp_path, server=server)
    cleanup(service)

    async def go():
        task = await synced_service(service, server)
        with pytest.raises(PermissionError):
            await service.ha.get_weather_forecast("switch.heater")
        with pytest.raises(PermissionError):
            await service.ha.get_weather_forecast("weather.forecast_home", "minutely")
        await stop(service, task)

    asyncio.run(go())
    assert server.service_calls == []
    assert {m["type"] for m in server.all_sent} <= {
        "auth",
        "subscribe_events",
        "get_states",
        "ping",
    }


# -- the schedule loop ----------------------------------------------------------


def run_loop(service, script):
    """Run the archive loop with a scripted fetch until the script is used up."""
    calls: list[datetime] = []

    async def fetch():
        calls.append(service.clock.now())
        return script[len(calls) - 1] if len(calls) <= len(script) else True

    service.archive_weather_forecast = fetch  # type: ignore[method-assign]

    async def go():
        task = asyncio.create_task(service._weather_archive_loop())
        await wait_until(lambda: len(calls) >= len(script))
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(go())
    return calls[: len(script)]


def test_loop_fetches_on_the_hourly_grid_and_not_at_start(tmp_path, cleanup):
    service = make_service(tmp_path)
    cleanup(service)
    calls = run_loop(service, [True, True, True])
    assert calls == [utc(2026, 10, 20, 11, 5), utc(2026, 10, 20, 12, 5), utc(2026, 10, 20, 13, 5)]


def test_loop_backs_off_after_failures_then_resumes_the_grid(tmp_path, cleanup):
    service = make_service(tmp_path)
    cleanup(service)
    calls = run_loop(service, [False, False, False, True, True])
    assert calls == [
        utc(2026, 10, 20, 11, 5),
        utc(2026, 10, 20, 11, 5, 30),  # +30 s
        utc(2026, 10, 20, 11, 6, 30),  # +60 s
        utc(2026, 10, 20, 11, 8, 30),  # +120 s: succeeds
        utc(2026, 10, 20, 12, 5),  # back on the grid
    ]


def test_loop_backoff_never_overruns_the_next_grid_instant(tmp_path, cleanup):
    service = make_service(tmp_path, **{"archive.weather_forecast.interval_minutes": 10})
    cleanup(service)
    # grid (offset 5 of every 10 min): 10:55, 11:05, 11:15, ...
    calls = run_loop(service, [False] * 8 + [True])
    grid = {utc(2026, 10, 20, 10, 55) + timedelta(minutes=10 * i) for i in range(6)}
    assert calls[0] == utc(2026, 10, 20, 10, 55)
    assert all(b > a for a, b in zip(calls, calls[1:], strict=False))
    # every grid instant inside the failure streak is still visited
    assert grid & set(calls) >= {utc(2026, 10, 20, 11, 5), utc(2026, 10, 20, 11, 15)}


def test_loop_survives_an_unexpected_exception(tmp_path, cleanup):
    service = make_service(tmp_path)
    cleanup(service)
    calls: list[datetime] = []

    async def fetch():
        calls.append(service.clock.now())
        if len(calls) == 1:
            raise RuntimeError("unexpected")
        return True

    service.archive_weather_forecast = fetch  # type: ignore[method-assign]

    async def go():
        task = asyncio.create_task(service._weather_archive_loop())
        await wait_until(lambda: len(calls) >= 2)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(go())
    assert len(calls) >= 2
    assert counter(service, "weather_forecast_fetches", result="failed") == 1


def test_start_runs_the_archive_task_only_when_enabled_and_connected(tmp_path, cleanup):
    async def names(service):
        await service.start()
        found = {t.get_name() for t in service._tasks}
        await service.stop()
        return found

    (tmp_path / "on").mkdir()
    (tmp_path / "off").mkdir()
    on = make_service(tmp_path / "on")
    off = make_service(tmp_path / "off", **{"archive.weather_forecast.enabled": False})
    for service in (on, off):
        cleanup(service)
    assert "weather-archive" in asyncio.run(names(on))
    assert "weather-archive" not in asyncio.run(names(off))
