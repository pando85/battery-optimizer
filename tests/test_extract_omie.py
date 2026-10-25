"""OMIE day-ahead extraction, validation and the ES-price reader.

Offline and synthetic: every file is built here in the public ``marginalpdbc`` format (never a
downloaded sample), served through ``httpx.MockTransport``, and written under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pyarrow.parquet as pq
import pytest
from test_build import (
    _T0,
    GRID,
    aligned_times,
    build,
    counter_rows,
    write_raw_part,
    write_tariff,
)

import battery_optimizer.extract_omie as eo
from battery_optimizer import cli
from battery_optimizer.coverage import EXIT_COVERAGE_GAPS, coverage_gap_flags
from battery_optimizer.extract_omie import (
    OmieError,
    OmieNotPublishedError,
    OmieParseError,
    expected_published_at,
    extract_omie,
    index_stored_days,
    local_day_window,
    parse_marginalpdbc,
    part_delivery_date,
    resolve_omie_dates,
    source_url,
)
from battery_optimizer.omie_prices import (
    OmieReadError,
    load_omie_es_prices,
    omie_usable_from,
)
from battery_optimizer.paths import iter_parts, parts_dir
from battery_optimizer.schemas import RAW_OMIE_DAY_AHEAD, SCHEMA_VERSION, validate_table

CFG = str(Path(__file__).resolve().parent.parent / "config" / "tariff.yaml")
MADRID = ZoneInfo("Europe/Madrid")
D = date(2026, 1, 15)
HTML = b"<!DOCTYPE html>\n<html><head><title>OMIE</title></head><body>no file</body></html>\n"
#: Fixed fetch time: well after the publication of any day used below.
LATE = datetime(2026, 12, 1, 12, 0, tzinfo=UTC)


def periods_in_day(day: date, minutes: int) -> int:
    start, end = local_day_window(day)
    return (end - start) // timedelta(minutes=minutes)


def omie_file(
    day: date,
    minutes: int | None = None,
    *,
    count: int | None = None,
    price: Callable[[int], tuple[float, float]] | None = None,
    eol: str = "\n",
) -> bytes:
    """A ``marginalpdbc`` file; default resolution follows the 2025-10-01 switch."""
    if minutes is None:
        minutes = 15 if day >= date(2025, 10, 1) else 60
    n = count if count is not None else periods_in_day(day, minutes)

    def default(i: int) -> tuple[float, float]:
        return round(50.0 + i * 0.5, 2), round(60.0 + i * 0.5, 2)  # (PT, ES)

    fn = price or default
    lines = ["MARGINALPDBC;"]
    for i in range(1, n + 1):
        pt, es = fn(i)
        lines.append(f"{day.year:04d};{day.month:02d};{day.day:02d};{i};{pt};{es};")
    lines.append("*")
    return (eol.join(lines) + eol).encode()


class Net:
    """A fake OMIE: ``files[date] -> (status, body)`` or a callable for custom behaviour."""

    def __init__(
        self,
        files: dict[date, tuple[int, bytes]] | None = None,
        *,
        default=None,
        versions: dict[tuple[date, int], tuple[int, bytes]] | None = None,
    ):
        """``files``/``default`` answer version ``.1``; later versions are 404 unless listed."""
        self.files = files or {}
        self.default = default
        self.versions = versions or {}
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        name = request.url.params["filename"]
        stem, _, suffix = name.rpartition(".")
        day = datetime.strptime(stem, "marginalpdbc_%Y%m%d").date()
        version = int(suffix)
        if (day, version) in self.versions:
            status, body = self.versions[(day, version)]
        elif version > 1:
            status, body = 404, HTML
        elif day in self.files:
            status, body = self.files[day]
        elif self.default is not None:
            status, body = self.default(day)
        else:
            status, body = 200, omie_file(day)
        return httpx.Response(status, content=body)

    def filenames(self) -> list[str]:
        return [r.url.params["filename"] for r in self.requests]

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


class FakeTime:
    """Deterministic clock/sleep pair: sleeping advances the clock."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    path = tmp_path / "data"
    path.mkdir()
    return path


def run(
    data_dir: Path,
    net: Net,
    start: date,
    end: date,
    *,
    now: datetime = LATE,
    **kwargs,
) -> dict:
    ft = kwargs.pop("ft", None) or FakeTime()
    path = extract_omie(
        data_dir=str(data_dir),
        config_path=CFG,
        start=start,
        end=end,
        client=net.client(),
        sleep=ft.sleep,
        clock=ft.clock,
        now=lambda: now,
        **kwargs,
    )
    return json.loads(path.read_text())


def manifests(data_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((data_dir / "manifests").glob("*.json"))]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


class TestParse:
    @pytest.mark.parametrize(
        ("day", "minutes", "count"),
        [
            (date(2026, 1, 15), 60, 24),
            (date(2026, 3, 29), 60, 23),  # spring forward, hourly era layout
            (date(2025, 10, 26), 60, 25),  # fall back, hourly layout
            (date(2026, 1, 15), 15, 96),
            (date(2026, 3, 29), 15, 92),
            (date(2026, 10, 25), 15, 100),
        ],
    )
    def test_period_counts_follow_the_local_day(self, day, minutes, count):
        parsed = parse_marginalpdbc(omie_file(day, minutes), day)
        assert (len(parsed.periods), parsed.period_minutes) == (count, minutes)
        start, end = local_day_window(day)
        assert parsed.periods[0].interval_start == start
        assert parsed.periods[-1].interval_end == end
        step = timedelta(minutes=minutes)
        assert all(p.interval_end - p.interval_start == step for p in parsed.periods)
        assert all(
            a.interval_end == b.interval_start
            for a, b in zip(parsed.periods, parsed.periods[1:], strict=False)
        )

    def test_spring_forward_skips_the_missing_local_hour(self):
        day = date(2026, 3, 29)
        parsed = parse_marginalpdbc(omie_file(day, 15), day)
        local_hours = [p.interval_start.astimezone(MADRID).hour for p in parsed.periods]
        assert 2 not in local_hours
        assert local_hours.count(3) == 4

    def test_fall_back_repeats_the_local_hour_in_utc_order(self):
        day = date(2026, 10, 25)
        parsed = parse_marginalpdbc(omie_file(day, 15), day)
        local_hours = [p.interval_start.astimezone(MADRID).hour for p in parsed.periods]
        assert local_hours.count(2) == 8  # 02:00-02:59 happens twice
        offsets = {p.interval_start.astimezone(MADRID).utcoffset() for p in parsed.periods}
        assert offsets == {timedelta(hours=2), timedelta(hours=1)}

    def test_pt_first_es_second_and_negative_prices(self):
        day = date(2026, 1, 15)
        body = omie_file(day, 60, price=lambda i: (10.5, -3.25) if i == 1 else (1.0, 2.0))
        first = parse_marginalpdbc(body, day).periods[0]
        assert (first.price_pt_eur_mwh, first.price_es_eur_mwh) == (10.5, -3.25)

    def test_crlf_bom_and_trailing_blank_lines(self):
        day = date(2026, 1, 15)
        body = b"\xef\xbb\xbf" + omie_file(day, 60, eol="\r\n") + b"\r\n\r\n"
        assert len(parse_marginalpdbc(body, day).periods) == 24

    def test_html_is_not_published(self):
        with pytest.raises(OmieNotPublishedError):
            parse_marginalpdbc(HTML, D)

    @pytest.mark.parametrize(
        ("mutate", "message"),
        [
            (lambda t: t.replace("MARGINALPDBC;", "OTHER;", 1), "first line"),
            (lambda t: t.rsplit("*", 1)[0], "terminator"),
            (lambda t: t.replace("2026;01;15;3;", "2026;01;16;3;", 1), "!= requested"),
            (lambda t: t.replace(";3;", ";4;", 1), "period 4 where 3"),
            (lambda t: t.replace("51.0;61.0;", "51,0;61,0;", 1), "not a price line"),
            (lambda t: t.replace("51.0;61.0;", "nan;61.0;", 1), "not a price line"),
            (lambda t: t.replace("51.0;61.0;", "51.0;", 1), "not a price line"),
            (lambda t: t + "2026;01;15;99;1;1;\n", "terminator"),
            (lambda t: "", "empty"),
        ],
    )
    def test_strict_rejection(self, mutate, message):
        text = omie_file(D, 60).decode()
        with pytest.raises(OmieParseError, match=message):
            parse_marginalpdbc(mutate(text).encode(), D)

    @pytest.mark.parametrize(
        ("day", "minutes", "count"),
        [
            (date(2026, 1, 15), 60, 23),
            (date(2026, 1, 15), 60, 25),
            (date(2026, 1, 15), 15, 95),
            (date(2026, 10, 25), 60, 24),  # a 24-hour file on a 25-hour day
            (date(2026, 3, 29), 15, 96),
        ],
    )
    def test_period_count_must_fit_the_local_day(self, day, minutes, count):
        with pytest.raises(OmieParseError, match="do not fit"):
            parse_marginalpdbc(omie_file(day, minutes, count=count), day)

    def test_non_utf8_is_a_parse_error(self):
        with pytest.raises(OmieParseError, match="UTF-8"):
            parse_marginalpdbc(b"\xff\xfe\x00MARGINALPDBC;", D)

    def test_url_names_the_day_file(self):
        assert source_url(date(2026, 1, 5)).endswith(
            "?parents=marginalpdbc&filename=marginalpdbc_20260105.1"
        )


class TestDates:
    def test_inclusive_range_and_default_end(self):
        today = date(2026, 10, 25)
        assert resolve_omie_dates("2026-10-01", "2026-10-03", None, today=today) == (
            date(2026, 10, 1),
            date(2026, 10, 3),
        )
        assert resolve_omie_dates("2026-10-01", None, None, today=today) == (
            date(2026, 10, 1),
            today,
        )
        assert resolve_omie_dates(None, None, 3, today=today) == (date(2026, 10, 22), today)

    @pytest.mark.parametrize("bad", ["2026-10-01T00:00:00Z", "20261001", "2026-13-01", ""])
    def test_timestamps_and_garbage_are_refused(self, bad):
        with pytest.raises(ValueError, match="date"):
            resolve_omie_dates(bad, "2026-10-03", None)

    def test_reversed_and_missing(self):
        with pytest.raises(ValueError, match="after"):
            resolve_omie_dates("2026-10-03", "2026-10-01", None)
        with pytest.raises(ValueError, match="required"):
            resolve_omie_dates(None, None, None)

    def test_part_name_window_roundtrips_to_the_local_date(self):
        for day in (date(2026, 3, 29), date(2026, 10, 25), date(2026, 1, 1)):
            s, e = local_day_window(day)
            name = f"omie-day-{s:%Y%m%dT%H%M%SZ}-{e:%Y%m%dT%H%M%SZ}-abcdef12.parquet"
            assert part_delivery_date(name) == day
        assert part_delivery_date("prom-samples-x.parquet") is None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class TestExtract:
    def test_one_part_per_day_with_provenance(self, data_dir):
        days = {date(2025, 9, 30): 24, date(2025, 10, 1): 96, date(2026, 3, 29): 92}
        net = Net()
        ft = FakeTime()
        manifest = {}
        for day in days:
            manifest = run(data_dir, net, day, day, ft=ft)
        parts = iter_parts(data_dir, "omie")
        assert len(parts) == 3
        for path in parts:
            table = pq.read_table(path)
            validate_table("RAW_OMIE_DAY_AHEAD", table)
            day = table.column("delivery_date")[0].as_py()
            assert table.num_rows == days[day]
            assert (
                path.name.rsplit("-", 1)[-1]
                == hashlib.sha256(path.read_bytes()).hexdigest()[:8] + ".parquet"
            )
            assert part_delivery_date(path) == day
            assert set(table.column("observed_at").to_pylist()) == {LATE}
            assert set(table.column("source_sha256").to_pylist()) == {
                hashlib.sha256(omie_file(day)).hexdigest()
            }
            # ES is the second price column of the file, PT the first.
            assert table.column("price_pt_eur_mwh")[0].as_py() == 50.5
            assert table.column("price_es_eur_mwh")[0].as_py() == 60.5
        assert manifest["source"] == "omie"
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert manifest["finished_at"]
        (entry,) = manifest["part_files"]
        assert entry["raw_sha256"] == hashlib.sha256(omie_file(date(2026, 3, 29))).hexdigest()
        assert (entry["http_status"], entry["periods"], entry["period_minutes"]) == (200, 92, 15)
        assert entry["observed_at"] == "2026-12-01T12:00:00Z"
        assert (
            hashlib.sha256((data_dir / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
        )
        # The recorded endpoint keeps the file name template (no query stripping).
        assert manifest["params"]["query_template"].endswith("marginalpdbc_{yyyymmdd}.{version}")
        assert manifest["params"]["endpoint"].startswith("https://www.omie.es/")
        assert manifest["params"]["days"][0]["raw_sha256"] == entry["raw_sha256"]

    def test_user_agent_is_browser_like(self, data_dir, monkeypatch):
        seen: list[str] = []
        real = httpx.Client

        def factory(**kw):
            seen.append(kw["headers"]["User-Agent"])
            return real(transport=httpx.MockTransport(Net()), **kw)

        monkeypatch.setattr(eo.httpx, "Client", factory)
        extract_omie(
            data_dir=str(data_dir),
            config_path=CFG,
            start=D,
            end=D,
            sleep=lambda _s: None,
            now=lambda: LATE,
        )
        assert seen and seen[0].startswith("Mozilla/5.0")

    def test_rerun_is_idempotent_and_makes_no_request(self, data_dir):
        net = Net()
        run(data_dir, net, date(2026, 1, 10), date(2026, 1, 12))
        before = sorted(p.name for p in iter_parts(data_dir, "omie"))
        assert len(net.requests) == 3
        second = run(data_dir, net, date(2026, 1, 10), date(2026, 1, 12))
        assert len(net.requests) == 3
        assert sorted(p.name for p in iter_parts(data_dir, "omie")) == before
        assert second["part_files"] == []
        assert second["row_count"] == 0
        assert second["params"]["day_counts"] == {"already_stored": 3}
        assert coverage_gap_flags(second["quality_flags"]) == []

    def test_overlapping_window_fetches_only_new_days(self, data_dir):
        net = Net()
        run(data_dir, net, date(2026, 1, 10), date(2026, 1, 12))
        net.requests.clear()
        manifest = run(data_dir, net, date(2026, 1, 11), date(2026, 1, 14))
        fetched = [r.url.params["filename"] for r in net.requests]
        assert fetched == ["marginalpdbc_20260113.1", "marginalpdbc_20260114.1"]
        assert len(iter_parts(data_dir, "omie")) == 5
        assert [p["delivery_date"] for p in manifest["part_files"]] == ["2026-01-13", "2026-01-14"]

    def test_requests_are_spaced_by_the_minimum_interval(self, data_dir):
        ft = FakeTime()
        run(data_dir, Net(), date(2026, 1, 10), date(2026, 1, 13), ft=ft, min_interval_seconds=0.5)
        assert len(ft.sleeps) == 3
        assert all(s == pytest.approx(0.5) for s in ft.sleeps)

    def test_default_interval_is_at_least_half_a_second(self):
        assert eo.MIN_REQUEST_INTERVAL_SECONDS >= 0.5

    def test_unpublished_before_the_expected_time_is_informational(self, data_dir):
        tomorrow = date(2026, 10, 26)
        net = Net({tomorrow: (200, HTML)})
        now = datetime(2026, 10, 25, 11, 24, tzinfo=UTC)  # 12:24 local: before 14:00 local
        manifest = run(data_dir, net, date(2026, 10, 25), tomorrow, now=now)
        assert manifest["params"]["day_counts"] == {"not_published_yet": 1, "stored": 1}
        assert [f.partition(":")[0] for f in manifest["quality_flags"]] == [
            "omie_not_published_yet"
        ]
        assert coverage_gap_flags(manifest["quality_flags"]) == []
        assert len(iter_parts(data_dir, "omie")) == 1  # the 100-period fall-back day

    def test_html_after_the_expected_time_is_a_gap(self, data_dir):
        net = Net({D: (200, HTML)})
        manifest = run(data_dir, net, D, D)
        assert coverage_gap_flags(manifest["quality_flags"]) == [
            f"omie_missing:{D}: no file (HTTP 200/404/404/404/404 for versions 1-5); "
            "expected since 14:00 local"
        ]
        assert iter_parts(data_dir, "omie") == []
        assert manifest["params"]["days"][0]["status"] == "missing"
        # a later run retries the day (nothing was stored for it)
        run(data_dir, Net(), D, D)
        assert len(iter_parts(data_dir, "omie")) == 1

    def test_later_publication_version_is_used_when_v1_is_absent(self, data_dir):
        # Seen live: 2025-10-30 exists only as .3 and 2025-11-27 only as .2.
        net = Net(
            {D: (404, HTML)},
            versions={(D, 3): (200, omie_file(D, 15))},
        )
        manifest = run(data_dir, net, D, D)
        assert net.filenames() == [f"marginalpdbc_{D:%Y%m%d}.{v}" for v in (1, 2, 3)]
        (entry,) = manifest["part_files"]
        assert (entry["file_version"], entry["http_status"]) == (3, 200)
        assert manifest["params"]["days"][0]["http_statuses"] == [404, 404, 200]
        assert manifest["params"]["requests_made"] == 3
        assert manifest["quality_flags"] == []
        (part,) = iter_parts(data_dir, "omie")
        assert set(pq.read_table(part).column("source_file_version").to_pylist()) == {3}

    def test_version_one_wins_and_stops_the_probe(self, data_dir):
        net = Net(versions={(D, 2): (200, omie_file(D, 15))})
        manifest = run(data_dir, net, D, D)
        assert net.filenames() == [f"marginalpdbc_{D:%Y%m%d}.1"]
        assert manifest["part_files"][0]["file_version"] == 1

    def test_html_200_for_a_version_also_moves_to_the_next(self, data_dir):
        net = Net({D: (200, HTML)}, versions={(D, 2): (200, omie_file(D, 15))})
        manifest = run(data_dir, net, D, D)
        assert manifest["part_files"][0]["file_version"] == 2

    def test_every_version_absent_probes_all_and_is_missing(self, data_dir):
        net = Net({D: (404, HTML)})
        manifest = run(data_dir, net, D, D)
        assert len(net.requests) == eo.MAX_FILE_VERSION
        assert [f.partition(":")[0] for f in manifest["quality_flags"]] == ["omie_missing"]

    def test_404_is_treated_like_an_unpublished_day(self, data_dir):
        manifest = run(data_dir, Net({D: (404, b"")}), D, D)
        assert [f.partition(":")[0] for f in manifest["quality_flags"]] == ["omie_missing"]

    def test_expected_publication_is_1400_local_the_day_before(self):
        assert expected_published_at(date(2026, 10, 26)) == datetime(
            2026, 10, 25, 13, 0, tzinfo=UTC
        )  # 14:00 CET: the clocks went back at 03:00 that day
        assert expected_published_at(date(2026, 1, 15)) == datetime(2026, 1, 14, 13, 0, tzinfo=UTC)

    def test_parse_error_is_a_gap_and_stores_nothing(self, data_dir):
        bad = omie_file(D, 60, count=23)
        manifest = run(data_dir, Net({D: (200, bad)}), D, D)
        (flag,) = coverage_gap_flags(manifest["quality_flags"])
        assert flag.startswith(f"omie_parse_error:{D}: 23 periods do not fit")
        assert iter_parts(data_dir, "omie") == []
        entry = manifest["params"]["days"][0]
        assert entry["status"] == "parse_error"
        assert entry["raw_sha256"] == hashlib.sha256(bad).hexdigest()

    def test_price_outside_limits_is_flagged_but_stored(self, data_dir):
        body = omie_file(D, 60, price=lambda i: (4500.0, 4500.0) if i == 1 else (1.0, 1.0))
        manifest = run(data_dir, Net({D: (200, body)}), D, D)
        assert len(iter_parts(data_dir, "omie")) == 1
        assert [f.partition(":")[0] for f in manifest["quality_flags"]] == [
            "omie_price_outside_expected_range"
        ]
        assert coverage_gap_flags(manifest["quality_flags"]) == []

    def test_transient_errors_are_retried_with_backoff(self, data_dir):
        calls = {"n": 0}

        def flaky(day):
            calls["n"] += 1
            if calls["n"] <= 2:
                return 503, b"busy"
            return 200, omie_file(day)

        ft = FakeTime()
        manifest = run(data_dir, Net(default=flaky), D, D, ft=ft, min_interval_seconds=0.0)
        assert manifest["params"]["days"][0]["attempts"] == 3
        assert ft.sleeps == [0.5, 1.0]
        assert len(iter_parts(data_dir, "omie")) == 1

    def test_transport_error_retries_then_fails_the_day(self, data_dir):
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down")

        client = httpx.Client(transport=httpx.MockTransport(boom))
        ft = FakeTime()
        path = extract_omie(
            data_dir=str(data_dir),
            config_path=CFG,
            start=D,
            end=D,
            client=client,
            max_retries=2,
            sleep=ft.sleep,
            clock=ft.clock,
            now=lambda: LATE,
        )
        manifest = json.loads(path.read_text())
        (flag,) = coverage_gap_flags(manifest["quality_flags"])
        assert flag == f"omie_fetch_failed:{D}: ConnectError after 3 attempt(s)"

    def test_other_4xx_is_not_retried(self, data_dir):
        net = Net({D: (403, b"no")})
        manifest = run(data_dir, net, D, D)
        assert len(net.requests) == 1
        assert [f.partition(":")[0] for f in manifest["quality_flags"]] == ["omie_fetch_failed"]

    def test_oversized_response_is_refused(self, data_dir, monkeypatch):
        monkeypatch.setattr(eo, "MAX_RESPONSE_BYTES", 100)
        manifest = run(data_dir, Net(), D, D)
        assert "larger than" in manifest["quality_flags"][0]

    def test_consecutive_failures_abort_without_a_manifest(self, data_dir):
        net = Net(default=lambda day: (403, b"no"))
        with pytest.raises(OmieError, match="3 consecutive"):
            run(data_dir, net, D, D + timedelta(days=9), max_consecutive_failures=3)
        assert manifests(data_dir) == []

    def test_a_success_resets_the_failure_streak(self, data_dir):
        bad = {D + timedelta(days=i) for i in (0, 1, 3, 4)}
        net = Net(default=lambda day: (403, b"no") if day in bad else (200, omie_file(day)))
        manifest = run(data_dir, net, D, D + timedelta(days=4), max_consecutive_failures=3)
        assert manifest["params"]["day_counts"] == {"fetch_failed": 4, "stored": 1}

    def test_orphan_part_is_adopted_not_refetched(self, data_dir):
        run(data_dir, Net(), D, D)
        for p in (data_dir / "manifests").glob("*.json"):
            p.unlink()  # a crash after the part was written, before the manifest
        net = Net()
        manifest = run(data_dir, net, D, D)
        assert net.requests == []
        (entry,) = manifest["part_files"]
        assert entry["adopted"] is True and entry["delivery_date"] == str(D)
        assert [f.partition(":")[0] for f in manifest["quality_flags"]] == [
            "omie_orphan_part_adopted"
        ]
        assert len(iter_parts(data_dir, "omie")) == 1
        # claimed now: a further run is silent
        third = run(data_dir, net, D, D)
        assert third["part_files"] == [] and third["quality_flags"] == []

    def test_unreadable_stored_part_is_a_gap_and_not_refetched(self, data_dir):
        run(data_dir, Net(), D, D)
        (part,) = iter_parts(data_dir, "omie")
        for p in (data_dir / "manifests").glob("*.json"):
            p.unlink()
        part.write_bytes(b"not parquet")
        net = Net()
        manifest = run(data_dir, net, D, D)
        assert net.requests == []
        assert [f.partition(":")[0] for f in coverage_gap_flags(manifest["quality_flags"])] == [
            "omie_part_unreadable"
        ]

    def test_index_reads_names_only(self, data_dir):
        run(data_dir, Net(), D, D + timedelta(days=1))
        assert sorted(index_stored_days(data_dir)) == [D, D + timedelta(days=1)]

    def test_part_is_never_overwritten_in_place(self, data_dir):
        run(data_dir, Net(), D, D)
        (part,) = iter_parts(data_dir, "omie")
        stat = part.stat()
        run(data_dir, Net(), D, D)
        assert part.stat().st_mtime_ns == stat.st_mtime_ns


# ---------------------------------------------------------------------------
# CLI: exit codes and validate
# ---------------------------------------------------------------------------


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("BATTERY_OPTIMIZER_DATA_DIR", str(data_dir))
    monkeypatch.setattr(eo, "MIN_REQUEST_INTERVAL_SECONDS", 0.0)
    net = Net({date(2026, 1, 17): (200, HTML)})
    real = httpx.Client

    def factory(**kw):
        return real(transport=httpx.MockTransport(net), **kw)

    monkeypatch.setattr(eo.httpx, "Client", factory)
    return data_dir, net


def _extract_args(*extra: str, start="2026-01-15", end="2026-01-16") -> list[str]:
    return ["extract", "--source", "omie", "--config", CFG, "--start", start, "--end", end, *extra]


class TestCli:
    def test_clean_run_exits_zero(self, cli_env, capsys):
        data_dir, _ = cli_env
        assert cli.main(_extract_args()) == 0
        assert len(iter_parts(data_dir, "omie")) == 2
        assert "coverage gap" not in capsys.readouterr().err

    def test_rerun_is_a_noop(self, cli_env):
        data_dir, net = cli_env
        assert cli.main(_extract_args()) == 0
        count = len(net.requests)
        assert cli.main(_extract_args()) == 0
        assert len(net.requests) == count
        assert len(iter_parts(data_dir, "omie")) == 2

    def test_coverage_gap_exits_three_and_keeps_data(self, cli_env, capsys):
        data_dir, _ = cli_env
        rc = cli.main(_extract_args(end="2026-01-17"))
        assert rc == EXIT_COVERAGE_GAPS == 3
        err = capsys.readouterr().err
        assert "omie recorded 1 coverage gap flag(s)" in err
        assert "omie_missing:2026-01-17" in err
        assert len(iter_parts(data_dir, "omie")) == 2
        assert len(manifests(data_dir)) == 1

    def test_allow_coverage_gaps_exits_zero(self, cli_env):
        assert cli.main(_extract_args("--allow-coverage-gaps", end="2026-01-17")) == 0

    def test_entities_and_timestamps_are_usage_errors(self, cli_env, capsys):
        assert cli.main(_extract_args("--entities", "sensor.x")) == 1
        assert cli.main(_extract_args(start="2026-01-15T00:00:00Z")) == 1
        assert "delivery dates" in capsys.readouterr().err

    def test_validate_omie_ok_then_detects_tampering(self, cli_env, capsys):
        data_dir, _ = cli_env
        assert cli.main(_extract_args()) == 0
        capsys.readouterr()
        assert cli.main(["validate", "--source", "omie"]) == 0
        out = capsys.readouterr().out
        assert "omie: OK — parts=2 rows=192" in out
        part = iter_parts(data_dir, "omie")[0]
        part.write_bytes(part.read_bytes() + b"x")
        assert cli.main(["validate", "--source", "omie"]) == 1
        assert "filename hash" in capsys.readouterr().out

    def test_validate_reports_a_day_stored_twice(self, cli_env, capsys, tmp_path):
        data_dir, _ = cli_env
        assert cli.main(_extract_args(end="2026-01-15")) == 0
        # a second part for the same window (different content hash): a duplicated day
        table = pq.read_table(iter_parts(data_dir, "omie")[0])
        import pyarrow as pa

        bumped = table.set_column(
            table.schema.get_field_index("price_es_eur_mwh"),
            "price_es_eur_mwh",
            pa.array([1.0] * table.num_rows, pa.float64()),
        )
        from battery_optimizer.extract_ha import table_to_bytes
        from battery_optimizer.paths import part_path, write_part_atomic

        payload = table_to_bytes(bumped)
        s, e = local_day_window(date(2026, 1, 15))
        write_part_atomic(
            part_path(data_dir, "omie", "omie-day", s, e, hashlib.sha256(payload).hexdigest()[:8]),
            payload,
        )
        capsys.readouterr()
        assert cli.main(["validate", "--source", "omie"]) == 1
        assert "2 parts for one delivery day" in capsys.readouterr().out

    def test_validate_all_still_means_prometheus_and_ha(self, cli_env, capsys):
        assert cli.main(_extract_args()) == 0
        capsys.readouterr()
        cli.main(["validate"])
        assert "omie:" not in capsys.readouterr().out


class TestCoverageClassification:
    def test_omie_gaps_and_informational_flags(self):
        gaps = [
            "omie_missing:2026-01-15: x",
            "omie_parse_error:2026-01-15: x",
            "omie_fetch_failed:2026-01-15: x",
            "omie_part_unreadable:2026-01-15: x",
        ]
        info = [
            "omie_not_published_yet:2026-01-16: x",
            "omie_orphan_part_adopted:2026-01-15: x",
            "omie_price_outside_expected_range:2026-01-15: x",
        ]
        assert coverage_gap_flags(gaps + info) == gaps


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_utc_timestamps_and_non_nullable(self):
        for name in ("interval_start", "interval_end", "observed_at"):
            assert str(RAW_OMIE_DAY_AHEAD.field(name).type) == "timestamp[us, tz=UTC]"
        assert all(not f.nullable for f in RAW_OMIE_DAY_AHEAD)

    def test_part_directory_is_its_own_subtree(self, data_dir):
        assert parts_dir(data_dir, "omie") == data_dir / "raw" / "omie" / "v1" / "parts"


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class TestReader:
    def test_hourly_periods_repeat_over_four_slots(self, data_dir):
        day = date(2025, 9, 30)
        run(data_dir, Net(), day, day)
        prices = load_omie_es_prices(data_dir, start_date=day, end_date=day)
        assert len(prices.slot_start) == 96
        expected = np.repeat([60.0 + i * 0.5 for i in range(1, 25)], 4)
        assert np.array_equal(prices.price_es_eur_mwh, expected)
        assert set(prices.source_period_minutes.tolist()) == {60}
        assert prices.slot_start[0] == local_day_window(day)[0]
        assert all(s.tzinfo is not None for s in prices.slot_start)
        assert prices.first_observed_at == {day: LATE}
        assert prices.missing_dates == () and prices.withheld_dates == ()

    def test_quarter_hours_and_the_2025_10_01_switch(self, data_dir):
        run(data_dir, Net(), date(2025, 9, 30), date(2025, 10, 1))
        prices = load_omie_es_prices(
            data_dir, start_date=date(2025, 9, 30), end_date=date(2025, 10, 1)
        )
        assert len(prices.slot_start) == 192
        assert set(prices.source_period_minutes[:96].tolist()) == {60}
        assert set(prices.source_period_minutes[96:].tolist()) == {15}
        assert prices.price_es_eur_mwh[96:100].tolist() == [60.5, 61.0, 61.5, 62.0]
        assert not np.isnan(prices.price_es_eur_mwh).any()
        assert set(prices.delivery_date[96:]) == {date(2025, 10, 1)}

    @pytest.mark.parametrize(
        ("day", "slots", "periods"),
        [
            (date(2026, 3, 29), 92, 92),
            (date(2026, 10, 25), 100, 100),
            (date(2025, 3, 30), 92, 23),  # hourly file on a spring-forward day
            (date(2025, 10, 26), 100, 25),
        ],
    )
    def test_dst_days_have_92_or_100_slots(self, data_dir, day, slots, periods):
        net = Net({day: (200, omie_file(day, 15 if periods > 30 else 60))})
        run(data_dir, net, day, day)
        prices = load_omie_es_prices(data_dir, start_date=day, end_date=day)
        assert len(prices.slot_start) == slots
        assert not np.isnan(prices.price_es_eur_mwh).any()
        steps = {b - a for a, b in zip(prices.slot_start, prices.slot_start[1:], strict=False)}
        assert steps == {timedelta(minutes=15)}

    def test_missing_days_are_nan_and_listed(self, data_dir):
        run(data_dir, Net(), D, D)
        prices = load_omie_es_prices(data_dir, start_date=D, end_date=D + timedelta(days=2))
        assert prices.missing_dates == (D + timedelta(days=1), D + timedelta(days=2))
        assert not np.isnan(prices.price_es_eur_mwh[:96]).any()
        assert np.isnan(prices.price_es_eur_mwh[96:]).all()
        assert prices.source_period_minutes[96:].tolist() == [0] * 192

    def test_as_of_withholds_days_that_are_not_usable_yet(self, data_dir):
        run(data_dir, Net(), D, D + timedelta(days=1))
        usable = omie_usable_from(D + timedelta(days=1))
        assert usable == datetime(2026, 1, 15, 12, 30, tzinfo=UTC)  # 13:30 local on D
        before = load_omie_es_prices(
            data_dir,
            start_date=D,
            end_date=D + timedelta(days=1),
            as_of=usable - timedelta(minutes=1),
        )
        assert before.withheld_dates == (D + timedelta(days=1),)
        assert np.isnan(before.price_es_eur_mwh[96:]).all()
        assert not np.isnan(before.price_es_eur_mwh[:96]).any()
        after = load_omie_es_prices(
            data_dir, start_date=D, end_date=D + timedelta(days=1), as_of=usable
        )
        assert after.withheld_dates == ()
        assert not np.isnan(after.price_es_eur_mwh).any()

    def test_naive_as_of_is_refused(self, data_dir):
        with pytest.raises(ValueError, match="timezone-aware"):
            load_omie_es_prices(data_dir, start_date=D, end_date=D, as_of=datetime(2026, 1, 1))

    def test_first_observed_is_the_fetch_time_per_day(self, data_dir):
        run(data_dir, Net(), D, D, now=datetime(2026, 1, 14, 12, 40, tzinfo=UTC))
        run(data_dir, Net(), D + timedelta(days=1), D + timedelta(days=1), now=LATE)
        prices = load_omie_es_prices(data_dir, start_date=D, end_date=D + timedelta(days=1))
        assert prices.first_observed_at == {
            D: datetime(2026, 1, 14, 12, 40, tzinfo=UTC),
            D + timedelta(days=1): LATE,
        }

    def test_only_manifest_claimed_parts_are_read(self, data_dir):
        run(data_dir, Net(), D, D)
        for p in (data_dir / "manifests").glob("*.json"):
            p.unlink()
        prices = load_omie_es_prices(data_dir, start_date=D, end_date=D)
        assert prices.missing_dates == (D,)

    def test_altered_part_is_refused(self, data_dir):
        run(data_dir, Net(), D, D)
        (part,) = iter_parts(data_dir, "omie")
        part.write_bytes(part.read_bytes() + b"x")
        with pytest.raises(OmieReadError, match="sha256"):
            load_omie_es_prices(data_dir, start_date=D, end_date=D)

    def test_missing_part_is_refused(self, data_dir):
        run(data_dir, Net(), D, D)
        iter_parts(data_dir, "omie")[0].unlink()
        with pytest.raises(OmieReadError, match="missing"):
            load_omie_es_prices(data_dir, start_date=D, end_date=D)

    def test_empty_store_reads_as_all_missing(self, data_dir):
        prices = load_omie_es_prices(data_dir, start_date=D, end_date=D)
        assert prices.missing_dates == (D,) and np.isnan(prices.price_es_eur_mwh).all()

    def test_negative_prices_pass_through(self, data_dir):
        body = omie_file(D, 60, price=lambda i: (-1.0, -0.01) if i == 5 else (1.0, 2.0))
        run(data_dir, Net({D: (200, body)}), D, D)
        prices = load_omie_es_prices(data_dir, start_date=D, end_date=D)
        assert prices.price_es_eur_mwh[16:20].tolist() == [-0.01] * 4


# ---------------------------------------------------------------------------
# The canonical dataset is never affected
# ---------------------------------------------------------------------------


class TestCanonicalDatasetIsUntouched:
    def test_omie_extraction_does_not_change_the_dataset_id(self, tmp_path, data_dir):
        cfg = write_tariff(tmp_path)
        rows = counter_rows(_T0, 13, times=aligned_times(_T0, 13, 300.0))
        write_raw_part(data_dir, rows)
        window = (_T0, _T0 + timedelta(hours=1))
        before = build(data_dir, cfg, *window, lookback=timedelta(minutes=5))
        manifest_before = (before / "build_manifest.json").read_bytes()

        run(data_dir, Net(), date(2026, 1, 3), date(2026, 1, 7))  # covers the window's days
        assert iter_parts(data_dir, "omie")
        assert any(m["source"] == "omie" for m in manifests(data_dir))

        after = build(data_dir, cfg, *window, lookback=timedelta(minutes=5))
        assert after == before, "an OMIE extraction minted a new dataset id"
        assert (after / "build_manifest.json").read_bytes() == manifest_before
        assert GRID in manifest_before.decode()

    def test_a_malformed_omie_manifest_is_ignored_by_build(self, tmp_path, data_dir):
        cfg = write_tariff(tmp_path)
        write_raw_part(data_dir, counter_rows(_T0, 13, times=aligned_times(_T0, 13, 300.0)))
        window = (_T0, _T0 + timedelta(hours=1))
        before = build(data_dir, cfg, *window, lookback=timedelta(minutes=5))
        (data_dir / "manifests" / "omie-junk.json").write_text(
            json.dumps({"source": "omie", "finished_at": "x", "part_files": [{"path": "../x"}]})
        )
        assert build(data_dir, cfg, *window, lookback=timedelta(minutes=5)) == before

    def test_schema_version_is_unchanged_by_the_new_schema(self):
        # SCHEMA_VERSION is part of the dataset id; registering a raw schema must not bump it.
        assert SCHEMA_VERSION == 1
