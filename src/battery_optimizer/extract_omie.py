"""OMIE day-ahead price extraction: the public ``marginalpdbc`` files into the raw store.

OMIE (the Iberian electricity market operator) publishes the day-ahead marginal price of
every delivery day as one small public file, no authentication.  The file for local delivery
day ``D`` is ``marginalpdbc_YYYYMMDD.1``::

    MARGINALPDBC;
    2025;10;25;1;115.2;115.2;
    ...
    *

Data lines are ``YYYY;MM;DD;period;price_PT;price_ES;`` in EUR/MWh (note the order: Portugal
first).  Periods are in local (Europe/Madrid) order starting at local midnight.  Until
2025-09-30 they are hourly (24, or 23/25 on the DST days); from 2025-10-01 they are quarter-hours
(96, or 92/100).  The resolution is inferred from the period count and checked against the
length of the local day, not hard-coded to a date.  The file for D+1 appears at about 13:00
local on D; before that the same URL answers with an HTML page and a 200 status.

Raw store layout (the same conventions as the other sources)::

    raw/omie/v1/parts/omie-day-<day_start_utc>-<day_end_utc>-<sha8>.parquet   one part per day
    manifests/<extraction_id>.json                                           source == "omie"

``<day_start_utc>``/``<day_end_utc>`` are the UTC instants of local midnight of the delivery day
and of the next day.  Each part is ``RAW_OMIE_DAY_AHEAD`` and carries ``observed_at`` (when this
tool fetched the file), ``source_sha256`` (SHA-256 of the raw bytes) and ``source_file_version``
(the file suffix), so the original file is verifiable even though it is stored parsed.

The URL suffix is the publication version of the day.  ``.1`` is the normal file, but for a few
days it answers 404 and only ``.2`` or ``.3`` exists (seen live for 2025-10-30 and 2025-11-27), so
versions ``1..MAX_FILE_VERSION`` are tried in order and the first present one is stored.

Rules this module enforces:

* **Idempotent**: a day with a part already on disk is never fetched or written again, so
  re-runs and overlapping windows do not duplicate.  Parts a crashed run left without a
  manifest are *adopted* into the next run's manifest instead of being fetched again.
* **Strict parsing**: the response is data, never code.  Anything that is not exactly the format
  above (header, line shape, terminator, contiguous periods, a count that fits the local day,
  finite prices) is a parse error and nothing is stored for that day.
* **Not published vs missing**: an HTML or 404 answer before the day's expected publication
  time (14:00 local on D-1) is informational (``omie_not_published_yet``); after it, a gap
  (``omie_missing``, after every version was tried).  Parse and fetch failures are gaps too, so
  ``extract`` exits 3.
* **Polite**: at least :data:`MIN_REQUEST_INTERVAL_SECONDS` between requests, retries with
  exponential backoff on 429/5xx/transport errors, a response size cap, and an abort after
  :data:`MAX_CONSECUTIVE_FETCH_FAILURES` consecutive failed days (the parts already written stay).

``observed_at`` of a backfilled day is the backfill time, not the publication time.  It is
evidence of publication only for days fetched live.

Nothing here touches ``prometheus`` or ``ha_history`` data, and OMIE manifests are never read
by ``build``, so the canonical ``dataset_id`` is unaffected (see ``docs/cli.md``).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from battery_optimizer.config import resolve_config_path
from battery_optimizer.extract_ha import iso_z, table_to_bytes
from battery_optimizer.manifest import new_manifest, write_manifest
from battery_optimizer.paths import iter_parts, part_path, write_part_atomic
from battery_optimizer.paths import resolve_data_dir as _resolve_data_dir
from battery_optimizer.schemas import RAW_OMIE_DAY_AHEAD, validate_table

log = logging.getLogger(__name__)

#: Source name: directory under ``raw/`` and ``source`` of the manifests.
SOURCE = "omie"
#: Part kind in the part file name.
PART_KIND = "omie-day"
#: Public download endpoint.  Kept apart from the query so the manifest sanitiser (which strips
#: the query of any URL-looking value) cannot drop the file name from the recorded provenance.
BASE_URL = "https://www.omie.es/es/file-download"
QUERY_TEMPLATE = "parents=marginalpdbc&filename=marginalpdbc_{yyyymmdd}.{version}"
#: The file suffix is the publication version of the day (``.1`` is the normal one).  Verified
#: against the live archive: for some days only ``.2`` or ``.3`` exists (the earlier versions answer
#: 404), and no day has two versions.  Versions are tried in order and the first present one wins.
MAX_FILE_VERSION = 5
#: OMIE answers some default client agents with an error page.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)
MARKET_TZ = ZoneInfo("Europe/Madrid")
MIN_REQUEST_INTERVAL_SECONDS = 0.5
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_CONSECUTIVE_FETCH_FAILURES = 5
_BACKOFF_BASE_SECONDS = 0.5
_BACKOFF_MAX_SECONDS = 30.0
_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
#: A day's file is expected by this local time on the previous day (published about 13:00).
PUBLICATION_EXPECTED_BY_LOCAL = dtime(14, 0)
#: Harmonised SDAC clearing price limits (EUR/MWh); a price outside is flagged, not rejected.
PRICE_EXPECTED_MIN = -500.0
PRICE_EXPECTED_MAX = 4000.0

#: Quality-flag prefixes.  The gaps (see :mod:`coverage`) make ``extract`` exit 3.
NOT_PUBLISHED_FLAG_PREFIX = "omie_not_published_yet"
MISSING_FLAG_PREFIX = "omie_missing"
PARSE_ERROR_FLAG_PREFIX = "omie_parse_error"
FETCH_FAILED_FLAG_PREFIX = "omie_fetch_failed"
PART_UNREADABLE_FLAG_PREFIX = "omie_part_unreadable"
ORPHAN_ADOPTED_FLAG_PREFIX = "omie_orphan_part_adopted"
PRICE_RANGE_FLAG_PREFIX = "omie_price_outside_expected_range"
OMIE_GAP_FLAG_PREFIXES = frozenset(
    {
        MISSING_FLAG_PREFIX,
        PARSE_ERROR_FLAG_PREFIX,
        FETCH_FAILED_FLAG_PREFIX,
        PART_UNREADABLE_FLAG_PREFIX,
    }
)

_HEADER = "MARGINALPDBC;"
_TERMINATOR = "*"
_NUM = r"-?\d+(?:\.\d+)?"
_LINE_RE = re.compile(rf"(\d{{4}});(\d{{2}});(\d{{2}});(\d{{1,3}});({_NUM});({_NUM});")
_PART_NAME_RE = re.compile(r"omie-day-(\d{8}T\d{6}Z)-(\d{8}T\d{6}Z)-([0-9a-f]{8})\.parquet")


class OmieError(RuntimeError):
    """Base class of every OMIE extraction failure."""


class OmieNotPublishedError(OmieError):
    """The server answered with an HTML page (or 404) instead of a price file."""


class OmieParseError(OmieError):
    """A response that claims to be a price file is not exactly one."""


class OmieFetchError(OmieError):
    """The file could not be downloaded (transport, status or size)."""


# ---------------------------------------------------------------------------
# Day arithmetic
# ---------------------------------------------------------------------------


def local_day_window(delivery_date: date) -> tuple[datetime, datetime]:
    """UTC instants of local midnight of *delivery_date* and of the following day."""
    start = datetime.combine(delivery_date, dtime(0), tzinfo=MARKET_TZ).astimezone(UTC)
    end = datetime.combine(
        delivery_date + timedelta(days=1), dtime(0), tzinfo=MARKET_TZ
    ).astimezone(UTC)
    return start, end


def expected_published_at(delivery_date: date) -> datetime:
    """UTC instant from which the file of *delivery_date* is expected to be public."""
    previous = delivery_date - timedelta(days=1)
    return datetime.combine(previous, PUBLICATION_EXPECTED_BY_LOCAL, tzinfo=MARKET_TZ).astimezone(
        UTC
    )


def date_range(start: date, end: date) -> list[date]:
    """Every date from *start* to *end*, both inclusive."""
    if start > end:
        raise ValueError(f"start date {start} is after end date {end}")
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _parse_date(value: str, name: str) -> date:
    text = (value or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError(
            f"{name}={value!r}: the omie source takes local delivery dates (YYYY-MM-DD), "
            "not timestamps"
        )
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{name}={value!r} is not a valid date: {exc}") from exc


def resolve_omie_dates(
    start: str | None,
    end: str | None,
    lookback_days: int | None,
    *,
    today: date | None = None,
) -> tuple[date, date]:
    """Resolve the delivery-date range of an ``extract --source omie`` run.

    Both bounds are **inclusive** local (Europe/Madrid) delivery dates.  ``--end`` defaults to
    the local date of today; ``--lookback-days N`` is ``[today - N, today]``.  Ask for tomorrow
    explicitly to fetch D+1 once it is published (about 13:00 local).
    """
    if today is None:
        today = datetime.now(UTC).astimezone(MARKET_TZ).date()
    if lookback_days is not None:
        if lookback_days < 1:
            raise ValueError(f"--lookback-days must be >= 1, got {lookback_days}")
        return today - timedelta(days=lookback_days), today
    if start is None:
        raise ValueError("either --start/--end or --lookback-days is required")
    start_date = _parse_date(start, "start")
    end_date = _parse_date(end, "end") if end else today
    if start_date > end_date:
        raise ValueError(f"start {start_date} is after end {end_date}")
    return start_date, end_date


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OmiePeriod:
    """One priced period of a delivery day."""

    period: int
    interval_start: datetime
    interval_end: datetime
    price_es_eur_mwh: float
    price_pt_eur_mwh: float


@dataclass(frozen=True)
class OmieDay:
    """A parsed delivery day: ``period_minutes`` is 60 or 15."""

    delivery_date: date
    period_minutes: int
    periods: tuple[OmiePeriod, ...]


def parse_marginalpdbc(payload: bytes, delivery_date: date) -> OmieDay:
    """Parse the bytes of ``marginalpdbc_<delivery_date>.1`` strictly.

    Raises:
        OmieNotPublishedError: the payload is an HTML page.
        OmieParseError: anything else that is not exactly a price file for *delivery_date*.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OmieParseError(f"not valid UTF-8: {exc}") from exc
    text = text.removeprefix("﻿").strip()
    if not text:
        raise OmieParseError("empty response")
    if text.startswith("<"):
        raise OmieNotPublishedError("HTML response instead of a price file")
    lines = [line.rstrip("\r") for line in text.split("\n")]
    if lines[0] != _HEADER:
        raise OmieParseError(f"unexpected first line {lines[0][:40]!r}")
    if lines[-1] != _TERMINATOR:
        raise OmieParseError("missing '*' terminator (truncated file?)")
    body = lines[1:-1]
    if not body:
        raise OmieParseError("no data lines")

    prices: list[tuple[float, float]] = []
    for index, line in enumerate(body, start=1):
        match = _LINE_RE.fullmatch(line)
        if match is None:
            raise OmieParseError(f"line {index + 1} is not a price line: {line[:60]!r}")
        year, month, day, period, price_pt, price_es = match.groups()
        try:
            line_date = date(int(year), int(month), int(day))
        except ValueError as exc:
            raise OmieParseError(f"line {index + 1}: invalid date: {exc}") from exc
        if line_date != delivery_date:
            raise OmieParseError(f"line {index + 1}: date {line_date} != requested {delivery_date}")
        if int(period) != index:
            raise OmieParseError(f"line {index + 1}: period {period} where {index} is expected")
        pt, es = float(price_pt), float(price_es)
        if not (math.isfinite(pt) and math.isfinite(es)):
            raise OmieParseError(f"line {index + 1}: non-finite price")
        prices.append((pt, es))

    day_start, day_end = local_day_window(delivery_date)
    hours = (day_end - day_start) // timedelta(hours=1)
    count = len(prices)
    if count == hours:
        minutes = 60
    elif count == hours * 4:
        minutes = 15
    else:
        raise OmieParseError(
            f"{count} periods do not fit a {hours}-hour local day "
            f"(expected {hours} hourly or {hours * 4} quarter-hour periods)"
        )
    step = timedelta(minutes=minutes)
    periods = tuple(
        OmiePeriod(
            period=i,
            interval_start=day_start + step * (i - 1),
            interval_end=day_start + step * i,
            price_es_eur_mwh=es,
            price_pt_eur_mwh=pt,
        )
        for i, (pt, es) in enumerate(prices, start=1)
    )
    return OmieDay(delivery_date=delivery_date, period_minutes=minutes, periods=periods)


def day_to_table(
    day: OmieDay,
    *,
    observed_at: datetime,
    source_sha256: str,
    source_file_version: int,
    extraction_id: str,
) -> pa.Table:
    """Build a ``RAW_OMIE_DAY_AHEAD`` table for one parsed day."""
    n = len(day.periods)
    ts = pa.timestamp("us", tz="UTC")
    return pa.table(
        {
            "delivery_date": pa.array([day.delivery_date] * n, pa.date32()),
            "period": pa.array([p.period for p in day.periods], pa.int16()),
            "period_minutes": pa.array([day.period_minutes] * n, pa.int16()),
            "interval_start": pa.array([p.interval_start for p in day.periods], ts),
            "interval_end": pa.array([p.interval_end for p in day.periods], ts),
            "price_es_eur_mwh": pa.array([p.price_es_eur_mwh for p in day.periods], pa.float64()),
            "price_pt_eur_mwh": pa.array([p.price_pt_eur_mwh for p in day.periods], pa.float64()),
            "observed_at": pa.array([observed_at] * n, ts),
            "source_sha256": pa.array([source_sha256] * n, pa.string()),
            "source_file_version": pa.array([source_file_version] * n, pa.int16()),
            "extraction_id": pa.array([extraction_id] * n, pa.string()),
        },
        schema=RAW_OMIE_DAY_AHEAD,
    )


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchResult:
    """One successful HTTP exchange: status 200 or 404, the body and when it arrived."""

    status: int
    body: bytes
    observed_at: datetime
    attempts: int


class _Throttle:
    """Spacing between request starts, with injectable clock and sleep for tests."""

    def __init__(
        self, interval: float, sleep: Callable[[float], None], clock: Callable[[], float]
    ) -> None:
        self._interval = interval
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            remaining = self._last + self._interval - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()


def source_url(delivery_date: date, version: int = 1) -> str:
    """The download URL of one delivery day's file (publication *version*, normally 1)."""
    query = QUERY_TEMPLATE.format(yyyymmdd=delivery_date.strftime("%Y%m%d"), version=version)
    return f"{BASE_URL}?{query}"


def fetch_day(
    client: httpx.Client,
    delivery_date: date,
    *,
    throttle: _Throttle,
    sleep: Callable[[float], None],
    now: Callable[[], datetime],
    max_retries: int,
    version: int = 1,
) -> FetchResult:
    """Download one day's file; retry 429/5xx/transport errors with exponential backoff.

    Returns status 200 with the body, or status 404 (treated like an unpublished day).

    Raises:
        OmieFetchError: retries exhausted, another 4xx, or a body above the size cap.
    """
    url = source_url(delivery_date, version)
    last_error = "no attempt"
    for attempt in range(max_retries + 1):
        if attempt:
            sleep(min(_BACKOFF_BASE_SECONDS * 2 ** (attempt - 1), _BACKOFF_MAX_SECONDS))
        throttle.wait()
        try:
            with client.stream("GET", url) as response:
                status = response.status_code
                if status == 429 or status >= 500:
                    last_error = f"HTTP {status}"
                    continue
                if status == 404:
                    return FetchResult(404, b"", now(), attempt + 1)
                if status != 200:
                    raise OmieFetchError(f"HTTP {status}")
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise OmieFetchError(f"response larger than {MAX_RESPONSE_BYTES} bytes")
                    chunks.append(chunk)
                return FetchResult(200, b"".join(chunks), now(), attempt + 1)
        except httpx.TransportError as exc:
            last_error = type(exc).__name__
    raise OmieFetchError(f"{last_error} after {max_retries + 1} attempt(s)")


def _is_absent(fetched: FetchResult) -> bool:
    """404, or the HTML page the site serves for a file that does not exist (yet)."""
    return fetched.status == 404 or fetched.body.lstrip()[:1] == b"<"


def fetch_day_versions(
    client: httpx.Client,
    delivery_date: date,
    *,
    throttle: _Throttle,
    sleep: Callable[[float], None],
    now: Callable[[], datetime],
    max_retries: int,
) -> tuple[int | None, FetchResult, list[int]]:
    """Fetch the first present publication version of a day.

    Tries ``.1`` to ``.MAX_FILE_VERSION`` while the answer is absent.  Returns
    ``(version, result, statuses)``: ``version`` is ``None`` when every version was absent (then
    *result* is the last answer), ``statuses`` the HTTP status of each version tried.

    Raises:
        OmieFetchError: a version could not be downloaded at all.
    """
    statuses: list[int] = []
    for version in range(1, MAX_FILE_VERSION + 1):
        fetched = fetch_day(
            client,
            delivery_date,
            throttle=throttle,
            sleep=sleep,
            now=now,
            max_retries=max_retries,
            version=version,
        )
        statuses.append(fetched.status)
        if not _is_absent(fetched):
            return version, fetched, statuses
    return None, fetched, statuses


# ---------------------------------------------------------------------------
# Stored days
# ---------------------------------------------------------------------------


def part_delivery_date(path: Path | str) -> date | None:
    """Local delivery date encoded in a part file name, ``None`` for a foreign name."""
    match = _PART_NAME_RE.fullmatch(Path(path).name)
    if match is None:
        return None
    start = datetime.strptime(match.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    return start.astimezone(MARKET_TZ).date()


def index_stored_days(root: Path) -> dict[date, list[Path]]:
    """Parts on disk by delivery date.  Coverage comes from the parts, not from manifests."""
    days: dict[date, list[Path]] = {}
    for path in iter_parts(root, SOURCE):
        day = part_delivery_date(path)
        if day is not None:
            days.setdefault(day, []).append(path)
    return days


def iter_omie_manifests(root: Path) -> list[dict]:
    """Every successful (``finished_at`` set) OMIE manifest, oldest file name first."""
    manifests_dir = root / "manifests"
    out: list[dict] = []
    if not manifests_dir.is_dir():
        return out
    for path in sorted(manifests_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except OSError, ValueError:
            continue
        if (
            isinstance(record, dict)
            and record.get("source") == SOURCE
            and record.get("finished_at") is not None
            and isinstance(record.get("part_files"), list)
        ):
            out.append(record)
    return out


def _claimed_part_names(root: Path) -> set[str]:
    names: set[str] = set()
    for record in iter_omie_manifests(root):
        for entry in record["part_files"]:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                names.add(entry["name"])
    return names


def table_violations(table: pa.Table, path_name: str) -> list[str]:
    """Semantic problems of one stored day beyond the schema (empty list: consistent).

    Checks one delivery date that matches the part's window, periods ``1..N`` in order, a
    uniform 60 or 15 minute resolution that fits the local day, contiguous UTC intervals from
    local midnight to the next local midnight, finite prices and one provenance per file.
    """
    out: list[str] = []
    n = table.num_rows
    if n == 0:
        return [f"{path_name}: zero rows"]
    dates = set(table.column("delivery_date").to_pylist())
    if len(dates) != 1:
        return [f"{path_name}: {len(dates)} delivery dates in one part"]
    (delivery_date,) = dates
    named = part_delivery_date(path_name)
    if named is not None and named != delivery_date:
        out.append(f"{path_name}: file name says {named}, rows say {delivery_date}")
    minutes = set(table.column("period_minutes").to_pylist())
    if minutes not in ({60}, {15}):
        return [*out, f"{path_name}: period_minutes {sorted(minutes)} is not 60 or 15"]
    (step_minutes,) = minutes
    day_start, day_end = local_day_window(delivery_date)
    expected = (day_end - day_start) // timedelta(minutes=step_minutes)
    if n != expected:
        out.append(f"{path_name}: {n} periods, a {step_minutes}-minute local day has {expected}")
    if table.column("period").to_pylist() != list(range(1, n + 1)):
        out.append(f"{path_name}: periods are not 1..{n} in order")
    starts = table.column("interval_start").to_pylist()
    ends = table.column("interval_end").to_pylist()
    step = timedelta(minutes=step_minutes)
    if starts[0] != day_start or ends[-1] != day_end:
        out.append(f"{path_name}: intervals do not span the local day {day_start}..{day_end}")
    if any(e - s != step for s, e in zip(starts, ends, strict=True)) or any(
        starts[i + 1] != ends[i] for i in range(n - 1)
    ):
        out.append(f"{path_name}: intervals are not contiguous {step_minutes}-minute steps")
    for column in ("price_es_eur_mwh", "price_pt_eur_mwh"):
        if any(not math.isfinite(v) for v in table.column(column).to_pylist()):
            out.append(f"{path_name}: non-finite value in {column}")
    for column in ("observed_at", "source_sha256", "source_file_version"):
        if len(set(table.column(column).to_pylist())) != 1:
            out.append(f"{path_name}: {column} differs between rows")
    return out


def validate_omie_store(root: Path) -> list[str]:
    """Violations across every stored OMIE part: duplicates of a day plus per-part checks."""
    out: list[str] = []
    for day, paths in sorted(index_stored_days(root).items()):
        if len(paths) > 1:
            out.append(f"{day}: {len(paths)} parts for one delivery day: {[p.name for p in paths]}")
    for path in iter_parts(root, SOURCE):
        if part_delivery_date(path) is None:
            out.append(f"{path.name}: not an {PART_KIND} part name")
            continue
        try:
            table = pq.read_table(path)
            validate_table("RAW_OMIE_DAY_AHEAD", table)
        except Exception:
            continue  # unreadable/schema problems are reported by the generic part check
        out.extend(table_violations(table, path.name))
    return out


def _summarise_stored_part(root: Path, path: Path) -> dict[str, object]:
    """Manifest entry for a part found on disk (read, schema- and semantics-checked)."""
    payload = path.read_bytes()
    table = pq.read_table(pa.BufferReader(payload))
    validate_table("RAW_OMIE_DAY_AHEAD", table)
    problems = table_violations(table, path.name)
    if problems:
        raise ValueError("; ".join(problems))
    day_start, day_end = local_day_window(table.column("delivery_date")[0].as_py())
    return {
        "path": str(path.relative_to(root)),
        "name": path.name,
        "kind": PART_KIND,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": table.num_rows,
        "chunk_start": iso_z(day_start),
        "chunk_end": iso_z(day_end),
        "delivery_date": table.column("delivery_date")[0].as_py().isoformat(),
        "observed_at": iso_z(table.column("observed_at")[0].as_py()),
        "raw_sha256": table.column("source_sha256")[0].as_py(),
        "file_version": table.column("source_file_version")[0].as_py(),
        "period_minutes": table.column("period_minutes")[0].as_py(),
        "periods": table.num_rows,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_omie(
    *,
    data_dir: str | None,
    config_path: str | None,
    start: date,
    end: date,
    client: httpx.Client | None = None,
    max_retries: int = 3,
    min_interval_seconds: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    max_consecutive_failures: int = MAX_CONSECUTIVE_FETCH_FAILURES,
) -> Path:
    """Extract OMIE day-ahead prices for the local delivery dates ``start..end`` (inclusive).

    One immutable ``omie-day`` part per delivery day under ``raw/omie/v1/parts/``, plus one
    manifest for the run.  Days already stored are skipped without a request.  See the module
    docstring for the parse, publication and gap rules.

    Args:
        data_dir: Data directory root; ``None`` resolves env/default.
        config_path: Config used for the manifest's ``config_hash`` (like every source).
        start, end: Local delivery dates, both inclusive.
        client: Pre-configured :class:`httpx.Client` for test injection.
        max_retries: Additional attempts after the first for 429/5xx/transport errors.
        min_interval_seconds: Minimum time between request starts (default
            :data:`MIN_REQUEST_INTERVAL_SECONDS`).
        sleep, clock, now: Injection points for tests.
        max_consecutive_failures: Abort after this many consecutive days that could not be
            downloaded (no manifest; parts already written are adopted by the next run).

    Returns:
        The path of the written manifest.

    Raises:
        ValueError: bad date range or config.
        OmieError: too many consecutive download failures.
    """
    root = _resolve_data_dir(None if data_dir is None else str(data_dir))
    cfg = resolve_config_path(None if config_path is None else str(config_path))
    if not cfg.is_file():
        raise FileNotFoundError(f"tariff config not found: {cfg}")
    if min_interval_seconds is None:
        min_interval_seconds = MIN_REQUEST_INTERVAL_SECONDS
    days = date_range(start, end)
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries!r}")

    window_start, _ = local_day_window(start)
    _, window_end = local_day_window(end)
    manifest = new_manifest(
        source=SOURCE,
        config_path=cfg,
        params={
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "interval_semantics": "inclusive local (Europe/Madrid) delivery dates",
            "endpoint": BASE_URL,
            "query_template": QUERY_TEMPLATE,
            "file": "marginalpdbc: YYYY;MM;DD;period;price_PT;price_ES; (EUR/MWh)",
            "min_request_interval_seconds": min_interval_seconds,
            "max_retries": max_retries,
            "publication_expected_by": (
                f"{PUBLICATION_EXPECTED_BY_LOCAL:%H:%M} Europe/Madrid on the previous day"
            ),
            "observed_at_semantics": "time this tool fetched the file, not the publication time",
        },
        entities=["marginalpdbc"],
        source_window_start=iso_z(window_start),
        source_window_end=iso_z(window_end),
        started_at=iso_z(now()),
    )
    extraction_id = manifest.extraction_id

    stored = index_stored_days(root)
    claimed = _claimed_part_names(root)
    flags: list[str] = []
    part_files: list[dict[str, object]] = []
    ledger: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    period_counts: Counter[int] = Counter()
    requests = 0
    consecutive_failures = 0

    own_client = client is None
    http_client = (
        client
        if client is not None
        else httpx.Client(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "text/plain,*/*;q=0.5"},
        )
    )
    throttle = _Throttle(min_interval_seconds, sleep, clock)
    try:
        for day in days:
            existing = stored.get(day)
            if existing:
                counts["already_stored"] += 1
                path = existing[0]
                if path.name in claimed:
                    continue
                try:
                    entry = _summarise_stored_part(root, path)
                except Exception as exc:
                    flags.append(
                        f"{PART_UNREADABLE_FLAG_PREFIX}:{day}: {path.name}: "
                        f"{type(exc).__name__}: {str(exc)[:200]}"
                    )
                    continue
                part_files.append({**entry, "adopted": True})
                counts["orphan_adopted"] += 1
                flags.append(
                    f"{ORPHAN_ADOPTED_FLAG_PREFIX}:{day}: {path.name} was on disk without a "
                    "manifest (a crashed run); this manifest now claims it"
                )
                continue

            record: dict[str, object] = {"delivery_date": day.isoformat()}
            try:
                version, fetched, statuses = fetch_day_versions(
                    http_client,
                    day,
                    throttle=throttle,
                    sleep=sleep,
                    now=now,
                    max_retries=max_retries,
                )
            except OmieFetchError as exc:
                consecutive_failures += 1
                counts["fetch_failed"] += 1
                flags.append(f"{FETCH_FAILED_FLAG_PREFIX}:{day}: {exc}")
                ledger.append({**record, "status": "fetch_failed", "error": str(exc)})
                log.warning("omie %s: fetch failed: %s", day, exc)
                if consecutive_failures >= max_consecutive_failures:
                    done = counts["stored"] + counts["orphan_adopted"] + counts["already_stored"]
                    raise OmieError(
                        f"{consecutive_failures} consecutive days could not be downloaded "
                        f"(last: {day}: {exc}); aborting with no manifest. {done} day(s) are "
                        "stored and a re-run skips them"
                    ) from exc
                continue
            consecutive_failures = 0
            requests += len(statuses)
            record.update(
                file_version=version,
                http_statuses=statuses,
                http_status=fetched.status,
                observed_at=iso_z(fetched.observed_at),
                attempts=fetched.attempts,
                raw_sha256=hashlib.sha256(fetched.body).hexdigest(),
                raw_bytes=len(fetched.body),
            )

            try:
                if version is None:
                    tried = "/".join(map(str, statuses))
                    raise OmieNotPublishedError(
                        f"no file (HTTP {tried} for versions 1-{len(statuses)})"
                    )
                parsed = parse_marginalpdbc(fetched.body, day)
            except OmieNotPublishedError as exc:
                if fetched.observed_at >= expected_published_at(day):
                    counts["missing"] += 1
                    flags.append(f"{MISSING_FLAG_PREFIX}:{day}: {exc}; expected since 14:00 local")
                    record["status"] = "missing"
                else:
                    counts["not_published_yet"] += 1
                    flags.append(f"{NOT_PUBLISHED_FLAG_PREFIX}:{day}: {exc}")
                    record["status"] = "not_published_yet"
                ledger.append(record)
                log.info("omie %s: %s", day, record["status"])
                continue
            except OmieParseError as exc:
                counts["parse_error"] += 1
                flags.append(f"{PARSE_ERROR_FLAG_PREFIX}:{day}: {exc}")
                ledger.append({**record, "status": "parse_error", "error": str(exc)})
                log.warning("omie %s: parse error: %s", day, exc)
                continue

            table = day_to_table(
                parsed,
                observed_at=fetched.observed_at,
                source_sha256=str(record["raw_sha256"]),
                source_file_version=int(version),
                extraction_id=extraction_id,
            )
            validate_table("RAW_OMIE_DAY_AHEAD", table)
            payload = table_to_bytes(table)
            content_sha = hashlib.sha256(payload).hexdigest()
            day_start, day_end = local_day_window(day)
            path = part_path(root, SOURCE, PART_KIND, day_start, day_end, content_sha[:8])
            write_part_atomic(path, payload)
            counts["stored"] += 1
            period_counts[len(parsed.periods)] += 1
            out_of_range = sum(
                1
                for p in parsed.periods
                for v in (p.price_es_eur_mwh, p.price_pt_eur_mwh)
                if not PRICE_EXPECTED_MIN <= v <= PRICE_EXPECTED_MAX
            )
            if out_of_range:
                flags.append(
                    f"{PRICE_RANGE_FLAG_PREFIX}:{day}: {out_of_range} price(s) outside "
                    f"[{PRICE_EXPECTED_MIN:g}, {PRICE_EXPECTED_MAX:g}] EUR/MWh"
                )
            record.update(
                status="stored",
                periods=len(parsed.periods),
                period_minutes=parsed.period_minutes,
                part=path.name,
            )
            ledger.append(record)
            part_files.append(
                {
                    "path": str(path.relative_to(root)),
                    "name": path.name,
                    "kind": PART_KIND,
                    "sha256": content_sha,
                    "rows": table.num_rows,
                    "chunk_start": iso_z(day_start),
                    "chunk_end": iso_z(day_end),
                    "delivery_date": day.isoformat(),
                    "observed_at": iso_z(fetched.observed_at),
                    "raw_sha256": record["raw_sha256"],
                    "file_version": version,
                    "http_status": fetched.status,
                    "period_minutes": parsed.period_minutes,
                    "periods": len(parsed.periods),
                    "adopted": False,
                }
            )
            log.info(
                "omie %s: stored %d x %d-minute period(s)",
                day,
                len(parsed.periods),
                parsed.period_minutes,
            )
    finally:
        if own_client:
            http_client.close()

    manifest.finished_at = iso_z(now())
    manifest.part_files = part_files
    manifest.row_count = sum(int(p["rows"]) for p in part_files)  # type: ignore[arg-type]
    manifest.quality_flags = sorted(set(flags))
    manifest.params["days_requested"] = len(days)
    manifest.params["requests_made"] = requests
    manifest.params["day_counts"] = dict(sorted(counts.items()))
    manifest.params["period_counts_stored"] = {str(k): v for k, v in sorted(period_counts.items())}
    manifest.params["days"] = ledger
    manifest.coverage_notes = (
        f"{len(days)} local delivery date(s) {start}..{end} inclusive; "
        f"{counts['stored']} fetched and stored, {counts['already_stored']} already stored "
        f"(skipped, {counts['orphan_adopted']} of them adopted from a crashed run), "
        f"{counts['not_published_yet']} not published yet, {counts['missing']} missing, "
        f"{counts['parse_error']} parse error(s), {counts['fetch_failed']} fetch failure(s). "
        "One part per day; observed_at is the fetch time, not the publication time."
    )
    path = write_manifest(root, manifest)
    log.info(
        "wrote manifest %s (extraction_id=%s, requests=%d, stored=%d)",
        path,
        extraction_id,
        requests,
        counts["stored"],
    )
    return path
