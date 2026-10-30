"""Independent outdoor-temperature reference: AEMET synoptic stations through Ogimet SYNOP.

AEMET OpenData has no free historical hourly data, but Ogimet serves the hourly SYNOP reports of
the AEMET stations without a key.  Source credit: "Fuente: AEMET (via Ogimet)".

* :func:`decode_synop_temperature` decodes the air temperature (group ``1sTTT`` of section 1) of
  one SYNOP report; :func:`parse_getsynop` turns a ``getsynop`` CSV response into a time series.
* :func:`fetch_station` downloads one month per request (at least ``min_interval_s`` apart, retry
  with backoff on a server error or timeout) and stores each month once and immutably under
  ``<data_dir>/climate/station/<WMO>/raw/``: the raw response plus a manifest (source, station,
  request range, ``fetched_at``, raw sha256, parser version).
* :func:`load_station` verifies the stored months against their manifests and returns the decoded
  series together with a manifest hash that changes whenever an input (a raw month, the parser)
  changes, so results that record it change id with the inputs.
* :func:`reference_at_slots` interpolates the series in time to slot midpoints, bridging station
  gaps of at most a configured length and leaving longer ones ``NaN``.

The chosen station reveals the site's area: its WMO id lives only in the private site file
(``reference_station``) and in the data dir, never in the repository.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import numpy as np

from battery_optimizer.paths import write_part_atomic

__all__ = [
    "PARSER_VERSION",
    "SOURCE",
    "StationError",
    "StationSeries",
    "decode_synop_temperature",
    "fetch_station",
    "load_station",
    "month_range",
    "parse_getsynop",
    "reference_at_slots",
    "station_dir",
]

PARSER_VERSION = "synop-1"
SOURCE = "ogimet-getsynop"
ATTRIBUTION = "Fuente: AEMET (via Ogimet)"
URL = "https://www.ogimet.com/cgi-bin/getsynop"
USER_AGENT = "battery-optimizer/climate-reference (github.com/pando85/home-energy-optimizer)"
STATION_DIR = Path("climate") / "station"

_STATION = re.compile(r"\d{5}")
_CSV_LINE = re.compile(r"^(\d{5}),(\d{4}),(\d\d),(\d\d),(\d\d),(\d\d),(.*)$")
_GROUP = re.compile(r"^[0-9/]{5}$")


class StationError(RuntimeError):
    """Expected, user-facing failure of the station fetcher or loader."""


# ---------------------------------------------------------------------------
# SYNOP decoding
# ---------------------------------------------------------------------------


def decode_synop_temperature(report: str, station: str | None = None) -> float | None:
    """Air temperature in degrees Celsius of one SYNOP (``AAXX``) report, or ``None``.

    Section 1 starts after ``YYGGi IIiii`` with ``iRiXhVV`` and ``Nddff`` (followed by ``00fff``
    when the wind speed ``ff`` is 99), and then comes group 1, ``1sTTT``: ``s`` 0 is positive, 1
    negative, ``TTT`` tenths of a degree.  The groups are read by position, never by searching
    for a token that starts with 1.  A ``NIL`` report, a report of another station, a missing
    group (``1////``) or anything malformed gives ``None``.
    """
    tokens = report.replace("=", " ").split()
    if len(tokens) < 3 or tokens[0] != "AAXX" or not _STATION.fullmatch(tokens[2]):
        return None
    if station is not None and tokens[2] != station:
        return None
    body = tokens[3:]
    if not body or body[0] == "NIL" or len(body) < 3:
        return None
    # body[0] = iRiXhVV, body[1] = Nddff, then 00fff when ff == 99, then group 1
    if not (_GROUP.match(body[0]) and _GROUP.match(body[1])):
        return None
    wind_extra = 1 if body[1][3:] == "99" else 0
    index = 2 + wind_extra
    if len(body) <= index:
        return None
    group = body[index]
    if wind_extra and not (_GROUP.match(body[2]) and body[2].startswith("00")):
        return None
    if len(group) != 5 or group[0] != "1" or group[1] not in "01" or not group[2:].isdigit():
        return None
    tenths = int(group[2:])
    return -tenths / 10.0 if group[1] == "1" else tenths / 10.0


def parse_getsynop(text: str, station: str) -> tuple[np.ndarray, np.ndarray]:
    """Decode a ``getsynop`` CSV response to ``(times_us, temp_c)`` sorted by time (UTC).

    One entry per distinct report time that holds a temperature; ``NIL`` and temperature-less
    reports are skipped, a repeated report time keeps its first.  A non-blank line that is not
    a CSV report line (an error or quota message) raises :class:`StationError`.
    """
    seen: dict[int, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _CSV_LINE.match(line)
        if m is None:
            raise StationError(f"station {station}: unexpected response line {line[:60]!r}")
        if m.group(1) != station:
            raise StationError(f"station {station}: the response holds station {m.group(1)}")
        t = decode_synop_temperature(m.group(7), station)
        if t is None:
            continue
        when = datetime(*(int(m.group(i)) for i in range(2, 7)), tzinfo=UTC)
        seen.setdefault(int(when.timestamp()) * 1_000_000, t)
    times = np.array(sorted(seen), dtype=np.int64)
    return times, np.array([seen[int(k)] for k in times], dtype=np.float64)


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def station_dir(data_dir: str | Path, station: str) -> Path:
    if not _STATION.fullmatch(station):
        raise StationError(f"{station!r} is not a 5-digit WMO station id")
    return Path(data_dir).expanduser() / STATION_DIR / station


def month_range(first: date, last: date) -> list[tuple[int, int]]:
    """``(year, month)`` of every month from the one holding *first* to the one holding *last*."""
    out: list[tuple[int, int]] = []
    y, m = first.year, first.month
    while (y, m) <= (last.year, last.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def _month_window(year: int, month: int) -> tuple[str, str]:
    ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
    last_day = (date(ny, nm, 1) - date(year, month, 1)).days
    return f"{year:04d}{month:02d}010000", f"{year:04d}{month:02d}{last_day:02d}2359"


def _month_name(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def _fetch_month(
    client: httpx.Client,
    station: str,
    year: int,
    month: int,
    *,
    retries: int,
    backoff_s: float,
    sleep: Callable[[float], None],
) -> bytes:
    begin, end = _month_window(year, month)
    params = {"block": station, "begin": begin, "end": end}
    for attempt in range(retries + 1):
        try:
            response = client.get(URL, params=params)
        except httpx.TransportError as exc:  # timeouts and connection failures
            error = f"{type(exc).__name__}"
        else:
            if response.status_code < 500:
                if response.status_code != 200:
                    raise StationError(f"ogimet answered HTTP {response.status_code}")
                return response.content
            error = f"HTTP {response.status_code}"
        if attempt == retries:
            raise StationError(f"{station} {_month_name(year, month)}: giving up after {error}")
        sleep(backoff_s * 3**attempt)
    raise AssertionError("unreachable")  # pragma: no cover


def fetch_station(
    data_dir: str | Path,
    station: str,
    months: Iterable[tuple[int, int]],
    *,
    client: httpx.Client | None = None,
    min_interval_s: float = 5.0,
    retries: int = 3,
    backoff_s: float = 30.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    log: Callable[[str], None] = lambda _msg: None,
) -> list[str]:
    """Fetch the missing *months* of *station* into the store, once each; returns those fetched.

    A month already stored is never requested again.  Requests are at least *min_interval_s*
    apart; a 5xx answer or a timeout is retried with exponential backoff.  A response that is
    not a clean ``getsynop`` CSV is refused and nothing is stored for the month.
    """
    root = station_dir(data_dir, station)
    raw_dir = root / "raw"
    own = client is None
    http = client or httpx.Client(
        timeout=httpx.Timeout(90.0, connect=15.0), headers={"User-Agent": USER_AGENT}
    )
    fetched: list[str] = []
    last_request: float | None = None
    try:
        for year, month in months:
            name = _month_name(year, month)
            if (raw_dir / f"{name}.manifest.json").exists():
                continue
            if last_request is not None:
                wait = min_interval_s - (clock() - last_request)
                if wait > 0:
                    sleep(wait)
            last_request = clock()
            payload = _fetch_month(
                http, station, year, month, retries=retries, backoff_s=backoff_s, sleep=sleep
            )
            last_request = clock()
            text = payload.decode("utf-8", errors="replace")
            times, _temp = parse_getsynop(text, station)  # refuses a malformed response
            begin, end = _month_window(year, month)
            manifest = {
                "source": SOURCE,
                "attribution": ATTRIBUTION,
                "station": station,
                "month": name,
                "request": {"begin": begin, "end": end},
                "fetched_at": now().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "reports_with_temperature": int(times.size),
                "parser_version": PARSER_VERSION,
            }
            raw_dir.mkdir(parents=True, exist_ok=True)
            write_part_atomic(raw_dir / f"{name}.csv", payload)
            write_part_atomic(
                raw_dir / f"{name}.manifest.json",
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
            )
            fetched.append(name)
            log(f"{name}: {times.size} reports with a temperature")
    finally:
        if own:
            http.close()
    return fetched


@dataclass(frozen=True)
class StationSeries:
    """Decoded station temperatures (UTC, sorted) and the hash of the inputs they came from."""

    station: str
    times_us: np.ndarray
    temp_c: np.ndarray
    months: tuple[str, ...]
    #: Hash of the station id, the parser version and every month's request range and raw sha256.
    manifest_sha256: str


def load_station(
    data_dir: str | Path, station: str, months: Iterable[tuple[int, int]]
) -> StationSeries:
    """Read and verify the stored *months*; a missing or altered month is an error."""
    raw_dir = station_dir(data_dir, station) / "raw"
    entries: list[dict[str, str]] = []
    times: list[np.ndarray] = []
    temps: list[np.ndarray] = []
    names: list[str] = []
    for year, month in months:
        name = _month_name(year, month)
        try:
            manifest = json.loads((raw_dir / f"{name}.manifest.json").read_text())
            payload = (raw_dir / f"{name}.csv").read_bytes()
        except (OSError, ValueError) as exc:
            raise StationError(
                f"station store lacks the month {name}: fetch it first ({type(exc).__name__})"
            ) from exc
        digest = hashlib.sha256(payload).hexdigest()
        if manifest.get("sha256") != digest or manifest.get("station") != station:
            raise StationError(f"station store: {name} does not match its manifest sha256")
        t, v = parse_getsynop(payload.decode("utf-8", errors="replace"), station)
        times.append(t)
        temps.append(v)
        names.append(name)
        entries.append(
            {
                "month": name,
                "begin": manifest["request"]["begin"],
                "end": manifest["request"]["end"],
                "sha256": digest,
            }
        )
    if not names:
        raise StationError("station store: no months requested")
    blob = json.dumps(
        {"station": station, "parser_version": PARSER_VERSION, "months": entries},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    all_t = np.concatenate(times)
    all_v = np.concatenate(temps)
    order = np.argsort(all_t, kind="stable")
    all_t, all_v = all_t[order], all_v[order]
    keep = np.concatenate(([True], np.diff(all_t) > 0)) if all_t.size else np.zeros(0, bool)
    return StationSeries(
        station=station,
        times_us=all_t[keep],
        temp_c=all_v[keep],
        months=tuple(names),
        manifest_sha256=hashlib.sha256(blob).hexdigest(),
    )


# ---------------------------------------------------------------------------
# Reference at slot midpoints
# ---------------------------------------------------------------------------


def reference_at_slots(
    times_us: np.ndarray, temp_c: np.ndarray, slot_mid_us: np.ndarray, max_gap_hours: float
) -> np.ndarray:
    """Station temperature linearly interpolated in time to *slot_mid_us*.

    A slot between two consecutive reports at most *max_gap_hours* apart is interpolated (hourly
    reports with up to two missing ones in a row, or a 3-hourly station, for a gap of 3 h); a
    slot in a longer gap, before the first or after the last report, is ``NaN``.
    """
    t = np.asarray(times_us, dtype=np.int64)
    v = np.asarray(temp_c, dtype=np.float64)
    mid = np.asarray(slot_mid_us, dtype=np.int64)
    out = np.full(mid.size, np.nan)
    if t.size == 0:
        return out
    nxt = np.searchsorted(t, mid, side="left")
    exact = (nxt < t.size) & (t[np.minimum(nxt, t.size - 1)] == mid)
    inside = (nxt > 0) & (nxt < t.size)
    lo = np.clip(nxt - 1, 0, t.size - 1)
    hi = np.clip(nxt, 0, t.size - 1)
    span = (t[hi] - t[lo]) / 3.6e9
    ok = inside & (span <= max_gap_hours + 1e-9)
    out[ok] = np.interp(mid[ok], t, v)
    out[exact] = v[nxt[exact]]
    return out
