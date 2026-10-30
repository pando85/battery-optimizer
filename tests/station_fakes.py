"""Synthetic Ogimet ``getsynop`` responses for the station tests (no network, no real data)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx

STATION = "99999"  # not a real station


def synop_line(station: str, when: datetime, temp_c: float | None, *, extra: str = "") -> str:
    """One CSV line as Ogimet serves it, with the temperature group ``1sTTT`` (or ``1////``)."""
    if temp_c is None:
        group = "1////"
    else:
        tenths = round(abs(temp_c) * 10)
        group = f"1{1 if temp_c < 0 else 0}{tenths:03d}"
    report = (
        f"AAXX {when.day:02d}{when.hour:02d}1 {station} 05/// /2401 {extra}{group} 20083 39394 333 "
        "59049=="
    )
    return f"{station},{when:%Y,%m,%d,%H,%M},{report}"


def synop_csv(station: str, times: list[datetime], temps: list[float | None]) -> str:
    return "\n".join(synop_line(station, t, v) for t, v in zip(times, temps, strict=True)) + "\n"


def hourly(start: datetime, hours: int) -> list[datetime]:
    return [start + timedelta(hours=h) for h in range(hours)]


def ogimet_transport(temp_of, calls: list[dict] | None = None) -> httpx.MockTransport:
    """A mock Ogimet: answers every request with hourly reports of ``temp_of(datetime)``."""

    def handler(request: httpx.Request) -> httpx.Response:
        q = request.url.params
        if calls is not None:
            calls.append(dict(q))
        station = q["block"]
        begin = datetime.strptime(q["begin"], "%Y%m%d%H%M").replace(tzinfo=UTC)
        end = datetime.strptime(q["end"], "%Y%m%d%H%M").replace(tzinfo=UTC)
        times = []
        t = begin
        while t <= end:
            times.append(t)
            t += timedelta(hours=1)
        return httpx.Response(200, text=synop_csv(station, times, [temp_of(t) for t in times]))

    return httpx.MockTransport(handler)
