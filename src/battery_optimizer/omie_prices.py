"""Read side of the OMIE day-ahead store: ES prices on the 15-minute UTC grid.

:func:`load_omie_es_prices` is the entry point of the price estimator.  Like ``build`` it reads
through the successful ``omie`` manifests (never by globbing the parts directory), verifies each
referenced part against the full SHA-256 and the schema, and refuses a part that is missing,
altered or inconsistent.  It never writes anything and never touches ``prometheus`` or
``ha_history`` data.

Grid rules:

* Slots are the 15-minute UTC slots from local midnight of the first requested delivery date to
  local midnight after the last one, so a DST day naturally has 92 or 100 slots.
* A quarter-hour period fills its slot.  An hourly period (files up to 2025-09-30) is **repeated
  over its four slots**, and ``source_period_minutes`` records which resolution each slot came from.
* A slot with no stored price is ``NaN`` (never filled or interpolated); ``missing_dates`` names the
  delivery dates of the request that have no stored day.

As-of rule: an OMIE D+1 value may be used only from **13:30 local on D** (the price outlook plan's
conservative rule, see :func:`omie_usable_from`).  Pass ``as_of`` to blank the slots of every
delivery day that is not yet usable at that instant.  ``first_observed_at`` is the earliest time
this tool fetched the day: evidence of publication only for days fetched live, since a backfilled
day is stamped with the backfill time.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from battery_optimizer.extract_omie import (
    MARKET_TZ,
    PART_KIND,
    date_range,
    iter_omie_manifests,
    local_day_window,
    table_violations,
)
from battery_optimizer.paths import resolve_data_dir
from battery_optimizer.schemas import validate_table

#: Local time on the previous day from which an OMIE value may be used (published about 13:00).
AS_OF_LOCAL_TIME = dtime(13, 30)
SLOT = timedelta(minutes=15)


class OmieReadError(RuntimeError):
    """A referenced OMIE part is missing, altered, inconsistent, or two parts disagree."""


def omie_usable_from(delivery_date: date) -> datetime:
    """UTC instant from which the OMIE price of *delivery_date* may be used (D-1 13:30 local)."""
    previous = delivery_date - timedelta(days=1)
    return datetime.combine(previous, AS_OF_LOCAL_TIME, tzinfo=MARKET_TZ).astimezone(UTC)


@dataclass(frozen=True)
class OmieEsPrices:
    """ES day-ahead prices on the 15-minute UTC grid.

    All per-slot sequences have one entry per slot and the same order.

    Attributes:
        slot_start: UTC start of every 15-minute slot (timezone-aware).
        price_es_eur_mwh: ES marginal price in EUR/MWh; ``NaN`` where nothing is stored or the
            day is not yet usable at ``as_of``.
        source_period_minutes: 60 or 15 for the file the slot came from, 0 where ``NaN``.
        delivery_date: Local (Europe/Madrid) delivery date of every slot.
        first_observed_at: Earliest fetch time per stored delivery date.
        missing_dates: Requested delivery dates with no stored day.
        withheld_dates: Stored dates blanked because ``as_of`` is before :func:`omie_usable_from`.
    """

    slot_start: tuple[datetime, ...]
    price_es_eur_mwh: np.ndarray
    source_period_minutes: np.ndarray
    delivery_date: tuple[date, ...]
    first_observed_at: dict[date, datetime]
    missing_dates: tuple[date, ...]
    withheld_dates: tuple[date, ...]


def _read_manifest_part(root: Path, entry: dict) -> pa.Table:
    rel = entry.get("path")
    declared = entry.get("sha256")
    if not isinstance(rel, str) or not isinstance(declared, str):
        raise OmieReadError(f"malformed omie manifest part entry: {entry!r}")
    resolved = (root / rel).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise OmieReadError(f"omie part path escapes the data directory: {rel}")
    if not resolved.is_file():
        raise OmieReadError(f"omie part missing: {rel}")
    payload = resolved.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != declared:
        raise OmieReadError(
            f"omie part {resolved.name}: sha256 {actual[:8]} != manifest {declared[:8]}"
        )
    table = pq.read_table(io.BytesIO(payload))
    try:
        validate_table("RAW_OMIE_DAY_AHEAD", table)
    except (TypeError, ValueError) as exc:
        raise OmieReadError(f"omie part {resolved.name}: {exc}") from exc
    problems = table_violations(table, resolved.name)
    if problems:
        raise OmieReadError("; ".join(problems))
    return table


def load_omie_es_prices(
    data_dir: str | Path | None,
    *,
    start_date: date,
    end_date: date,
    as_of: datetime | None = None,
) -> OmieEsPrices:
    """ES prices for the local delivery dates ``start_date..end_date`` (inclusive) on the UTC grid.

    Args:
        data_dir: Data directory root; ``None`` resolves env/default.
        start_date, end_date: Local (Europe/Madrid) delivery dates, both inclusive.
        as_of: When given (timezone-aware), the slots of every delivery day whose
            :func:`omie_usable_from` is after it are ``NaN`` and listed in ``withheld_dates``.

    Raises:
        OmieReadError: a referenced part is missing, altered or inconsistent, or two stored
            parts of one day disagree on a price.
        ValueError: bad range, or a naive ``as_of``.
    """
    if as_of is not None and as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    root = resolve_data_dir(None if data_dir is None else str(data_dir))
    wanted = date_range(start_date, end_date)
    wanted_set = set(wanted)
    window_start, _ = local_day_window(start_date)
    _, window_end = local_day_window(end_date)

    tables: dict[date, pa.Table] = {}
    first_observed: dict[date, datetime] = {}
    seen_parts: set[tuple[str, str]] = set()
    for record in iter_omie_manifests(root):
        for entry in record["part_files"]:
            if not isinstance(entry, dict) or entry.get("kind") != PART_KIND:
                continue
            delivery = entry.get("delivery_date")
            try:
                day = date.fromisoformat(str(delivery))
            except ValueError:
                raise OmieReadError(f"manifest part entry without a delivery_date: {entry!r}")
            if day not in wanted_set:
                continue
            key = (str(entry.get("name")), str(entry.get("sha256")))
            if key in seen_parts:
                continue
            seen_parts.add(key)
            table = _read_manifest_part(root, entry)
            observed = table.column("observed_at")[0].as_py()
            if day in tables:
                same = (
                    table.column("price_es_eur_mwh").to_pylist()
                    == tables[day].column("price_es_eur_mwh").to_pylist()
                    and table.column("period_minutes").to_pylist()
                    == tables[day].column("period_minutes").to_pylist()
                )
                if not same:
                    raise OmieReadError(f"two stored parts of {day} disagree on the ES prices")
                first_observed[day] = min(first_observed[day], observed)
                continue
            tables[day] = table
            first_observed[day] = observed

    n_slots = (window_end - window_start) // SLOT
    slot_start = tuple(window_start + SLOT * i for i in range(n_slots))
    index_of = {s: i for i, s in enumerate(slot_start)}
    prices = np.full(n_slots, np.nan, dtype=np.float64)
    minutes = np.zeros(n_slots, dtype=np.int16)
    local_dates = tuple(s.astimezone(MARKET_TZ).date() for s in slot_start)

    withheld: list[date] = []
    for day, table in sorted(tables.items()):
        if as_of is not None and as_of < omie_usable_from(day):
            withheld.append(day)
            continue
        step = int(table.column("period_minutes")[0].as_py())
        repeats = step // 15
        for start, price in zip(
            table.column("interval_start").to_pylist(),
            table.column("price_es_eur_mwh").to_pylist(),
            strict=True,
        ):
            for k in range(repeats):
                i = index_of[start + SLOT * k]
                prices[i] = price
                minutes[i] = step

    missing = tuple(d for d in wanted if d not in tables)
    return OmieEsPrices(
        slot_start=slot_start,
        price_es_eur_mwh=prices,
        source_period_minutes=minutes,
        delivery_date=local_dates,
        first_observed_at=dict(sorted(first_observed.items())),
        missing_dates=missing,
        withheld_dates=tuple(withheld),
    )


__all__ = [
    "AS_OF_LOCAL_TIME",
    "OmieEsPrices",
    "OmieReadError",
    "load_omie_es_prices",
    "omie_usable_from",
]
