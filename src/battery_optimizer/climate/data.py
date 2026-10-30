"""Climate feature set and dataset prices as numpy arrays, with the signals C2 derives from them.

Input: the climate feature table (``build-features --signal-set climate``: a uniform 15-minute UTC
grid, every column the mean or integral over its slot, ``NaN`` where the slot is unknown) and,
optionally, a canonical dataset's PVPC prices.  Output: :class:`ClimateData`, one array per signal
on the feature grid plus a local-day partition.  Nothing is written and nothing is zero-filled; the
only interpolation is the outdoor-temperature de-biasing below.

Derived signals
---------------
* ``t_air``: mean temperature of the comfort rooms (``NaN`` unless all of them are known, so the
  composition never jumps).  :func:`room_offsets` gives each room's median offset from it by local
  hour; it is a function, not a field, so a fold fits it on its own training mask and no held-out
  or reserved slot can leak into it.
* ``q_heat_kwh``: the power-integrated thermal heating output of a slot; :meth:`ClimateData.
  counter_check` compares its daily sum with the heat-produced energy counter.
* ``dhw`` and ``cooling``: slots in which the pump produced hot water or cooling.
* ``t_out`` and ``t_out_debiased``: the Aquarea outdoor sensor reads 1.5 to 2.5 K low while the
  compressor runs and recovers after it stops, so ``t_out_debiased`` (the model's T_out input) keeps
  only slots in which the compressor has been off at least ``signals.t_out_debias.min_off_minutes``
  and interpolates linearly in time across running periods up to ``max_gap_hours``; longer
  periods are ``NaN``.  The raw ``t_out`` stays for diagnostics.
* ``valve_open_inferred``: from room temperature and setpoint (a room below its setpoint wants
  heat); ``climate_action`` has no history before 2026-10-30.  Not the plant's valve cap.
* ``solar``: clear-sky proxy, the positive part of the sine of the solar elevation at the slot
  midpoint (:func:`solar_proxy`); the site coordinates come from a private YAML in the data dir.
* ``u``: the legacy ``climate_control`` permission replayed per local hour through
  :func:`battery_optimizer.structured_forecast.replay_mask_for_day`, decided at the start of each
  local day.  ``u`` is ``NaN`` on every local day without a replayed schedule: ``partial_day``,
  ``rule_disabled``, ``enable_unknown``, ``min_hours_unknown``, ``rule_not_deployed``,
  ``prices_incomplete``, ``price_history_short``, ``rule_error_negative_price`` (see
  :attr:`ClimateData.u_reason`).  ``u_known`` marks the slots that have one; u-driven code must
  use only those and never treat ``NaN`` as "not permitted".
* ``drop``: one boolean array per training filter of ``config/climate.yaml``; ``train`` is the
  slots no filter drops.  Season and month membership is per local day and left to the caller.

The feature table is verified through its manifest by :func:`load_climate_data`.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import yaml

from battery_optimizer.climate.config import ClimateConfig
from battery_optimizer.features import FEATURES_DIRNAME, column_array, load_features
from battery_optimizer.load_forecast import SideInputs
from battery_optimizer.paths import resolve_data_dir
from battery_optimizer.policies import contiguous_day_bounds
from battery_optimizer.structured_forecast import replay_mask_for_day

__all__ = [
    "ClimateData",
    "ClimateDataError",
    "Site",
    "build_climate_data",
    "debias_outdoor",
    "dataset_prices",
    "load_climate_data",
    "load_site",
    "room_offsets",
    "solar_proxy",
    "steady_circulation",
]

SLOT_SECONDS = 900
SLOT_US = SLOT_SECONDS * 1_000_000
SLOTS_PER_HOUR = 4
ROOMS = ("bedroom", "hall_kitchen", "office_t", "office_a")
SITE_FILE = Path("climate") / "site.yaml"

#: Reasons a local day has no replayed ``u`` that are not a data problem.
REASON_PARTIAL_DAY = "partial_day"
REASON_RULE_DISABLED = "rule_disabled"

#: Training filters, in the order of the plan.
FILTERS = (
    "unknown_quality",
    "outdoor_unknown",
    "window_open",
    "valves_shut",
    "thermostat_outage",
    "cooling",
    "rule_disabled_day",
)


class ClimateDataError(RuntimeError):
    """Expected, user-facing failure of the climate loader."""


# ---------------------------------------------------------------------------
# Private site file
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Site:
    """Site coordinates (and optional always-open valve share) from the private YAML."""

    latitude: float
    longitude: float
    phi: float | None
    sha256: str


def load_site(path: str | Path) -> Site:
    """Read ``<data_dir>/climate/site.yaml``: ``latitude``, ``longitude`` and optional ``phi``.

    The file lives in the private data dir and is never committed; its hash is recorded in
    results so a different site cannot go unnoticed.
    """
    p = Path(path).expanduser()
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ClimateDataError(f"site file {p}: cannot read ({exc.strerror})") from exc
    raw = yaml.safe_load(payload)
    if not isinstance(raw, dict) or not {"latitude", "longitude"} <= raw.keys():
        raise ClimateDataError(f"site file {p}: needs latitude and longitude")
    extra = set(raw) - {"latitude", "longitude", "phi"}
    if extra:
        raise ClimateDataError(f"site file {p}: unknown key(s) {sorted(extra)}")

    def number(key: str, lo: float, hi: float) -> float:
        v = raw[key]
        if isinstance(v, bool) or not isinstance(v, int | float) or not lo <= v <= hi:
            raise ClimateDataError(f"site file {p}: {key} must be a number in [{lo}, {hi}]")
        return float(v)

    phi = number("phi", 0.0, 1.0) if raw.get("phi") is not None else None
    return Site(
        latitude=number("latitude", -90.0, 90.0),
        longitude=number("longitude", -180.0, 180.0),
        phi=phi,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


# ---------------------------------------------------------------------------
# Solar proxy
# ---------------------------------------------------------------------------


def solar_proxy(t_us: np.ndarray, latitude: float, longitude: float) -> np.ndarray:
    """``max(0, sin(solar elevation))`` at the UTC instants *t_us* (int64 microseconds).

    The NOAA low-precision series: declination and equation of time from the fractional year,
    hour angle from true solar time (longitude east positive).  Accurate to a fraction of a
    degree, which is far below what a clear-sky proxy for an irradiance-free site needs.
    """
    t = np.asarray(t_us, dtype=np.int64)
    day = t.astype("datetime64[us]").astype("datetime64[D]")
    doy = (day - day.astype("datetime64[Y]")).astype(np.int64) + 1
    year_days = np.where(
        (day.astype("datetime64[Y]").astype(np.int64) + 1970) % 4 == 0, 366.0, 365.0
    )
    minutes = (t - day.astype("datetime64[us]").astype(np.int64)) / 60_000_000.0
    g = 2.0 * np.pi / year_days * (doy - 1 + (minutes / 60.0 - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(g)
        - 0.032077 * np.sin(g)
        - 0.014615 * np.cos(2 * g)
        - 0.040849 * np.sin(2 * g)
    )
    decl = (
        0.006918
        - 0.399912 * np.cos(g)
        + 0.070257 * np.sin(g)
        - 0.006758 * np.cos(2 * g)
        + 0.000907 * np.sin(2 * g)
        - 0.002697 * np.cos(3 * g)
        + 0.00148 * np.sin(3 * g)
    )
    solar_minutes = minutes + eqtime + 4.0 * longitude
    hour_angle = np.radians(solar_minutes / 4.0 - 180.0)
    lat = math.radians(latitude)
    sin_elevation = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    return np.maximum(0.0, sin_elevation)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


def _expected_slots(day: date, tz: ZoneInfo) -> int:
    """Slots of a whole local day: 96, or 92 / 100 across a DST change."""
    start = datetime.combine(day, time.min, tzinfo=tz).astimezone(UTC)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=tz).astimezone(UTC)
    return int(round((end - start).total_seconds() / SLOT_SECONDS))


def _local_calendar(starts_us: np.ndarray, tz: ZoneInfo) -> tuple[list[datetime], np.ndarray, list]:
    starts = [datetime.fromtimestamp(int(u) / 1e6, tz=UTC) for u in starts_us]
    local = [s.astimezone(tz) for s in starts]
    hour = np.array([t.hour for t in local], dtype=np.int16)
    return starts, hour, [t.date() for t in local]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ClimateData:
    """Climate signals on the feature grid.  Arrays have one entry per 15-minute slot."""

    feature_set_id: str
    dataset_id: str | None
    config_sha256: str
    site_sha256: str
    tz: ZoneInfo
    starts_us: np.ndarray
    local_hour: np.ndarray
    #: ``(local date, first slot, one past the last slot)`` of every local day on the grid.
    days: list[tuple[date, int, int]]
    #: ``True`` for a day with all its 92 / 96 / 100 slots on the grid.
    day_complete: np.ndarray
    day_of_slot: np.ndarray
    #: ``"S1"`` / ``"S2"`` / ``"S3"`` per local day, ``""`` outside the seasons.
    day_season: np.ndarray
    day_fit_month: np.ndarray
    day_gate_month: np.ndarray
    rooms: tuple[str, ...]
    comfort_rooms: tuple[str, ...]
    room_temp: dict[str, np.ndarray]
    room_setpoint: dict[str, np.ndarray]
    t_air: np.ndarray
    #: Raw Aquarea outdoor temperature (diagnostics); models read ``t_out_debiased``.
    t_out: np.ndarray
    t_out_debiased: np.ndarray
    #: Slots whose raw reading is trusted (compressor off long enough).
    t_out_valid: np.ndarray
    #: Slots of ``t_out_debiased`` that were interpolated across a running period.
    t_out_interpolated: np.ndarray
    q_heat_kwh: np.ndarray
    e_heat_kwh: np.ndarray
    dhw: np.ndarray
    cooling: np.ndarray
    valve_open_inferred: dict[str, np.ndarray]
    solar: np.ndarray
    price: np.ndarray
    #: Replayed legacy permission (1.0 / 0.0); ``NaN`` where there is none, see ``u_known``.
    u: np.ndarray
    #: Why the replay gave no ``u`` for a local day (``""`` when it did).
    u_reason: list[str]
    drop: dict[str, np.ndarray]
    train: np.ndarray
    #: Raw feature columns that later stages use directly (flow, return, tank, starts, ...).
    columns: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.starts_us.size)

    @property
    def u_known(self) -> np.ndarray:
        """Slots with a replayed ``u``; the only slots on which u-driven code may read it."""
        return np.isfinite(self.u)

    @property
    def day_u_known(self) -> np.ndarray:
        """Per local day: ``True`` when the replay gave a schedule."""
        return np.array([r == "" for r in self.u_reason], dtype=bool)

    @property
    def q_heat_kw(self) -> np.ndarray:
        return self.q_heat_kwh * (3600.0 / SLOT_SECONDS)

    def day_mask(self, flags: np.ndarray) -> np.ndarray:
        """Spread a per-day boolean over the slots of each day."""
        return np.asarray(flags, dtype=bool)[self.day_of_slot]

    def season_slots(self, season: str, *, months: str = "fit") -> np.ndarray:
        """Slots of complete days in *season*; ``months`` is ``"fit"`` or ``"gate"``."""
        flag = self.day_fit_month if months == "fit" else self.day_gate_month
        return self.day_mask((self.day_season == season) & self.day_complete & flag)

    def counter_check(self, min_daily_kwh: float) -> dict[str, np.ndarray]:
        """Daily thermal energy from the power integral against the energy counter.

        A day counts when every slot of it is known in both series.  Returns per-day arrays
        ``days``, ``power_kwh``, ``counter_kwh`` and ``ratio`` (power / counter) for the days
        whose counter total is at least *min_daily_kwh*.
        """
        power = self.q_heat_kwh
        counter = self.columns["hp_heat_energy_out_delta"]
        rows: list[tuple[date, float, float]] = []
        for (label, lo, hi), complete in zip(self.days, self.day_complete, strict=True):
            if not complete:
                continue
            p, c = power[lo:hi], counter[lo:hi]
            if not (np.isfinite(p).all() and np.isfinite(c).all()):
                continue
            if c.sum() >= min_daily_kwh:
                rows.append((label, float(p.sum()), float(c.sum())))
        power_kwh = np.array([r[1] for r in rows])
        counter_kwh = np.array([r[2] for r in rows])
        return {
            "days": np.array([r[0] for r in rows], dtype=object),
            "power_kwh": power_kwh,
            "counter_kwh": counter_kwh,
            "ratio": power_kwh / counter_kwh if rows else np.zeros(0),
        }


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def room_offsets(
    room_temp: Mapping[str, np.ndarray],
    comfort_rooms: Sequence[str],
    t_air: np.ndarray,
    local_hour: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Median offset of each comfort room from ``t_air`` by local hour over the *mask* slots.

    ``(len(comfort_rooms), 24)``; ``NaN`` for an hour without a usable slot.  A room's
    temperature is then estimated as ``t_air + offsets[room, hour]``.
    """
    out = np.full((len(comfort_rooms), 24), np.nan)
    ok = np.asarray(mask, dtype=bool) & np.isfinite(t_air)
    for i, room in enumerate(comfort_rooms):
        diff = room_temp[room] - t_air
        for hour in range(24):
            sel = ok & (local_hour == hour) & np.isfinite(diff)
            if sel.any():
                out[i, hour] = float(np.median(diff[sel]))
    return out


def steady_circulation(
    circulating: np.ndarray, min_minutes: float, step_minutes: float = 15.0
) -> np.ndarray:
    """Slots at least *min_minutes* into a circulation run, guaranteed at the slot's start.

    A slot with ``k`` consecutive circulating slots before it has been running for at least
    ``k * step_minutes`` when it begins (the run may have started earlier inside the first slot),
    so it qualifies when ``k * step_minutes >= min_minutes``.
    """
    need = max(int(np.ceil(min_minutes / step_minutes)), 0)
    steady = np.asarray(circulating, dtype=bool).copy()
    for k in range(1, need + 1):
        before = np.zeros(steady.size, dtype=bool)
        before[k:] = circulating[:-k]
        steady &= before
    return steady


def debias_outdoor(
    t_out: np.ndarray,
    compressor_freq: np.ndarray,
    *,
    min_off_slots: int,
    max_gap_slots: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outdoor temperature without the compressor's own influence.

    A slot is *valid* when the compressor was off (mean frequency exactly 0, known) in the slot and
    in the *min_off_slots* slots before it, and the reading is known.  Between valid slots the
    series is interpolated linearly in time when at most *max_gap_slots* slots lie between them;
    a longer run, and anything before the first or after the last valid slot, is ``NaN``.

    Returns ``(debiased, valid, interpolated)``.
    """
    t = np.asarray(t_out, dtype=np.float64)
    freq = np.asarray(compressor_freq, dtype=np.float64)
    n = t.size
    with np.errstate(invalid="ignore"):
        off = np.isfinite(freq) & (freq == 0.0)
    ok = off.copy()
    for k in range(1, min_off_slots + 1):
        before = np.zeros(n, dtype=bool)
        before[k:] = off[:-k]
        ok &= before
    valid = ok & np.isfinite(t)
    out = np.full(n, np.nan)
    out[valid] = t[valid]
    idx = np.flatnonzero(valid)
    if idx.size >= 2:
        slots = np.arange(n)
        nxt = np.searchsorted(idx, slots)
        inside = (nxt > 0) & (nxt < idx.size) & ~valid
        gap = np.zeros(n, dtype=np.int64)
        gap[inside] = idx[nxt[inside]] - idx[nxt[inside] - 1] - 1
        fill = inside & (gap <= max_gap_slots)
        out[fill] = np.interp(slots[fill], idx, t[idx])
    else:
        fill = np.zeros(n, dtype=bool)
    return out, valid, fill


def dataset_prices(dataset_table: pa.Table, starts_us: np.ndarray) -> np.ndarray:
    """PVPC price (EUR/kWh) of the dataset aligned to *starts_us*; ``NaN`` where it has none.

    A price the dataset marks ``missing`` or ``ambiguous`` is unknown; a forward-filled one is
    kept (the legacy rule read the sensor state, which is also held).
    """
    from battery_optimizer.benchmark import NEVER_PRICE

    d_ts = dataset_table.column("interval_start").cast(pa.int64()).to_numpy()
    price = np.array(dataset_table.column("price_eur_per_kwh").to_pylist(), dtype=np.float64)
    quality = dataset_table.column("price_quality").to_pylist()
    price[np.array([q in NEVER_PRICE for q in quality], dtype=bool)] = np.nan
    pos = np.clip(np.searchsorted(d_ts, starts_us), 0, max(d_ts.size - 1, 0))
    found = (d_ts.size > 0) & (d_ts[pos] == starts_us)
    return np.where(found, price[pos], np.nan)


def _require(table: pa.Table, names: Sequence[str]) -> None:
    missing = [n for n in names if n not in table.column_names]
    if missing:
        raise ClimateDataError(f"the feature table lacks the column(s) {missing}")


def _replay_u(
    *,
    starts: list[datetime],
    tz: ZoneInfo,
    local_hour: np.ndarray,
    days: list[tuple[date, int, int]],
    day_complete: np.ndarray,
    price: np.ndarray,
    enable: np.ndarray,
    min_hours: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Replay the legacy rule for every complete local day.

    The decision is at 00:00:40 local, inside the day's first slot, so the flags are the ones
    held at that slot's start (``cut = first slot + 1``) and the prices are the day's own."""
    side = SideInputs(
        interval_starts_utc=starts,
        tz=tz,
        prices=price,
        series={"climate_enable": enable, "climate_min_hours": min_hours},
    )
    view = side.view(len(starts), days)
    u = np.full(len(starts), np.nan)
    reasons: list[str] = []
    for j, (_label, lo, hi) in enumerate(days):
        if not day_complete[j]:
            reasons.append(REASON_PARTIAL_DAY)
            continue
        mask, reason = replay_mask_for_day(view, days, price[:hi], j, lo + 1)
        reasons.append(reason)
        if mask is not None:
            u[lo:hi] = mask[local_hour[lo:hi]].astype(float)
    return u, reasons


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

_ROOM_COLUMNS = tuple(
    c for r in ROOMS for c in (f"room_{r}_temp_mean_c", f"room_{r}_setpoint_mean_c")
)
_SIGNAL_COLUMNS = (
    "outdoor_temp_mean_c",
    "hp_heat_kwh",
    "hp_heat_out_kwh",
    "hp_dhw_kwh",
    "hp_dhw_out_kwh",
    "hp_cool_kwh",
    "hp_cool_out_kwh",
    "hp_heat_energy_out_delta",
    "dhw_force_mean",
    "climate_enable_at_start",
    "climate_min_hours_at_start",
)
#: Raw columns kept for later stages (COP, slab proxy, heating curve, audits).
_KEPT_COLUMNS = (
    "tank_temp_mean_c",
    "flow_temp_mean_c",
    "return_temp_mean_c",
    "flow_target_temp_mean_c",
    "compressor_freq_mean",
    "compressor_starts_delta",
    "pump_flow_mean",
    "servers_kwh",
    "hp_dhw_kwh",
    "hp_cool_kwh",
    "hp_dhw_out_kwh",
    "hp_heat_energy_out_delta",
    "hp_heat_energy_delta",
    "branch_aero_kwh",
    "branch_aero_energy_delta",
    "climate_enable_at_start",
    "climate_min_hours_at_start",
    "dhw_force_mean",
)
REQUIRED_COLUMNS = ("interval_start", *_ROOM_COLUMNS, *_SIGNAL_COLUMNS, *_KEPT_COLUMNS)


def build_climate_data(
    table: pa.Table,
    *,
    config: ClimateConfig,
    site: Site,
    prices: np.ndarray | None = None,
    feature_set_id: str = "",
    dataset_id: str | None = None,
) -> ClimateData:
    """Derive the climate signals from a feature table (and prices on the same grid)."""
    _require(table, REQUIRED_COLUMNS)
    tz = ZoneInfo(config.timezone)
    n = table.num_rows
    if n == 0:
        raise ClimateDataError("the feature table is empty")
    starts_us = table.column("interval_start").cast(pa.int64()).to_numpy()
    if (np.diff(starts_us) != SLOT_US).any() or starts_us[0] % (3600 * 1_000_000):
        raise ClimateDataError("the feature table needs a uniform 15-minute grid on full hours")
    price = np.full(n, np.nan) if prices is None else np.asarray(prices, dtype=np.float64)
    if price.size != n:
        raise ClimateDataError("prices must be aligned to the feature grid")

    def col(name: str) -> np.ndarray:
        return column_array(table, name)

    starts, local_hour, local_dates = _local_calendar(starts_us, tz)
    days = contiguous_day_bounds(local_dates)
    day_of_slot = np.empty(n, dtype=np.int64)
    for j, (_label, lo, hi) in enumerate(days):
        day_of_slot[lo:hi] = j
    day_complete = np.array(
        [hi - lo == _expected_slots(label, tz) for label, lo, hi in days], dtype=bool
    )
    seasons = config.seasons
    day_season = np.array(
        [
            next((s for s in ("S1", "S2", "S3") if seasons[s].start <= d <= seasons[s].end), "")
            for d, _lo, _hi in days
        ],
        dtype=object,
    )
    day_fit_month = np.array([d.month in seasons.fit_months for d, _l, _h in days], dtype=bool)
    day_gate_month = np.array([d.month in seasons.gate_months for d, _l, _h in days], dtype=bool)

    room_temp = {r: col(f"room_{r}_temp_mean_c") for r in ROOMS}
    room_setpoint = {r: col(f"room_{r}_setpoint_mean_c") for r in ROOMS}
    comfort = tuple(config.comfort.rooms)
    comfort_temp = np.vstack([room_temp[r] for r in comfort])
    comfort_sp = np.vstack([room_setpoint[r] for r in comfort])
    t_air = comfort_temp.mean(axis=0)  # NaN unless every comfort room is known

    t_out = col("outdoor_temp_mean_c")
    q_heat = col("hp_heat_out_kwh")
    e_heat = col("hp_heat_kwh")
    sig = config.signals

    def producing(electric: str, thermal: str, e_min: float, q_min: float):
        e, q = col(electric), col(thermal)
        with np.errstate(invalid="ignore"):
            on = (e >= e_min) | (q >= q_min)
        known = on | (np.isfinite(e) & np.isfinite(q))
        return on, known

    dhw, dhw_known = producing(
        "hp_dhw_kwh", "hp_dhw_out_kwh", sig.dhw.min_electric_kwh, sig.dhw.min_thermal_kwh
    )
    with np.errstate(invalid="ignore"):
        forced = col("dhw_force_mean") >= sig.dhw.force_share
    dhw |= forced
    dhw_known |= forced
    cooling, cooling_known = producing(
        "hp_cool_kwh", "hp_cool_out_kwh", sig.cooling.min_electric_kwh, sig.cooling.min_thermal_kwh
    )

    margin = sig.valve.inferred_state_margin_k
    with np.errstate(invalid="ignore"):
        valve_open = {r: room_temp[r] < room_setpoint[r] - margin for r in ROOMS}
    step_min = SLOT_SECONDS // 60
    t_out_debiased, t_out_valid, t_out_interpolated = debias_outdoor(
        t_out,
        col("compressor_freq_mean"),
        min_off_slots=sig.t_out_debias.min_off_minutes // step_min,
        max_gap_slots=int(sig.t_out_debias.max_gap_hours * 60 // step_min),
    )

    # --- training filters (True = drop the slot) ----------------------------------------------
    flt = config.filters
    required = [
        np.isfinite(t_air),
        np.isfinite(comfort_sp).all(axis=0),
        np.isfinite(q_heat),
        dhw_known,
        cooling_known,
    ]
    unknown = ~np.logical_and.reduce(required) if flt.drop_unknown_quality else np.zeros(n, bool)
    # kept apart from unknown_quality: a long compressor run leaves T_out unknown by design, and
    # diagnostics that do not need T_out must not inherit that
    outdoor_unknown = (
        ~np.isfinite(t_out_debiased) if flt.drop_unknown_quality else np.zeros(n, bool)
    )

    window_open = np.zeros(n, dtype=bool)
    k = flt.window_open.slots
    with np.errstate(invalid="ignore"):
        fall = comfort_temp[:, k:] - comfort_temp[:, :-k] < -flt.window_open.drop_k
    window_open[k:] = fall.any(axis=0)

    with np.errstate(invalid="ignore"):
        at_setpoint = (comfort_temp >= comfort_sp).sum(axis=0)
    valves_shut = at_setpoint >= flt.valves_shut.min_rooms_at_setpoint

    outage_days = np.zeros(len(days), dtype=bool)
    for w in flt.thermostat_outage.windows:
        outage_days |= np.array([w.start <= d <= w.end for d, _l, _h in days], dtype=bool)
    outage = outage_days[day_of_slot]

    drop_cooling = cooling if flt.drop_cooling_slots else np.zeros(n, dtype=bool)

    u, u_reason = _replay_u(
        starts=starts,
        tz=tz,
        local_hour=local_hour,
        days=days,
        day_complete=day_complete,
        price=price,
        enable=col("climate_enable_at_start"),
        min_hours=col("climate_min_hours_at_start"),
    )
    disabled_days = np.array([r == REASON_RULE_DISABLED for r in u_reason], dtype=bool)
    rule_disabled = disabled_days[day_of_slot] if flt.drop_rule_disabled_days else np.zeros(n, bool)

    drop = {
        "unknown_quality": unknown,
        "outdoor_unknown": outdoor_unknown,
        "window_open": window_open,
        "valves_shut": valves_shut,
        "thermostat_outage": outage,
        "cooling": drop_cooling,
        "rule_disabled_day": rule_disabled,
    }
    train = ~np.logical_or.reduce(list(drop.values()))

    return ClimateData(
        feature_set_id=feature_set_id,
        dataset_id=dataset_id,
        config_sha256=config.sha256,
        site_sha256=site.sha256,
        tz=tz,
        starts_us=starts_us,
        local_hour=local_hour,
        days=days,
        day_complete=day_complete,
        day_of_slot=day_of_slot,
        day_season=day_season,
        day_fit_month=day_fit_month,
        day_gate_month=day_gate_month,
        rooms=ROOMS,
        comfort_rooms=comfort,
        room_temp=room_temp,
        room_setpoint=room_setpoint,
        t_air=t_air,
        t_out=t_out,
        t_out_debiased=t_out_debiased,
        t_out_valid=t_out_valid,
        t_out_interpolated=t_out_interpolated,
        q_heat_kwh=q_heat,
        e_heat_kwh=e_heat,
        dhw=dhw,
        cooling=cooling,
        valve_open_inferred=valve_open,
        solar=solar_proxy(starts_us + SLOT_US // 2, site.latitude, site.longitude),
        price=price,
        u=u,
        u_reason=u_reason,
        drop=drop,
        train=train,
        columns={name: col(name) for name in dict.fromkeys(_KEPT_COLUMNS)},
    )


def load_climate_data(
    config: ClimateConfig,
    *,
    feature_set: str,
    dataset: str | Path | None = None,
    data_dir: str | None = None,
    site: Site | None = None,
) -> ClimateData:
    """Load a climate feature set (id or unique id prefix) and the prices of a dataset.

    The feature table and the dataset are verified through their manifests.  *site* defaults to
    ``<data_dir>/climate/site.yaml``.  Without a *dataset* there are no prices, so no ``u``.
    """
    root = resolve_data_dir(data_dir)
    base = root / "analysis" / "v1" / FEATURES_DIRNAME
    matches = sorted(p for p in base.glob(f"{feature_set}*") if p.is_dir())
    if len(matches) != 1:
        raise ClimateDataError(f"feature set {feature_set!r} matches {len(matches)} under {base}")
    table, manifest = load_features(matches[0])
    if "room_bedroom_temp_mean_c" not in table.column_names:
        raise ClimateDataError(f"{matches[0].name} is not a climate feature set")
    if site is None:
        site = load_site(root / SITE_FILE)
    prices = None
    dataset_id = None
    if dataset is not None:
        from battery_optimizer.benchmark import load_dataset

        loaded = load_dataset(dataset, data_dir=root)
        starts_us = table.column("interval_start").cast(pa.int64()).to_numpy()
        prices = dataset_prices(loaded.table, starts_us)
        dataset_id = loaded.dataset_id
        tz_local = loaded.table.column("tz_local")[0].as_py()
        if tz_local != config.timezone:
            raise ClimateDataError(
                f"the dataset's time zone {tz_local!r} differs from the config's "
                f"{config.timezone!r}"
            )
    return build_climate_data(
        table,
        config=config,
        site=site,
        prices=prices,
        feature_set_id=str(manifest.get("feature_set_id") or matches[0].name),
        dataset_id=dataset_id,
    )
