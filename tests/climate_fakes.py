"""Synthetic climate feature tables shared by the climate tests (no real data, no network)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa

from battery_optimizer.climate.config import load_climate_config
from battery_optimizer.climate.data import REQUIRED_COLUMNS, SLOT_US, Site, build_climate_data

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_climate_config(REPO_ROOT / "config" / "climate.yaml")
MADRID = ZoneInfo("Europe/Madrid")
SITE = Site(latitude=40.0, longitude=0.0, phi=None, sha256="synthetic")
SLOTS_DAY = 96


def local_midnight(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=MADRID).astimezone(UTC)


def grid(first_day: date, n_days: int) -> np.ndarray:
    """UTC slot starts (int64 microseconds) from the local midnight of *first_day*, whole days."""
    start = local_midnight(first_day)
    end = local_midnight(first_day + timedelta(days=n_days))
    return np.arange(int(start.timestamp()) * 1_000_000, int(end.timestamp()) * 1_000_000, SLOT_US)


def make_table(starts_us: np.ndarray, **overrides: np.ndarray | float) -> pa.Table:
    """A feature table with every column the loader reads, at benign constant values."""
    n = starts_us.size
    defaults: dict[str, float] = {
        "outdoor_temp_mean_c": 5.0,
        "hp_heat_kwh": 0.1,
        "hp_heat_out_kwh": 0.4,
        "hp_dhw_kwh": 0.0,
        "hp_dhw_out_kwh": 0.0,
        "hp_cool_kwh": 0.0,
        "hp_cool_out_kwh": 0.0,
        "hp_heat_energy_out_delta": 0.4,
        "hp_heat_energy_delta": 0.1,
        "dhw_force_mean": 0.0,
        "climate_enable_at_start": 1.0,
        "climate_min_hours_at_start": 6.0,
    }
    for room in ("bedroom", "hall_kitchen", "office_t", "office_a"):
        defaults[f"room_{room}_temp_mean_c"] = 21.0
        defaults[f"room_{room}_setpoint_mean_c"] = 24.0
    cols: dict[str, np.ndarray] = {}
    for name in REQUIRED_COLUMNS:
        if name == "interval_start":
            continue
        value = overrides.get(name, defaults.get(name, 1.0))
        cols[name] = np.array(np.broadcast_to(np.asarray(value, dtype=float), n), dtype=float)
    arrays = {"interval_start": pa.array(starts_us, pa.timestamp("us", tz="UTC"))}
    arrays.update({k: pa.array(v, pa.float64()) for k, v in cols.items()})
    return pa.table(arrays)


def build(table: pa.Table, prices: np.ndarray | None = None, config=CONFIG):
    return build_climate_data(table, config=config, site=SITE, prices=prices)
