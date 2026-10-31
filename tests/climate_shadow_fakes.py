"""Synthetic raw samples and a synthetic prior for the climate-shadow tests (no real data)."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from battery_optimizer.climate import mpc_replay, plant, shadow
from battery_optimizer.climate.config import load_climate_config
from battery_optimizer.climate.cop import COP_MODEL_VERSION, CopModel
from battery_optimizer.climate.data import Site
from battery_optimizer.climate.mpc import PlanSettings
from battery_optimizer.climate.mpc_config import load_mpc_config, override
from battery_optimizer.climate.mpc_model import HeatLaw
from battery_optimizer.features import RawSeries

REPO = Path(__file__).resolve().parents[1]
CCFG = load_climate_config(REPO / "config" / "climate.yaml")
# a 7-day refit window keeps the tests fast; everything else is the committed config
MPC = override(
    load_mpc_config(REPO / "config" / "climate-mpc.yaml"), ("model", "reestimate", "window_days"), 7
)
SITE = Site(latitude=40.0, longitude=0.0, phi=None, sha256="synthetic")
ROOMS = ("bedroom", "hall_kitchen", "office_t")
PARAMS = {"ua": 0.1, "c_a": 2.0, "c_s": 12.0, "k": 2.0, "q0": 0.3, "g_s": 0.8}
PARAMS.update({f"g{i}": 0.05 * (i % 3) for i in range(8)})
SETTINGS = replace(PlanSettings.from_config(MPC), relinearize_iterations=1, time_limit_s=3.0)
SAMPLE_S = 300  # one sample every 5 minutes (the max-gap rule is 300 s)


def cop() -> CopModel:
    return CopModel(
        version=COP_MODEL_VERSION,
        selected="lift_table",
        coefficients=(0.2, 0.01, 0.0, 0.05, 0.05),
        knee_c=5.0,
        table_bin_k=2.0,
        table=((0.0, 4.0, 100),),
        cop_range=(1.0, 9.0),
        train_seasons=("S3",),
        n_train_rows=100,
        feature_set_id="synthetic",
        config_sha256="synthetic",
        lift_bin_k=2.0,
        lift_table=tuple((float(lo), 6.0 - lo / 10.0, 100) for lo in range(10, 40, 2)),
    )


def prior(offsets: dict[str, float] | None = None) -> shadow.ShadowPrior:
    offsets = offsets or {}
    p = mpc_replay.Prior(
        season="S3",
        rooms=ROOMS,
        params=dict(PARAMS),
        n_diurnal=8,
        fit_info={"on_bound": []},
        law=HeatLaw(0.5, 4.0, 0.1, 22.0),
        law_stats={},
        cap=plant.ValveCap(0.3, 0.25, np.zeros((3, 24))),
        process_var=0.01,
        gain=np.array([0.5, 0.05]),
    )
    return shadow.ShadowPrior(
        prior=p,
        cop_model=cop(),
        room_offsets={r: [offsets.get(r, 0.0)] * 24 for r in shadow.DESIGN_ROOMS},
        lineage={"synthetic": True},
    )


def house(start: datetime, end: datetime, *, price_entity: str = "sensor.esios_pvpc") -> dict:
    """Raw samples of every live source on ``[start, end)``: a heated house with a daily DHW
    run at 03:00 UTC, heating 02:00 to 08:00 UTC, PVPC-like prices and the legacy rule on."""
    t = np.arange(int(start.timestamp()), int(end.timestamp()), SAMPLE_S, dtype=np.int64)
    hours = (t % 86400) / 3600.0
    rng = np.random.default_rng(int(t[0]) % 1000)
    heating = (hours >= 2.0) & (hours < 8.0)
    dhw = (hours >= 3.0) & (hours < 3.25)
    space = heating & ~dhw
    room = 21.6 + 0.4 * np.sin(2 * np.pi * (hours - 9.0) / 24.0) + rng.normal(0, 0.03, t.size)
    values: dict[str, np.ndarray] = {
        "outdoor_temp": 6.0 + 3.0 * np.sin(2 * np.pi * (hours - 15.0) / 24.0),
        "flow_target_temp": np.full(t.size, 27.0),
        "compressor_freq": np.where(heating, 40.0, 0.0),
        "hp_heat": np.where(space, 400.0, 0.0),
        "hp_heat_out": np.where(space, 1600.0, 0.0),
        "hp_dhw": np.where(dhw, 800.0, 0.0),
        "hp_dhw_out": np.where(dhw, 2400.0, 0.0),
        "hp_cool": np.zeros(t.size),
        "hp_cool_out": np.zeros(t.size),
        "dhw_force": np.zeros(t.size),
        "climate_enable": np.ones(t.size),
        "climate_min_hours": np.full(t.size, 6.0),
    }
    out: dict[str, RawSeries] = {}
    t_us = t * 1_000_000
    for sig in shadow.LIVE_SIGNALS:
        if sig.alias.startswith("room_"):
            v = np.full(t.size, 22.0) if sig.alias.endswith("_setpoint") else room
            if sig.alias == "room_office_a_temp":
                v = room + 1.5
        else:
            v = values[sig.alias]
        out[sig.source] = RawSeries(sig.entity, "m", "", t_us, np.asarray(v, float))
    price = 0.12 + 0.05 * np.sin(2 * np.pi * np.floor(hours) / 24.0)
    out[shadow.PRICE_KEY] = RawSeries(price_entity, "m", "", t_us, price)
    return out


class FakeSamples:
    """A sample source over a precomputed house; records every call."""

    def __init__(self, start: datetime, end: datetime) -> None:
        self.data = house(start, end)
        self.calls: list[datetime] = []

    def __call__(self, now: datetime) -> dict[str, RawSeries]:
        self.calls.append(now)
        us = int(now.timestamp() * 1e6)
        return {
            k: RawSeries(s.entity, s.metric, s.unit, s.t_us[s.t_us < us], s.values[s.t_us < us])
            for k, s in self.data.items()
        }


def utc(*args: int) -> datetime:
    return datetime(*args, tzinfo=UTC)


def forecast_record(at: datetime, hours: int = 30, level: float = 6.0) -> dict:
    return {
        "kind": "weather_forecast",
        "requested_at": at.isoformat(),
        "entity_id": "weather.forecast_home",
        "forecast": [
            {"datetime": (at + timedelta(hours=k)).isoformat(), "temperature": level + 0.1 * k}
            for k in range(hours)
        ],
    }
