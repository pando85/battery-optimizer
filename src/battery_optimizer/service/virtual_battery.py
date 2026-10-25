"""Open-loop replay of shadow decisions against measured load.

One slot at a time, a decision intent is turned into the energy the battery would
have moved and projected onto that slot's **measured** feasible set by
:func:`battery_optimizer.policies._clip_plan_to_model_bounds` — the same clipping
the rolling-horizon benchmark applies when it executes a forecast-made plan against
measured load (power limit, SOC window, import ceiling headroom, no export).  Flows
are only ever reduced, so the replay cannot invent an arbitrage.

Used twice, with one code path:

* online, by the service, as the counterfactual battery (and, in
  ``soc_source=simulated`` mode, as the SOC the planner starts from);
* offline, by ``shadow-report``, to recompute the counterfactual from the
  recorded intents and observations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from battery_optimizer.battery import BatterySpec, Horizon
from battery_optimizer.policies import _clip_plan_to_model_bounds

__all__ = ["SlotStep", "intent_energy_kwh", "soc_violation", "step_virtual_battery"]


@dataclass(frozen=True)
class SlotStep:
    soc_start_frac: float
    soc_end_frac: float
    charge_kwh: float
    discharge_kwh: float
    clipped_charge_kwh: float
    clipped_discharge_kwh: float
    grid_kwh: float | None
    """Counterfactual net import: ``load + charge - discharge`` (``None`` if load unknown)."""
    applied: bool
    """False when the load was unknown and the slot was held (no flows)."""


def intent_energy_kwh(
    mode: str | None, target_power_w: float, interval_seconds: float
) -> tuple[float, float]:
    """``(charge_kwh, discharge_kwh)`` an intent requests over one slot."""
    energy = max(0.0, float(target_power_w)) * interval_seconds / 3_600_000.0
    if mode == "charge":
        return energy, 0.0
    if mode == "discharge":
        return 0.0, energy
    return 0.0, 0.0


def soc_violation(
    spec: BatterySpec, soc_start_frac: float, charge_kwh: float, discharge_kwh: float
) -> bool:
    """Would applying the request from *soc_start_frac* leave the SOC window?"""
    soc_kwh = soc_start_frac * spec.capacity_kwh
    end = soc_kwh + charge_kwh * spec.charge_efficiency - discharge_kwh / spec.discharge_efficiency
    tol = 1e-6
    return end < spec.soc_bound_kwh("min") - tol or end > spec.soc_bound_kwh("max") + tol


def step_virtual_battery(
    spec: BatterySpec,
    *,
    soc_start_frac: float,
    mode: str | None,
    target_power_w: float,
    load_kwh: float | None,
    interval_seconds: float,
    import_limit_kw: float | None,
) -> SlotStep:
    """Apply one decision to the virtual battery for one slot."""
    lo, hi = spec.soc_min_frac, spec.soc_max_frac
    soc_start_frac = min(max(soc_start_frac, lo), hi)
    charge, discharge = intent_energy_kwh(mode, target_power_w, interval_seconds)
    if load_kwh is None or not np.isfinite(load_kwh):
        return SlotStep(soc_start_frac, soc_start_frac, 0.0, 0.0, charge, discharge, None, False)
    slot_spec = replace(spec, initial_soc_frac=soc_start_frac, terminal_soc_frac=None)
    horizon = Horizon(
        interval_seconds=interval_seconds,
        prices=np.zeros(1),
        load_kwh=np.array([max(0.0, float(load_kwh))]),
        usable=np.ones(1, dtype=bool),
        import_limit_kw=import_limit_kw,
    )
    c, d, end_soc_kwh, clip_c, clip_d = _clip_plan_to_model_bounds(
        slot_spec,
        horizon,
        np.array([charge]),
        np.array([discharge]),
        floor_kwh=spec.soc_bound_kwh("min"),
    )
    end_frac = min(max(end_soc_kwh / spec.capacity_kwh, lo), hi)
    grid = max(0.0, float(load_kwh)) + float(c[0]) - float(d[0])
    return SlotStep(
        soc_start_frac=soc_start_frac,
        soc_end_frac=end_frac,
        charge_kwh=float(c[0]),
        discharge_kwh=float(d[0]),
        clipped_charge_kwh=float(clip_c),
        clipped_discharge_kwh=float(clip_d),
        grid_kwh=grid,
        applied=True,
    )
