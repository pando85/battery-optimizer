"""Live planning: the offline MPC kernel driven from a frozen input snapshot.

There is no second optimizer here.  A plan is
:func:`battery_optimizer.oracle.solve_oracle` in ``terminal_mode="free"`` with an
explicit terminal-inventory value and the configured throughput cost — the exact
call the rolling-horizon benchmark makes per event — followed by
:func:`battery_optimizer.dispatch.intents_from_schedule`, which re-checks the plan
against the shared battery kernel and emits the TODO-4.6 dispatch contract.  The
battery parameters come from ``tariff.yaml`` through
:func:`battery_optimizer.benchmark.resolve_battery_assumptions`, the mapping the
benchmark uses (``sqrt(RTE)`` per leg, configured SOC window and import ceiling).

Two deliberate differences from the benchmark, both forced by running live:

* **SOC floor.**  The benchmark floors every step at its evaluation window's
  opening SOC (seed protection, a fairness device for comparing policies).  A
  live plan has no evaluation window: the energy in the battery is real, so the
  floor is the physical window minimum (``soc_window_min``) and the leftover is
  valued at the terminal value instead.
* **Price horizon.**  The benchmark may plan past the published prices with
  exact or noise-perturbed *true* prices.  Live prices beyond the last published
  hour do not exist, so the horizon ends at the last published slot
  (``max_horizon_slots`` caps it); the terminal value is what keeps the plan from
  emptying the battery at the end of the published window.

Everything the solve depends on is frozen in :class:`PlanningSnapshot` before it
is handed to the worker thread, and its canonical hash (``inputs_hash``) is stored
with every plan and intent.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from battery_optimizer.battery import BatterySpec, Horizon
from battery_optimizer.config import TariffConfig
from battery_optimizer.dispatch import DispatchMode, DispatchPlan, intents_from_schedule
from battery_optimizer.oracle import solve_oracle

__all__ = [
    "PlanResult",
    "PlanningSnapshot",
    "ShadowIntent",
    "battery_spec_from_tariff",
    "clamp_soc",
    "run_planner",
    "shadow_intents",
]


def battery_spec_from_tariff(
    tariff: TariffConfig, soc_frac: float
) -> tuple[BatterySpec, float | None, dict[str, Any]]:
    """``(spec, import_limit_kw, provenance)`` via the benchmark's resolver."""
    from battery_optimizer.benchmark import resolve_battery_assumptions

    assumptions = resolve_battery_assumptions(tariff, initial_soc_frac=soc_frac)
    spec = BatterySpec(**assumptions.spec_kwargs())
    provenance = {
        "efficiency_mapping": assumptions.efficiency_mapping,
        "import_limit_source": assumptions.import_limit_source,
    }
    return spec, assumptions.import_limit_kw, provenance


def clamp_soc(soc_frac: float, tariff: TariffConfig) -> tuple[float, bool]:
    """Clamp a measured SOC into the configured planning window."""
    lo = float(tariff.battery.soc_window_min or 0.0)
    hi = float(tariff.battery.soc_window_max or 1.0)
    clamped = min(max(soc_frac, lo), hi)
    return clamped, not math.isclose(clamped, soc_frac, abs_tol=1e-12)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


@dataclass(frozen=True)
class PlanningSnapshot:
    """Every input of one solve, frozen before it leaves the event loop."""

    decision_time: datetime
    interval_seconds: int
    slot_starts: tuple[datetime, ...]
    prices: tuple[float, ...]
    load_kwh: tuple[float, ...]
    soc_frac: float
    soc_measured_frac: float | None
    soc_clamped: bool
    soc_source: str
    soc_observed_at: datetime | None
    spec: BatterySpec
    import_limit_kw: float | None
    terminal_value_eur_per_kwh: float
    terminal_value_source: str
    throughput_cost_eur_per_kwh: float
    solver_time_limit_s: float
    plan_sequence: int
    expires_at: datetime
    trigger: str
    code_version: str
    config_hash: str
    forecast_meta: dict[str, Any] = field(default_factory=dict)
    price_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.slot_starts)
        if n == 0 or len(self.prices) != n or len(self.load_kwh) != n:
            raise ValueError("snapshot slots, prices and load must be non-empty and aligned")

    @property
    def first_slot(self) -> datetime:
        return self.slot_starts[0]

    def input_payload(self) -> dict[str, Any]:
        """The solve's inputs (no decision instant, trigger or sequence)."""
        return {
            "interval_seconds": self.interval_seconds,
            "slot_starts": [_iso(s) for s in self.slot_starts],
            "prices": [float(p) for p in self.prices],
            "load_kwh": [round(float(v), 9) for v in self.load_kwh],
            "soc_frac": round(self.soc_frac, 9),
            "soc_source": self.soc_source,
            "spec": {
                k: getattr(self.spec, k)
                for k in (
                    "capacity_kwh",
                    "power_kw",
                    "charge_efficiency",
                    "discharge_efficiency",
                    "soc_min_frac",
                    "soc_max_frac",
                )
            },
            "import_limit_kw": self.import_limit_kw,
            "terminal_value_eur_per_kwh": self.terminal_value_eur_per_kwh,
            "throughput_cost_eur_per_kwh": self.throughput_cost_eur_per_kwh,
            "solver_time_limit_s": self.solver_time_limit_s,
            "config_hash": self.config_hash,
        }

    def inputs_hash(self) -> str:
        text = json.dumps(self.input_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()

    def summary(self) -> dict[str, Any]:
        return {
            "decision_time": _iso(self.decision_time),
            "first_slot": _iso(self.first_slot),
            "slots": len(self.slot_starts),
            "horizon_end": _iso(self.slot_starts[-1] + timedelta(seconds=self.interval_seconds)),
            "soc_frac": self.soc_frac,
            "soc_measured_frac": self.soc_measured_frac,
            "soc_clamped": self.soc_clamped,
            "soc_source": self.soc_source,
            "soc_observed_at": _iso(self.soc_observed_at) if self.soc_observed_at else None,
            "terminal_value_eur_per_kwh": self.terminal_value_eur_per_kwh,
            "terminal_value_source": self.terminal_value_source,
            "import_limit_kw": self.import_limit_kw,
            "trigger": self.trigger,
            "forecast": self.forecast_meta,
            "prices": self.price_meta,
            "load_forecast_kwh_total": float(sum(self.load_kwh)),
        }


@dataclass(frozen=True)
class ShadowIntent:
    """One time-bounded **shadow** intent: recorded, never sent anywhere."""

    plan_id: str
    plan_sequence: int
    sequence_id: int
    valid_from: datetime
    valid_until: datetime
    """``min(interval end, plan expiry)``; an intent is never current outside it."""
    interval_end: datetime
    mode: str
    target_power_w: float
    signed_power_w: float
    """``+`` charge, ``-`` discharge (requested AC power)."""
    soc_target_frac: float
    expected_buy_price_eur_per_kwh: float | None
    generated_at: datetime
    expires_at: datetime
    inputs_hash: str
    code_version: str
    shadow: bool = True

    @property
    def actionable(self) -> bool:
        """False for intents that start after the plan expired (forecast only)."""
        return self.valid_from < self.valid_until

    def is_current(self, now: datetime) -> bool:
        return self.actionable and self.valid_from <= now < self.valid_until

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_sequence": self.plan_sequence,
            "sequence_id": self.sequence_id,
            "valid_from": _iso(self.valid_from),
            "valid_until": _iso(self.valid_until),
            "interval_end": _iso(self.interval_end),
            "actionable": self.actionable,
            "mode": self.mode,
            "target_power_w": self.target_power_w,
            "signed_power_w": self.signed_power_w,
            "soc_target_frac": self.soc_target_frac,
            "expected_buy_price_eur_per_kwh": self.expected_buy_price_eur_per_kwh,
            "generated_at": _iso(self.generated_at),
            "expires_at": _iso(self.expires_at),
            "inputs_hash": self.inputs_hash,
            "code_version": self.code_version,
            "shadow": self.shadow,
        }


def shadow_intents(
    plan: DispatchPlan, *, plan_id: str, inputs_hash: str, code_version: str
) -> list[ShadowIntent]:
    out: list[ShadowIntent] = []
    for intent in plan.intents:
        mode = DispatchMode(intent.mode)
        sign = (
            1.0 if mode is DispatchMode.CHARGE else -1.0 if mode is DispatchMode.DISCHARGE else 0.0
        )
        out.append(
            ShadowIntent(
                plan_id=plan_id,
                plan_sequence=plan.plan_sequence,
                sequence_id=intent.sequence_id,
                valid_from=intent.start,
                valid_until=min(intent.end, plan.expires_at),
                interval_end=intent.end,
                mode=mode.value,
                target_power_w=intent.target_power_w,
                signed_power_w=sign * intent.target_power_w,
                soc_target_frac=intent.soc_target_frac,
                expected_buy_price_eur_per_kwh=intent.expected_buy_price_eur_per_kwh,
                generated_at=plan.generated_at,
                expires_at=plan.expires_at,
                inputs_hash=inputs_hash,
                code_version=code_version,
            )
        )
    return out


@dataclass(frozen=True)
class PlanResult:
    snapshot: PlanningSnapshot
    plan: DispatchPlan
    plan_id: str
    inputs_hash: str
    intents: tuple[ShadowIntent, ...]
    solver: dict[str, Any]
    solve_seconds: float

    def current_intent(self, now: datetime) -> ShadowIntent | None:
        for intent in self.intents:
            if intent.is_current(now):
                return intent
        return None

    def intent_for_slot(self, slot_start: datetime) -> ShadowIntent | None:
        """The actionable intent valid at *slot_start*, if this plan has one."""
        return self.current_intent(slot_start)

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": "shadow_plan",
            "shadow": True,
            "plan_id": self.plan_id,
            "plan_sequence": self.plan.plan_sequence,
            "generated_at": _iso(self.plan.generated_at),
            "expires_at": _iso(self.plan.expires_at),
            "content_hash": self.plan.content_hash,
            "inputs_hash": self.inputs_hash,
            "code_version": self.snapshot.code_version,
            "config_hash": self.snapshot.config_hash,
            "solver": self.solver,
            "solve_seconds": self.solve_seconds,
            "snapshot": self.snapshot.summary(),
            "inputs": self.snapshot.input_payload(),
            "intents": [i.to_dict() for i in self.intents],
        }


def run_planner(snapshot: PlanningSnapshot) -> PlanResult:
    """Solve one snapshot.  CPU-bound and pure: runs in the bounded worker."""
    started = time.perf_counter()
    spec = replace(snapshot.spec, initial_soc_frac=snapshot.soc_frac, terminal_soc_frac=None)
    n = len(snapshot.slot_starts)
    horizon = Horizon(
        interval_seconds=snapshot.interval_seconds,
        prices=np.asarray(snapshot.prices, dtype=float),
        load_kwh=np.asarray(snapshot.load_kwh, dtype=float),
        usable=np.ones(n, dtype=bool),
        import_limit_kw=snapshot.import_limit_kw,
    )
    schedule = solve_oracle(
        spec,
        horizon,
        time_limit_s=snapshot.solver_time_limit_s,
        terminal_mode="free",
        terminal_value_eur_per_kwh=snapshot.terminal_value_eur_per_kwh,
        floor_kwh=spec.soc_bound_kwh("min"),
        throughput_cost_eur_per_kwh=snapshot.throughput_cost_eur_per_kwh,
    )
    plan = intents_from_schedule(
        schedule,
        horizon,
        battery_spec=spec,
        interval_starts_utc=list(snapshot.slot_starts),
        generated_at=snapshot.decision_time,
        expires_at=snapshot.expires_at,
        plan_sequence=snapshot.plan_sequence,
        expected_buy_prices=list(snapshot.prices),
    )
    inputs_hash = snapshot.inputs_hash()
    plan_id = (
        f"{snapshot.decision_time.strftime('%Y%m%dT%H%M%SZ')}-{snapshot.plan_sequence}-"
        f"{plan.content_hash[:8]}"
    )
    intents = shadow_intents(
        plan, plan_id=plan_id, inputs_hash=inputs_hash, code_version=snapshot.code_version
    )
    solver = {
        "kernel": "oracle.solve_oracle(terminal_mode='free') + dispatch.intents_from_schedule",
        "status": schedule.solver_status,
        "gap": schedule.solver_gap,
        "certified": schedule.is_certified,
        "cost_eur": schedule.cost_eur,
        "meter_cost_eur": schedule.meter_cost_eur,
        "terminal_soc_kwh": schedule.terminal_soc_kwh,
        "charge_kwh": float(np.sum(schedule.charge_kwh)),
        "discharge_kwh": float(np.sum(schedule.discharge_kwh)),
    }
    return PlanResult(
        snapshot=snapshot,
        plan=plan,
        plan_id=plan_id,
        inputs_hash=inputs_hash,
        intents=tuple(intents),
        solver=solver,
        solve_seconds=time.perf_counter() - started,
    )
