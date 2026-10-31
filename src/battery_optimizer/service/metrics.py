"""Prometheus metrics (prefix ``battery_optimizer_``) on a private registry.

Labels are bounded by configuration (roles, configured entity IDs, fixed reason
and result enums).  No plan ID, timestamp, price or household reading is ever a
label: plan identity lives in the store and ``/plan``, not in time-series
cardinality.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    disable_created_metrics,
    generate_latest,
)
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector

__all__ = ["NOT_READY_REASONS", "ServiceMetrics"]

PREFIX = "battery_optimizer"

#: Every reason the service can report for not being ready (fixed label set).
NOT_READY_REASONS: tuple[str, ...] = (
    "ha_unconfigured",
    "ha_disconnected",
    "ha_not_synced",
    "grid_power_stale",
    "soc_unconfigured",
    "soc_stale",
    "prices_missing",
    "prices_insufficient",
    "history_unavailable",
    "forecast_unavailable",
    "tariff_invalid",
    "no_recent_plan",
)


class _RuntimeCollector(Collector):
    """Values computed at scrape time from the runtime's state."""

    def __init__(self, snapshot: Callable[[], dict[str, Any]]) -> None:
        self._snapshot = snapshot

    def collect(self) -> Iterable[Metric]:
        snap = self._snapshot()
        ha = snap.get("ha", {})

        def gauge(name: str, doc: str, value: float | None, labels: dict[str, str] | None = None):
            family = GaugeMetricFamily(f"{PREFIX}_{name}", doc, labels=list((labels or {}).keys()))
            if value is not None:
                family.add_metric(list((labels or {}).values()), float(value))
            return family

        yield gauge("ready", "1 when inputs are fresh and a recent plan exists", snap["ready"])
        yield gauge("inputs_ready", "1 when every planning input is fresh", snap["inputs_ready"])
        reasons = GaugeMetricFamily(
            f"{PREFIX}_not_ready_reason", "1 for each active not-ready reason", labels=["reason"]
        )
        active = set(snap.get("reasons", ()))
        for reason in NOT_READY_REASONS:
            reasons.add_metric([reason], 1.0 if reason in active else 0.0)
        yield reasons

        yield gauge(
            "ha_connected", "1 while the HA WebSocket is authenticated", ha.get("connected")
        )
        yield gauge("ha_synced", "1 after the post-connect get_states resync", ha.get("synced"))
        for name, key, doc in (
            ("ha_reconnects", "reconnects", "WebSocket reconnect attempts"),
            ("ha_resyncs", "resyncs", "Full get_states resynchronisations"),
            ("ha_state_events", "events", "state_changed events received for tracked entities"),
        ):
            family = CounterMetricFamily(f"{PREFIX}_{name}", doc)
            family.add_metric([], float(ha.get(key, 0)))
            yield family
        failures = CounterMetricFamily(
            f"{PREFIX}_ha_failures", "HA session failures by stage", labels=["stage"]
        )
        for stage in (
            "connect",
            "auth",
            "subscribe",
            "get_states",
            "heartbeat",
            "disconnect",
            "protocol",
        ):
            failures.add_metric([stage], float(ha.get("failures", {}).get(stage, 0)))
        yield failures
        updates = CounterMetricFamily(
            f"{PREFIX}_ha_state_updates",
            "Entity observations by classification (accepted, refreshed, duplicate, out_of_order)",
            labels=["result"],
        )
        for result in ("accepted", "refreshed", "duplicate", "out_of_order"):
            updates.add_metric([result], float(snap.get("updates", {}).get(result, 0)))
        yield updates

        ages = GaugeMetricFamily(
            f"{PREFIX}_input_age_seconds",
            "Seconds since HA last reported the entity (absent if never seen)",
            labels=["role", "entity"],
        )
        for item in snap.get("inputs", ()):
            if item.get("age_s") is not None:
                ages.add_metric([item["role"], item["entity"]], float(item["age_s"]))
        yield ages
        stale = GaugeMetricFamily(
            f"{PREFIX}_input_stale",
            "1 when the input is missing or older than its limit",
            labels=["role", "entity"],
        )
        for item in snap.get("inputs", ()):
            stale.add_metric([item["role"], item["entity"]], 0.0 if item.get("fresh") else 1.0)
        yield stale

        yield gauge(
            "price_horizon_seconds",
            "Seconds from now to the end of the last published price",
            snap.get("price_horizon_s"),
        )
        yield gauge(
            "prices_tomorrow_published",
            "1 when next-day PVPC prices are present",
            snap.get("tomorrow_published"),
        )
        yield gauge(
            "plan_age_seconds",
            "Seconds since the active plan was generated",
            snap.get("plan_age_s"),
        )
        yield gauge(
            "plan_horizon_slots", "Slots covered by the latest plan", snap.get("plan_slots")
        )
        yield gauge(
            "current_intent_power_watts",
            "Shadow intent for the current slot: + charge, - discharge (never sent)",
            snap.get("current_intent_w"),
        )
        yield gauge(
            "current_intent_active",
            "1 when a valid shadow intent covers now",
            snap.get("current_intent_active"),
        )
        yield gauge(
            "solve_in_progress", "1 while the planner worker is busy", snap.get("worker_busy")
        )
        yield gauge(
            "history_usable_slots",
            "Usable gross-load slots in the forecast history",
            snap.get("history_usable_slots"),
        )
        yield gauge(
            "history_age_seconds",
            "Seconds since the load history was fetched",
            snap.get("history_age_s"),
        )
        yield gauge(
            "terminal_value_eur_per_kwh",
            "EUR/DC-kWh value of end-of-horizon SOC used by the planner",
            snap.get("terminal_value"),
        )
        yield gauge(
            "virtual_battery_soc_ratio",
            "SOC of the open-loop virtual (counterfactual) battery",
            snap.get("virtual_soc"),
        )
        yield gauge(
            "observer_late_samples",
            "Samples that arrived after their slot had been closed",
            snap.get("late_samples"),
        )
        climate = snap.get("climate_shadow")
        if climate is not None:
            yield from _climate_families(climate)


#: Values of the ``status`` label of the gate state set.
GATE_STATUSES = ("PASS", "FAIL", "UNDETERMINED")
CLIMATE_REGIMES = ("legacy", "comfort", "comfort_off", "unknown")
CLIMATE_SOLVE_STATUSES = ("optimal", "gap", "time_limit", "infeasible", "error")


def _climate_families(c: dict[str, Any]) -> Iterable[Metric]:
    """The climate shadow's scrape-time gauges (prefix ``battery_optimizer_climate_shadow_``)."""
    name = f"{PREFIX}_climate_shadow"

    def family(suffix: str, doc: str, labels: list[str]) -> GaugeMetricFamily:
        return GaugeMetricFamily(f"{name}_{suffix}", doc, labels=labels)

    up = family("up", "1 when the job is loaded (configs, prior, site file)", [])
    up.add_metric([], 0.0 if c.get("error") else 1.0)
    yield up
    for suffix, key, doc in (
        ("last_run_timestamp_seconds", "last_run_at", "Unix time of the last decision run"),
        ("last_refit_timestamp_seconds", "last_refit_at", "Unix time of the last weekly refit"),
    ):
        g = family(suffix, doc, [])
        if c.get(key) is not None:
            g.add_metric([], c[key].timestamp())
        yield g
    regime = family("regime", "1 for the control regime recorded at the last run", ["regime"])
    current = (c.get("last") or {}).get("regime")
    for r in CLIMATE_REGIMES:
        regime.add_metric([r], 1.0 if r == current else 0.0)
    yield regime
    queue = family("scoring_queue", "Decisions waiting for their horizon to be observed", [])
    queue.add_metric([], float(c.get("queue") or 0))
    yield queue
    top_last = family(
        "last_plan_top_offset_share_ratio",
        "Share of the last plan's heating slots at the top water offset",
        [],
    )
    if (c.get("last") or {}).get("plan_top_offset_share") is not None:
        top_last.add_metric([], float(c["last"]["plan_top_offset_share"]))
    yield top_last

    report = c.get("report") or {}
    windows = report.get("windows") or {}
    rmse = family(
        "rmse_kelvin", "Room-mean RMSE of the open-loop prediction", ["window", "horizon"]
    )
    skill = family("skill_ratio", "1 - RMSE(model) / RMSE(persistence)", ["window", "horizon"])
    bias = family("abs_bias_kelvin", "Absolute mean room-mean error", ["window", "horizon"])
    n = family("origins", "Scored forecast origins on valid heating days", ["window", "horizon"])
    mdape = family(
        "daily_heat_mdape_ratio", "Median absolute percentage error of daily heat", ["window"]
    )
    weeks = family("valid_weeks", "Valid heating days in the window / 7", ["window"])
    top = family(
        "top_offset_share_ratio", "Share of planned heating slots at the top offset", ["window"]
    )
    gate = family(
        "gate",
        "Preregistered section-A criterion status (state set; criterion verdict = the window)",
        ["window", "criterion", "status"],
    )
    for w, win in windows.items():
        for h, st in (win.get("horizons") or {}).items():
            n.add_metric([w, h], float(st.get("n") or 0))
            for fam, value in ((rmse, st.get("rmse")), (skill, st.get("skill"))):
                if value is not None:
                    fam.add_metric([w, h], float(value))
            if st.get("bias") is not None:
                bias.add_metric([w, h], abs(float(st["bias"])))
        if (win.get("daily_heat") or {}).get("mdape") is not None:
            mdape.add_metric([w], float(win["daily_heat"]["mdape"]))
        weeks.add_metric([w], float(win.get("valid_weeks") or 0.0))
        if (win.get("plan") or {}).get("top_offset_share") is not None:
            top.add_metric([w], float(win["plan"]["top_offset_share"]))
        for criterion, status in {**win.get("criteria", {}), "verdict": win.get("verdict")}.items():
            for st in GATE_STATUSES:
                gate.add_metric([w, criterion, st], 1.0 if st == status else 0.0)
    yield from (rmse, skill, bias, n, mdape, weeks, top, gate)


class ServiceMetrics:
    def __init__(self, snapshot: Callable[[], dict[str, Any]] | None = None) -> None:
        # ``*_created`` series double the exposition for no use here (process-wide switch).
        disable_created_metrics()
        self.registry = CollectorRegistry(auto_describe=False)
        r = self.registry
        self.build_info = Gauge(
            f"{PREFIX}_build_info",
            "Build and configuration identity (value is always 1)",
            ["code_version", "version", "soc_source", "mode"],
            registry=r,
        )
        self.solve_duration = Histogram(
            f"{PREFIX}_solve_duration_seconds",
            "Wall time of planner solves (successful ones)",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
            registry=r,
        )
        self.solves = Counter(f"{PREFIX}_solves", "Planner runs by result", ["result"], registry=r)
        self.skipped = Counter(
            f"{PREFIX}_skipped_solves",
            "Solve requests not started, by reason",
            ["reason"],
            registry=r,
        )
        self.history_refresh = Counter(
            f"{PREFIX}_history_refresh",
            "Prometheus history refreshes by result",
            ["result"],
            registry=r,
        )
        self.sink_errors = Counter(
            f"{PREFIX}_sink_write_errors", "Failed shadow-store writes", registry=r
        )
        self.pvpc_snapshots = Counter(
            f"{PREFIX}_pvpc_snapshots",
            "PVPC publication snapshots by result (written, failed)",
            ["result"],
            registry=r,
        )
        self.weather_forecast_fetches = Counter(
            f"{PREFIX}_weather_forecast_fetches",
            "weather.get_forecasts archive fetches by result (ok = fetched and written)",
            ["result"],
            registry=r,
        )
        # Set only on a success, so the series is absent until the first one.
        self.archive_last_success = Gauge(
            f"{PREFIX}_archive_last_success_timestamp_seconds",
            "Unix time of the last successful archive write",
            ["archive"],
            registry=r,
        )
        self.rest_polls = Counter(
            f"{PREFIX}_ha_rest_polls", "REST GET refreshes by result", ["result"], registry=r
        )
        self.shadow_slots = Counter(
            f"{PREFIX}_shadow_slots", "Closed slots by decision outcome", ["decision"], registry=r
        )
        self.shadow_planned_energy = Counter(
            f"{PREFIX}_shadow_planned_energy_kwh",
            "Energy the shadow decisions requested",
            ["direction"],
            registry=r,
        )
        self.shadow_actual_energy = Counter(
            f"{PREFIX}_shadow_actual_battery_energy_kwh",
            "Battery energy the native controller actually moved",
            ["direction"],
            registry=r,
        )
        # Gauges (cumulative since process start): PVPC can be negative, so a running
        # cost can fall, which a Counter cannot express.  The report is authoritative.
        self.shadow_native_cost = Gauge(
            f"{PREFIX}_shadow_native_cost_eur",
            "Pretax PVPC cost of the measured grid import over comparable slots since start",
            registry=r,
        )
        self.shadow_cf_cost = Gauge(
            f"{PREFIX}_shadow_counterfactual_cost_eur",
            "Pretax PVPC cost had the shadow decisions run (open-loop replay) since start",
            registry=r,
        )
        self.shadow_comparable = Counter(
            f"{PREFIX}_shadow_comparable_slots",
            "Slots with price, measured load and a replayable decision outcome",
            registry=r,
        )
        self.shadow_soc_violations = Counter(
            f"{PREFIX}_shadow_soc_violations",
            "Decisions that would have left the SOC window from the measured SOC",
            registry=r,
        )
        self.shadow_clipped = Counter(
            f"{PREFIX}_shadow_clipped_energy_kwh",
            "Requested energy the replay had to clip (SOC, power, import ceiling, no export)",
            registry=r,
        )
        self.climate_runs = Counter(
            f"{PREFIX}_climate_shadow_runs",
            "Climate shadow decision runs by outcome",
            ["result"],
            registry=r,
        )
        self.climate_solves = Counter(
            f"{PREFIX}_climate_shadow_solves",
            "Comfort-MPC MILP solves by status",
            ["status"],
            registry=r,
        )
        self.climate_solve_duration = Histogram(
            f"{PREFIX}_climate_shadow_solve_duration_seconds",
            "Wall time of one comfort-MPC MILP solve",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
            registry=r,
        )
        self.climate_scores = Counter(
            f"{PREFIX}_climate_shadow_scores",
            "Forecast origins leaving the scoring queue by outcome",
            ["result"],
            registry=r,
        )
        self.climate_refits = Counter(
            f"{PREFIX}_climate_shadow_refits",
            "Weekly re-estimations by outcome (refitted, prior_kept)",
            ["result"],
            registry=r,
        )
        for result in ("planned", "plan_failed", "no_state", "no_inputs", "error"):
            self.climate_runs.labels(result=result)
        for status in CLIMATE_SOLVE_STATUSES:
            self.climate_solves.labels(status=status)
        for result in ("scored", "expired", "failed"):
            self.climate_scores.labels(result=result)
        for result in ("refitted", "prior_kept"):
            self.climate_refits.labels(result=result)
        for result in ("written", "failed"):
            self.pvpc_snapshots.labels(result=result)
        for result in ("ok", "failed"):
            self.weather_forecast_fetches.labels(result=result)
        for reason in ("busy", *(r for r in NOT_READY_REASONS if r != "no_recent_plan")):
            self.skipped.labels(reason=reason)
        for result in ("ok", "error", "timeout", "discarded"):
            self.solves.labels(result=result)
        for decision in ("intent", "no_plan", "plan_expired", "not_ready", "not_observed"):
            self.shadow_slots.labels(decision=decision)
        for direction in ("charge", "discharge"):
            self.shadow_planned_energy.labels(direction=direction)
            self.shadow_actual_energy.labels(direction=direction)
        if snapshot is not None:
            r.register(_RuntimeCollector(snapshot))

    def render(self) -> bytes:
        return generate_latest(self.registry)
