# Dispatch Optimizer — component roadmap

- **Status (2026-10-25):** V0 receding-horizon MPC built and running in **shadow only**; no physical writer/safety guard.
- **Domain responsibility:** choose economically useful **and physically feasible** charging/discharging over a horizon from independently supplied **Load Forecaster**, **Price Outlook**, **Battery Model**, measured state, and objective parameters.
- **Current code:** `src/battery_optimizer/service/planner.py` (`PlanningSnapshot`, `run_planner`), `src/battery_optimizer/oracle.py` (shared HiGHS solver), `src/battery_optimizer/rolling_horizon.py` (offline replan/policy), `src/battery_optimizer/dispatch.py` (`DispatchPlan`, `DispatchIntent`), `src/battery_optimizer/economic_sensitivity.py` (parameter sweeps).
- **Architecture:** [four DECIDE components](../../architecture.md#planning-components); [ADR-0001](../../adr/0001-home-battery-economic-dispatch.md); [production and ACT gate](../home-battery-optimizer-implementation.md#16-production-packaging-and-runtime-rollout).

## Contract and non-goals

**Input:** frozen and timestamped load curve, published/estimated price curve with availability metadata, battery physical constraints, current SOC, import ceiling, plan horizon, terminal-value/degradation assumptions and source quality. **Output:** a bounded-time, validated `DispatchPlan` and expiring `DispatchIntent` values with input hash, solver provenance and generation ordering.

The MPC's decisions are **not device setpoints**. Only a separately authorized, safety-tested HA/Omnibattery layer could ever apply them; the optimizer must never compete for direct Modbus ownership or replace Omnibattery's fast Shelly feedback.

## Milestones

| Stage | Status | Work / deliverable | Exit evidence |
|---|---|---|---|
| **DO-0 — Causal offline + shadow MPC** | **Built** | Shared solver/physics, five-policy ladder (`no_battery`, `causal_price`, `day_ahead`, `rolling_horizon`, oracle), causal forecast alignment, 15-minute service replanning, early solve, trigger debouncing, one in-flight worker, stale-plan invalidation and expiring intents. | Reproducible chronological goldens, no simultaneous solver work, stale inputs/expired plans produce no actionable intent, shadow store and metrics work. |
| **DO-1 — Economic accounting and objective calibration** | **Next; offline and shadow** | Audit PVPC pretax vs billed variable components, losses and SOC terminal inventory. Run existing throughput/terminal-value sensitivity tooling; keep `throughput_cost_eur_per_kwh = 0` baseline until degradation assumptions are substantiated. | Like-for-like net EUR savings and regret against existing policies on matched windows; physical RTE/limits kept explicit; no double-counting degradation vs energy losses. |
| **DO-2 — Better constraints and replanning policy** | **Candidate** | Use measured battery/Shelly limits, chosen import headroom and SOC reserve; test significant-change replans (price publication, SOC, scheduled loads) vs 15-minute cadence, balancing compute/churn. Model configured reserve for critical backup loads **only after wiring/EPS is characterized**. | Power/SOC/import/no-export invariants, solver latency and number of solves, economic effect measured in chronological replay and read-only shadow. |
| **DO-3 — Uncertainty-aware planning** | **Optional, after forecast evidence** | Consume uncertainty from [Load Forecaster](load-forecaster.md) and estimated **unpublished** prices from [Price Outlook](price-outlook.md), when available; compare conservative deterministic decisions, correlated multi-hour scenarios and robust/chance/CVaR-like formulations. | Reliable uncertainty calibration, feasible scenarios and improved **held-out risk-adjusted net EUR** after compute cost; preserve exact prices whenever published. |
| **DO-4 — Longitudinal shadow acceptance** | **Battery-dependent** | Compare proposed intents vs physical native behavior for at least **two representative weeks**, normalize starting/terminal SOC economics, monitor missing data and decision attribution; inspect candidate changes side by side against DO-0. | Positive reproducible benefit, bounded churn/latency, physical-model confidence and documented anomalies. No physical commands at this stage. |
| **DO-5 — ACT handoff** | **Explicitly not authorized** | Only after separate ADR/runbook and hardware fault tests: design high-level bounded intent translation, ownership, reconciliation, verified safe idle/rollback and HA-loss behavior while Omnibattery retains PD/Modbus control. | Formal independent safety gate; this milestone is **not** permission to enable live mode. |

## Metrics and experiment design

1. Optimize **billed-variable net EUR savings**, energy throughput, constraint feasibility, plan-versus-actual deviation, battery cycling and terminal SOC valuation; always report what is provisional or excluded (tax, degradation, backup).
2. Compare candidates on the **same** as-of prices, gross load, physical model, initial/terminal SOC assumptions, observation masks and price-publication schedule. Avoid changing forecast quality and planner logic in the same un-attributed experiment.
3. Monitor solver p50/p95/max wall time, timeout/skipped/replanned counts, one-worker occupancy, solver status, predicted vs realized import and plan staleness. All solving must stay off the HA event loop.
4. Gate each upstream model promotion on incremental **policy savings**, not local predictor accuracy. `shadow-report` is a counterfactual estimate with assumptions, not a guaranteed bill saving; there is no valid native-device comparison while SOC is simulated.

## Dependencies, boundaries and stop rule

- DO-1 works today on reproducible offline data; DO-2/DO-4 depend partly on real Marstek/Omnibattery telemetry and validated Shelly sign. DO-3 requires causal upstream uncertainty and honest publication snapshots, not invented future information.
- The fast no-export feedback and inverter safety remain with Omnibattery/hardware. **Loss of Home Assistant cannot be repaired by an in-HA lease**: that is a physical commissioning question, not a solver tweak.
- If extra solver complexity cannot beat the causal baseline in consistent net EUR outcomes without increasing safety/operational risk, do not deploy it. Joint HVAC/DHW actuation remains a separate future ADR, not part of this roadmap.
