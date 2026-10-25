# Battery Model — component roadmap

- **Status (2026-10-25):** V0 numeric physics implemented with **assumed** round-trip efficiency; no calibrated Marstek measurements yet.
- **Domain responsibility:** describe physically feasible charge/discharge, SOC evolution, conversion losses, energy inventory and site import/no-export constraints, **independently of price/load prediction**.
- **Current code:** `src/battery_optimizer/battery.py` (`BatterySpec`, `Horizon` and pure numeric kernel), `src/battery_optimizer/benchmark.py` (assumption resolution), `src/battery_optimizer/service/virtual_battery.py` (shadow replay), `src/battery_optimizer/service/planner.py` (spec assembly).
- **Architecture:** [four DECIDE components](../../architecture.md#planning-components); [physical constraints](../../adr/0001-home-battery-economic-dispatch.md); [production integration gate](../home-battery-optimizer-implementation.md#16-production-packaging-and-runtime-rollout).

## Contract and non-goals

**Input:** device capabilities, measured SOC when available, elapsed slot duration, charge/discharge flows, site limits and calibrated parameters. **Output:** feasible SOC/flow constraints and predicted AC energy/round-trip losses, shareable by backtest, MPC and shadow replay. The model is a **physical constraint**, *not* a forecast of prices or load and not an Omnibattery controller.

Existing physics enforces energy-unit consistency, one active direction per slot, SOC and power limits, a `grid = gross_load + charge - discharge` balance, **no intentional export**, and charging constrained by configured site import headroom. It assumes an **RTE of 0.80** (split into one-way efficiencies), not a measurement of the user's device. Keep that assumption visible in every evaluation.

## Milestones

| Stage | Status | Work / deliverable | Exit evidence |
|---|---|---|---|
| **BM-0 — Feasible scalar physics** | **Built** | Deterministic AC-side balance, power/SOC limits and constant-efficiency model reused in offline benchmark, planning and replay. | Existing conservation, non-simultaneous flow, no-export and import-ceiling golden/property tests remain green. |
| **BM-1 — Real telemetry and instrumentation** | **Hardware-gated: next** | Install/configure Marstek Venus E/Omnibattery in HA **read-only first**; map SOC, AC charge/discharge power and cumulative energy counters; validate Shelly CT topology/sign, timestamps, meter offsets and missing-data semantics. | Comparable meter/battery energy series with coverage and sign audit; `soc_source: entity` only after a fresh, consistent snapshot. No actuation by optimizer. |
| **BM-2 — Bench-calibrated losses** | **Hardware-gated** | Measure charge/discharge AC energy, energy stored/retrieved, actual capacity, different power setpoints and SOC regions, **standby draw**, and effect of AC bypass/BACKUP (including the intended ~250 W server load, subject to physical verification). Separate metering/auxiliary loads from converter losses. | Measured uncertainty bands for each curve; independent held-out SOC/energy/efficiency residuals improve on BM-0, including slow/partial cycles. |
| **BM-3 — Solver-compatible physical upgrades** | **Candidate** | Introduce simple calibrated one-way/piecewise efficiency and idle-loss/auxiliary terms only where needed; evaluate observed power ramp/deadband, SOC boundaries and thermal derating. Prefer linear/MILP-compatible formulations, or an explicit approximation with error bounds. | Equivalence at original scalar settings, conservation at each timestep, monotonic feasibility and bounded HiGHS solve time; physical behavior no worse in held-out tests. |
| **BM-4 — Degradation and backup constraints** | **Later, evidence-gated** | Estimate lifetime/throughput degradation assumptions separately from conversion losses; represent reserve SOC for backup **only after EPS/BACKUP transfer and loads are verified**. Calibrate economics without conflating battery aging and energy losses. | Realistic model supported by measurements/vendor information; backup reserve has an explicit objective and tested behavior, never a claimed UPS guarantee without tests. |

## Measurement and validation protocol

1. Collect synchronized **SOC**, signed AC power, charging/discharging energy counters, Shelly grid/load and known Omnibattery mode; mark actual start/end timestamps and any discontinuity.
2. Compare multiple power/SOC bands and partial/full cycles. SOC percentages alone are not direct energy measurements: account for SOC quantization, battery's hidden reserve and auxiliary consumption; do not infer round-trip efficiency from an unbalanced partial cycle.
3. Cross-validate model predictions against **held-out physical charge/discharge intervals**, not only against a synthetic battery trajectory; report Wh error, SOC error, RTE uncertainty and feasibility violations by power band.
4. Run the same calibrated physics through offline backtesting and shadow before attempting physical control. Reject a more complex model if its measured error improvement is negligible or it creates pathological/infeasible MILP plans.

## Dependencies, boundaries and stop rule

- **BM-1/BM-2 require physical commissioning.** HA and inverter safety protections are not guaranteed by this software, and `soc_source: simulated` is pre-commissioning only.
- **Safety is not reducible to a battery simulation:** verify Omnibattery's response, power-loss behavior, grid export transients and BACKUP/AC bypass separately with the [live ACT gate](../home-battery-optimizer-implementation.md#16-production-packaging-and-runtime-rollout).
- Stop at the simplest physically adequate model; do not add temperature, aging or a nonlinear battery digital twin absent measured value.
