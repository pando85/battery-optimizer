# Load Forecaster — component roadmap

- **Status (2026-10-25):** V0 implemented; enhancements proposed, not implemented.
- **Domain responsibility:** forecast **gross household demand** in kWh per aligned 15-minute UTC slot, as information existed when a decision was made.
- **Current code:** `src/battery_optimizer/load_forecast.py` (forecaster registry and `run_forecaster` seam: `trailing_mean` = LF-0, `perfect` = offline oracle), `src/battery_optimizer/service/history.py` (`LoadForecast`, `forecast_load`), `src/battery_optimizer/policies.py` (`forecast_load_trailing_slot_mean`), `src/battery_optimizer/rolling_horizon.py` (causal/frozen offline evaluation), `src/battery_optimizer/ha_states.py` (as-of appliance-state reader).
- **Architecture:** [four DECIDE components](../../architecture.md#planning-components); [ADR-0002](../../adr/0002-appliance-aware-load-forecasting.md); [implementation TODO-3.1–3.9](../home-battery-optimizer-implementation.md#6-phase-3--predict-demand-only-if-it-creates-savings).

## Contract and non-goals

**Input:** immutable decision timestamp, slot horizon, historical measurements and their quality mask, known calendar/schedules, and only appliance attributes actually available at that timestamp. **Output:** aligned load forecast (and eventually uncertainty), forecast provenance, missing-data/coverage flags, and input-availability cutoff. Production currently uses `LoadForecast`. The history-cutoff and provenance obligations are implemented by the seam in `load_forecast.py` (`run_forecaster`); the rest remain **logical obligations**.

Do not conflate household gross demand with signed grid import after the battery starts operating: when charge/discharge energy is measured, reconstruct gross load with the correctly signed battery flows. Do not treat unknown samples as zero, leak tomorrow's observed appliance state into today's prediction, infer appliance consumption directly from aggregate power without evidence, or send commands to HVAC/appliances.

## Milestones

| Stage | Status | Work / deliverable | Exit evidence |
|---|---|---|---|
| **LF-0 — Causal baseline** | **Built** | Historical trailing 28-day same-slot mean with decision-local-midnight cutoff; compatible offline and shadow code path. | Golden/causal comparisons remain reproducible; missing history blocks rather than silently producing a zero-load forecast. |
| **LF-1 — Observable deterministic features** | **Next, offline first** | Build an *as-of* feature audit for the roughly 250 W server base load (**owner estimate, verify by Shelly/Prometheus**), weekday/holiday patterns, and AppDaemon's known aerothermal/DHW schedules. Compare both whole-home feature models and properly trained residuals; prevent double-counting scheduled demand. | Feature provenance and availability documented; chronological holdout comparison against LF-0 on identical data/price assumptions; no worse physical feasibility. |
| **LF-2 — Appliance-aware aggregate forecast** | **Candidate** | Reuse HA program, phase, on/off, elapsed/remaining-time states for washing machine, dishwasher and other observable devices. Start with one aggregate load model conditioned on *currently known* appliance states, including uncertainty about future starts. With trustworthy labels, optionally learn per-program duration/remaining-energy profiles. | Per-appliance and whole-home ablations, session-disjoint validation when appropriate, interval coverage/error reported for 15 min, 1 h, 6 h and 24 h; no retrospective future-state leakage. |
| **LF-3 — Statistical/ML challengers** | **Optional, evidence-gated** | Compare feature-based baselines with LightGBM/CatBoost; add P10/P50/P90 **calibrated conditional quantiles** for uncertain loads. NILM/HSMM only for *unobserved* appliances when reliable ground truth exists and benefit justifies maintenance. | Improvement survives independent chronological folds/seasons and end-to-end battery-dispatch evaluation. If not, retain the simple model. |
| **LF-4 — Production promotion** | **Blocked pending evidence** | Run new forecasts beside LF-0 in shadow; version features/model, track drift, missing states, fallback and decision-time latency; rollback without changing MPC internals. | Pre-agreed economic and reliability gates pass; safe fallback to LF-0 on stale inputs, low confidence, or model errors. |

## Metrics and experiment design

1. Report **MAE/WAPE** and calibration/coverage of P10/P50/P90, *with* sample counts and horizon-specific breakdowns (including expensive periods, weekdays/weekends and DST days).
2. Primary promotion metric: **held-out net EUR dispatch saving / regret relative to an identical baseline**. Keep the same tariffs, battery physics, SOC seed/terminal valuation and solver across comparisons; do not choose based on MAE alone.
3. Record feature availability **as of** each decision (including HA recorder timestamps, program transitions and published schedule changes). Use walk-forward training and no overlap/leakage across labeled appliance cycles.
4. Run ablations in order: LF-0 → +verified server baseline → +known AppDaemon schedules → +appliance states → +ML → +probabilistic forecast. Keep raw and predicted series private.

## Dependencies, boundaries and stop rule

- Requires trustworthy SENSE coverage. Until actual battery telemetry exists, historical gross load uses the recorded **battery-absent** assumption; revalidate that calculation after commissioning.
- A model supplying quantiles does **not** require the MPC to become stochastic immediately. Retain a deterministic central forecast until the [Dispatch Optimizer](dispatch-optimizer.md) has a justified uncertainty-handling strategy.
- If added features or ML do not produce reproducible material **net EUR** improvement under a predeclared threshold, stop at the simpler forecaster. This component never controls the battery or appliances.
