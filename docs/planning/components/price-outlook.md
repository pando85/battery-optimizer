# Price Outlook — component roadmap

- **Status (2026-10-25):** V0 published-price adapter built; live estimation of unpublished prices **does not exist**.
- **Domain responsibility:** provide the electricity-price curve **available at decision time**, including publication cutoff and (only in a later phase) clearly marked estimates beyond that cutoff.
- **Current code:** `src/battery_optimizer/service/pvpc.py` (`PriceCurve`, `parse_pvpc_state`); offline publication/price-noise experiments in `src/battery_optimizer/rolling_horizon.py`.
- **Architecture:** [four DECIDE components](../../architecture.md#planning-components); [implementation plan](../home-battery-optimizer-implementation.md); [price experiment records](../energy-baseline-forecast-experiments.md).

## Contract and non-goals

**Input:** decision timestamp, price source/publication timestamps, PVPC tariff components and optional *past* publications of external market data. **Output:** UTC-aligned EUR/kWh prices, interval-specific **published vs estimated vs unavailable** provenance, the *last known published slot*, and assumptions about billing components. The existing `PriceCurve` supports **published** prices only; a richer object is a future requirement, **not an existing API**.

Use published prices verbatim when valid; never overwrite a known value with a model estimate. The live controller currently **ends the MPC horizon at the last published price** and values remaining stored energy through a terminal heuristic; offline simulations of noisy next-day prices do **not** mean production already predicts them. An archived final/realized price is **not proof** of what was published earlier.

## Milestones

| Stage | Status | Work / deliverable | Exit evidence |
|---|---|---|---|
| **PO-0 — Exact published PVPC** | **Built** | Parse `sensor.esios_pvpc` as-of state attributes, turn hourly prices into 15-minute UTC slots, handle Europe/Madrid's 23/25-hour days and the repeated-hour suffix. | No gaps or shifted hours within the published horizon; missing/old publications produce a visible reason, not invented values. |
| **PO-1 — Publication evidence** | **Next, highest priority** | Persist private snapshots of source values, `published_at`/`observed_at`, and decision-time information sets (not just final prices). Compare archived snapshots with subsequent published/realized PVPC and invoice inputs. | Reconstruct exactly which hours and values each historical decision could know; verify or reject the current *published = realized* working assumption. |
| **PO-2 — Early next-day estimate** | **Candidate, offline first** | Investigate OMIE day-ahead market data once available and estimate the remaining regulated PVPC components for hours before the complete next-day PVPC publication (e.g. the afternoon ~14:00–20:30 window). Simple historical/calendar models first; maintain explicit uncertainty and external-data availability timestamps. | Backtest against decision-time snapshots, not hindsight; show improved **net EUR dispatch value** versus published-only MPC and no leakage of final regulated components. |
| **PO-3 — Scenario prices** | **Optional** | Produce P10/P50/P90 or coherent day-level scenarios for **unpublished** hours, preserving temporal correlation and publication-confidence metadata. Feed scenarios to the MPC only after it has an approved scenario interface. | Calibrated interval/day error and decision regret; robust optimizer must not exploit unrealistically independent or perfect future prices. |
| **PO-4 — Shadow promotion** | **Blocked pending evidence** | Run estimate-augmented horizon as a challenger next to published-only; record when extra predicted hours actually changed a decision. On uncertainty/fetch failure, fall back to exact-only horizon and terminal value. | Positive held-out gain after extra churn/cycles; transparent provenance and no worsening of safety constraints. |

## Metrics and experiment design

1. Exact data: freshness, coverage, DST alignment, publication lag, consistency with realized/billed variable tariffs; distinguish pretax energy component from taxes, fees and fixed charges.
2. Estimates: signed bias, MAE and calibration, particularly **hour ordering and spreads** that determine charge/discharge choices, separately before and after actual publication.
3. **Primary metric:** incremental net EUR savings / policy regret on identical historical load forecasts, solver, physical assumptions, terminal valuation and decision timestamps.
4. Persist the exact price information set at decision time to avoid publication-lookahead bias; register evaluation windows and promotion threshold before tuning.

## Dependencies, boundaries and stop rule

- PO-1 depends on a reliable timestamped SENSE archive. The earlier owner's observation that published and realized prices match is **not yet independently verified**.
- PO-2/PO-3 are optional; they only help when acting **before** complete PVPC publication has an economic advantage. Today's exact-only policy remains a supported baseline.
- Do not develop a full price-forecasting service or introduce new APIs until the replay shows decisions worth changing. No electricity market purchases, billing changes or device commands are in scope.
