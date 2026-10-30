# Regime log

Dated log of real plant and control changes. Each row says what changed, the evidence for the date and
which component pages it touches. **Add a row (or turn a `planned` row into a dated one) before anything
else when the house or a control rule changes**; see [how to use these pages](README.md#how-to-use-these-pages).

How the dates were derived (2026-10-30), so they can be re-checked:

- **data**: from the climate feature set `da6d678016ed` (window 2023-11-07 to 2026-05-01, 15-minute
  slots) in the pulled cluster store, read only. Water target = median `flow_target_temp` over slots with
  heat-mode compressor energy; setpoints = mean `*_setpoint` of the three comfort rooms (bedroom,
  hall-kitchen, office T); `min_hours` and the enable flag = value at the start of the slot.
- **code**: the rule-version tables in `structured_forecast.py` (`CLIMATE_RULE_VERSIONS`,
  `DHW_RULE_VERSIONS`), which mirror the homelab commits named there.
- **doc**: the [climate plan](../planning/climate-optimizer-implementation.md) and the homelab page
  `docs/user-guide/heat-pump-water-temperature.md` (audit of 2026-10-30).
- **unverified**: no evidence in this repository; the entry says which bounds are known.

Seasons are those of [`config/climate.yaml`](../../config/climate.yaml): S1 2023-11-08 to 2024-04-30,
S2 2024-10-01 to 2025-04-30, S3 2025-10-01 to 2026-04-30.

## Log

| Date (local) | Change | Evidence | Touches |
|---|---|---|---|
| 2023-11-08 | Start of the `hass_*` history at 1-minute cadence (S1 begins). The direct water-temperature mode is verified only for 2026-10-30 and its 90-day history; the mode in earlier seasons is **unverified** (a constant target per season is consistent with direct mode) | doc | data-pipeline |
| 2023-11-11 | Water target 35 degC (S1 regime; thermostat setpoints 24.0 hall-kitchen, 23.7 mean of the three comfort rooms) | data | cop-model, thermal-model |
| 2023-11-25 | `climate_control` first release: heat only in hours priced below two thirds of the trailing mean, else the cheapest `min_hours` hours (`climate_control@fda18eb2`) | code | heat-pump-controller, battery-load-forecast |
| 2023-11-26 to 2023-12-31 | `min_hours` tuned by hand (5, 7, 9, 6), then 11 from 2023-12-31 (S1 and S2 run 11 h) | data | heat-pump-controller |
| 2024-04-27 to 2024-11-08 | Water target 35 -> 28 degC. **Date unverified**: the last 35 degC heating slot is 2024-04-27, the first 28 degC one is 2024-11-09, and the pump only made hot water in between | data (bounds only) | cop-model, thermal-model |
| 2024-11-09 | S2 heating starts, `min_hours` 11 for all 147 heating days up to 2025-04-23; setpoints 24.8 hall-kitchen in the gate months, 24.4 mean of the three rooms | data | all climate components |
| 2025-02-20 | Water target 28 -> 26 degC (exact day; 28 on 2025-02-19). Two isolated days at 32 to 34 degC on 2025-03-22/23 are excursions, not a regime | data | cop-model, thermal-model |
| 2025-03-16 | `dhw_control` first release: one forced hot-water hour per day (`dhw_control@363047d3`) | code | heat-pump-controller |
| 2025-04-01 | Thermostat setpoints fall from 24.5 to 25.5 (hall-kitchen 25.5) to 23.0 in every room and stay at 23.0 (S3 regime). A temporary 24 to 25 in 2025-08-25 to 2025-09-24 is off-season | data | thermal-model, cop-model |
| 2025-09-22 to 2025-11-03 | Legacy rule disabled; S3 heating starts 2025-11-04 at `min_hours` 10 (`min_hours` 8 before, 6 for the first two days) | data | heat-pump-controller |
| 2025-12-29 | `dhw_control` interval 24 h -> 12 h (`dhw_control@153d7f88`) | code | heat-pump-controller |
| 2026-01-01 23:30 | `min_hours` 10 -> 15 (first slot at 15; the rule decision of 2026-01-02 uses it): 53 heating days at 10 h, then 84 heating days at 15 h up to 2026-04-03 | data | heat-pump-controller, battery-load-forecast |
| 2026-02-09 to 2026-02-12 | Thermostat outage (excluded by `filters.thermostat_outage`) | data | thermal-model |
| 2026-04-07/08 | `min_hours` 15 -> 10; rule disabled from 2026-04-08, enabled again 2026-04-22 (last day in the feature set: 2026-04-30) | data | heat-pump-controller |
| 2026-04-28 | `climate_control@e734f5e9`: history window 10 -> 7 days with retries | code | heat-pump-controller |
| 2026-09-05 | `climate_control@5af36d49` and `dhw_control@5af36d49`: negative prices accepted | code | heat-pump-controller |
| 2026-10-xx | C0 AppDaemon fixes (homelab): the enable toggle cancels timers and restores the heat mode, a heartbeat watchdog restores it after 15 minutes, the price-retry loop re-checks the toggle, all nine heat modes map to `SetOperationMode`, numeric sensors for operating mode, defrost and 3-way valve. Legacy hour selection unchanged. **Merge and deploy date unverified** (homelab files last modified 2026-10-30) | doc | heat-pump-controller, data-pipeline |
| 2026-10-30 | Template room-temperature sensors and `hass_climate_action` (real valve demand) start to exist; earlier demand must be inferred. Pump verified in direct mode, water target 26 degC, space heating off (hot water only) | doc | data-pipeline, thermal-model |
| 2026-10-30 | Decoupled design agreed (ADR-0006): the heat pump does comfort with the fewest kWh and ignores price, the battery does price arbitrage | doc | all |

## Planned (not yet real: dates to be filled in)

| Planned change | Components that go stale when it happens | Notes |
|---|---|---|
| **Comfort schedule live**: hall-kitchen at least 23.5 degC from 12:00 on weekdays and from 09:00 at weekends, every room at least 21 degC always; optimum start | battery-load-forecast, heat-pump-controller, hp-consumption-forecast, system-replay-benchmark | Start the 28-day settling clock here |
| **Legacy cheapest-hours rule retired**, `dhw_control` forcing retired (one hot-water writer) | hp-consumption-forecast (LF-1), battery-load-forecast | Never run two price reactors |
| **Boost mode live** for very cheap hours (rare in winter) | boost-mode, battery-load-forecast | Interlock reads battery SOC |
| **Battery installed and go-live** through OmniBattery driven by this optimizer | data-pipeline, battery-model, battery-optimizer, battery-load-forecast | `ch1` becomes net grid; commissioning measures RTE and idle draw |
| **Heating-curve change / water-temperature trial** (C7-F field trial arms) | cop-model, thermal-model, battery-load-forecast | Alternating-arm weeks are excluded from forecast and gate windows |
