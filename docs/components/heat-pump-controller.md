# Heat-pump controller (legacy cheapest hours now, comfort schedule next)

## Purpose and owner

Decides when the Panasonic Aquarea heats rooms and makes hot water. **Owner: the homelab repository**
(AppDaemon `climate_control.py` and `dhw_control.py`, HA helpers, HeishaMon automations); this repository
only observes it and replays it. Water-temperature control, its modes and the commands the unit accepts
are documented in the homelab repository at `docs/user-guide/heat-pump-water-temperature.md`. This
repository never writes to it (shadow phase, [AGENTS.md](../../AGENTS.md)).

## Interfaces

- **In:** the PVPC price sensor, `input_number.appdaemon_climate_min_hours_per_day`, the enable toggles,
  manual thermostat setpoints, and the water-target helper (absolute value, direct mode).
- **Out:** operating mode `Heat+DHW` or `DHW`, `switch.aquarea_force_dhw_mode`; the unit's own water target.
- **Consumers:** the whole-house load that the [battery load forecast](battery-load-forecast.md) learns
  from; [cop-model](cop-model.md) and [thermal-model](thermal-model.md) data; LF-1
  ([hp-consumption-forecast](hp-consumption-forecast.md)) replays it.

## Calibrated on

Not fitted: the rules and `min_hours` were chosen by the owner. What exists is the observed behaviour:

- Legacy rule: hours below two thirds of the trailing mean price, else the cheapest `min_hours`; hot water
  forced in the cheapest hour of each 12 h block. Versions `climate_control@5af36d49`,
  `dhw_control@5af36d49` ([regime log](regime-log.md)).
- Observed regimes: S2 11 h; S3 10 h then 15 h from 2026-01-02. Water target 28 then 26 degC; setpoints
  23.0 in S3. Feature set `da6d678016ed`.
- Comfort evidence is an **association**: 15 h days against 10 h days in S3 show about 12% against 39% of
  time below 22 degC for +16% heating electricity ([climate plan](../planning/climate-optimizer-implementation.md),
  [replay](../planning/system-replay.md)). Months apart, so confounded.

## Assumptions

- Mode command path `input_select.heishamon_heatmode` is reliable (years of operation, owner decision).
- The pump runs a fixed direct water target; the stored heating curve is inactive
  ([C7-F](../planning/climate-flow-calibration-optimization.md)).
- Setpoints are manual and high enough that comfort-room valves stay open.
- The comfort schedule (ADR-0006: hall-kitchen at least 23.5 degC from 12:00 on weekdays and from 09:00
  at weekends, all rooms at least 21 degC) can be met by optimum start with the fewest kWh. Untested:
  the schedule shape was never replayed.

## Invalidated when

- Any change of rule version, `min_hours`, enable flag, water target, curve or setpoint (log it first).
  A changed hour selection also needs a matching `structured_forecast.py` replay update.
- The comfort schedule goes live, the legacy rule or the `dhw_control` forcing is retired, or boost
  starts: the load shape changes for every consumer listed above.
- Two price-reactive writers run at once (legacy rule plus boost plus a battery with native price
  features).

## Recalibrate by

There is no fit; the check is measurement. After a change: wait 28 days, then
`uv run battery-optimizer build-features --signal-set climate` and `uv run python hack/climate_audit.py`
on the new window. Acceptance: all rooms stay at or above 21 degC, the hall-kitchen reaches 23.5 degC at
the scheduled time, and kWh per heating degree-day and Kh below 22 degC are no worse than the 15 h days
(replay: about 0.75 kWh/HDD, about 1.15 Kh/day).

## Status

**valid (legacy rule), being replaced** (2026-10-30). Space heating is off for the season; the comfort
schedule is planned ([regime log](regime-log.md#planned-not-yet-real-dates-to-be-filled-in)).
