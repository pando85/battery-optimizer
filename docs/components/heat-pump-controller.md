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

## Heater facts (import-limit inputs)

From the Panasonic service manual (indoor WH-ADC0309J3E5 with outdoor WH-UD09JE5 documented; the owner's
WH-ADC0309JE5C and WH-UD09JE5-1 are assumed equivalent). Page details in the homelab
`docs/user-guide/heat-pump-water-temperature.md`.

- One integrated 3.00 kW electric heater (13.0 A) on its own supply (own 16 A breaker); backup for space
  heating and, with the tank heater set to Internal, the DHW booster (p.25-26, 138, 146).
- Heat pump alone at most 3.47 kW (about 2.0 kW rated at A7W35); **worst case with the heater is about
  6.5 kW**, above the 5.0 kW import limit, which the smart meter enforces by cutting supply (cuts have
  happened).
- Automatic starts: space heating below the outdoor "O/D temp for heater ON" (default 0 degC, live reading
  0 degC; p.144); defrost if the backup heater is enabled (p.145); hot water when the tank target exceeds the
  heat pump reach, about 51 to 55 degC (p.26, 107, 137-138); the **weekly sterilization**, default 65 degC for
  10 min, a scheduled 3 kW load (p.150); anti-freeze below 6 degC (p.152); a forced heater on a heat pump
  error only if "Force Heater" = Auto (default Manual; p.144, 147-148). SMART DHW (default 20:00 to 05:00)
  can suppress reheat (p.152).
- The current 49 degC tank target (2026-10-30) is within heat pump range. Any DHW target above about
  51 degC starts the heater.
- Power-limit inputs exist (demand control 0-10 V, external compressor switch, SG-ready, CN-THERMO; most need
  the optional CZ-NS4P board; p.92-93, 98, 103, 159-163). The manual does not say demand control limits the
  heater. Whether any is fitted is unknown.
- Observed (read-only, 2023-11-08 to 2026-10-25): no heater state or heater power signal is exported, and the
  tank temperature never exceeded 51 degC, so no 65 degC sterilization or heater-boosted hot water is visible;
  the outdoor temperature was below 0 degC on only 11 days (minimum -3 degC).
- HeishaMon observation of 2026-10-30 ([regime log](regime-log.md)): the legionella/sterilization function is
  **off** (`Sterilization_State` 0; the configured 65 degC / 10 min is unused; the tank never exceeded 51 degC
  from 2023-11 to 2026-10); DHW heater lifetime hours **0**; room/backup heater lifetime hours **29**; the
  heater-on outdoor temperature is 0 degC. So the booster has never run, and the backup heater has run for
  space heating or defrost for 29 h in total, over an unknown period and not attributable to a day.
- **Placeholder (entity ids to follow):** read-only HA sensors for the heater, sterilization and `Heating_Mode`
  are being added in the homelab repository. Entity ids: `TODO`. Until they exist the heater stays
  unobserved at 15-minute resolution.

## Calibrated on

Not fitted: the rules and `min_hours` were chosen by the owner. What exists is the observed behaviour:

- Legacy rule: hours below two thirds of the trailing mean price, else the cheapest `min_hours`; hot water
  forced in the cheapest hour of each 12 h block. Versions `climate_control@5af36d49`,
  `dhw_control@5af36d49` ([regime log](regime-log.md)).
- Observed regimes: S2 11 h; S3 10 h then 15 h from 2026-01-02. Water target 28 then 26 degC; setpoints
  23.0 in S3. Feature set `da6d678016ed`.
- Comfort history, hall-kitchen (local time, whole winters Nov to Mar, slots with a reading; S1 coverage 92%,
  S2 100%, S3 97 to 98%). Temperatures in degC:

  | | S1 | S2 | S3 |
  |---|---|---|---|
  | Setpoint mean (weekday 12 to 22) | 24.0 | 24.8 | 23.0 |
  | Weekday 12 to 22: mean / p10 / p90 | 23.4 / 22.0 / 24.4 | 23.4 / 22.4 / 24.5 | 22.2 / 21.0 / 23.2 |
  | Weekday 12 to 22: share at or above 23.5 / 25 | 56% / 5.0% | 47% / 0.7% | 2.9% / 0.2% |
  | Weekend 09 to 22: mean, share at or above 23.5 | 23.7, 71% | 23.6, 59% | 22.2, 4.5% |
  | Night 00 to 07: mean | 23.1 | 22.9 | 21.7 |
  | Share of time below 21: bedroom / hall-kitchen | 6.4% / 5.2% | 2.5% / 1.5% | 15.5% / 12.3% |
  | Bedroom p10 | 21.5 | 21.8 | 20.4 |

  The 23.5 degC target of the comfort schedule was met about half the weekday afternoons under the S1 and
  S2 setpoints and almost never under S3 (setpoint 23.0), where the bedroom spent 15.5% of the winter below
  21 degC. Reproduce: `uv run python hack/comfort_history.py --feature-set da6d678016ed
  --data-dir <store>` (writes `<store>/analysis/v1/comfort-history/<run-id>/`).
- Comfort evidence is an **association**: 15 h days against 10 h days in S3 show about 12% against 39% of
  time below 22 degC for +16% heating electricity ([climate plan](../planning/climate-optimizer-implementation.md),
  [replay](../planning/system-replay.md)). Months apart, so confounded.

## Assumptions

- Mode command path `input_select.heishamon_heatmode` is reliable (years of operation, owner decision).
- The pump runs a fixed direct water target; the stored heating curve is inactive
  ([C7-F](../planning/climate-flow-calibration-optimization.md)).
- Setpoints are manual and high enough that comfort-room valves stay open.
- The heater stays off in normal operation (hot water at or below about 50 degC, outdoor above 0 degC); the
  weekly sterilization, the one scheduled exception, is observed off on 2026-10-30 (the facts above).
- The comfort schedule ([ADR-0006](../adr/0006-decouple-heat-pump-comfort-from-battery-price-optimisation.md): hall-kitchen at least 23.5 degC from 12:00 on weekdays and from 09:00
  at weekends, all rooms at least 21 degC) can be met by optimum start with the fewest kWh. Untested:
  the schedule shape was never replayed.

## Invalidated when

- Any change of rule version, `min_hours`, enable flag, water target, curve or setpoint (log it first).
  A changed hour selection also needs a matching `structured_forecast.py` replay update.
- The comfort schedule goes live, the legacy rule or the `dhw_control` forcing is retired, or boost
  starts: the load shape changes for every consumer listed above.
- The heater facts change (tank target above about 51 degC, sterilization or "Force Heater" enabled, a
  power-limit input fitted, a model difference from the manual).
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
