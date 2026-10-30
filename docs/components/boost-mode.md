# Boost mode (planned)

## Purpose and owner

In very cheap hours, heat the slab and the hot-water tank beyond the comfort need (a thermal soak), as the
only price-reactive behaviour left on the heat-pump side. Rare in winter, frequent in April to October on
weekends around midday. Decided in [ADR-0006](../adr/0006-decouple-heat-pump-comfort-from-battery-price-optimisation.md); **owner: the homelab repository**
(AppDaemon), with the battery side read-only. Not built.

## Interfaces

- **In (AppDaemon):** the pre-tax PVPC sensor (same basis as the battery, state it in the homelab config),
  the Omnibattery SOC and charge-power sensors read-only, the current comfort and hot-water state.
- **Out:** heat and hot-water commands through the existing AppDaemon path; the hot-water boost target stays
  at or below about 50 degC (heat pump only, see Assumptions); optionally `kind: boost` slots
  in [hp-consumption-forecast](hp-consumption-forecast.md) (not needed: with the interlock the HP waits
  for the battery, so the battery plan for the window is already right).
- **Consumers:** [heat-pump-controller](heat-pump-controller.md); [battery-load-forecast](battery-load-forecast.md)
  sees boost only as smeared load.

## Calibrated on

Nothing. The [whole-system replay](../planning/system-replay.md) did not simulate boost; the S3 load it
used came from 15 h price-chosen days. The ordering argument is arithmetic only: the usable 3.84 kWh
fills in about 1.7 h at 2.5 kW, so a window of 3 h or more holds both.

## Assumptions

- Battery first, then space-heating soak, then hot water last; the order comes from the heat pump
  waiting (soak starts at SOC about 90 to 95%, or charge power below a threshold, or
  `window_end - soak_duration`). Do not expect the battery to yield through grid-target control: it would
  yield to load, so the heat pump would effectively go first.
- Boost and the battery charge land in the same cheapest slots; the sum with the house load must stay under
  the 5.0 kW import limit.
- **The DHW boost must stay at or below about 50 degC (heat pump only).** The Panasonic service manual
  (p.26, 107, 137-138, 146) gives a heat-pump-only ceiling of about 51 to 55 degC (control reference 51 degC;
  "above 55 degC only possible with backup heater"). A target of 55 to 60 degC, as in ADR-0006 decision 3,
  starts the integrated 3.00 kW electric heater (p.25-26, 138, 146): heat pump plus heater is about 6.5 kW
  (heat pump alone at most 3.47 kW), above the 5.0 kW limit on its own and far above it with the battery
  charging. The smart meter cuts supply on overload, which has happened.
- The sterilization cycle (default 65 degC for 10 min, weekly, p.150) also uses the heater, so it is a
  scheduled 3 kW load that boost must not overlap with the battery charge or the soak.
- Legionella and pump hygiene cycles stay authoritative; every other price reactor is off.

## Invalidated when

- The import-limit basis turns out to be an instantaneous trip rather than a 15-minute average.
- The Omnibattery SOC entity, its grid-target semantics or its native price features change.
- The legacy cheapest-hours rule or the `dhw_control` forcing still runs (a second price reactor).
- The heater facts above change (a firmware or model difference from the assumed-equivalent manual, a
  power-limit input fitted, "Force Heater" or the tank heater setting changed), or the hot-water target or
  the soak duration changes; the boost price basis
  differs from the battery's.

## Recalibrate by

Before the first boost, predeclare thresholds and the acceptance check; then measure on the first boost
days (log the heat-pump sub-meter, the battery counters, SOC and the main meter at 15-minute and
instantaneous resolution; [data-pipeline](data-pipeline.md) gross reconstruction must be in place).
Acceptance: no interval above the import limit minus the margin while the battery charges; the soak
starts after the battery reached its SOC trigger or the fallback time; comfort is not degraded the day
after; then rerun [battery-load-forecast](battery-load-forecast.md) with boost days flagged.

## Status

**planned** (2026-10-30): no rule, threshold, interlock or code. Blocked on the import-limit basis and the
retirement of the legacy rule.

**Open adjustment to ADR-0006 decision 3** (not yet made; the ADR text is unchanged): the "55 to 60 degC"
DHW step conflicts with the heater facts above and should become "at or below about 50 degC, heat pump only"
unless the owner accepts the heater with a hard interlock (no heater while the battery charges or any other
large load runs). The booster power is no longer unverified: 3.00 kW, from the service manual.
