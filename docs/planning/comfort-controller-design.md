# Comfort controller design (homelab AppDaemon)

- **Status:** Design agreed with the owner on 2026-10-31; **for owner review, not yet implemented**. No physical control is authorised by this document.
- **Date:** 2026-10-31
- **Decisions:** [ADR-0006](../adr/0006-decouple-heat-pump-comfort-from-battery-price-optimisation.md) and its "Amendment 2026-10-31" (this design amends decisions 3 and 4 of the ADR).
- **Parent plans:** [Climate optimizer implementation](climate-optimizer-implementation.md) (section 6 work list), [C7-F field trial](climate-flow-calibration-optimization.md)
- **Facts about the unit:** homelab `docs/user-guide/heat-pump-water-temperature.md` (modes, commands, safety notes); heater facts in [heat-pump-controller](../components/heat-pump-controller.md#heater-facts-import-limit-inputs).
- **Owner of the code:** `pando85/homelab` (AppDaemon and HA configuration). This repository holds the design, the evaluation and the shadow MPC (track B), and writes nothing to the plant.

## 1. Purpose, scope and rollout gate

The comfort controller is a new homelab AppDaemon app that **replaces the legacy cheapest-hours
`climate_control.py`**. It meets a comfort schedule with the fewest kWh and ignores price for space
heating. Hot water stays price-optimised (section 7).

**Rollout gate.** The implementation goes on a homelab branch. It is **merged and tested only once the
house battery is working** (installation expected the week of 2026-11-02). Until then the legacy app keeps
running unchanged. The reason is evaluation and safety: the import-limit cap (section 9) and the
heat-pump metering basis only matter once the battery runs, and the regime change should not be mixed
with the battery commissioning.

Out of scope: any battery actuation, cooling, and any change to the unit's internal protections.

## 2. Run logic

AppDaemon decides whether heat-pump heating is on or off. It has to, because the pump's zone sensor is the
water temperature (verified 2026-10-30), so the room thermostats cannot call for heat; they only open and
close the floor valves.

| Transition | Condition |
|---|---|
| **On** | any thermostat room is below its setpoint minus a hysteresis |
| **Off** | every room is at its setpoint |

There is no separate optimum-start condition: optimum start works by moving the setpoint earlier (section 6), because a thermostat whose room is above its setpoint keeps its floor valve closed, so switching the heat pump on early would deliver no heat to that room.

Machine limits (in code, not helpers): compressor **minimum on 30 min, minimum off 20 min**; a
**starts-per-day alarm** (a notification, not a control action).

The on/off action is the existing, proven command path (operating mode `Heat+DHW` / `DHW` through
`input_select.heishamon_heatmode`), as in the C0 fixes of 2026-10-30.

## 3. Comfort schedule and setpoints

Minimums, as in ADR-0006 decision 2:

- hall/kitchen at least 23.5 degC from **12:00 on weekdays and from 09:00 at weekends**;
- every room at least 21 degC at all times.

Setpoints carry a **+0.5 K margin** over the minimum:

| Room (thermostat entity) | Setpoint in the day window | Setpoint otherwise |
|---|---|---|
| Hall/kitchen (`climate.thermostat_hall_kitchen`) | 24.0 | 21.5 |
| Bedroom (`climate.thermostat_bedroom`) | 21.5 | 21.5 |
| Office A (`climate.thermostat_office_a`) | 21.5 | 21.5 |
| Office T (`climate.thermostat_office_t`) | 21.5 | 21.5 |

The **day window** starts at the weekday or weekend start above and ends at **22:00 by default**. The end
time is an assumption, editable as a helper (open item).

## 4. Thermostat semantics

- **The setpoint is the room's current target.** AppDaemon writes setpoints only **at schedule
  boundaries** (the early-start write of section 6, window end) and for away mode.
- **A manual change becomes the target until the next boundary.** A manual change made before the early-start write holds until that early-start boundary, which overwrites it.
  The same applies to every other boundary. If a manual setpoint is below the 21 degC
  floor, that is the owner's explicit choice and the controller does not fight it.
- **Every manual override is logged** (room, time, old value, new value). Repeated patterns produce
  **suggested** schedule changes (a notification or a report), **never silent changes**.
- **The thermostats must not run their own device programs**, or the setpoint would change behind the
  controller's back (open item: confirm the current state of the programs).

## 5. Away mode

One HA switch sets every room to 21.5 degC. When away is switched off, AppDaemon writes the schedule
setpoints that apply at that moment, including the day-window setpoint if the early-start time of the next
window has already passed. The compressor limits stay in force.

## 6. Optimum start

**Optimum start is the schedule boundary moved earlier.** AppDaemon writes the day-window setpoint of the
hall/kitchen at **(window start minus lead time)** instead of at the window start. The thermostat then opens
its valve because the room is below the new setpoint, the run logic of section 2 turns the heat pump on, and
the room reaches the minimum at the window start. The run logic stays purely demand-driven. The lead time
is a table by outdoor-temperature band, **editable as HA helpers**.

The initial values come from the warm-up analysis (track B, step B0, in progress). Until it reports, the
table is a marked placeholder:

| Outdoor band (degC) | Lead time: early-start write before the window start (minutes) |
|---|---|
| TODO (below -2) | TODO from B0 |
| TODO (-2 to 2) | TODO from B0 |
| TODO (2 to 6) | TODO from B0 |
| TODO (6 to 10) | TODO from B0 |
| TODO (10 and above) | TODO from B0 |

The band edges above are also placeholders. No lead time is guessed in this document; the first deployment
uses the B0 values or, if B0 is late, a conservative owner-chosen table recorded in the
[regime log](../components/regime-log.md).

## 7. Water temperature and hot water

### 7.1 Space-heating water temperature

AppDaemon controls it **directly**. The unit stays in **direct mode** (`Heating_Mode` 1). AppDaemon writes
`SetZ1HeatRequestTemperature` as an absolute degC value.

Requirements:

- a **software curve** with editable points; initial points: outdoor **-5 degC to 32 degC** water and
  **15 degC to 25 degC** water, interpolated linearly and clamped outside the two points;
- an optional **raise during optimum start** (a small, editable number of kelvin), applied **from the early-start write until the hall/kitchen reaches its setpoint**;
- **integers only, clamped to 20 to 35 degC**;
- **write only when `sensor.aquarea_heating_mode` = 1** (direct); in curve mode the same command means a
  shift in K, so never write;
- **MQTT retain false and a validated payload**: the fork turns an empty or garbage payload into 0 degC;
- **write only when the target changes by at least 1 K and at most once per 15 min**, because the unit
  probably persists settings (avoid memory wear);
- **log every write** (time, old target, new target, outdoor input, reason).

The curve is software, not the unit's native curve, so the unit's four curve points are untouched.

**Open: the outdoor-temperature source for the curve.** The unit's own sensor reads about 1.6 to 2.5 K low
while the compressor runs ([outdoor-temperature](../components/outdoor-temperature.md)). Options: an offset
correction of the unit sensor, or an HA weather entity.

### 7.2 DHW stays price-optimised (amends ADR-0006 decision 4)

ADR-0006 decision 4 retired DHW price forcing. This design **keeps DHW price-optimised**. Shifting heat into
the tank is lossless, whereas the battery loses 10 to 20% round trip and has limited capacity, so hot water
is the better place to use a cheap hour.

- **Rule:** the cheapest **2 h window of the day**, ranked by **price divided by expected COP** (warmer
  hours rank better), avoiding the optimum-start warm-up.
- **Tank target at most 50 degC**, so the heat pump alone reaches it and the 3 kW heater never starts
  (service manual: above about 51 to 55 degC needs the heater).
- **Legionella / sterilization** is currently off (`binary_sensor.aquarea_sterilization_state` 0). Keeping
  it off is an open owner decision.

### 7.3 Price reactors become three

The price-reactive writers are now: the **battery optimiser**, the **DHW scheduler** and **boost**
(section 8). DHW and battery charging will often share the cheapest hours.

## 8. Boost (phase 2, after normal mode works)

Boost is built only after normal mode has run in production. Parameters as in ADR-0006 decision 3, as
amended:

- **Space-heating soak:** room ceiling about 25.5 degC, water about 30 to 32 degC.
- **Start condition:** battery state of charge about 90 to 95%, **or** charge power drops, **or** window end
  minus soak time is reached, whichever comes first.
- **DHW in boost also stays at most 50 degC** (corrects the "55 to 60 degC" of the original decision 3).
- Boost never relaxes the comfort minimums, the compressor limits or the manufacturer protections.

## 9. Import limit and the heater

The **5.0 kW smart-meter limit is hard**: the meter cuts supply, and this has happened. DHW and battery
charging will often share the cheapest hours, so:

- the **battery grid-target cap below the import limit (work-list item 8 in the
  [climate plan](climate-optimizer-implementation.md)) is mandatory before go-live**;
- **AppDaemon must never trigger the 3 kW heater**: DHW target at most 50 degC, no sterilization from
  AppDaemon, no forced heater.

Open: the import-limit basis (instantaneous trip or 15-minute average), and whether the backup heater stays
enabled for space heating (below the unit's heater-on outdoor temperature of 0 degC it starts by itself;
see the heater facts in [heat-pump-controller](../components/heat-pump-controller.md)).

## 10. Rollback and failure mode

- **No runtime strategy switch.** Rollback is **`git revert` in homelab**; the legacy app is **archived in
  the homelab repository** so the revert is easy.
- **If AppDaemon is down,** the existing watchdog leaves the heat pump in heating mode at the last water
  temperature, and the thermostats keep protecting comfort through the floor valves. The worst case is
  extra kWh, never no heating.

Per the [rollback rules](climate-optimizer-implementation.md#rollback-and-operational-handoff), the
implementation PR states the expected `Heating_Mode`, who controls DHW, and how to verify the pump after
stopping AppDaemon.

## 11. HA helpers (editable)

Editable without a deployment: day minimums and times (weekday and weekend), night floor, margin, lead
times, curve points, the optional optimum-start raise, away. **Machine limits stay in code** (minimum on/off,
water clamp 20 to 35 degC, write rate, tank cap 50 degC).

## 12. Evaluation

Without a runtime switch, the [C7-F](climate-flow-calibration-optimization.md) field trial cannot alternate
arms. It becomes a **before/after comparison**: past winters plus this winter until go-live, against the new
controller, **normalised for outdoor temperature** (heating degree-days or temperature-band matching). The
shadow MPC (track B) is the second reference.

Metrics:

- Kh below each minimum, per room, and the share of time below;
- heat-pump kWh per HDD, from the heat-pump meter or HeishaMon, **not the whole-house meter** once the
  battery runs;
- compressor starts per day;
- DHW cost share.

**Preregister the criteria before go-live** (the same discipline as the F2a preregistration: criteria and
thresholds committed before the first controller day). A before/after comparison is **weaker evidence than a
randomised trial**: weather, occupancy and tariff differ between periods and the normalisation is
imperfect. The result is read with that limit stated.

## 13. Track B: shadow thermal-model MPC

A thermal model plus an MPC (objective: **minimum kWh subject to the same comfort schedule, price-blind**),
prepared in this repository and run in **shadow** (read-only, plan and prediction metrics only) in parallel
this winter. It is compared with the controller after winter. The shadow rule of `AGENTS.md` is unchanged:
no writes, no new credentials, no Home Assistant service calls.

Shadow can validate predictions and plan differences. It **cannot** show the realised comfort of the MPC,
because the MPC never acts.

| Step | Content | Status |
|---|---|---|
| **B0** | Diagnostics: [C2 recovery](climate-c2-recovery.md) R0 to R2 and the warm-up rates (feeds the optimum-start table) | in progress |
| **B1** | Offline MPC replay on the committed comfort schedule | planned |
| **B2** | Shadow runner in the service (read-only plan and prediction metrics) | planned |
| **B3** | Preregistered comparison after winter | planned |

## 14. Open items for the owner

| Item | Needed for | Options or default |
|---|---|---|
| Outdoor-temperature source for the curve | 7.1 | Offset-corrected unit sensor, or an HA weather entity |
| Day window end time | 3 | 22:00 default (assumption) |
| Lead-time values | 6 | B0 warm-up analysis (table is a TODO) |
| Legionella policy | 7.2 | Currently off; keep off or schedule a controlled cycle outside the cheap hours |
| Backup heater enabled for space heating | 9 | The heater starts below 0 degC outdoors; keep, restrict or disable |
| Import-limit basis | 9 | Instantaneous trip or 15-minute average (contract or distributor) |
| Thermostat device programs | 4 | Confirm they are off, or disable them |
