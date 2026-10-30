# Climate flow-temperature calibration and optimization (C7-F)

- **Status:** Proposed extension; design and data collection only. No authority to change heat-pump settings. Reconciled 2026-10-30 with the C2 outcome (UNDETERMINED, no-go) and the C1a/C2 data findings; the identification, modeling and simulated-gate stages were replaced by a joint field trial agreed in principle with the owner (arms and preregistration still to be agreed)
- **Date:** 2026-10-30
- **Decision context:** [ADR-0006](../adr/0006-decouple-heat-pump-comfort-from-battery-price-optimisation.md) (2026-10-30) changed the F2 arms: A legacy cheapest-hours, B comfort schedule + efficiency-first (proposed, to be preregistered in F2a); the preheat/boost arm is dropped.
- **Parent plan:** [Climate optimizer implementation](climate-optimizer-implementation.md) (C0–C7), [C2 recovery plan](climate-c2-recovery.md) (R0–R5), [ADR-0005](../adr/0005-shared-energy-planning-climate.md)
- **Repositories:** `pando85/home-energy-optimizer` (datasets, preregistration, trial analysis); `pando85/homelab` (HA/AppDaemon helper semantics, guarded physical command writer)
- **Scope:** Space heating and supply/flow temperature only. No DHW temperature, cooling, battery dispatch or joint MPC changes.

## 1. Why this is a separate extension

The C2 models estimate the building dynamics and the electrical energy needed for measured heat output, and accept measured flow temperature as an input (none was accepted: C2 ended UNDETERMINED, see the [C2 outcome](climate-optimizer-implementation.md#c2-design-and-predeclared-gate)). That is **not** the same as knowing the counterfactual consequences of *commanding* a new supply temperature. The historical data predominantly reflects a fixed manually changed target; it may not identify an independent flow-temperature effect, an attainable target, or its interaction with valve closures and compressor modulation.

The purpose of C7-F is now a **model-free field trial** of the two levers that interact, permitted heating hours and water temperature (fixed target vs native curve), measured against current operation. Models of the plant are no longer a precondition: the history could not identify them, and a simulated gate would rest on them.

**Do not modify or re-register the predeclared C2 `heat-v1` gate, historic season split, or comfort thresholds.** C2 continues to evaluate the binary heat-permission lever. C7-F has its own separately preregistered dataset, baselines, success criteria and rollout decision. The C7-F trial is optional: C0–C6 and any future heating-permission pilot must not depend on its success, and C7-F does not depend on an accepted C2 model or on C3 either.

There is **no new service, OCI image, solver deployment, MQTT client or hardware writer**. Reuse the offline feature set, PVPC, forecast timestamps and the AppDaemon control boundary.

## 2. What we know, and what must be verified

The currently checked-in `homelab` configuration exposes `input_number.heishamon_heatshift_z1`, connected to MQTT command `SetZ1HeatRequestTemperature`, plus the Aquarea inlet/outlet/target, pump flow, compressor and thermal/electric heating-power sensors. The owner's `GoHeishaMon` fork has a handler for the command, but implementation of an MQTT command alone does **not** prove that a requested absolute supply temperature is applied.

In HeishaMon's documented modes, `SetZ1HeatRequestTemperature` may mean a **direct absolute water setpoint** or a **compensation-curve offset**. Direct-temperature behavior also differs under room-thermostat/thermistor modes; updating the underlying curve may require `SetCurves`, a command that must not be assumed to exist in the deployed owner's fork. Panasonic has separate direct-flow and compensation-curve configurations. Confirm the actual installed model, heating control mode, zone count, active curve and hardware limitations *on the device* before any automated flow actuation.

Audit the existing controller mode without altering it: distinguish commanded value, controller-reported zone request, controller-reported main target and measured water outlet. Map the applicable control surface explicitly as one of `direct_absolute`, `curve_offset`, `curve_definition` or `unsupported`. `unsupported` is a valid outcome: collect observational data and keep native control.

Read-only sources:
- Aquarea outdoor, water inlet, outlet, outlet-target, pump flow, compressor frequency/starts, heating thermal and electrical energy/power, defrost and 3-way valve, heating/DHW mode, and force-DHW state.
- Room thermostat measured temperatures, setpoints, valve/demand proxy and the always-open floor area; keep the office with server gains identifiable.
- Weather forecast as issued at decision time, archived outside temperature, solar proxy, PVPC publication snapshots, heat-permission commands and *reported device states*.
- Shelly EM heating-branch electricity with identified contamination from washer/dishwasher/extractor. Its historic ~1.28x ratio to HeishaMon is **not** a validated device COP correction factor.
- For every trial settings change: UTC issue time, intended zone, control mode/units, desired value, state acknowledgement, observed main target/outlet, actuation owner, override state and rollback outcome.

### Starting facts from the C1a/C2 audit

F0, F1 and the trial design start from these observations (qualitative; plant-level only). They are evidence to confirm, not settled model structure.

- **Fixed flow target.** The flow target was a constant per season: about 35 °C in 2023–24, 28 °C then 26 °C in 2024–25, and 26 °C in 2025–26. Within a month it had zero correlation with outdoor temperature.
- **F0 finding from Prometheus (2026-10-30), to be confirmed on the device.** The pump runs in direct water-temperature mode: `sensor.aquarea_z1_heat_request_temperature` equals `sensor.aquarea_z1_water_target_temperature`, which is 26 °C even at an outdoor temperature of 6 °C. A Z1 heating curve is configured but inactive: its sensors read −5 °C outdoor → 35 °C water and 15 °C outdoor → 26 °C water, and its high-end target followed the manual value (28, then 26). `input_number.heishamon_heatshift_z1` holds the absolute target (35, 28 and 26 over the seasons). Still open for F0: confirm the mode on the device, and whether the owner's GoHeishaMon fork or the Panasonic controller switches the heating mode.
- **Valves and flow temperature.** At about 35 °C the comfort-room valves were shut about half the heating time and heat went to the always-open zones and the slab; the same rooms then delivered about twice the heat per kelvin of indoor–outdoor difference. COP was lower at the higher flow temperature.
- **Passive natural experiments.** With all valves open, the heat delivered per kelvin of water-to-room difference was similar across seasons. The three seasons are F1's passive natural experiments; together they span the 26–35 °C range, with few points between 28 and 35 °C.
- **Outdoor sensor.** The heat-pump outdoor sensor is biased by the compressor and by sun. Use `t_out_debiased`, which for long gaps can be filled from a fused series that uses an external reference station (`t_out_fused`); do not use the raw sensor for flow-response fitting.
- **COP basis.** A COP table binned by lift (flow − outdoor) met the daily validation limit but not the season limit. Treat it as a starting baseline for later model identification, not as an accepted model.
- **Slab identifiability.** The two-state slab model was not identifiable from historical data. The trial's deliberately varied data may help distinguish the plant's response, but is **not guaranteed** to identify an effective slab temperature. C2-R checks the return-water proxy, sensitivity and parameter profiles; if these fail, favor a simpler empirically validated model over retaining B.

Carry provenance, quality flags, instrument calibration/resolution and decision-time availability with each 15-minute feature row. Exclude ACS, cooling, defrost and abnormal sensor periods from steady-state COP fitting; preserve these periods separately for robustness and operating-cost accounting.

## 3. Stages and dependencies

| Stage | Where | Deliverable | Start condition and exit gate |
|---|---|---|---|
| **F0 — Control-surface audit** | homelab + optimizer docs | Actual Panasonic control mode and zone map (starting from the finding above), command-to-response semantics, manufacturer's flow limits, how the configured curve would be activated and restored, manual-override/failsafe behavior | No automated test commands; documented observation and supervised verification. If the mode or curve cannot be switched safely, the trial has no native-curve arm |
| **F1 — Passive coverage and baseline** | optimizer | Historical flow/ambient/thermal/COP coverage report built on the three fixed-target seasons; baseline distributions of the trial metrics under current operation | Use existing C1a/C2 datasets and the facts above (`t_out_debiased`, lift-based COP); verify electrical/thermal meter discrepancies and sensor quality |
| **F2 — Joint field trial** (replaces the former identification campaign, flow-aware models and simulated gate) | optimizer (preregistration, analysis), homelab (settings changes) | Model-free, preregistered trial of permitted heating hours and water temperature against current operation, run this winter ([section 4](#4-joint-field-trial)); GO / NO-GO / UNDETERMINED per arm | Arms, preregistration and the small homelab change agreed with the owner **before** the first block; F0 done for any native-curve arm; C0 safety/override handling |
| **F5 — Guarded actuation** | homelab | **Parked** | Revisited only if the trial shows a static configuration leaves substantial value; then needs its own plan, single-writer gates and physical authorization |
| **F6 — Maintenance** | optimizer + homelab | **Parked** | As F5 |

**Concurrency:** C2 ended UNDETERMINED (no-go), so the model-based permission path (C1b, C3–C5 as specified) does not start. [C2-R](climate-c2-recovery.md) diagnoses the observation and action-model failures; F0 and F1 can run in parallel with it and share the evidence. The trial needs no accepted C2 model, no C3 and no permission optimizer, and never replaces or reopens the C2 acceptance sequence or its preregistered gate.

## 4. Joint field trial

A model-free comparison on the real house. It varies the two levers that interact, permitted heating hours and water temperature, because varying one at a time would miss the interaction (flow temperature changed valve behaviour and COP in the history).

**Arms.** Two, proposed after [ADR-0006](../adr/0006-decouple-heat-pump-comfort-from-battery-price-optimisation.md) and **to be preregistered in F2a** (not yet agreed in detail):
- **A, legacy cheapest-hours:** current operation, the baseline.
- **B, comfort schedule + efficiency-first:** hall/kitchen at 23.5 °C or more from 12:00 on weekdays and 09:00 at weekends, every room at 21 °C or more, optimum start from learned warm-up rates, the native heating curve (low water temperature) and long gentle runs, price ignored.

The weather-dependent-hours candidate and the preheating arm are dropped: price-driven preheating is out of scope because very cheap windows are rare in winter, and boost is not part of this trial.

**Blocks.** Whole-week blocks in randomised order, so weather and price trends do not line up with one arm. The slab and building time constant is about 150 h, so the first 1–2 days of each block are discarded as washout. That is short against 150 h: carry-over is not removed, so the preregistration states how it is handled (randomised order, previous-arm adjustment) rather than assuming it away.

**Metrics, preregistered before the first block.** All adjusted for heating degree-days:
- degree-hours (Kh) below the comfort minimums in the comfort rooms (bedroom, hall-kitchen, office T); F2a re-bases the former 22 °C threshold on the ADR-0006 schedule (hall/kitchen 23.5 °C in the scheduled hours, every room 21 °C) and keeps 22 °C as a comparison with history;
- heating electricity;
- cost relative to each day's price level (price-neutral), so arms are not rewarded for luckier price weeks (secondary; heating electricity and comfort are primary);
- compressor starts.

Also preregistered: arms and their exact settings, block order and seed, washout rule, day-exclusion rules (DHW, defrost, outages), the HDD adjustment, the electrical metering basis, per-arm decision thresholds and the abort conditions. Do not invent thresholds in this document; agree them with the owner and commit them before the first block. A comfort regression vetoes any cost or electricity gain.

**Result.** GO, NO-GO or UNDETERMINED **per arm against baseline**. With few blocks and weather variance, UNDETERMINED is an expected and valid outcome; it is not "safe to deploy". Never promote on a single day or week.

**How settings change.** The owner changes them manually, or AppDaemon follows a calendar. Either way it is a small homelab change agreed separately; the optimizer service writes nothing and holds no new credential. Record, for every change, the UTC time, intended arm, requested value, the reported device state afterwards, who changed it and any override (`experiment_id`, `policy_version` in the existing history format).

**Second purpose.** The trial's deliberately varied data also feeds later model identification (the old F3 goal: command to heat to room to electricity, including the slab), under C2-R's separate gates. That identification is a by-product, not a condition of the trial, and does not make an arm's verdict depend on a model.

**Safety.** Manual override wins. DHW and defrost take precedence. Respect manufacturer and screed surface limits, freeze protection and minimum compressor run/rest. Suspend or abort an arm on stale data, large comfort error, excessive starts, failure of the device to report the new setting, manual override or unexpected hydraulics, and restore the previous configuration. Rollback to current operation and pump-native control must remain possible throughout. Do not modify compressor or internal Panasonic protection loops.

## 5. Separation from C2

The trial does not use, rewrite or re-register the C2 `heat-v1` gate, its season split or comfort thresholds. It has its own preregistration and verdicts. It is independent of the C2-R gates, and neither can convert C2's no-go into a pass. If no arm clearly beats baseline, keep current operation (or whatever simpler rule the model-free `min_hours` analysis supports); that is a successful simplification, not a failed project.

## 6. Parked: guarded actuation and maintenance

F5 (a flow-target executor) and F6 (drift and recalibration automation) are parked. They are revisited only if the trial shows that a static configuration leaves substantial value on the table, and then under their own plan: single writer in AppDaemon, real acknowledgement from the reported device state, bounded rate and conservative fallback to the native curve.

A future permission-mode actuation (C5, if its prerequisites are ever met) would not need any flow writer, and nothing here authorizes battery actuation or DHW/cooling setpoint changes.

## 7. Reviewable PR sequence

| PR | Area | Required evidence |
|---|---|---|
| **F0** | Homelab field audit, measurement source and control semantics | Device mode, zone and actual vs requested targets documented, including the open questions above; no unsafe commands |
| **F1** | Historical coverage and trial-metric baselines | Correct UTC/units, meter reconciliation flags, independent days and exposure gaps |
| **F2a** | Trial design and preregistration | Owner-agreed arms, block order, metrics, thresholds, abort conditions; committed before the first block |
| **F2b** | Homelab settings change (manual procedure or calendar) | Agreed separately; reversible; reported-state check; manual override and rollback |
| **F2c** | Trial analysis and verdict | Per-arm GO / NO-GO / UNDETERMINED against baseline; retained raw evidence |

No F-stage code or experiment authorization is implied by publishing this document. Physical changes require separate approval.

## 8. Sources and related work

- [Climate C2 acceptance design and gate](climate-optimizer-implementation.md#c2-design-and-predeclared-gate)
- [Climate package and thermal/plant/COP models](../architecture.md#climate-subpackage-climate)
- [Home Assistant / HeishaMon integration](https://github.com/pando85/homelab/blob/master/apps/home-assistant/config/integrations/heishamon.yaml)
- [GoHeishaMon CZ-TAW1 fork](https://github.com/pando85/GoHeishaMon/blob/main/src/taw2mqtt.go)
- [HeishaMon MQTT command modes](https://github.com/heishamon/HeishaMon/blob/main/MQTT-Topics.md)
- [Panasonic Aquarea heating configuration manual](https://www.panasonicproclub.com/uploads/GB/catalogues/2023/UK_INSTALLATION_MANUAL_H%202023v3.pdf)
