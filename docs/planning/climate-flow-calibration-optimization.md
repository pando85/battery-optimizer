# Climate flow-temperature calibration and optimization (C7-F)

- **Status:** Proposed extension; design and data collection only. No authority to change heat-pump settings. Reconciled 2026-10-30 with the C2 outcome (UNDETERMINED, no-go) and the C1a/C2 data findings
- **Date:** 2026-10-30
- **Parent plan:** [Climate optimizer implementation](climate-optimizer-implementation.md) (C0–C7), [C2 recovery plan](climate-c2-recovery.md) (R0–R5), [ADR-0005](../adr/0005-shared-energy-planning-climate.md)
- **Repositories:** `pando85/home-energy-optimizer` (datasets, physics, calibration, offline/shadow evaluation); `pando85/homelab` (HA/AppDaemon helper semantics, guarded physical command writer)
- **Scope:** Space heating and supply/flow temperature only. No DHW temperature, cooling, battery dispatch or joint MPC changes.

## 1. Why this is a separate extension

The C2 models estimate the building dynamics and the electrical energy needed for measured heat output, and accept measured flow temperature as an input (none was accepted: C2 ended UNDETERMINED, see the [C2 outcome](climate-optimizer-implementation.md#c2-design-and-predeclared-gate)). That is **not** the same as knowing the counterfactual consequences of *commanding* a new supply temperature. The historical data predominantly reflects a fixed manually changed target; it may not identify an independent flow-temperature effect, an attainable target, or its interaction with valve closures and compressor modulation.

The purpose of C7-F is to **characterize the heat pump and floor response over the feasible supply-temperature envelope**, then compare the incremental value of controlling the flow temperature on top of the existing heating-permission scheduler.

**Do not modify or re-register the predeclared C2 `heat-v1` gate, historic season split, or comfort thresholds.** C2 continues to evaluate the binary heat-permission lever. C7-F has its own separately preregistered dataset, baselines, success criteria and rollout decision. The C7-F experiment is optional: C0–C6 and the existing C5 heating-permission pilot must not depend on its success, and C7-F does not depend on an accepted C2 model or on C3 either.

There is **no new service, OCI image, solver deployment, MQTT client or hardware writer**. Reuse the offline feature set, existing climate thermal/COP/plant modules, versioned model artifacts, shadow service, PVPC, forecast timestamps and AppDaemon control boundary.

## 2. What we know, and what must be verified

The currently checked-in `homelab` configuration exposes `input_number.heishamon_heatshift_z1`, connected to MQTT command `SetZ1HeatRequestTemperature`, plus the Aquarea inlet/outlet/target, pump flow, compressor and thermal/electric heating-power sensors. The owner's `GoHeishaMon` fork has a handler for the command, but implementation of an MQTT command alone does **not** prove that a requested absolute supply temperature is applied.

In HeishaMon's documented modes, `SetZ1HeatRequestTemperature` may mean a **direct absolute water setpoint** or a **compensation-curve offset**. Direct-temperature behavior also differs under room-thermostat/thermistor modes; updating the underlying curve may require `SetCurves`, a command that must not be assumed to exist in the deployed owner's fork. Panasonic has separate direct-flow and compensation-curve configurations. Confirm the actual installed model, heating control mode, zone count, active curve and hardware limitations *on the device* before any automated flow actuation.

Audit the existing controller mode without altering it: distinguish commanded value, controller-reported zone request, controller-reported main target and measured water outlet. Map the applicable control surface explicitly as one of `direct_absolute`, `curve_offset`, `curve_definition` or `unsupported`. `unsupported` is a valid outcome: collect observational data and keep native control.

Read-only sources:
- Aquarea outdoor, water inlet, outlet, outlet-target, pump flow, compressor frequency/starts, heating thermal and electrical energy/power, defrost and 3-way valve, heating/DHW mode, and force-DHW state.
- Room thermostat measured temperatures, setpoints, valve/demand proxy and the always-open floor area; keep the office with server gains identifiable.
- Weather forecast as issued at decision time, archived outside temperature, solar proxy, PVPC publication snapshots, heat-permission commands and *reported device states*.
- Shelly EM heating-branch electricity with identified contamination from washer/dishwasher/extractor. Its historic ~1.28x ratio to HeishaMon is **not** a validated device COP correction factor.
- For every tested flow command: UTC issue time, intended zone, control mode/units, desired value, state acknowledgement, observed main target/outlet, actuation owner, override state and rollback outcome.

### Starting facts from the C1a/C2 audit

F0 and F1 start from these observations (qualitative; plant-level only). They are evidence to confirm, not settled model structure.

- **Fixed flow target.** The flow target was a constant per season: about 35 °C in 2023–24, 28 °C then 26 °C in 2024–25, and 26 °C in 2025–26. Within a month it had zero correlation with outdoor temperature. The pump therefore appears to run with a fixed water target, not a native compensation curve. F0 must confirm the actual mode on the device.
- **Valves and flow temperature.** At about 35 °C the comfort-room valves were shut about half the heating time and heat went to the always-open zones and the slab; the same rooms then delivered about twice the heat per kelvin of indoor–outdoor difference. COP was lower at the higher flow temperature.
- **Passive natural experiments.** With all valves open, the heat delivered per kelvin of water-to-room difference was similar across seasons. The three seasons are F1's passive natural experiments; together they span the 26–35 °C range, with few points between 28 and 35 °C.
- **Outdoor sensor.** The heat-pump outdoor sensor is biased by the compressor and by sun. Use `t_out_debiased`, which for long gaps can be filled from a fused series that uses an external reference station (`t_out_fused`); do not use the raw sensor for flow-response fitting.
- **COP basis.** A COP table binned by lift (flow − outdoor) met the daily validation limit but not the season limit. Treat it as the starting COP baseline for F3, not as an accepted model.
- **Slab identifiability.** The two-state slab model was not identifiable from historical data. Flow-varied experiments (F2) may help distinguish the plant's response, but are **not guaranteed** to identify an effective slab temperature. C2-R checks the return-water proxy, sensitivity and parameter profiles; if these fail, favor a simpler empirically validated model over retaining B.

Carry provenance, quality flags, instrument calibration/resolution and decision-time availability with each 15-minute feature row. Exclude ACS, cooling, defrost and abnormal sensor periods from steady-state COP fitting; preserve these periods separately for robustness and operating-cost accounting.

## 3. Stages and dependencies

| Stage | Where | Deliverable | Start condition and exit gate |
|---|---|---|---|
| **F0 — Control-surface audit** | homelab + optimizer docs | Actual Panasonic control mode and zone map, command-to-response semantics, manufacturer's flow limits, native curve and manual-override/failsafe behavior | No automated test commands; documented observation and supervised verification. If not controllable, stop before F2 active tests |
| **F1 — Passive coverage and baseline** | optimizer | Historical flow/ambient/thermal/COP coverage report and a flow-aware dataset contract; rank missing operating points | Use existing C1a/C2 datasets and the facts above (the three fixed-target seasons as natural experiments, `t_out_debiased`, lift-based COP); verify electrical/thermal meter discrepancies and sensor quality. Baseline heat/flow response established; whether any compensation curve was ever active is an F0 finding |
| **F2 — Supervised identification campaign** | homelab execution, optimizer archive | Timestamped, labelled low-risk experiments that fill informative gaps in observed flow/ambient conditions | Owner-approved site configuration and hardware-safe ranges; C0 safety/override handling. Before C5, any setpoint change is **manual**, not agent-controlled |
| **F3 — Flow-dependent models** | optimizer | Reusable `Q_heat` response + `E_electric` models conditioned on ambient and *actual* flow, with uncertainty/domain checks and model provenance; cooperate with [C2-R identifiability and plant/energy decomposition](climate-c2-recovery.md#3-diagnostic-experiments-before-new-optimization) | Held-out comparison against simple, existing and diagnostic challenger models; **not** a claim that A/B have been accepted; no unsupported extrapolation |
| **F4 — Incremental planning evaluation** | optimizer | Offline comparison of actual legacy and conditionally supported native compensation / flow strategies; any newly accepted permission model is an optional baseline | F3's **entire command→heat→room→electricity chain** passes independent held-out / prospective checks before an optimizer claim; otherwise observational analysis only. Does not require C2/C3, has its own preregistered gate |
| **F5 — Optional guarded actuation** | homelab | Feature-flagged, reversible flow-target/offset executor alongside the existing heating-permission owner | C5 single-writer, runtime safety and handoff gates passed; F4 evidence positive; separate explicit physical authorization |
| **F6 — Maintenance** | optimizer + homelab | Coverage/drift dashboard, periodic offline recalibration, shadow challenger evaluation, pinned model versions and rollback | Only promote new models through the same holdout and safety gates; no opportunistic unbounded experimentation |

**Concurrency:** C2 ended UNDETERMINED (no-go), so the model-based permission path (C1b, C3–C5 as specified) does not start. [C2-R](climate-c2-recovery.md) diagnoses the observation and action-model failures; F0, F1 and F3 can run in parallel and share the evidence. F2 passive/manual experiments do not require permission MPC; automatic experiment commands and F5 are blocked on the relevant supervised actuation safeguards. F4 does not need an accepted C2 model: it compares against the historical legacy baseline and native compensation, and adds an accepted permission policy only if one exists later. F4/F5 never replace or reopen the C2 acceptance sequence or its preregistered gate.

## 4. Identification campaign: collect the *right* data

First generate a two-dimensional **effective sample coverage matrix** over outdoor-temperature bands and **actual achieved supply-temperature bands**, keeping compressor state, valve availability and mode homogeneous. Count *independent heating sessions/days* and effective steady-state hours, not raw one-minute samples. Highlight combinations that are absent, heavily defrost-contaminated or confounded by solar gains and DHW.

The initial candidate supply temperatures **30, 32 and 35 °C are exploratory only**; the historical seasons already cover about 26, 28 and 35 °C, so prefer candidates that fill F1's gaps over repeating 35 °C. Derive approved absolute values, or equivalently small curve offsets, from the machine's actual configuration and previous comfortable operating envelope. Do not force 30 °C in severe cold simply to populate a dataset, or test a fixed temperature if the equipment uses curve offsets.

Campaign method:

1. **Passive first.** Inventory past manually changed targets, temperature differences and comparable winter days; use existing natural experiments (the fixed-target seasons) wherever possible. Separate valve-open from valve-shut periods, since valve closure changed the delivered heat per kelvin by about a factor of two.
2. **Supervised, bounded crossover only for missing conditions.** Use matched outdoor-temperature and occupancy windows; alternate A/B/B/A order where practicable rather than always running high flow in colder weather. Record which values were *requested* and *actually achieved*.
3. **Separate time scales.** For compressor/COP identification, require stable heating intervals after transients; for slab/room identification, observe the longer 12–48 h response and recovery, since flow-varied runs are the most promising way to make the slab identifiable. Do not assume an hourly temperature switch exposes the house's thermal response.
4. **Change one control lever at a time.** Keep heating permission, thermostat bands, zone availability and DHW scheduling as constant as practicable. Log unavoidable deviations; never treat command-on as delivered heat.
5. **Bound and abort.** Respect manufacturer/screed surface limits, freeze protection, minimum compressor run/rest and defrost/DHW precedence; suspend trials on stale data, large comfort error, excessive starts, failure to acknowledge, manual override or unexpected hydraulics.
6. **Track marginal information.** Stop testing an operating cell once held-out estimates stabilize; return to passive collection outside the characterized weather envelope.

As a *planning estimate*, a few weeks of appropriately varied winter operation may establish useful first flow-response curves. This is **not** a promised calibration duration or a substitute for low-outdoor-temperature coverage. Seasonal extremes can only be learned when they occur (or constrained conservatively from manufacturer data).

Maintain an internal `experiment_id` and `policy_version` in the existing history/artifact format. Preserve a holdout period never touched during candidate selection or tuning. Avoid data leakage between overlapping windows, bootstrapped sessions and model versions.

## 5. Modeling, cost and uncertainty

Keep the building model (`climate/thermal.py`) separate from the heat-pump **command-to-delivered-heat** model (`climate/plant.py`) and **delivered-heat-to-electricity** model (`climate/cop.py`):

- **Thermal building state:** indoor air and slab effective temperature (the slab state was not identifiable from historical data alone); losses and solar/internal gains. Refitting may be needed as the training envelope expands, but do not create one house model per flow setpoint.
- **Plant response:** predict attainable heating power and actual supply temperature given control mode, requested target/offset, ambient, slab/return state, valve availability, compressor state and hydraulic bounds. The current `u`-driven response already includes flow target, but identifying a counterfactual command response requires evidence at varied setpoints.
- **Electricity/COP:** fit **electricity** for measured thermal output as a function of *actual* flow/outdoor temperature and operating state (including starts and separate defrost handling); compare existing simple/table baselines (the lift-binned table first) before adding interactions, splines or ML. Do not call an observed COP ratio an independently calibrated actuator model.
- **Coverage and uncertainty:** publish allowed actuation envelope, effective sample counts, confidence/error intervals and in-/out-of-distribution status. If a proposed setpoint is outside measured or manufacturer-bounded coverage, **fall back to the native curve or an accepted fixed baseline**. Uncertainty should affect both predicted economics and comfort constraints.
- **Separate costs:** report thermal kWh, electrical kWh, euros under decision-time prices, compressor starts and comfort degree-hours. Reconcile the HeishaMon/Shelly electrical basis before claiming euro savings, and avoid counting HVAC demand twice when exchanging a committed climate forecast with the battery planner.

Use a clear flow model version, training-period boundaries, input manifest hash and a rollback-safe versioned artifact on the existing PVC. No broad real-time auto-fitting in the actuation process.

## 6. Comparison and promotion: a new gate, not a rewrite of C2

Prerecord the experiment coverage, acceptable hardware envelope, model-error thresholds, baseline and minimum incremental benefit **before** testing the final holdout. The following are the required comparisons on matched chronological periods:

| Policy | Purpose |
|---|---|
| Existing AppDaemon cheap-hour scheduling + current fixed/owner-configured flow | Historical operational baseline |
| Panasonic native compensation curve with the *same* (legacy) permission schedule | Measure value of the native weather compensation without a new flow writer |
| Any later accepted heating-permission policy at the same fixed flow (not available while C2 is undetermined) | Isolate value of better on/off scheduling |
| Heating-permission schedule (legacy, or an accepted policy) + proposed bounded flow modulation | Measure the additional value of C7-F |

The set must be meaningful without an accepted C2 model. The historical data show a fixed flow target, so native compensation was probably never operated: do not fabricate its observed counterfactual; label simulation-only results explicitly or exclude that baseline until a safe prospective comparison.

Required evaluation:
- Chronological held-out **actual-weather** model diagnostics, separately from **decision-time-weather-forecast** backtests; both matter and must never be mixed.
- Command-to-achieved target error, measured heat-power and electricity prediction error by outdoor/flow bands, compressor start frequency and valve-closure sensitivity.
- 1/6/12/24 h room-temperature errors, degree-hours below 22 °C and above 28 °C, worst excursion and terminal stored-heat consistency.
- Matched thermal kWh, electrical kWh and euros, with paired confidence intervals and sensitivity to the electrical metering uncertainty. Distinguish less energy from cheaper timing, and use the same end-of-horizon valuation across policies.
- Runtime feasibility, command rate limits, plan validity/ack timeouts, and sustained shadow-vs-observed performance. Safety/comfort regression vetoes any apparent monetary benefit.

**Pass criterion:** the flow-aware challenger must demonstrably improve the chosen objective compared with the *strongest applicable* accepted baseline, with no materially worse comfort or machine cycling and an improvement greater than model/metering uncertainty. Do not promote based on a single selected day's cost reduction. If there is insufficient comparable coverage, return **UNDETERMINED** (not “safe to deploy”). F4 keeps its own gate, preregistered separately. Reusing or rewriting the C2 `heat-v1` permission-only gate for this new lever would invalidate its preregistration.

If improvement is small or not reproducible, prefer **native weather compensation (when verified/safely commissioned) plus the current AppDaemon legacy schedule**, or a separately validated simpler permission policy. An accepted C2 permission optimizer does not currently exist. Avoid deploying additional water-temperature control without measured benefit.

## 7. Deployment and continuing learning

- **SENSE / LEARN:** reuse the existing Prometheus extraction and parquet pipeline, model registry / shadow reporting, daily or periodic offline recalibration jobs. New jobs use the same OCI image with a separate CLI entrypoint, not a new Deployment.
- **DECIDE:** reuse the same read-only resident optimizer. A `ClimatePlan` may later carry `flow_control_mode`, `requested_offset_or_target`, `range_confidence`, `model_id` and expiry/quality metadata. Do not conflate an advisory target with a measured or committed action. Add versioned fields only after F4.
- **ACT:** retain *one* write authority in AppDaemon (or a future explicitly approved replacement). Avoid introducing a second direct MQTT writer from the optimization service. Apply a bounded change with acknowledgement based on actual reported device state; never assume that writing an HA helper proves a new flow target was reached.
- **Control rate:** a 15-minute MPC cadence does not imply changing supply temperature every 15 minutes. Add a minimum dwell time, small bounded adjustments, hysteresis and conservative native-curve fallback. Do not modify compressor/internal Panasonic protection loops.
- **Operational safeguards:** manual override wins; DHW and defrost win; stale sensors, API failure, implausible command response, missed acknowledgement or out-of-envelope forecasts disable *flow modulation* and restore the previously approved safe/native configuration. Document what physically happens on AppDaemon restart and on release rollback.
- **Online learning:** collect continuously; recalibrate offline periodically when sufficient informative new data arrives; evaluate challengers in shadow with frozen holdouts; promote only versioned accepted artifacts. Alerts for drift, uncovered outdoor/flow bins, excess starts, room-disagreement and COP residuals.

Existing C5 permission-mode actuation can be accepted and operated **without** any C7-F flow writer. Changes to heating flow targets do not authorize battery actuation or DHW/cooling setpoint changes.

## 8. Reviewable PR sequence and acceptance checklist

| PR | Area | Required evidence |
|---|---|---|
| **F0** | Homelab field audit, measurement source and control semantics | Device mode, zone and actual vs requested targets documented; no unsafe commands |
| **F1** | Flow signal provenance and historical coverage matrix | Correct UTC/units, meter reconciliation flags, independent samples and exposure gaps |
| **F2** | Supervised experiment journal / site-specific approved profile | Owner-reviewed envelope, AB/BA comparisons, abort conditions and retained raw evidence |
| **F3** | Flow-aware heat/COP model, training CLI and artifacts | Reproducible fit; improved held-out performance versus existing simpler models; uncertainty and out-of-support rejection |
| **F4** | Backtest, competing policy baselines, dedicated preregistered gate and shadow | Causal weather/price inputs, matched safety/economic metrics, **GO/NO-GO/UNDETERMINED** verdict |
| **F5** | Separately authorized AppDaemon executor + homelab GitOps feature flag | Single writer, real acknowledgement, bounded rate, manual override, failure tests, native fallback and pilot rollback |
| **F6** | Drift/coverage/refresh automation | Recalibration never automatically changes live model without promotion; measured ongoing benefit and rollback alerting |

No F-stage code or experiment authorization is implied by publishing this document. Physical changes require separate approval; rollback to the existing heating policy and pump-native control must remain possible throughout.

## 9. Sources and related work

- [Climate C2 acceptance design and gate](climate-optimizer-implementation.md#c2-design-and-predeclared-gate)
- [Climate package and thermal/plant/COP models](../architecture.md#climate-subpackage-climate)
- [Home Assistant / HeishaMon integration](https://github.com/pando85/homelab/blob/master/apps/home-assistant/config/integrations/heishamon.yaml)
- [GoHeishaMon CZ-TAW1 fork](https://github.com/pando85/GoHeishaMon/blob/main/src/taw2mqtt.go)
- [HeishaMon MQTT command modes](https://github.com/heishamon/HeishaMon/blob/main/MQTT-Topics.md)
- [Panasonic Aquarea heating configuration manual](https://www.panasonicproclub.com/uploads/GB/catalogues/2023/UK_INSTALLATION_MANUAL_H%202023v3.pdf)
