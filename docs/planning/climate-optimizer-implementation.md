# Climate optimizer: shared-runtime architecture and phased implementation plan

- **Status:** Proposed (documentation only; no application change or permission to actuate)
- **Date:** 2026-10-30
- **Architecture decision:** [ADR-0005](../adr/0005-shared-energy-planning-climate.md)
- **Repository ownership:** `pando85/battery-optimizer` for models, planning, history and evaluations; [`pando85/homelab`](https://github.com/pando85/homelab) for AppDaemon controls, HA entities, GitOps, manifests and deployment
- **Related:** [ADR-0001](../adr/0001-home-battery-economic-dispatch.md), [ADR-0004](../adr/0004-battery-optimizer-production-architecture.md), [battery runtime](../service.md), [component architecture](../architecture.md), [load forecaster](components/load-forecaster.md)

## 1. Goal, baseline and scope

Move from **"run for the cheapest N hours"** to **"find the lowest-cost physically feasible heat-pump schedule that maintains room comfort, hot-water availability and compressor health"**. Start with space heating; integrate domestic hot water (DHW) planning once the shared safety/ownership boundary is ready. Cooling follows only after separate model and condensation-safety work.

This is an **incremental extension** of the existing resident battery-optimizer service, **not** a third Deployment. Reuse data and runtime infrastructure, but keep battery and climate optimizers domain-specific. No new background database, MQTT broker, Kubernetes service or broad HA write credential is justified.

### Verified current state (repository review, 2026-10-30)

| Area | Exists today | Limitation / planning implication |
|---|---|---|
| AppDaemon runtime | Single-replica `home-assistant` chart Deployment; `acockburn/appdaemon:4.5.13`, `apps.yaml`; deployed with HA via ConfigMaps | Existing control authority and user toggles must remain intact |
| Space heating/cooling | `climate_control.py` chooses PVPC below 2/3 of a historical mean, otherwise the cheapest minimum number of hours | No room-temperature, COP, heat-loss or weather forecast in the planning objective |
| DHW | `dhw_control.py` forces hot water in the cheapest hour of each 12-hour block | No tank-need or water-demand prediction; separate app can compete with unified planning |
| Plant control | `input_select.heishamon_heatmode` maps via HA automation to HeishaMon `SetOperationMode`; `switch.aquarea_force_dhw_mode` is a separate actuator | Input-helper state is not equivalent to confirmed heat-pump mode; avoid unbounded state waits |
| Available telemetry | HeishaMon exposes heating/DHW thermal/electric power, COP-related measurements, tank/flow/outdoor temps, compressor/defrost state; HA LocalTuya has bedroom, hall-kitchen and office thermostats | Verify actual entity IDs, timestamps, units, calibration, representative rooms and physical zone topology before modeling |
| Existing thermal observability | Grafana climate/room/energy dashboards, a Shelly EM branch reported as aerothermal load | Check topology and measured electric power before treating this as ground truth |
| Shared data/runtime | `service/ha_client.py`, `entity_store.py`, `pvpc.py`, `timegrid.py`, `history.py`; weather/PVPC archives, 15-minute UTC time grid, Prometheus, shadow sink and metrics | Reuse; do not build another data-ingestion stack |
| Battery planning | `battery-optimizer serve` solves read-only battery MPC every 15 minutes, publishing shadow plans; hardware SOC may be simulated | Keep all existing battery endpoints/semantics and HA write guard unchanged |
| Schedule-aware battery forecasting | `structured_forecast.py` already replays versioned legacy AppDaemon climate/DHW rules **offline** | Preserve this as baseline; replace replay with versioned `ClimatePlan` forecasts only after validation, without double counting |

Relevant source:
[`homelab/apps.yaml`](https://github.com/pando85/homelab/blob/master/apps/home-assistant/appdaemon/apps/apps.yaml),
[`climate_control.py`](https://github.com/pando85/homelab/blob/master/apps/home-assistant/appdaemon/apps/climate_control.py),
[`dhw_control.py`](https://github.com/pando85/homelab/blob/master/apps/home-assistant/appdaemon/apps/dhw_control.py),
[`heishamon.yaml`](https://github.com/pando85/homelab/blob/master/apps/home-assistant/config/integrations/heishamon.yaml),
[`home-assistant/values.yaml`](https://github.com/pando85/homelab/blob/master/apps/home-assistant/values.yaml).

**Out of scope for the initial rollout:** hardware flow-temperature changes, direct Modbus control, multi-pump dispatch, automatic battery actuation, replacing Omnibattery, installing PV generation, retraining large ML models, and removing AppDaemon. Any projected savings before controlled validation are **model estimates**, not measured bill reductions.

## 2. Target architecture and responsibility map

~~~mermaid
flowchart TD
    subgraph SOURCES[Existing sensors and history]
        HA[HA and HeishaMon events]
        ARCH[PVPC and weather archives]
        PROM[Prometheus and HA history]
    end
    subgraph ENGINE[Existing battery-optimizer Deployment]
        INPUT[Read adapters, freshness, UTC/DST and tariffs]
        TM[Thermal model and COP estimator]
        DHW[DHW tank and demand model]
        CMPC[Climate planner]
        BMPC[Existing battery forecast and dispatch planner]
        STORE[Existing shadow store and metrics]
        API[Read-only GET /climate/plan]
        INPUT --> TM
        INPUT --> DHW
        TM --> CMPC
        DHW --> CMPC
        INPUT --> BMPC
        CMPC --> STORE
        CMPC --> API
        CMPC -. forecast HVAC electrical kWh .-> BMPC
        BMPC --> STORE
    end
    HA --> INPUT
    ARCH --> INPUT
    PROM --> INPUT
    API -. gated, future .-> AD[Existing AppDaemon executor]
    AD --> HACTL[HA / HeishaMon commands]
    HACTL --> PUMP[Panasonic Aquarea]
~~~

The **engine is read-only** in this plan. The existing AppDaemon -> HA -> pump arrow is deployed; the plan-consuming dotted arrow is a **proposal**, not a current implementation. Battery dispatch has no actuation arrow.

### Components, boundaries and proposed module locations

| Component | Input / responsibility | Output | Reuse or new |
|---|---|---|---|
| **Acquisition and freshness** | HA state/event ordering, sensor health, observed plant mode, weather forecasts and historical telemetry | Timestamped validated `ClimateObservation` | Reuse `service/ha_client.py`, `entity_store.py`, `history.py`; extend role mapping |
| **Price outlook and time base** | Published PVPC and tariff assumptions at the decision timestamp; UTC quarter-hours, local labeling, DST | `PriceCurve`, known/estimated/provenance flags | Reuse `service/pvpc.py` and `service/timegrid.py`, no second parser |
| **Building thermal model** | Indoor/outdoor temperature, heat-pump output, off/on intervals and passive gains | Predicted indoor temperature trajectory and confidence/fit diagnostics | New `climate/thermal.py` |
| **COP model** | Heat/electrical measurements, ambient/flow temperature, operation mode and defrost | Estimate marginal electricity per delivered thermal kWh | New `climate/cop.py`; bounded fallback |
| **DHW model** | Tank temperature, reheating events, tank loss, demand windows, hardware hygiene schedule | Tank trajectory and DHW availability constraints | New `climate/dhw.py` |
| **Climate planner** | Frozen snapshot, comfort bands, thermal models, price/forecast horizon, compressor constraints | `ClimatePlan` with rationale, no direct actions | New `climate/planner.py`, `climate/contracts.py` |
| **Climate scheduler/runtime** | Replans triggered by slots and meaningful input changes; solver budget/cancellation | Shadow decisions, `GET /climate/plan`/status/metrics | Extend existing `service/runtime.py` and `service/api.py` behind a feature flag |
| **AppDaemon actuator** | Current selected intent, plan TTL, HA control ownership and safety telemetry | Verified HA/HeishaMon command or conservative fallback | Refactor in `homelab`; use existing writable HA connection, not battery-optimizer credentials |
| **Forecast exchange** | Versioned predicted HVAC + DHW electric kWh by slot, quality/provenance | Battery load-forecast feature; never a device command | Extend battery load forecaster and `structured_forecast.py` replay/evaluation path |

Treat file paths as **candidate boundaries**, not a mandate to split every component into its own class or microservice. Refactor shared runtime only when a second implementation requires it. Preserve offline/live use of the same domain kernels and input semantics.

### Separation from the battery domain

The battery planner manages SOC, conversion efficiency, import constraints and dispatch intents; the climate planner manages room/tank temperatures, thermal energy, COP, comfort and compressor operation. Their optimizers must remain independent initially. Reusing SciPy/HiGHS does not imply reusing `battery.py` or `oracle.solve_oracle` as the thermal solver.

Avoid a feedback loop: the battery planner consumes a **frozen** climate electrical-demand forecast for a given generation. The climate plan does **not** assume a future battery discharge is guaranteed or arbitrarily free. Joint cross-domain optimization would require a new ADR and counterfactual evaluation.

## 3. Data and model design

### 3.1 Input inventory and validation

| Source role | Candidate HA data | Model / quality checks |
|---|---|---|
| Indoor comfort | Existing LocalTuya room thermostats (bedroom, hall-kitchen and offices) | Select representative sensor(s); flag stale/unknown, sensor offset, open-window events and heterogeneous zone temperatures |
| Outdoor temperature | `sensor.aquarea_outside_ambient_temperature` (verify actual registry), `weather.forecast_home` hourly forecasts | Sensor/forecast reconciliation and forecast issue time; do not substitute realized future weather in backtests |
| Heat and electricity | Aquarea heating produced/consumed power, compressor state, defrost; Shelly EM channel 2 | Validate energy integrals, W/kW units, zero/negative readings, standby, compressor start count and DHW separation |
| Tank heat | Aquarea tank temperature/target, force DHW and operating mode | Exclude draws, reheats and anti-legionella from standing-loss fitting; no assumed tank sensor identity |
| Electricity price | `sensor.esios_pvpc` | Reuse published/next-day prices, `_d` fallback-hour handling, pretax-vs-billed tariff reconciliation; permit negative prices |
| Occupancy/comfort policy | HA schedules/helpers and manual overrides | Confirm actual schedules; explicit time-of-day comfort bands and authorized setback, not inferred from device names |
| Hardware bounds | Aquarea/HeishaMon capabilities and configured minimum run/rest times | Verify manufacturer/install-specific limits before enabling control; preserve pump-native frost/hygiene protection |

Store UTC-aware, half-open 15-minute slots with provenance, sample age, availability cutoff and quality flags. Preserve raw inputs; train only on measurements available **as of** the historical decision. Recorder may purge after 10 days, so use the existing archival pipeline rather than training solely against short HA history.

Initially, model one representative comfort zone **only when** the room sensors/heat distribution justify that simplification. If independent valve states or rooms disagree materially, choose a conservative comfort aggregation or defer automated optimization until topology is understood.

### 3.2 Learn physics progressively

Start with an identifiable first-order model (heating):

~~~text
T_in[t+1] = T_in[t] + dt/C *
            (Q_heat[t] - UA*(T_in[t]-T_out[t]) + Q_passive[t])
~~~

Learn/estimate `UA` (heat-loss coefficient), `C` (effective capacity/time constant) and bounded passive gain from filtered heating/coasting intervals. Distinguish heat delivered by the machine from electricity consumed; COP is a function of ambient/flow temperature and operating conditions, not a universal constant. Compare a simple temperature-conditioned COP baseline with a fitted model, rejecting implausible values.

DHW uses a **different** tank-loss/draw model and demand windows; a household draw is not a building heat-loss observation. Pump-side safety and legionella cycles remain authoritative. A slab/two-state building model is deferred until held-out residuals justify it; cooling requires its own model and dew-point constraints.

Models must be versioned, bounded, replayable and persisted on the existing PVC **only inside the planning runtime**. Do not store learned parameters solely in AppDaemon's current `emptyDir`.

### 3.3 Optimization and solver behavior

Optimize a 24–48 h lookahead on the existing 15-minute grid, bounded by actual **published** prices and reliable forecast coverage; unknown later prices must be labeled as estimates with an uncertainty policy, never silently treated as known. A terminal heat-storage value/penalty must prevent artificially cheap end-of-horizon cooling.

Indicative objective, with hard physical and safety constraints and softened comfort only for unavoidable infeasibility:

~~~text
min Σ_t [ actual_import_price[t] * predicted_electric_kWh[t]
        + comfort_deviation_penalty[t]
        + compressor_start_penalty[t]
        + terminal_energy_penalty ]
~~~

Constraints include room minimum/maximum comfort bands, maximum/minimum thermal output or admissible on/off states, manufacturer-verified minimum on/off durations, DHW availability windows and no contradictory DHW+space-heating demand on a single compressor. Distinguish fixed hardware protection from tunable comfort preferences. A negative import price must never create an unbounded incentive to overheat the house or tank.

Start with a temperature-aware rule heuristic and a small bounded solver; adopt mixed-integer/advanced formulations only if the simpler method demonstrably fails. Preserve the original cheap-hours policy for comparative baselines and rollback.

## 4. Plan publication and executor contract

The first implementation is **shadow only**. Introduce `GET /climate/plan` under the existing internal FastAPI service, with optional read-only status and metrics. Existing `/plan`, `/status` and `/metrics` battery semantics remain compatible.

Define a versioned, immutable `ClimatePlan` contract containing:

| Field | Semantics |
|---|---|
| `schema_version`, `plan_id`, `generation` and `input_hash` | Detect incompatible, replaced or duplicated decisions |
| `mode` | `shadow` (non-actionable) or, after a separate explicit rollout, `active` |
| `created_at`, `valid_from`, `expires_at` | UTC-aware plan freshness; never execute an old intent |
| `price_provenance`, `weather_provenance`, `model_version` | Audit input availability and uncertainty |
| `slots[]` with UTC start/end, action, expected electric kWh, heat/DHW allocation, predicted temperatures and reason | Explain current decision and make load demand reusable by battery planning |
| `constraints` and `readiness` | Explain rejected or degraded plans; unknown is not safe-by-default |

The read-only endpoint publishes **proposals**, not a write API or a claim that a plan was executed. Return a clearly invalid/unavailable state if inputs or solver results are stale. Keep internal access scoped; never expose home occupancy or temperatures publicly.

### Executor state machine

| Mode | Climate actuator behavior |
|---|---|
| **Disabled / manual override** | Do not issue optimization writes; hand control back to the user/pump according to an explicit ownership policy; cancel only owned timers |
| **Legacy** | Existing AppDaemon price schedule remains the authoritative writer |
| **Shadow** | Read and compare proposed plans, but **never** issue plan-derived writes |
| **Active (future gate)** | Execute only the current valid, acknowledged intent while supervising temperatures, hardware mode and minimum run times |
| **Degraded / fault** | Expire optimization authority; transition through a documented safe controller/hardware fallback rather than assuming the last HA helper state is safe |

Only one climate writer may be authoritative at a time. When replacing independent legacy climate/DHW timers, transition both to a single coordinated executor or retain DHW hardware ownership explicitly—never run overlapping force-DHW schedules from two controllers. AppDaemon may call HA services, but must confirm the *reported* HeishaMon operating mode; `input_select` reflecting a requested option is insufficient. Every actuation needs deadline, bounded retries, idempotent intent handling and mismatch alerting.

Do **not** equate `disabled`, transport loss or optimizer failure with `off`. Document how the pump's native thermostats/frost protection continue to operate, how a stale heating request is revoked, and what manual takeover does during a restart. The proposed climate plan consumer is a separately reviewed homelab PR; no runtime control changes in this planning PR.

### Packaging/deployment specifics

- The homelab chart packages `appdaemon/apps/*` directly into a ConfigMap and copies it to an `emptyDir` at startup. If AppDaemon code later becomes a Python package, update chart globs/mount paths and import tests; do not assume nested files appear automatically.
- Existing AppDaemon is `replicas: 1`, but a single replica is **not** a reliable single-writer lock across restarts/reconciliations. Reconcile actual pump state and plan generation on startup.
- Keep the read-only HA credential and `ha_guard.py` exactly restricted in the battery-optimizer process. The future AppDaemon plan consumer uses its existing appropriately scoped write-capable integration, not the battery optimizer's token.
- Run climate solver work off the shared event loop; bound runtime and queueing. Do not let climate solves starve battery slot closing, price archiving, API liveness, or existing battery MPC work. Benchmark CPU/memory on the current Deployment before raising limits. If demand warrants it, separate worker pools inside the process, not new pods by default.
- Keep GitOps ownership of image digest, explicit enable flags, secrets, monitoring and rollbacks in `pando85/homelab`. No auto-promotion from shadow to active on image or ConfigMap update.

### Packaging, release and rename migration

**Near-term: retain the current release name and deploy the same image for its two existing workloads.** Do not introduce `climate-optimizer` or an additional image build. The package contains domain-specific `battery` and `climate` planning modules, with separate CLI commands but a common dependency set and OCI release.

| Kubernetes resource | Container image | Process | Operational separation |
|---|---|---|---|
| Existing `battery-optimizer` Deployment | Existing pinned `ghcr.io/pando85/battery-optimizer` OCI image | `battery-optimizer serve`, eventually scheduling both battery and climate planners in-process | Read-only HA token; independent planner snapshots, readiness, solve budgets and metrics |
| Existing `battery-optimizer-pipeline` CronJob | **Same image and pinned digest** | Existing `extract`, `validate`, `build`; future offline `climate-evaluate`/training mode if justified | Short-lived batch process; dataset/PVC and historical API access only |
| Existing AppDaemon Deployment | Its **existing separate AppDaemon image** | Existing `climate_control` / `dhw_control` code; later a thin `ClimatePlan` reader/executor | Sole current HA/HeishaMon climate writer, with manual override and rollback |

Planned CLI shape after the optional project-wide rename (illustrative, **not implemented**):

~~~console
home-energy-optimizer serve
home-energy-optimizer extract
home-energy-optimizer validate
home-energy-optimizer build
home-energy-optimizer benchmark
home-energy-optimizer climate-evaluate   # proposed
home-energy-optimizer climate-actuator   # future, separately authorized
~~~

**Optional replacement for AppDaemon:** package `climate-actuator` in the same OCI artifact but run it as a **separate Kubernetes Deployment/process**, with its own service account, restricted HA write credential, supervised reconciliation and exclusive command ownership. The read-only `serve` process must still reject all actuation calls. Do not run the actuator as a side effect of starting `serve`, embed its writable HA client in the planner, or deploy it until the C5/retirement-specific gates have been reviewed. When enabled, the intent is to replace AppDaemon's actuation Deployment, **not** to add a permanently competing writer. The separate battery/Omnibattery controller remains independently owned.

**Staged naming/operational migration** (a separate follow-up PR or PR series; do not combine with climate runtime development):

1. **Repository URL:** Rename GitHub `pando85/battery-optimizer` to `pando85/home-energy-optimizer` once the architectural documentation is approved; audit consumers, GitHub Actions, release tooling, issue/PR links and configured source URLs. Do not rely solely on GitHub's old-URL redirects.
2. **Python and CLI compatibility:** Decide an import/entrypoint compatibility period before switching `battery_optimizer` to `home_energy_optimizer`. Preserve running scripts, packaging metadata and existing command usage until migrated in tested increments.
3. **OCI identity and supply chain:** Publish and scan the new GHCR image name separately, ensure actual manifest availability and signed/provenance requirements, then update homelab's image **by digest**. GHCR names are not guaranteed to redirect after a repository rename. Do not remove old image tags before pin consumers move.
4. **GitOps rollout:** Update the optimizer Deployment and pipeline CronJob to the **same verified new digest**, keeping their commands, PVC/mounts, ServiceMonitor/PrometheusRule, secrets, service names and ArgoCD identities stable unless a later explicit migration demands a change.
5. **Persistent dataset/API compatibility:** Preserve content-addressed manifests, `dataset_id`, `code_version`, archived source hashes, existing `/plan` and `/status` contracts and historical report reproducibility. Record any migration alias or metadata format version rather than rewriting historical artifacts.
6. **Rollback:** Keep a tested reference to the prior image digest, command and GitOps configuration. Repository rename or branding must **not** toggle battery or climate actuation.

The target is **one image build and release pipeline**; existing battery planner Deployment and pipeline CronJob remain separate processes/workloads, and AppDaemon remains a separately built image until a distinct actuator-replacement decision. Creating the actuator entrypoint does not authorize running it, and this plan intentionally avoids introducing a third planner Deployment.

## 5. Climate ↔ battery integration

The existing `structured_forecast.py` (LF-1) **already** replays the AppDaemon policy and estimates HVAC electricity offline. Do not duplicate or delete it when adding climate MPC. Make load forecast provenance explicit:

1. Legacy periods: forecast/replay the versioned cheap-hours and DHW rules as currently implemented.
2. Shadow period: record the candidate `ClimatePlan` separately, alongside **actual legacy** execution. Do not label a shadow plan as realized electricity demand.
3. Active climate period (future): use the published **committed/executable** climate schedule where known, plus residual household demand; account for plant deviations separately.
4. Validate that `P_house = P_nonHVAC + P_climate + P_DHW` is a mutually exclusive decomposition: if the forecast baseline already contains heat-pump load, **subtract/re-fit** it before adding a new climate load schedule. Never add predicted HVAC load twice.
5. Keep battery optimization separately constrained by SOC, grid import, conversion losses and Omnibattery's eventual physical safety rules. An inactive/simulated battery cannot make real HVAC electricity free.

Forecast interfaces should carry UTC slot IDs, kWh, quality and model/plan version, with a clear information cutoff. No cyclic re-optimization of climate and battery without a later architecture decision and measured benefit.

## 6. Iteration sequence (separately reviewable PRs)

| Stage | Repository | Deliverable | Exit gate |
|---|---|---|---|
| **C0 — Correctness audit and stabilization** | homelab | Fix legacy set-state vs HA service calls, real HeishaMon ack/timeout, timer cancellation and manual ownership, recursive retry, DST, stale PVPC and fallback safety; preserve existing user-facing toggles | Tests cover invalid/missing price, negative price, both DST days, restart/disable/manual override, command mismatch and lost HA |
| **C1 — Shared observation contract** | battery-optimizer + homelab configuration | Map verified room, tank, compressor, power and weather data into the existing entity store/history/archives; quality dashboard; no control writes | Enough measured history and documented gaps to fit a model without inventing data; battery baseline regression-free |
| **C2 — Thermal and DHW identification** | battery-optimizer | Pure model(s), fitting, safe parameter envelopes, recorded versions and chronological validation | Held-out trajectory error and bias outperform a defined simple/persistence baseline; bad sensors trigger model fallback |
| **C3 — Climate shadow planner** | battery-optimizer | Temperature-aware heuristic then climate MPC, versioned plan, read-only GET, shadow metrics, executor not connected | Deterministic, physically feasible plans, bounded solve latency, no interference with existing battery runtime |
| **C4 — Evaluation and plan handoff** | battery-optimizer, then homelab | Compare cheap-hours, comfort-rule and MPC using identical historical decision-time inputs; add AppDaemon **shadow** plan reader and telemetry | Comfort, model-cost and cycling comparisons recorded; shadow reads cannot actuate; tested expires/parse errors |
| **C5 — Gated heating control** | homelab (plus explicit optimizer flag/config) | Single writer for heat and DHW; controlled schedule activation with acknowledgement and safety fallback | Manual supervised pilot, recovery and hardware-state checks, acceptance of temperature and cycling behavior; easy switch back to legacy |
| **C6 — Demand forecast exchange** | battery-optimizer | Expose committed climate electric load to the battery forecaster; validate residual and historical reconciliation | No double counting; battery shadow savings/forecast regret improve on matched windows; no battery writes |
| **C7 — Optional evolution** | both, new ADR as needed | Multi-zone/slab model, DHW optimization refinements, flow modulation, cooling, joint MPC, AppDaemon retirement | Separate evidence and safety/comfort decision per addition; do not presume adoption |

**Ordering:** C0 can begin independently in `homelab`; C1 and C2 can run while legacy heating stays active. C3–C4 must precede any C5 actuation. C6 can start read-only before C5 but must distinguish hypothetical shadow schedules from committed ones. The battery Phase D commissioning/authorization is **independent** of C5. No timeline, energy savings percentage or thermal parameter is assumed.

### Recommended first implementation PRs

- **homelab — C0:** safe/observable HeishaMon mode change, a single explicit timer/fallback ownership policy, and exhaustive time/price tests. Preserve current schedules by default.
- **battery-optimizer — C1:** typed validated observations and instrumented datasets using existing HA client/Prometheus archive; model input audits for room, tank and HVAC electric/thermal power.
- **battery-optimizer — C2:** offline first-order thermal baseline, COP validation and test fixtures; no runtime configuration or actuator.
- **battery-optimizer — C3:** opt-in shadow planning and `GET /climate/plan`; prove battery shadow behavior unaffected.
- **homelab — C4/C5:** explicit read-only plan consumption then a separately approved, reversible controlled heating rollout.

## 7. Validation, metrics and non-negotiable safety

### Evaluation rules

- **Causality:** replay prices as published at the decision, use weather *forecast issue time* (not realized future weather), measured building/tank state as-of the slot and chronological train/test splits.
- **Baselines:** existing AppDaemon policy; same policy with temperature guard; first-order heuristic; MPC; perfect foresight only as an explicitly labeled diagnostic upper bound.
- **Costs:** PVPC marginal purchase prices with verified tariff/tax assumptions, electricity measured by mode, COP and transition/cycling cost assumptions; separate fixed charges. No generated savings figure is accepted as a bill measurement.
- **Comfort:** report degree-hours below minimum/above maximum, absolute worst excursion, per-room disagreement, recovery delay and expected-versus-realized temperature trajectory.
- **Machine:** starts/day, short cycles, compressor runtime, high-flow operation, defrost effects, forced DHW cycles and unsuccessful acknowledgements.
- **Model:** 15/30/60-minute and full-horizon temperature MAE/bias, DHW temperature trajectory, COP error, confidence, fit drift and invalid-sensor counts.
- **Operational:** valid-plan coverage, stale input time, solution feasibility/failures/latency, cancelled/superseded plans, applied-vs-observed mode, fallback frequency, service CPU/memory, impact on battery MPC and price/weather archival.
- **Promotion:** predeclare comfort bounds, minimum on/off constraints, accepted model errors, runtime budgets, rollback thresholds and economic improvement criteria based on observed baseline performance; do not invent thresholds now. Regressions in comfort or plant safety override nominal cost gains.

### Mandatory failure-scenario tests

| Scenario | Required expectation |
|---|---|
| Europe/Madrid spring-forward and fall-back, including duplicated `02h_d` | UTC-aligned 92/100 quarter-hour slots with correct tariff association |
| Missing, stale, negative and partly published PVPC prices | Correct availability/error attribution, no random coercion and explicit policy for estimated tail |
| Stale room/outdoor/tank sensors; open windows; contradictory zone temperatures | No false thermal learning; conservative limits or no optimizer actuation |
| Solver timeout, CPU saturation, failed persistence, restart or overlapping generations | No old/stale plan execution, no battery shadow starvation, bounded recovery |
| HA disconnect, wrong helper state, MQTT acknowledgement mismatch | No infinite wait or false command-success metric; hardware mode reconciled |
| Manual override, disable during heating, competing DHW event | Deterministic owner transition; no orphaned timer or conflicting writer |
| Defrost, compressor short cycle, tank draw, pump hygiene/legionella cycle | Machine/hygiene safeguards remain authoritative, do not train on misleading intervals |
| Cooling requests while heating-only planner active | Refuse to produce cooling control; no unsafe water-temperature or dew-point commands |
| Battery SoC absent or simulated, missing grid measurement | Existing battery shadow service unchanged; climate cannot turn hypothetical battery discharge into real savings |

### Rollback and operational handoff

For each control-stage PR, specify the HA helper/feature flag that restores **legacy** ownership, the expected actual HeishaMon mode after handoff, who controls DHW, the actuator-reconciliation sequence and how to verify the pump after stopping AppDaemon. A stopped process or expired plan is not evidence that the heat pump stopped or entered a safe thermostat mode. Keep native pump protections enabled. Alert on unexpected state and investigate rather than blindly retrying unlimited commands.

## 8. Acceptance criteria and unresolved decisions

The architecture is ready for implementation when this document and ADR are reviewed, but not when the physical controller is ready for actuation. Before C1, confirm:

- [ ] Which room(s) and LocalTuya thermostat measurements define comfort, and whether the radiators/UFH/valves are independently controlled.
- [ ] Actual HeishaMon sensor IDs and the agreement between HeishaMon energy and the Shelly EM aerothermal electrical branch, including idle/DHW.
- [ ] Panasonic operating mode, room/zone sensor configuration, manufacturer-safe cycling bounds and native frost/hygiene behavior.
- [ ] User-approved heating and DHW comfort bands and manual/holiday/disable semantics.
- [ ] Whether battery-optimizer's current deployment can accommodate thermal-data ingestion and solves without interrupting existing slots/archival.
- [ ] Source of reliable historical weather forecasts (existing `weather.forecast_home` archive and limits) and price snapshots for causal replay.
- [ ] Measurable pilot/rollback criteria agreed before the first active heating plan.

**Completion definition:** the user can compare a causal, reproducible cheap-hours baseline with temperature-aware and MPC forecasts; the climate planner runs read-only on the existing service; AppDaemon executes only explicitly authorized, current and verified plans; DHW and heating share one command owner; battery planning consumes climate demand without double counting; and any upstream failure has a tested comfort-preserving fallback. Removing AppDaemon or enabling battery hardware dispatch is **not** required for this plan to be complete.

## 9. References

- [ODIN (documentation and user manual; solver source not published)](https://github.com/gekkekoe/heatpump-optimizer)
- [tvofi/heatpump_optimizer (MIT-licensed, inspectable MPC reference)](https://github.com/tvofi/heatpump_optimizer)
- [EMHASS (thermal/energy scheduling reference)](https://github.com/davidusb-geek/emhass)
- [Homelab AppDaemon controls](https://github.com/pando85/homelab/tree/master/apps/home-assistant/appdaemon/apps)
- [Homelab battery optimizer deployment](https://github.com/pando85/homelab/blob/master/docs/deployment/battery-optimizer.md)
