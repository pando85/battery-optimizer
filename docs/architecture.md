# Architecture

The component diagram is in the [README](../README.md#architecture).

The package has two halves. The offline pipeline (`extract`, `validate`, `build`, `benchmark`, `report`;
see the [CLI reference](cli.md)) ingests Prometheus and Home Assistant history into an immutable,
content-addressed store and scores dispatch policies on it. The resident [shadow service](service.md) reads
Home Assistant and Prometheus, runs the same MPC kernel live and records plans that are never sent anywhere.
Decisions are recorded in [`adr/`](adr/); the production architecture is
[ADR-0004](adr/0004-battery-optimizer-production-architecture.md).

## Planning components

The README [ASCII component diagram](../README.md#architecture) is a **logical architecture**:
the four named DECIDE components are responsibilities within one Python package, not four
microservices, new runtime processes or existing four plugin interfaces. The offline tools and
the resident shadow service reuse the same battery/optimization primitives, with intentionally
different information availability and orchestration rules. **No code refactor is required**
to make these boundaries explicit.

| Component | Owns / input | Current implementation and output |
|---|---|---|
| **[Load Forecaster](planning/components/load-forecaster.md)** | Predict future **gross household demand** (kWh per UTC slot) from causal load history | `load_forecast.py` registers the named forecasters behind one call contract (`run_forecaster`); `service/history.py` (`forecast_load`, `LoadForecast`) runs `planning.load_forecaster` through it; `rolling_horizon.py` supplies causal/frozen offline forecast modes. Live baseline is the trailing 28-day same-slot mean with a local-midnight information cutoff. |
| **[Price Outlook](planning/components/price-outlook.md)** | Supply prices **known at decision time**, publication timestamps and the available priced horizon | `service/pvpc.py` (`parse_pvpc_state`, `PriceCurve`) reads published PVPC attributes and handles Europe/Madrid DST; `rolling_horizon.py` implements offline price-publication/perturbation experiments. **The live service does not forecast unpublished prices:** its horizon ends at the last published slot. |
| **[Battery Model](planning/components/battery-model.md)** | Supply physical state and constraints: capacity, SOC bounds, charge/discharge power, one-way efficiencies, grid import/export accounting | `battery.py` (`BatterySpec`, `Horizon`, numeric physics/validation), battery configuration resolved via `benchmark.resolve_battery_assumptions`; `service/virtual_battery.py` replays shadow decisions. The shipped **0.80 round-trip efficiency is an assumption**, not a device measurement. |
| **[Dispatch Optimizer](planning/components/dispatch-optimizer.md)** | Find feasible, low-cost charge/discharge decisions from **Load Forecast + Price Outlook + Battery Model**, current SOC and economic assumptions | `service/planner.py` freezes a `PlanningSnapshot`, invokes the shared `oracle.solve_oracle` (SciPy/HiGHS), and produces `DispatchPlan`/`DispatchIntent` via `dispatch.py`; `rolling_horizon.py` is the offline policy/replanning harness. The service re-solves every 15 minutes and on selected changes, but remains **shadow-only**. |

### Boundaries and information flow

- The **Load Forecaster** emits time-aligned demand predictions. It does not choose battery
  actions or know future measured consumption. AppDaemon schedules and appliance-state
  history are **potential future features**, not currently incorporated in its baseline.
- The **Price Outlook** distinguishes published prices from possible estimates. There is
  currently no live price prediction; offline simulations with noisy future prices are
  not evidence of a production forecasting service. Persist **decision-time publication
  snapshots** before evaluating any new estimator against future outcomes.
- The **Battery Model** describes physics and allowable state transitions. It is an
  independent input to the optimizer, **not a third forecast** nor a downstream stage
  of either forecaster. When the real battery is commissioned, measured SOC replaces
  `soc_source: simulated`; efficiency/power characteristics must be measured before
  treating model upgrades as calibrated.
- The **Dispatch Optimizer** does not ingest raw HA events directly and does not own
  the physical regulation loop. The service assembles a frozen, hashable
  `PlanningSnapshot`, solves off the asyncio event loop and records intents.
  `DispatchPlan` is **not** a battery command or proof of achieved power.
- **SENSE** includes the HA/Prometheus read adapters, freshness and offline
  extraction/validation. **LEARN** contains immutable offline benchmarking, the
  shadow sink, virtual replay, reports and operational metrics. Neither controls
  hardware. **ACT** remains a future safety guard and HA/Omnibattery adapter;
  Omnibattery must retain sole Modbus ownership and fast Shelly-based feedback.

### Evolution: improve one component at a time

The table below is a **navigation summary**, not a second implementation checklist. Each
component has its own detailed roadmap (linked above) with milestones, measurable
acceptance tests, dependencies and stop rules. The existing
[implementation plan](planning/home-battery-optimizer-implementation.md) remains the
source of truth for cross-component production/hardware safety gates; the
[system roadmap](planning/energy-battery-roadmap.md) remains a global status overview.

| Component | Current baseline | Next candidate | Later / acceptance evidence |
|---|---|---|---|
| **Load Forecaster** | Causal trailing-slot-mean, 28 days | Condition on known server base load, AppDaemon schedules and HA appliance states | Compare LightGBM/CatBoost and P10/P50/P90 models using chronological holdouts; **promote only on out-of-sample net EUR savings/regret**, not MAE alone. Prevent target leakage and record feature availability at decision time. |
| **Price Outlook** | Published PVPC values only, with UTC/DST normalization | Archive decision-time publication snapshots and audit their completeness | Predict only the *unpublished* horizon where useful; compare estimates, probabilistic scenarios and **decision regret** to the published-only baseline. Never replace published prices with guesses. |
| **Battery Model** | Assumed constant RTE 0.80, nameplate power/capacity and SOC constraints | Measure AC efficiency vs power/SOC and standby draw; characterize BACKUP / AC bypass | Introduce calibrated rate-dependent or piecewise-linear physics **while retaining solver feasibility**; validate SOC, power and energy error on held-out cycles, including uncertainty. |
| **Dispatch Optimizer** | 15-minute shadow MPC using HiGHS, published price horizon and explicit terminal/throughput knobs | Calibrate throughput/degradation cost, residual energy value, import ceiling and replan triggers | Explore robust/scenario MPC only if it improves feasible net EUR savings, SOC/risk constraints and solver latency in like-for-like backtests and shadow comparison. |

**Iteration rule:** treat these as named seams first, not a mandate to create
`LoadForecaster`/`PriceOutlook`/`BatteryModel`/`DispatchOptimizer` classes or move
files now. Make an implementation-level interface change only when a second
implementation or measurable benefit warrants it. New algorithms must work
through the **same shared physical/solver kernel and causal input semantics**
used by benchmark and shadow operation; report forecast metrics *and* dispatch
economics with matching datasets, costs and terminal SOC assumptions.

**Safety and commissioning:** HA/Omnibattery battery telemetry is absent from
the current installation, so the shadow service runs with simulated SOC.
The ACT stage (device writes, safety/failover, HA-loss behavior) is neither
implemented nor authorized; [ADR-0004](adr/0004-battery-optimizer-production-architecture.md)
and the [production plan](planning/home-battery-optimizer-implementation.md)
keep its commissioning gate separate.

## Modules with no CLI surface

These have no CLI surface (or `hack/` only), deliberately:

- `dispatch.py` — the dispatch **intent contract** (see [Dispatch intent contract](#dispatch-intent-contract-dispatchpy)).
  Typed, validated, canonically serialized `DispatchIntent` / `DispatchPlan`. No writer, no I/O,
  no subcommand. This package still actuates nothing.
- `artifact_integrity.py` — output-manifest sidecars for the benchmark and report stages
  (see [Artifact integrity](#artifact-integrity-artifact_integritypy)).
- `economic_sensitivity.py` — pure rescoring / re-optimisation under user-supplied throughput-cost
  and terminal-value sweeps (see [Economic sensitivity](#economic-sensitivity-economic_sensitivitypy)). Experiment only, not part of the
  canonical benchmark pipeline.
- `replay_runtime.py` — replay-only asyncio dispatch runtime with a recording sink. No writer,
  no actuation, no HA adapter, no wall clock. See [Replay runtime](#replay-runtime-replay_runtimepy).

## Dispatch intent contract (`dispatch.py`)

Typed, validated, canonically serialized `DispatchIntent` / `DispatchPlan`. Clock is a **required
caller argument with no default**. **No writer, no I/O, no CLI subcommand.** See module docstring.

## Artifact integrity (`artifact_integrity.py`)

**Status: complete — committed and wired into the canonical writers.** Output-manifest sidecars for benchmark
and report stages: `finalize_run_outputs` runs after the three benchmark artifact files, and the report
reader verifies the benchmark sidecar before consuming the payload and writes its own sidecar after.
**Detects integrity, not authenticity** (no signature claim). Legacy runs without a sidecar are
accepted but marked unverified. CLI: `hack/verify_artifacts.py <run-dir> [--report]`. See module
docstring.

## Economic sensitivity (`economic_sensitivity.py`)

**Status: complete — committed, experiment module.** Sweeps under **user-supplied hypothetical**
throughput costs and terminal values, re-optimizing at every parameter pair rather than rescoring a
fixed schedule. Break-even uses the **closed-form algebraic intersection** of the two schedules'
cost lines (accounting for unequal initial SOC), not a binary search. **Not a physical aging model,
no calibration.** Terminal optimization vs scoring separated. Runner: `hack/economic_sensitivity.py`.
See module docstring.

## Replay runtime (`replay_runtime.py`)

**Status: complete — committed. NOT deployed, NOT a resident service, NO HA adapter, recording
only.** Deterministic plan content; record ids are per-run random (`generation_id` is `uuid.uuid4()`). Caller-supplied time; recording-only sink; injectable planner.
Demo: `python -m battery_optimizer.hack.replay_runtime_demo`. See module docstring.

## Shadow service (`service/`)

**Status: Phase C (PROD-C1 to C3) built — shadow only, no actuation path.** `battery-optimizer serve`
subscribes to Home Assistant (read-only allowlist, reconnect and full resync), reads load history from
Prometheus, solves the offline MPC kernel every 15 minutes in a bounded worker, records time-bounded
shadow intents and exposes `/health/live`, `/health/ready`, `/status`, `/plan` and `/metrics`.
`battery-optimizer shadow-report` compares the recorded plans with the native battery behaviour.
No battery entity exists in Home Assistant yet, so the committed configuration runs a virtual battery
(`soc_source: simulated`) until the Omnibattery entities are mapped. Architecture, configuration,
metrics, failure modes and the comparison method: [service.md](service.md).
