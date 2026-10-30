# Architecture

The component diagram is in the [README](../README.md#architecture).

The package has two halves. The offline pipeline (`extract`, `validate`, `build`, `benchmark`, `report`;
see the [CLI reference](cli.md)) ingests Prometheus and Home Assistant history into an immutable,
content-addressed store and scores dispatch policies on it. The resident [shadow service](service.md) reads
Home Assistant and Prometheus, runs the same MPC kernel live and records plans that are never sent anywhere.
Decisions are recorded in [`adr/`](adr/); the production architecture is
[ADR-0004](adr/0004-battery-optimizer-production-architecture.md).

## Proposed climate-planning extension

[ADR-0005](adr/0005-shared-energy-planning-climate.md) proposes reusing this **read-only** resident runtime for a domain-specific thermal/DHW planner rather than creating a second ingest/optimization deployment. It keeps AppDaemon's existing HeishaMon actuation separate and explicitly gated. No climate planner, ClimatePlan endpoint or AppDaemon plan consumer is implemented yet; see the [phased climate plan](planning/climate-optimizer-implementation.md). Battery dispatch remains shadow-only and unchanged.

## Planning components

The README [ASCII component diagram](../README.md#architecture) is a **logical architecture**:
the four named DECIDE components are responsibilities within one Python package, not four
microservices, new runtime processes or existing four plugin interfaces. The offline tools and
the resident shadow service reuse the same battery/optimization primitives, with intentionally
different information availability and orchestration rules. **No code refactor is required**
to make these boundaries explicit.

| Component | Owns / input | Current implementation and output |
|---|---|---|
| **[Load Forecaster](planning/components/load-forecaster.md)** | Predict future **gross household demand** (kWh per UTC slot) from causal load history | `load_forecast.py` registers the named forecasters behind one call contract (`run_forecaster`); `service/history.py` (`forecast_load`, `LoadForecast`) runs `planning.load_forecaster` through it; `rolling_horizon.py` supplies causal/frozen offline forecast modes; `structured_forecast.py` registers the offline structured forecaster `structured` (LF-1: replayed AppDaemon heat-pump schedule, fitted energy per hour, residual forecaster); `forecast_eval.py` is the offline evaluation harness (`forecast-eval`). Live baseline is the trailing 28-day same-slot mean with a local-midnight information cutoff. |
| **[Price Outlook](planning/components/price-outlook.md)** | Supply prices **known at decision time**, publication timestamps and the available priced horizon | `service/pvpc.py` (`parse_pvpc_state`, `PriceCurve`) reads published PVPC attributes and handles Europe/Madrid DST; `price_outlook.py` registers the offline price outlooks behind one call contract (`run_price_outlook`: `exact`, `published_only`, `persistence`) that `rolling_horizon.py` plans on, with `price_eval.py` as their evaluation harness (`price-eval`). **The live service does not forecast unpublished prices:** its horizon ends at the last published slot. |
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
- `omie_prices.py` — read side of the OMIE day-ahead store (`extract --source omie`, see
  [CLI](cli.md#extract---source-omie-omie-day-ahead-prices)): ES prices on the 15-minute UTC grid,
  per-day `first_observed_at` and the as-of rule. A library function for the price estimator; it
  never writes and is not a `build` input.
- `replay_runtime.py` — replay-only asyncio dispatch runtime with a recording sink. No writer,
  no actuation, no HA adapter, no wall clock. See [Replay runtime](#replay-runtime-replay_runtimepy).
- `climate/` — offline data layer of the climate optimizer (C2). See
  [Climate subpackage](#climate-subpackage-climate).

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

## Climate subpackage (`climate/`)

**Status: C2 PR 1 — configuration, data loader and audit; no model, no runtime, no actuation.** The
offline half of the [climate optimizer plan](planning/climate-optimizer-implementation.md#c2-design-and-predeclared-gate).
It reads the climate feature set (`build-features --signal-set climate`) and the PVPC prices of a
canonical dataset, and never writes into either.

- `config.py` and [`config/climate.yaml`](../config/climate.yaml): the heat-v1 comfort band, training
  filters, parameter envelope, model-acceptance criteria and gate thresholds, committed before any
  model result exists. Loading is strict (unknown, missing or duplicated keys, wrong types and
  inconsistent ranges are errors) and the file's SHA-256 identifies it in every result.
- `data.py`: `load_climate_data` verifies the feature set through its manifest and returns
  `ClimateData`, numpy arrays on the 15-minute UTC grid with a local-day partition (92 / 96 / 100
  slots): `T_air` (comfort-room mean), `Q_heat` with its daily
  cross-check against the energy counter, the DHW and cooling masks, the inferred valve state, the
  clear-sky solar proxy, the legacy permission `u` replayed through
  `structured_forecast.replay_mask_for_day` (decided 40 s into each local day) and one boolean per
  training filter. `u` is `NaN` (and `u_known` false) on every day without a replayed schedule,
  including disabled-rule days, which are excluded; u-driven code must use `u_known` only.
  The Aquarea outdoor sensor reads low while the compressor runs, so the model's `t_out_debiased`
  keeps only slots with the compressor off for at least 30 minutes and interpolates linearly across
  running periods up to a configured cap (longer ones are unknown); raw `t_out` stays for
  diagnostics. Per-room offsets are the function `room_offsets(..., mask)`, not a field, so a fold
  fits them on its own training mask and nothing held out can leak.
- The site coordinates (and an optional always-open valve share `phi`) are read from
  `<data_dir>/climate/site.yaml` (`latitude`, `longitude`, optional `phi`). The file is private,
  never committed, and its hash is recorded with the data.
- `hack/climate_audit.py` (read-only) writes `report.json` and `report.md` into
  `<data_dir>/analysis/v1/climate-audit/<feature set>-<config>[-name]/`: thermostat resolution, gaps,
  counter quantization, the return temperature as a slab proxy, the heating curve, the legacy-mask
  hit rate, the channel-2 ratio, the outdoor-sensor sun test, filter survival per season and the
  `min_hours` regimes. The report holds private figures: it stays in the data dir.

```bash
uv run python hack/climate_audit.py --feature-set <climate feature set id> \
    --dataset <data_dir>/analysis/v1/<dataset_id>
```

## Shadow service (`service/`)

**Status: Phase C (PROD-C1 to C3) built — shadow only, no actuation path.** `battery-optimizer serve`
subscribes to Home Assistant (read-only allowlist, reconnect and full resync; passive PVPC-snapshot and
weather-forecast archives, see [service.md](service.md#passive-archives)), reads load history from
Prometheus, solves the offline MPC kernel every 15 minutes in a bounded worker, records time-bounded
shadow intents and exposes `/health/live`, `/health/ready`, `/status`, `/plan` and `/metrics`.
`battery-optimizer shadow-report` compares the recorded plans with the native battery behaviour.
No battery entity exists in Home Assistant yet, so the committed configuration runs a virtual battery
(`soc_source: simulated`) until the Omnibattery entities are mapped. Architecture, configuration,
metrics, failure modes and the comparison method: [service.md](service.md).
