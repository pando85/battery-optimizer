# ADR-0005: Share the energy-planning runtime with a predictive climate planner

- **Status:** Proposed (architecture only; no authorization to change physical controls)
- **Date:** 2026-10-30
- **Decision owners:** Homelab maintainers
- **Extends:** [ADR-0001](0001-home-battery-economic-dispatch.md), [ADR-0002](0002-appliance-aware-load-forecasting.md), and [ADR-0004](0004-battery-optimizer-production-architecture.md)
- **Execution plan:** [Climate optimizer implementation plan](../planning/climate-optimizer-implementation.md)
- **Affected repositories:** `pando85/battery-optimizer` (models, planning and evaluation) and `pando85/homelab` (AppDaemon/HA configuration, deployment and actuation)

## Context

The homelab's AppDaemon `climate_control.py` selects inexpensive PVPC hours and `dhw_control.py` schedules forced hot water within twelve-hour windows. Neither currently predicts indoor temperature, building thermal inertia or DHW tank demand. Their HeishaMon/Panasonic Aquarea control surface is already operational and must not be replaced without an explicit migration.

The deployed `battery-optimizer` already provides a resident **read-only** Home Assistant/Prometheus runtime, an as-of and DST-aware PVPC time grid, price/weather archives, persistent shadow plans, metrics and offline benchmarking. Its `structured_forecast.py` already replays AppDaemon's climate/DHW rules as a versioned **offline** demand-forecast baseline. Creating a second always-on forecasting/MPC runtime would duplicate these foundations, while importing the entire battery service into AppDaemon would duplicate state, scheduling and storage.

ADR-0001 originally deferred joint heat-pump/battery optimization and reserved HVAC/DHW authority for AppDaemon. ADR-0004 authorized only a shadow battery service and retained strict Home Assistant write guards. Those protections remain in force.

## Decision

1. **One shared planning deployment, not one universal physical solver.** Evolve the existing `battery-optimizer serve` process into a shared planning *runtime*. The climate and battery domains retain independent state equations, constraints, objective functions, snapshots and plans. Share only proven infrastructure: HA read adapters, entity freshness, UTC/DST grid, published prices, forecast ingestion, historical storage, solver execution patterns and observability. Reuse SciPy/HiGHS where mathematically appropriate; do not force climate control through the battery dispatch kernel.
2. **No new application or Kubernetes workload for climate planning.** Extend the existing deployed service behind a disabled-by-default climate feature. The package, CLI and image names remain unchanged for this phase (only the repository was renamed, see below). The current battery behavior, API, datasets and deployment cadence must remain backward compatible.
3. **Keep planning read-only.** The existing service's `ha_guard`, read-only HA credential and no-actuation guarantee remain intact. Climate plans are emitted via a new, internal, read-only endpoint and persisted as shadow decisions. No HA write permission, raw MQTT/Modbus command or control-side API is introduced to this service by this ADR.
4. **AppDaemon initially retains exclusive climate/DHW actuation.** During migration it will optionally consume validated `ClimatePlan` decisions and translate them into HA/HeishaMon commands. The existing scheduler stays available as an explicit fallback/baseline. Mode changes require device-state acknowledgement, timeout, manual-override ownership and recovery behavior. The separate DHW app must not remain a competing force-DHW writer when joint planning is activated.
5. **Battery/heat coordination is forecast exchange first, not joint dispatch.** Publish the future HVAC+DHW *electrical* demand forecast to the battery load forecaster. Avoid double-counting this energy in gross household demand. Battery operation remains shadow-only until ADR-0004's separate physical-commissioning gates pass. The climate executor must never issue battery commands, and the battery planner must never command the heat pump.
6. **Model complexity is evidence-driven.** Begin with a temperature-aware baseline and identifiable first-order building/DHW models. Add slab, multiple zones, flow-temperature modulation, cooling, uncertainty-aware planning or joint optimization only when measured prediction or economic/comfort improvements justify their complexity.
7. **AppDaemon retirement is a separate future decision.** It can be removed only after a replacement HA actuator preserves single-writer semantics, permissions isolation, manual controls, hardware safety and operational rollback. Co-location does not itself justify granting the planning runtime actuator credentials.

## Component and control boundaries

~~~mermaid
flowchart TD
    HA[Home Assistant and HeishaMon telemetry] --> READ[Existing read-only HA client and entity store]
    HIST[Existing Prometheus and archived history] --> READ
    READ --> SHARED[Shared time grid, prices, weather and validation]
    SHARED --> CLIM[Climate thermal, COP and DHW models]
    SHARED --> BATT[Battery load forecast and physical model]
    CLIM --> CMPC[Climate MPC]
    CMPC --> CP[Versioned ClimatePlan in shadow store / GET endpoint]
    CP --> AD[AppDaemon climate actuator, gated]
    AD --> HP[HA / HeishaMon / Aquarea]
    CP -. electrical demand forecast .-> BATT
    BATT --> BMPC[Existing battery MPC, shadow only]
    BMPC --> BP[Existing DispatchPlan and metrics]
~~~

The diagram's AppDaemon-to-pump connection is **existing**. The planned `ClimatePlan` consumer is **not implemented**. The battery MPC-to-hardware path is intentionally absent.

**Single-writer rule:** the planning process writes neither device. AppDaemon owns HVAC and forced-DHW writes until an independently reviewed migration; HA/Omnibattery owns battery-device regulation once commissioned. A plan is advice, not acknowledgement of execution.

## Packaging and workload decision

**One source repository, one versioned Python distribution, one OCI image, multiple entrypoints.** In the near-term deployment, the same image continues to power:

| Workload | Image | Command / responsibility | Permissions |
|---|---|---|---|
| Existing **optimizer Deployment** (one replica) | `battery-optimizer` today; future `home-energy-optimizer` | `serve`: read-only battery and climate planning in one process, separately scheduled/limited solver work | Read-only HA/Prometheus access; no physical write capability |
| Existing **pipeline CronJob** | **Exactly the same OCI image and pinned digest** | `extract` / `validate` / `build`; future offline training/evaluation subcommands | Read-only telemetry plus scoped dataset/PVC writes; no HA actuation |
| Existing **AppDaemon Deployment** | Its **own existing image**, unchanged by the planning migration | Climate/DHW actuation, with legacy schedules and optional validated ClimatePlan consumption | HA service permissions needed for HeishaMon commands |

This is **one optimizer image**, **not one image per optimizer domain**. A shared image does **not** imply one container, process, service account or failure boundary for every responsibility. The optimizer Deployment and pipeline CronJob already have different lifecycles and permissions; they only share an artifact.

**Optional post-migration actuator:** If AppDaemon is eventually retired, the preferred candidate is a dedicated `climate-actuator` Deployment using the **same optimizer OCI image** with a separate `climate-actuator` entrypoint, separate narrowly scoped credentials, independently configurable availability/restarts and exclusive HA/HeishaMon write ownership. It replaces AppDaemon's climate execution role, rather than granting write access to `serve`. Such a deployment is **not approved or needed for the initial climate planner**; its creation, credential scope, failsafe behavior and exclusive ownership must be decided and verified separately. No direct Modbus writer should be added. Packaging together does not weaken runtime least privilege.

### Repository and artifact naming

The repository was renamed from `pando85/battery-optimizer` to `pando85/home-energy-optimizer` because the software will cover battery dispatch *planning* and HVAC/DHW optimization. **A GitHub repository rename is not an automatic OCI/Python/CLI migration.** The existing `battery_optimizer` import name, `battery-optimizer` command and `ghcr.io/pando85/battery-optimizer` image remain supported through the first climate milestones; renaming those identifiers requires separate reviewed compatibility and GitOps changes. Preserve archived dataset provenance, package imports, binary command usage, existing API paths and pinned digest deployment references.

The full staged migration and proposed commands are specified in the [implementation plan](../planning/climate-optimizer-implementation.md#packaging-release-and-rename-migration).

## Rejected alternatives

| Alternative | Reason |
|---|---|
| A standalone `climate-optimizer` Deployment with its own HA client, archive and HTTP API | Duplicates mature read/forecast/metrics infrastructure and operational overhead before proving value |
| Embed the battery-optimizer resident runtime inside AppDaemon | Couples battery read-only guarantees to an HA write-capable process, duplicates scheduling/state, and complicates releases |
| Directly replace the AppDaemon scheduler with an MPC rollout | No verified thermal model, end-to-end actuator tests, backtest or comfort safety gate yet |
| Make one universal battery+HVAC mathematical solver immediately | Different physics and real-time safety owners; makes validation, isolation and regression attribution harder |
| Grant the existing read-only service unrestricted HA writes | Breaks ADR-0004's technical write barrier and increases the blast radius of an optimizer or model bug |

## Consequences and explicit non-decisions

- **Positive:** reuse the resident service, existing GitOps image/PVC, publication-aware PVPC parser, weather archive, shadow store, benchmarks and monitoring. A future climate plan can inform the existing structured load-forecast seam.
- **Costs:** additional runtime CPU/memory/solve scheduling, domain-specific telemetry, thermal identification and separation of independent plan validity. The current 200 MiB AppDaemon limit and the battery service's limits must be validated before extending workloads.
- **No new deployable now:** only docs and a future optional in-process planner. The legacy AppDaemon remains a Kubernetes Deployment, not a DaemonSet.
- **No change to current battery safety:** the battery service stays shadow-only, with simulated SOC where actual entities are absent. Nothing in this ADR authorizes Phase D battery actuation.
- **No direct cooling or flow-temperature control in v1:** cooling needs distinct thermodynamics and dew-point/condensation protection; Panasonic direct flow targets require verified equipment configuration.
- **No assumed household PV system:** do not introduce solar self-consumption into the cost model until installed generation and metering are verified.
- **Unresolved commissioning inputs:** representative room sensors, thermal distribution/zone topology, DHW usage periods, measured COP validity, compressor minimum-run constraints, user comfort bands and existing heat-pump fallback configuration. These are explicit verification tasks, not implementation assumptions.

## Approval and migration gates

Acceptance of this ADR authorizes the **software architecture and a read-only implementation roadmap only**. Each phase in the linked plan requires its own PR with independent acceptance criteria. Changing HVAC authority requires an explicit homelab GitOps rollout and controlled device-level tests. Changing battery authority requires the separate ADR-0004 Phase D approval; this ADR does not supersede it.
