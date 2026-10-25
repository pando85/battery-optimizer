# Documentation

Design records and planning for the battery optimizer. The project began in
[`pando85/homelab`](https://github.com/pando85/homelab); see [provenance](provenance.md) for what moved,
how identities map, and the caveats when comparing old artifacts.

## Architecture decision records

| ADR | Title | Status |
|---|---|---|
| [0001](adr/0001-home-battery-economic-dispatch.md) | Forecast-informed economic dispatch | Proposed |
| [0002](adr/0002-appliance-aware-load-forecasting.md) | Appliance-aware load forecasting | Proposed |
| [0003](adr/0003-phase1-measurement-and-baseline.md) | Phase 1 measurement and baseline | Accepted (2026-10-23) |
| [0004](adr/0004-battery-optimizer-production-architecture.md) | Production runtime and independent source repository | Proposed |

## Service

- [Resident shadow service](service.md): architecture, endpoints, metrics, configuration, failure modes and the
  shadow-vs-native comparison method.

## Planning

| Document | Purpose |
|---|---|
| [Implementation plan](planning/home-battery-optimizer-implementation.md) | Phased plan; [§16](planning/home-battery-optimizer-implementation.md#16-production-packaging-and-runtime-rollout) is the single production sequencing checklist (Phases A to D) |
| [Battery roadmap](planning/energy-battery-roadmap.md) | Component-level progress summary |
| [Baseline progress](planning/energy-baseline-progress.md) | Phase 1 status, built datasets and benchmark runs |
| [Forecast experiments](planning/energy-baseline-forecast-experiments.md) | Rolling-horizon and forecast experiment records |
| [AppDaemon baseline context](planning/energy-baseline-appdaemon-baseline-context.md) | Context on the existing AppDaemon heating/DHW control the baseline is compared against |

## Provenance

- [Provenance of the homelab extraction](provenance.md): source commit, rename, full commit map, `code_version` /
  `dataset_id` rule, signatures, `source_hashes` caveat and environment variable renames.

## Owned elsewhere

The Kubernetes deployment, ArgoCD/Helm values, secrets references and the
[production execution plan](https://github.com/pando85/homelab/blob/master/planning/energy-baseline-production-execution.md)
(extraction schedule, retention) live in `pando85/homelab`.
