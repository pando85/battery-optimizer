# Documentation

Guides, design records and planning for the battery optimizer. The project began in
[`pando85/homelab`](https://github.com/pando85/homelab); see [provenance](provenance.md) for what moved,
how identities map, and the caveats when comparing old artifacts. The project overview and quick start are in
the [top-level README](../README.md).

## Guides

| Document | Purpose |
|---|---|
| [Configuration](configuration.md) | Environment variables, data directory, development and production locations, Home Assistant authentication, secrets policy |
| [CLI reference](cli.md) | Every command, the `extract` exit codes and coverage gaps, `ha-states`, `inspect-sql` |
| [Benchmark reference](benchmark.md) | `benchmark` flags, the five-rung policy ladder, battery assumptions, methodology notes |
| [Architecture](architecture.md) | Module notes: dispatch intent contract, artifact integrity, economic sensitivity, replay runtime, shadow service |
| [Container image](container.md) | Published tags, running the batch commands and `serve` from the image, provenance verification |
| [Known issues](known-issues.md) | HA retention, per-chunk emptiness, row-level re-extraction dedup, `carry_in` flags |

## Architecture decision records

| ADR | Title | Status |
|---|---|---|
| [0001](adr/0001-home-battery-economic-dispatch.md) | Forecast-informed economic dispatch | Proposed |
| [0002](adr/0002-appliance-aware-load-forecasting.md) | Appliance-aware load forecasting | Proposed |
| [0003](adr/0003-phase1-measurement-and-baseline.md) | Phase 1 measurement and baseline | Accepted (2026-10-23) |
| [0004](adr/0004-battery-optimizer-production-architecture.md) | Production runtime and independent source repository | Accepted (2026-10-25; software architecture, shadow only) |

## Delivery snapshot (2026-10-25)

- **Application and releases:** v0.2.1 published with wheel, sdist and multi-arch OCI image; associated CI, image and scan jobs succeeded. See [releases](https://github.com/pando85/battery-optimizer/releases).
- **Deployed via homelab GitOps:** a daily extract/validate/periodic-build CronJob and resident read-only shadow service, using a ZFS PVC, ServiceMonitor, Grafana dashboard and alert rules. See the [operator runbook](https://github.com/pando85/homelab/blob/master/docs/deployment/battery-optimizer.md).
- **Not accepted yet:** repeated unattended pipeline success, alert and backup-restore drills, two-week live native-battery shadow comparison. The service uses **simulated SOC** before Marstek/Omnibattery commissioning. **No device writer or live dispatch exists.**
- **Release policy follow-up:** image publication runs independently of the CI workflow and a tagged release does not require the scan job to pass, although v0.2.1 checks succeeded. See the [production gates](planning/home-battery-optimizer-implementation.md#16-production-packaging-and-runtime-rollout).

## Service

- [Resident shadow service](service.md): architecture, endpoints, metrics, configuration, failure modes and the
  shadow-vs-native comparison method. Environment variables shared with the batch commands are in
  [configuration](configuration.md#environment-variables).

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
