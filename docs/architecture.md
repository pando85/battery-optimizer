# Architecture

<!-- COMPONENT DIAGRAM -->

The package has two halves. The offline pipeline (`extract`, `validate`, `build`, `benchmark`, `report`;
see the [CLI reference](cli.md)) ingests Prometheus and Home Assistant history into an immutable,
content-addressed store and scores dispatch policies on it. The resident [shadow service](service.md) reads
Home Assistant and Prometheus, runs the same MPC kernel live and records plans that are never sent anywhere.
Decisions are recorded in [`adr/`](adr/); the production architecture is
[ADR-0004](adr/0004-battery-optimizer-production-architecture.md).

## Modules with no CLI surface

**Modules with no CLI surface (or `hack/` only), deliberately:**

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
