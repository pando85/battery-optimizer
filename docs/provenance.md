# Provenance of the homelab extraction

This repository (`pando85/battery-optimizer`) began life as the `scripts/energy-baseline/` Python package
inside [`pando85/homelab`](https://github.com/pando85/homelab). This page records exactly what was moved,
what changed, how to relate old identities to new ones, and what could not be preserved.

## Source

| | |
|---|---|
| Source repository | [`pando85/homelab`](https://github.com/pando85/homelab) |
| Import point | [`b7286b874f34f732c7bf9d69940e05feb61ba8c6`](https://github.com/pando85/homelab/commit/b7286b874f34f732c7bf9d69940e05feb61ba8c6) (homelab `master`) |
| Method | `git filter-repo`, history of the moved paths preserved (26 commits) |

## What was moved and renamed

| Homelab path (at the import point) | Path here |
|---|---|
| `scripts/energy-baseline/` (package, tests, `hack/`, `config/`, README, lockfile) | repository root |
| `scripts/energy-baseline/src/energy_baseline/` | `src/battery_optimizer/` (renamed after the import, commit `3b5a93ab0992`) |
| `docs/adr/0001` .. `0004` | [`docs/adr/`](adr/) |
| `planning/*.md` (five files: roadmap, implementation plan, progress, forecast experiments, AppDaemon baseline context) | [`docs/planning/`](planning/) |

Not moved (still owned by homelab): `planning/energy-baseline-production-execution.md`, the extraction and
deployment plan, and everything Kubernetes: `apps/battery-optimizer/` (upcoming), ArgoCD/Helm values,
`ExternalSecret` references, NetworkPolicy, ServiceMonitor, alerts, dashboards, PVC and runbooks.
Links from the moved documents to homelab-only files are absolute
`https://github.com/pando85/homelab/blob/master/...` URLs.

Documents were otherwise moved verbatim. Historical statements (for example "built in
`scripts/energy-baseline/...`" in dated records, command transcripts, `/tmp/opencode/energy-baseline/` run
paths and ADR decision text) keep their original wording; where they describe current code, file paths were
updated to the new layout and the CLI/variable renames are listed below.

## Commit map

Every imported commit has a new SHA because its tree was rewritten (paths moved, then the package
renamed in later commits). The commit subjects are unchanged except that `(#N)` pull-request references
were rewritten to `pando85/homelab#N`, because `#N` would otherwise resolve to this repository. The old
SHA links to the original commit in homelab.

| Date | Old SHA (homelab) | New SHA | Subject |
|---|---|---|---|
| 2026-10-23 | [`2fdbae621f38`](https://github.com/pando85/homelab/commit/2fdbae621f383c53df59fde398d91e46a1259607) | `3e01138231f4` | docs: Propose phased home battery economic optimizer ADR (pando85/homelab#4651) |
| 2026-10-23 | [`10f579323fe8`](https://github.com/pando85/homelab/commit/10f579323fe82d90ae5d8a57a9733cf260b6a7be) | `5a2cb0d5c2e8` | docs: ADR-0002 appliance-aware load forecasting with HA telemetry and NILM (pando85/homelab#4652) |
| 2026-10-23 | [`f3a1c75f49df`](https://github.com/pando85/homelab/commit/f3a1c75f49df3baf9347e84cd6b5cf8f5b7ddd92) | `bccfb0b25fe0` | docs: Define reduced-scope energy baseline ADR |
| 2026-10-23 | [`edac40b35edd`](https://github.com/pando85/homelab/commit/edac40b35edd60984ff95225a22170f9fa3e0358) | `e4701037635b` | docs: Record production execution path for energy baseline |
| 2026-10-23 | [`9d5be140ce1b`](https://github.com/pando85/homelab/commit/9d5be140ce1bf8280dbaa947a091ada9081b0aff) | `0b9e72647e1c` | scripts: Add energy-baseline package with Prometheus and HA extraction (cited as `code_version`) |
| 2026-10-23 | [`a6cb9045f0d3`](https://github.com/pando85/homelab/commit/a6cb9045f0d3298e4bd492b6febd7e54f272e9c0) | `fc6375d1eef1` | scripts: Complete energy-baseline benchmark and correct build facts (cited as `code_version`) |
| 2026-10-23 | [`f536cbd44426`](https://github.com/pando85/homelab/commit/f536cbd4442690d520378281cfb8c5ede0269ced) | `5b735a5b03a1` | scripts: Deduplicate HA re-extraction and record Phase 1 decisions (cited as `code_version`) |
| 2026-10-23 | [`c156931472e2`](https://github.com/pando85/homelab/commit/c156931472e2a20025ccd1be4dfbd8558136ac96) | `0ce3447e1d69` | scripts: Add day-ahead policy on a shared scoring basis (cited as `code_version`) |
| 2026-10-23 | [`a4837f6dc82e`](https://github.com/pando85/homelab/commit/a4837f6dc82e5074e732186e7174437eb622964b) | `b6a01e856170` | docs: Record conditional acceptance of the four-policy baseline (cited as `code_version`) |
| 2026-10-24 | [`7e0aae2a2d6a`](https://github.com/pando85/homelab/commit/7e0aae2a2d6a76680ac98cfc324a31761353c971) | `3eb41a9f7ab1` | scripts: Add two-publication rolling-horizon experiment |
| 2026-10-24 | [`272a3cf02488`](https://github.com/pando85/homelab/commit/272a3cf02488caa7ee274e63e422f5575101f14d) | `43a696d6e2d3` | docs: Record owner-supplied battery and price facts |
| 2026-10-24 | [`5fc27b9f2631`](https://github.com/pando85/homelab/commit/5fc27b9f2631de10f905b7ffa6460e9b682f94f1) | `bda3d7d522be` | scripts: Promote rolling horizon to a fifth policy at RTE 0.80 (cited as `code_version`) |
| 2026-10-24 | [`6fca56afc33d`](https://github.com/pando85/homelab/commit/6fca56afc33d8ffc39a18aed17833906002b3af1) | `395a408f968e` | scripts: Correct false forecast-basis comparability claim (cited as `code_version`) |
| 2026-10-24 | [`5ad8b4d892d6`](https://github.com/pando85/homelab/commit/5ad8b4d892d615781ea0aa6666ffc5d1a158dedb) | `a379e3fa5440` | docs: Fill rebuild placeholders with the 0.80 ladder results |
| 2026-10-24 | [`90feb857356f`](https://github.com/pando85/homelab/commit/90feb857356fc4cc107ef69b8a13721f0dea3bfb) | `97fdf1734f9b` | docs(battery): decide production runtime and independent repo ownership (pando85/homelab#4653) |
| 2026-10-24 | [`6a3b634e189b`](https://github.com/pando85/homelab/commit/6a3b634e189b6c19a5e3ebdd2d2b54e59311e355) | `0bfec3ba438f` | scripts: Add artifact-integrity manifest module and report wiring |
| 2026-10-24 | [`a3d22316d597`](https://github.com/pando85/homelab/commit/a3d22316d597399264942f7621a179aedb0baeff) | `d7f8b062f5d8` | scripts: Align rolling-horizon load forecast with the day-ahead basis |
| 2026-10-24 | [`39d2f3bcb92d`](https://github.com/pando85/homelab/commit/39d2f3bcb92ddec38334761959e56e7bdd7c2af9) | `87afbb6356d0` | scripts: Add sensitivity sweeps, replay runtime, and validation runner |
| 2026-10-24 | [`818e16aadcde`](https://github.com/pando85/homelab/commit/818e16aadcdee2ded500689fcf3f64481915cb7a) | `f9033ba5f499` | docs: Record the bounded pre-migration milestone |
| 2026-10-24 | [`b42c379d7356`](https://github.com/pando85/homelab/commit/b42c379d735644cc0e2efa808e75ab15e7deeca6) | `173fb3ffb158` | scripts: Preserve plan sequence across restart in the replay runtime |
| 2026-10-24 | [`22133bd30196`](https://github.com/pando85/homelab/commit/22133bd301962606bc67138e7886a1c6456a76ab) | `428b0190bec4` | scripts: Correct docstring overpromises and derive slots per day |
| 2026-10-24 | [`8b899ed0e06e`](https://github.com/pando85/homelab/commit/8b899ed0e06e626f0114800dbf78639cb78f392e) | `f57c27ba4c4d` | scripts: Pin code provenance in both experiment runners |
| 2026-10-24 | [`3c7ddc9ca9c4`](https://github.com/pando85/homelab/commit/3c7ddc9ca9c423555d8ee40331337040b552f430) | `d7650139555e` | docs: Resolve contradictory milestone claims and disclose provenance (cited as `code_version`) |
| 2026-10-24 | [`071d3caeea0d`](https://github.com/pando85/homelab/commit/071d3caeea0d661eba2ab3c4288019faf14d184b) | `7d64b5be9bac` | docs: Record pinned re-runs and the causal five-rung ladder |
| 2026-10-25 | [`07e408d743de`](https://github.com/pando85/homelab/commit/07e408d743de0d322894493744c50bde7799b04c) | `b184ed2ca622` | docs(battery): document Omnibattery HA control contract and integration gates (pando85/homelab#4654) |
| 2026-10-25 | [`c5218745277c`](https://github.com/pando85/homelab/commit/c5218745277cf90ec8842093e69fdf2d485b4a65) | `0dd35be8ff05` | docs: Record the independent review of the pinned evidence |

The map covers the 26 imported commits. Commits made in this repository afterwards (rename, build identity,
tooling) have no homelab counterpart.

## Code identity: `code_version` and `dataset_id`

- Manifests written by `extract`, `build`, `benchmark` and the experiment runners carry
  `code_version` (and `git_head`/`git_dirty` where recorded) equal to the **git SHA of the tree that
  produced them**. `compute_dataset_id` hashes `code_version` together with source-part hashes, config
  hash, schema version, window and parameters, so a dataset id is bound to that SHA.
- **Everything built before the extraction is identified by a homelab SHA**, for example `9d5be140…`,
  `c1569314…`, `a4837f6d…`, `a6cb9045…`, `5fc27b9f…`, `f536cbd4…-dirty`, `6fca56af…` or `3c7ddc9c…`.
  Those SHAs resolve **in homelab history, not in this repository**. To inspect the code that produced such
  an artifact, use a homelab checkout: `git show <code_version>:scripts/energy-baseline/<path>`. The
  new SHA in the commit map is the corresponding commit here, but its tree differs (paths and package
  name), so it is a traceability aid and **not** an equivalence claim.
- **Historical identities are immutable.** Existing dataset and benchmark ids are never recomputed,
  migrated or re-labelled. Nothing in this repository claims that a pre-extraction dataset id would be
  reproduced bit-for-bit by a rebuild here: any fresh build re-mints its id from the new SHA.
- **New identities are SHAs of this repository.** Locally, `code_version()` reads `git rev-parse HEAD`
  of the package directory. Containers have no git metadata and receive the commit through the build-time
  variable `BATTERY_OPTIMIZER_GIT_SHA` (40 lowercase hex characters, optionally `-dirty`; anything else is
  rejected).
- The private data directory (default `~/data/energy-baseline`) is deliberately **unchanged**, so old
  and new artifacts live side by side. It is not part of this repository and is never published.

## `source_hashes` caveat when comparing old artifacts

Benchmark and experiment-runner artifacts record `source_hashes`: SHA-256 of the raw bytes of the source
files on the measurement path, keyed by project-relative path. Those keys changed:

| Era | Key shape |
|---|---|
| Possibly earlier runs | homelab-rooted paths such as `scripts/energy-baseline/src/energy_baseline/<module>.py` (check the artifact; the key shape was not uniform across homelab commits) |
| Before the rename | `src/energy_baseline/<module>.py` |
| Current | `src/battery_optimizer/<module>.py` |

Hash-compare an old artifact's `source_hashes` with the current tree in two steps. First map the key
(`energy_baseline` becomes `battery_optimizer`, and strip any `scripts/energy-baseline/` prefix). Second,
**do not expect equal hashes even for unchanged logic**: the rename rewrote the imports and strings inside
the modules, so their bytes (and therefore hashes) differ. The authoritative comparison is against the
old bytes, taken from homelab at the artifact's `code_version`:
`git show <code_version>:scripts/energy-baseline/<old key>`.

`hack/verify_artifacts.py` is **not affected**. It verifies the files listed in a run's
`outputs_manifest.json` / `report_outputs_manifest.json` against their recorded digests and never
reads `source_hashes` or `code_version`. The tools that do hash source files are
`hack/compare_forecasts.py`, `hack/economic_sensitivity.py` and `hack/validate_pre_migration.py`; they
hash the current tree under the new keys. One documentation detail: the docstring of
`_collect_source_hashes` in `hack/validate_pre_migration.py` still suggests
`git show <sha>:scripts/energy-baseline/<key>`, which only applies to artifacts produced in homelab.

## Environment variable and CLI renames

The CLI is `battery-optimizer` (was `energy-baseline`) and the Python package is `battery_optimizer`
(was `energy_baseline`). The `BATTERY_OPTIMIZER_*` name is read first; the legacy `ENERGY_BASELINE_*`
name is still honoured as a fallback and emits a `DeprecationWarning` when it supplies the value. If both
are set, the new name wins silently. An empty value counts as unset.

| Current | Legacy fallback | Purpose |
|---|---|---|
| `BATTERY_OPTIMIZER_DATA_DIR` | `ENERGY_BASELINE_DATA_DIR` | Private data directory (default `~/data/energy-baseline`, unchanged) |
| `BATTERY_OPTIMIZER_CONFIG` | `ENERGY_BASELINE_CONFIG` | Tariff/config file (default `config/tariff.yaml` relative to CWD) |
| `BATTERY_OPTIMIZER_PROMETHEUS_URL` | `ENERGY_BASELINE_PROMETHEUS_URL` | Prometheus base URL for `extract` |
| `BATTERY_OPTIMIZER_HA_URL` | `ENERGY_BASELINE_HA_URL` | Home Assistant base URL |
| `BATTERY_OPTIMIZER_HA_TOKEN` | `ENERGY_BASELINE_HA_TOKEN` | Home Assistant bearer token (never committed; Vault/ExternalSecret in deployment) |
| `BATTERY_OPTIMIZER_GIT_SHA` | (new) | Build-time commit identity for images without git |

## Signatures

`git filter-repo` rewrites commits, so the **original GPG signatures of the imported commits were
dropped**; this is unavoidable and not a tampering signal. The originals remain verifiable in homelab
history through the linked old SHAs. Commits made in this repository after the import are signed
normally.
