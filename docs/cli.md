# CLI reference

Every command is run as `uv run battery-optimizer <command>` from a checkout, as `battery-optimizer <command>`
when installed, or as the container image's entrypoint (see [Container image](container.md)). Paths, environment
variables and secrets are described in [Configuration](configuration.md). There is no recurring scheduler yet:
every command here is run manually (see [No recurring scheduler yet](configuration.md#no-recurring-scheduler-yet)).
The modules that have no CLI surface are described in [Architecture](architecture.md#modules-with-no-cli-surface).

| Command        | Status       | Purpose                                        |
|----------------|--------------|------------------------------------------------|
| `extract`      | **functional** | Ingest from Prometheus / HA history to Parquet |
| `validate`     | **functional** | Validate raw Parquet against schemas + manifest hashes |
| `build`        | **functional** | Build 15-min analysis dataset                  |
| `build-features` | **functional** | Build the slot-aligned sub-load feature table, apart from the canonical dataset |
| `decompose-load` | **functional** | Private report: share of load and of LF-0 error per sub-load |
| `forecast-eval` | **functional** | Pre-registered load-forecaster evaluation: accuracy and paired EUR against LF-0 |
| `benchmark`    | **functional** | Run policy benchmarks on a dataset             |
| `report`       | **functional** | Generate monthly reconciliation report         |
| `inspect-sql`  | **functional** | Inspect DuckDB SQL queries                     |
| `ha-states`    | **functional** | Read Home Assistant enum state history (read-side, deduplicated, as-of) |
| `serve`        | **functional** | Resident shadow service: read-only HA/Prometheus, live MPC, intents recorded never sent ([docs](service.md)) |
| `shadow-report`| **functional** | Shadow plans vs native battery behaviour: coverage, energy, cost, SOC violations |
| `mirror`       | deferred     | Deferred — not implemented in Phase 1          |

## `extract`

```bash
# Prometheus (no auth, default 4 entities)
uv run battery-optimizer extract \
  --source prometheus \
  --config config/tariff.yaml \
  --lookback-days 9

# HA history (requires BATTERY_OPTIMIZER_HA_TOKEN env var)
uv run battery-optimizer extract \
  --source ha-history \
  --config config/tariff.yaml \
  --start 2026-10-14T00:00:00Z \
  --end 2026-10-23T10:59:56Z \
  --entities sensor.lavadora_current_status \
  --entities sensor.lavavajillas_current_status

# Both sources in one run
uv run battery-optimizer extract --source all --lookback-days 7
```

Time window: either `--start`/`--end` (ISO-8601 UTC, end defaults to now) or
`--lookback-days N` (start = N days ago at midnight UTC, end = now).

### Exit codes and coverage gaps

| Code | Meaning |
|------|---------|
| `0` | Success, with no coverage gap recorded (or `--allow-coverage-gaps` was passed) |
| `1` | The run failed (extractor error, bad arguments or window); no manifest for that source |
| `2` | Usage error (argparse) |
| `3` | The run **completed and committed its data and manifest**, but the manifest records a coverage gap |

Exit `3` exists so a scheduled job (the Kubernetes CronJob) does not look green while data has holes.
The data and manifest stay on disk and inspectable; stderr names every gap flag (chunk windows
included). With `--source all` both sources always run before the code is chosen, so a Prometheus
gap never skips the non-re-extractable HA pull. Pass `--allow-coverage-gaps` for interactive or
backfill runs (for example a window that starts before a series existed) to get exit `0`; the
summary is still printed.

Flags that are **gaps** (exit `3`): `empty_chunk:`, `empty_chunk_after_window_filter:`,
`chunk_accounting_mismatch`, `no_rows_for_entity:`, `no_rows_extracted:all` (HA) and
`no_samples_extracted:`, `no_series_chunks:`, `series_without_samples_chunks:` (Prometheus). A
known-dead entity (`sensor.esios_pvpc_2`) is expected to be empty and never counts.

Flags that stay **informational** (no exit change): `duplicate_rows_suppressed:`,
`duplicate_rows_already_stored:`, `carry_in_classification_disagreement:`,
`prior_part_unreadable:`, `duplicate_samples_dropped:`, `boundary_duplicates_dropped:`,
`non_finite_samples_dropped:`, `overlapping_target_samples:`, `dead_duplicate_entity:`. They say how
the run went, not that data is missing. An HA entity that simply did not change is *not* an empty
chunk: the history API returns its state at the chunk start (the `carry_in` row), so a zero-row
chunk means purged history or an absent entity.

Per-request `httpx`/`httpcore` INFO lines are hidden by default so the pipeline's own progress lines
stay readable in container logs; `-v`/`--verbose` shows them again (HA extraction still suppresses
`httpx` itself, because its URLs name the recorder host).

Re-running an overlapping or identical window is safe for the HA source: rows
already in the store are not written again, and the manifest discloses how many
were suppressed. See [Overlapping re-extraction is deduplicated at row
level](known-issues.md#overlapping-re-extraction-is-deduplicated-at-row-level).

## `validate`

```bash
uv run battery-optimizer validate --source ha-history
uv run battery-optimizer validate --source prometheus
uv run battery-optimizer validate --source all   # default
```

Checks per source:
- Every part file matches `RAW_SAMPLE` / `RAW_ENUM_STATE` schema
- Filename content hash matches actual SHA-256 of file bytes
- Every manifest's recorded part hashes match the actual files on disk
- No zero-row parts, no unreadable parts

Exit code 0 if all checks pass; 1 otherwise.

## `build`

```bash
uv run battery-optimizer build \
  --start 2026-10-01T00:00:00+00:00 \
  --end 2026-10-23T00:00:00+00:00 \
  --assume-no-battery  # default; use --no-assume-no-battery to refuse
```

The `--assume-no-battery` flag is a **recorded assumption, not a detection**: missing battery
telemetry cannot prove a battery does not exist. The assumption is pinned in the build manifest.
When `--no-assume-no-battery` is set, the build requires registered battery telemetry to
reconstruct gross load; without it the build refuses.

**Immutable store:** the canonical data directory is write-once, content-addressed. Re-running
`build` for a window that already exists is idempotent when the bytes match. When they do not —
`refusing to overwrite .../build_manifest.json with different content (existing=…, new=…)` —
the build **refuses** with one line on stderr and exit code 1; it does not print a traceback, and
it never overwrites or deletes anything. The dataset id covers source part hashes, config hash,
code version, schema version, window and builder parameters, so an identical id with different
bytes means something outside those inputs changed since the original build — in practice code in
a dirty working tree, whose version string records only a `-dirty` marker, or a provenance field
the id does not cover. Fix it on your side of the boundary: point `--data-dir` at a fresh data
directory, or remove that one dataset directory yourself and re-run.

## `build-features`

```bash
uv run battery-optimizer extract --source prometheus --config config/tariff.yaml \
  --start 2024-01-01T00:00:00Z --end 2026-10-01T00:00:00Z --allow-coverage-gaps \
  --entities sensor.shellypmmini_servers_power,sensor.aquarea_heat_power_consumed
uv run battery-optimizer build-features \
  --start 2024-01-01T00:00:00Z --end 2026-10-01T00:00:00Z \
  --dataset ~/data/energy-baseline/analysis/v1/<dataset_id>
```

Builds a slot-aligned 15-minute UTC table of the observable sub-loads from the raw samples of the
feature entities (`extract_prom.FEATURE_ENTITY_IDS`: servers circuit, heat-pump consumed and
produced power, outdoor temperature, AppDaemon control states). Each power series becomes kWh per
slot by sample-and-hold integration; a gap longer than `--max-gap-seconds` (300) is not held, and
a slot with less than `--min-coverage` (0.8) covered is `unknown`, never zero. The rule is spelled
out in the `features` module docstring and pinned in the manifest.

The table is written to `<data_dir>/analysis/v1/features/<feature_set_id>/` with a
`features_manifest.json` (hashes of every raw part read, code version, parameters, quality checks:
units, cadence, gaps per month, COP sanity, long constant runs). It never touches a canonical
dataset directory, and the feature entities are not default entities, so `extract` without
`--entities` and the canonical `dataset_id` are unchanged. A raw manifest that declares only
feature entities is also ignored by `build`'s lineage hash. With `--dataset`, the
`servers + heat pump <= main` check against that dataset's gross load is written to
`quality/main_meter-<dataset_id12>.json` next to the manifest.

## `decompose-load`

```bash
uv run battery-optimizer decompose-load \
  --features ~/data/energy-baseline/analysis/v1/features/<feature_set_id> \
  --dataset ~/data/energy-baseline/analysis/v1/<dataset_id>
```

Writes `p3-report/summary.{json,md}` inside the feature-set directory (private: it holds energy and
price figures). For each season and month it reports the share of kWh from servers, heat pump and
the residual, runs LF-0 (28-day trailing mean) on the main load and on every component with one
common mask, and reports each component's error, its share of the error variance and its
price-weighted error, overall and for the top price tercile. It also replays the AppDaemon climate
and DHW rules on the day's prices to show how much heat-pump energy they explain.

## `forecast-eval`

```bash
uv run battery-optimizer forecast-eval \
  --dataset ~/data/energy-baseline/analysis/v1/<dataset_id> \
  --name p2-cheap \
  --candidate trailing_mean:window_days=14 \
  --candidate <registered name>[:key=value,...] \
  --p0-summary ~/data/energy-baseline/analysis/v1/<dataset_id>/forecast-eval/<p0 run>/summary.json
```

The evaluation harness of the [load-forecaster plan](planning/load-forecaster-improvement-plan.md)
(P1). Each `--candidate` is a registered forecaster with optional param overrides; LF-0
(`trailing_mean`, default params) is always the reference and is added when missing. `--period`
(repeatable) picks the evaluation windows, by default both P0 periods: `2025`
(2025-01-01 to 2026-01-01 UTC) and `2026-01..09` (2026-01-01 to 2026-10-01 UTC). `--resamples`
(default 10000, at least 5000) and `--seed` (default 20261025) set the bootstrap.

It writes a private run to `<dataset>/forecast-eval/<name>-<UTC timestamp>/`:

- `preregistration.json`, written before any candidate runs: candidates and params, periods,
  policies, metric definitions, bootstrap seed, and gate G2 with the P0 gaps it reads.
- `summary.json` and `summary.md`: forecast accuracy, the paired economic result, the gate verdict,
  sanity checks and provenance (git commit and dirty flag, tariff and evaluation config hashes,
  dataset id, forecasters, every benchmark run id, runtime per candidate).

The method (decision instants, metrics, block bootstrap, gate, folds) is described in
[Benchmark reference § Forecast evaluation](benchmark.md#forecast-evaluation-forecast-eval).
`benchmark --eval-end` is the flag the harness uses to cut a period out of a longer dataset.

## `benchmark`

Runs the policy ladder (`no_battery`, `causal_price`, `day_ahead`, `rolling_horizon`, `oracle`) on a built
dataset and writes the benchmark artifacts into the dataset's `benchmarks/` directory. `--train-end` is
required and has no default. The flags, the policy semantics, the battery assumptions and the
methodology notes are long enough to have their own page: see the
[benchmark reference](benchmark.md).

## `report`

```bash
uv run battery-optimizer report --dataset ~/data/energy-baseline/analysis/v1/<dataset_id>
uv run battery-optimizer report --benchmark ~/data/energy-baseline/analysis/v1/<dataset_id>/benchmarks/<run_id>
```

`--dataset` must contain exactly one benchmark run; if multiple runs exist, the error lists the
available run ids rather than silently picking the newest. `--benchmark` overrides `--dataset`.
`--config` is optional but recommended: when passed, its canonical hash is checked against the one
pinned in the run and a mismatch refuses (`ReportError`) instead of reporting against another tariff.

**The report states its own horizon scope.** `report.md` carries a *Horizon scope* line under
*Evaluation window*, and `report.json` carries the same sentence as `horizon_scope_note`: the horizon
is a **UTC-scoped window, not a calendar year in the dataset's local timezone**, unless both bounds
genuinely fall on local midnights, in which case the line says it is a local calendar year too. The
sentence is derived from the run's own bound labels and the build manifest's pinned `params.timezone`,
so it hardcodes no year — the rule in the [benchmark reference](benchmark.md#usage-and-flags) ("the claim is UTC-scoped and the report
must say so") is now enforced by the report instead of left to `Z`-suffixed labels to imply it.

Reports are written into the run directory with the same no-clobber rule as every other artifact, so a
changed report does not silently rewrite a run that has already been reviewed: re-generating a report
whose bytes differ raises `FileExistsError`, which the CLI surfaces as one line and exit code 1 — it is
intended. An existing run's `report.md` and `report.json` are evidence, not a rendering to refresh; the
tool never deletes canonical data on your behalf, so if a report genuinely must be rebuilt, remove that
one file by hand and know that its bytes will differ from what was reviewed.

## `inspect-sql`

```bash
uv run battery-optimizer inspect-sql --list-queries
uv run battery-optimizer inspect-sql --dataset ~/data/energy-baseline/analysis/v1/<dataset_id> \
  --query price_distribution
uv run battery-optimizer inspect-sql --dataset ~/data/energy-baseline/analysis/v1/<dataset_id> \
  --sql "SELECT * FROM intervals WHERE price_eur_per_kwh > 0.20" \
  --benchmark ~/data/energy-baseline/analysis/v1/<dataset_id>/benchmarks/<run_id>
```

Exactly one action is required: `--sql`, `--query`, or `--list-queries`. `--sql` and `--query` are
mutually exclusive. Results print as aligned text by default, or JSON with `--json`. The SQL is
validated as a single read-only statement (SELECT / WITH ... SELECT / EXPLAIN SELECT); anything
else is refused at the grammar level.

`benchmark_results` and `benchmark_schedules` exist only when `--benchmark PATH` is given. Curated
queries that read them (`monthly_cost`, `excluded_intervals`, `benchmark_summary`) and any ad-hoc
SQL naming them are therefore refused up front, before DuckDB is opened, with one line naming the
missing view and telling you to pass `--benchmark` — instead of leaking a raw
`duckdb.CatalogException` traceback. Dataset-only queries (`price_distribution`,
`quality_summary`, `dataset_meta`) work without it.

## `ha-states`

```bash
# Summary of the store: parts, rows before and after dedup, entities present
uv run battery-optimizer ha-states

# Every deduplicated state for one entity, in order
uv run battery-optimizer ha-states --entity sensor.lavadora_current_status --list-states

# What that entity's state was at an instant (inclusive at the transition boundary)
uv run battery-optimizer ha-states --entity sensor.lavadora_current_status \
  --as-of 2026-10-20T12:00:00Z
```

Reads the Home Assistant appliance enum history that `extract --source ha-history` preserves, through
`ha_states.py`. It exists because the durable Prometheus archive cannot carry these values at all —
the exporter emits state-value series only for numerically convertible states — and because the
recorder keeps them for about ten days, so the Parquet store is the only surviving copy.

- **Deduplication happens at read time**, on the same identity the extractor uses:
  `(entity, last_changed, last_updated, state, attributes_json)`. Part files are immutable and 12 known
  pre-dedup duplicate rows were **deliberately left in place** rather than rewriting history, so collapsing
  them is the reader's job; see
  [Overlapping re-extraction is deduplicated at row level](known-issues.md#overlapping-re-extraction-is-deduplicated-at-row-level).
- **As-of queries are inclusive at the transition boundary**: at exactly a transition instant the new state
  is visible, because `last_changed` is when the state became effective. Only rows with
  `last_changed <= T` are returned, which makes it *structurally* impossible to leak a later transition into
  an earlier snapshot — ADR-0002's as-of requirement, not a convention to remember.
- `carry_in` and both timestamps (`last_changed` = event time, `last_updated` = observation time) are
  preserved verbatim; consumers must filter `carry_in=False` before treating `last_changed` as a real
  event. Unavailable/unknown states are surfaced, not silently dropped.
- `--as-of` and `--list-states` both require `--entity`; a refusal is a `ValueError`, which the CLI prints
  as one line with exit code 1.
- **Verified live against the real store, read-only:** `parts=12 rows_before_dedup=51 rows_after_dedup=39
  duplicates=12 entities=2` for `sensor.lavadora_current_status` and `sensor.lavavajillas_current_status`.

**This is a read-side module.** It does not feed `build`, does not enter `dataset_id`, and no policy or
benchmark reads it. No appliance modelling, NILM or forecasting was built on it: it closes the long-deferred
"there is no reader" gap and nothing more, and ADR-0002 Phase 3 remains the place where consumption would
start.

## `mirror`

Deferred — not implemented in Phase 1 (ADR-0003 §optional). The optional TimescaleDB / SQL mirror
is out of scope for this phase.
