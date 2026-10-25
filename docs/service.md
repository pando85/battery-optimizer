# Resident shadow service

`battery-optimizer serve` is the Phase C runtime of
[ADR-0004](adr/0004-battery-optimizer-production-architecture.md) (plan
[§16](planning/home-battery-optimizer-implementation.md#16-production-packaging-and-runtime-rollout),
PROD-C1 to C3). It is a single resident process that reads Home Assistant and Prometheus, runs the
same MPC kernel as the offline benchmarks every 15 minutes, records the resulting **shadow** plans,
and compares them with what the native battery controller actually did.

**There is no actuation path.** The Home Assistant adapters cannot write a state or call an actuating
service (see [Safety](#safety)), there is no Modbus code, and plans are stored as intents that are never
sent anywhere. The one service call they can make is the non-mutating `weather.get_forecasts` query used by
the [weather-forecast archive](#passive-archives). Live actuation is Phase D and is neither built nor
authorized.

## Architecture

```
                 Home Assistant                       Prometheus (long-term)
   WebSocket (subscribe + get_states)  REST GET       raw range-vector samples
                │                         │                     │
        ha_client.HAWebSocketClient  HARestClient       history.PrometheusHistory
                └──────────┬──────────────┘                     │  (worker thread, once a day)
                     entity_store.EntityStore                   │
            (source-time ordering, freshness)                   │
              │              │            │                     │
        observer.SlotObserver  pvpc.parse_pvpc_state   history.forecast_load
              │                        │                        │
              │        runtime.ShadowService.build_snapshot ────┘
              │                        │  PlanningSnapshot (frozen, hashed)
              │            ThreadPoolExecutor(max_workers=1) + timeout
              │                        │
              │        planner.run_planner = oracle.solve_oracle(free terminal)
              │                          + dispatch.intents_from_schedule
              │                        │
              └──────► slot open/close ─┴─► sink.ShadowSink (append-only JSONL / plan files)
                                │                     │
                     virtual_battery (counterfactual)  shadow_report (offline comparison)
                                │
                         api (FastAPI): /health/live /health/ready /status /plan /metrics
```

| Module (`battery_optimizer.service`) | Role |
|---|---|
| `config` | `config/service.yaml` + environment; only `mode: shadow` is accepted |
| `ha_guard` | Read-only allowlist for every WebSocket frame and REST request (plus the one `weather.get_forecasts` exception) |
| `ha_client` | WebSocket session (auth, subscribe, resync, heartbeat, reconnect) and GET-only REST |
| `entity_store` | Latest state per entity, ordered by HA source time, freshness from `last_reported` |
| `pvpc` | Publication-aware price curve from `sensor.esios_pvpc` attributes |
| `history` | Prometheus load history (offline primitives) and the causal load forecast |
| `planner` | Snapshot → offline MPC kernel → time-bounded shadow intents |
| `observer` | Per-slot measurements (counter energy, power averages, SOC edges) |
| `virtual_battery` | Open-loop replay of decisions against measured load |
| `runtime` | Scheduling, readiness, bounded executor, slot cycle, metrics snapshot |
| `archive` | Record builders and the fetch schedule of the passive [archives](#passive-archives) |
| `sink` | Append-only store under the data directory |
| `shadow_report` | Shadow-vs-native comparison (`shadow-report`) |
| `api`, `serve`, `metrics` | HTTP surface, uvicorn wiring, Prometheus metrics |

### What is reused from the offline code

* **Optimizer:** `oracle.solve_oracle(terminal_mode="free", terminal_value, floor_kwh, throughput_cost)`
  — the call `rolling_horizon.run_rolling_horizon` makes per event — then
  `dispatch.intents_from_schedule`, which re-checks the plan against the battery kernel.
* **Battery parameters:** `benchmark.resolve_battery_assumptions` on `tariff.yaml` (`sqrt(RTE)` per
  leg, SOC window, import ceiling).
* **Forecast:** the registered load forecaster `planning.load_forecaster` (default `trailing_mean`,
  the 28-day same-slot mean `policies.forecast_load_trailing_slot_mean`), run through
  `load_forecast.run_forecaster` with the information cut at the local midnight of the decision day,
  which is the `CausalForecastProvider` rule. Config validation rejects oracle forecasters such as
  `perfect`.
* **History:** `extract_prom.query_range_vector` (instant queries over range vectors, never
  `query_range`), `build.stitch_series` / `counter_deltas` / `allocate_energy`.
* **Execution clipping:** `policies._clip_plan_to_model_bounds`, used by the rolling-horizon benchmark
  to execute a forecast-made plan against measured load.
* **Store:** `paths.write_part_atomic` for plan files.

### Deliberate deviations (and why)

| Topic | Benchmark | Service | Reason |
|---|---|---|---|
| SOC floor | Evaluation window's opening SOC (seed protection) | Physical window minimum (`soc_window_min`) | Seed protection is a fairness device between policies; live energy in the battery is real and is valued at the terminal value instead |
| Price horizon | May plan past publication with exact or noise-perturbed true prices | Ends at the last published slot (capped by `max_horizon_slots`) | Unpublished prices do not exist live; the trailing-mean terminal value keeps the plan from emptying the battery at the edge |
| Orchestration | `replay_runtime.ReplayRuntime` | `runtime.ShadowService` | The replay runtime requires a temperature reading, keeps an unbounded in-memory record list and uses a fixed-terminal oracle adapter; the service borrows its rules (one worker, timeouts never free the worker, obsolete plans discarded) |
| SOC before commissioning | — | `soc_source: simulated` | No battery entity exists in HA yet; a virtual battery evolves by applying the service's own decisions against measured load (see below) |

## Scheduling and the slot cycle

All slots are half-open UTC intervals on a 15-minute grid; Europe/Madrid only labels them, so the
2026-03-29 and 2026-10-25 local days have 92 and 100 slots with no ambiguous or missing slot. For
every boundary `B`:

1. `B - solve_lead_s` (120 s): scheduled solve; the plan's first slot is `B`.
2. `B`: the slot opens and its decision is **frozen**: the actionable intent valid at `B` from the
   newest unexpired plan made before `B`, or no decision with a reason (`no_plan`, `plan_expired`,
   `not_ready`). If any input is stale at `B`, the decision is `not_ready` even if a plan exists.
3. `B + slot_close_grace_s` (20 s): the previous slot closes — measurements, the frozen decision,
   the virtual-battery step and the costs are appended to the store.

Extra solves are requested after a completed resync, newly published next-day prices, a refreshed
history or an SOC change of at least `replan.soc_change_frac`; they are debounced and rate-limited.
Solves run one at a time in a single-thread executor with `solve_timeout_s`. A request while the
worker is busy is skipped and counted (`busy`); a timed-out solve keeps the worker until its thread
really ends. A plan whose first slot has started by the time it lands is discarded as obsolete.

### Intents

Each plan yields one intent per slot with `plan_id`, `plan_sequence`, `valid_from`, `valid_until`
(`min(interval end, plan expiry)`), mode, requested power, SOC target, expected price,
`inputs_hash` (SHA-256 of the frozen snapshot) and `code_version`; every record carries
`shadow: true`. A plan expires `plan_ttl_slots` (2) slots after its first slot, so intents beyond the
expiry are recorded as forecast only (`actionable: false`) and an old plan can never be "current".
A restart never resumes a persisted plan.

## Readiness

`/health/ready` is 200 only when every input is fresh **and** an unexpired plan was generated in
the last `interval + solve_lead + 60 s`; otherwise 503 with the reasons (bounded set, also exported as
`battery_optimizer_not_ready_reason{reason}`):

| Reason | Meaning |
|---|---|
| `ha_unconfigured` | `BATTERY_OPTIMIZER_HA_URL` / token not set |
| `ha_disconnected`, `ha_not_synced` | WebSocket down, or connected but the post-connect `get_states` has not completed |
| `grid_power_stale` | Grid power missing, unavailable or older than `freshness_s.grid_power` |
| `soc_unconfigured`, `soc_stale` | SOC entity not mapped (entity mode) / missing, stale or out of range |
| `prices_missing`, `prices_insufficient` | No PVPC state, or fewer than `min_horizon_slots` published slots ahead |
| `history_unavailable`, `forecast_unavailable` | Load history not fetched for today / no usable history for a target slot |
| `tariff_invalid` | `tariff.yaml` missing or unconfirmed battery fields |
| `no_recent_plan` | No plan recent enough (readiness only, never a reason to skip a solve) |

Freshness is measured from HA's own `last_reported`, not from when the service received the
value. HA fires no `state_changed` event while a sensor keeps reporting the same value, so the
planning entities are also refreshed by REST GET every `poll_interval_s` (60 s).

## Configuration

`config/service.yaml` (override with `--service-config` or `BATTERY_OPTIMIZER_SERVICE_CONFIG`) maps
roles to entity IDs and holds freshness, planning and observation settings; the file documents every
key. Entity IDs verified against the live registry on 2026-10-25 with the read-only token:

| Role | Entity | State |
|---|---|---|
| `grid_power` | `sensor.shellyem_main_channel_1_power` | present (W; sign is a user assertion, the export counter has never moved) |
| `grid_energy` | `sensor.shellyem_main_channel_1_energy` | present (kWh import counter) |
| `price` | `sensor.esios_pvpc` | present (`pvpc_updated`; `price_HHh`, `price_next_day_HHh` after ~20:00, `_d` on the repeated hour) |
| `battery_*` | — | **absent**: no Marstek / Omnibattery entity exists in HA yet (typical IDs are listed as comments) |
| observed | `sensor.shellyem_main_channel_2_power` | present (aerotherm branch, context only) |

Until the battery is integrated the committed config uses `soc_source: simulated`. Once the
Omnibattery entities exist, map `battery_soc` plus either `battery_power` (signed, + charge) or the
`battery_charge_power`/`battery_discharge_power` pair (and, ideally, the two energy counters), add the
control surface (`number.*_pd_target_grid_power`, `switch.*_battery_allow_charge/discharge`, system
caps — where `0` means *unlimited*) to `observed`, and switch to `soc_source: entity`.

The optional `archive:` section switches the two [passive archives](#passive-archives) on; without it
both are off, so a deployed config must carry it.

The variables the service reads (the complete table for every command is in
[Configuration](configuration.md#environment-variables)):

| Variable | Purpose | Default |
|---|---|---|
| `BATTERY_OPTIMIZER_HA_URL` | Home Assistant base URL (the in-cluster Service URL, set by the homelab manifest) | unset → not ready |
| `BATTERY_OPTIMIZER_HA_TOKEN` / `BATTERY_OPTIMIZER_HA_TOKEN_FILE` | Read-only, non-admin long-lived token (value or mounted file) | unset |
| `BATTERY_OPTIMIZER_PROMETHEUS_URL` | The long-term Prometheus holding the `hass_*` series (in-cluster Service URL) | unset → not ready |
| `BATTERY_OPTIMIZER_DATA_DIR` | Store root (`/data` on a PVC); must be outside a git tree | `~/data/energy-baseline` |
| `BATTERY_OPTIMIZER_CONFIG` | `tariff.yaml` | `config/tariff.yaml` |
| `BATTERY_OPTIMIZER_SERVICE_CONFIG` | `service.yaml` | `config/service.yaml` |
| `BATTERY_OPTIMIZER_PORT` | HTTP port | `8080` |
| `BATTERY_OPTIMIZER_GIT_SHA` | Build-time code identity (set by the image) | — |

The image contains neither config file: mount them (a ConfigMap at `/config`) and point the two
config variables at them. The process writes only under the data directory, so it runs with a
read-only root filesystem and needs no `/tmp`.

## HTTP endpoints

All `GET`, all read-only; there is no OpenAPI/docs route and no writable route (other methods
return 405). Bind address `0.0.0.0`, container port **8080**.

| Path | Body |
|---|---|
| `/health/live` | `{"status": "alive"}` — the process and its event loop are up |
| `/health/ready` | `{ready, inputs_ready, reasons, details}`; 200 or 503 |
| `/status` | Adapter state (connects, reconnects, resyncs, failures, last error), per-entity freshness, price curve summary, history summary, last solve (timing, outcome), virtual battery, code version, config hash |
| `/plan` | Latest plan with its snapshot summary, solver result and intents, plus the current intent (all flagged `shadow`) |
| `/metrics` | Prometheus exposition |

Kubernetes probes: use `/health/live` for liveness **and readiness**. `/health/ready` answers 503
whenever an input is stale, which would remove the pod from the Service endpoints and stop the
ServiceMonitor scrape exactly when the metrics matter; alert on `battery_optimizer_ready` instead.

## Metrics

Prefix `battery_optimizer_`; labels are bounded (roles, configured entity IDs, fixed enums). No plan
ID, price, timestamp or household reading is a label.

| Metric | Type | Labels |
|---|---|---|
| `build_info` | gauge (1) | `code_version`, `version`, `soc_source`, `mode` |
| `ready`, `inputs_ready` | gauge 0/1 | — |
| `not_ready_reason` | gauge 0/1 | `reason` |
| `ha_connected`, `ha_synced` | gauge 0/1 | — |
| `ha_reconnects_total`, `ha_resyncs_total`, `ha_state_events_total` | counter | — |
| `ha_failures_total` | counter | `stage` (`connect`, `auth`, `subscribe`, `get_states`, `heartbeat`, `disconnect`, `protocol`) |
| `ha_state_updates_total` | counter | `result` (`accepted`, `refreshed`, `duplicate`, `out_of_order`) |
| `ha_rest_polls_total` | counter | `result` |
| `input_age_seconds`, `input_stale` | gauge | `role`, `entity` |
| `price_horizon_seconds`, `prices_tomorrow_published` | gauge | — |
| `solve_duration_seconds` | histogram | — |
| `solves_total` | counter | `result` (`ok`, `error`, `timeout`, `discarded`) |
| `skipped_solves_total` | counter | `reason` (`busy` + not-ready reasons) |
| `solve_in_progress` | gauge | — |
| `plan_age_seconds`, `plan_horizon_slots` | gauge | — |
| `current_intent_power_watts`, `current_intent_active` | gauge | — |
| `history_refresh_total` | counter | `result` |
| `history_usable_slots`, `history_age_seconds`, `terminal_value_eur_per_kwh` | gauge | — |
| `shadow_slots_total` | counter | `decision` (`intent`, `no_plan`, `plan_expired`, `not_ready`, `not_observed`) |
| `shadow_planned_energy_kwh_total`, `shadow_actual_battery_energy_kwh_total` | counter | `direction` |
| `shadow_native_cost_eur`, `shadow_counterfactual_cost_eur` | gauge (since start; PVPC can be negative) | — |
| `shadow_comparable_slots_total`, `shadow_soc_violations_total`, `shadow_clipped_energy_kwh_total` | counter | — |
| `virtual_battery_soc_ratio`, `observer_late_samples` | gauge | — |
| `sink_write_errors_total` | counter | — |
| `pvpc_snapshots_total` | counter | `result` (`written`, `failed`) |
| `weather_forecast_fetches_total` | counter | `result` (`ok` = fetched and written, `failed`) |
| `archive_last_success_timestamp_seconds` | gauge (absent until the first success) | `archive` (`pvpc_snapshots`, `weather_forecast`) |

Suggested alerts (homelab): `ready == 0` for 30 m; `ha_connected == 0` for 10 m;
`input_stale{role="grid_power"} == 1` for 15 m; `plan_age_seconds > 1800`;
`increase(solves_total{result=~"error|timeout"}[1h]) > 0`; `increase(sink_write_errors_total[1h]) > 0`;
`history_age_seconds > 93600`; archives:
`time() - archive_last_success_timestamp_seconds{archive="weather_forecast"} > 10800` or
`increase(weather_forecast_fetches_total{result="failed"}[3h]) > 2`. A PVPC snapshot only exists when the sensor
changes, so alert on `increase(pvpc_snapshots_total{result="failed"}[1h]) > 0` rather than on staleness.

## Store

Under `<data_dir>/service/v1/`, partitioned by UTC day (a DST day is never split or doubled),
files `0600`, directories `0700`:

| Path | Content |
|---|---|
| `plans/date=YYYY-MM-DD/<plan_id>.json` | One immutable file per plan (atomic, no-clobber): snapshot summary and inputs, solver result, intents |
| `slots/date=YYYY-MM-DD/slots.jsonl` | One row per closed slot: decision, measurements, price, virtual-battery step, costs |
| `events/date=YYYY-MM-DD/events.jsonl` | Start/stop, HA sync/disconnect, solve skipped/error/timeout/discarded, history refresh, price publication |
| `state/virtual_battery.json` | Virtual SOC (restored after a restart if younger than 6 h) |
| `pvpc_snapshots/date=YYYY-MM-DD/pvpc_snapshots.jsonl` | One row per PVPC sensor state change (see [Passive archives](#passive-archives)) |
| `weather_forecasts/date=YYYY-MM-DD/weather_forecasts.jsonl` | One row per fetched HA weather forecast (see [Passive archives](#passive-archives)) |

JSONL rows are appended with a single `O_APPEND` write plus `fsync`; readers skip a torn last line.
The store is private household data: it never leaves the PVC and is never committed.

## Passive archives

Two append-only private archives collect data for later offline work (Price Outlook PO-B and Load
Forecaster P4). They are written through the same sink as slots and events (one `O_APPEND` write of one
complete JSON line, `fsync`, files `0600`, directories `0700`, UTC-day partitions, `schema_version: 1`),
are **never read by the planner**, and can never change readiness: a failed collection or write is logged,
counted and dropped. **There is no retention yet:** the files grow until pruned by hand (roughly a few MB
per day: one PVPC snapshot of a few kB per sensor change and one forecast of tens of kB per hour).

```yaml
archive:                      # section absent => both archives off (the code default)
  pvpc_snapshots:
    enabled: true             # committed config/service.yaml: true
  weather_forecast:
    enabled: true             # committed config/service.yaml: true
    entity_id: weather.forecast_home   # a weather.* entity
    type: hourly              # hourly | daily | twice_daily
    interval_minutes: 60      # must divide 1440
    offset_minutes: 5         # fetch at HH:05 UTC; 0 <= offset < interval
```

Unknown keys under `archive` are refused at startup. The archive settings are not part of the planning
`config_hash`.

**PVPC publication snapshots** (`pvpc_snapshots/.../pvpc_snapshots.jsonl`). One row for every accepted state
change of the `price` entity (`sensor.esios_pvpc`), and one for the first state the process sees after a
start (`trigger: "initial"`, later ones `"change"`). The 60 s REST refresh of an unchanged state is not a
change and writes nothing. Rows are written even when the state is `unavailable` or cannot be parsed into a
curve.

| Field | Content |
|---|---|
| `kind`, `schema_version`, `code_version`, `entity_id`, `trigger` | `pvpc_snapshot`, `1`, build identity, the price entity, `initial` / `change` |
| `observed_at` | Service clock when the change was applied (UTC ISO) |
| `last_changed`, `last_updated`, `last_reported` | Home Assistant's own timestamps (UTC ISO) |
| `state` | The state string |
| `attributes` | The raw attribute dict, verbatim (`price_HHh`, `price_next_day_HHh`, ...) |
| `next_day_present` | `true` when any `price_next_day_*` attribute exists |
| `reference_day` | Local date (`timezone`) of `last_updated`: the day the `price_HHh` keys refer to |

`observed_at - last_updated` is the delivery delay; the first row with `next_day_present: true` per
`reference_day` is the observed publication time.

**Weather-forecast archive** (`weather_forecasts/.../weather_forecasts.jsonl`). Every `interval_minutes`
(on a UTC grid at `offset_minutes` past; not at startup) the service calls `weather.get_forecasts` for
`entity_id` with `return_response` over the existing WebSocket session and appends one row. A fetch needs a
synced HA session; without one it fails and is retried.

| Field | Content |
|---|---|
| `kind`, `schema_version`, `code_version` | `weather_forecast`, `1`, build identity |
| `requested_at`, `received_at` | Service clock when the call was sent and answered (UTC ISO); the row is partitioned by `requested_at` |
| `entity_id`, `type` | As configured |
| `forecast` | HA's raw list for the entity (`response[entity_id].forecast`), entries untouched (`datetime`, `condition`, `temperature`, ... as the integration provides them) |

An empty or missing forecast list is a failure and is not written.

**Permission.** The only service call the service can make is `weather.get_forecasts`, and only in the form
the guard accepts (see [Safety](#safety)). The Home Assistant user behind the token must be allowed to call
that service (a normal non-admin user is); the token stays read-only in every other respect.

## Failure modes

| Failure | Detection | Behaviour |
|---|---|---|
| HA unreachable / socket drops | `ha_connected=0`, `ha_failures_total{stage}` | Reconnect with capped backoff (1 s → 60 s, ±20 % jitter); full `get_states` resync on every connect; decisions `not_ready` meanwhile |
| Token rejected | `ha_failures_total{stage="auth"}` | Retried only at the maximum backoff |
| Subscribe or `get_states` refused or not acknowledged | `stage="subscribe"` / `"get_states"` | Session ends, reconnect |
| Missed pong | `stage="heartbeat"` | Session ends, reconnect |
| Stale grid power or SOC | `input_stale`, `not_ready_reason` | No solve (`skipped_solves_total{reason}`), no decision |
| Prices missing, or next day not yet published late in the day | `prices_missing` / `prices_insufficient` | No solve until `min_horizon_slots` are published |
| Prometheus down at the daily refresh | `history_refresh_total{result="error"}` | Retried every 10 min; yesterday's history is not reused for today |
| Solver error or timeout | `solves_total{result}` | Reported; a timed-out thread keeps the worker (`busy`) |
| Store write fails | `sink_write_errors_total` | Logged; service continues |
| Archive write fails, or a forecast fetch fails (no session, call refused, timeout, empty or malformed response) | `pvpc_snapshots_total{result="failed"}`, `weather_forecast_fetches_total{result="failed"}` | Logged and counted; the record is dropped; a failed fetch is retried after 30 s, 60 s, ... capped at 10 min (never past the next grid instant); readiness, solves and slot closes are untouched |
| Restart | `build_info`, events | No plan is resumed; decisions start after the first fresh solve |

## Shadow-vs-native comparison

```bash
battery-optimizer shadow-report --since 2026-11-01 --until 2026-11-15            # Markdown
battery-optimizer shadow-report --since 2026-11-01 --until 2026-11-15 --format json
```

`--since`/`--until` accept local dates (that day's local midnight) or ISO instants with an offset;
the default is the 14 complete local days before today. The report is recomputed from the store
alone, per local day and overall:

* **Coverage:** slots with a valid intent at their start divided by the real quarter-hours of the
  window (92/96/100 per local day), with decision outcomes, skipped-solve reasons, solve failures,
  plans and HA disconnects.
* **Planned vs actual:** requested charge/discharge energy against the energy the native controller
  moved (battery counters, else integrated battery power, labelled), and the mean absolute and mean
  signed difference of slot-average power.
* **Cost:** native = `price × measured grid import`. Counterfactual = the frozen decisions replayed
  open-loop on a virtual battery starting from the first measured SOC, against the measured gross
  load `grid − battery charge + battery discharge`, clipped by power, SOC window, import ceiling and
  no-export — `price × (load + charge − discharge)`. Same slots, prices and loads for both.
  `savings_eur` is native minus counterfactual; `savings_inventory_adjusted_eur` also values the
  difference in stored energy at the end of the window at the plans' terminal value.
* **SOC-limit violations:** decisions that, applied to the measured SOC at slot start, would have
  left the configured SOC window; plus the energy the replay had to clip.
* **Assumptions** are listed in every report: pretax energy only (taxes unconfirmed), open-loop
  replay (plans saw the measured SOC, the replay carries its own), provisional battery parameters,
  no idle self-discharge, and the slots excluded for missing load or price.

`comparison_basis` is `native_battery` only when battery telemetry is mapped and `soc_source` is
`entity`. Before commissioning it is `no_battery_pre_commissioning`: the "native" side is the house
with no battery and the counterfactual is the virtual battery, which estimates the battery's value
but does **not** satisfy exit gate C. `exit_gate.met` requires at least 14 local days with ≥ 90 %
coverage on the `native_battery` basis.

## Safety

* WebSocket: every outgoing frame is checked against the allowlist `auth`, `ping`,
  `subscribe_events` (only `state_changed`), `subscribe_entities`, `subscribe_trigger` (only state
  triggers), `unsubscribe_events`, `get_states`, `get_config`. Anything else raises
  `ForbiddenHACallError` before reaching the socket.
* REST: `GET` only, on `/api/`, `/api/config`, `/api/states`, `/api/states/<entity_id>`, enforced by
  an `httpx` request hook.
* The single exception to "no service call" is **`weather.get_forecasts` with `return_response: true`**,
  accepted only as the exact frame `{"id": <int>, "type": "call_service", "domain": "weather", "service":
  "get_forecasts", "return_response": true, "service_data": {"type": "hourly" | "daily" | "twice_daily"},
  "target": {"entity_id": "weather.<name>" | [...]}}`. Another domain or service, a missing or non-`true`
  `return_response`, any additional field (in the frame, `service_data` or `target`, so no area, device or
  label targets), a non-`weather.` entity or a non-integer `id` raises `ForbiddenHACallError` before the frame
  reaches the socket. It is non-mutating: the service only returns the forecast the weather entity already
  holds, it changes no state and operates no device. The only caller is the passive forecast archive, which
  builds the frame with `ha_guard.weather_forecast_request`. REST `POST` stays forbidden.
* The HTTP API has no writable route. Tests assert all of the above and statically keep Modbus/MQTT/
  HA libraries and HTTP write calls out of the package.
* Use a **non-admin, read-only** HA user's token; the service never needs more.

## Local run

```bash
export BATTERY_OPTIMIZER_HA_URL=https://homeassistant.example.org
export BATTERY_OPTIMIZER_HA_TOKEN_FILE=/path/outside/repo/ha_token   # 0600
export BATTERY_OPTIMIZER_PROMETHEUS_URL=https://prometheus.example.org
export BATTERY_OPTIMIZER_DATA_DIR=/path/outside/repo/data
battery-optimizer serve --port 8080
curl -s localhost:8080/health/ready
```
