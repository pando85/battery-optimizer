# Load Forecaster — improvement plan

- **Status (2026-10-25):** P0 and P3 are done. The P1 forecaster seam (`load_forecast.py`) and the
  P1 evaluation harness (`forecast-eval`, see
  [Benchmark reference](../benchmark.md#forecast-evaluation-forecast-eval)) are built. The P0 run
  is done and its results are private. P2 waits for the harness. P4 is planned but blocked on the
  [Price Outlook](components/price-outlook.md) next-day estimate. The plan refines the
- **Question:** can a better household load forecast reduce the planned battery's net cost, and by
  how much? Answer it with the least machinery first. Stop as soon as the evidence says the remaining
  gain is not worth the maintenance.
- **Scope:** offline evaluation first. Then a **forecast-only** shadow in the existing service, which
  needs no battery. Read-only throughout: no actuation and no new write credentials. Raw and
  predicted series stay in the private data dir, and no monetary figure goes into this repo.

## Progress

| Phase | Status | Where | Gate |
|---|---|---|---|
| P0 headroom | **Done** | `perfect` oracle in #14; private run `forecast-eval/p0-*` | G0: owner threshold pending |
| P1 harness | Seam **done** (#14); harness in review (#16) | `load_forecast.py`, `forecast-eval` | G1: seam passed (bit-identical, leakage test) |
| P2 cheap candidates | Pending P1 | — | G2 pre-registered (see P2) |
| P3 sub-load audit | **Done** | #15: `build-features`, `load_decomposition.py`; private feature set and P3 report | **G3 passed** |
| P4 structured forecaster | **Planned, blocked** on Price Outlook PO-1/PO-2 | this document, §P4 | G4 |
| P5–P7 | Not started (P5 collection continues passively) | — | — |

All phases run on the refreshed private dataset `dd5a2178…`, which covers 2024-01-01 to 2026-10-01.
Its rows for 2024–2025 are identical to the canonical `2e01620b…`, so 2026-01…09 can serve as a
second evaluation period.

## 1. Where we start

**LF-0 (production and benchmark):** `forecast_load_trailing_slot_mean`. For each 15-minute slot
position it takes the mean of the same position over the previous 28 local days, cut at the
decision's local midnight. It uses no weekday, holiday, temperature or schedule information.

Evidence already on record:

| Finding | Source | Implication |
|---|---|---|
| Ranking windows by MAE gives the opposite order to ranking them by dispatch cost (7 d best MAE, 28 d cheapest; on the frozen holdout, 28 d best MAE, 365 d cheapest) | [progress §Forecast experiments](energy-baseline-progress.md#forecast-experiments), [experiments §11](energy-baseline-forecast-experiments.md) | Never select a forecaster on MAE. EUR is the primary metric. |
| About 24 % of planned discharge is clipped because measured load fell below the forecast | [progress](energy-baseline-progress.md) | Forecast error does reach execution, but its EUR value has never been isolated. |
| The ordering of `rolling_horizon` against `day_ahead` flipped with the load-forecast basis (frozen vs causal) | progress, *bounded pre-migration milestone* | The ladder is sensitive to the load basis, so the experiments need paired comparisons. |
| The 81-arm grid has no pinned evaluation code | progress, *provenance blast radius* | New runs must stamp code and config identity. |

**Signals verified in long-term Prometheus on 2026-10-25 (read-only):**

| Signal | Series | History from | Notes |
|---|---|---|---|
| Main meter | `sensor.shellyem_main_channel_1_*` | 2024-01 | Already the canonical gross load (battery absent). |
| Servers | `sensor.shellypmmini_servers_power` (`hass_sensor_power_w`) | 2024-01-29 | 30-day mean about 215–245 W. That is consistent with the owner's "~250 W" estimate, a little lower. |
| Heat pump, electrical input | `sensor.aquarea_{heat,dhw,cool}_power_consumed` | 2024-01-29 | Split into heating, DHW and cooling. The total `sensor.aquarea_power_consumed` only exists from 2025-12-10. |
| Outdoor temperature | `sensor.aquarea_outside_ambient_temperature` (`hass_sensor_unit_celsius`) | 2024-01-29 | Measured, not forecast. Using it as a feature for future slots leaks future information (see P4). |
| AppDaemon controls | `input_boolean.appdaemon_climate_enable`, `input_number.appdaemon_climate_min_hours_per_day`, `input_boolean.appdaemon_dhw_enable`, `switch.aquarea_force_dhw_mode` | 2024-01 (DHW enable from 2025-03) | At 00:00:40, `climate_control` takes every hour priced below ⅔ of the trailing 7-day mean, or the cheapest `min_hours_per_day` hours if that gives fewer. It runs in `Heat+DHW` **or `Cool+DHW`** mode (`input_select.appdaemon_climate_heat_mode`, which has no value series), and the off mode is `DHW`. `dhw_control` forces DHW at the cheapest hour of each 12-hour block. Both rules are price-driven and can be replayed (see P3 and P4). |
| Washer energy | `sensor.terrace_lavadora_energy_today` (from 2026-07), `sensor.lavadora_cycles` / `_energy_yesterday` (from 2025-11) | see notes | Daily washer energy labels for P5. `sensor.smart_plug_v2_terrace_power` is **not** the washer: it reported only between April and July 2024, with small loads. |
| Appliance enums | `sensor.lavadora_*`, `sensor.lavavajillas_*` (status, remaining, delayed start/end) | HA recorder only, extracted since 2026-10-15 | Prometheus has only `hass_entity_info` for the enum states, so history is short and has to be accumulated. |

## 2. Phases

```
P0 headroom ──► P1 harness ──► P2 cheap baselines ──► P4 structured ──► P6 ML/quantiles ──► P7 shadow
                   ▲                                    ▲   ▲               ▲
P3 data audit ─────┴────────────────────────────────────┘   │               │
Price Outlook PO-1/PO-2 (next-day PVPC estimate) ───────────┘               │
P5 appliance data (passive accumulation starts now) ────────────────────────┘
```

P0 and P3 can run in parallel. P5 only collects data until it has enough history. Each phase ends
at a gate. Failing a gate stops the plan there, with the simpler forecaster kept and the negative
result recorded.

### P0 — Headroom: what is perfect load knowledge worth? (≈1 day)

The aim is to bound the prize before building anything.

- Add an offline-only `load_forecast_mode: perfect`. It feeds the measured load of the target day
  to `day_ahead` and `rolling_horizon` as the forecast. Prices, battery, SOC seed, terminal value
  and solver stay identical. The service must reject this mode.
- Run `causal` (LF-0) against `perfect` on the full 2025 year and on 2026-01…09.
- Report:
  - the EUR gap per year, per month and per season;
  - the change in clipped charge/discharge kWh;
  - the per-day distribution of the gap, to see whether it comes from a few days or from everywhere;
  - which slots carry it (price tercile × hour).
- **Gate G0:** the owner writes down a minimum worthwhile annual gain before seeing the result. If
  the whole gap is below it, stop after P2, because only near-zero-cost tweaks are justified. Every
  later phase reports its result as a **share of this gap captured**.

**Result (2026-10-25; amounts private).** Runs were made for 2025 and for 2026-01…09, under
`day_ahead` and `rolling_horizon`, with exact prices.
- Perfect load knowledge is worth **about a fifth of the battery's net saving** over the
  no-battery baseline, and the share is similar in both periods and under both policies.
- The gap is **broad**. It is positive on most days, the ten largest days carry only about
  13–17 % of it, and every month contributes.
- It sits in the **most expensive price tercile, mainly 18:00–21:00 local**. A perfect forecast
  discharges more into the evening peak, partly by charging in different cheap slots.
- Clipped charge and discharge disappear entirely under perfect load.
- LF-0's WAPE is about 47 % per slot, with negligible bias.
- G0 is pending the owner's threshold, which stays outside the repo.

### P1 — Evaluation harness (≈2–3 days)

This makes every candidate cheap to compare and impossible to leak future data.

- **Forecaster seam (built):** `load_forecast.run_forecaster(name, load, usable, days, cutoff_idx,
  target_days)` returns the forecast with provenance. A non-oracle forecaster sees only read-only
  history before the cutoff. LF-0 (`trailing_mean`) is the first implementation, and `perfect` is
  the P0 oracle. Select a forecaster with `--load-forecaster` in the benchmark CLI and with
  `planning.load_forecaster` in `service.yaml` (default `trailing_mean`; oracles are rejected).
  To add a candidate, write one `@register_forecaster` function. Planner and MPC internals stay
  unchanged.
- **Walk-forward folds:** use monthly folds from 2025-01 to 2026-09, which gives 21 folds covering
  every season twice. Each fold fits only on data before each decision's cutoff. Decision instants
  are the policy's real events: 00:00 for `day_ahead`, and 14:00 and 20:30 for `rolling_horizon`.
- **Forecast metrics:**
  - MAE, WAPE, bias and P95 absolute error;
  - each of those by horizon bucket (0–6 h, 6–24 h, 24–36 h), price tercile, weekday/weekend and season;
  - a **price-weighted error** `Σ price·|error|` as a cheap EUR proxy. Check that it agrees with real EUR before trusting it.
- **Economic metric:** paired EUR difference against LF-0 on identical days, with a **weekly block
  bootstrap** 95 % interval, plus the share of the P0 gap captured.
- **Provenance:** every run stamps the git commit, dirty flag, config hash, dataset id and
  forecaster name and parameters. This closes the provenance gap in the 81-arm grid.
- **Pre-registration:** a short file in the private run dir lists the candidates, folds and
  thresholds before the first candidate runs.
- **Gate G1:** LF-0 passed through the seam reproduces the canonical five-policy run bit-for-bit
  (golden digest), and a leakage test passes. The test perturbs every post-cutoff value and checks
  that the forecast does not change.

### P2 — Cheap statistical candidates, no new data (≈2 days)

All of these are variations of the existing function:

| Candidate | Idea |
|---|---|
| `window_N` | N ∈ {7, 14, 28, 56}, re-run on the folds rather than one year |
| `median` / `trimmed_mean` | Robust to occasional appliance spikes; the median is the MAE-optimal point estimate |
| `ewma_h` | Exponentially weighted over days, half-life h ∈ {3, 7, 14} |
| `daytype` | Separate profiles for weekday, Saturday and Sunday/holiday. Holidays come from the `holidays` package (Spain plus a configured subdivision). |
| `level_adjusted` | Profile × (last 24 h measured ÷ last 24 h profile), clipped. At 14:00 and 20:30 events it also uses today's observed slots. |
| `blend` | Weighted mix of recent and same-season-last-year profiles |

**Gate G2 (pre-registered in each harness run):** the primary policy is `rolling_horizon` and the
secondary is `day_ahead`. A candidate passes only if, under the primary policy and in **both**
periods (2025 and 2026-01…09):
- the 95 % weekly-block-bootstrap interval of its paired daily saving against `trailing_mean` lies
  entirely above zero, **and**
- its point estimate is at least 10 % of that period's P0 gap.

The best passing candidate becomes the new reference for P4. Otherwise LF-0 remains the reference.

### P3 — Sub-load data audit and extraction (≈2 days, parallel with P0–P2)

This phase finds out which part of the load is predictable and where the error comes from.

- Extend `extract_prom.ENTITY_SPECS` with:
  - servers power;
  - heat-pump heat, DHW and cool consumed power;
  - outdoor temperature;
  - the AppDaemon enable, min-hours and DHW switch series.

  Write them to a **separate feature table** (`analysis/v1/features/`) so the canonical `dataset_id`
  and the golden runs stay unchanged.
- Quality checks:
  - units and sample cadence;
  - gaps, using the same quality mask as the meter;
  - `servers + heat_pump ≤ main` per slot, with a tolerance and a count of violations;
  - that `*_consumed` is electrical input and not thermal output.
- **Decomposition report** (private): for each season, the share of kWh and the share of LF-0 error
  variance that come from servers, the heat pump and the residual. Error during expensive slots is
  reported separately.
- **Gate G3:** P4 goes ahead only if the heat pump (or another observable component) carries a
  material share of the error. If the residual dominates, skip to P5 or P6.

**Result (2026-10-25, #15; tables in the private P3 report).**
- **Delivered:** 15 entities registered in `ENTITY_SPECS`, none of them default extraction. They
  were extracted for 2024-01 to 2026-10. The `build-features` subcommand writes a separate
  slot-aligned feature table with its own manifest.
  - **Integration rule:** each sample's power is held until the next sample. A span longer than
    300 s counts as a gap. Coverage of 0.8 or more is `partial`; below that the slot is `unknown`
    (null, never zero).
  - `build` now ignores manifests that contain only feature entities, so extracting them never
    changes the canonical `dataset_id`.
- **Quality:**
  - Every series is sampled every 60 s.
  - The produced/consumed ratios (heat ≈ 4.5, DHW ≈ 2.9, cool ≈ 5.1) confirm that `*_consumed` is
    electrical input.
  - `servers + heat_pump ≤ main` fails in 0.16 % of slots. The causes are main-meter dropouts in
    March 2024 and one stuck heat-pump reading in December 2024, and those slots are excluded from
    the decomposition.
- **Decomposition of LF-0 error:**

  | Component | Share of error variance | In the expensive tercile |
  |---|---|---|
  | Heat pump | about 65 % | about 61 % |
  | ↳ DHW | about 32 % | |
  | ↳ heating | about 18 % | |
  | ↳ cooling | about 15 % | |
  | Residual | about 34 % | |
  | Servers | about 0 % | |

  A perfect heat-pump forecast would remove about half of the price-weighted error. Its share is
  55–70 % in every season.
- **G3 passed:** the heat pump is observable and carries the majority of the error.
- **Correction from P4 planning:** the P3 report treated cooling as unscheduled. In fact
  `climate_control` schedules cooling as well. See P4 for the replay of the full rule.

### P4 — Structured component forecaster (LF-1, planned 2026-10-25, blocked on Price Outlook)

`forecast = servers + heat_pump + residual`. The residual model is trained on
`main − servers − heat_pump`, which prevents double counting.

**Evidence from P3 and the rule replay (private numbers in the P3 report):**
- The heat pump carries about two thirds of LF-0's error variance, both overall and in the most
  expensive tercile. Servers carry almost none.
- `climate_control` schedules **both heating and cooling**. `input_select.appdaemon_climate_heat_mode`
  switches between `Heat+DHW` and `Cool+DHW`, while the off mode is `DHW`. The real rule takes every
  hour priced below ⅔ of the trailing 7-day mean price, and falls back to the cheapest
  `min_hours_per_day` hours when there are fewer than that. Replaying it on enabled days places
  about 97 % of heating energy and about 98 % of cooling energy inside the selected hours, which
  cover about half the day. The simplified "cheapest N hours" rule captures less.
- So **when** the heat pump runs is almost fully known as-of. **How much** it uses per selected hour
  is the hard part.
- On disabled days cooling also runs under manual control, and DHW follows its 12-hour rule only
  partly.

**Steps:**

| Step | Component | As-of inputs | Model |
|---|---|---|---|
| P4a | Schedule mask | Published PVPC for the day, trailing 7-day mean price, `min_hours_per_day`, enable flag, mode (heat or cool) inferred from the last 48 h of heat vs cool consumption, because the mode select has no value series | Exact replay of `climate_control`, versioned against the AppDaemon source and validated against observed heat/cool consumption |
| P4b | Energy per selected hour | Mask, outdoor temperature (as-of), local hour, block length and position, number of selected hours, mode | Degree-hour or binned regression per mode, fitted walk-forward per monthly fold (owner choice: simple first; LightGBM stays in P6) |
| P4c | DHW | `dhw_control` 12-hour cheapest-slot replay plus trailing profile of the remainder | Rule-hour energy plus profile |
| P4d | Servers and residual | `main − heat_pump` | Best P2 forecaster |
| P4e | Fallbacks | Rule disabled, mode unknown, prices or temperature stale | Heat-pump part falls back to LF-0, with the reason recorded in provenance |

**Owner decisions (2026-10-25):**
- **Temperature:** backtest with persistence, i.e. the same hour yesterday as measured. Start
  archiving HA `weather.forecast_home` hourly forecasts now, so a real forecast can be evaluated
  later. That needs the `ha_guard` allowlist to permit the read-only `weather.get_forecasts` call.
- **No extra climate signals for now.** Leave out thermostat setpoints, zone request temperatures and
  compressor frequency, and add them only if P4b underperforms.
- **Model:** simple regression first.
- **Dependency:** at the 14:00 decision tomorrow's PVPC is not published, so tomorrow's schedule is
  unknown. The owner wants this solved properly in the
  [Price Outlook](components/price-outlook.md) first. That means PO-1, decision-time publication
  snapshots, and PO-2, an early next-day PVPC estimate. **P4 implementation waits for that work.**
  Then P4a replays tomorrow's rule on the estimated prices, flagged as estimated, before
  publication.

**Can start now** (passive data collection, independent of the block): the weather-forecast archive,
and PO-1's PVPC publication snapshots.

- **Gate G4:** the paired EUR result beats the P2 reference with a bootstrap interval that excludes
  zero, in at least one heating and one cooling season. A mid-run change to the AppDaemon rule is
  handled by versioning, not by refitting silently.

### P5 — Appliance-aware signals (LF-2, collect now, evaluate later)

- Data accumulates passively starting now. Keep `extract_ha` running for the washer and dishwasher
  enums, because the recorder only keeps 8 days of lookback. Confirm the terrace plug is the washer
  so it can serve as an energy label.
- **The highest-value feature is a *known future start*.** A dishwasher `delayed_start` or a washer
  `delayed_end` that has been programmed is a load at a known time and fully available as-of. The
  next most useful is the remaining energy of a cycle already running. Probabilities of unknown
  future starts come last.
- Evaluate once there are about 3 months of cycles. The split is session-disjoint, and results are
  reported separately for short horizons (0–6 h), where most of the benefit is expected.
- **Gate G5:** same EUR rule as G4. Classification accuracy or MAE alone do not count.

### P6 — ML challenger and quantiles (LF-3, only if gap remains)

Start only if P2–P5 leave a large share of the P0 gap uncaptured.

- LightGBM on:
  - slot of day, day type and holiday;
  - lags at 24 h and 168 h;
  - temperature (as-of forecast);
  - the replayed schedule mask;
  - appliance states.
- Train with quantile objectives for P10/P50/P90.
- Only P50 feeds the deterministic MPC. Quantiles are reported for calibration and kept for a future
  risk-aware planner. Wheels exist for amd64 and arm64, but adding the dependency to the image needs
  owner sign-off.
- **Gate G6:** improvement over P4 holds across folds and seasons, and P10/P90 coverage is within
  ±5 points. Otherwise keep the simpler model.

### P7 — Forecast-only shadow and promotion (LF-4, ≈2 days + 4 weeks)

**This phase does not need the battery.** Forecast accuracy can be shadowed live today.

- The service computes LF-0 and the candidate on each plan. Only the configured forecaster feeds the
  planner.
- Once each slot's measurement is in, record realized errors. Expose them as
  `battery_optimizer_load_forecast_abs_error_kwh{forecaster,horizon_bucket}` and add a Grafana panel.
- Fall back to LF-0 when any input is stale or missing (temperature, schedule replay, appliance
  state). The fallback reason goes on the existing provenance and is exported as a counter.
- After 4 weeks of shadow in which the live error ordering matches the offline one, promote by
  changing `planning.load_forecaster` in `config/service.yaml`, re-vendored into the homelab chart.
  Rollback is the reverse one-line change.
- **Revalidate after battery commissioning:** gross load becomes
  `import − export + discharge − charge`. Re-run the P1 harness on the post-commissioning data
  before trusting the forecaster on it.

## 3. Decision gates summary

| Gate | Pass condition | If it fails |
|---|---|---|
| G0 | P0 gap ≥ owner's predeclared minimum | Do only P2, then stop |
| G1 | LF-0 through the seam is bit-identical; leakage test passes | Fix the harness before any candidate runs |
| G2 | A P2 candidate's EUR bootstrap interval is entirely better than LF-0's | Keep LF-0 as reference |
| G3 | An observable component carries material error | Skip P4 |
| G4 | P4 beats the reference with an interval excluding zero, in ≥2 seasons | Keep the P2 reference |
| G5 | Appliance features add EUR on top of the reference | Keep collecting, or drop |
| G6 | ML is stable across folds and quantiles are calibrated | Keep the simpler model |
| G7 | 4-week live shadow is consistent with offline results | Do not promote |

## 4. Risks

- **Small prize:** the battery is 5.12 kWh, and a perfect forecast may be worth little. P0 exists to
  find this out first.
- **Behaviour drift:** occupancy, new servers and changes to AppDaemon rules. Mitigated by walk-forward
  folds, rule versioning and the live shadow error metric.
- **Temperature leakage:** only as-of forecasts are allowed, and the leakage test covers feature
  inputs as well as load.
- **Dataset churn:** sub-load features live in a separate table, so canonical runs stay reproducible.

## 5. Owner decisions needed

1. G0 minimum worthwhile annual gain (kept private, outside the repo).
2. ~~The temperature source for P4~~ — decided: persistence for the backtest, plus archiving the HA
   weather forecast (see P4).
3. ~~Whether the terrace plug meters the washer~~ — P3: it does not. It reported only in
   April–July 2024 and its loads are small. Washer energy labels come from
   `sensor.terrace_lavadora_energy_today` and the Home Connect counters instead.
4. The holiday subdivision for `daytype`.
5. Whether LightGBM may be added as an image dependency (P6 only).
