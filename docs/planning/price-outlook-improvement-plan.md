# Price Outlook — improvement plan

- **Status (2026-10-25):** PO-A done, **gate GA failed**: knowing tomorrow's prices at 14:00 is worth
  very little to dispatch. PO-B shipped in 0.3.0 and is collecting. **PO-C was restarted the same
  day** because Load Forecaster P4 met its restart condition (the 14:00 schedule guess on
  `persistence` prices misses 17-24 % of the measured climate energy, see the
  [Load Forecaster plan](load-forecaster-improvement-plan.md)); it is built (`omie_outlook.py`) and
  evaluated, **gate GC passed** (see §PO-C). PO-D and PO-E stay **deferred**. Refines the
  [Price Outlook roadmap](components/price-outlook.md) stages PO-1…PO-4 into executable phases.
- **Why now:** P4 of the [Load Forecaster plan](load-forecaster-improvement-plan.md) replays
  AppDaemon's price-driven heat-pump schedule. At the 14:00 decision tomorrow's PVPC is not
  published, so neither tomorrow's prices nor tomorrow's schedule are known. The owner decided this
  must be solved here first.
- **Question:** how much is knowing tomorrow's prices at 14:00 worth, and how close can a
  decision-time estimate get to it?
- **Scope:** read-only. Use public market data only. The MPC sees no estimated price until the
  shadow gate passes, and no monetary figure goes into this repo.

## 1. Where we start

| Fact | Evidence | Implication |
|---|---|---|
| PVPC is **hourly**, with no variation within an hour through 2026-09. REE publishes D+1 at about 20:15, and the HA integration fetches from 20:00 into `price_next_day_HHh`. | `service/pvpc.py`; dataset `dd5a2178…` | Between about 13:00 and 20:15, tomorrow is unknown to the service. |
| The service plans **only up to the last published price** and then uses a terminal value. | `service/runtime.py` | The live 14:00 plan does not see tomorrow at all. |
| The canonical `rolling_horizon` rung gives the 14:00 event **exact** prices for tomorrow (σ = 0). | `rolling_horizon.py`, [benchmark](../benchmark.md) | The benchmark overstates what is achievable today. That rung is an upper bound, not the service's behaviour. |
| No archive records when a price was published. Prometheus keeps only the current-hour value, and the HA recorder keeps 8 days. | [roadmap PO-1](components/price-outlook.md) | The *published = realized* assumption is still unverified. Snapshots have to start being collected. |
| OMIE day-ahead marginal prices are **public files without auth** (`marginalpdbc_YYYYMMDD.1`). They are published at about 13:00 local for D+1, are **15-minute since 2025-10-01** (96, or 92/100 on DST days), and history goes back years. | Fetched 2026-10-25: D+1 was not yet present at 12:24 local | A realistic estimate source exists before the 14:00 decision. |
| **Feasibility probe:** PVPC ≈ β·OMIE + an intercept per tariff period (2.0TD P1/P2/P3), fitted on 2026-08 and tested on 2026-09. | Scratch analysis, private | Compared with persistence (yesterday's PVPC): MAE about halved, within-day rank correlation about 0.97 against 0.87, overlap of the cheapest 10 hours about 93 % against 81 %. Ordering, which is what dispatch and the AppDaemon rule depend on, is well preserved. |
| Earlier synthetic-noise grid: price-estimate noise cost about a tenth of the policy margin | [experiments §11](energy-baseline-forecast-experiments.md) | A 0.88-era result on synthetic noise. PO-A re-measures it on real data. |

## 2. Phases

```
PO-A headroom ──► PO-C OMIE estimator (offline) ──► PO-E shadow ──► (feeds Load Forecaster P4a)
PO-B snapshots (passive, start now) ──┘ (as-of times)    ▲
PO-D scenarios (optional) ───────────────────────────────┘
```

### PO-A — Headroom: what is tomorrow at 14:00 worth? (≈1–2 days)

- Add `rolling_horizon` price-information modes. The canonical default stays unchanged.
  - **`published_only`:** the 14:00 event plans only through the last published price and uses
    the terminal value. This is what the service does today.
  - **`exact`:** today's canonical behaviour, the upper bound.
  - **`persistence`:** tomorrow is estimated as today's prices, a naive estimate.
- Run 2025 and 2026-01…09 with the P1 harness machinery: paired daily EUR and a weekly block
  bootstrap.
- Report how often the extra hours change the 14:00–20:30 actions.
- **Gate GA:** the gap `published_only − exact` must be material, against an owner threshold kept
  private. Independent of the dispatch value, P4 still needs a tomorrow schedule at 14:00, so PO-C
  runs anyway, but GA decides whether the MPC should ever use the estimate.

**Result (2026-10-25, private run `price-eval/po-a-*`, `price_outlook.py`, `battery-optimizer price-eval`):**
- `exact` reproduces the canonical rung bit for bit; leakage tests show `published_only` and
  `persistence` never see an unpublished price.
- The gap `published_only − exact` is about **3 %** of the P0 load-forecast gap in 2025 (bootstrap
  interval above zero) and about **1 %** in 2026-01…09 (interval includes zero). The extra hours
  change the 14:00–20:30 actions on more than half of the days, but by very little money: the 20:30
  replan already sees the published prices.
- `persistence` captures about half of that gap in 2025 and almost none in 2026. As a price
  estimate it has negligible bias, a within-day rank correlation of about 0.85 (median) and about
  75 % overlap with the true cheapest 10 hours.
- **GA failed** (owner, 2026-10-25). An estimated tomorrow is not worth building for dispatch.

### PO-B — Publication evidence (start now, passive)

- The service appends a private snapshot of every `sensor.esios_pvpc` state change: raw
  attributes, `last_updated`, `observed_at`, and whether `price_next_day_*` is present. It goes
  into `data_dir/service/v1/` through the existing sink.
- From the snapshots, measure the actual publication time distribution and verify *published =
  realized* against the Prometheus realized series.
- This ships in the same release as the HA weather-forecast archive agreed in Load Forecaster P4.
  Both are read-only, and the weather archive needs the `ha_guard` allowlist to permit
  `weather.get_forecasts`.

### PO-C — OMIE-based next-day estimator (offline) — restarted and built 2026-10-25

> Deferred after GA on 2026-10-25 and restarted the same day: Load Forecaster P4 showed that its
> 14:00 guess of tomorrow's heat-pump schedule (from `persistence` prices) is what keeps the
> structured forecaster from winning in the heating season. The OMIE extract source and its
> 2024-01-01 onward backfill already existed (`extract --source omie`, `omie_prices.py`, as-of
> 13:30 local), so the work was the estimators and their evaluation.

**Built** (`omie_outlook.py`, registered in the price-outlook seam; evaluated by `price-eval`):

| Name | Built | Notes |
|---|---|---|
| `omie_period` | yes | `PVPC_hour = beta * mean(OMIE ES over the hour) + intercept(2.0TD period)`: one slope and one intercept per period (P3 is also weekends and the nine national fixed-date holidays), least squares on the hourly prices of the last 28 fully published local days |
| `omie_period_month` | yes | The same with a level offset per calendar month of a 120-day window, predicted at the latest month's level |
| `omie_quantile` | **no** | It exists to feed PO-D, which stays deferred after GA |

- **Seam.** An outlook registered with `needs_omie=True` receives the OMIE series as a side input
  (`OutlookSide`), cut by the seam at the first OMIE day not yet usable at the decision
  (`omie_end_idx`, D+1 from 13:30 local on D), as it cuts the published PVPC. Before 13:30 the event
  falls back to `published_only`. A missing side input, a missing OMIE quarter in a target hour, or
  fewer than 7 published days in the fit window leaves the slot `unavailable` (the
  `published_only` behaviour), with the reason recorded per event and counted in the telemetry.
- **Leakage tests** (`tests/test_omie_outlook.py`): an OMIE day that is not usable at the decision,
  and any realized PVPC after the publication boundary, cannot change the estimate; D+1's OMIE
  changes it from 13:30 and not before; the seam's cut and `omie_prices.omie_usable_from` are one
  rule; the dispatcher plans on the estimate and records the side input.
- **Fit windows chosen on 2024 data only, before the 2025 and 2026 evaluation was run.** On the
  274 days of 2024-04-01 to 2024-12-31 (14:00 events, price metrics of `price-eval`; no 2025 or 2026
  row was read for this): `omie_period` windows 14, 28, 56 and 91 days gave a price MAE within
  8 % of each other, the lowest at 28 days (14 days a close second); within-day Spearman and the
  cheapest-10 overlap were equal to two decimals, so **28 days** was kept. `omie_period_month`
  windows 56, 91 and 120 days gave MAE decreasing with the window, so **120 days** was kept (the
  longest the 2024 history allows before the first evaluated day, so the grid edge, not an
  interior optimum). On the same 2024 days `persistence` had an MAE about four times larger.
  The choice rule (lowest MAE; ties on rank metrics go to the shorter window) was stated after
  looking at the 2024 table, not before; it only used 2024 rows.

**Result (2026-10-25, private run `price-eval/po-c-*`, pre-registered with gate GC before any
benchmark ran; relative figures only).** 2025 and 2026-01...09, 14:00 events, OMIE complete over both
periods (no fallback event).

- **Gate GC passed** for both estimators: in both periods each beats `persistence` on within-day
  rank, on the cheapest-10 overlap, on the replayed climate-mask overlap and on the climate energy
  inside the replayed mask. The dispatch criterion of the gate is not applied because GA failed.
- **Price estimate.** The mean absolute error of `omie_period` is about half of `persistence`'s
  (51 % in 2025, 46 % in 2026), with a small bias (a slight over-estimate in 2026, under a quarter
  of the error). Within-day Spearman, mean (median): 0.95 (0.98) in 2025 and 0.93 (0.97) in 2026,
  against 0.70 (0.84) and 0.75 (0.87) for `persistence`. Overlap of the 10 cheapest hours: 93 % and
  92 % against 74 % and 76 %; of the 4 most expensive: 90 % and 90 % against 73 % and 77 %. On the
  four 2026 days with a negative price the error is about a third of `persistence`'s. On the three
  DST days of the periods the estimators are **not** better than `persistence` (a sample of three).
- **Overlap with the AppDaemon rule's hours (what P4 needs).** Replaying `climate_control` on the
  estimate against on the realized prices, over the 223-227 days per period with a schedule: mean
  hour overlap (Jaccard) 0.89 and 0.93 against 0.67 and 0.76 for `persistence`. The share of measured
  climate energy that falls inside the replayed hours is **92 % and 94 %** against **76 % and 83 %**
  for `persistence`; with the realized prices it would be 97 % and 98 %. So `omie_period` closes
  about three quarters of the gap between persistence and perfect price knowledge, in both periods.
- **Variants.** `omie_period_month` is indistinguishable on every metric (mask overlap 93 % and
  94 % of the climate energy, price error within 5 % of `omie_period`), so the simpler `omie_period`
  stays the one P4 uses.
- **Dispatch (information only).** In 2025 `omie_period` captures about 70 % of the GA gap, with an
  interval that touches zero; in 2026 it captures none, but that period's gap is itself
  indistinguishable from zero. GA stays failed: the estimate is for the load forecaster, not the MPC.

**Original spec (kept for reference).**

- **Data** (built before, PR #24):
  - A new `extract` source `omie` writes the public day-ahead files into the raw store with
    manifests and hashes, covering 2024-01 to now.
  - Parse the hourly files before 2025-10 and the quarter-hour files after it, map them to UTC
    slots, and handle DST days.
  - Record the as-of rule: an OMIE D+1 value may be used only from **13:30 local**. That is
    conservative until PO-B and the shadow measure the real fetch time.
- **Price-estimator seam:** mirror the load-forecaster seam. Registered estimators return slot
  prices with `published`, `estimated` or `unavailable` provenance per slot, and a published value
  is never overwritten.
- **Candidates:**

  | Name | Inputs | Idea |
  |---|---|---|
  | `persistence` | Today's PVPC | Baseline |
  | `omie_period` | OMIE D+1, 2.0TD period (weekday, hour, national holidays) | β·OMIE + intercept per period, fitted on a trailing window |
  | `omie_period_month` | Same plus month or trailing-level terms | Tracks drift in regulated components and in the futures-basket share |
  | `omie_quantile` | Same | P10/P50/P90 residual quantiles (feeds PO-D) |

- **Metrics:**
  - MAE, bias, and within-day Spearman rank correlation;
  - overlap of the cheapest N hours and of the most expensive 4 hours;
  - **overlap of the AppDaemon rule's selected hours**, which is what P4a needs;
  - errors on negative-price and DST days;
  - paired EUR in `rolling_horizon` with the estimate at 14:00, against `published_only` and
    `exact`, reported as the share of the GA gap captured.
- **Walk-forward:** monthly folds from 2025-01 to 2026-09, with each fit using only data before the
  decision. Pre-register before running.
- **Gate GC:** the estimate beats persistence on rank and rule-hour overlap in both periods, and if
  GA was material, it captures a meaningful share of the gap with a bootstrap interval above zero.

### PO-D — Scenario prices (optional)

This phase starts only if PO-C's error is concentrated on specific days, so that a point estimate
misleads the MPC. It produces coherent day-level scenarios that preserve within-day correlation.
They feed the MPC only once the Dispatch Optimizer has an approved scenario interface.

### PO-E — Live shadow (≈2 days + 4 weeks)

- The service fetches OMIE D+1 once a day, after publication, with retry and a staleness flag.
  Egress is unrestricted once the NetworkPolicy is removed (see §3).
- It computes the estimate and records it with provenance. At the 20:15 publication it scores the
  estimate and exports `battery_optimizer_price_estimate_abs_error{metric}` along with rule-hour
  overlap metrics.
- The MPC keeps using published prices only. A config flag extends the horizon with estimated slots
  **only after** 4 weeks of shadow consistent with offline results. The P4a schedule replay may
  consume the estimate as soon as GC passes.
- Fallbacks: if OMIE is missing, stale or fails to parse, use the published-only horizon and P4's
  profile fallback, and record the reason.

## 3. Owner decisions

Decided 2026-10-25:
- **Source:** OMIE public files. No ESIOS token.
- **PO-B ships now**, together with the HA weather-forecast archive from Load Forecaster P4, as a
  service release and homelab deploy.
- **Egress:** the service may reach `www.omie.es`. The owner also questioned why the
  battery-optimizer namespace has a restrictive egress NetworkPolicy at all. First-party,
  controlled software in this cluster normally runs without one; network policies are kept for
  uncontrolled workloads such as LLM harnesses. The policy is removed in the homelab deploy that
  ships PO-B. The read-only guarantee stays in code: `ha_guard` and no write credentials.

- **Benchmark convention:** the canonical `rolling_horizon` rung keeps `exact` prices for tomorrow
  at 14:00, so it stays bit-identical and comparable with earlier results as the upper bound. The
  realistic modes, `published_only` and the PO-C estimate, are reported beside it. Once PO-C has
  measured how precise the estimate is, the owner decides which realistic mode becomes the headline
  and reworks what is needed.

- **GA (after PO-A):** failed. PO-C and everything after it are deferred; P4 uses published prices
  where they exist and `persistence` for tomorrow at 14:00.
