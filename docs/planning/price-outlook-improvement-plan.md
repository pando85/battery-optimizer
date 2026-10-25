# Price Outlook — improvement plan

- **Status (2026-10-25):** agreed with the owner; PO-B in progress, everything else proposed. Refines the
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

### PO-B — Publication evidence (start now, passive)

- The service appends a private snapshot of every `sensor.esios_pvpc` state change: raw
  attributes, `last_updated`, `observed_at`, and whether `price_next_day_*` is present. It goes
  into `data_dir/service/v1/` through the existing sink.
- From the snapshots, measure the actual publication time distribution and verify *published =
  realized* against the Prometheus realized series.
- This ships in the same release as the HA weather-forecast archive agreed in Load Forecaster P4.
  Both are read-only, and the weather archive needs the `ha_guard` allowlist to permit
  `weather.get_forecasts`.

### PO-C — OMIE-based next-day estimator (offline, ≈3–4 days)

- **Data:**
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

Still open:
1. GA minimum worthwhile gain (private).
