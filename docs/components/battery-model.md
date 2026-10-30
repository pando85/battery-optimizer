# Battery model (physics, limits, losses, wear)

## Purpose and owner

The feasible charge, discharge and state-of-charge transitions of the house battery, shared by the
benchmark, the live planner and the shadow replay. Owner: this repository (`battery.py`; assumptions
resolved by `benchmark.resolve_battery_assumptions` and `service/planner.battery_spec_from_tariff` from
the `battery:` block of `config/tariff.yaml`). Roadmap: [Battery Model](../planning/components/battery-model.md).

## Interfaces

- **In:** `config/tariff.yaml` battery block (and, after commissioning, measured SOC and counters).
- **Out:** `BatterySpec` and `Horizon`; `simulate`, `check_feasible`, `compute_efc`.
- **Consumers:** [battery-optimizer](battery-optimizer.md), [boost-mode](boost-mode.md) (interlock reads
  SOC), [system-replay-benchmark](system-replay-benchmark.md), every `dataset_id` (the tariff file is hashed).

## Calibrated on

Nothing measured: the battery (Marstek Venus E class AC unit) is bought and **not commissioned**, so there
is no telemetry. Confirmed from specification or contract: capacity 5.12 kWh nameplate, 2.5 kW AC power,
5.0 kW contracted import limit. Provisional values are in the next section. Datasets built at
`config_hash` `7a901801...` (RTE 0.80); everything computed at the earlier 0.88 is superseded.

## Assumptions

| Parameter | Value | Basis |
|---|---|---|
| Round-trip efficiency | 0.80 (0.89443 per leg) | Owner choice 2026-10-24, bottom of the 0.80 to 0.84 range reported by others; a scalar, not a curve |
| SOC window | 0.20 to 0.95 (3.84 kWh usable) | Owner's life-extension choice; real controller limits not read |
| Initial SOC | 0.20 | Floor, so no free stored energy |
| Idle draw | 0, not modelled | `battery.py`; unmeasured |
| Wear / throughput cost | 0.0 EUR/kWh | Roadmap keeps 0 until substantiated; **not sensible for a live planner** |
| Import limit | 5.0 kW, 15-minute average, no margin | Contract; whether enforcement is the 15-minute average or an instantaneous trip is unverified |
| Export | none, no credit | Invariant of the kernel; no PV |

Evidence: [`config/tariff.yaml`](../../config/tariff.yaml), [benchmark](../benchmark.md), [ADR-0001](../adr/0001-home-battery-economic-dispatch.md).

## Invalidated when

- The battery is commissioned and a measurement disagrees with any row above (RTE vs power, idle draw,
  usable window, latency, firmware limits).
- The Omnibattery or firmware configuration changes limits; the warranty terms change the wear value.
- Any edit to the battery block of `config/tariff.yaml`: every `dataset_id` is re-minted and the old
  datasets and benchmark results are superseded (record it).
- The import-limit basis turns out to be an instantaneous trip: the averaged constraint is then not a
  sufficient guard.

## Recalibrate by

1. Commissioning measurement ([roadmap BM-1, BM-2](../planning/components/battery-model.md)): synchronized
   SOC, signed AC power and energy counters; several power and SOC bands; standby draw; full and partial
   cycles. Do not infer RTE from an unbalanced partial cycle.
2. Update the battery block in `config/tariff.yaml`, rebuild datasets, supersede the old ones, re-run
   `uv run battery-optimizer benchmark ...` ([benchmark](../benchmark.md)).
3. Wear: run the throughput-cost sweep of `economic_sensitivity` on new-regime data, pick the knee and set
   `planning.throughput_cost_eur_per_kwh` (a service value; no dataset re-mint).
4. Acceptance: predicted against measured Wh and SOC on held-out physical intervals, by power band;
   a model refinement is rejected if the measured gain is negligible or it breaks MILP feasibility.

## Status

**provisional** (2026-10-30): RTE, SOC window, idle draw and wear cost are assumptions. Idle draw and wear
must be set before go-live.
