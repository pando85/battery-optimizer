# Battery optimizer and live planner

## Purpose and owner

Chooses charge and discharge per 15-minute slot to minimise the cost of importing the household load.
One MILP kernel (`oracle.solve_oracle`, HiGHS) serves the benchmark policies (`policies.py`,
`rolling_horizon.py`) and the live shadow planner (`service/planner.py` `run_planner`, free terminal mode,
re-solved every 15 minutes). Owner: this repository. Roadmap:
[Dispatch Optimizer](../planning/components/dispatch-optimizer.md). Go-live is through OmniBattery, driven
by this optimizer, under a separate ADR; the shadow phase has no writer.

## Interfaces

- **In:** [battery-load-forecast](battery-load-forecast.md), published prices ([tariff-prices](tariff-prices.md)),
  [battery-model](battery-model.md), current SOC (simulated until commissioning).
- **Out:** `DispatchPlan` and `DispatchIntent` (`dispatch.py`, types only), shadow records and metrics.
  Not a battery command.
- **Consumers:** [system-replay-benchmark](system-replay-benchmark.md); later the battery actuator.

## Calibrated on

- Nothing is fitted; the evidence is the benchmark on the legacy regime: dataset `2e01620ba727`
  (2024-01-01 to 2025-12-31, RTE 0.80), run `fe3cbfec46b6`, oracle certified. Share of the causal-to-oracle
  gap closed: `day_ahead` 65.1%, `rolling_horizon` 69.8% (causal load basis).
- Replay on S2 and S3: `rolling_horizon` removes about 13 to 16% of no-battery cost under either heat-pump
  regime; the battery sits at its SOC floor while importing in 0 to 3.3% of expensive cold-day slots
  ([system replay](../planning/system-replay.md)).
- Plant regime: no battery, legacy heat pump, pre-tax PVPC.

## Assumptions

- Load is exogenous. The decoupled design keeps the objective unchanged; only price-driven heat-pump
  planning and a joint MPC are dropped ([ADR-0006](../adr/0006-decouple-heat-pump-comfort-from-battery-price-optimisation.md)).
- The horizon ends at the last published price slot; there is no unpublished-price forecast live.
- **Hard constraint (owner-confirmed 2026-10-30):** the 5.0 kW import limit is enforced by the smart
  meter, which cuts supply on overload; this has happened. A trip is a safety and availability failure,
  never an acceptable cost trade-off.
- Charge is limited only through forecast load (`charge <= limit * h - forecast load`) on a 15-minute
  average with no margin; load that is not in the forecast (boost, hot-water booster) is invisible to that
  constraint. At go-live the real-time grid-target cap below the limit is therefore **required**, not optional.
- **Heater inputs to the import limit** (Panasonic service manual; details in
  [heat-pump-controller](heat-pump-controller.md#heater-facts-import-limit-inputs)): the heat pump alone draws
  at most 3.47 kW, and its integrated 3.00 kW electric heater brings the worst case to about 6.5 kW, above the
  5.0 kW limit without any other load. The heater starts for hot water above about 51 degC, for the weekly
  sterilization (65 degC, 10 min, default) and below the outdoor heater-on temperature (default 0 degC). The
  sterilization is a **scheduled 3 kW load**, and none of it is in the load forecast, so the grid-target cap
  must hold the battery charge below the limit minus the heater, not only the forecast load.
- Throughput cost 0.0, so marginal cycles that merely clear the 0.80 round trip are taken
  (about 1.2 equivalent full cycles per winter day in the replay).
- Omnibattery's own price features will be off, so this is the only battery price reactor.

## Invalidated when

- Any [battery-model](battery-model.md) parameter changes, or the wear cost is set.
- [tariff-prices](tariff-prices.md) changes (taxes confirmed, holidays added, new publication timing).
- The load forecast changes regime (comfort schedule, boost): re-check floor-hit share and clipping.
- At go-live: measured SOC replaces simulated SOC; the charge adapter caps the grid target below the
  import limit; a second price-reactive writer appears.
- The heater facts change (hot-water target above about 51 degC, sterilization schedule, a power-limit input
  fitted), or the contracted power or the meter's trip behaviour changes, or a meter cut-off happens with the battery
  charging (treat as a failed acceptance and review the margin).

## Recalibrate by

`uv run battery-optimizer benchmark --dataset <store>/analysis/v1/<dataset_id> --train-end <ISO>
--policies all --terminal-mode free --time-limit-s 600` ([benchmark](../benchmark.md)), on a window in the
new regime after the 28-day settling period, with that window's measured load as `no_battery`.
Acceptance: oracle certified; `rolling_horizon` gap closure and relative saving against `no_battery` on the
same load are reported next to the legacy figures above (same RTE basis) and any material drop is
explained before go-live; the floor-hit share on expensive cold-day slots stays near the replay's 0 to 3.3%.

## Status

**valid, shadow only** (2026-10-30). Not go-live ready: wear cost, idle draw, measured SOC, the import-cap
adapter and the actuator ADR are open.
