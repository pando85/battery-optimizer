# Comfort MPC (price-blind, track B; offline, shadow pending B2)

## Purpose and owner

Plans space heating for the comfort schedule with the fewest predicted heat-pump kWh, **ignoring
price**, as the second reference for the rule-based AppDaemon controller
([design](../planning/comfort-controller-design.md#13-track-b-shadow-thermal-model-mpc) section 13,
[ADR-0006](../adr/0006-decouple-heat-pump-comfort-from-battery-price-optimisation.md) amendment). It only
plans: nothing writes to Home Assistant or the plant. Owner: this repository (`climate/mpc*.py`,
`climate/b2d.py`, `hack/climate_mpc_replay.py`, [`config/climate-mpc.yaml`](../../config/climate-mpc.yaml)).

## Interfaces

- **MILP** (`climate/mpc.py`, HiGHS): one binary per slot and water level (software curve plus 0 to
  +3 K, integer, clamped 20 to 35 degC: 4 binaries per slot instead of 16 temperatures); minimise COP
  electricity of the level plus a start penalty; linear state equation; comfort minimums soft
  (20 kWh-eq per Kh); minimum on 30 min, off 20 min; DHW slots off and counted as compressor runs;
  stored heat valued at the horizon end. Heat per level: the model's heat table along a reference,
  re-linearised twice along the plan's simulation.
- **Model interface** (`climate/mpc_model.py`, `MpcModel`): `dynamics()`, `drive()`, `room_outputs()`,
  `heat_table()`, `simulate()`, `capacities`. A new model structure implements these; the MILP does not
  change. Implementations: `b2d` (default) and `c2` (comparison).
- **In:** climate feature set, [outdoor-temperature](outdoor-temperature.md), [cop-model](cop-model.md),
  comfort rooms and offsets as in [thermal-model](thermal-model.md). **Out:**
  `<data_dir>/analysis/v1/climate-mpc/<run-id>/`. **Consumers:** none live; B2 shadow, B3 comparison.

## Calibrated on

- `b2d`: B0's room-air two-state model with a weekday/weekend diurnal gain, prior fitted on the *other*
  season, `ua` and `q0` re-estimated weekly on the trailing 28 days; heat law (C2's response B) and COP
  lift table fitted on the prior season. `c2`: C2's model B, static S2 prior (in-sample on S2).
- Replays `b1-20261031-b2d-perfect`, `-b2d-persistence`, `-c2-perfect`: S2 and S3, November to March,
  hourly; code `26dcbb8`, dataset `cdd448abb006`, feature set `da6d678016ed`, MPC config `06ec157c`.
  Legacy regime (10 to 15 h, water 26 to 28 degC). [Results](../planning/comfort-controller-design.md#b1-result-offline-replay).

## Assumptions

- **Planned kWh and comfort are model predictions of plans never executed**; each day starts from the
  legacy-run house; no thermal model is accepted. Valves follow thermostats at the schedule setpoints
  (minimum + 0.5 K); a room is the comfort-room mean plus its hourly offset (office A not modelled).
- `perfect` weather is the measured outdoor series (declared foresight); `persistence` holds the last
  valid reading. The horizon's DHW mask is known. Beyond about 12 h predictions are low-confidence.
  Lifts outside the COP table read the nearest bin; the start penalty stands in for cycling losses.

## Invalidated when

- The schedule, setpoint margin, water curve or machine limits change (edit the MPC config).
- The comfort controller goes live (legacy-regime heat law, offsets and COP no longer hold), a
  successor to B0's model is accepted, or the outdoor source or heat measurement changes.

## Recalibrate by

`uv run python hack/climate_mpc_replay.py --feature-set <id> --dataset <dir> --workers 24` (plus
`--weather persistence`, `--model c2`) on new-regime data after the 28-day settling period. Acceptance:
section A of the [2026-27 preregistration](../planning/climate-preregistration-2026-27.md) on live data
(B3); this offline replay is exploratory and grants nothing.
TODO (future extension, not code): schedule the legionella cycle (owner wish, design section 7.2).

## Status

**provisional** (2026-10-31): implemented, offline; shadow pending B2. Exploratory model, no gate.
