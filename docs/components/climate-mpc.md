# Comfort MPC (price-blind, track B; offline, shadow pending B2)

## Purpose and owner

Plans space heating for the comfort schedule with the fewest predicted heat-pump kWh, **ignoring
price**, as the second reference for the rule-based AppDaemon controller
([design](../planning/comfort-controller-design.md#13-track-b-shadow-thermal-model-mpc) section 13,
[ADR-0006](../adr/0006-decouple-heat-pump-comfort-from-battery-price-optimisation.md) amendment). It only
plans: nothing writes to Home Assistant or the plant. Owner: this repository (`climate/mpc*.py`,
`climate/b2d.py`, `hack/climate_mpc_replay.py`, [`config/climate-mpc.yaml`](../../config/climate-mpc.yaml)).

## Interfaces

- **MILP** (`climate/mpc.py`, HiGHS): one binary per slot and water level (curve plus 0 to +6 K,
  integer, 20 to 35 degC); minimises the COP electricity of the level plus a start penalty. Comfort
  minimums are soft (20 kWh-eq per Kh) at the slot end; no heat where every modelled valve is shut.
  Minimum on 30 min; minimum off 20 min is 2 slots, so **effectively 30 min**. Starts and limits count
  space heating only (DHW transparent); stored heat is valued at the end. Heat per level comes from the
  heat table along a reference (free-running house, then three re-linearisations; best iterate kept).
- **Model interface** (`climate/mpc_model.py`, `MpcModel`): `dynamics()`, `drive()`, `room_outputs()`,
  `heat_table()`, `valves()`, `simulate()`, `capacities`. Implementations: `b2d` (default) and `c2`.
- **In:** climate feature set, [outdoor-temperature](outdoor-temperature.md), [cop-model](cop-model.md),
  comfort rooms and offsets as in [thermal-model](thermal-model.md); setpoints from design sections 3
  and 6 (`optimum_start`). **Out:** `<data_dir>/analysis/v1/climate-mpc/<run-id>/`. **Consumers:** B2, B3.

## Calibrated on

- `b2d`: B0's 2-state model with weekday/weekend diurnal gain; prior, heat law and COP table from the
  *other* season (S2 scored with a later-season prior), weekly `ua`/`q0` refits. `c2`: S2 in-sample.
- Replays `b1r2-20261031-*` (code `84b05f0`, MPC config `f1c25db4`, dataset `cdd448abb006`, feature set
  `da6d678016ed`), legacy regime, S2 and S3 November to March; [results](../planning/comfort-controller-design.md#b1-result-offline-replay). The first replay `b1-20261031-*` is **withdrawn** (artefact of the setpoint assumption plus the placeholder `phi`).

## Assumptions

- **Planned kWh and comfort are model predictions of plans never executed**; each day starts from the
  legacy-run house; no model is accepted. Valves follow the setpoints the controller writes; a room is
  the comfort-room mean plus its hourly offset (office A not modelled); `phi` 0.3 is a placeholder.
- `perfect` weather is declared foresight and the observer reads it too (fused station series in long
  gaps); `persistence` holds the last valid reading; the DHW mask is known. Beyond about 12 h
  predictions are low-confidence. Lifts outside the COP table read the nearest bin, which understates
  the cost of high water (13 to 22% of planned slots at +6 K).
- Residual: a slot forced on by a minimum on time in progress may be planned against shut valves, where
  the MILP credits `phi` heat that `simulate` does not deliver (run logic).

## Invalidated when

- The schedule, lead table, setpoint margin, water curve or machine limits change (edit the MPC config).
- The comfort controller goes live (legacy heat law, offsets and COP no longer hold), a successor to
  B0's model is accepted, or the outdoor source or heat measurement changes.

## Recalibrate by

`uv run python hack/climate_mpc_replay.py --feature-set <id> --dataset <dir> --workers 30` (options
`--weather persistence`, `--model c2`, `--offsets 0,1,2,3`) on new-regime data after the 28-day
settling period. Acceptance is section A of the [2026-27 preregistration](../planning/climate-preregistration-2026-27.md)
on live data (B3); this offline replay grants nothing. TODO (future extension, not code): schedule the legionella cycle (owner wish, design section 7.2).

## Status

**provisional** (2026-10-31): implemented, offline; shadow pending B2. Exploratory model, no gate.
