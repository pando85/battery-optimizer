# Thermal model (parked, C2 UNDETERMINED)

## Purpose and owner

Predicts room temperature from delivered heat and outdoor temperature (model A, first order; model B,
air plus slab) and the heat response to a heating permission, to plan heating against price. Owner: this
repository (`climate/thermal.py`, `plant.py`, `fit.py`, the `climate-fit` command). Its price-driven use
was dropped by the decoupling decision (ADR-0006); what remains is diagnostic and, possibly, warm-up-rate
learning for optimum start.

## Interfaces

- **In:** climate feature set, [outdoor-temperature](outdoor-temperature.md), [cop-model](cop-model.md)
  for the electricity criteria, the replayed legacy permission `u`, the private site file.
- **Out:** `<data_dir>/analysis/v1/climate-models/<model_id>/` (`model.json`, `validation.json`).
- **Consumers:** none live. The offline whole-system replay deliberately does **not** use it.

## Calibrated on

- Fold: fit **S2** (2024-10-01 to 2025-04-30), test **S3** (2025-10-01 to 2026-04-30), causal weekly refits
  shrunk toward the S2 fit. S1 is excluded as a different plant regime (35 degC water, valves shut about
  half the heating time, apparent UA about three times S2's and S3's).
- Runs: `c1794f600291` (config v2, hash `22888468...`, code `9055b4d`, dataset `cdd448abb006`, feature
  set `da6d678016ed`, verdict UNDETERMINED) and the earlier `c521d2733188` (config v1, hash `e01282e5...`, trained on S1 and S2, also UNDETERMINED).
- Plant regime: water 28 then 26 degC, setpoints 24 to 25 (S2) and 23.0 (S3), legacy 10 to 15 h.

## Assumptions

- Mean comfort-room temperature represents comfort; always-open share `phi` is 0.3 until floor areas
  exist; valve closure is binary.
- Return-water temperature is a proxy for slab temperature (a hypothesis; B's slab capacity sits on its
  lower bound, so the slab is **not identifiable**).
- Evidence: [C2 outcome](../planning/climate-optimizer-implementation.md#c2-model-iteration-the-one-predeclared-iteration-config-version-2)
  and [C2 recovery](../planning/climate-c2-recovery.md).

## Invalidated when

Already unaccepted, so these are triggers against any future fit:

- Water target, heating curve, thermostat setpoints or valve behaviour change (each moved the apparent UA
  between S1, S2 and S3).
- The permission lever disappears: under the comfort schedule there is no binary `u`, so the response
  model of `u` has nothing to learn from.
- The outdoor series or the heat measurement changes.

## Recalibrate by

Not by re-running `climate-fit`: the one predeclared iteration is spent and S3 is no longer an untouched
holdout. Follow the diagnostic-first plan R0 to R5 ([C2 recovery](../planning/climate-c2-recovery.md)):
predeclare the new candidates, thresholds and partition, then test on data from 2026-10-01 onward
(`seasons.reserved_from`). Acceptance: the new gates G-R1 to G-R3 of that plan, written before the run.

## Status

**parked** (2026-10-30). Nothing consumes it; the battery side does not depend on it. Un-park only with a
new predeclared evaluation.
