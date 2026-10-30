# Outdoor temperature (debias, reference station, fused fill)

## Purpose and owner

The outdoor temperature the climate models read. The heat pump's own sensor reads 1.5 to 2.5 K low
while the compressor runs and warm in the sun, so the series is rebuilt from the sensor's trusted slots
(`t_out_debiased`), optionally fused with a nearby AEMET SYNOP reference through a Kalman-smoothed offset
(`t_out_fused`). Owner: this repository (`climate/data.py`, `climate/tout.py`, `climate/station.py`).

## Interfaces

- **In:** the Aquarea outdoor sensor, compressor frequency, a clear-sky proxy from the private site file,
  and the monthly reference-station files in the data directory (manifest-hashed).
- **Out:** `t_out`, `t_out_debiased`, `t_out_fused` and the filled-slot mask in `ClimateData`;
  `t_out_model` is the one the models read.
- **Consumers:** [cop-model](cop-model.md), [thermal-model](thermal-model.md). Nothing battery-side.

## Calibrated on

- Offset model fitted on **S1 and S2 only** (`signals.t_out_fused.fit_seasons`, a held-out season is
  refused); evaluated on S3 (October to April). Config `config/climate.yaml` v2, hash `22888468...`.
- Feature set `da6d678016ed`, station manifest hash `0183591c...` (parser version in the manifest).
- Plant regime: legacy heat-pump control, compressor on for the 10 to 15 permitted hours.

## Assumptions

- A slot is a trusted reading once the compressor has been off 30 minutes; a longer running period is
  interpolated only up to 6 h, else unknown (`signals.t_out_debias`).
- The sensor-minus-station offset is observed only in the dark (clear-sky proxy below 0.05) and is a
  constant plus a slowly varying AR(1) term.
- Station gaps up to 3 h can be bridged; the station represents the site.
- Result on S3: on hidden trusted blocks the fused series has a **higher** RMSE than the interpolated
  debiased series, so `t_out_source: debiased` stays and fused only fills unknown slots
  (`t_out_fill_unknown_from_fused: true`). Evidence: the station `evaluate` report in the data directory,
  [architecture](../architecture.md#fused-outdoor-temperature-stationpy-toutpy).

## Invalidated when

- The heat pump's outdoor sensor is moved, shielded or replaced (the bias and the offset change).
- The compressor duty rises. Longer, gentler runs (efficiency-first or comfort-schedule control) leave
  fewer trusted off-slots: in S2/S3 about 60% of slots were trusted and about a third interpolated, so
  the unknown share grows and the fill matters more.
- The reference station changes, relocates or stops reporting; the parser or the raw months change
  (the manifest hash changes, so `input_sha256` and every dependent `model_id` change).
- The fit seasons change, or a new season is added to the offset fit.

## Recalibrate by

1. `uv run python hack/climate_station.py fetch --stations <WMO>` for the new months (5 s between
   requests, each month kept immutable).
2. `uv run python hack/climate_station.py evaluate --feature-set <id> --dataset <dir>`
   ([architecture](../architecture.md#fused-outdoor-temperature-stationpy-toutpy)).
3. Acceptance: the predeclared switch rule. Fused replaces debiased only if its RMSE is lower on held-out
   trusted readings of a season the offset fit has not seen; otherwise keep debiased and report the
   unknown share and the filled share of the new regime.

## Status

**valid** (2026-10-30) for the legacy compressor duty. Re-run `evaluate` once the comfort schedule has
produced 28 days of data and record the trusted, interpolated and unknown shares.
