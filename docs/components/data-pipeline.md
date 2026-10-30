# Data extraction and dataset build

## Purpose and owner

Turns the Prometheus archive and the Home Assistant history into an immutable, content-addressed store:
15-minute UTC intervals of whole-house load and PVPC price (`build`), and a climate feature set of heat
pump, thermostat and sensor signals (`build-features`). Owner: this repository. The extraction CronJob
and the data volume are deployed from the homelab repository.

## Interfaces

- **In:** Prometheus `hass_*` history (1-minute, from 2023-11-08), HA history for enum states
  (not re-extractable), `config/tariff.yaml` (hashed into every `dataset_id`).
- **Out:** `<data_dir>/analysis/v1/<dataset_id>/` (`gross_kwh`, `raw_grid_kwh`, price, quality flags,
  `battery_absent`) and `<data_dir>/analysis/v1/features/<feature_set_id>/`.
- **Consumers:** every offline result ([benchmark](system-replay-benchmark.md), forecasters, climate
  models) and, for the same definitions, the shadow service history.

## Calibrated on

- Frozen baseline dataset `2e01620ba727` (2024-01-01 to 2025-12-31, `config_hash` `7a901801...`, RTE 0.80);
  benchmark run `fe3cbfec46b6`.
- Current cluster-store dataset `cdd448abb006` (2024-01-01 to 2026-10-01, built at code `0e5887f`,
  same `config_hash`); climate feature set `da6d678016ed` (2023-11-07 to 2026-05-01, code `ac90586`).
- Plant regime: no battery, legacy heat-pump control, S1 to S3 (see [regime log](regime-log.md)).

## Assumptions

- `gross_kwh` is the whole-house feed, Shelly main channel 1, user confirmed 2026-10-23; the heat pump
  and the servers are inside it, not added to it ([ADR-0003](../adr/0003-phase1-measurement-and-baseline.md)).
- No battery: `battery_absent=True` is a recorded assumption, because the telemetry registry is empty
  (`build.py` `BATTERY_TELEMETRY_ENTITIES = ()`, `resolve_gross_load`).
- HA recorder history cannot be re-extracted; string states are not exported ([known issues](../known-issues.md)).
- Heat-pump electricity: the branch meter (channel 2) reads about 1.28 times the HeishaMon counters and
  also feeds appliances ([climate plan](../planning/climate-optimizer-implementation.md)).

## Invalidated when

- **A battery is commissioned.** Channel 1 becomes net grid and `build` silently stores it as gross
  (battery charging counted as house load, discharge hidden). Every dataset built after commissioning
  is unusable until reconstruction exists.
- The meter, its CT orientation or the channel mapping changes; an HA entity is renamed.
- `config/tariff.yaml` changes in any field: every `dataset_id` is re-minted and the old one superseded.
- A new heat-pump signal enters the feature set (new feature-set id), or the exporter changes what it
  retains.

## Recalibrate by

1. Before go-live: register the Omnibattery counters in `BATTERY_TELEMETRY_ENTITIES`, implement
   `gross = raw + charge - discharge` in `build` (mirror `service/history.py`) with tests, and make
   `battery_absent=True` an error for windows after the commissioning date.
2. Then `extract`, `validate`, `build --no-assume-no-battery` for the new window
   ([CLI](../cli.md#build)); `build-features --signal-set climate` for the feature set.
3. Acceptance: `build_manifest.json` energy reconciliation closes, `battery_absent` is false, and gross
   minus raw equals the metered charge minus discharge on a sampled week.

## Status

**valid** until battery go-live (2026-10-30); **blocking** for any post-commissioning window until step 1.
