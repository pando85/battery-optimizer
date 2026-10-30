# Whole-system replay: heat-pump regime x house battery

- **Status:** Result recorded (offline, read-only, no actuation). An empirical estimate, not a thermal simulation.
- **Date:** 2026-10-30
- **Runner:** `hack/system_sim.py` (added in #46, repository commit `f5be13d`)
- **Feeds:** the decoupled-architecture ADR (to be written next) and the [C7-F field trial](climate-flow-calibration-optimization.md)
- **Privacy:** this page carries relative figures only. Euro and energy totals stay in the private data
  directory (see [Where the private figures live](#where-the-private-figures-live)).

## 1. Question

Does a decoupled design pay off? The heat pump would optimise comfort and its own electricity (longer,
gentler runs, price ignored) and the house battery (5.12 kWh, 2.5 kW) would do the price arbitrage. The
legacy AppDaemon control instead picks the cheapest N hours for the heat pump, so price and comfort are
coupled. The replay asks what each choice does to cost and comfort over a whole heating season, with and
without the battery.

## 2. Method

A 2 x 2 grid per season:

| | no battery | simulated battery |
|---|---|---|
| **A** legacy, cheapest-N hours (load as observed) | A0 | A1 |
| **B** efficiency-first, like the 15 h days | B0 | B1 |

Seasons are S2 (2024-10 to 2025-04) and S3 (2025-10 to 2026-04) of `config/climate.yaml`. A is the load as
observed: S2 ran the legacy rule with 11 permitted hours (not 10 h), S3 with 10 h in Nov-Dec and 15 h from
January. Cost is pre-tax PVPC over the usable intervals of the season window; every cost below is relative
to **A0 of the same season**.

**No thermal model.** C2 ended UNDETERMINED, so B is an *empirical transfer* from the S3 days the legacy
rule ran with 15 permitted hours:

- Daily heating electricity from the 15 h days' kWh-versus-HDD line (base 15 degC, R2 about 0.75).
- Hourly shape from their kWh-weighted mean profile, split at 9 degC daily mean outdoor (cold and mild).
- Comfort (time below 22 degC, degree-hours), COP and compressor starts from the 15 h days in the same
  outdoor band (bands at 3, 6, 9, 12 and 15 degC, nearest band with at least 3 days).
- `Bs` (sensitivity): the day's observed heating times the S3 stratified 15 h / 10 h ratio (days already at
  15 h unchanged), same shape and comfort as B. It keeps each season's own plant state, so it is the
  realistic variant for S2.
- B replaces heating only on complete days with a replayed legacy schedule (146 days in S2, 144 in S3);
  other days, hot water and the rest of the house stay as observed.

**Meter basis.** The dataset load is the whole-house meter and the heat pump is measured by HeishaMon, so
B load = gross + k (E_B - E_A), with k the season's median ratio of the heat-pump branch meter to
HeishaMon: 1.27 (S2) and 1.28 (S3). The branch also carries some appliances, so k leans high.

**Battery.** The unchanged benchmark on a copy of the dataset with only `gross_kwh` replaced in memory:
the rungs `no_battery`, `day_ahead` and `rolling_horizon` (realistic, causal trailing-mean load forecast,
published prices) and `oracle` (perfect foresight). `config/tariff.yaml` battery block, terminal mode
free, pre-tax, round-trip efficiency 0.80 (provisional).

## 3. Sanity check: it reproduces the earlier 10 h to 15 h finding

S3 normal days, 10 h against 15 h (the observational comparison already recorded in the
[climate plan](climate-optimizer-implementation.md)):

| | below 22 degC | heat electricity | COP | starts/day |
|---|---|---|---|---|
| prior finding | 39% to 12% | +16% | 5.1 to 4.75 | 14 to 17 |
| replay, raw (55 vs 76 days) | 34.6% to 11.7% | +18% | 5.08 to 4.78 | 13.2 to 16.5 |
| replay, band-matched (131 days) | 36% to 11.4% | +21% | 5.02 to 4.82 | 13.0 to 16.5 |

The transfer is built from these same days, so this checks the plumbing and the band stratification, not
the causal claim.

## 4. Results

Comfort and efficiency are intensities over the replayed days. Cost columns are the change against the
same season's A0.

| season | scenario | below 22 degC | Kh/day below 22 | kWh/HDD | COP | starts/day | cost, no battery | day-ahead | rolling horizon | oracle |
|---|---|---|---|---|---|---|---|---|---|---|
| S2 | A (11 h legacy) | 5.3% | 0.63 | 0.98 | 4.58 | 11.2 | 0 | -12.6% | -13.7% | -16.1% |
| S2 | B | 7.3% | 0.49 | 0.75 | 4.93 | 16.7 | -1.9% | -15.3% | -16.3% | -18.6% |
| S2 | Bs | 7.3% | 0.49 | 1.19 | 4.90 | 16.7 | +8.3% | -5.2% | -6.2% | -8.5% |
| S3 | A (10 h then 15 h) | 20.5% | 2.03 | 0.69 | 4.90 | 15.4 | 0 | -14.4% | -16.0% | -19.0% |
| S3 | B | 11.4% | 1.15 | 0.75 | 4.83 | 16.6 | +2.1% | -12.7% | -14.3% | -17.2% |
| S3 | Bs | 11.4% | 1.15 | 0.75 | 4.81 | 16.6 | +2.4% | -12.4% | -14.0% | -16.9% |

Battery saving as a share of the same scenario's no-battery cost (rolling horizon): S2 A 13.7%, B 14.6%,
Bs 13.4%; S3 A 16.0%, B 16.1%, Bs 16.0%.

Key findings:

1. **Battery value is nearly independent of the heat-pump regime.** In S3 the battery removes 16.0% of
   cost with the legacy load and 16.1% with the efficiency-first load; the decoupled design keeps the
   arbitrage the battery would deliver anyway. S2 shows the same within about one point (13.4 to 14.6%).
2. **Efficiency-first adds a small cost that the battery more than covers.** In S3 B0 costs +2.1% (Bs0
   +2.4%) against A0, and the realistic battery turns that into -14.3% (B1) against -16.0% for A1: B1 is
   about 2% above A1 in cost, with the battery saving 115% of the net B1 saving (the battery pays for the
   regime change and more).
3. **S3, B1 against A0: about -14% cost with roughly half the time below 22 degC** (11.4% against 20.5%,
   a ratio of 0.56; degree-hours 1.15 against 2.03 Kh/day). The comfort gain is smaller than the
   10 h to 15 h comparison (36% to 11%) because S3's A is already a mix that ran 15 h from January.
4. **The efficiency price is small in S3:** COP barely moves (4.90 to 4.83) while kWh/HDD rises from 0.69
   to 0.75 and starts from 15.4 to 16.6 per day.
5. **The battery is rarely empty when it matters.** Share of expensive slots (top quarter of the day's
   prices) on cold days (mean outdoor below 8 degC) that end at the SOC floor while importing, rolling
   horizon: 0 to 3.3% in every scenario (S3: 2.7% A, 2.8% B, 3.3% Bs); day-ahead 0 to 3.0%. Only the
   oracle goes lower in cost and hits the floor more (4 to 16%).
6. **Cycling is about 1.2 equivalent full cycles per day** in winter (rolling horizon, 246 to 262 over the
   roughly 212-day Oct to Apr window; B adds about 4 to 6% more than A). Battery wear is **not modelled**
   in these costs.
7. **S2 is read through Bs.** S2 B (-1.9% no battery, -16.3% with) is flattered by borrowed S3 plant
   state; the like-for-like S2 estimate is Bs: the observed 11 h days moved to 15 h cost +8.3% before the
   battery (kWh/HDD 0.98 to 1.19) and -6.2% after it. S2 therefore shows a larger regime premium than S3
   because its A was not at 15 h at all.

## 5. Caveats

- **The comfort of B is an association, not an effect.** It comes from 15 h days (January onward) set
  against 10 h days (Nov-Dec), months apart, so weather, occupancy and plant settings are confounded. The
  [C7-F field trial](climate-flow-calibration-optimization.md) is what confirms or refutes it.
- **S2 B borrows S3 plant state.** Thermostat setpoints were about 24.5 degC and the flow target 28 degC
  in S2 against 23 degC and 26 degC in S3, so comparing S2 comfort or kWh/HDD across A and B is not
  like-for-like (S2 B reads 7.3% below 22 degC against 5.3% for A, the opposite of S3). Use Bs for S2.
- **The B hourly shape is inherited from the price-chosen 15 h days.** Those hours were picked by the
  legacy cheapest-hours rule, so B0 cost is somewhat flattered (the load already sits in cheaper hours
  than an indifferent efficiency-first schedule would) and the battery share of the B1 saving is
  probably understated.
- **Battery assumptions are unconfirmed:** round-trip efficiency 0.80, idle draw and taxes are provisional
  or absent (pre-tax). Absolute savings would change with them; findings 1 and 2 are relative and
  plausibly less sensitive to them, but that was not tested.
- **Review notes from #46:** at most 76 of roughly 20k slots per scenario had B load clipped to zero
  (about 0.4%, S2 B 76, S2 Bs 60, S3 B 3, S3 Bs 0), negligible. Oracle solves are not guaranteed to certify
  within the time limit (default 600 s, `--oracle-time-limit-s`); in the recorded run all six oracle
  solves reported a certified optimum and the causal rungs report `feasible`. The S2 "legacy" regime
  actually ran 11 h, not 10 h.
- No thermal model and no DHW changes: every result is for this installation and these two seasons.

## 6. Reproduce

Read-only: the runner writes only below the data directory. Use the pulled cluster store (`just pull-store`,
see [configuration](../configuration.md)); never point it at the frozen `~/data/energy-baseline` store.

**Always use a fresh run id.** The runner does not refuse an existing id: it writes into the same
directory and mixes the new outputs with the old ones. The reference run below is the clean
reproduction `v1-20261030-repro`, made from a scratch worktree of `f5be13d`.

Identities recorded in the report of the reference run (they must match on a faithful reproduction):

| field | value |
|---|---|
| repository commit | `f5be13da8fb7c3eeef958002de205e06f8a59374` (`f5be13d`, clean tree) |
| climate config | `config/climate.yaml`, `config_version: 2`, hash `228884683bc52be1c85ad2edb167900f6e58d7548678adb1277a21042ff15d18` |
| climate feature set | `da6d678016ed5b27f997640e5cfa804856daf987a2550ab04517f93c6bd4554b` |
| dataset | `cdd448abb006d401854b193f7a240c367c92908ecfa75a76d76a413828e80e9d` |
| tariff | `config/tariff.yaml` (battery block, 5.12 kWh, 2.5 kW, RTE 0.80) |
| run id | `v1-20261030-repro` |

```bash
git checkout f5be13d
export BATTERY_OPTIMIZER_DATA_DIR=<store>      # pulled cluster store, outside the repository
uvx uv@0.13.5 run python hack/system_sim.py \
    --feature-set da6d678016ed5b27f997640e5cfa804856daf987a2550ab04517f93c6bd4554b \
    --dataset <store>/analysis/v1/cdd448abb006d401854b193f7a240c367c92908ecfa75a76d76a413828e80e9d \
    --config config/climate.yaml --tariff config/tariff.yaml \
    --run-id <new-run-id>        # never reuse an existing id (it is not refused and mixes outputs)
```

Other flags: `--seasons S2,S3` (default), `--oracle-time-limit-s 600` (default), `--no-battery` (skip the
benchmark runs and keep the heat-pump and sanity-check sections), `--data-dir` (overrides the variable).
The pinned uv version is the one in `.github/actions/setup-uv/action.yml`.

Runtime: 451 s wall (7.5 minutes, including the environment build) for the reference run, so budget
about ten minutes. The oracle solves dominate.

**Reproduction check.** The first run of this analysis (id `v1-20261030`) printed code version
`109432e...-dirty`, i.e. the runner as an uncommitted working tree on top of `109432e`. The reference
run repeats it at the committed `f5be13d` and reproduces it exactly:

- every numeric field of `report.json` (heat-pump, sanity-check and transfer statistics, meter factors,
  costs, savings, oracle figures and solver statuses) is identical, so the tables on this page are
  unchanged;
- the per-scenario `results.parquet` of all six benchmark runs are identical, and the `schedules.parquet`
  differ only in the `run_id` column;
- the only differences are bookkeeping: the recorded code version (and with it the benchmark run
  directory names), and `no_battery` `floor_hit_share`, which the original run reported as a share and
  the committed runner reports as null (not applicable: that rung sits at the SOC floor by
  construction). No table here uses it.

The committed runner is therefore the one behind these results as far as costs and heat-pump figures
go, which is evidence from identical outputs rather than from a shared commit.

Outputs, all in `<store>/analysis/v1/system-sim/<run-id>/`:

- `report.json` (every input, per-band statistics, per-scenario heat-pump and battery figures, savings),
- `report.md` (the tables above with absolute figures),
- one benchmark run per scenario and season: `A-S2`, `B-S2`, `Bs-S2`, `A-S3`, `B-S3`, `Bs-S3`
  (each `benchmarks/<run>/` with `benchmark.json`, `results.parquet`, `schedules.parquet` and the
  integrity manifest).

### Where the private figures live

`<store>/analysis/v1/system-sim/<run-id>/report.md` (for example `v1-20261030-repro`) holds the euro and energy totals. They stay in the
data directory and never enter the repository.

## 7. Decision it feeds

- The decoupled-architecture ADR (to be written next) cites findings 1 and 2: separating comfort from
  price loses little and leaves the battery arbitrage intact, subject to the caveats above.
- The [C7-F joint field trial](climate-flow-calibration-optimization.md) (permitted heating hours and
  water temperature against current operation) replaces the confounded association behind the B comfort
  numbers with a measured one. Until it reports, treat the comfort side of B as a hypothesis.
