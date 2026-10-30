# C2 recovery: diagnostic-first thermal identification after UNDETERMINED

- **Status:** Proposed recovery plan (not an amendment to C2 or approval for actuation).
- **Recorded outcome:** 2026-10-30, C2 config v2, first and only predeclared iteration: **UNDETERMINED / no-go**.
- **Progress:** 2026-10-31, steps R0 to R2 done as track-B step B0 ([section 7](#7-status-2026-10-31-b0-results-r0-to-r2)); G-R1 to G-R3 not reached.
- **Parent:** [climate optimizer implementation](climate-optimizer-implementation.md), especially the immutable [C2 outcome](climate-optimizer-implementation.md#c2-model-iteration-the-one-predeclared-iteration-config-version-2).
- **Related:** [C7-F water-flow identification](climate-flow-calibration-optimization.md), [ADR-0005](../adr/0005-shared-energy-planning-climate.md).
- **Scope:** Diagnosis and replacement validation strategy for the *building/plant/electricity* models. Existing AppDaemon and pump settings remain untouched.

## 1. Decision: do not promote A or B, and do not relax the failed gate

The original C2 heat-v1 gate and its version 2 iteration **stay unchanged**. In particular, do not widen a fitted parameter envelope just because the optimizer stopped at a bound, report an unaccepted model as valid, recycle S3 as a pristine holdout or start C3/C5 on the strength of a visually plausible fit. A bound hit is **a symptom**: data/parameter non-identifiability, omitted dynamics, sensor-model mismatch, excessive aggregation and real model misspecification are distinct possibilities. The numerical bounds are not evidence that the true slab capacity lies just outside them.

Grounded findings from the completed v2 run:

| Finding | Consequence |
|---|---|
| Models A and B both miss the 6 h Q-driven RMSE, u-driven daily-energy MAPE/season bias, the 10 h `min_hours` regime bias and COP season error | Neither is a safe predictive model of *actions* and electricity |
| A also misses 12 h RMSE, 24 h bias, 15 h regime bias and the full legacy replay | Do not use A as a fallback actuator predictor merely because it has fewer parameters |
| B passes the historical legacy electricity, cost and comfort replay but hits the S2 slab-capacity lower bound and many weekly parameter bounds | Replay agreement is insufficient: B is **not identifiable**, not an accepted controller |
| The earlier S1 season is a different water-target/valve regime; S2 to S3 shifts remain a generalization challenge | Do not pool regimes unconditionally or interpret changing apparent UA as actual wall insulation |
| T_out bias/gaps, control-valve closures, weak slab-return proxy and HeishaMon-vs-Shelly electrical mismatch are documented | Fix the observation/actuation model *before* adding degrees of freedom |

**Hypotheses, not established causes:** return water is a hydraulically coupled proxy and may not observe the effective slab temperature; mean comfort-room temperature may hide distribution of heating into always-open zones; a fixed `phi` and binary valve cap may hide modulation/valve hysteresis; solar and internal gains can correlate with schedule and outdoor temperature; starts, defrost, DHW and water-target changes can bias a coarse COP model. Test these against real observations rather than assuming a larger RC model will fix them.

## 2. Two independent tracks, kept distinct

**D — diagnostic / scientific model recovery (offline):** find what is identifiable, compare alternative minimal models, quantify uncertainty and preserve reproducibility. A useful thermal description may be possible without passing a decision-grade heat-permission model.

**E — observational/model-free policy analysis (offline):** evaluate the existing `min_hours` settings, comfort and actual delivered electricity/cost using matched historical conditions. The 10 h versus 15 h comparisons are **observational**: weather, setpoint, season, day type, valve state and policy timing confound the causal effect. Use matching/stratification, overlap checks, block confidence intervals and sensitivity analysis; if overlap is poor, label the result *association only*. Never claim a counterfactual electricity saving from naive across-month averages.

Track E may produce a **proposed** comfort-aware legacy heuristic, but without adequate evidence it remains advisory. No Home Assistant or AppDaemon behavior changes in this plan. Any later live test needs its own preregistered protocol, approved safeguards and homelab rollout.

## 3. Diagnostic experiments before new optimization

| Step | Question / diagnostic | Deliverable and stop condition |
|---|---|---|
| **R0 — Evidence ledger** | Can all failed C2 criteria be reproduced for the exact pinned dataset/model/config/commit? Which errors are at 1, 6, 12, 24 h, by weather, water target, valve occupancy and `min_hours` regime? | Versioned diagnostic report referencing **private** full results, with data coverage, sample counts, residual charts and per-gate status. No copied private measurement figures to git |
| **R1 — Sensor/energy audit** | Do reported heat kWh, water flow × delta-T, compressor runtime/starts and COP reconcile in steady non-DHW heating? How do temperature quantization, solar/external `T_out` fusion, room offsets, valve state and Shelly appliances affect uncertainty? | Meter/temperature uncertainty budget; explicit missing-data mask; independently measured vs inferred signals, with no future-leaking feature |
| **R2 — Identifiability tests** | Is the second state supported? Compare profile likelihood and Jacobian/SVD condition numbers, parameter correlation, parameter estimates under constrained initial slab temperature, leave-session-out stability, sensitivity to `b_ret` and its flow dependence. | A/B are *diagnostic references*, not accepted control models. If slab capacity or coupling stays non-identifiable, **drop the latent slab state from the actuator model**, retain uncertainty in the building response |
| **R3 — Fair challenger models** | What is the simplest candidate that generalizes? Compare physical first-order with exogenous solar/internal gains, a regularized delayed/impulse-response or ARX model driven by **measured heat**, optional physically constrained two-state model with externally sourced priors, and a separate empirical `permission→Q_heat` model | Rolling-origin error at 1/6/12/24 h on identical mask and matched information set. Include persistence/diurnal and original A/B as reference. Do not default to ML or more states |
| **R4 — Plant/energy error decomposition** | Does the error originate in `Q_heat→room`, `permission→Q_heat`, or `Q_heat→electricity`? | Distinct failure budgets and baselines for each link. Even a good Q-driven house fit **does not** authorize u-driven scheduling if the permission/valve model fails |
| **R5 — Data value and experiment design** | Which missing outdoor × achieved-flow × valve × compressor sessions would distinguish competing models? | Ranked coverage matrix, expected information gain and a proposed supervised experiment batch, coordinated with the C7-F F0/F1 audits and field trial (F2). Avoid experimental changes for cells with already stable estimates |

Use the existing `climate/data.py`, `thermal.py`, `plant.py`, `cop.py`, parquet/manifest datasets, historical GoHeishaMon telemetry and `t_out_fused`; add diagnostics as offline functions/CLI reports, not a new resident service. Do not store training outputs under the repository or retrofit a previous C2 `model_id`. Inputs and outputs must be versioned, content-addressed and replayable.

### Model contestants and important constraints

- **Minimal grey-box:** fit stable `UA` and effective `C` using periods with sufficiently different `T_in-T_out`, measured heat and identified solar/internal load; calibrate at multiple initial indoor temperatures. Fixed physical heat-loss priors are allowed only as externally sourced **declared** priors, not unexplained tuning to S3.
- **Empirical response (ARX/finite impulse response):** infer the lag from *measured thermal heat* to room temperature with regularization and bounded state; preserve physical response signs and no unbounded thermal storage. Test multi-hour extrapolation and out-of-distribution intervals, not just one-step fit.
- **Two-state slab:** preserve only as a competitor when independently observable; optional extra floor/return probes can help. Return-water temperature is a water-side measurement, **not** a direct slab thermometer.
- **Plant model:** replace the simplistic valve share only if session-level evidence demands it; distinguish always-open floor, thermostat valve hysteresis, actual circulation and Panasonic internal regulation. First verify which target/offset the real pump achieved in each episode.
- **Electricity:** model measured electrical kWh from measured thermal kWh and actual (not merely requested) water/outdoor conditions. Compare a lift table, constrained regression and manufacturer performance bounds. Reconcile energy attribution and defrost before claiming a cost advantage.

## 4. Evaluation, data leakage and independent gates

C2 S3 has now been inspected to diagnose and amend the model. It **cannot** serve as an untouched confirmatory test of a new post-hoc approach. Use S2/S3 for exploratory development with nested blocked validation that holds whole heating sessions/days out, *and clearly label these reports exploratory*. Before any new evaluation claim, publish the new model candidates, the diagnostic-to-controller promotion criteria, the frozen dataset partition and the policy comparator. Reserve data **from 2026-10 onwards** for a subsequent chronological/prospective test, without repeatedly tuning on its results. If prospective data is insufficient or doesn't cover cold conditions, result is **UNDETERMINED**.

Keep three separate gates:

1. **G-R1 Observation-model gate:** meaningful held-out improvement over simple persistence/diurnal models, calibrated error/uncertainty bounds, credible parameters or explicitly nonparametric model, no material drift by outside/flow bands. Predeclare numerical thresholds after R0 and **before** touching the reserved test period.
2. **G-R2 Action-response gate:** with historical commands/information cutoff, `permission→thermal heat→room temp` plus electricity passes separate held-out energy, comfort, cycling and 24 h error tolerances. The reported weather has to be an issued forecast at decision time; realized weather can only be used for a separately labeled diagnostic.
3. **G-R3 Policy gate:** using the accepted complete action model, compare legacy vs simple conservative temperature guard vs a policy challenger on matched inputs/terminal heat, comfort ≤ legacy violations and device safety. Report cost *and* electricity and uncertainty. No apparent gain below model/metering error qualifies.

Failing G-R1/R2 means **no MPC permission/flow actuation**; this does not prevent building dashboards, passive experiments, or model-free observational analysis. G-R3 passing would permit **read-only shadow C3** under a separate documented approval and API contract, followed by the existing C4 validation, C5 single-writer device safety gate and supervised commissioning. G-R1/R2 alone do not approve even shadow plans being treated as feasible.

C7-F remains an **independent** control-lever study but cannot skip the action-model and safety evidence needed for an actual optimizer. F0/F1 and the deliberately varied data of its field trial (F2) can contribute the data needed for recovery. The trial is model-free: it compares each arm with current operation and needs no validated permission baseline; any model-based flow claim stays non-actionable until the gates above pass. Neither C7-F nor C2-R retroactively converts C2's no-go to a pass.

## 5. Planned PRs, responsibilities and concrete exit criteria

| PR | Area | Result |
|---|---|---|
| **R0** | Reproducible C2 failure ledger with grouped residuals and pinned manifests | Reviewer can see *which stage* fails; does not publish raw/private house data |
| **R1** | Sensor, hydraulic, meter and slab-observability audit | Quantify bias/uncertainty; choose minimal trustworthy features |
| **R2** | Identifiability diagnostics and simple challenger fitting | Demonstrate parameter stability and held-out gains, or explicitly reject the model family; no threshold-tuning after viewing held-out results |
| **R3** | Experiment planner integrated with the C7-F F1 coverage and F2 trial design | Approved, low-risk session requirements with abort conditions; no default actuation |
| **R4** | Independent frozen preregistration, seasonal/prospective comparison and gate report | PASS / FAIL / UNDETERMINED for observation, action and policy; only then consider C3 shadow |
| **R5** | Optional C3 shadow proposal | New PR if R4 passes; no homelab writes, no battery changes |

**When to stop:** if a simple weather/temperature-aware legacy rule can satisfy comfort with equal or lower measured electricity/cost, that may be the preferable end state. If sensors/actuation are unobservable, choose manufacturer's native weather compensation and conservative legacy scheduling over a complex unidentified MPC.

## 6. Architectural invariants

- Keep read-only optimizer credentials, frozen original C2 config v2, manifest provenance, UTC/DST semantics and separate battery planner unchanged.
- Keep one climate/DHW hardware writer in AppDaemon, with manual override, DHW/defrost precedence, stale-input protection and safe rollback; no experiments through the optimizer service.
- A 15-minute planning grid is not a command frequency and must not drive high-frequency water-setpoint changes.
- Keep no-go and inconclusive reports visible in status pages; do not silently substitute the failed B fit as a production default.

## 7. Status 2026-10-31: B0 results (R0 to R2)

Track-B step B0 ([comfort controller design](comfort-controller-design.md#13-track-b-shadow-thermal-model-mpc))
ran R0 to R2 on 2026-10-31 as a read-only analysis. The full results are private (per-step JSON in the data
directory); this section records only what they mean. **Everything that fits a model on one season and
scores it on the other (S2 to S3, S3 to S2) is exploratory**: S3 was already inspected in C2, so it is not a
pristine holdout, and no number below is a confirmatory gate result.

### 7.1 Gate outcomes

| Step / gate | Outcome |
|---|---|
| R0 reproduce C2 v2 | **Done: reproduced exactly.** Every value of `validation.json` equals run `c1794f600291` (no differences); only the model id differs, because the code version moved on (the climate code and config are unchanged). |
| R1 sensor and energy audit | Done (deliverable, no gate). |
| R2 identifiability | Done (deliverable, no gate). The stop rule ("drop the latent slab") is triggered with the return-water channel in the fit and **not** triggered without it. |
| **G-R1** observation model | **Not reached.** The plan requires numerical thresholds predeclared after R0 and before the reserved period; none exist. There is also no heating data after the cutoff: space heating has been off since the reserved period began ([regime log](../components/regime-log.md)). The exploratory numbers below are not a pass. |
| **G-R2** action response | **Not reached.** No issued-forecast archive covers heating (it starts 2026-10-25). The lever it validates, the binary on/off permission, **no longer exists under ADR-0006**, so the gate is written around a removed action: a **plan gap** to be closed in the next gate revision (which action is validated: start time, setpoint or water target?). The early signal points to fail: the permission-to-heat link alone has a daily energy error of about 21% against the 15% limit. |
| **G-R3** policy | **Not reached.** No accepted action model; ADR-0006 also changed the metric (electricity primary) and the comparator is no longer the running policy. |

### 7.2 Root causes

1. **The slab bound is a measurement artefact, not a property of the slab.** The return-water sensor
   observes the water loop with a **flow-dependent offset** (about 0.1 K at low pump flow, 2.4 to 3.0 K at
   high flow in S3), so `y2 = T_s + constant` is misspecified. With the return channel in the fit the
   profile of the slab capacity is pulled to the lower bound (6 of 7 leave-one-month-out folds hit it, and
   leaving one month out flips the fit to a second optimum). Fitted on **room air only**, the slab is
   identified in the interior at **about 10.6 to 12.5 kWh/K** in both seasons, and the other parameters are
   stable. Model A's parameters are stable as well.
2. **A missing daily cycle, with a weekday/weekend shape, drives the 6 to 12 h errors.** The open-loop
   error has the same time-of-day pattern in both seasons (too cool at night, too warm in the evening) and
   weekends differ from weekdays. Likely drivers: occupancy and internal gains, plus a diurnal error of the
   outdoor sensor (sun at midday, a night offset).
3. **The season shift explains the 24 h and the energy bias.** The apparent UA fell by about 22% from S2 to
   S3 while the solar gain term collapsed, with setpoint, water target, valve-shut time and outdoor sensor
   all changing: a regime confound, not insulation. The simulated valve share is higher than the measured
   one, so the closed loop adds heat.
4. **The COP error comes from extra starts.** S3 had about 37% more heating runs than S2; hours with two
   or more starts are under-predicted by about 25%, while the error is uniform across lift bins. A
   starts/cycling term is missing from the lift table; this is not a lift error.

### 7.3 Most promising model (exploratory)

A **2-state (air plus slab) model fitted on room air only**, with a **weekday/weekend daily gain**, and the
trailing-window **weekly re-estimation** of the level parameters (UA, q0). Room-mean RMSE at 6 and 12 h
(S2 to S3, measured-heat driven): **0.249 and 0.325 K**, against 0.334 and 0.426 for C2's model A and
0.447 and 0.536 for persistence. The reverse fold (S3 to S2) improves too. The daily gain is the largest
single contribution (about 20% at 6 and 12 h in both directions); the air-only 2-state structure adds a
few percent; valve handling adds about 0.01 K.

Limits: at 24 h no structure passes a 0.15 K bias limit across seasons without adaptive re-estimation, 1 h
ties persistence (sensor resolution), and 3 h is only 5 to 20% better. **The useful horizon is 6 to
12 h.** Hall/kitchen is the hardest room, and a per-room model is useful to the controller but no shortcut
for accuracy. A good heat-driven fit does not license actions while the permission-to-heat link fails.
The preregistered thresholds for the next test are in the
[2026-27 preregistration](climate-preregistration-2026-27.md).

### 7.4 Reproduction

The scripts are **not committed**: they hard-code local paths, cache pickles and carry exploration-order
names, so they are not a clean runner. They live next to the outputs in the private data directory under
`analysis/v1/climate-b0/b0-20261031/` (`scripts/`, run id `b0-20261031`). The inputs are the climate
feature set `da6d678016ed`, dataset `cdd448abb006`, `config/climate.yaml` version 2 (hash `22888468...`)
and the repository at `4d1dadf` (climate code identical to the C2 run's `9055b4d`). Run them with the pinned
uv from the scripts directory, for example `uvx uv@<pinned> run --frozen --project <repo> python r0.py`.
They would need to become a clean `hack/` runner before any confirmatory use.
