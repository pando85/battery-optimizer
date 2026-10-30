# C2 recovery: diagnostic-first thermal identification after UNDETERMINED

- **Status:** Proposed recovery plan (not an amendment to C2 or approval for actuation).
- **Recorded outcome:** 2026-10-30, C2 config v2, first and only predeclared iteration: **UNDETERMINED / no-go**.
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
