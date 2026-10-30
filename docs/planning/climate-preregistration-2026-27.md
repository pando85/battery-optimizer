# Climate preregistration, winter 2026-27

- **Committed:** 2026-10-31, **before any 2026-27 heating data has been seen**. The last heating data in the
  feature set ends 2026-05-01 and space heating has been off since (see the [regime log](../components/regime-log.md)).
- **Covers:** (A) the shadow thermal model and MPC (track B, step B3), (B) the comfort controller
  before/after comparison ([design section 12](comfort-controller-design.md#12-evaluation)).
- **Review:** the owner reviews these criteria **together with the homelab comfort-controller PR before
  go-live**. Thresholds change **only before go-live**, through a dated, visible amendment (section 4). After
  go-live nothing here is edited except to add a result.
- **Not in scope:** euro figures, savings and absolute energy totals. Every criterion is relative to a
  stated baseline or a physical unit.
- **Context:** [C2 recovery, section 7](climate-c2-recovery.md#7-status-2026-10-31-b0-results-r0-to-r2) and the
  [thermal model](../components/thermal-model.md) page. The numbers of B0 are exploratory and are not these
  criteria's result.

## 1. Shared definitions

- **Regimes.** *Legacy regime:* the shadow deployment until go-live of the comfort controller. *New regime:*
  go-live plus 7 days (settling) through **2027-03-31**. The days between are excluded from both.
- **Valid heating day.** A day with space heating running in at least part of the day, and with the
  thermostat-room temperatures, the heat-pump energy source and the outdoor series present for at least 90%
  of the 15-minute slots. The owner confirms this definition at review.
- **Room-mean.** The mean of the thermostat rooms listed in [design section 3](comfort-controller-design.md#3-comfort-schedule-and-setpoints).
- **Persistence.** The last observed value held over the horizon. **Skill** is
  `1 - RMSE(model) / RMSE(persistence)` on the same forecast origins.
- **UNDETERMINED** means the data requirement was not met. It is never read as a pass.

## 2. A. Shadow thermal model and MPC (track B, B3)

The candidate is the 2-state, room-air-only model with a weekday/weekend daily gain and weekly re-estimation
(C2 recovery section 7.3), predicting from forecast origins in the live data. **Two separate windows, two
separate verdicts:** the legacy regime and the new regime. Each verdict needs **at least 6 weeks of valid
heating days**, otherwise **UNDETERMINED**.

| Criterion | Threshold |
|---|---|
| Room-mean RMSE at 6 h | at most 0.30 K |
| Room-mean RMSE at 12 h | at most 0.40 K |
| Skill versus persistence | at least 25% at **both** horizons |
| Mean bias at 6 h | absolute value at most 0.15 K |
| Daily heat-delivery error | median absolute percentage error at most 15%, on days whose heat is above 20% of the window's median daily heat |

**Verdict.** PASS on all criteria: the MPC becomes a **candidate** for a real alternating-arm trial in a later
season, which is a separate decision with its own preregistration. FAIL on any criterion: the MPC is not
promoted.

The MPC plan-versus-rule comparison (predicted kWh of the MPC plan against the realised kWh of the rule) is
**descriptive only** and grants nothing: the MPC never acts, so its realised comfort cannot be observed.

## 3. B. Comfort controller before/after (design section 12)

The **new regime** is compared with the **S3 legacy baseline** (winter 2025-26), **matched by outdoor-temperature
band** (the bands of the warm-up analysis: below 4, 4 to 8, 8 to 12 and 12 degC and above; a band with fewer than
10 days in either regime is pooled with its neighbour). Heat-pump electricity is taken on the **heat-pump
branch meter or HeishaMon basis, never the whole-house meter**. At least **4 weeks of valid heating days**
are required, otherwise **UNDETERMINED**.

| Group | Criterion | Threshold |
|---|---|---|
| Comfort | Hall/kitchen below 23.5 degC | at most 10% of the day-window time |
| Comfort | Each thermostat room below 21 degC | at most 3% of all time |
| Efficiency | Heat-pump electricity per heating degree-day (base 15 degC, as `HDD_BASE_C` in `hack/system_sim.py:64`, so it stays on the replay's basis; daily mean outdoor) | at most +20% against the S3 band-matched baseline |
| Machine | Compressor starts per day | median at most 20 |
| Machine | Electric heater | no controller-caused activation (DHW heater hours unchanged) |
| DHW | Share of DHW heat-pump energy in the scheduled window | at least 80% |
| DHW | Tank temperature from the scheduler | never above 50 degC |

**Rationale for the efficiency margin.** The comfort target is about **1.3 K above** S3's realised day mean in
the hall/kitchen, so a higher use is expected: roughly **+8 to 13%**. The +20% limit allows that plus
estimation error and no more.

**Verdict.** **GO** (keep the controller) if every criterion passes. **NO-GO** if any fails: `git revert` in
homelab (design section 10). **UNDETERMINED** if the data is insufficient; the controller stays and the
period is extended or the comparison is repeated next winter.

**Strength of evidence.** A before/after comparison is **weaker than a randomised trial**. Weather, occupancy,
tariff, the setpoint and the water target all differ between periods, and the band matching and the
degree-day normalisation are imperfect. A GO means "no sign of harm against these limits", not a measured
saving. Every result is reported with this limit stated.

## 4. Amendments

Allowed only **before go-live**, one dated entry each, naming the threshold, the old and the new value and the
reason. The commit history of this file is the audit trail. No amendment has been made.

| Date | Criterion | Old | New | Reason |
|---|---|---|---|---|
| (none) | | | | |

## 5. Items for the owner at review

- The validity definition (section 1) and the room list (design section 3). The degree-day base follows `hack/system_sim.py`.
- The go-live date: it fixes the start of the new regime and the end of the legacy window.
