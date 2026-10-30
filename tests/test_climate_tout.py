"""Kalman filter, RTS smoother and maximum likelihood of the outdoor offset (synthetic data)."""

from __future__ import annotations

import numpy as np
import pytest

from battery_optimizer.climate.tout import (
    OffsetParams,
    compressor_start_step,
    fit_offset_model,
    fuse,
    interpolate_trusted,
    kalman_filter,
    leave_block_out,
    offset_observations,
    rts_smoother,
)

TRUE = OffsetParams(mu=-0.8, phi=0.97, q=0.02, r=0.05)


def simulate(n: int, params: OffsetParams, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """``(state x, observation y = mu + x + v)`` of the offset model."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = rng.normal(0.0, np.sqrt(params.stationary_variance))
    w = rng.normal(0.0, np.sqrt(params.q), n)
    for t in range(1, n):
        x[t] = params.phi * x[t - 1] + w[t]
    return x, params.mu + x + rng.normal(0.0, np.sqrt(params.r), n)


def gappy(y: np.ndarray, seed: int = 2, share: float = 0.3, run: int = 24) -> np.ndarray:
    """Keep *share* of the slots, in runs of *run* slots, the way trusted readings come."""
    rng = np.random.default_rng(seed)
    keep = np.zeros(y.size, dtype=bool)
    for s in range(0, y.size, run):
        keep[s : s + run] = rng.random() < share
    return np.where(keep, y, np.nan)


def test_mle_recovers_the_known_parameters():
    _x, y = simulate(40_000, TRUE)
    fit = fit_offset_model(gappy(y))
    assert fit.converged and fit.n_obs > 8_000
    p = fit.params
    assert p.mu == pytest.approx(TRUE.mu, abs=0.15)
    assert p.phi == pytest.approx(TRUE.phi, abs=0.01)
    assert p.q == pytest.approx(TRUE.q, rel=0.25)
    assert p.r == pytest.approx(TRUE.r, rel=0.15)


def test_mle_likelihood_is_higher_at_the_fit_than_at_wrong_parameters():
    _x, y = simulate(6_000, TRUE, seed=5)
    y = gappy(y, seed=6)
    fit = fit_offset_model(y)
    for wrong in (
        OffsetParams(TRUE.mu, 0.5, TRUE.q, TRUE.r),
        OffsetParams(TRUE.mu, TRUE.phi, TRUE.q * 20, TRUE.r),
        OffsetParams(TRUE.mu + 2, TRUE.phi, TRUE.q, TRUE.r),
    ):
        assert kalman_filter(y, wrong).loglik < fit.loglik


def test_mle_needs_enough_observations():
    with pytest.raises(ValueError, match="at least"):
        fit_offset_model(np.full(1_000, np.nan))


def test_smoother_beats_the_filter_inside_gaps():
    x, y = simulate(30_000, TRUE, seed=3)
    obs = gappy(y, seed=4, share=0.5, run=20)
    hidden = np.isnan(obs)
    filt = kalman_filter(obs, TRUE)
    smooth = rts_smoother(obs, TRUE)
    rmse_f = np.sqrt(np.mean((filt.x[hidden] - x[hidden]) ** 2))
    rmse_s = np.sqrt(np.mean((smooth.x[hidden] - x[hidden]) ** 2))
    assert rmse_s < 0.9 * rmse_f
    assert (smooth.p <= filt.p + 1e-12).all()  # two-sided information never adds variance
    assert smooth.x[-1] == filt.x[-1] and smooth.p[-1] == filt.p[-1]
    # the reported variance is honest: about 95% of the truth falls inside two sd
    inside = np.abs(smooth.x[hidden] - x[hidden]) < 2 * np.sqrt(smooth.p[hidden])
    assert 0.92 < inside.mean() < 0.99


def test_uncertainty_grows_in_a_gap_and_returns_to_the_stationary_variance():
    y = np.full(2_000, np.nan)
    y[:50] = 0.0
    p = kalman_filter(y, TRUE).p
    assert p[49] < p[100] < p[1_000] <= TRUE.stationary_variance + 1e-9
    assert p[-1] == pytest.approx(TRUE.stationary_variance, rel=1e-3)


def test_filter_is_causal_and_the_smoother_is_not():
    _x, y = simulate(500, TRUE, seed=7)
    changed = y.copy()
    changed[400:] += 5.0
    a, b = kalman_filter(y, TRUE), kalman_filter(changed, TRUE)
    assert np.array_equal(a.x[:400], b.x[:400])
    assert not np.allclose(rts_smoother(y, TRUE).x[:400], rts_smoother(changed, TRUE).x[:400])


def test_fuse_is_reference_plus_offset_and_nan_without_reference():
    ref = np.linspace(0.0, 10.0, 400)
    x, y_true = simulate(400, TRUE, seed=8)
    ref[100:120] = np.nan
    y = offset_observations(ref + y_true, ref, np.ones(400, dtype=bool))
    for mode in ("smoother", "filter"):
        fused, sd = fuse(ref, y, TRUE, mode=mode)
        assert np.isnan(fused[100:120]).all() and np.isnan(sd[100:120]).all()
        assert np.allclose(
            fused[:100],
            ref[:100]
            + TRUE.mu
            + (rts_smoother(y, TRUE).x if mode == "smoother" else kalman_filter(y, TRUE).x)[:100],
        )
        assert (sd[:100] > 0).all()
    with pytest.raises(ValueError, match="mode"):
        fuse(ref, y, TRUE, mode="oracle")


def test_offset_observations_need_both_series_and_the_mask():
    house = np.array([1.0, 2.0, np.nan, 4.0])
    ref = np.array([0.5, np.nan, 0.5, 1.0])
    obs = np.array([True, True, True, False])
    out = offset_observations(house, ref, obs)
    assert out[0] == 0.5 and np.isnan(out[1:]).all()


def test_parameters_are_validated():
    for bad in ((0.0, 1.0, 0.1, 0.1), (0.0, 0.5, 0.0, 0.1), (0.0, 0.5, 0.1, -1.0)):
        with pytest.raises(ValueError):
            OffsetParams(*bad)


# ---------------------------------------------------------------------------
# Leave-block-out and the compressor-start step
# ---------------------------------------------------------------------------


def _scene(n=6_000, seed=11):
    """A diurnal true temperature, a station that sees it (hourly noise-free), a house sensor
    with a persistent offset and noise, trusted slots in long runs."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    truth = 8.0 + 5.0 * np.sin(2 * np.pi * t / 96.0) + 2.0 * np.sin(2 * np.pi * t / 37.0)
    ref = truth.copy()
    offset = -0.8 + np.cumsum(rng.normal(0, 0.02, n)) * 0.2
    house = truth + offset + rng.normal(0, 0.15, n)
    trusted = np.ones(n, dtype=bool)
    for s in range(0, n, 150):  # a running period every ~37 h: 40 slots untrusted
        trusted[s : s + 40] = False
    return house, ref, trusted


def test_interpolate_trusted_matches_the_debiased_rule():
    values = np.arange(20, dtype=float)
    trusted = np.zeros(20, dtype=bool)
    trusted[[2, 3, 8, 15]] = True
    out = interpolate_trusted(values, trusted, max_gap_slots=4)
    assert np.isnan(out[:2]).all() and np.allclose(out[2:9], np.arange(2, 9))
    assert np.isnan(out[9:15]).all() and out[15] == 15 and np.isnan(out[16:]).all()


def test_leave_block_out_hides_the_blocks_from_both_predictors():
    house, ref, trusted = _scene()
    dark = trusted.copy()
    kwargs = dict(
        reference=ref,
        trusted=trusted,
        observed=dark,
        params=OffsetParams(-0.8, 0.99, 0.0005, 0.02),
        region=np.ones(house.size, dtype=bool),
        block_slots=8,
        stride_slots=96,
        phase_slots=48,
        max_gap_slots=24,
    )
    out = leave_block_out(house=house, **kwargs)
    assert out["slot"].size and (np.diff(out["slot"])[np.diff(out["slot"]) > 1] >= 96 - 8).all()
    assert np.array_equal(out["truth"], house[out["slot"]])
    # the hidden readings cannot leak: changing them changes neither prediction
    altered = house.copy()
    altered[out["slot"]] += 50.0
    again = leave_block_out(house=altered, **kwargs)
    assert np.array_equal(out["slot"], again["slot"])
    assert np.allclose(out["fused"], again["fused"]) and np.allclose(
        out["debiased"], again["debiased"], equal_nan=True
    )
    # blocks with an untrusted slot are not tested
    assert not trusted[out["slot"]].size == 0 and trusted[out["slot"]].all()


def test_fused_beats_interpolation_when_the_reference_tracks_the_weather():
    house, ref, trusted = _scene()
    params = OffsetParams(-0.8, 0.99, 0.0005, 0.02)
    errs = {"debiased": [], "fused": []}
    for phase in range(0, 96, 12):
        out = leave_block_out(
            house=house,
            reference=ref,
            trusted=trusted,
            observed=trusted,
            params=params,
            region=np.ones(house.size, dtype=bool),
            block_slots=24,
            stride_slots=96,
            phase_slots=phase,
            max_gap_slots=24,
        )
        ok = np.isfinite(out["debiased"]) & np.isfinite(out["fused"])
        for key in errs:
            errs[key].append(out[key][ok] - out["truth"][ok])
    rmse = {k: float(np.sqrt(np.mean(np.concatenate(v) ** 2))) for k, v in errs.items()}
    assert rmse["fused"] < 0.5 * rmse["debiased"]


def test_no_blocks_when_no_block_is_fully_trusted():
    house, ref, _ = _scene(600)
    out = leave_block_out(
        house=house,
        reference=ref,
        trusted=np.zeros(600, dtype=bool),
        observed=np.zeros(600, dtype=bool),
        params=TRUE,
        region=np.ones(600, dtype=bool),
        block_slots=4,
        stride_slots=12,
        phase_slots=0,
        max_gap_slots=24,
    )
    assert out["slot"].size == 0


def test_compressor_start_step_nets_out_the_weather():
    n = 400
    freq = np.zeros(n)
    freq[100:160] = 50.0
    freq[300:330] = 50.0
    weather = 0.01 * np.arange(n)
    raw = weather.copy()
    for s, e in ((100, 160), (300, 330)):
        raw[s:e] -= 2.0
    steps = compressor_start_step(raw, weather, freq, min_off_slots=2, window_slots=1)
    assert steps.size == 2 and np.allclose(steps, -2.0, atol=0.03)
    clean = compressor_start_step(weather, weather, freq, min_off_slots=2, window_slots=1)
    assert np.allclose(clean, 0.0)
    # a start right after another run (not off long enough) is not a start
    freq[160:162] = 0.0
    freq[162:170] = 50.0
    assert compressor_start_step(raw, weather, freq, min_off_slots=4).size == 2
