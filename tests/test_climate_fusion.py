"""The fused outdoor temperature in the climate loader: fold discipline, ids and fallbacks.

Synthetic station and house series: a diurnal weather, a house sensor with a persistent offset.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import numpy as np
import pytest
from climate_fakes import REPO_ROOT, SITE, grid, make_table
from station_fakes import STATION, ogimet_transport
from test_climate_data import _write_feature_set

from battery_optimizer.climate.config import ClimateConfigError, load_climate_config
from battery_optimizer.climate.data import (
    SLOT_US,
    ClimateDataError,
    Site,
    build_climate_data,
    load_climate_data,
    load_site,
)
from battery_optimizer.climate.station import StationSeries, fetch_station

FIRST = date(2025, 12, 1)
N_DAYS = 30
CONFIG_TEXT = (REPO_ROOT / "config" / "climate.yaml").read_text()


def _config(tmp_path: Path, *, source="debiased", fit_seasons="[S1, S2]", fill=False):
    text = CONFIG_TEXT
    for name, (a, b) in {"S1": ("01", "10"), "S2": ("11", "20"), "S3": ("21", "30")}.items():
        old = next(line for line in text.splitlines() if line.startswith(f"  {name}: {{start"))
        text = text.replace(old, f"  {name}: {{start: 2025-12-{a}, end: 2025-12-{b}}}")
    text = text.replace("t_out_source: debiased", f"t_out_source: {source}")
    text = text.replace("fit_seasons: [S1, S2]", f"fit_seasons: {fit_seasons}")
    old = "t_out_fill_unknown_from_fused: true"
    assert old in text
    text = text.replace(old, f"t_out_fill_unknown_from_fused: {str(fill).lower()}")
    path = tmp_path / "climate.yaml"
    path.write_text(text)
    return load_climate_config(path)


def weather(t_us: np.ndarray) -> np.ndarray:
    hours = t_us / 3.6e9
    return 5.0 + 4.0 * np.sin(2 * np.pi * hours / 24.0) + 1.5 * np.sin(2 * np.pi * hours / 7.3)


def _station(manifest: str = "m1", skip=None) -> StationSeries:
    start = int(datetime(2025, 11, 30, tzinfo=UTC).timestamp()) * 10**6
    times = start + np.arange(0, 34 * 24) * 3600 * 10**6
    keep = np.ones(times.size, dtype=bool) if skip is None else ~skip(times)
    t = times[keep]
    return StationSeries(STATION, t, np.round(weather(t), 1), ("2025-12",), manifest)


def _house(starts: np.ndarray, *, noise=0.1, day_bias=0.0, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mid = starts + SLOT_US // 2
    offset = -0.8 + 0.3 * np.sin(2 * np.pi * (mid / 3.6e9) / 150.0)
    return weather(mid) + offset + rng.normal(0.0, noise, starts.size)


def _build(cfg, raw, station, site=SITE):
    starts = grid(FIRST, N_DAYS)
    table = make_table(starts, outdoor_temp_mean_c=raw, compressor_freq_mean=0.0)
    return build_climate_data(table, config=cfg, site=site, station=station)


@pytest.fixture
def starts():
    return grid(FIRST, N_DAYS)


def test_fused_series_tracks_the_weather_and_keeps_raw_and_debiased(tmp_path, starts):
    cfg = _config(tmp_path)
    raw = _house(starts)
    d = _build(cfg, raw, _station())
    f = d.fusion
    assert f is not None and f.n_fit_obs > 500
    assert np.array_equal(d.t_out, raw) and d.t_out_source == "debiased"
    assert d.t_out_model is d.t_out_debiased  # the default stays debiased
    mid = starts + SLOT_US // 2
    inner = slice(8, -8)
    assert np.abs(f.fused[inner] - raw[inner]).mean() < 0.25  # fused follows the house sensor ...
    # ... and the offset is the one that was put in (about -0.8 K, see _house)
    assert f.params.mu == pytest.approx(-0.8, abs=0.3)
    assert np.isfinite(f.sd[inner]).all() and f.from_reference[inner].all()
    assert np.abs(f.reference[inner] - weather(mid)[inner]).max() < 0.5
    fused_cfg = _config(tmp_path, source="fused")
    assert _build(fused_cfg, raw, _station()).t_out_model.shape == raw.shape


def test_no_s3_reading_tunes_the_parameters(tmp_path, starts):
    cfg = _config(tmp_path)
    raw = _house(starts)
    base = _build(cfg, raw, _station())
    slot_s3 = base.day_mask(base.day_season == "S3")
    assert slot_s3.any() and base.fusion.observed[slot_s3].any()  # S3 has trusted dark readings
    poisoned = raw.copy()
    poisoned[slot_s3] += 7.0
    other = _build(cfg, poisoned, _station())
    assert other.fusion.params == base.fusion.params  # not one S3 reading entered the fit
    assert other.fusion.n_fit_obs == base.fusion.n_fit_obs
    # the smoother still reads them: the fused series moves in S3 and not before it
    assert np.abs(other.fusion.fused[slot_s3] - base.fusion.fused[slot_s3]).max() > 1.0
    s1 = base.day_mask(base.day_season == "S1")
    assert np.abs(other.fusion.fused[s1][:200] - base.fusion.fused[s1][:200]).max() < 0.5
    # fitting on the held-out season is refused by the configuration
    with pytest.raises(ClimateConfigError, match="held-out"):
        _config(tmp_path, fit_seasons="[S1, S3]")


def test_daytime_readings_do_not_enter_the_offset(tmp_path, starts):
    cfg = _config(tmp_path)
    raw = _house(starts)
    base = _build(cfg, raw, _station())
    sunny = raw.copy()
    day = base.solar >= cfg.signals.t_out_fused.dark_solar_max
    assert day.any() and base.fusion.observed.any() and not base.fusion.observed[day].any()
    sunny[day] += 3.0  # the sensor's midday sun bias
    other = _build(cfg, sunny, _station())
    assert other.fusion.params == base.fusion.params
    assert np.allclose(other.fusion.fused, base.fusion.fused)


def test_long_reference_gaps_fall_back_to_the_debiased_series(tmp_path, starts):
    cfg = _config(tmp_path)
    raw = _house(starts)
    day = int(datetime(2025, 12, 12, tzinfo=UTC).timestamp()) * 10**6
    gap = lambda t: (t >= day + 6 * 3600 * 10**6) & (t < day + 12 * 3600 * 10**6)  # noqa: E731
    d = _build(cfg, raw, _station(skip=gap))
    f = d.fusion
    mid = starts + SLOT_US // 2
    inside = (mid > day + 7 * 3600 * 10**6) & (mid < day + 11 * 3600 * 10**6)
    assert (~f.from_reference[inside]).all() and np.isnan(f.sd[inside]).all()
    assert np.array_equal(f.fused[inside], d.t_out_debiased[inside], equal_nan=True)
    short = lambda t: (t >= day + 6 * 3600 * 10**6) & (t < day + 8 * 3600 * 10**6)  # noqa: E731
    bridged = _build(cfg, raw, _station(skip=short)).fusion
    assert bridged.from_reference[(mid > day + 6.2 * 3600 * 10**6) & (mid < day + 8 * 3600e6)].all()


def test_fused_source_follows_into_the_outdoor_filter(tmp_path, starts):
    raw = _house(starts)
    raw[300:340] = np.nan  # the house sensor is unknown, the station is not
    debiased = _build(_config(tmp_path), raw, _station())
    fused = _build(_config(tmp_path, source="fused"), raw, _station())
    assert debiased.drop["outdoor_unknown"][310] and not fused.drop["outdoor_unknown"][310]
    assert np.isfinite(fused.t_out_model[310]) and np.isnan(fused.t_out_debiased[310])


def test_long_gap_fill_touches_only_the_unknown_debiased_slots(tmp_path, starts):
    raw = _house(starts)
    raw[300:340] = np.nan  # the house sensor is unknown, the station is not
    freq = np.zeros(starts.size)
    freq[1000:1040] = 40.0  # a 10 h compressor run: longer than the 6 h interpolation cap
    table = make_table(starts, outdoor_temp_mean_c=raw, compressor_freq_mean=freq)
    plain = build_climate_data(table, config=_config(tmp_path), site=SITE, station=_station())
    filled = build_climate_data(
        table, config=_config(tmp_path, fill=True), site=SITE, station=_station()
    )
    unknown = ~np.isfinite(plain.t_out_debiased)
    assert unknown[310] and unknown[1020] and plain.t_out_filled is None
    mask = filled.t_out_filled
    assert np.array_equal(mask, unknown & filled.fusion.from_reference) and mask[[310, 1020]].all()
    # every other slot keeps the debiased value bit for bit; the filled ones take the fused value
    assert np.array_equal(filled.t_out_model[~mask], plain.t_out_debiased[~mask], equal_nan=True)
    assert np.array_equal(filled.t_out_model[mask], filled.fusion.fused[mask])
    assert np.array_equal(filled.t_out_debiased, plain.t_out_debiased, equal_nan=True)
    assert not filled.drop["outdoor_unknown"][[310, 1020]].any()
    assert plain.drop["outdoor_unknown"][[310, 1020]].all()
    assert filled.input_sha256 != plain.input_sha256  # a different config: a different input id
    # without a station there is nothing to fill from: the series stays debiased
    alone = build_climate_data(table, config=_config(tmp_path, fill=True), site=SITE)
    assert alone.t_out_filled is None and alone.t_out_model is alone.t_out_debiased


def test_ids_follow_the_station_manifest(tmp_path, starts):
    cfg = _config(tmp_path)
    raw = _house(starts)
    a = _build(cfg, raw, _station("manifest-a"))
    b = _build(cfg, raw, _station("manifest-b"))
    again = _build(cfg, raw, _station("manifest-a"))
    assert a.station_sha256 == "manifest-a" and b.station_sha256 == "manifest-b"
    assert a.input_sha256 == again.input_sha256 != b.input_sha256
    plain = build_climate_data(
        make_table(starts, outdoor_temp_mean_c=raw, compressor_freq_mean=0.0),
        config=cfg,
        site=SITE,
    )
    assert plain.station_sha256 == "" and plain.input_sha256 != a.input_sha256
    assert plain.fusion is None and plain.t_out_model is plain.t_out_debiased
    # the source, the feature set and the site are inputs too
    fused = _build(_config(tmp_path, source="fused"), raw, _station("manifest-a"))
    assert fused.input_sha256 != a.input_sha256
    other_site = _build(cfg, raw, _station("manifest-a"), site=Site(40.0, 0.0, None, "other"))
    assert other_site.input_sha256 != a.input_sha256


def test_fused_source_without_a_station_is_refused(tmp_path, starts):
    cfg = _config(tmp_path, source="fused")
    table = make_table(starts, compressor_freq_mean=0.0)
    with pytest.raises(ClimateDataError, match="no station series"):
        build_climate_data(table, config=cfg, site=SITE)


def test_too_few_training_observations_is_an_error(tmp_path, starts):
    cfg = _config(tmp_path)
    raw = np.full(starts.size, np.nan)
    with pytest.raises(ClimateDataError, match="cannot be fitted"):
        _build(cfg, raw, _station())


def test_site_file_reference_station_is_optional_validated_and_hashed(tmp_path):
    p = tmp_path / "site.yaml"
    p.write_text('latitude: 40\nlongitude: 0\nreference_station: "12346"\n')
    site = load_site(p)
    assert site.reference_station == "12346"
    assert site.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()
    p.write_text("latitude: 40\nlongitude: 0\n")
    assert load_site(p).reference_station is None
    for bad in ("8227", "abcde", "1234567", "[1]", "true"):
        p.write_text(f"latitude: 40\nlongitude: 0\nreference_station: {bad}\n")
        with pytest.raises(ClimateDataError, match="reference_station"):
            load_site(p)


def test_loader_reads_the_store_of_the_site_reference_station(tmp_path, starts):
    cfg = _config(tmp_path)
    root = tmp_path / "store"
    mid = starts + SLOT_US // 2
    table = make_table(starts, outdoor_temp_mean_c=_house(starts), compressor_freq_mean=0.0)
    _write_feature_set(root, table, "fs1")
    (root / "climate").mkdir()
    site_file = root / "climate" / "site.yaml"
    site_file.write_text(f'latitude: 40\nlongitude: 0\nreference_station: "{STATION}"\n')

    def truth(t: datetime) -> float:
        return float(weather(np.array([t.timestamp() * 1e6]))[0])

    # nothing fetched yet: a clear error, not a silent fallback
    with pytest.raises(ClimateDataError, match="lacks the month"):
        load_climate_data(cfg, feature_set="fs1", data_dir=str(root))
    client = httpx.Client(transport=ogimet_transport(truth))
    months = [(2025, 11), (2025, 12), (2026, 1)]
    fetch_station(root, STATION, months, client=client, sleep=lambda s: None)
    d = load_climate_data(cfg, feature_set="fs1", data_dir=str(root))
    assert d.fusion is not None and len(d.station_sha256) == 64
    assert np.abs(d.fusion.reference[8:-8] - weather(mid)[8:-8]).max() < 0.5
    assert d.site_sha256 == load_site(site_file).sha256
    # without the key there is no fusion; the loader never reaches for a station
    site_file.write_text("latitude: 40\nlongitude: 0\n")
    assert load_climate_data(cfg, feature_set="fs1", data_dir=str(root)).fusion is None
