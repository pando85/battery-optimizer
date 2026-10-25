"""Structured component forecaster (LF-1): rule replay, as-of inputs, fallbacks and leakage.

Everything here is synthetic.  The rule-replay tests use hand-built days; the end-to-end
tests build a small world whose heat-pump energy follows the replayed rule, so the forecaster
has something to find, and then perturb every value the decision must not know.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from battery_optimizer import load_forecast as lf
from battery_optimizer import structured_forecast as sf
from battery_optimizer.policies import contiguous_day_bounds

MADRID = ZoneInfo("Europe/Madrid")
# 2025-08-31T22:00Z is local midnight (CEST) of 2025-09-01; 80 days reach 2025-11-19 and
# cross the 2025-10-26 fall-back day (100 intervals).
START = datetime(2025, 8, 31, 22, tzinfo=UTC)
N_DAYS = 80


# ---------------------------------------------------------------------------
# Rule replay on hand-built days
# ---------------------------------------------------------------------------


def _day(prices: list[float], present: list[bool] | None = None):
    p = np.array(prices + [np.nan] * (24 - len(prices)), dtype=float)
    pr = np.array(present if present is not None else [True] * 24, dtype=bool)
    return p, pr


class TestReplayClimateRule:
    def test_cheap_branch_takes_every_hour_below_two_thirds_of_the_mean(self) -> None:
        prices = [0.30] * 8 + [0.10] * 6 + [0.30] * 10
        p, present = _day(prices)
        schedule = sf.replay_climate_rule(p, present, trailing_mean=0.24, min_hours=4)
        assert schedule is not None and schedule.cheap_branch
        assert np.flatnonzero(schedule.selected).tolist() == list(range(8, 14))
        assert schedule.limit == pytest.approx(0.16)

    def test_the_limit_is_strict(self) -> None:
        p, present = _day([0.16] * 6 + [0.50] * 18)
        schedule = sf.replay_climate_rule(p, present, trailing_mean=0.24, min_hours=2)
        assert schedule is not None and not schedule.cheap_branch
        # 0.16 is not below 0.24 * 2/3 (up to rounding): fall back to the cheapest two.
        assert schedule.hours == 2

    def test_fewer_cheap_hours_than_min_hours_falls_back_to_the_cheapest_n(self) -> None:
        prices = [0.30, 0.31, 0.12, 0.29, 0.28, 0.13, 0.27, 0.26] + [0.30] * 16
        p, present = _day(prices)
        schedule = sf.replay_climate_rule(p, present, trailing_mean=0.24, min_hours=5)
        assert schedule is not None and not schedule.cheap_branch
        # Two hours are cheap (< 0.16) but five are wanted: the five cheapest, clock order
        # breaking the 0.26 / 0.27 / 0.28 ties exactly as the stable sort of the app does.
        assert np.flatnonzero(schedule.selected).tolist() == [2, 4, 5, 6, 7]

    def test_equal_prices_are_taken_in_clock_order(self) -> None:
        p, present = _day([0.20] * 24)
        schedule = sf.replay_climate_rule(p, present, trailing_mean=0.20, min_hours=3)
        assert schedule is not None
        assert np.flatnonzero(schedule.selected).tolist() == [0, 1, 2]

    def test_more_cheap_hours_than_min_hours_keeps_them_all(self) -> None:
        p, present = _day([0.05] * 20 + [0.40] * 4)
        schedule = sf.replay_climate_rule(p, present, trailing_mean=0.20, min_hours=6)
        assert schedule is not None and schedule.cheap_branch and schedule.hours == 20

    def test_zero_min_hours_selects_only_cheap_hours(self) -> None:
        p, present = _day([0.30] * 24)
        schedule = sf.replay_climate_rule(p, present, trailing_mean=0.30, min_hours=0)
        assert schedule is not None and schedule.cheap_branch and schedule.hours == 0

    def test_a_day_with_fewer_priced_hours_than_min_hours_has_no_schedule(self) -> None:
        p, present = _day([0.1] * 5)
        assert sf.replay_climate_rule(p, present, trailing_mean=0.2, min_hours=6) is None

    def test_a_negative_trailing_mean_reproduces_the_apps_arithmetic(self) -> None:
        # 2/3 of a negative mean is above the mean, so every hour at the mean is "cheap".
        p, present = _day([-0.02] * 24)
        schedule = sf.replay_climate_rule(p, present, trailing_mean=-0.02, min_hours=3)
        assert schedule is not None and schedule.cheap_branch and schedule.hours == 24

    def test_a_spring_forward_day_has_no_hour_two(self) -> None:
        # 23 clock hours: hour 02 does not exist.
        present = [h != 2 for h in range(24)]
        prices = [0.30] * 24
        prices[3] = 0.05
        prices[1] = 0.06
        p, pr = _day(prices, present)
        p[2] = np.nan
        schedule = sf.replay_climate_rule(p, pr, trailing_mean=0.30, min_hours=2)
        assert schedule is not None
        assert np.flatnonzero(schedule.selected).tolist() == [1, 3]
        assert not schedule.selected[2]


class TestReplayDhwRule:
    def test_cheapest_hour_of_each_twelve_hour_block(self) -> None:
        prices = np.array([0.3] * 24)
        prices[5], prices[17] = 0.1, 0.1
        assert sf.replay_dhw_rule(prices, np.ones(24, bool), interval_hours=12) == (5, 17)

    def test_one_hour_per_day_with_a_24_hour_interval(self) -> None:
        prices = np.array([0.3] * 24)
        prices[19], prices[4] = 0.1, 0.2
        assert sf.replay_dhw_rule(prices, np.ones(24, bool), interval_hours=24) == (19,)

    def test_the_first_cheapest_hour_wins_a_tie(self) -> None:
        assert sf.replay_dhw_rule(np.full(24, 0.2), np.ones(24, bool), interval_hours=12) == (0, 12)

    def test_a_23_hour_day_is_cut_into_blocks_of_12_and_11_priced_hours(self) -> None:
        present = np.array([h != 2 for h in range(24)])
        prices = np.full(24, 0.3)
        prices[23] = 0.1  # priced hours: 0,1,3..23; the second chunk is the last 11 of them
        prices[13] = 0.2
        hours = sf.replay_dhw_rule(prices, present, interval_hours=12)
        assert hours == (0, 23)
        # The first chunk is hours 0,1,3..11,12 (12 entries): 13 starts the second.
        prices2 = np.full(24, 0.3)
        prices2[12] = 0.05
        prices2[13] = 0.06
        assert sf.replay_dhw_rule(prices2, present, interval_hours=12) == (12, 13)


class TestHelpers:
    def test_block_positions(self) -> None:
        sel = np.zeros(24, bool)
        sel[[3, 4, 5, 6, 10, 23]] = True
        pos = sf.block_positions(sel)
        assert pos[[3, 4, 5, 6, 10, 23]].tolist() == [0, 1, 2, 2, 0, 0]
        assert pos[0] == -1

    def test_mode_is_the_dominant_energy_of_the_last_48_hours(self) -> None:
        n = 400
        heat = np.zeros(n)
        cool = np.zeros(n)
        cool[200:300] = 0.1
        heat[200:210] = 0.05
        assert sf.infer_mode(heat, cool, 300) == ("cool", "")
        heat[250:290] = 0.5
        assert sf.infer_mode(heat, cool, 300) == ("heat", "")

    def test_mode_is_unknown_without_energy_data_or_history(self) -> None:
        zeros = np.zeros(400)
        assert sf.infer_mode(zeros, zeros, 300) == (None, "mode_no_recent_climate_energy")
        assert sf.infer_mode(zeros, zeros, 100) == (None, "mode_history_short")
        gappy = np.full(400, np.nan)
        assert sf.infer_mode(gappy, gappy, 300) == (None, "mode_data_missing")

    def test_rule_versions_follow_the_homelab_history(self) -> None:
        v = sf._version_for
        assert v(sf.CLIMATE_RULE_VERSIONS, date(2025, 6, 1)).history_days == 10
        assert v(sf.CLIMATE_RULE_VERSIONS, date(2026, 4, 28)).history_days == 7
        assert not v(sf.CLIMATE_RULE_VERSIONS, date(2026, 9, 4)).accepts_negative_prices
        assert v(sf.CLIMATE_RULE_VERSIONS, date(2026, 9, 5)).accepts_negative_prices
        assert v(sf.DHW_RULE_VERSIONS, date(2025, 3, 15)) is None
        assert v(sf.DHW_RULE_VERSIONS, date(2025, 3, 16)).interval_hours == 24
        assert v(sf.DHW_RULE_VERSIONS, date(2025, 12, 28)).interval_hours == 24
        assert v(sf.DHW_RULE_VERSIONS, date(2025, 12, 29)).interval_hours == 12
        assert sf.RULE_MIRROR["climate_control"][0]["label"] == "climate_control@fda18eb2"


# ---------------------------------------------------------------------------
# A synthetic world
# ---------------------------------------------------------------------------


class World:
    """Quarter-hour series whose heat-pump energy follows the replayed AppDaemon rules."""

    def __init__(self, *, start: datetime = START, n_days: int = N_DAYS, seed: int = 5) -> None:
        self.starts = [start + timedelta(minutes=15 * i) for i in range(n_days * 96 + 4)]
        local = [s.astimezone(MADRID) for s in self.starts]
        self.local_days = [t.date() for t in local]
        # Trim to whole local days.
        last = self.local_days[-1]
        keep = [i for i, d in enumerate(self.local_days) if d != last]
        self.starts = [self.starts[i] for i in keep]
        self.local_days = [self.local_days[i] for i in keep]
        local = [local[i] for i in keep]
        self.days = contiguous_day_bounds(self.local_days)
        n = len(self.starts)
        rng = np.random.default_rng(seed)
        hour_idx = np.arange(n) // 4
        day_noise = rng.normal(0.0, 0.01, size=hour_idx.max() + 1)
        clock = np.array([t.hour for t in local])
        hourly = 0.12 + 0.06 * np.sin((clock[::4][: hour_idx.max() + 1] - 6) / 24 * 2 * np.pi)
        self.prices = (hourly + day_noise)[hour_idx] + 0.0
        self.temp = 29.0 + 3.0 * np.sin((clock - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.3, n)
        self.clock = clock
        self.heat = np.zeros(n)
        self.cool = np.zeros(n)
        self.dhw = np.zeros(n)
        self.enable = np.ones(n)
        self.min_hours = np.full(n, 8.0)
        self.dhw_enable = np.ones(n)
        self.servers = 0.06 + rng.random(n) * 0.01
        self.usable = np.ones(n, dtype=bool)
        # Heat-pump energy from the replayed rule on the true prices.
        side = self.side()
        view = side.view(n, self.days, published_end=n)
        prices = sf._PriceHours(view, self.days, self.prices)
        for j, (label, lo, hi) in enumerate(self.days):
            schedule, reason, _ = sf._climate_schedule_for_day(
                j=j, view=view, days=self.days, prices=prices, min_hours=8.0
            )
            if schedule is None:
                continue
            slots = np.arange(lo, hi)
            on = schedule.selected[clock[slots]]
            pos = sf.block_positions(schedule.selected)[clock[slots]]
            level = 0.03 * (self.temp[slots] - 20.0) * (1.0 + 0.8 * (pos == 0))
            self.cool[slots] = np.where(on, np.maximum(level, 0.0), 0.0)
        self.hp = self.heat + self.cool + self.dhw
        self.load = self.servers + self.hp

    def side(self, **override) -> lf.SideInputs:
        series = {
            "hp_heat_kwh": self.heat,
            "hp_cool_kwh": self.cool,
            "hp_dhw_kwh": self.dhw,
            "outdoor_temp_c": self.temp,
            "climate_enable": self.enable,
            "climate_min_hours": self.min_hours,
            "dhw_enable": self.dhw_enable,
        }
        series.update(override.pop("series", {}))
        return lf.SideInputs(
            interval_starts_utc=self.starts,
            tz=MADRID,
            prices=override.pop("prices", self.prices),
            series=series,
            source={"kind": "test"},
            omie_eur_per_kwh=override.pop("omie", None),
        )


@pytest.fixture(scope="module")
def world() -> World:
    return World()


def _run(world: World, side: lf.SideInputs, *, cut: int, targets, load=None, **params):
    return lf.run_forecaster(
        "structured",
        load_kwh=world.load if load is None else load,
        usable=world.usable,
        days=world.days,
        cutoff_idx=cut,
        target_days=targets,
        params=params or None,
        side=side,
    )


def _day_index(world: World, day: date) -> int:
    return next(j for j, (label, _lo, _hi) in enumerate(world.days) if label == day)


class TestStructuredForecast:
    def test_the_world_has_a_fall_back_day_with_100_intervals(self, world) -> None:
        j = _day_index(world, date(2025, 10, 26))
        assert world.days[j][2] - world.days[j][1] == 100

    def test_the_replayed_mask_finds_the_energy_and_beats_the_trailing_mean(self, world) -> None:
        side = world.side()
        target = _day_index(world, date(2025, 11, 5))
        lo, hi = world.days[target][1], world.days[target][2]
        result = lf.forecast_each_day_at_its_start(
            "structured",
            load_kwh=world.load,
            usable=world.usable,
            days=world.days,
            target_days=[target],
            side=side,
        )
        lf0 = lf.forecast_each_day_at_its_start(
            "trailing_mean",
            load_kwh=world.load,
            usable=world.usable,
            days=world.days,
            target_days=[target],
        )
        day = result.per_day[0]
        assert day["labels"] == {
            "prices": "published",
            "climate": "structured",
            "dhw": "structured",
        }
        assert day["climate"]["mode"] == "cool"
        assert day["climate"]["rule_version"] == "climate_control@fda18eb2"
        err = np.abs(result.values[lo:hi] - world.load[lo:hi]).sum()
        err0 = np.abs(lf0.values[lo:hi] - world.load[lo:hi]).sum()
        assert err < 0.6 * err0

    def test_without_side_inputs_it_is_lf0_and_says_so(self, world) -> None:
        target = _day_index(world, date(2025, 11, 5))
        kwargs = {
            "load_kwh": world.load,
            "usable": world.usable,
            "days": world.days,
            "cutoff_idx": world.days[target][1],
            "target_days": [target],
        }
        plain = lf.run_forecaster("structured", **kwargs)
        lf0 = lf.run_forecaster("trailing_mean", **kwargs)
        assert np.array_equal(plain.values, lf0.values)
        assert plain.per_day[0]["climate_fallback_reason"] == "no_side_inputs"

    def test_the_provenance_declares_the_side_rule_and_the_cutoff(self) -> None:
        record = lf.get_forecaster("structured").provenance({})
        assert record["intraday_cutoff"] is True
        assert record["side_inputs"] == lf.SIDE_INPUT_RULE
        assert record["fit"] == lf.FIT_FOLD_RULE

    def test_a_disabled_rule_falls_back_to_lf0_with_the_reason(self, world) -> None:
        side = world.side(series={"climate_enable": np.zeros(len(world.starts))})
        target = _day_index(world, date(2025, 11, 5))
        result = _run(world, side, cut=world.days[target][1], targets=[target])
        day = result.per_day[0]
        assert day["climate_source"] == "fallback_lf0"
        assert day["climate_fallback_reason"] == "rule_disabled"

    @pytest.mark.parametrize(
        ("override", "reason"),
        [
            ({"climate_enable": None}, "enable_unknown"),
            ({"climate_min_hours": None}, "min_hours_unknown"),
        ],
    )
    def test_unknown_flags_fall_back(self, world, override, reason) -> None:
        series = {k: np.full(len(world.starts), np.nan) for k in override}
        side = world.side(series=series)
        target = _day_index(world, date(2025, 11, 5))
        result = _run(world, side, cut=world.days[target][1], targets=[target])
        assert result.per_day[0]["climate_fallback_reason"] == reason

    def test_an_unknown_mode_falls_back(self, world) -> None:
        n = len(world.starts)
        side = world.side(series={"hp_cool_kwh": np.zeros(n), "hp_heat_kwh": np.zeros(n)})
        target = _day_index(world, date(2025, 11, 5))
        result = _run(world, side, cut=world.days[target][1], targets=[target])
        assert result.per_day[0]["climate_fallback_reason"] == "mode_no_recent_climate_energy"

    def test_missing_temperature_falls_back(self, world) -> None:
        # The fit (before November) keeps its temperatures; the last three days are gone.
        temp = world.temp.copy()
        temp[world.days[_day_index(world, date(2025, 11, 1))][1] :] = np.nan
        side = world.side(series={"outdoor_temp_c": temp})
        target = _day_index(world, date(2025, 11, 5))
        result = _run(world, side, cut=world.days[target][1], targets=[target])
        assert result.per_day[0]["climate_fallback_reason"] == "temperature_missing"

    def test_missing_prices_fall_back(self, world) -> None:
        side = world.side(prices=np.full(len(world.starts), np.nan))
        target = _day_index(world, date(2025, 11, 5))
        result = _run(world, side, cut=world.days[target][1], targets=[target])
        assert result.per_day[0]["climate_fallback_reason"] in {
            "prices_incomplete",
            "price_history_short",
        }

    def test_no_fitted_model_before_there_is_history(self, world) -> None:
        side = world.side()
        target = _day_index(world, date(2025, 9, 10))
        result = _run(world, side, cut=world.days[target][1], targets=[target])
        assert result.per_day[0]["climate_source"] == "fallback_lf0"
        assert result.per_day[0]["climate_fallback_reason"] in {
            "model_not_fitted",
            "mode_history_short",
            "price_history_short",
        }

    def test_a_fall_back_day_is_forecast_with_100_intervals(self, world) -> None:
        side = world.side()
        target = _day_index(world, date(2025, 10, 26))
        result = lf.forecast_each_day_at_its_start(
            "structured",
            load_kwh=world.load,
            usable=world.usable,
            days=world.days,
            target_days=[target],
            side=side,
        )
        lo, hi = world.days[target][1], world.days[target][2]
        assert hi - lo == 100
        assert np.isfinite(result.values[lo:hi]).all()
        assert result.per_day[0]["labels"]["climate"] == "structured"

    def test_the_residual_forecaster_is_a_param(self, world) -> None:
        side = world.side()
        target = _day_index(world, date(2025, 11, 5))
        base = _run(world, side, cut=world.days[target][1], targets=[target])
        other = _run(
            world,
            side,
            cut=world.days[target][1],
            targets=[target],
            residual_forecaster="trailing_median",
        )
        assert other.per_day[0]["residual_forecaster"] == "trailing_median"
        assert other.provenance["params"]["residual_forecaster"] == "trailing_median"
        assert not np.array_equal(base.values, other.values)

    def test_bad_residual_forecasters_and_outlooks_are_refused(self, world) -> None:
        side = world.side()
        target = _day_index(world, date(2025, 11, 5))
        for params in (
            {"residual_forecaster": "perfect"},
            {"residual_forecaster": "structured"},
            {"price_outlook": "exact"},
        ):
            with pytest.raises(ValueError):
                _run(world, side, cut=world.days[target][1], targets=[target], **params)


class TestPriceProvenance:
    """14:00 plans tomorrow on estimated prices, 20:30 on published ones."""

    def _event_cut(self, world, day: date, hh: int, mm: int) -> int:
        j = _day_index(world, day)
        lo = world.days[j][1]
        return lo + (hh * 60 + mm) // 15 if hh >= 3 else lo + (hh * 60 + mm) // 15

    def test_before_publication_tomorrow_is_estimated_and_after_it_is_published(
        self, world
    ) -> None:
        side = world.side()
        today = _day_index(world, date(2025, 11, 5))
        targets = [today, today + 1]
        afternoon = _run(
            world, side, cut=self._event_cut(world, date(2025, 11, 5), 14, 0), targets=targets
        )
        evening = _run(
            world, side, cut=self._event_cut(world, date(2025, 11, 5), 20, 30), targets=targets
        )
        assert [d["labels"]["prices"] for d in afternoon.per_day] == ["published", "estimated"]
        assert [d["labels"]["prices"] for d in evening.per_day] == ["published", "published"]

    def test_published_only_leaves_tomorrow_unavailable_and_falls_back(self, world) -> None:
        side = world.side()
        today = _day_index(world, date(2025, 11, 5))
        cut = self._event_cut(world, date(2025, 11, 5), 14, 0)
        result = _run(
            world, side, cut=cut, targets=[today, today + 1], price_outlook="published_only"
        )
        tomorrow = result.per_day[1]
        assert tomorrow["labels"]["prices"] == "unavailable"
        assert tomorrow["climate_fallback_reason"] == "prices_unavailable"
        assert tomorrow["dhw_fallback_reason"] == "prices_unavailable"
        assert result.per_day[0]["labels"]["climate"] == "structured"

    def test_the_rolling_provider_records_what_each_event_used(self, world) -> None:
        from battery_optimizer.rolling_horizon import CausalForecastProvider, compute_rolling_events

        side = world.side()
        start = world.days[_day_index(world, date(2025, 11, 5))][1]
        events = compute_rolling_events(
            world.starts,
            world.local_days,
            eval_start_idx=start,
            eval_end_idx=len(world.starts),
            tz=MADRID,
        )
        event = next(ev for ev in events if ev.kind == "afternoon")
        provider = CausalForecastProvider(
            world.load,
            world.local_days,
            world.usable,
            world.starts,
            tz=MADRID,
            load_forecaster="structured",
            side=side,
        )
        _forecast, meta = provider.forecast_for_event(event)
        assert meta["cutoff_idx"] == event.start_idx  # the declared intraday cutoff
        assert meta["horizon_forecaster_labels"][0]["prices"] == "estimated"
        plain = CausalForecastProvider(
            world.load, world.local_days, world.usable, world.starts, tz=MADRID
        )
        _f, plain_meta = plain.forecast_for_event(event)
        assert "horizon_forecaster_labels" not in plain_meta


# ---------------------------------------------------------------------------
# Leakage: what the decision must not know
# ---------------------------------------------------------------------------


def _scramble(values: np.ndarray, start: int) -> np.ndarray:
    out = values.copy()
    pattern = np.array([1e9, -5e8, np.nan, np.inf, 0.0, 3e5])
    out[start:] = np.resize(pattern, out.size - start)
    return out


class TestLeakage:
    def _decision(self, world, hh: int, mm: int):
        j = _day_index(world, date(2025, 11, 5))
        cut = world.days[j][1] + (hh * 60 + mm) // 15
        return cut, [j, j + 1]

    def _base(self, world, cut, targets):
        return _run(world, world.side(), cut=cut, targets=targets)

    @pytest.mark.parametrize("clock", [(0, 0), (14, 0), (20, 30)])
    def test_future_load_and_measured_side_series_cannot_move_the_forecast(
        self, world, clock
    ) -> None:
        cut, targets = self._decision(world, *clock)
        base = self._base(world, cut, targets)
        series = {name: _scramble(world.side().series[name], cut) for name in sf.SERIES}
        scrambled = world.side(series=series)
        moved = _run(world, scrambled, cut=cut, targets=targets, load=_scramble(world.load, cut))
        assert np.array_equal(base.values, moved.values)
        assert base.per_day == moved.per_day

    @pytest.mark.parametrize("clock", [(0, 0), (14, 0), (20, 14), (20, 30)])
    def test_unpublished_prices_cannot_move_the_forecast(self, world, clock) -> None:
        cut, targets = self._decision(world, *clock)
        base = self._base(world, cut, targets)
        end = lf.SideInputs.view(world.side(), cut, world.days).published_end_idx
        moved = _run(
            world, world.side(prices=_scramble(world.prices, end)), cut=cut, targets=targets
        )
        assert np.array_equal(base.values, moved.values)
        assert base.per_day == moved.per_day

    def test_publication_moves_at_20_15_local(self, world) -> None:
        j = _day_index(world, date(2025, 11, 5))
        before = world.side().view(world.days[j][1] + 80, world.days)  # 20:00
        after = world.side().view(world.days[j][1] + 81, world.days)  # 20:15
        assert before.published_end_idx == world.days[j][2]
        assert after.published_end_idx == world.days[j + 1][2]

    def test_published_prices_do_move_it(self, world) -> None:
        cut, targets = self._decision(world, 20, 30)
        base = self._base(world, cut, targets)
        j = targets[1]
        prices = world.prices.copy()
        prices[world.days[j][1] : world.days[j][2]] = np.roll(
            prices[world.days[j][1] : world.days[j][2]], 24
        )
        moved = _run(world, world.side(prices=prices), cut=cut, targets=targets)
        assert not np.array_equal(base.values, moved.values)

    def test_measured_history_does_move_it(self, world) -> None:
        cut, targets = self._decision(world, 14, 0)
        base = self._base(world, cut, targets)
        temp = world.temp.copy()
        temp[cut - 400 : cut] += 6.0
        moved = _run(world, world.side(series={"outdoor_temp_c": temp}), cut=cut, targets=targets)
        assert not np.array_equal(base.values, moved.values)

    def test_the_fold_fit_never_sees_its_own_fold(self, world) -> None:
        j = _day_index(world, date(2025, 11, 5))
        fold_start = _day_index(world, date(2025, 11, 1))
        lo = world.days[fold_start][1]
        cut = world.days[j][1]
        base = _run(world, world.side(), cut=cut, targets=[j])
        # Everything from the fold start up to the decision is replaced for the FIT; the
        # decision itself keeps its own as-of history, so only the fit can differ.
        side = world.side()
        view = side.view(lo, world.days, published_end=lo)
        state = lf.fold_state("structured", {}, world.load, world.usable, world.days, cut, side)
        assert state["fit_cutoff_idx"] == lo
        assert max(len(a) for a in view.series.values()) == lo
        assert view.published_prices.size == lo
        perturbed = world.side(
            series={name: _scramble(world.side().series[name], lo) for name in sf.SERIES},
            prices=_scramble(world.prices, lo),
        )
        assert (
            lf.fold_state(
                "structured",
                {},
                _scramble(world.load, lo),
                _scramble(world.usable.astype(float), lo) > 0,
                world.days,
                cut,
                perturbed,
            )["fit_cutoff_idx"]
            == lo
        )
        for mode in ("heat", "cool"):
            a = state["models"].get(mode)
            b = lf.fold_state(
                "structured", {}, world.load, world.usable, world.days, cut, perturbed
            )["models"].get(mode)
            assert (a is None) == (b is None)
            if a is not None:
                assert np.array_equal(a.beta, b.beta)
        assert base.provenance["fit_cutoff_idx"] == lo


class TestSeam:
    def test_a_side_input_forecaster_is_never_handed_side_data_unless_declared(self, world) -> None:
        # trailing_mean ignores side data entirely.
        target = _day_index(world, date(2025, 11, 5))
        a = lf.run_forecaster(
            "trailing_mean",
            load_kwh=world.load,
            usable=world.usable,
            days=world.days,
            cutoff_idx=world.days[target][1],
            target_days=[target],
            side=world.side(),
        )
        b = lf.run_forecaster(
            "trailing_mean",
            load_kwh=world.load,
            usable=world.usable,
            days=world.days,
            cutoff_idx=world.days[target][1],
            target_days=[target],
        )
        assert np.array_equal(a.values, b.values)

    def test_a_misaligned_side_is_refused(self, world) -> None:
        small = World(n_days=10)
        with pytest.raises(ValueError, match="side inputs cover"):
            _run(world, small.side(), cut=world.days[40][1], targets=[40])

    def test_side_inputs_need_a_uniform_hourly_grid(self) -> None:
        starts = [START + timedelta(minutes=15 * i) for i in range(8)]
        starts[5] += timedelta(minutes=1)
        with pytest.raises(ValueError, match="uniform"):
            lf.SideInputs(
                interval_starts_utc=starts, tz=MADRID, prices=np.zeros(8), series={"a": np.zeros(8)}
            )

    def test_the_fingerprint_follows_the_content(self, world) -> None:
        a = world.side()
        b = world.side()
        assert a.fingerprint == b.fingerprint
        assert world.side(series={"outdoor_temp_c": world.temp + 1.0}).fingerprint != a.fingerprint

    def test_the_service_refuses_a_forecaster_that_needs_side_inputs(self, tmp_path) -> None:
        from test_service_inputs import write_service_config

        from battery_optimizer.service.config import load_service_config

        with pytest.raises(ValueError, match="side inputs"):
            load_service_config(
                write_service_config(tmp_path, **{"planning.load_forecaster": "structured"})
            )

    def test_the_calendar_of_a_fall_back_day_keeps_the_first_pass(self, world) -> None:
        j = _day_index(world, date(2025, 10, 26))
        view = world.side().view(world.days[j][1], world.days)
        first = int(view.first_real_hour[j, 2])
        lo = world.days[j][1]
        assert first == lo // 4 + 2
        assert view.first_real_hour[j].min() >= 0  # every clock hour exists on a fall-back day
        k = _day_index(world, date(2025, 10, 25))
        assert (view.first_real_hour[k] >= 0).all()


class TestRecentScale:
    def test_the_level_follows_the_last_days_and_can_be_switched_off(self, world) -> None:
        target = _day_index(world, date(2025, 11, 12))
        cut = world.days[target][1]
        cool = world.cool.copy()
        cool[world.days[target - 6][1] : cut] *= 2.0  # the unit suddenly draws twice as much
        side = world.side(series={"hp_cool_kwh": cool})
        calibrated = _run(world, side, cut=cut, targets=[target])
        plain = _run(world, side, cut=cut, targets=[target], calibration_days=0)
        scale = calibrated.per_day[0]["climate"]["recent_scale"]
        assert 1.5 < scale <= 2.0
        assert plain.per_day[0]["climate"]["recent_scale"] == 1.0
        lo, hi = world.days[target][1], world.days[target][2]
        assert calibrated.values[lo:hi].sum() > plain.values[lo:hi].sum()

    def test_without_two_enabled_days_the_scale_is_one(self, world) -> None:
        side = world.side(series={"climate_enable": np.zeros(len(world.starts))})
        scale = sf._recent_scale(
            model=None,
            mode="cool",
            view=side.view(world.days[40][1], world.days),
            days=world.days,
            prices=None,
            hourly_temp=np.zeros(1),
            cut=world.days[40][1],
            window=7,
        )
        assert scale == (1.0, 0)
