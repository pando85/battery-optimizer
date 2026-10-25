"""P2 statistical load forecasters: median, trimmed mean, EWMA, day type, level, blend.

The generic leakage tests in ``test_load_forecast.py`` already cover every registered
forecaster on a 40-day series.  That series is too short for ``blend`` (no same-season
last year) and too regular to exercise the day-type search, so the leakage perturbation is
repeated here on a series long enough to reach every branch, next to a check that the
perturbation does change the forecast when it is applied to the *history* (so a pass is
not vacuous).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from test_load_forecast import _absurd, _afternoon_event, _provider, _series

from battery_optimizer import load_forecast as lf
from battery_optimizer.policies import contiguous_day_bounds

P2_NAMES = (
    "trailing_median",
    "trailing_trimmed_mean",
    "ewma",
    "daytype",
    "level_adjusted",
    "blend",
)
SLOTS = 96


def _flat_days(first: date, n_days: int) -> list[tuple[date, int, int]]:
    return [(first + timedelta(days=d), d * SLOTS, (d + 1) * SLOTS) for d in range(n_days)]


def _run(name, load, usable, days, target, *, cutoff=None, **params):
    cut = days[target][1] if cutoff is None else cutoff
    result = lf.run_forecaster(
        name,
        load_kwh=load,
        usable=usable,
        days=days,
        cutoff_idx=cut,
        target_days=[target],
        params=params or None,
    )
    lo, hi = days[target][1], days[target][2]
    return result.values[lo:hi], result.per_day[0]


def _per_day_levels(levels: list[float]) -> np.ndarray:
    return np.repeat(np.asarray(levels, dtype=float), SLOTS)


def _ones(load: np.ndarray) -> np.ndarray:
    return np.ones(load.size, dtype=bool)


class TestRegistration:
    def test_all_candidates_are_registered_non_oracle_with_documented_defaults(self) -> None:
        for name in P2_NAMES:
            assert name in lf.forecaster_names(include_oracle=False)
        assert lf.get_forecaster("trailing_median").defaults == {"window_days": 28}
        assert lf.get_forecaster("trailing_trimmed_mean").defaults == {
            "window_days": 28,
            "trim": 0.1,
        }
        assert lf.get_forecaster("ewma").defaults == {"half_life_days": 7.0, "window_days": 56}
        assert lf.get_forecaster("daytype").defaults == {
            "n_days": 6,
            "window_days": 84,
            "subdivision": None,
        }
        assert lf.get_forecaster("level_adjusted").defaults == {
            "window_days": 28,
            "lo": 0.7,
            "hi": 1.4,
            "min_slots": 24,
        }
        assert lf.get_forecaster("blend").defaults["w"] == 0.5

    def test_only_level_adjusted_declares_an_intraday_cutoff(self) -> None:
        declared = {n for n in P2_NAMES if lf.get_forecaster(n).intraday_cutoff}
        assert declared == {"level_adjusted"}

    @pytest.mark.parametrize(
        ("name", "params"),
        [
            ("trailing_median", {"window_days": 0}),
            ("trailing_trimmed_mean", {"trim": 0.5}),
            ("trailing_trimmed_mean", {"trim": -0.1}),
            ("ewma", {"half_life_days": 0}),
            ("ewma", {"window_days": 0}),
            ("daytype", {"n_days": 0}),
            ("daytype", {"subdivision": "XX"}),
            ("level_adjusted", {"lo": 1.5, "hi": 1.0}),
            ("blend", {"w": 1.5}),
        ],
    )
    def test_bad_params_are_refused(self, name: str, params: dict) -> None:
        load, usable, days = _series()
        with pytest.raises(ValueError):
            _run(name, load, usable, days, 30, **params)


class TestEveryDayShape:
    @pytest.mark.parametrize("name", P2_NAMES)
    def test_every_target_day_has_its_own_length_across_the_dst_days(self, name: str) -> None:
        load, usable, days = _series()
        lengths = {hi - lo for _l, lo, hi in days}
        assert lengths >= {96, 100}  # the window crosses the fall-back day
        result = lf.run_forecaster(
            name,
            load_kwh=load,
            usable=usable,
            days=days,
            cutoff_idx=days[20][1],
            target_days=range(20, len(days)),
        )
        assert np.isfinite(result.values).all()
        for diag in result.per_day:
            assert diag["forecast_fallback_rule"]
            assert diag["slots_zero_no_history"] == 0

    @pytest.mark.parametrize("name", P2_NAMES)
    def test_no_usable_history_is_flagged_and_forecasts_zero_not_a_guess(self, name: str) -> None:
        load, _usable, days = _series()
        nothing = np.zeros(load.size, dtype=bool)
        values, diag = _run(name, load, nothing, days, 30)
        assert np.all(values == 0.0)
        assert diag["fallback"] == "zero_no_history"
        assert diag["slots_zero_no_history"] == values.size
        assert diag["slots_pooled_fallback"] == 0

    @pytest.mark.parametrize("name", P2_NAMES)
    def test_the_first_day_has_no_history_and_does_not_fail(self, name: str) -> None:
        load, usable, days = _series()
        values, diag = _run(name, load, usable, days, 0, cutoff=0)
        assert np.all(values == 0.0)
        assert diag["slots_zero_no_history"] == values.size

    @pytest.mark.parametrize("name", P2_NAMES)
    def test_a_non_finite_usable_value_is_refused_like_lf0(self, name: str) -> None:
        load, usable, days = _series()
        load = load.copy()
        usable = usable.copy()
        load[days[29][1] + 5] = np.nan
        usable[days[29][1] + 5] = True
        with pytest.raises(ValueError, match="non-finite"):
            _run(name, load, usable, days, 30)

    @pytest.mark.parametrize("name", P2_NAMES)
    def test_history_before_the_cutoff_moves_the_forecast(self, name: str) -> None:
        """Non-vacuity of the leakage tests: the forecaster does read the history."""
        load, usable, days = _series()
        cut = days[30][1] + 57 if name == "level_adjusted" else days[30][1]
        base, _ = _run(name, load, usable, days, 31, cutoff=cut)
        bumped = load.copy()
        bumped[cut - 2 * SLOTS : cut] += 0.5
        moved, _ = _run(name, bumped, usable, days, 31, cutoff=cut)
        assert not np.array_equal(base, moved)


class TestRobustProfiles:
    def test_median_of_one_day_is_the_trailing_mean_of_one_day(self) -> None:
        load, usable, days = _series()
        a = lf.run_forecaster(
            "trailing_median",
            load_kwh=load,
            usable=usable,
            days=days,
            cutoff_idx=days[10][1],
            target_days=[10],
            params={"window_days": 1},
        )
        b = lf.run_forecaster(
            "trailing_mean",
            load_kwh=load,
            usable=usable,
            days=days,
            cutoff_idx=days[10][1],
            target_days=[10],
            params={"window_days": 1},
        )
        assert np.array_equal(a.values, b.values)

    def test_trimmed_mean_without_trimming_is_the_trailing_mean(self) -> None:
        load, usable, days = _series()
        a, _ = _run("trailing_trimmed_mean", load, usable, days, 30, trim=0.0)
        b, _ = _run("trailing_mean", load, usable, days, 30)
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12)

    def test_a_spike_moves_the_mean_but_not_the_median_or_the_trimmed_mean(self) -> None:
        levels = [1.0] * 10 + [0.0]
        load = _per_day_levels(levels)
        load[3 * SLOTS + 40] = 100.0
        days = _flat_days(date(2025, 3, 3), 11)
        usable = _ones(load)
        mean, _ = _run("trailing_mean", load, usable, days, 10, window_days=10)
        median, _ = _run("trailing_median", load, usable, days, 10, window_days=10)
        trimmed, diag = _run("trailing_trimmed_mean", load, usable, days, 10, window_days=10)
        assert mean[40] == pytest.approx(10.9)
        assert median[40] == 1.0
        assert trimmed[40] == 1.0  # floor(0.1 * 10) = 1 dropped at each end
        assert diag["trim"] == 0.1
        assert np.all(median[:40] == 1.0)

    def test_even_count_median_averages_the_middle_pair(self) -> None:
        load = _per_day_levels([1.0, 2.0, 4.0, 10.0, 0.0])
        days = _flat_days(date(2025, 3, 3), 5)
        median, _ = _run("trailing_median", load, _ones(load), days, 4, window_days=4)
        assert np.all(median == 3.0)

    @pytest.mark.parametrize("name", ["trailing_median", "trailing_trimmed_mean", "ewma"])
    def test_unusable_slots_are_unknown_not_zero(self, name: str) -> None:
        load = _per_day_levels([2.0, 2.0, 2.0, 2.0, 0.0])
        usable = _ones(load)
        for day in (0, 1):
            usable[day * SLOTS + 7] = False  # slot 7 unknown on two of the four days
        for day in range(4):
            usable[day * SLOTS + 9] = False  # slot 9 unknown on every day
        days = _flat_days(date(2025, 3, 3), 5)
        values, diag = _run(name, load, usable, days, 4, window_days=4)
        assert values[7] == pytest.approx(2.0)  # the two known days, not a mean with zeros
        assert values[9] == pytest.approx(2.0)  # pooled fallback of the window, not 0
        assert diag["slots_pooled_fallback"] == 1
        assert diag["fallback"] == "pooled_window_mean"
        assert diag["window_days_with_usable_data"] == 4


class TestEwma:
    def test_weights_follow_the_half_life_by_age_in_days(self) -> None:
        load = _per_day_levels([4.0, 1.0, 0.0])
        days = _flat_days(date(2025, 3, 3), 3)
        values, diag = _run("ewma", load, _ones(load), days, 2, half_life_days=1.0)
        # age 1 -> weight 1/2 (level 1.0), age 2 -> weight 1/4 (level 4.0)
        assert np.all(values == pytest.approx((0.5 * 1.0 + 0.25 * 4.0) / 0.75))
        assert diag["half_life_days"] == 1.0

    def test_a_very_long_half_life_approaches_the_trailing_mean(self) -> None:
        load, usable, days = _series()
        a, _ = _run("ewma", load, usable, days, 30, half_life_days=1e9, window_days=28)
        b, _ = _run("trailing_mean", load, usable, days, 30)
        assert np.allclose(a, b, rtol=1e-6)

    def test_the_window_caps_the_history(self) -> None:
        load = _per_day_levels([100.0, 1.0, 1.0, 0.0])
        days = _flat_days(date(2025, 3, 3), 4)
        values, _ = _run("ewma", load, _ones(load), days, 3, half_life_days=50.0, window_days=2)
        assert np.all(values == pytest.approx(1.0))

    def test_the_age_is_measured_from_the_target_not_from_the_cutoff(self) -> None:
        # A day-ahead cut at the target's own midnight and a rolling cut one day earlier
        # differ only in the history they hold, never in how a day is weighted.
        load = _per_day_levels([4.0, 1.0, 9.0, 0.0])
        days = _flat_days(date(2025, 3, 3), 4)
        values, _ = _run("ewma", load, _ones(load), days, 3, cutoff=days[2][1], half_life_days=1.0)
        # day 2 is not in the history; age 2 (level 1.0) -> 1/4, age 3 (level 4.0) -> 1/8
        assert np.all(values == pytest.approx((0.25 * 1.0 + 0.125 * 4.0) / 0.375))


class TestDaytype:
    def test_classes_follow_weekday_saturday_and_sunday_or_holiday(self) -> None:
        assert lf.daytype_of(date(2025, 3, 3)) == "weekday"  # Monday
        assert lf.daytype_of(date(2025, 3, 8)) == "saturday"
        assert lf.daytype_of(date(2025, 3, 9)) == "sunday_holiday"
        assert lf.daytype_of(date(2025, 12, 25)) == "sunday_holiday"  # a Thursday holiday
        assert lf.daytype_of(date(2025, 8, 15)) == "sunday_holiday"
        # A holiday on a Saturday is a holiday (2025-11-01).
        assert date(2025, 11, 1).weekday() == 5
        assert lf.daytype_of(date(2025, 11, 1)) == "sunday_holiday"

    def test_the_subdivision_adds_its_own_holidays_only_when_asked(self) -> None:
        assert lf.daytype_of(date(2025, 5, 2)) == "weekday"
        assert lf.daytype_of(date(2025, 5, 2), "MD") == "sunday_holiday"

    def _weekly(self, *, holidays_as_sundays: bool = True, weeks: int = 12):
        # Monday 2025-03-03 ...; levels: weekday 1, Saturday 5, Sunday (and holiday) 9
        days = _flat_days(date(2025, 3, 3), weeks * 7)
        levels = []
        for label, _lo, _hi in days:
            kind = (
                lf.daytype_of(label)
                if holidays_as_sundays
                else {5: "saturday", 6: "sunday_holiday"}.get(label.weekday(), "weekday")
            )
            levels.append({"weekday": 1.0, "saturday": 5.0, "sunday_holiday": 9.0}[kind])
        return _per_day_levels(levels), days

    @pytest.mark.parametrize(
        ("target", "expected", "kind"),
        [
            (date(2025, 5, 7), 1.0, "weekday"),
            (date(2025, 5, 10), 5.0, "saturday"),
            (date(2025, 5, 11), 9.0, "sunday_holiday"),
            (date(2025, 5, 1), 9.0, "sunday_holiday"),  # Labour Day, a Thursday
        ],
    )
    def test_each_day_type_uses_only_its_own_days(self, target, expected, kind) -> None:
        load, days = self._weekly()
        j = next(i for i, d in enumerate(days) if d[0] == target)
        values, diag = _run("daytype", load, _ones(load), days, j)
        assert np.all(values == expected)
        assert diag["daytype"] == kind
        assert diag["daytype_days_found"] == 6

    def test_past_holidays_join_the_sunday_pool_and_n_days_caps_it(self) -> None:
        # Loads follow the weekday only, so Good Friday (2025-04-18) keeps the weekday level.
        load, days = self._weekly(holidays_as_sundays=False)
        j = next(i for i, d in enumerate(days) if d[0] == date(2025, 4, 20))  # a Sunday
        values, diag = _run("daytype", load, _ones(load), days, j, n_days=3)
        assert diag["daytype_days_found"] == 3
        # The last three Sunday-or-holiday days: Good Friday (1.0), Sundays 04-13 and 04-06 (9.0).
        assert np.all(values == pytest.approx((1.0 + 9.0 + 9.0) / 3.0))

    def test_the_search_stops_at_the_window(self) -> None:
        load, days = self._weekly()
        j = next(i for i, d in enumerate(days) if d[0] == date(2025, 5, 11))
        _v, diag = _run("daytype", load, _ones(load), days, j, window_days=7)
        assert diag["daytype_days_found"] == 1  # 05-04 (05-01 is eight days back)
        _v, diag = _run("daytype", load, _ones(load), days, j, window_days=14)
        assert diag["daytype_days_found"] == 3  # 05-04, 05-01 (holiday) and 04-27

    def test_days_without_usable_data_do_not_count_towards_n_days(self) -> None:
        load, days = self._weekly()
        usable = _ones(load)
        j = next(i for i, d in enumerate(days) if d[0] == date(2025, 5, 11))
        usable[(j - 7) * SLOTS : (j - 6) * SLOTS] = False  # last Sunday is a gap
        values, diag = _run("daytype", load, usable, days, j, n_days=2)
        assert np.all(values == 9.0)
        assert diag["daytype_days_found"] == 2
        assert diag["window_days_with_usable_data"] == 2


class TestLevelAdjusted:
    def _flat(self, scale_yesterday: float, observed_days: int = 40):
        """*observed_days* days, the last at *scale_yesterday*, then a target day at level 1."""
        days = _flat_days(date(2025, 3, 3), observed_days + 1)
        load = _per_day_levels([1.0] * (observed_days - 1) + [scale_yesterday, 1.0])
        return load, days, observed_days  # target index

    def _trailing(self, load, days, j, cut):
        values, _ = _run("trailing_mean", load, _ones(load), days, j, cutoff=cut)
        return values

    def test_the_factor_is_the_observed_to_profile_ratio_of_the_last_24_hours(self) -> None:
        load, days, j = self._flat(1.2)
        cut = j * SLOTS  # local midnight: the last 24 h are the whole previous day
        values, diag = _run("level_adjusted", load, _ones(load), days, j, cutoff=cut)
        assert diag["level_factor"] == pytest.approx(1.2)
        assert diag["level_clipped"] is False
        assert np.allclose(values, self._trailing(load, days, j, cut) * 1.2)

    @pytest.mark.parametrize(("scale", "clipped_to"), [(3.0, 1.4), (0.1, 0.7)])
    def test_the_factor_is_clipped(self, scale: float, clipped_to: float) -> None:
        load, days, j = self._flat(scale)
        cut = j * SLOTS
        values, diag = _run("level_adjusted", load, _ones(load), days, j, cutoff=cut)
        assert diag["level_factor"] == clipped_to
        assert diag["level_clipped"] is True
        assert diag["level_factor_raw"] == pytest.approx(scale)
        assert np.allclose(values, self._trailing(load, days, j, cut) * clipped_to)

    def test_custom_clip_bounds_apply(self) -> None:
        load, days, j = self._flat(3.0)
        _v, diag = _run("level_adjusted", load, _ones(load), days, j, cutoff=j * SLOTS, hi=2.0)
        assert diag["level_factor"] == 2.0

    def test_a_mid_day_cut_uses_todays_observed_slots(self) -> None:
        load, days, j = self._flat(1.0)
        cut = j * SLOTS + 56  # 14:00
        load = load.copy()
        load[j * SLOTS : cut] = 1.3  # this morning ran 30 % high
        values, diag = _run("level_adjusted", load, _ones(load), days, j, cutoff=cut)
        expected = (40 * 1.0 + 56 * 1.3) / 96  # 40 slots of yesterday afternoon, 56 of today
        assert diag["level_slots_observed"] == SLOTS
        assert diag["level_factor"] == pytest.approx(expected)
        assert np.allclose(values, self._trailing(load, days, j, cut) * expected)

    def test_too_few_observed_slots_leave_the_profile_unscaled(self) -> None:
        load, days, j = self._flat(3.0)
        usable = np.zeros(load.size, dtype=bool)
        usable[: j * SLOTS - 10] = True
        usable[j * SLOTS - 10 : j * SLOTS] = True
        usable[(j - 1) * SLOTS : j * SLOTS - 10] = False  # only 10 observed slots in 24 h
        _v, diag = _run("level_adjusted", load, usable, days, j, cutoff=j * SLOTS)
        assert diag["level_slots_observed"] == 10
        assert diag["level_factor"] == 1.0
        assert diag["level_factor_raw"] is None

    def test_unusable_observed_slots_are_not_counted_as_zero(self) -> None:
        load, days, j = self._flat(1.0)
        usable = _ones(load)
        cut = j * SLOTS + 56
        usable[j * SLOTS : j * SLOTS + 20] = False  # a gap this morning
        values, diag = _run("level_adjusted", load, usable, days, j, cutoff=cut)
        assert diag["level_factor"] == pytest.approx(1.0)
        assert diag["level_slots_observed"] == SLOTS - 20
        assert np.allclose(values, 1.0)

    def test_the_declared_cutoff_is_the_event_instant(self) -> None:
        load, usable, days = _series()
        event, starts, local_days = _afternoon_event(days, load.size)
        day_start = next(lo for label, lo, _hi in days if label == event.local_day)
        provider = _provider("level_adjusted", load, usable, starts, local_days)
        base, meta = provider.forecast_for_event(event)
        assert meta["cutoff_idx"] == event.start_idx > day_start
        plan = slice(event.start_idx, event.plan_end_idx)
        bumped = load.copy()
        bumped[day_start : event.start_idx] += 5.0  # today's morning, observed before 14:00
        moved, _ = _provider(
            "level_adjusted", bumped, usable, starts, local_days
        ).forecast_for_event(event)
        assert not np.array_equal(base[plan], moved[plan])
        # ... but nothing from the event instant on can.
        late = load.copy()
        late[event.start_idx :] += 5.0
        same, _ = _provider("level_adjusted", late, usable, starts, local_days).forecast_for_event(
            event
        )
        assert np.array_equal(base[plan], same[plan])


def _year_series(seed: int = 11, n_days: int = 440):
    """Hourly-shaped weekly load over ~14.5 months crossing both DST changes."""
    madrid = ZoneInfo("Europe/Madrid")
    first = datetime(2024, 9, 30, 22, tzinfo=UTC)  # 2024-10-01 00:00 CEST
    starts = [first + timedelta(minutes=15 * i) for i in range(n_days * 96)]
    local = [s.astimezone(madrid) for s in starts]
    local_days = [s.date() for s in local]
    rng = np.random.default_rng(seed)
    hour = np.array([s.hour + s.minute / 60 for s in local])
    weekday = np.array([s.weekday() for s in local])
    doy = np.array([s.timetuple().tm_yday for s in local])
    load = (
        0.2
        + 0.15 * np.sin(2 * np.pi * (hour - 7) / 24) ** 2
        + 0.05 * (weekday >= 5)
        + 0.1 * np.cos(2 * np.pi * doy / 365.0)
        + rng.random(len(starts)) * 0.05
    )
    usable = rng.random(len(starts)) > 0.03
    return load, usable, contiguous_day_bounds(local_days)


class TestBlend:
    def _levels(self, ly: float, recent: float, total: int = 420):
        """Constant level per era: last year's window at *ly*, the rest at *recent*."""
        days = _flat_days(date(2024, 1, 1), total)
        levels = [recent] * total
        target = date(2025, 2, 20)
        centre = date(2024, 2, 20)
        for d, (label, _lo, _hi) in enumerate(days):
            if abs((label - centre).days) <= 14:
                levels[d] = ly
        load = _per_day_levels(levels)
        j = next(i for i, d in enumerate(days) if d[0] == target)
        return load, days, j

    @pytest.mark.parametrize("w", [0.5, 0.75, 0.0, 1.0])
    def test_the_blend_weights_recent_and_last_year(self, w: float) -> None:
        load, days, j = self._levels(ly=4.0, recent=2.0)
        values, diag = _run("blend", load, _ones(load), days, j, w=w)
        assert np.all(values == pytest.approx(w * 2.0 + (1 - w) * 4.0))
        assert diag["blend_source"] == "blend"
        assert diag["last_year_days_with_usable_data"] == 29

    def test_it_falls_back_to_trailing_when_last_year_is_unavailable(self) -> None:
        load, days, j = self._levels(ly=4.0, recent=2.0)
        trailing, _ = _run("trailing_mean", load, _ones(load), days, j)
        # Start the history a month before the target: no last-year days exist.
        sub_days = days[j - 40 : j + 1]
        base = sub_days[0][1]
        sub_days = [(label, lo - base, hi - base) for label, lo, hi in sub_days]
        sub_load = load[base : sub_days[-1][2] + base]
        values, diag = _run("blend", sub_load, _ones(sub_load), sub_days, len(sub_days) - 1)
        assert np.array_equal(values, trailing)
        assert diag["blend_source"] == "trailing_only"
        assert diag["last_year_days_with_usable_data"] == 0

    def test_a_thin_last_year_window_is_not_trusted(self) -> None:
        load, days, j = self._levels(ly=4.0, recent=2.0)
        usable = _ones(load)
        centre = next(i for i, d in enumerate(days) if d[0] == date(2024, 2, 20))
        keep_days = {centre, centre + 1, centre + 2}
        for d in range(centre - 14, centre + 15):
            if d not in keep_days:
                usable[d * SLOTS : (d + 1) * SLOTS] = False
        values, diag = _run("blend", load, usable, days, j)
        assert np.all(values == pytest.approx(2.0))
        assert diag["blend_source"] == "trailing_only"
        assert diag["last_year_days_with_usable_data"] == 3
        values, diag = _run("blend", load, usable, days, j, min_last_year_days=3)
        assert diag["blend_source"] == "blend"

    def test_slots_unknown_last_year_keep_the_trailing_value(self) -> None:
        load, days, j = self._levels(ly=4.0, recent=2.0)
        usable = _ones(load)
        centre = next(i for i, d in enumerate(days) if d[0] == date(2024, 2, 20))
        for d in range(centre - 14, centre + 15):
            usable[d * SLOTS + 10] = False  # slot 10 unknown on all 29 days
        values, diag = _run("blend", load, usable, days, j)
        assert values[10] == pytest.approx(2.0)  # trailing only: not blended with zero
        assert values[11] == pytest.approx(3.0)
        assert diag["last_year_slots_blended"] == SLOTS - 1

    def test_without_recent_history_last_year_alone_is_used(self) -> None:
        load, days, j = self._levels(ly=4.0, recent=2.0)
        usable = _ones(load)
        usable[(j - 40) * SLOTS : j * SLOTS] = False  # a long outage right before the target
        values, diag = _run("blend", load, usable, days, j)
        assert np.all(values == pytest.approx(4.0))
        assert diag["blend_source"] == "last_year_only"

    def test_29_february_looks_back_to_28_february(self) -> None:
        assert lf._one_year_earlier(date(2024, 2, 29)) == date(2023, 2, 28)
        assert lf._one_year_earlier(date(2025, 3, 1)) == date(2024, 3, 1)


@pytest.mark.parametrize("name", P2_NAMES)
class TestLeakageOnAYearOfData:
    """The generic leakage test, on a series long enough to reach last year's window."""

    def _cases(self, name: str):
        load, usable, days = _year_series()
        j = 400
        cuts = [days[j][1]]
        if lf.get_forecaster(name).intraday_cutoff:
            cuts.append(days[j][1] + 57)
        return load, usable, days, j, cuts

    def test_post_cutoff_values_cannot_move_the_forecast(self, name: str) -> None:
        load, usable, days, j, cuts = self._cases(name)
        targets = list(range(j, j + 3))
        for cut in cuts:
            base = lf.run_forecaster(
                name, load_kwh=load, usable=usable, days=days, cutoff_idx=cut, target_days=targets
            )
            bad_load, bad_mask = _absurd(load, usable, cut)
            moved = lf.run_forecaster(
                name,
                load_kwh=bad_load,
                usable=bad_mask,
                days=days,
                cutoff_idx=cut,
                target_days=targets,
            )
            assert np.array_equal(base.values, moved.values)
            assert base.per_day == moved.per_day

    def test_the_perturbation_is_not_vacuous(self, name: str) -> None:
        load, usable, days, j, cuts = self._cases(name)
        cut = cuts[-1]
        base = lf.run_forecaster(
            name, load_kwh=load, usable=usable, days=days, cutoff_idx=cut, target_days=[j]
        )
        # Move the slice just *before* the cutoff (and last year's window for blend).
        earlier = load.copy()
        earlier[cut - 3 * 96 : cut] *= 1.7
        year_ago = days[j - 365][1]
        earlier[year_ago - 10 * 96 : year_ago + 10 * 96] *= 1.7
        moved = lf.run_forecaster(
            name, load_kwh=earlier, usable=usable, days=days, cutoff_idx=cut, target_days=[j]
        )
        assert not np.array_equal(base.values, moved.values)

    def test_last_years_window_reaches_blend_and_only_blend(self, name: str) -> None:
        load, usable, days, j, cuts = self._cases(name)
        cut = cuts[0]
        year_ago = days[j - 366][1]
        changed = load.copy()
        changed[year_ago - 20 * 96 : year_ago + 20 * 96] *= 2.0
        a = lf.run_forecaster(
            name, load_kwh=load, usable=usable, days=days, cutoff_idx=cut, target_days=[j]
        )
        b = lf.run_forecaster(
            name, load_kwh=changed, usable=usable, days=days, cutoff_idx=cut, target_days=[j]
        )
        # ``daytype`` looks back 84 days and ``ewma`` 56, all others at most 28: last
        # year's season is out of every window except blend's.
        assert np.array_equal(a.values, b.values) == (name != "blend")

    def test_the_day_ahead_vintage_ignores_the_target_day_and_later(self, name: str) -> None:
        load, usable, days, j, _cuts = self._cases(name)
        base = lf.forecast_each_day_at_its_start(
            name, load_kwh=load, usable=usable, days=days, target_days=[j]
        )
        bad_load, bad_mask = _absurd(load, usable, days[j][1])
        moved = lf.forecast_each_day_at_its_start(
            name, load_kwh=bad_load, usable=bad_mask, days=days, target_days=[j]
        )
        assert np.array_equal(base.values, moved.values)
