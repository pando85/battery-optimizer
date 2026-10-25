"""P4 second look: OMIE price outlook, the disabled-rule fallback and the seasonal hybrid.

Synthetic, on the structured forecaster's own world.  PVPC there is hourly; the OMIE series is
built so that the PVPC is an exact affine function of it per tariff period, which makes the
OMIE estimate exact and lets the tests compare it with the published prices.
"""

from __future__ import annotations

from datetime import date, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from test_structured_forecast import World, _day_index, _run

from battery_optimizer import forecast_eval as fe
from battery_optimizer import load_forecast as lf
from battery_optimizer import omie_outlook as oo
from battery_optimizer import structured_forecast as sf

MADRID = ZoneInfo("Europe/Madrid")
BETA = 1.3
INTERCEPT = {1: 0.12, 2: 0.08, 3: 0.04}


@pytest.fixture(scope="module")
def world() -> World:
    return World()


def _omie(world: World) -> np.ndarray:
    local = [s.astimezone(MADRID) for s in world.starts]
    period = np.array([oo.tariff_period(t.date(), t.hour) for t in local])
    return (world.prices - np.array([INTERCEPT[int(p)] for p in period])) / BETA


def _cut(world: World, day: date, hh: int, mm: int = 0) -> int:
    return world.days[_day_index(world, day)][1] + (hh * 60 + mm) // 15


NOV5 = date(2025, 11, 5)


class TestOmieOutlook:
    def _afternoon(self, world: World, **kwargs):
        j = _day_index(world, NOV5)
        omie = kwargs.pop("omie", _omie(world))
        side = world.side(omie=omie, **kwargs)
        return side, j

    def test_tomorrows_schedule_on_the_omie_estimate_is_the_published_schedule(self, world) -> None:
        side, j = self._afternoon(world)
        afternoon = _run(
            world, side, cut=_cut(world, NOV5, 14), targets=[j, j + 1], price_outlook="omie_period"
        )
        evening = _run(world, side, cut=_cut(world, NOV5, 20, 30), targets=[j, j + 1])
        est, pub = afternoon.per_day[1], evening.per_day[1]
        assert est["labels"]["prices"] == "estimated" and pub["labels"]["prices"] == "published"
        assert est["labels"]["climate"] == "structured"
        assert est["climate"]["selected_hours"] == pub["climate"]["selected_hours"]
        assert est["price_provenance"] if "price_provenance" in est else True

    def test_without_the_omie_series_tomorrow_is_unavailable(self, world) -> None:
        j = _day_index(world, NOV5)
        side = world.side()  # no OMIE series
        result = _run(
            world, side, cut=_cut(world, NOV5, 14), targets=[j, j + 1], price_outlook="omie_period"
        )
        assert result.per_day[1]["labels"]["prices"] == "unavailable"
        assert result.per_day[1]["climate_fallback_reason"] == "prices_unavailable"
        assert result.per_day[0]["labels"]["climate"] == "structured"

    def test_before_13_30_tomorrow_is_unavailable_even_with_the_series(self, world) -> None:
        side, j = self._afternoon(world)
        result = _run(
            world, side, cut=_cut(world, NOV5, 13), targets=[j, j + 1], price_outlook="omie_period"
        )
        assert result.per_day[1]["labels"]["prices"] == "unavailable"
        at_13_30 = _run(
            world,
            side,
            cut=_cut(world, NOV5, 13, 30),
            targets=[j, j + 1],
            price_outlook="omie_period",
        )
        assert at_13_30.per_day[1]["labels"]["prices"] == "estimated"

    def test_the_view_holds_only_the_omie_days_usable_at_the_cutoff(self, world) -> None:
        side, j = self._afternoon(world)
        early = side.view(_cut(world, NOV5, 13), world.days)
        late = side.view(_cut(world, NOV5, 13, 30), world.days)
        assert early.omie_eur_per_kwh.size == world.days[j][2]  # through today
        assert late.omie_eur_per_kwh.size == world.days[j + 1][2]  # through tomorrow
        assert not late.omie_eur_per_kwh.flags.writeable
        assert world.side().view(10, world.days).omie_eur_per_kwh is None

    def test_omie_after_the_usable_day_and_unpublished_prices_cannot_move_the_forecast(
        self, world
    ) -> None:
        j = _day_index(world, NOV5)
        cut = _cut(world, NOV5, 14)
        omie = _omie(world)
        base = _run(
            world, world.side(omie=omie), cut=cut, targets=[j, j + 1], price_outlook="omie_period"
        )
        beyond = omie.copy()
        beyond[world.days[j + 1][2] :] = 9.0
        end = world.side().view(cut, world.days).published_end_idx
        scrambled_prices = world.prices.copy()
        scrambled_prices[end:] = 99.0
        moved = _run(
            world,
            world.side(omie=beyond, prices=scrambled_prices),
            cut=cut,
            targets=[j, j + 1],
            price_outlook="omie_period",
        )
        assert np.array_equal(base.values, moved.values)
        assert base.per_day == moved.per_day

    def test_tomorrows_omie_does_move_the_forecast(self, world) -> None:
        j = _day_index(world, NOV5)
        cut = _cut(world, NOV5, 14)
        omie = _omie(world)
        base = _run(
            world, world.side(omie=omie), cut=cut, targets=[j, j + 1], price_outlook="omie_period"
        )
        flipped = omie.copy()
        lo, hi = world.days[j + 1][1], world.days[j + 1][2]
        flipped[lo:hi] = omie[lo:hi].max() + omie[lo:hi].min() - omie[lo:hi]
        moved = _run(
            world,
            world.side(omie=flipped),
            cut=cut,
            targets=[j, j + 1],
            price_outlook="omie_period",
        )
        assert not np.array_equal(base.values, moved.values)

    def test_the_side_fingerprint_and_slice_follow_the_omie_series(self, world) -> None:
        omie = _omie(world)
        a, b = world.side(omie=omie), world.side(omie=omie * 1.01)
        assert a.fingerprint != b.fingerprint != world.side().fingerprint
        assert a.slice(500).omie.size == 500

    def test_an_oracle_outlook_is_still_refused(self, world) -> None:
        j = _day_index(world, NOV5)
        with pytest.raises(ValueError, match="oracle"):
            _run(world, world.side(), cut=_cut(world, NOV5, 14), targets=[j], price_outlook="exact")


class TestDisabledFallback:
    @staticmethod
    def _disabled_world(world: World, first_disabled: date, manual: float = 0.01):
        """Rule enabled before *first_disabled*; after it, a small manual cooling pattern."""
        n = len(world.starts)
        enable = np.ones(n)
        cool = world.cool.copy()
        j0 = _day_index(world, first_disabled)
        start = world.days[j0][1]
        enable[start:] = 0.0
        cool[start:] = 0.0
        hour = np.array([t.astimezone(MADRID).hour for t in world.starts])
        cool[start:] = np.where((hour[start:] >= 12) & (hour[start:] < 18), manual, 0.0)
        load = world.servers + world.heat + cool + world.dhw
        return world.side(series={"climate_enable": enable, "hp_cool_kwh": cool}), load, cool

    def test_the_default_is_the_first_p4_behaviour(self, world) -> None:
        side, load, _cool = self._disabled_world(world, date(2025, 11, 1))
        j = _day_index(world, NOV5)
        cut = world.days[j][1]
        default = _run(world, side, cut=cut, targets=[j], load=load)
        explicit = _run(world, side, cut=cut, targets=[j], load=load, disabled_fallback="lf0")
        assert np.array_equal(default.values, explicit.values)
        assert default.per_day[0]["climate_source"] == "fallback_lf0"
        assert "disabled_history" not in default.per_day[0]

    def test_the_climate_part_follows_the_recent_disabled_days(self, world) -> None:
        side, load, cool = self._disabled_world(world, date(2025, 11, 1))
        j = _day_index(world, NOV5)
        cut = world.days[j][1]
        lf0 = _run(world, side, cut=cut, targets=[j], load=load)
        fixed = _run(
            world, side, cut=cut, targets=[j], load=load, disabled_fallback="disabled_history"
        )
        day = fixed.per_day[0]
        assert day["climate_source"] == "disabled_history"
        assert day["climate_fallback_reason"] == "rule_disabled"
        assert day["labels"]["climate"] == "fallback:rule_disabled"
        assert day["disabled_history"]["samples"] == 4  # 1-4 November
        assert day["disabled_history"]["source"] == "disabled_state_mean"
        lo, hi = world.days[j][1], world.days[j][2]
        # The manual pattern is exact in the history, so the whole-day total moves from the
        # trailing mean of every day (which carries October's cooling) to the manual energy.
        assert fixed.values[lo:hi].sum() < lf0.values[lo:hi].sum()
        err_fixed = np.abs(fixed.values[lo:hi] - load[lo:hi]).sum()
        err_lf0 = np.abs(lf0.values[lo:hi] - load[lo:hi]).sum()
        assert err_fixed < 0.5 * err_lf0

    def test_the_profile_is_the_slot_mean_of_the_disabled_days_only(self, world) -> None:
        side, load, cool = self._disabled_world(world, date(2025, 11, 1))
        j = _day_index(world, NOV5)
        series = world.heat + cool
        keep = np.ones(world.days[j][1], dtype=bool)
        values, record = sf.disabled_state_profile(
            series[: world.days[j][1]],
            keep,
            side.series["climate_enable"],
            world.days,
            j,
            history_days=28,
            lookback_days=60,
        )
        lo = world.days[_day_index(world, date(2025, 11, 4))][1]
        assert record["samples"] == 4
        assert np.allclose(values, series[lo : lo + 96])  # every disabled day has one pattern

    def test_the_history_is_capped_and_limited_in_time(self, world) -> None:
        side, load, cool = self._disabled_world(world, date(2025, 11, 1))
        j = _day_index(world, date(2025, 11, 12))
        series = (world.heat + cool)[: world.days[j][1]]
        keep = np.ones(series.size, dtype=bool)
        enable = side.series["climate_enable"]

        def run(**kw):
            return sf.disabled_state_profile(series, keep, enable, world.days, j, **kw)[1]

        assert run(history_days=3, lookback_days=60)["samples"] == 3
        assert run(history_days=28, lookback_days=60)["samples"] == 11
        assert run(history_days=28, lookback_days=5)["samples"] == 5

    def test_with_no_disabled_day_the_climate_part_is_zero(self, world) -> None:
        side, load, _cool = self._disabled_world(world, date(2025, 11, 1))
        j = _day_index(world, date(2025, 11, 1))  # the first disabled day: no disabled history
        cut = world.days[j][1] + 4  # 01:00, once the day's first flag sample is known
        result = _run(
            world, side, cut=cut, targets=[j], load=load, disabled_fallback="disabled_history"
        )
        day = result.per_day[0]
        assert day["climate_source"] == "disabled_history"
        assert day["disabled_history"] == {
            "samples": 0,
            "history_days": 28,
            "lookback_days": 60,
            "source": "zero_no_disabled_history",
        }

    def test_an_incomplete_or_unusable_day_is_not_a_sample(self, world) -> None:
        side, load, cool = self._disabled_world(world, date(2025, 11, 1))
        j = _day_index(world, NOV5)
        series = (world.heat + cool)[: world.days[j][1]]
        keep = np.ones(series.size, dtype=bool)
        bad = world.days[_day_index(world, date(2025, 11, 3))]
        keep[bad[1] : bad[2] - 20] = False  # fewer than 90 % of that day usable
        out = sf.disabled_state_profile(
            series,
            keep,
            side.series["climate_enable"],
            world.days,
            j,
            history_days=28,
            lookback_days=60,
        )[1]
        assert out["samples"] == 3

    def test_future_climate_energy_and_flags_cannot_move_the_forecast(self, world) -> None:
        side, load, cool = self._disabled_world(world, date(2025, 11, 1))
        j = _day_index(world, NOV5)
        cut = world.days[j][1]
        base = _run(
            world, side, cut=cut, targets=[j], load=load, disabled_fallback="disabled_history"
        )
        junk = np.full(len(world.starts), 7.0)
        junk[:cut] = side.series["hp_cool_kwh"][:cut]
        enable = side.series["climate_enable"].copy()
        enable[cut + 1 :] = 1.0
        moved = _run(
            world,
            world.side(series={"climate_enable": enable, "hp_cool_kwh": junk}),
            cut=cut,
            targets=[j],
            load=np.where(np.arange(load.size) < cut, load, 55.0),
            disabled_fallback="disabled_history",
        )
        assert np.array_equal(base.values, moved.values)
        assert base.per_day == moved.per_day

    def test_bad_parameters_are_refused(self, world) -> None:
        side = world.side()
        j = _day_index(world, NOV5)
        for params in (
            {"disabled_fallback": "zero"},
            {"disabled_history_days": 0},
            {"disabled_history_days": 30, "disabled_lookback_days": 10},
            {"climate_lf0_season": "autumn"},
        ):
            with pytest.raises(ValueError):
                _run(world, side, cut=world.days[j][1], targets=[j], **params)


class TestSeasonalHybrid:
    def test_the_season_months_are_the_g4_seasons(self) -> None:
        assert set(sf.HEATING_MONTHS) == set(fe.G4_SEASONS["heating"])
        assert set(sf.COOLING_MONTHS) == set(fe.G4_SEASONS["cooling"])

    def _day(self, world: World, day: date, **params):
        j = _day_index(world, day)
        return _run(world, world.side(), cut=world.days[j][1], targets=[j], **params).per_day[0]

    def test_heating_months_use_the_lf0_climate_part_and_cooling_months_the_model(
        self, world
    ) -> None:
        nov = self._day(world, NOV5, climate_lf0_season="heating")
        assert nov["climate_source"] == "fallback_lf0"
        assert nov["climate_fallback_reason"] == "season_hybrid"
        # September is a cooling month: the hybrid does not touch it (whether the model runs
        # that early in the synthetic history is another matter).
        sep = self._day(world, date(2025, 9, 20), climate_lf0_season="heating")
        assert sep["climate_fallback_reason"] != "season_hybrid"
        # October is a shoulder month: it belongs to neither season, so the model runs.
        octo = self._day(world, date(2025, 10, 7), climate_lf0_season="heating")
        assert octo["climate_source"] == "structured"

    def test_the_opposite_season_flips_it_and_the_default_changes_nothing(self, world) -> None:
        assert self._day(world, NOV5, climate_lf0_season="cooling")["climate_source"] == (
            "structured"
        )
        assert (
            self._day(world, date(2025, 9, 20), climate_lf0_season="cooling")[
                "climate_fallback_reason"
            ]
            == "season_hybrid"
        )
        assert self._day(world, NOV5)["climate_source"] == "structured"

    def test_the_hybrid_climate_part_is_what_the_lf0_fallback_gives(self, world) -> None:
        j = _day_index(world, NOV5)
        cut = world.days[j][1]
        hybrid = _run(world, world.side(), cut=cut, targets=[j], climate_lf0_season="heating")
        disabled = _run(
            world,
            world.side(series={"climate_enable": np.zeros(len(world.starts))}),
            cut=cut,
            targets=[j],
        )
        lo, hi = world.days[j][1], world.days[j][2]
        # Same climate part (the LF-0 trailing mean), same hot-water and residual parts.
        assert np.allclose(hybrid.values[lo:hi], disabled.values[lo:hi])

    def test_a_disabled_day_in_the_season_takes_the_disabled_fallback_first(self, world) -> None:
        j = _day_index(world, NOV5)
        side = world.side(series={"climate_enable": np.zeros(len(world.starts))})
        day = _run(
            world,
            side,
            cut=world.days[j][1],
            targets=[j],
            climate_lf0_season="heating",
            disabled_fallback="disabled_history",
        ).per_day[0]
        assert day["climate_fallback_reason"] == "rule_disabled"
        assert day["climate_source"] == "disabled_history"


class TestDiagnostics:
    def test_the_component_report_runs_with_the_second_look_params(self, world) -> None:
        j = _day_index(world, NOV5)
        side = world.side(omie=_omie(world))
        decisions = [
            {
                "kind": "afternoon",
                "cutoff_idx": _cut(world, NOV5, 14),
                "target_days": [j, j + 1],
                "start_idx": _cut(world, NOV5, 14),
                "stop_idx": world.days[j + 1][2],
            }
        ]
        forecaster = lf.get_forecaster("structured")
        params = {
            "features": "x",
            "price_outlook": "omie_period",
            "disabled_fallback": "disabled_history",
            "climate_lf0_season": "heating",
        }
        out = forecaster.diagnose(
            side=side,
            load=world.load,
            usable=world.usable,
            days=world.days,
            decisions=decisions,
            params=params,
        )
        assert out["decisions"] == 1
        assert "afternoon/prices" in out["counts"]
        assert out["counts"]["afternoon/prices"].get("estimated", 0) == 1
        assert timedelta(0) == timedelta(0)
