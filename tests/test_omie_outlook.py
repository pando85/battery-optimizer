"""PO-C: the OMIE next-day estimators (``omie_period``, ``omie_period_month``).

Synthetic throughout.  PVPC is built as ``BETA * OMIE + intercept(period)`` so the fit has a
known answer, and the leakage tests perturb what must not matter: OMIE days that are not usable
at the decision, and realized PVPC after the publication boundary.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from test_extract_omie import Net, data_dir, omie_file, run  # noqa: F401
from test_rolling_horizon import QUARTER, _make_dataset, _spec

from battery_optimizer import omie_outlook as oo
from battery_optimizer import price_outlook as po
from battery_optimizer.omie_prices import omie_usable_from
from battery_optimizer.policies import contiguous_day_bounds
from battery_optimizer.rolling_horizon import build_frozen_load_profile, run_rolling_horizon

MADRID = ZoneInfo("Europe/Madrid")
BETA = 1.3
INTERCEPT = {1: 0.12, 2: 0.08, 3: 0.04}
OMIE_OUTLOOKS = ("omie_period", "omie_period_month")


def _local(d: date, hh: int, mm: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=MADRID).astimezone(UTC)


class World:
    """40 local days from 2025-06-02 with PVPC = BETA * OMIE + intercept(period)."""

    def __init__(self, n_days: int = 40, seed: int = 1, shift_from: date | None = None) -> None:
        start = datetime(2025, 6, 1, 22, tzinfo=UTC)  # local midnight of 2025-06-02
        _p, self.load, self.usable, self.local_days, self.starts = _make_dataset(start, n_days)
        rng = np.random.default_rng(seed)
        hours = len(self.starts) // 4
        self.omie = np.repeat(0.04 + 0.1 * rng.random(hours), 4)
        local = [s.astimezone(MADRID) for s in self.starts]
        period = np.array([oo.tariff_period(t.date(), t.hour) for t in local])
        self.pvpc = BETA * self.omie + np.array([INTERCEPT[int(p)] for p in period])
        if shift_from is not None:
            after = np.array([t.date() >= shift_from for t in local])
            self.pvpc = self.pvpc + 0.05 * after
        self.days = contiguous_day_bounds(self.local_days)

    def at(self, d: date, hh: int, mm: int = 0) -> int:
        return next(i for i, s in enumerate(self.starts) if s == _local(d, hh, mm))

    def day_start(self, d: date) -> int:
        return next(i for i, day in enumerate(self.local_days) if day == d)

    def run(
        self,
        name: str,
        decision: int,
        *,
        pvpc: np.ndarray | None = None,
        omie: np.ndarray | None = None,
        side: po.OutlookSide | None | str = "default",
        params: dict | None = None,
        plan_end: int | None = None,
    ) -> po.OutlookResult:
        prices = self.pvpc if pvpc is None else pvpc
        if side == "default":
            side = po.OutlookSide(self.omie if omie is None else omie)
        end = plan_end if plan_end is not None else self.day_start(self.d1(decision)) + 96
        return po.run_price_outlook(
            name,
            prices=prices,
            published_end_idx=po.published_end_idx(self.starts[decision], self.days, MADRID),
            decision_idx=decision,
            plan_end_idx=end,
            interval_starts_utc=self.starts,
            days=self.days,
            tz=MADRID,
            params=params,
            side=side,  # type: ignore[arg-type]
        )

    def d1(self, decision: int) -> date:
        return self.starts[decision].astimezone(MADRID).date() + timedelta(days=1)


D = date(2025, 7, 9)  # a Wednesday; its 14:00 event plans 2025-07-10


# ---------------------------------------------------------------------------
# Tariff periods
# ---------------------------------------------------------------------------


class TestTariffPeriod:
    def test_weekday_bands(self) -> None:
        wed = date(2025, 7, 9)
        got = {h: oo.tariff_period(wed, h) for h in range(24)}
        assert {h for h, p in got.items() if p == 3} == set(range(0, 8))
        assert {h for h, p in got.items() if p == 2} == {8, 9, 14, 15, 16, 17, 22, 23}
        assert {h for h, p in got.items() if p == 1} == {10, 11, 12, 13, 18, 19, 20, 21}

    def test_weekend_and_national_holidays_are_valle_all_day(self) -> None:
        for day in (date(2025, 7, 12), date(2025, 7, 13), date(2025, 12, 8), date(2026, 1, 6)):
            assert {oo.tariff_period(day, h) for h in range(24)} == {3}, day
        # 8 December 2025 is a Monday; a regional holiday is not a period rule.
        assert oo.tariff_period(date(2025, 8, 18), 12) == 1

    def test_the_period_bands_match_the_tariff_config(self) -> None:
        from pathlib import Path

        from battery_optimizer.config import load_config, tariff_period_for

        cfg = load_config(Path(__file__).resolve().parent.parent / "config" / "tariff.yaml")
        # Plain days only: the configured holiday list is empty on purpose (see tariff.yaml),
        # the estimator adds the nine national fixed-date holidays on top of the bands.
        for day in (date(2025, 7, 9), date(2025, 7, 12), date(2025, 7, 13)):
            for h in range(24):
                assert oo.tariff_period(day, h) == tariff_period_for(cfg.tariff_periods, day, h)


# ---------------------------------------------------------------------------
# The estimators
# ---------------------------------------------------------------------------


class TestEstimators:
    def setup_method(self) -> None:
        self.w = World()
        self.decision = self.w.at(D, 14)
        self.target = slice(self.w.day_start(D + timedelta(days=1)), None)

    def test_registered_as_non_oracle_omie_outlooks(self) -> None:
        for name in OMIE_OUTLOOKS:
            outlook = po.get_price_outlook(name)
            assert not outlook.oracle and outlook.needs_omie
            assert outlook.inputs_usable_from == po.OMIE_NEXT_DAY_USABLE_FROM
            assert "omie_rule" in outlook.provenance({})
        assert po.get_price_outlook("persistence").needs_omie is False
        assert "omie_rule" not in po.get_price_outlook("persistence").provenance({})
        assert po.price_outlook_names()[:3] == ("exact", "published_only", "persistence")

    @pytest.mark.parametrize("name", OMIE_OUTLOOKS)
    def test_it_recovers_an_exact_affine_relation(self, name: str) -> None:
        r = self.w.run(name, self.decision)
        n_pub = self.w.day_start(D + timedelta(days=1)) - self.decision
        assert (r.provenance[:n_pub] == po.SLOT_PUBLISHED).all()
        assert (r.provenance[n_pub:] == po.SLOT_ESTIMATED).all()
        want = self.w.pvpc[self.target][:96]
        assert np.allclose(r.values[n_pub:], want, atol=1e-9)
        assert r.diagnostics["fit_days"] >= oo.MIN_FIT_DAYS
        assert abs(r.diagnostics["beta"] - BETA) < 1e-3
        assert r.diagnostics["oracle"] is False

    def test_the_estimate_is_flat_inside_an_hour(self) -> None:
        r = self.w.run("omie_period", self.decision)
        n_pub = self.w.day_start(D + timedelta(days=1)) - self.decision
        est = r.values[n_pub:].reshape(24, 4)
        assert (est == est[:, :1]).all()

    def test_a_published_price_is_never_overwritten(self) -> None:
        r = self.w.run("omie_period", self.decision)
        n_pub = self.w.day_start(D + timedelta(days=1)) - self.decision
        assert np.array_equal(r.values[:n_pub], self.w.pvpc[self.decision : self.decision + n_pub])

    def test_the_default_windows_are_the_2024_choices(self) -> None:
        assert po.get_price_outlook("omie_period").defaults == {
            "window_days": oo.OMIE_PERIOD_WINDOW_DAYS
        }
        assert po.get_price_outlook("omie_period_month").defaults == {
            "window_days": oo.OMIE_PERIOD_MONTH_WINDOW_DAYS
        }

    def test_the_month_variant_follows_a_level_shift_a_plain_fit_averages(self) -> None:
        w = World(shift_from=date(2025, 7, 1))
        decision = w.at(D, 14)
        target = slice(w.day_start(D + timedelta(days=1)), None)
        want = w.pvpc[target][:96]
        n_pub = w.day_start(D + timedelta(days=1)) - decision
        month = w.run("omie_period_month", decision, params={"window_days": 40})
        plain = w.run("omie_period", decision, params={"window_days": 40})
        assert np.allclose(month.values[n_pub:], want, atol=1e-9)
        assert np.abs(plain.values[n_pub:] - want).max() > 0.005
        assert month.diagnostics["fit_months"] == 2


# ---------------------------------------------------------------------------
# As-of rules and leakage
# ---------------------------------------------------------------------------


class TestAsOf:
    def setup_method(self) -> None:
        self.w = World()
        self.decision = self.w.at(D, 14)
        self.boundary = self.w.day_start(D + timedelta(days=1))
        self.omie_end = self.w.day_start(D + timedelta(days=2))

    @pytest.mark.parametrize("name", OMIE_OUTLOOKS)
    def test_omie_after_the_usable_day_cannot_change_the_estimate(self, name: str) -> None:
        base = self.w.run(name, self.decision)
        perturbed = self.w.omie.copy()
        perturbed[self.omie_end :] = 9.0 + np.arange(perturbed.size - self.omie_end)
        other = self.w.run(name, self.decision, omie=perturbed)
        assert np.array_equal(base.values, other.values, equal_nan=True)
        assert np.array_equal(base.provenance, other.provenance)

    @pytest.mark.parametrize("name", OMIE_OUTLOOKS)
    def test_realized_pvpc_after_the_publication_boundary_cannot_change_it(self, name: str) -> None:
        base = self.w.run(name, self.decision)
        perturbed = self.w.pvpc.copy()
        perturbed[self.boundary :] = 99.0 + np.arange(perturbed.size - self.boundary)
        other = self.w.run(name, self.decision, pvpc=perturbed)
        assert np.array_equal(base.values, other.values, equal_nan=True)
        assert np.array_equal(base.provenance, other.provenance)

    @pytest.mark.parametrize("name", OMIE_OUTLOOKS)
    def test_tomorrows_omie_matters_from_13_30_and_not_before(self, name: str) -> None:
        early = self.w.at(D, 13, 0)
        late = self.w.at(D, 13, 30)
        bumped = self.w.omie.copy()
        bumped[self.boundary : self.omie_end] += 0.05
        a, b = self.w.run(name, early), self.w.run(name, early, omie=bumped)
        assert (a.provenance == po.SLOT_UNAVAILABLE).sum() == 96
        assert "inputs not usable yet" in a.diagnostics["fallback"]
        assert np.array_equal(a.values, b.values, equal_nan=True)
        c, d = self.w.run(name, late), self.w.run(name, late, omie=bumped)
        assert (c.provenance == po.SLOT_ESTIMATED).sum() == 96
        assert not np.allclose(c.values, d.values)

    def test_the_seam_cut_agrees_with_the_omie_store_rule(self) -> None:
        # The seam's 13:30 boundary and the read side's omie_usable_from are one rule.
        for d in (date(2025, 7, 9), date(2025, 10, 25), date(2026, 3, 28)):
            nxt = d + timedelta(days=1)
            instant = omie_usable_from(nxt)
            assert instant.astimezone(MADRID).date() == d
            assert (
                instant.astimezone(MADRID).time().replace(tzinfo=None)
                == po.OMIE_NEXT_DAY_USABLE_FROM
            )
            days = [(d, 0, 96), (nxt, 96, 192), (nxt + timedelta(days=1), 192, 288)]
            before = po.omie_end_idx(instant - timedelta(seconds=1), days, MADRID)
            after = po.omie_end_idx(instant, days, MADRID)
            assert before == 96 and after == 192

    @pytest.mark.parametrize("name", OMIE_OUTLOOKS)
    def test_the_fit_reads_only_published_days(self, name: str) -> None:
        # At 14:00 the fit window ends with D (published); D+1's PVPC is unpublished.
        r = self.w.run(name, self.decision)
        assert r.diagnostics["fit_days"] <= po.get_price_outlook(name).defaults["window_days"]
        poisoned = self.w.pvpc.copy()
        poisoned[self.boundary :] = np.nan  # unpublished days unknown: still the same fit
        assert np.allclose(
            self.w.run(name, self.decision, pvpc=poisoned).values,
            r.values,
            equal_nan=True,
        )


# ---------------------------------------------------------------------------
# Fallbacks
# ---------------------------------------------------------------------------


class TestFallbacks:
    def setup_method(self) -> None:
        self.w = World()
        self.decision = self.w.at(D, 14)
        self.boundary = self.w.day_start(D + timedelta(days=1))

    @pytest.mark.parametrize("name", OMIE_OUTLOOKS)
    def test_no_side_input_leaves_every_unpublished_slot_unavailable(self, name: str) -> None:
        r = self.w.run(name, self.decision, side=None)
        assert (r.provenance[self.boundary - self.decision :] == po.SLOT_UNAVAILABLE).all()
        assert r.diagnostics["fallback"] == "omie_side_input_missing"
        assert r.usable_end_idx == self.boundary

    @pytest.mark.parametrize("name", OMIE_OUTLOOKS)
    def test_a_missing_omie_quarter_blanks_its_hour_and_records_why(self, name: str) -> None:
        omie = self.w.omie.copy()
        omie[self.boundary + 40 + 2] = np.nan  # one quarter of 10:00
        r = self.w.run(name, self.decision, omie=omie)
        k = self.boundary - self.decision
        bad = np.flatnonzero(r.provenance[k:] == po.SLOT_UNAVAILABLE)
        assert bad.tolist() == [40, 41, 42, 43]
        assert r.diagnostics["fallback"] == "omie_missing_for_target_hours"
        assert r.diagnostics["slots_without_omie"] == 4
        # The planner stops at the first unavailable slot.
        assert r.usable_end_idx == self.boundary + 40

    @pytest.mark.parametrize("name", OMIE_OUTLOOKS)
    def test_a_missing_day_of_omie_in_the_fit_window_only_thins_the_fit(self, name: str) -> None:
        omie = self.w.omie.copy()
        gap = self.w.day_start(D - timedelta(days=3))
        omie[gap : gap + 96] = np.nan
        r = self.w.run(name, self.decision, omie=omie)
        assert (r.provenance[self.boundary - self.decision :] == po.SLOT_ESTIMATED).all()
        assert np.allclose(
            r.values[self.boundary - self.decision :],
            self.w.pvpc[self.boundary : self.boundary + 96],
            atol=1e-9,
        )

    @pytest.mark.parametrize("name", OMIE_OUTLOOKS)
    def test_a_short_fit_window_is_a_recorded_fallback(self, name: str) -> None:
        early = self.w.at(date(2025, 6, 6), 14)  # four published days
        r = self.w.run(name, early)
        assert r.diagnostics["fallback"] == "fit_window_short"
        assert r.counts()["estimated"] == 0

    def test_a_plan_beyond_the_usable_omie_day_is_cut_there(self) -> None:
        end = self.boundary + 192  # two days past the boundary
        r = self.w.run("omie_period", self.decision, plan_end=end)
        k = self.boundary - self.decision
        assert (r.provenance[k : k + 96] == po.SLOT_ESTIMATED).all()
        assert (r.provenance[k + 96 :] == po.SLOT_UNAVAILABLE).all()
        assert r.diagnostics["fallback"] == "plan_reaches_beyond_usable_omie_day"

    def test_too_short_a_window_param_is_refused(self) -> None:
        with pytest.raises(ValueError, match="window_days must be >="):
            self.w.run("omie_period", self.decision, params={"window_days": 3})


# ---------------------------------------------------------------------------
# The dispatcher
# ---------------------------------------------------------------------------


def _roll(w: World, price_outlook: str, side, pvpc=None):
    eval_start = w.day_start(D)
    frozen, _ = build_frozen_load_profile(
        w.load,
        w.local_days,
        w.usable,
        training_end_idx=eval_start,
        window_days=28,
        interval_starts_utc=w.starts,
    )
    return run_rolling_horizon(
        _spec(),
        prices=w.pvpc if pvpc is None else pvpc,
        load_kwh=w.load,
        usable=w.usable,
        interval_seconds=QUARTER,
        local_days=w.local_days,
        interval_starts_utc=w.starts,
        eval_start_idx=eval_start,
        eval_end_idx=eval_start + 2 * 96,
        tz=MADRID,
        frozen_load=frozen,
        terminal_value_eur_per_kwh=0.10,
        price_outlook=price_outlook,
        price_outlook_side=side,
    )


def _afternoon(result, day: date) -> dict:
    return next(
        e for e in result.per_event if e["kind"] == "afternoon" and e["local_day"] == str(day)
    )


class TestDispatcher:
    def setup_method(self) -> None:
        self.w = World()
        self.boundary = self.w.day_start(D + timedelta(days=1))
        self.side = po.OutlookSide(self.w.omie, {"kind": "test"})

    def test_the_14_00_plan_uses_the_estimate_and_records_the_side_input(self) -> None:
        res = _roll(self.w, "omie_period", self.side)
        ev = _afternoon(res, D)
        assert ev["estimated_price_intervals"] == 96
        assert ev["unavailable_price_intervals"] == 0
        assert ev["planned_through_idx"] == ev["plan_end_idx"]
        tel = res.telemetry["price_outlook"]
        assert tel["name"] == "omie_period" and tel["oracle"] is False
        assert tel["side_input"]["kind"] == "test" and "fingerprint" in tel["side_input"]
        assert "fallback_events" not in tel

    def test_tomorrows_realized_pvpc_does_not_reach_the_plan_but_tomorrows_omie_does(self) -> None:
        base = _roll(self.w, "omie_period", self.side)
        pvpc = self.w.pvpc.copy()
        pvpc[self.boundary : self.boundary + 96] = np.linspace(-0.2, 1.5, 96)
        same = _roll(self.w, "omie_period", self.side, pvpc=pvpc)
        assert (
            _afternoon(base, D)["plan_prices_sha256"] == _afternoon(same, D)["plan_prices_sha256"]
        )
        bumped = self.w.omie.copy()
        bumped[self.boundary : self.boundary + 96] += 0.05
        other = _roll(self.w, "omie_period", po.OutlookSide(bumped, {"kind": "test"}))
        assert (
            _afternoon(base, D)["plan_prices_sha256"] != _afternoon(other, D)["plan_prices_sha256"]
        )

    def test_a_missing_omie_day_falls_back_to_published_only_and_is_counted(self) -> None:
        omie = self.w.omie.copy()
        omie[self.boundary : self.boundary + 96] = np.nan
        res = _roll(self.w, "omie_period", po.OutlookSide(omie, {"kind": "test"}))
        only = _roll(self.w, "published_only", None)
        ev = _afternoon(res, D)
        assert ev["unavailable_price_intervals"] == 96
        assert ev["price_outlook_fallback"] == "omie_missing_for_target_hours"
        assert res.telemetry["price_outlook"]["fallback_events"] == {
            "omie_missing_for_target_hours": 1
        }
        assert ev["plan_prices_sha256"] == _afternoon(only, D)["plan_prices_sha256"]
        assert np.array_equal(res.schedule.charge_kwh, only.schedule.charge_kwh)

    def test_without_a_side_input_it_is_published_only(self) -> None:
        res = _roll(self.w, "omie_period", None)
        only = _roll(self.w, "published_only", None)
        assert np.array_equal(res.schedule.discharge_kwh, only.schedule.discharge_kwh)
        assert res.telemetry["price_outlook"]["fallback_events"] == {"omie_side_input_missing": 1}


# ---------------------------------------------------------------------------
# The side input from the verified store
# ---------------------------------------------------------------------------


class TestLoadOmieSide:
    def _grid(self, day: date, days: int) -> list[datetime]:
        start = _local(day, 0)
        return [start + timedelta(minutes=15 * i) for i in range(96 * days)]

    def test_values_are_eur_per_kwh_on_the_dataset_grid_and_gaps_are_nan(self, data_dir) -> None:  # noqa: F811
        d1, d2 = date(2025, 9, 30), date(2025, 10, 1)  # hourly file, then quarter-hour file
        run(data_dir, Net(), d1, d1)
        run(data_dir, Net(), d2, d2)
        grid = self._grid(d1, 3)  # the third day is not stored
        side = oo.load_omie_side(grid, data_dir=data_dir)
        assert side.n == len(grid)
        first = side.omie_eur_per_kwh
        assert first[0] == pytest.approx((60.0 + 0.5) / 1000.0)
        assert first[3] == pytest.approx((60.0 + 0.5) / 1000.0)  # hourly file repeats 4 times
        assert first[4] == pytest.approx((60.0 + 1.0) / 1000.0)
        assert first[96 + 1] == pytest.approx((60.0 + 1.0) / 1000.0)  # quarter-hour file
        assert np.isnan(first[192:]).all() and np.isfinite(first[:192]).all()
        assert side.source["missing_dates"] == ["2025-10-02"]
        assert side.source["slots_without_price"] == 96
        assert len(side.fingerprint) == 64

    def test_a_grid_that_is_not_15_minute_is_refused(self, data_dir) -> None:  # noqa: F811
        hourly = [_local(date(2025, 9, 30), 0) + timedelta(hours=i) for i in range(24)]
        with pytest.raises(ValueError, match="uniform 15-minute"):
            oo.load_omie_side(hourly, data_dir=data_dir)
