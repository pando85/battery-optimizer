"""Price-outlook seam: publication rule, causality, registered outlooks and the dispatcher.

The leakage tests are the point of the seam: a non-oracle outlook sees only the prices
published at the decision instant, so perturbing anything after the publication boundary
(tomorrow's realized prices at the 14:00 event) cannot change what the event plans on.
They are parametrised over every registered non-oracle outlook, so a new outlook (PO-C's
OMIE estimators) is covered without writing a new test.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from test_rolling_horizon import (
    QUARTER,
    _dst_fall_back_window,
    _dst_spring_forward_window,
    _make_dataset,
    _spec,
)

from battery_optimizer import benchmark as bm
from battery_optimizer import price_outlook as po
from battery_optimizer.policies import contiguous_day_bounds
from battery_optimizer.rolling_horizon import (
    build_frozen_load_profile,
    compute_rolling_events,
    run_rolling_horizon,
)

MADRID = ZoneInfo("Europe/Madrid")
NON_ORACLE = po.price_outlook_names(include_oracle=False)


def _local(d: date, hh: int, mm: int) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=MADRID).astimezone(UTC)


def _grid(start: datetime, n_days: int, seed: int = 3):
    prices, load, usable, local_days, starts = _make_dataset(start, n_days)
    prices = 0.05 + np.random.default_rng(seed).random(prices.size) * 0.3
    return prices, load, usable, local_days, starts


def _call(name: str, prices, starts, local_days, decision_idx: int, plan_end: int):
    days = contiguous_day_bounds(local_days)
    return po.run_price_outlook(
        name,
        prices=prices,
        published_end_idx=po.published_end_idx(starts[decision_idx], days, MADRID),
        decision_idx=decision_idx,
        plan_end_idx=plan_end,
        interval_starts_utc=starts,
        days=days,
        tz=MADRID,
        allow_oracle=True,
    )


# ---------------------------------------------------------------------------
# Publication rule
# ---------------------------------------------------------------------------


class TestPublicationRule:
    def test_next_day_is_published_from_20_15_local(self) -> None:
        d = date(2025, 6, 16)
        assert po.PVPC_NEXT_DAY_PUBLISHED_FROM == time(20, 15)
        assert po.last_published_day(_local(d, 14, 0), MADRID) == d
        assert po.last_published_day(_local(d, 20, 0), MADRID) == d
        assert po.last_published_day(_local(d, 20, 15), MADRID) == d + timedelta(days=1)
        assert po.last_published_day(_local(d, 20, 30), MADRID) == d + timedelta(days=1)
        assert po.OMIE_NEXT_DAY_USABLE_FROM == time(13, 30)

    @pytest.mark.parametrize(
        "window",
        [
            lambda: _make_dataset(datetime(2025, 6, 14, 22, tzinfo=UTC), 5),
            _dst_fall_back_window,
            _dst_spring_forward_window,
        ],
    )
    def test_it_matches_the_legacy_events_information_set(self, window) -> None:
        """At every legacy event the 20:15 rule and the dispatcher's 20:30 boundary agree."""
        _prices, _load, _usable, local_days, starts = window()
        days = contiguous_day_bounds(local_days)
        events = compute_rolling_events(
            starts, local_days, eval_start_idx=10, eval_end_idx=len(starts), tz=MADRID
        )
        assert {e.kind for e in events} >= {"warmup", "afternoon", "evening"}
        for ev in events:
            boundary = min(
                po.published_end_idx(starts[ev.start_idx], days, MADRID), ev.plan_end_idx
            )
            assert boundary == ev.prices_exact_end_idx, ev


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


class TestSeam:
    def setup_method(self) -> None:
        self.prices, _l, _u, self.local_days, self.starts = _grid(
            datetime(2025, 6, 14, 22, tzinfo=UTC), 4
        )
        # The 14:00 event of the second local day, planning to the end of the third.
        self.decision = next(
            i for i, s in enumerate(self.starts) if s == _local(date(2025, 6, 16), 14, 0)
        )
        self.plan_end = next(i for i, d in enumerate(self.local_days) if d == date(2025, 6, 18))
        self.boundary = next(i for i, d in enumerate(self.local_days) if d == date(2025, 6, 17))

    def test_registry(self) -> None:
        assert po.price_outlook_names()[:3] == ("exact", "published_only", "persistence")
        assert po.DEFAULT_PRICE_OUTLOOK == bm.DEFAULT_ROLLING_HORIZON_PRICE_OUTLOOK == "exact"
        assert po.get_price_outlook("exact").oracle is True
        assert "exact" not in NON_ORACLE and "persistence" in NON_ORACLE
        with pytest.raises(ValueError, match="unknown price outlook 'nope'"):
            po.get_price_outlook("nope")
        with pytest.raises(ValueError, match="no param"):
            po.resolve_price_outlook_params("persistence", {"alpha": 1})

    def test_oracle_needs_allow_oracle(self) -> None:
        days = contiguous_day_bounds(self.local_days)
        with pytest.raises(ValueError, match="only allowed in offline evaluation"):
            po.run_price_outlook(
                "exact",
                prices=self.prices,
                published_end_idx=self.boundary,
                decision_idx=self.decision,
                plan_end_idx=self.plan_end,
                interval_starts_utc=self.starts,
                days=days,
                tz=MADRID,
            )

    @pytest.mark.parametrize("name", po.price_outlook_names())
    def test_published_slots_are_copied_and_labelled(self, name: str) -> None:
        r = _call(name, self.prices, self.starts, self.local_days, self.decision, self.plan_end)
        n_pub = self.boundary - self.decision
        assert np.array_equal(r.values[:n_pub], self.prices[self.decision : self.boundary])
        assert (r.provenance[:n_pub] == po.SLOT_PUBLISHED).all()
        assert set(r.provenance[n_pub:]) <= {po.SLOT_ESTIMATED, po.SLOT_UNAVAILABLE}
        assert not r.values.flags.writeable
        assert sum(r.counts().values()) == self.plan_end - self.decision

    @pytest.mark.parametrize("name", NON_ORACLE)
    def test_no_price_after_the_publication_boundary_reaches_an_outlook(self, name: str) -> None:
        base = _call(name, self.prices, self.starts, self.local_days, self.decision, self.plan_end)
        perturbed = self.prices.copy()
        perturbed[self.boundary :] = 99.0 + np.arange(perturbed.size - self.boundary)
        other = _call(name, perturbed, self.starts, self.local_days, self.decision, self.plan_end)
        assert np.array_equal(base.values, other.values, equal_nan=True)
        assert np.array_equal(base.provenance, other.provenance)
        assert base.usable_end_idx == other.usable_end_idx

    def test_exact_reads_the_realized_prices(self) -> None:
        r = _call("exact", self.prices, self.starts, self.local_days, self.decision, self.plan_end)
        assert np.array_equal(r.values, self.prices[self.decision : self.plan_end])
        assert r.usable_end_idx == self.plan_end
        assert r.counts()["estimated"] == self.plan_end - self.boundary

    def test_published_only_stops_at_the_last_published_price(self) -> None:
        r = _call(
            "published_only",
            self.prices,
            self.starts,
            self.local_days,
            self.decision,
            self.plan_end,
        )
        assert r.usable_end_idx == self.boundary
        assert r.counts() == {
            "published": self.boundary - self.decision,
            "estimated": 0,
            "unavailable": self.plan_end - self.boundary,
        }
        assert np.array_equal(r.usable_values, self.prices[self.decision : self.boundary])

    def test_persistence_repeats_today_at_the_same_local_time(self) -> None:
        r = _call(
            "persistence", self.prices, self.starts, self.local_days, self.decision, self.plan_end
        )
        today = self.prices[self.boundary - 96 : self.boundary]
        assert np.array_equal(r.values[self.boundary - self.decision :], today)
        assert r.diagnostics["source_day"] == "2025-06-16"
        assert r.diagnostics["slots_from_earlier_day"] == 0

    def test_persistence_skips_a_price_the_source_day_lacks(self) -> None:
        prices = self.prices.copy()
        hole = self.boundary - 96 + 40  # one slot of the last published day is unknown
        prices[hole] = np.nan
        r = _call("persistence", prices, self.starts, self.local_days, self.decision, self.plan_end)
        k = self.boundary - self.decision + 40
        # Taken from the day before, which has the clock time, never a NaN estimate.
        assert r.values[k] == prices[hole - 96]
        assert r.diagnostics["slots_from_earlier_day"] == 1

    def test_malformed_answers_are_refused(self, monkeypatch) -> None:
        def short(request):
            return np.zeros(1), np.ones(1, dtype=bool), {}

        def nan(request):
            n = request.target_idx.size
            return np.full(n, np.nan), np.ones(n, dtype=bool), {}

        for fn, match in ((short, "for 96 unpublished"), (nan, "non-finite")):
            monkeypatch.setitem(po._REGISTRY, "bad", po.PriceOutlook("bad", fn, {}, False, "test"))
            with pytest.raises(ValueError, match=match):
                _call(
                    "bad", self.prices, self.starts, self.local_days, self.decision, self.plan_end
                )

    def test_inputs_not_usable_yet_fall_back_to_published_only(self, monkeypatch) -> None:
        """The PO-C hook: an outlook declaring an as-of time is not run before it."""
        calls: list[int] = []

        def omie_like(request):
            calls.append(request.decision_idx)
            n = request.target_idx.size
            return np.full(n, 0.1), np.ones(n, dtype=bool), {}

        monkeypatch.setitem(
            po._REGISTRY,
            "omie_like",
            po.PriceOutlook(
                "omie_like", omie_like, {}, False, "test", po.OMIE_NEXT_DAY_USABLE_FROM
            ),
        )
        early = self.decision - 4 * 1  # 13:00 local
        r = _call("omie_like", self.prices, self.starts, self.local_days, early, self.plan_end)
        assert calls == [] and r.usable_end_idx == self.boundary
        assert "inputs not usable yet" in r.diagnostics["fallback"]
        assert r.diagnostics["inputs_usable_from_local"] == "13:30"
        r = _call(
            "omie_like", self.prices, self.starts, self.local_days, self.decision, self.plan_end
        )
        assert calls == [self.decision] and r.usable_end_idx == self.plan_end


class TestPersistenceAcrossDst:
    def _run(self, window, d: date):
        prices, _l, _u, local_days, starts = window()
        prices = 0.05 + np.arange(len(prices)) * 1e-4
        decision = next(i for i, s in enumerate(starts) if s == _local(d, 14, 0))
        nxt = d + timedelta(days=1)
        boundary = local_days.index(nxt)
        plan_end = boundary + local_days.count(nxt)
        r = _call("persistence", prices, starts, local_days, decision, plan_end)
        return r, prices, starts, boundary, plan_end

    def test_fall_back_target_day_repeats_the_hour(self) -> None:
        # 2025-10-25 14:00 estimates the 100-slot 2025-10-26.
        r, prices, starts, boundary, plan_end = self._run(_dst_fall_back_window, date(2025, 10, 25))
        assert plan_end - boundary == 100
        assert r.counts()["estimated"] == 100 and r.counts()["unavailable"] == 0
        decision = r.usable_end_idx - r.values.size
        source = {
            (s.astimezone(MADRID).hour, s.astimezone(MADRID).minute): i
            for i, s in enumerate(starts[boundary - 96 : boundary], start=boundary - 96)
        }
        for t in range(boundary, plan_end):
            local = starts[t].astimezone(MADRID)
            assert r.values[t - decision] == prices[source[(local.hour, local.minute)]]
        # Both 02:00 slots (CEST, then CET) carry yesterday's 02:00 price.
        twos = [t for t in range(boundary, plan_end) if starts[t].astimezone(MADRID).hour == 2]
        assert len(twos) == 8
        assert r.values[twos[0] - decision] == r.values[twos[4] - decision]

    def test_fall_back_source_day_uses_the_first_occurrence(self) -> None:
        # 2025-10-26 14:00 estimates 2025-10-27 from the 100-slot day.
        r, prices, starts, boundary, plan_end = self._run(_dst_fall_back_window, date(2025, 10, 26))
        decision = r.usable_end_idx - r.values.size
        first_two = next(
            i for i in range(boundary - 100, boundary) if starts[i].astimezone(MADRID).hour == 2
        )
        two = next(t for t in range(boundary, plan_end) if starts[t].astimezone(MADRID).hour == 2)
        assert r.values[two - decision] == prices[first_two]
        assert r.diagnostics["slots_from_earlier_day"] == 0

    def test_spring_forward_source_day_borrows_the_missing_hour(self) -> None:
        # 2026-03-29 has no 02:xx; its 14:00 estimates 2026-03-30 and takes 02:xx from the 28th.
        r, prices, starts, boundary, plan_end = self._run(
            _dst_spring_forward_window, date(2026, 3, 29)
        )
        assert r.counts()["estimated"] == 96
        assert r.diagnostics["slots_from_earlier_day"] == 4
        decision = r.usable_end_idx - r.values.size
        two = next(t for t in range(boundary, plan_end) if starts[t].astimezone(MADRID).hour == 2)
        src = next(
            i
            for i, s in enumerate(starts)
            if s.astimezone(MADRID).date() == date(2026, 3, 28) and s.astimezone(MADRID).hour == 2
        )
        assert r.values[two - decision] == prices[src]


# ---------------------------------------------------------------------------
# The dispatcher
# ---------------------------------------------------------------------------


def _scenario():
    """Flat prices except an expensive morning two local days in: only tomorrow matters."""
    start = datetime(2025, 6, 14, 22, tzinfo=UTC)  # local midnight
    prices, load, usable, local_days, starts = _make_dataset(start, 5)
    prices[:] = 0.10
    spike_day = local_days.index(date(2025, 6, 17))
    prices[spike_day + 24 : spike_day + 40] = 0.9
    frozen, _ = build_frozen_load_profile(
        load, local_days, usable, training_end_idx=96, window_days=28, interval_starts_utc=starts
    )
    return prices, load, usable, local_days, starts, frozen


def _roll(prices, load, usable, local_days, starts, frozen, **kwargs):
    return run_rolling_horizon(
        _spec(),
        prices=prices,
        load_kwh=load,
        usable=usable,
        interval_seconds=QUARTER,
        local_days=local_days,
        interval_starts_utc=starts,
        eval_start_idx=96,
        eval_end_idx=len(starts),
        tz=MADRID,
        frozen_load=frozen,
        terminal_value_eur_per_kwh=0.10,
        **kwargs,
    )


def _afternoon(result, day: date) -> dict:
    return next(
        e for e in result.per_event if e["kind"] == "afternoon" and e["local_day"] == str(day)
    )


class TestDispatcher:
    def test_default_outlook_adds_nothing_to_the_output(self) -> None:
        args = _scenario()
        default = _roll(*args)
        explicit = _roll(*args, price_outlook="exact")
        assert "price_outlook" not in default.telemetry
        assert "plan_prices_sha256" not in default.per_event[0]
        assert np.array_equal(default.schedule.charge_kwh, explicit.schedule.charge_kwh)
        assert default.cost_eur == explicit.cost_eur

    @pytest.mark.parametrize("name", NON_ORACLE)
    def test_tomorrows_realized_prices_do_not_reach_the_14_00_plan(self, name: str) -> None:
        """Leakage: perturbing D+1's realized prices leaves the 14:00 plan inputs unchanged."""
        prices, load, usable, local_days, starts, frozen = _scenario()
        day = date(2025, 6, 16)
        perturbed = prices.copy()
        nxt = local_days.index(day + timedelta(days=1))
        perturbed[nxt : nxt + 96] = np.linspace(-0.2, 1.5, 96)
        a = _roll(prices, load, usable, local_days, starts, frozen, price_outlook=name)
        b = _roll(perturbed, load, usable, local_days, starts, frozen, price_outlook=name)
        ea, eb = _afternoon(a, day), _afternoon(b, day)
        assert ea["plan_prices_sha256"] == eb["plan_prices_sha256"]
        assert ea["soc_start_kwh"] == eb["soc_start_kwh"]
        lo, hi = ea["start_idx"] - 96, ea["exec_end_idx"] - 96
        assert np.array_equal(a.schedule.charge_kwh[lo:hi], b.schedule.charge_kwh[lo:hi])
        assert np.array_equal(a.schedule.discharge_kwh[lo:hi], b.schedule.discharge_kwh[lo:hi])

    def test_exact_sees_tomorrow_and_published_only_does_not(self) -> None:
        """Non-vacuity: the scenario's spike is worth charging for at 14:00 under exact."""
        args = _scenario()
        day = date(2025, 6, 16)
        exact = _afternoon(_roll(*args), day)
        live = _roll(*args, price_outlook="published_only")
        ev = _afternoon(live, day)
        assert exact["charge_kwh"] > 0.0
        assert ev["charge_kwh"] == 0.0
        nxt = args[3].index(day + timedelta(days=1))
        assert ev["planned_through_idx"] == nxt
        assert ev["estimated_price_intervals"] == 0
        assert ev["unavailable_price_intervals"] == 96
        assert ev["plan_intervals"] == nxt - ev["start_idx"]
        tel = live.telemetry["price_outlook"]
        assert tel["name"] == "published_only" and tel["oracle"] is False
        assert tel["estimated_price_intervals_planned"] == 0
        assert tel["events_planned_short_of_their_horizon"] >= 3

    def test_refusals(self) -> None:
        args = _scenario()
        with pytest.raises(ValueError, match="unknown price outlook"):
            _roll(*args, price_outlook="nope")
        with pytest.raises(ValueError, match="needs sigma=0"):
            _roll(*args, price_outlook="persistence", sigma=0.1)
        with pytest.raises(ValueError, match="needs sigma=0"):
            _roll(*args, price_outlook="persistence", bias_sigma=0.1)
        with pytest.raises(ValueError, match="legacy 14:00 / 20:30 cadence"):
            _roll(*args, price_outlook="published_only", replan_interval_slots=4)
