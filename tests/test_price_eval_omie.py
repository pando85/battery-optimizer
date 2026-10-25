"""PO-C through the price-eval harness: OMIE estimators, side input plumbing, mask overlap."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from test_benchmark import requires_core
from test_extract_omie import Net, omie_file, run
from test_forecast_eval import PERIODS, _dataset
from test_structured_forecast import World

from battery_optimizer import benchmark as bm
from battery_optimizer import price_eval as pe


class TestOmieEndToEnd:
    @pytest.fixture(scope="class")
    @classmethod
    def runs(cls, tmp_path_factory) -> dict[str, object]:
        requires_core()
        tmp = tmp_path_factory.mktemp("pec")
        dataset, config = _dataset(tmp)
        store = tmp / "store"
        store.mkdir()

        def hourly(day: date):
            def price(i: int) -> tuple[float, float]:
                hour = i - 1
                level = 0.05 if hour < 6 else (0.30 if hour >= 18 else 0.15)
                return 0.0, round(level * 1000.0 - 5.0, 2)

            return 200, omie_file(day, price=price)

        first = date(2025, 3, 3)
        files = {first + timedelta(days=k): hourly(first + timedelta(days=k)) for k in range(24)}
        run(store, Net(files), first, first + timedelta(days=23))
        out = pe.run_price_eval(
            dataset=dataset,
            config_path=config,
            name="po-c",
            outlooks=["persistence", "omie_period", "omie_period_month"],
            periods=PERIODS[:1],
            resamples=5000,
            now=datetime(2026, 1, 1, tzinfo=UTC),
            data_dir=store,
            label="plumbing test",
        )
        return {
            "dataset": dataset,
            "run": out,
            "summary": json.loads((out / pe.SUMMARY_JSON_NAME).read_text()),
            "prereg": json.loads((out / pe.PREREGISTRATION_NAME).read_text()),
        }

    def test_the_preregistration_records_the_omie_rule_and_the_side_input(self, runs) -> None:
        prereg = runs["prereg"]
        assert prereg["label"] == "plumbing test"
        assert "13:30" in prereg["omie_rule"]
        assert prereg["omie_side_input"]["kind"] == "omie_es_day_ahead"
        names = [o["name"] for o in prereg["outlooks"]]
        assert names == [
            "exact",
            "published_only",
            "persistence",
            "omie_period",
            "omie_period_month",
        ]
        omie = next(o for o in prereg["outlooks"] if o["name"] == "omie_period")
        assert omie["oracle"] is False and omie["inputs_usable_from_local"] == "13:30"

    def test_the_estimate_beats_persistence_on_the_synthetic_relation(self, runs) -> None:
        period = runs["summary"]["periods"]["a"]["outlooks"]
        omie = period["omie_period"]["price_metrics"]
        persistence = period["persistence"]["price_metrics"]
        assert omie["estimated_slots"] == persistence["estimated_slots"]
        assert omie["mae_eur_per_kwh"] < persistence["mae_eur_per_kwh"]
        assert omie["full_days"] == 6 and omie["fallback_events"] == {}
        assert "climate_mask" not in omie

    def test_each_estimator_has_its_own_run_with_the_side_in_its_identity(self, runs) -> None:
        period = runs["summary"]["periods"]["a"]["outlooks"]
        run_id = period["omie_period"]["benchmark_run_id"]
        assert run_id != period["persistence"]["benchmark_run_id"]
        payload = json.loads(
            (runs["dataset"] / bm.BENCHMARKS_DIRNAME / run_id / bm.RESULTS_JSON_NAME).read_text()
        )
        frozen = payload["frozen_inputs"]
        assert frozen["rolling_horizon_price_outlook"] == "omie_period"
        assert frozen["rolling_horizon_price_outlook_side"]["kind"] == "omie_es_day_ahead"
        block = payload["rolling_horizon"]["price_outlook"]
        assert (
            block["side_input"]["fingerprint"]
            == (frozen["rolling_horizon_price_outlook_side"]["fingerprint"])
        )
        assert "omie_rule" in block

    def test_gate_gc_is_preregistered_and_undetermined_without_the_mask_metrics(self, runs) -> None:
        gc = runs["prereg"]["gate_gc"]
        assert gc["id"] == "GC" and gc["baseline"] == "persistence"
        assert gc["candidates"] == ["omie_period", "omie_period_month"]
        assert "NOT computed" in gc["rule_hour_metrics"]
        verdict = runs["summary"]["gate_gc"]["omie_period"]
        assert verdict["passed"] is None  # the rule-hour criteria need a feature set
        checks = verdict["periods"]["a"]
        assert checks["spearman_mean"] is not None and checks["mask_jaccard"] is None
        md = (runs["run"] / pe.SUMMARY_MD_NAME).read_text()
        assert "Gate GC" in md and "UNDETERMINED" in md

    def test_an_empty_omie_store_is_a_recorded_fallback_not_a_failure(self, tmp_path) -> None:
        requires_core()
        dataset, config = _dataset(tmp_path)
        run_dir = bm.benchmark_dataset(
            dataset=dataset,
            config_path=config,
            train_end=PERIODS[0].train_end,
            eval_end=PERIODS[0].eval_end,
            train_days=PERIODS[0].train_days,
            policies="no_battery,rolling_horizon",
            rolling_horizon_price_outlook="omie_period",
            data_dir=tmp_path / "empty-store",
        )
        payload = json.loads((run_dir / bm.RESULTS_JSON_NAME).read_text())
        block = payload["rolling_horizon"]["price_outlook"]
        assert block["side_input"]["slots_without_price"] == block["side_input"]["slots"]
        assert block["estimated_price_intervals_planned"] == 0
        assert sum(block["fallback_events"].values()) == 7
        assert set(block["fallback_events"]) == {"fit_rows_short"}


@pytest.fixture(scope="module")
def setup():
    world = World()
    days = world.days
    j = next(i for i, (d, _lo, _hi) in enumerate(days) if d == date(2025, 11, 5))
    prev = next(i for i, (d, _lo, _hi) in enumerate(days) if d == date(2025, 11, 4))
    lo, hi = days[j][1], days[j][2]
    return SimpleNamespace(
        world=world,
        data=SimpleNamespace(days=days, prices=world.prices),
        cut=days[prev][1] + 14 * 4,  # 14:00 local on the day before
        j=j,
        slots=np.arange(lo, hi),
        plan_end=hi,
    )


class TestClimateMaskOverlap:
    """The mask metric on the structured forecaster's synthetic world."""

    def _add(self, s, values, feat=None) -> dict:
        acc = pe._MaskAccumulator()
        acc.add(
            feat=feat or s.world.side(),
            data=s.data,
            cut=s.cut,
            plan_end=s.plan_end,
            est_slots=s.slots,
            est_values=values,
            j=s.j,
        )
        return acc.result()

    def test_identical_prices_give_identical_masks_and_a_perfect_overlap(self, setup) -> None:
        out = self._add(setup, setup.world.prices[setup.slots])
        assert out["days_with_both_schedules"] == 1 and out["mean_jaccard"] == 1.0
        assert out["days_identical_mask"] == 1
        assert out["climate_energy_hit_rate_estimated_mask"] == pytest.approx(1.0, abs=1e-9)
        assert (
            out["climate_energy_hit_rate_estimated_mask"]
            == (out["climate_energy_hit_rate_realized_mask"])
        )

    def test_inverted_prices_pick_other_hours_and_lose_the_energy(self, setup) -> None:
        prices = setup.world.prices
        inverted = prices.max() + prices.min() - prices[setup.slots]
        out = self._add(setup, inverted)
        assert out["mean_jaccard"] < 1.0
        assert (
            out["climate_energy_hit_rate_estimated_mask"]
            < out["climate_energy_hit_rate_realized_mask"]
        )

    def test_a_disabled_rule_is_counted_and_not_scored(self, setup) -> None:
        feat = setup.world.side(series={"climate_enable": np.zeros_like(setup.world.enable)})
        out = self._add(setup, setup.world.prices[setup.slots], feat=feat)
        assert out["days"] == 1 and out["days_with_both_schedules"] == 0
        assert out["days_without_schedule"] == {"rule_disabled": 1}


class TestGateGc:
    @staticmethod
    def _metrics(rho: float, c10: float, jac: float | None, hit: float | None) -> dict:
        out = {
            "estimated_slots": 1,
            "spearman_mean": rho,
            "spearman_median": rho,
            "cheapest_10h_overlap_mean": c10,
        }
        if jac is not None:
            out["climate_mask"] = {
                "mean_jaccard": jac,
                "climate_energy_hit_rate_estimated_mask": hit,
            }
        return out

    def _periods(self, est_2026: dict) -> dict:
        base = self._metrics(0.8, 0.7, 0.7, 0.78)
        good = self._metrics(0.97, 0.9, 0.9, 0.9)
        return {
            "a": {
                "outlooks": {"persistence": {"price_metrics": base}, "est": {"price_metrics": good}}
            },
            "b": {
                "outlooks": {
                    "persistence": {"price_metrics": base},
                    "est": {"price_metrics": est_2026},
                }
            },
        }

    def test_it_passes_only_when_every_criterion_beats_persistence_in_every_period(self) -> None:
        good = self._metrics(0.97, 0.9, 0.9, 0.9)
        out = pe._gc_evaluate(self._periods(good), ["est"], ["a", "b"])
        assert out["est"]["passed"] is True
        worse_mask = self._metrics(0.97, 0.9, 0.9, 0.7)
        out = pe._gc_evaluate(self._periods(worse_mask), ["est"], ["a", "b"])
        assert out["est"]["passed"] is False
        assert out["est"]["periods"]["b"]["mask_energy_hit_rate"] is False
        assert out["est"]["periods"]["a"]["mask_energy_hit_rate"] is True

    def test_a_tie_does_not_beat(self) -> None:
        tie = self._metrics(0.8, 0.9, 0.9, 0.9)
        out = pe._gc_evaluate(self._periods(tie), ["est"], ["a", "b"])
        assert (
            out["est"]["passed"] is False and out["est"]["periods"]["b"]["spearman_mean"] is False
        )

    def test_missing_mask_metrics_leave_it_undetermined_not_failed(self) -> None:
        no_mask = self._metrics(0.97, 0.9, None, None)
        out = pe._gc_evaluate(self._periods(no_mask), ["est"], ["a", "b"])
        assert out["est"]["passed"] is None
