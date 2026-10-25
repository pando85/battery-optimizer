"""Price-outlook evaluation: benchmark plumbing, CLI and a synthetic end-to-end run.

The end-to-end run uses a small synthetic dataset (no real data) and checks what the
harness exists for: the pre-registration is on disk before the first benchmark runs,
the references against themselves are exactly zero, the default outlook reproduces a
pre-seam benchmark run bit for bit, and every artifact stays inside the dataset directory.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from test_benchmark import requires_core
from test_forecast_eval import PERIODS, _dataset

from battery_optimizer import benchmark as bm
from battery_optimizer import cli
from battery_optimizer import forecast_eval as fe
from battery_optimizer import price_eval as pe


def _bench(dataset: Path, config: Path, **kwargs) -> dict:
    run = bm.benchmark_dataset(
        dataset=dataset,
        config_path=config,
        train_end=PERIODS[0].train_end,
        eval_end=PERIODS[0].eval_end,
        train_days=PERIODS[0].train_days,
        policies="no_battery,rolling_horizon",
        **kwargs,
    )
    return json.loads((run / bm.RESULTS_JSON_NAME).read_text())


class TestBenchmarkPlumbing:
    def test_outlook_enters_the_identity_only_when_not_default(self, tmp_path: Path) -> None:
        requires_core()
        dataset, config = _dataset(tmp_path)
        default = _bench(dataset, config)
        live = _bench(dataset, config, rolling_horizon_price_outlook="published_only")
        assert "rolling_horizon_price_outlook" not in default["frozen_inputs"]
        assert live["frozen_inputs"]["rolling_horizon_price_outlook"] == "published_only"
        assert live["run_id"] != default["run_id"]
        assert default["rolling_horizon"]["price_outlook"]["name"] == "exact"
        assert default["rolling_horizon"]["price_outlook"]["oracle"] is True
        assert (
            "price_outlook" in live["rolling_horizon"]
            and "price outlook 'published_only'"
            in (live["rolling_horizon"]["information_set"]["prices"])
        )
        assert live["rolling_horizon"]["price_outlook"]["events_planned_short_of_their_horizon"]
        assert (
            live["policies"]["no_battery"]["cost_eur"]
            == default["policies"]["no_battery"]["cost_eur"]
        )

    def test_impossible_combinations_are_refused_before_loading(self, tmp_path: Path) -> None:
        for kwargs, match in (
            ({"rolling_horizon_price_outlook": "nope"}, "unknown price outlook"),
            (
                {"rolling_horizon_price_outlook": "persistence", "rolling_horizon_sigma": 0.1},
                "needs rolling_horizon_sigma=0",
            ),
            (
                {
                    "rolling_horizon_price_outlook": "persistence",
                    "rolling_horizon_replan_interval_slots": 4,
                },
                "legacy 14:00",
            ),
        ):
            with pytest.raises(bm.BenchmarkInputError, match=match):
                bm.benchmark_dataset(
                    dataset=tmp_path / "missing",
                    config_path=tmp_path / "missing.yaml",
                    train_end=PERIODS[0].train_end,
                    policies="rolling_horizon",
                    **kwargs,
                )

    @staticmethod
    def _capture(monkeypatch, argv: list[str]) -> dict:
        monkeypatch.setattr(bm, "core_available", lambda: True)
        captured: dict = {}

        def fake(**kw):
            captured.update(kw)
            return Path("/tmp/run")

        monkeypatch.setattr(bm, "benchmark_dataset", fake)
        assert cli.main(argv) == 0
        return captured

    def test_cli_flag(self, monkeypatch) -> None:
        argv = ["benchmark", "--dataset", "/tmp/ds", "--train-end", "2026-01-01T00:00:00+00:00"]
        assert self._capture(monkeypatch, argv)["rolling_horizon_price_outlook"] == "exact"
        flagged = [*argv, "--rolling-horizon-price-outlook", "persistence"]
        assert self._capture(monkeypatch, flagged)["rolling_horizon_price_outlook"] == (
            "persistence"
        )


class TestEndToEnd:
    @pytest.fixture(scope="class")
    @classmethod
    def runs(cls, tmp_path_factory) -> dict[str, object]:
        requires_core()
        tmp = tmp_path_factory.mktemp("pe")
        dataset, config = _dataset(tmp)
        # A pre-seam style reference: the forecast-eval benchmark params, LF-0, period a.
        reference = bm.benchmark_dataset(
            dataset=dataset,
            config_path=config,
            train_end=PERIODS[0].train_end,
            eval_end=PERIODS[0].eval_end,
            train_days=PERIODS[0].train_days,
            **fe.BENCHMARK_PARAMS,
        )
        ref_summary = tmp / "fe-summary.json"
        ref_summary.write_text(
            json.dumps({"provenance": {"benchmark_runs": {"a": {"trailing_mean": reference.name}}}})
        )
        original = bm.benchmark_dataset
        seen: list[bool] = []

        def spy(**kwargs):
            run_dirs = list((dataset / pe.EVAL_DIRNAME).glob("*/"))
            seen.append(all((d / pe.PREREGISTRATION_NAME).is_file() for d in run_dirs))
            return original(**kwargs)

        mp = pytest.MonkeyPatch()
        mp.setattr(bm, "benchmark_dataset", spy)
        try:
            run = pe.run_price_eval(
                dataset=dataset,
                config_path=config,
                name="po-a",
                outlooks=["persistence"],
                periods=PERIODS,
                reference_summary=ref_summary,
                resamples=5000,
                now=datetime(2026, 1, 1, tzinfo=UTC),
            )
        finally:
            mp.undo()
        return {
            "dataset": dataset,
            "run": run,
            "seen": seen,
            "summary": json.loads((run / pe.SUMMARY_JSON_NAME).read_text()),
            "prereg": json.loads((run / pe.PREREGISTRATION_NAME).read_text()),
        }

    def test_layout_and_preregistration_first(self, runs) -> None:
        run = runs["run"]
        assert run.parent == runs["dataset"] / pe.EVAL_DIRNAME
        assert run.name == "po-a-20260101T000000Z"
        assert {p.name for p in run.iterdir()} == {
            pe.PREREGISTRATION_NAME,
            pe.SUMMARY_JSON_NAME,
            pe.SUMMARY_MD_NAME,
        }
        assert runs["seen"] == [True] * 6
        prereg = runs["prereg"]
        assert [o["name"] for o in prereg["outlooks"]] == ["exact", "published_only", "persistence"]
        assert prereg["gate"]["id"] == "GA"
        assert prereg["benchmark_params"]["policies"] == ["no_battery", "rolling_horizon"]
        for key in ("git_commit", "git_dirty", "tariff_config_hash", "eval_config_hash"):
            assert key in prereg

    def test_references_and_gap(self, runs) -> None:
        for label in ("a", "b"):
            period = runs["summary"]["periods"][label]
            outlooks = period["outlooks"]
            for ref in ("exact", "published_only"):
                row = outlooks[ref]["vs"][ref]
                assert (row["point_eur"], row["ci_low_eur"], row["ci_high_eur"]) == (0.0, 0.0, 0.0)
            gap = period["ga_gap"]["point_eur"]
            assert gap == pytest.approx(
                outlooks["published_only"]["cost_eur"] - outlooks["exact"]["cost_eur"]
            )
            if gap:
                assert outlooks["exact"]["share_of_ga_gap"] == pytest.approx(1.0)
                assert outlooks["published_only"]["share_of_ga_gap"] == 0.0
            ids = {o["benchmark_run_id"] for o in outlooks.values()}
            assert len(ids) == 3
            for run_id in ids:
                assert (runs["dataset"] / bm.BENCHMARKS_DIRNAME / run_id).is_dir()

    def test_action_differences_and_price_metrics(self, runs) -> None:
        period = runs["summary"]["periods"]["a"]["outlooks"]
        diff = period["persistence"]["action_differences"]["exact"]
        assert diff["afternoon_events"] == 7
        assert diff["afternoon_slots"] == 7 * 26
        assert 0 <= diff["slots_differing"] <= diff["afternoon_slots"]
        assert diff == period["exact"]["action_differences"]["persistence"]
        exact = period["exact"]["price_metrics"]
        assert exact["mae_eur_per_kwh"] == 0.0 and exact["bias_eur_per_kwh"] == 0.0
        assert exact["cheapest_10h_overlap_mean"] == 1.0
        assert period["published_only"]["price_metrics"]["estimated_slots"] == 0
        persistence = period["persistence"]["price_metrics"]
        # The last 14:00 event's tomorrow is cut by the window end after its first hour.
        assert persistence["estimated_slots"] == 6 * 96 + 4
        assert persistence["full_days"] == 6 and persistence["partial_days_skipped"] == 1
        assert persistence["mae_eur_per_kwh"] > 0.0
        assert -1.0 <= persistence["spearman_mean"] <= 1.0
        assert 0.0 <= persistence["dearest_4h_overlap_mean"] <= 1.0

    def test_default_outlook_reproduces_the_reference_run(self, runs) -> None:
        repro = runs["summary"]["periods"]["a"]["reference_reproduction"]
        assert repro["bit_identical"] is True
        assert repro["this_run_cost_eur"] == repro["reference_cost_eur"]
        assert "reference_reproduction" not in runs["summary"]["periods"]["b"]
        md = (runs["run"] / pe.SUMMARY_MD_NAME).read_text()
        assert "bit-identical yes" in md and "GA gap" in md

    def test_the_cli_drives_the_harness(self, runs, monkeypatch, capsys) -> None:
        captured: dict[str, object] = {}

        def fake(**kwargs):
            captured.update(kwargs)
            return runs["run"]

        monkeypatch.setattr(pe, "run_price_eval", fake)
        argv = ["price-eval", "--dataset", str(runs["dataset"]), "--name", "x"]
        assert cli.main([*argv, "--period", "2025"]) == 0
        assert captured["outlooks"] == list(pe.DEFAULT_OUTLOOKS)
        assert [p.label for p in captured["periods"]] == ["2025"]
        assert "price-eval: run directory" in capsys.readouterr().out
        assert cli.main([*argv, "--period", "1999"]) == 1
        assert "unknown period" in capsys.readouterr().err

    def test_bad_inputs_are_refused(self, runs) -> None:
        with pytest.raises(ValueError, match="name must match"):
            pe.run_price_eval(dataset=runs["dataset"], config_path="x", name="Bad Name")
        with pytest.raises(ValueError, match="resamples"):
            pe.run_price_eval(dataset=runs["dataset"], config_path="x", name="ok", resamples=10)
        with pytest.raises(ValueError, match="unknown price outlook"):
            pe.run_price_eval(dataset=runs["dataset"], config_path="x", name="ok", outlooks=["n"])
