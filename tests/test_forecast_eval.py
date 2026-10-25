"""Forecast-evaluation harness: statistics, benchmark plumbing and a synthetic end-to-end run.

The end-to-end run uses a small synthetic dataset (no real data) and checks the
properties the harness exists for: the pre-registration is on disk before the first
candidate runs, LF-0 against itself is exactly zero, a second run that names the first
as its "P0 reference" reproduces it bit for bit, and every artifact stays inside the
dataset directory.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from test_benchmark import (
    make_analysis_table,
    make_tariff_config,
    requires_core,
    write_synthetic_dataset,
    write_tariff_config,
)

from battery_optimizer import benchmark as bm
from battery_optimizer import cli
from battery_optimizer import forecast_eval as fe

# 2025-03-02T23:00Z is local midnight (CET); the window ends before the 2025-03-30 DST day.
START = datetime(2025, 3, 2, 23, tzinfo=UTC)
DAYS = 24
PERIODS = (
    fe.Period("a", datetime(2025, 3, 10, tzinfo=UTC), datetime(2025, 3, 17, tzinfo=UTC), 7),
    fe.Period("b", datetime(2025, 3, 17, tzinfo=UTC), datetime(2025, 3, 25, tzinfo=UTC), 7),
)


def _dataset(tmp_path: Path) -> tuple[Path, Path]:
    n = DAYS * 96
    rng = np.random.default_rng(7)
    hour = (np.arange(n) % 96) // 4
    load = 0.15 + 0.5 * (hour >= 18) + rng.random(n) * 0.3
    prices = np.where(hour < 6, 0.05, np.where(hour >= 18, 0.30, 0.15)) + rng.random(n) * 0.02
    table = make_analysis_table(
        start=START, intervals=n, gross_kwh=load.tolist(), prices=prices.tolist()
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    config = tmp_path / "tariff.yaml"
    write_tariff_config(config, make_tariff_config())
    return write_synthetic_dataset(tmp_path, table, config_path=config), config


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


class TestCandidates:
    def test_name_alone_is_the_default_params(self) -> None:
        c = fe.parse_candidate("trailing_mean")
        assert (c.label, c.params, c.overrides) == ("trailing_mean", {"window_days": 28}, {})
        assert c.benchmark_kwargs() == {
            "load_forecaster": "trailing_mean",
            "load_forecaster_params": None,
            "day_ahead_forecast_window_days": 28,
        }

    def test_overrides_are_parsed_and_labelled(self) -> None:
        c = fe.parse_candidate("trailing_mean:window_days=7")
        assert c.label == "trailing_mean[window_days=7]"
        assert c.params == {"window_days": 7}
        assert c.benchmark_kwargs()["day_ahead_forecast_window_days"] == 7
        # Restating a default is the default candidate, not a new one.
        assert fe.parse_candidate("trailing_mean:window_days=28").label == "trailing_mean"

    def test_unknown_names_and_params_are_refused(self) -> None:
        with pytest.raises(ValueError, match="unknown load forecaster"):
            fe.parse_candidate("nope")
        with pytest.raises(ValueError, match="no param 'alpha'"):
            fe.parse_candidate("trailing_mean:alpha=1")
        with pytest.raises(ValueError, match="key=value"):
            fe.parse_candidate("trailing_mean:window_days")


class TestBootstrap:
    def test_identical_runs_give_exactly_zero(self) -> None:
        out = fe.block_bootstrap(np.zeros(30), resamples=5000, seed=1)
        assert (out["point_eur"], out["ci_low_eur"], out["ci_high_eur"]) == (0.0, 0.0, 0.0)
        assert out["blocks"] == 5  # 4 full weeks and one 2-day block

    def test_equal_weeks_have_no_spread_and_draws_are_seeded(self) -> None:
        out = fe.block_bootstrap(np.full(28, 0.5), resamples=5000, seed=1)
        assert out["ci_low_eur"] == pytest.approx(14.0)
        assert out["ci_high_eur"] == pytest.approx(14.0)
        rng = np.random.default_rng(3)
        daily = rng.normal(0.1, 1.0, 70)
        a = fe.block_bootstrap(daily, resamples=5000, seed=(9, 0))
        b = fe.block_bootstrap(daily, resamples=5000, seed=(9, 0))
        assert a == b
        assert a["ci_low_eur"] < a["point_eur"] < a["ci_high_eur"]

    def test_spearman_needs_three_candidates(self) -> None:
        assert fe._spearman([1.0, 2.0], [3.0, 4.0])["status"].startswith("not computed")
        agree = fe._spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        assert agree["rho"] == pytest.approx(1.0) and agree["same_order"] is True
        disagree = fe._spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0])
        assert disagree["rho"] == pytest.approx(-1.0) and disagree["same_order"] is False


# ---------------------------------------------------------------------------
# Benchmark plumbing the harness relies on
# ---------------------------------------------------------------------------


class TestBenchmarkEvalEnd:
    def test_eval_end_equals_a_dataset_that_ends_there(self, tmp_path: Path) -> None:
        requires_core()
        dataset, config = _dataset(tmp_path / "full")
        period = PERIODS[0]
        # The same rows, built as a dataset that simply ends at eval_end.
        full = bm.load_dataset(dataset).table
        kept = sum(1 for s in full.column("interval_start").to_pylist() if s < period.eval_end)
        short_dir = tmp_path / "short"
        short_dir.mkdir()
        short = write_synthetic_dataset(short_dir, full.slice(0, kept), config_path=config)
        kwargs = {
            "config_path": config,
            "train_end": period.train_end,
            "train_days": period.train_days,
            "policies": "no_battery,day_ahead,rolling_horizon",
        }
        cut_run = bm.benchmark_dataset(dataset=dataset, eval_end=period.eval_end, **kwargs)
        short_run = bm.benchmark_dataset(dataset=short, **kwargs)
        a = json.loads((cut_run / bm.RESULTS_JSON_NAME).read_text())
        b = json.loads((short_run / bm.RESULTS_JSON_NAME).read_text())
        for policy in ("no_battery", "day_ahead", "rolling_horizon"):
            assert a["policies"][policy]["cost_eur"] == b["policies"][policy]["cost_eur"]
        assert a["frozen_inputs"]["eval_end"] == "2025-03-17T00:00:00Z"
        assert "eval_end" not in b["frozen_inputs"]
        assert a["evaluation"]["horizon_end"] == "2025-03-17T00:00:00Z"
        sa = pq.read_table(cut_run / bm.SCHEDULES_PARQUET_NAME)
        sb = pq.read_table(short_run / bm.SCHEDULES_PARQUET_NAME)
        assert sa.column("interval_cost_eur").equals(sb.column("interval_cost_eur"))

    def test_eval_end_must_follow_train_end(self, tmp_path: Path) -> None:
        requires_core()
        dataset, config = _dataset(tmp_path)
        with pytest.raises(bm.BenchmarkInputError, match="not after train_end"):
            bm.benchmark_dataset(
                dataset=dataset,
                config_path=config,
                train_end=PERIODS[0].train_end,
                eval_end=PERIODS[0].train_end,
                policies="no_battery",
            )

    def test_forecaster_params_enter_the_identity(self, tmp_path: Path) -> None:
        requires_core()
        dataset, config = _dataset(tmp_path)
        with pytest.raises(bm.BenchmarkInputError, match="no param 'alpha'"):
            bm.benchmark_dataset(
                dataset=dataset,
                config_path=config,
                train_end=PERIODS[0].train_end,
                policies="day_ahead",
                load_forecaster_params={"alpha": 1},
            )


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.fixture(scope="class")
    @classmethod
    def runs(cls, tmp_path_factory) -> dict[str, object]:
        requires_core()
        tmp = tmp_path_factory.mktemp("fe")
        dataset, config = _dataset(tmp)
        original = bm.benchmark_dataset
        seen: list[bool] = []

        def spy(**kwargs):
            run_dirs = list((dataset / fe.EVAL_DIRNAME).glob("*/"))
            seen.append(all((d / fe.PREREGISTRATION_NAME).is_file() for d in run_dirs))
            return original(**kwargs)

        mp = pytest.MonkeyPatch()
        mp.setattr(bm, "benchmark_dataset", spy)
        try:
            first = fe.run_forecast_eval(
                dataset=dataset,
                config_path=config,
                name="g1",
                candidates=["perfect", "trailing_mean:window_days=3"],
                periods=PERIODS,
                resamples=5000,
                now=datetime(2026, 1, 1, tzinfo=UTC),
            )
        finally:
            mp.undo()
        summary = json.loads((first / fe.SUMMARY_JSON_NAME).read_text())
        # A "P0 summary" in the P0 layout, built from the first run's own numbers.
        p0 = {
            "periods": [
                {
                    "period": p.label,
                    "evaluation": p.window,
                    "runs": {},
                    "policies": {
                        policy: {
                            "lf0_cost_eur": summary["periods"][p.label]["economic"][
                                "trailing_mean"
                            ][policy]["cost_eur"],
                            "perfect_cost_eur": summary["periods"][p.label]["economic"]["perfect"][
                                policy
                            ]["cost_eur"],
                            "gap_eur": summary["in_run_p0_gap_eur"][p.label][policy],
                        }
                        for policy in fe.EVAL_POLICIES
                    },
                }
                for p in PERIODS
            ]
        }
        p0_path = tmp / "p0-summary.json"
        p0_path.write_text(json.dumps(p0))
        second = fe.run_forecast_eval(
            dataset=dataset,
            config_path=config,
            name="g1",
            candidates=["trailing_mean", "perfect"],
            periods=PERIODS,
            p0_summary=p0_path,
            resamples=5000,
            now=datetime(2026, 1, 2, tzinfo=UTC),
        )
        return {
            "dataset": dataset,
            "first": first,
            "second": second,
            "summary": summary,
            "summary2": json.loads((second / fe.SUMMARY_JSON_NAME).read_text()),
            "seen": seen,
        }

    def test_layout_and_preregistration_first(self, runs) -> None:
        first = runs["first"]
        assert first.parent == runs["dataset"] / fe.EVAL_DIRNAME
        assert first.name == "g1-20260101T000000Z"
        assert {p.name for p in first.iterdir()} == {
            fe.PREREGISTRATION_NAME,
            fe.SUMMARY_JSON_NAME,
            fe.SUMMARY_MD_NAME,
        }
        # 3 candidates x 2 periods benchmark calls, each after the preregistration existed.
        assert runs["seen"] == [True] * 6
        prereg = json.loads((first / fe.PREREGISTRATION_NAME).read_text())
        assert prereg["kind"] == "preregistration"
        assert [c["label"] for c in prereg["candidates"]] == [
            "trailing_mean",
            "perfect",
            "trailing_mean[window_days=3]",
        ]
        assert prereg["gate"]["id"] == "G2"
        assert prereg["gate"]["primary_policy"] == "rolling_horizon"
        assert prereg["gate"]["min_share_of_p0_gap"] == 0.10
        assert prereg["bootstrap"] == {
            **prereg["bootstrap"],
            "block_days": 7,
            "resamples": 5000,
            "seed": fe.DEFAULT_SEED,
        }
        assert [p["label"] for p in prereg["periods"]] == ["a", "b"]
        for key in ("git_commit", "git_dirty", "tariff_config_hash", "eval_config_hash"):
            assert key in prereg

    def test_reference_against_itself_is_exactly_zero(self, runs) -> None:
        for period in runs["summary"]["sanity"]["reference_vs_itself"].values():
            for row in period.values():
                assert row == {"point_eur": 0.0, "ci_low_eur": 0.0, "ci_high_eur": 0.0}

    def test_accuracy_breakdowns(self, runs) -> None:
        period = runs["summary"]["periods"]["a"]
        perfect = period["candidates"]["perfect"]["accuracy"]
        lf0 = period["candidates"]["trailing_mean"]["accuracy"]
        for policy in fe.EVAL_POLICIES:
            assert perfect[policy]["overall"]["mae_kwh"] == 0.0
            assert perfect[policy]["overall"]["price_weighted_abs_error_eur"] == 0.0
            overall = lf0[policy]["overall"]
            assert overall["mae_kwh"] > 0.0
            assert set(lf0[policy]["by_price_tercile"]) == {"T1", "T2", "T3"}
            assert sum(c["slots"] for c in lf0[policy]["by_horizon"].values()) == overall["slots"]
            assert sum(c["slots"] for c in lf0[policy]["by_month"].values()) == overall["slots"]
            assert set(lf0[policy]["by_day_type"]) == {"weekday", "weekend"}
        assert set(lf0["day_ahead"]["by_horizon"]) == {"0-6h", "6-24h"}
        assert set(lf0["rolling_horizon"]["by_horizon"]) == {"0-6h", "6-24h", "24-36h"}
        # 7 local days of 00:00 decisions (the UTC window opens at 01:00 local).
        assert lf0["day_ahead"]["decisions"] == 8
        assert lf0["rolling_horizon"]["decisions"] == 14

    def test_economics_gate_and_proxy(self, runs) -> None:
        summary = runs["summary"]
        for label in ("a", "b"):
            period = summary["periods"][label]
            for policy in fe.EVAL_POLICIES:
                check = period["proxy_check"][policy]
                assert check["status"] == "computed" and check["candidates"] == 3
                cmp = period["economic"]["perfect"][policy]["vs_reference"]
                # Without a P0 summary the gap is the in-run perfect one, so perfect
                # captures exactly 100 % of it.
                assert cmp["p0_gap_eur"] == summary["in_run_p0_gap_eur"][label][policy]
                assert cmp["share_of_p0_gap"] == pytest.approx(1.0)
                assert cmp["ci_low_eur"] <= cmp["point_eur"] <= cmp["ci_high_eur"]
        gate = summary["gate"]
        assert gate["trailing_mean"]["passed"] is None
        assert gate["perfect"]["eligible"] is False
        assert gate["trailing_mean[window_days=3]"]["eligible"] is True
        assert gate["trailing_mean[window_days=3]"]["passed"] in (True, False)
        prov = summary["provenance"]
        assert set(prov["benchmark_runs"]) == {"a", "b"}
        for runs_of_period in prov["benchmark_runs"].values():
            for run_id in runs_of_period.values():
                assert (runs["dataset"] / bm.BENCHMARKS_DIRNAME / run_id).is_dir()
        assert set(summary["folds"]["a"][0]) == {
            "fold",
            "day_ahead_decisions",
            "rh_decisions",
            "fit_before_utc",
        }
        assert summary["runtime_s"]["perfect"]["a"]["benchmark_s"] >= 0.0

    def test_a_rerun_reproduces_its_reference_bit_for_bit(self, runs) -> None:
        summary2 = runs["summary2"]
        repro = summary2["sanity"]["p0_reproduction"]
        assert repro["bit_identical"] is True
        assert set(repro["periods"]) == {"a", "b"}
        # Same code, same params: the LF-0 and perfect benchmark runs are the same runs.
        first = runs["summary"]["provenance"]["benchmark_runs"]
        second = summary2["provenance"]["benchmark_runs"]
        for label in ("a", "b"):
            assert second[label]["trailing_mean"] == first[label]["trailing_mean"]
            assert second[label]["perfect"] == first[label]["perfect"]
        prereg = json.loads((runs["second"] / fe.PREREGISTRATION_NAME).read_text())
        assert prereg["gate"]["p0_gap_eur"] == {
            p: runs["summary"]["in_run_p0_gap_eur"][p]["rolling_horizon"] for p in ("a", "b")
        }
        md = (runs["second"] / fe.SUMMARY_MD_NAME).read_text()
        assert "P0 reproduction bit-identical: yes" in md

    def test_the_cli_drives_the_harness(self, runs, monkeypatch, capsys) -> None:
        captured: dict[str, object] = {}

        def fake(**kwargs):
            captured.update(kwargs)
            return runs["first"]

        monkeypatch.setattr(fe, "run_forecast_eval", fake)
        code = cli.main(
            [
                "forecast-eval",
                "--dataset",
                str(runs["dataset"]),
                "--config",
                str(runs["dataset"]),
                "--name",
                "x",
                "--candidate",
                "trailing_mean:window_days=7",
                "--period",
                "2025",
            ]
        )
        assert code == 0
        assert captured["candidates"] == ["trailing_mean:window_days=7"]
        assert [p.label for p in captured["periods"]] == ["2025"]
        assert captured["resamples"] == fe.DEFAULT_RESAMPLES
        out = capsys.readouterr().out
        assert "gate G2 perfect: " in out and "trailing_mean[window_days=3]" in out

    def test_unknown_period_and_bad_names_are_refused(self, runs, capsys) -> None:
        code = cli.main(
            ["forecast-eval", "--dataset", str(runs["dataset"]), "--name", "x", "--period", "1999"]
        )
        assert code == 1
        assert "unknown period" in capsys.readouterr().err
        with pytest.raises(ValueError, match="name must match"):
            fe.run_forecast_eval(
                dataset=runs["dataset"], config_path="x", name="Bad Name", candidates=[]
            )
        with pytest.raises(ValueError, match="resamples"):
            fe.run_forecast_eval(
                dataset=runs["dataset"], config_path="x", name="ok", candidates=[], resamples=10
            )


def test_default_periods_are_the_p0_windows() -> None:
    assert [p.window for p in fe.DEFAULT_PERIODS] == [
        "2025-01-01T00:00:00Z .. 2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z .. 2026-10-01T00:00:00Z",
    ]
    assert [p.train_days for p in fe.DEFAULT_PERIODS] == [366, 365]
    assert fe.DEFAULT_PERIODS[0].train_end - timedelta(days=366) == datetime(2024, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Gate G4 and the structured forecaster's side inputs
# ---------------------------------------------------------------------------


def _write_feature_set(data_dir: Path, *, n: int) -> str:
    """A tiny feature set aligned with the synthetic dataset grid; returns its id."""
    import hashlib

    import pyarrow as pa

    rng = np.random.default_rng(11)
    starts = [START + timedelta(minutes=15 * i) for i in range(n)]
    hour = np.array([(s.hour + 1) % 24 for s in starts])
    columns: dict[str, object] = {
        "interval_start": pa.array(starts, type=pa.timestamp("us", tz="UTC")),
        "hp_heat_kwh": 0.2 * (hour < 8) + rng.random(n) * 0.01,
        "hp_cool_kwh": np.zeros(n),
        "hp_dhw_kwh": rng.random(n) * 0.01,
        "outdoor_temp_mean_c": 8.0 + rng.random(n),
        "climate_enable_at_start": np.ones(n),
        "climate_min_hours_at_start": np.full(n, 6.0),
        "dhw_enable_at_start": np.ones(n),
    }
    table = pa.table(columns)
    feature_id = "ab" * 32
    base = data_dir / "analysis" / "v1" / "features" / feature_id
    (base / "parts").mkdir(parents=True)
    part = base / "parts" / "features-test.parquet"
    pq.write_table(table, part)
    manifest = {
        "feature_set_id": feature_id,
        "part_files": [
            {
                "name": part.name,
                "path": f"parts/{part.name}",
                "rows": n,
                "sha256": hashlib.sha256(part.read_bytes()).hexdigest(),
            }
        ],
    }
    (base / "features_manifest.json").write_text(json.dumps(manifest))
    return feature_id


class TestCandidateLabels:
    def test_an_alias_names_the_candidate(self) -> None:
        c = fe.parse_candidate("structured@lf1:residual_forecaster=trailing_median")
        assert c.label == "lf1" and c.name == "structured"
        assert c.params["residual_forecaster"] == "trailing_median"

    def test_long_values_are_shortened_in_automatic_labels(self) -> None:
        c = fe.parse_candidate("structured:features=" + "ab" * 32)
        assert c.label == "structured[features=abababababab]"
        assert c.as_dict()["side_inputs"] is not None
        assert fe.parse_candidate("trailing_mean").as_dict()["side_inputs"] is None


class TestSeasonPairs:
    def _record(self, cost: np.ndarray, days: list[str]) -> dict[str, object]:
        n = len(days)
        return {"local_day": np.array(days), "usable": np.ones(n, dtype=bool), "cost": cost}

    def test_a_steady_saving_holds_in_the_season_it_covers(self) -> None:
        start = datetime(2025, 1, 1)
        days = [(start + timedelta(days=i)).date().isoformat() for i in range(365)]
        ref = self._record(np.ones(365), days)
        saving = np.where(np.array([int(d[5:7]) for d in days]) <= 3, 0.1, 0.0)
        cand = self._record(1.0 - saving, days)
        out = fe._season_pairs(ref, cand, resamples=5000, seed=(1, 0, 0))
        assert out["heating"]["holds"] is True
        assert out["cooling"]["holds"] is False
        assert out["period"]["point_eur"] == pytest.approx(saving.sum())
        assert out["heating"]["days"] == 90 + 61  # Jan-Mar plus Nov-Dec

    def test_a_short_season_is_not_evaluated(self) -> None:
        days = [f"2025-03-{d:02d}" for d in range(1, 11)]
        ref = self._record(np.ones(10), days)
        out = fe._season_pairs(ref, ref, resamples=5000, seed=(1, 0, 0))
        assert out["heating"]["status"] == "too few days" and out["heating"]["holds"] is False

    def test_the_g4_definition_states_the_seasons_before_any_run(self) -> None:
        g4 = fe._g4_definition("lf1", "trailing_mean", [("lf1", "trailing_median")], PERIODS)
        assert g4["seasons"] == {"heating": [11, 12, 1, 2, 3], "cooling": [6, 7, 8, 9]}
        assert g4["secondary_comparisons"][0]["gating"] is False
        assert "at least one heating season instance AND at least one cooling" in g4["rule"]


class TestG4EndToEnd:
    @pytest.fixture(scope="class")
    @classmethod
    def run(cls, tmp_path_factory) -> dict[str, object]:
        requires_core()
        tmp = tmp_path_factory.mktemp("g4")
        dataset, config = _dataset(tmp)
        feature_id = _write_feature_set(tmp, n=DAYS * 96)
        out = fe.run_forecast_eval(
            dataset=dataset,
            config_path=config,
            name="g4",
            candidates=[
                "trailing_mean",
                f"structured@lf1:features={feature_id}",
                f"structured@lf1_median:features={feature_id},residual_forecaster=trailing_median",
                "trailing_median",
            ],
            periods=PERIODS[:1],
            resamples=5000,
            data_dir=tmp,
            now=datetime(2026, 1, 1, tzinfo=UTC),
            gate="G4",
            gate_candidate="lf1",
            gate_reference="trailing_mean",
            compare=[("lf1", "trailing_median"), ("lf1_median", "trailing_mean")],
        )
        return {
            "dir": out,
            "prereg": json.loads((out / fe.PREREGISTRATION_NAME).read_text()),
            "summary": json.loads((out / fe.SUMMARY_JSON_NAME).read_text()),
        }

    def test_the_gate_and_seasons_are_preregistered(self, run) -> None:
        prereg = run["prereg"]
        assert prereg["gate_g4"]["seasons"]["cooling"] == [6, 7, 8, 9]
        assert prereg["gate_g4"]["candidate"] == "lf1"
        assert prereg["structured"]["rule_mirror"]["climate_control"][0]["history_days"] == 10
        assert prereg["side_inputs"]["lf1"]["feature_set_id"] == "ab" * 32
        assert {c["label"] for c in prereg["candidates"]} >= {"lf1", "lf1_median"}

    def test_the_summary_carries_g4_pairwise_and_diagnostics(self, run) -> None:
        summary = run["summary"]
        assert summary["g4"]["candidate"] == "lf1" and summary["g4"]["passed"] is False
        # One week of March: no season instance has 14 days.
        assert all("status" in row for row in summary["g4"]["instances"].values())
        period = summary["periods"]["a"]
        assert set(period["pairwise"]) == {
            "lf1 vs trailing_mean",
            "lf1 vs trailing_median",
            "lf1_median vs trailing_mean",
        }
        assert period["pairwise"]["lf1 vs trailing_median"]["role"] == "secondary"
        diag = period["candidates"]["lf1"]["diagnostics"]
        assert set(diag) == {"day_ahead", "rolling_horizon"}
        assert diag["rolling_horizon"]["decisions"] > 0
        assert "afternoon/prices" in diag["rolling_horizon"]["counts"]
        assert period["candidates"]["trailing_mean"]["diagnostics"] is None
        assert "Gate G4" in (run["dir"] / fe.SUMMARY_MD_NAME).read_text()

    def test_the_benchmark_ran_the_structured_forecaster_on_its_side_inputs(self, run) -> None:
        summary = run["summary"]
        run_id = summary["provenance"]["benchmark_runs"]["a"]["lf1"]
        bench = Path(summary["provenance"]["dataset_dir"]) / "benchmarks" / run_id
        results = json.loads((bench / bm.RESULTS_JSON_NAME).read_text())
        record = results["lineage"]["load_forecaster"]
        assert record["name"] == "structured"
        assert record["side_inputs_source"]["feature_set_id"] == "ab" * 32
