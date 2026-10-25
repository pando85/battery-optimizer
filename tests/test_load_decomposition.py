"""Tests for the load decomposition: seasons, rule replay, additive LF-0 error attribution."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from battery_optimizer import features
from battery_optimizer import load_decomposition as ld
from battery_optimizer.cli import main
from battery_optimizer.load_decomposition import (
    Frame,
    climate_evidence,
    decompose,
    dhw_evidence,
    dhw_rule_hours,
    error_stats,
    g3_indicator,
    joint_mask,
    lf0_errors,
    local_day_prices,
    season_label,
    select_climate_hours,
)
from battery_optimizer.schemas import ANALYSIS_INTERVAL

DAY0 = date(2026, 1, 1)


def make_frame(days: int = 45, seed: int = 7) -> Frame:
    """Synthetic grid: constant servers, a heat pump that runs in the cheap hours, noise."""
    rng = np.random.default_rng(seed)
    n = days * 96
    local_day = [DAY0 + timedelta(days=i // 96) for i in range(n)]
    local_hour = np.array([(i % 96) // 4 for i in range(n)])
    # cheap hours 2-5, expensive evening, flat otherwise; a slow drift between days.
    base = np.where((local_hour >= 2) & (local_hour < 6), 0.05, 0.12)
    base = np.where((local_hour >= 19) & (local_hour < 22), 0.20, base)
    price = base + 0.001 * np.array([i // 96 for i in range(n)])
    servers = np.full(n, 0.06)
    hp_heat = np.where((local_hour >= 2) & (local_hour < 6), 0.15, 0.0) * rng.uniform(0.5, 1.5, n)
    hp_dhw = np.where(rng.uniform(size=n) < 0.03, 0.2, 0.0)
    hp_cool = np.zeros(n)
    residual = 0.05 + 0.03 * rng.standard_normal(n)
    main = servers + hp_heat + hp_dhw + hp_cool + residual
    return Frame(
        start_us=np.arange(n, dtype=np.int64) * 900_000_000,
        local_day=local_day,
        local_hour=local_hour,
        price=price,
        usable=np.ones(n, dtype=bool),
        main=main,
        servers=servers,
        hp_heat=hp_heat,
        hp_dhw=hp_dhw,
        hp_cool=hp_cool,
        climate_min_hours=np.full(n, 4.0),
        climate_enable=np.ones(n),
        dhw_enable=np.ones(n),
        dhw_force_mean=np.zeros(n),
    )


class TestSeasons:
    def test_meteorological_seasons_and_december_joins_the_next_winter(self) -> None:
        assert season_label(date(2025, 12, 31)) == "2026-winter"
        assert season_label(date(2026, 1, 1)) == "2026-winter"
        assert season_label(date(2026, 4, 30)) == "2026-spring"
        assert season_label(date(2026, 7, 1)) == "2026-summer"
        assert season_label(date(2026, 11, 30)) == "2026-autumn"


class TestRuleReplay:
    PRICES = {h: 0.10 + 0.01 * ((h * 7) % 24) for h in range(24)}

    def test_cheapest_min_hours_when_few_hours_are_cheap(self) -> None:
        chosen = select_climate_hours(self.PRICES, limit=0.0, min_hours=3)
        expected = sorted(sorted(self.PRICES, key=lambda h: self.PRICES[h])[:3])
        assert chosen == expected

    def test_all_cheap_hours_when_there_are_at_least_min_hours_of_them(self) -> None:
        limit = 0.14
        cheap = sorted(h for h, p in self.PRICES.items() if p < limit)
        assert len(cheap) >= 3
        assert select_climate_hours(self.PRICES, limit=limit, min_hours=3) == cheap

    def test_dhw_picks_the_cheapest_hour_of_each_block(self) -> None:
        prices = {h: 1.0 for h in range(24)}
        prices[5] = 0.1
        prices[20] = 0.2
        assert dhw_rule_hours(prices, 12) == [5, 20]
        assert dhw_rule_hours(prices, 24) == [5]

    def test_dhw_ties_keep_the_earlier_hour(self) -> None:
        assert dhw_rule_hours({h: 1.0 for h in range(24)}, 12) == [0, 12]


class TestJointMask:
    def test_faults_and_unknowns_are_excluded(self) -> None:
        frame = make_frame(days=2)
        frame.servers[0] = np.nan
        frame.hp_heat[1] = np.nan
        frame.main[2] = 0.0  # servers + heat pump above main: a fault
        frame.usable[3] = False
        joint = joint_mask(frame)
        assert not joint[:4].any()
        assert joint[4:].mean() > 0.9


class TestLf0Attribution:
    def test_component_errors_add_up_to_the_main_error(self) -> None:
        frame = make_frame()
        joint = joint_mask(frame)
        errors, evaluation, meta = lf0_errors(frame, joint, window_days=7)
        assert evaluation.any()
        assert meta["max_additivity_deviation_kwh"] < 1e-9
        total = errors["servers"] + errors["heat_pump"] + errors["residual"]
        assert np.allclose(total[evaluation], errors["main"][evaluation])

    def test_variance_shares_sum_to_one_and_a_constant_series_has_no_error(self) -> None:
        frame = make_frame()
        joint = joint_mask(frame)
        errors, evaluation, _ = lf0_errors(frame, joint, window_days=7)
        stats = error_stats(errors, frame.price, evaluation)
        shares = [stats[c]["var_share"] for c in ("servers", "heat_pump", "residual")]
        assert sum(shares) == pytest.approx(1.0)
        assert stats["servers"]["mae_kwh"] == pytest.approx(0.0, abs=1e-12)
        assert stats["heat_pump"]["headroom_share"] > 0
        # shares of the heat pump parts also add up (heat + dhw + cool = heat pump)
        parts = sum(stats[c]["var_share"] for c in ("hp_heat", "hp_dhw", "hp_cool"))
        assert parts == pytest.approx(stats["heat_pump"]["var_share"])

    def test_only_causal_history_is_used(self) -> None:
        frame = make_frame()
        joint = joint_mask(frame)
        before, evaluation, _ = lf0_errors(frame, joint, window_days=7)
        # Perturb the last day: errors of every earlier day must not move.
        last = np.array([d == frame.local_day[-1] for d in frame.local_day])
        frame.hp_heat[last] += 5.0
        frame.main[last] += 5.0
        after, _, _ = lf0_errors(frame, joint_mask(frame), window_days=7)
        early = evaluation & ~last
        assert np.allclose(before["main"][early], after["main"][early])

    def test_cold_start_days_are_not_evaluated(self) -> None:
        frame = make_frame()
        joint = joint_mask(frame)
        _errors, evaluation, meta = lf0_errors(frame, joint, window_days=28)
        assert meta["min_history_days"] == 14
        assert not evaluation[: 14 * 96].any()
        assert evaluation[20 * 96 :].any()


class TestDecompose:
    def test_tables_cover_overall_months_and_top_tercile(self) -> None:
        frame = make_frame()
        joint = joint_mask(frame)
        errors, evaluation, _ = lf0_errors(frame, joint, window_days=7)
        table = decompose(frame, errors, evaluation, joint)
        assert set(table["by_month"]) == {"2026-01", "2026-02"}
        assert set(table["by_season_of_year"]) == {"winter"}
        overall = table["overall"]
        assert overall["energy"]["servers_share"] + overall["energy"]["heat_pump_share"] + overall[
            "energy"
        ]["residual_share"] == pytest.approx(1.0)
        top = overall["error_top_price_tercile"]
        assert 0 < top["slots"] < overall["error"]["slots"]
        verdict = g3_indicator(table)
        assert verdict["error"]["var_share"]["heat_pump"] is not None
        # the pump is the only spiky, price-driven component in the fixture
        assert "heat_pump" in verdict["error"]["material_observable"]
        assert "servers" not in verdict["error"]["material_observable"]


class TestEvidence:
    def test_hourly_prices_skip_hours_whose_slots_disagree(self) -> None:
        frame = make_frame(days=2)
        frame.price[0] = 0.5  # first slot of hour 0 differs from its siblings
        prices = local_day_prices(frame)
        assert 0 not in prices[DAY0]
        assert 1 in prices[DAY0]

    def test_climate_replay_finds_the_heating_in_the_selected_hours(self) -> None:
        frame = make_frame(days=20)
        evidence = climate_evidence(frame)
        assert evidence["days_used"] > 0
        overall = evidence["by_group"]["all"]
        # the fixture heats in hours 2-5, which are the four cheapest hours every day
        assert overall["share_in_selected"] == pytest.approx(1.0)
        assert overall["selected_time_share"] == pytest.approx(4 / 24)
        assert overall["enrichment"] == pytest.approx(6.0)

    def test_climate_replay_skips_disabled_days(self) -> None:
        frame = make_frame(days=20)
        frame.climate_enable[:] = 0.0
        evidence = climate_evidence(frame)
        assert evidence["days_used"] == 0
        assert evidence["days_skipped"]["climate_disabled"] > 0

    def test_dhw_evidence_counts_force_on_inside_the_rule_hour(self) -> None:
        frame = make_frame(days=3)
        day_hours = {h: frame.price[(frame.local_hour == h)][0] for h in range(24)}
        rule = dhw_rule_hours(day_hours, 12)
        frame.dhw_force_mean[:] = 0.0
        # switch on during the rule hours of every day
        for h in rule:
            frame.dhw_force_mean[frame.local_hour == h] = 1.0
        evidence = dhw_evidence(frame)
        row = evidence["interval_12h"]["by_group"]["all"]
        assert row["share_force_on_slots_in_rule_hour"] == pytest.approx(1.0)
        assert row["share_rule_hours_with_force_on"] == pytest.approx(1.0)

    def test_after_force_on_mask_extends_three_hours(self) -> None:
        frame = make_frame(days=2)
        frame.dhw_force_mean[:] = 0.0
        frame.dhw_force_mean[10] = 1.0
        mask = ld.after_force_on_mask(frame)
        assert mask[10] and mask[10 + 11] and not mask[10 + 12] and not mask[9]


def _write_dataset(root: Path, frame: Frame) -> Path:
    """A minimal canonical dataset directory for the frame (UTC as the local zone)."""
    n = frame.n
    start = datetime(2026, 1, 1, tzinfo=UTC)
    starts = [start + timedelta(minutes=15 * i) for i in range(n)]
    dataset_id = "d" * 64
    table = pa.table(
        {
            "interval_start": pa.array(starts, pa.timestamp("us", tz="UTC")),
            "interval_end": pa.array(
                [s + timedelta(minutes=15) for s in starts], pa.timestamp("us", tz="UTC")
            ),
            "tz_local": ["UTC"] * n,
            "tariff_period": pa.array([1] * n, pa.int8()),
            "local_day": pa.array(frame.local_day, pa.date32()),
            "dst_transition": pa.array([None] * n, pa.string()),
            "raw_grid_kwh": pa.array(frame.main, pa.float64()),
            "gross_kwh": pa.array(frame.main, pa.float64()),
            "battery_absent": [True] * n,
            "energy_quality": ["measured"] * n,
            "allocation_quality": ["estimated_boundary"] * n,
            "sample_count": pa.array([15] * n, pa.int32()),
            "gap_seconds": pa.array([0.0] * n, pa.float64()),
            "price_eur_per_kwh": pa.array(frame.price, pa.float64()),
            "price_quality": ["measured"] * n,
            "dataset_id": [dataset_id] * n,
        },
        schema=ANALYSIS_INTERVAL,
    )
    directory = root / "analysis" / "v1" / dataset_id
    (directory / "parts").mkdir(parents=True)
    path = directory / "parts" / "intervals-test.parquet"
    pq.write_table(table, path)
    manifest = {
        "dataset_id": dataset_id,
        "requested_window": {"start": "2026-01-01T00:00:00Z", "end": "2026-02-15T00:00:00Z"},
        "part_files": [
            {
                "path": "parts/intervals-test.parquet",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rows": n,
            }
        ],
    }
    (directory / "build_manifest.json").write_text(json.dumps(manifest))
    return directory


def _write_features(root: Path, frame: Frame) -> Path:
    schema = features.feature_schema()
    n = frame.n
    start = datetime(2026, 1, 1, tzinfo=UTC)
    starts = [start + timedelta(minutes=15 * i) for i in range(n)]
    columns: dict[str, pa.Array] = {
        "interval_start": pa.array(starts, pa.timestamp("us", tz="UTC")),
        "interval_end": pa.array(
            [s + timedelta(minutes=15) for s in starts], pa.timestamp("us", tz="UTC")
        ),
    }
    given = {
        "servers_kwh": frame.servers,
        "hp_heat_kwh": frame.hp_heat,
        "hp_dhw_kwh": frame.hp_dhw,
        "hp_cool_kwh": frame.hp_cool,
        "climate_min_hours_at_start": frame.climate_min_hours,
        "climate_enable_at_start": frame.climate_enable,
        "dhw_enable_at_start": frame.dhw_enable,
        "dhw_force_mean": frame.dhw_force_mean,
    }
    for field in schema:
        if field.name in columns:
            continue
        if field.name in given:
            columns[field.name] = pa.array(given[field.name], pa.float64())
        elif field.type == pa.string():
            columns[field.name] = pa.array(["unknown"] * n, pa.string())
        else:
            columns[field.name] = pa.array([None] * n, pa.float64())
    table = pa.table(columns, schema=schema)
    directory = root / "analysis" / "v1" / "features" / "f"
    (directory / "parts").mkdir(parents=True)
    path = directory / "parts" / "features-test.parquet"
    pq.write_table(table, path)
    (directory / "features_manifest.json").write_text(
        json.dumps(
            {
                "feature_set_id": "f",
                "part_files": [
                    {
                        "path": "parts/features-test.parquet",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "rows": n,
                    }
                ],
            }
        )
    )
    return directory


class TestWriteReport:
    def test_writes_json_and_markdown_next_to_the_features(self, tmp_path: Path) -> None:
        frame = make_frame()
        root = tmp_path / "data"
        dataset = _write_dataset(root, frame)
        feats = _write_features(root, frame)
        out = ld.write_report(
            features_dir=feats, dataset_dir=dataset, data_dir=str(root), window_days=7
        )
        assert out == feats / "p3-report"
        summary = json.loads((out / "summary.json").read_text())
        assert summary["inputs"]["feature_set_id"] == "f"
        assert summary["joint"]["slots"] > 0
        assert summary["lf0"]["max_additivity_deviation_kwh"] < 1e-9
        assert summary["g3"]["error"]["material_observable"] == ["heat_pump"]
        text = (out / "summary.md").read_text()
        assert "# P3 sub-load decomposition" in text
        assert "G3 indicator" in text
        # regenerating overwrites the derived report instead of failing
        ld.write_report(features_dir=feats, dataset_dir=dataset, data_dir=str(root), window_days=7)

    def test_cli_subcommand(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        frame = make_frame()
        root = tmp_path / "data"
        dataset = _write_dataset(root, frame)
        feats = _write_features(root, frame)
        rc = main(
            [
                "decompose-load",
                "--data-dir",
                str(root),
                "--features",
                str(feats),
                "--dataset",
                str(dataset),
                "--window-days",
                "7",
            ]
        )
        assert rc == 0
        assert "report directory" in capsys.readouterr().out
