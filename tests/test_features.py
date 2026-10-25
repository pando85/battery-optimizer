"""Tests for the sub-load feature table: integration rule, quality, placement, immutability."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from battery_optimizer import features
from battery_optimizer.cli import main
from battery_optimizer.extract_prom import compute_series_key
from battery_optimizer.features import (
    FeatureError,
    Signal,
    assemble_table,
    build_features,
    cop_check,
    integrate_slots,
    load_features,
    main_meter_check,
    merge_samples,
    signal_columns,
)
from battery_optimizer.paths import parts_dir
from battery_optimizer.schemas import RAW_SAMPLE

T0 = datetime(2026, 1, 5, tzinfo=UTC)
SERVERS = "sensor.shellypmmini_servers_power"
HEAT = "sensor.aquarea_heat_power_consumed"
TEMP = "sensor.aquarea_outside_ambient_temperature"
ENABLE = "input_boolean.appdaemon_climate_enable"
POWER_SIGNAL = Signal(SERVERS, "servers", features.KIND_POWER)
STATE_SIGNAL = Signal(ENABLE, "climate_enable", features.KIND_STATE)
TEMP_SIGNAL = Signal(TEMP, "outdoor_temp", features.KIND_TEMPERATURE)


def edges(n_slots: int, step: float = 900.0) -> np.ndarray:
    return np.arange(n_slots + 1, dtype=float) * step


def us_edges(n_slots: int) -> np.ndarray:
    base = int(T0.timestamp()) * 1_000_000
    return base + np.arange(n_slots + 1, dtype=np.int64) * 900_000_000


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


class TestMergeSamples:
    def test_sorts_and_drops_repeated_timestamps_keeping_the_first(self) -> None:
        t, v, dropped, conflicts = merge_samples(
            np.array([3, 1, 2, 2], dtype=np.int64), np.array([30.0, 10.0, 20.0, 20.0])
        )
        assert t.tolist() == [1, 2, 3]
        assert v.tolist() == [10.0, 20.0, 30.0]
        assert (dropped, conflicts) == (1, 0)

    def test_a_conflicting_duplicate_is_counted(self) -> None:
        _t, v, dropped, conflicts = merge_samples(
            np.array([1, 1], dtype=np.int64), np.array([5.0, 6.0])
        )
        assert v.tolist() == [5.0]
        assert (dropped, conflicts) == (1, 1)

    def test_empty_input(self) -> None:
        t, v, dropped, conflicts = merge_samples(np.zeros(0, dtype=np.int64), np.zeros(0))
        assert t.size == v.size == 0
        assert (dropped, conflicts) == (0, 0)


class TestIntegrateSlots:
    def test_sample_and_hold_is_exact_across_a_slot_edge(self) -> None:
        # 100 W held from 0 to 450 s, 300 W from 450 s on; one 900 s slot.
        t = np.array([0.0, 450.0, 900.0])
        v = np.array([100.0, 300.0, 300.0])
        integral, covered, at_start = integrate_slots(t, v, edges(1), 1000.0)
        assert integral[0] == pytest.approx(100 * 450 + 300 * 450)
        assert covered[0] == pytest.approx(900.0)
        assert at_start[0] == 100.0

    def test_a_span_is_split_in_proportion_to_time_at_the_slot_edge(self) -> None:
        t = np.array([0.0, 600.0, 1800.0])
        v = np.array([200.0, 400.0, 400.0])
        integral, covered, _ = integrate_slots(t, v, edges(2), 2000.0)
        assert integral.tolist() == pytest.approx([200 * 600 + 400 * 300, 400 * 900])
        assert covered.tolist() == pytest.approx([900.0, 900.0])

    def test_a_span_longer_than_the_max_gap_is_not_held(self) -> None:
        t = np.array([0.0, 60.0, 120.0, 1200.0, 1260.0])
        v = np.array([500.0] * 5)
        _integral, covered, at_start = integrate_slots(t, v, edges(2), 300.0)
        # slot 0: spans 0-60 and 60-120 covered (120 s); 120-1200 is a gap.
        assert covered[0] == pytest.approx(120.0)
        assert covered[1] == pytest.approx(60.0)
        assert np.isnan(at_start[1])  # the slot starts inside the gap

    def test_the_time_after_the_last_sample_is_not_covered(self) -> None:
        t = np.array([0.0, 60.0])
        v = np.array([10.0, 10.0])
        _integral, covered, _ = integrate_slots(t, v, edges(1), 300.0)
        assert covered[0] == pytest.approx(60.0)

    def test_fewer_than_two_samples_cover_nothing(self) -> None:
        integral, covered, at_start = integrate_slots(
            np.array([5.0]), np.array([1.0]), edges(2), 300.0
        )
        assert integral.tolist() == [0.0, 0.0]
        assert covered.tolist() == [0.0, 0.0]
        assert np.isnan(at_start).all()


class TestSignalColumns:
    def _minute_samples(self, watts: float, minutes: int) -> tuple[np.ndarray, np.ndarray]:
        base = int(T0.timestamp()) * 1_000_000
        t = base + np.arange(minutes + 1, dtype=np.int64) * 60_000_000
        return t, np.full(t.size, watts)

    def test_constant_power_gives_kwh_and_full_quality(self) -> None:
        t, v = self._minute_samples(400.0, 60)
        cols = signal_columns(
            POWER_SIGNAL, t, v, us_edges(3), max_gap_seconds=300.0, min_coverage=0.8
        )
        assert cols["servers_kwh"].tolist() == pytest.approx([0.1, 0.1, 0.1])
        assert cols["servers_coverage"].tolist() == pytest.approx([1.0, 1.0, 1.0])
        assert cols["servers_quality"].tolist() == ["measured"] * 3

    def test_partial_slot_is_scaled_and_labelled_and_a_thin_slot_is_unknown_not_zero(
        self,
    ) -> None:
        base = int(T0.timestamp()) * 1_000_000
        # slot 0: samples each minute for 12 minutes then a 3 minute pause (still held, gap of
        # 180 s <= 300 s) -> full; slot 1: samples for the first 2 minutes only, then a gap.
        minutes = list(range(0, 16)) + [16, 17, 18, 45, 46, 47, 48]
        t = base + np.array(minutes, dtype=np.int64) * 60_000_000
        v = np.full(t.size, 600.0)
        cols = signal_columns(
            POWER_SIGNAL, t, v, us_edges(3), max_gap_seconds=300.0, min_coverage=0.8
        )
        # slot 1 (15-30 min): covered 15-18 min = 3 min of 15 -> unknown.
        assert cols["servers_quality"][1] == "unknown"
        assert np.isnan(cols["servers_kwh"][1])
        assert cols["servers_coverage"][1] == pytest.approx(3 / 15)
        # slot 0 is fully covered.
        assert cols["servers_quality"][0] == "measured"

    def test_partial_coverage_above_the_threshold_is_scaled_to_the_full_slot(self) -> None:
        base = int(T0.timestamp()) * 1_000_000
        # 12 of 15 minutes covered, 500 W: energy is the mean power times the full slot.
        minutes = list(range(0, 12)) + [12.0]
        t = base + (np.array(minutes) * 60_000_000).astype(np.int64)
        v = np.full(t.size, 500.0)
        # a second slot's worth of samples far away so the 12 -> 40 min span is a gap
        t = np.concatenate([t, base + np.array([40 * 60_000_000, 41 * 60_000_000])])
        v = np.concatenate([v, [500.0, 500.0]])
        cols = signal_columns(
            POWER_SIGNAL, t, v, us_edges(1), max_gap_seconds=300.0, min_coverage=0.8
        )
        assert cols["servers_coverage"][0] == pytest.approx(12 / 15)
        assert cols["servers_quality"][0] == "partial"
        assert cols["servers_kwh"][0] == pytest.approx(500 * 0.25 / 1000)

    def test_no_samples_is_unknown_never_zero(self) -> None:
        cols = signal_columns(
            POWER_SIGNAL,
            np.zeros(0, dtype=np.int64),
            np.zeros(0),
            us_edges(2),
            max_gap_seconds=300.0,
            min_coverage=0.8,
        )
        assert np.isnan(cols["servers_kwh"]).all()
        assert cols["servers_coverage"].tolist() == [0.0, 0.0]
        assert cols["servers_quality"].tolist() == ["unknown", "unknown"]

    def test_state_has_value_at_start_and_fraction_on(self) -> None:
        base = int(T0.timestamp()) * 1_000_000
        # off for the first 6 minutes of the slot, then on.
        minutes = np.arange(0, 31)
        t = base + minutes.astype(np.int64) * 60_000_000
        v = np.where(minutes < 6, 0.0, 1.0)
        cols = signal_columns(
            STATE_SIGNAL, t, v, us_edges(2), max_gap_seconds=300.0, min_coverage=0.8
        )
        assert cols["climate_enable_at_start"].tolist() == [0.0, 1.0]
        assert cols["climate_enable_mean"].tolist() == pytest.approx([9 / 15, 1.0])

    def test_temperature_is_the_time_weighted_mean(self) -> None:
        base = int(T0.timestamp()) * 1_000_000
        t = base + np.array([0, 450, 900], dtype=np.int64) * 1_000_000
        v = np.array([10.0, 20.0, 20.0])
        cols = signal_columns(
            TEMP_SIGNAL, t, v, us_edges(1), max_gap_seconds=1000.0, min_coverage=0.8
        )
        assert cols["outdoor_temp_mean_c"][0] == pytest.approx(15.0)


class TestAssembleTable:
    def test_missing_series_yield_nulls_and_the_schema_is_stable(self) -> None:
        table = assemble_table({}, us_edges(4))
        assert table.schema.equals(features.feature_schema())
        assert table.num_rows == 4
        assert table.column("servers_kwh").null_count == 4
        assert table.column("climate_enable_at_start").null_count == 4
        assert set(table.column("servers_quality").to_pylist()) == {"unknown"}

    def test_slots_are_a_15_minute_utc_grid(self) -> None:
        table = assemble_table({}, us_edges(3))
        starts = table.column("interval_start").to_pylist()
        ends = table.column("interval_end").to_pylist()
        assert starts[1] - starts[0] == timedelta(minutes=15)
        assert ends[0] == starts[1]
        assert starts[0].tzinfo is not None


class TestSlotEdges:
    def test_unaligned_bounds_are_refused(self) -> None:
        with pytest.raises(FeatureError, match="15 minutes"):
            features.slot_edges_us(T0 + timedelta(minutes=7), T0 + timedelta(hours=1))

    def test_empty_window_is_refused(self) -> None:
        with pytest.raises(FeatureError, match="before end"):
            features.slot_edges_us(T0, T0)


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------


def _feature_table(**columns: list[float]) -> pa.Table:
    n = len(next(iter(columns.values())))
    base = {
        name: pa.array(
            [None if x is None or x != x else x for x in values],  # NaN -> null
            pa.float64(),
        )
        for name, values in columns.items()
    }
    starts = [T0 + timedelta(minutes=15 * i) for i in range(n)]
    return pa.table(
        {
            "interval_start": pa.array(starts, pa.timestamp("us", tz="UTC")),
            "interval_end": pa.array(
                [s + timedelta(minutes=15) for s in starts], pa.timestamp("us", tz="UTC")
            ),
            **base,
        }
    )


def _dataset_table(gross: list[float], quality: str = "measured") -> pa.Table:
    n = len(gross)
    starts = [T0 + timedelta(minutes=15 * i) for i in range(n)]
    return pa.table(
        {
            "interval_start": pa.array(starts, pa.timestamp("us", tz="UTC")),
            "gross_kwh": pa.array(gross, pa.float64()),
            "energy_quality": pa.array([quality] * n),
            "allocation_quality": pa.array(["estimated_boundary"] * n),
        }
    )


class TestMainMeterCheck:
    def _table(self) -> pa.Table:
        nan = float("nan")
        return _feature_table(
            servers_kwh=[0.05, 0.05, 0.05, 0.05, nan],
            hp_heat_kwh=[0.10, 0.20, 0.0, 0.0, 0.0],
            hp_dhw_kwh=[0.0, 0.0, 0.0, 0.0, 0.0],
            hp_cool_kwh=[0.0, 0.0, 0.0, 0.0, 0.0],
        )

    def test_counts_violations_and_their_size_per_tolerance(self) -> None:
        dataset = _dataset_table([0.30, 0.20, 0.0, 0.07, 0.30])
        check = main_meter_check(self._table(), dataset)
        # slot 4 has no servers value -> not compared.
        assert check["slots_compared"] == 4
        by_tol = check["violations_by_tolerance"]
        # excess: 0.15-0.30=-0.15 ok; 0.25-0.20=0.05 viol; 0.05-0=0.05 viol; 0.05-0.07<0.
        assert by_tol["0"]["count"] == 2
        assert by_tol["0.02"]["count"] == 2
        assert by_tol["0.05"]["count"] == 0
        assert by_tol["0"]["excess_kwh_max"] == pytest.approx(0.05)

    def test_slots_with_unusable_main_are_not_compared(self) -> None:
        dataset = _dataset_table([0.0, 0.0, 0.0, 0.0, 0.0], quality="gap_total_unknown")
        check = main_meter_check(self._table(), dataset)
        assert check["slots_compared"] == 0

    def test_a_missing_component_is_unknown_not_zero(self) -> None:
        nan = float("nan")
        table = _feature_table(
            servers_kwh=[0.05],
            hp_heat_kwh=[nan],
            hp_dhw_kwh=[0.0],
            hp_cool_kwh=[0.0],
        )
        assert np.isnan(features.heat_pump_input(table)).all()
        assert main_meter_check(table, _dataset_table([0.0]))["slots_compared"] == 0


class TestCopCheck:
    def test_ratio_of_produced_to_consumed_is_a_cop(self) -> None:
        table = _feature_table(
            hp_heat_kwh=[0.1, 0.2, 0.0],
            hp_heat_out_kwh=[0.4, 0.8, 0.0],
            hp_dhw_kwh=[0.1, 0.0, 0.0],
            hp_dhw_out_kwh=[0.25, 0.0, 0.0],
            hp_cool_kwh=[0.0, 0.0, 0.0],
            hp_cool_out_kwh=[0.0, 0.0, 0.0],
            hp_total_reported_kwh=[0.2, 0.2, 0.0],
        )
        out = cop_check(table)
        assert out["heat"]["cop_ratio_of_sums"] == pytest.approx(4.0)
        assert out["dhw"]["cop_ratio_of_sums"] == pytest.approx(2.5)
        assert out["heat"]["slots_active"] == 2
        assert out["reported_total_vs_components"]["ratio_of_sums"] == pytest.approx(1.0)


class TestConstantRuns:
    def test_a_long_non_zero_constant_run_is_listed_and_zero_is_ignored(self) -> None:
        minutes = np.arange(0, 600)
        t = minutes.astype(np.int64) * 60_000_000
        v = np.where(minutes < 100, 3000.0, np.where(minutes < 300, 0.0, 41.0 + minutes % 7))
        runs = features.constant_runs(t, v, min_hours=1.5)
        assert runs["count"] == 1
        assert runs["longest"][0]["value"] == 3000.0
        assert runs["longest"][0]["hours"] == pytest.approx(99 / 60)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def _raw_rows(entity: str, metric: str, unit: str, start: datetime, values: list[float]):
    labels = json.dumps({"__name__": metric, "entity": entity, "job": "home-assistant"})
    key = compute_series_key({"__name__": metric, "entity": entity, "job": "home-assistant"})
    return [
        {
            "entity": entity,
            "metric": metric,
            "unit": unit,
            "series_key": key,
            "ts": start + timedelta(seconds=60 * i + 0.25),
            "value": value,
            "scrape_instance": "10.0.0.1:8080",
            "extraction_id": "11111111-1111-1111-1111-111111111111",
            "labels_json": labels,
        }
        for i, value in enumerate(values)
    ]


def _write_raw(data_dir: Path, rows: list[dict[str, Any]], *, entities: list[str]) -> str:
    table = pa.table(
        {k: [r[k] for r in rows] for k in RAW_SAMPLE.names},
        schema=RAW_SAMPLE,
    )
    name = f"prom-samples-20260105T000000Z-20260106T000000Z-{len(entities):08x}.parquet"
    path = parts_dir(data_dir, "prometheus") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "manifest_version": 1,
        "extraction_id": f"ex-{sha[:12]}",
        "finished_at": "2026-01-06T00:00:00Z",
        "source": "prometheus",
        "entities": sorted(entities),
        "source_window_start": "2026-01-05T00:00:00Z",
        "source_window_end": "2026-01-06T00:00:00Z",
        "row_count": table.num_rows,
        "part_files": [
            {
                "path": f"raw/prometheus/v1/parts/{name}",
                "name": name,
                "sha256": sha,
                "rows": table.num_rows,
                "chunk_start": "2026-01-05T00:00:00Z",
                "chunk_end": "2026-01-06T00:00:00Z",
            }
        ],
    }
    manifests = data_dir / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / f"ex-{sha[:12]}.json").write_text(json.dumps(manifest))
    return sha


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "data"
    directory.mkdir()
    rows = _raw_rows(SERVERS, "hass_sensor_power_w", "W", T0, [240.0] * 121)
    rows += _raw_rows(HEAT, "hass_sensor_power_w", "W", T0, [0.0] * 60 + [800.0] * 61)
    rows += _raw_rows(ENABLE, "hass_input_boolean_state", "bool", T0, [1.0] * 121)
    _write_raw(directory, rows, entities=[SERVERS, HEAT, ENABLE])
    return directory


class TestBuildFeatures:
    WINDOW = ("2026-01-05T00:00:00Z", "2026-01-05T02:00:00Z")

    def test_writes_a_separate_feature_set_with_hashed_inputs(self, data_dir: Path) -> None:
        out = build_features(data_dir=str(data_dir), start=self.WINDOW[0], end=self.WINDOW[1])
        assert out.parent == data_dir / "analysis" / "v1" / "features"
        assert not (out / "build_manifest.json").exists()
        table, manifest = load_features(out)
        assert out.name == manifest["feature_set_id"]
        assert table.num_rows == 8
        # first hour: servers 240 W -> 0.06 kWh per slot, heat pump idle.
        assert table.column("servers_kwh").to_pylist()[0] == pytest.approx(0.06)
        assert table.column("hp_heat_kwh").to_pylist()[0] == pytest.approx(0.0)
        # second hour at 800 W -> 0.2 kWh per slot; the last sample ends the data, so the final
        # slot loses the last minute but stays above the coverage threshold.
        assert table.column("hp_heat_kwh").to_pylist()[5] == pytest.approx(0.2)
        # series that were not extracted stay unknown, not zero
        assert table.column("hp_dhw_kwh").null_count == 8
        assert manifest["inputs"]["raw_parts"][0]["sha256"]
        assert manifest["code_version"]
        assert manifest["params"]["max_gap_seconds"] == 300.0
        assert "Sample-and-hold" in manifest["integration_rule"]
        assert manifest["quality"]["series"]["servers"]["cadence_seconds"]["median"] == 60.0
        assert (
            "sensor.aquarea_dhw_power_consumed" in manifest["quality"]["entities_without_samples"]
        )

    def test_rebuilding_the_same_inputs_is_idempotent(self, data_dir: Path) -> None:
        first = build_features(data_dir=str(data_dir), start=self.WINDOW[0], end=self.WINDOW[1])
        second = build_features(data_dir=str(data_dir), start=self.WINDOW[0], end=self.WINDOW[1])
        assert first == second

    def test_a_different_parameter_gives_a_different_id(self, data_dir: Path) -> None:
        first = build_features(data_dir=str(data_dir), start=self.WINDOW[0], end=self.WINDOW[1])
        second = build_features(
            data_dir=str(data_dir),
            start=self.WINDOW[0],
            end=self.WINDOW[1],
            max_gap_seconds=120.0,
        )
        assert first != second

    def test_a_tampered_part_is_refused_on_load(self, data_dir: Path) -> None:
        out = build_features(data_dir=str(data_dir), start=self.WINDOW[0], end=self.WINDOW[1])
        part = next((out / "parts").glob("*.parquet"))
        part.write_bytes(part.read_bytes() + b"x")
        with pytest.raises(FeatureError, match="sha256"):
            load_features(out)

    def test_a_tampered_raw_part_is_refused(self, data_dir: Path) -> None:
        raw = next(parts_dir(data_dir, "prometheus").glob("*.parquet"))
        raw.write_bytes(raw.read_bytes() + b"x")
        with pytest.raises(FeatureError, match="sha256"):
            build_features(data_dir=str(data_dir), start=self.WINDOW[0], end=self.WINDOW[1])

    def test_no_raw_input_is_an_error(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(FeatureError, match="no raw parts"):
            build_features(data_dir=str(empty), start=self.WINDOW[0], end=self.WINDOW[1])

    def test_bad_parameters_are_refused(self, data_dir: Path) -> None:
        with pytest.raises(FeatureError, match="min_coverage"):
            build_features(
                data_dir=str(data_dir), start=self.WINDOW[0], end=self.WINDOW[1], min_coverage=0.0
            )
        with pytest.raises(FeatureError, match="max_gap"):
            build_features(
                data_dir=str(data_dir),
                start=self.WINDOW[0],
                end=self.WINDOW[1],
                max_gap_seconds=0,
            )


class TestCli:
    def test_build_features_subcommand(
        self, data_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(
            [
                "build-features",
                "--data-dir",
                str(data_dir),
                "--start",
                "2026-01-05T00:00:00Z",
                "--end",
                "2026-01-05T02:00:00Z",
            ]
        )
        assert rc == 0
        assert "feature set directory" in capsys.readouterr().out

    def test_user_errors_are_one_line_exit_1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = main(
            [
                "build-features",
                "--data-dir",
                str(tmp_path),
                "--start",
                "2026-01-05T00:00:00Z",
                "--end",
                "2026-01-05T02:00:00Z",
            ]
        )
        assert rc == 1
        assert "no raw parts" in capsys.readouterr().err
