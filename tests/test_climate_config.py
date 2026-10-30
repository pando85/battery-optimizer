"""Strict validation of ``config/climate.yaml`` (the heat-v1 preregistered values)."""

from __future__ import annotations

import copy
import re
from datetime import date
from pathlib import Path
from typing import Any

import pytest
import yaml

from battery_optimizer import forecast_eval
from battery_optimizer.climate.config import ClimateConfigError, load_climate_config

REPO_ROOT = Path(__file__).resolve().parents[1]
SHIPPED = REPO_ROOT / "config" / "climate.yaml"


def _raw() -> dict[str, Any]:
    return yaml.safe_load(SHIPPED.read_text())


def _write(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "climate.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return path


def test_shipped_config_holds_the_plan_values():
    cfg = load_climate_config(SHIPPED)
    assert cfg.config_version == 1
    assert (cfg.comfort.band, cfg.comfort.low_c, cfg.comfort.high_c) == ("heat-v1", 22.0, 28.0)
    assert cfg.comfort.rooms == ("bedroom", "hall_kitchen", "office_t")
    assert cfg.comfort.excluded_rooms == ("office_a",)
    assert cfg.seasons.S3.start == date(2025, 10, 1) and cfg.seasons.S3.end == date(2026, 4, 30)
    assert cfg.seasons.gate_months == (11, 12, 1, 2, 3)
    assert cfg.filters.window_open.drop_k == 0.5
    assert cfg.filters.valves_shut.min_rooms_at_setpoint == 2
    assert cfg.model.envelope.tau_h == (10.0, 500.0)
    assert cfg.model.envelope.ua_kw_per_k == (0.03, 0.5)
    assert cfg.model.envelope.c_slab_kwh_per_k == (2.0, 60.0)
    assert cfg.model.envelope.k_kw_per_k == (0.1, 3.0)
    assert cfg.model.envelope.q0_kw == (0.0, 3.0)
    assert cfg.acceptance.q_driven.rmse_k == {6: 0.30, 12: 0.40, 24: 0.50}
    assert cfg.acceptance.q_driven.abs_bias_24h_k == 0.15
    assert cfg.acceptance.u_driven.rmse_24h_k == 0.60
    assert cfg.acceptance.u_driven.regime_energy_bias == 0.10
    assert cfg.acceptance.parameters.fold_rel_diff == 0.30
    assert cfg.cop.daily_mape == 0.08 and cfg.cop.season_error == 0.03
    assert cfg.gate.cost.min_saving == 0.10
    assert cfg.gate.comfort.min_kh_reduction == 0.25
    assert cfg.gate.hard.starts_per_day_ratio_max == 1.10
    assert cfg.gate.hard.max_starts_per_day == 30
    assert cfg.gate.hard.max_blocks_per_day == 4


def test_statistics_follow_forecast_eval():
    cfg = load_climate_config(SHIPPED)
    assert cfg.gate.statistics.block_days == forecast_eval.BLOCK_DAYS
    assert cfg.gate.statistics.confidence == forecast_eval.CONFIDENCE


def test_hash_identifies_the_file(tmp_path):
    cfg = load_climate_config(SHIPPED)
    other = _raw()
    other["gate"]["cost"]["min_saving"] = 0.11
    assert load_climate_config(_write(tmp_path, other)).sha256 != cfg.sha256
    assert load_climate_config(SHIPPED).sha256 == cfg.sha256


def test_config_is_read_only():
    cfg = load_climate_config(SHIPPED)
    with pytest.raises(AttributeError):
        cfg.comfort.low_c = 20.0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        cfg.comfort.nope  # noqa: B018


def _mutations() -> list[tuple[str, Any]]:
    def drop(*path: str):
        def run(raw):
            node = raw
            for key in path[:-1]:
                node = node[key]
            del node[path[-1]]

        return run

    def put(value, *path: str):
        def run(raw):
            node = raw
            for key in path[:-1]:
                node = node[key]
            node[path[-1]] = value

        return run

    return [
        ("unknown top-level key", put(1, "surprise")),
        ("unknown nested key", put(1, "gate", "cost", "surprise")),
        ("missing top-level key", drop("gate")),
        ("missing nested key", drop("filters", "window_open", "drop_k")),
        ("bool as number", put(True, "comfort", "low_c")),
        ("string as number", put("22", "comfort", "low_c")),
        ("nan", put(float("nan"), "gate", "cost", "min_saving")),
        ("inf", put(float("inf"), "gate", "cost", "min_saving")),
        ("fraction above one", put(1.5, "gate", "cost", "min_saving")),
        ("negative threshold", put(-0.1, "acceptance", "u_driven", "rmse_24h_k")),
        ("float as integer", put(2.0, "filters", "valves_shut", "min_rooms_at_setpoint")),
        ("reversed envelope", put([500.0, 10.0], "model", "envelope", "tau_h")),
        ("envelope of three", put([1.0, 2.0, 3.0], "model", "envelope", "tau_h")),
        ("band inverted", put(30.0, "comfort", "low_c")),
        ("unknown room", put(["bedroom", "attic"], "comfort", "rooms")),
        ("room in both lists", put(["bedroom", "hall_kitchen", "office_a"], "comfort", "rooms")),
        ("room dropped", put(["bedroom"], "comfort", "rooms")),
        ("gate band differs", put("heat-v2", "gate", "band")),
        ("version", put(2, "config_version")),
        ("string for bool", put("yes", "filters", "drop_cooling_slots")),
        ("overlapping seasons", put({"start": "2024-04-01", "end": "2025-04-30"}, "seasons", "S2")),
        ("season reversed", put({"start": "2024-04-30", "end": "2024-04-01"}, "seasons", "S2")),
        ("reserved before S3 end", put("2026-01-01", "seasons", "reserved_from")),
        ("gate months not in fit months", put([5], "seasons", "gate_months")),
        ("fold tests on its training season", put({"train": ["S1"], "test": "S1"}, "folds", 0)),
        ("bad date", put("not-a-date", "seasons", "S1", "start")),
        ("rmse horizons", put({6: 0.3, 12: 0.4}, "acceptance", "q_driven", "rmse_k")),
        ("blocks disagree", put(5, "gate", "hard", "max_blocks_per_day")),
        (
            "window reversed",
            put(
                [{"start": "2026-02-12", "end": "2026-02-09"}],
                "filters",
                "thermostat_outage",
                "windows",
            ),
        ),
        (
            "more rooms at setpoint than comfort rooms",
            put(4, "filters", "valves_shut", "min_rooms_at_setpoint"),
        ),
    ]


@pytest.mark.parametrize(("label", "mutate"), _mutations(), ids=[m[0] for m in _mutations()])
def test_malformed_config_is_refused(tmp_path, label, mutate):
    raw = copy.deepcopy(_raw())
    mutate(raw)
    with pytest.raises(ClimateConfigError, match="climate config"):
        load_climate_config(_write(tmp_path, raw))


def test_duplicate_key_is_refused(tmp_path):
    text = SHIPPED.read_text().replace(
        "band: heat-v1\n  low_c", "band: heat-v1\n  band: heat-v1\n  low_c", 1
    )
    path = tmp_path / "climate.yaml"
    path.write_text(text)
    with pytest.raises(ClimateConfigError, match="duplicate key 'band'"):
        load_climate_config(path)


def test_unreadable_and_invalid_yaml_are_refused(tmp_path):
    with pytest.raises(ClimateConfigError, match="cannot read"):
        load_climate_config(tmp_path / "absent.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("a: [unclosed\n")
    with pytest.raises(ClimateConfigError, match="not valid YAML"):
        load_climate_config(bad)
    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("42\n")
    with pytest.raises(ClimateConfigError, match="expected a mapping"):
        load_climate_config(scalar)


def test_no_site_coordinates_or_money_in_the_shipped_config():
    text = SHIPPED.read_text().lower()
    assert not re.search(r"latitude|longitude|\beur\b|€", text)
