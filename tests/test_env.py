"""Tests for the ``BATTERY_OPTIMIZER_*`` env lookup with legacy ``ENERGY_BASELINE_*`` fallback."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from battery_optimizer._env import get_env, legacy_name

NEW = "BATTERY_OPTIMIZER_HA_TOKEN"
OLD = "ENERGY_BASELINE_HA_TOKEN"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(NEW, raising=False)
    monkeypatch.delenv(OLD, raising=False)


def test_legacy_name_mapping():
    assert legacy_name("BATTERY_OPTIMIZER_DATA_DIR") == "ENERGY_BASELINE_DATA_DIR"


def test_legacy_name_rejects_foreign_prefix():
    with pytest.raises(ValueError):
        legacy_name("SOMETHING_ELSE")


def test_unset_returns_none_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert get_env(NEW) is None


def test_new_name_read_without_warning(monkeypatch):
    monkeypatch.setenv(NEW, "new")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert get_env(NEW) == "new"


def test_legacy_fallback_warns_naming_both(monkeypatch):
    monkeypatch.setenv(OLD, "old")
    with pytest.warns(DeprecationWarning) as rec:
        assert get_env(NEW) == "old"
    msg = str(rec[0].message)
    assert OLD in msg and NEW in msg


def test_new_wins_over_legacy_silently(monkeypatch):
    monkeypatch.setenv(NEW, "new")
    monkeypatch.setenv(OLD, "old")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert get_env(NEW) == "new"


def test_empty_new_falls_back_to_legacy(monkeypatch):
    monkeypatch.setenv(NEW, "")
    monkeypatch.setenv(OLD, "old")
    with pytest.warns(DeprecationWarning):
        assert get_env(NEW) == "old"


class TestCallSites:
    def test_data_dir_legacy_fallback(self, monkeypatch, tmp_path: Path):
        from battery_optimizer.paths import resolve_data_dir

        monkeypatch.delenv("BATTERY_OPTIMIZER_DATA_DIR", raising=False)
        monkeypatch.setenv("ENERGY_BASELINE_DATA_DIR", str(tmp_path / "legacy"))
        with pytest.warns(DeprecationWarning):
            assert resolve_data_dir() == (tmp_path / "legacy").resolve()

    def test_data_dir_new_wins(self, monkeypatch, tmp_path: Path):
        from battery_optimizer.paths import resolve_data_dir

        monkeypatch.setenv("BATTERY_OPTIMIZER_DATA_DIR", str(tmp_path / "new"))
        monkeypatch.setenv("ENERGY_BASELINE_DATA_DIR", str(tmp_path / "legacy"))
        assert resolve_data_dir() == (tmp_path / "new").resolve()

    def test_default_data_dir_keeps_legacy_name(self, monkeypatch):
        from battery_optimizer.paths import resolve_data_dir

        monkeypatch.delenv("BATTERY_OPTIMIZER_DATA_DIR", raising=False)
        monkeypatch.delenv("ENERGY_BASELINE_DATA_DIR", raising=False)
        assert resolve_data_dir() == (Path.home() / "data" / "energy-baseline").resolve()

    def test_config_path_new_and_legacy(self, monkeypatch, tmp_path: Path):
        from battery_optimizer.config import resolve_config_path

        monkeypatch.delenv("BATTERY_OPTIMIZER_CONFIG", raising=False)
        monkeypatch.setenv("ENERGY_BASELINE_CONFIG", str(tmp_path / "old.yaml"))
        with pytest.warns(DeprecationWarning):
            assert resolve_config_path() == (tmp_path / "old.yaml").resolve()
        monkeypatch.setenv("BATTERY_OPTIMIZER_CONFIG", str(tmp_path / "new.yaml"))
        assert resolve_config_path() == (tmp_path / "new.yaml").resolve()

    def test_ha_token_error_names_new_variable(self):
        from battery_optimizer.extract_ha import resolve_token

        with pytest.raises(ValueError, match="BATTERY_OPTIMIZER_HA_TOKEN"):
            resolve_token(None)

    def test_ha_token_legacy_fallback(self, monkeypatch):
        from battery_optimizer.extract_ha import resolve_token

        monkeypatch.setenv(OLD, "legacy-token")
        with pytest.warns(DeprecationWarning):
            assert resolve_token(None) == "legacy-token"

    def test_prometheus_url_legacy_fallback(self, monkeypatch):
        from battery_optimizer.extract_prom import resolve_base_url

        monkeypatch.delenv("BATTERY_OPTIMIZER_PROMETHEUS_URL", raising=False)
        monkeypatch.setenv("ENERGY_BASELINE_PROMETHEUS_URL", "http://legacy.example:9090")
        with pytest.warns(DeprecationWarning):
            assert resolve_base_url() == "http://legacy.example:9090"
        monkeypatch.setenv("BATTERY_OPTIMIZER_PROMETHEUS_URL", "http://new.example:9090")
        assert resolve_base_url() == "http://new.example:9090"

    def test_ha_url_legacy_fallback(self, monkeypatch):
        from battery_optimizer.extract_ha import resolve_base_url

        monkeypatch.delenv("BATTERY_OPTIMIZER_HA_URL", raising=False)
        monkeypatch.setenv("ENERGY_BASELINE_HA_URL", "http://legacy-ha.example")
        with pytest.warns(DeprecationWarning):
            assert resolve_base_url() == "http://legacy-ha.example"
