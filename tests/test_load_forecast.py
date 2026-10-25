"""Load-forecaster seam: registry, call contract, LF-0 bit-identity and leakage.

The leakage test is parametrized over every registered **non-oracle** forecaster, so a
forecaster added later is covered without a new test; the oracle (``perfect``) is
excluded explicitly through its registry flag, and a separate test proves that the
same perturbation does reach it (so the leakage test is not vacuous).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from battery_optimizer import benchmark as bm
from battery_optimizer import load_forecast as lf
from battery_optimizer.policies import (
    FORECAST_FALLBACK_RULE,
    FORECAST_METHOD,
    contiguous_day_bounds,
    forecast_load_trailing_slot_mean,
)

MADRID = ZoneInfo("Europe/Madrid")
# 2025-10-04T22:00Z is local midnight (CEST); the window crosses the 2025-10-26
# fall-back day, so one target day has 100 slots.
START = datetime(2025, 10, 4, 22, tzinfo=UTC)
N_DAYS = 40


def _series(seed: int = 3) -> tuple[np.ndarray, np.ndarray, list[tuple[object, int, int]]]:
    starts = [START + timedelta(minutes=15 * i) for i in range(N_DAYS * 96 + 4)]
    local_days = [s.astimezone(MADRID).date() for s in starts]
    rng = np.random.default_rng(seed)
    load = rng.random(len(starts)) * 0.4 + 0.05
    usable = rng.random(len(starts)) > 0.05  # a few gap-excluded intervals
    usable[96 * 3 : 96 * 4] = False  # and one fully missing day
    return load, usable, contiguous_day_bounds(local_days)


def _absurd(values: np.ndarray, usable: np.ndarray, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    """Every post-cutoff value replaced by an absurd one, and the mask scrambled."""
    load = values.copy()
    tail = load.size - cutoff
    pattern = np.array([1e12, -1e9, np.nan, np.inf, -np.inf, 0.0, 7.5e300])
    load[cutoff:] = np.resize(pattern, tail)
    mask = usable.copy()
    mask[cutoff:] = ~mask[cutoff:]
    return load, mask


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_defaults_agree_across_modules(self) -> None:
        assert lf.DEFAULT_LOAD_FORECASTER == "trailing_mean"
        assert bm.DEFAULT_LOAD_FORECASTER == lf.DEFAULT_LOAD_FORECASTER

    def test_lf0_and_the_oracle_are_registered(self) -> None:
        assert lf.forecaster_names()[:2] == ("trailing_mean", "perfect")
        assert "perfect" not in lf.forecaster_names(include_oracle=False)
        trailing = lf.get_forecaster("trailing_mean")
        assert trailing.oracle is False
        assert dict(trailing.defaults) == {"window_days": 28}
        assert trailing.method == FORECAST_METHOD
        assert trailing.fallback_rule == FORECAST_FALLBACK_RULE
        perfect = lf.get_forecaster("perfect")
        assert perfect.oracle is True
        assert perfect.provenance({})["oracle"] is True

    def test_unknown_name_lists_the_registered_ones(self) -> None:
        with pytest.raises(ValueError, match="trailing_mean"):
            lf.get_forecaster("nope")

    def test_params_resolution(self) -> None:
        assert lf.resolve_forecaster_params("trailing_mean", window_days=7) == {"window_days": 7}
        # The shared window knob only feeds forecasters that declare a window.
        assert lf.resolve_forecaster_params("perfect", window_days=7) == {}
        with pytest.raises(ValueError, match="no param 'alpha'"):
            lf.resolve_forecaster_params("trailing_mean", overrides={"alpha": 1})

    def test_adding_a_forecaster_is_one_decorated_function(self, monkeypatch) -> None:
        monkeypatch.setattr(lf, "_REGISTRY", dict(lf._REGISTRY))

        @lf.register_forecaster("flat_test")
        def _flat(request: lf.ForecastRequest, *, level: float = 0.25) -> list[lf.DayForecast]:
            """Constant level for every target slot."""
            return [
                (np.full(request.days[j][2] - request.days[j][1], level), {"local_day": "x"})
                for j in request.target_days
            ]

        assert "flat_test" in lf.forecaster_names(include_oracle=False)
        assert lf.get_forecaster("flat_test").defaults == {"level": 0.25}
        load, usable, days = _series()
        out = lf.run_forecaster(
            "flat_test", load_kwh=load, usable=usable, days=days, cutoff_idx=96, target_days=[1]
        )
        assert np.all(out.values[days[1][1] : days[1][2]] == 0.25)
        assert out.provenance["params"] == {"level": 0.25}

    def test_params_without_default_are_refused(self, monkeypatch) -> None:
        monkeypatch.setattr(lf, "_REGISTRY", dict(lf._REGISTRY))
        with pytest.raises(ValueError, match="needs a default"):

            @lf.register_forecaster("bad_test")
            def _bad(request: lf.ForecastRequest, *, k: int) -> list[lf.DayForecast]:
                return []


# ---------------------------------------------------------------------------
# LF-0 through the seam is the pre-seam function, bit for bit
# ---------------------------------------------------------------------------


class TestTrailingMeanBitIdentity:
    @pytest.mark.parametrize("window_days", [1, 7, 28])
    def test_day_ahead_vintage_matches_whole_array_function(self, window_days: int) -> None:
        load, usable, days = _series()
        canonical, diag = forecast_load_trailing_slot_mean(
            load, days, window_days=window_days, usable=usable
        )
        result = lf.forecast_each_day_at_its_start(
            "trailing_mean",
            load_kwh=load,
            usable=usable,
            days=days,
            target_days=range(len(days)),
            params={"window_days": window_days},
        )
        assert np.array_equal(result.values, canonical)
        for record, full in zip(result.per_day, diag, strict=True):
            assert {k: full[k] for k in record} == record

    @pytest.mark.parametrize("cutoff_day", [0, 5, 22, 39])
    def test_single_cutoff_matches_truncated_mask(self, cutoff_day: int) -> None:
        load, usable, days = _series()
        cutoff = days[cutoff_day][1]
        truncated = usable.copy()
        truncated[cutoff:] = False
        canonical, _diag = forecast_load_trailing_slot_mean(
            load, days, window_days=28, usable=truncated
        )
        result = lf.run_forecaster(
            "trailing_mean",
            load_kwh=load,
            usable=usable,
            days=days,
            cutoff_idx=cutoff,
            target_days=range(len(days)),
        )
        assert np.array_equal(result.values, canonical)


# ---------------------------------------------------------------------------
# Leakage: every non-oracle forecaster, automatically
# ---------------------------------------------------------------------------


NON_ORACLE = lf.forecaster_names(include_oracle=False)


class TestLeakage:
    @pytest.mark.parametrize("name", NON_ORACLE)
    @pytest.mark.parametrize("cutoff", ["day_start", "mid_day"])
    def test_post_cutoff_values_cannot_move_the_forecast(self, name: str, cutoff: str) -> None:
        load, usable, days = _series()
        cut = days[30][1] if cutoff == "day_start" else days[30][1] + 57
        targets = [j for j, (_l, start, _s) in enumerate(days) if start >= days[30][1]]
        base = lf.run_forecaster(
            name, load_kwh=load, usable=usable, days=days, cutoff_idx=cut, target_days=targets
        )
        bad_load, bad_mask = _absurd(load, usable, cut)
        moved = lf.run_forecaster(
            name,
            load_kwh=bad_load,
            usable=bad_mask,
            days=days,
            cutoff_idx=cut,
            target_days=targets,
        )
        assert np.array_equal(base.values, moved.values)
        assert base.per_day == moved.per_day

    @pytest.mark.parametrize("name", NON_ORACLE)
    def test_day_ahead_vintage_ignores_the_target_day_and_later(self, name: str) -> None:
        load, usable, days = _series()
        target = 31
        cut = days[target][1]
        base = lf.forecast_each_day_at_its_start(
            name, load_kwh=load, usable=usable, days=days, target_days=[target]
        )
        bad_load, bad_mask = _absurd(load, usable, cut)
        moved = lf.forecast_each_day_at_its_start(
            name, load_kwh=bad_load, usable=bad_mask, days=days, target_days=[target]
        )
        assert np.array_equal(base.values, moved.values)

    def test_the_oracle_is_excluded_by_its_flag_and_is_moved_by_the_same_perturbation(
        self,
    ) -> None:
        oracles = [n for n in lf.forecaster_names() if lf.get_forecaster(n).oracle]
        assert oracles == ["perfect"]
        assert not set(oracles) & set(NON_ORACLE)
        load, usable, days = _series()
        cut = days[30][1]
        kwargs = {"days": days, "cutoff_idx": cut, "target_days": [30], "allow_oracle": True}
        base = lf.run_forecaster("perfect", load_kwh=load, usable=usable, **kwargs)
        bad_load, bad_mask = _absurd(load, usable, cut)
        moved = lf.run_forecaster("perfect", load_kwh=bad_load, usable=bad_mask, **kwargs)
        assert not np.array_equal(base.values, moved.values)


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------


class TestPerfect:
    def test_refused_without_allow_oracle(self) -> None:
        load, usable, days = _series()
        with pytest.raises(ValueError, match="oracle"):
            lf.run_forecaster(
                "perfect", load_kwh=load, usable=usable, days=days, cutoff_idx=0, target_days=[0]
            )

    def test_returns_the_measured_load_including_unusable_slots(self) -> None:
        load, usable, days = _series()
        load[days[5][1] + 3] = np.nan  # a non-finite value is zeroed and counted
        result = lf.run_forecaster(
            "perfect",
            load_kwh=load,
            usable=usable,
            days=days,
            cutoff_idx=days[5][1],
            target_days=[5, 6],
            allow_oracle=True,
        )
        lo, hi = days[5][1], days[6][2]
        expected = np.where(np.isfinite(load[lo:hi]), load[lo:hi], 0.0)
        assert np.array_equal(result.values[lo:hi], expected)
        assert result.per_day[0]["slots_nonfinite_zeroed"] == 1
        assert result.per_day[0]["oracle"] is True
        assert result.per_day[0]["leaks_future"] is True
        assert result.provenance["oracle"] is True


class TestSeamValidation:
    def test_wrong_length_from_a_forecaster_is_refused(self, monkeypatch) -> None:
        monkeypatch.setattr(lf, "_REGISTRY", dict(lf._REGISTRY))

        @lf.register_forecaster("short_test")
        def _short(request: lf.ForecastRequest) -> list[lf.DayForecast]:
            return [(np.zeros(3), {}) for _ in request.target_days]

        load, usable, days = _series()
        with pytest.raises(ValueError, match="returned 3 values"):
            lf.run_forecaster(
                "short_test", load_kwh=load, usable=usable, days=days, cutoff_idx=0, target_days=[0]
            )

    def test_cutoff_outside_the_measurements_is_refused(self) -> None:
        load, usable, days = _series()
        with pytest.raises(ValueError, match="cutoff_idx"):
            lf.run_forecaster(
                "trailing_mean",
                load_kwh=load[:10],
                usable=usable[:10],
                days=days,
                cutoff_idx=11,
                target_days=[0],
            )

    def test_history_handed_to_a_forecaster_is_read_only_and_cut(self, monkeypatch) -> None:
        monkeypatch.setattr(lf, "_REGISTRY", dict(lf._REGISTRY))
        seen: dict[str, object] = {}

        @lf.register_forecaster("spy_test")
        def _spy(request: lf.ForecastRequest) -> list[lf.DayForecast]:
            seen["len"] = request.history_load_kwh.size
            seen["writeable"] = request.history_load_kwh.flags.writeable
            seen["measured"] = request.measured_load_kwh
            return [
                (np.zeros(request.days[j][2] - request.days[j][1]), {}) for j in request.target_days
            ]

        load, usable, days = _series()
        lf.run_forecaster(
            "spy_test", load_kwh=load, usable=usable, days=days, cutoff_idx=200, target_days=[3]
        )
        assert seen == {"len": 200, "writeable": False, "measured": None}
