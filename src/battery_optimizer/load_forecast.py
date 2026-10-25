"""Load-forecaster seam: one call contract and a registry of named forecasters.

Every load forecast the benchmark rungs (``day_ahead``, ``rolling_horizon`` in causal
mode) and the shadow service plan on goes through :func:`run_forecaster`.  The caller
states **what was known** (the measured series, its usability mask and the decision's
information cutoff) and **what to forecast** (whole local days of a partition); the
seam hands the forecaster only the history strictly before the cutoff and returns
the forecast, per-day diagnostics and provenance.

The contract, stated once:

* ``days`` is a ``(label, start, stop)`` partition of local calendar days
  (:func:`battery_optimizer.policies.contiguous_day_bounds`) over the whole index
  space, including days past the cutoff.  Day structure (lengths, DST) is calendar
  knowledge, not measured data, so a forecaster may read it freely.
* ``cutoff_idx`` is the first index the decision may **not** know.  A non-oracle
  forecaster receives ``load[:cutoff_idx]`` and ``usable[:cutoff_idx]`` as read-only
  views and nothing else from the measured series, so a post-cutoff value cannot
  reach it.  ``tests/test_load_forecast.py`` pins that for every registered
  non-oracle forecaster, so a new one is covered without writing a new test.
* ``target_days`` are indices into ``days``.  The forecaster returns one array per
  target day (the day's full length) and one diagnostics dict per target day.

Adding a forecaster is one decorated function::

    @register_forecaster("median", window_days=28)
    def _median(request: ForecastRequest, *, window_days: int) -> list[DayForecast]:
        ...

Keyword-only parameters with their defaults become the forecaster's recorded
``params``.  A forecaster that needs future measurements (an upper bound, never a
deployable model) is registered with ``oracle=True``: it is the only kind that sees
``request.measured_load_kwh``, :func:`run_forecaster` refuses it unless the caller
passes ``allow_oracle=True``, and the service configuration rejects it outright.

Scoring a forecast against the measured load needs hindsight and therefore lives
outside the forecasters (:func:`battery_optimizer.policies.score_day_forecast`).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .policies import (
    DEFAULT_FORECAST_WINDOW_DAYS,
    FORECAST_FALLBACK_RULE,
    FORECAST_METHOD,
    trailing_slot_mean_for_day,
)

__all__ = [
    "DEFAULT_LOAD_FORECASTER",
    "DayForecast",
    "ForecastRequest",
    "ForecastResult",
    "LoadForecaster",
    "forecast_each_day_at_its_start",
    "forecaster_names",
    "get_forecaster",
    "register_forecaster",
    "resolve_forecaster_params",
    "run_forecaster",
]

#: The production and benchmark default (LF-0).
DEFAULT_LOAD_FORECASTER = "trailing_mean"

#: One target day's forecast: ``(values, diagnostics)``.
DayForecast = tuple[np.ndarray, dict[str, Any]]


@dataclass(frozen=True)
class ForecastRequest:
    """What a forecaster may see for one decision."""

    history_load_kwh: np.ndarray
    """Measured load at indices ``[0, cutoff_idx)``, read-only."""

    history_usable: np.ndarray
    """Usability mask at indices ``[0, cutoff_idx)``, read-only."""

    days: Sequence[tuple[Any, int, int]]
    """Local-day partition of the whole index space (history and beyond)."""

    cutoff_idx: int
    """First index the decision does not know; equals ``len(history_load_kwh)``."""

    target_days: tuple[int, ...]
    """Indices into :attr:`days` to forecast."""

    measured_load_kwh: np.ndarray | None = None
    """The whole measured series, past the cutoff too.  ``None`` unless the forecaster
    is registered ``oracle=True``."""


@dataclass(frozen=True)
class LoadForecaster:
    """A registered forecaster: its function, default params and oracle flag."""

    name: str
    fn: Callable[..., list[DayForecast]]
    defaults: Mapping[str, Any]
    oracle: bool
    method: str
    fallback_rule: str
    description: str

    def provenance(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The record every artifact carries next to its lineage."""
        return {
            "name": self.name,
            "params": dict(params),
            "oracle": self.oracle,
            "method": self.method,
            "description": self.description,
        }


@dataclass(frozen=True)
class ForecastResult:
    """Output of :func:`run_forecaster`."""

    values: np.ndarray
    """Full-length (``days[-1][2]``) forecast; zero outside the target days.  Read-only."""

    per_day: list[dict[str, Any]]
    """One diagnostics dict per target day, in ``target_days`` order."""

    provenance: dict[str, Any] = field(default_factory=dict)


_REGISTRY: dict[str, LoadForecaster] = {}


def register_forecaster(
    name: str,
    *,
    oracle: bool = False,
    method: str | None = None,
    fallback_rule: str = "",
    description: str = "",
) -> Callable[[Callable[..., list[DayForecast]]], Callable[..., list[DayForecast]]]:
    """Register ``fn(request, **params) -> list[DayForecast]`` under *name*.

    The function's keyword-only parameters, with their defaults, are the forecaster's
    params; every one must have a default so the forecaster is runnable by name alone.
    """

    def decorate(fn: Callable[..., list[DayForecast]]) -> Callable[..., list[DayForecast]]:
        if name in _REGISTRY:
            raise ValueError(f"load forecaster {name!r} is already registered")
        defaults: dict[str, Any] = {}
        for param in inspect.signature(fn).parameters.values():
            if param.kind is not inspect.Parameter.KEYWORD_ONLY:
                continue
            if param.default is inspect.Parameter.empty:
                raise ValueError(f"load forecaster {name!r}: param {param.name!r} needs a default")
            defaults[param.name] = param.default
        _REGISTRY[name] = LoadForecaster(
            name=name,
            fn=fn,
            defaults=defaults,
            oracle=oracle,
            method=method or name,
            fallback_rule=fallback_rule,
            description=description or (inspect.getdoc(fn) or "").split("\n", 1)[0],
        )
        return fn

    return decorate


def forecaster_names(*, include_oracle: bool = True) -> tuple[str, ...]:
    """Registered names, in registration order."""
    return tuple(n for n, f in _REGISTRY.items() if include_oracle or not f.oracle)


def get_forecaster(name: str) -> LoadForecaster:
    """The registered forecaster called *name*, or ``ValueError`` naming the known ones."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown load forecaster {name!r}; registered: {', '.join(forecaster_names())}"
        ) from None


def resolve_forecaster_params(
    name: str,
    *,
    window_days: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Defaults of *name*, with the shared ``window_days`` knob and *overrides* applied.

    ``window_days`` is the existing forecast-window setting (benchmark
    ``--day-ahead-forecast-window-days``, service ``planning.forecast_window_days``); it
    only feeds forecasters that declare a ``window_days`` param.  Unknown override keys
    are refused rather than silently ignored.
    """
    forecaster = get_forecaster(name)
    params = dict(forecaster.defaults)
    if window_days is not None and "window_days" in params:
        params["window_days"] = int(window_days)
    for key, value in (overrides or {}).items():
        if key not in params:
            raise ValueError(
                f"load forecaster {name!r} has no param {key!r}; params: {sorted(params)}"
            )
        params[key] = value
    return params


def _read_only_view(values: np.ndarray, stop: int) -> np.ndarray:
    view = values[:stop].view()
    view.setflags(write=False)
    return view


def run_forecaster(
    name: str,
    *,
    load_kwh: np.ndarray | Sequence[float],
    usable: np.ndarray | Sequence[bool],
    days: Sequence[tuple[Any, int, int]],
    cutoff_idx: int,
    target_days: Sequence[int],
    params: Mapping[str, Any] | None = None,
    allow_oracle: bool = False,
) -> ForecastResult:
    """Forecast *target_days* with the forecaster *name*, knowing only ``[0, cutoff_idx)``.

    Args:
        load_kwh / usable: the measured series and its mask over ``[0, len)``; may be
            shorter than the partition (the service has no measurements past the
            cutoff), never shorter than *cutoff_idx*.
        days: local-day partition; the result has ``days[-1][2]`` entries.
        cutoff_idx: first index the decision does not know.
        target_days: indices into *days*.
        params: forecaster params; ``None`` means its registered defaults.
        allow_oracle: must be ``True`` to run an ``oracle=True`` forecaster.  Only the
            offline benchmark passes it.

    Raises:
        ValueError: unknown forecaster, an oracle without ``allow_oracle``, malformed
            inputs, or a forecaster returning the wrong shape or a non-finite value.
    """
    forecaster = get_forecaster(name)
    if forecaster.oracle and not allow_oracle:
        raise ValueError(
            f"load forecaster {name!r} is an oracle (it reads measurements past the "
            "decision cutoff) and is only allowed in offline evaluation"
        )
    resolved = resolve_forecaster_params(name, overrides=params)
    load = np.asarray(load_kwh, dtype=float).reshape(-1)
    keep = np.asarray(usable, dtype=bool).reshape(-1)
    if keep.size != load.size:
        raise ValueError(f"usable has {keep.size} entries, load_kwh has {load.size}")
    if not days:
        raise ValueError("days is empty: there is no local day to forecast")
    total = int(days[-1][2])
    if not 0 <= cutoff_idx <= load.size:
        raise ValueError(f"cutoff_idx {cutoff_idx} is outside 0..{load.size}")
    targets = tuple(int(j) for j in target_days)
    for j in targets:
        if not 0 <= j < len(days):
            raise ValueError(f"target day index {j} is outside the partition of {len(days)}")
    request = ForecastRequest(
        history_load_kwh=_read_only_view(load, cutoff_idx),
        history_usable=_read_only_view(keep, cutoff_idx),
        days=days,
        cutoff_idx=int(cutoff_idx),
        target_days=targets,
        measured_load_kwh=_read_only_view(load, load.size) if forecaster.oracle else None,
    )
    out = forecaster.fn(request, **resolved)
    if len(out) != len(targets):
        raise ValueError(
            f"load forecaster {name!r} returned {len(out)} days for {len(targets)} targets"
        )
    values = np.zeros(total, dtype=float)
    per_day: list[dict[str, Any]] = []
    for j, (day_values, record) in zip(targets, out, strict=True):
        _label, start, stop = days[j]
        day_values = np.asarray(day_values, dtype=float).reshape(-1)
        if day_values.size != stop - start:
            raise ValueError(
                f"load forecaster {name!r} returned {day_values.size} values for a "
                f"{stop - start}-interval day"
            )
        if not np.isfinite(day_values).all():
            raise ValueError(f"load forecaster {name!r} returned a non-finite value")
        values[start:stop] = day_values
        per_day.append(record)
    values.setflags(write=False)
    provenance = forecaster.provenance(resolved)
    provenance["cutoff_idx"] = int(cutoff_idx)
    return ForecastResult(values=values, per_day=per_day, provenance=provenance)


def forecast_each_day_at_its_start(
    name: str,
    *,
    load_kwh: np.ndarray,
    usable: np.ndarray,
    days: Sequence[tuple[Any, int, int]],
    target_days: Sequence[int],
    params: Mapping[str, Any] | None = None,
    allow_oracle: bool = False,
) -> ForecastResult:
    """The day-ahead vintage: each target day cut at its **own** local-day start.

    One :func:`run_forecaster` call per target day with ``cutoff_idx = days[j][1]`` —
    what a controller standing at that local midnight knows.  The result's
    ``provenance['cutoff_idx']`` is ``None`` because the cutoff differs per day.
    """
    load = np.asarray(load_kwh, dtype=float).reshape(-1)
    keep = np.asarray(usable, dtype=bool).reshape(-1)
    total = int(days[-1][2]) if days else 0
    values = np.zeros(total, dtype=float)
    per_day: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    for j in target_days:
        result = run_forecaster(
            name,
            load_kwh=load,
            usable=keep,
            days=days,
            cutoff_idx=int(days[j][1]),
            target_days=(int(j),),
            params=params,
            allow_oracle=allow_oracle,
        )
        _label, start, stop = days[j]
        values[start:stop] = result.values[start:stop]
        per_day.extend(result.per_day)
        provenance = result.provenance
    if not provenance:
        provenance = get_forecaster(name).provenance(
            resolve_forecaster_params(name, overrides=params)
        )
    provenance["cutoff_idx"] = None
    provenance["cutoff_rule"] = "each target local day is cut at its own local-day start"
    values.setflags(write=False)
    return ForecastResult(values=values, per_day=per_day, provenance=provenance)


# ---------------------------------------------------------------------------
# Registered forecasters
# ---------------------------------------------------------------------------


@register_forecaster(
    "trailing_mean",
    method=FORECAST_METHOD,
    fallback_rule=FORECAST_FALLBACK_RULE,
)
def _trailing_mean(
    request: ForecastRequest, *, window_days: int = DEFAULT_FORECAST_WINDOW_DAYS
) -> list[DayForecast]:
    """LF-0: same-slot-position mean over the previous window_days local days."""
    if int(window_days) < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")
    load = request.history_load_kwh
    keep = request.history_usable
    if not np.isfinite(load[keep]).all():
        bad = int(np.flatnonzero(keep & ~np.isfinite(load))[0])
        raise ValueError(f"load_kwh contains a non-finite value at index {bad}")
    return [
        trailing_slot_mean_for_day(load, keep, request.days, j, int(window_days))
        for j in request.target_days
    ]


#: How the ``perfect`` oracle treats a gap-excluded interval, recorded in every run.
PERFECT_UNUSABLE_RULE = (
    "the measured value is passed through verbatim on every interval, usable or not — the "
    "same load array the whole-horizon oracle optimises against (the benchmark already maps "
    "a null measurement to 0.0). A gap-excluded interval carries zero flow for every policy "
    "through the shared usable mask, so its value cannot change a plan; a non-finite value "
    "(none occurs in a built dataset) is replaced by 0.0 and counted"
)


@register_forecaster(
    "perfect",
    oracle=True,
    method="perfect_measured_load",
    fallback_rule=PERFECT_UNUSABLE_RULE,
)
def _perfect(request: ForecastRequest) -> list[DayForecast]:
    """ORACLE, offline only: the measured load of each target day (leaks the future)."""
    measured = request.measured_load_kwh
    if measured is None:
        raise ValueError("the perfect forecaster needs the measured series")
    out: list[DayForecast] = []
    for j in request.target_days:
        label, start, stop = request.days[j]
        raw = np.zeros(stop - start, dtype=float)
        have = max(0, min(stop, measured.size) - start)
        raw[:have] = measured[start : start + have]
        finite = np.isfinite(raw)
        values = np.where(finite, raw, 0.0)
        out.append(
            (
                values,
                {
                    "local_day": str(label),
                    "intervals": int(stop - start),
                    "oracle": True,
                    "leaks_future": True,
                    "slots_nonfinite_zeroed": int((~finite).sum()),
                    "slots_past_measurements": int(stop - start - have),
                    "slots_pooled_fallback": 0,
                    "slots_zero_no_history": 0,
                    "fallback": "none",
                    "forecast_fallback_rule": PERFECT_UNUSABLE_RULE,
                },
            )
        )
    return out
