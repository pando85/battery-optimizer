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

Two optional capabilities, both declared at registration:

* ``fit=`` — a trained model.  ``fit(FitRequest, **params)`` returns any state; the
  seam fits **once per monthly walk-forward fold** on history strictly before the
  fold's first local day (:func:`fold_start_idx`) and hands the state to every
  decision in that fold as ``request.fitted``.  A fold is the local calendar month of
  the decision's cutoff, so the fit never sees the fold it forecasts.
* ``intraday_cutoff=True`` — the rolling-horizon rung cuts this forecaster's history
  at the event instant (14:00 / 20:30) instead of the event's local midnight, so it
  may use today's observed slots.  The default (``False``) keeps the conservative
  local-midnight cutoff.  ``day_ahead`` decides at local midnight, where the two
  cutoffs coincide.

* ``side_loader=`` — a forecaster that reads more than the load: aligned side series (a
  sub-metered heat pump, an outdoor temperature, control states, the PVPC prices).  The
  caller builds one :class:`SideInputs` for the whole index space
  (:func:`build_side_inputs`) and passes it as ``side=``.  The seam hands the forecaster a
  :class:`SideView` only: every measured side series is cut at ``cutoff_idx`` exactly like
  the load, and the prices at the first index that is **not yet public** at the cutoff
  instant (:func:`battery_optimizer.price_outlook.published_end_idx`), so an unpublished
  price cannot reach it.  A forecaster that does not declare a ``side_loader`` is never
  handed side data, and one that does must still run (degraded, with the reason in its
  diagnostics) when no side data is passed.

Scoring a forecast against the measured load needs hindsight and therefore lives
outside the forecasters (:func:`battery_optimizer.policies.score_day_forecast`).
"""

from __future__ import annotations

import bisect
import hashlib
import inspect
import json
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from .policies import (
    DEFAULT_FORECAST_WINDOW_DAYS,
    FORECAST_FALLBACK_RULE,
    FORECAST_METHOD,
    trailing_slot_mean_for_day,
)
from .price_outlook import PVPC_NEXT_DAY_PUBLISHED_FROM, published_end_idx

__all__ = [
    "DEFAULT_LOAD_FORECASTER",
    "FIT_FOLD_RULE",
    "SIDE_INPUT_RULE",
    "DayForecast",
    "FitRequest",
    "ForecastRequest",
    "ForecastResult",
    "LoadForecaster",
    "SideInputs",
    "SideView",
    "build_side_inputs",
    "fold_start_idx",
    "fold_state",
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

#: How a forecaster with a ``fit`` step is refit, recorded in its provenance.
FIT_FOLD_RULE = (
    "monthly walk-forward: one fit per local calendar month (the month of the decision's "
    "cutoff), on history strictly before that month's first local day; every decision in "
    "the month reuses that fit"
)

#: How a forecaster with side inputs sees them, recorded in its provenance.
SIDE_INPUT_RULE = (
    "side inputs reach the forecaster as an as-of view: every measured side series is cut at "
    "the decision cutoff, exactly like the load, and the prices at the first index that is not "
    "yet public at the cutoff instant (PVPC for the next local day counts as published from "
    f"{PVPC_NEXT_DAY_PUBLISHED_FROM:%H:%M} local); nothing at or past either boundary is "
    "readable"
)

#: Every interval is a quarter of an hour; side calendars are built on that grid.
SIDE_INTERVAL_SECONDS = 900


class SideInputs:
    """Aligned side series over the whole index space, built once by the caller.

    ``prices`` (EUR/kWh) and every entry of ``series`` have one value per interval of the
    local-day partition, ``NaN`` where unknown; ``interval_starts_utc`` is the grid.  The
    object is immutable (arrays are read-only) and carries a :attr:`fingerprint` of its
    content, which keys the fit cache.  Forecasters never receive it: they get a
    :class:`SideView` cut by :meth:`view`.
    """

    def __init__(
        self,
        *,
        interval_starts_utc: Sequence[datetime],
        tz: ZoneInfo,
        prices: np.ndarray | Sequence[float],
        series: Mapping[str, np.ndarray | Sequence[float]],
        source: Mapping[str, Any] | None = None,
    ) -> None:
        starts = tuple(
            s if s.tzinfo is not None else s.replace(tzinfo=UTC) for s in interval_starts_utc
        )
        self.n = len(starts)
        self.tz = tz
        self.interval_starts_utc = starts
        self.prices = self._frozen(prices, "prices")
        self.series = {name: self._frozen(values, name) for name, values in series.items()}
        self.source = dict(source or {})
        local = [s.astimezone(tz) for s in starts]
        self.local_hour = np.array([t.hour for t in local], dtype=np.int16)
        self.quarter = np.array([t.minute // 15 for t in local], dtype=np.int16)
        self.local_hour.setflags(write=False)
        self.quarter.setflags(write=False)
        # Real hours (four intervals each) and the local day they belong to.  A new local
        # day starts where the clock hour does not increase, which holds for the DST days
        # too (the repeated hour of a fall-back day does not wrap, the skipped one of a
        # spring-forward day only jumps).
        nreal = self.n // 4
        self.clock_of_real = self.local_hour[: 4 * nreal : 4].astype(np.int16)
        wrap = np.zeros(nreal, dtype=np.int64)
        if nreal > 1:
            wrap[1:] = (self.clock_of_real[1:] < self.clock_of_real[:-1]).astype(np.int64)
        self.day_of_real = np.cumsum(wrap)
        self.clock_of_real.setflags(write=False)
        self.day_of_real.setflags(write=False)
        if self.n:
            first = int(starts[0].timestamp())
            stamps = np.array([int(s.timestamp()) for s in starts], dtype=np.int64)
            if (stamps - first != np.arange(self.n) * SIDE_INTERVAL_SECONDS).any():
                raise ValueError("side inputs need a uniform 15-minute grid without gaps")
            if first % 3600 != 0:
                raise ValueError("side inputs need the grid to start on a full hour")
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.prices).tobytes())
        for name in sorted(self.series):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(self.series[name]).tobytes())
        digest.update(repr((self.n, starts[0] if starts else None, str(tz))).encode())
        self.fingerprint = digest.hexdigest()

    def _frozen(self, values: np.ndarray | Sequence[float], name: str) -> np.ndarray:
        arr = np.array(values, dtype=float).reshape(-1)
        if arr.size != self.n:
            raise ValueError(f"side series {name!r} has {arr.size} entries, the grid has {self.n}")
        arr.setflags(write=False)
        return arr

    def slice(self, stop: int) -> SideInputs:
        """The first *stop* intervals, for a dataset cut at an earlier instant."""
        if not 0 <= stop <= self.n:
            raise ValueError(f"cannot slice {self.n} intervals at {stop}")
        return SideInputs(
            interval_starts_utc=self.interval_starts_utc[:stop],
            tz=self.tz,
            prices=self.prices[:stop],
            series={k: v[:stop] for k, v in self.series.items()},
            source=self.source,
        )

    def view(
        self,
        cutoff_idx: int,
        days: Sequence[tuple[Any, int, int]],
        *,
        published_end: int | None = None,
    ) -> SideView:
        """What a decision cut at *cutoff_idx* may read.

        Measured series are cut at the cutoff.  Prices are cut at the first index not public
        at the cutoff instant, or at *published_end* when the caller states it (the fit uses
        the fold start).  The decision instant is the start of interval ``cutoff_idx``.
        """
        cut = min(max(int(cutoff_idx), 0), self.n)
        if published_end is None:
            probe = self.interval_starts_utc[min(cut, self.n - 1)] if self.n else None
            end = published_end_idx(probe, days, self.tz) if probe is not None else 0
        else:
            end = int(published_end)
        end = min(max(end, 0), self.n)

        def _ro(values: np.ndarray, stop: int) -> np.ndarray:
            v = values[:stop].view()
            v.setflags(write=False)
            return v

        n_days = len(days)
        if self.day_of_real.size and int(self.day_of_real[-1]) >= n_days:
            raise ValueError("the side inputs' local calendar and the day partition disagree")
        first_real = np.full((n_days, 24), -1, dtype=np.int64)
        # Assign in reverse so the FIRST real hour of a repeated clock hour wins.
        reverse = np.arange(self.day_of_real.size)[::-1]
        first_real[self.day_of_real[reverse], self.clock_of_real[reverse]] = reverse
        first_real.setflags(write=False)
        return SideView(
            cutoff_idx=cut,
            published_end_idx=end,
            interval_starts_utc=self.interval_starts_utc,
            tz=self.tz,
            local_hour=self.local_hour,
            quarter=self.quarter,
            day_of_real_hour=self.day_of_real,
            first_real_hour=first_real,
            series={name: _ro(values, cut) for name, values in self.series.items()},
            published_prices=_ro(self.prices, end),
            source=self.source,
        )

    def describe(self) -> dict[str, Any]:
        """The identity record of these side inputs (no values)."""
        return {
            **self.source,
            "fingerprint": self.fingerprint,
            "slots": self.n,
            "series": sorted(self.series),
        }


@dataclass(frozen=True)
class SideView:
    """The as-of side data of one decision (see :meth:`SideInputs.view`)."""

    cutoff_idx: int
    """First index the decision does not know; every ``series`` entry has this length."""

    published_end_idx: int
    """First index whose price is not public at the decision; ``published_prices`` length."""

    interval_starts_utc: Sequence[datetime]
    """UTC start of every interval of the partition (calendar knowledge)."""

    tz: ZoneInfo
    """The tariff's local time zone."""

    local_hour: np.ndarray
    """Local clock hour of every interval of the partition (calendar knowledge)."""

    quarter: np.ndarray
    """Quarter of the local clock hour (0..3) of every interval (calendar knowledge)."""

    day_of_real_hour: np.ndarray
    """Index into the day partition of every real hour (four intervals), calendar knowledge."""

    first_real_hour: np.ndarray
    """``(days, 24)``: index of the first real hour of each local clock hour of each local day,
    ``-1`` where the clock hour does not exist (the skipped hour of a spring-forward day)."""

    series: Mapping[str, np.ndarray]
    """Measured side series at indices ``[0, cutoff_idx)``, read-only, ``NaN`` unknown."""

    published_prices: np.ndarray
    """Prices at indices ``[0, published_end_idx)``, read-only, ``NaN`` unknown."""

    source: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FitRequest:
    """What a forecaster's ``fit`` step may see for one walk-forward fold."""

    history_load_kwh: np.ndarray
    """Measured load at indices ``[0, fit_cutoff_idx)``, read-only."""

    history_usable: np.ndarray
    """Usability mask at indices ``[0, fit_cutoff_idx)``, read-only."""

    days: Sequence[tuple[Any, int, int]]
    """Local-day partition of the whole index space (calendar knowledge)."""

    fit_cutoff_idx: int
    """First index of the fold; equals ``len(history_load_kwh)``."""

    fold: str
    """The fold's local month, ``YYYY-MM``."""

    side: SideView | None = None
    """Side data as of the fold start (prices only up to it), for a forecaster with a
    ``side_loader`` when the caller passed side inputs; else ``None``."""


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

    fitted: Any = None
    """The fold's fitted state for a forecaster registered with ``fit=``, else ``None``."""

    fit_cutoff_idx: int | None = None
    """First index of the fold the state was fitted before (``<= cutoff_idx``)."""

    side: SideView | None = None
    """The as-of side data for a forecaster registered with ``side_loader=`` when the caller
    passed side inputs; else ``None``."""


@dataclass(frozen=True)
class LoadForecaster:
    """A registered forecaster: its function, default params and capability flags."""

    name: str
    fn: Callable[..., list[DayForecast]]
    defaults: Mapping[str, Any]
    oracle: bool
    method: str
    fallback_rule: str
    description: str
    fit: Callable[..., Any] | None = None
    intraday_cutoff: bool = False
    side_loader: Callable[..., SideInputs] | None = None
    diagnose: Callable[..., dict[str, Any]] | None = None

    def provenance(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The record every artifact carries next to its lineage.

        The capability keys appear only when declared, so LF-0's record is unchanged.
        """
        record: dict[str, Any] = {
            "name": self.name,
            "params": dict(params),
            "oracle": self.oracle,
            "method": self.method,
            "description": self.description,
        }
        if self.fit is not None:
            record["fit"] = FIT_FOLD_RULE
        if self.intraday_cutoff:
            record["intraday_cutoff"] = True
        if self.side_loader is not None:
            record["side_inputs"] = SIDE_INPUT_RULE
        return record


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
    fit: Callable[..., Any] | None = None,
    intraday_cutoff: bool = False,
    side_loader: Callable[..., SideInputs] | None = None,
    diagnose: Callable[..., dict[str, Any]] | None = None,
) -> Callable[[Callable[..., list[DayForecast]]], Callable[..., list[DayForecast]]]:
    """Register ``fn(request, **params) -> list[DayForecast]`` under *name*.

    The function's keyword-only parameters, with their defaults, are the forecaster's
    params; every one must have a default so the forecaster is runnable by name alone.

    Args:
        fit: optional ``fit(FitRequest, **params) -> state``, called with the same params
            as the forecaster, once per monthly walk-forward fold (:data:`FIT_FOLD_RULE`).
        intraday_cutoff: the rolling-horizon rung hands this forecaster history up to
            the event instant instead of the event's local midnight.
        side_loader: ``side_loader(params, *, table, data_dir) -> SideInputs`` builds the
            side series from a built dataset table (:func:`build_side_inputs`); the
            forecaster then receives the as-of :class:`SideView`.
        diagnose: optional ``diagnose(*, side, load, usable, days, decisions, params) ->
            dict`` that the evaluation harness calls to report the forecaster's components
            (see :mod:`battery_optimizer.structured_forecast`); it needs hindsight and is
            never part of a forecast.
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
            fit=fit,
            intraday_cutoff=bool(intraday_cutoff),
            side_loader=side_loader,
            diagnose=diagnose,
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


def _label_date(label: Any) -> date:
    return label if isinstance(label, date) else date.fromisoformat(str(label)[:10])


def fold_start_idx(days: Sequence[tuple[Any, int, int]], cutoff_idx: int) -> tuple[int, str]:
    """``(first index, "YYYY-MM")`` of the monthly fold a decision cut at *cutoff_idx* is in.

    The fold is the local calendar month of the day holding *cutoff_idx* (the last day
    when the cutoff is past the partition); its first index is that month's first local
    day in the partition, never past *cutoff_idx*.
    """
    if not days:
        raise ValueError("days is empty: there is no local day to fold")
    probe = min(max(int(cutoff_idx), 0), int(days[-1][2]) - 1)
    j = next((k for k, (_l, lo, hi) in enumerate(days) if lo <= probe < hi), len(days) - 1)
    month = _label_date(days[j][0]).strftime("%Y-%m")
    while j > 0 and _label_date(days[j - 1][0]).strftime("%Y-%m") == month:
        j -= 1
    return int(min(days[j][1], cutoff_idx)), month


#: Fitted states, keyed by forecaster, params, fold cutoff and a digest of the history.
_FIT_CACHE: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_FIT_CACHE_SIZE = 64


def _fitted_state(
    forecaster: LoadForecaster,
    params: Mapping[str, Any],
    load: np.ndarray,
    keep: np.ndarray,
    days: Sequence[tuple[Any, int, int]],
    fit_cutoff: int,
    fold: str,
    side: SideInputs | None = None,
) -> Any:
    """The fold's state: fitted once on ``[0, fit_cutoff)`` and cached by content.

    Side inputs are immutable, so their fingerprint and the fold start identify exactly the
    side data the fit reads (``[0, fit_cutoff)``).
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(load[:fit_cutoff]).tobytes())
    digest.update(np.ascontiguousarray(keep[:fit_cutoff]).tobytes())
    digest.update(repr([(str(label), lo, hi) for label, lo, hi in days]).encode())
    key = (
        forecaster.name,
        id(forecaster.fit),
        json.dumps(dict(params), sort_keys=True, default=str),
        int(fit_cutoff),
        digest.hexdigest(),
        None if side is None else side.fingerprint,
    )
    if key in _FIT_CACHE:
        _FIT_CACHE.move_to_end(key)
        return _FIT_CACHE[key]
    if forecaster.fit is None:  # pragma: no cover - guarded by the caller
        raise ValueError(f"load forecaster {forecaster.name!r} has no fit step")
    state = forecaster.fit(
        FitRequest(
            history_load_kwh=_read_only_view(load, fit_cutoff),
            history_usable=_read_only_view(keep, fit_cutoff),
            days=days,
            fit_cutoff_idx=int(fit_cutoff),
            fold=fold,
            side=(
                None if side is None else side.view(fit_cutoff, days, published_end=int(fit_cutoff))
            ),
        ),
        **params,
    )
    _FIT_CACHE[key] = state
    while len(_FIT_CACHE) > _FIT_CACHE_SIZE:
        _FIT_CACHE.popitem(last=False)
    return state


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
    side: SideInputs | None = None,
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
        side: side inputs for the whole partition; only a forecaster registered with
            ``side_loader=`` receives them, as an as-of :class:`SideView`.

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
    use_side = side if forecaster.side_loader is not None else None
    if use_side is not None and use_side.n != total:
        raise ValueError(f"side inputs cover {use_side.n} intervals, the partition {total}")
    fitted: Any = None
    fit_cutoff: int | None = None
    fold: str | None = None
    if forecaster.fit is not None:
        fit_cutoff, fold = fold_start_idx(days, cutoff_idx)
        fitted = _fitted_state(
            forecaster, resolved, load, keep, days, fit_cutoff, fold, side=use_side
        )
    request = ForecastRequest(
        history_load_kwh=_read_only_view(load, cutoff_idx),
        history_usable=_read_only_view(keep, cutoff_idx),
        days=days,
        cutoff_idx=int(cutoff_idx),
        target_days=targets,
        measured_load_kwh=_read_only_view(load, load.size) if forecaster.oracle else None,
        fitted=fitted,
        fit_cutoff_idx=fit_cutoff,
        side=None if use_side is None else use_side.view(cutoff_idx, days),
    )
    out = forecaster.fn(request, **resolved)
    if len(out) != len(targets):
        raise ValueError(
            f"load forecaster {name!r} returned {len(out)} days for {len(targets)} targets"
        )
    values = np.zeros(total, dtype=float)
    per_day: list[dict[str, Any]] = []
    for j, (day_values, record) in zip(targets, out, strict=True):
        label, start, stop = days[j]
        day_values = np.asarray(day_values, dtype=float).reshape(-1)
        if day_values.size != stop - start:
            raise ValueError(
                f"load forecaster {name!r} returned {day_values.size} values for a "
                f"{stop - start}-interval day"
            )
        if not np.isfinite(day_values).all():
            raise ValueError(f"load forecaster {name!r} returned a non-finite value")
        values[start:stop] = day_values
        if "local_day" not in record:
            # Callers key diagnostics by day; a forecaster need not repeat the label.
            record = {**record, "local_day": str(label)}
        per_day.append(record)
    values.setflags(write=False)
    provenance = forecaster.provenance(resolved)
    provenance["cutoff_idx"] = int(cutoff_idx)
    if fit_cutoff is not None:
        provenance["fit_cutoff_idx"] = int(fit_cutoff)
        provenance["fold"] = fold
    return ForecastResult(values=values, per_day=per_day, provenance=provenance)


def build_side_inputs(
    name: str,
    params: Mapping[str, Any] | None = None,
    *,
    table: Any,
    data_dir: Path | str | None = None,
) -> SideInputs | None:
    """The side inputs forecaster *name* needs for a built dataset *table*, or ``None``.

    ``None`` for a forecaster that does not declare a ``side_loader``.  *table* is the
    (possibly ``--eval-end``-truncated) dataset table the run plans on; the loader aligns
    its side series to the table's own grid.
    """
    forecaster = get_forecaster(name)
    if forecaster.side_loader is None:
        return None
    resolved = resolve_forecaster_params(name, overrides=params)
    return forecaster.side_loader(resolved, table=table, data_dir=data_dir)


def fold_state(
    name: str,
    params: Mapping[str, Any],
    load_kwh: np.ndarray,
    usable: np.ndarray,
    days: Sequence[tuple[Any, int, int]],
    cutoff_idx: int,
    side: SideInputs | None = None,
) -> Any:
    """The fitted state the monthly fold of a decision cut at *cutoff_idx* uses.

    The same state :func:`run_forecaster` hands the forecaster as ``request.fitted`` (fitted
    once per fold and cached); ``None`` for a forecaster without a ``fit`` step.  The
    evaluation diagnostics use it to look inside a forecast.
    """
    forecaster = get_forecaster(name)
    if forecaster.fit is None:
        return None
    resolved = resolve_forecaster_params(name, overrides=params)
    fit_cutoff, fold = fold_start_idx(days, cutoff_idx)
    return _fitted_state(
        forecaster,
        resolved,
        np.asarray(load_kwh, dtype=float).reshape(-1),
        np.asarray(usable, dtype=bool).reshape(-1),
        days,
        fit_cutoff,
        fold,
        side=side if forecaster.side_loader is not None else None,
    )


def forecast_each_day_at_its_start(
    name: str,
    *,
    load_kwh: np.ndarray,
    usable: np.ndarray,
    days: Sequence[tuple[Any, int, int]],
    target_days: Sequence[int],
    params: Mapping[str, Any] | None = None,
    allow_oracle: bool = False,
    side: SideInputs | None = None,
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
            side=side,
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
    provenance.pop("fit_cutoff_idx", None)
    provenance.pop("fold", None)
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


# ---------------------------------------------------------------------------
# P2: cheap statistical candidates (same history, same slot positions as LF-0)
# ---------------------------------------------------------------------------
#
# Every candidate below reads the same thing LF-0 reads: the usable load of the
# local days before the target day, aligned by **slot position inside the local-day
# partition** (``days[p]``), never by an offset into a sliced array — a 92- or
# 100-interval DST day therefore stays one local day, exactly as in
# :func:`battery_optimizer.policies.trailing_slot_mean_for_day`.  An interval that is
# not usable is *unknown* (NaN in the matrices below): it never contributes and is
# never counted as zero.  A slot no day in the window knows falls back to the pooled
# mean of the window, and a window with no usable interval at all yields 0.0 kWh,
# flagged ``zero_no_history`` in the per-day record — the same contract as LF-0, so
# the rolling-horizon support diagnostics read every candidate the same way.

_CANDIDATE_FALLBACK_RULE = (
    "per slot k of the target local day: the statistic over the usable load at position k "
    "of the candidate's history days (only days that have position k and are usable there); "
    "if none does (a 25-hour fall-back day against shorter history days), the pooled mean "
    "of every usable interval of those days — a flat profile; if they hold no usable "
    "interval at all, 0.0 kWh, flagged zero_no_history. Unusable intervals are unknown, "
    "never zero"
)


def _history_day_count(request: ForecastRequest) -> int:
    """How many leading local days start before the cutoff (the only ones with data)."""
    return bisect.bisect_left(request.days, request.cutoff_idx, key=lambda d: d[1])


def _history_rows(request: ForecastRequest, day_indices: Sequence[int], m: int) -> np.ndarray:
    """Matrix ``(len(day_indices), m)`` of the usable load of those local days, by slot position.

    Unknown entries (unusable intervals, positions the day does not have, days past
    the cutoff) are NaN.  Only ``history_load_kwh`` / ``history_usable`` are read.
    """
    load = request.history_load_kwh
    keep = request.history_usable
    seen = load.size
    rows = np.full((len(day_indices), m), np.nan, dtype=float)
    for r, p in enumerate(day_indices):
        _label, p_start, p_stop = request.days[p]
        if p_start >= seen:
            continue
        p_stop = min(int(p_stop), seen)
        usable = keep[p_start:p_stop]
        if not usable.any():
            continue
        values = load[p_start:p_stop]
        if not np.isfinite(values[usable]).all():
            bad = p_start + int(np.flatnonzero(usable & ~np.isfinite(values))[0])
            raise ValueError(f"load_kwh contains a non-finite value at index {bad}")
        shared = min(m, p_stop - p_start)
        take = usable[:shared]
        rows[r, :shared][take] = values[:shared][take]
    return rows


def _weighted_mean(rows: np.ndarray, weights: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-slot (weighted) mean of the known entries, and the per-slot sample counts."""
    known = np.isfinite(rows)
    w = known.astype(float) if weights is None else known * weights[:, None]
    total = w.sum(axis=0)
    safe = np.where(total > 0, total, 1.0)
    return (np.where(known, rows, 0.0) * w).sum(axis=0) / safe, known.sum(axis=0)


def _median_of_columns(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-slot median of the known entries (unknown sorts last), and the sample counts."""
    ordered = np.sort(rows, axis=0)
    counts = np.isfinite(rows).sum(axis=0)
    if rows.shape[0] == 0:
        return np.zeros(rows.shape[1]), counts
    low = np.take_along_axis(ordered, np.maximum((counts - 1) // 2, 0)[None, :], axis=0)[0]
    high = np.take_along_axis(ordered, (counts // 2).clip(max=rows.shape[0] - 1)[None, :], axis=0)[
        0
    ]
    return np.where(counts > 0, (low + high) / 2.0, 0.0), counts


def _trimmed_mean_of_columns(rows: np.ndarray, trim: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean of each slot's known values after dropping ``floor(trim * n)`` at each end."""
    ordered = np.sort(rows, axis=0)
    counts = np.isfinite(rows).sum(axis=0)
    if rows.shape[0] == 0:
        return np.zeros(rows.shape[1]), counts
    cut = np.floor(trim * counts + 1e-9).astype(int)
    rank = np.arange(rows.shape[0])[:, None]
    inside = (rank >= cut[None, :]) & (rank < (counts - cut)[None, :])
    kept = inside.sum(axis=0)
    total = np.where(inside, ordered, 0.0).sum(axis=0)
    return np.where(kept > 0, total / np.maximum(kept, 1), 0.0), counts


def _complete_day(
    label: Any,
    m: int,
    profile: np.ndarray,
    counts: np.ndarray,
    rows: np.ndarray,
    weights: np.ndarray | None,
    *,
    requested: int,
    extra: Mapping[str, Any] | None = None,
) -> DayForecast:
    """Fill slots without a sample from the pooled mean and build the per-day record."""
    known = np.isfinite(rows)
    pooled_count = int(known.sum())
    if pooled_count:
        w = known.astype(float) if weights is None else known * weights[:, None]
        pooled_mean = float((np.where(known, rows, 0.0) * w).sum() / w.sum())
    else:
        pooled_mean = 0.0
    used = counts > 0
    values = np.where(used, profile, pooled_mean)
    pooled_slots = int((~used).sum()) if pooled_count else 0
    empty_slots = int((~used).sum()) if not pooled_count else 0
    fallback = "none"
    if pooled_slots:
        fallback = "pooled_window_mean"
    if empty_slots:
        fallback = "zero_no_history"
    record: dict[str, Any] = {
        "local_day": str(label),
        "intervals": int(m),
        "window_days_requested": int(requested),
        "window_days_available": int(rows.shape[0]),
        "window_days_with_usable_data": int(known.any(axis=1).sum()),
        "slots_using_history": int(used.sum()),
        "slots_pooled_fallback": pooled_slots,
        "slots_zero_no_history": empty_slots,
        "samples_min": int(counts[used].min()) if used.any() else 0,
        "samples_max": int(counts[used].max()) if used.any() else 0,
        "fallback": fallback,
        "forecast_fallback_rule": _CANDIDATE_FALLBACK_RULE,
    }
    if extra:
        record.update(extra)
    return values, record


def _window_days_before(request: ForecastRequest, j: int, window: int, history: int) -> range:
    """Day indices ``[j - window, j)`` that start before the cutoff."""
    return range(max(0, j - window), min(j, history))


def _require_finite_history(request: ForecastRequest) -> None:
    """Refuse a non-finite usable value, as ``trailing_mean`` does (kernel-based candidates)."""
    load = request.history_load_kwh
    keep = request.history_usable
    if not np.isfinite(load[keep]).all():
        bad = int(np.flatnonzero(keep & ~np.isfinite(load))[0])
        raise ValueError(f"load_kwh contains a non-finite value at index {bad}")


def _check_window(window_days: int) -> int:
    if int(window_days) < 1:
        raise ValueError(f"window_days must be >= 1, got {window_days}")
    return int(window_days)


@register_forecaster("trailing_median", fallback_rule=_CANDIDATE_FALLBACK_RULE)
def _trailing_median(request: ForecastRequest, *, window_days: int = 28) -> list[DayForecast]:
    """Same-slot-position median over the previous window_days local days (robust to spikes)."""
    window = _check_window(window_days)
    history = _history_day_count(request)
    out: list[DayForecast] = []
    for j in request.target_days:
        label, start, stop = request.days[j]
        rows = _history_rows(
            request, _window_days_before(request, j, window, history), stop - start
        )
        profile, counts = _median_of_columns(rows)
        out.append(
            _complete_day(label, stop - start, profile, counts, rows, None, requested=window)
        )
    return out


@register_forecaster("trailing_trimmed_mean", fallback_rule=_CANDIDATE_FALLBACK_RULE)
def _trailing_trimmed_mean(
    request: ForecastRequest, *, window_days: int = 28, trim: float = 0.1
) -> list[DayForecast]:
    """Same-slot-position trimmed mean (trim fraction dropped at each end) over window_days."""
    window = _check_window(window_days)
    if not 0.0 <= float(trim) < 0.5:
        raise ValueError(f"trim must be in [0, 0.5), got {trim}")
    history = _history_day_count(request)
    out: list[DayForecast] = []
    for j in request.target_days:
        label, start, stop = request.days[j]
        rows = _history_rows(
            request, _window_days_before(request, j, window, history), stop - start
        )
        profile, counts = _trimmed_mean_of_columns(rows, float(trim))
        out.append(
            _complete_day(
                label,
                stop - start,
                profile,
                counts,
                rows,
                None,
                requested=window,
                extra={"trim": float(trim)},
            )
        )
    return out


@register_forecaster("ewma", fallback_rule=_CANDIDATE_FALLBACK_RULE)
def _ewma(
    request: ForecastRequest, *, half_life_days: float = 7.0, window_days: int = 56
) -> list[DayForecast]:
    """Exponentially weighted mean over days per slot position (half-life, capped window)."""
    window = _check_window(window_days)
    if not float(half_life_days) > 0.0:
        raise ValueError(f"half_life_days must be > 0, got {half_life_days}")
    history = _history_day_count(request)
    out: list[DayForecast] = []
    for j in request.target_days:
        label, start, stop = request.days[j]
        prior = _window_days_before(request, j, window, history)
        rows = _history_rows(request, prior, stop - start)
        age = np.array([j - p for p in prior], dtype=float)
        weights = 0.5 ** (age / float(half_life_days))
        profile, counts = _weighted_mean(rows, weights)
        out.append(
            _complete_day(
                label,
                stop - start,
                profile,
                counts,
                rows,
                weights,
                requested=window,
                extra={"half_life_days": float(half_life_days)},
            )
        )
    return out


#: The three day-type profiles of ``daytype``.
DAYTYPE_WEEKDAY = "weekday"
DAYTYPE_SATURDAY = "saturday"
DAYTYPE_SUNDAY_HOLIDAY = "sunday_holiday"


@lru_cache(maxsize=8)
def _holiday_calendar(subdivision: str | None) -> Any:
    import holidays  # noqa: PLC0415 - kept lazy: the package is large and only daytype needs it

    try:
        return holidays.country_holidays("ES", subdiv=subdivision)
    except NotImplementedError as exc:
        raise ValueError(f"unknown Spanish holiday subdivision {subdivision!r}") from exc


@lru_cache(maxsize=4096)
def _daytype_of_ordinal(ordinal: int, subdivision: str | None) -> str:
    day = date.fromordinal(ordinal)
    if day.weekday() == 6 or day in _holiday_calendar(subdivision):
        return DAYTYPE_SUNDAY_HOLIDAY
    return DAYTYPE_SATURDAY if day.weekday() == 5 else DAYTYPE_WEEKDAY


def daytype_of(day: date, subdivision: str | None = None) -> str:
    """``weekday`` / ``saturday`` / ``sunday_holiday`` of a local date.

    Spain's national holidays, plus the optional subdivision's, count as Sundays.
    """
    return _daytype_of_ordinal(day.toordinal(), subdivision)


@register_forecaster("daytype", fallback_rule=_CANDIDATE_FALLBACK_RULE)
def _daytype(
    request: ForecastRequest,
    *,
    n_days: int = 6,
    window_days: int = 84,
    subdivision: str | None = None,
) -> list[DayForecast]:
    """Mean profile of the last n_days local days of the target's day type.

    Day types are weekday, Saturday and Sunday-or-holiday (Spanish holidays from the
    ``holidays`` package, national plus the optional subdivision); the search goes back
    at most window_days.  A day with no usable interval does not count towards n_days.
    """
    window = _check_window(window_days)
    if int(n_days) < 1:
        raise ValueError(f"n_days must be >= 1, got {n_days}")
    _holiday_calendar(subdivision)  # refuses an unknown subdivision up front
    history = _history_day_count(request)
    out: list[DayForecast] = []
    for j in request.target_days:
        label, start, stop = request.days[j]
        kind = daytype_of(_label_date(label), subdivision)
        matching: list[int] = []
        for p in reversed(_window_days_before(request, j, window, history)):
            if len(matching) >= int(n_days):
                break
            _p_label, p_start, p_stop = request.days[p]
            if (
                daytype_of(_label_date(_p_label), subdivision) == kind
                and request.history_usable[p_start:p_stop].any()
            ):
                matching.append(p)
        matching.reverse()
        rows = _history_rows(request, matching, stop - start)
        profile, counts = _weighted_mean(rows, None)
        out.append(
            _complete_day(
                label,
                stop - start,
                profile,
                counts,
                rows,
                None,
                requested=int(n_days),
                extra={"daytype": kind, "daytype_days_found": len(matching)},
            )
        )
    return out


def _slots_per_day(request: ForecastRequest) -> int:
    lengths = [hi - lo for _label, lo, hi in request.days]
    return max(set(lengths), key=lengths.count)


@register_forecaster("level_adjusted", intraday_cutoff=True, fallback_rule=FORECAST_FALLBACK_RULE)
def _level_adjusted(
    request: ForecastRequest,
    *,
    window_days: int = 28,
    lo: float = 0.7,
    hi: float = 1.4,
    min_slots: int = 24,
) -> list[DayForecast]:
    """Trailing-mean profile scaled by clip(observed last 24 h / profile of those slots, lo, hi).

    Declares ``intraday_cutoff``: at a 14:00 / 20:30 event the last 24 h include today's
    observed slots.  The same factor scales every target day.  Fewer than ``min_slots``
    usable observed slots, or a non-positive profile sum, leave the factor at 1.0.
    """
    window = _check_window(window_days)
    if not 0.0 < float(lo) <= float(hi):
        raise ValueError(f"need 0 < lo <= hi, got lo={lo}, hi={hi}")
    _require_finite_history(request)
    load = request.history_load_kwh
    keep = request.history_usable
    cutoff = request.cutoff_idx
    first = max(0, cutoff - _slots_per_day(request))
    observed = 0.0
    expected = 0.0
    slots = 0
    history = _history_day_count(request)
    for p in range(
        max(0, bisect.bisect_right(request.days, first, key=lambda d: d[1]) - 1), history
    ):
        _label, p_start, p_stop = request.days[p]
        lo_i, hi_i = max(p_start, first), min(p_stop, cutoff)
        usable = keep[lo_i:hi_i]
        if not usable.any():
            continue
        profile, _record = trailing_slot_mean_for_day(load, keep, request.days, p, window)
        positions = np.flatnonzero(usable)
        values = load[lo_i:hi_i][positions]
        observed += float(values.sum())
        expected += float(profile[lo_i - p_start + positions].sum())
        slots += int(positions.size)
    raw = observed / expected if slots >= int(min_slots) and expected > 0.0 else None
    factor = 1.0 if raw is None else float(np.clip(raw, float(lo), float(hi)))
    extra = {
        "level_factor": factor,
        "level_factor_raw": raw,
        "level_slots_observed": slots,
        "level_clipped": raw is not None and factor != raw,
    }
    out: list[DayForecast] = []
    for j in request.target_days:
        values, record = trailing_slot_mean_for_day(load, keep, request.days, j, window)
        out.append((values * factor, {**record, **extra}))
    return out


def _one_year_earlier(day: date) -> date:
    try:
        return day.replace(year=day.year - 1)
    except ValueError:  # 29 February
        return day.replace(year=day.year - 1, day=28)


@register_forecaster("blend", fallback_rule=_CANDIDATE_FALLBACK_RULE)
def _blend(
    request: ForecastRequest,
    *,
    w: float = 0.5,
    window_days: int = 28,
    last_year_half_width_days: int = 14,
    min_last_year_days: int = 7,
) -> list[DayForecast]:
    """w * trailing mean + (1 - w) * same-season-last-year profile.

    The last-year profile is the mean of the local days within last_year_half_width_days
    of the same date one year earlier.  Trailing alone is used where that window holds
    fewer than min_last_year_days days with usable data.
    """
    window = _check_window(window_days)
    if not 0.0 <= float(w) <= 1.0:
        raise ValueError(f"w must be in [0, 1], got {w}")
    half = int(last_year_half_width_days)
    if half < 0:
        raise ValueError(f"last_year_half_width_days must be >= 0, got {half}")
    _require_finite_history(request)
    history = _history_day_count(request)
    ordinals = np.fromiter(
        (_label_date(label).toordinal() for label, _lo, _hi in request.days),
        dtype=np.int64,
        count=len(request.days),
    )
    load = request.history_load_kwh
    keep = request.history_usable
    out: list[DayForecast] = []
    for j in request.target_days:
        label, start, stop = request.days[j]
        recent, record = trailing_slot_mean_for_day(load, keep, request.days, j, window)
        centre = _one_year_earlier(_label_date(label)).toordinal()
        a = int(np.searchsorted(ordinals, centre - half, side="left"))
        b = min(int(np.searchsorted(ordinals, centre + half, side="right")), history)
        rows = _history_rows(request, range(a, max(a, b)), stop - start)
        ly_profile, ly_counts = _weighted_mean(rows, None)
        ly_days = int(np.isfinite(rows).any(axis=1).sum()) if rows.size else 0
        usable_ly = ly_days >= max(1, int(min_last_year_days))
        no_recent = int(record["window_days_with_usable_data"]) == 0
        if not usable_ly:
            values = recent
            source = "trailing_only"
        elif no_recent:
            values = np.where(ly_counts > 0, ly_profile, recent)
            source = "last_year_only"
        else:
            values = np.where(ly_counts > 0, w * recent + (1.0 - w) * ly_profile, recent)
            source = "blend"
        out.append(
            (
                values,
                {
                    **record,
                    "blend_weight_recent": float(w),
                    "blend_source": source,
                    "last_year_days_with_usable_data": ly_days,
                    "last_year_slots_blended": int((ly_counts > 0).sum()) if usable_ly else 0,
                },
            )
        )
    return out


# Registers the structured forecaster (LF-1).  It lives in its own module and needs the names
# above, so it is imported last.
from . import structured_forecast  # noqa: E402,F401
