"""Price-outlook seam: what a planning event may know about the prices it plans over.

The rolling-horizon rung plans each event over ``[decision, plan end)``.  Part of that
window is **published** (PVPC is released for the next local day in the evening), and
the rest is not.  A *price outlook* decides what the planner uses for the unpublished
part.  Every outlook goes through :func:`run_price_outlook`, which mirrors the
load-forecaster seam (:mod:`battery_optimizer.load_forecast`):

* The caller states **what was published** (``published_end_idx``, the first index whose
  price is not yet public at the decision instant, see :func:`published_end_idx`) and
  the plan window.
* A non-oracle outlook receives the published prices ``prices[:published_end_idx]`` as a
  read-only view and nothing past it, so a realized price that was not public at the
  decision cannot reach it.  ``tests/test_price_outlook.py`` pins that for every
  registered non-oracle outlook, so a new one is covered without writing a new test.
* The outlook answers only for the unpublished slots.  The seam copies the published
  slots itself, so **a published value is never overwritten**.
* Every slot of the plan window gets a provenance: ``published``, ``estimated`` or
  ``unavailable``.  The rolling-horizon rung plans only up to the first ``unavailable``
  slot and values the energy left at that point with the run's terminal value, which is
  what the live service does when it reaches the last published price.

Registered outlooks:

* ``exact`` (default, **oracle**): the realized prices, optionally degraded by the
  synthetic sigma / bias noise channels of :mod:`battery_optimizer.rolling_horizon`.
  It is the canonical benchmark convention and an upper bound, never a deployable
  outlook.
* ``published_only``: no estimate.  Every unpublished slot is ``unavailable``, so the
  14:00 event plans through the end of its own local day.  This is the live service.
* ``persistence``: tomorrow's price at a local wall-clock time is the last published
  day's price at the same local wall-clock time.

Adding an outlook is one decorated function::

    @register_price_outlook("my_outlook")
    def _my_outlook(request: OutlookRequest, *, alpha: float = 0.5) -> OutlookSlots: ...

Keyword-only parameters with their defaults become the outlook's recorded ``params``.

**Extension point for PO-C (OMIE estimators).**  An OMIE-based estimator needs a side
series the dataset does not hold (OMIE day-ahead marginal prices) and an as-of rule of
its own: OMIE D+1 values may be used from :data:`OMIE_NEXT_DAY_USABLE_FROM` local.  The
intended shape is: add an optional, as-of-filtered side input to :class:`OutlookRequest`
(the seam truncates it by the decision instant exactly as it truncates the published
prices), and register the estimator with ``@register_price_outlook("omie_period",
inputs_usable_from=OMIE_NEXT_DAY_USABLE_FROM)``.  The seam does not run an outlook
whose declared inputs are not usable yet at the decision instant, so a 14:00 event can
use OMIE and a 13:00 one falls back to ``published_only``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

__all__ = [
    "DEFAULT_PRICE_OUTLOOK",
    "OMIE_NEXT_DAY_USABLE_FROM",
    "PVPC_NEXT_DAY_PUBLISHED_FROM",
    "SLOT_ESTIMATED",
    "SLOT_PUBLISHED",
    "SLOT_UNAVAILABLE",
    "OutlookRequest",
    "OutlookResult",
    "OutlookSlots",
    "PriceOutlook",
    "get_price_outlook",
    "last_published_day",
    "price_outlook_names",
    "published_end_idx",
    "register_price_outlook",
    "resolve_price_outlook_params",
    "run_price_outlook",
]

#: The canonical benchmark convention: tomorrow's realized prices at 14:00 (upper bound).
DEFAULT_PRICE_OUTLOOK = "exact"

SLOT_PUBLISHED = "published"
SLOT_ESTIMATED = "estimated"
SLOT_UNAVAILABLE = "unavailable"

#: Local wall-clock time on day D from which PVPC prices for D+1 count as published.
#: REE releases the next-day PVPC at about 20:15 local, and the Home Assistant
#: integration polls for it from 20:00 (``service/pvpc.py``).  So the 20:30 event sees
#: D+1 and the 14:00 event does not.  This is an assumption until the PO-B snapshots
#: measure the real publication times.  ``rolling_horizon._last_published_day`` keeps its
#: own 20:30 boundary for the synthetic noise model; at the 14:00 and 20:30 events the
#: two rules give the same answer.
PVPC_NEXT_DAY_PUBLISHED_FROM = time(20, 15)

#: Local wall-clock time on day D from which OMIE's day-ahead prices for D+1 may be used.
#: OMIE publishes at about 13:00 local; 13:30 is a conservative margin until PO-B and the
#: PO-E shadow measure the real fetch time.  Unused until PO-C registers an OMIE estimator.
OMIE_NEXT_DAY_USABLE_FROM = time(13, 30)

#: What an outlook returns for the unpublished slots it is asked about:
#: ``(values, estimated, diagnostics)``.  ``values`` and ``estimated`` have one entry per
#: slot; a slot with ``estimated[k] == False`` is ``unavailable`` and its value is ignored.
OutlookSlots = tuple[np.ndarray, np.ndarray, dict[str, Any]]


# ---------------------------------------------------------------------------
# Publication rule
# ---------------------------------------------------------------------------


def _as_utc(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _label_date(label: Any) -> date:
    return label if isinstance(label, date) else date.fromisoformat(str(label)[:10])


def last_published_day(
    decision_utc: datetime, tz: ZoneInfo, *, published_from: time = PVPC_NEXT_DAY_PUBLISHED_FROM
) -> date:
    """Last local day whose PVPC prices are public at *decision_utc*.

    Before *published_from* local on day D it is D; from then on it is D+1.
    """
    local = _as_utc(decision_utc).astimezone(tz)
    if local.time().replace(tzinfo=None) >= published_from:
        return local.date() + timedelta(days=1)
    return local.date()


def published_end_idx(
    decision_utc: datetime,
    days: Sequence[tuple[Any, int, int]],
    tz: ZoneInfo,
    *,
    published_from: time = PVPC_NEXT_DAY_PUBLISHED_FROM,
) -> int:
    """First index whose price is not yet public at *decision_utc*.

    *days* is the ``(label, start, stop)`` local-day partition.  Returns the start of
    the first day after :func:`last_published_day`, or ``days[-1][2]`` when every day
    of the partition is already published.
    """
    if not days:
        raise ValueError("days is empty: there is no local day to publish")
    last = last_published_day(decision_utc, tz, published_from=published_from)
    for label, start, _stop in days:
        if _label_date(label) > last:
            return int(start)
    return int(days[-1][2])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutlookRequest:
    """What an outlook may see for one decision."""

    published_prices: np.ndarray
    """Prices at indices ``[0, published_end_idx)``, read-only."""

    published_end_idx: int
    """First index not public at the decision; equals ``len(published_prices)``."""

    decision_idx: int
    """First interval of the plan (the decision instant)."""

    target_idx: np.ndarray
    """The unpublished slots of the plan, ``[max(decision, published_end), plan_end)``."""

    interval_starts_utc: Sequence[datetime]
    """UTC start of every interval (calendar knowledge, not measured data)."""

    days: Sequence[tuple[Any, int, int]]
    """Local-day partition of the whole index space (calendar knowledge)."""

    tz: ZoneInfo
    """The tariff's local time zone."""

    oracle_prices: np.ndarray | None = None
    """The whole realized (or synthetic-noise) series.  ``None`` unless the outlook is
    registered ``oracle=True``."""

    @property
    def decision_utc(self) -> datetime:
        return _as_utc(self.interval_starts_utc[self.decision_idx])


@dataclass(frozen=True)
class PriceOutlook:
    """A registered outlook: its function, default params and flags."""

    name: str
    fn: Callable[..., OutlookSlots]
    defaults: Mapping[str, Any]
    oracle: bool
    description: str
    inputs_usable_from: time | None = None

    def provenance(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The record every artifact carries for the outlook it ran."""
        record: dict[str, Any] = {
            "name": self.name,
            "params": dict(params),
            "oracle": self.oracle,
            "description": self.description,
            "publication_rule": (
                f"PVPC for local day D+1 counts as published from "
                f"{PVPC_NEXT_DAY_PUBLISHED_FROM:%H:%M} local on day D; a published price is "
                "never replaced by an estimate"
            ),
        }
        if self.inputs_usable_from is not None:
            record["inputs_usable_from_local"] = f"{self.inputs_usable_from:%H:%M}"
        return record


@dataclass(frozen=True)
class OutlookResult:
    """Output of :func:`run_price_outlook` for one plan window ``[decision, plan_end)``."""

    values: np.ndarray
    """One price per plan slot; ``nan`` on an ``unavailable`` slot.  Read-only."""

    provenance: np.ndarray
    """One of ``published`` / ``estimated`` / ``unavailable`` per plan slot."""

    usable_end_idx: int
    """First ``unavailable`` index, or the plan end: the window the planner may use."""

    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def usable_values(self) -> np.ndarray:
        """The prices of ``[decision, usable_end_idx)``, all finite."""
        bad = np.flatnonzero(self.provenance == SLOT_UNAVAILABLE)
        return self.values[: int(bad[0]) if bad.size else self.values.size]

    def counts(self) -> dict[str, int]:
        return {
            kind: int(np.count_nonzero(self.provenance == kind))
            for kind in (SLOT_PUBLISHED, SLOT_ESTIMATED, SLOT_UNAVAILABLE)
        }


_REGISTRY: dict[str, PriceOutlook] = {}


def register_price_outlook(
    name: str,
    *,
    oracle: bool = False,
    description: str = "",
    inputs_usable_from: time | None = None,
) -> Callable[[Callable[..., OutlookSlots]], Callable[..., OutlookSlots]]:
    """Register ``fn(request, **params) -> OutlookSlots`` under *name*.

    Args:
        oracle: the outlook reads realized prices that were not public at the decision.
            Only offline evaluation may run it (``allow_oracle=True``).
        inputs_usable_from: local wall-clock time from which the outlook's own inputs
            exist on the decision day (OMIE: :data:`OMIE_NEXT_DAY_USABLE_FROM`).  Before it
            the seam does not call the outlook and marks every unpublished slot
            ``unavailable`` (the ``published_only`` fallback), recording why.
    """

    def decorate(fn: Callable[..., OutlookSlots]) -> Callable[..., OutlookSlots]:
        if name in _REGISTRY:
            raise ValueError(f"price outlook {name!r} is already registered")
        defaults: dict[str, Any] = {}
        for param in inspect.signature(fn).parameters.values():
            if param.kind is not inspect.Parameter.KEYWORD_ONLY:
                continue
            if param.default is inspect.Parameter.empty:
                raise ValueError(f"price outlook {name!r}: param {param.name!r} needs a default")
            defaults[param.name] = param.default
        _REGISTRY[name] = PriceOutlook(
            name=name,
            fn=fn,
            defaults=defaults,
            oracle=oracle,
            description=description or (inspect.getdoc(fn) or "").split("\n", 1)[0],
            inputs_usable_from=inputs_usable_from,
        )
        return fn

    return decorate


def price_outlook_names(*, include_oracle: bool = True) -> tuple[str, ...]:
    """Registered names, in registration order."""
    return tuple(n for n, o in _REGISTRY.items() if include_oracle or not o.oracle)


def get_price_outlook(name: str) -> PriceOutlook:
    """The registered outlook called *name*, or ``ValueError`` naming the known ones."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown price outlook {name!r}; registered: {', '.join(price_outlook_names())}"
        ) from None


def resolve_price_outlook_params(
    name: str, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Defaults of *name* with *overrides* applied; unknown keys are refused."""
    outlook = get_price_outlook(name)
    params = dict(outlook.defaults)
    for key, value in (overrides or {}).items():
        if key not in params:
            raise ValueError(
                f"price outlook {name!r} has no param {key!r}; params: {sorted(params)}"
            )
        params[key] = value
    return params


def _read_only_view(values: np.ndarray, stop: int | None) -> np.ndarray:
    view = values[:stop].view()
    view.setflags(write=False)
    return view


def run_price_outlook(
    name: str,
    *,
    prices: np.ndarray | Sequence[float],
    published_end_idx: int,
    decision_idx: int,
    plan_end_idx: int,
    interval_starts_utc: Sequence[datetime],
    days: Sequence[tuple[Any, int, int]],
    tz: ZoneInfo,
    oracle_prices: np.ndarray | None = None,
    params: Mapping[str, Any] | None = None,
    allow_oracle: bool = False,
) -> OutlookResult:
    """The prices one plan ``[decision_idx, plan_end_idx)`` may use, slot by slot.

    Args:
        prices: realized prices; only ``[0, published_end_idx)`` reaches a non-oracle
            outlook.  May be shorter than the partition, never shorter than
            ``min(published_end_idx, plan_end_idx)``.
        published_end_idx: first index not public at the decision
            (:func:`published_end_idx`).
        oracle_prices: what an ``oracle=True`` outlook reads past the publication
            boundary; defaults to *prices*.
        allow_oracle: must be ``True`` to run an oracle outlook (offline only).

    Raises:
        ValueError: unknown outlook, an oracle without ``allow_oracle``, malformed inputs
            or a malformed answer.
    """
    outlook = get_price_outlook(name)
    if outlook.oracle and not allow_oracle:
        raise ValueError(
            f"price outlook {name!r} is an oracle (it reads prices that were not published at "
            "the decision) and is only allowed in offline evaluation"
        )
    resolved = resolve_price_outlook_params(name, params)
    realized = np.asarray(prices, dtype=float).reshape(-1)
    start, stop = int(decision_idx), int(plan_end_idx)
    if not 0 <= start < stop:
        raise ValueError(f"plan window [{start}, {stop}) is empty or negative")
    published = int(published_end_idx)
    if published < 0:
        raise ValueError(f"published_end_idx must be >= 0, got {published}")
    known = min(published, stop)
    if realized.size < known:
        raise ValueError(f"prices has {realized.size} entries, the plan needs {known} published")
    decision_utc = _as_utc(interval_starts_utc[start])
    inputs_ready = True
    if outlook.inputs_usable_from is not None:
        local = decision_utc.astimezone(tz).time().replace(tzinfo=None)
        inputs_ready = local >= outlook.inputs_usable_from

    m = stop - start
    values = np.full(m, np.nan, dtype=float)
    provenance = np.full(m, SLOT_UNAVAILABLE, dtype=object)
    n_pub = max(0, known - start)
    values[:n_pub] = realized[start : start + n_pub]
    provenance[:n_pub] = SLOT_PUBLISHED
    targets = np.arange(start + n_pub, stop)
    diagnostics: dict[str, Any] = {}
    if targets.size and not inputs_ready:
        # Before the outlook's inputs exist the decision falls back to published_only.
        diagnostics = {"fallback": "inputs not usable yet: unpublished slots unavailable"}
    elif targets.size:
        oracle_series = None
        if outlook.oracle:
            source = realized if oracle_prices is None else oracle_prices
            oracle_series = _read_only_view(np.asarray(source, dtype=float).reshape(-1), None)
        request = OutlookRequest(
            published_prices=_read_only_view(realized, known),
            published_end_idx=known,
            decision_idx=start,
            target_idx=targets,
            interval_starts_utc=interval_starts_utc,
            days=days,
            tz=tz,
            oracle_prices=oracle_series,
        )
        est_values, estimated, diagnostics = outlook.fn(request, **resolved)
        est_values = np.asarray(est_values, dtype=float).reshape(-1)
        estimated = np.asarray(estimated, dtype=bool).reshape(-1)
        if est_values.size != targets.size or estimated.size != targets.size:
            raise ValueError(
                f"price outlook {name!r} answered {est_values.size} values / {estimated.size} "
                f"flags for {targets.size} unpublished slots"
            )
        if not np.isfinite(est_values[estimated]).all():
            raise ValueError(f"price outlook {name!r} returned a non-finite estimate")
        values[n_pub:] = np.where(estimated, est_values, np.nan)
        provenance[n_pub:] = np.where(estimated, SLOT_ESTIMATED, SLOT_UNAVAILABLE)
    values.setflags(write=False)
    bad = np.flatnonzero(provenance == SLOT_UNAVAILABLE)
    usable_end = start + (int(bad[0]) if bad.size else m)
    return OutlookResult(
        values=values,
        provenance=provenance,
        usable_end_idx=usable_end,
        diagnostics={**outlook.provenance(resolved), **dict(diagnostics)},
    )


# ---------------------------------------------------------------------------
# Registered outlooks
# ---------------------------------------------------------------------------


@register_price_outlook(
    "exact",
    oracle=True,
    description=(
        "UPPER BOUND, offline only: the realized prices of the unpublished slots (degraded by "
        "the synthetic sigma / bias noise channels when those are set)"
    ),
)
def _exact(request: OutlookRequest) -> OutlookSlots:
    """The realized (or synthetic-noise) price of every unpublished slot."""
    if request.oracle_prices is None:
        raise ValueError("the exact outlook needs the realized series")
    values = np.asarray(request.oracle_prices)[request.target_idx]
    return values, np.ones(values.size, dtype=bool), {}


@register_price_outlook("published_only")
def _published_only(request: OutlookRequest) -> OutlookSlots:
    """No estimate: plan up to the last published price, then the terminal value."""
    n = request.target_idx.size
    return np.full(n, np.nan), np.zeros(n, dtype=bool), {}


@register_price_outlook("persistence")
def _persistence(request: OutlookRequest) -> OutlookSlots:
    """Tomorrow at a local wall-clock time = the last published day at that time."""
    tz = request.tz
    starts = request.interval_starts_utc
    known = request.published_end_idx
    # Published local days, most recent first.  A day is a source only if all of it is
    # published (``hi <= known``).
    sources = [(lbl, lo, hi) for lbl, lo, hi in request.days if hi <= known and hi > lo]
    sources.reverse()

    def clock_map(lo: int, hi: int) -> dict[tuple[int, int], int]:
        # Keyed by local wall-clock (hour, minute).  On a fall-back day the repeated hour
        # keeps its first occurrence; a spring-forward day simply lacks 02:xx.
        out: dict[tuple[int, int], int] = {}
        for i in range(lo, hi):
            local = _as_utc(starts[i]).astimezone(tz)
            out.setdefault((local.hour, local.minute), i)
        return out

    maps = [(str(lbl), clock_map(lo, hi)) for lbl, lo, hi in sources[:2]]
    values = np.full(request.target_idx.size, np.nan)
    estimated = np.zeros(request.target_idx.size, dtype=bool)
    from_earlier = 0
    for k, t in enumerate(request.target_idx):
        local = _as_utc(starts[int(t)]).astimezone(tz)
        key = (local.hour, local.minute)
        for depth, (_label, mapping) in enumerate(maps):
            idx = mapping.get(key)
            if idx is not None:
                values[k] = request.published_prices[idx]
                estimated[k] = True
                from_earlier += depth > 0
                break
    return (
        values,
        estimated,
        {
            "source_day": maps[0][0] if maps else None,
            "slots_from_earlier_day": int(from_earlier),
        },
    )
