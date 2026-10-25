"""Load-forecaster evaluation harness (P1 of the load-forecaster improvement plan).

One call compares a list of registered forecasters (with params) against LF-0
(``trailing_mean``) on a built dataset and writes a private, pre-registered run under
``<dataset>/forecast-eval/<name>-<UTC timestamp>/``:

1. ``preregistration.json`` — candidates, periods, policies, metrics, bootstrap seed and
   the G2 gate, written **before** any candidate runs.
2. **Forecast accuracy** at the policies' real decision instants: 00:00 local for
   ``day_ahead`` (one forecast per local day) and the 14:00 / 20:30 events for
   ``rolling_horizon`` (one forecast per event, over that event's plan horizon), on
   usable slots only.  Forecasts come from the same seam calls the benchmark plans on
   (:func:`~battery_optimizer.load_forecast.forecast_each_day_at_its_start` and
   :class:`~battery_optimizer.rolling_horizon.CausalForecastProvider`), so the error is
   the error the planner saw.  MAE, WAPE, bias and P95 absolute error, overall and by
   horizon bucket from the decision instant, price tercile, day type, season and
   month, plus the price-weighted absolute error ``sum(price * |error|)``.
3. **Economic evaluation**: one ``benchmark`` run per candidate and period
   (``no_battery``, ``day_ahead``, ``rolling_horizon``; exact prices), the paired
   per-local-day cost difference against LF-0 with a weekly (7-local-day) block
   bootstrap interval, and the share of the P0 gap (LF-0 minus ``perfect``) captured.
   With three or more candidates, the Spearman correlation between price-weighted
   error and EUR says whether the cheap proxy ranks forecasters like the money does.
4. ``summary.json`` / ``summary.md`` with provenance: git commit and dirty flag, tariff
   config hash, evaluation-config hash, dataset id, forecaster names and params, and
   every benchmark run id.

Walk-forward folds are a property of the seam, not of this module: a forecaster
registered with ``fit=`` is refit once per local month on history strictly before the
month (:data:`~battery_optimizer.load_forecast.FIT_FOLD_RULE`), for the benchmark runs
and the accuracy metrics alike.  The harness records which folds each period's
decisions fall in.

Everything this module writes holds monetary figures or forecasts of the private series,
so it only ever writes inside the dataset directory, which lives in the private data dir.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq

from battery_optimizer import benchmark as bm
from battery_optimizer.config import load_config
from battery_optimizer.load_forecast import (
    DEFAULT_LOAD_FORECASTER,
    FIT_FOLD_RULE,
    SIDE_INPUT_RULE,
    SideInputs,
    build_side_inputs,
    forecast_each_day_at_its_start,
    get_forecaster,
    resolve_forecaster_params,
)
from battery_optimizer.manifest import code_version, config_hash
from battery_optimizer.policies import contiguous_day_bounds
from battery_optimizer.rolling_horizon import CausalForecastProvider, compute_rolling_events

__all__ = [
    "BENCHMARK_PARAMS",
    "DEFAULT_PERIODS",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "EVAL_DIRNAME",
    "Candidate",
    "Period",
    "block_bootstrap",
    "parse_candidate",
    "run_forecast_eval",
]

EVAL_DIRNAME = "forecast-eval"
PREREGISTRATION_NAME = "preregistration.json"
SUMMARY_JSON_NAME = "summary.json"
SUMMARY_MD_NAME = "summary.md"
SCHEMA = "battery-optimizer/forecast-eval/v1"

REFERENCE = DEFAULT_LOAD_FORECASTER
ORACLE_REFERENCE = "perfect"
PRIMARY_POLICY = bm.POLICY_ROLLING_HORIZON
SECONDARY_POLICY = bm.POLICY_DAY_AHEAD
EVAL_POLICIES = (bm.POLICY_DAY_AHEAD, bm.POLICY_ROLLING_HORIZON)
BENCHMARK_POLICIES = (bm.POLICY_NO_BATTERY, bm.POLICY_DAY_AHEAD, bm.POLICY_ROLLING_HORIZON)

#: Canonical benchmark params of the P0 runs (exact prices, legacy 14:00 / 20:30 cadence).
#: Everything not listed is the benchmark module's default.
BENCHMARK_PARAMS: dict[str, Any] = {
    "policies": list(BENCHMARK_POLICIES),
    "min_usable_fraction": 0.8,
    "quality_policy": bm.QUALITY_POLICY_REPORT,
    "time_limit_s": 900.0,
    "terminal_mode": "free",
    "rolling_horizon_load_forecast_mode": "causal",
    "rolling_horizon_sigma": 0.0,
    "rolling_horizon_bias_sigma": 0.0,
}

DEFAULT_SEED = 20261025
DEFAULT_RESAMPLES = 10_000
MIN_RESAMPLES = 5_000
BLOCK_DAYS = 7
CONFIDENCE = 0.95
GATE_MIN_SHARE_OF_P0_GAP = 0.10

#: Gate G4 (structured forecaster): season definitions, pre-registered before any run.  A
#: season instance is one period's days in these local months; April, May and October are
#: transitional and reported only in the whole-period figures.
G4_SEASONS: dict[str, tuple[int, ...]] = {
    "heating": (11, 12, 1, 2, 3),
    "cooling": (6, 7, 8, 9),
}
G4_MIN_SEASON_DAYS = 14

#: ``(label, lower_h, upper_h)`` — hours from the decision instant, half-open.
HORIZON_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0-6h", 0.0, 6.0),
    ("6-24h", 6.0, 24.0),
    ("24-36h", 24.0, 36.0),
)
SEASON_OF_MONTH = {
    12: "DJF",
    1: "DJF",
    2: "DJF",
    3: "MAM",
    4: "MAM",
    5: "MAM",
    6: "JJA",
    7: "JJA",
    8: "JJA",
    9: "SON",
    10: "SON",
    11: "SON",
}
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


@dataclass(frozen=True)
class Period:
    """One evaluation window: ``[train_end, eval_end)`` in UTC, plus the train window."""

    label: str
    train_end: datetime
    eval_end: datetime
    train_days: int

    @property
    def window(self) -> str:
        return f"{_iso(self.train_end)} .. {_iso(self.eval_end)}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "train_end": _iso(self.train_end),
            "eval_end": _iso(self.eval_end),
            "train_days": self.train_days,
        }


#: The P0 periods (UTC): calendar 2025, and 2026-01..09.  ``train_days`` covers the
#: previous calendar year (366 for leap 2024), which fixes the terminal SOC value.
DEFAULT_PERIODS: tuple[Period, ...] = (
    Period("2025", datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC), 366),
    Period("2026-01..09", datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 10, 1, tzinfo=UTC), 365),
)


@dataclass(frozen=True)
class Candidate:
    """A registered forecaster with resolved params and a stable label."""

    label: str
    name: str
    params: dict[str, Any]
    overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def oracle(self) -> bool:
        return get_forecaster(self.name).oracle

    def benchmark_kwargs(self) -> dict[str, Any]:
        """``window_days`` goes through the benchmark's window knob, the rest as overrides."""
        rest = {k: v for k, v in self.overrides.items() if k != "window_days"}
        return {
            "load_forecaster": self.name,
            "load_forecaster_params": rest or None,
            "day_ahead_forecast_window_days": int(
                self.params.get("window_days", bm.DEFAULT_FORECAST_WINDOW_DAYS)
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        forecaster = get_forecaster(self.name)
        return {
            "label": self.label,
            "name": self.name,
            "params": dict(self.params),
            "oracle": forecaster.oracle,
            "fit": FIT_FOLD_RULE if forecaster.fit is not None else None,
            "intraday_cutoff": forecaster.intraday_cutoff,
            "side_inputs": SIDE_INPUT_RULE if forecaster.side_loader is not None else None,
        }


def _parse_value(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_candidate(spec: str | Candidate) -> Candidate:
    """``"name"`` or ``"name:key=value,key=value"`` (values parsed as JSON when possible)."""
    if isinstance(spec, Candidate):
        return spec
    name, _, rest = str(spec).strip().partition(":")
    name, _, alias = name.strip().partition("@")
    name = name.strip()
    overrides: dict[str, Any] = {}
    for item in filter(None, (part.strip() for part in rest.split(","))):
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"candidate {spec!r}: expected key=value, got {item!r}")
        overrides[key.strip()] = _parse_value(value.strip())
    params = resolve_forecaster_params(name, overrides=overrides)
    defaults = dict(get_forecaster(name).defaults)
    changed = {k: v for k, v in params.items() if defaults.get(k) != v}
    label = alias.strip() or name
    if changed and not alias.strip():
        label += "[" + ",".join(f"{k}={_short(changed[k])}" for k in sorted(changed)) + "]"
    return Candidate(label=label, name=name, params=params, overrides=changed)


def _short(value: Any) -> str:
    """A label-sized value: long strings (feature set ids) keep their first 12 characters."""
    text = str(value)
    return text if len(text) <= 16 else text[:12]


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _canonical_sha(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: Any) -> bytes:
    data = (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(data)
    path.chmod(0o600)
    return data


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def block_bootstrap(
    daily: np.ndarray,
    *,
    resamples: int,
    seed: int | Sequence[int],
    block_days: int = BLOCK_DAYS,
    confidence: float = CONFIDENCE,
) -> dict[str, Any]:
    """Percentile interval of the period total under a non-overlapping block bootstrap.

    *daily* is chronological (one value per local day).  Days are cut into consecutive
    blocks of *block_days* (the last block may be shorter); each resample draws as many
    blocks as there are, with replacement, and sums them.  The same *seed* gives the same
    draws, so comparisons in one period share their resamples.
    """
    values = np.asarray(daily, dtype=float)
    blocks = np.array([values[k : k + block_days].sum() for k in range(0, values.size, block_days)])
    if blocks.size == 0:
        raise ValueError("no days to bootstrap")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, blocks.size, size=(int(resamples), blocks.size))
    totals = blocks[draws].sum(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(totals, [alpha, 1.0 - alpha])
    return {
        "point_eur": float(values.sum()),
        "ci_low_eur": float(low),
        "ci_high_eur": float(high),
        "confidence": confidence,
        "days": int(values.size),
        "blocks": int(blocks.size),
        "block_days": int(block_days),
        "resamples": int(resamples),
    }


def _metrics(err: np.ndarray, measured: np.ndarray, price: np.ndarray) -> dict[str, Any]:
    if err.size == 0:
        return {"slots": 0}
    absolute = np.abs(err)
    total = float(measured.sum())
    return {
        "slots": int(err.size),
        "mae_kwh": float(absolute.mean()),
        "wape": float(absolute.sum() / total) if total > 0 else None,
        "bias_kwh": float(err.mean()),
        "p95_abs_error_kwh": float(np.quantile(absolute, 0.95)),
        "price_weighted_abs_error_eur": float((price * absolute).sum()),
        "measured_kwh": total,
    }


def _spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    from scipy.stats import rankdata, spearmanr

    if len(x) < 3:
        return {"status": "not computed: needs at least 3 candidates", "candidates": len(x)}
    rho = spearmanr(x, y).statistic
    return {
        "status": "computed",
        "candidates": len(x),
        "rho": None if not math.isfinite(float(rho)) else float(rho),
        "same_order": bool(np.array_equal(rankdata(x), rankdata(y))),
    }


# ---------------------------------------------------------------------------
# Period data and forecast accuracy
# ---------------------------------------------------------------------------


@dataclass
class _PeriodData:
    period: Period
    starts: list[datetime]
    local_days: list[Any]
    usable: np.ndarray
    load: np.ndarray
    prices: np.ndarray
    eval_start: int
    eval_end: int
    days: list[tuple[Any, int, int]]
    eval_day_idx: list[int]
    events: list[Any]
    interval_seconds: float
    tz: ZoneInfo
    tercile: np.ndarray
    tercile_edges: tuple[float, float]
    month: np.ndarray
    season: np.ndarray
    daytype: np.ndarray
    table: Any = None


def _as_date(label: Any) -> date:
    return label if isinstance(label, date) else date.fromisoformat(str(label)[:10])


def _period_data(loaded: bm.LoadedDataset, period: Period, tz: ZoneInfo) -> _PeriodData:
    table = loaded.table
    all_starts = table.column("interval_start").to_pylist()
    kept = sum(1 for s in all_starts if s < period.eval_end)
    table = table.slice(0, kept)
    starts = all_starts[:kept]
    local_days = table.column("local_day").to_pylist()
    usable, _reasons = bm.usable_mask(table, allow_reconstructed=False)

    def _col(name: str) -> np.ndarray:
        return np.array([0.0 if v is None else float(v) for v in table.column(name).to_pylist()])

    load = _col("gross_kwh")
    prices = _col("price_eur_per_kwh")
    eval_start = next((i for i, s in enumerate(starts) if s >= period.train_end), kept)
    if eval_start >= kept:
        raise ValueError(f"period {period.label}: no intervals in {period.window}")
    days = contiguous_day_bounds(local_days)
    eval_day_idx = [j for j, (_l, lo, hi) in enumerate(days) if hi > eval_start and lo < kept]
    events = [
        ev
        for ev in compute_rolling_events(
            starts, local_days, eval_start_idx=eval_start, eval_end_idx=kept, tz=tz
        )
        if ev.kind in ("afternoon", "evening")
    ]
    in_eval = np.zeros(kept, dtype=bool)
    in_eval[eval_start:] = True
    q1, q2 = np.quantile(prices[in_eval & usable], [1 / 3, 2 / 3])
    tercile = np.where(prices <= q1, "T1", np.where(prices <= q2, "T2", "T3"))
    dates = [_as_date(d) for d in local_days]
    return _PeriodData(
        period=period,
        starts=starts,
        local_days=local_days,
        usable=usable,
        load=load,
        prices=prices,
        eval_start=eval_start,
        eval_end=kept,
        days=days,
        eval_day_idx=eval_day_idx,
        events=events,
        interval_seconds=loaded.interval_seconds,
        tz=tz,
        tercile=tercile,
        tercile_edges=(float(q1), float(q2)),
        month=np.array([d.strftime("%Y-%m") for d in dates]),
        season=np.array([SEASON_OF_MONTH[d.month] for d in dates]),
        daytype=np.array(["weekend" if d.weekday() >= 5 else "weekday" for d in dates]),
        table=table,
    )


def _instances(
    candidate: Candidate, data: _PeriodData, policy: str, side: SideInputs | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """``(slot index, decision index, forecast)`` of every forecast slot, and the decisions."""
    idx_parts: list[np.ndarray] = []
    dec_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    window = int(candidate.params.get("window_days", bm.DEFAULT_FORECAST_WINDOW_DAYS))
    rest = {k: v for k, v in candidate.overrides.items() if k != "window_days"}
    if policy == bm.POLICY_DAY_AHEAD:
        params = resolve_forecaster_params(candidate.name, window_days=window, overrides=rest)
        result = forecast_each_day_at_its_start(
            candidate.name,
            load_kwh=data.load,
            usable=data.usable,
            days=data.days,
            target_days=data.eval_day_idx,
            params=params,
            allow_oracle=True,
            side=side,
        )
        for j in data.eval_day_idx:
            _label, lo, hi = data.days[j]
            idx = np.arange(max(lo, data.eval_start), min(hi, data.eval_end))
            idx_parts.append(idx)
            dec_parts.append(np.full(idx.size, lo))
            val_parts.append(result.values[idx])
        decisions = len(data.eval_day_idx)
    else:
        provider = CausalForecastProvider(
            data.load,
            data.local_days,
            data.usable,
            data.starts,
            window_days=window,
            interval_seconds=data.interval_seconds,
            tz=data.tz,
            load_forecaster=candidate.name,
            load_forecaster_params=rest or None,
            side=side,
        )
        for ev in data.events:
            forecast, _meta = provider.forecast_for_event(ev)
            idx = np.arange(ev.start_idx, ev.plan_end_idx)
            idx_parts.append(idx)
            dec_parts.append(np.full(idx.size, ev.start_idx))
            val_parts.append(np.asarray(forecast)[idx])
        decisions = len(data.events)
    idx = np.concatenate(idx_parts) if idx_parts else np.zeros(0, dtype=int)
    dec = np.concatenate(dec_parts) if dec_parts else np.zeros(0, dtype=int)
    val = np.concatenate(val_parts) if val_parts else np.zeros(0)
    keep = data.usable[idx]
    return idx[keep], dec[keep], val[keep], decisions


def _accuracy(
    candidate: Candidate, data: _PeriodData, policy: str, side: SideInputs | None = None
) -> dict[str, Any]:
    idx, dec, forecast, decisions = _instances(candidate, data, policy, side)
    measured = data.load[idx]
    price = data.prices[idx]
    err = forecast - measured
    hours = (idx - dec) * data.interval_seconds / 3600.0

    def by(keys: np.ndarray, order: Sequence[str] | None = None) -> dict[str, Any]:
        labels = order if order is not None else sorted(set(keys.tolist()))
        return {
            str(k): _metrics(err[keys == k], measured[keys == k], price[keys == k])
            for k in labels
            if (keys == k).any()
        }

    bucket = np.full(idx.size, "36h+", dtype=object)
    for label, lo, hi in HORIZON_BUCKETS:
        bucket[(hours >= lo) & (hours < hi)] = label
    return {
        "decision_instants": (
            "00:00 local of each local day"
            if policy == bm.POLICY_DAY_AHEAD
            else "14:00 and 20:30 local events (warm-up events excluded)"
        ),
        "decisions": decisions,
        "overall": _metrics(err, measured, price),
        "by_horizon": by(bucket.astype(str), [b[0] for b in HORIZON_BUCKETS] + ["36h+"]),
        "by_price_tercile": by(data.tercile[idx], ["T1", "T2", "T3"]),
        "by_day_type": by(data.daytype[idx], ["weekday", "weekend"]),
        "by_season": by(data.season[idx], ["DJF", "MAM", "JJA", "SON"]),
        "by_month": by(data.month[idx]),
    }


def _diagnostics(
    candidate: Candidate, data: _PeriodData, side: SideInputs | None
) -> dict[str, Any] | None:
    """The forecaster's own component report over the decisions of both policies, or ``None``."""
    forecaster = get_forecaster(candidate.name)
    if forecaster.diagnose is None or side is None:
        return None
    params = dict(candidate.params)
    out: dict[str, Any] = {}
    day_decisions = [
        {
            "kind": "00:00",
            "cutoff_idx": data.days[j][1],
            "target_days": [j],
            "start_idx": max(data.days[j][1], data.eval_start),
            "stop_idx": min(data.days[j][2], data.eval_end),
        }
        for j in data.eval_day_idx
    ]
    out[bm.POLICY_DAY_AHEAD] = forecaster.diagnose(
        side=side,
        load=data.load,
        usable=data.usable,
        days=data.days,
        decisions=day_decisions,
        params=params,
    )
    rolling = []
    for ev in data.events:
        touched = [
            j
            for j, (_l, lo, hi) in enumerate(data.days)
            if hi > ev.start_idx and lo < ev.plan_end_idx
        ]
        rolling.append(
            {
                "kind": str(ev.kind),
                "cutoff_idx": (
                    ev.start_idx if forecaster.intraday_cutoff else data.days[touched[0]][1]
                ),
                "target_days": touched,
                "start_idx": ev.start_idx,
                "stop_idx": ev.plan_end_idx,
            }
        )
    out[bm.POLICY_ROLLING_HORIZON] = forecaster.diagnose(
        side=side,
        load=data.load,
        usable=data.usable,
        days=data.days,
        decisions=rolling,
        params=params,
    )
    return out


def _folds(data: _PeriodData) -> list[dict[str, Any]]:
    """Monthly walk-forward folds this period's decisions fall in (decision local month)."""
    out: dict[str, dict[str, Any]] = {}
    for j in data.eval_day_idx:
        label, lo, _hi = data.days[j]
        month = _as_date(label).strftime("%Y-%m")
        out.setdefault(month, {"fold": month, "day_ahead_decisions": 0, "rh_decisions": 0})
        out[month]["day_ahead_decisions"] += 1
    for ev in data.events:
        month = _as_date(ev.local_day).strftime("%Y-%m")
        out.setdefault(month, {"fold": month, "day_ahead_decisions": 0, "rh_decisions": 0})
        out[month]["rh_decisions"] += 1
    for month, record in out.items():
        first = next(
            lo for label, lo, _hi in data.days if _as_date(label).strftime("%Y-%m") == month
        )
        record["fit_before_utc"] = _iso(data.starts[first])
    return [out[k] for k in sorted(out)]


# ---------------------------------------------------------------------------
# Economic evaluation
# ---------------------------------------------------------------------------


def _run_costs(run_dir: Path) -> dict[str, dict[str, Any]]:
    """Per policy: interval costs in schedule order, local days and the usable mask."""
    table = pq.read_table(
        run_dir / bm.SCHEDULES_PARQUET_NAME,
        columns=["policy", "local_day", "usable", "interval_cost_eur"],
    ).to_pydict()
    out: dict[str, dict[str, list[Any]]] = {}
    for i, policy in enumerate(table["policy"]):
        record = out.setdefault(policy, {"local_day": [], "usable": [], "cost": []})
        record["local_day"].append(table["local_day"][i].isoformat())
        record["usable"].append(bool(table["usable"][i]))
        value = table["interval_cost_eur"][i]
        record["cost"].append(0.0 if value is None else float(value))
    return {
        policy: {
            "local_day": np.array(r["local_day"]),
            "usable": np.array(r["usable"], dtype=bool),
            "cost": np.array(r["cost"], dtype=float),
        }
        for policy, r in out.items()
    }


def _daily(record: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    days = record["local_day"]
    starts = np.flatnonzero(np.r_[True, days[1:] != days[:-1]])
    return [str(days[i]) for i in starts], np.add.reduceat(record["cost"], starts)


def _paired(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    resamples: int,
    seed: Sequence[int],
) -> dict[str, Any]:
    """Saving of *candidate* against *reference* (positive = candidate cheaper)."""
    if not (
        np.array_equal(reference["local_day"], candidate["local_day"])
        and np.array_equal(reference["usable"], candidate["usable"])
    ):
        raise ValueError("paired runs do not share the same intervals and usable mask")
    days, ref_daily = _daily(reference)
    _days, cand_daily = _daily(candidate)
    out = block_bootstrap(ref_daily - cand_daily, resamples=resamples, seed=seed)
    out["point_eur"] = float(np.sum(reference["cost"] - candidate["cost"]))
    out["mean_daily_eur"] = out["point_eur"] / len(days)
    out["first_day"], out["last_day"] = days[0], days[-1]
    return out


# ---------------------------------------------------------------------------
# P0 reference
# ---------------------------------------------------------------------------


def _p0_reference(path: Path | None, periods: Sequence[Period]) -> dict[str, Any] | None:
    if path is None:
        return None
    raw = Path(path).expanduser().read_bytes()
    payload = json.loads(raw)
    by_window = {str(p.get("evaluation")): p for p in payload.get("periods") or []}
    out: dict[str, Any] = {
        "source": str(Path(path).expanduser()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "periods": {},
    }
    for period in periods:
        record = by_window.get(period.window)
        if record is None:
            continue
        out["periods"][period.label] = {
            "p0_period": record.get("period"),
            "runs": {k: v.get("run_id") for k, v in (record.get("runs") or {}).items()},
            "policies": {
                policy: {
                    key: (record["policies"].get(policy) or {}).get(key)
                    for key in ("gap_eur", "lf0_cost_eur", "perfect_cost_eur")
                }
                for policy in EVAL_POLICIES
                if policy in (record.get("policies") or {})
            },
        }
    return out


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def _gate_definition(p0: dict[str, Any] | None, periods: Sequence[Period]) -> dict[str, Any]:
    gaps = {
        period.label: ((p0 or {}).get("periods", {}).get(period.label, {}).get("policies", {}))
        .get(PRIMARY_POLICY, {})
        .get("gap_eur")
        for period in periods
    }
    return {
        "id": "G2",
        "primary_policy": PRIMARY_POLICY,
        "secondary_policy": SECONDARY_POLICY,
        "periods": [p.label for p in periods],
        "min_share_of_p0_gap": GATE_MIN_SHARE_OF_P0_GAP,
        "confidence": CONFIDENCE,
        "rule": (
            "a non-oracle candidate PASSES if, for the primary policy in EVERY period, the "
            f"{CONFIDENCE:.0%} weekly block-bootstrap interval of its paired saving against "
            f"{REFERENCE} lies entirely above 0 AND the point saving is at least "
            f"{GATE_MIN_SHARE_OF_P0_GAP:.0%} of that period's P0 gap "
            f"({REFERENCE} minus {ORACLE_REFERENCE}) for the primary policy. The secondary "
            "policy is reported, not gated"
        ),
        "p0_gap_eur": gaps,
        "p0_gap_source": (
            "the P0 summary named under p0_reference"
            if p0 is not None
            else f"none pre-registered: the in-run {ORACLE_REFERENCE} candidate, if any"
        ),
    }


def _evaluate_gate(
    candidate: Candidate,
    comparisons: dict[str, dict[str, Any]],
    gaps: dict[str, float | None],
    periods: Sequence[Period],
) -> dict[str, Any]:
    if candidate.label == REFERENCE:
        return {"eligible": False, "reason": "the reference itself", "passed": None}
    per_period: dict[str, Any] = {}
    passed: bool | None = True
    for period in periods:
        result = comparisons[period.label][PRIMARY_POLICY]["vs_reference"]
        gap = gaps.get(period.label)
        ci_ok = result["ci_low_eur"] > 0.0
        share = None if not gap else result["point_eur"] / gap
        share_ok = None if share is None else share >= GATE_MIN_SHARE_OF_P0_GAP
        per_period[period.label] = {
            "point_eur": result["point_eur"],
            "ci_low_eur": result["ci_low_eur"],
            "ci_high_eur": result["ci_high_eur"],
            "ci_entirely_above_zero": ci_ok,
            "p0_gap_eur": gap,
            "share_of_p0_gap": share,
            "share_at_least_min": share_ok,
        }
        if not ci_ok or share_ok is False:
            passed = False
        elif share_ok is None and passed is not False:
            passed = None
    record: dict[str, Any] = {
        "eligible": not candidate.oracle,
        "passed": passed,
        "periods": per_period,
    }
    if candidate.oracle:
        record["reason"] = "an oracle is an upper bound, never a deployable candidate"
    return record


def _g4_definition(
    candidate: str,
    reference: str,
    comparisons: Sequence[tuple[str, str]],
    periods: Sequence[Period],
) -> dict[str, Any]:
    """Gate G4, pre-registered: the structured forecaster against the P2 reference."""
    return {
        "id": "G4",
        "candidate": candidate,
        "reference": reference,
        "primary_policy": PRIMARY_POLICY,
        "secondary_policy": SECONDARY_POLICY,
        "periods": [p.label for p in periods],
        "seasons": {name: list(months) for name, months in G4_SEASONS.items()},
        "season_definition": (
            "local calendar months of each period's days: heating = Nov, Dec, Jan, Feb, Mar; "
            "cooling = Jun, Jul, Aug, Sep; Apr, May and Oct are transitional (reported in the "
            "whole-period figures only). A season instance is one period's days in a season"
        ),
        "min_season_days": G4_MIN_SEASON_DAYS,
        "confidence": CONFIDENCE,
        "rule": (
            f"{candidate} PASSES if, for the primary policy, the {CONFIDENCE:.0%} weekly "
            f"block-bootstrap interval of its paired saving against {reference} lies entirely "
            "above 0 in at least one heating season instance AND at least one cooling season "
            "instance (blocks of 7 consecutive days of that season instance). Whole-period "
            "figures, the secondary policy and the comparisons below are reported, not gated"
        ),
        "secondary_comparisons": [
            {"candidate": c, "reference": r, "gating": False} for c, r in comparisons
        ],
    }


def _season_pairs(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    resamples: int,
    seed: Sequence[int],
) -> dict[str, Any]:
    """Whole-period and per-season paired saving of *candidate* against *reference*."""
    out: dict[str, Any] = {
        "period": _paired(reference, candidate, resamples=resamples, seed=(*seed, 0))
    }
    days, ref_daily = _daily(reference)
    _days, cand_daily = _daily(candidate)
    diff = ref_daily - cand_daily
    months = np.array([int(d[5:7]) for d in days])
    for k, (name, selected) in enumerate(G4_SEASONS.items(), start=1):
        mask = np.isin(months, selected)
        if int(mask.sum()) < G4_MIN_SEASON_DAYS:
            out[name] = {"days": int(mask.sum()), "status": "too few days", "holds": False}
            continue
        record = block_bootstrap(diff[mask], resamples=resamples, seed=(*seed, k))
        record["holds"] = bool(record["ci_low_eur"] > 0.0)
        record["worse"] = bool(record["ci_high_eur"] < 0.0)
        out[name] = record
    return out


def run_forecast_eval(
    *,
    dataset: Path | str,
    config_path: Path | str,
    name: str,
    candidates: Sequence[str | Candidate],
    periods: Sequence[Period] = DEFAULT_PERIODS,
    p0_summary: Path | str | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    data_dir: Path | str | None = None,
    benchmark_overrides: Mapping[str, Any] | None = None,
    now: datetime | None = None,
    gate: str = "G2",
    gate_candidate: str | None = None,
    gate_reference: str | None = None,
    compare: Sequence[tuple[str, str]] = (),
    label: str | None = None,
    notes: Sequence[str] = (),
) -> Path:
    """Pre-register, run and summarise one forecast evaluation; return its directory.

    Args:
        dataset: built dataset directory (``analysis/v<schema>/<dataset_id>``).
        config_path: the tariff config the dataset was built with.
        name: short run name (``[a-z0-9][a-z0-9_-]*``); the directory is
            ``<dataset>/forecast-eval/<name>-<UTC timestamp>``.
        candidates: registered forecasters as ``name`` or ``name:key=value,...``.
            LF-0 (``trailing_mean`` with its defaults) is the reference and is added
            first when missing.
        periods: evaluation windows (default: the P0 periods).
        p0_summary: the P0 ``summary.json`` whose gaps the gate and the "share of the
            P0 gap" read.  Without it the in-run ``perfect`` candidate, if any, is used.
        resamples / seed: weekly block bootstrap (at least ``MIN_RESAMPLES``).
        benchmark_overrides: extra ``benchmark_dataset`` keywords (tests use it for
            shorter solver limits); recorded in the pre-registration.
        now: the timestamp of the run directory (tests).
        gate: ``"G2"`` (default: the P2 gate against LF-0) or ``"G4"``.  Either way the
            G2 figures are reported; G4 additionally pre-registers and evaluates the
            structured forecaster *gate_candidate* against *gate_reference* (default LF-0)
            per season instance.
        compare: ``(candidate label, reference label)`` pairs reported like G4 but never
            gated (secondary comparisons).
        label / notes: free text recorded in the pre-registration, for the role of the run
            (for example a second look at periods that were already seen) and what a result
            may and may not be used for.  Both absent leaves the pre-registration unchanged.
    """
    if gate not in ("G2", "G4"):
        raise ValueError(f"gate must be 'G2' or 'G4', got {gate!r}")
    if not _NAME_RE.match(name):
        raise ValueError(f"forecast-eval name must match {_NAME_RE.pattern}, got {name!r}")
    if int(resamples) < MIN_RESAMPLES:
        raise ValueError(f"resamples must be >= {MIN_RESAMPLES}, got {resamples}")
    if not periods:
        raise ValueError("no evaluation period")
    if len({p.label for p in periods}) != len(periods):
        raise ValueError("period labels must be unique")
    parsed = [parse_candidate(c) for c in candidates]
    reference = parse_candidate(REFERENCE)
    if reference.label not in {c.label for c in parsed}:
        parsed.insert(0, reference)
    else:
        parsed.sort(key=lambda c: c.label != reference.label)
    labels = [c.label for c in parsed]
    if len(set(labels)) != len(labels):
        raise ValueError(f"duplicate candidates: {labels}")

    gate_ref_label = gate_reference or reference.label
    gate_cand_label = gate_candidate or ""
    if gate == "G4":
        for needed in (gate_ref_label, gate_cand_label):
            if needed not in labels:
                raise ValueError(f"G4 names {needed!r}, which is not a candidate: {labels}")
    for pair in compare:
        for needed in pair:
            if needed not in labels:
                raise ValueError(f"comparison names {needed!r}, not a candidate: {labels}")
    dataset_dir = Path(dataset).expanduser()
    cfg_path = Path(config_path).expanduser()
    loaded = bm.load_dataset(dataset_dir, data_dir=data_dir)
    cfg_hash = config_hash(cfg_path)
    if loaded.manifest.get("config_hash") != cfg_hash:
        raise ValueError(
            f"tariff config drift: dataset {loaded.dataset_id} pins config_hash="
            f"{str(loaded.manifest.get('config_hash'))[:12]} but {cfg_path} hashes to "
            f"{cfg_hash[:12]}"
        )
    tz = ZoneInfo(load_config(cfg_path).tariff_periods.timezone)
    p0 = _p0_reference(Path(p0_summary) if p0_summary else None, periods)
    version = code_version()
    commit = version.removesuffix("-dirty")
    params = {**BENCHMARK_PARAMS, **dict(benchmark_overrides or {})}

    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = loaded.directory / EVAL_DIRNAME / f"{name}-{stamp}"
    run_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    run_dir.mkdir(mode=0o700)

    config_block = {
        "candidates": [c.as_dict() for c in parsed],
        "reference": reference.label,
        "periods": [p.as_dict() for p in periods],
        "policies": {"primary": PRIMARY_POLICY, "secondary": SECONDARY_POLICY},
        "benchmark_params": params,
        "folds": {
            "rule": FIT_FOLD_RULE,
            "planned": "monthly folds over every period (2025-01 .. 2026-09 for the P0 "
            "periods); forecasters without a fit step need none",
        },
        "metrics": {
            "accuracy": (
                "on usable slots at the policies' real decision instants (day_ahead: 00:00 "
                "local; rolling_horizon: the 14:00 and 20:30 events over each plan "
                "horizon): MAE, WAPE, bias (forecast minus measured), P95 |error| and "
                "price-weighted |error| sum(price*|error|); overall and by horizon bucket "
                f"{[b[0] for b in HORIZON_BUCKETS]} from the decision instant, price "
                "tercile (terciles of the period's usable evaluation prices), day type "
                "(weekday / weekend, holidays ignored), meteorological season and month"
            ),
            "economic": (
                "per candidate and period, a benchmark run with the params above; paired "
                f"per-local-day cost difference against {reference.label} (positive = "
                "candidate cheaper), its sum over the period as the point estimate, and the "
                "share of the P0 gap it captures"
            ),
            "proxy_check": (
                "Spearman correlation between price-weighted |error| and EUR cost across "
                "all candidates, per period and policy, once there are at least 3"
            ),
        },
        "bootstrap": {
            "kind": "non-overlapping block bootstrap, percentile interval",
            "block_days": BLOCK_DAYS,
            "resamples": int(resamples),
            "seed": int(seed),
            "confidence": CONFIDENCE,
            "statistic": "sum over the period of the daily paired saving",
            "shared_draws": "one draw matrix per period (seed, period index), shared by "
            "every comparison in that period",
        },
        "gate": _gate_definition(p0, periods),
        "p0_reference": p0,
    }
    side_full: dict[str, SideInputs] = {}
    for candidate in parsed:
        built = build_side_inputs(
            candidate.name, candidate.params, table=loaded.table, data_dir=data_dir
        )
        if built is not None:
            side_full[candidate.label] = built
    if side_full:
        from battery_optimizer.structured_forecast import RULE_MIRROR

        config_block["side_inputs"] = {label: side.describe() for label, side in side_full.items()}
        config_block["structured"] = {
            "rule_mirror": RULE_MIRROR,
            "temperature": "persistence: the same local clock hour of the most recent of the "
            "last three days it was measured by the decision cutoff (usually yesterday)",
            "price_outlook_for_unpublished_days": "the candidate's price_outlook param "
            "(default persistence; omie_period reads the OMIE day-ahead price usable from 13:30 "
            "local), flagged estimated; published prices are used wherever they are public at "
            "the decision (D+1 from 20:15 local)",
            "second_look_params": "disabled_fallback (lf0 | disabled_history: the climate part "
            "of a day whose rule is disabled), climate_lf0_season (the climate part is LF-0 on "
            "the days of that season even when the rule is enabled); both default to the first "
            "P4 behaviour",
            "diagnostics": "per policy and decision kind: component WAPE of the heat pump "
            "(climate, hot water, total) against the LF-0 trailing mean of the same component, "
            "mask hit-rate (observed climate kWh inside the replayed hours) by price "
            "provenance, and counts of price provenance and climate / hot-water fallbacks",
        }
    if label or notes:
        config_block["run_label"] = {"label": label, "notes": list(notes)}
    if gate == "G4":
        config_block["gate_g4"] = _g4_definition(gate_cand_label, gate_ref_label, compare, periods)
    elif compare:
        config_block["secondary_comparisons"] = [
            {"candidate": c, "reference": r, "gating": False} for c, r in compare
        ]
    preregistration = {
        "schema": SCHEMA,
        "kind": "preregistration",
        "name": name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": loaded.dataset_id,
        "dataset_dir": str(loaded.directory),
        "code_version": version,
        "git_commit": commit,
        "git_dirty": version.endswith("-dirty"),
        "tariff_config_hash": cfg_hash,
        "eval_config_hash": _canonical_sha(config_block),
        **config_block,
    }
    prereg_bytes = _write_json(run_dir / PREREGISTRATION_NAME, preregistration)

    results: dict[str, Any] = {}
    runtime: dict[str, Any] = {c.label: {} for c in parsed}
    folds: dict[str, Any] = {}
    for period in periods:
        data = _period_data(loaded, period, tz)
        folds[period.label] = _folds(data)
        sides = {
            label: side.slice(data.eval_end) if data.eval_end < side.n else side
            for label, side in side_full.items()
        }
        period_out: dict[str, Any] = {
            "window": period.window,
            "price_tercile_edges_eur_per_kwh": list(data.tercile_edges),
            "candidates": {},
        }
        for candidate in parsed:
            t0 = time.perf_counter()
            side = sides.get(candidate.label)
            accuracy = {
                policy: _accuracy(candidate, data, policy, side) for policy in EVAL_POLICIES
            }
            diagnostics = _diagnostics(candidate, data, side)
            t1 = time.perf_counter()
            bench_dir = bm.benchmark_dataset(
                dataset=loaded.directory,
                config_path=cfg_path,
                train_end=period.train_end,
                eval_end=period.eval_end,
                train_days=period.train_days,
                data_dir=data_dir,
                **params,
                **candidate.benchmark_kwargs(),
            )
            t2 = time.perf_counter()
            runtime[candidate.label][period.label] = {
                "accuracy_s": round(t1 - t0, 3),
                "benchmark_s": round(t2 - t1, 3),
            }
            period_out["candidates"][candidate.label] = {
                "accuracy": accuracy,
                "diagnostics": diagnostics,
                "benchmark_run_id": Path(bench_dir).name,
                "benchmark_dir": str(bench_dir),
                "_costs": _run_costs(Path(bench_dir)),
            }
        results[period.label] = period_out

    summary = _summarise(
        parsed=parsed,
        periods=periods,
        results=results,
        p0=p0,
        resamples=int(resamples),
        seed=int(seed),
        g4=config_block.get("gate_g4"),
        compare=compare,
    )
    summary.update(
        {
            "schema": SCHEMA,
            "kind": "summary",
            "name": name,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "run_dir": str(run_dir),
            "provenance": {
                "dataset_id": loaded.dataset_id,
                "dataset_dir": str(loaded.directory),
                "code_version": version,
                "git_commit": commit,
                "git_dirty": version.endswith("-dirty"),
                "tariff_config_hash": cfg_hash,
                "eval_config_hash": preregistration["eval_config_hash"],
                "preregistration_sha256": hashlib.sha256(prereg_bytes).hexdigest(),
                "candidates": [c.as_dict() for c in parsed],
                "benchmark_runs": {
                    period.label: {
                        label: results[period.label]["candidates"][label]["benchmark_run_id"]
                        for label in labels
                    }
                    for period in periods
                },
            },
            "folds": folds,
            "runtime_s": runtime,
        }
    )
    _write_json(run_dir / SUMMARY_JSON_NAME, summary)
    md = run_dir / SUMMARY_MD_NAME
    with md.open("x", encoding="utf-8") as handle:
        handle.write(_markdown(summary, parsed, periods))
    md.chmod(0o600)
    return run_dir


def _summarise(
    *,
    parsed: Sequence[Candidate],
    periods: Sequence[Period],
    results: dict[str, Any],
    p0: dict[str, Any] | None,
    resamples: int,
    seed: int,
    g4: dict[str, Any] | None = None,
    compare: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    ref_label = parsed[0].label
    labels = [c.label for c in parsed]
    perfect = next((c for c in parsed if c.name == ORACLE_REFERENCE and not c.overrides), None)
    out_periods: dict[str, Any] = {}
    in_run_gaps: dict[str, dict[str, float]] = {}
    for k, period in enumerate(periods):
        res = results[period.label]
        costs = {label: res["candidates"][label].pop("_costs") for label in labels}
        ref = costs[ref_label]
        economic: dict[str, Any] = {}
        for label in labels:
            cand = costs[label]
            economic[label] = {
                "no_battery_cost_eur": float(cand[bm.POLICY_NO_BATTERY]["cost"].sum()),
                **{
                    policy: {
                        "cost_eur": float(cand[policy]["cost"].sum()),
                        "vs_reference": _paired(
                            ref[policy], cand[policy], resamples=resamples, seed=(seed, k)
                        ),
                    }
                    for policy in EVAL_POLICIES
                },
            }
        if perfect is not None:
            in_run_gaps[period.label] = {
                policy: float(np.sum(ref[policy]["cost"] - costs[perfect.label][policy]["cost"]))
                for policy in EVAL_POLICIES
            }
        p0_period = ((p0 or {}).get("periods") or {}).get(period.label) or {}
        for policy in EVAL_POLICIES:
            gap = (p0_period.get("policies") or {}).get(policy, {}).get("gap_eur")
            if gap is None:
                gap = in_run_gaps.get(period.label, {}).get(policy)
            for label in labels:
                record = economic[label][policy]["vs_reference"]
                record["p0_gap_eur"] = gap
                record["share_of_p0_gap"] = record["point_eur"] / gap if gap else None
        proxy = {
            policy: _spearman(
                [
                    res["candidates"][label]["accuracy"][policy]["overall"][
                        "price_weighted_abs_error_eur"
                    ]
                    for label in labels
                ],
                [economic[label][policy]["cost_eur"] for label in labels],
            )
            for policy in EVAL_POLICIES
        }
        pairwise: dict[str, Any] = {}
        pair_specs = [(g4["candidate"], g4["reference"], "gate")] if g4 else []
        pair_specs += [(c, r, "secondary") for c, r in compare]
        for cand_label, ref_name, role in pair_specs:
            pairwise[f"{cand_label} vs {ref_name}"] = {
                "role": role,
                **{
                    policy: _season_pairs(
                        costs[ref_name][policy],
                        costs[cand_label][policy],
                        resamples=resamples,
                        seed=(seed, k, 1000 + 10 * len(pairwise) + e),
                    )
                    for e, policy in enumerate(EVAL_POLICIES)
                },
            }
        out_periods[period.label] = {
            **res,
            "economic": economic,
            "proxy_check": proxy,
            "pairwise": pairwise,
        }

    gate_gaps: dict[str, float | None] = {}
    for period in periods:
        p0_period = ((p0 or {}).get("periods") or {}).get(period.label) or {}
        gap = (p0_period.get("policies") or {}).get(PRIMARY_POLICY, {}).get("gap_eur")
        if gap is None:
            gap = in_run_gaps.get(period.label, {}).get(PRIMARY_POLICY)
        gate_gaps[period.label] = gap
    gate = {
        c.label: _evaluate_gate(
            c,
            {p.label: out_periods[p.label]["economic"][c.label] for p in periods},
            gate_gaps,
            periods,
        )
        for c in parsed
    }

    sanity: dict[str, Any] = {
        "reference_vs_itself": {
            period.label: {
                policy: {
                    key: out_periods[period.label]["economic"][ref_label][policy]["vs_reference"][
                        key
                    ]
                    for key in ("point_eur", "ci_low_eur", "ci_high_eur")
                }
                for policy in EVAL_POLICIES
            }
            for period in periods
        }
    }
    if perfect is not None and p0 is not None:
        reproduction: dict[str, Any] = {}
        exact = True
        for period in periods:
            p0_period = (p0.get("periods") or {}).get(period.label)
            if not p0_period:
                continue
            economic = out_periods[period.label]["economic"]
            for policy in EVAL_POLICIES:
                ref_p0 = p0_period["policies"].get(policy) or {}
                row = {
                    "lf0_cost_eur": (
                        economic[ref_label][policy]["cost_eur"],
                        ref_p0.get("lf0_cost_eur"),
                    ),
                    "perfect_cost_eur": (
                        economic[perfect.label][policy]["cost_eur"],
                        ref_p0.get("perfect_cost_eur"),
                    ),
                    "gap_eur": (in_run_gaps[period.label][policy], ref_p0.get("gap_eur")),
                }
                cells = {
                    key: {"this_run": a, "p0": b, "identical": a == b}
                    for key, (a, b) in row.items()
                }
                exact = exact and all(c["identical"] for c in cells.values())
                reproduction.setdefault(period.label, {})[policy] = cells
        sanity["p0_reproduction"] = {
            # None when the P0 summary covers none of these periods: nothing was compared.
            "bit_identical": exact if reproduction else None,
            "periods": reproduction,
        }
    out: dict[str, Any] = {
        "periods": out_periods,
        "gate": gate,
        "in_run_p0_gap_eur": in_run_gaps,
        "sanity": sanity,
    }
    if g4:
        key = f"{g4['candidate']} vs {g4['reference']}"
        instances: dict[str, Any] = {}
        holds: dict[str, list[str]] = {name: [] for name in G4_SEASONS}
        for period in periods:
            record = out_periods[period.label]["pairwise"][key][PRIMARY_POLICY]
            for name in G4_SEASONS:
                row = record[name]
                instances[f"{period.label}/{name}"] = row
                if row.get("holds"):
                    holds[name].append(period.label)
        passed = all(holds[name] for name in G4_SEASONS)
        out["g4"] = {
            "id": "G4",
            "candidate": g4["candidate"],
            "reference": g4["reference"],
            "passed": passed,
            "holds": holds,
            "instances": instances,
            "periods": {
                period.label: {
                    policy: out_periods[period.label]["pairwise"][key][policy]["period"]
                    for policy in EVAL_POLICIES
                }
                for period in periods
            },
        }
    return out


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _f(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: Any) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _markdown(
    summary: dict[str, Any], parsed: Sequence[Candidate], periods: Sequence[Period]
) -> str:
    prov = summary["provenance"]
    lines = [
        f"# Forecast evaluation `{summary['name']}`",
        "",
        "Private. Saving = cost(reference) − cost(candidate); positive means the candidate "
        "is cheaper. EUR on the shared pre-tax scoring basis.",
        "",
        f"- Dataset `{prov['dataset_id']}`; code `{prov['code_version']}` "
        f"(dirty: {_f(prov['git_dirty'])}); tariff config `{prov['tariff_config_hash'][:12]}`; "
        f"eval config `{prov['eval_config_hash'][:12]}`; preregistration "
        f"`{prov['preregistration_sha256'][:12]}`.",
        "- Candidates: "
        + ", ".join(
            f"`{c.label}` ({c.name} {json.dumps(c.params, sort_keys=True)})" for c in parsed
        ),
        "",
    ]
    for period in periods:
        res = summary["periods"][period.label]
        lines += [f"## {period.label} ({res['window']})", ""]
        lines += [
            "| candidate | policy | cost | saving vs ref | 95 % CI | mean/day | share of P0 gap |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in parsed:
            econ = res["economic"][c.label]
            for policy in EVAL_POLICIES:
                cmp = econ[policy]["vs_reference"]
                lines.append(
                    f"| {c.label} | {policy} | {_f(econ[policy]['cost_eur'])} | "
                    f"{_f(cmp['point_eur'])} | [{_f(cmp['ci_low_eur'])}, {_f(cmp['ci_high_eur'])}] "
                    f"| {_f(cmp['mean_daily_eur'], 4)} | {_pct(cmp['share_of_p0_gap'])} |"
                )
        lines += [
            "",
            "| candidate | policy | decisions | MAE kWh | WAPE | bias kWh | P95 kWh "
            "| price-weighted |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in parsed:
            for policy in EVAL_POLICIES:
                acc = res["candidates"][c.label]["accuracy"][policy]
                o = acc["overall"]
                lines.append(
                    f"| {c.label} | {policy} | {acc['decisions']} | {_f(o.get('mae_kwh'), 4)} | "
                    f"{_pct(o.get('wape'))} | {_f(o.get('bias_kwh'), 4)} | "
                    f"{_f(o.get('p95_abs_error_kwh'), 4)} | "
                    f"{_f(o.get('price_weighted_abs_error_eur'))} |"
                )
        lines += ["", "MAE (kWh/slot) by horizon bucket:", ""]
        buckets = [b[0] for b in HORIZON_BUCKETS]
        lines += [
            "| candidate | policy | " + " | ".join(buckets) + " |",
            "|---|---|" + "---|" * len(buckets),
        ]
        for c in parsed:
            for policy in EVAL_POLICIES:
                by_h = res["candidates"][c.label]["accuracy"][policy]["by_horizon"]
                cells = [_f(by_h.get(b, {}).get("mae_kwh"), 4) for b in buckets]
                lines.append(f"| {c.label} | {policy} | " + " | ".join(cells) + " |")
        lines += ["", "Proxy check (price-weighted error vs EUR, Spearman):", ""]
        for policy, check in res["proxy_check"].items():
            if check["status"] == "computed":
                lines.append(
                    f"- {policy}: rho {_f(check['rho'], 3)}, same order: "
                    f"{_f(check['same_order'])} ({check['candidates']} candidates)"
                )
            else:
                lines.append(f"- {policy}: {check['status']}")
        lines.append("")
    lines += ["## Gate G2", ""]
    for c in parsed:
        gate = summary["gate"][c.label]
        if c.label == parsed[0].label:
            continue
        status = {True: "PASS", False: "FAIL", None: "UNDETERMINED"}[gate["passed"]]
        if not gate["eligible"]:
            status += " (not eligible: oracle)"
        lines.append(f"- `{c.label}`: {status}")
        for label, row in gate["periods"].items():
            lines.append(
                f"  - {label}: point {_f(row['point_eur'])}, CI low {_f(row['ci_low_eur'])}, "
                f"share of P0 gap {_pct(row['share_of_p0_gap'])}"
            )
    g4 = summary.get("g4")
    if g4:
        verdict = "PASS" if g4["passed"] else "FAIL"
        lines += [
            "",
            f"## Gate G4: `{g4['candidate']}` vs `{g4['reference']}`: {verdict}",
            "",
            "Primary policy rolling_horizon; a season instance holds when its 95 % interval "
            "is entirely above zero.",
            "",
            f"- seasons holding: {json.dumps(g4['holds'])}",
        ]
        for label, row in g4["instances"].items():
            if "status" in row:
                lines.append(f"  - {label}: {row['status']} ({row['days']} days)")
            else:
                lines.append(
                    f"  - {label}: {row['days']} days, saving {_f(row['point_eur'])}, "
                    f"[{_f(row['ci_low_eur'])}, {_f(row['ci_high_eur'])}], "
                    f"holds {_f(row['holds'])}"
                )
    for period in periods:
        pairwise = summary["periods"][period.label].get("pairwise") or {}
        if not pairwise:
            continue
        lines += ["", f"### Paired comparisons, {period.label}", ""]
        lines += [
            "| comparison | role | policy | scope | days | saving | 95 % CI |",
            "|---|---|---|---|---|---|---|",
        ]
        for key, record in pairwise.items():
            for policy in EVAL_POLICIES:
                for scope in ("period", *G4_SEASONS):
                    row = record[policy][scope]
                    if "status" in row:
                        continue
                    lines.append(
                        f"| {key} | {record['role']} | {policy} | {scope} | {row['days']} | "
                        f"{_f(row['point_eur'])} | [{_f(row['ci_low_eur'])}, "
                        f"{_f(row['ci_high_eur'])}] |"
                    )
    for period in periods:
        for label, cand in summary["periods"][period.label]["candidates"].items():
            diag = cand.get("diagnostics")
            if not diag:
                continue
            lines += ["", f"### Component diagnostics, `{label}`, {period.label}", ""]
            for policy, report in diag.items():
                lines.append(f"- {policy}: {report['decisions']} decisions")
                for group, row in report["groups"].items():
                    cells = ", ".join(
                        f"{k} {_pct(v) if isinstance(v, float) else v}" for k, v in row.items()
                    )
                    lines.append(f"  - {group}: {cells}")
                for group, row in report["counts"].items():
                    lines.append(f"  - {group}: {json.dumps(row, sort_keys=True)}")
    lines += ["", "## Sanity", ""]
    for label, by_policy in summary["sanity"]["reference_vs_itself"].items():
        for policy, row in by_policy.items():
            lines.append(
                f"- reference vs itself, {label} {policy}: point {row['point_eur']!r}, "
                f"CI [{row['ci_low_eur']!r}, {row['ci_high_eur']!r}]"
            )
    repro = summary["sanity"].get("p0_reproduction")
    if repro is not None:
        lines.append(f"- P0 reproduction bit-identical: {_f(repro['bit_identical'])}")
        for label, by_policy in repro["periods"].items():
            for policy, cells in by_policy.items():
                parts = [
                    f"{key} {cell['this_run']!r} vs {cell['p0']!r}" for key, cell in cells.items()
                ]
                lines.append(f"  - {label} {policy}: " + "; ".join(parts))
    lines += ["", "## Runtime (s)", ""]
    for label, by_period in summary["runtime_s"].items():
        parts = [
            f"{p}: accuracy {r['accuracy_s']}, benchmark {r['benchmark_s']}"
            for p, r in by_period.items()
        ]
        lines.append(f"- `{label}`: " + "; ".join(parts))
    lines += ["", "Benchmark runs:", ""]
    for label, runs in summary["provenance"]["benchmark_runs"].items():
        lines.append(f"- {label}: " + ", ".join(f"`{k}` {v}" for k, v in runs.items()))
    return "\n".join(lines) + "\n"
