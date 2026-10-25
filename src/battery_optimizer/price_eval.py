"""Price-outlook evaluation harness (PO-A of the price-outlook improvement plan).

One call compares registered price outlooks (:mod:`battery_optimizer.price_outlook`) on
the ``rolling_horizon`` rung and writes a private, pre-registered run under
``<dataset>/price-eval/<name>-<UTC timestamp>/``:

1. ``preregistration.json`` — outlooks, periods, benchmark params, metrics, bootstrap
   seed and the GA gap definition, written **before** any benchmark runs.
2. **Economic evaluation**: one ``benchmark`` run per outlook and period
   (``no_battery`` and ``rolling_horizon``, the P0 parameters of
   :data:`battery_optimizer.forecast_eval.BENCHMARK_PARAMS`), and the paired
   per-local-day cost difference of every outlook against ``exact`` and against
   ``published_only`` with the weekly block bootstrap of the forecast-eval harness.  The
   GA gap is ``published_only`` minus ``exact``; each outlook's share of it is reported.
3. **Action differences** at the 14:00 events: how many executed slots of the
   14:00-20:30 windows (and how many such days) differ between two outlooks, and how many
   of those events started from the same state of charge, so the difference comes from
   the price information alone.
4. **Price-estimate metrics** for the slots an outlook estimated at the 14:00 events,
   from the same seam call the planner used: MAE, bias (estimate minus realized),
   within-day Spearman rank correlation of the hourly prices, and the overlap of the
   cheapest 10 and the most expensive 4 hours of each fully estimated local day.
5. **Climate-mask overlap** (with ``features=``): at each 14:00 event, the hours AppDaemon's
   ``climate_control`` would select for the fully estimated day when replayed on the
   outlook's prices against when replayed on the realized prices
   (:func:`~battery_optimizer.structured_forecast.replay_mask_for_day`): hour overlap
   (Jaccard) and the share of the day's measured heat-pump climate energy that falls inside
   each mask.  This is the quantity the P4 forecaster needs from a price outlook.
6. ``summary.json`` / ``summary.md`` with provenance: git commit and dirty flag, tariff
   and evaluation config hashes, dataset id, outlook provenance and every benchmark run.

Everything this module writes holds monetary figures or private prices, so it only ever
writes inside the dataset directory, which lives in the private data dir.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from datetime import time as dt_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq

from battery_optimizer import benchmark as bm
from battery_optimizer import forecast_eval as fe
from battery_optimizer.config import load_config
from battery_optimizer.load_forecast import SideInputs
from battery_optimizer.manifest import code_version, config_hash
from battery_optimizer.price_outlook import (
    DEFAULT_PRICE_OUTLOOK,
    OMIE_NEXT_DAY_USABLE_FROM,
    PVPC_NEXT_DAY_PUBLISHED_FROM,
    SLOT_ESTIMATED,
    OutlookSide,
    get_price_outlook,
    published_end_idx,
    run_price_outlook,
)
from battery_optimizer.rolling_horizon import _clock_slot_position
from battery_optimizer.structured_forecast import replay_mask_for_day

__all__ = [
    "BENCHMARK_PARAMS",
    "DEFAULT_OUTLOOKS",
    "EVAL_DIRNAME",
    "run_price_eval",
]

EVAL_DIRNAME = "price-eval"
PREREGISTRATION_NAME = fe.PREREGISTRATION_NAME
SUMMARY_JSON_NAME = fe.SUMMARY_JSON_NAME
SUMMARY_MD_NAME = fe.SUMMARY_MD_NAME
SCHEMA = "battery-optimizer/price-eval/v1"

UPPER = DEFAULT_PRICE_OUTLOOK
LIVE = "published_only"
REFERENCES = (UPPER, LIVE)
DEFAULT_OUTLOOKS = (UPPER, LIVE, "persistence")
#: The estimators of PO-C, evaluated together with the baselines.
PO_C_OUTLOOKS = (UPPER, LIVE, "persistence", "omie_period", "omie_period_month")
POLICY = bm.POLICY_ROLLING_HORIZON

#: The P0 parameters, on the two policies an outlook can be judged by.
BENCHMARK_PARAMS: dict[str, Any] = {
    **fe.BENCHMARK_PARAMS,
    "policies": [bm.POLICY_NO_BATTERY, bm.POLICY_ROLLING_HORIZON],
}

CHEAPEST_HOURS = 10
DEAREST_HOURS = 4
#: Gate GC compares an estimator with this baseline (the naive estimate of PO-A).
GC_BASELINE = "persistence"
GC_ID = "GC"

_FLOW_TOL_KWH = 1e-9


# ---------------------------------------------------------------------------
# Action differences
# ---------------------------------------------------------------------------


def _schedule(run_dir: Path) -> dict[str, np.ndarray]:
    table = pq.read_table(
        run_dir / bm.SCHEDULES_PARQUET_NAME,
        columns=["policy", "charge_kwh", "discharge_kwh"],
        filters=[("policy", "=", POLICY)],
    ).to_pydict()
    return {
        key: np.array([0.0 if v is None else float(v) for v in table[key]])
        for key in ("charge_kwh", "discharge_kwh")
    }


def _afternoon_events(run_dir: Path) -> list[dict[str, Any]]:
    payload = json.loads((run_dir / bm.RESULTS_JSON_NAME).read_text())
    return [e for e in payload[POLICY]["per_event"] if e["kind"] == "afternoon"]


def _action_differences(a: dict[str, Any], b: dict[str, Any], eval_start: int) -> dict[str, Any]:
    """Executed 14:00-20:30 slots that differ between two runs of one period."""
    events_a, events_b = a["events"], b["events"]
    if [e["start_idx"] for e in events_a] != [e["start_idx"] for e in events_b]:
        raise ValueError("the two runs do not share their 14:00 events")
    slots = days = same_soc = same_soc_days = total_slots = 0
    for ea, eb in zip(events_a, events_b, strict=True):
        lo, hi = ea["start_idx"] - eval_start, ea["exec_end_idx"] - eval_start
        total_slots += hi - lo
        differ = (
            np.abs(a["schedule"]["charge_kwh"][lo:hi] - b["schedule"]["charge_kwh"][lo:hi])
            > _FLOW_TOL_KWH
        ) | (
            np.abs(a["schedule"]["discharge_kwh"][lo:hi] - b["schedule"]["discharge_kwh"][lo:hi])
            > _FLOW_TOL_KWH
        )
        n = int(differ.sum())
        slots += n
        days += n > 0
        if abs(ea["soc_start_kwh"] - eb["soc_start_kwh"]) <= _FLOW_TOL_KWH:
            same_soc += 1
            same_soc_days += n > 0
    return {
        "afternoon_events": len(events_a),
        "afternoon_slots": total_slots,
        "slots_differing": slots,
        "days_differing": days,
        "events_with_same_start_soc": same_soc,
        "days_differing_from_same_start_soc": same_soc_days,
    }


# ---------------------------------------------------------------------------
# Price-estimate metrics
# ---------------------------------------------------------------------------


def _overlap(est: np.ndarray, real: np.ndarray, k: int, *, cheapest: bool) -> float:
    k = min(k, est.size)
    order_e = np.argsort(est if cheapest else -est, kind="stable")[:k]
    order_r = np.argsort(real if cheapest else -real, kind="stable")[:k]
    return len(set(order_e.tolist()) & set(order_r.tolist())) / k


def _day_slots(label: Any, tz: ZoneInfo, interval_seconds: float) -> int:
    """Slots of a whole local day (92 / 96 / 100 at 15 minutes across DST)."""
    day = fe._as_date(label)
    start = datetime.combine(day, dt_time.min, tzinfo=tz).astimezone(UTC)
    end = datetime.combine(day + timedelta(days=1), dt_time.min, tzinfo=tz).astimezone(UTC)
    return int(round((end - start).total_seconds() / interval_seconds))


class _MaskAccumulator:
    """Overlap of the replayed climate mask on estimated against realized prices."""

    def __init__(self) -> None:
        self.days = 0
        self.reasons: dict[str, int] = {}
        self.both = 0
        self.jaccard: list[float] = []
        self.hours_est: list[int] = []
        self.hours_real: list[int] = []
        self.same_hours = 0
        self.energy = 0.0
        self.in_est = 0.0
        self.in_real = 0.0

    def add(
        self,
        *,
        feat: SideInputs,
        data: fe._PeriodData,
        cut: int,
        plan_end: int,
        est_slots: np.ndarray,
        est_values: np.ndarray,
        j: int,
    ) -> None:
        view = feat.view(cut, data.days)
        known = min(view.published_end_idx, plan_end)
        est_prices = np.full(plan_end, np.nan)
        est_prices[:known] = data.prices[:known]
        est_prices[est_slots] = est_values
        mask_est, why_est = replay_mask_for_day(view, data.days, est_prices, j, cut)
        mask_real, why_real = replay_mask_for_day(view, data.days, data.prices[:plan_end], j, cut)
        self.days += 1
        if mask_est is None or mask_real is None:
            reason = why_real or why_est
            self.reasons[reason] = self.reasons.get(reason, 0) + 1
            return
        self.both += 1
        union = int((mask_est | mask_real).sum())
        self.jaccard.append(int((mask_est & mask_real).sum()) / union if union else 1.0)
        self.hours_est.append(int(mask_est.sum()))
        self.hours_real.append(int(mask_real.sum()))
        self.same_hours += int(np.array_equal(mask_est, mask_real))
        _label, lo, hi = data.days[j]
        slots = np.arange(lo, hi)
        energy = feat.series["hp_heat_kwh"][slots] + feat.series["hp_cool_kwh"][slots]
        ok = np.isfinite(energy)
        hour = view.local_hour[slots]
        self.energy += float(energy[ok].sum())
        self.in_est += float(energy[ok & mask_est[hour]].sum())
        self.in_real += float(energy[ok & mask_real[hour]].sum())

    def result(self) -> dict[str, Any]:
        if not self.days:
            return {"days": 0}
        return {
            "days": self.days,
            "days_with_both_schedules": self.both,
            "days_without_schedule": dict(sorted(self.reasons.items())),
            "mean_jaccard": float(np.mean(self.jaccard)) if self.jaccard else None,
            "days_identical_mask": self.same_hours,
            "mean_selected_hours_on_estimate": (
                float(np.mean(self.hours_est)) if self.hours_est else None
            ),
            "mean_selected_hours_on_realized": (
                float(np.mean(self.hours_real)) if self.hours_real else None
            ),
            "climate_energy_hit_rate_estimated_mask": (
                self.in_est / self.energy if self.energy > 0 else None
            ),
            "climate_energy_hit_rate_realized_mask": (
                self.in_real / self.energy if self.energy > 0 else None
            ),
        }


def _price_metrics(
    outlook: str,
    data: fe._PeriodData,
    side: OutlookSide | None = None,
    feat: SideInputs | None = None,
) -> dict[str, Any]:
    """Estimate-vs-realized metrics on the slots *outlook* estimated at the 14:00 events."""
    from scipy.stats import spearmanr

    per_hour = int(round(3600.0 / data.interval_seconds))
    est_all: list[np.ndarray] = []
    real_all: list[np.ndarray] = []
    day_rows: list[dict[str, Any]] = []
    partial_days = 0
    masks = _MaskAccumulator() if feat is not None else None
    fallbacks: dict[str, int] = {}
    for ev in data.events:
        if ev.kind != "afternoon":
            continue
        result = run_price_outlook(
            outlook,
            prices=data.prices,
            published_end_idx=published_end_idx(data.starts[ev.start_idx], data.days, data.tz),
            decision_idx=ev.start_idx,
            plan_end_idx=ev.plan_end_idx,
            interval_starts_utc=data.starts,
            days=data.days,
            tz=data.tz,
            allow_oracle=True,
            side=None if side is None else side.slice(len(data.starts)),
        )
        reason = result.diagnostics.get("fallback")
        if reason:
            fallbacks[str(reason)] = fallbacks.get(str(reason), 0) + 1
        idx = ev.start_idx + np.flatnonzero(result.provenance == SLOT_ESTIMATED)
        if idx.size == 0:
            continue
        est = result.values[idx - ev.start_idx]
        real = data.prices[idx]
        est_all.append(est)
        real_all.append(real)
        for j, (label, lo, hi) in enumerate(data.days):
            inside = (idx >= lo) & (idx < hi)
            if not inside.any():
                continue
            if int(inside.sum()) != _day_slots(label, data.tz, data.interval_seconds):
                # A day cut by the evaluation window has no meaningful ranking.
                partial_days += 1
                continue
            hours = np.array(
                [
                    _clock_slot_position(data.starts[i], data.tz, data.interval_seconds) // per_hour
                    for i in idx[inside]
                ]
            )
            keys = np.unique(hours)
            h_est = np.array([est[inside][hours == h].mean() for h in keys])
            h_real = np.array([real[inside][hours == h].mean() for h in keys])
            rho = float(spearmanr(h_est, h_real).statistic) if keys.size > 2 else math.nan
            if masks is not None and feat is not None:
                masks.add(
                    feat=feat,
                    data=data,
                    cut=ev.start_idx,
                    plan_end=ev.plan_end_idx,
                    est_slots=idx,
                    est_values=est,
                    j=j,
                )
            day_slots = int(inside.sum())
            day_abs = np.abs(est[inside] - real[inside])
            day_rows.append(
                {
                    "local_day": str(label),
                    "hours": int(keys.size),
                    "abs_err_sum": float(day_abs.sum()),
                    "slots": day_slots,
                    "negative_price": bool((real[inside] < 0).any()),
                    "dst_day": day_slots != round(86400.0 / data.interval_seconds),
                    "spearman": None if not math.isfinite(rho) else rho,
                    "cheapest_overlap": _overlap(h_est, h_real, CHEAPEST_HOURS, cheapest=True),
                    "dearest_overlap": _overlap(h_est, h_real, DEAREST_HOURS, cheapest=False),
                }
            )
    if not est_all:
        return {
            "estimated_slots": 0,
            "note": "this outlook estimated no slot",
            "fallback_events": dict(sorted(fallbacks.items())),
        }
    est_v = np.concatenate(est_all)
    real_v = np.concatenate(real_all)
    err = est_v - real_v
    rhos = [r["spearman"] for r in day_rows if r["spearman"] is not None]

    def _mean(key: str) -> float | None:
        return float(np.mean([r[key] for r in day_rows])) if day_rows else None

    def _day_class(pick: Any) -> dict[str, Any]:
        rows = [r for r in day_rows if pick(r)]
        slots = sum(r["slots"] for r in rows)
        return {
            "days": len(rows),
            "mae_eur_per_kwh": (sum(r["abs_err_sum"] for r in rows) / slots) if slots else None,
        }

    return {
        "estimated_slots": int(err.size),
        "mae_eur_per_kwh": float(np.abs(err).mean()),
        "bias_eur_per_kwh": float(err.mean()),
        "realized_mean_eur_per_kwh": float(real_v.mean()),
        "full_days": len(day_rows),
        "partial_days_skipped": partial_days,
        "spearman_mean": float(np.mean(rhos)) if rhos else None,
        "spearman_median": float(np.median(rhos)) if rhos else None,
        "spearman_days": len(rhos),
        f"cheapest_{CHEAPEST_HOURS}h_overlap_mean": _mean("cheapest_overlap"),
        f"dearest_{DEAREST_HOURS}h_overlap_mean": _mean("dearest_overlap"),
        "mae_on_days_with_a_negative_price": _day_class(lambda r: r["negative_price"]),
        "mae_on_dst_days": _day_class(lambda r: r["dst_day"]),
        "fallback_events": dict(sorted(fallbacks.items())),
        **({"climate_mask": masks.result()} if masks is not None else {}),
    }


# ---------------------------------------------------------------------------
# Reference reproduction (the default outlook is the pre-seam code)
# ---------------------------------------------------------------------------


def _reference_runs(path: Path | None, dataset_dir: Path) -> dict[str, Any] | None:
    """The LF-0 benchmark run of each period of a forecast-eval ``summary.json``."""
    if path is None:
        return None
    raw = Path(path).expanduser().read_bytes()
    payload = json.loads(raw)
    runs = (payload.get("provenance") or {}).get("benchmark_runs") or {}
    return {
        "source": str(Path(path).expanduser()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "runs": {
            label: str(dataset_dir / bm.BENCHMARKS_DIRNAME / by_label[fe.REFERENCE])
            for label, by_label in runs.items()
            if fe.REFERENCE in by_label
        },
    }


def _reproduction(reference_dir: Path, run_dir: Path) -> dict[str, Any]:
    cols = ["charge_kwh", "discharge_kwh", "soc_kwh", "grid_kwh", "interval_cost_eur"]

    def read(d: Path) -> dict[str, list[Any]]:
        return pq.read_table(
            d / bm.SCHEDULES_PARQUET_NAME, columns=cols, filters=[("policy", "=", POLICY)]
        ).to_pydict()

    a, b = read(reference_dir), read(run_dir)
    identical = {c: a[c] == b[c] for c in cols}
    return {
        "reference_run": reference_dir.name,
        "columns_identical": identical,
        "bit_identical": all(identical.values()),
        "reference_cost_eur": float(sum(v or 0.0 for v in a["interval_cost_eur"])),
        "this_run_cost_eur": float(sum(v or 0.0 for v in b["interval_cost_eur"])),
    }


# ---------------------------------------------------------------------------
# The harness
# ---------------------------------------------------------------------------


def _outlook_record(name: str) -> dict[str, Any]:
    return get_price_outlook(name).provenance({})


def _gc_candidates(names: Sequence[str]) -> list[str]:
    """The estimators gate GC judges: registered OMIE outlooks among *names*."""
    return [o for o in names if get_price_outlook(o).needs_omie]


def _gc_definition(candidates: Sequence[str], features: str | None) -> dict[str, Any] | None:
    """Gate GC of the price-outlook plan, pre-registered with the run."""
    if not candidates:
        return None
    return {
        "id": GC_ID,
        "candidates": list(candidates),
        "baseline": GC_BASELINE,
        "rule": (
            f"an estimator PASSES if, in EVERY period, it beats {GC_BASELINE} on rank and on "
            "rule-hour overlap, all of: mean and median within-day Spearman rank correlation "
            f"of the hourly prices; mean overlap of the {CHEAPEST_HOURS} cheapest hours; mean "
            "Jaccard overlap of the hours the replayed climate_control rule selects on the "
            "estimate against on the realized prices; share of measured climate energy inside "
            "the replayed mask of the estimate. Point comparisons on the same 14:00 events; no "
            "interval is attached because the margins are far larger than the day-to-day "
            "spread"
        ),
        "dispatch_criterion": (
            "not applied: gate GA failed (owner, 2026-10-25), so the share of the GA gap "
            "captured is reported for information only"
        ),
        "rule_hour_metrics": (
            "computed" if features is not None else "NOT computed: no feature set was given"
        ),
    }


def _gc_evaluate(
    periods: Mapping[str, Any], candidates: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    """Per candidate and period, whether each GC criterion holds against the baseline."""

    def beats(a: Any, b: Any) -> bool | None:
        return None if a is None or b is None else bool(a > b)

    out: dict[str, Any] = {}
    for name in candidates:
        per_period: dict[str, Any] = {}
        verdict: bool | None = True
        for label in labels:
            outlooks = periods[label]["outlooks"]
            m, b = outlooks[name]["price_metrics"], outlooks[GC_BASELINE]["price_metrics"]
            cm, cb = m.get("climate_mask") or {}, b.get("climate_mask") or {}
            checks = {
                "spearman_mean": beats(m.get("spearman_mean"), b.get("spearman_mean")),
                "spearman_median": beats(m.get("spearman_median"), b.get("spearman_median")),
                f"cheapest_{CHEAPEST_HOURS}h_overlap": beats(
                    m.get(f"cheapest_{CHEAPEST_HOURS}h_overlap_mean"),
                    b.get(f"cheapest_{CHEAPEST_HOURS}h_overlap_mean"),
                ),
                "mask_jaccard": beats(cm.get("mean_jaccard"), cb.get("mean_jaccard")),
                "mask_energy_hit_rate": beats(
                    cm.get("climate_energy_hit_rate_estimated_mask"),
                    cb.get("climate_energy_hit_rate_estimated_mask"),
                ),
            }
            per_period[label] = checks
            if any(v is False for v in checks.values()):
                verdict = False
            elif any(v is None for v in checks.values()) and verdict is not False:
                verdict = None
        out[name] = {"passed": verdict, "periods": per_period}
    return out


def run_price_eval(
    *,
    dataset: Path | str,
    config_path: Path | str,
    name: str,
    outlooks: Sequence[str] = DEFAULT_OUTLOOKS,
    periods: Sequence[fe.Period] = fe.DEFAULT_PERIODS,
    reference_summary: Path | str | None = None,
    resamples: int = fe.DEFAULT_RESAMPLES,
    seed: int = fe.DEFAULT_SEED,
    data_dir: Path | str | None = None,
    benchmark_overrides: Mapping[str, Any] | None = None,
    now: datetime | None = None,
    features: str | None = None,
    label: str | None = None,
) -> Path:
    """Pre-register, run and summarise one price-outlook evaluation; return its directory.

    Args:
        outlooks: registered price outlooks.  ``exact`` and ``published_only`` are the
            references and are added first when missing.
        reference_summary: a forecast-eval ``summary.json``; its LF-0 benchmark runs are
            checked for bit-identity against this run's ``exact`` rolling-horizon rows.
        resamples / seed: weekly block bootstrap (at least ``forecast_eval.MIN_RESAMPLES``).
        benchmark_overrides: extra ``benchmark_dataset`` keywords (tests use it for
            shorter solver limits); recorded in the pre-registration.
        now: the timestamp of the run directory (tests).
        features: a feature set id (:mod:`battery_optimizer.features`).  When given, the
            climate-mask overlap metrics are computed and its identity is pre-registered.
        label: free text recorded in the pre-registration (for example that a run is a
            second look at already seen periods).
    """
    if not fe._NAME_RE.match(name):
        raise ValueError(f"price-eval name must match {fe._NAME_RE.pattern}, got {name!r}")
    if int(resamples) < fe.MIN_RESAMPLES:
        raise ValueError(f"resamples must be >= {fe.MIN_RESAMPLES}, got {resamples}")
    if not periods or len({p.label for p in periods}) != len(periods):
        raise ValueError("periods must be non-empty with unique labels")
    names = list(REFERENCES) + [o for o in outlooks if o not in REFERENCES]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate outlooks: {list(outlooks)}")
    for outlook in names:
        get_price_outlook(outlook)

    loaded = bm.load_dataset(Path(dataset).expanduser(), data_dir=data_dir)
    cfg_path = Path(config_path).expanduser()
    cfg_hash = config_hash(cfg_path)
    if loaded.manifest.get("config_hash") != cfg_hash:
        raise ValueError(
            f"tariff config drift: dataset {loaded.dataset_id} pins config_hash="
            f"{str(loaded.manifest.get('config_hash'))[:12]} but {cfg_path} hashes to "
            f"{cfg_hash[:12]}"
        )
    tz = ZoneInfo(load_config(cfg_path).tariff_periods.timezone)
    omie_side: OutlookSide | None = None
    if any(get_price_outlook(o).needs_omie for o in names):
        from battery_optimizer.omie_outlook import load_omie_side

        omie_side = load_omie_side(
            loaded.table.column("interval_start").to_pylist(), data_dir=data_dir
        )
    reference = _reference_runs(
        Path(reference_summary) if reference_summary else None, loaded.directory
    )
    version = code_version()
    params = {**BENCHMARK_PARAMS, **dict(benchmark_overrides or {})}

    stamp = (now or datetime.now(UTC)).astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = loaded.directory / EVAL_DIRNAME / f"{name}-{stamp}"
    run_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    run_dir.mkdir(mode=0o700)

    config_block = {
        "outlooks": [_outlook_record(o) for o in names],
        "references": {"upper_bound": UPPER, "live_service": LIVE},
        "periods": [p.as_dict() for p in periods],
        "policy": POLICY,
        "benchmark_params": params,
        "publication_rule": (
            f"PVPC for local day D+1 is published from {PVPC_NEXT_DAY_PUBLISHED_FROM:%H:%M} "
            "local on day D: the 14:00 event does not see it, the 20:30 event does"
        ),
        "omie_rule": (
            None
            if omie_side is None
            else f"OMIE for local day D+1 is usable from {OMIE_NEXT_DAY_USABLE_FROM:%H:%M} "
            "local on day D: the 14:00 event sees it, an earlier decision falls back to "
            "published_only; the fit of an OMIE estimator reads only PVPC days published at "
            "the decision"
        ),
        "omie_side_input": None if omie_side is None else omie_side.describe(),
        "features": features,
        "label": label,
        "metrics": {
            "economic": (
                f"per outlook and period, a benchmark run with the params above and "
                f"rolling_horizon_price_outlook=<outlook>; the paired per-local-day {POLICY} "
                f"cost difference against {UPPER} and against {LIVE} (saving = cost(reference) "
                "- cost(outlook), positive = outlook cheaper). GA gap = cost(published_only) - "
                "cost(exact); share of the GA gap captured = (cost(published_only) - "
                "cost(outlook)) / GA gap"
            ),
            "climate_mask": (
                None
                if features is None
                else "on each fully estimated day of the 14:00 events: AppDaemon's "
                "climate_control replayed (rule version of the day, enable flag and "
                "min_hours_per_day as of the decision, trailing mean from published prices) "
                "on the outlook's prices against on the realized prices: hour overlap "
                "(Jaccard) and the share of the day's measured heat-pump heat + cool energy "
                "inside each mask (pooled over days with both schedules)"
            ),
            "action_differences": (
                "over the slots executed by the 14:00 events (14:00 to 20:30 local): the "
                "number of slots whose charge or discharge differs by more than 1e-9 kWh "
                "between two outlooks, the number of such days, and how many of those events "
                "started from the same SOC (difference due to price information alone)"
            ),
            "price_estimate": (
                "on the slots an outlook estimated at the 14:00 events, from the seam call the "
                "planner used: MAE and bias (estimate - realized) in EUR/kWh; per fully "
                "estimated local day, on hourly mean prices: Spearman rank correlation, overlap "
                f"of the {CHEAPEST_HOURS} cheapest and of the {DEAREST_HOURS} most expensive "
                "hours (shared hours / N), averaged over days"
            ),
        },
        "bootstrap": {
            "kind": "non-overlapping block bootstrap, percentile interval",
            "block_days": fe.BLOCK_DAYS,
            "resamples": int(resamples),
            "seed": int(seed),
            "confidence": fe.CONFIDENCE,
            "statistic": "sum over the period of the daily paired saving",
            "shared_draws": "one draw matrix per period (seed, period index)",
        },
        "gate_gc": _gc_definition(_gc_candidates(names), features),
        "gate": {
            "id": "GA",
            "definition": (
                f"the GA gap (cost of {LIVE} minus cost of {UPPER}, {POLICY}) with its "
                "bootstrap interval, per period. Whether it is material is judged against an "
                "owner threshold that is kept private and is not recorded here; this run "
                "reports the gap and gives no verdict"
            ),
        },
        "reference_reproduction": reference,
    }
    preregistration = {
        "schema": SCHEMA,
        "kind": "preregistration",
        "name": name,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "dataset_id": loaded.dataset_id,
        "dataset_dir": str(loaded.directory),
        "code_version": version,
        "git_commit": version.removesuffix("-dirty"),
        "git_dirty": version.endswith("-dirty"),
        "tariff_config_hash": cfg_hash,
        "eval_config_hash": fe._canonical_sha(config_block),
        **config_block,
    }
    prereg_bytes = fe._write_json(run_dir / PREREGISTRATION_NAME, preregistration)

    from battery_optimizer.structured_forecast import _load_side

    out_periods: dict[str, Any] = {}
    runtime: dict[str, Any] = {}
    bench_runs: dict[str, dict[str, str]] = {}
    for k, period in enumerate(periods):
        data = fe._period_data(loaded, period, tz)
        feat = (
            None
            if features is None
            else _load_side(
                {
                    "features": features,
                    "residual_forecaster": "trailing_mean",
                    "price_outlook": "persistence",
                },
                table=data.table,
                data_dir=data_dir,
            )
        )
        runs: dict[str, dict[str, Any]] = {}
        for outlook in names:
            t0 = time.perf_counter()
            bench_dir = Path(
                bm.benchmark_dataset(
                    dataset=loaded.directory,
                    config_path=cfg_path,
                    train_end=period.train_end,
                    eval_end=period.eval_end,
                    train_days=period.train_days,
                    data_dir=data_dir,
                    rolling_horizon_price_outlook=outlook,
                    **params,
                )
            )
            t1 = time.perf_counter()
            metrics = _price_metrics(outlook, data, omie_side, feat)
            t2 = time.perf_counter()
            runtime.setdefault(outlook, {})[period.label] = {
                "benchmark_s": round(t1 - t0, 3),
                "price_metrics_s": round(t2 - t1, 3),
            }
            runs[outlook] = {
                "dir": bench_dir,
                "costs": fe._run_costs(bench_dir),
                "schedule": _schedule(bench_dir),
                "events": _afternoon_events(bench_dir),
                "price_metrics": metrics,
            }
        bench_runs[period.label] = {o: r["dir"].name for o, r in runs.items()}
        out_periods[period.label] = _period_summary(
            runs, data, resamples=int(resamples), seed=(int(seed), k)
        )
        out_periods[period.label]["window"] = period.window
        if reference is not None and period.label in reference["runs"]:
            out_periods[period.label]["reference_reproduction"] = _reproduction(
                Path(reference["runs"][period.label]), runs[UPPER]["dir"]
            )

    gc_candidates = _gc_candidates(names)
    summary = {
        "schema": SCHEMA,
        "kind": "summary",
        "name": name,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "periods": out_periods,
        "gate_gc": (
            _gc_evaluate(out_periods, gc_candidates, [p.label for p in periods])
            if gc_candidates and GC_BASELINE in names
            else None
        ),
        "provenance": {
            "dataset_id": loaded.dataset_id,
            "dataset_dir": str(loaded.directory),
            "code_version": version,
            "git_commit": version.removesuffix("-dirty"),
            "git_dirty": version.endswith("-dirty"),
            "tariff_config_hash": cfg_hash,
            "eval_config_hash": preregistration["eval_config_hash"],
            "preregistration_sha256": hashlib.sha256(prereg_bytes).hexdigest(),
            "outlooks": [_outlook_record(o) for o in names],
            "benchmark_runs": bench_runs,
        },
        "runtime_s": runtime,
    }
    fe._write_json(run_dir / SUMMARY_JSON_NAME, summary)
    md = run_dir / SUMMARY_MD_NAME
    with md.open("x", encoding="utf-8") as handle:
        handle.write(_markdown(summary, names, periods))
    md.chmod(0o600)
    return run_dir


def _period_summary(
    runs: dict[str, dict[str, Any]],
    data: fe._PeriodData,
    *,
    resamples: int,
    seed: Sequence[int],
) -> dict[str, Any]:
    names = list(runs)
    costs = {o: r["costs"][POLICY] for o, r in runs.items()}
    gap = fe._paired(costs[LIVE], costs[UPPER], resamples=resamples, seed=seed)
    outlooks: dict[str, Any] = {}
    for outlook in names:
        record: dict[str, Any] = {
            "benchmark_run_id": runs[outlook]["dir"].name,
            "cost_eur": float(costs[outlook]["cost"].sum()),
            "no_battery_cost_eur": float(
                runs[outlook]["costs"][bm.POLICY_NO_BATTERY]["cost"].sum()
            ),
            "vs": {
                ref: fe._paired(costs[ref], costs[outlook], resamples=resamples, seed=seed)
                for ref in REFERENCES
            },
            "price_metrics": runs[outlook]["price_metrics"],
            "action_differences": {
                other: _action_differences(runs[outlook], runs[other], data.eval_start)
                for other in names
                if other != outlook
            },
        }
        saving = record["vs"][LIVE]["point_eur"]
        record["share_of_ga_gap"] = saving / gap["point_eur"] if gap["point_eur"] else None
        outlooks[outlook] = record
    return {"ga_gap": gap, "outlooks": outlooks}


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def _markdown(summary: dict[str, Any], names: Sequence[str], periods: Sequence[fe.Period]) -> str:
    f, pct = fe._f, fe._pct
    prov = summary["provenance"]
    lines = [
        f"# Price-outlook evaluation `{summary['name']}`",
        "",
        "Private. Saving = cost(reference) - cost(outlook); positive means the outlook is "
        f"cheaper. `{POLICY}` EUR on the shared pre-tax scoring basis.",
        "",
        f"- Dataset `{prov['dataset_id']}`; code `{prov['code_version']}` "
        f"(dirty: {f(prov['git_dirty'])}); tariff config `{prov['tariff_config_hash'][:12]}`; "
        f"eval config `{prov['eval_config_hash'][:12]}`; preregistration "
        f"`{prov['preregistration_sha256'][:12]}`.",
        "- Outlooks: " + ", ".join(f"`{o}`" for o in names),
        "",
    ]
    for period in periods:
        res = summary["periods"][period.label]
        gap = res["ga_gap"]
        lines += [
            f"## {period.label} ({res['window']})",
            "",
            f"GA gap ({LIVE} - {UPPER}): {f(gap['point_eur'])} "
            f"[{f(gap['ci_low_eur'])}, {f(gap['ci_high_eur'])}], {gap['days']} days.",
            "",
            "| outlook | cost | vs exact | 95 % CI | vs published_only | 95 % CI | share of GA |",
            "|---|---|---|---|---|---|---|",
        ]
        for o in names:
            r = res["outlooks"][o]
            a, b = r["vs"][UPPER], r["vs"][LIVE]
            lines.append(
                f"| {o} | {f(r['cost_eur'])} | {f(a['point_eur'])} | "
                f"[{f(a['ci_low_eur'])}, {f(a['ci_high_eur'])}] | {f(b['point_eur'])} | "
                f"[{f(b['ci_low_eur'])}, {f(b['ci_high_eur'])}] | {pct(r['share_of_ga_gap'])} |"
            )
        lines += [
            "",
            "Executed 14:00-20:30 actions that differ (slots / days; days from the same "
            "start SOC):",
            "",
            "| outlook | other | slots | of | days | of | same-SOC days |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, o in enumerate(names):
            for other in names[i + 1 :]:
                d = res["outlooks"][o]["action_differences"][other]
                lines.append(
                    f"| {o} | {other} | {d['slots_differing']} | {d['afternoon_slots']} | "
                    f"{d['days_differing']} | {d['afternoon_events']} | "
                    f"{d['days_differing_from_same_start_soc']} of "
                    f"{d['events_with_same_start_soc']} |"
                )
        lines += [
            "",
            "Price estimates at 14:00 (estimated slots only):",
            "",
            "| outlook | slots | MAE EUR/kWh | bias | Spearman mean | median "
            f"| cheapest-{CHEAPEST_HOURS} | dearest-{DEAREST_HOURS} | days |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for o in names:
            m = res["outlooks"][o]["price_metrics"]
            if not m.get("estimated_slots"):
                lines.append(f"| {o} | 0 | n/a | n/a | n/a | n/a | n/a | n/a | 0 |")
                continue
            lines.append(
                f"| {o} | {m['estimated_slots']} | {f(m['mae_eur_per_kwh'], 4)} | "
                f"{f(m['bias_eur_per_kwh'], 4)} | {f(m['spearman_mean'], 3)} | "
                f"{f(m['spearman_median'], 3)} | "
                f"{pct(m[f'cheapest_{CHEAPEST_HOURS}h_overlap_mean'])} | "
                f"{pct(m[f'dearest_{DEAREST_HOURS}h_overlap_mean'])} | {m['full_days']} |"
            )
        masks = {
            o: res["outlooks"][o]["price_metrics"].get("climate_mask")
            for o in names
            if res["outlooks"][o]["price_metrics"].get("climate_mask")
        }
        if masks:
            lines += [
                "",
                "Replayed climate mask on the estimated day against on the realized prices "
                "(days with both schedules):",
                "",
                "| outlook | days | both | mean Jaccard | identical | hours (est / real) "
                "| energy in estimated mask | in realized mask |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for o, m in masks.items():
                lines.append(
                    f"| {o} | {m['days']} | {m.get('days_with_both_schedules', 0)} | "
                    f"{f(m.get('mean_jaccard'), 3)} | {m.get('days_identical_mask', 0)} | "
                    f"{f(m.get('mean_selected_hours_on_estimate'), 1)} / "
                    f"{f(m.get('mean_selected_hours_on_realized'), 1)} | "
                    f"{pct(m.get('climate_energy_hit_rate_estimated_mask'))} | "
                    f"{pct(m.get('climate_energy_hit_rate_realized_mask'))} |"
                )
        extra = [
            f"- `{o}`: MAE on days with a negative price "
            f"{f(m['mae_on_days_with_a_negative_price']['mae_eur_per_kwh'], 4)} "
            f"({m['mae_on_days_with_a_negative_price']['days']} days), on DST days "
            f"{f(m['mae_on_dst_days']['mae_eur_per_kwh'], 4)} ({m['mae_on_dst_days']['days']} "
            f"days); fallback events {m['fallback_events'] or 'none'}"
            for o in names
            if (m := res["outlooks"][o]["price_metrics"]).get("estimated_slots")
        ]
        if extra:
            lines += ["", *extra]
        repro = res.get("reference_reproduction")
        if repro is not None:
            lines += [
                "",
                f"Reference reproduction (`{UPPER}` against forecast-eval run "
                f"`{repro['reference_run']}`): bit-identical {f(repro['bit_identical'])}; "
                f"cost {repro['this_run_cost_eur']!r} vs {repro['reference_cost_eur']!r}.",
            ]
        lines.append("")
    gc = summary.get("gate_gc")
    if gc:
        lines += ["## Gate GC (estimator against persistence, every period)", ""]
        for o, verdict in gc.items():
            word = {True: "PASS", False: "FAIL", None: "UNDETERMINED"}[verdict["passed"]]
            lines.append(f"- `{o}`: **{word}**")
            for label, checks in verdict["periods"].items():
                failed = [k for k, v in checks.items() if v is False]
                unknown = [k for k, v in checks.items() if v is None]
                note = "all criteria beat persistence"
                if failed:
                    note = "does not beat persistence on " + ", ".join(failed)
                elif unknown:
                    note = "not computed: " + ", ".join(unknown)
                lines.append(f"  - {label}: {note}")
        lines.append("")
    lines += ["## Runtime (s)", ""]
    for o, by_period in summary["runtime_s"].items():
        parts = [
            f"{p}: benchmark {r['benchmark_s']}, price metrics {r['price_metrics_s']}"
            for p, r in by_period.items()
        ]
        lines.append(f"- `{o}`: " + "; ".join(parts))
    lines += ["", "Benchmark runs:", ""]
    for label, runs in prov["benchmark_runs"].items():
        lines.append(f"- {label}: " + ", ".join(f"`{k}` {v}" for k, v in runs.items()))
    return "\n".join(lines) + "\n"
