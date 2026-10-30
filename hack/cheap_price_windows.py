#!/usr/bin/env python3
"""Read-only statistics of the cheap-price windows in a dataset, per month and per season.

From the usable intervals of a built dataset (``benchmark.usable_mask``), per local month:

* the hours at a price at or below each ``--thresholds`` value (EUR/kWh, default 0, 0.02, 0.04,
  0.06) and the hours below ``--below-fraction`` (default 25%) of the local day's mean price;
* the days with a run of at least ``--min-run-hours`` (default 3) consecutive hours at or below
  each threshold, split weekday / weekend, with the median start and end of the day's longest run
  (local hours) and the median and maximum length of those longest runs;
* the minimum price.

``--heating-months`` (default 11,12,1,2,3) splits the same figures into a heating and a
non-heating aggregate.  A run is a chain of usable intervals adjacent in time; the day of a run
is the local day it starts in, and each day counts once with its longest run.  Prices are the
dataset's pre-tax ``price_eur_per_kwh``.

Usage:
    uv run python hack/cheap_price_windows.py \\
        --dataset ~/data/<store>/analysis/v1/<dataset_id> \\
        [--data-dir ~/data/<store>] [--run-id <name>] \\
        [--thresholds 0,0.02,0.04,0.06] [--min-run-hours 3] [--below-fraction 0.25] \\
        [--heating-months 11,12,1,2,3]

``BATTERY_OPTIMIZER_DATA_DIR`` replaces ``--data-dir``.  Everything is written to
``<data_dir>/analysis/v1/cheap-price-windows/<run-id>/`` (``report.json``, ``report.md``); the
repository is never touched.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "src"))

from battery_optimizer import benchmark  # noqa: E402
from battery_optimizer.features import json_bytes  # noqa: E402
from battery_optimizer.paths import resolve_data_dir, write_part_atomic  # noqa: E402


def _run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value) or ".." in value:
        raise argparse.ArgumentTypeError(f"{value!r}: use only [A-Za-z0-9._-] and no '..'")
    return value


def _floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(","))


def _months(text: str) -> frozenset[int]:
    return frozenset(int(x) for x in text.split(","))


def runs(low: np.ndarray, adjacent: np.ndarray) -> list[tuple[int, int]]:
    """``(start, length)`` of every maximal chain of ``low`` slots; ``adjacent[i]`` says that slot
    ``i`` directly follows slot ``i - 1`` in time."""
    out: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(low):
        if flag and start is not None and adjacent[i]:
            continue
        if start is not None:
            out.append((start, i - start))
            start = None
        if flag:
            start = i
    if start is not None:
        out.append((start, low.size - start))
    return out


def longest_run_per_day(
    low: np.ndarray, adjacent: np.ndarray, day: np.ndarray, min_len: int
) -> list[tuple[Any, int, int]]:
    """``(local day, start index, length)`` of each day's longest run of at least ``min_len``.

    The day is that of the run's first slot; the first of equally long runs wins.
    """
    best: dict[Any, tuple[int, int]] = {}
    for start, length in runs(low, adjacent):
        if length < min_len:
            continue
        d = day[start]
        if d not in best or length > best[d][1]:
            best[d] = (start, length)
    return [(d, s, n) for d, (s, n) in sorted(best.items(), key=lambda kv: kv[1][0])]


def _med(x: list[float]) -> float | None:
    return float(np.median(x)) if x else None


def _run_stats(days: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "days": len(days),
        "weekend_days": sum(1 for d in days if d["weekend"]),
        "weekday_days": sum(1 for d in days if not d["weekend"]),
        "start_median_h": _med([d["start_h"] for d in days]),
        "end_median_h": _med([d["end_h"] for d in days]),
        "longest_run_median_h": _med([d["hours"] for d in days]),
        "longest_run_max_h": max((d["hours"] for d in days), default=None),
    }


def analyse(
    *,
    starts_us: np.ndarray,
    price: np.ndarray,
    usable: np.ndarray,
    local_day: np.ndarray,
    interval_s: int,
    tz: ZoneInfo,
    thresholds: tuple[float, ...],
    min_run_hours: float,
    below_fraction: float,
    heating_months: frozenset[int],
) -> dict[str, Any]:
    slot_h = interval_s / 3600.0
    min_len = int(round(min_run_hours / slot_h))
    keep = np.flatnonzero(usable)
    t_us, p, day = starts_us[keep], price[keep], local_day[keep]
    adjacent = np.concatenate(([False], np.diff(t_us) == interval_s * 1_000_000))
    local = [datetime.fromtimestamp(int(u) / 1e6, tz) for u in t_us]
    month = np.array([f"{d.year}-{d.month:02d}" for d in local])
    hour = np.array([d.hour + d.minute / 60 for d in local])
    # The day mean is over the usable intervals of the same local day.
    day_mean = np.empty_like(p)
    for d in np.unique(day):
        sel = day == d
        day_mean[sel] = p[sel].mean()
    below = p < below_fraction * day_mean
    weekend = np.array([datetime.fromisoformat(str(d)).weekday() >= 5 for d in day])

    qualifying: dict[float, list[dict[str, Any]]] = {}
    for thr in thresholds:
        found = []
        for d, s, n in longest_run_per_day(p <= thr, adjacent, day, min_len):
            start_h = float(hour[s])
            found.append(
                {
                    "day": str(d),
                    "month": str(month[s]),
                    "weekend": bool(weekend[s]),
                    "start_h": start_h,
                    "end_h": (start_h + n * slot_h) % 24 or 24.0,
                    "hours": n * slot_h,
                }
            )
        qualifying[thr] = found

    def block(sel: np.ndarray, months: set[str] | None) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "usable_days": int(np.unique(day[sel]).size),
            "min_price": float(p[sel].min()) if sel.any() else None,
            "hours_below_fraction_of_day_mean": float(below[sel].sum() * slot_h),
            "thresholds": {},
        }
        for thr in thresholds:
            days = [q for q in qualifying[thr] if months is None or q["month"] in months]
            entry["thresholds"][f"{thr:g}"] = {
                "hours_at_or_below": float((p[sel] <= thr).sum() * slot_h),
            } | _run_stats(days)
        return entry

    months_sorted = sorted(set(month))
    per_month = {m: block(month == m, {m}) for m in months_sorted}
    is_heating = {m: int(m[-2:]) in heating_months for m in months_sorted}
    seasons = {}
    for label, flag in (("heating", True), ("non_heating", False)):
        ms = {m for m in months_sorted if is_heating[m] == flag}
        sel = np.isin(month, sorted(ms))
        seasons[label] = block(sel, ms)
    return {
        "usable_intervals": int(keep.size),
        "months": per_month,
        "seasons": seasons,
        "overall": block(np.ones(keep.size, dtype=bool), None),
    }


def _fmt(v: float | None, spec: str = ".2f") -> str:
    return "n/a" if v is None else format(v, spec)


def render_markdown(report: dict[str, Any]) -> str:
    p = report["parameters"]
    keys = [f"{t:g}" for t in p["thresholds"]]
    lines = [
        "# Cheap price windows",
        "",
        f"Dataset `{report['dataset_id']}`, time zone {p['timezone']}, prices in EUR/kWh "
        f"(pre-tax). A qualifying day has a run of at least {p['min_run_hours']:g} h at or below "
        f"the threshold; start / end are medians of the day's longest run (local hours). "
        f"Heating months {p['heating_months']}.",
        "",
    ]
    for key in keys:
        lines += [
            f"## Threshold {key}",
            "",
            "| Period | Hours at or below | Hours below fraction of day mean | Qualifying days "
            "(weekday / weekend) | Start | End | Longest run median / max h | Min price |",
            "|---|---|---|---|---|---|---|---|",
        ]
        rows = list(report["months"].items()) + list(report["seasons"].items())
        rows.append(("overall", report["overall"]))
        for name, b in rows:
            t = b["thresholds"][key]
            lines.append(
                f"| {name} | {_fmt(t['hours_at_or_below'], '.2f')} | "
                f"{_fmt(b['hours_below_fraction_of_day_mean'], '.2f')} | "
                f"{t['days']} ({t['weekday_days']} / {t['weekend_days']}) | "
                f"{_fmt(t['start_median_h'])} | {_fmt(t['end_median_h'])} | "
                f"{_fmt(t['longest_run_median_h'])} / {_fmt(t['longest_run_max_h'])} | "
                f"{_fmt(b['min_price'], '.4f')} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", required=True, help="canonical dataset directory")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--thresholds", type=_floats, default=(0.0, 0.02, 0.04, 0.06))
    parser.add_argument("--min-run-hours", type=float, default=3.0)
    parser.add_argument("--below-fraction", type=float, default=0.25)
    parser.add_argument("--heating-months", type=_months, default=frozenset({11, 12, 1, 2, 3}))
    parser.add_argument(
        "--run-id", type=_run_id, default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    args = parser.parse_args(argv)

    root = resolve_data_dir(args.data_dir)
    out_dir = root / "analysis" / "v1" / "cheap-price-windows" / args.run_id
    loaded = benchmark.load_dataset(args.dataset, data_dir=root)
    table = loaded.table
    usable, _ = benchmark.usable_mask(table)
    tz_name = table.column("tz_local")[0].as_py()
    result = analyse(
        starts_us=table.column("interval_start").cast(pa.int64()).to_numpy(),
        price=table.column("price_eur_per_kwh").to_numpy(zero_copy_only=False).astype(float),
        usable=np.asarray(usable, dtype=bool),
        local_day=np.array([str(d) for d in table.column("local_day").to_pylist()]),
        interval_s=int(loaded.interval_seconds),
        tz=ZoneInfo(tz_name),
        thresholds=args.thresholds,
        min_run_hours=args.min_run_hours,
        below_fraction=args.below_fraction,
        heating_months=args.heating_months,
    )
    report = {
        "dataset_id": loaded.dataset_id,
        "parameters": {
            "timezone": tz_name,
            "thresholds": list(args.thresholds),
            "min_run_hours": args.min_run_hours,
            "below_fraction": args.below_fraction,
            "heating_months": sorted(args.heating_months),
        },
    } | result
    write_part_atomic(out_dir / "report.json", json_bytes(report))
    write_part_atomic(out_dir / "report.md", render_markdown(report).encode("utf-8"))
    print(render_markdown(report))
    print(f"written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
