#!/usr/bin/env python3
"""Read-only comfort history per winter: how warm the rooms were and what the setpoints said.

For each winter (default S1 2023-11..2024-03, S2 2024-11..2025-03, S3 2025-11..2026-03, whole
months in the local time zone) it reports, from the climate feature set:

* the hall-kitchen temperature mean / median / p10 / p90 and the hall-kitchen setpoint mean in
  three windows: weekday 12-22, weekend 09-22 and night 00-07 (local hours, ``[start, end)``);
* the share of time at or above ``--hot`` (default 25 degC) and ``--comfort`` (default 23.5 degC)
  in the two day windows;
* each room's share of time below ``--floor`` (default 21 degC) and its p10 over the winter;
* the data coverage (share of slots with a reading) of every figure.

Shares and statistics use the slots with a reading only; the coverage says how many that is.

Usage:
    uv run python hack/comfort_history.py \\
        --feature-set <climate feature set id or prefix> \\
        [--data-dir ~/data/<store>] [--run-id <name>] [--timezone Europe/Madrid] \\
        [--winters S1:2023-11:2024-03,S2:2024-11:2025-03,S3:2025-11:2026-03] \\
        [--hot 25] [--comfort 23.5] [--floor 21]

``BATTERY_OPTIMIZER_DATA_DIR`` replaces ``--data-dir``.  Everything is written to
``<data_dir>/analysis/v1/comfort-history/<run-id>/`` (``report.json``, ``report.md``); the
repository is never touched.  The report holds private figures: keep it in the data dir.
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

from battery_optimizer.climate.data import ROOMS  # noqa: E402
from battery_optimizer.features import FEATURES_DIRNAME, json_bytes, load_features  # noqa: E402
from battery_optimizer.paths import resolve_data_dir, write_part_atomic  # noqa: E402

DEFAULT_WINTERS = "S1:2023-11:2024-03,S2:2024-11:2025-03,S3:2025-11:2026-03"
#: Window name -> (weekday only / weekend only / any, first hour, last hour exclusive).
WINDOWS = {
    "weekday_12_22": ("weekday", 12.0, 22.0),
    "weekend_09_22": ("weekend", 9.0, 22.0),
    "night_00_07": ("any", 0.0, 7.0),
}
DAY_WINDOWS = ("weekday_12_22", "weekend_09_22")
FOCUS_ROOM = "hall_kitchen"


def _run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value) or ".." in value:
        raise argparse.ArgumentTypeError(f"{value!r}: use only [A-Za-z0-9._-] and no '..'")
    return value


def parse_winters(text: str) -> dict[str, tuple[int, int]]:
    """``name:YYYY-MM:YYYY-MM`` (first and last month, inclusive) -> month indices."""
    winters: dict[str, tuple[int, int]] = {}
    for part in text.split(","):
        name, first, last = part.split(":")
        y0, m0 = (int(x) for x in first.split("-"))
        y1, m1 = (int(x) for x in last.split("-"))
        winters[name] = (y0 * 12 + m0 - 1, y1 * 12 + m1 - 1)
    return winters


def _finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _stats(x: np.ndarray) -> dict[str, float | int | None]:
    v = _finite(x)
    if v.size == 0:
        return {"mean": None, "median": None, "p10": None, "p90": None, "n": 0}
    return {
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
        "n": int(v.size),
    }


def _share(x: np.ndarray, test: np.ndarray) -> float | None:
    ok = np.isfinite(x)
    return float(test[ok].mean()) if ok.any() else None


def _coverage(x: np.ndarray) -> float | None:
    return float(np.isfinite(x).mean()) if x.size else None


def local_calendar(starts_us: np.ndarray, tz: ZoneInfo) -> dict[str, np.ndarray]:
    """Local month index (``year * 12 + month - 1``), weekday and hour of day per slot."""
    local = [datetime.fromtimestamp(int(u) / 1e6, tz) for u in starts_us]
    return {
        "month": np.array([d.year * 12 + d.month - 1 for d in local]),
        "weekday": np.array([d.weekday() for d in local]),
        "hour": np.array([d.hour + d.minute / 60 for d in local]),
    }


def window_mask(cal: dict[str, np.ndarray], kind: str, h0: float, h1: float) -> np.ndarray:
    day = {
        "weekday": cal["weekday"] < 5,
        "weekend": cal["weekday"] >= 5,
        "any": np.ones(cal["weekday"].shape, dtype=bool),
    }[kind]
    return day & (cal["hour"] >= h0) & (cal["hour"] < h1)


def analyse(
    temp: dict[str, np.ndarray],
    setpoint: dict[str, np.ndarray],
    cal: dict[str, np.ndarray],
    winters: dict[str, tuple[int, int]],
    *,
    hot: float,
    comfort: float,
    floor: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (m0, m1) in winters.items():
        in_winter = (cal["month"] >= m0) & (cal["month"] <= m1)
        t_f, sp_f = temp[FOCUS_ROOM], setpoint[FOCUS_ROOM]
        windows: dict[str, Any] = {}
        for wname, (kind, h0, h1) in WINDOWS.items():
            sel = in_winter & window_mask(cal, kind, h0, h1)
            x = t_f[sel]
            entry: dict[str, Any] = {
                "slots": int(sel.sum()),
                "temperature": _stats(x),
                "temperature_coverage": _coverage(x),
                "setpoint_mean": _stats(sp_f[sel])["mean"],
                "setpoint_coverage": _coverage(sp_f[sel]),
            }
            if wname in DAY_WINDOWS:
                entry["share_at_or_above_hot"] = _share(x, x >= hot)
                entry["share_at_or_above_comfort"] = _share(x, x >= comfort)
            windows[wname] = entry
        rooms = {}
        for room in ROOMS:
            x = temp[room][in_winter]
            v = _finite(x)
            rooms[room] = {
                "share_below_floor": _share(x, x < floor),
                "p10": float(np.percentile(v, 10)) if v.size else None,
                "mean": float(v.mean()) if v.size else None,
                "coverage": _coverage(x),
            }
        out[name] = {"slots": int(in_winter.sum()), "windows": windows, "rooms": rooms}
    return out


def _fmt(v: float | None, spec: str = ".1f", scale: float = 1.0) -> str:
    return "n/a" if v is None else format(v * scale, spec)


def render_markdown(report: dict[str, Any]) -> str:
    p = report["parameters"]
    lines = [
        "# Comfort history",
        "",
        f"Feature set `{report['feature_set']}`, time zone {p['timezone']}, winters "
        f"{p['winters']}. Hot >= {p['hot_c']} degC, comfort >= {p['comfort_c']} degC, "
        f"floor < {p['floor_c']} degC. Statistics use slots with a reading; coverage in %.",
        "",
        f"## {FOCUS_ROOM} by window (temperature mean / median / p10 / p90, setpoint mean)",
        "",
        "| Winter | Window | Slots | Temperature | Setpoint | >= hot % | >= comfort % "
        "| Cov T/SP % |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, w in report["winters"].items():
        for wname, e in w["windows"].items():
            t = e["temperature"]
            lines.append(
                f"| {name} | {wname} | {e['slots']} | "
                f"{_fmt(t['mean'])} / {_fmt(t['median'])} / {_fmt(t['p10'])} / {_fmt(t['p90'])} | "
                f"{_fmt(e['setpoint_mean'])} | "
                f"{_fmt(e.get('share_at_or_above_hot'), '.1f', 100)} | "
                f"{_fmt(e.get('share_at_or_above_comfort'), '.1f', 100)} | "
                f"{_fmt(e['temperature_coverage'], '.0f', 100)} / "
                f"{_fmt(e['setpoint_coverage'], '.0f', 100)} |"
            )
    lines += [
        "",
        "## Rooms over the whole winter",
        "",
        "| Winter | Room | Below floor % | p10 | Mean | Coverage % |",
        "|---|---|---|---|---|---|",
    ]
    for name, w in report["winters"].items():
        for room, r in w["rooms"].items():
            lines.append(
                f"| {name} | {room} | {_fmt(r['share_below_floor'], '.1f', 100)} | "
                f"{_fmt(r['p10'])} | {_fmt(r['mean'])} | {_fmt(r['coverage'], '.0f', 100)} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--feature-set", required=True, help="climate feature set id (or prefix)")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--timezone", default="Europe/Madrid")
    parser.add_argument("--winters", default=DEFAULT_WINTERS)
    parser.add_argument("--hot", type=float, default=25.0, help="degC")
    parser.add_argument("--comfort", type=float, default=23.5, help="degC")
    parser.add_argument("--floor", type=float, default=21.0, help="degC")
    parser.add_argument(
        "--run-id", type=_run_id, default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    args = parser.parse_args(argv)

    root = resolve_data_dir(args.data_dir)
    base = root / "analysis" / "v1" / FEATURES_DIRNAME
    matches = sorted(p for p in base.glob(f"{args.feature_set}*") if p.is_dir())
    if len(matches) != 1:
        parser.error(f"feature set {args.feature_set!r} matches {len(matches)} under {base}")
    table, _ = load_features(matches[0])
    out_dir = root / "analysis" / "v1" / "comfort-history" / args.run_id

    def col(name: str) -> np.ndarray:
        return table.column(name).to_numpy(zero_copy_only=False).astype(float)

    temp = {r: col(f"room_{r}_temp_mean_c") for r in ROOMS}
    setpoint = {r: col(f"room_{r}_setpoint_mean_c") for r in ROOMS}
    starts_us = table.column("interval_start").cast(pa.int64()).to_numpy()
    cal = local_calendar(starts_us, ZoneInfo(args.timezone))
    result = analyse(
        temp,
        setpoint,
        cal,
        parse_winters(args.winters),
        hot=args.hot,
        comfort=args.comfort,
        floor=args.floor,
    )
    report = {
        "feature_set": matches[0].name,
        "parameters": {
            "timezone": args.timezone,
            "winters": args.winters,
            "hot_c": args.hot,
            "comfort_c": args.comfort,
            "floor_c": args.floor,
            "windows": {k: list(v) for k, v in WINDOWS.items()},
        },
        "winters": result,
    }
    write_part_atomic(out_dir / "report.json", json_bytes(report))
    write_part_atomic(out_dir / "report.md", render_markdown(report).encode("utf-8"))
    print(render_markdown(report))
    print(f"written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
