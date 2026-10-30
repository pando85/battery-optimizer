#!/usr/bin/env python3
"""Throughput-cost sweep on the realistic rolling-horizon policy (and the oracle).

``hack/economic_sensitivity.py`` sweeps the oracle only.  The live planner is a rolling horizon
that re-solves with a causal load forecast, so the wear cost must be chosen against *that*
policy.  This runner calls the unchanged benchmark once per ``(window, throughput cost)`` pair
with ``policies=no_battery,rolling_horizon,oracle`` and tabulates relative figures.

Convention.  ``--throughput-cost-eur-per-kwh`` is a planning-only term, EUR per AC kWh **per
direction**: the objective is ``sum((price + tp) * charge - (price - tp) * discharge)``, so a kWh
charged and a kWh discharged each cost ``tp`` (one cycle of E kWh in and 0.8 E out costs
``1.8 tp E``).  It never enters the reported cost.  The saving columns are therefore the true
meter-bill saving of a plan made with that wear price.  It is a convention, not a fitted aging
model.

Usage::

    uv run python hack/throughput_sweep.py \\
        --dataset ~/data/<store>/analysis/v1/<dataset_id> --data-dir ~/data/<store> \\
        --run-id <fresh-name> [--costs 0,0.005,0.01,0.015,0.02,0.03,0.05] \\
        [--windows S2:2024-10-01:2025-05-01,S3:2025-10-01:2026-05-01,SU25:2025-05-01:2025-10-01] \\
        [--workers 8] [--time-limit-s 600]

Windows are ``name:first-local-day:day-after-last``.  The benchmark runs are written under
``<data-dir>/analysis/v1/throughput-sweep/<run-id>/`` (the dataset directory is not touched) and
``table.json`` / ``table.md`` next to them.  Use a fresh run id.  Both outputs hold private euro
figures only in the benchmark artifacts; the table has relative figures, cycles per day and
throughput per day, which are the only things to copy into the repository.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "src"))

from battery_optimizer import benchmark  # noqa: E402
from battery_optimizer.config import load_config  # noqa: E402
from battery_optimizer.features import json_bytes  # noqa: E402
from battery_optimizer.paths import resolve_data_dir, write_part_atomic  # noqa: E402

DEFAULT_COSTS = "0,0.005,0.01,0.015,0.02,0.03,0.05"
DEFAULT_WINDOWS = "S2:2024-10-01:2025-05-01,S3:2025-10-01:2026-05-01,SU25:2025-05-01:2025-10-01"
POLICIES = "no_battery,rolling_horizon,oracle"
#: A saving that falls by more than this share of the cost-0 saving is past the knee.
KNEE_LOSS = 0.05


@contextlib.contextmanager
def _dataset_directory(loaded: benchmark.LoadedDataset, directory: Path):
    """Write the run next to the sweep, not into the dataset directory."""
    original = benchmark.load_dataset
    patched = dataclasses.replace(loaded, directory=directory)
    benchmark.load_dataset = lambda *_a, **_k: patched
    try:
        yield
    finally:
        benchmark.load_dataset = original


def _bounds(spec: str, tz: ZoneInfo) -> tuple[str, datetime, datetime]:
    name, first, after = spec.split(":")
    start = datetime.combine(date.fromisoformat(first), time(0), tz).astimezone(UTC)
    end = datetime.combine(date.fromisoformat(after), time(0), tz).astimezone(UTC)
    return name, start, end


def _run_one(task: dict) -> dict:
    root = Path(task["data_dir"])
    loaded = benchmark.load_dataset(task["dataset"], data_dir=root)
    out = Path(task["out_dir"])
    with _dataset_directory(loaded, out):
        run_dir = benchmark.benchmark_dataset(
            dataset=out,
            config_path=task["tariff"],
            train_end=datetime.fromisoformat(task["start"]),
            eval_end=datetime.fromisoformat(task["end"]),
            policies=POLICIES,
            time_limit_s=task["time_limit_s"],
            throughput_cost_eur_per_kwh=task["cost"],
            data_dir=root,
        )
    rows = {
        r["policy"]: r for r in pq.read_table(run_dir / benchmark.RESULTS_PARQUET_NAME).to_pylist()
    }
    return {
        "window": task["window"],
        "cost": task["cost"],
        "run_dir": str(run_dir.relative_to(root)),
        "policies": {
            p: {
                k: rows[p][k]
                for k in ("cost_eur", "efc", "throughput_kwh", "solver_status", "is_certified")
            }
            for p in rows
        },
    }


def tabulate(results: list[dict], days: dict[str, float]) -> dict:
    """Relative figures per window and policy, plus the knee of each rolling-horizon curve."""
    out: dict = {}
    for w in sorted({r["window"] for r in results}):
        rs = sorted((r for r in results if r["window"] == w), key=lambda r: r["cost"])
        base = rs[0]
        assert base["cost"] == 0.0, "the sweep must include cost 0"
        tbl: dict = {"days": days[w], "policies": {}}
        for pol in ("rolling_horizon", "oracle"):
            rows = []
            s0 = None
            for r in rs:
                nb = r["policies"]["no_battery"]["cost_eur"]
                p = r["policies"][pol]
                saving = (nb - p["cost_eur"]) / nb
                s0 = saving if s0 is None else s0
                rows.append(
                    {
                        "cost_eur_per_kwh": r["cost"],
                        "saving_rel": saving,
                        "saving_lost_vs_0": 1.0 - saving / s0,
                        "efc_per_day": p["efc"] / days[w],
                        "throughput_kwh_per_day": p["throughput_kwh"] / days[w],
                        "status": p["solver_status"],
                        "certified": p["is_certified"],
                    }
                )
            tbl["policies"][pol] = rows
        out[w] = tbl
    return out


def knee(rows: list[dict]) -> float:
    """Largest cost whose saving loss against cost 0 is still within :data:`KNEE_LOSS`."""
    ok = [r["cost_eur_per_kwh"] for r in rows if r["saving_lost_vs_0"] <= KNEE_LOSS]
    return max(ok)


def render(table: dict) -> str:
    lines = ["# Throughput-cost sweep (relative figures)", ""]
    for w, t in table.items():
        for pol, rows in t["policies"].items():
            lines += [
                f"## {w}, {pol} ({t['days']:.0f} days; knee {knee(rows):g})",
                "",
                "| tp EUR/kWh | saving vs no_battery | lost vs tp 0 | EFC/day | throughput kWh/day "
                "| status |",
                "|---|---|---|---|---|---|",
            ]
            for r in rows:
                lines.append(
                    f"| {r['cost_eur_per_kwh']:g} | {100 * r['saving_rel']:.2f}% "
                    f"| {100 * r['saving_lost_vs_0']:.1f}% | {r['efc_per_day']:.3f} "
                    f"| {r['throughput_kwh_per_day']:.2f} | {r['status']} |"
                )
            lines.append("")
    return "\n".join(lines)


def _run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value) or ".." in value:
        raise argparse.ArgumentTypeError(f"{value!r}: use only [A-Za-z0-9._-] and no '..'")
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tariff", default="config/tariff.yaml")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--run-id", type=_run_id, required=True)
    ap.add_argument("--costs", default=DEFAULT_COSTS)
    ap.add_argument("--windows", default=DEFAULT_WINDOWS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--time-limit-s", type=float, default=600.0)
    args = ap.parse_args(argv)

    root = resolve_data_dir(args.data_dir)
    out_dir = root / "analysis" / "v1" / "throughput-sweep" / args.run_id
    if out_dir.exists():
        ap.error(f"{out_dir} exists: use a fresh --run-id")
    costs = [float(c) for c in args.costs.split(",")]
    if 0.0 not in costs:
        ap.error("--costs must include 0")
    tz = ZoneInfo(load_config(Path(args.tariff)).tariff_periods.timezone)
    windows = [_bounds(s, tz) for s in args.windows.split(",")]
    days = {n: (e - s).total_seconds() / 86400.0 for n, s, e in windows}
    tasks = [
        {
            "dataset": args.dataset,
            "data_dir": str(root),
            "tariff": args.tariff,
            "out_dir": str(out_dir / f"{n}-tp{c:g}"),
            "window": n,
            "cost": c,
            "start": s.isoformat(),
            "end": e.isoformat(),
            "time_limit_s": args.time_limit_s,
        }
        for n, s, e in windows
        for c in costs
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_run_one, tasks))
    table = tabulate(results, days)
    report = {
        "dataset": args.dataset,
        "code_version": benchmark.code_version(),
        "costs": costs,
        "windows": args.windows,
        "knee_rule": f"largest cost with saving lost vs cost 0 <= {KNEE_LOSS:.0%}",
        "runs": results,
        "table": table,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_part_atomic(out_dir / "table.json", json_bytes(report))
    write_part_atomic(out_dir / "table.md", render(table).encode("utf-8"))
    print(f"throughput_sweep: written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
