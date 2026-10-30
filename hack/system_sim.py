#!/usr/bin/env python3
"""Read-only whole-system replay: heat-pump regime x house battery, per heating season.

Answers, on historical data, what a decoupled design would look like: the heat pump runs for
comfort and efficiency (it ignores price) and the house battery does the price arbitrage.  It is a
2 x 2 grid per season (S2, S3 of ``config/climate.yaml``):

* ``A0`` legacy heat-pump control (cheapest-N hours), load as observed, no battery.
* ``A1`` the same load with a simulated battery.
* ``B0`` efficiency-first heat pump (longer, gentler runs, like the S3 days run with
  ``min_hours`` 15), no battery.
* ``B1`` the B load with a simulated battery.

Nothing is simulated thermally.  The B heat-pump load is an **empirical transfer** from the S3
days the legacy rule ran with 15 permitted hours: the daily heating electricity comes from their
kWh-versus-heating-degree-days line, the hourly shape from their mean normalised profile (by
outdoor band) and the comfort, starts and COP of a B day from the 15 h days in the same outdoor
band.  The heat-pump power is HeishaMon-measured; the dataset load is the whole-house meter, so the
B load is ``gross + k (E_B - E_A)`` with ``k`` the season's median ratio of the heat-pump branch
meter to HeishaMon.  The battery runs through the unchanged benchmark (``benchmark_dataset``) on a
copy of the dataset table whose ``gross_kwh`` column is replaced in memory; ``oracle`` is perfect
foresight, ``day_ahead`` and ``rolling_horizon`` the benchmark's causal forecasts.

Usage:
    uv run python hack/system_sim.py \\
        --feature-set <climate feature set id or prefix> \\
        --dataset ~/data/<store>/analysis/v1/<dataset_id> \\
        [--config config/climate.yaml] [--tariff config/tariff.yaml] [--data-dir ~/data/<store>] \\
        [--run-id <name>] [--seasons S2,S3] [--no-battery]

Everything is written to ``<data_dir>/analysis/v1/system-sim/<run-id>/`` (``report.json``,
``report.md`` and one benchmark run per scenario and season); the repository is never touched.
The report holds private figures (energy, money): keep it in the data dir.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import re
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "src"))

from battery_optimizer import benchmark  # noqa: E402
from battery_optimizer.climate.config import load_climate_config  # noqa: E402
from battery_optimizer.climate.data import ClimateData, load_climate_data  # noqa: E402
from battery_optimizer.features import json_bytes  # noqa: E402
from battery_optimizer.paths import resolve_data_dir, write_part_atomic  # noqa: E402

SLOT_H = 0.25
COMFORT_C = 22.0
HDD_BASE_C = 15.0
LEGACY_HOURS = (10.0, 11.0)  # S3 ran 10 h, S2 ran 11 h
EFFICIENT_HOURS = 15.0
#: Outdoor bands (daily mean, degC) for matching and the sanity check.
BANDS = (-np.inf, 3.0, 6.0, 9.0, 12.0, 15.0, np.inf)
#: Coarser bands for the hourly shape (more days per band).
SHAPE_SPLIT_C = 9.0
MIN_BAND_DAYS = 3
COLD_DAY_C = 8.0
#: Expensive slots: the top quarter of a local day's prices.
EXPENSIVE_QUANTILE = 0.75
FLOOR_TOL_KWH = 0.01
POLICIES = ("no_battery", "day_ahead", "rolling_horizon", "oracle")


# ---------------------------------------------------------------------------
# Per-day table
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Day:
    label: date
    season: str
    lo: int
    hi: int
    complete: bool
    replayed: bool
    hours: float | None
    outage: bool
    t_out: float | None
    e_heat: float
    e_heat_cov: float
    q_heat: float
    hp_total: float
    branch: float | None
    starts: float | None
    t_air_cov: float
    share_below: float | None
    kh_below: float | None

    @property
    def hdd(self) -> float:
        return max(0.0, HDD_BASE_C - self.t_out) if self.t_out is not None else float("nan")

    @property
    def band(self) -> int:
        return int(np.searchsorted(BANDS, self.t_out, side="right") - 1)

    @property
    def normal(self) -> bool:
        """A day usable for the empirical comparison: complete, replayed, a regular rule
        setting, thermostats and heat pump known, and outside the thermostat outage."""
        return (
            self.complete
            and self.replayed
            and self.hours in (*LEGACY_HOURS, EFFICIENT_HOURS)
            and not self.outage
            and self.t_out is not None
            and self.t_air_cov >= 0.9
            and self.e_heat_cov >= 0.95
            and self.starts is not None
        )


def _nansum(x: np.ndarray) -> float:
    return float(np.nansum(x))


def build_days(d: ClimateData, seasons: tuple[str, ...]) -> list[Day]:
    outage = d.drop["thermostat_outage"]
    starts = d.columns["compressor_starts_delta"]
    branch = d.columns["branch_aero_kwh"]
    hp = d.e_heat_kwh, d.columns["hp_dhw_kwh"], d.columns["hp_cool_kwh"]
    t_out = d.t_out_model
    out: list[Day] = []
    for j, (label, lo, hi) in enumerate(d.days):
        if d.day_season[j] not in seasons:
            continue
        n = hi - lo
        sl = slice(lo, hi)
        to = t_out[sl]
        ta = d.t_air[sl]
        known = np.isfinite(ta)
        st = starts[sl]
        br = branch[sl]
        out.append(
            Day(
                label=label,
                season=str(d.day_season[j]),
                lo=lo,
                hi=hi,
                complete=bool(d.day_complete[j]),
                replayed=bool(d.day_u_known[j]),
                hours=float(np.nansum(d.u[sl]) / 4) if d.day_u_known[j] else None,
                outage=bool(outage[lo]),
                t_out=float(np.nanmean(to)) if np.isfinite(to).sum() >= n / 2 else None,
                e_heat=_nansum(d.e_heat_kwh[sl]),
                e_heat_cov=float(np.isfinite(d.e_heat_kwh[sl]).mean()),
                q_heat=_nansum(d.q_heat_kwh[sl]),
                hp_total=sum(_nansum(x[sl]) for x in hp),
                branch=_nansum(br) if np.isfinite(br).mean() >= 0.95 else None,
                starts=_nansum(st) if np.isfinite(st).mean() >= 0.95 else None,
                t_air_cov=float(known.mean()),
                share_below=float((ta[known] < COMFORT_C).mean()) if known.any() else None,
                kh_below=(
                    float(np.maximum(0.0, COMFORT_C - ta[known]).mean() * 24)
                    if known.any()
                    else None
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Empirical comparison (sanity check) and the B transfer
# ---------------------------------------------------------------------------


def _summary(days: list[Day]) -> dict[str, Any]:
    if not days:
        return {"days": 0}
    e = sum(x.e_heat for x in days)
    return {
        "days": len(days),
        "t_out_mean_c": float(np.mean([x.t_out for x in days])),
        "share_below_22": float(np.mean([x.share_below for x in days])),
        "kh_below_per_day": float(np.mean([x.kh_below for x in days])),
        "heat_kwh_per_day": e / len(days),
        "heat_kwh_per_hdd": e / max(sum(x.hdd for x in days), 1e-9),
        "cop": sum(x.q_heat for x in days) / e if e > 0 else None,
        "starts_per_day": float(np.mean([x.starts for x in days])),
    }


def regime_comparison(days: list[Day]) -> dict[str, Any]:
    """S3 normal days, 10 h against 15 h: raw, and re-weighted onto a common outdoor mix."""
    s3 = [x for x in days if x.season == "S3" and x.normal]
    ten = [x for x in s3 if x.hours == 10.0]
    fif = [x for x in s3 if x.hours == EFFICIENT_HOURS]
    bands = []
    for b in range(len(BANDS) - 1):
        a, c = [x for x in ten if x.band == b], [x for x in fif if x.band == b]
        bands.append(
            {
                "band_c": [float(BANDS[b]), float(BANDS[b + 1])],
                "h10": _summary(a),
                "h15": _summary(c),
            }
        )
    # Stratified: weight each band by its pooled day count, only bands holding both regimes.
    keys = ("share_below_22", "kh_below_per_day", "heat_kwh_per_day", "starts_per_day")
    both = [r for r in bands if r["h10"]["days"] and r["h15"]["days"]]
    w = np.array([r["h10"]["days"] + r["h15"]["days"] for r in both], dtype=float)
    strat: dict[str, Any] = {"bands_used": len(both), "days_weighted": int(w.sum())}
    for regime in ("h10", "h15"):
        strat[regime] = {k: float(np.dot(w, [r[regime][k] for r in both]) / w.sum()) for k in keys}
        q = np.dot(w, [r[regime]["cop"] * r[regime]["heat_kwh_per_day"] for r in both])
        strat[regime]["cop"] = float(q / np.dot(w, [r[regime]["heat_kwh_per_day"] for r in both]))
    strat["heat_kwh_change"] = (
        strat["h15"]["heat_kwh_per_day"] / strat["h10"]["heat_kwh_per_day"] - 1
    )
    return {
        "raw": {"h10": _summary(ten), "h15": _summary(fif)},
        "raw_heat_kwh_change": _summary(fif)["heat_kwh_per_day"] / _summary(ten)["heat_kwh_per_day"]
        - 1,
        "stratified": strat,
        "by_band": bands,
    }


@dataclasses.dataclass
class Transfer:
    """The B regime as learnt from the S3 15 h days."""

    slope: float
    intercept: float
    r2: float
    n_days: int
    #: Local-hour fractions of daily heating electricity, keyed by "cold"/"mild".
    shape: dict[str, np.ndarray]
    #: Per outdoor band: share below, Kh/day, starts/day, COP (from the 15 h days).
    band: dict[int, dict[str, float]]

    def daily_kwh(self, hdd: float) -> float:
        return max(0.0, self.intercept + self.slope * hdd)

    def band_stats(self, b: int) -> dict[str, float]:
        """The band's 15 h statistics, or the nearest band that has enough days."""
        have = sorted(self.band, key=lambda k: (abs(k - b), k))
        return self.band[have[0]]


def fit_transfer(d: ClimateData, days: list[Day]) -> Transfer:
    fif = [x for x in days if x.season == "S3" and x.normal and x.hours == EFFICIENT_HOURS]
    hdd = np.array([x.hdd for x in fif])
    e = np.array([x.e_heat for x in fif])
    slope, intercept = np.polyfit(hdd, e, 1)
    pred = intercept + slope * hdd
    r2 = 1 - float(((e - pred) ** 2).sum() / ((e - e.mean()) ** 2).sum())
    shape: dict[str, np.ndarray] = {}
    for name, sel in (
        ("cold", [x for x in fif if x.t_out < SHAPE_SPLIT_C]),
        ("mild", [x for x in fif if x.t_out >= SHAPE_SPLIT_C]),
    ):
        hourly = np.zeros(24)
        for x in sel:
            h = d.local_hour[x.lo : x.hi]
            np.add.at(hourly, h, np.nan_to_num(d.e_heat_kwh[x.lo : x.hi]))
        shape[name] = hourly / hourly.sum()
    band: dict[int, dict[str, float]] = {}
    for b in range(len(BANDS) - 1):
        sel = [x for x in fif if x.band == b]
        if len(sel) >= MIN_BAND_DAYS:
            s = _summary(sel)
            band[b] = {
                "days": s["days"],
                "share_below_22": s["share_below_22"],
                "kh_below_per_day": s["kh_below_per_day"],
                "starts_per_day": s["starts_per_day"],
                "cop": s["cop"],
            }
    return Transfer(float(slope), float(intercept), r2, len(fif), shape, band)


def meter_factor(days: list[Day], season: str) -> dict[str, Any]:
    """Median daily ratio of the heat-pump branch meter to HeishaMon on heating days."""
    r = np.array(
        [
            x.branch / x.hp_total
            for x in days
            if x.season == season and x.normal and x.branch is not None and x.hp_total >= 2.0
        ]
    )
    return {
        "k": float(np.median(r)),
        "days": int(r.size),
        "p25": float(np.percentile(r, 25)),
        "p75": float(np.percentile(r, 75)),
    }


def b_daily_kwh(x: Day, variant: str, tr: Transfer, ratio: float) -> float:
    """Daily B heating electricity (HeishaMon basis).

    ``B``: the 15 h days' kWh-versus-HDD line (a level transfer, carries S3's plant state).
    ``Bs``: the day's observed heating scaled by the S3 15 h / 10 h stratified ratio (a relative
    transfer that keeps the season's own plant state); days already run at 15 h stay as observed.
    """
    if variant == "B":
        return tr.daily_kwh(x.hdd)
    return x.e_heat * (1.0 if x.hours == EFFICIENT_HOURS else ratio)


def b_heat_profile(d: ClimateData, x: Day, tr: Transfer, daily_kwh: float) -> np.ndarray:
    """Synthetic B heating electricity (HeishaMon basis) on the slots of day *x*."""
    hours = d.local_hour[x.lo : x.hi]
    frac = tr.shape["cold" if x.t_out < SHAPE_SPLIT_C else "mild"][hours]
    per_hour = np.bincount(hours, minlength=24)[hours]  # 4, or 8 on the fall-back hour
    return daily_kwh * frac / per_hour


def substituted(day: Day) -> bool:
    """Days on which the B regime replaces the observed heating: the legacy rule ran (it is
    off outside the heating period) and the HeishaMon heating counter is known."""
    return day.complete and day.replayed and day.t_out is not None and day.e_heat_cov >= 0.95


# ---------------------------------------------------------------------------
# Battery: the unchanged benchmark on an in-memory load
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _dataset_override(loaded: benchmark.LoadedDataset):
    original = benchmark.load_dataset
    benchmark.load_dataset = lambda *_a, **_k: loaded
    try:
        yield
    finally:
        benchmark.load_dataset = original


def run_battery(
    loaded: benchmark.LoadedDataset,
    gross: np.ndarray,
    out_dir: Path,
    *,
    tariff: str,
    start: datetime,
    end: datetime,
    oracle_time_limit_s: float,
) -> Path:
    arr = pa.array([None if not np.isfinite(v) else float(v) for v in gross], pa.float64())
    idx = loaded.table.column_names.index("gross_kwh")
    table = loaded.table.set_column(idx, "gross_kwh", arr)
    patched = dataclasses.replace(loaded, table=table, directory=out_dir)
    with _dataset_override(patched):
        return benchmark.benchmark_dataset(
            dataset=out_dir,
            config_path=tariff,
            train_end=start,
            eval_end=end,
            policies=",".join(POLICIES),
            time_limit_s=oracle_time_limit_s,
        )


def soc_floor_kwh(tariff: str) -> float:
    """The SOC every solver rung may not go below: ``max(soc_min, initial)`` of nameplate."""
    from battery_optimizer.config import load_config

    a = benchmark.resolve_battery_assumptions(load_config(Path(tariff)))
    return a.capacity_kwh * max(a.soc_min_frac, a.initial_soc_frac)


def battery_kpis(run_dir: Path, cold_days: set[date], floor: float) -> dict[str, Any]:
    res = pq.read_table(run_dir / benchmark.RESULTS_PARQUET_NAME).to_pylist()
    sched = pq.read_table(
        run_dir / benchmark.SCHEDULES_PARQUET_NAME,
        columns=["policy", "local_day", "usable", "price_eur_per_kwh", "soc_kwh", "grid_kwh"],
    )
    pol = np.array(sched.column("policy").to_pylist())
    day = np.array(sched.column("local_day").to_pylist())
    usable = np.array(sched.column("usable").to_pylist(), dtype=bool)
    price = np.array(sched.column("price_eur_per_kwh").to_pylist(), dtype=float)
    soc = np.array(sched.column("soc_kwh").to_pylist(), dtype=float)
    grid = np.array(sched.column("grid_kwh").to_pylist(), dtype=float)
    first = pol == pol[0]
    # Expensive slots of each cold local day (same for every policy: one price series).
    expensive = np.zeros(int(first.sum()), dtype=bool)
    d0, p0, u0 = day[first], price[first], usable[first]
    for label in set(d0) & cold_days:
        sel = (d0 == label) & u0
        if sel.any():
            expensive |= sel & (p0 >= np.quantile(p0[sel], EXPENSIVE_QUANTILE))
    out: dict[str, Any] = {}
    for row in res:
        name = row["policy"]
        m = pol == name
        s, g = soc[m], grid[m]
        hit = expensive & (s <= floor + FLOOR_TOL_KWH) & (g > 1e-6)
        out[name] = {
            "cost_eur": row["cost_eur"],
            "efc": row["efc"],
            "throughput_kwh": row["throughput_kwh"],
            "solver_status": row["solver_status"],
            "is_certified": row["is_certified"],
            "import_kwh": float(np.nansum(g[usable[m]])),
            "expensive_cold_slots": int(expensive.sum()),
            # no_battery sits at the initial SOC (= the floor) throughout: not applicable
            "floor_hit_share": (
                float(hit.sum() / expensive.sum())
                if expensive.any() and name != "no_battery"
                else None
            ),
        }
    return out


# ---------------------------------------------------------------------------
# Scenario KPIs and the report
# ---------------------------------------------------------------------------


def hp_kpis(
    days: list[Day], season: str, regime: str, tr: Transfer, k: float, ratio: float
) -> dict[str, Any]:
    """Heat-pump and comfort KPIs over the substituted days of *season*."""
    sel = [x for x in days if x.season == season and substituted(x)]
    hdd = sum(x.hdd for x in sel)
    if regime == "A":
        e = sum(x.e_heat for x in sel)
        q = sum(x.q_heat for x in sel)
        comfort = [x for x in sel if x.share_below is not None and x.t_air_cov >= 0.5]
        share = float(np.mean([x.share_below for x in comfort]))
        kh = float(np.mean([x.kh_below for x in comfort]))
        st = [x.starts for x in sel if x.starts is not None]
        starts = float(np.mean(st))
    else:
        e_day = [b_daily_kwh(x, regime, tr, ratio) for x in sel]
        e = sum(e_day)
        stats = [tr.band_stats(x.band) for x in sel]
        q = sum(ed * s["cop"] for ed, s in zip(e_day, stats, strict=True))
        share = float(np.mean([s["share_below_22"] for s in stats]))
        kh = float(np.mean([s["kh_below_per_day"] for s in stats]))
        starts = float(np.mean([s["starts_per_day"] for s in stats]))
    return {
        "days": len(sel),
        "hdd": hdd,
        "share_below_22": share,
        "kh_below_per_day": kh,
        "heat_kwh_heishamon": e,
        "heat_kwh_meter": e * k,
        "heat_kwh_per_hdd": e / hdd if hdd else None,
        "cop": q / e if e else None,
        "starts_per_day": starts,
    }


def savings(sc: dict[str, Any]) -> dict[str, Any]:
    """Savings against A0 per battery policy; the battery's part of X1 is X0 - X1."""
    a = sc["A"]["cost"]
    out: dict[str, Any] = {"A0_cost": a["no_battery"]}
    for regime in ("B", "Bs"):
        out[f"{regime}0_minus_A0"] = sc[regime]["cost"]["no_battery"] - a["no_battery"]
    for pol in ("perfect_foresight", "rolling_horizon", "day_ahead"):
        row: dict[str, Any] = {"A1_saving": a["no_battery"] - a[pol]}
        for regime in ("B", "Bs"):
            b = sc[regime]["cost"]
            row[f"{regime}1_saving"] = a["no_battery"] - b[pol]
            row[f"{regime}1_battery_part"] = b["no_battery"] - b[pol]
        out[pol] = row
    return out


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1000 else f"{x:.0f}"
    return str(x)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# System replay: heat-pump regime x battery (private: real installation figures)",
        "",
        "Empirical-transfer estimates, not simulations of room temperature. See report.json for "
        "every input and per-band figure.",
        "",
        f"- inputs: `{json.dumps(report['identity'])}`",
        "",
        "## Sanity check: S3 normal days, 10 h vs 15 h",
        "",
        "| | days | T_out | below 22 | Kh/day | kWh/day | kWh/HDD | COP | starts/day |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    rc = report["sanity_check"]
    for name in ("h10", "h15"):
        s = rc["raw"][name]
        lines.append(
            f"| raw {name} | {s['days']} | {_fmt(s['t_out_mean_c'])} | {_fmt(s['share_below_22'])} "
            f"| {_fmt(s['kh_below_per_day'])} | {_fmt(s['heat_kwh_per_day'])} "
            f"| {_fmt(s['heat_kwh_per_hdd'])} | {_fmt(s['cop'])} | {_fmt(s['starts_per_day'])} |"
        )
    for name in ("h10", "h15"):
        s = rc["stratified"][name]
        lines.append(
            f"| stratified {name} | {rc['stratified']['days_weighted']} | - "
            f"| {_fmt(s['share_below_22'])} | {_fmt(s['kh_below_per_day'])} "
            f"| {_fmt(s['heat_kwh_per_day'])} | - | {_fmt(s['cop'])} "
            f"| {_fmt(s['starts_per_day'])} |"
        )
    lines += [
        "",
        f"Heat electricity change 15 h vs 10 h: raw {_fmt(rc['raw_heat_kwh_change'])}, "
        f"stratified {_fmt(rc['stratified']['heat_kwh_change'])}.",
        "",
        "## Scenarios",
        "",
        "| season | scen | below 22 | Kh/day | HP kWh (meter) | kWh/HDD | COP | starts/d "
        "| cost pf | cost rh | cost da | EFC rh | floor-hit rh |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for season, sc in report["scenarios"].items():
        for regime, s in sc.items():
            name = f"{regime}0/{regime}1"
            hp = s["hp"]
            b = s.get("cost", {})
            lines.append(
                f"| {season} | {name} | {_fmt(hp['share_below_22'])} "
                f"| {_fmt(hp['kh_below_per_day'])} | {_fmt(hp['heat_kwh_meter'])} "
                f"| {_fmt(hp['heat_kwh_per_hdd'])} | {_fmt(hp['cop'])} "
                f"| {_fmt(hp['starts_per_day'])} | {_fmt(b.get('perfect_foresight'))} "
                f"| {_fmt(b.get('rolling_horizon'))} | {_fmt(b.get('day_ahead'))} "
                f"| {_fmt(b.get('efc_rolling_horizon'))} "
                f"| {_fmt(b.get('floor_hit_share_rolling_horizon'))} |"
            )
    lines += ["", "## Savings against A0", ""]
    for season, sv in report.get("savings", {}).items():
        lines.append(f"- {season}: `{json.dumps(sv)}`")
    lines += ["", "## Assumptions", ""] + [f"- {a}" for a in report["assumptions"]]
    return "\n".join(lines) + "\n"


def _season_window(cfg, season: str) -> tuple[datetime, datetime]:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(cfg.timezone)
    s = cfg.seasons[season]
    start = datetime.combine(s.start, time(0), tz).astimezone(UTC)
    end = datetime.combine(s.end + timedelta(days=1), time(0), tz).astimezone(UTC)
    return start, end


def _run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value) or ".." in value:
        raise argparse.ArgumentTypeError(f"{value!r}: use only [A-Za-z0-9._-] and no '..'")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--feature-set", required=True, help="climate feature set id (or prefix)")
    parser.add_argument("--dataset", required=True, help="canonical dataset directory")
    parser.add_argument("--config", default="config/climate.yaml")
    parser.add_argument("--tariff", default="config/tariff.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seasons", default="S2,S3")
    parser.add_argument(
        "--run-id", type=_run_id, default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    parser.add_argument("--oracle-time-limit-s", type=float, default=600.0)
    parser.add_argument("--no-battery", action="store_true", help="skip the benchmark runs")
    args = parser.parse_args(argv)

    root = resolve_data_dir(args.data_dir)
    out_dir = root / "analysis" / "v1" / "system-sim" / args.run_id
    seasons = tuple(args.seasons.split(","))
    cfg = load_climate_config(args.config)
    d = load_climate_data(cfg, feature_set=args.feature_set, dataset=args.dataset, data_dir=root)
    days = build_days(d, tuple({*seasons, "S3"}))
    sanity = regime_comparison(days)
    tr = fit_transfer(d, days)

    loaded = benchmark.load_dataset(args.dataset, data_dir=root)
    ds_start = loaded.table.column("interval_start").cast(pa.int64()).to_numpy()
    gross_a = np.array(
        [np.nan if v is None else v for v in loaded.table.column("gross_kwh").to_pylist()]
    )
    pos = np.searchsorted(ds_start, d.starts_us)
    on_grid = (pos < ds_start.size) & (ds_start[np.minimum(pos, ds_start.size - 1)] == d.starts_us)

    report: dict[str, Any] = {
        "identity": {
            "feature_set_id": d.feature_set_id,
            "dataset_id": loaded.dataset_id,
            "climate_config_sha256": d.config_sha256,
            "code_version": benchmark.code_version(),
        },
        "sanity_check": sanity,
        "transfer": {
            "daily_kwh_line": {
                "intercept": tr.intercept,
                "slope_per_hdd": tr.slope,
                "r2": tr.r2,
                "days": tr.n_days,
            },
            "shape": {k: v.tolist() for k, v in tr.shape.items()},
            "band_stats": {f"{BANDS[b]}..{BANDS[b + 1]}": v for b, v in tr.band.items()},
        },
        "meter_factor": {},
        "scenarios": {},
        "savings": {},
    }
    ratio = 1.0 + sanity["stratified"]["heat_kwh_change"]
    floor = soc_floor_kwh(args.tariff)
    for season in seasons:
        mf = meter_factor(days, season)
        k = mf["k"]
        report["meter_factor"][season] = mf
        sub = [x for x in days if x.season == season and substituted(x)]
        loads = {"A": gross_a}
        clipped: dict[str, int] = {}
        for variant in ("B", "Bs"):
            gross = gross_a.copy()
            clipped[variant] = 0
            for x in sub:
                e_b = b_heat_profile(d, x, tr, b_daily_kwh(x, variant, tr, ratio))
                delta = k * (e_b - np.nan_to_num(d.e_heat_kwh[x.lo : x.hi]))
                ok = on_grid[x.lo : x.hi]
                rows = pos[x.lo : x.hi][ok]
                new = gross_a[rows] + delta[ok]
                clipped[variant] += int((new < 0).sum())
                gross[rows] = np.where(np.isfinite(gross_a[rows]), np.maximum(new, 0.0), np.nan)
            loads[variant] = gross
        cold = {
            x.label
            for x in days
            if x.season == season and x.t_out is not None and x.t_out < COLD_DAY_C
        }
        start, end = _season_window(cfg, season)
        report["scenarios"][season] = {}
        for regime, gross in loads.items():
            hp = hp_kpis(days, season, regime, tr, k, ratio)
            if regime != "A":
                hp["slots_clipped_to_zero"] = clipped[regime]
            hp["days_not_substituted"] = sum(
                1 for x in days if x.season == season and not substituted(x)
            )
            entry: dict[str, Any] = {"hp": hp}
            if not args.no_battery:
                run_dir = run_battery(
                    loaded,
                    gross,
                    out_dir / f"{regime}-{season}",
                    tariff=args.tariff,
                    start=start,
                    end=end,
                    oracle_time_limit_s=args.oracle_time_limit_s,
                )
                kp = battery_kpis(run_dir, cold, floor)
                entry["battery"] = kp
                entry["run_dir"] = str(run_dir.relative_to(out_dir))
                entry["cost"] = {
                    "no_battery": kp["no_battery"]["cost_eur"],
                    "perfect_foresight": kp["oracle"]["cost_eur"],
                    "rolling_horizon": kp["rolling_horizon"]["cost_eur"],
                    "day_ahead": kp["day_ahead"]["cost_eur"],
                    "efc_rolling_horizon": kp["rolling_horizon"]["efc"],
                    "floor_hit_share_rolling_horizon": kp["rolling_horizon"]["floor_hit_share"],
                }
            report["scenarios"][season][regime] = entry
        if not args.no_battery:
            report["savings"][season] = savings(report["scenarios"][season])
    report["assumptions"] = [
        "B is an empirical transfer from the S3 days the legacy rule ran with 15 permitted hours, "
        "not a room-temperature simulation; 10 h and 15 h days are also months apart (confounded).",
        f"B daily heating kWh = max(0, a + b HDD{HDD_BASE_C:g}) from the 15 h days; hourly shape = "
        f"their kWh-weighted local-hour profile, split at {SHAPE_SPLIT_C:g} C daily mean outdoor.",
        "B comfort, starts and COP = the mean of the 15 h days in the same outdoor band "
        f"(bands {list(BANDS[1:-1])} C, nearest band with >= {MIN_BAND_DAYS} days).",
        "Bs (sensitivity): the day's observed heating x the S3 stratified 15 h / 10 h ratio "
        "(days already at 15 h unchanged), same shape and comfort as B; it keeps each season's "
        "own plant state, which the level transfer B does not.",
        "A is the load as observed: S2 ran 11 h; S3 ran 10 h in Nov-Dec and 15 h from Jan.",
        "B replaces heating only on complete days with a replayed legacy schedule (the rule was "
        "on); other days, DHW and the rest of the house stay as observed.",
        "Meter basis: B load = gross + k (E_B - E_A), k = median daily heat-pump branch meter / "
        "HeishaMon (heat+DHW+cool) on normal days of the season; the branch also carries some "
        "appliances, so k is an upper-leaning estimate.",
        "Battery: the unchanged benchmark (config/tariff.yaml battery block, terminal mode free, "
        "default forecasts); oracle = perfect foresight, rolling_horizon and day_ahead = causal "
        "trailing-mean load forecast with published prices. Costs are pre-tax PVPC over usable "
        "intervals of the season window (Oct-Apr).",
        f"Floor hit: share of expensive slots (top {100 * (1 - EXPENSIVE_QUANTILE):g}% of the "
        f"day's prices) on cold days (mean T_out < {COLD_DAY_C:g} C) that end at the SOC floor "
        "while importing.",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_part_atomic(out_dir / "report.json", json_bytes(report))
    write_part_atomic(out_dir / "report.md", render_markdown(report).encode("utf-8"))
    print(f"system_sim: report written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
