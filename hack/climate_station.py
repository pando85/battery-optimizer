#!/usr/bin/env python3
"""Fetch, compare and evaluate the AEMET reference station for the outdoor temperature (C2).

Source credit: "Fuente: AEMET (via Ogimet)".  Three subcommands, all writing only under the data
dir (``<data_dir>/climate/station/`` for the raw store, ``<data_dir>/analysis/v1/climate-station/``
for reports); the repository is never touched and no site coordinate is ever printed.

``fetch``     download the SYNOP months of candidate stations once each and store them immutably
              (monthly requests at least 5 s apart, retry with backoff).
``compare``   per candidate station against the house sensor on *trusted* slots (compressor off
              long enough): coverage, bias by local hour and season, RMSE, correlation of 15-minute
              and hourly changes and lag, for all trusted slots and for trusted dark ones.
``evaluate``  the fused series of the site file's ``reference_station``: fitted parameters, the
              leave-block-out test against ``t_out_debiased``'s interpolation on held-out trusted
              readings (the switch rule), the residual compressor-start step and coverage.

Usage:
    uv run python hack/climate_station.py fetch --stations A B C \\
        [--from 2023-11-01 --to 2026-05-01] [--data-dir ~/data/<store>]
    uv run python hack/climate_station.py compare --stations A B C --feature-set <id or prefix> \\
        --dataset <canonical dataset dir> [--data-dir ~/data/<store>] [--name <suffix>]
    uv run python hack/climate_station.py evaluate --feature-set <id> --dataset <dir> \\
        [--station <WMO id>] [--data-dir ~/data/<store>] [--name <suffix>]

Reports hold private figures: they stay in the data dir.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "src"))

from battery_optimizer.climate.config import load_climate_config  # noqa: E402
from battery_optimizer.climate.data import (  # noqa: E402
    SITE_FILE,
    ClimateData,
    load_climate_data,
    load_site,
    station_months,
)
from battery_optimizer.climate.station import (  # noqa: E402
    ATTRIBUTION,
    StationError,
    fetch_station,
    load_station,
    month_range,
    reference_at_slots,
)
from battery_optimizer.climate.tout import compressor_start_step, leave_block_out  # noqa: E402
from battery_optimizer.paths import resolve_data_dir  # noqa: E402

SLOT_US = 900 * 1_000_000
SEASONS = {"winter": (12, 1, 2), "spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11)}
BLOCK_HOURS = (1, 2, 3, 4, 6)
#: Beyond the switch rule (1-6 h): where `t_out_debiased` is unknown or the gap is long; the
#: interpolation is then not capped, so the comparison is against the uncapped line.
EXTENDED_BLOCK_HOURS = (8, 12, 24)
PHASE_STEP_SLOTS = 12  # block phases every 3 h, one block per phase per day
REPORT_DIR = Path("analysis") / "v1" / "climate-station"


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _f(x: float | np.floating | None, nd: int = 3) -> float | None:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 10 or a.std() == 0 or b.std() == 0:
        return None
    return _f(np.corrcoef(a, b)[0, 1])


def _local_month(d: ClimateData) -> np.ndarray:
    return np.array([day[0].month for day in d.days], dtype=np.int64)[d.day_of_slot]


def _season_of(month: np.ndarray) -> np.ndarray:
    out = np.empty(month.size, dtype=object)
    for name, months in SEASONS.items():
        out[np.isin(month, months)] = name
    return out


def _diff_stats(diff: np.ndarray, hour: np.ndarray, season: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(diff)
    n = int(ok.sum())
    if n == 0:
        return {"n": 0}
    d = diff[ok]
    return {
        "n": n,
        "bias_k": _f(d.mean()),
        "rmse_k": _f(np.sqrt(np.mean(d**2))),
        "rmse_after_bias_k": _f(d.std()),
        "bias_by_local_hour_k": [
            _f(diff[ok & (hour == h)].mean()) if (ok & (hour == h)).any() else None
            for h in range(24)
        ],
        "bias_by_season_k": {
            s: {"n": int((ok & (season == s)).sum()), "bias_k": _f(diff[ok & (season == s)].mean())}
            for s in SEASONS
            if (ok & (season == s)).any()
        },
    }


def _change_corr(house: np.ndarray, ref: np.ndarray, ok: np.ndarray, k: int) -> dict[str, Any]:
    """Correlation of the *k*-slot changes of the house sensor and the reference."""
    both = ok[k:] & ok[:-k]
    a = (house[k:] - house[:-k])[both]
    b = (ref[k:] - ref[:-k])[both]
    return {"n": int(both.sum()), "corr": _corr(a, b)}


def _lag(house: np.ndarray, ref: np.ndarray, ok: np.ndarray, max_slots: int = 12) -> dict[str, Any]:
    """Shift of the reference (15-minute slots) that best matches the house sensor's hourly
    changes: positive means the house sensor lags the station."""
    k = 4
    best: tuple[float, int] | None = None
    curve: dict[int, float | None] = {}
    for lag in range(-max_slots, max_slots + 1):
        n = house.size
        t = np.arange(max(k + abs(lag), 0), n - abs(lag))
        h_dt = house[t] - house[t - k]
        r_dt = ref[t - lag] - ref[t - lag - k]
        good = ok[t] & ok[t - k] & ok[t - lag] & ok[t - lag - k]
        c = _corr(h_dt[good], r_dt[good])
        curve[lag] = c
        if c is not None and (best is None or c > best[0]):
            best = (c, lag)
    return {
        "best_lag_slots": None if best is None else best[1],
        "best_lag_minutes": None if best is None else best[1] * 15,
        "corr_at_best": None if best is None else _f(best[0]),
        "corr_at_zero": curve.get(0),
    }


def compare_one(d: ClimateData, ref: np.ndarray, dark_max: float) -> dict[str, Any]:
    trusted = d.t_out_valid & np.isfinite(d.t_out)
    dark = trusted & (d.solar < dark_max)
    hour = d.local_hour
    season = _season_of(_local_month(d))
    out: dict[str, Any] = {
        "coverage": {
            "grid_slots_with_reference": _f(np.isfinite(ref).mean()),
            "trusted_with_reference": _f(np.isfinite(ref[trusted]).mean())
            if trusted.any()
            else None,
            "dark_with_reference": _f(np.isfinite(ref[dark]).mean()) if dark.any() else None,
        },
    }
    for name, subset in (("trusted_all", trusted), ("trusted_dark", dark)):
        ok = subset & np.isfinite(ref)
        diff = np.where(ok, d.t_out - ref, np.nan)
        stats = _diff_stats(diff, hour, season)
        stats["change_15min"] = _change_corr(d.t_out, ref, ok, 1)
        stats["change_1h"] = _change_corr(d.t_out, ref, ok, 4)
        stats["lag"] = _lag(d.t_out, ref, ok)
        out[name] = stats
    return out


# ---------------------------------------------------------------------------
# Leave-block-out
# ---------------------------------------------------------------------------


def _bootstrap_rel(sq_a: np.ndarray, sq_b: np.ndarray, block: np.ndarray, seed: int = 20251030):
    """95% interval of ``rmse(b) / rmse(a) - 1`` resampling whole blocks (a: debiased)."""
    ids, inverse = np.unique(block, return_inverse=True)
    if ids.size < 5:
        return None
    sa = np.bincount(inverse, weights=sq_a)
    sb = np.bincount(inverse, weights=sq_b)
    cnt = np.bincount(inverse).astype(float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, ids.size, size=(2000, ids.size))
    rel = np.sqrt(sb[draw].sum(1) / cnt[draw].sum(1)) / np.sqrt(sa[draw].sum(1) / cnt[draw].sum(1))
    return [_f(np.percentile(rel, 2.5) - 1), _f(np.percentile(rel, 97.5) - 1)]


def _summarise(parts: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    if not parts:
        return {"n": 0}
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    both = np.isfinite(cat["debiased"]) & np.isfinite(cat["fused"])
    n_hidden = int(cat["truth"].size)
    if not both.any():
        return {"n_hidden": n_hidden, "n_paired": 0}
    e_a = cat["debiased"][both] - cat["truth"][both]
    e_b = cat["fused"][both] - cat["truth"][both]
    return {
        "n_hidden": n_hidden,
        "n_paired": int(both.sum()),
        "debiased_defined": _f(np.isfinite(cat["debiased"]).mean()),
        "fused_defined": _f(np.isfinite(cat["fused"]).mean()),
        "rmse_debiased_k": _f(np.sqrt(np.mean(e_a**2))),
        "rmse_fused_k": _f(np.sqrt(np.mean(e_b**2))),
        "bias_debiased_k": _f(e_a.mean()),
        "bias_fused_k": _f(e_b.mean()),
        "rel_change_rmse": _f(np.sqrt(np.mean(e_b**2)) / np.sqrt(np.mean(e_a**2)) - 1),
        "rel_change_ci95": _bootstrap_rel(e_a**2, e_b**2, cat["block_key"][both]),
        "dark_only": {
            "n": int((both & cat["dark"]).sum()),
            "rmse_debiased_k": _f(
                np.sqrt(np.mean((cat["debiased"] - cat["truth"])[both & cat["dark"]] ** 2))
            )
            if (both & cat["dark"]).any()
            else None,
            "rmse_fused_k": _f(
                np.sqrt(np.mean((cat["fused"] - cat["truth"])[both & cat["dark"]] ** 2))
            )
            if (both & cat["dark"]).any()
            else None,
        },
    }


def leave_block_out_report(d: ClimateData, cfg) -> dict[str, Any]:
    """The switch-rule test, on blocks of any trusted readings and on night blocks only."""
    assert d.fusion is not None
    f = d.fusion
    trusted = d.t_out_valid & np.isfinite(d.t_out)
    s3 = d.day_mask((d.day_season == "S3") & d.day_complete & d.day_fit_month)
    season = _season_of(_local_month(d))
    max_gap = int(cfg.signals.t_out_debias.max_gap_hours * 60 // 15)

    def run(
        block_ok: np.ndarray | None, hours_list: tuple[int, ...], max_gap_slots: int
    ) -> dict[str, Any]:
        by_len: dict[str, list] = {}
        pooled: list = []
        by_season: dict[str, dict[str, list]] = {s: {} for s in ("autumn", "winter", "spring")}
        for hours in hours_list:
            for phase in range(0, 96, PHASE_STEP_SLOTS):
                r = leave_block_out(
                    house=d.t_out,
                    reference=f.reference,
                    trusted=trusted,
                    observed=f.observed,
                    params=f.params,
                    region=s3,
                    block_slots=hours * 4,
                    stride_slots=96,
                    phase_slots=phase,
                    max_gap_slots=max_gap_slots,
                    block_ok=block_ok,
                )
                if r["slot"].size == 0:
                    continue
                slots = r["slot"].astype(int)
                part = {k: r[k] for k in ("truth", "debiased", "fused", "dark")}
                part["block_key"] = (r["block"] + 10_000 * phase + 10**7 * hours).astype(np.int64)
                by_len.setdefault(str(hours), []).append(part)
                pooled.append(part)
                for s in by_season:
                    sel = season[slots] == s
                    if sel.any():
                        by_season[s].setdefault(str(hours), []).append(
                            {k: v[sel] for k, v in part.items()}
                        )
        return {
            "pooled": _summarise(pooled),
            "by_block_hours": {h: _summarise(p) for h, p in by_len.items()},
            "by_season_and_block_hours": {
                s: {h: _summarise(p) for h, p in per.items()} for s, per in by_season.items()
            },
            "by_season_pooled": {
                s: _summarise([x for p in per.values() for x in p]) for s, per in by_season.items()
            },
        }

    dark = d.solar < cfg.signals.t_out_fused.dark_solar_max
    return {
        "region": "S3 (October-April)",
        "any_trusted_blocks": run(None, BLOCK_HOURS, max_gap),
        "night_blocks": run(dark, BLOCK_HOURS, max_gap),
        "extended_any_trusted_blocks_uncapped_interpolation": run(
            None, EXTENDED_BLOCK_HOURS, 10_000
        ),
    }


def step_report(d: ClimateData, cfg) -> dict[str, Any]:
    assert d.fusion is not None
    f = d.fusion
    min_off = cfg.signals.t_out_debias.min_off_minutes // 15
    freq = d.columns["compressor_freq_mean"]
    out: dict[str, Any] = {}
    for label, series in (("raw", d.t_out), ("debiased", d.t_out_debiased), ("fused", f.fused)):
        row = {}
        for lag in (0, 2, 4):
            steps = compressor_start_step(
                series, f.reference, freq, min_off_slots=min_off, window_slots=2, lag_slots=lag
            )
            row[f"after_{lag * 15}min"] = {
                "n": int(steps.size),
                "mean_k": _f(steps.mean()) if steps.size else None,
                "median_k": _f(np.median(steps)) if steps.size else None,
            }
        out[label] = row
    return out


def evaluate(d: ClimateData, cfg) -> dict[str, Any]:
    assert d.fusion is not None
    f = d.fusion
    trusted = d.t_out_valid & np.isfinite(d.t_out)
    complete = d.day_mask(d.day_complete)
    seasons = {s: d.season_slots(s, months="fit") for s in ("S1", "S2", "S3")}
    return {
        "parameters": {**f.params.to_dict(), "n_fit_observations": f.n_fit_obs},
        "fit_seasons": list(cfg.signals.t_out_fused.fit_seasons),
        "station_manifest_sha256": f.station_sha256,
        "coverage": {
            name: {
                "t_out_debiased_finite": _f(np.isfinite(d.t_out_debiased[m]).mean()),
                "t_out_debiased_interpolated": _f(d.t_out_interpolated[m].mean()),
                "t_out_fused_finite": _f(np.isfinite(f.fused[m]).mean()),
                "fused_from_reference": _f(f.from_reference[m].mean()),
                "reference_finite": _f(np.isfinite(f.reference[m]).mean()),
                "trusted": _f(trusted[m].mean()),
                "fused_sd_median_k": _f(np.nanmedian(f.sd[m])),
            }
            for name, m in {**seasons, "all_complete_days": complete}.items()
            if m.any()
        },
        "leave_block_out": leave_block_out_report(d, cfg),
        "compressor_start_step": step_report(d, cfg),
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _md(obj: Any, indent: int = 0) -> list[str]:
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict | list) and v and not (isinstance(v, list) and len(v) <= 24):
                lines.append(f"{pad}- **{k}**")
                lines += _md(v, indent + 1)
            else:
                lines.append(f"{pad}- **{k}**: {json.dumps(v)}")
    return lines


def _write(root: Path, kind: str, name: str, report: dict[str, Any]) -> Path:
    out = root / REPORT_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{kind}.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    md = [f"# Outdoor reference: {kind}", "", f"{ATTRIBUTION}.", "", *_md(report), ""]
    (out / f"{kind}.md").write_text("\n".join(md))
    return out


def _name(value: str) -> str:
    if value and (not re.fullmatch(r"[A-Za-z0-9._-]+", value) or ".." in value):
        raise argparse.ArgumentTypeError(f"{value!r}: use only [A-Za-z0-9._-] and no '..'")
    return value


def _load(args: argparse.Namespace, *, with_station: bool):
    root = resolve_data_dir(args.data_dir)
    cfg = load_climate_config(args.config)
    site = load_site(root / SITE_FILE)
    if not with_station:
        site = replace(site, reference_station=None)
    elif args.station:
        site = replace(site, reference_station=args.station)
    d = load_climate_data(
        cfg, feature_set=args.feature_set, dataset=args.dataset, data_dir=str(root), site=site
    )
    return root, cfg, d


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("fetch", "compare", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--data-dir", default=None)
        p.add_argument("--config", default="config/climate.yaml")
        p.add_argument("--name", default="", type=_name)
        if name != "evaluate":
            p.add_argument("--stations", nargs="+", required=True, help="5-digit WMO ids")
        else:
            p.add_argument(
                "--station",
                default=None,
                help="WMO id (default: the site file's reference_station)",
            )
        if name == "fetch":
            p.add_argument("--from", dest="first", default="2023-11-01", type=date.fromisoformat)
            p.add_argument("--to", dest="last", default="2026-05-01", type=date.fromisoformat)
        else:
            p.add_argument("--feature-set", required=True)
            p.add_argument("--dataset", required=True)
    args = parser.parse_args(argv)
    root = resolve_data_dir(args.data_dir)

    if args.cmd == "fetch":
        months = month_range(args.first, args.last)
        for station in args.stations:
            try:
                got = fetch_station(root, station, months, log=lambda m, s=station: print(s, m))
            except StationError as exc:
                print(f"{station}: {exc}", file=sys.stderr)
                return 1
            print(f"{station}: fetched {len(got)} new month(s) of {len(months)}")
        return 0

    _root, cfg, d = _load(args, with_station=args.cmd == "evaluate")
    label = f"{d.feature_set_id[:12]}{'-' + args.name if args.name else ''}"
    if args.cmd == "compare":
        report: dict[str, Any] = {"attribution": ATTRIBUTION, "stations": {}}
        for station in args.stations:
            try:
                series = load_station(root, station, station_months(d.starts_us))
            except StationError as exc:
                print(f"{station}: {exc}", file=sys.stderr)
                return 1
            ref = reference_at_slots(
                series.times_us,
                series.temp_c,
                d.starts_us + SLOT_US // 2,
                cfg.signals.t_out_fused.max_ref_gap_hours,
            )
            report["stations"][station] = compare_one(
                d, ref, cfg.signals.t_out_fused.dark_solar_max
            ) | {
                "station_manifest_sha256": series.manifest_sha256,
                "reports_decoded": int(series.times_us.size),
            }
        print(_write(root, "compare", label, report))
        return 0

    if d.fusion is None:
        print("the site file has no reference_station", file=sys.stderr)
        return 1
    print(_write(root, "evaluate", label, evaluate(d, cfg)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
