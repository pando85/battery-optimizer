#!/usr/bin/env python3
"""Offline receding-horizon replay of the price-blind comfort MPC (track B, step B1). Read-only.

The MPC plans 24 h every hour over the replayed heating days (``config/climate-mpc.yaml``) from
the observer's state, while the house was run by the legacy controller. **Planned kWh and planned
comfort are model predictions of plans that were never executed, not realised outcomes.** See
``battery_optimizer.climate.mpc_replay`` for the causality rules and the declared foresight.

Usage:
    uv run python hack/climate_mpc_replay.py \\
        --feature-set <climate feature set id or prefix> --dataset <dataset dir or id> \\
        [--data-dir ~/data/<store>] [--run-id <name>] [--workers 8] \\
        [--config config/climate.yaml] [--mpc-config config/climate-mpc.yaml] \\
        [--model b2d|c2] [--weather perfect|persistence] [--seasons S2,S3] [--max-days N] \\
        [--offsets 0,1,2,3]

``BATTERY_OPTIMIZER_DATA_DIR`` replaces ``--data-dir``. Everything is written to
``<data_dir>/analysis/v1/climate-mpc/<run-id>/`` (``report.json``, ``report.md``,
``days.jsonl``); the repository is never touched. The figures are private: keep them in the data
dir. ``--model``, ``--weather``, ``--seasons``, ``--max-days`` and ``--offsets`` override the
configuration and are recorded in the report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# one BLAS thread per worker process (the replay parallelises over days)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT / "src"))

from battery_optimizer.climate import b2d, mpc_replay  # noqa: E402
from battery_optimizer.climate.config import load_climate_config  # noqa: E402
from battery_optimizer.climate.data import SITE_FILE, load_climate_data, load_site  # noqa: E402
from battery_optimizer.climate.mpc import PlanSettings  # noqa: E402
from battery_optimizer.climate.mpc_config import load_mpc_config, override  # noqa: E402
from battery_optimizer.features import json_bytes  # noqa: E402
from battery_optimizer.manifest import code_version  # noqa: E402
from battery_optimizer.paths import resolve_data_dir, write_part_atomic  # noqa: E402

log = logging.getLogger("climate_mpc_replay")


def _run_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]+", value) or ".." in value:
        raise argparse.ArgumentTypeError(f"{value!r}: use only [A-Za-z0-9._-] and no '..'")
    return value


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:+.1f}%"


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _relin(r: dict[str, Any]) -> str:
    if not r.get("decisions"):
        return "n/a"
    last = r["last_iterate_changed_slots"]
    return (
        f"iterate kept {r['chosen_share']}, last iterate changed a median of "
        f"{_f(last['median'], 1)} slots (unchanged in {_pct(last['share_zero'])})"
    )


def _slots(p: dict[str, Any]) -> str:
    if not p.get("heat_slots"):
        return "no planned heating"
    return (
        f"top offset {_pct(p['top_offset_share'])} (midnight plans "
        f"{_pct(p.get('midnight_top_offset_share'))}); mean valve factor "
        f"{_f(p.get('valve_factor_mean'))}; open rooms share {p['open_rooms_share']}; "
        f"switched off by the run logic in simulation {_pct(p['runlogic_off_share'])}"
    )


def render(report: dict[str, Any]) -> str:
    lines = [
        "# Comfort MPC offline replay (private: numbers from the real installation)",
        "",
        "**Planned kWh and planned comfort are MODEL PREDICTIONS of plans that were never",
        "executed, not realised outcomes.** The house ran the legacy controller. The model is",
        "exploratory",
        "(no accepted thermal model: C2 UNDETERMINED, B0 exploratory). Predictions beyond about",
        "12 h are low-confidence.",
        "",
        f"- run `{report['run_id']}`, code `{report['lineage']['code_version']}`, model "
        f"`{report['settings']['model']}`, weather `{report['settings']['weather']}`"
        + (" (perfect foresight of weather)" if report["settings"]["weather"] == "perfect" else ""),
        f"- overrides: {report['settings']['overrides'] or 'none'}",
        "",
    ]
    for season, s in report["seasons"].items():
        acc, cmp_, sol = (
            s["summary"]["accuracy"],
            s["summary"]["comparison"],
            s["summary"]["solver"],
        )
        lines += [f"## {season} (model prior fitted on {s['model']['prior_season']})", ""]
        lines += [
            "### (a) Open-loop accuracy (legacy actions through the model)",
            "",
            "| horizon | air MAE / bias (K) | hall MAE / bias (K) | air, measured Q "
            "| persistence air MAE |",
            "|---|---|---|---|---|",
        ]
        for h, row in acc.items():
            if not h.endswith("h") or "air" not in row:
                continue
            a, hl, aq, pa = row["air"], row["hall"], row["air_q"], row["pers_air"]
            lines.append(
                f"| {h} | {_f(a.get('mae'))} / {_f(a.get('bias'))} (n {a.get('n')}) | "
                f"{_f(hl.get('mae'))} / {_f(hl.get('bias'))} | {_f(aq.get('mae'))} / "
                f"{_f(aq.get('bias'))} | {_f(pa.get('mae'))} |"
            )
        dh, de = acc["daily_heat"], acc["daily_electricity"]
        lines += [
            "",
            f"- daily heat delivered (permission to heat link): MAPE {_pct(dh['mape'])}, "
            f"season bias {_pct(dh['season_bias'])} over {dh['days']} days",
            f"- daily electricity: MAPE {_pct(de['mape'])}, season bias "
            f"{_pct(de['season_bias'])} over {de['days']} days",
            "",
            "### (b) Plan (model prediction) against legacy (measured), same days",
            "",
        ]
        if cmp_.get("days"):
            kh = cmp_["kh_per_day"]
            lines += [
                f"- days {cmp_['days']}; planned / legacy measured kWh "
                f"{_f(cmp_['plan_over_legacy_kwh'])}; planned / legacy through the same model "
                f"{_f(cmp_['plan_over_legacy_model_kwh'])} (model / measured on legacy "
                f"{_f(cmp_['legacy_model_over_legacy_kwh'])})",
                f"- Kh/day below the day minimum: plan {_f(kh['plan_day'])}, legacy "
                f"{_f(kh['legacy_day'])};"
                f" below the floor: plan {_f(kh['plan_floor'])}, legacy {_f(kh['legacy_floor'])}",
                "",
                "| HDD band | days | plan / legacy kWh |",
                "|---|---|---|",
            ]
            for b in cmp_.get("by_hdd_tercile", []):
                lines.append(
                    f"| {b['hdd_from']:.1f} to {b['hdd_to']:.1f} | {b['days']} | "
                    f"{_f(b['plan_over_legacy_kwh'])} |"
                )
            m = cmp_["material"]
            lines += [
                "",
                "### (d) Material differences",
                "",
                f"- kWh differs by more than the threshold on {_pct(m['kwh_share'])} of days, the "
                f"schedule overlap is below the threshold on {_pct(m['schedule_share'])}, "
                "either on "
                f"{_pct(m['either_share'])}; median heating-slot overlap {_f(m['median_overlap'])}",
                "",
            ]
        lines += [
            "### (c) Solver",
            "",
            f"- decisions {sol['decisions']}, solves {sol['solves']}, statuses "
            f"{sol['status_counts']}, infeasible or error {sol['infeasible_or_error']}",
            f"- solve time median / p95 / max {_f(sol['solve_s']['median'])} / "
            f"{_f(sol['solve_s']['p95'])} / {_f(sol['solve_s']['max'])} s; gap median / max "
            f"{_f(sol['gap']['median'], 5)} / {_f(sol['gap']['max'], 5)}",
            f"- planned lift outside the COP table {_pct(sol['lift_outside_table_mean'])}; "
            f"re-linearisation: {_relin(sol['relinearization'])}",
            "",
            "### Planned heating slots",
            "",
            f"- {_slots(s['summary']['plan_slots'])}",
            "",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--feature-set", required=True, help="climate feature set id (or prefix)")
    parser.add_argument("--dataset", required=True, help="canonical dataset directory or id")
    parser.add_argument("--config", default=str(_PROJECT / "config" / "climate.yaml"))
    parser.add_argument("--mpc-config", default=str(_PROJECT / "config" / "climate-mpc.yaml"))
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--run-id", type=_run_id, default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model", choices=("b2d", "c2"), default=None)
    parser.add_argument("--weather", choices=("perfect", "persistence"), default=None)
    parser.add_argument("--seasons", default=None, help="comma-separated, e.g. S2,S3")
    parser.add_argument(
        "--max-days", type=int, default=None, help="evenly spaced subset per season"
    )
    parser.add_argument("--offsets", default=None, help="water offsets in K, e.g. 0,1,2,3")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    ccfg = load_climate_config(args.config)
    cfg = load_mpc_config(args.mpc_config)
    try:
        mpc_replay.check_configs(ccfg, cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    overrides: dict[str, Any] = {}
    if args.model:
        cfg = override(cfg, ("model", "kind"), args.model)
        overrides["model.kind"] = args.model
    if args.weather:
        cfg = override(cfg, ("replay", "weather"), args.weather)
        overrides["replay.weather"] = args.weather
    seasons = tuple(args.seasons.split(",")) if args.seasons else tuple(cfg.replay.seasons)
    if args.seasons:
        overrides["replay.seasons"] = list(seasons)
    if args.max_days:
        overrides["max_days"] = args.max_days
    if args.offsets:
        offsets = [int(v) for v in args.offsets.split(",")]
        cfg = override(cfg, ("water", "offsets_k"), tuple(offsets))
        overrides["water.offsets_k"] = offsets

    root = resolve_data_dir(args.data_dir)
    out_dir = mpc_replay.outputs_dir(root, args.run_id)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; choose another --run-id")
    site = load_site(root / SITE_FILE)
    started = time.time()
    log.info("loading climate data")
    d = load_climate_data(
        ccfg, feature_set=args.feature_set, dataset=args.dataset, data_dir=str(root), site=site
    )
    hour_frac, weekend = b2d.slot_calendar(d)
    diurnal = b2d.diurnal_matrix(hour_frac, weekend, cfg.model.diurnal_harmonics)
    t_out_causal = mpc_replay.causal_outdoor(d)
    hist_t_out = d.t_out_model if cfg.replay.weather == "perfect" else t_out_causal
    w_hist = b2d.full_inputs(d, diurnal)
    w_hist[:, 1] = np.where(d.cooling, np.nan, hist_t_out)
    c2_record = None
    c2_id = None
    if cfg.model.kind == "c2":
        base = root / "analysis" / "v1" / "climate-models"
        matches = sorted(p for p in base.glob(f"{cfg.model.c2_model_id}*") if p.is_dir())
        if len(matches) != 1:
            raise SystemExit(f"C2 model {cfg.model.c2_model_id!r} matches {len(matches)}")
        c2_record = json.loads((matches[0] / "model.json").read_text())
        c2_id = c2_record.get("model_id")
    phi = site.phi if site.phi is not None else float(ccfg.model.response.phi_default)
    settings = PlanSettings.from_config(cfg)

    report: dict[str, Any] = {
        "run_id": args.run_id,
        "notice": "Planned figures are model predictions of plans never executed, not outcomes.",
        "settings": {
            "model": cfg.model.kind,
            "weather": cfg.replay.weather,
            "declared_foresight": (
                ["outdoor temperature over the horizon (perfect)"]
                if cfg.replay.weather == "perfect"
                else []
            )
            + ["DHW mask over the horizon", "sun proxy and calendar (deterministic)"],
            "overrides": overrides,
            "workers": args.workers,
        },
        "lineage": {
            "code_version": code_version(),
            "climate_config_sha256": ccfg.sha256,
            "mpc_config_sha256": cfg.sha256,
            "feature_set_id": d.feature_set_id,
            "dataset_id": d.dataset_id,
            "input_sha256": d.input_sha256,
            "c2_model_id": c2_id,
        },
        "seasons": {},
    }
    all_days: list[dict[str, Any]] = []
    for season in seasons:
        log.info("preparing %s", season)
        sm = mpc_replay.prepare_season(
            d, ccfg, cfg, season, w_hist=w_hist, diurnal=diurnal, phi=phi, c2_record=c2_record
        )
        cop_season = "S2" if cfg.model.kind == "c2" else sm.prior_season
        cop_model = mpc_replay.fit_cop(d, ccfg, cop_season)
        ctx = mpc_replay.Context(
            d=d,
            cfg=cfg,
            settings=settings,
            season=sm,
            cop_model=cop_model,
            diurnal=diurnal,
            hour_frac=hour_frac,
            weekend=weekend,
            t_out_model=hist_t_out,
            t_out_causal=t_out_causal,
            space_heat=mpc_replay.space_heat_state(d),
        )
        days = mpc_replay.replay_day_indices(ctx, season, args.max_days)
        log.info("replaying %d days of %s with %d workers", len(days), season, args.workers)
        t1 = time.time()
        records = mpc_replay.replay_days(ctx, days, workers=args.workers)
        log.info("%s done in %.0f s", season, time.time() - t1)
        for r in records:
            r["season"] = season
        all_days += records
        report["seasons"][season] = {
            "model": {
                **{k: v for k, v in sm.info.items()},
                "prior_season": sm.prior_season,
                "cop_model_id": cop_model.model_id,
                "cop_train_season": cop_season,
                "cop_selected": cop_model.selected,
            },
            "days": len(days),
            "summary": mpc_replay.summarise(records, cfg),
        }
    report["elapsed_s"] = time.time() - started
    write_part_atomic(out_dir / "report.json", json_bytes(report))
    write_part_atomic(out_dir / "report.md", render(report).encode())
    lines = "\n".join(json.dumps(r, default=float) for r in all_days) + "\n"
    write_part_atomic(out_dir / "days.jsonl", lines.encode())
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
