"""The preregistered section-A gates of the live shadow, from its persisted records.

Source of every number here: ``docs/planning/climate-preregistration-2026-27.md`` (committed
2026-10-31, before any 2026-27 heating data).  The thresholds may change only through a dated
amendment in that file **before go-live**; a test pins them to it.

Windows (preregistration section 1): the **legacy** window is the shadow deployment until the
comfort controller's go-live (records in the ``legacy`` regime); the **new** window is go-live
plus 7 days of settling through 2027-03-31 (records in the ``comfort`` regime).  Go-live is the
first local day of a record in the ``comfort`` regime unless the service config names it.  A
record whose regime does not match its window (``unknown``, ``comfort_off``, ...) counts in
neither.  Each window needs at least 6 weeks (42) of valid heating days, else every criterion is
UNDETERMINED, which is never read as a pass.

Rows: the forecast origins whose local day is a valid heating day of the window.  RMSE, bias and
skill use the origins where both the model and persistence have an error, so the skill is on the
same origins.  The model error is the room-mean prediction driven by the measured heat and the
measured outdoor temperature (:func:`battery_optimizer.climate.shadow.score`, ``heat``).

The daily heat-delivery error reads the midnight origins: the heat the model's heat law delivers
under the actions (the replayed permission, or the measured run after the rule is retired) against
the measured heat, on the days whose measured heat is above 20% of the window's median daily heat.

The plan-versus-rule electricity ratio is **descriptive only** and grants nothing.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np

__all__ = [
    "CRITERIA",
    "HEAT_DAY_SHARE",
    "MIN_VALID_DAYS",
    "SEASON_END",
    "SETTLING_DAYS",
    "WINDOWS",
    "Criterion",
    "evaluate",
    "find_go_live",
    "render_markdown",
    "window_of",
]


@dataclass(frozen=True)
class Criterion:
    name: str
    metric: str
    op: str  # "<=" or ">="
    threshold: float
    text: str

    def passes(self, value: float) -> bool:
        return value <= self.threshold if self.op == "<=" else value >= self.threshold


#: Preregistration section 2, in its order.
CRITERIA: tuple[Criterion, ...] = (
    Criterion("rmse_6h", "rmse_6h", "<=", 0.30, "Room-mean RMSE at 6 h at most 0.30 K"),
    Criterion("rmse_12h", "rmse_12h", "<=", 0.40, "Room-mean RMSE at 12 h at most 0.40 K"),
    Criterion("skill_6h", "skill_6h", ">=", 0.25, "Skill versus persistence at 6 h at least 25%"),
    Criterion(
        "skill_12h", "skill_12h", ">=", 0.25, "Skill versus persistence at 12 h at least 25%"
    ),
    Criterion("abs_bias_6h", "abs_bias_6h", "<=", 0.15, "Mean bias at 6 h, |value| at most 0.15 K"),
    Criterion(
        "daily_heat_mdape",
        "daily_heat_mdape",
        "<=",
        0.15,
        "Daily heat-delivery median absolute percentage error at most 15%",
    ),
)
MIN_VALID_DAYS = 42  # 6 weeks of valid heating days per verdict
SETTLING_DAYS = 7
SEASON_END = date(2027, 3, 31)
HEAT_DAY_SHARE = 0.20
WINDOWS = ("legacy", "new")
GATE_HORIZONS = (6, 12)
REPORT_HORIZONS = (1, 6, 12, 24)
PASS, FAIL, UNDETERMINED = "PASS", "FAIL", "UNDETERMINED"


def find_go_live(records: Iterable[Mapping[str, Any]], override: date | None = None) -> date | None:
    """The configured go-live day, else the first local day of a ``comfort`` record."""
    if override is not None:
        return override
    days = [date.fromisoformat(r["local_day"]) for r in records if r.get("regime") == "comfort"]
    return min(days) if days else None


def window_of(day: date, regime: str | None, go_live: date | None) -> str | None:
    """The preregistered window of a record of *day* in *regime*, or ``None`` (excluded)."""
    if day > SEASON_END:
        return None
    if go_live is None or day < go_live:
        return "legacy" if regime == "legacy" else None
    if day >= go_live + timedelta(days=SETTLING_DAYS):
        return "new" if regime == "comfort" else None
    return None


def _day_regime(record: Mapping[str, Any]) -> str | None:
    known = {r for r in record.get("regimes") or () if r != "unknown"}
    return next(iter(known)) if len(known) == 1 else None


def _rmse(x: np.ndarray) -> float | None:
    return float(np.sqrt(np.mean(x**2))) if x.size else None


def _horizon_stats(rows: list[Mapping[str, Any]], hh: int) -> dict[str, Any]:
    pairs = []
    for r in rows:
        cell = (r.get("room_mean") or {}).get(f"h{hh}") or {}
        if cell.get("heat") is not None and cell.get("persistence") is not None:
            pairs.append((cell["heat"], cell["persistence"]))
    if not pairs:
        return {"n": 0, "rmse": None, "rmse_persistence": None, "skill": None, "bias": None}
    e = np.array(pairs, float)
    rmse, rmse_p = _rmse(e[:, 0]), _rmse(e[:, 1])
    return {
        "n": len(pairs),
        "rmse": rmse,
        "rmse_persistence": rmse_p,
        "skill": (1.0 - rmse / rmse_p) if rmse is not None and rmse_p else None,
        "bias": float(e[:, 0].mean()),
    }


def _daily_heat(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    days = [
        r["accuracy"]["day"]
        for r in rows
        if r.get("midnight") and "day" in (r.get("accuracy") or {})
    ]
    meas = np.array([x["q_meas_kwh"] for x in days], float)
    pred = np.array([x["q_pred_kwh"] for x in days], float)
    if not meas.size:
        return {"days": 0, "days_scored": 0, "mdape": None}
    floor = HEAT_DAY_SHARE * float(np.median(meas))
    keep = (meas > floor) & (meas > 0)
    ape = np.abs(pred[keep] - meas[keep]) / meas[keep]
    return {
        "days": int(meas.size),
        "days_scored": int(keep.sum()),
        "mdape": float(np.median(ape)) if ape.size else None,
    }


def _plan_stats(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    plans = [r.get("plan") or {} for r in rows]
    statuses = Counter(str(p.get("status")) for p in plans)
    times = np.array([p["solve_s"] for p in plans if p.get("solve_s") is not None], float)
    heat = sum(p.get("heat_slots") or 0 for p in plans)
    top = sum(p.get("top_slots") or 0 for p in plans)
    mid = [
        (r["plan"]["kwh"], r["accuracy"]["day"]["e_meas_kwh"])
        for r in rows
        if r.get("midnight")
        and (r.get("plan") or {}).get("kwh") is not None
        and "day" in (r.get("accuracy") or {})
    ]
    e_meas = sum(m for _p, m in mid)
    return {
        "decisions": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "solve_s": {
            "median": float(np.median(times)) if times.size else None,
            "p95": float(np.quantile(times, 0.95)) if times.size else None,
        },
        "top_offset_share": (top / heat) if heat else None,
        "plan_over_rule_electricity": {
            "days": len(mid),
            "ratio": (sum(p for p, _m in mid) / e_meas) if e_meas > 0 else None,
            "note": "descriptive only: the MPC never acts, its comfort is not observed",
        },
    }


def _verdict(metrics: Mapping[str, Any], valid_days: int) -> tuple[dict[str, str], str]:
    if valid_days < MIN_VALID_DAYS:
        return {c.name: UNDETERMINED for c in CRITERIA}, UNDETERMINED
    status = {}
    for c in CRITERIA:
        value = metrics.get(c.metric)
        status[c.name] = UNDETERMINED if value is None else (PASS if c.passes(value) else FAIL)
    values = set(status.values())
    verdict = FAIL if FAIL in values else UNDETERMINED if UNDETERMINED in values else PASS
    return status, verdict


def evaluate(
    scores: Iterable[Mapping[str, Any]],
    days: Iterable[Mapping[str, Any]],
    *,
    go_live: date | None = None,
) -> dict[str, Any]:
    """The gate report of both windows from the score and day records."""
    scores = list(scores)
    days = list(days)
    live = find_go_live(scores, go_live)
    latest: dict[str, Mapping[str, Any]] = {}
    for rec in days:  # a re-scored day keeps its last record
        latest[rec["day"]] = rec
    report: dict[str, Any] = {
        "go_live": live.isoformat() if live else None,
        "go_live_source": "config" if go_live else ("records" if live else None),
        "settling_days": SETTLING_DAYS,
        "season_end": SEASON_END.isoformat(),
        "min_valid_days": MIN_VALID_DAYS,
        "criteria": [c.__dict__ for c in CRITERIA],
        "windows": {},
    }
    for w in WINDOWS:
        valid = sorted(
            d
            for d, rec in latest.items()
            if rec.get("valid_heating_day")
            and window_of(date.fromisoformat(d), _day_regime(rec), live) == w
        )
        valid_set = set(valid)
        in_window = [
            r
            for r in scores
            if window_of(date.fromisoformat(r["local_day"]), r.get("regime"), live) == w
        ]
        rows = [r for r in in_window if r["local_day"] in valid_set]
        horizons = {f"{hh}h": _horizon_stats(rows, hh) for hh in REPORT_HORIZONS}
        heat = _daily_heat(rows)
        metrics: dict[str, Any] = {"daily_heat_mdape": heat["mdape"]}
        for hh in GATE_HORIZONS:
            s = horizons[f"{hh}h"]
            metrics[f"rmse_{hh}h"] = s["rmse"]
            metrics[f"skill_{hh}h"] = s["skill"]
        bias = horizons["6h"]["bias"]
        metrics["abs_bias_6h"] = abs(bias) if bias is not None else None
        status, verdict = _verdict(metrics, len(valid))
        report["windows"][w] = {
            "valid_heating_days": len(valid),
            "valid_weeks": len(valid) / 7.0,
            "first_day": valid[0] if valid else None,
            "last_day": valid[-1] if valid else None,
            "origins": len(rows),
            "horizons": horizons,
            "daily_heat": heat,
            "metrics": metrics,
            "criteria": status,
            "verdict": verdict,
            "plan": _plan_stats(in_window),
        }
    return report


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def render_markdown(report: Mapping[str, Any]) -> str:
    """The gate report as Markdown (private figures: keep it with the data)."""
    lines = [
        "# Climate shadow: preregistered gates (section A)",
        "",
        "Shadow only: the MPC never acts. Thresholds from "
        "docs/planning/climate-preregistration-2026-27.md; UNDETERMINED is never a pass.",
        "",
        f"- go-live: {report['go_live'] or 'not yet'} ({report['go_live_source'] or 'none'}); "
        f"settling {report['settling_days']} days; season end {report['season_end']}; "
        f"each verdict needs {report['min_valid_days']} valid heating days",
        "",
    ]
    names = {c["name"]: c for c in report["criteria"]}
    for w, win in report["windows"].items():
        lines += [
            f"## {w} window: {win['verdict']}",
            "",
            f"- valid heating days {win['valid_heating_days']} ({win['valid_weeks']:.1f} weeks), "
            f"{win['first_day'] or '-'} to {win['last_day'] or '-'}; origins {win['origins']}",
            "",
            "| criterion | value | threshold | status |",
            "|---|---|---|---|",
        ]
        for name, status in win["criteria"].items():
            c = names[name]
            value = win["metrics"].get(c["metric"])
            shown = _pct(value) if "skill" in name or "mdape" in name else _f(value)
            limit = (
                _pct(c["threshold"])
                if "skill" in name or "mdape" in name
                else f"{c['threshold']:.2f} K"
            )
            lines.append(f"| {c['text']} | {shown} | {c['op']} {limit} | {status} |")
        lines += [
            "",
            "| horizon | n | RMSE (K) | persistence RMSE (K) | skill | bias (K) |",
            "|---|---|---|---|---|---|",
        ]
        for h, s in win["horizons"].items():
            lines.append(
                f"| {h} | {s['n']} | {_f(s['rmse'])} | {_f(s['rmse_persistence'])} | "
                f"{_pct(s['skill'])} | {_f(s['bias'])} |"
            )
        heat, plan = win["daily_heat"], win["plan"]
        ratio = plan["plan_over_rule_electricity"]
        lines += [
            "",
            f"- daily heat delivery: MdAPE {_pct(heat['mdape'])} on {heat['days_scored']} of "
            f"{heat['days']} midnight days",
            f"- plans: {plan['decisions']} decisions, statuses {plan['status_counts']}, solve "
            f"median / p95 {_f(plan['solve_s']['median'])} / {_f(plan['solve_s']['p95'])} s, "
            f"top-offset share {_pct(plan['top_offset_share'])}",
            f"- plan / rule electricity (descriptive only, grants nothing): "
            f"{_f(ratio['ratio'])} over {ratio['days']} days",
            "",
        ]
    return "\n".join(lines)
