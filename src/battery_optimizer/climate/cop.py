"""Heat-pump electricity (COP) model of the climate optimizer (C2, PR 2).

The thermal models give the heat the machine delivers; this module gives the electricity it takes
to deliver it, which is what a price is multiplied by.  It fits **electricity, not COP**, on hourly
aggregates of heat-mode hours without domestic hot water::

    E_el = Q * (a + b * (T_flow - T_out) + c * max(0, knee - T_out)) + d * starts + e * hours_on

``Q`` is the thermal output and ``E_el`` the electricity of the hour (kWh), ``T_flow`` the measured
flow temperature, ``T_out`` the model's outdoor temperature (``ClimateData.t_out_model``),
``starts`` the compressor starts in the hour and ``hours_on`` its on-time. ``a`` is the electricity
per heat unit with no lift, ``b`` the lift penalty, ``c`` the defrost penalty below the knee, ``d``
the cost of one start (kWh) and ``e`` the running overhead (kW). The fit is weighted least squares
with weight ``Q``. There is no weather-compensated heating curve on this plant (the flow target is a
per-season constant), which is harmless here because the model reads the *measured* flow.

Two tables sit beside it: the COP binned by ``T_out``, and the COP binned by the lift
``T_flow - T_out`` (the flow temperature changed by hand between and within seasons, which a
``T_out`` table cannot follow).  All three are validated on the held-out season given the measured
``Q``; the model the simulator uses is predeclared in ``cop.model`` (:attr:`CopModel.selected`),
and the candidate with the lowest held-out daily MAPE is reported beside it.

Folds, thresholds, the knee and the acceptance limits come from ``config/climate.yaml``
(``seasons``, ``folds``, ``cop``).  Fitting uses the fit months of the training seasons and
validation the gate months of the held-out one, as the plan prescribes for every model.

For the simulator: :func:`predict_electricity` and the versioned, hashable :class:`CopModel`.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from battery_optimizer.climate.config import ClimateConfig
from battery_optimizer.climate.data import SLOT_SECONDS, SLOTS_PER_HOUR, ClimateData

__all__ = [
    "COP_MODEL_VERSION",
    "CopError",
    "CopModel",
    "HourlyRows",
    "fit_and_validate",
    "fit_lift_table",
    "fit_regression",
    "fit_table",
    "hourly_rows",
    "predict_electricity",
    "render_markdown",
    "validate",
]

#: Version of the model structure and of the parameter record; any change of either bumps it.
COP_MODEL_VERSION = 2

COEFFICIENTS = ("a", "b", "c", "d", "e")
CANDIDATES = ("regression", "table", "lift_table")

#: A compressor-frequency slot mean above this is "on" (the counter data has small negative noise).
ON_FREQUENCY_HZ = 0.0
#: A table bin needs this many training hours of its own; a sparser bin borrows the nearest one.
MIN_BIN_ROWS = 10
#: A validation day is scored when at least this many of its hours are eligible rows.
MIN_DAY_HOURS = 6

#: Plausible physical ranges, reported next to the fitted terms.  They are a sanity check, not a
#: gate: a fit outside them is flagged and read by a person.
SANITY_RANGES: dict[str, tuple[float, float, str]] = {
    "a": (0.05, 0.60, "kWh_el per kWh_th at zero lift (COP at zero lift between 1.7 and 20)"),
    "b": (0.0, 0.03, "kWh_el per kWh_th per K of lift (cannot be negative)"),
    "c": (0.0, 0.05, "kWh_el per kWh_th per K below the defrost knee (a penalty, not a bonus)"),
    "d": (0.0, 0.30, "kWh_el per compressor start (inrush and restart losses are small)"),
    "e": (0.0, 0.50, "kW_el while the compressor is on (crankcase, fan and pump overhead)"),
}
#: Outdoor temperatures at which the implied COP curve is reported.
CURVE_T_OUT = (-5.0, 0.0, 5.0, 10.0, 15.0)


class CopError(ValueError):
    """The COP model cannot be fitted or evaluated on the data it was given."""


# ---------------------------------------------------------------------------
# Hourly rows
# ---------------------------------------------------------------------------


@dataclass
class HourlyRows:
    """Heat-mode hours without hot water, one entry per eligible hour.

    ``eligible`` counts rows surviving every exclusion; the ``excluded_*`` fields give the
    sequential reasons so the report can say where the data went.
    """

    hour_start_us: np.ndarray
    day: np.ndarray  # index into ClimateData.days
    season: np.ndarray  # object array: "S1" / "S2" / "S3" / ""
    fit_month: np.ndarray
    gate_month: np.ndarray
    q: np.ndarray
    e: np.ndarray
    t_flow: np.ndarray
    t_out: np.ndarray
    starts: np.ndarray
    hours_on: np.ndarray
    exclusions: dict[str, int]

    @property
    def n(self) -> int:
        return int(self.q.size)

    def select(self, mask: np.ndarray) -> HourlyRows:
        m = np.asarray(mask, dtype=bool)
        return HourlyRows(
            hour_start_us=self.hour_start_us[m],
            day=self.day[m],
            season=self.season[m],
            fit_month=self.fit_month[m],
            gate_month=self.gate_month[m],
            q=self.q[m],
            e=self.e[m],
            t_flow=self.t_flow[m],
            t_out=self.t_out[m],
            starts=self.starts[m],
            hours_on=self.hours_on[m],
            exclusions=dict(self.exclusions),
        )


def _hours(x: np.ndarray, n_hours: int) -> np.ndarray:
    return np.asarray(x)[: n_hours * SLOTS_PER_HOUR].reshape(n_hours, SLOTS_PER_HOUR)


def hourly_rows(data: ClimateData, config: ClimateConfig) -> HourlyRows:
    """Aggregate the 15-minute slots into UTC hours and keep the eligible heat-mode hours.

    An hour is dropped when any of its four slots lacks a value the model needs (thermal output,
    electricity, hot-water, cooling and compressor columns, starts, the flow temperature while
    heat flows, or the model's outdoor temperature), when hot water or cooling ran in it, or
    when its thermal output is below ``cop.min_q_kwh`` or its implied COP is outside
    ``cop.cop_range``.  Local time zones are whole hours from UTC, so a UTC hour is one local hour.
    """
    n_hours = data.n // SLOTS_PER_HOUR
    if n_hours == 0:
        raise CopError("the feature grid has no full hour")
    cop_cfg = config.cop
    lo_cop, hi_cop = (float(v) for v in cop_cfg.cop_range)
    col = data.columns

    q_slot = _hours(data.q_heat_kwh, n_hours)
    e_slot = _hours(data.e_heat_kwh, n_hours)
    flow_slot = _hours(col["flow_temp_mean_c"], n_hours)
    freq_slot = _hours(col["compressor_freq_mean"], n_hours)
    starts_slot = _hours(col["compressor_starts_delta"], n_hours)
    t_out_slot = _hours(data.t_out_model, n_hours)
    dhw_inputs = [
        _hours(col[name], n_hours) for name in ("hp_dhw_kwh", "hp_dhw_out_kwh", "dhw_force_mean")
    ]
    cool_slot = _hours(col["hp_cool_kwh"], n_hours)
    dhw_flag = _hours(data.dhw, n_hours).any(axis=1)
    cool_flag = _hours(data.cooling, n_hours).any(axis=1)

    fin = np.isfinite
    with np.errstate(invalid="ignore"):
        heat_flows = q_slot > 0.0
        known = (
            fin(q_slot).all(axis=1)
            & fin(e_slot).all(axis=1)
            & fin(freq_slot).all(axis=1)
            & fin(starts_slot).all(axis=1)
            & fin(cool_slot).all(axis=1)
            & np.logical_and.reduce([fin(x).all(axis=1) for x in dhw_inputs])
            & (fin(flow_slot) | ~heat_flows).all(axis=1)
        )
        q = np.where(known, q_slot.sum(axis=1), np.nan)
        e = np.where(known, e_slot.sum(axis=1), np.nan)
        starts = np.where(known, starts_slot.sum(axis=1), np.nan)
        hours_on = np.where(
            known, (freq_slot > ON_FREQUENCY_HZ).sum(axis=1) * (SLOT_SECONDS / 3600.0), np.nan
        )
        # the flow temperature of the hour is the heat-weighted mean of its slots
        weights = np.where(heat_flows & known[:, None], q_slot, 0.0)
        wsum = weights.sum(axis=1)
        t_flow = np.where(
            wsum > 0.0,
            (np.where(weights > 0.0, flow_slot, 0.0) * weights).sum(axis=1)
            / np.where(wsum > 0.0, wsum, 1.0),
            np.nan,
        )
        t_out = np.where(fin(t_out_slot).all(axis=1), t_out_slot.mean(axis=1), np.nan)
        implied_cop = q / np.where(e > 0.0, e, np.nan)

    steps: list[tuple[str, np.ndarray]] = [
        ("unknown_input", ~known),
        ("dhw_or_cooling", dhw_flag | cool_flag),
        ("outdoor_unknown", ~fin(t_out)),
        ("q_below_min", ~(q >= cop_cfg.min_q_kwh)),
        ("cop_out_of_range", ~((implied_cop >= lo_cop) & (implied_cop <= hi_cop))),
        ("flow_unknown", ~fin(t_flow)),
    ]
    keep = np.ones(n_hours, dtype=bool)
    exclusions: dict[str, int] = {"hours": n_hours}
    for name, drop in steps:
        newly = keep & drop
        exclusions[name] = int(newly.sum())
        keep &= ~drop
    exclusions["eligible"] = int(keep.sum())

    first = np.arange(n_hours) * SLOTS_PER_HOUR
    day = data.day_of_slot[first]
    return HourlyRows(
        hour_start_us=data.starts_us[first][keep],
        day=day[keep],
        season=data.day_season[day][keep],
        fit_month=data.day_fit_month[day][keep] & data.day_complete[day][keep],
        gate_month=data.day_gate_month[day][keep] & data.day_complete[day][keep],
        q=q[keep],
        e=e[keep],
        t_flow=t_flow[keep],
        t_out=t_out[keep],
        starts=starts[keep],
        hours_on=hours_on[keep],
        exclusions=exclusions,
    )


# ---------------------------------------------------------------------------
# Parameter record
# ---------------------------------------------------------------------------


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class CopModel:
    """The fitted electricity model: a versioned, hashable parameter record.

    Both candidates are kept; :attr:`selected` says which one answers :func:`predict_electricity`.
    Nothing else is hidden state, so two records with equal fields are the same model and
    :attr:`model_id` (a SHA-256 over the canonical JSON) identifies it in results.
    """

    version: int
    selected: str  # one of CANDIDATES
    coefficients: tuple[float, float, float, float, float]  # a, b, c, d, e
    knee_c: float
    table_bin_k: float
    #: ``(lower edge of the bin in C, COP, training hours)``, ascending, populated bins only.
    table: tuple[tuple[float, float, int], ...]
    cop_range: tuple[float, float]
    train_seasons: tuple[str, ...]
    n_train_rows: int
    feature_set_id: str
    config_sha256: str
    lift_bin_k: float = 2.0
    #: ``(lower edge of the lift bin in K, COP, training hours)``, ascending, populated bins only.
    lift_table: tuple[tuple[float, float, int], ...] = ()

    def __post_init__(self) -> None:
        if self.selected not in CANDIDATES:
            raise CopError(f"selected must be one of {list(CANDIDATES)}, not {self.selected!r}")
        if not self.table:
            raise CopError("the COP table is empty")
        if self.selected == "lift_table" and not self.lift_table:
            raise CopError("the lift table is empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "selected": self.selected,
            "coefficients": dict(zip(COEFFICIENTS, self.coefficients, strict=True)),
            "knee_c": self.knee_c,
            "table_bin_k": self.table_bin_k,
            "table": [list(row) for row in self.table],
            "cop_range": list(self.cop_range),
            "train_seasons": list(self.train_seasons),
            "n_train_rows": self.n_train_rows,
            "feature_set_id": self.feature_set_id,
            "config_sha256": self.config_sha256,
            "lift_bin_k": self.lift_bin_k,
            "lift_table": [list(row) for row in self.lift_table],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> CopModel:
        if raw.get("version") != COP_MODEL_VERSION:
            raise CopError(f"unsupported COP model version {raw.get('version')!r}")
        coef = raw["coefficients"]
        return cls(
            version=int(raw["version"]),
            selected=str(raw["selected"]),
            coefficients=tuple(float(coef[k]) for k in COEFFICIENTS),  # type: ignore[arg-type]
            knee_c=float(raw["knee_c"]),
            table_bin_k=float(raw["table_bin_k"]),
            table=tuple((float(lo), float(c), int(n)) for lo, c, n in raw["table"]),
            cop_range=(float(raw["cop_range"][0]), float(raw["cop_range"][1])),
            train_seasons=tuple(str(s) for s in raw["train_seasons"]),
            n_train_rows=int(raw["n_train_rows"]),
            feature_set_id=str(raw["feature_set_id"]),
            config_sha256=str(raw["config_sha256"]),
            lift_bin_k=float(raw["lift_bin_k"]),
            lift_table=tuple((float(lo), float(c), int(n)) for lo, c, n in raw["lift_table"]),
        )

    @property
    def model_id(self) -> str:
        return hashlib.sha256(_canonical(self.to_dict())).hexdigest()

    def with_selected(self, selected: str) -> CopModel:
        return CopModel(**{**self.__dict__, "selected": selected})

    def cop_table(self, t_out: np.ndarray | float) -> np.ndarray:
        """COP of the table at *t_out*; a bin without training hours takes the nearest bin."""
        return _lookup(self.table, self.table_bin_k, t_out)

    def cop_lift(self, lift: np.ndarray | float) -> np.ndarray:
        """COP of the lift table at *lift* (``T_flow - T_out``), with the same nearest-bin rule."""
        if not self.lift_table:
            raise CopError("the model has no lift table")
        return _lookup(self.lift_table, self.lift_bin_k, lift)

    def predict_regression(
        self,
        q: np.ndarray | float,
        t_flow: np.ndarray | float,
        t_out: np.ndarray | float,
        starts: np.ndarray | float,
        hours_on: np.ndarray | float,
    ) -> np.ndarray:
        a, b, c, d, e = self.coefficients
        q_, tf, to, st, ho = np.broadcast_arrays(
            *(np.asarray(x, dtype=np.float64) for x in (q, t_flow, t_out, starts, hours_on))
        )
        per_heat = a + b * (tf - to) + c * np.maximum(0.0, self.knee_c - to)
        return np.maximum(0.0, q_ * per_heat + d * st + e * ho)

    def predict_table(self, q: np.ndarray | float, t_out: np.ndarray | float) -> np.ndarray:
        q_, to = np.broadcast_arrays(
            np.asarray(q, dtype=np.float64), np.asarray(t_out, dtype=np.float64)
        )
        return q_ / self.cop_table(to)

    def predict_lift_table(
        self, q: np.ndarray | float, t_flow: np.ndarray | float, t_out: np.ndarray | float
    ) -> np.ndarray:
        q_, tf, to = np.broadcast_arrays(
            *(np.asarray(x, dtype=np.float64) for x in (q, t_flow, t_out))
        )
        return q_ / self.cop_lift(tf - to)


def _lookup(table: Sequence[tuple[float, float, int]], bin_k: float, x) -> np.ndarray:
    """The table value of the bin holding *x*; a value in an unpopulated bin reads the bin whose
    centre is nearest (a tie: the lower bin), ``NaN`` stays ``NaN``."""
    t = np.asarray(x, dtype=np.float64)
    lo = np.array([row[0] for row in table])
    cop = np.array([row[1] for row in table])
    centre = lo + 0.5 * bin_k
    flat = t.reshape(-1)
    want = np.floor(flat / bin_k) * bin_k
    pos = np.searchsorted(lo, want)
    exact = (pos < lo.size) & (lo[np.clip(pos, 0, lo.size - 1)] == want)
    nearest = np.abs(flat[:, None] - centre[None, :]).argmin(axis=1)
    idx = np.where(exact, np.clip(pos, 0, lo.size - 1), nearest)
    out = cop[idx]
    out[~np.isfinite(flat)] = np.nan
    return out.reshape(t.shape)


def predict_electricity(
    model: CopModel,
    q: np.ndarray | float,
    t_flow: np.ndarray | float,
    t_out: np.ndarray | float,
    starts: np.ndarray | float = 0.0,
    hours_on: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Electricity (kWh) of the heat pump for *q* kWh of heat, as the selected model predicts it.

    The inputs broadcast together: *q* in kWh, *t_flow* and *t_out* in C, *starts* the compressor
    starts and *hours_on* its on-time in hours, all over the same period.  The ``T_out`` table
    reads only *q* and *t_out*, the lift table *q* and ``t_flow - t_out``.  The regression is
    clipped at zero.
    """
    if model.selected == "table":
        return model.predict_table(q, t_out)
    if model.selected == "lift_table":
        return model.predict_lift_table(q, t_flow, t_out)
    return model.predict_regression(q, t_flow, t_out, starts, hours_on)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _design(rows: HourlyRows, knee_c: float) -> np.ndarray:
    return np.column_stack(
        [
            rows.q,
            rows.q * (rows.t_flow - rows.t_out),
            rows.q * np.maximum(0.0, knee_c - rows.t_out),
            rows.starts,
            rows.hours_on,
        ]
    )


def fit_regression(rows: HourlyRows, knee_c: float) -> dict[str, Any]:
    """WLS (weight ``Q``) of the hourly electricity on the five terms.

    Returns the coefficients ``a..e``, their standard errors, the weighted residual scale, the
    condition number of the weighted design and the row count.
    """
    if rows.n < 20:
        raise CopError(f"only {rows.n} training hours: too few for the regression")
    x = _design(rows, knee_c)
    sw = np.sqrt(rows.q)
    xw, yw = x * sw[:, None], rows.e * sw
    beta, _res, rank, sing = np.linalg.lstsq(xw, yw, rcond=None)
    if rank < x.shape[1]:
        raise CopError("the regression design is rank deficient")
    dof = max(rows.n - x.shape[1], 1)
    resid = yw - xw @ beta
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.inv(xw.T @ xw)
    return {
        "coefficients": tuple(float(v) for v in beta),
        "std_errors": tuple(float(v) for v in np.sqrt(np.diag(cov))),
        "weighted_residual_scale": math.sqrt(sigma2),
        "condition_number": float(sing[0] / sing[-1]),
        "n_rows": rows.n,
    }


def _bin_table(
    x: np.ndarray, rows: HourlyRows, bin_k: float, what: str
) -> tuple[tuple[float, float, int], ...]:
    if rows.n == 0:
        raise CopError(f"no training hours for the COP {what} table")
    lower = np.floor(x / bin_k) * bin_k
    table: list[tuple[float, float, int]] = []
    for lo in np.unique(lower):
        sel = lower == lo
        if int(sel.sum()) >= MIN_BIN_ROWS:
            table.append((float(lo), float(rows.q[sel].sum() / rows.e[sel].sum()), int(sel.sum())))
    if not table:
        raise CopError(f"no {what} bin has {MIN_BIN_ROWS} training hours")
    return tuple(table)


def fit_table(rows: HourlyRows, bin_k: float) -> tuple[tuple[float, float, int], ...]:
    """The COP table: the heat-over-electricity ratio of the training hours of each ``T_out`` bin.

    A bin needs :data:`MIN_BIN_ROWS` hours of its own to appear; a sparser one is left out and
    its temperatures read the nearest populated bin.
    """
    return _bin_table(rows.t_out, rows, bin_k, "T_out")


def fit_lift_table(rows: HourlyRows, bin_k: float) -> tuple[tuple[float, float, int], ...]:
    """The COP table by lift ``T_flow - T_out`` (heat-weighted flow of the hour), with the same
    minimum count and nearest-bin rule as :func:`fit_table`."""
    return _bin_table(rows.t_flow - rows.t_out, rows, bin_k, "lift")


def _record(
    rows: HourlyRows,
    config: ClimateConfig,
    fit: Mapping[str, Any],
    table: tuple[tuple[float, float, int], ...],
    lift_table: tuple[tuple[float, float, int], ...],
    *,
    train_seasons: Sequence[str],
    feature_set_id: str,
    selected: str,
) -> CopModel:
    lo, hi = (float(v) for v in config.cop.cop_range)
    return CopModel(
        version=COP_MODEL_VERSION,
        selected=selected,
        coefficients=tuple(fit["coefficients"]),  # type: ignore[arg-type]
        knee_c=float(config.cop.defrost_knee_c),
        table_bin_k=float(config.cop.table_bin_k),
        table=table,
        cop_range=(lo, hi),
        train_seasons=tuple(train_seasons),
        n_train_rows=rows.n,
        feature_set_id=feature_set_id,
        config_sha256=config.sha256,
        lift_bin_k=float(config.cop.lift_bin_k),
        lift_table=lift_table,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _score(
    rows: HourlyRows, predicted: np.ndarray, config: ClimateConfig, bin_k: float
) -> dict[str, Any]:
    """Daily MAPE, season error and the residuals by ``T_out`` bin of one prediction."""
    days = np.unique(rows.day)
    meas, pred = [], []
    for d in days:
        sel = rows.day == d
        if int(sel.sum()) >= MIN_DAY_HOURS:
            meas.append(float(rows.e[sel].sum()))
            pred.append(float(predicted[sel].sum()))
    meas_a, pred_a = np.array(meas), np.array(pred)
    if meas_a.size == 0 or meas_a.sum() <= 0.0:
        raise CopError("no validation day has enough eligible hours")
    daily_mape = float(np.mean(np.abs(pred_a - meas_a) / meas_a))
    season_error = float((pred_a.sum() - meas_a.sum()) / meas_a.sum())

    lower = np.floor(rows.t_out / bin_k) * bin_k
    by_bin = []
    for lo in np.unique(lower):
        sel = lower == lo
        m, p = float(rows.e[sel].sum()), float(predicted[sel].sum())
        by_bin.append(
            {
                "t_out_from_c": float(lo),
                "hours": int(sel.sum()),
                "measured_kwh": m,
                "predicted_kwh": p,
                "relative_error": (p - m) / m if m > 0.0 else None,
                "mean_hourly_residual_kwh": float((predicted[sel] - rows.e[sel]).mean()),
            }
        )
    return {
        "days": int(meas_a.size),
        "hours": rows.n,
        "daily_mape": daily_mape,
        "season_error": season_error,
        "passes": bool(
            daily_mape <= config.cop.daily_mape and abs(season_error) <= config.cop.season_error
        ),
        "residuals_by_t_out": by_bin,
    }


def validate(model: CopModel, rows: HourlyRows, config: ClimateConfig) -> dict[str, Any]:
    """Score the regression and both tables on held-out *rows*, given their measured ``Q``."""
    reg = model.predict_regression(rows.q, rows.t_flow, rows.t_out, rows.starts, rows.hours_on)
    tab = model.predict_table(rows.q, rows.t_out)
    bin_k = float(config.cop.table_bin_k)
    out = {
        "regression": _score(rows, reg, config, bin_k),
        "table": _score(rows, tab, config, bin_k),
    }
    if model.lift_table:
        lift = model.predict_lift_table(rows.q, rows.t_flow, rows.t_out)
        out["lift_table"] = _score(rows, lift, config, bin_k)
    return out


def select_model(validation: Mapping[str, Any]) -> tuple[str, str]:
    """The candidate with the lowest held-out daily MAPE (a tie: the earlier of
    :data:`CANDIDATES`).  Reported beside the predeclared ``cop.model``, never replacing it.

    Returns ``(best, reason)``.  Whether a model also meets the acceptance limits is a separate
    fact (``passes``): losing the comparison and failing the limits are reported apart.
    """
    names = [c for c in CANDIDATES if c in validation]
    best = min(names, key=lambda c: (validation[c]["daily_mape"], names.index(c)))
    return best, f"{best} has the lowest held-out daily MAPE of {names}"


def _sanity(fit: Mapping[str, Any], rows: HourlyRows, knee_c: float) -> dict[str, Any]:
    terms = {}
    for name, value, se in zip(COEFFICIENTS, fit["coefficients"], fit["std_errors"], strict=True):
        lo, hi, meaning = SANITY_RANGES[name]
        terms[name] = {
            "value": value,
            "std_error": se,
            "plausible_range": [lo, hi],
            "ok": bool(lo <= value <= hi),
            "meaning": meaning,
        }
    a, b, c = fit["coefficients"][:3]
    flow = float(np.median(rows.t_flow))
    curve = []
    for t_out in CURVE_T_OUT:
        per_heat = a + b * (flow - t_out) + c * max(0.0, knee_c - t_out)
        curve.append(
            {"t_out_c": t_out, "cop_heat_terms_only": (1.0 / per_heat) if per_heat > 0 else None}
        )
    return {
        "terms": terms,
        "all_ok": all(t["ok"] for t in terms.values()),
        "reference_t_flow_c": flow,
        "implied_cop_curve": curve,
    }


def dhw_cop(data: ClimateData, seasons: Sequence[str]) -> dict[str, Any]:
    """Hot-water COP per season (information only: DHW is fixed in every policy).

    The thermal over the electrical energy of the slots that produced hot water, summed per
    season over the fit months, from slots where both are known.
    """
    e, q = data.columns["hp_dhw_kwh"], data.columns["hp_dhw_out_kwh"]
    out: dict[str, Any] = {}
    for season in seasons:
        m = data.season_slots(season, months="fit") & data.dhw & np.isfinite(e) & np.isfinite(q)
        out[season] = {
            "slots": int(m.sum()),
            "cop": float(q[m].sum() / e[m].sum()) if m.any() and e[m].sum() > 0 else None,
        }
    return out


def fit_and_validate(data: ClimateData, config: ClimateConfig) -> dict[str, Any]:
    """Fit and validate the electricity model on both folds of ``config.folds``.

    Returns a JSON-ready report: the hourly row accounting, and per fold the fitted
    :class:`CopModel` (``model``), the regression terms with their physical sanity check, the
    validation of the regression and of the table, the explicit selection, and the hot-water COP.
    """
    rows = hourly_rows(data, config)
    folds = []
    for fold in config.folds:
        train_seasons = [str(s) for s in fold["train"]]
        test_season = str(fold["test"])
        fit_mask = np.zeros(rows.n, dtype=bool)
        for s in train_seasons:
            fit_mask |= (rows.season == s) & rows.fit_month
        test_mask = (rows.season == test_season) & rows.gate_month
        train, test = rows.select(fit_mask), rows.select(test_mask)

        fit = fit_regression(train, float(config.cop.defrost_knee_c))
        table = fit_table(train, float(config.cop.table_bin_k))
        lift_table = fit_lift_table(train, float(config.cop.lift_bin_k))
        selected = str(config.cop.model)
        model = _record(
            train,
            config,
            fit,
            table,
            lift_table,
            train_seasons=train_seasons,
            feature_set_id=data.feature_set_id,
            selected=selected,
        )
        validation = validate(model, test, config)
        best, reason = select_model(validation)
        folds.append(
            {
                "train": train_seasons,
                "test": test_season,
                "train_hours": train.n,
                "test_hours": test.n,
                "model": model.to_dict(),
                "model_id": model.model_id,
                "selected": selected,
                "selection_reason": "predeclared in config (cop.model)",
                "held_out_best": best,
                "held_out_best_reason": reason,
                "validation": validation,
                "limits": {
                    "daily_mape": config.cop.daily_mape,
                    "season_error": config.cop.season_error,
                },
                "selected_passes": validation[selected]["passes"],
                "regression_fit": {
                    k: list(v) if isinstance(v, tuple) else v
                    for k, v in fit.items()
                    if k != "coefficients"
                },
                "sanity": _sanity(fit, train, float(config.cop.defrost_knee_c)),
            }
        )
    return {
        "model_version": COP_MODEL_VERSION,
        "feature_set_id": data.feature_set_id,
        "dataset_id": data.dataset_id,
        "config_sha256": config.sha256,
        "hourly_rows": rows.exclusions,
        "folds": folds,
        "dhw_cop_information_only": dhw_cop(data, ("S1", "S2", "S3")),
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:+.1f}%"


def render_markdown(report: Mapping[str, Any]) -> str:
    """A readable summary of :func:`fit_and_validate`; its figures are private (data dir only)."""
    lines = [
        "# Heat-pump electricity (COP) model (private: numbers from the real installation)",
        "",
        f"- model version {report['model_version']}, feature set `{report['feature_set_id']}`, "
        f"dataset `{report['dataset_id']}`, config `{report['config_sha256']}`",
        "- hourly rows: " + ", ".join(f"{k} {v}" for k, v in report["hourly_rows"].items()),
        "",
    ]
    for fold in report["folds"]:
        limits = fold["limits"]
        lines += [
            f"## Fit {'+'.join(fold['train'])}, test {fold['test']}",
            "",
            f"- model id `{fold['model_id']}`: **{fold['selected']}** selected "
            f"({fold['selection_reason']}); selected model passes the limits: "
            f"{fold['selected_passes']}; lowest held-out MAPE: {fold['held_out_best']}",
            f"- training hours {fold['train_hours']}, validation hours {fold['test_hours']}",
            f"- limits: daily MAPE <= {limits['daily_mape']:.0%}, "
            f"|season error| <= {limits['season_error']:.0%}",
            "",
            "| model | days | daily MAPE | season error | passes |",
            "|---|---|---|---|---|",
        ]
        for name in [c for c in CANDIDATES if c in fold["validation"]]:
            v = fold["validation"][name]
            lines.append(
                f"| {name} | {v['days']} | {100 * v['daily_mape']:.1f}% | "
                f"{_pct(v['season_error'])} | {v['passes']} |"
            )
        lines += ["", "Terms (physical sanity check, not a gate):", ""]
        lines += ["| term | value | std error | plausible range | ok |", "|---|---|---|---|---|"]
        for name, t in fold["sanity"]["terms"].items():
            lo, hi = t["plausible_range"]
            lines.append(
                f"| {name} | {t['value']:.4g} | {t['std_error']:.2g} | {lo} to {hi} | {t['ok']} |"
            )
        lines += ["", "Residuals by outdoor temperature bin (predicted over measured - 1):", ""]
        names = [c for c in CANDIDATES if c in fold["validation"]]
        lines += [
            "| T_out from (C) | hours | " + " | ".join(names) + " |",
            "|---|---|" + "---|" * len(names),
        ]
        by_name = {
            n: {b["t_out_from_c"]: b for b in fold["validation"][n]["residuals_by_t_out"]}
            for n in names
        }
        for b in fold["validation"]["regression"]["residuals_by_t_out"]:
            cells = " | ".join(_pct(by_name[n][b["t_out_from_c"]]["relative_error"]) for n in names)
            lines.append(f"| {b['t_out_from_c']:g} | {b['hours']} | {cells} |")
        lines.append("")
    lines += ["## Hot water COP (information only)", ""]
    for season, v in report["dhw_cop_information_only"].items():
        cop = "n/a" if v["cop"] is None else f"{v['cop']:.2f}"
        lines.append(f"- {season}: {cop} over {v['slots']} slots")
    lines.append("")
    return "\n".join(lines)
