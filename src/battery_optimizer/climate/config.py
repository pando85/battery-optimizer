"""Strict loader of ``config/climate.yaml`` (heat-v1 comfort band, filters, envelope, gate).

The file is committed before any model result exists, so the loader refuses anything that is not
exactly the declared shape: an unknown, missing or duplicated key, a wrong type, a boolean where a
number belongs, a non-finite number, a range that is empty or reversed, or a cross-field
inconsistency (comfort rooms, seasons, folds).  The result is a read-only tree whose sections are
reachable by attribute (``cfg.comfort.low_c``) and whose SHA-256 identifies the file in every
result that used it.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator, Mapping
from datetime import date
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

__all__ = [
    "CONFIG_VERSION",
    "ClimateConfig",
    "ClimateConfigError",
    "Section",
    "load_climate_config",
]

CONFIG_VERSION = 2
KNOWN_ROOMS = ("bedroom", "hall_kitchen", "office_t", "office_a")


class ClimateConfigError(ValueError):
    """The climate configuration is malformed; the message names the offending key."""


class Section(Mapping[str, Any]):
    """An immutable mapping that also answers attribute access."""

    def __init__(self, items: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_items", dict(items))

    def __getitem__(self, key: str) -> Any:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("climate configuration is read-only")

    def to_dict(self) -> dict[str, Any]:
        return {k: v.to_dict() if isinstance(v, Section) else v for k, v in self._items.items()}


Spec = Callable[[Any, str], Any]


def _fail(path: str, message: str) -> ClimateConfigError:
    return ClimateConfigError(f"climate config: {path}: {message}")


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _fail(path, f"expected a number, got {value!r}")
    if not math.isfinite(value):
        raise _fail(path, "must be finite")
    return float(value)


def num(lo: float | None = None, hi: float | None = None) -> Spec:
    def check(value: Any, path: str) -> float:
        x = _number(value, path)
        if lo is not None and x < lo:
            raise _fail(path, f"{x} is below the minimum {lo}")
        if hi is not None and x > hi:
            raise _fail(path, f"{x} is above the maximum {hi}")
        return x

    return check


def integer(lo: int | None = None, hi: int | None = None) -> Spec:
    def check(value: Any, path: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise _fail(path, f"expected an integer, got {value!r}")
        if lo is not None and value < lo:
            raise _fail(path, f"{value} is below the minimum {lo}")
        if hi is not None and value > hi:
            raise _fail(path, f"{value} is above the maximum {hi}")
        return value

    return check


def boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise _fail(path, f"expected true or false, got {value!r}")
    return value


def const(expected: bool | float) -> Spec:
    """A value fixed by the preregistration: any other value is an error."""

    def check(value: Any, path: str) -> bool | float:
        if isinstance(expected, bool):
            ok = isinstance(value, bool) and value is expected
        else:
            ok = (
                not isinstance(value, bool) and isinstance(value, int | float) and value == expected
            )
        if not ok:
            raise _fail(path, f"is fixed by the preregistration at {expected!r}, got {value!r}")
        return expected

    return check


def timezone(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(path, f"expected a time zone name, got {value!r}")
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError, OSError) as exc:
        raise _fail(path, f"unknown time zone {value!r}") from exc
    return value


def text(*choices: str) -> Spec:
    def check(value: Any, path: str) -> str:
        if not isinstance(value, str) or not value:
            raise _fail(path, f"expected a non-empty string, got {value!r}")
        if choices and value not in choices:
            raise _fail(path, f"{value!r} is not one of {list(choices)}")
        return value

    return check


def day(value: Any, path: str) -> date:
    if isinstance(value, date) and not hasattr(value, "hour"):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
    raise _fail(path, f"expected an ISO date, got {value!r}")


def seq(item: Spec, *, min_len: int = 0, unique: bool = False) -> Spec:
    def check(value: Any, path: str) -> tuple[Any, ...]:
        if not isinstance(value, list):
            raise _fail(path, f"expected a list, got {value!r}")
        out = tuple(item(v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(out) < min_len:
            raise _fail(path, f"needs at least {min_len} entries")
        if unique and len(set(out)) != len(out):
            raise _fail(path, "entries must be unique")
        return out

    return check


def optional(spec: Spec) -> Spec:
    """*spec*, or ``null`` (a value not chosen yet: the code that needs it refuses to run)."""

    def check(value: Any, path: str) -> Any:
        return None if value is None else spec(value, path)

    return check


def bounds(lo: float, hi: float) -> Spec:
    """A ``[low, high]`` pair with ``low < high``, both inside ``[lo, hi]``."""
    one = num(lo, hi)

    def check(value: Any, path: str) -> tuple[float, float]:
        if not isinstance(value, list) or len(value) != 2:
            raise _fail(path, f"expected [low, high], got {value!r}")
        a, b = one(value[0], f"{path}[0]"), one(value[1], f"{path}[1]")
        if not a < b:
            raise _fail(path, f"low {a} must be below high {b}")
        return a, b

    return check


def table(key: Spec, value: Spec) -> Spec:
    """A mapping with arbitrary keys (checked by *key*) and values (checked by *value*)."""

    def check(raw: Any, path: str) -> Section:
        if not isinstance(raw, dict) or not raw:
            raise _fail(path, "expected a non-empty mapping")
        return Section({key(k, f"{path}.{k}"): value(v, f"{path}.{k}") for k, v in raw.items()})

    return check


def section(fields: Mapping[str, Spec]) -> Spec:
    """A mapping with exactly the *fields*: unknown and missing keys are errors."""

    def check(raw: Any, path: str) -> Section:
        if not isinstance(raw, dict):
            raise _fail(path, "expected a mapping")
        for k in raw:
            if k not in fields:
                raise _fail(f"{path}.{k}".lstrip("."), "unknown key")
        for k in fields:
            if k not in raw:
                raise _fail(f"{path}.{k}".lstrip("."), "missing key")
        return Section({k: fields[k](raw[k], f"{path}.{k}".lstrip(".")) for k in fields})

    return check


_ROOM = text(*KNOWN_ROOMS)
_FRACTION = num(0.0, 1.0)
_POSITIVE = num(1e-9)
_SEASON = section({"start": day, "end": day})
_HOURS = seq(integer(1, 48), min_len=1, unique=True)

SCHEMA: Spec = section(
    {
        "config_version": integer(CONFIG_VERSION, CONFIG_VERSION),
        "timezone": timezone,
        "comfort": section(
            {
                "band": text(),
                "low_c": num(-50, 60),
                "high_c": num(-50, 60),
                "rooms": seq(_ROOM, min_len=1, unique=True),
                "excluded_rooms": seq(_ROOM, unique=True),
            }
        ),
        "seasons": section(
            {
                "S1": _SEASON,
                "S2": _SEASON,
                "S3": _SEASON,
                "fit_months": seq(integer(1, 12), min_len=1, unique=True),
                "gate_months": seq(integer(1, 12), min_len=1, unique=True),
                "reserved_from": day,
                "training_excluded": seq(text("S1", "S2", "S3"), unique=True),
            }
        ),
        "folds": seq(
            section(
                {"train": seq(text("S1", "S2", "S3"), min_len=1), "test": text("S1", "S2", "S3")}
            ),
            min_len=1,
        ),
        "signals": section(
            {
                "step_seconds": integer(900, 900),
                "dhw": section(
                    {
                        "min_electric_kwh": _POSITIVE,
                        "min_thermal_kwh": _POSITIVE,
                        "force_share": num(1e-9, 1.0),
                    }
                ),
                "cooling": section({"min_electric_kwh": _POSITIVE, "min_thermal_kwh": _POSITIVE}),
                # inferred valve state (room below setpoint - margin), not the plant response
                "valve": section({"inferred_state_margin_k": num(0.0, 5.0)}),
                "t_out_debias": section(
                    {"min_off_minutes": integer(15, 240), "max_gap_hours": num(0.25, 48.0)}
                ),
                "counter_check": section({"min_daily_kwh": _POSITIVE}),
                "t_out_source": text("debiased", "fused"),
                "t_out_fill_unknown_from_fused": boolean,
                "t_out_fused": section(
                    {
                        "max_ref_gap_hours": num(0.25, 12.0),
                        "dark_solar_max": num(0.0, 0.5),
                        "fit_seasons": seq(text("S1", "S2", "S3"), min_len=1, unique=True),
                    }
                ),
            }
        ),
        "filters": section(
            {
                "drop_unknown_quality": boolean,
                "window_open": section({"drop_k": _POSITIVE, "slots": integer(1, 8)}),
                "valves_shut": section({"min_rooms_at_setpoint": integer(1, 3)}),
                "thermostat_outage": section({"windows": seq(section({"start": day, "end": day}))}),
                "drop_cooling_slots": boolean,
                "drop_rule_disabled_days": boolean,
            }
        ),
        "model": section(
            {
                "step_seconds": integer(900, 900),
                "window_hours": integer(24, 168),
                "forecast_origins_local_hours": seq(integer(0, 23), min_len=1, unique=True),
                "forecast_horizons_hours": _HOURS,
                "envelope": section(
                    {
                        "tau_h": bounds(0.0, 1e5),
                        "ua_kw_per_k": bounds(0.0, 100.0),
                        "c_slab_kwh_per_k": bounds(0.0, 1e4),
                        "k_kw_per_k": bounds(0.0, 100.0),
                        "q0_kw": bounds(0.0, 100.0),
                    }
                ),
                "slab_proxy": section(
                    {"min_flow_l_per_min": _POSITIVE, "min_minutes_into_run": integer(0, 240)}
                ),
                "response": section(
                    {"phi_default": _FRACTION, "plant_valve_cap_margin_k": num(0.0, 5.0)}
                ),
                "selection": section({"b_min_rmse_gain": _FRACTION, "horizons_hours": _HOURS}),
                "adaptive": section(
                    {
                        "window_days": integer(1, 366),
                        "refit_every_days": integer(1, 366),
                        "shrinkage": section(
                            {"A": optional(num(0.0, 1e12)), "B": optional(num(0.0, 1e12))}
                        ),
                        "selection": section(
                            {
                                "season": text("S1", "S2", "S3"),
                                "prior_until": day,
                                "grid": seq(num(0.0, 1e12), min_len=2, unique=True),
                            }
                        ),
                    }
                ),
            }
        ),
        "acceptance": section(
            {
                "q_driven": section(
                    {
                        "rmse_k": table(integer(1, 48), _POSITIVE),
                        "abs_bias_24h_k": _POSITIVE,
                        "min_gain_vs_baseline": _FRACTION,
                        "gain_horizons_hours": _HOURS,
                    }
                ),
                "u_driven": section(
                    {
                        "rmse_24h_k": _POSITIVE,
                        "daily_energy_mape": _FRACTION,
                        "season_energy_bias": _FRACTION,
                        "regime_energy_bias": _FRACTION,
                        "regime_min_heating_days": integer(1, 366),
                    }
                ),
                "legacy_replay": section(
                    {
                        "cost_rel_error": _FRACTION,
                        "electricity_rel_error": _FRACTION,
                        "kh_below_low_rel_error": _FRACTION,
                        "kh_below_low_abs_error": _POSITIVE,
                    }
                ),
                "parameters": section({"weekly_refit_ok_share": _FRACTION}),
                "max_iterations": const(1),
            }
        ),
        "cop": section(
            {
                "min_q_kwh": _POSITIVE,
                "cop_range": bounds(0.0, 20.0),
                "table_bin_k": _POSITIVE,
                "lift_bin_k": _POSITIVE,
                "model": text("regression", "table", "lift_table"),
                "defrost_knee_c": num(-30, 30),
                "daily_mape": _FRACTION,
                "season_error": _FRACTION,
            }
        ),
        "policies": section(
            {
                "heuristic": section(
                    {
                        "decision_minute": integer(0, 59),
                        "min_on_hours": integer(1, 24),
                        "min_off_hours": integer(1, 24),
                        "max_horizon_hours": integer(1, 48),
                        "never_heat_above_c": num(-50, 60),
                    }
                ),
                "perfect_foresight": section({"max_blocks_per_day": integer(1, 24)}),
                "setpoint_offset": section({"offset_k": _POSITIVE, "max_setpoint_c": num(-50, 60)}),
            }
        ),
        "gate": section(
            {
                "band": text(),
                "hard": section(
                    {
                        "max_kh_above_high": const(0.0),
                        "starts_per_day_ratio_max": num(1.0),
                        "max_starts_per_day": integer(1, 96),
                        "max_blocks_per_day": integer(1, 24),
                        "mismatch_run_keeps_sign": const(True),
                        "model_accepted": const(True),
                    }
                ),
                "cost": section(
                    {
                        "min_saving": num(1e-9, 1.0),
                        "bootstrap_lower_bound_above": const(0.0),
                        "saving_exceeds_replay_error": const(True),
                        "comfort_kh_tolerance_per_room": num(0.0),
                        "worst_excursion_floor_c": num(-50, 60),
                    }
                ),
                "comfort": section(
                    {
                        "min_kh_reduction": num(1e-9, 1.0),
                        "bootstrap_lower_bound_above": const(0.0),
                        "reduction_exceeds_replay_error": const(True),
                        "cost_point_delta_max": const(0.0),
                        "cost_bootstrap_upper_bound_max": num(0.0, 1.0),
                    }
                ),
                "model_error": section({"ood_share_warn": _FRACTION}),
                "statistics": section(
                    {"block_days": integer(1, 60), "confidence": num(0.5, 0.999)}
                ),
            }
        ),
    }
)


def _cross_checks(cfg: Section) -> None:
    comfort = cfg.comfort
    if not comfort.low_c < comfort.high_c:
        raise _fail("comfort", "low_c must be below high_c")
    if set(comfort.rooms) & set(comfort.excluded_rooms):
        raise _fail("comfort", "a room cannot be both a comfort room and excluded")
    if set(comfort.rooms) | set(comfort.excluded_rooms) != set(KNOWN_ROOMS):
        raise _fail("comfort", f"rooms and excluded_rooms must cover exactly {list(KNOWN_ROOMS)}")
    if cfg.gate.band != comfort.band:
        raise _fail("gate.band", "must match comfort.band")
    if cfg.signals.t_out_debias.min_off_minutes % (cfg.signals.step_seconds // 60):
        raise _fail("signals.t_out_debias.min_off_minutes", "must be a whole number of slots")
    if cfg.filters.valves_shut.min_rooms_at_setpoint > len(comfort.rooms):
        raise _fail("filters.valves_shut", "min_rooms_at_setpoint exceeds the comfort rooms")
    if "S3" in cfg.signals.t_out_fused.fit_seasons:
        raise _fail("signals.t_out_fused.fit_seasons", "must not include the held-out season S3")
    seasons = [cfg.seasons[name] for name in ("S1", "S2", "S3")]
    for name, s in zip(("S1", "S2", "S3"), seasons, strict=True):
        if s.start > s.end:
            raise _fail(f"seasons.{name}", "start is after end")
    for earlier, later in zip(seasons, seasons[1:], strict=False):
        if earlier.end >= later.start:
            raise _fail("seasons", "seasons must be chronological and disjoint")
    if seasons[-1].end >= cfg.seasons.reserved_from:
        raise _fail("seasons.reserved_from", "must be after the end of S3")
    if not set(cfg.seasons.gate_months) <= set(cfg.seasons.fit_months):
        raise _fail("seasons.gate_months", "must be a subset of fit_months")
    excluded = set(cfg.seasons.training_excluded)
    for i, fold in enumerate(cfg.folds):
        if fold.test in fold.train:
            raise _fail(f"folds[{i}]", "the test season is also a training season")
        if excluded & set(fold.train):
            raise _fail(f"folds[{i}]", "trains on a season in seasons.training_excluded")
    adaptive = cfg.model.adaptive
    sel = adaptive.selection
    if sel.season == "S3" or sel.season in excluded:
        raise _fail("model.adaptive.selection.season", "must be a training season, never S3")
    for i, w in enumerate(cfg.filters.thermostat_outage.windows):
        if w.start > w.end:
            raise _fail(f"filters.thermostat_outage.windows[{i}]", "start is after end")
    if cfg.acceptance.q_driven.rmse_k.keys() != {6, 12, 24}:
        raise _fail("acceptance.q_driven.rmse_k", "must give exactly the 6, 12 and 24 h horizons")
    if not set(cfg.model.selection.horizons_hours) <= set(cfg.model.forecast_horizons_hours):
        raise _fail("model.selection.horizons_hours", "must be forecast horizons")
    if not set(cfg.acceptance.q_driven.gain_horizons_hours) <= set(
        cfg.model.forecast_horizons_hours
    ):
        raise _fail("acceptance.q_driven.gain_horizons_hours", "must be forecast horizons")
    if cfg.gate.hard.max_blocks_per_day != cfg.policies.perfect_foresight.max_blocks_per_day:
        raise _fail("gate.hard.max_blocks_per_day", "must match the perfect-foresight blocks")


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that refuses a duplicated mapping key (PyYAML keeps the last silently)."""


def _construct_mapping(loader: _StrictLoader, node: yaml.MappingNode, deep: bool = False) -> Any:
    seen: set[Any] = set()
    for key_node, _value in node.value:
        key = loader.construct_object(key_node, deep=True)
        if key in seen:
            raise ClimateConfigError(f"climate config: duplicate key {key!r}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


class ClimateConfig(Section):
    """The validated configuration; ``sha256`` identifies the exact file."""

    sha256: str
    path: Path


def load_climate_config(path: str | Path = "config/climate.yaml") -> ClimateConfig:
    """Read, strictly validate and hash a climate configuration file."""
    p = Path(path).expanduser()
    try:
        payload = p.read_bytes()
    except OSError as exc:
        raise ClimateConfigError(f"climate config: cannot read {p}: {exc.strerror}") from exc
    try:
        raw = yaml.load(payload, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ClimateConfigError(f"climate config: {p}: not valid YAML ({exc})") from exc
    tree = SCHEMA(raw, "")
    cfg = ClimateConfig(tree)
    object.__setattr__(cfg, "sha256", hashlib.sha256(payload).hexdigest())
    object.__setattr__(cfg, "path", p)
    _cross_checks(cfg)
    return cfg
