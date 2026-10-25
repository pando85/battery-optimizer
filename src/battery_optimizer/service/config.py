"""Service configuration: ``config/service.yaml`` plus environment overrides.

Endpoints and credentials never live in the YAML: ``BATTERY_OPTIMIZER_HA_URL``,
``BATTERY_OPTIMIZER_HA_TOKEN`` (or ``BATTERY_OPTIMIZER_HA_TOKEN_FILE``, the form a
mounted Kubernetes secret takes) and ``BATTERY_OPTIMIZER_PROMETHEUS_URL``.  The
token is held in a :class:`Secret` whose ``repr`` never shows the value.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml

from battery_optimizer._env import get_env

__all__ = [
    "BATTERY_ROLES",
    "ENTITY_ROLES",
    "SOC_SOURCES",
    "EntityMap",
    "HomeAssistantSettings",
    "ObservationSettings",
    "PlanningSettings",
    "PrometheusSettings",
    "Secret",
    "ServiceConfig",
    "load_service_config",
    "resolve_service_config_path",
]

#: Roles the service understands.  Each maps to at most one HA entity ID.
ENTITY_ROLES: tuple[str, ...] = (
    "grid_power",
    "grid_energy",
    "price",
    "battery_soc",
    "battery_power",
    "battery_charge_power",
    "battery_discharge_power",
    "battery_charge_energy",
    "battery_discharge_energy",
)

#: The roles that describe the physical battery.
BATTERY_ROLES: tuple[str, ...] = tuple(r for r in ENTITY_ROLES if r.startswith("battery_"))

SOC_SOURCES: tuple[str, ...] = ("entity", "simulated")

DEFAULT_PORT = 8080


class Secret:
    """A string that never appears in ``repr``/``str``/logs."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "Secret('***')"

    __str__ = __repr__

    def __bool__(self) -> bool:
        return bool(self._value)


@dataclass(frozen=True)
class EntityMap:
    """Role -> entity ID mapping plus read-only context entities."""

    roles: Mapping[str, str | None]
    observed: tuple[str, ...] = ()

    def get(self, role: str) -> str | None:
        if role not in ENTITY_ROLES:
            raise KeyError(f"unknown entity role {role!r}")
        return self.roles.get(role)

    def configured(self) -> dict[str, str]:
        """Roles that have an entity, in canonical role order."""
        return {r: e for r in ENTITY_ROLES if (e := self.roles.get(r))}

    def all_entities(self) -> tuple[str, ...]:
        """Every entity the service subscribes to (roles first, then observed)."""
        seen: dict[str, None] = {}
        for entity in (*self.configured().values(), *self.observed):
            seen.setdefault(entity, None)
        return tuple(seen)

    def role_of(self, entity_id: str) -> str | None:
        for role, entity in self.configured().items():
            if entity == entity_id:
                return role
        return None

    @property
    def has_battery(self) -> bool:
        return any(self.roles.get(r) for r in BATTERY_ROLES)


@dataclass(frozen=True)
class HomeAssistantSettings:
    url: str | None
    token: Secret | None = field(repr=False, default=None)
    request_timeout_s: float = 15.0
    reconnect_initial_delay_s: float = 1.0
    reconnect_max_delay_s: float = 60.0
    reconnect_jitter_frac: float = 0.2
    heartbeat_interval_s: float = 30.0
    heartbeat_timeout_s: float = 10.0
    poll_interval_s: float = 60.0


@dataclass(frozen=True)
class PrometheusSettings:
    url: str | None
    timeout_s: float = 60.0
    history_days: int = 28


@dataclass(frozen=True)
class PlanningSettings:
    soc_source: str = "simulated"
    interval_seconds: int = 900
    solve_lead_s: float = 120.0
    solve_timeout_s: float = 60.0
    solver_time_limit_s: float = 20.0
    plan_ttl_slots: int = 2
    min_horizon_slots: int = 4
    max_horizon_slots: int = 192
    forecast_window_days: int = 28
    terminal_value: str | float = "trailing_mean"
    throughput_cost_eur_per_kwh: float = 0.0
    replan_soc_change_frac: float = 0.05
    replan_debounce_s: float = 30.0
    replan_min_interval_s: float = 120.0


@dataclass(frozen=True)
class ObservationSettings:
    slot_close_grace_s: float = 20.0
    counter_max_age_s: float = 300.0


@dataclass(frozen=True)
class ServiceConfig:
    """Fully resolved service configuration."""

    entities: EntityMap
    home_assistant: HomeAssistantSettings
    prometheus: PrometheusSettings
    planning: PlanningSettings
    observation: ObservationSettings
    freshness_s: Mapping[str, float]
    timezone: str = "Europe/Madrid"
    mode: str = "shadow"
    source_path: str | None = None

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    def freshness_for(self, role: str) -> float:
        """Maximum age for a role; power roles share ``battery_power``'s limit."""
        if role in self.freshness_s:
            return float(self.freshness_s[role])
        if role in ("battery_charge_power", "battery_discharge_power"):
            return float(self.freshness_s.get("battery_power", 300.0))
        if role in ("battery_charge_energy", "battery_discharge_energy"):
            return float(self.freshness_s.get("grid_energy", 900.0))
        return 900.0

    def public_dict(self) -> dict[str, Any]:
        """Configuration without secrets, for ``/status`` and hashing."""
        ha = asdict(self.home_assistant)
        ha.pop("token", None)
        ha["token_configured"] = bool(self.home_assistant.token)
        return {
            "mode": self.mode,
            "timezone": self.timezone,
            "entities": {
                "roles": dict(self.entities.roles),
                "observed": list(self.entities.observed),
            },
            "home_assistant": ha,
            "prometheus": asdict(self.prometheus),
            "planning": asdict(self.planning),
            "observation": asdict(self.observation),
            "freshness_s": dict(self.freshness_s),
            "source_path": self.source_path,
        }

    def config_hash(self) -> str:
        """SHA-256 of the planning-relevant configuration (URLs excluded)."""
        public = self.public_dict()
        public["home_assistant"].pop("url", None)
        public["prometheus"].pop("url", None)
        public.pop("source_path", None)
        text = json.dumps(public, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def resolve_service_config_path(cli_arg: str | None = None) -> Path:
    """``--service-config`` > ``BATTERY_OPTIMIZER_SERVICE_CONFIG`` > ``config/service.yaml``."""
    if cli_arg:
        return Path(cli_arg).expanduser().resolve()
    env = get_env("BATTERY_OPTIMIZER_SERVICE_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.cwd() / "config" / "service.yaml").resolve()


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"service config: {name} must be a mapping, got {type(value).__name__}")
    return value


def _positive(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"service config: {name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"service config: {name} must be a number, got {value!r}") from exc
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"service config: {name} must be {'>=' if allow_zero else '>'} 0")
    return number


def _int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"service config: {name} must be an integer >= {minimum}, got {value!r}")
    return value


def _read_token() -> Secret | None:
    token = get_env("BATTERY_OPTIMIZER_HA_TOKEN")
    if token:
        return Secret(token.strip())
    token_file = get_env("BATTERY_OPTIMIZER_HA_TOKEN_FILE")
    if token_file:
        text = Path(token_file).expanduser().read_text().strip()
        return Secret(text) if text else None
    return None


def _parse_entities(raw: Mapping[str, Any]) -> EntityMap:
    unknown = sorted(k for k in raw if k not in (*ENTITY_ROLES, "observed"))
    if unknown:
        raise ValueError(f"service config: unknown entity roles {unknown}")
    roles: dict[str, str | None] = {}
    for role in ENTITY_ROLES:
        value = raw.get(role)
        if value is not None and (not isinstance(value, str) or "." not in value):
            raise ValueError(f"service config: entities.{role} must be an entity ID or null")
        roles[role] = value
    for required in ("grid_power", "price"):
        if not roles[required]:
            raise ValueError(f"service config: entities.{required} is required")
    observed_raw = raw.get("observed") or []
    if not isinstance(observed_raw, list) or not all(isinstance(e, str) for e in observed_raw):
        raise ValueError("service config: entities.observed must be a list of entity IDs")
    if roles["battery_power"] and (
        roles["battery_charge_power"] or roles["battery_discharge_power"]
    ):
        raise ValueError(
            "service config: map either battery_power (signed) or the "
            "battery_charge_power/battery_discharge_power pair, not both"
        )
    if bool(roles["battery_charge_power"]) != bool(roles["battery_discharge_power"]):
        raise ValueError(
            "service config: battery_charge_power and battery_discharge_power go together"
        )
    return EntityMap(roles=roles, observed=tuple(observed_raw))


def load_service_config(path: Path | None = None) -> ServiceConfig:
    """Load the YAML at *path* (or the resolved default) and apply env overrides."""
    path = path or resolve_service_config_path()
    if not path.is_file():
        raise ValueError(
            f"service config not found: {path} (set BATTERY_OPTIMIZER_SERVICE_CONFIG or run "
            "from a directory containing config/service.yaml)"
        )
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"service config must be a mapping, got {type(raw).__name__}")
    mode = raw.get("mode", "shadow")
    if mode != "shadow":
        raise ValueError(
            f"service config: mode={mode!r} is not supported — only 'shadow' exists; live "
            "actuation is Phase D and is not built"
        )
    timezone = str(raw.get("timezone", "Europe/Madrid"))
    ZoneInfo(timezone)  # validate early

    ha_raw = _section(raw, "home_assistant")
    reconnect = _section(ha_raw, "reconnect")
    heartbeat = _section(ha_raw, "heartbeat")
    ha = HomeAssistantSettings(
        url=(get_env("BATTERY_OPTIMIZER_HA_URL") or "").rstrip("/") or None,
        token=_read_token(),
        request_timeout_s=_positive(ha_raw.get("request_timeout_s", 15), "request_timeout_s"),
        reconnect_initial_delay_s=_positive(reconnect.get("initial_delay_s", 1), "initial_delay_s"),
        reconnect_max_delay_s=_positive(reconnect.get("max_delay_s", 60), "max_delay_s"),
        reconnect_jitter_frac=_positive(
            reconnect.get("jitter_frac", 0.2), "jitter_frac", allow_zero=True
        ),
        heartbeat_interval_s=_positive(heartbeat.get("interval_s", 30), "heartbeat.interval_s"),
        heartbeat_timeout_s=_positive(heartbeat.get("timeout_s", 10), "heartbeat.timeout_s"),
        poll_interval_s=_positive(ha_raw.get("poll_interval_s", 60), "poll_interval_s"),
    )
    prom_raw = _section(raw, "prometheus")
    prom = PrometheusSettings(
        url=(get_env("BATTERY_OPTIMIZER_PROMETHEUS_URL") or "").rstrip("/") or None,
        timeout_s=_positive(prom_raw.get("timeout_s", 60), "prometheus.timeout_s"),
        history_days=_int(prom_raw.get("history_days", 28), "prometheus.history_days"),
    )

    pl_raw = _section(raw, "planning")
    replan = _section(pl_raw, "replan")
    soc_source = pl_raw.get("soc_source", "simulated")
    if soc_source not in SOC_SOURCES:
        raise ValueError(f"service config: planning.soc_source must be one of {SOC_SOURCES}")
    terminal_value = pl_raw.get("terminal_value", "trailing_mean")
    if terminal_value != "trailing_mean":
        if isinstance(terminal_value, bool) or not isinstance(terminal_value, (int, float)):
            raise ValueError("service config: terminal_value must be 'trailing_mean' or a number")
        if not math.isfinite(float(terminal_value)):
            raise ValueError("service config: terminal_value must be finite")
        terminal_value = float(terminal_value)
    interval_seconds = _int(pl_raw.get("interval_seconds", 900), "interval_seconds", 60)
    if 3600 % interval_seconds:
        raise ValueError("service config: interval_seconds must divide an hour")
    planning = PlanningSettings(
        soc_source=soc_source,
        interval_seconds=interval_seconds,
        solve_lead_s=_positive(pl_raw.get("solve_lead_s", 120), "solve_lead_s"),
        solve_timeout_s=_positive(pl_raw.get("solve_timeout_s", 60), "solve_timeout_s"),
        solver_time_limit_s=_positive(pl_raw.get("solver_time_limit_s", 20), "solver_time_limit_s"),
        plan_ttl_slots=_int(pl_raw.get("plan_ttl_slots", 2), "plan_ttl_slots"),
        min_horizon_slots=_int(pl_raw.get("min_horizon_slots", 4), "min_horizon_slots"),
        max_horizon_slots=_int(pl_raw.get("max_horizon_slots", 192), "max_horizon_slots"),
        forecast_window_days=_int(pl_raw.get("forecast_window_days", 28), "forecast_window_days"),
        terminal_value=terminal_value,
        throughput_cost_eur_per_kwh=_positive(
            pl_raw.get("throughput_cost_eur_per_kwh", 0.0), "throughput_cost", allow_zero=True
        ),
        replan_soc_change_frac=_positive(
            replan.get("soc_change_frac", 0.05), "replan.soc_change_frac"
        ),
        replan_debounce_s=_positive(
            replan.get("debounce_s", 30), "replan.debounce_s", allow_zero=True
        ),
        replan_min_interval_s=_positive(
            replan.get("min_interval_s", 120), "replan.min_interval_s", allow_zero=True
        ),
    )
    if planning.solve_timeout_s >= planning.solve_lead_s:
        raise ValueError(
            "service config: solve_timeout_s must be below solve_lead_s, or a solve could "
            "finish after its first slot began"
        )
    if planning.solver_time_limit_s > planning.solve_timeout_s:
        raise ValueError("service config: solver_time_limit_s must not exceed solve_timeout_s")
    if planning.min_horizon_slots > planning.max_horizon_slots:
        raise ValueError("service config: min_horizon_slots must not exceed max_horizon_slots")

    entities = _parse_entities(_section(raw, "entities"))
    if soc_source == "entity" and not entities.get("battery_soc"):
        raise ValueError(
            "service config: planning.soc_source=entity needs entities.battery_soc; use "
            "soc_source=simulated until the battery is integrated in Home Assistant"
        )

    obs_raw = _section(raw, "observation")
    observation = ObservationSettings(
        slot_close_grace_s=_positive(obs_raw.get("slot_close_grace_s", 20), "slot_close_grace_s"),
        counter_max_age_s=_positive(obs_raw.get("counter_max_age_s", 300), "counter_max_age_s"),
    )
    fresh_raw = _section(raw, "freshness_s")
    freshness = {str(k): _positive(v, f"freshness_s.{k}") for k, v in fresh_raw.items()}
    return ServiceConfig(
        entities=entities,
        home_assistant=ha,
        prometheus=prom,
        planning=planning,
        observation=observation,
        freshness_s=freshness,
        timezone=timezone,
        mode=mode,
        source_path=str(path),
    )


def resolve_port(cli_arg: int | None = None) -> int:
    """``--port`` > ``BATTERY_OPTIMIZER_PORT`` > 8080."""
    if cli_arg is not None:
        return cli_arg
    env = get_env("BATTERY_OPTIMIZER_PORT")
    if env:
        port = int(env)
        if not 0 < port < 65536:
            raise ValueError(f"BATTERY_OPTIMIZER_PORT out of range: {port}")
        return port
    return DEFAULT_PORT
