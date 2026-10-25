"""Per-entity latest state with source-time ordering and freshness.

Home Assistant delivers the same logical state through several paths — the
``state_changed`` stream, the ``get_states`` snapshot taken after every
(re)connect, and the periodic REST refresh — and they can arrive interleaved.
Ordering is therefore decided on **source time**, never on arrival:

* ``last_updated`` newer than the stored one -> accepted;
* ``last_updated`` older -> rejected as out of order (a stale snapshot must never
  overwrite a newer event, which is exactly what happens when ``get_states``
  answers after an event already landed);
* ``last_updated`` equal and ``last_reported`` newer -> *refreshed*: same value,
  re-confirmed by HA later (HA fires no ``state_changed`` for an unchanged value,
  so this is how an idle SOC stays fresh);
* both equal -> duplicate.

Freshness is ``now - max(last_updated, last_reported)``: the age of the last
moment HA itself vouched for the value, not when we happened to receive it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from .timegrid import require_aware

__all__ = [
    "UNAVAILABLE_STATES",
    "EntityState",
    "EntityStore",
    "UpdateResult",
    "parse_ha_state",
]

UNAVAILABLE_STATES = frozenset({"unavailable", "unknown", "none", ""})


class UpdateResult(StrEnum):
    ACCEPTED = "accepted"
    REFRESHED = "refreshed"
    DUPLICATE = "duplicate"
    OUT_OF_ORDER = "out_of_order"
    IGNORED = "ignored"


@dataclass(frozen=True)
class EntityState:
    """One HA state object, with source timestamps kept separate from arrival."""

    entity_id: str
    state: str
    attributes: Mapping[str, Any]
    last_changed: datetime
    last_updated: datetime
    last_reported: datetime
    received_at: datetime | None = None

    @property
    def observed_at(self) -> datetime:
        """Latest instant HA vouched for this value."""
        return max(self.last_updated, self.last_reported)

    @property
    def available(self) -> bool:
        return self.state.strip().lower() not in UNAVAILABLE_STATES

    def numeric(self) -> float | None:
        """The state as a finite float, or ``None`` (unavailable / non-numeric)."""
        if not self.available:
            return None
        try:
            value = float(self.state)
        except ValueError:
            return None
        return value if math.isfinite(value) else None

    def same_content(self, other: EntityState) -> bool:
        return self.state == other.state and dict(self.attributes) == dict(other.attributes)


def _parse_ts(value: Any, name: str) -> datetime:
    if isinstance(value, datetime):
        return require_aware(value, name)
    if isinstance(value, (int, float)):
        from datetime import UTC

        return datetime.fromtimestamp(float(value), tz=UTC)
    if not isinstance(value, str):
        raise ValueError(f"{name} missing or not a timestamp: {value!r}")
    return require_aware(datetime.fromisoformat(value.replace("Z", "+00:00")), name)


def parse_ha_state(payload: Mapping[str, Any], received_at: datetime | None = None) -> EntityState:
    """Build an :class:`EntityState` from an HA state dict (REST or WebSocket)."""
    entity_id = payload.get("entity_id")
    if not isinstance(entity_id, str) or "." not in entity_id:
        raise ValueError(f"state payload has no valid entity_id: {entity_id!r}")
    last_updated = _parse_ts(payload.get("last_updated"), "last_updated")
    last_changed = _parse_ts(payload.get("last_changed", last_updated), "last_changed")
    reported_raw = payload.get("last_reported")
    last_reported = _parse_ts(reported_raw, "last_reported") if reported_raw else last_updated
    attributes = payload.get("attributes") or {}
    if not isinstance(attributes, Mapping):
        attributes = {}
    return EntityState(
        entity_id=entity_id,
        state=str(payload.get("state", "")),
        attributes=dict(attributes),
        last_changed=last_changed,
        last_updated=last_updated,
        last_reported=last_reported,
        received_at=received_at,
    )


Listener = Callable[[EntityState, UpdateResult], None]


@dataclass
class EntityStore:
    """Latest accepted state per tracked entity plus per-result counters."""

    tracked: frozenset[str]
    _states: dict[str, EntityState] = field(default_factory=dict)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)
    _listeners: list[Listener] = field(default_factory=list)

    @classmethod
    def for_entities(cls, entities: Iterable[str]) -> EntityStore:
        return cls(tracked=frozenset(entities))

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def _count(self, entity_id: str, result: UpdateResult) -> None:
        per = self.counts.setdefault(entity_id, {})
        per[result.value] = per.get(result.value, 0) + 1

    def apply(self, new: EntityState) -> UpdateResult:
        """Apply one observation; return how it was classified."""
        if new.entity_id not in self.tracked:
            return UpdateResult.IGNORED
        old = self._states.get(new.entity_id)
        if old is None or new.last_updated > old.last_updated:
            result = UpdateResult.ACCEPTED
        elif new.last_updated < old.last_updated:
            result = UpdateResult.OUT_OF_ORDER
        elif new.last_reported > old.last_reported:
            result = UpdateResult.REFRESHED
        elif new.last_reported == old.last_reported:
            result = UpdateResult.DUPLICATE
        else:
            result = UpdateResult.OUT_OF_ORDER
        self._count(new.entity_id, result)
        if result in (UpdateResult.ACCEPTED, UpdateResult.REFRESHED):
            self._states[new.entity_id] = new
            for listener in self._listeners:
                listener(new, result)
        return result

    def get(self, entity_id: str | None) -> EntityState | None:
        if entity_id is None:
            return None
        return self._states.get(entity_id)

    def age_s(self, entity_id: str | None, now: datetime) -> float | None:
        """Seconds since HA last vouched for the entity, or ``None`` if never seen."""
        state = self.get(entity_id)
        if state is None:
            return None
        return max(0.0, (now - state.observed_at).total_seconds())

    def fresh_value(self, entity_id: str | None, now: datetime, max_age_s: float) -> float | None:
        """Numeric value if seen, available and younger than *max_age_s*; else ``None``."""
        state = self.get(entity_id)
        if state is None:
            return None
        age = (now - state.observed_at).total_seconds()
        if age > max_age_s:
            return None
        return state.numeric()

    def totals(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for per in self.counts.values():
            for key, value in per.items():
                out[key] = out.get(key, 0) + value
        return out
