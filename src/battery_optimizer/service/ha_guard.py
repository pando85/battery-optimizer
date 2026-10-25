"""Read-only guard for every byte the service sends to Home Assistant.

Phase C is shadow only: the service must be **unable** to operate a device, write a
state, fire an event or change configuration — not merely unwilling.  So the
allowlist is enforced at the lowest layer that sends anything:

* WebSocket: every outgoing frame is parsed and its ``type`` checked against
  :data:`ALLOWED_WS_TYPES` (authentication, heartbeat, subscriptions and reads).
  ``fire_event``, ``execute_script``, config writes and every other type raise
  :class:`ForbiddenHACallError` before the frame reaches the socket.
  ``subscribe_events`` is further restricted to ``state_changed``.
* ``call_service`` is refused, with **one** exception: ``weather.get_forecasts``
  with ``return_response: true`` (:func:`check_weather_forecast_call`).  It is a
  query, not an action: the service is registered ``SupportsResponse.ONLY``, so HA
  rejects it without a response and it only returns the forecast the weather
  entity already holds; it changes no state and operates no device.  The frame is
  accepted only when it is exactly::

      {"id": <int>, "type": "call_service", "domain": "weather",
       "service": "get_forecasts", "return_response": true,
       "service_data": {"type": "hourly" | "daily" | "twice_daily"},
       "target": {"entity_id": "weather.<name>" | ["weather.<name>", ...]}}

  Any other domain, service or ``return_response``, any additional field, and any
  entity outside the ``weather.`` domain raise.  The only caller is the passive
  forecast archive, which builds its frame with :func:`weather_forecast_request`
  (so no other module needs the literal service-call frame).
* REST: only ``GET`` on ``/api/``, ``/api/config``, ``/api/states`` and
  ``/api/states/<entity_id>``.  ``POST``/``PUT``/``DELETE`` and any other path raise.

The guard holds no credentials and does no I/O; the transports call it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    "ALLOWED_REST_METHODS",
    "ALLOWED_SUBSCRIBE_EVENT_TYPES",
    "ALLOWED_WEATHER_FORECAST_TYPES",
    "ALLOWED_WS_TYPES",
    "ForbiddenHACallError",
    "check_rest_request",
    "check_weather_forecast_call",
    "check_ws_frame",
    "check_ws_message",
    "weather_forecast_request",
]

#: The complete set of WebSocket message types the service may send.
ALLOWED_WS_TYPES: frozenset[str] = frozenset(
    {
        "auth",
        "ping",
        "subscribe_events",
        "subscribe_entities",
        "subscribe_trigger",
        "unsubscribe_events",
        "get_states",
        "get_config",
    }
)

#: ``subscribe_events`` may only follow state changes.
ALLOWED_SUBSCRIBE_EVENT_TYPES: frozenset[str] = frozenset({"state_changed"})

#: ``weather.get_forecasts`` forecast types (the service's ``type`` field).
ALLOWED_WEATHER_FORECAST_TYPES: frozenset[str] = frozenset({"hourly", "daily", "twice_daily"})

ALLOWED_REST_METHODS: frozenset[str] = frozenset({"GET"})

_WEATHER_CALL_KEYS = frozenset(
    {"id", "type", "domain", "service", "service_data", "target", "return_response"}
)
_WEATHER_ENTITY = re.compile(r"^weather\.[a-z0-9_]+$")
_MAX_WEATHER_ENTITIES = 8

_REST_PATH = re.compile(r"^/api/(?:config|states(?:/[a-z_]+\.[A-Za-z0-9_]+)?)?$")


class ForbiddenHACallError(PermissionError):
    """An attempt to send something outside the read-only allowlist."""


def _weather_entity_ids(value: Any, where: str) -> list[str]:
    if value is None:
        return []
    ids = value if isinstance(value, list) else [value]
    if not ids or len(ids) > _MAX_WEATHER_ENTITIES:
        raise ForbiddenHACallError(f"weather.get_forecasts: {where}.entity_id has a bad length")
    for entity_id in ids:
        if not isinstance(entity_id, str) or not _WEATHER_ENTITY.match(entity_id):
            raise ForbiddenHACallError(
                f"weather.get_forecasts: {where}.entity_id {entity_id!r} is not a weather entity"
            )
    return list(ids)


def check_weather_forecast_call(message: Mapping[str, Any]) -> None:
    """Raise unless *message* is exactly the one allowed ``call_service`` frame.

    Only ``weather.get_forecasts`` with ``return_response: true``, a ``type`` in
    :data:`ALLOWED_WEATHER_FORECAST_TYPES` and ``weather.*`` entities (in ``target``
    or ``service_data``) passes.  Everything else, including any extra field, raises.
    """
    extra = sorted(str(k) for k in set(message) - _WEATHER_CALL_KEYS)
    if extra:
        raise ForbiddenHACallError(f"call_service: unexpected field(s) {extra}")
    if message.get("domain") != "weather" or message.get("service") != "get_forecasts":
        raise ForbiddenHACallError(
            f"call_service {message.get('domain')!r}.{message.get('service')!r} is not allowed: "
            "the only permitted service call is weather.get_forecasts"
        )
    if message.get("return_response") is not True:
        raise ForbiddenHACallError("weather.get_forecasts is only allowed with return_response")
    if "id" in message and (isinstance(message["id"], bool) or not isinstance(message["id"], int)):
        raise ForbiddenHACallError("weather.get_forecasts: id must be an integer")
    service_data = message.get("service_data")
    if not isinstance(service_data, Mapping) or not set(service_data) <= {"type", "entity_id"}:
        raise ForbiddenHACallError(
            "weather.get_forecasts: service_data may only contain type and entity_id"
        )
    if service_data.get("type") not in ALLOWED_WEATHER_FORECAST_TYPES:
        raise ForbiddenHACallError(
            f"weather.get_forecasts: type must be one of {sorted(ALLOWED_WEATHER_FORECAST_TYPES)}"
        )
    target = message.get("target", {})
    if not isinstance(target, Mapping) or not set(target) <= {"entity_id"}:
        raise ForbiddenHACallError("weather.get_forecasts: target may only contain entity_id")
    entities = _weather_entity_ids(service_data.get("entity_id"), "service_data")
    entities += _weather_entity_ids(target.get("entity_id"), "target")
    if not entities:
        raise ForbiddenHACallError("weather.get_forecasts: no weather entity named")


def weather_forecast_request(
    msg_id: int, entity_id: str, forecast_type: str = "hourly"
) -> dict[str, Any]:
    """The one permitted ``call_service`` frame (validated before it is returned)."""
    message: dict[str, Any] = {
        "id": msg_id,
        "type": "call_service",
        "domain": "weather",
        "service": "get_forecasts",
        "service_data": {"type": forecast_type},
        "target": {"entity_id": entity_id},
        "return_response": True,
    }
    check_ws_message(message)
    return message


def check_ws_message(message: Mapping[str, Any]) -> None:
    """Raise :class:`ForbiddenHACallError` unless *message* is an allowed read-only command."""
    if not isinstance(message, Mapping):
        raise ForbiddenHACallError(
            f"WebSocket message must be an object, got {type(message).__name__}"
        )
    msg_type = message.get("type")
    if msg_type == "call_service":
        check_weather_forecast_call(message)  # the single, narrow, non-mutating exception
        return
    if msg_type not in ALLOWED_WS_TYPES:
        raise ForbiddenHACallError(
            f"WebSocket message type {msg_type!r} is not in the read-only allowlist "
            f"{sorted(ALLOWED_WS_TYPES)}: the shadow service has no actuation path"
        )
    if msg_type == "subscribe_events":
        event_type = message.get("event_type")
        if event_type not in ALLOWED_SUBSCRIBE_EVENT_TYPES:
            raise ForbiddenHACallError(
                f"subscribe_events for event_type={event_type!r} is not allowed; only "
                f"{sorted(ALLOWED_SUBSCRIBE_EVENT_TYPES)}"
            )
    if msg_type == "subscribe_trigger":
        trigger = message.get("trigger")
        triggers = trigger if isinstance(trigger, list) else [trigger]
        for item in triggers:
            platform = (
                item.get("platform", item.get("trigger")) if isinstance(item, Mapping) else None
            )
            if platform != "state":
                raise ForbiddenHACallError("subscribe_trigger is limited to state triggers")


def check_ws_frame(frame: str | bytes) -> None:
    """Check a serialized outgoing frame (a JSON object or a coalesced JSON array)."""
    try:
        payload = json.loads(frame)
    except (TypeError, ValueError) as exc:
        raise ForbiddenHACallError("outgoing WebSocket frame is not JSON") from exc
    messages = payload if isinstance(payload, list) else [payload]
    for message in messages:
        check_ws_message(message)


def check_rest_request(method: str, path: str) -> None:
    """Raise :class:`ForbiddenHACallError` unless this is a GET of an allowed read path."""
    if method.upper() not in ALLOWED_REST_METHODS:
        raise ForbiddenHACallError(f"REST method {method!r} is not allowed: GET only")
    if not _REST_PATH.match(path):
        raise ForbiddenHACallError(f"REST path {path!r} is not in the read-only allowlist")
