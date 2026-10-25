"""Read-only guard for every byte the service sends to Home Assistant.

Phase C is shadow only: the service must be **unable** to call a service, write a
state, fire an event or change configuration — not merely unwilling.  So the
allowlist is enforced at the lowest layer that sends anything:

* WebSocket: every outgoing frame is parsed and its ``type`` checked against
  :data:`ALLOWED_WS_TYPES` (authentication, heartbeat, subscriptions and reads).
  ``call_service``, ``fire_event``, ``execute_script``, config writes and every
  other type raise :class:`ForbiddenHACallError` before the frame reaches the socket.
  ``subscribe_events`` is further restricted to ``state_changed``.
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
    "ALLOWED_WS_TYPES",
    "ForbiddenHACallError",
    "check_rest_request",
    "check_ws_frame",
    "check_ws_message",
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

ALLOWED_REST_METHODS: frozenset[str] = frozenset({"GET"})

_REST_PATH = re.compile(r"^/api/(?:config|states(?:/[a-z_]+\.[A-Za-z0-9_]+)?)?$")


class ForbiddenHACallError(PermissionError):
    """An attempt to send something outside the read-only allowlist."""


def check_ws_message(message: Mapping[str, Any]) -> None:
    """Raise :class:`ForbiddenHACallError` unless *message* is an allowed read-only command."""
    if not isinstance(message, Mapping):
        raise ForbiddenHACallError(
            f"WebSocket message must be an object, got {type(message).__name__}"
        )
    msg_type = message.get("type")
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
