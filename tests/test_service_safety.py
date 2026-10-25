"""The shadow service cannot actuate: allowlist guard on every outgoing byte."""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import httpx
import pytest

from battery_optimizer.service.config import HomeAssistantSettings, Secret
from battery_optimizer.service.ha_client import GuardedConnection, HARestClient
from battery_optimizer.service.ha_guard import (
    ALLOWED_WS_TYPES,
    ForbiddenHACallError,
    check_rest_request,
    check_ws_frame,
    check_ws_message,
)

SERVICE_DIR = Path(__file__).resolve().parents[1] / "src" / "battery_optimizer" / "service"

FORBIDDEN_WS = [
    {"type": "call_service", "domain": "switch", "service": "turn_on"},
    {"type": "execute_script", "sequence": []},
    {"type": "fire_event", "event_type": "x"},
    {"type": "config/entity_registry/update", "entity_id": "number.x"},
    {"type": "render_template", "template": "{{ 1 }}"},
    {"type": "subscribe_events", "event_type": "call_service"},
    {"type": "subscribe_events"},  # all events
    {"type": "subscribe_trigger", "trigger": {"platform": "event", "event_type": "x"}},
    {"id": 3},
]


@pytest.mark.parametrize("message", FORBIDDEN_WS)
def test_forbidden_ws_messages_raise(message):
    with pytest.raises(ForbiddenHACallError):
        check_ws_message(message)


@pytest.mark.parametrize(
    "message",
    [
        {"type": "auth", "access_token": "t"},
        {"id": 1, "type": "ping"},
        {"id": 2, "type": "subscribe_events", "event_type": "state_changed"},
        {"id": 3, "type": "get_states"},
        {"id": 4, "type": "get_config"},
        {"id": 5, "type": "unsubscribe_events", "subscription": 2},
        {"id": 6, "type": "subscribe_entities", "entity_ids": ["sensor.x"]},
        {
            "id": 7,
            "type": "subscribe_trigger",
            "trigger": {"platform": "state", "entity_id": "a.b"},
        },
    ],
)
def test_allowed_ws_messages_pass(message):
    check_ws_message(message)


def test_allowlist_is_read_only():
    assert ALLOWED_WS_TYPES == {
        "auth",
        "ping",
        "subscribe_events",
        "subscribe_entities",
        "subscribe_trigger",
        "unsubscribe_events",
        "get_states",
        "get_config",
    }


def test_coalesced_frame_with_one_bad_message_is_refused():
    frame = json.dumps([{"id": 1, "type": "ping"}, {"id": 2, "type": "call_service"}])
    with pytest.raises(ForbiddenHACallError):
        check_ws_frame(frame)
    with pytest.raises(ForbiddenHACallError):
        check_ws_frame("not json")


def test_guarded_connection_blocks_service_call_before_the_socket():
    class Recorder:
        def __init__(self):
            self.sent = []

        async def send(self, text):
            self.sent.append(text)

        async def recv(self):
            return ""

        async def close(self):
            pass

    inner = Recorder()
    conn = GuardedConnection(inner)

    async def go():
        await conn.send(json.dumps({"id": 1, "type": "get_states"}))
        with pytest.raises(ForbiddenHACallError):
            await conn.send(
                json.dumps(
                    {"id": 2, "type": "call_service", "domain": "number", "service": "set_value"}
                )
            )

    asyncio.run(go())
    assert len(inner.sent) == 1


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("POST", "/api/services/switch/turn_on"),
        ("POST", "/api/states/sensor.x"),
        ("DELETE", "/api/states/sensor.x"),
        ("PUT", "/api/config"),
        ("GET", "/api/services"),
        ("GET", "/api/template"),
        ("GET", "/api/states/../services"),
    ],
)
def test_rest_guard_refuses(method, path):
    with pytest.raises(ForbiddenHACallError):
        check_rest_request(method, path)


def test_rest_client_is_get_only_even_through_the_raw_client():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"entity_id": "sensor.x", "state": "1"})

    settings = HomeAssistantSettings(url="http://ha.test", token=Secret("t"))
    client = HARestClient(settings=settings, transport=httpx.MockTransport(handler))

    async def go():
        assert (await client.get_state("sensor.x"))["state"] == "1"
        with pytest.raises(ForbiddenHACallError):
            await client._client.post("/api/services/switch/turn_on", json={})
        with pytest.raises(ForbiddenHACallError):
            await client._client.get("/api/services")
        await client.aclose()

    asyncio.run(go())
    assert [r.method for r in seen] == ["GET"]
    assert seen[0].url.path == "/api/states/sensor.x"


def test_token_never_in_repr():
    settings = HomeAssistantSettings(url="http://ha.test", token=Secret("super-secret"))
    assert "super-secret" not in repr(settings)
    assert "super-secret" not in str(settings.token)


def test_service_package_has_no_actuation_or_modbus_code():
    """Static guard: no service-call strings, no Modbus/MQTT library, no POST."""
    banned_imports = {"pymodbus", "paho", "homeassistant", "aiohttp", "subprocess"}
    for path in SERVICE_DIR.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [(node.module or "").split(".")[0]]
            else:
                names = []
            assert not set(names) & banned_imports, f"{path.name} imports {names}"
            if isinstance(node, ast.Attribute) and node.attr in ("post", "put", "delete", "patch"):
                pytest.fail(f"{path.name}: HTTP write method .{node.attr} referenced")
        text = path.read_text()
        if path.name != "ha_guard.py":
            assert "call_service" not in text, path.name
