"""Join your own LiveKit room: livekit_room= is sent as avatar_livekit in the first session.update, the refusals are typed,
the token never shows in repr or logs, and the disconnect event reaches the callback. A fake server answers like Core."""
import asyncio
import json
import logging
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.asyncio.server import serve

from newera import (AvatarError, ByoRefused, InvalidRequest, LineDone, LiveKitDisconnected, LiveKitJoin, LiveKitRoom,
                    Newera, WrongTransport)
from newera.avatar import AUDIO_RATE, error_for, parse_event

TOKEN = "eyJ.secret-customer-token.sig"
URL = "wss://livekit.example.com"
BYO_ROOM = {"byo": True, "identity": "avatar-newera", "room": "clinic-1"}


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


def _select(connection, subprotocols):
    return "realtime" if "realtime" in subprotocols else None


class Server:
    """refuse=<code>: answers the first session.update with that error and closes 4400. drop=<reason>: after a line, sends
    avatar.livekit.disconnected and closes 1000 livekit_room_disconnected."""

    def __init__(self, *, refuse=None, drop=None):
        self.refuse, self.drop = refuse, drop
        self.received, self.paths = [], []

    async def handler(self, ws):
        self.paths.append(ws.request.path)
        q = {k: v[0] for k, v in parse_qs(urlparse(ws.request.path).query).items()}
        send = lambda ev: ws.send(json.dumps(ev))                                   # noqa: E731
        first = json.loads(await ws.recv())
        self.received.append(first)
        if self.refuse:
            await send({"type": "error", "event_id": "e", "error": {
                "type": "invalid_request_error", "code": self.refuse, "message": "refused", "param": "session.avatar_livekit",
                "event_id": None}})
            await ws.close(code=4400, reason=self.refuse)
            return
        livekit = dict(BYO_ROOM) if "avatar_livekit" in first["session"] else {"url": "wss://rtc.example.test", "token": "viewer"}
        await send({"type": "avatar.ready", "fps": 25, "width": 512, "height": 512, "audio_sample_rate": AUDIO_RATE,
                    "frames_per_event": 7, "mode": q.get("mode"), "output": "square", "video_format": "webrtc",
                    "transport": "livekit", "livekit": livekit})
        await send({"type": "session.created", "session": {"id": "sess_1"}})
        async for raw in ws:
            ev = json.loads(raw)
            self.received.append(ev)
            if ev["type"] == "avatar.line.text":
                await send({"type": "avatar.line.started", "line_id": ev["line_id"], "at_ms": 0})
                if self.drop:
                    await send({"type": "avatar.livekit.disconnected", "reason": self.drop})
                    await ws.close(code=1000, reason="livekit_room_disconnected")
                    return
                await send({"type": "avatar.line.done", "line_id": ev["line_id"], "status": "completed"})
            elif ev["type"] == "session.end":
                await ws.close()

    async def __aenter__(self):
        self.server = await serve(self.handler, "127.0.0.1", 0, select_subprotocol=_select)
        self.base_url = "http://127.0.0.1:%d" % self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *a):
        self.server.close()
        await self.server.wait_closed()

    def client(self):
        return Newera(api_key="nwc_test", base_url=self.base_url)


# ------------------------------------------------------------------------------------------------ what is sent

def test_session_update_carries_avatar_livekit_exactly():
    async def go():
        async with Server() as srv:
            join = LiveKitJoin(URL, TOKEN, publish_on_behalf="agent-1")
            async with srv.client().avatar.connect(mode="render-audio", livekit_room=join) as call:
                assert call.transport == "livekit"
            first = srv.received[0]
            assert first["type"] == "session.update"
            assert first["session"]["avatar_livekit"] == {"url": URL, "token": TOKEN, "publish_on_behalf": "agent-1"}
            assert parse_qs(urlparse(srv.paths[0]).query)["transport"] == ["livekit"]
    run(go())


def test_dict_form_and_publish_on_behalf_is_optional():
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect(mode="render-text", livekit_room={"url": URL, "token": TOKEN}):
                pass
            assert srv.received[0]["session"]["avatar_livekit"] == {"url": URL, "token": TOKEN}
    run(go())


def test_without_livekit_room_nothing_is_added():
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect(mode="render-text"):
                pass
            assert "avatar_livekit" not in srv.received[0]["session"]
    run(go())


def test_call_livekit_reports_the_customers_room_without_a_viewer_token():
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect(mode="render-audio", livekit_room={"url": URL, "token": TOKEN}) as call:
                lk = call.livekit
                assert isinstance(lk, LiveKitRoom) and lk.byo is True
                assert (lk.identity, lk.room, lk.token) == ("avatar-newera", "clinic-1", "")
        async with Server() as srv:                       # our own room: byo is False and there is a viewer token
            async with srv.client().avatar.connect(mode="render-audio") as call:
                assert call.livekit.byo is False and call.livekit.token == "viewer"
    run(go())


# ------------------------------------------------------------------------------------------------ validation

def test_livekit_room_needs_transport_livekit():
    client = Newera(api_key="nwc_test", base_url="http://127.0.0.1:1")
    with pytest.raises(ValueError, match="transport"):
        client.avatar.connect(mode="render-audio", transport="websocket", livekit_room={"url": URL, "token": TOKEN})


@pytest.mark.parametrize("bad", [{"url": "", "token": TOKEN}, {"url": URL, "token": ""}, {"url": URL},
                                 {"url": URL, "token": TOKEN, "room": "x"}])
def test_bad_livekit_room_is_a_value_error_before_connecting(bad):
    client = Newera(api_key="nwc_test", base_url="http://127.0.0.1:1")
    with pytest.raises(ValueError):
        client.avatar.connect(mode="render-audio", livekit_room=bad)


def test_wrong_type_is_a_type_error():
    client = Newera(api_key="nwc_test", base_url="http://127.0.0.1:1")
    with pytest.raises(TypeError):
        client.avatar.connect(mode="render-audio", livekit_room="wss://x")


# ------------------------------------------------------------------------------------------------ the token stays secret

def test_token_is_never_in_repr_or_logs(caplog):
    caplog.set_level(logging.DEBUG)

    async def go():
        async with Server() as srv:
            join = LiveKitJoin(URL, TOKEN, "agent-1")
            assert TOKEN not in repr(join) and TOKEN not in str(join)
            conn = srv.client().avatar.connect(mode="render-audio", livekit_room=join)
            async with conn as call:
                assert TOKEN not in repr(call) and TOKEN not in repr(call.livekit) and TOKEN not in repr(call.ready)
                assert "avatar_livekit" not in call._session        # not kept past the first session.update
    run(go())
    assert TOKEN not in caplog.text


# ------------------------------------------------------------------------------------------------ errors

CODES = ["byo_not_enabled", "byo_host_not_allowed", "invalid_livekit_url", "invalid_livekit_token", "invalid_livekit",
         "avatar_locked"]


@pytest.mark.parametrize("code", CODES)
def test_refusals_map_to_byo_refused(code):
    async def go():
        async with Server(refuse=code) as srv:
            with pytest.raises(ByoRefused) as ei:
                async with srv.client().avatar.connect(mode="render-audio", livekit_room={"url": URL, "token": TOKEN}):
                    pass
            assert ei.value.code == code and ei.value.close_code == 4400
            assert isinstance(ei.value, InvalidRequest) and isinstance(ei.value, AvatarError)
    run(go())


def test_error_for_maps_each_code_and_wrong_transport_stays_wrong_transport():
    for code in CODES:
        assert type(error_for(code, "m", 4400)) is ByoRefused
    assert isinstance(error_for("wrong_transport", "m", 4400), WrongTransport)


# ------------------------------------------------------------------------------------------------ disconnect

def test_parse_disconnect_event():
    ev = parse_event({"type": "avatar.livekit.disconnected", "reason": "participant_removed"})
    assert isinstance(ev, LiveKitDisconnected) and ev.reason == "participant_removed"


def test_disconnect_event_reaches_callback_events_and_the_call_ends_normally():
    async def go():
        async with Server(drop="room_deleted") as srv:
            async with srv.client().avatar.connect(mode="render-text", livekit_room={"url": URL, "token": TOKEN}) as call:
                seen = []
                call.on_livekit_disconnected(lambda ev: seen.append(ev.reason))
                await call.speak("hello", "l1")
                types = []
                async for ev in call.events():
                    types.append(type(ev))
                assert LiveKitDisconnected in types and LineDone not in types
                assert seen == ["room_deleted"]
                assert call.livekit_disconnected.reason == "room_deleted"
                assert call.close_code == 1000 and call.close_reason == "livekit_room_disconnected"
                assert call.last_error is None
    run(go())
