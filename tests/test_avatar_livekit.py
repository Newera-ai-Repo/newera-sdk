"""The transport option: livekit by default (the media is in a LiveKit room, only events on the socket) and
transport="websocket" for video chunks on the socket. A fake server answers like the Core gateway for each transport."""
import asyncio
import base64
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.asyncio.server import serve

from newera import (AvatarReady, AvatarUnavailable, ClientSecret, DEFAULT_TRANSPORT, InvalidRequest, LineDone, LineStarted,
                    LipSyncPlayer, LiveKitRoom, Newera, TransportUnavailable, VideoChunk, WrongTransport)
from newera.avatar import AUDIO_RATE, parse_event

FRAME_BYTES = AUDIO_RATE // 25 * 2
ROOM = {"transport": "livekit", "url": "wss://rtc.example.test", "room": "avatar-sess_1", "token": "jwt.viewer.token",
        "identity": "viewer-abc", "avatar_identity": "newera-avatar",
        "video": {"codec": "h264", "width": 512, "height": 512, "fps": 25},
        "audio": {"sample_rate": 24000, "channels": 1},
        "data_topics": {"events": "avatar.event", "control": "avatar.control"}}


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


def _select(connection, subprotocols):
    return "realtime" if "realtime" in subprotocols else None


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


class Server:
    """Answers per the transport in the URL. livekit: avatar.ready with the room, lines and events on the socket, no video.
    websocket: a JPEG video chunk per line. A line named "hold" waits for an interrupt."""

    def __init__(self, *, refuse_livekit=False):
        self.refuse_livekit = refuse_livekit
        self.paths, self.protocols, self.received = [], [], []

    async def handler(self, ws):
        self.paths.append(ws.request.path)
        self.protocols.append(ws.request.headers.get("Sec-WebSocket-Protocol"))
        q = {k: v[0] for k, v in parse_qs(urlparse(ws.request.path).query).items()}
        mode, transport = q.get("mode", "conversational"), q.get("transport", "websocket")
        send = lambda ev: ws.send(json.dumps(ev))                                   # noqa: E731
        if transport == "livekit" and self.refuse_livekit:
            await send({"type": "error", "event_id": "e", "error": {"type": "server_error", "code": "transport_unavailable",
                                                                    "message": "not available", "param": None, "event_id": None}})
            await ws.close(code=4503, reason="transport_unavailable")
            return
        self.received.append(json.loads(await ws.recv()))
        ready = {"type": "avatar.ready", "fps": 25, "width": 512, "height": 512, "audio_sample_rate": AUDIO_RATE,
                 "frames_per_event": 7, "mode": mode, "output": "square"}
        if transport == "livekit":
            ready.update({"video_format": "webrtc", "transport": "livekit", "livekit": ROOM})
        else:
            ready.update({"video_format": q.get("video_format", "jpeg")})
        await send(ready)
        await send({"type": "session.created", "session": {"id": "sess_1"}})
        audio: dict = {}
        frame = 0

        async def play(line):
            nonlocal frame
            await send({"type": "avatar.line.started", "line_id": line, "at_ms": frame * 40})
            if transport == "websocket":
                pcm = b"\x01\x00" * (FRAME_BYTES // 2 * 7)
                await send({"type": "response.video.chunk", "index": 0, "frame_index": frame, "start_ms": frame * 40,
                            "frames": [b64(b"jpeg%d" % i) for i in range(7)], "fps": 25, "width": 512, "height": 512,
                            "speech": True, "audio": b64(pcm), "audio_sample_rate": AUDIO_RATE,
                            "audio_start_ms": frame * 40, "audio_end_ms": frame * 40 + 280, "line_id": line})
            frame += 7
            await send({"type": "avatar.line.done", "line_id": line, "status": "completed"})

        open_lines = []
        async for raw in ws:
            ev = json.loads(raw)
            self.received.append(ev)
            kind, line = ev["type"], ev.get("line_id")
            if kind == "avatar.line.append":
                audio[line] = audio.get(line, b"") + base64.b64decode(ev["audio"])
            elif kind == "avatar.line.end":
                await play(line)
            elif kind == "avatar.line.text":
                if ev["text"] == "hold":
                    open_lines.append(line)
                else:
                    await play(line)
            elif kind == "avatar.interrupt":
                await send({"type": "avatar.interrupted"})
                for line in open_lines:
                    await send({"type": "avatar.line.done", "line_id": line, "status": "interrupted"})
                open_lines.clear()
            elif kind == "response.create":
                await send({"type": "response.audio_transcript.delta", "delta": "Hi", "response_id": "r"})
                await send({"type": "response.done", "response": {"status": "completed", "output_transcript": "Hi there"}})
            elif kind == "session.end":
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


def query(path):
    return {k: v[0] for k, v in parse_qs(urlparse(path).query).items()}


# ---------------------------------------------------------------------------------------- the default and the URL

def test_default_transport_is_livekit():
    assert DEFAULT_TRANSPORT == "livekit"


def test_default_connect_asks_for_livekit_and_sends_no_video_format(monkeypatch):
    monkeypatch.setitem(sys.modules, "av", None)            # no PyAV: livekit needs no decoder, so no warning and no error

    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect(mode="render-text") as call:
                assert call.transport == "livekit" and call.ready.transport == "livekit"
            assert query(srv.paths[0]) == {"mode": "render-text", "output": "original", "transport": "livekit"}
    run(go())


def test_websocket_opt_out_keeps_the_old_url_and_video_chunks():
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect(mode="render-text", transport="websocket", video_format="jpeg",
                                                   record=True) as call:
                assert call.transport == "websocket" and call.livekit is None and call.video_format == "jpeg"
                line = await call.speak("hello")
                assert await line.wait(timeout=5) == "completed"
                async for ev in call.events():
                    if isinstance(ev, VideoChunk):
                        break
                assert call.timeline is not None and len(call.timeline) == 7
            assert query(srv.paths[0]) == {"mode": "render-text", "video_format": "jpeg", "output": "original",
                                           "transport": "websocket"}
    run(go())


def test_conversational_default_url_has_model_and_transport():
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect("lama"):
                pass
            assert query(srv.paths[0]) == {"model": "newera-s2s", "output": "original", "transport": "livekit"}
    run(go())


def test_bad_transport_is_a_value_error_before_connecting():
    with pytest.raises(ValueError, match="transport"):
        Newera(api_key="k").avatar.connect(transport="webrtc")


def test_secret_connect_defaults_to_livekit_and_follows_the_secret_and_the_argument():
    async def go():
        async with Server() as srv:
            c = srv.client()
            async with c.avatar.connect(client_secret="avs_1", mode="render-text"):
                pass
            async with c.avatar.connect(client_secret=ClientSecret("avs_2", 1, "newera-s2s", mode="render-text",
                                                                   transport="websocket"), video_format="jpeg"):
                pass
            async with c.avatar.connect(client_secret=ClientSecret("avs_3", 1, "newera-s2s", mode="render-text",
                                                                   transport="websocket"), transport="livekit"):
                pass
            assert [query(p).get("transport") for p in srv.paths] == ["livekit", "websocket", "livekit"]
            assert srv.protocols[0] == "realtime, newera-client-secret.avs_1"
    run(go())


# ------------------------------------------------------------------------------------------------ the room

def test_livekit_object_is_parsed_on_ready_and_on_the_call():
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect(mode="render-audio") as call:
                lk = call.livekit
                assert isinstance(lk, LiveKitRoom) and lk is call.ready.livekit
                assert (lk.url, lk.room, lk.token, lk.identity, lk.avatar_identity) == (
                    "wss://rtc.example.test", "avatar-sess_1", "jwt.viewer.token", "viewer-abc", "newera-avatar")
                assert lk.video == {"codec": "h264", "width": 512, "height": 512, "fps": 25}
                assert lk.audio == {"sample_rate": 24000, "channels": 1}
                assert lk.data_topics == {"events": "avatar.event", "control": "avatar.control"}
                assert call.video_format == "webrtc" and call.ready.video_format == "webrtc"
                assert "jwt.viewer.token" not in repr(lk)          # the token stays out of logs
    run(go())


def test_parse_ready_transport_variants():
    ev = parse_event({"type": "avatar.ready", "video_format": "webrtc", "livekit": ROOM})
    assert isinstance(ev, AvatarReady) and ev.transport == "livekit" and ev.livekit.url == ROOM["url"]
    ev = parse_event({"type": "avatar.ready", "video_format": "jpeg"})
    assert ev.transport == "websocket" and ev.livekit is None
    ev = parse_event({"type": "avatar.ready", "video_format": "webrtc", "transport": "livekit"})
    assert ev.transport == "livekit" and ev.livekit is None


# ------------------------------------------------------------------------ frame APIs raise under livekit

def test_record_is_refused_at_connect_under_livekit():
    with pytest.raises(WrongTransport, match='transport="websocket"'):
        Newera(api_key="k").avatar.connect(record=True)


def test_frame_apis_raise_with_a_clear_error_under_livekit():
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect(mode="render-text") as call:
                for fn in (lambda: call.save_mp4("x.mp4"), lambda: call.save_h264("x.h264"),
                           lambda: call.save_wav("x.wav"), lambda: call.save_frames("x")):
                    with pytest.raises(WrongTransport, match='transport="websocket"'):
                        fn()
                for gen in (call.frames, call.units, call.audio):
                    with pytest.raises(WrongTransport, match='transport="websocket"'):
                        async for _ in gen():
                            pass
                with pytest.raises(WrongTransport, match="request_keyframe"):
                    await call.request_keyframe()
                with pytest.raises(WrongTransport, match="LipSyncPlayer"):
                    LipSyncPlayer(lambda b: None, lambda f, n: None, call=call)
                with pytest.raises(WrongTransport, match="LipSyncPlayer"):
                    LipSyncPlayer(lambda b: None, lambda f, n: None, request_keyframe=call.request_keyframe)
    run(go())


def test_wrong_transport_is_an_avatar_error():
    from newera import AvatarError
    assert issubclass(WrongTransport, AvatarError)


# ------------------------------------------------------------------------- line APIs work under livekit

def test_render_text_lines_work_under_livekit():
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect(mode="render-text") as call:
                seen = []
                call.on_line_started(lambda e: seen.append(("started", e.line_id)))
                call.on_line_done(lambda e: seen.append(("done", e.line_id, e.status)))
                line = await call.speak("Hello there", "l1", voice="female")
                assert await line.wait_started(timeout=5)
                assert await line.wait(timeout=5, strict=True) == "completed"
                evs = []
                async for ev in call.events():
                    evs.append(ev)
                    if isinstance(ev, LineDone):
                        break
                assert [type(e) for e in evs] == [LineStarted, LineDone]
                assert not any(isinstance(e, VideoChunk) for e in evs)
                assert seen == [("started", "l1"), ("done", "l1", "completed")]
            assert {"type": "avatar.line.text", "line_id": "l1", "event_id": "line:l1:1", "text": "Hello there",
                    "voice": "female"} in srv.received
    run(go())


def test_render_audio_stream_and_interrupt_work_under_livekit():
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect(mode="render-audio") as call:
                async def pieces():
                    for _ in range(3):
                        yield b"\x00\x00" * 2400
                line = call.start_line("a")
                assert await line.stream(pieces()) is True
                assert await line.wait(timeout=5) == "completed"
                kinds = [e["type"] for e in srv.received]
                assert kinds.count("avatar.line.append") == 3 and kinds.count("avatar.line.end") == 1
            async with srv.client().avatar.connect(mode="render-text") as call:
                held = await call.speak("hold", "h")
                await call.interrupt()
                assert await held.wait(timeout=5) == "interrupted"
                assert any(e["type"] == "avatar.interrupt" for e in srv.received)
    run(go())


def test_conversational_wait_reply_works_but_its_media_properties_raise():
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect("lama") as call:
                await call.send_text("hi")
                reply = await call.wait_reply(timeout=10, quiet=0.2)
                assert reply.text == "Hi there" and reply.status == "completed" and reply.chunks == []
                for name in ("frames", "units", "audio"):
                    with pytest.raises(WrongTransport):
                        getattr(reply, name)
    run(go())


def test_websocket_media_apis_still_work(tmp_path):
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect(mode="render-text", transport="websocket", video_format="jpeg",
                                                   record=True) as call:
                line = await call.speak("hello")
                await line.wait(timeout=5)
                got = []
                async for n, jpeg in call.frames():
                    got.append(n)
                    if len(got) == 7:
                        break
                assert got == list(range(7))
                call.save_wav(str(tmp_path / "a.wav"))
                call.save_frames(str(tmp_path / "f"))
                await call.request_keyframe()               # a JPEG call: nothing to send, no error
                LipSyncPlayer(lambda b: None, lambda f, n: None, call=call)
    run(go())


# ------------------------------------------------------------------------------ errors and client secrets

def test_transport_unavailable_is_a_typed_error_that_says_what_to_do():
    async def go():
        async with Server(refuse_livekit=True) as srv:
            with pytest.raises(TransportUnavailable) as exc:
                async with srv.client().avatar.connect(mode="render-text"):
                    pass
            assert exc.value.code == "transport_unavailable" and exc.value.close_code == 4503
            assert isinstance(exc.value, AvatarUnavailable)
            async with srv.client().avatar.connect(mode="render-text", transport="websocket"):
                pass                                        # the opt-out still works
    run(go())


def test_error_codes_for_the_new_server_errors():
    from newera.avatar import error_for
    assert isinstance(error_for("transport_unavailable", close_code=4503), TransportUnavailable)
    assert isinstance(error_for("avatar_render_stalled", close_code=4503), AvatarUnavailable)
    assert isinstance(error_for("invalid_transport", http_status=400), InvalidRequest)


class _Http(BaseHTTPRequestHandler):
    seen: list = []
    status, body = 200, {}

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        _Http.seen.append((self.path, json.loads(self.rfile.read(n) or b"{}")))
        data = json.dumps(_Http.body).encode()
        self.send_response(_Http.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def http():
    srv = HTTPServer(("127.0.0.1", 0), _Http)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Http.seen, _Http.status = [], 200
    yield "http://127.0.0.1:%d" % srv.server_port
    srv.shutdown()


def test_create_session_passes_transport_into_the_body_and_the_secret(http):
    _Http.body = {"model": "newera-s2s", "mode": "render-text", "client_secret": {"value": "avs_t", "expires_at": 5}}

    async def go():
        c = Newera(api_key="nwc_test", base_url=http)
        secret = await c.avatar.create_session(mode="render-text", transport="livekit")
        assert secret.transport == "livekit" and secret.as_dict()["transport"] == "livekit"
        plain = await c.avatar.create_session(mode="render-text")
        assert plain.transport is None and "transport" not in plain.as_dict()
        with pytest.raises(ValueError, match="transport"):
            await c.avatar.create_session(transport="rtc")
    run(go())
    assert _Http.seen[0] == ("/v1/avatar/sessions", {"mode": "render-text", "transport": "livekit"})
    assert _Http.seen[1] == ("/v1/avatar/sessions", {"mode": "render-text"})


def test_create_session_transport_unavailable_maps_from_http(http):
    _Http.status, _Http.body = 503, {"error": {"code": "transport_unavailable", "message": "no"}}

    async def go():
        with pytest.raises(TransportUnavailable) as exc:
            await Newera(api_key="nwc_test", base_url=http).avatar.create_session(transport="livekit")
        assert exc.value.http_status == 503
    run(go())
