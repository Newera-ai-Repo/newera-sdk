"""The render modes (render-audio, render-text), lines, interrupt, typed errors and the lip-sync player, against a
fake server (a local WebSocket); no network, no key."""
import asyncio
import base64
import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.asyncio.server import serve

from newera import (AvatarBusy, AvatarError, ClientSecret, ErrorEvent, InsufficientQuota, Interrupted, LineDone,
                    LineError, LineStarted, LipSyncPlayer, Newera, NotEnabled, RateLimited, VideoChunk, WrongMode)
from newera.avatar import AUDIO_RATE, error_for, parse_event

FRAME_BYTES = AUDIO_RATE // 25 * 2


def bare(path):
    """A request path without the video_format and output options (every call sends them, see test_avatar_output.py)."""
    from urllib.parse import parse_qsl, urlencode, urlsplit
    u = urlsplit(path)
    q = [(k, v) for k, v in parse_qsl(u.query) if k not in ("video_format", "output", "transport")]
    return u.path + ("?" + urlencode(q) if q else "")


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


def _select(connection, subprotocols):
    return "realtime" if "realtime" in subprotocols else None


def chunk(frame_index, line_id, *, n=7, speech=True, tag=1, idle=False):
    start = frame_index * 40
    pcm = struct.pack("<h", tag) * (FRAME_BYTES // 2 * n)
    return {"type": "response.video.chunk", "event_id": "evt_x", "index": 0, "frame_index": frame_index,
            "start_ms": start, "frames": [b64(("jpeg-%d-%d" % (frame_index, i)).encode()) for i in range(n)],
            "fps": 25, "width": 512, "height": 512, "speech": speech, "audio": b64(pcm) if speech else None,
            "audio_sample_rate": AUDIO_RATE, "audio_start_ms": start, "audio_end_ms": start + n * 40,
            "line_id": line_id if speech else None, "generation_ms": 900, "queued_ms": 100}


class FakeRender:
    """The render modes of Core's /v1/avatar. A line plays (started, video, done) when its audio is complete.
    A line called "hold" never finishes by itself: it waits for an interrupt, which answers like Core does:
    avatar.interrupted, then line.done interrupted for every open line. The text "FAIL" is `tts_failed`, "BAD" an
    `invalid_text` error."""

    def __init__(self, *, refuse=None, drop_after_first=False):
        self.refuse, self.drop_after_first = refuse, drop_after_first
        self.paths, self.protocols, self.headers, self.received = [], [], [], []
        self.audio: dict[str, bytes] = {}
        self.open: list[str] = []
        self.frame = 0

    async def handler(self, ws):
        self.paths.append(ws.request.path)
        self.protocols.append(ws.request.headers.get("Sec-WebSocket-Protocol"))
        self.headers.append(dict(ws.request.headers))
        mode = (parse_qs(urlparse(ws.request.path).query).get("mode") or ["conversational"])[0]
        send = lambda ev: ws.send(json.dumps(ev))                                   # noqa: E731
        if self.refuse:
            code, close, extra = self.refuse
            await send({"type": "error", "event_id": "e", "error": {"type": "x", "code": code, "message": "no: " + code,
                                                                    "param": None, "event_id": None, **extra}})
            await ws.close(code=close, reason=code)
            return
        first = json.loads(await ws.recv())
        self.received.append(first)
        await send({"type": "avatar.ready", "fps": 25, "width": 512, "height": 512, "video_format": "jpeg",
                    "audio_sample_rate": AUDIO_RATE, "frames_per_event": 7, "mode": mode})
        await send({"type": "session.created", "session": {"id": "sess_r"}})

        async def play(line, pcm):
            await send({"type": "avatar.line.started", "line_id": line, "at_ms": self.frame * 40})
            for i in range(max(1, len(pcm) // (FRAME_BYTES * 7))):
                await send(chunk(self.frame, line, tag=i + 1))
                self.frame += 7
            await send({"type": "avatar.line.done", "line_id": line, "status": "completed"})
            self.open.remove(line)

        def err(code, ev, param=None):
            return send({"type": "error", "event_id": "e", "error": {"type": "invalid_request_error", "code": code,
                         "message": code, "param": param, "event_id": ev.get("event_id")}})

        async for raw in ws:
            ev = json.loads(raw)
            self.received.append(ev)
            kind, line = ev["type"], ev.get("line_id")
            if kind == "avatar.line.append" and mode == "render-audio":
                self.audio[line] = self.audio.get(line, b"") + base64.b64decode(ev["audio"])
                if line not in self.open:
                    self.open.append(line)
            elif kind == "avatar.line.end" and mode == "render-audio":
                if line != "hold":
                    await play(line, self.audio.get(line, b""))
                if self.drop_after_first:
                    await ws.close()
                    return
            elif kind == "avatar.line.text" and mode == "render-text":
                if ev["text"] == "BAD":
                    await err("invalid_text", ev, "text")
                    continue
                self.open.append(line)
                if ev["text"] == "FAIL":
                    await send({"type": "error", "event_id": "e", "error": {"type": "server_error", "code": "tts_failed",
                                "message": "x", "param": "line_id", "event_id": None}})
                    await send({"type": "avatar.line.done", "line_id": line, "status": "failed"})
                    self.open.remove(line)
                elif ev["text"] != "hold":
                    await play(line, bytes(FRAME_BYTES * 7 * 2))
            elif kind.startswith("avatar.line."):
                await err("wrong_mode", ev)
            elif kind == "avatar.interrupt":
                await send({"type": "avatar.interrupted"})
                for line in self.open:
                    await send({"type": "avatar.line.done", "line_id": line, "status": "interrupted"})
                self.open.clear()
            elif kind == "session.end":
                await send({"type": "session.summary", "summary": {"billed": {"connected_seconds": 30}}})
                await ws.close()

    async def __aenter__(self):
        self.server = await serve(self.handler, "127.0.0.1", 0, select_subprotocol=_select)
        self.base_url = "http://127.0.0.1:%d" % self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *a):
        self.server.close()
        await self.server.wait_closed()


def client_for(srv):
    return Newera(api_key="nwc_test", base_url=srv.base_url)


async def collect(call, until):
    """Events up to and including the first one `until` accepts."""
    out = []
    async for ev in call.events():
        out.append(ev)
        if until(ev):
            return out
    return out


# ------------------------------------------------------------------------------------------ parsing

def test_new_events_and_chunk_timestamps_parse():
    ev = parse_event(chunk(14, "a", n=7, tag=2))
    assert isinstance(ev, VideoChunk)
    assert (ev.line_id, ev.audio_start_ms, ev.audio_end_ms) == ("a", 560, 840)
    assert ev.frame_times_ms() == [560 + i * 40 for i in range(7)]
    assert len(ev.audio) == 7 * FRAME_BYTES
    old = parse_event({"type": "response.video.chunk", "frame_index": 3, "start_ms": 120, "frames": [b64(b"x")] * 2,
                       "fps": 25})                                   # a server without the new fields
    assert (old.audio_start_ms, old.audio_end_ms, old.line_id) == (120, 200, None)
    s = parse_event({"type": "avatar.line.started", "line_id": "a", "at_ms": 80})
    d = parse_event({"type": "avatar.line.done", "line_id": "a", "status": "interrupted"})
    assert isinstance(s, LineStarted) and (s.line_id, s.at_ms) == ("a", 80)
    assert isinstance(d, LineDone) and d.status == "interrupted"
    assert parse_event({"type": "avatar.ready", "mode": "render-text"}).mode == "render-text"
    assert parse_event({"type": "response.audio.delta", "delta": b64(b"\0\0"), "line_id": "a"}).line_id == "a"


def test_error_mapping_including_retry_after():
    e = parse_event({"type": "error", "error": {"code": "avatar_busy", "message": "m", "retry_after_s": 7, "event_id": "z"}})
    assert isinstance(e, ErrorEvent) and e.retry_after_s == 7 and e.event_id == "z"
    exc = e.exception(4503)
    assert isinstance(exc, AvatarBusy) and exc.retry_after_s == 7 and exc.close_code == 4503
    assert isinstance(error_for("avatar_concurrency_limit", close_code=4429), RateLimited)
    assert isinstance(error_for("model_not_entitled", close_code=4403), NotEnabled)
    assert isinstance(error_for("insufficient_quota", close_code=4402), InsufficientQuota)
    assert isinstance(error_for("wrong_mode"), WrongMode)
    for code in ("invalid_line_id", "duplicate_line", "too_many_lines", "line_too_long", "invalid_audio",
                 "invalid_text", "text_too_long"):
        assert isinstance(error_for(code), LineError)
    assert type(error_for("session_time_limit")).__name__ == "CallTimeLimit"
    assert error_for("x").retry_after_s is None


# ------------------------------------------------------------------------------------- render-audio

def test_render_audio_line_streams_audio_and_reports_events():
    async def go():
        async with FakeRender() as srv:
            async with client_for(srv).avatar.connect("lama", mode="render-audio", idle_video=True, transport="websocket") as call:
                assert bare(srv.paths[0]) == "/v1/avatar?mode=render-audio"           # no model in a render mode
                assert srv.received[0] == {"type": "session.update",
                                           "session": {"avatar": "lama", "avatar_idle_video": True}}
                assert call.mode == "render-audio" and call.ready.mode == "render-audio"
                started, done = [], []
                call.on_line_started(started.append)
                call.on_line_done(done.append)

                async def synth():                                  # audio arriving while it is being synthesised
                    for tag in (1, 2, 3):
                        yield struct.pack("<h", tag) * (FRAME_BYTES // 2 * 7)
                        await asyncio.sleep(0)

                line = call.start_line("greeting")
                assert not line.done and srv.received[1:] == []      # nothing is sent before the first chunk
                assert await line.stream(synth()) is True
                assert await line.wait(5) == "completed" and line.started
                events = await collect(call, lambda e: isinstance(e, LineDone))
                assert [type(e).__name__ for e in events if not isinstance(e, VideoChunk)] == ["LineStarted", "LineDone"]
                chunks = [e for e in events if isinstance(e, VideoChunk)]
                assert chunks and all(c.line_id == "greeting" and c.audio and c.frames for c in chunks)
                assert chunks[0].audio_start_ms == 0 and chunks[0].audio_end_ms == 280
                assert chunks[1].audio_start_ms == chunks[0].audio_end_ms          # one continuous clock
                kinds = [e["type"] for e in srv.received[1:]]
                assert kinds == ["avatar.line.append"] * 3 + ["avatar.line.end"]
                assert len(srv.audio["greeting"]) == 3 * 7 * FRAME_BYTES and srv.audio["greeting"][:2] == struct.pack("<h", 1)
                assert all(e["line_id"] == "greeting" for e in srv.received[1:])
                assert [e.line_id for e in started] == ["greeting"] and [(e.line_id, e.status) for e in done] == [("greeting", "completed")]
    run(go())


def test_send_line_one_shot_and_async_with():
    async def go():
        async with FakeRender() as srv:
            async with client_for(srv).avatar.connect(mode="render-audio", transport="websocket") as call:
                pcm = bytes(FRAME_BYTES * 7)
                one = await call.send_line(pcm, "one", chunk_ms=100)
                assert await one.wait(5) == "completed"
                assert [e["type"] for e in srv.received[1:]].count("avatar.line.append") > 1
                async with call.start_line() as auto:                    # an id is made for you; async with ends it
                    await auto.append(pcm)
                assert auto.id == "line-%d" % int(auto.id.split("-")[1]) and await auto.wait(5) == "completed"
                with pytest.raises(RuntimeError):
                    await auto.append(pcm)                               # ended
    run(go())


# -------------------------------------------------------------------------------------- render-text

def test_render_text_sends_exact_text_and_voice():
    async def go():
        async with FakeRender() as srv:
            async with client_for(srv).avatar.connect(mode="render-text", transport="websocket") as call:
                assert bare(srv.paths[0]) == "/v1/avatar?mode=render-text"
                assert "voice" not in srv.received[0]["session"]
                a = await call.speak("مرحبا, welcome!", "t1", voice="female")
                b = await call.speak("second line", "t2")
                assert [await a.wait(5), await b.wait(5)] == ["completed", "completed"]
                sent = [e for e in srv.received if e["type"] == "avatar.line.text"]
                assert [(e["line_id"], e["text"], e.get("voice")) for e in sent] == [
                    ("t1", "مرحبا, welcome!", "female"), ("t2", "second line", None)]
                with pytest.raises(WrongMode):
                    call.start_line("x")                                 # audio lines belong to render-audio
    run(go())


def test_audio_methods_refused_in_render_text_and_the_reverse():
    async def go():
        async with FakeRender() as srv:
            async with client_for(srv).avatar.connect(mode="render-audio", transport="websocket") as call:
                with pytest.raises(WrongMode):
                    await call.speak("hi")
            # the server's own refusal of a line event of the wrong mode is typed too
            async with client_for(srv).avatar.connect(mode="render-text", transport="websocket") as call:
                line = call._new_line("raw", "audio")
                await line.append(b"\0\0")
                assert await line.wait(5) == "failed"
                with pytest.raises(WrongMode):
                    await line.wait(strict=True)
    run(go())


def test_failed_and_refused_lines_are_typed():
    async def go():
        async with FakeRender() as srv:
            async with client_for(srv).avatar.connect(mode="render-text", transport="websocket") as call:
                bad = await call.speak("FAIL", "f1")
                assert await bad.wait(5) == "failed"
                with pytest.raises(AvatarError) as e:
                    await bad.wait(strict=True)
                assert e.value.code == "tts_failed"
                refused = await call.speak("BAD", "f2")                  # the server answers an error, no done
                assert await refused.wait(5) == "failed"
                with pytest.raises(LineError) as e:
                    await refused.wait(strict=True)
                assert e.value.code == "invalid_text"
                ok = await call.speak("fine", "f3")                      # the call goes on
                assert await ok.wait(5) == "completed"
    run(go())


# ---------------------------------------------------------------------------------------- interrupt

def test_interrupt_drops_unsent_audio_and_ends_open_lines():
    async def go():
        async with FakeRender() as srv:
            async with client_for(srv).avatar.connect(mode="render-audio", transport="websocket") as call:
                pcm = bytes(FRAME_BYTES * 7)
                hold = call.start_line("hold")
                await hold.append(pcm)
                never = call.start_line("never-sent")                    # opened, but no audio yet
                sent_before = len(srv.received)

                async def endless():
                    for _ in range(1000):
                        yield pcm
                        await asyncio.sleep(0.001)

                feeder = asyncio.ensure_future(hold.stream(endless()))
                await asyncio.sleep(0.02)
                await call.interrupt()
                assert await feeder is False                              # the generator is dropped at once
                assert await hold.append(pcm) is False
                assert hold.interrupted and await hold.wait(5) == "interrupted"
                assert await never.wait(5) == "interrupted"               # local: the server never heard of it
                ev = await collect(call, lambda e: isinstance(e, LineDone))
                assert any(isinstance(e, Interrupted) for e in ev)        # avatar.interrupted, then the done
                assert [(e.line_id, e.status) for e in ev if isinstance(e, LineDone)] == [("hold", "interrupted")]
                n_append = [e["type"] for e in srv.received[sent_before:]].count("avatar.line.append")
                assert n_append < 1000 and srv.received[-1]["type"] == "avatar.interrupt"
                await hold.end()                                          # safe after an interrupt
                fresh = await call.send_line(pcm, "fresh")                # lines after it play normally
                assert await fresh.wait(5) == "completed"
    run(go())


def test_interrupt_in_render_text():
    async def go():
        async with FakeRender() as srv:
            async with client_for(srv).avatar.connect(mode="render-text", transport="websocket") as call:
                line = await call.speak("hold", "h")
                await call.interrupt()
                assert await line.wait(5) == "interrupted"
    run(go())


# ------------------------------------------------------------------------------- errors and drops

@pytest.mark.parametrize("mode", ["render-audio", "render-text"])
def test_avatar_busy_has_retry_after(mode):
    async def go():
        async with FakeRender(refuse=("avatar_busy", 4503, {"retry_after_s": 9})) as srv:
            with pytest.raises(AvatarBusy) as e:
                async with client_for(srv).avatar.connect(mode=mode, transport="websocket"):
                    pass
            assert e.value.retry_after_s == 9 and e.value.close_code == 4503 and e.value.code == "avatar_busy"
    run(go())


@pytest.mark.parametrize("code,close,cls", [
    ("model_not_entitled", 4403, NotEnabled), ("insufficient_quota", 4402, InsufficientQuota),
    ("avatar_concurrency_limit", 4429, RateLimited)])
def test_entitlement_credit_and_limit_errors(code, close, cls):
    async def go():
        async with FakeRender(refuse=(code, close, {})) as srv:
            with pytest.raises(cls) as e:
                async with client_for(srv).avatar.connect(mode="render-text", transport="websocket"):
                    pass
            assert e.value.code == code and e.value.close_code == close
    run(go())


def test_dropped_call_fails_open_lines():
    async def go():
        async with FakeRender(drop_after_first=True) as srv:
            async with client_for(srv).avatar.connect(mode="render-audio", transport="websocket") as call:
                await call.send_line(bytes(FRAME_BYTES * 7), "a")
                b = call.start_line("b")
                await b.append(bytes(FRAME_BYTES))
                assert await b.wait(5) == "failed" and b.error.code == "closed"
    run(go())


# ------------------------------------------------------------------------ options and secret flow

def test_render_modes_reject_conversation_options_and_bad_mode():
    c = Newera(api_key="k")
    for kw in ({"instructions": "x"}, {"language": "ar"}, {"model": "newera-s2s-mini"}, {"turn_detection": None}):
        with pytest.raises(ValueError):
            c.avatar.connect(mode="render-text", **kw, transport="websocket")
    with pytest.raises(ValueError):
        c.avatar.connect(mode="video", transport="websocket")


def test_client_secret_flow_for_render_mode():
    async def go():
        async with FakeRender() as srv:
            client = Newera(api_key=None, base_url=srv.base_url)
            async with client.avatar.connect(client_secret=ClientSecret("avs_q", 1, "newera-s2s", "render-text"), transport="websocket") as call:
                assert call.mode == "render-text"
            assert srv.protocols[0] == "realtime, newera-client-secret.avs_q"
            assert bare(srv.paths[0]) == "/v1/avatar?mode=render-text" and "authorization" not in srv.headers[0]
    run(go())


class _Http(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        _Http.seen.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"))
        data = json.dumps({"object": "avatar.session", "mode": "render-audio",
                           "client_secret": {"value": "avs_r", "expires_at": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_create_session_for_render_mode_and_conversational_body_unchanged():
    srv = HTTPServer(("127.0.0.1", 0), _Http)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        async def go():
            client = Newera(api_key="k", base_url="http://127.0.0.1:%d" % srv.server_port)
            s = await client.avatar.create_session(mode="render-audio")
            assert (s.value, s.mode) == ("avs_r", "render-audio") and s.as_dict()["mode"] == "render-audio"
            await client.avatar.create_session(model="newera-s2s-mini")
        run(go())
    finally:
        srv.shutdown()
    assert _Http.seen == [{"mode": "render-audio"}, {"model": "newera-s2s-mini"}]


# ------------------------------------------------------------------------------------------- player

def test_player_plays_audio_and_frames_on_one_clock():
    import time
    audio, frames = [], []
    t0 = time.monotonic()

    def play_audio(pcm):
        audio.append((time.monotonic() - t0, len(pcm)))

    def show(jpeg, n):
        frames.append((time.monotonic() - t0, n))

    async def go():
        p = LipSyncPlayer(play_audio, show, buffer_ms=100, audio_lead_ms=30)
        evs = [parse_event(chunk(0, "a", n=5)), parse_event(chunk(5, "a", n=5))]

        async def source():
            for e in evs:
                yield e
        await p.play(source())
        return p
    p = run(go())
    assert [n for _, n in frames] == list(range(10)) and len(audio) == 2
    gaps = [b[0] - a[0] for a, b in zip(frames, frames[1:])]
    assert all(0.02 < g < 0.09 for g in gaps)                         # 40 ms apart, on the audio's clock
    assert frames[0][0] == pytest.approx(0.1, abs=0.04)               # buffered 100 ms before the first frame
    assert audio[0][0] < frames[0][0] and audio[1][0] < frames[5][0]  # each chunk's audio is written before its frames
    assert p.late_ms_total == 0


def test_player_keeps_sync_when_a_chunk_is_late_and_drops_on_interrupt():
    import time
    frames, flushed = [], []

    async def go():
        p = LipSyncPlayer(lambda pcm: None, lambda j, n: frames.append((time.monotonic(), n)), buffer_ms=40,
                          flush_audio=lambda: flushed.append(1), max_late_ms=50)
        task = asyncio.ensure_future(p.play())
        p.feed(parse_event(chunk(0, "a", n=2)))
        await asyncio.sleep(0.4)                                       # the next chunk is 300 ms late
        t_late = time.monotonic()
        p.feed(parse_event(chunk(2, "a", n=2)))
        await asyncio.sleep(0.25)
        assert [n for _, n in frames] == [0, 1, 2, 3]
        assert frames[2][0] - t_late < 0.12 and p.late_ms_total > 100  # clock moved, not skipped frames
        p.feed(parse_event(chunk(4, "a", n=7)))
        p.feed(Interrupted())                                          # drops what was queued and scheduled
        assert flushed == [1]
        await asyncio.sleep(0.4)
        assert [n for _, n in frames] == [0, 1, 2, 3]
        p.feed(parse_event(chunk(30, "b", n=1)))                       # a new line starts cleanly afterwards
        await asyncio.sleep(0.2)
        p.close()
        await task
        assert [n for _, n in frames][-1] == 30
    run(go())
