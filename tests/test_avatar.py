"""The avatar API against a fake server (a local WebSocket and a local HTTP endpoint); no network, no key."""
import asyncio
import base64
import json
import os
import shutil
import struct
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from websockets.asyncio.server import serve

from newera import (AudioDelta, AuthenticationError, AvatarBusy, AvatarError, AvatarReady, ErrorEvent, Interrupted,
                    NeweraError, NotEnabled, Newera, RateLimited, ReplyTextDelta, ResponseDone,
                    UserTranscript, VideoChunk)
from newera.avatar import AUDIO_RATE, Timeline, build_session, error_for, parse_event, read_wav_24k, resample_pcm16
from newera.talk import Playback

FRAME_BYTES = AUDIO_RATE // 25 * 2     # 1920 bytes of audio per frame


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def chunk_event(frame_index, n=7, speech=True, tag=1):
    frames = [("jpeg-%d-%d" % (frame_index, i)).encode() for i in range(n)]
    pcm = b"".join(struct.pack("<h", tag) * (FRAME_BYTES // 2) for _ in range(n)) if speech else None
    return {"type": "response.video.chunk", "event_id": "evt_x", "index": frame_index // 28, "frame_index": frame_index,
            "start_ms": frame_index * 40, "frames": [b64(f) for f in frames], "fps": 25, "width": 512, "height": 512,
            "speech": speech, "audio": b64(pcm) if pcm else None, "audio_sample_rate": AUDIO_RATE,
            "generation_ms": 1300, "queued_ms": 500}


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


def _select(connection, subprotocols):
    return "realtime" if "realtime" in subprotocols else None


class FakeAvatar:
    """session.update first, then avatar.ready + session.created. A scripted reply follows `response.create`
    or `reply_after` bytes of audio."""

    def __init__(self, *, refuse=None, reply_after=None, chunks=3, video_delay=0.0):
        self.refuse, self.reply_after, self.chunks, self.video_delay = refuse, reply_after, chunks, video_delay
        self.received, self.paths, self.headers, self.protocols = [], [], [], []
        self.audio_bytes = 0
        self.server = None
        self.replied = False

    async def handler(self, ws):
        self.paths.append(ws.request.path)
        self.headers.append(dict(ws.request.headers))
        self.protocols.append(ws.request.headers.get("Sec-WebSocket-Protocol"))
        if self.refuse:
            code, close = self.refuse
            await ws.send(json.dumps({"type": "error", "event_id": "e", "error": {
                "type": "x", "code": code, "message": "refused: " + code, "param": None}}))
            await ws.close(code=close, reason=code)
            return
        first = json.loads(await ws.recv())
        self.received.append(first)
        assert first["type"] == "session.update"
        await ws.send(json.dumps({"type": "avatar.ready", "event_id": "e", "fps": 25, "width": 512, "height": 512,
                                  "video_format": "jpeg", "audio_sample_rate": AUDIO_RATE, "frames_per_event": 7}))
        await ws.send(json.dumps({"type": "session.created", "session": {"id": "sess_1", "voice": "female"}}))
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "input_audio_buffer.append":
                self.audio_bytes += len(base64.b64decode(msg["audio"]))
                if self.reply_after is not None and not self.replied and self.audio_bytes >= self.reply_after:
                    self.replied = True
                    await self.reply(ws)
                continue
            self.received.append(msg)
            if msg["type"] == "response.create":
                await self.reply(ws)
            elif msg["type"] == "session.end":
                await ws.send(json.dumps({"type": "session.summary", "summary": {"duration_s": 1.0,
                                          "billed": {"connected_seconds": 30}}}))
                await ws.close()

    async def reply(self, ws):
        send = lambda ev: ws.send(json.dumps(ev))      # noqa: E731
        await send({"type": "conversation.item.input_audio_transcription.completed", "transcript": "hello there", "item_id": "i1"})
        await send({"type": "response.created", "response": {"id": "r1"}})
        await send({"type": "response.audio_transcript.delta", "response_id": "r1", "delta": "Hi, "})
        await send({"type": "response.audio_transcript.delta", "response_id": "r1", "delta": "welcome."})
        await send({"type": "response.audio_transcript.done", "response_id": "r1", "transcript": "Hi, welcome."})
        await send({"type": "response.done", "response": {"id": "r1", "status": "completed", "output_transcript": "Hi, welcome.",
                                                          "usage": {"input_tokens": 3}}})
        await asyncio.sleep(self.video_delay)         # the voice finishes before the face renders
        for i in range(self.chunks):
            await send(chunk_event(28 + i * 7, tag=i + 1))

    async def __aenter__(self):
        self.server = await serve(self.handler, "127.0.0.1", 0, select_subprotocol=_select)
        port = self.server.sockets[0].getsockname()[1]
        self.base_url = "http://127.0.0.1:%d" % port
        return self

    async def __aexit__(self, *a):
        self.server.close()
        await self.server.wait_closed()


def client_for(srv, **kw):
    return Newera(api_key="nwc_test", base_url=srv.base_url, **kw)


# ----------------------------------------------------------------------------- parsing

def test_video_chunk_decodes_frames_and_pairs_audio():
    ev = parse_event(chunk_event(91, n=7, tag=5))
    assert isinstance(ev, VideoChunk)
    assert (ev.frame_index, ev.start_ms, ev.fps, ev.speech, ev.sample_rate) == (91, 3640, 25, True, AUDIO_RATE)
    assert ev.frames[0] == b"jpeg-91-0" and len(ev.frames) == 7
    assert len(ev.audio) == 7 * FRAME_BYTES                  # the audio of exactly these frames
    assert ev.duration_ms == pytest.approx(280)
    assert ev.type == "response.video.chunk"


def test_listening_face_chunk_has_no_audio():
    ev = parse_event(chunk_event(0, n=3, speech=False))
    assert ev.speech is False and ev.audio is None and len(ev.frames) == 3


def test_other_events_are_typed():
    assert isinstance(parse_event({"type": "avatar.ready", "fps": 25, "frames_per_event": 7}), AvatarReady)
    assert isinstance(parse_event({"type": "avatar.interrupted"}), Interrupted)
    assert parse_event({"type": "avatar.unavailable", "message": "gone"}).message == "gone"
    d = parse_event({"type": "response.audio.delta", "delta": b64(b"\x01\x02")})
    assert isinstance(d, AudioDelta) and d.audio == b"\x01\x02"
    assert parse_event({"type": "conversation.item.input_audio_transcription.completed", "transcript": "hi"}).text == "hi"
    assert parse_event({"type": "response.audio_transcript.delta", "delta": "x"}).text == "x"
    done = parse_event({"type": "response.done", "response": {"status": "completed", "output_transcript": "ok"}})
    assert isinstance(done, ResponseDone) and done.text == "ok" and not done.dropped
    assert parse_event({"type": "response.done", "response": {"status_details": {"reason": "turn_detected"}}}).dropped
    other = parse_event({"type": "conversation.item.created"})
    assert other.type == "conversation.item.created"


# ----------------------------------------------------------------------------- session

def test_build_session_defaults_and_options(tmp_path):
    s = build_session()
    assert s == {"avatar": "lama", "avatar_idle_video": False, "voice": "female"}
    img = tmp_path / "me.jpg"
    img.write_bytes(b"\xff\xd8face")
    s = build_session(image=str(img), face_crop=False, idle_video=True, language="ar", instructions="be brief",
                      settings={"turn_detection": None, "newera": {"voice_quality": "balanced"}})
    assert "avatar" not in s and s["avatar_image"] == b64(b"\xff\xd8face") and s["avatar_face_crop"] is False
    assert s["avatar_idle_video"] is True and s["instructions"] == "be brief" and s["turn_detection"] is None
    assert s["newera"] == {"voice_quality": "balanced", "detect_language": "ar"}


def test_bad_arguments_raise_before_connecting():
    client = Newera(api_key="k")
    with pytest.raises(ValueError):
        client.avatar.connect(model="gpt")
    with pytest.raises(ValueError):
        client.avatar.connect(voice="robot")
    with pytest.raises(ValueError):
        client.avatar.connect(language="fr")


def test_ws_url_from_base_url():
    assert Newera(api_key="k").avatar_ws_url == "wss://api.newera.ai/v1/avatar"
    assert Newera(api_key="k", base_url="http://localhost:9/").avatar_ws_url == "ws://localhost:9/v1/avatar"


# ----------------------------------------------------------------------------- a call

def test_call_sends_session_update_first_and_gets_typed_events():
    async def go():
        async with FakeAvatar() as srv:
            async with client_for(srv).avatar.connect("lama", voice="male", language="en", model="newera-s2s-mini",
                                                       record=True, instructions="be brief") as call:
                assert srv.paths[0] == "/v1/avatar?model=newera-s2s-mini"
                assert srv.headers[0]["authorization"] == "Bearer nwc_test"
                assert call.ready.frames_per_event == 7 and call.session["id"] == "sess_1"
                await call.send_text("hello")
                reply = await call.wait_reply(quiet=0.3)
                assert reply.user_text == "hello there" and reply.text == "Hi, welcome." and reply.status == "completed"
                assert len(reply.chunks) == 3 and len(reply.frames) == 21
                assert len(reply.audio) == 21 * FRAME_BYTES
                assert any(isinstance(e, ReplyTextDelta) for e in reply.events)
                assert any(isinstance(e, UserTranscript) for e in reply.events)
                tl = call.timeline
                assert len(tl) == 21
                frames, pcm = tl.timeline()
                assert frames[0] == b"jpeg-28-0" and frames[-1] == b"jpeg-42-6"
                assert len(pcm) == 21 * FRAME_BYTES
                assert pcm[:2] == struct.pack("<h", 1) and pcm[-2:] == struct.pack("<h", 3)     # chunk audio stays with its frames
            first = srv.received[0]
            assert first["type"] == "session.update"
            s = first["session"]
            assert s["avatar"] == "lama" and s["voice"] == "male" and s["newera"] == {"detect_language": "en"}
            assert s["avatar_idle_video"] is False and s["instructions"] == "be brief"
            assert any(m["type"] == "session.end" for m in srv.received)
            assert call.summary["billed"]["connected_seconds"] == 30
    run(go())


def test_events_iterator_ends_when_call_closes():
    async def go():
        async with FakeAvatar(chunks=1) as srv:
            async with client_for(srv).avatar.connect() as call:
                await call.say("greet")
                kinds = []
                async for ev in call.events():
                    kinds.append(type(ev).__name__)
                    if isinstance(ev, VideoChunk):
                        break
                assert "UserTranscript" in kinds and kinds[-1] == "VideoChunk"
    run(go())


def test_frames_and_audio_helpers():
    async def go():
        async with FakeAvatar(chunks=2) as srv:
            async with client_for(srv).avatar.connect() as call:
                await call.say("greet")
                got = []
                async for index, jpeg in call.frames():
                    got.append(index)
                    if len(got) == 14:
                        break
                assert got == list(range(28, 42))
    run(go())


def test_not_recorded_call_cannot_save():
    async def go():
        async with FakeAvatar() as srv:
            async with client_for(srv).avatar.connect() as call:
                with pytest.raises(RuntimeError, match="record=True"):
                    call.save_mp4("x.mp4")
    run(go())


def test_save_wav_frames_and_mp4_without_ffmpeg(tmp_path, monkeypatch):
    tl = Timeline()
    tl.add(parse_event(chunk_event(0, n=2, tag=7)))
    tl.add(parse_event(chunk_event(100, n=1, tag=9)))          # a later reply: the gap is shortened and silent
    frames, pcm = tl.timeline(max_gap_s=0.2)
    assert len(frames) == 2 + 5 + 1 and len(pcm) == 8 * FRAME_BYTES
    assert pcm[2 * FRAME_BYTES:7 * FRAME_BYTES] == bytes(5 * FRAME_BYTES)
    with wave.open(tl.save_wav(tmp_path / "a.wav")) as w:
        assert w.getframerate() == AUDIO_RATE and w.getnchannels() == 1
    assert len(os.listdir(tl.save_frames(tmp_path / "f"))) == 28
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        tl.save_mp4(tmp_path / "x.mp4")


def test_send_file_resamples_and_paces(tmp_path):
    path = tmp_path / "q.wav"
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x10\x00" * 3200)                      # 0.2 s at 16 kHz
    assert len(read_wav_24k(path)) == 4800 * 2                 # 0.2 s at 24 kHz

    async def go():
        async with FakeAvatar() as srv:
            async with client_for(srv).avatar.connect() as call:
                t0 = asyncio.get_running_loop().time()
                await call.send_file(path, tail=0.1, keep_silence=False)
                took = asyncio.get_running_loop().time() - t0
                assert 0.25 <= took < 1.0                      # real time: 0.2 s of audio + 0.1 s tail
                await asyncio.sleep(0.05)
                assert srv.audio_bytes == (4800 + 2400) * 2
    run(go())


def test_resample_stereo_to_mono():
    stereo = struct.pack("<4h", 100, 300, -100, -300)
    assert struct.unpack("<2h", resample_pcm16(stereo, 24000, 24000, channels=2)) == (200, -200)


# ----------------------------------------------------------------------------- client secret

class _Http(BaseHTTPRequestHandler):
    status, body, seen = 200, {}, []

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        _Http.seen.append((self.path, self.headers.get("Authorization"), json.loads(self.rfile.read(n) or b"{}")))
        self._reply()

    def do_GET(self):
        _Http.seen.append((self.path, self.headers.get("Authorization"), None))
        self._reply()

    def _reply(self):
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
    _Http.status, _Http.seen = 200, []
    yield "http://127.0.0.1:%d" % srv.server_port
    srv.shutdown()


def test_create_session_and_availability(http):
    _Http.body = {"object": "avatar.session", "model": "newera-s2s-mini",
                  "client_secret": {"value": "avs_abc", "expires_at": 1763300000}}

    async def go():
        client = Newera(api_key="nwc_test", base_url=http)
        secret = await client.avatar.create_session(model="newera-s2s-mini")
        assert (secret.value, secret.expires_at, secret.model) == ("avs_abc", 1763300000, "newera-s2s-mini")
        assert secret.as_dict()["client_secret"]["value"] == "avs_abc"
        _Http.body = {"object": "avatar.availability", "busy": True, "available": True}
        a = await client.avatar.availability()
        assert a.busy and not a.free
        _Http.body = {"busy": False, "available": True}
        assert (await client.avatar.availability()).free
    run(go())
    assert _Http.seen[0] == ("/v1/avatar/sessions", "Bearer nwc_test", {"model": "newera-s2s-mini"})
    assert _Http.seen[1][:2] == ("/v1/avatar/availability", "Bearer nwc_test")


def test_connect_with_client_secret_sends_subprotocol_and_no_key(monkeypatch):
    monkeypatch.delenv("NEWERA_API_KEY", raising=False)
    async def go():
        async with FakeAvatar() as srv:
            client = Newera(api_key=None, base_url=srv.base_url)
            from newera import ClientSecret
            async with client.avatar.connect(client_secret=ClientSecret("avs_xyz", 1, "newera-s2s")):
                pass
            assert srv.protocols[0] == "realtime, newera-client-secret.avs_xyz"
            assert "authorization" not in srv.headers[0]
            assert srv.paths[0] == "/v1/avatar"                       # the model was fixed when the secret was made
    run(go())


def test_create_session_http_errors_map(http, monkeypatch):
    monkeypatch.delenv("NEWERA_API_KEY", raising=False)
    _Http.status = 403
    _Http.body = {"error": {"type": "permission_error", "code": "avatar_not_enabled", "message": "not enabled"}}

    async def go():
        with pytest.raises(NotEnabled) as e:
            await Newera(api_key="k", base_url=http).avatar.create_session()
        assert e.value.code == "avatar_not_enabled" and e.value.http_status == 403
        with pytest.raises(AuthenticationError):
            await Newera(api_key=None, base_url=http).avatar.create_session()
    run(go())


# ----------------------------------------------------------------------------- errors

@pytest.mark.parametrize("code,close,cls", [
    ("avatar_busy", 4503, AvatarBusy), ("avatar_not_enabled", 4403, NotEnabled),
    ("model_not_entitled", 4403, NotEnabled), ("invalid_api_key", 4401, AuthenticationError),
    ("invalid_client_secret", 4401, AuthenticationError), ("concurrent_call_limit", 4429, RateLimited),
])
def test_refused_connection_raises_matching_error(code, close, cls):
    async def go():
        async with FakeAvatar(refuse=(code, close)) as srv:
            with pytest.raises(cls) as e:
                async with client_for(srv).avatar.connect():
                    pass
            assert e.value.code == code and e.value.close_code == close
            assert isinstance(e.value, NeweraError)
    run(go())


def test_error_for_falls_back_to_close_code_and_status():
    from newera import AvatarUnavailable, InsufficientQuota, InvalidRequest
    assert isinstance(error_for("weird", "m", close_code=4402), InsufficientQuota)
    assert isinstance(error_for("weird", "m", close_code=4400), InvalidRequest)
    assert isinstance(error_for(None, "m", http_status=503), AvatarUnavailable)
    assert type(error_for("weird", "m")) is AvatarError


def test_error_event_has_exception():
    ev = parse_event({"type": "error", "error": {"type": "t", "code": "avatar_busy", "message": "busy"}})
    assert isinstance(ev, ErrorEvent) and isinstance(ev.exception(4503), AvatarBusy)


# ----------------------------------------------------------------------------- the player's clock

def chunk(frame_index, n=7, speech=True):
    return parse_event(chunk_event(frame_index, n=n, speech=speech))


def test_playback_starts_a_reply_after_the_prebuffer_and_shows_frames_by_the_audio_clock():
    p = Playback(prebuffer=2.0)
    p.add(chunk(0))
    assert p.due_frame() is None                        # nothing before the cushion has played
    p.read(int(1.9 * AUDIO_RATE) * 2)
    assert p.due_frame() is None
    p.read(int(0.11 * AUDIO_RATE) * 2)                  # 2.01 s: the first frame's time has come
    assert p.due_frame() == b"jpeg-0-0"
    p.read(FRAME_BYTES * 2)                             # two more frames: the newest is shown, older skipped
    assert p.due_frame() == b"jpeg-0-2"


def test_playback_contiguous_chunks_follow_each_other_and_interrupt_drops_everything():
    p = Playback(prebuffer=0.5)
    p.add(chunk(0))
    p.add(chunk(7))                                      # contiguous: no new cushion
    assert p.buffered == pytest.approx(0.5 + 14 / 25, abs=0.01)
    p.interrupt()
    assert p.buffered == 0 and p.due_frame() is None and p.idle and p.interrupts == 1
    assert p.read(8) == bytes(8)                        # silence
    p.add(chunk(100))                                    # not contiguous after an interrupt: a cushion again
    assert p.buffered > 0.5


def test_playback_buffer_ran_dry_restarts_with_a_short_cushion():
    p = Playback(prebuffer=0.5)
    p.add(chunk(0, n=1))
    p.read(int(2 * AUDIO_RATE) * 2)                     # played out, then silence
    p.add(chunk(1, n=1))                                 # contiguous but dry
    assert 0.35 < p.buffered < 0.5


def test_playback_idle_video_keeps_the_clock_with_silence():
    p = Playback()
    p.add(chunk(0, n=3, speech=False))
    assert p.buffered == pytest.approx(0.2 + 3 / 25, abs=0.01)


def test_playback_audio_delta_fallback():
    p = Playback()
    p.add_audio(bytes(AUDIO_RATE * 2))
    assert p.buffered == pytest.approx(1.2, abs=0.01)


# ----------------------------------------------------------------------------- the CLI

def test_cli_parser_and_missing_key(monkeypatch, capsys):
    from newera.cli import build_parser, main
    args = build_parser().parse_args(["avatar", "--input-file", "q.wav", "--no-window", "--voice", "male"])
    assert args.input_file == "q.wav" and args.no_window and args.voice == "male"
    monkeypatch.delenv("NEWERA_API_KEY", raising=False)
    assert main(["avatar", "--no-window", "--mute"]) == 2
    assert "NEWERA_API_KEY" in capsys.readouterr().out
    assert main([]) == 0


def test_cli_busy_message(monkeypatch, capsys, http):
    _Http.body = {"busy": True, "available": True}
    from newera.cli import main
    monkeypatch.setenv("NEWERA_API_KEY", "k")
    monkeypatch.setenv("NEWERA_BASE_URL", http)
    assert main(["avatar", "--no-window", "--mute"]) == 3
    assert "busy" in capsys.readouterr().out


def test_wait_reply_keeps_collecting_video_that_arrives_after_response_done():
    async def go():
        async with FakeAvatar(chunks=3, video_delay=1.0) as srv:
            async with client_for(srv).avatar.connect() as call:
                await call.send_text("hello")
                reply = await call.wait_reply(quiet=1.5)
                assert len(reply.chunks) == 3 and len(reply.frames) == 21
    run(go())
