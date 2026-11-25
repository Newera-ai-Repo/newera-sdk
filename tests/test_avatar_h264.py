"""The opt-in H.264 video format: option and URL, parsing, the decoder's keyframe rules (fake codec), the player with a
fake decoder, saving, and (skipped without PyAV) real encode/decode/mux round trips. No network, no key.

The tests that need PyAV start with `pytest.importorskip("av")` and are named `test_av_*`."""
import asyncio
import base64
import json
import os
import struct
import subprocess
import sys
import threading
import time
import types
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.asyncio.server import serve

from newera import (AvatarError, AvatarReady, AvatarUnavailable, InvalidRequest, Interrupted, LipSyncPlayer, Newera,
                    VideoChunk)
from newera.avatar import AUDIO_RATE, Timeline, error_for, parse_event
from newera.h264 import DecodedFrame, H264Decoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAME_BYTES = AUDIO_RATE // 25 * 2
START = b"\x00\x00\x00\x01"


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


def _select(connection, subprotocols):
    return "realtime" if "realtime" in subprotocols else None


def fake_units(n, keys=(0,), tag="u"):
    """Fake access units: a start code, 'K' or 'P', then a payload of a different length for every unit."""
    return [START + (b"K" if i in keys else b"P") + ("%s%d" % (tag, i)).encode() * (i + 1) for i in range(n)]


def h264_event(frame_index, units=None, keys=(0,), *, speech=True, line_id="a", tag=1, fps=25):
    units = fake_units(7) if units is None else units
    n = len(units)
    start = frame_index * 1000 // fps
    pcm = struct.pack("<h", tag) * (FRAME_BYTES // 2 * n)
    return {"type": "response.video.chunk", "event_id": "evt_x", "index": 0, "frame_index": frame_index,
            "start_ms": start, "video": b64(b"".join(units)), "frame_sizes": [len(u) for u in units],
            "keyframes": list(keys), "frame_count": n, "video_format": "h264", "fps": fps, "width": 512,
            "height": 512, "speech": speech, "audio": b64(pcm) if speech else None, "audio_sample_rate": AUDIO_RATE,
            "audio_start_ms": start, "audio_end_ms": start + n * 1000 // fps, "line_id": line_id if speech else None,
            "generation_ms": 900, "queued_ms": 100}


def jpeg_event(frame_index, n=7, line_id="a"):
    start = frame_index * 40
    return {"type": "response.video.chunk", "event_id": "evt_x", "index": 0, "frame_index": frame_index,
            "start_ms": start, "frames": [b64(("jpeg-%d-%d" % (frame_index, i)).encode()) for i in range(n)],
            "fps": 25, "width": 512, "height": 512, "speech": True, "audio": b64(bytes(FRAME_BYTES * n)),
            "audio_sample_rate": AUDIO_RATE, "audio_start_ms": start, "audio_end_ms": start + n * 40,
            "line_id": line_id, "generation_ms": 900, "queued_ms": 100}


H264_INFO = {"codec": "avc1.4D401F", "stream": "annexb", "gop_frames": 50, "bitrate_kbps": 700, "keyframe_on_request": True}


class FakeServer:
    """A /v1/avatar that answers avatar.ready (jpeg or h264), and after `avatar.line.end` sends `events`."""

    def __init__(self, events=(), h264=True, refuse=None):
        self.events, self.h264, self.refuse = list(events), h264, refuse
        self.paths, self.received = [], []

    async def handler(self, ws):
        self.paths.append(ws.request.path)
        query = parse_qs(urlparse(ws.request.path).query)
        send = lambda ev: ws.send(json.dumps(ev))                                    # noqa: E731
        if self.refuse:
            code, close = self.refuse
            await send({"type": "error", "error": {"type": "x", "code": code, "message": code}})
            await ws.close(code=close, reason=code)
            return
        self.received.append(json.loads(await ws.recv()))
        ready = {"type": "avatar.ready", "fps": 25, "width": 512, "height": 512, "audio_sample_rate": AUDIO_RATE,
                 "frames_per_event": 7, "mode": (query.get("mode") or ["conversational"])[0]}
        if self.h264:
            ready.update(video_format="h264", video=H264_INFO)
        else:
            ready["video_format"] = "jpeg"
        await send(ready)
        await send({"type": "session.created", "session": {"id": "s"}})
        async for raw in ws:
            ev = json.loads(raw)
            self.received.append(ev)
            if ev["type"] == "avatar.line.end":
                for e in self.events:
                    await send(e)
                await send({"type": "avatar.line.done", "line_id": ev["line_id"], "status": "completed"})
            elif ev["type"] == "session.end":
                await send({"type": "session.summary", "summary": {}})
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


@pytest.fixture
def fake_av(monkeypatch):
    """A stand-in `av` module, so connect(video_format="h264") passes its PyAV check without PyAV."""
    monkeypatch.setitem(sys.modules, "av", types.ModuleType("av"))


# ------------------------------------------------------------------------------- option, URL, errors

def test_default_jpeg_url_and_session_are_unchanged(monkeypatch):
    monkeypatch.setitem(sys.modules, "av", None)                       # a JPEG call never needs PyAV

    async def go():
        async with FakeServer(h264=False) as srv:
            async with srv.client().avatar.connect(mode="render-audio") as call:
                assert call.video_format == "jpeg" and call.ready.video is None
            async with srv.client().avatar.connect(video_format="jpeg") as call:
                pass
            async with srv.client().avatar.connect() as call:
                pass
            assert srv.paths == ["/v1/avatar?mode=render-audio", "/v1/avatar?model=newera-s2s",
                                 "/v1/avatar?model=newera-s2s"]
    run(go())


def test_h264_option_goes_into_the_url_and_ready_carries_the_video_dict(fake_av):
    async def go():
        async with FakeServer() as srv:
            async with srv.client().avatar.connect(mode="render-audio", video_format="h264") as call:
                assert call.video_format == "h264" and isinstance(call.ready, AvatarReady)
                assert call.ready.video["codec"] == "avc1.4D401F" and call.ready.video["gop_frames"] == 50
            async with srv.client().avatar.connect(video_format="h264"):
                pass
            assert srv.paths == ["/v1/avatar?mode=render-audio&video_format=h264",
                                 "/v1/avatar?model=newera-s2s&video_format=h264"]
    run(go())


def test_h264_with_a_client_secret_keeps_the_option_in_the_url(fake_av):
    from newera import ClientSecret

    async def go():
        async with FakeServer() as srv:
            client = Newera(api_key=None, base_url=srv.base_url)
            secret = ClientSecret("avs_q", 1, "newera-s2s", "render-audio", "h264")
            assert secret.as_dict()["video_format"] == "h264"
            assert "video_format" not in ClientSecret("avs_q", 1, "newera-s2s").as_dict()
            async with client.avatar.connect(client_secret=secret):
                pass
            assert srv.paths == ["/v1/avatar?mode=render-audio&video_format=h264"]
    run(go())


def test_unknown_video_format_is_a_value_error():
    c = Newera(api_key="k")
    with pytest.raises(ValueError):
        c.avatar.connect(video_format="vp9")
    with pytest.raises(ValueError):
        run(c.avatar.create_session(video_format="vp9"))


def test_missing_pyav_is_a_clear_error_before_any_connection(monkeypatch):
    monkeypatch.setitem(sys.modules, "av", None)
    c = Newera(api_key="k", base_url="http://127.0.0.1:1")              # nothing listens here: it must not be tried
    with pytest.raises(AvatarError) as e:
        c.avatar.connect(video_format="h264")
    assert e.value.code == "av_not_installed" and "newera-sdk[h264]" in e.value.message


def test_error_codes_for_video_format():
    assert isinstance(error_for("invalid_video_format", close_code=4400), InvalidRequest)
    assert isinstance(error_for("video_format_unavailable", close_code=4503), AvatarUnavailable)
    assert isinstance(error_for(None, close_code=4503), AvatarUnavailable)


def test_server_refusal_of_h264_is_typed(fake_av):
    async def go():
        async with FakeServer(refuse=("video_format_unavailable", 4503)) as srv:
            with pytest.raises(AvatarUnavailable) as e:
                async with srv.client().avatar.connect(video_format="h264"):
                    pass
            assert e.value.code == "video_format_unavailable" and e.value.close_code == 4503
    run(go())


class _Http(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        _Http.seen.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"))
        data = json.dumps({"object": "avatar.session", "client_secret": {"value": "avs_h", "expires_at": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_create_session_body_only_changes_for_h264():
    srv = HTTPServer(("127.0.0.1", 0), _Http)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        async def go():
            client = Newera(api_key="k", base_url="http://127.0.0.1:%d" % srv.server_port)
            await client.avatar.create_session(model="newera-s2s-mini")
            await client.avatar.create_session(model="newera-s2s-mini", video_format="jpeg")
            s = await client.avatar.create_session(mode="render-audio", video_format="h264")
            assert s.video_format == "h264" and s.as_dict()["video_format"] == "h264"
        run(go())
    finally:
        srv.shutdown()
    assert _Http.seen == [{"model": "newera-s2s-mini"}, {"model": "newera-s2s-mini"},
                          {"mode": "render-audio", "video_format": "h264"}]


# ----------------------------------------------------------------------------------------------- parsing

def test_h264_chunk_is_split_by_frame_sizes():
    units = fake_units(7, keys=(0, 4))
    ev = parse_event(h264_event(14, units, keys=(0, 4)))
    assert isinstance(ev, VideoChunk) and ev.video_format == "h264"
    assert ev.units == units and ev.keyframes == [0, 4] and ev.frames == [] and ev.frame_count == 7
    assert (ev.frame_index, ev.audio_start_ms, ev.audio_end_ms, ev.line_id) == (14, 560, 840, "a")
    assert ev.duration_ms == pytest.approx(280) and ev.frame_times_ms() == [560 + 40 * i for i in range(7)]
    assert len(ev.audio) == 7 * FRAME_BYTES and ev.sample_rate == AUDIO_RATE and (ev.width, ev.height) == (512, 512)
    assert "h264 units=7" in repr(ev) and "keyframes=[0, 4]" in repr(ev) and "audio=%d bytes" % (7 * FRAME_BYTES) in repr(ev)


def test_h264_chunk_without_audio_and_short_chunk():
    ev = parse_event(h264_event(3, fake_units(2), speech=False))
    assert ev.audio is None and ev.frame_count == 2 and ev.speech is False and ev.line_id is None
    assert ev.audio_end_ms - ev.audio_start_ms == pytest.approx(80)
    assert "audio=None" in repr(ev)


def test_inconsistent_frame_sizes_are_rejected():
    ev = h264_event(0)
    ev["frame_sizes"][0] += 1
    with pytest.raises(ValueError):
        parse_event(ev)


def test_jpeg_chunk_parse_and_repr_are_unchanged():
    ev = parse_event(jpeg_event(14, n=5))
    assert ev.video_format == "jpeg" and ev.units == [] and ev.keyframes == [] and ev.frame_count == 5
    assert ev.frames[0] == b"jpeg-14-0" and ev.duration_ms == pytest.approx(200)
    assert ev.frame_times_ms() == [560 + 40 * i for i in range(5)] and ev.audio_end_ms == 760
    assert repr(ev) == ("VideoChunk(frame_index=14, start_ms=560, frames=5, audio=%d bytes, speech=True, "
                        "generation_ms=900)" % (5 * FRAME_BYTES))
    old = parse_event({"type": "response.video.chunk", "frame_index": 3, "start_ms": 120, "frames": [b64(b"x")] * 2, "fps": 25})
    assert (old.audio_start_ms, old.audio_end_ms, old.frame_count) == (120, 200, 2)


# --------------------------------------------------------------------------------------- the decoder

class FakeCodec:
    """`decode(unit)` -> one DecodedFrame per unit; a unit starting `BAD` raises. K / P are keyframe / delta."""
    made = 0

    def __init__(self):
        FakeCodec.made += 1
        self.closed = False

    def decode(self, unit):
        if unit.startswith(START + b"BAD"):
            raise RuntimeError("corrupt unit")
        return [DecodedFrame(2, 2, bytes(12))]

    def close(self):
        self.closed = True


def make_decoder(**kw):
    FakeCodec.made = 0
    lost = []
    clock = {"t": 0.0}
    dec = H264Decoder(on_lost=lambda: lost.append(clock["t"]), codec_factory=FakeCodec, clock=lambda: clock["t"], **kw)
    return dec, lost, clock


def chunk_of(frame_index, units, keys):
    return parse_event(h264_event(frame_index, units, keys))


def test_decoder_drops_units_until_the_first_keyframe_then_decodes_in_order():
    dec, lost, _ = make_decoder()
    out = dec.decode(chunk_of(10, fake_units(5, keys=(2,)), (2,)))
    assert [f.index for f in out] == [12, 13, 14] and [f.pts for f in out] == [12, 13, 14]
    assert dec.dropped == 2 and dec.keyframes == 1 and dec.errors == 0 and dec.synced and dec.frames == 3
    assert len(lost) == 1                                   # joining mid-stream: one request for a keyframe
    out = dec.decode(chunk_of(15, fake_units(3, keys=()), ()))
    assert [f.index for f in out] == [15, 16, 17] and len(lost) == 1


def test_decoder_resets_and_resyncs_at_every_later_keyframe():
    dec, _, _ = make_decoder()
    dec.decode(chunk_of(0, fake_units(4, keys=(0,)), (0,)))
    assert FakeCodec.made == 1
    dec.decode(chunk_of(4, fake_units(4, keys=(2,)), (2,)))          # a keyframe in the middle: a fresh decoder
    assert FakeCodec.made == 2 and dec.keyframes == 2 and dec.synced and dec.dropped == 0


def test_decoder_counts_a_failed_unit_asks_for_a_keyframe_and_waits_for_it():
    dec, lost, clock = make_decoder()
    units = [START + b"K0", START + b"P1", START + b"BAD2", START + b"P3", START + b"P4", START + b"K5", START + b"P6"]
    out = dec.decode(chunk_of(0, units, (0, 5)))
    assert dec.errors == 1 and [f.index for f in out] == [0, 1, 5, 6]
    assert dec.dropped == 2 and dec.synced                       # P3, P4 dropped until the keyframe K5
    assert len(lost) == 1                                        # asked at the failure; the drops within 1 s do not ask again
    clock["t"] = 5.0
    dec.decode(chunk_of(7, [START + b"BAD", START + b"P"], ()))
    assert dec.errors == 2 and len(lost) == 2 and not dec.synced
    clock["t"] = 5.2
    dec.decode(chunk_of(9, [START + b"P"], ()))                  # still waiting; asked 0.2 s ago
    assert len(lost) == 2
    clock["t"] = 6.5
    dec.decode(chunk_of(10, [START + b"P"], ()))                 # still no keyframe after a second: ask again
    assert len(lost) == 3


def test_decoder_reset_drops_the_picture_until_the_next_keyframe():
    dec, lost, _ = make_decoder()
    dec.decode(chunk_of(0, fake_units(3, keys=(0,)), (0,)))
    dec.reset()
    assert not dec.synced
    out = dec.decode(chunk_of(3, fake_units(3, keys=(1,)), (1,)))
    assert [f.index for f in out] == [4, 5] and dec.dropped == 1 and dec.synced


def test_decoder_notices_a_hole_in_the_stream():
    dec, lost, _ = make_decoder()
    dec.decode(chunk_of(0, fake_units(7, keys=(0,)), (0,)))
    out = dec.decode(chunk_of(30, fake_units(3, keys=()), ()))       # frames 7..29 never came and no keyframe
    assert out == [] and dec.dropped == 3 and not dec.synced and len(lost) == 1
    out = dec.decode(chunk_of(40, fake_units(2, keys=(0,)), (0,)))   # a keyframe heals it, wherever it is
    assert [f.index for f in out] == [40, 41]


def test_decoder_callback_errors_do_not_break_decoding_and_jpeg_chunks_are_refused():
    def boom():
        raise RuntimeError("callback")
    dec = H264Decoder(on_lost=boom, codec_factory=FakeCodec)
    assert dec.decode(chunk_of(0, fake_units(2, keys=()), ())) == [] and dec.dropped == 2
    with pytest.raises(ValueError):
        dec.decode(parse_event(jpeg_event(0)))


# ------------------------------------------------------------------------- the player, fake decoder

class FakeDecoder:
    """What LipSyncPlayer needs of a decoder: decode(chunk), reset(), on_lost. Logs where and in which order."""

    def __init__(self, delay=0.0, lose_on=None):
        self.on_lost, self.delay, self.lose_on = None, delay, lose_on
        self.decoded, self.threads, self.resets = [], set(), 0

    def decode(self, chunk):
        self.threads.add(threading.current_thread().name)
        if self.delay:
            time.sleep(self.delay)
        self.decoded.append(chunk.frame_index)
        if self.lose_on == chunk.frame_index and self.on_lost:
            self.on_lost()
        return [DecodedFrame(1, 1, b"\0\0\0", chunk.frame_index + i, chunk.frame_index + i) for i in range(chunk.frame_count)]

    def reset(self):
        self.resets += 1


class JpegFrame(DecodedFrame):
    def to_jpeg(self, quality=90):
        return b"jpeg:%d" % self.index


def test_player_shows_decoded_frames_in_order_on_the_same_clock_as_jpeg():
    shown, audio = [], []
    t0 = time.monotonic()

    async def go():
        dec = FakeDecoder()
        p = LipSyncPlayer(lambda pcm: audio.append((time.monotonic() - t0, len(pcm))),
                          lambda f, n: shown.append((time.monotonic() - t0, n, f)), buffer_ms=100, audio_lead_ms=30,
                          decoder=dec)

        async def source():
            for fi in (0, 5):
                yield parse_event(h264_event(fi, fake_units(5, keys=(0,))))
        await p.play(source())
        return p, dec
    p, dec = run(go())
    assert [n for _, n, _ in shown] == list(range(10)) and all(isinstance(f, DecodedFrame) for _, _, f in shown)
    assert dec.decoded == [0, 5] and threading.main_thread().name not in dec.threads    # decoded off the loop's thread
    gaps = [b[0] - a[0] for a, b in zip(shown, shown[1:])]
    assert all(0.02 < g < 0.09 for g in gaps)                          # 40 ms apart, as for JPEG
    assert shown[0][0] == pytest.approx(0.1, abs=0.04) and len(audio) == 2 and audio[0][0] < shown[0][0]
    assert p.late_ms_total == 0


def test_player_decodes_without_blocking_the_event_loop_and_keeps_chunk_order():
    shown, stalls = [], []

    async def ticker():
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.005)
            now = time.monotonic()
            stalls.append(now - last)
            last = now

    async def go():
        dec = FakeDecoder(delay=0.08)                                   # a slow decode: 80 ms per chunk
        p = LipSyncPlayer(lambda pcm: None, lambda f, n: shown.append(n), buffer_ms=60, decoder=dec)
        tick = asyncio.ensure_future(ticker())
        task = asyncio.ensure_future(p.play())
        for fi in (0, 7, 14, 21):
            p.feed(parse_event(h264_event(fi)))
        await asyncio.sleep(0.9)
        p.close()
        await task
        tick.cancel()
        return dec
    dec = run(go())
    assert dec.decoded == [0, 7, 14, 21] and shown == list(range(28))
    assert max(stalls) < 0.06                                            # the 80 ms sleeps never ran on the loop


def test_player_interrupt_resets_the_decoder_and_drops_scheduled_frames():
    shown = []

    async def go():
        dec = FakeDecoder()
        p = LipSyncPlayer(lambda pcm: None, lambda f, n: shown.append(n), buffer_ms=50, decoder=dec)
        task = asyncio.ensure_future(p.play())
        p.feed(parse_event(h264_event(0)))
        p.feed(parse_event(h264_event(7)))
        await asyncio.sleep(0.02)
        p.feed(Interrupted())
        await asyncio.sleep(0.5)
        assert dec.resets == 1 and shown == []                           # nothing was due yet: all dropped
        before = list(shown)
        p.feed(parse_event(h264_event(50, line_id="b")))                 # the server's keyframe after the interrupt
        await asyncio.sleep(0.5)
        p.close()
        await task
        return dec, before
    dec, before = run(go())
    assert dec.decoded == [0, 7, 50] and shown[len(before):] == list(range(50, 57))


def test_player_asks_for_a_keyframe_when_the_decoder_loses_sync():
    asked = []

    async def request():
        asked.append(1)

    async def go():
        dec = FakeDecoder(lose_on=7)
        p = LipSyncPlayer(lambda pcm: None, lambda f, n: None, buffer_ms=30, decoder=dec, request_keyframe=request)
        task = asyncio.ensure_future(p.play())
        for fi in (0, 7):
            p.feed(parse_event(h264_event(fi)))
        await asyncio.sleep(0.5)
        p.close()
        await task
        return p
    p = run(go())
    assert asked == [1] and p.keyframe_requests == 1


def test_player_decodes_idle_chunks_it_does_not_show_and_converts_to_jpeg_on_request():
    shown = []

    async def go():
        dec = FakeDecoder()
        orig = dec.decode
        dec.decode = lambda chunk: [JpegFrame(1, 1, b"\0\0\0", f.pts, f.index) for f in orig(chunk)]
        p = LipSyncPlayer(lambda pcm: None, lambda f, n: shown.append((n, f)), buffer_ms=30, decoder=dec,
                          show_idle=False, frame_format="jpeg")
        task = asyncio.ensure_future(p.play())
        p.feed(parse_event(h264_event(0, fake_units(2), speech=False)))     # idle: decoded (never skip), not shown
        p.feed(parse_event(h264_event(2, fake_units(2))))
        await asyncio.sleep(0.4)
        p.close()
        await task
        return dec
    dec = run(go())
    assert dec.decoded == [0, 2] and shown == [(2, b"jpeg:2"), (3, b"jpeg:3")]
    with pytest.raises(ValueError):
        LipSyncPlayer(lambda p: None, lambda f, n: None, frame_format="png")


def test_jpeg_player_still_gets_bytes_and_never_makes_a_decoder(monkeypatch):
    monkeypatch.setitem(sys.modules, "av", None)
    shown = []

    async def go():
        p = LipSyncPlayer(lambda pcm: None, lambda f, n: shown.append((n, f)), buffer_ms=30)
        task = asyncio.ensure_future(p.play())
        p.feed(parse_event(jpeg_event(0, n=3)))
        await asyncio.sleep(0.3)
        p.close()
        await task
        assert p.decoder is None
    run(go())
    assert shown == [(i, b"jpeg-0-%d" % i) for i in range(3)]


# ------------------------------------------------------------------------------ call: events, saving

def test_call_h264_events_keyframe_request_and_raw_save(fake_av, tmp_path):
    units = fake_units(7, keys=(0,))
    ev0, ev1 = h264_event(0, units, keys=(0,)), h264_event(7, fake_units(7, keys=(), tag="v"), keys=())

    async def go():
        async with FakeServer(events=[ev0, ev1]) as srv:
            async with srv.client().avatar.connect(mode="render-audio", video_format="h264", record=True) as call:
                got = []
                line = call.start_line("a")
                await line.append(bytes(FRAME_BYTES * 14))
                await line.end()
                async for ev in call.events():
                    got.append(ev)
                    if type(ev).__name__ == "LineDone":
                        break
                await call.request_keyframe()
                await call.close()
                return call, got, srv
    call, got, srv = run(go())
    chunks = [e for e in got if isinstance(e, VideoChunk)]
    assert [c.video_format for c in chunks] == ["h264", "h264"] and chunks[0].units == units
    assert {"type": "avatar.video.keyframe"} in srv.received
    assert len(call.timeline) == 14 and call.timeline.is_h264
    out = call.save_h264(tmp_path / "out.h264")
    assert open(out, "rb").read() == b"".join(units) + b"".join(fake_units(7, keys=(), tag="v"))
    call.save_wav(tmp_path / "a.wav")
    assert os.path.getsize(tmp_path / "a.wav") == 44 + 14 * FRAME_BYTES


def test_request_keyframe_is_harmless_on_a_jpeg_call(monkeypatch):
    async def go():
        async with FakeServer(h264=False) as srv:
            async with srv.client().avatar.connect() as call:
                await call.request_keyframe()
                await call.close()
            assert all(e["type"] != "avatar.video.keyframe" for e in srv.received)
    run(go())


def test_timeline_h264_starts_at_a_keyframe_and_shrinks_long_gaps():
    tl = Timeline()
    tl.add(parse_event(h264_event(100, fake_units(2, keys=(), tag="x"), keys=())))          # joined mid-stream
    tl.add(parse_event(h264_event(102, fake_units(3, keys=(0,), tag="a"), keys=(0,), tag=2)))
    tl.add(parse_event(h264_event(105, fake_units(2, keys=(), tag="b"), keys=(), tag=3)))
    tl.add(parse_event(h264_event(400, fake_units(2, keys=(0,), tag="c"), keys=(0,), tag=4)))   # 295 frames later
    layout, pcm = tl._h264_layout()
    assert [n for n, _ in layout] == [102, 103, 104, 105, 106, 400, 401]
    assert [s for _, s in layout] == [0, 1, 2, 3, 4, 30, 31]               # the gap is cut to 25 frames, none repeated
    assert len(pcm) == 32 * FRAME_BYTES and pcm[5 * FRAME_BYTES + 2:5 * FRAME_BYTES + 4] == bytes(2)    # silence in the gap
    assert pcm[:2] == struct.pack("<h", 2) and pcm[-2:] == struct.pack("<h", 4)
    assert tl.seconds == pytest.approx(7 / 25) and len(tl) == 9


def test_jpeg_timeline_and_saves_are_unchanged(tmp_path):
    tl = Timeline()
    tl.add(parse_event(jpeg_event(0, n=3)))
    tl.add(parse_event(jpeg_event(5, n=2)))
    frames, pcm = tl.timeline()
    assert frames == [b"jpeg-0-0", b"jpeg-0-1", b"jpeg-0-2", b"jpeg-0-2", b"jpeg-0-2", b"jpeg-5-0", b"jpeg-5-1"]
    assert len(pcm) == 7 * FRAME_BYTES and not tl.is_h264 and len(tl) == 5
    tl.save_frames(tmp_path / "f")
    assert sorted(os.listdir(tmp_path / "f")) == ["%06d.jpg" % i for i in range(7)]
    with pytest.raises(RuntimeError):
        tl.save_h264(tmp_path / "x.h264")


# ------------------------------------------------------------------ JPEG paths never load PyAV

def test_importing_the_sdk_and_a_jpeg_call_never_load_pyav(monkeypatch):
    code = ("import sys; import newera, newera.h264, newera.player, newera.talk, newera.cli; "
            "assert 'av' not in sys.modules, 'av was imported'")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [os.path.join(ROOT, "src"), os.environ.get("PYTHONPATH", "")]))},
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_the_jpeg_suites_pass_with_pyav_blocked():
    code = ("import sys; sys.modules['av'] = None; import pytest; "
            "sys.exit(pytest.main(['-q', '-x', '-p', 'no:cacheprovider', 'tests/test_avatar.py', 'tests/test_avatar_render.py']))")
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [os.path.join(ROOT, "src"), os.environ.get("PYTHONPATH", "")]))},
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-1000:]


# =========================================================================== real PyAV (skipped here)

def _fill(plane, value, width, height):
    stride = plane.line_size
    plane.update((bytes([value]) * width + bytes(stride - width)) * height)


def encode_units(av, n, *, width=64, height=64, gop=6, rgb=(200, 60, 60), shift=0):
    """n frames of a flat colour, encoded to Annex-B access units with libx264 the way the server does it
    (no B-frames, SPS and PPS in front of every keyframe). Returns [(unit bytes, is_keyframe)]."""
    r, g, b = rgb
    y = round(16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255) + shift
    u = round(128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255)
    v = round(128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255)
    ctx = av.CodecContext.create("libx264", "w")
    ctx.width, ctx.height, ctx.pix_fmt = width, height, "yuv420p"
    ctx.time_base, ctx.framerate = Fraction(1, 25), Fraction(25, 1)
    ctx.options = {"preset": "ultrafast", "tune": "zerolatency",
                   "x264-params": "keyint=%d:min-keyint=%d:scenecut=0:bframes=0:repeat-headers=1:aud=0:"
                                  "colorprim=smpte170m:transfer=smpte170m:colormatrix=smpte170m:fullrange=off" % (gop, gop)}
    ctx.open()
    out = []
    for i in range(n):
        frame = av.VideoFrame(width, height, "yuv420p")
        _fill(frame.planes[0], min(235, y + i), width, height)
        _fill(frame.planes[1], u, width // 2, height // 2)
        _fill(frame.planes[2], v, width // 2, height // 2)
        frame.pts = i
        for p in ctx.encode(frame):
            out.append((bytes(p), bool(p.is_keyframe)))
    for p in ctx.encode(None):
        out.append((bytes(p), bool(p.is_keyframe)))
    assert len(out) == n and out[0][1], "the encoder must give one unit per frame and start with a keyframe"
    return out


def chunks_of(encoded, first_frame=0, per=7, tag=1):
    out = []
    for off in range(0, len(encoded), per):
        part = encoded[off:off + per]
        keys = [i for i, (_, k) in enumerate(part) if k]
        out.append(parse_event(h264_event(first_frame + off, [u for u, _ in part], keys, tag=tag)))
    return out


def nals_annexb(data):
    """NAL units (payload bytes, start codes removed) of an Annex-B byte string."""
    parts, i, n, starts = [], 0, len(data), []
    while i + 3 <= n:
        if data[i:i + 3] == b"\x00\x00\x01":
            starts.append(i + 3)
            i += 3
        else:
            i += 1
    for k, s in enumerate(starts):
        e = (starts[k + 1] - 3) if k + 1 < len(starts) else n
        parts.append(data[s:e].rstrip(b"\x00") if k + 1 < len(starts) else data[s:e])
    return parts


def nals_avcc(data, length_size=4):
    """NAL units of a length-prefixed (MP4) sample."""
    out, i = [], 0
    while i < len(data):
        size = int.from_bytes(data[i:i + length_size], "big")
        out.append(data[i + length_size:i + length_size + size])
        i += length_size + size
    return out


def test_av_decoder_round_trip_order_size_colour_and_jpeg():
    av = pytest.importorskip("av")
    enc = encode_units(av, 12, gop=6)
    lost = []
    dec = H264Decoder(on_lost=lambda: lost.append(1))
    frames = []
    for c in chunks_of(enc):
        frames += dec.decode(c)
    assert [f.index for f in frames] == list(range(12)) and dec.errors == 0 and dec.dropped == 0 and dec.keyframes == 2
    f = frames[0]
    assert (f.width, f.height) == (64, 64) and len(f.rgb) == 64 * 64 * 3
    centre = f.rgb[(32 * 64 + 32) * 3:(32 * 64 + 32) * 3 + 3]
    for got, want in zip(centre, (200, 60, 60)):                       # BT.601 limited range, as signalled
        assert abs(got - want) <= 14, (list(centre), (200, 60, 60))
    assert frames[11].rgb != frames[0].rgb                              # the pictures differ frame to frame
    assert f.to_jpeg(90)[:2] == b"\xff\xd8" and f._jpeg_with_av(80)[:2] == b"\xff\xd8"
    try:
        import numpy
        assert f.to_ndarray().shape == (64, 64, 3)
    except ImportError:
        with pytest.raises(ImportError):
            f.to_ndarray()


def test_av_decoder_joins_mid_stream_at_the_next_keyframe():
    av = pytest.importorskip("av")
    enc = encode_units(av, 12, gop=6)
    lost = []
    dec = H264Decoder(on_lost=lambda: lost.append(1))
    late = chunks_of(enc[2:], first_frame=2)                            # we missed units 0 and 1
    frames = []
    for c in late:
        frames += dec.decode(c)
    assert dec.dropped == 4 and [f.index for f in frames] == list(range(6, 12)) and lost == [1]
    dec.reset()
    assert dec.decode(chunks_of(enc[7:9], first_frame=7)[0]) == [] and not dec.synced      # waits for a keyframe again
    assert dec.dropped == 6


def test_av_player_shows_every_frame_in_order():
    av = pytest.importorskip("av")
    enc = encode_units(av, 12, gop=6)
    shown = []

    async def go():
        p = LipSyncPlayer(lambda pcm: None, lambda f, n: shown.append((n, f)), buffer_ms=60)

        async def source():
            for c in chunks_of(enc):
                yield c
        await p.play(source())
    run(go())
    assert [n for n, _ in shown] == list(range(12)) and all(isinstance(f, DecodedFrame) for _, f in shown)
    assert all(f.index == n and len(f.rgb) == 64 * 64 * 3 for n, f in shown)


def test_av_save_mp4_copies_the_access_units_without_re_encoding(tmp_path):
    av = pytest.importorskip("av")
    enc_a, enc_b = encode_units(av, 6, gop=6), encode_units(av, 4, gop=6, shift=20)
    lead = encode_units(av, 3, gop=6)[1:]                               # units that cannot start a picture
    tl = Timeline()
    for c in chunks_of(lead, first_frame=100, tag=5):                    # received before any keyframe
        tl.add(c)
    for c in chunks_of(enc_a, first_frame=102, tag=1):
        tl.add(c)
    for c in chunks_of(enc_b, first_frame=112, tag=2):                   # frames 108..111 never came
        tl.add(c)
    path = str(tmp_path / "out.mp4")
    assert tl.save_mp4(path) == path
    with av.open(path) as mp4:
        video, audio = mp4.streams.video[0], mp4.streams.audio[0]
        assert video.codec_context.name == "h264" and audio.codec_context.name == "aac"
        assert (video.width, video.height) == (64, 64) and audio.codec_context.sample_rate == AUDIO_RATE
        packets = [p for p in mp4.demux(video) if p.size]
        pts = [round(p.pts * video.time_base * 25) for p in packets]
        assert pts == [0, 1, 2, 3, 4, 5, 10, 11, 12, 13]                # from frame numbers; none repeated or invented
        assert packets[0].is_keyframe
        sent = [u for u, _ in enc_a] + [u for u, _ in enc_b]
        assert len(packets) == len(sent)
        for pkt, unit in zip(packets, sent):                            # the same NAL units: the video was not re-encoded
            want = [n for n in nals_annexb(unit)]
            have = nals_avcc(bytes(pkt))
            assert [n for n in have if n[0] & 0x1F in (1, 5)] == [n for n in want if n[0] & 0x1F in (1, 5)]
    with av.open(path) as mp4:
        total = sum(p.duration * p.time_base for p in mp4.demux(mp4.streams.audio[0]) if p.size)
        assert abs(float(total) - 14 / 25) < 0.12                        # audio runs on the same clock as the video


def test_av_save_mp4_through_a_call_and_save_frames(tmp_path):
    av = pytest.importorskip("av")
    enc = encode_units(av, 12, gop=6)

    async def go():
        async with FakeServer(events=[h264_event(c.frame_index, c.units, c.keyframes) for c in chunks_of(enc)]) as srv:
            async with srv.client().avatar.connect(mode="render-audio", video_format="h264", record=True) as call:
                line = call.start_line("a")
                await line.append(bytes(FRAME_BYTES * 12))
                await line.end()
                assert await line.wait(10) == "completed"
                await asyncio.sleep(0.2)
                path = call.save_mp4(tmp_path / "call.mp4")
                frames_dir = call.save_frames(tmp_path / "frames")
                await call.close()
                return path, frames_dir
    path, frames_dir = run(go())
    with av.open(path) as mp4:
        assert len([p for p in mp4.demux(mp4.streams.video[0]) if p.size]) == 12
    names = sorted(os.listdir(frames_dir))
    assert names == ["%06d.jpg" % i for i in range(12)]
    assert open(os.path.join(frames_dir, names[0]), "rb").read(2) == b"\xff\xd8"
