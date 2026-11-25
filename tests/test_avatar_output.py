"""The `output` option ("original" | "square"), the new defaults (h264 + original), `image=` uploads and pictures that
are not square (the player, save_mp4, the window). The fake server answers like the Core gateway: the ready event says the
output it really serves, and why it could not serve "original"."""

import asyncio
import base64
import json
import os
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.asyncio.server import serve

from newera import AvatarReady, LipSyncPlayer, Newera, VideoChunk
from newera.avatar import AUDIO_RATE, Timeline, error_for, parse_event, build_session
from newera.avatar import InvalidRequest
from newera.talk import _fit

FRAME_BYTES = AUDIO_RATE // 25 * 2


def b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


def _select(connection, subprotocols):
    return "realtime" if "realtime" in subprotocols else None


@pytest.fixture
def fake_av(monkeypatch):
    monkeypatch.setitem(sys.modules, "av", types.ModuleType("av"))


class Server:
    """Serves ready for the options in the URL; `photo=False` is a call without an uploaded photo (a preset face)."""

    def __init__(self, width=1024, height=1280):
        self.width, self.height = width, height
        self.queries, self.sessions = [], []

    async def handler(self, ws):
        q = {k: v[0] for k, v in parse_qs(urlparse(ws.request.path).query).items()}
        self.queries.append(q)
        first = json.loads(await ws.recv())
        self.sessions.append(first["session"])
        fmt = q.get("video_format", "jpeg")
        want = q.get("output", "original")
        has_photo = "avatar_image" in first["session"]
        out = "original" if want == "original" and has_photo else "square"
        ready = {"type": "avatar.ready", "fps": 25, "video_format": fmt, "output": out, "audio_sample_rate": AUDIO_RATE,
                 "frames_per_event": 7, "mode": q.get("mode", "conversational"),
                 "width": self.width if out == "original" else 512, "height": self.height if out == "original" else 512}
        if want == "original" and out == "square":
            ready["output_reason"] = "preset_face"
        if fmt == "h264":
            ready["video"] = {"codec": "avc1.4D401F", "stream": "annexb", "gop_frames": 50, "bitrate_kbps": 700,
                              "keyframe_on_request": True}
        await ws.send(json.dumps(ready))
        await ws.send(json.dumps({"type": "session.created", "session": {"id": "s"}}))
        async for raw in ws:
            if json.loads(raw)["type"] == "session.end":
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


# ------------------------------------------------------------------------------------ the options in the URL

def test_defaults_are_h264_and_original_and_are_always_said(fake_av):
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect(mode="render-text", image=b"\xff\xd8photo", transport="websocket") as call:
                assert call.video_format == "h264" and call.ready.output == "original"
                assert (call.ready.width, call.ready.height) == (1024, 1280)
            assert srv.queries[0] == {"mode": "render-text", "video_format": "h264", "output": "original", "transport": "websocket"}
    run(go())


def test_square_and_jpeg_options_and_every_combination(fake_av):
    async def go():
        async with Server() as srv:
            seen = {}
            for output in ("original", "square"):
                for fmt in ("h264", "jpeg"):
                    async with srv.client().avatar.connect(mode="render-audio", image=b"\xff\xd8photo", output=output,
                                                           video_format=fmt, transport="websocket") as call:
                        seen[(output, fmt)] = (call.ready.output, call.ready.video_format, call.ready.width, call.ready.height)
            assert seen == {("original", "h264"): ("original", "h264", 1024, 1280), ("original", "jpeg"): ("original", "jpeg", 1024, 1280),
                            ("square", "h264"): ("square", "h264", 512, 512), ("square", "jpeg"): ("square", "jpeg", 512, 512)}
            assert [(q["output"], q["video_format"]) for q in srv.queries] == [
                ("original", "h264"), ("original", "jpeg"), ("square", "h264"), ("square", "jpeg")]
    run(go())


def test_original_without_a_photo_is_served_square_and_the_reason_is_in_ready(fake_av):
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect("lama", mode="render-audio", transport="websocket") as call:        # a preset face
                r = call.ready
                assert (r.output, r.output_reason, r.width, r.height) == ("square", "preset_face", 512, 512)
    run(go())


def test_an_older_server_that_does_not_know_output_reads_as_square():
    ev = parse_event({"type": "avatar.ready", "fps": 25, "width": 512, "height": 512, "video_format": "jpeg"})
    assert isinstance(ev, AvatarReady) and ev.output == "square" and ev.output_reason is None


def test_ready_parses_output_and_reason_and_the_size():
    ev = parse_event({"type": "avatar.ready", "fps": 25, "width": 1024, "height": 1280, "video_format": "h264",
                      "output": "original", "frames_per_event": 7, "mode": "render-text"})
    assert (ev.output, ev.width, ev.height, ev.output_reason) == ("original", 1024, 1280, None)
    ev = parse_event({"type": "avatar.ready", "output": "square", "output_reason": "no_face_found"})
    assert ev.output_reason == "no_face_found"


def test_unknown_values_are_value_errors_and_the_server_codes_map():
    c = Newera(api_key="k", base_url="http://127.0.0.1:1")
    with pytest.raises(ValueError, match="output"):
        c.avatar.connect(output="circle", transport="websocket")
    with pytest.raises(ValueError, match="output"):
        run(c.avatar.create_session(output="circle"))
    assert isinstance(error_for("invalid_output", close_code=4400), InvalidRequest)


def test_without_pyav_the_default_falls_back_to_jpeg_with_a_warning_but_h264_is_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "av", None)
    c = Newera(api_key="k", base_url="http://127.0.0.1:1")
    with pytest.warns(UserWarning, match="PyAV"):
        conn = c.avatar.connect(transport="websocket")
    assert conn.video_format == "jpeg" and conn.output == "original"
    from newera import AvatarError
    with pytest.raises(AvatarError) as e:
        c.avatar.connect(video_format="h264", transport="websocket")
    assert e.value.code == "av_not_installed"


# ------------------------------------------------------------------------------------ image upload

def test_image_as_bytes_path_base64_and_data_url(tmp_path):
    photo = b"\xff\xd8\xff\xe0photo"
    p = tmp_path / "me.jpg"
    p.write_bytes(photo)
    for image in (photo, str(p), p, b64(photo), "data:image/jpeg;base64," + b64(photo)):
        s = build_session(image=image)
        assert "avatar" not in s and base64.b64decode(s["avatar_image"].split(",")[-1]) == photo
    assert build_session(image=photo, face_crop=False)["avatar_face_crop"] is False
    with pytest.raises(TypeError):
        build_session(image=12)


def test_the_photo_goes_in_the_first_session_update(fake_av):
    async def go():
        async with Server() as srv:
            async with srv.client().avatar.connect(mode="render-text", image=b"\xff\xd8photo", transport="websocket"):
                pass
            assert base64.b64decode(srv.sessions[0]["avatar_image"]) == b"\xff\xd8photo"
    run(go())


# ------------------------------------------------------------------------------------ client secrets

class _Http(BaseHTTPRequestHandler):
    seen = []

    def do_POST(self):
        _Http.seen.append(json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"))
        data = json.dumps({"object": "avatar.session", "mode": "render-audio",
                           "client_secret": {"value": "avs_o", "expires_at": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_create_session_passes_both_options_and_the_secret_keeps_them(fake_av):
    http = HTTPServer(("127.0.0.1", 0), _Http)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    _Http.seen = []
    try:
        async def go():
            c = Newera(api_key="k", base_url="http://127.0.0.1:%d" % http.server_port)
            s = await c.avatar.create_session(mode="render-audio", output="square", video_format="jpeg")
            assert (s.output, s.video_format) == ("square", "jpeg")
            d = s.as_dict()
            assert d["output"] == "square" and d["video_format"] == "jpeg"
            plain = await c.avatar.create_session(mode="render-audio")
            assert plain.output is None and "output" not in plain.as_dict()
            async with Server() as srv:
                client = Newera(api_key=None, base_url=srv.base_url)
                async with client.avatar.connect(client_secret=s, image=b"\xff\xd8photo", transport="websocket"):
                    pass
                async with client.avatar.connect(client_secret=plain, image=b"\xff\xd8photo", transport="websocket"):
                    pass
                assert (srv.queries[0]["output"], srv.queries[0]["video_format"]) == ("square", "jpeg")   # the secret's
                assert (srv.queries[1]["output"], srv.queries[1]["video_format"]) == ("original", "h264")  # the defaults
        run(go())
    finally:
        http.shutdown()
    assert _Http.seen == [{"mode": "render-audio", "video_format": "jpeg", "output": "square"}, {"mode": "render-audio"}]


# ------------------------------------------------------------------------------------ pictures that are not square

def jpeg_chunk(frame_index, width, height, n=3):
    start = frame_index * 40
    return {"type": "response.video.chunk", "event_id": "e", "index": 0, "frame_index": frame_index, "start_ms": start,
            "frames": [b64(("f%d-%d" % (frame_index, i)).encode()) for i in range(n)], "fps": 25, "width": width,
            "height": height, "speech": True, "audio": b64(bytes(FRAME_BYTES * n)), "audio_sample_rate": AUDIO_RATE,
            "audio_start_ms": start, "audio_end_ms": start + n * 40, "line_id": "a", "generation_ms": 1, "queued_ms": 1}


def test_a_video_chunk_carries_the_picture_size_and_the_player_does_not_care():
    ev = parse_event(jpeg_chunk(0, 1024, 1280))
    assert isinstance(ev, VideoChunk) and (ev.width, ev.height) == (1024, 1280)
    shown = []

    async def go():
        p = LipSyncPlayer(lambda pcm: None, lambda f, n: shown.append((n, f)), buffer_ms=30)
        task = asyncio.ensure_future(p.play())
        p.feed(ev)
        await asyncio.sleep(0.3)
        p.close()
        await task
    run(go())
    assert shown == [(i, b"f0-%d" % i) for i in range(3)]


def test_the_window_takes_the_shape_of_the_picture():
    assert _fit(1024, 1280, 512) == (410, 512)
    assert _fit(1280, 720, 640) == (640, 360)
    assert _fit(512, 512, 512) == (512, 512)
    assert _fit(0, 0, 512) == (512, 512)


def test_cli_flags_square_and_jpeg(monkeypatch):
    from newera import cli
    got = {}
    monkeypatch.setattr("newera.talk.talk", lambda **kw: got.update(kw) or 0)
    cli.main(["avatar", "--image", "me.jpg", "--square", "--jpeg", "--no-window"])
    assert got["video_format"] == "jpeg" and got["output"] == "square" and got["image"] == "me.jpg"
    got.clear()
    cli.main(["avatar", "--no-window"])
    assert got["video_format"] is None and got["output"] is None           # the SDK's defaults: h264 + original


def test_the_examples_take_the_flags():
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name, positional in (("avatar_render_text.py", ["hello"]), ("avatar_render_audio.py", ["a.wav"]),
                             ("avatar_quickstart.py", ["--text", "hi"])):
        done = subprocess.run([sys.executable, os.path.join(root, "examples", name), *positional, "--image", "x.jpg",
                               "--square", "--jpeg", "--help"], capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": os.path.join(root, "src") + os.pathsep + os.environ.get("PYTHONPATH", "")})
        assert done.returncode == 0, done.stderr
        for flag in ("--image", "--square", "--jpeg"):
            assert flag in done.stdout, (name, flag)


# ------------------------------------------------------------------------------------ real PyAV, any frame size

def test_av_a_landscape_h264_call_decodes_and_saves_an_mp4_of_that_size(tmp_path):
    av = pytest.importorskip("av")
    if not hasattr(av.container.OutputContainer, "add_stream_from_template"):
        pytest.skip("PyAV 14 or newer is needed to mux without re-encoding")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from test_avatar_h264 import chunks_of, encode_units
    from newera.h264 import H264Decoder
    enc = encode_units(av, 12, width=320, height=180, gop=6)
    dec, frames = H264Decoder(), []
    for c in chunks_of(enc):
        frames += dec.decode(c)
    assert len(frames) == 12 and all((f.width, f.height) == (320, 180) for f in frames)
    tl = Timeline()
    for c in chunks_of(enc):
        tl.add(c)
    path = tl.save_mp4(str(tmp_path / "wide.mp4"))
    with av.open(path) as mp4:
        v = mp4.streams.video[0]
        assert (v.width, v.height) == (320, 180)
    tall = encode_units(av, 8, width=256, height=320, gop=6)
    tl = Timeline()
    for c in chunks_of(tall):
        tl.add(c)
    with av.open(tl.save_mp4(str(tmp_path / "tall.mp4"))) as mp4:
        assert (mp4.streams.video[0].width, mp4.streams.video[0].height) == (256, 320)
