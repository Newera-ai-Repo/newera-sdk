"""`client.audio.transcribe` and `client.audio.transcribe_stream` against fakes (a local HTTP server and a local
WebSocket); no network, no key."""
import asyncio
import json
import struct
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.asyncio.server import serve

from newera import (AuthenticationError, Done, InsufficientQuota, InvalidRequest, Newera, NotEnabled, RateLimited,
                    Segment, StreamError, Transcription, stt)
from newera.stt import SpeechStarted, SpeechStopped, parse_stream_event


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


# ------------------------------------------------------------------------------------------------- files

def parse_multipart(body: bytes, ctype: str) -> dict:
    boundary = ctype.split("boundary=")[1].encode()
    out = {}
    for part in body.split(b"--" + boundary)[1:-1]:
        head, _, value = part.partition(b"\r\n\r\n")
        value = value[:-2]
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        if name == "file":
            out["file"] = value
            out["filename"] = head.split(b'filename="')[1].split(b'"')[0].decode()
        else:
            out[name] = value.decode()
    return out


class FakeHttp:
    def __init__(self):
        self.requests: list[dict] = []
        self.reply = (200, {"text": "مرحبا"}, {"x-newera-audio-seconds": "3"})

    def __enter__(self):
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                n = int(self.headers["Content-Length"])
                body = self.rfile.read(n)
                form = parse_multipart(body, self.headers["Content-Type"])
                outer.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "form": form})
                status, payload, headers = outer.reply
                raw = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "text/plain" if isinstance(payload, str) else "application/json")
                self.send_header("Content-Length", str(len(raw)))
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = "http://127.0.0.1:%d" % self.server.server_address[1]
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()


def wav_bytes(path, seconds=1.0, rate=16000, value=0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", value) * int(rate * seconds))
    return path


def test_transcribe_bytes_defaults(tmp_path):
    with FakeHttp() as srv:
        client = Newera(api_key="nwc_test", base_url=srv.base_url)
        data = wav_bytes(tmp_path / "a.wav").read_bytes()
        result = run(client.audio.transcribe(data))
    assert isinstance(result, Transcription)
    assert (result.text, result.audio_seconds) == ("مرحبا", 3.0)
    req = srv.requests[0]
    assert req["path"] == "/v1/audio/transcriptions" and req["auth"] == "Bearer nwc_test"
    form = req["form"]
    assert form["model"] == "newera-stt" and form["vad"] == "true" and form["response_format"] == "json"
    assert "language" not in form and "prompt" not in form and "temperature" not in form
    assert form["file"] == data and form["filename"] == "audio.wav"


def test_transcribe_path_options_and_vad_off(tmp_path):
    path = wav_bytes(tmp_path / "clip.wav")
    with FakeHttp() as srv:
        client = Newera(api_key="nwc_test", base_url=srv.base_url)
        run(client.audio.transcribe(path, language="ar", prompt="نيو إيرا", vad=False, temperature=0.2))
        run(client.audio.transcribe(str(path), vad=True))
        with open(path, "rb") as f:
            run(client.audio.transcribe(f, language="en"))
    a, b, c = (r["form"] for r in srv.requests)
    assert (a["language"], a["prompt"], a["vad"], a["temperature"], a["filename"]) == ("ar", "نيو إيرا", "false", "0.2", "clip.wav")
    assert b["vad"] == "true"
    assert c["language"] == "en" and c["file"] == path.read_bytes()


def test_transcribe_empty_text_for_silence_and_no_header():
    with FakeHttp() as srv:
        srv.reply = (200, {"text": ""}, {})
        result = run(Newera(api_key="k", base_url=srv.base_url).audio.transcribe(b"RIFFxxxx", vad=True))
    assert result.text == "" and result.audio_seconds is None


def test_transcribe_text_format():
    with FakeHttp() as srv:
        srv.reply = (200, "hello world\n", {"x-newera-audio-seconds": "2.5"})
        result = run(Newera(api_key="k", base_url=srv.base_url).audio.transcribe(b"ID3abc", response_format="text"))
    assert (result.text, result.audio_seconds) == ("hello world", 2.5)
    assert srv.requests[0]["form"]["response_format"] == "text" and srv.requests[0]["form"]["filename"] == "audio.mp3"


@pytest.mark.parametrize("status,code,cls", [
    (401, "invalid_api_key", AuthenticationError), (402, "insufficient_quota", InsufficientQuota),
    (403, "model_not_entitled", NotEnabled), (429, "rate_limit_exceeded", RateLimited), (400, "invalid_value", InvalidRequest)])
def test_transcribe_errors_are_mapped(status, code, cls):
    with FakeHttp() as srv:
        srv.reply = (status, {"error": {"code": code, "message": "nope", "retry_after_s": 7}}, {})
        with pytest.raises(cls) as exc:
            run(Newera(api_key="k", base_url=srv.base_url).audio.transcribe(b"RIFFxxxx"))
    assert exc.value.http_status == status and exc.value.code == code
    if status == 429:
        assert exc.value.retry_after_s == 7


def test_transcribe_error_without_json_body_uses_status():
    with FakeHttp() as srv:
        srv.reply = (401, "unauthorized", {})
        with pytest.raises(AuthenticationError):
            run(Newera(api_key="k", base_url=srv.base_url).audio.transcribe(b"RIFFxxxx"))


def test_transcribe_validates_before_sending():
    client = Newera(api_key="k", base_url="http://127.0.0.1:1")
    for kw in ({"response_format": "srt"}, {"vad": "yes"}, {"temperature": 2}):
        with pytest.raises(ValueError):
            run(client.audio.transcribe(b"RIFFxxxx", **kw))
    with pytest.raises(ValueError):
        run(client.audio.transcribe(b""))
    with pytest.raises(ValueError):
        run(client.audio.transcribe(123))
    with pytest.raises(AuthenticationError):
        run(Newera(api_key="", base_url="http://127.0.0.1:1").audio.transcribe(b"RIFFxxxx"))


def test_transcribe_unreachable_is_connection_error():
    from newera import AvatarError
    with pytest.raises(AvatarError) as exc:
        run(Newera(api_key="k", base_url="http://127.0.0.1:1").audio.transcribe(b"RIFFxxxx"))
    assert exc.value.code == "connection_error"


# ------------------------------------------------------------------------------------------------ events

def test_events_parse():
    assert isinstance(parse_stream_event({"type": "speech_started", "audio_start_ms": 120}), SpeechStarted)
    assert parse_stream_event({"type": "speech_stopped", "audio_end_ms": 900}).audio_end_ms == 900.0
    seg = parse_stream_event({"type": "transcript.segment", "id": "s1", "text": "hi", "start_ms": 100, "end_ms": 900, "final": True})
    assert isinstance(seg, Segment) and (seg.id, seg.text, seg.start_ms, seg.end_ms, seg.final) == ("s1", "hi", 100.0, 900.0, True)
    done = parse_stream_event({"type": "transcript.done", "text": "hi", "audio_ms": 1500, "billed_seconds": 2,
                               "segments": [{"id": "s1", "text": "hi", "start_ms": 100, "end_ms": 900, "final": True}]})
    assert isinstance(done, Done) and done.billed_seconds == 2.0 and done.audio_ms == 1500.0
    assert isinstance(done.segments[0], Segment) and done.text == "hi" and done.segment_count == 1
    counted = parse_stream_event({"type": "transcript.done", "text": "", "segments": 3, "audio_ms": 1, "billed_seconds": 1})
    assert counted.segment_count == 3 and counted.segments == []
    err = parse_stream_event({"type": "error", "error": {"code": "stt_failed", "message": "m"}})
    assert isinstance(err, StreamError) and err.code == "stt_failed"
    assert stt.OtherStreamEvent is type(parse_stream_event({"type": "whatever"}))


# ----------------------------------------------------------------------------------------------- stream

class FakeStream:
    """/v1/audio/transcriptions/stream: after 3200 bytes of audio it says speech_started, then at commit it sends
    speech_stopped, one segment per 6400 bytes received, transcript.done, and closes 1000. `refuse` is a (code, close)
    pair sent at connect; `mid_error` sends a non-fatal error with the segments."""

    def __init__(self, refuse=None, mid_error=False, drop=False):
        self.refuse, self.mid_error, self.drop = refuse, mid_error, drop
        self.paths, self.headers, self.audio, self.commands = [], [], bytearray(), []

    async def handler(self, ws):
        self.paths.append(ws.request.path)
        self.headers.append(ws.request.headers.get("Authorization"))
        send = lambda ev: ws.send(json.dumps(ev))                                     # noqa: E731
        if self.refuse:
            code, close = self.refuse
            await send({"type": "error", "error": {"type": "invalid_request_error", "code": code, "message": code}})
            await ws.close(code=close, reason=code)
            return
        q = parse_qs(urlparse(ws.request.path).query)
        await send({"type": "session.created", "session": {"id": "sess_1", "model": q["model"][0], "language": q.get("language", [None])[0],
                                                           "sample_rate": int(q["sample_rate"][0]), "encoding": "pcm_s16le"}})
        started = False
        async for msg in ws:
            if isinstance(msg, bytes):
                self.audio += msg
                if not started and len(self.audio) >= 3200:
                    started = True
                    await send({"type": "speech_started", "audio_start_ms": 100})
                if self.drop:
                    await ws.close(code=4429, reason="concurrent_call_limit")
                    return
                continue
            ev = json.loads(msg)
            self.commands.append(ev)
            if ev["type"] == "commit":
                n = len(self.audio) // 6400
                segs = [{"id": "seg_%d" % i, "text": "كلمة%d" % i, "start_ms": i * 200, "end_ms": i * 200 + 150, "final": True}
                        for i in range(n)]
                await send({"type": "speech_stopped", "audio_end_ms": 150})
                for s in segs:
                    await send({"type": "transcript.segment", **s})
                if self.mid_error:
                    await send({"type": "error", "error": {"type": "server_error", "code": "segment_failed", "message": "x"}})
                await send({"type": "transcript.done", "text": " ".join(s["text"] for s in segs), "segments": len(segs),
                            "audio_ms": len(self.audio) // 32, "billed_seconds": 1})
                await ws.close(1000)

    async def __aenter__(self):
        self.server = await serve(self.handler, "127.0.0.1", 0)
        self.base_url = "http://127.0.0.1:%d" % self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *a):
        self.server.close()
        await self.server.wait_closed()


def test_stream_flow_events_in_order_and_commit():
    async def go():
        async with FakeStream() as srv:
            client = Newera(api_key="nwc_test", base_url=srv.base_url)
            seen = []
            async with client.audio.transcribe_stream(language="ar", sample_rate=16000, on_segment=seen.append) as stream:
                assert stream.session.id == "sess_1" and stream.session.sample_rate == 16000
                await stream.send(b"\x01\x00" * 1600)
                await stream.send(b"\x01\x00" * 3200)
                await stream.commit()
                events = [ev async for ev in stream]
            return srv, events, seen, stream
    srv, events, seen, stream = run(go())
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["SpeechStarted", "SpeechStopped", "Segment", "Done"]
    assert events[0].audio_start_ms == 100 and events[1].audio_end_ms == 150
    assert [s.text for s in seen] == ["كلمة0"] and seen[0] is events[2]
    assert events[3].text == "كلمة0" and events[3].billed_seconds == 1 and stream.done is events[3]
    assert srv.commands == [{"type": "commit"}] and len(srv.audio) == 9600
    q = parse_qs(urlparse(srv.paths[0]).query)
    assert urlparse(srv.paths[0]).path == "/v1/audio/transcriptions/stream"
    assert q == {"model": ["newera-stt"], "language": ["ar"], "sample_rate": ["16000"]}
    assert srv.headers[0] == "Bearer nwc_test"


def test_stream_on_segment_async_and_result_without_iterating():
    async def go():
        async with FakeStream() as srv:
            got = []

            async def cb(seg):
                got.append(seg.text)
            async with Newera(api_key="k", base_url=srv.base_url).audio.transcribe_stream(on_segment=cb) as stream:
                await stream.send(b"\x00\x00" * 6400)
                await stream.commit()
                done = await stream.result()
            return got, done, srv
    got, done, srv = run(go())
    assert got == ["كلمة0", "كلمة1"] and done.text == "كلمة0 كلمة1" and len(done.segments) == 2 and done.segment_count == 2
    assert "language" not in parse_qs(urlparse(srv.paths[0]).query)


def test_stream_non_fatal_error_is_an_event():
    async def go():
        async with FakeStream(mid_error=True) as srv:
            async with Newera(api_key="k", base_url=srv.base_url).audio.transcribe_stream() as stream:
                await stream.send(b"\x00\x00" * 3200)
                await stream.commit()
                return [ev async for ev in stream]
    events = run(go())
    errs = [e for e in events if isinstance(e, StreamError)]
    assert len(errs) == 1 and errs[0].code == "segment_failed" and isinstance(events[-1], Done)


@pytest.mark.parametrize("code,close,cls", [
    ("invalid_api_key", 4401, AuthenticationError), ("insufficient_quota", 4402, InsufficientQuota),
    ("model_not_entitled", 4403, NotEnabled), ("concurrent_call_limit", 4429, RateLimited),
    ("invalid_model", 4400, InvalidRequest)])
def test_stream_refusals_raise_typed_errors(code, close, cls):
    async def go():
        async with FakeStream(refuse=(code, close)) as srv:
            async with Newera(api_key="k", base_url=srv.base_url).audio.transcribe_stream():
                pass
    with pytest.raises(cls) as exc:
        run(go())
    assert exc.value.code == code and exc.value.close_code == close


def test_stream_closed_with_error_code_raises_when_iterating():
    async def go():
        async with FakeStream(drop=True) as srv:
            async with Newera(api_key="k", base_url=srv.base_url).audio.transcribe_stream() as stream:
                await stream.send(b"\x00\x00" * 3200)
                return [ev async for ev in stream]
    with pytest.raises(RateLimited):
        run(go())


def test_stream_validation_and_state():
    client = Newera(api_key="k", base_url="http://127.0.0.1:1")
    with pytest.raises(ValueError):
        client.audio.transcribe_stream(sample_rate=44100)
    with pytest.raises(AuthenticationError):
        run(Newera(api_key="", base_url="http://127.0.0.1:1").audio.transcribe_stream().__aenter__())

    async def go():
        async with FakeStream() as srv:
            async with Newera(api_key="k", base_url=srv.base_url).audio.transcribe_stream() as stream:
                await stream.commit()
                await stream.commit()                       # a second commit is a no-op
                with pytest.raises(RuntimeError):
                    await stream.send(b"\x00\x00")
                await stream.result()
            return srv
    assert len(run(go()).commands) == 1


def test_send_wav_paces_and_checks_rate(tmp_path):
    wav = wav_bytes(tmp_path / "s.wav", seconds=0.5, rate=16000)
    assert stt.wav_info(wav) == {"sample_rate": 16000, "seconds": 0.5}

    async def go():
        async with FakeStream() as srv:
            client = Newera(api_key="k", base_url=srv.base_url)
            async with client.audio.transcribe_stream(sample_rate=16000) as stream:
                t0 = asyncio.get_running_loop().time()
                await stream.send_wav(wav, realtime=True, chunk_ms=100)
                elapsed = asyncio.get_running_loop().time() - t0
                done = await stream.result()
            async with client.audio.transcribe_stream(sample_rate=24000) as stream:
                with pytest.raises(ValueError):
                    await stream.send_wav(wav)
            return srv, elapsed, done
    srv, elapsed, done = run(go())
    assert len(srv.audio) == 16000 and srv.commands == [{"type": "commit"}]
    assert 0.4 <= elapsed < 2.0
    assert done.audio_ms == 500


def test_send_wav_fast_and_rejects_stereo(tmp_path):
    stereo = tmp_path / "st.wav"
    with wave.open(str(stereo), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00" * 3200)
    with pytest.raises(ValueError):
        stt.wav_info(stereo)
    wav = wav_bytes(tmp_path / "s.wav", seconds=1.0)

    async def go():
        async with FakeStream() as srv:
            async with Newera(api_key="k", base_url=srv.base_url).audio.transcribe_stream() as stream:
                t0 = asyncio.get_running_loop().time()
                await stream.send_wav(wav, realtime=False)
                await stream.result()
                return asyncio.get_running_loop().time() - t0
    assert run(go()) < 0.9
