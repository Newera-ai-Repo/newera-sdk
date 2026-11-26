"""Text lines that stream (render-text), per-line quality / normalize / pronunciations, the piece-timing events, and
`client.audio.speech(..., stream=True)`, against fakes (a local WebSocket and a local HTTP server); no network, no key."""
import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
from websockets.asyncio.server import serve

from newera import (AvatarError, InvalidRequest, LineDone, LineError, LinePiece, LineStarted, LineWords, Newera, Word,
                    WrongMode)
from newera.avatar import AUDIO_RATE, parse_event, text_options

PIECE_MS, GAP_MS = 1000, 190


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 30))


def _select(connection, subprotocols):
    return "realtime" if "realtime" in subprotocols else None


class FakeText:
    """render-text of Core's /v1/avatar: a line's text is cut at full stops; each sentence is a piece, announced with
    avatar.line.piece (1000 ms, 190 ms between pieces) as soon as it is complete. `done` follows the whole text
    (avatar.line.text) or avatar.line.text.end. A text of "REFUSE" on a first event is `invalid_quality`; a delta
    whose text is "TOOLONG" is refused as `text_too_long` (the line goes on)."""

    def __init__(self, words=True):
        self.words = words
        self.received: list[dict] = []
        self.paths: list[str] = []
        self.state: dict[str, dict] = {}

    async def handler(self, ws):
        self.paths.append(ws.request.path)
        send = lambda ev: ws.send(json.dumps(ev))                                   # noqa: E731
        self.received.append(json.loads(await ws.recv()))
        await send({"type": "avatar.ready", "fps": 25, "width": 512, "height": 512, "video_format": "jpeg",
                    "audio_sample_rate": AUDIO_RATE, "frames_per_event": 7, "mode": "render-text"})
        await send({"type": "session.created", "session": {"id": "sess_t"}})

        def err(code, ev, param=None):
            return send({"type": "error", "event_id": "e", "error": {"type": "invalid_request_error", "code": code,
                         "message": code, "param": param, "event_id": ev.get("event_id")}})

        async def pieces(line, final):
            st = self.state[line]
            while True:
                cut = st["buf"].find(".")
                if cut < 0:
                    break
                text, st["buf"] = st["buf"][:cut + 1].strip(), st["buf"][cut + 1:]
                await piece(line, text)
            if final and st["buf"].strip():
                await piece(line, st["buf"].strip())
                st["buf"] = ""

        async def piece(line, text):
            st = self.state[line]
            start = st["pos"] + (GAP_MS if st["n"] else 0)
            await send({"type": "avatar.line.piece", "event_id": "evt_p", "line_id": line, "index": st["n"], "text": text,
                        "start_ms": start, "end_ms": start + PIECE_MS})
            st["pos"], st["n"] = start + PIECE_MS, st["n"] + 1
            if self.words:
                names = text.rstrip(".").split()
                step = PIECE_MS // len(names)
                await send({"type": "avatar.line.words", "event_id": "evt_w", "line_id": line, "piece_index": st["n"] - 1,
                            "words": [{"word": n, "start_ms": start + k * step, "end_ms": start + (k + 1) * step}
                                      for k, n in enumerate(names)], "approximate": text.startswith("Approx")})

        async def finish(line):
            await send({"type": "avatar.line.started", "line_id": line, "at_ms": 480})
            await send({"type": "avatar.line.done", "line_id": line, "status": "completed"})
            self.state[line]["done"] = True

        async for raw in ws:
            ev = json.loads(raw)
            self.received.append(ev)
            kind, line = ev["type"], ev.get("line_id")
            if kind in ("avatar.line.text", "avatar.line.text.delta"):
                if line not in self.state:
                    if ev.get("quality") == "REFUSE":
                        await err("invalid_quality", ev, "quality")
                        continue
                    self.state[line] = {"buf": "", "pos": 0, "n": 0, "done": False}
                if ev["text"] == "TOOLONG":
                    await err("text_too_long", ev, "text")
                    continue
                if self.state[line]["done"]:
                    await err("line_closed", ev, "line_id")
                    continue
                self.state[line]["buf"] += ev["text"]
                await pieces(line, kind == "avatar.line.text")
                if kind == "avatar.line.text":
                    await finish(line)
            elif kind == "avatar.line.text.end":
                if line not in self.state:
                    await err("unknown_line", ev, "line_id")
                    continue
                await pieces(line, True)
                await finish(line)
            elif kind == "avatar.interrupt":
                await send({"type": "avatar.interrupted"})
                for name, st in self.state.items():
                    if not st["done"]:
                        st["done"] = True
                        await send({"type": "avatar.line.done", "line_id": name, "status": "interrupted"})
            elif kind == "session.end":
                await ws.close()

    async def __aenter__(self):
        self.server = await serve(self.handler, "127.0.0.1", 0, select_subprotocol=_select)
        self.base_url = "http://127.0.0.1:%d" % self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *a):
        self.server.close()
        await self.server.wait_closed()


def connect(srv, **kw):
    return Newera(api_key="nwc_test", base_url=srv.base_url).avatar.connect("lama", mode="render-text", transport="websocket", **kw)


def sent(srv, kind):
    return [e for e in srv.received if e["type"] == kind]


# ---------------------------------------------------------------------------------------------- the event

def test_line_piece_event_parses():
    ev = parse_event({"type": "avatar.line.piece", "event_id": "evt_1", "line_id": "a", "index": 2, "text": "Hello.",
                      "start_ms": 1190, "end_ms": 2190})
    assert isinstance(ev, LinePiece)
    assert (ev.line_id, ev.index, ev.text, ev.start_ms, ev.end_ms) == ("a", 2, "Hello.", 1190.0, 2190.0)


def test_text_options_are_validated():
    assert text_options() == {}
    assert text_options(voice="male", quality="instant", normalize=False, pronunciations={"Newera": "نيو إيرا"}) == {
        "voice": "male", "quality": "instant", "normalize": False, "pronunciations": {"Newera": "نيو إيرا"}}
    for bad in ({"quality": "ultra"}, {"normalize": "yes"}, {"pronunciations": {"a": ""}}, {"pronunciations": ["a"]},
                {"voice": "robot"}):
        with pytest.raises(ValueError):
            text_options(**bad)


# ------------------------------------------------------------------------------- whole text and options

def test_speak_sends_the_line_options_and_collects_the_pieces():
    async def go():
        async with FakeText() as srv:
            async with connect(srv) as call:
                got = []
                call.on_line_piece(got.append)
                line = await call.speak("Hello there. How are you.", "a", quality="instant", normalize=False,
                                        pronunciations={"Newera": "Nu-era"})
                assert await line.wait(timeout=5, strict=True) == "completed"
                ev = sent(srv, "avatar.line.text")[0]
                assert (ev["text"], ev["quality"], ev["normalize"], ev["pronunciations"]) == (
                    "Hello there. How are you.", "instant", False, {"Newera": "Nu-era"})
                assert "voice" not in ev
                assert [(p.index, p.text, p.start_ms, p.end_ms) for p in line.pieces] == [
                    (0, "Hello there.", 0, 1000), (1, "How are you.", 1000 + GAP_MS, 2000 + GAP_MS)]
                assert got == line.pieces                                          # the callback saw the same events
    run(go())


def test_speak_without_options_sends_exactly_what_it_sent_before():
    async def go():
        async with FakeText() as srv:
            async with connect(srv) as call:
                line = await call.speak("Hello.", "a")
                await line.wait(timeout=5)
                ev = sent(srv, "avatar.line.text")[0]
                assert {k: v for k, v in ev.items() if k != "event_id"} == {"type": "avatar.line.text", "line_id": "a", "text": "Hello."}
                line = await call.speak("Hello.", "b", voice="male")
                await line.wait(timeout=5)
                assert sent(srv, "avatar.line.text")[1]["voice"] == "male"
    run(go())


def test_session_defaults_are_sent_in_the_first_session_update():
    async def go():
        async with FakeText() as srv:
            async with connect(srv, quality="balanced", normalize=False, pronunciations={"GPU": "جي بي يو"}):
                s = srv.received[0]["session"]
                assert (s["voice_quality"], s["normalize"], s["pronunciations"]) == ("balanced", False, {"GPU": "جي بي يو"})
            async with connect(srv):
                assert not {"voice_quality", "normalize", "pronunciations"} & set(srv.received[-1]["session"])
    run(go())
    for bad in ({"quality": "ultra"}, {"normalize": 1}):
        with pytest.raises(ValueError):
            Newera(api_key="k").avatar.connect("lama", mode="render-text", **bad)
    with pytest.raises(ValueError):                       # the defaults of render-text lines only
        Newera(api_key="k").avatar.connect("lama", mode="render-audio", quality="instant")
    with pytest.raises(ValueError):
        Newera(api_key="k").avatar.connect("lama", quality="instant")


def test_session_without_defaults_is_unchanged():
    async def go():
        async with FakeText() as srv:
            async with connect(srv):
                assert srv.received[0] == {"type": "session.update", "session": {"avatar": "lama", "avatar_idle_video": False}}
    run(go())


# ---------------------------------------------------------------------------------------- streamed text

def test_stream_text_from_an_async_iterator_speaks_while_text_arrives():
    async def go():
        async with FakeText() as srv:
            async with connect(srv) as call:
                first_piece = asyncio.Event()
                call.on_line_piece(lambda ev: first_piece.set())
                gate = asyncio.Event()
                seen_before_rest = []

                async def llm():
                    yield "Welcome to the show"
                    yield " today. We have"
                    await asyncio.wait_for(first_piece.wait(), 5)     # the first piece is voiced before the rest is sent
                    seen_before_rest.append(len(call.lines["s1"].pieces))
                    gate.set()
                    yield " three guests."

                line = await call.stream_text(llm(), "s1", quality="instant")
                assert await line.wait(timeout=5, strict=True) == "completed"
                assert gate.is_set() and seen_before_rest == [1]
                deltas = sent(srv, "avatar.line.text.delta")
                assert [d["text"] for d in deltas] == ["Welcome to the show", " today. We have", " three guests."]
                assert deltas[0]["quality"] == "instant" and "quality" not in deltas[1] and "quality" not in deltas[2]
                assert len(sent(srv, "avatar.line.text.end")) == 1
                assert [p.text for p in line.pieces] == ["Welcome to the show today.", "We have three guests."]
    run(go())


def test_start_text_line_append_text_end_and_async_with():
    async def go():
        async with FakeText() as srv:
            async with connect(srv) as call:
                async with call.start_text_line("t1", voice="male", normalize=False) as line:
                    assert not srv.received[1:]                         # nothing is sent until the first text
                    assert await line.append_text("One two three.") is True
                    assert await line.append_text("") is True           # nothing to send
                    await line.append_text(" Four five.")
                assert await line.wait(timeout=5) == "completed"
                deltas = sent(srv, "avatar.line.text.delta")
                assert [d["text"] for d in deltas] == ["One two three.", " Four five."]
                assert (deltas[0]["voice"], deltas[0]["normalize"]) == ("male", False)
                assert sent(srv, "avatar.line.text.end")[0]["line_id"] == "t1"
                with pytest.raises(RuntimeError):
                    await line.append_text("more")
    run(go())


def test_ending_a_streamed_line_that_had_no_text_still_works():
    async def go():
        async with FakeText() as srv:
            async with connect(srv) as call:
                line = call.start_text_line("e1")
                await line.end()
                assert await line.wait(timeout=5) == "completed"
                assert [d["text"] for d in sent(srv, "avatar.line.text.delta")] == [""]
    run(go())


def test_events_iterator_has_the_pieces_in_order():
    async def go():
        async with FakeText() as srv:
            async with connect(srv) as call:
                await call.speak("A b. C d.", "x")
                got = []
                async for ev in call.events():
                    got.append(ev)
                    if isinstance(ev, LineDone):
                        break
                kinds = [type(e).__name__ for e in got if isinstance(e, (LinePiece, LineStarted, LineDone))]
                assert kinds == ["LinePiece", "LinePiece", "LineStarted", "LineDone"]
    run(go())


def test_refused_first_event_fails_the_line_but_a_later_refusal_does_not():
    async def go():
        async with FakeText() as srv:
            async with connect(srv) as call:
                bad = call.start_text_line("bad")
                bad.options = {"quality": "REFUSE"}                     # the fake refuses this on the first event
                await bad.append_text("x")
                assert await bad.wait(timeout=5) == "failed"
                assert isinstance(bad.error, LineError) and bad.error.code == "invalid_quality"

                line = call.start_text_line("ok")
                await line.append_text("Fine start.")
                await line.append_text("TOOLONG")                       # refused, the line carries on
                await asyncio.sleep(0.2)
                assert not line.done and isinstance(line.error, LineError) and line.error.code == "text_too_long"
                await line.append_text(" More text.")
                await line.end()
                assert await line.wait(timeout=5) == "completed"
                assert [p.text for p in line.pieces] == ["Fine start.", "More text."]
    run(go())


def test_interrupt_stops_feeding_a_streamed_line():
    async def go():
        async with FakeText() as srv:
            async with connect(srv) as call:
                line = call.start_text_line("i1")
                await line.append_text("Some text here")
                await call.interrupt()
                assert await line.append_text("more") is False
                assert await line.wait(timeout=5) == "interrupted"
                assert await line.stream_text(["a", "b"]) is False
    run(go())


def test_streamed_text_needs_render_text():
    async def go():
        async with FakeText() as srv:
            async with connect(srv) as call:
                audio_line = call._new_line("a", "audio")
                with pytest.raises(WrongMode):
                    await audio_line.append_text("x")
                with pytest.raises(ValueError):
                    call.start_text_line("q", quality="ultra")
    run(go())


# ----------------------------------------------------------------------------- client.audio.speech

class FakeSpeechServer:
    """POST /v1/audio/speech. Non-stream: a WAV-ish body and headers. stream: chunked, one chunk per sentence, the
    second only after the test says the first was received (so a client that waits for the whole body hangs)."""

    def __init__(self):
        self.bodies: list[dict] = []
        self.release = threading.Event()
        self.closed_early = threading.Event()
        self.mode = "ok"                      # ok | error | cut
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append({**body, "_auth": self.headers.get("Authorization")})
                if outer.mode == "error" or body.get("pronunciations") == {"bad": "x"}:
                    data = json.dumps({"error": {"code": "invalid_value", "message": "nope", "param": "pronunciations"}}).encode()
                    self.send_response(400)
                    self.send_header("content-type", "application/json")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                if not body.get("stream"):
                    data = b"RIFF" + b"\0" * 40
                    self.send_response(200)
                    self.send_header("content-type", "audio/wav")
                    self.send_header("content-length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(200)
                self.send_header("content-type", "audio/pcm")
                self.send_header("x-sample-rate", "24000")
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()

                def chunk(b):
                    self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")
                    self.wfile.flush()
                try:
                    chunk(b"\x01\x00" * 100)
                    if outer.mode == "cut":
                        self.wfile.flush()
                        self.connection.close()
                        return
                    if not outer.release.wait(5):
                        return
                    chunk(b"\x02\x00" * 100)
                    chunk(b"\x03\x00" * 100)
                    if body.get("input", "").startswith("LONG"):          # keeps writing: a closed connection shows up
                        for _ in range(100):
                            chunk(b"\x04\x00" * 100)
                            time.sleep(0.02)
                    self.wfile.write(b"0\r\n\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    outer.closed_early.set()

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.base_url = "http://127.0.0.1:%d" % self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def speech_server():
    srv = FakeSpeechServer()
    yield srv
    srv.release.set()
    srv.close()


def test_speech_without_stream_returns_the_audio_and_sends_only_what_was_given(speech_server):
    client = Newera(api_key="nwc_k", base_url=speech_server.base_url)
    audio = run(client.audio.speech("Hello there."))
    assert audio.startswith(b"RIFF")
    assert speech_server.bodies[0] == {"model": "newera-tts", "input": "Hello there.", "response_format": "wav", "_auth": "Bearer nwc_k"}
    run(client.audio.speech("Hi", voice="male", quality="instant", normalize=False, pronunciations={"Newera": "Nu-era"},
                            response_format="pcm", speed=1.2))
    b = speech_server.bodies[1]
    assert (b["voice"], b["quality"], b["normalize"], b["pronunciations"], b["response_format"], b["speed"]) == (
        "male", "instant", False, {"Newera": "Nu-era"}, "pcm", 1.2) and "stream" not in b


def test_speech_stream_yields_chunks_as_they_arrive(speech_server):
    client = Newera(api_key="nwc_k", base_url=speech_server.base_url)

    async def go():
        chunks = []
        t0 = time.monotonic()
        async for c in client.audio.speech("One. Two. Three.", response_format="pcm", stream=True):
            chunks.append(c)
            if len(chunks) == 1:
                assert time.monotonic() - t0 < 3 and not speech_server.release.is_set()   # the first came before the rest
                speech_server.release.set()
        return chunks
    chunks = run(go())
    data = b"".join(chunks)
    assert data == b"\x01\x00" * 100 + b"\x02\x00" * 100 + b"\x03\x00" * 100
    assert speech_server.bodies[0]["stream"] is True and speech_server.bodies[0]["response_format"] == "pcm"


def test_speech_stream_stopped_early_closes_the_connection(speech_server):
    client = Newera(api_key="nwc_k", base_url=speech_server.base_url)

    async def go():
        agen = client.audio.speech("LONG One. Two.", response_format="pcm", stream=True)
        first = await agen.__anext__()
        await agen.aclose()
        speech_server.release.set()
        return first
    assert run(go()) == b"\x01\x00" * 100
    assert speech_server.closed_early.wait(5)


def test_speech_errors_are_typed_before_the_first_chunk(speech_server):
    client = Newera(api_key="nwc_k", base_url=speech_server.base_url)
    with pytest.raises(InvalidRequest) as exc:
        run(client.audio.speech("x", pronunciations={"bad": "x"}))
    assert exc.value.code == "invalid_value" and exc.value.http_status == 400

    async def go():
        async for _ in client.audio.speech("x", pronunciations={"bad": "x"}, stream=True):
            pass
    with pytest.raises(InvalidRequest):
        run(go())


def test_speech_stream_cut_after_the_first_chunk_raises_stream_interrupted(speech_server):
    speech_server.mode = "cut"
    client = Newera(api_key="nwc_k", base_url=speech_server.base_url)

    async def go():
        got = []
        async for c in client.audio.speech("One. Two.", response_format="pcm", stream=True):
            got.append(c)
    with pytest.raises(AvatarError) as exc:
        run(go())
    assert exc.value.code == "stream_interrupted"


def test_speech_checks_its_arguments_before_any_request():
    client = Newera(api_key="nwc_k", base_url="http://127.0.0.1:1")
    for kw in ({"quality": "ultra"}, {"response_format": "mp3"}, {"normalize": "no"}, {"pronunciations": {"": "x"}}):
        with pytest.raises(ValueError):
            client.audio.speech("x", **kw)
    with pytest.raises(ValueError):
        client.audio.speech("   ")
    no_key = Newera(api_key=None, base_url="http://127.0.0.1:1")
    no_key.api_key = None
    with pytest.raises(AvatarError) as exc:
        run(no_key.audio.speech("x"))
    assert exc.value.code == "invalid_api_key"


# ------------------------------------------------------------------------------------------------ word timings

def test_line_words_event_parses():
    ev = parse_event({"type": "avatar.line.words", "event_id": "evt_1", "line_id": "a", "piece_index": 1, "approximate": True,
                      "words": [{"word": "Hello", "start_ms": 1190, "end_ms": 1500}, {"word": "there", "start_ms": 1500, "end_ms": 2190}]})
    assert isinstance(ev, LineWords)
    assert (ev.line_id, ev.piece_index, ev.approximate) == ("a", 1, True)
    assert ev.words == [Word("Hello", 1190.0, 1500.0), Word("there", 1500.0, 2190.0)]
    bare = parse_event({"type": "avatar.line.words", "line_id": "a"})
    assert bare.words == [] and bare.approximate is False and bare.piece_index == 0


def test_words_arrive_by_callback_events_and_line_words():
    async def go():
        async with FakeText() as srv:
            async with connect(srv) as call:
                got, evs = [], []
                call.on_line_words(got.append)
                line = await call.speak("Hello there. Approx one two.", "a")
                assert await line.wait(timeout=5, strict=True) == "completed"
                async for ev in call.events():
                    evs.append(ev)
                    if isinstance(ev, LineDone):
                        break
                assert [(w.piece_index, [x.word for x in w.words], w.approximate) for w in line.words] == [
                    (0, ["Hello", "there"], False), (1, ["Approx", "one", "two"], True)]
                assert got == line.words
                assert line.words[0].words[1].start_ms == 500 and line.words[0].words[1].end_ms == 1000
    run(go())


def test_words_come_in_events_after_their_piece():
    async def go():
        async with FakeText() as srv:
            async with connect(srv) as call:
                line = await call.speak("One two.", "a")
                kinds = []
                async for ev in call.events():
                    kinds.append(type(ev).__name__)
                    if isinstance(ev, LineDone):
                        break
                assert kinds.index("LinePiece") < kinds.index("LineWords")
    run(go())


def test_word_timings_false_is_sent_in_the_session_and_validated():
    async def go():
        async with FakeText(words=False) as srv:
            async with connect(srv, word_timings=False) as call:
                line = await call.speak("One two.", "a")
                await line.wait(timeout=5, strict=True)
                assert line.words == []
            assert srv.received[0]["session"]["word_timings"] is False
        async with FakeText() as srv:
            async with connect(srv) as call:
                pass
            assert "word_timings" not in srv.received[0]["session"]
    run(go())
    with pytest.raises(ValueError):
        Newera(api_key="nwc_test").avatar.connect("lama", mode="render-text", word_timings="no")
    with pytest.raises(ValueError):
        Newera(api_key="nwc_test").avatar.connect("lama", mode="render-audio", word_timings=False)
