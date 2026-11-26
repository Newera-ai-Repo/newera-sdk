"""Speech to text: `client.audio.transcribe(...)` for files and `client.audio.transcribe_stream(...)` for live audio.

    client = Newera()

    result = await client.audio.transcribe("meeting.wav", language="ar")
    print(result.text, result.audio_seconds)

    async with client.audio.transcribe_stream(language="ar", sample_rate=16000) as stream:
        await stream.send(pcm_bytes)                  # PCM 16-bit little-endian mono, any chunk size
        await stream.commit()                         # no more audio: the rest is transcribed
        async for event in stream:                    # SpeechStarted, SpeechStopped, Segment ..., Done
            if isinstance(event, Segment):
                print(event.start_ms, event.text)
"""

from __future__ import annotations

import asyncio
import dataclasses
import http.client
import json
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
import wave
from typing import Any, AsyncIterator, Callable

import websockets

from .avatar import error_for

MODEL = "newera-stt"
RESPONSE_FORMATS = ("json", "text")
SAMPLE_RATES = (8000, 16000, 24000, 48000)


# ------------------------------------------------------------------------------------------------ files

@dataclasses.dataclass
class Transcription:
    """The result of `client.audio.transcribe`. `text` is "" for a clip without speech (vad on).
    `audio_seconds` is the billed length from the `x-newera-audio-seconds` header, None when the server did not send it."""
    text: str
    audio_seconds: float | None = None

    def __str__(self) -> str:
        return self.text


def _sniff(data: bytes) -> str:
    if data[:4] == b"RIFF":
        return "audio.wav"
    if data[:4] == b"OggS":
        return "audio.ogg"
    if data[:4] == b"fLaC":
        return "audio.flac"
    if data[:4] == b"\x1aE\xdf\xa3":
        return "audio.webm"
    if data[4:8] == b"ftyp":
        return "audio.m4a"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio.mp3"
    return "audio.wav"


def _read_input(source) -> tuple[bytes, str]:
    """The bytes and a file name for a path, bytes or a binary file object."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source)
        name = _sniff(data)
    elif isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        with open(path, "rb") as f:
            data = f.read()
        name = os.path.basename(path) or "audio.wav"
    elif hasattr(source, "read"):
        data = source.read()
        name = os.path.basename(getattr(source, "name", "") or "") or _sniff(data)
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError("the file must be opened in binary mode")
        data = bytes(data)
    else:
        raise ValueError("audio must be a path, bytes or a binary file object, not %s" % type(source).__name__)
    if not data:
        raise ValueError("the audio is empty")
    return data, name


def _multipart(fields: dict, filename: str, data: bytes) -> tuple[bytes, str]:
    boundary = "newera" + uuid.uuid4().hex
    parts = []
    for key, value in fields.items():
        parts.append(('--%s\r\nContent-Disposition: form-data; name="%s"\r\n\r\n%s\r\n' % (boundary, key, value)).encode())
    safe = filename.replace('"', "").replace("\r", "").replace("\n", "")
    parts.append(('--%s\r\nContent-Disposition: form-data; name="file"; filename="%s"\r\n'
                  "Content-Type: application/octet-stream\r\n\r\n" % (boundary, safe)).encode())
    parts.append(data)
    parts.append(("\r\n--%s--\r\n" % boundary).encode())
    return b"".join(parts), "multipart/form-data; boundary=" + boundary


def _fields(language, prompt, vad, response_format, temperature, model) -> dict:
    if response_format not in RESPONSE_FORMATS:
        raise ValueError("response_format must be one of %s, not %r" % (", ".join(RESPONSE_FORMATS), response_format))
    if not isinstance(vad, bool):
        raise ValueError("vad must be True or False, not %r" % (vad,))
    fields: dict = {"model": model, "response_format": response_format, "vad": "true" if vad else "false"}
    if language is not None:
        fields["language"] = language
    if prompt is not None:
        fields["prompt"] = prompt
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 1:
            raise ValueError("temperature must be a number from 0 to 1, not %r" % (temperature,))
        fields["temperature"] = temperature
    return fields


def transcribe_blocking(client, source, *, language=None, prompt=None, vad=True, response_format="json",
                        temperature=None, model=MODEL) -> Transcription:
    fields = _fields(language, prompt, vad, response_format, temperature, model)
    data, name = _read_input(source)
    if not client.api_key:
        raise error_for("invalid_api_key", "No API key. Pass Newera(api_key=...) or set NEWERA_API_KEY.", http_status=401)
    body, ctype = _multipart(fields, name, data)
    req = urllib.request.Request(client.base_url + "/v1/audio/transcriptions", data=body, method="POST",
                                 headers={**client._headers(), "Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=300, context=client._ssl_context()) as resp:
            raw = resp.read()
            seconds = resp.headers.get("x-newera-audio-seconds")
    except urllib.error.HTTPError as exc:
        try:
            err = json.loads(exc.read().decode()).get("error") or {}
        except Exception:  # noqa: BLE001 - not JSON
            err = {}
        retry = err.get("retry_after_s") or exc.headers.get("Retry-After")
        try:
            retry = int(retry) if retry is not None else None
        except (TypeError, ValueError):
            retry = None
        raise error_for(err.get("code"), err.get("message") or "HTTP %s" % exc.code, http_status=exc.code,
                        retry_after_s=retry) from exc
    except urllib.error.URLError as exc:
        raise error_for("connection_error", "Could not reach %s: %s" % (client.base_url, exc.reason)) from exc
    try:
        audio_seconds = float(seconds) if seconds is not None else None
    except ValueError:
        audio_seconds = None
    text = raw.decode("utf-8", "replace")
    if response_format == "json":
        try:
            text = json.loads(text).get("text") or ""
        except (ValueError, AttributeError) as exc:
            raise error_for("bad_response", "The server did not return JSON.") from exc
    else:
        text = text.strip()
    return Transcription(text=text, audio_seconds=audio_seconds)


# ---------------------------------------------------------------------------------------------- streaming

@dataclasses.dataclass
class StreamEvent:
    """A server event of the stream. `raw` is the JSON as sent."""
    raw: dict = dataclasses.field(default_factory=dict, repr=False)


@dataclasses.dataclass
class SessionCreated(StreamEvent):
    id: str = ""
    model: str = ""
    language: str | None = None
    sample_rate: int = 0


@dataclasses.dataclass
class SpeechStarted(StreamEvent):
    """Speech begins, `audio_start_ms` from the first sample you sent."""
    audio_start_ms: float = 0.0


@dataclasses.dataclass
class SpeechStopped(StreamEvent):
    """The speaker paused; the segment is being transcribed."""
    audio_end_ms: float = 0.0


@dataclasses.dataclass
class Segment(StreamEvent):
    """The text of one spoken segment, final. Times are milliseconds from the first sample you sent."""
    id: str = ""
    text: str = ""
    start_ms: float = 0.0
    end_ms: float = 0.0
    final: bool = True


@dataclasses.dataclass
class Done(StreamEvent):
    """After `commit()`: the whole text, the segments, the audio length and what is billed."""
    text: str = ""
    segments: list = dataclasses.field(default_factory=list)   # the Segment events of the stream, in order
    segment_count: int = 0
    audio_ms: float = 0.0
    billed_seconds: float | None = None


@dataclasses.dataclass
class StreamError(StreamEvent):
    """A problem that did not end the stream (a segment that could not be transcribed, with its times in `raw`).
    `exception()` is the matching SDK exception."""
    code: str = "error"
    message: str = ""
    param: str | None = None

    def exception(self, close_code: int | None = None):
        return error_for(self.code, self.message, close_code)


@dataclasses.dataclass
class OtherStreamEvent(StreamEvent):
    pass


def _num(v, default=0.0):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else default


def parse_stream_event(raw: dict) -> StreamEvent:
    kind = raw.get("type")
    if kind == "session.created":
        s = raw.get("session") or {}
        return SessionCreated(raw=raw, id=s.get("id", ""), model=s.get("model", ""), language=s.get("language"),
                              sample_rate=int(s.get("sample_rate") or 0))
    if kind == "speech_started":
        return SpeechStarted(raw=raw, audio_start_ms=_num(raw.get("audio_start_ms")))
    if kind == "speech_stopped":
        return SpeechStopped(raw=raw, audio_end_ms=_num(raw.get("audio_end_ms")))
    if kind == "transcript.segment":
        return Segment(raw=raw, id=str(raw.get("id", "")), text=raw.get("text") or "", start_ms=_num(raw.get("start_ms")),
                       end_ms=_num(raw.get("end_ms")), final=bool(raw.get("final", True)))
    if kind == "transcript.done":
        raw_segs = raw.get("segments")            # the server sends the count; a list of segments is accepted too
        segs = [parse_stream_event({"type": "transcript.segment", **x}) for x in raw_segs if isinstance(x, dict)] \
            if isinstance(raw_segs, list) else []
        count = len(segs) if isinstance(raw_segs, list) else int(_num(raw_segs))
        billed = raw.get("billed_seconds")
        return Done(raw=raw, text=raw.get("text") or "", segments=segs, segment_count=count, audio_ms=_num(raw.get("audio_ms")),
                    billed_seconds=_num(billed) if billed is not None else None)
    if kind == "error":
        e = raw.get("error") or {}
        return StreamError(raw=raw, code=e.get("code") or "error", message=e.get("message") or "", param=e.get("param"))
    return OtherStreamEvent(raw=raw)


_CLOSED = object()


class TranscribeStream:
    """`async with client.audio.transcribe_stream(language="ar") as stream:`. See the module docstring.

    A refusal at connect (bad key, no balance, model not entitled, too many calls) raises the matching exception from
    `async with`. Events are read in the background, so `on_segment` is called even while you are only sending; iterate
    the stream (or `await stream.result()`) to get them all."""

    def __init__(self, client, *, language=None, sample_rate=16000, model=MODEL, on_segment: Callable | None = None):
        if sample_rate not in SAMPLE_RATES:
            raise ValueError("sample_rate must be one of %s, not %r" % (", ".join(map(str, SAMPLE_RATES)), sample_rate))
        self.client, self.language, self.sample_rate, self.model = client, language, sample_rate, model
        self.on_segment = on_segment
        self.session: SessionCreated | None = None
        self.segments: list[Segment] = []
        self.done: Done | None = None
        self.close_code: int | None = None
        self._ws = None
        self._queue: asyncio.Queue = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._finished = False
        self._committed = False

    def __await__(self):
        return self.__aenter__().__await__()

    @property
    def url(self) -> str:
        base = self.client.base_url
        for http_, ws in (("https://", "wss://"), ("http://", "ws://")):
            if base.startswith(http_):
                base = ws + base[len(http_):]
        query = {"model": self.model, "sample_rate": self.sample_rate}
        if self.language:
            query["language"] = self.language
        return base + "/v1/audio/transcriptions/stream?" + urllib.parse.urlencode(query)

    async def __aenter__(self) -> "TranscribeStream":
        client = self.client
        if not client.api_key:
            raise error_for("invalid_api_key", "No API key. Pass Newera(api_key=...) or set NEWERA_API_KEY.", http_status=401)
        kwargs: dict = {"max_size": None, "additional_headers": client._headers()}
        ctx = client._ssl_context()
        if ctx is not None:
            kwargs["ssl"] = ctx
        try:
            self._ws = await websockets.connect(self.url, **kwargs)
        except websockets.InvalidStatus as exc:
            raise error_for(None, "The server refused the connection (HTTP %s)." % exc.response.status_code,
                            http_status=exc.response.status_code) from exc
        except OSError as exc:
            raise error_for("connection_error", "Could not reach %s: %s" % (client.base_url, exc)) from exc
        try:
            first = await self._next_raw()
            ev = parse_stream_event(first) if first is not None else None
            if isinstance(ev, StreamError):
                raise ev.exception(await self._wait_close_code())
            if not isinstance(ev, SessionCreated):
                raise error_for("closed", "The stream closed before it started.", self._ws.close_code)
            self.session = ev
        except BaseException:
            await self._ws.close()
            raise
        self._reader = asyncio.create_task(self._read())
        return self

    async def _wait_close_code(self) -> int | None:
        try:
            await asyncio.wait_for(self._ws.wait_closed(), 2)
        except Exception:  # noqa: BLE001
            pass
        return self._ws.close_code

    async def _next_raw(self) -> dict | None:
        try:
            while True:
                msg = await self._ws.recv()
                if isinstance(msg, bytes):
                    continue
                try:
                    data = json.loads(msg)
                except ValueError:
                    continue
                if isinstance(data, dict):
                    return data
        except websockets.ConnectionClosed:
            return None

    async def _read(self) -> None:
        try:
            while True:
                raw = await self._next_raw()
                if raw is None:
                    break
                ev = parse_stream_event(raw)
                if isinstance(ev, Segment):
                    self.segments.append(ev)
                    if self.on_segment is not None:
                        res = self.on_segment(ev)
                        if asyncio.iscoroutine(res):
                            await res
                elif isinstance(ev, Done):
                    if not ev.segments:
                        ev.segments = list(self.segments)
                    self.done = ev
                self._queue.put_nowait(ev)
            self.close_code = self._ws.close_code
            if self.done is None and self.close_code not in (None, 1000, 1005):
                self._queue.put_nowait(error_for("closed", "The stream closed (code %s)." % self.close_code, self.close_code))
        except BaseException as exc:  # noqa: BLE001 - handed to the consumer
            self._queue.put_nowait(exc if isinstance(exc, Exception) else error_for("cancelled", "cancelled"))
            if not isinstance(exc, Exception):
                raise
        finally:
            self._queue.put_nowait(_CLOSED)

    async def send(self, pcm: bytes) -> None:
        """Audio: PCM 16-bit little-endian mono at `sample_rate`, any size."""
        if self._committed:
            raise RuntimeError("commit() was already called")
        if pcm:
            await self._ws.send(bytes(pcm))

    async def commit(self) -> None:
        """No more audio: speech in progress is transcribed, every result is delivered, then `Done`."""
        if not self._committed:
            self._committed = True
            await self._ws.send(json.dumps({"type": "commit"}))

    async def send_wav(self, path, *, realtime: bool = True, chunk_ms: int = 100, commit: bool = True) -> None:
        """Stream a 16-bit mono PCM WAV file (the rate of the file must be the stream's `sample_rate`).
        `realtime=True` paces it like a microphone, one chunk every `chunk_ms`."""
        info = wav_info(path)
        if info["sample_rate"] != self.sample_rate:
            raise ValueError("the WAV is %d Hz but the stream was opened with sample_rate=%d" % (info["sample_rate"], self.sample_rate))
        async for chunk in wav_chunks(path, chunk_ms=chunk_ms, realtime=realtime):
            await self.send(chunk)
        if commit:
            await self.commit()

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[StreamEvent]:
        while not self._finished:
            item = await self._queue.get()
            if item is _CLOSED:
                self._finished = True
                return
            if isinstance(item, BaseException):
                self._finished = True
                raise item
            yield item
            if isinstance(item, Done):
                self._finished = True
                return

    async def result(self) -> Done:
        """Wait for `Done` (after `commit()`), skipping the other events."""
        async for _ in self:
            pass
        if self.done is None:
            raise error_for("closed", "The stream closed before the transcript was done.", self.close_code)
        return self.done

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
        if self._reader is not None:
            try:
                await asyncio.wait_for(self._reader, 2)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):  # noqa: BLE001
                self._reader.cancel()

    async def __aexit__(self, *exc) -> None:
        await self.close()


def wav_info(path) -> dict:
    """`{"sample_rate", "seconds"}` of a 16-bit mono PCM WAV; ValueError for any other format."""
    with wave.open(os.fspath(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getcomptype() != "NONE":
            raise ValueError("the WAV must be 16-bit mono PCM (convert: ffmpeg -i in.wav -ac 1 -ar 16000 out.wav)")
        return {"sample_rate": w.getframerate(), "seconds": w.getnframes() / float(w.getframerate())}


async def wav_chunks(path, *, chunk_ms: int = 100, realtime: bool = True) -> AsyncIterator[bytes]:
    """The PCM of a 16-bit mono WAV in chunks of `chunk_ms`, one per `chunk_ms` of wall time when `realtime`."""
    loop = asyncio.get_running_loop()
    with wave.open(os.fspath(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getcomptype() != "NONE":
            raise ValueError("the WAV must be 16-bit mono PCM (convert: ffmpeg -i in.wav -ac 1 -ar 16000 out.wav)")
        frames = max(1, w.getframerate() * chunk_ms // 1000)
        start, n = loop.time(), 0
        while True:
            data = w.readframes(frames)
            if not data:
                return
            yield data
            n += 1
            if realtime:
                await asyncio.sleep(max(0.0, start + n * chunk_ms / 1000.0 - loop.time()))
