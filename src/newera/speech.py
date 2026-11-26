"""Text to speech over HTTP: `client.audio.speech(...)`, with `stream=True` to receive the audio as it is made.

    client = Newera()

    wav = await client.audio.speech("Welcome to the clinic.", voice="female")           # bytes of a WAV file

    async for chunk in client.audio.speech("A long text ...", response_format="pcm", stream=True):
        play(chunk)                       # raw 24 kHz 16-bit mono PCM, the first sentence after about a second

The text is voiced a sentence or clause at a time and each piece is sent as soon as it is ready. Pauses
(`<break time="500ms"/>` in the text) and `pronunciations` work with and without `stream`.
"""

from __future__ import annotations

import asyncio
import http.client
import json
import socket
import threading
import urllib.error
import urllib.request
from typing import Any, AsyncIterator

from .avatar import QUALITIES, error_for
from . import stt

RESPONSE_FORMATS = ("wav", "pcm")
SAMPLE_RATE = 24000
_DONE = object()


class Audio:
    """`client.audio`: text to speech (`speech`) and speech to text (`transcribe`, `transcribe_stream`)."""

    def __init__(self, client):
        self._client = client

    def speech(self, input: str, *, model: str = "newera-tts", voice: str | None = None, quality: str | None = None,
               normalize: bool | None = None, pronunciations: dict | None = None, response_format: str = "wav",
               speed: float | None = None, language: str | None = None, stream: bool = False) -> Any:
        """Speak `input` (`POST /v1/audio/speech`).

        Without `stream`: `await client.audio.speech(...)` returns the audio, the bytes of a WAV file (or raw PCM with
        `response_format="pcm"`: 24 kHz, 16-bit, mono).

        With `stream=True`: `async for chunk in client.audio.speech(..., stream=True)` yields the audio as it is made,
        one chunk or more per sentence, so playback can start after the first one. With `response_format="pcm"` the
        chunks are raw audio; with "wav" (the default) the first chunk is a WAV header whose length fields are open
        ("until the stream ends"), which players read, followed by the same audio. You are billed for the audio produced,
        also if you stop iterating early. An error before the first chunk raises the matching exception; one after it
        raises `AvatarError("stream_interrupted")`.

        voice            "male" or "female".
        quality          "instant", "balanced" or "studio" (the default).
        normalize        False speaks the text exactly as written; by default numbers, times and amounts are read as words.
        pronunciations   {"Newera": "نيو إيرا"}: how to say a word, whole words, any case, applied after numbers are read.
        text             `<break time="500ms"/>` is a pause of exactly that length (at most 5 s each).
        """
        body = _body(input, model, voice, quality, normalize, pronunciations, response_format, speed, language, stream)
        return self._stream(body) if stream else self._once(body)

    async def transcribe(self, file, *, language: str | None = None, prompt: str | None = None, vad: bool = True,
                         response_format: str = "json", temperature: float | None = None,
                         model: str = stt.MODEL) -> "stt.Transcription":
        """Speech to text for a file (`POST /v1/audio/transcriptions`, up to 25 MB; wav, mp3, m4a, ogg, flac, webm).

        file             a path, the bytes, or a binary file object.
        language         "ar" or "en". Send "ar" for Arabic: the language is not detected.
        prompt           words to help the model (names, terms).
        vad              True (default): a clip with no speech returns "" without running the model (it is still billed
                         for its length). False if you already remove silence.
        response_format  "json" (default) or "text"; `Transcription.text` is the text either way.
        temperature      0 to 1.

        Returns `Transcription(text, audio_seconds)`; `audio_seconds` is the billed length from the
        `x-newera-audio-seconds` header. Errors raise the SDK exceptions (`AuthenticationError`, `InsufficientQuota`,
        `NotEnabled`, `RateLimited`, `InvalidRequest`, ...).
        """
        return await asyncio.to_thread(stt.transcribe_blocking, self._client, file, language=language, prompt=prompt,
                                       vad=vad, response_format=response_format, temperature=temperature, model=model)

    def transcribe_stream(self, *, language: str | None = None, sample_rate: int = 16000, model: str = stt.MODEL,
                          on_segment=None) -> "stt.TranscribeStream":
        """Live speech to text: `async with client.audio.transcribe_stream(language="ar") as stream:` then
        `await stream.send(pcm)`, `await stream.commit()` and `async for event in stream` (`Segment`, `SpeechStarted`,
        `SpeechStopped`, `Done`). `on_segment(segment)` (sync or async) is called for each segment. Audio is PCM 16-bit
        little-endian mono at `sample_rate` (8000, 16000, 24000 or 48000). `stream.send_wav(path)` streams a WAV file at
        real-time pace."""
        return stt.TranscribeStream(self._client, language=language, sample_rate=sample_rate, model=model,
                                    on_segment=on_segment)

    async def _once(self, body: dict) -> bytes:
        return await asyncio.to_thread(self._read_all, body)

    def _open(self, body: dict, timeout: float):
        client = self._client
        if not client.api_key:
            raise error_for("invalid_api_key", "No API key. Pass Newera(api_key=...) or set NEWERA_API_KEY.", http_status=401)
        headers = {**client._headers(), "Content-Type": "application/json"}
        req = urllib.request.Request(client.base_url + "/v1/audio/speech", data=json.dumps(body).encode(),
                                     method="POST", headers=headers)
        try:
            return urllib.request.urlopen(req, timeout=timeout, context=client._ssl_context())
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

    def _read_all(self, body: dict) -> bytes:
        with self._open(body, timeout=300) as resp:
            return resp.read()

    async def _stream(self, body: dict) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        holder: dict = {}

        def put(item) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def pump() -> None:
            try:
                resp = self._open(body, timeout=120)
                holder["resp"] = resp
                put(b"")                              # the response is open: an error status was raised above
                while not stop.is_set():
                    chunk = resp.read1(65536)
                    if not chunk:
                        break
                    put(chunk)
            except BaseException as exc:  # noqa: BLE001 - handed to the consumer
                if not stop.is_set():
                    if isinstance(exc, (http.client.IncompleteRead, http.client.HTTPException, OSError)):
                        exc = error_for("stream_interrupted", "The audio stream was cut before its end.")
                    put(exc)
            finally:
                put(_DONE)

        thread = threading.Thread(target=pump, name="newera-speech-stream", daemon=True)
        thread.start()
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                if item:
                    yield item
        finally:
            stop.set()                           # leaving early: close the connection, so the server stops voicing
            resp = holder.get("resp")
            if resp is not None:
                try:
                    sock = getattr(getattr(resp.fp, "raw", None), "_sock", None)
                    if sock is not None:             # a read may be blocked in the thread: shut the socket, not only the file
                        sock.shutdown(socket.SHUT_RDWR)
                    resp.close()
                except Exception:  # noqa: BLE001
                    pass


def _body(input, model, voice, quality, normalize, pronunciations, response_format, speed, language, stream) -> dict:
    if not isinstance(input, str) or not input.strip():
        raise ValueError("input must be the text to speak")
    if response_format not in RESPONSE_FORMATS:
        raise ValueError("response_format must be one of %s, not %r" % (", ".join(RESPONSE_FORMATS), response_format))
    body: dict = {"model": model, "input": input, "response_format": response_format}
    if voice is not None:
        body["voice"] = voice
    if quality is not None:
        if quality not in QUALITIES:
            raise ValueError("quality must be one of %s, not %r" % (", ".join(QUALITIES), quality))
        body["quality"] = quality
    if normalize is not None:
        if not isinstance(normalize, bool):
            raise ValueError("normalize must be True or False, not %r" % (normalize,))
        body["normalize"] = normalize
    if pronunciations is not None:
        if (not isinstance(pronunciations, dict)
                or not all(isinstance(k, str) and k.strip() and isinstance(v, str) and v.strip() for k, v in pronunciations.items())):
            raise ValueError('pronunciations must be a dict of words and how to say them: {"Newera": "نيو إيرا"}')
        body["pronunciations"] = dict(pronunciations)
    if speed is not None:
        body["speed"] = speed
    if language is not None:
        body["language"] = language
    if stream:
        body["stream"] = True
    return body
