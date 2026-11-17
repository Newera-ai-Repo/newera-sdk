"""The Newera avatar: a talking face on top of realtime voice.

    from newera import Newera

    client = Newera(api_key="nwc_live_...")
    async with client.avatar.connect(avatar="lama", voice="female", record=True) as call:
        await call.send_file("question.wav")          # the caller's turn, streamed in real time
        reply = await call.wait_reply()
        print(reply.text)
        call.save_mp4("reply.mp4")                    # JPEG frames and their audio, muxed with ffmpeg

You stream the caller's audio in (PCM16 mono 24 kHz); you get back the assistant's speech as video: JPEG
frames at 25 fps each paired with the audio of exactly those frames, plus the transcripts. This module
is the protocol (`wss://api.newera.ai/v1/avatar`): typed events, errors mapped to exceptions, the
client-secret flow for browsers, and a recorder. `newera avatar` (see talk.py) plays it live.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import wave
from array import array
from typing import Any, AsyncIterator

import websockets

from .agent import NeweraError

DEFAULT_BASE_URL = "https://api.newera.ai"
MODELS = ("newera-s2s", "newera-s2s-mini")
VOICES = ("male", "female")
FPS = 25
AUDIO_RATE = 24000
CHUNK_MS = 20                       # microphone block: 20 ms, as the protocol recommends


# ------------------------------------------------------------------------------------------ errors

class AvatarError(NeweraError):
    """A refusal or failure from the avatar service. `code` is the error code ("avatar_busy"),
    `close_code` the WebSocket close code (4503) when the socket was closed with one."""

    def __init__(self, code: str, message: str, close_code: int | None = None, http_status: int | None = None):
        super().__init__(code, message)
        self.close_code, self.http_status = close_code, http_status


class InvalidRequest(AvatarError):
    """4400: a bad session field or model (invalid_avatar, invalid_avatar_image, invalid_model)."""


class AuthenticationError(AvatarError):
    """4401: missing or wrong key, or a used or expired client secret."""


class InsufficientQuota(AvatarError):
    """4402: the prepaid balance cannot cover the call."""


class NotEnabled(AvatarError):
    """4403: the organisation is not enabled for the avatar or for the voice model."""


class CallTimeLimit(AvatarError):
    """4408: the service ended the call."""


class RateLimited(AvatarError):
    """4429: too many requests, or too many calls on this key right now."""


class AvatarBusy(AvatarError):
    """4503 avatar_busy: the avatar can't take this call right now. Retry shortly."""


class AvatarUnavailable(AvatarError):
    """4503 avatar_unavailable: the face or Newera Core is temporarily unavailable."""


_BY_CODE = {
    "invalid_avatar": InvalidRequest, "invalid_avatar_image": InvalidRequest, "invalid_model": InvalidRequest,
    "invalid_api_key": AuthenticationError, "invalid_client_secret": AuthenticationError,
    "insufficient_quota": InsufficientQuota,
    "avatar_not_enabled": NotEnabled, "model_not_entitled": NotEnabled,
    "call_time_limit": CallTimeLimit,
    "rate_limit_exceeded": RateLimited, "concurrent_call_limit": RateLimited,
    "avatar_busy": AvatarBusy, "avatar_unavailable": AvatarUnavailable,
}
_BY_CLOSE = {4400: InvalidRequest, 4401: AuthenticationError, 4402: InsufficientQuota, 4403: NotEnabled,
             4408: CallTimeLimit, 4429: RateLimited, 4503: AvatarUnavailable}
_BY_HTTP = {400: InvalidRequest, 401: AuthenticationError, 402: InsufficientQuota, 403: NotEnabled,
            429: RateLimited, 503: AvatarUnavailable}


def error_for(code: str | None, message: str = "", close_code: int | None = None,
              http_status: int | None = None) -> AvatarError:
    """The exception for an error code, falling back to the close code or HTTP status."""
    cls = _BY_CODE.get(code or "") or _BY_CLOSE.get(close_code or 0) or _BY_HTTP.get(http_status or 0) or AvatarError
    return cls(code or "error", message, close_code, http_status)


# ------------------------------------------------------------------------------------------ events

@dataclasses.dataclass
class AvatarEvent:
    """A server event. `raw` is the JSON as sent (without the frames of a video chunk)."""
    raw: dict = dataclasses.field(default_factory=dict, repr=False)

    @property
    def type(self) -> str:
        return self.raw.get("type", "")


@dataclasses.dataclass
class AvatarReady(AvatarEvent):
    """The face has started (sent once, before session.created)."""
    fps: int = FPS
    width: int = 512
    height: int = 512
    video_format: str = "jpeg"
    audio_sample_rate: int = AUDIO_RATE
    frames_per_event: int = 7


@dataclasses.dataclass
class VideoChunk(AvatarEvent):
    """`response.video.chunk`: 1 to 7 JPEG frames and the audio of exactly those frames.

    `frames` are JPEG bytes in display order; `audio` is PCM16 mono bytes at `sample_rate` (None for the
    listening face, `speech=False`, which only comes with idle_video). Show frame `frame_index` at
    `frame_index / fps` seconds on the same clock as the audio."""
    frame_index: int = 0
    start_ms: int = 0
    frames: list = dataclasses.field(default_factory=list, repr=False)
    fps: int = FPS
    width: int = 512
    height: int = 512
    speech: bool = True
    audio: bytes | None = dataclasses.field(default=None, repr=False)
    sample_rate: int = AUDIO_RATE
    generation_ms: int = 0
    queued_ms: int = 0
    index: int = 0

    def __repr__(self) -> str:
        return ("VideoChunk(frame_index=%d, start_ms=%d, frames=%d, audio=%s, speech=%s, generation_ms=%d)"
                % (self.frame_index, self.start_ms, len(self.frames),
                   "%d bytes" % len(self.audio) if self.audio is not None else "None", self.speech,
                   self.generation_ms))

    @property
    def duration_ms(self) -> float:
        return len(self.frames) * 1000.0 / self.fps


@dataclasses.dataclass
class Interrupted(AvatarEvent):
    """`avatar.interrupted`: the caller started speaking. Stop playback and drop everything queued."""


@dataclasses.dataclass
class Unavailable(AvatarEvent):
    """`avatar.unavailable`: the face stopped mid-call; the voice goes on as AudioDelta events."""
    message: str = ""


@dataclasses.dataclass
class AudioDelta(AvatarEvent):
    """`response.audio.delta`, sent only after `avatar.unavailable`: PCM16 mono 24 kHz bytes."""
    audio: bytes = dataclasses.field(default=b"", repr=False)


@dataclasses.dataclass
class UserTranscript(AvatarEvent):
    """What the caller said (a finished turn)."""
    text: str = ""
    item_id: str = ""


@dataclasses.dataclass
class ReplyTextDelta(AvatarEvent):
    """A piece of the avatar's reply text. It arrives before the video of that part."""
    text: str = ""
    response_id: str = ""


@dataclasses.dataclass
class ReplyText(AvatarEvent):
    """The whole reply text of one response."""
    text: str = ""
    response_id: str = ""


@dataclasses.dataclass
class SpeechStarted(AvatarEvent):
    """The caller started speaking."""


@dataclasses.dataclass
class SpeechStopped(AvatarEvent):
    """The caller stopped speaking."""


@dataclasses.dataclass
class ResponseDone(AvatarEvent):
    """The reply was generated (its video may still be arriving). `status` is "completed", "cancelled"..."""
    status: str = ""
    text: str = ""
    usage: dict = dataclasses.field(default_factory=dict)
    reason: str | None = None

    @property
    def dropped(self) -> bool:
        """A draft the server dropped because the caller kept talking (not a real reply)."""
        return self.reason == "turn_detected"


@dataclasses.dataclass
class SessionSummary(AvatarEvent):
    """`session.summary`, sent after `session.end`: duration, billed seconds, cost, latency."""
    summary: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class ErrorEvent(AvatarEvent):
    """An `error` event. `exception()` is the matching AvatarError (AvatarBusy, NotEnabled, ...)."""
    code: str = "error"
    message: str = ""
    error_type: str = ""
    param: str | None = None

    def exception(self, close_code: int | None = None) -> AvatarError:
        return error_for(self.code, self.message, close_code)


@dataclasses.dataclass
class OtherEvent(AvatarEvent):
    """Any other server event of the realtime protocol (session.created, conversation.item.created, ...)."""


def parse_event(data: dict) -> AvatarEvent:
    """A server JSON event as its typed class."""
    t = data.get("type", "")
    if t == "response.video.chunk":
        audio = data.get("audio")
        return VideoChunk(
            raw={"type": t, "event_id": data.get("event_id")},
            frame_index=int(data.get("frame_index", 0)), start_ms=int(data.get("start_ms", 0)),
            frames=[base64.b64decode(f) for f in data.get("frames") or []],
            fps=int(data.get("fps") or FPS), width=int(data.get("width") or 512), height=int(data.get("height") or 512),
            speech=bool(data.get("speech", True)), audio=base64.b64decode(audio) if audio else None,
            sample_rate=int(data.get("audio_sample_rate") or AUDIO_RATE), generation_ms=int(data.get("generation_ms") or 0),
            queued_ms=int(data.get("queued_ms") or 0), index=int(data.get("index") or 0))
    if t == "avatar.ready":
        return AvatarReady(raw=data, fps=int(data.get("fps") or FPS), width=int(data.get("width") or 512),
                           height=int(data.get("height") or 512), video_format=data.get("video_format", "jpeg"),
                           audio_sample_rate=int(data.get("audio_sample_rate") or AUDIO_RATE),
                           frames_per_event=int(data.get("frames_per_event") or 7))
    if t == "avatar.interrupted":
        return Interrupted(raw=data)
    if t == "avatar.unavailable":
        return Unavailable(raw=data, message=data.get("message", ""))
    if t == "response.audio.delta":
        return AudioDelta(raw={"type": t}, audio=base64.b64decode(data.get("delta") or ""))
    if t == "conversation.item.input_audio_transcription.completed":
        return UserTranscript(raw=data, text=data.get("transcript") or "", item_id=data.get("item_id", ""))
    if t == "response.audio_transcript.delta":
        return ReplyTextDelta(raw=data, text=data.get("delta") or "", response_id=data.get("response_id", ""))
    if t == "response.audio_transcript.done":
        return ReplyText(raw=data, text=data.get("transcript") or "", response_id=data.get("response_id", ""))
    if t == "input_audio_buffer.speech_started":
        return SpeechStarted(raw=data)
    if t == "input_audio_buffer.speech_stopped":
        return SpeechStopped(raw=data)
    if t == "response.done":
        r = data.get("response") or {}
        reason = (r.get("status_details") or {}).get("reason")
        return ResponseDone(raw=data, status=r.get("status", ""), text=r.get("output_transcript") or _output_text(r),
                            usage=r.get("usage") or {}, reason=reason)
    if t == "session.summary":
        return SessionSummary(raw=data, summary=data.get("summary") or {})
    if t == "error":
        e = data.get("error") or {}
        return ErrorEvent(raw=data, code=e.get("code") or "error", message=e.get("message") or "",
                          error_type=e.get("type") or "", param=e.get("param"))
    return OtherEvent(raw=data)


def _output_text(response: dict) -> str:
    parts = []
    for item in response.get("output") or []:
        for c in item.get("content") or []:
            if c.get("transcript"):
                parts.append(c["transcript"])
    return " ".join(parts)


# ----------------------------------------------------------------------------------- recorded video

class Timeline:
    """The video of a call, placed by `frame_index`: every frame with the audio of that frame."""

    def __init__(self, fps: int = FPS, sample_rate: int = AUDIO_RATE):
        self.fps, self.sample_rate = fps, sample_rate
        self.jpegs: dict[int, bytes] = {}
        self.pcm: dict[int, bytes] = {}

    def add(self, chunk: VideoChunk) -> None:
        self.fps, self.sample_rate = chunk.fps, chunk.sample_rate
        n = len(chunk.frames)
        per = (len(chunk.audio) // n) // 2 * 2 if chunk.audio and n else 0       # bytes of audio per frame
        for i, jpeg in enumerate(chunk.frames):
            self.jpegs[chunk.frame_index + i] = jpeg
            if per:
                self.pcm[chunk.frame_index + i] = chunk.audio[i * per:(i + 1) * per]

    def __len__(self) -> int:
        return len(self.jpegs)

    @property
    def seconds(self) -> float:
        return len(self.timeline()[0]) / self.fps

    def timeline(self, max_gap_s: float = 1.0) -> tuple[list[bytes], bytes]:
        """(JPEG frames, PCM16 audio) from the first to the last frame. A gap between replies repeats the
        last frame in silence, at most `max_gap_s` seconds of it."""
        if not self.jpegs:
            return [], b""
        silent = bytes(self.sample_rate // self.fps * 2)
        order = sorted(self.jpegs)
        frames: list[bytes] = []
        pcm = bytearray()
        prev = None
        for n in order:
            if prev is not None and n - prev > 1:
                for _ in range(min(n - prev - 1, int(max_gap_s * self.fps))):
                    frames.append(self.jpegs[prev])
                    pcm += silent
            frames.append(self.jpegs[n])
            pcm += self.pcm.get(n, silent)
            prev = n
        return frames, bytes(pcm)

    def save_wav(self, path: str | os.PathLike) -> str:
        """The audio of the call as a PCM16 mono WAV."""
        _, pcm = self.timeline()
        with wave.open(os.fspath(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm)
        return os.fspath(path)

    def save_frames(self, directory: str | os.PathLike) -> str:
        """The frames as 000000.jpg, 000001.jpg, ... in `directory`."""
        frames, _ = self.timeline()
        os.makedirs(directory, exist_ok=True)
        for i, jpeg in enumerate(frames):
            with open(os.path.join(directory, "%06d.jpg" % i), "wb") as fh:
                fh.write(jpeg)
        return os.fspath(directory)

    def save_mp4(self, path: str | os.PathLike) -> str:
        """Frames and audio muxed into an MP4 (H.264 and AAC). Needs `ffmpeg` on the PATH; without it,
        use `save_frames()` and `save_wav()`."""
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg was not found on the PATH. Install it (macOS: brew install ffmpeg), or use "
                               "save_frames(dir) and save_wav(path) and mux them yourself.")
        if not self.jpegs:
            raise RuntimeError("No video was received, so there is nothing to save.")
        with tempfile.TemporaryDirectory(prefix="newera-avatar-") as tmp:
            self.save_frames(os.path.join(tmp, "frames"))
            wav = self.save_wav(os.path.join(tmp, "audio.wav"))
            cmd = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(self.fps), "-i",
                   os.path.join(tmp, "frames", "%06d.jpg"), "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-c:a", "aac", "-shortest", os.fspath(path)]
            done = subprocess.run(cmd, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError("ffmpeg failed: %s" % done.stderr.strip()[-300:])
        return os.fspath(path)


@dataclasses.dataclass
class AvatarReply:
    """One finished reply, from AvatarCall.wait_reply()."""
    user_text: str = ""             # what the caller said, as understood
    text: str = ""                  # the reply text
    status: str = ""
    chunks: list = dataclasses.field(default_factory=list, repr=False)   # the VideoChunk events
    events: list = dataclasses.field(default_factory=list, repr=False)
    first_text_s: float | None = None       # seconds after the question audio ended (send_file) ...
    first_video_s: float | None = None      # ... until the first reply text and the first reply video

    @property
    def frames(self) -> list:
        """All the reply's JPEG frames, in order."""
        return [f for c in self.chunks if c.speech for f in c.frames]

    @property
    def audio(self) -> bytes:
        """All the reply's audio: PCM16 mono 24 kHz."""
        return b"".join(c.audio for c in self.chunks if c.speech and c.audio)


# ------------------------------------------------------------------------------------------- audio

def read_wav_24k(path: str | os.PathLike) -> bytes:
    """A WAV file as PCM16 mono 24 kHz bytes (16-bit PCM WAV of any channel count and rate;
    stereo is averaged to mono and other rates are resampled linearly)."""
    with wave.open(os.fspath(path), "rb") as w:
        channels, width, rate, raw = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("%s: need 16-bit PCM WAV (convert with: ffmpeg -i in.m4a -ac 1 -ar 24000 out.wav)" % path)
    return resample_pcm16(raw, rate, AUDIO_RATE, channels)


def resample_pcm16(raw: bytes, rate: int, to_rate: int = AUDIO_RATE, channels: int = 1) -> bytes:
    """PCM16 little-endian bytes: mix to mono, then resample from `rate` to `to_rate` (linear)."""
    samples = array("h")
    samples.frombytes(raw[:len(raw) // 2 * 2])
    if sys.byteorder == "big":       # pragma: no cover
        samples.byteswap()
    if channels > 1:
        usable = len(samples) // channels * channels
        samples = array("h", (sum(samples[i:i + channels]) // channels for i in range(0, usable, channels)))
    if rate != to_rate and len(samples) > 1:
        n_out = int(len(samples) * to_rate / rate)
        step = rate / to_rate
        out = array("h", bytes(2 * n_out))
        last = len(samples) - 1
        for j in range(n_out):
            pos = j * step
            i = int(pos)
            f = pos - i
            a = samples[i]
            b = samples[i + 1] if i < last else a
            out[j] = int(a + (b - a) * f)
        samples = out
    if sys.byteorder == "big":       # pragma: no cover
        samples.byteswap()
    return samples.tobytes()


# ------------------------------------------------------------------------------------------- the API

@dataclasses.dataclass
class ClientSecret:
    """A one-use secret, valid for 60 seconds, that a browser connects with instead of an API key."""
    value: str
    expires_at: int
    model: str

    def as_dict(self) -> dict:
        return {"client_secret": {"value": self.value, "expires_at": self.expires_at}, "model": self.model}


@dataclasses.dataclass
class Availability:
    busy: bool
    available: bool

    @property
    def free(self) -> bool:
        """True when the avatar can take a call right now."""
        return self.available and not self.busy


class Avatar:
    """`client.avatar`: availability, client secrets and calls."""

    def __init__(self, client):
        self._client = client

    async def availability(self) -> Availability:
        """Whether the avatar can take a call right now. Still handle AvatarBusy when connecting."""
        data = await asyncio.to_thread(self._client._http, "GET", "/v1/avatar/availability")
        return Availability(busy=bool(data.get("busy")), available=bool(data.get("available", True)))

    async def create_session(self, model: str = "newera-s2s") -> ClientSecret:
        """Trade the API key for a client secret (`avs_...`, one use, 60 seconds) that a browser opens the
        WebSocket with; call it from your server, never from a page."""
        _check_model(model)
        data = await asyncio.to_thread(self._client._http, "POST", "/v1/avatar/sessions", {"model": model})
        secret = data.get("client_secret") or {}
        if not secret.get("value"):
            raise AvatarError("invalid_response", "The server did not return a client secret.")
        return ClientSecret(value=secret["value"], expires_at=int(secret.get("expires_at") or 0),
                            model=data.get("model") or model)

    def connect(self, avatar: str = "lama", *, voice: str = "female", language: str | None = None,
                model: str = "newera-s2s", image: Any = None, face_crop: bool = True, idle_video: bool = False,
                instructions: str | None = None, input_sample_rate: int = AUDIO_RATE, record: bool = False,
                client_secret: str | ClientSecret | None = None, **realtime_settings) -> "_Connect":
        """Open a call: `async with client.avatar.connect(...) as call:`.

        avatar         a preset face ("lama") ...
        image          ... or your own: a path, bytes, base64 or data URL of a JPEG, PNG or WebP of one person who
                       agreed to be animated. Replaces `avatar`. `face_crop=False` if it is already a head shot.
        voice          "male" or "female", independent of the face.
        language       "ar" or "en" to fix the call's language; None detects it on every turn.
        model          "newera-s2s" or "newera-s2s-mini".
        idle_video     also stream the face while it listens (about ten times the data).
        record         keep every frame and its audio so `call.save_mp4()` works (about 1 MB per second of video).
        client_secret  connect with a secret from `create_session()` instead of the API key.
        realtime_settings  anything else of the realtime `session.update` (turn_detection, tools, ...).
        """
        _check_model(model)
        if voice is not None and voice not in VOICES:
            raise ValueError("voice must be one of %s, not %r" % (", ".join(VOICES), voice))
        if language not in (None, "auto", "ar", "en"):
            raise ValueError('language must be "ar", "en" or None (detect), not %r' % (language,))
        session = build_session(avatar=avatar, voice=voice, language=language, image=image, face_crop=face_crop,
                                idle_video=idle_video, instructions=instructions,
                                input_sample_rate=input_sample_rate, settings=realtime_settings)
        if isinstance(client_secret, ClientSecret):
            model, client_secret = client_secret.model, client_secret.value
        return _Connect(self._client, session, model, record, client_secret)


def _check_model(model: str) -> None:
    if model not in MODELS:
        raise ValueError("model must be one of %s, not %r" % (", ".join(MODELS), model))


def build_session(*, avatar: str = "lama", voice: str | None = "female", language: str | None = None, image: Any = None,
                  face_crop: bool = True, idle_video: bool = False, instructions: str | None = None,
                  input_sample_rate: int = AUDIO_RATE, settings: dict | None = None) -> dict:
    """The `session` of the first session.update."""
    s: dict = dict(settings or {})
    if image is not None:
        s["avatar_image"] = _image_b64(image)
        if not face_crop:
            s["avatar_face_crop"] = False
    else:
        s["avatar"] = avatar
    s["avatar_idle_video"] = bool(idle_video)
    if voice is not None:
        s["voice"] = voice
    if instructions:
        s["instructions"] = instructions
    if input_sample_rate != AUDIO_RATE:
        s["input_audio_sample_rate"] = input_sample_rate
    if language and language != "auto":
        newera = dict(s.get("newera") or {})
        newera["detect_language"] = language
        s["newera"] = newera
    return s


def _image_b64(image: Any) -> str:
    if isinstance(image, (bytes, bytearray)):
        return base64.b64encode(bytes(image)).decode()
    if isinstance(image, os.PathLike) or (isinstance(image, str) and len(image) < 4096 and os.path.isfile(image)):
        with open(image, "rb") as fh:
            return base64.b64encode(fh.read()).decode()
    if isinstance(image, str):
        return image                    # already base64 or a data: URL
    raise TypeError("image must be a path, bytes, or a base64 string")


class _Connect:
    def __init__(self, client, session: dict, model: str, record: bool, secret: str | None):
        self.client, self.session, self.model, self.record, self.secret = client, session, model, record, secret
        self.call: AvatarCall | None = None

    def __await__(self):
        return self.__aenter__().__await__()

    async def __aenter__(self) -> "AvatarCall":
        kwargs: dict = {"max_size": None}       # a video event is a few hundred KB
        url = self.client.avatar_ws_url
        if self.secret:
            kwargs["subprotocols"] = ["realtime", "newera-client-secret." + self.secret]
        else:
            kwargs["additional_headers"] = self.client._headers()
            url += "?" + urllib.parse.urlencode({"model": self.model})
        ctx = self.client._ssl_context()
        if ctx is not None:
            kwargs["ssl"] = ctx
        try:
            ws = await websockets.connect(url, **kwargs)
        except websockets.InvalidStatus as exc:
            raise error_for(None, "The server refused the connection (HTTP %s)." % exc.response.status_code,
                            http_status=exc.response.status_code) from exc
        self.call = AvatarCall(ws, self.session, record=self.record)
        try:
            await self.call._start()
        except BaseException:
            await self.call._abort()
            raise
        return self.call

    async def __aexit__(self, *exc) -> None:
        if self.call is not None:
            await self.call.close()


class AvatarCall:
    """An open call. Send audio in, iterate `events()` out; `close()` ends it (or leave the `async with`)."""

    def __init__(self, ws, session: dict, *, record: bool = False):
        self._ws = ws
        self._session = session
        self._queue: asyncio.Queue = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._silence: asyncio.Task | None = None
        self.ready: AvatarReady | None = None
        self.session: dict = {}
        self.summary: dict = {}
        self.close_code: int | None = None
        self.close_reason: str = ""
        self.timeline: Timeline | None = Timeline() if record else None
        self.question_end: float | None = None        # time.monotonic() when send_file finished
        self.input_sample_rate = int(session.get("input_audio_sample_rate") or AUDIO_RATE)
        self._closed = False

    # ------------------------------------------------------------------ lifecycle

    async def _start(self) -> None:
        # The server waits up to 2 s for the first session.update, because the face is chosen in it, and
        # sends session.created only after the face has started: so send it first, do not wait.
        try:
            try:
                await self._send({"type": "session.update", "session": self._session})
            except websockets.ConnectionClosed:
                pass        # a refused call: the error event is already waiting to be read
            while True:
                try:
                    raw = await asyncio.wait_for(self._ws.recv(), 60)
                except asyncio.TimeoutError:
                    raise AvatarUnavailable("timeout", "The avatar did not start within 60 seconds.") from None
                ev = parse_event(json.loads(raw))
                if isinstance(ev, AvatarReady):
                    self.ready = ev
                elif isinstance(ev, ErrorEvent):
                    await self._wait_close()
                    raise ev.exception(self.close_code)
                elif ev.type == "session.created":
                    self.session = ev.raw.get("session", {})
                    break
        except websockets.ConnectionClosed as exc:
            self._note_close()
            raise error_for(self.close_reason or None, "The connection closed before the call started.",
                            self.close_code) from exc
        self._reader = asyncio.create_task(self._read())

    async def _wait_close(self) -> None:
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._ws.wait_closed(), 2)
        self._note_close()

    def _note_close(self) -> None:
        self.close_code = getattr(self._ws, "close_code", None)
        self.close_reason = getattr(self._ws, "close_reason", "") or ""

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                ev = parse_event(json.loads(raw))
                if isinstance(ev, VideoChunk) and self.timeline is not None and ev.speech:
                    self.timeline.add(ev)
                elif isinstance(ev, SessionSummary):
                    self.summary = ev.summary
                await self._queue.put(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - connection broken
            if not isinstance(exc, websockets.ConnectionClosed):
                await self._queue.put(exc)
        finally:
            self._note_close()
            await self._queue.put(None)

    async def _abort(self) -> None:
        with contextlib.suppress(Exception):
            await self._ws.close()

    async def close(self) -> None:
        """End the call: `session.end`, the summary, then close the socket. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._stop_silence()
        if self._reader and not self._reader.done():
            with contextlib.suppress(Exception):
                await self._send({"type": "session.end"})
            with contextlib.suppress(asyncio.TimeoutError, Exception):
                await asyncio.wait_for(asyncio.shield(self._reader), 3)
        if self._reader:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
        with contextlib.suppress(Exception):
            await self._ws.close()
        self._note_close()

    async def _send(self, event: dict) -> None:
        await self._ws.send(json.dumps(event, ensure_ascii=False))

    # ----------------------------------------------------------------------- send

    async def send_audio(self, pcm: bytes) -> None:
        """PCM16 mono at the call's input rate (24 kHz by default), in real time, about 20 ms at a time.
        Keep sending (silence too) so that turn detection works."""
        self._stop_silence()
        await self._raw_audio(pcm)

    async def _raw_audio(self, pcm: bytes) -> None:
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})

    async def send_file(self, path: str | os.PathLike, *, tail: float = 1.5, keep_silence: bool = True) -> None:
        """Stream a WAV file as the caller, at real time, then `tail` seconds of silence so the turn
        ends. With `keep_silence` the call goes on receiving silence in the background (until the next
        send or close), which the server needs to keep detecting turns."""
        pcm = read_wav_24k(path)
        if self.input_sample_rate != AUDIO_RATE:
            pcm = resample_pcm16(pcm, AUDIO_RATE, self.input_sample_rate)
        await self.stream_audio(pcm, tail=tail)
        self.question_end = time.monotonic() - tail
        if keep_silence:
            self._silence = asyncio.create_task(self._send_silence())

    async def stream_audio(self, pcm: bytes, *, tail: float = 0.0) -> None:
        """Send PCM16 bytes (at the call's input rate) paced at real time, then `tail` seconds of silence."""
        self._stop_silence()
        step = self.input_sample_rate * CHUNK_MS // 1000 * 2
        data = pcm + bytes(int(tail * self.input_sample_rate) * 2)
        start = time.monotonic()
        for n, i in enumerate(range(0, len(data), step)):
            await self._raw_audio(data[i:i + step])
            delay = start + (n + 1) * CHUNK_MS / 1000 - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

    async def _send_silence(self) -> None:
        silence = bytes(self.input_sample_rate * CHUNK_MS // 1000 * 2)
        nxt = time.monotonic()
        try:
            while True:
                await self._raw_audio(silence)
                nxt += CHUNK_MS / 1000
                await asyncio.sleep(max(0.0, nxt - time.monotonic()))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the call ended
            pass

    def _stop_silence(self) -> None:
        if self._silence is not None:
            self._silence.cancel()
            self._silence = None

    async def send_text(self, text: str, *, respond: bool = True) -> None:
        """A typed turn."""
        await self._send({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})
        if respond:
            await self._send({"type": "response.create"})

    async def say(self, instructions: str) -> None:
        """Make the avatar say something (a greeting, for example)."""
        await self._send({"type": "response.create", "response": {"instructions": instructions}})

    async def cancel(self) -> None:
        """Stop the reply that is being generated."""
        await self._send({"type": "response.cancel"})

    async def update(self, **session) -> None:
        """Change realtime settings during the call. The face is fixed: avatar fields get `avatar_locked`."""
        await self._send({"type": "session.update", "session": session})

    # -------------------------------------------------------------------- receive

    def events(self) -> AsyncIterator[AvatarEvent]:
        """Every server event, typed (see the classes above), until the call closes."""
        return self._events()

    def __aiter__(self) -> AsyncIterator[AvatarEvent]:
        return self._events()

    async def _events(self) -> AsyncIterator[AvatarEvent]:
        while True:
            item = await self._queue.get()
            if item is None:
                self._queue.put_nowait(None)    # a later iteration also ends
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def frames(self) -> AsyncIterator[tuple[int, bytes]]:
        """(frame_index, JPEG bytes) of every reply frame, in order, as they arrive."""
        async for ev in self.events():
            if isinstance(ev, VideoChunk) and ev.speech:
                for i, jpeg in enumerate(ev.frames):
                    yield ev.frame_index + i, jpeg

    async def audio(self) -> AsyncIterator[bytes]:
        """The reply audio (PCM16 mono 24 kHz) in order, as it arrives."""
        async for ev in self.events():
            if isinstance(ev, VideoChunk) and ev.speech and ev.audio:
                yield ev.audio
            elif isinstance(ev, AudioDelta):
                yield ev.audio

    async def wait_reply(self, timeout: float = 90.0, quiet: float = 3.0) -> AvatarReply:
        """Events up to the end of the next reply: `response.done`, then until no reply video has come for
        `quiet` seconds (the voice finishes before the video does). Raises the matching AvatarError on an
        `error` event. Collects text, video and audio."""
        reply = AvatarReply()
        deadline = time.monotonic() + timeout
        done_at: float | None = None
        last_video = time.monotonic()
        while True:
            now = time.monotonic()
            if now > deadline:
                raise asyncio.TimeoutError("no finished reply within %s s" % timeout)
            if done_at is not None and now - last_video >= quiet:
                return reply
            wait = min(deadline - now, 0.5 if done_at is not None else deadline - now)
            try:
                item = await asyncio.wait_for(self._queue.get(), max(wait, 0.01))
            except asyncio.TimeoutError:
                continue
            if item is None:
                self._queue.put_nowait(None)
                if done_at is not None:
                    return reply
                raise AvatarError("closed", "The call closed before the avatar replied.", self.close_code)
            if isinstance(item, BaseException):
                raise item
            ev = item
            reply.events.append(ev)
            if isinstance(ev, UserTranscript):
                reply.user_text = ev.text
            elif isinstance(ev, ReplyTextDelta):
                if reply.first_text_s is None and self.question_end is not None:
                    reply.first_text_s = time.monotonic() - self.question_end
            elif isinstance(ev, VideoChunk):
                if ev.speech:
                    reply.chunks.append(ev)
                    last_video = time.monotonic()
                    if reply.first_video_s is None and self.question_end is not None:
                        reply.first_video_s = last_video - self.question_end
            elif isinstance(ev, ResponseDone):
                if ev.dropped:
                    continue
                reply.text, reply.status = ev.text or reply.text, ev.status
                done_at, last_video = time.monotonic(), time.monotonic()
            elif isinstance(ev, ErrorEvent):
                raise ev.exception(self.close_code)

    # ----------------------------------------------------------------------- save

    def _recorded(self) -> Timeline:
        if self.timeline is None:
            raise RuntimeError("This call was not recorded. Open it with connect(..., record=True).")
        return self.timeline

    def save_mp4(self, path: str | os.PathLike) -> str:
        """The recorded video and audio as an MP4 (needs ffmpeg; needs `record=True`)."""
        return self._recorded().save_mp4(path)

    def save_wav(self, path: str | os.PathLike) -> str:
        """The recorded reply audio as a WAV (needs `record=True`)."""
        return self._recorded().save_wav(path)

    def save_frames(self, directory: str | os.PathLike) -> str:
        """The recorded frames as numbered JPEGs (needs `record=True`)."""
        return self._recorded().save_frames(directory)
