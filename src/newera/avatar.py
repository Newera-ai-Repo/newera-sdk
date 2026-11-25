"""The Newera avatar: a talking face on top of realtime voice.

    from newera import Newera

    client = Newera(api_key="nwc_live_...")
    async with client.avatar.connect(avatar="lama", mode="render-text") as call:     # transport="livekit" (default)
        print(call.livekit.url, call.livekit.token)   # a browser or app joins this LiveKit room and plays the avatar
        line = await call.speak("Welcome.")
        await line.wait()

    async with client.avatar.connect(avatar="lama", voice="female", record=True, transport="websocket") as call:
        await call.send_file("question.wav")          # the caller's turn, streamed in real time
        reply = await call.wait_reply()
        print(reply.text)
        call.save_mp4("reply.mp4")                    # the video chunks received on the socket, muxed

You stream the caller's audio in (PCM16 mono 24 kHz); you get back the assistant's speech as video with its audio,
plus the transcripts. With `transport="livekit"` (the default) the video and audio are published in a LiveKit room for
a browser or mobile app to play, and the socket carries only your commands and the events. With
`transport="websocket"` the video arrives on the socket: JPEG frames or H.264 units at 25 fps, each paired with the
audio of exactly those frames. This module is the protocol (`wss://api.newera.ai/v1/avatar`): typed events, errors
mapped to exceptions, the client-secret flow for browsers, and a recorder. `newera avatar` (see talk.py) plays it live.
"""

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
import warnings
import wave
from array import array
from typing import Any, AsyncIterator, Callable, Iterable

import websockets

from .agent import NeweraError

DEFAULT_BASE_URL = "https://api.newera.ai"
MODELS = ("newera-s2s", "newera-s2s-mini")
MODES = ("conversational", "render-audio", "render-text")
VOICES = ("male", "female")
VIDEO_FORMATS = ("h264", "jpeg")
OUTPUTS = ("original", "square")
TRANSPORTS = ("livekit", "websocket")
DEFAULT_TRANSPORT = "livekit"       # media in a LiveKit room; "websocket" puts the video chunks on the socket
DEFAULT_VIDEO_FORMAT = "h264"
DEFAULT_OUTPUT = "original"
FPS = 25
AUDIO_RATE = 24000
CHUNK_MS = 20                       # microphone block: 20 ms, as the protocol recommends


# ------------------------------------------------------------------------------------------ errors

class AvatarError(NeweraError):
    """A refusal or failure from the avatar service. `code` is the error code ("avatar_busy"),
    `close_code` the WebSocket close code (4503) when the socket was closed with one. `retry_after_s` is how
    many seconds to wait before trying again, when the server said (avatar_busy)."""

    def __init__(self, code: str, message: str, close_code: int | None = None, http_status: int | None = None,
                 retry_after_s: int | None = None):
        super().__init__(code, message)
        self.close_code, self.http_status, self.retry_after_s = close_code, http_status, retry_after_s


class InvalidRequest(AvatarError):
    """4400: a bad session field or model (invalid_avatar, invalid_avatar_image, invalid_model, invalid_video_format, invalid_output)."""


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
    """4503 avatar_busy: the avatar can't take this call right now. Retry after `retry_after_s` seconds
    (None when the server did not say)."""


class WrongMode(InvalidRequest):
    """wrong_mode: the event does not belong to this call's mode (a line event in conversational mode, an audio
    line in render-text mode, ...)."""


class LineError(InvalidRequest):
    """A refused line: invalid_line_id, duplicate_line, too_many_lines, line_too_long, invalid_audio, invalid_text,
    text_too_long."""


class AvatarUnavailable(AvatarError):
    """4503 avatar_unavailable: the face or Newera Core is temporarily unavailable. Also `video_format_unavailable`:
    this gateway cannot produce H.264 right now (connect again with the default JPEG)."""


class ByoRefused(InvalidRequest):
    """4400: the server refused to join your own LiveKit room (`livekit_room=`). `code` is one of
    `byo_not_enabled` (not enabled for this account), `byo_host_not_allowed` (that LiveKit host is not enabled by Newera
    for the account), `invalid_livekit_url`, `invalid_livekit_token`, `invalid_livekit` (wrong shape, bad
    `publish_on_behalf`) or `avatar_locked` (the room can only be given in the first session.update). Nothing is billed."""


class WrongTransport(AvatarError):
    """The call uses the LiveKit transport, so the video and its audio are in the LiveKit room and not on the
    socket. The API you used needs them on the socket: connect with `transport="websocket"`."""


class TransportUnavailable(AvatarUnavailable):
    """transport_unavailable (503): the LiveKit transport is not available right now. Connect again with
    `transport="websocket"`, or retry later."""


_BY_CODE = {
    "invalid_avatar": InvalidRequest, "invalid_avatar_image": InvalidRequest, "invalid_model": InvalidRequest,
    "invalid_api_key": AuthenticationError, "invalid_client_secret": AuthenticationError,
    "insufficient_quota": InsufficientQuota,
    "avatar_not_enabled": NotEnabled, "model_not_entitled": NotEnabled,
    "call_time_limit": CallTimeLimit,
    "rate_limit_exceeded": RateLimited, "concurrent_call_limit": RateLimited,
    "avatar_busy": AvatarBusy, "avatar_unavailable": AvatarUnavailable, "model_unavailable": AvatarUnavailable,
    "avatar_concurrency_limit": RateLimited, "session_time_limit": CallTimeLimit,
    "invalid_mode": InvalidRequest, "wrong_mode": WrongMode,
    "byo_not_enabled": ByoRefused, "byo_host_not_allowed": ByoRefused, "invalid_livekit_url": ByoRefused,
    "invalid_livekit_token": ByoRefused, "invalid_livekit": ByoRefused, "avatar_locked": ByoRefused,
    "invalid_transport": InvalidRequest, "transport_unavailable": TransportUnavailable,
    "avatar_render_stalled": AvatarUnavailable, "wrong_transport": WrongTransport,
    "invalid_video_format": InvalidRequest, "invalid_output": InvalidRequest, "video_format_unavailable": AvatarUnavailable,
    "invalid_line_id": LineError, "duplicate_line": LineError, "too_many_lines": LineError,
    "line_too_long": LineError, "invalid_audio": LineError, "invalid_text": LineError, "text_too_long": LineError,
}
_BY_CLOSE = {4400: InvalidRequest, 4401: AuthenticationError, 4402: InsufficientQuota, 4403: NotEnabled,
             4408: CallTimeLimit, 4429: RateLimited, 4503: AvatarUnavailable}
_BY_HTTP = {400: InvalidRequest, 401: AuthenticationError, 402: InsufficientQuota, 403: NotEnabled,
            429: RateLimited, 503: AvatarUnavailable}


def error_for(code: str | None, message: str = "", close_code: int | None = None,
              http_status: int | None = None, retry_after_s: int | None = None) -> AvatarError:
    """The exception for an error code, falling back to the close code or HTTP status."""
    cls = _BY_CODE.get(code or "") or _BY_CLOSE.get(close_code or 0) or _BY_HTTP.get(http_status or 0) or AvatarError
    return cls(code or "error", message, close_code, http_status, retry_after_s)


# ------------------------------------------------------------------------------------------ events

@dataclasses.dataclass
class AvatarEvent:
    """A server event. `raw` is the JSON as sent (without the frames of a video chunk)."""
    raw: dict = dataclasses.field(default_factory=dict, repr=False)

    @property
    def type(self) -> str:
        return self.raw.get("type", "")


@dataclasses.dataclass
class LiveKitJoin:
    """Your own LiveKit room for the avatar to join (`connect(..., livekit_room=LiveKitJoin(...))`).

    `url` is your LiveKit server (`wss://livekit.example.com`); Newera must have enabled that host for your account.
    `token` is a LiveKit access token YOU minted for the avatar's participant (identity containing "avatar", kind
    "agent", grants room_join + can_publish + can_publish_data for that room only). `publish_on_behalf` is optional:
    your agent's identity, so your front end shows the avatar as the agent. The token is hidden from `repr()` and the
    SDK never logs it."""
    url: str
    token: str = dataclasses.field(repr=False)
    publish_on_behalf: str | None = None

    def as_dict(self) -> dict:
        d = {"url": self.url, "token": self.token}
        if self.publish_on_behalf:
            d["publish_on_behalf"] = self.publish_on_behalf
        return d


def _livekit_join(value: "LiveKitJoin | dict") -> LiveKitJoin:
    if isinstance(value, dict):
        extra = set(value) - {"url", "token", "publish_on_behalf"}
        if extra:
            raise ValueError("livekit_room takes url, token and publish_on_behalf, not %s" % ", ".join(sorted(extra)))
        value = LiveKitJoin(url=value.get("url"), token=value.get("token"),
                            publish_on_behalf=value.get("publish_on_behalf"))
    if not isinstance(value, LiveKitJoin):
        raise TypeError("livekit_room must be a LiveKitJoin or a dict {url, token, publish_on_behalf}")
    if not isinstance(value.url, str) or not value.url:
        raise ValueError("livekit_room needs a url, like wss://livekit.example.com")
    if not isinstance(value.token, str) or not value.token:
        raise ValueError("livekit_room needs a token (a LiveKit access token for the avatar's participant)")
    if value.publish_on_behalf is not None and not isinstance(value.publish_on_behalf, str):
        raise ValueError("livekit_room publish_on_behalf must be a string (your agent's identity)")
    return value


@dataclasses.dataclass
class LiveKitRoom:
    """The LiveKit room of a call made with `transport="livekit"` (`call.livekit`). Hand `url` and `token` to the
    page or app that shows the avatar: `await room.connect(url, token)` with any LiveKit client SDK, then play
    the avatar's video and audio tracks. The token is for this room and one viewer, and has to be used within
    10 minutes. When you connected with `livekit_room=` the avatar joined YOUR room instead: `byo` is True, `identity`
    is the avatar's participant identity there and `token` is empty (your clients are already in your room). Line events are also sent in the room as data messages on `data_topics["events"]`."""
    url: str = ""
    token: str = dataclasses.field(default="", repr=False)
    room: str = ""
    identity: str = ""
    avatar_identity: str = ""           # the participant that publishes the avatar's tracks
    video: dict = dataclasses.field(default_factory=dict)         # {"codec": "h264", "width": 512, "height": 512, "fps": 25}
    audio: dict = dataclasses.field(default_factory=dict)         # {"sample_rate": 24000, "channels": 1}
    data_topics: dict = dataclasses.field(default_factory=dict)   # {"events": "avatar.event", "control": "avatar.control"}
    byo: bool = False                   # True when the avatar joined YOUR room (`livekit_room=`): there is no viewer token


def parse_livekit(data: Any) -> LiveKitRoom | None:
    """The `livekit` object of `avatar.ready` as a LiveKitRoom (None when it is missing)."""
    if not isinstance(data, dict) or not data:
        return None
    return LiveKitRoom(url=str(data.get("url") or ""), token=str(data.get("token") or ""),
                       room=str(data.get("room") or ""), identity=str(data.get("identity") or ""),
                       avatar_identity=str(data.get("avatar_identity") or ""),
                       video=dict(data.get("video") or {}), audio=dict(data.get("audio") or {}),
                       data_topics=dict(data.get("data_topics") or {}), byo=bool(data.get("byo")))


@dataclasses.dataclass
class AvatarReady(AvatarEvent):
    """The face has started (sent once, before session.created)."""
    fps: int = FPS
    width: int = 512
    height: int = 512
    video_format: str = "jpeg"
    audio_sample_rate: int = AUDIO_RATE
    frames_per_event: int = 7
    mode: str = "conversational"
    video: dict | None = None           # h264 only: {"codec": "avc1.4D401F", "stream": "annexb", "gop_frames": 50, ...}
    output: str = "square"              # the output the call really has: "original" (your photo's size) or "square" (512 x 512)
    output_reason: str | None = None    # why "original" was not possible, when you asked for it and got "square"
    transport: str = "websocket"        # "livekit": the media is in the LiveKit room (`livekit`), video_format is "webrtc"
    livekit: LiveKitRoom | None = None  # the room to join, with transport "livekit"


@dataclasses.dataclass
class VideoChunk(AvatarEvent):
    """`response.video.chunk`: 1 to 7 frames and the audio of exactly those frames.

    JPEG stream (the default): `frames` are JPEG bytes in display order. H.264 stream (`video_format="h264"`):
    `frames` is empty; `units` are the Annex-B access units, one per frame in stream order, and `keyframes` the
    indices (within this chunk) of the units that are IDR keyframes. Feed them to `newera.h264.H264Decoder` (or
    `LipSyncPlayer`) in order and never skip one. `frame_count` is the number of frames in both forms.
    `video_format` says which form this chunk is.

    `audio` is PCM16 mono bytes at `sample_rate` (None for the
    listening face, `speech=False`, which only comes with idle_video). Show frame `frame_index` at
    `frame_index / fps` seconds on the same clock as the audio.

    `audio_start_ms` / `audio_end_ms` place this chunk's audio on the stream clock (every chunk has them, idle
    ones too); frame `i` of the chunk is due at `audio_start_ms + i * 1000 / fps`. `line_id` is the line that is
    speaking (None for the listening face). `newera.player.LipSyncPlayer` plays this."""
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
    audio_start_ms: float = 0.0
    audio_end_ms: float = 0.0
    line_id: str | None = None
    video_format: str = "jpeg"
    units: list = dataclasses.field(default_factory=list, repr=False)        # h264: one access unit per frame
    keyframes: list = dataclasses.field(default_factory=list)                # h264: indices of IDR units
    frame_count: int = 0                                                      # 0 means: len(frames)

    def __post_init__(self) -> None:
        if not self.frame_count:
            self.frame_count = len(self.frames) or len(self.units)

    def __repr__(self) -> str:
        audio = "%d bytes" % len(self.audio) if self.audio is not None else "None"
        if self.video_format == "h264":
            return ("VideoChunk(frame_index=%d, start_ms=%d, h264 units=%d, keyframes=%s, audio=%s, speech=%s, "
                    "generation_ms=%d)" % (self.frame_index, self.start_ms, len(self.units), list(self.keyframes),
                                           audio, self.speech, self.generation_ms))
        return ("VideoChunk(frame_index=%d, start_ms=%d, frames=%d, audio=%s, speech=%s, generation_ms=%d)"
                % (self.frame_index, self.start_ms, len(self.frames), audio, self.speech, self.generation_ms))

    @property
    def duration_ms(self) -> float:
        return (self.frame_count or len(self.frames)) * 1000.0 / self.fps

    def frame_times_ms(self) -> list:
        """When each frame is due, in ms on the stream clock (the clock of `audio_start_ms`)."""
        return [self.audio_start_ms + i * 1000.0 / self.fps for i in range(self.frame_count or len(self.frames))]


@dataclasses.dataclass
class Interrupted(AvatarEvent):
    """`avatar.interrupted`: the caller started speaking. Stop playback and drop everything queued."""


@dataclasses.dataclass
class Unavailable(AvatarEvent):
    """`avatar.unavailable`: the face stopped mid-call; the voice goes on as AudioDelta events."""
    message: str = ""


@dataclasses.dataclass
class LiveKitDisconnected(AvatarEvent):
    """`avatar.livekit.disconnected`: your LiveKit room dropped the avatar (kicked, room deleted, token expired,
    network). The call then ends normally (close 1000 `livekit_room_disconnected`)."""
    reason: str = ""


@dataclasses.dataclass
class AudioDelta(AvatarEvent):
    """`response.audio.delta`, sent only after `avatar.unavailable`: PCM16 mono 24 kHz bytes."""
    audio: bytes = dataclasses.field(default=b"", repr=False)
    line_id: str | None = None


@dataclasses.dataclass
class LineStarted(AvatarEvent):
    """`avatar.line.started`: just before the first speaking frame of the line. `at_ms` is on the stream clock."""
    line_id: str = ""
    at_ms: float = 0.0


@dataclasses.dataclass
class LineDone(AvatarEvent):
    """`avatar.line.done`: `status` is "completed", "interrupted" or "failed"."""
    line_id: str = ""
    status: str = ""


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
    retry_after_s: int | None = None
    event_id: str | None = None       # the event_id of the client event it answers, when it had one

    def exception(self, close_code: int | None = None) -> AvatarError:
        return error_for(self.code, self.message, close_code, retry_after_s=self.retry_after_s)


@dataclasses.dataclass
class OtherEvent(AvatarEvent):
    """Any other server event of the realtime protocol (session.created, conversation.item.created, ...)."""


def parse_event(data: dict) -> AvatarEvent:
    """A server JSON event as its typed class."""
    t = data.get("type", "")
    if t == "response.video.chunk":
        audio = data.get("audio")
        fmt = data.get("video_format") or "jpeg"
        units: list = []
        if fmt == "h264":
            units = _split_units(base64.b64decode(data.get("video") or ""), data.get("frame_sizes") or [])
        count = int(data.get("frame_count") or 0) or len(data.get("frames") or []) or len(units)
        return VideoChunk(
            raw={"type": t, "event_id": data.get("event_id")},
            frame_index=int(data.get("frame_index", 0)), start_ms=int(data.get("start_ms", 0)),
            frames=[base64.b64decode(f) for f in data.get("frames") or []],
            fps=int(data.get("fps") or FPS), width=int(data.get("width") or 512), height=int(data.get("height") or 512),
            speech=bool(data.get("speech", True)), audio=base64.b64decode(audio) if audio else None,
            sample_rate=int(data.get("audio_sample_rate") or AUDIO_RATE), generation_ms=int(data.get("generation_ms") or 0),
            queued_ms=int(data.get("queued_ms") or 0), index=int(data.get("index") or 0),
            audio_start_ms=_ms(data, "audio_start_ms", data.get("start_ms")),
            audio_end_ms=_ms(data, "audio_end_ms", None) or (
                _ms(data, "audio_start_ms", data.get("start_ms")) + count * 1000.0 / int(data.get("fps") or FPS)),
            line_id=data.get("line_id"), video_format=fmt, units=units,
            keyframes=[int(k) for k in data.get("keyframes") or []], frame_count=count)
    if t == "avatar.ready":
        return AvatarReady(raw=data, fps=int(data.get("fps") or FPS), width=int(data.get("width") or 512),
                           height=int(data.get("height") or 512), video_format=data.get("video_format", "jpeg"),
                           audio_sample_rate=int(data.get("audio_sample_rate") or AUDIO_RATE),
                           frames_per_event=int(data.get("frames_per_event") or 7),
                           output=data.get("output") or "square", output_reason=data.get("output_reason"),
                           mode=data.get("mode") or "conversational",
                           video=data.get("video") if isinstance(data.get("video"), dict) else None,
                           transport=data.get("transport") or ("livekit" if data.get("video_format") == "webrtc" else "websocket"),
                           livekit=parse_livekit(data.get("livekit")))
    if t == "avatar.livekit.disconnected":
        return LiveKitDisconnected(raw=data, reason=str(data.get("reason") or ""))
    if t == "avatar.interrupted":
        return Interrupted(raw=data)
    if t == "avatar.unavailable":
        return Unavailable(raw=data, message=data.get("message", ""))
    if t == "response.audio.delta":
        return AudioDelta(raw={"type": t}, audio=base64.b64decode(data.get("delta") or ""), line_id=data.get("line_id"))
    if t == "avatar.line.started":
        return LineStarted(raw=data, line_id=data.get("line_id") or "", at_ms=_ms(data, "at_ms", 0))
    if t == "avatar.line.done":
        return LineDone(raw=data, line_id=data.get("line_id") or "", status=data.get("status") or "")
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
        retry = e.get("retry_after_s")
        return ErrorEvent(raw=data, code=e.get("code") or "error", message=e.get("message") or "",
                          error_type=e.get("type") or "", param=e.get("param"),
                          retry_after_s=int(retry) if isinstance(retry, (int, float)) else None,
                          event_id=e.get("event_id"))
    return OtherEvent(raw=data)


def _split_units(video: bytes, sizes: list) -> list:
    """The access units of an h264 event: `video` cut at the byte lengths in `sizes`."""
    sizes = [int(n) for n in sizes]
    if sum(sizes) != len(video) or any(n < 0 for n in sizes):
        raise ValueError("h264 video chunk: frame_sizes add up to %d bytes but the video has %d" % (sum(sizes), len(video)))
    units, pos = [], 0
    for n in sizes:
        units.append(video[pos:pos + n])
        pos += n
    return units


def _ms(data: dict, key: str, default) -> float:
    v = data.get(key)
    v = default if v is None else v
    return float(v or 0)


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
        self.units: dict[int, bytes] = {}           # h264 stream: the access unit of each frame, as received
        self.keyframes: set[int] = set()            # h264 stream: the frame numbers of the IDR units

    @property
    def is_h264(self) -> bool:
        return bool(self.units)

    def add(self, chunk: VideoChunk) -> None:
        self.fps, self.sample_rate = chunk.fps, chunk.sample_rate
        n = chunk.frame_count or len(chunk.frames)
        per = (len(chunk.audio) // n) // 2 * 2 if chunk.audio and n else 0       # bytes of audio per frame
        if chunk.video_format == "h264":
            for i, unit in enumerate(chunk.units):
                self.units[chunk.frame_index + i] = unit
                if i in chunk.keyframes:
                    self.keyframes.add(chunk.frame_index + i)
        else:
            for i, jpeg in enumerate(chunk.frames):
                self.jpegs[chunk.frame_index + i] = jpeg
        for i in range(n):
            if per:
                self.pcm[chunk.frame_index + i] = chunk.audio[i * per:(i + 1) * per]

    def __len__(self) -> int:
        return len(self.units) if self.units else len(self.jpegs)

    @property
    def seconds(self) -> float:
        if self.is_h264:
            return len(self._h264_layout()[0]) / self.fps
        return len(self.timeline()[0]) / self.fps

    def timeline(self, max_gap_s: float = 1.0) -> tuple[list[bytes], bytes]:
        """(JPEG frames, PCM16 audio) from the first to the last frame. A gap between replies repeats the
        last frame in silence, at most `max_gap_s` seconds of it. For an h264 call the frames are decoded
        (needs PyAV) and, unlike the MP4, the gap frames are repeated like for JPEG."""
        if self.is_h264:
            return self._h264_timeline(max_gap_s)
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

    # ---- h264 streams: the received access units are kept as they are

    def _h264_layout(self, max_gap_s: float = 1.0) -> tuple[list[tuple[int, int]], bytes]:
        """([(frame number, slot)], PCM16 audio) of the playable part of an h264 stream: it starts at the first
        keyframe (leading units cannot be decoded). A slot is the frame's position on the output clock; a gap
        between frames shrinks to at most `max_gap_s` seconds and is silent. No frame is repeated."""
        order = sorted(self.units)
        first = next((n for n in order if n in self.keyframes), None)
        if first is None:
            return [], b""
        silent = bytes(self.sample_rate // self.fps * 2)
        max_gap = int(max_gap_s * self.fps)
        layout: list[tuple[int, int]] = []
        pcm = bytearray()
        prev, slot = None, 0
        for n in order:
            if n < first:
                continue
            if prev is not None:
                step = min(n - prev, max_gap + 1)
                pcm += silent * (step - 1)
                slot += step
            layout.append((n, slot))
            pcm += self.pcm.get(n, silent)
            prev = n
        return layout, bytes(pcm)

    def _h264_timeline(self, max_gap_s: float = 1.0) -> tuple[list[bytes], bytes]:
        layout, pcm = self._h264_layout(max_gap_s)
        return self._decode_jpegs([n for n, _ in layout]), pcm

    def _decode_jpegs(self, numbers: list[int]) -> list[bytes]:
        """The JPEG of every decodable frame among `numbers` (frame numbers of `units`), in order."""
        from .h264 import H264Decoder
        dec = H264Decoder()
        out: list[bytes] = []
        try:
            for n in numbers:
                out += [f.to_jpeg() for f in dec.decode_units([self.units[n]], [0] if n in self.keyframes else [], n)]
            out += [f.to_jpeg() for f in dec.flush()]
        finally:
            dec.close()
        return out

    def save_h264(self, path: str | os.PathLike) -> str:
        """The video of an h264 call as a raw Annex-B file (`.h264`), exactly the access units that were received,
        from the first keyframe on. Play it with `ffplay -f h264 file.h264`."""
        if not self.is_h264:
            raise RuntimeError("This call has no h264 video. Open it with connect(..., video_format=\"h264\").")
        layout, _ = self._h264_layout()
        if not layout:
            raise RuntimeError("No keyframe was received, so there is nothing to save.")
        with open(os.fspath(path), "wb") as fh:
            for n, _ in layout:
                fh.write(self.units[n])
        return os.fspath(path)

    def _save_mp4_h264(self, path: str | os.PathLike) -> str:
        """MP4 of an h264 call without re-encoding the video: the received access units are muxed as they are, with
        timestamps from their frame numbers (frames that never came leave a gap; none is repeated), and only the
        audio is encoded (PCM to AAC). Needs PyAV."""
        from .h264 import mux_mp4
        layout, pcm = self._h264_layout()
        if not layout:
            raise RuntimeError("No keyframe was received, so there is nothing to save.")
        mux_mp4([(self.units[n], n in self.keyframes, slot) for n, slot in layout], pcm, self.fps, self.sample_rate,
                os.fspath(path))
        return os.fspath(path)

    def save_wav(self, path: str | os.PathLike) -> str:
        """The audio of the call as a PCM16 mono WAV."""
        pcm = self._h264_layout()[1] if self.is_h264 else self.timeline()[1]
        with wave.open(os.fspath(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(pcm)
        return os.fspath(path)

    def save_frames(self, directory: str | os.PathLike) -> str:
        """The frames as 000000.jpg, 000001.jpg, ... in `directory`. An h264 call is decoded to JPEG (needs PyAV)."""
        frames, _ = self.timeline()
        os.makedirs(directory, exist_ok=True)
        for i, jpeg in enumerate(frames):
            with open(os.path.join(directory, "%06d.jpg" % i), "wb") as fh:
                fh.write(jpeg)
        return os.fspath(directory)

    def save_mp4(self, path: str | os.PathLike) -> str:
        """Frames and audio muxed into an MP4 (H.264 and AAC). Needs `ffmpeg` on the PATH; without it,
        use `save_frames()` and `save_wav()`. An h264 call is muxed without re-encoding the video and needs PyAV
        instead of ffmpeg (see `_save_mp4_h264`)."""
        if self.is_h264:
            return self._save_mp4_h264(path)
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
    transport: str = "websocket"            # "livekit": there are no frames, units or audio here (they are in the room)

    @property
    def frames(self) -> list:
        """All the reply's JPEG frames, in order (WrongTransport on a livekit call)."""
        _need_socket_media(self.transport, "reply.frames")
        return [f for c in self.chunks if c.speech for f in c.frames]

    @property
    def units(self) -> list:
        """All the reply's h264 access units, in order (empty for a JPEG call; WrongTransport on a livekit call)."""
        _need_socket_media(self.transport, "reply.units")
        return [u for c in self.chunks if c.speech for u in c.units]

    @property
    def audio(self) -> bytes:
        """All the reply's audio: PCM16 mono 24 kHz (WrongTransport on a livekit call)."""
        _need_socket_media(self.transport, "reply.audio")
        return b"".join(c.audio for c in self.chunks if c.speech and c.audio)


def _need_socket_media(transport: str, what: str) -> None:
    """Raise WrongTransport when `what` needs video or audio on the socket and the call is a livekit call."""
    if transport == "livekit":
        raise WrongTransport("wrong_transport", "%s needs the video on the socket, but this call uses the LiveKit "
                             "transport (the video and audio are in the LiveKit room: see call.livekit). Connect with "
                             "transport=\"websocket\" to use it." % what)


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
    mode: str = "conversational"
    video_format: str | None = None     # what the secret was made for; None = the server's default (h264)
    output: str | None = None           # likewise: None = the server's default (original)
    transport: str | None = None        # "livekit" or "websocket" as asked for; None = not said (the server's default)

    def as_dict(self) -> dict:
        d = {"client_secret": {"value": self.value, "expires_at": self.expires_at}, "model": self.model}
        if self.mode != "conversational":
            d["mode"] = self.mode
        if self.video_format:
            d["video_format"] = self.video_format
        if self.output:
            d["output"] = self.output
        if self.transport:
            d["transport"] = self.transport
        return d


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

    async def create_session(self, model: str = "newera-s2s", *, mode: str = "conversational",
                             video_format: str | None = None, output: str | None = None,
                             transport: str | None = None) -> ClientSecret:
        """Trade the API key for a client secret (`avs_...`, one use, 60 seconds) that a browser opens the
        WebSocket with; call it from your server, never from a page. `mode` is checked against your
        entitlements now (the page still opens the socket with `?mode=` itself). `model` is for conversational.
        `video_format` ("h264" or "jpeg") and `output` ("original" or "square") are checked now and kept in the
        secret as the socket's defaults (the page may still pass `&video_format=` / `&output=` itself); left out,
        the server's defaults apply (h264, original).

        `transport` ("livekit" or "websocket") is checked now and bound into the secret. Left out, nothing is said and
        the server's default (websocket) applies to a page that does not pass `&transport=` itself; pass
        `transport="livekit"` for a page that shows the avatar with the LiveKit client SDK. A `transport_unavailable`
        error means the LiveKit transport is not available right now."""
        _check_mode(mode)
        if transport is not None:
            _check_transport(transport)
        if video_format is not None:
            _check_video_format(video_format)
        if output is not None:
            _check_output(output)
        body: dict = {}
        if mode == "conversational":
            _check_model(model)
            body["model"] = model
        else:
            body["mode"] = mode
        if video_format is not None:
            body["video_format"] = video_format
        if output is not None:
            body["output"] = output
        if transport is not None:
            body["transport"] = transport
        data = await asyncio.to_thread(self._client._http, "POST", "/v1/avatar/sessions", body)
        secret = data.get("client_secret") or {}
        if not secret.get("value"):
            raise AvatarError("invalid_response", "The server did not return a client secret.")
        return ClientSecret(value=secret["value"], expires_at=int(secret.get("expires_at") or 0),
                            model=data.get("model") or model, mode=data.get("mode") or mode,
                            video_format=video_format, output=output, transport=transport)

    def connect(self, avatar: str = "lama", *, voice: str = "female", language: str | None = None,
                model: str = "newera-s2s", image: Any = None, face_crop: bool = True, idle_video: bool = False,
                instructions: str | None = None, input_sample_rate: int = AUDIO_RATE, record: bool = False,
                client_secret: str | ClientSecret | None = None, mode: str | None = None,
                video_format: str | None = None, output: str | None = None, transport: str | None = None,
                livekit_room: "LiveKitJoin | dict | None" = None, **realtime_settings) -> "_Connect":
        """Open a call: `async with client.avatar.connect(...) as call:`.

        mode           "conversational" (default: the avatar listens and answers), "render-audio" (you stream
                       your own PCM16 24 kHz audio, line by line; the face lip-syncs it verbatim) or
                       "render-text" (you send the exact text of each line; Newera voices it). In the two render
                       modes only the face options apply (avatar, image, face_crop, idle_video, record,
                       client_secret); `voice` is chosen per line (`call.speak(text, voice=...)`), and
                       `language`, `instructions`, `model` and realtime settings raise ValueError.

        avatar         a preset face ("lama") ...
        image          ... or your own: a path, bytes, base64 or data URL of a JPEG, PNG or WebP of one person who
                       agreed to be animated. Replaces `avatar`. `face_crop=False` if it is already a head shot.
        voice          "male" or "female", independent of the face.
        language       "ar" or "en" to fix the call's language; None detects it on every turn.
        model          "newera-s2s" or "newera-s2s-mini".
        idle_video     also stream the face while it listens (about ten times the data).
        record         keep every frame and its audio so `call.save_mp4()` works (about 1 MB per second of video).
                       Needs `transport="websocket"`: with livekit there are no frames to keep (WrongTransport).
        client_secret  connect with a secret from `create_session()` instead of the API key.
        transport      "livekit" (default): the avatar's video and audio are sent in a LiveKit (WebRTC) room, to a
                       browser or mobile app: `call.livekit` has the `url` and `token` to join it with any LiveKit client
                       SDK. Lines, interrupts and line events work as before on the socket. Nothing arrives as video on the
                       socket, so `record`, `save_mp4`, `save_frames`, `save_wav`, `frames()`, `units()`, `audio()` and
                       `LipSyncPlayer` raise WrongTransport. "websocket": the video chunks arrive on the socket (the
                       behaviour of 0.5 and before), for when your server processes or saves the frames.
        livekit_room   `LiveKitJoin(url, token, publish_on_behalf=None)` or a dict with those keys: the avatar joins
                       YOUR LiveKit room (with the identity in your token) and publishes its video and audio there,
                       instead of in a Newera room. Needs `transport="livekit"` (ValueError otherwise) and the host
                       enabled by Newera for your account (else `ByoRefused`). `call.livekit` then has `byo=True` and no
                       viewer token. See `call.on_livekit_disconnected`.
        video_format   websocket transport only (the livekit transport has no video_format).
                       "h264" (default: H.264 access units, under 1 Mbit/s; decoded with PyAV, which the SDK installs;
                       `newera.h264.H264Decoder` or `LipSyncPlayer` do it for you) or "jpeg" (JPEG frames, several
                       times the data; no decoder needed).
        output         "original" (default) or "square". "original": the video has the size and framing of the photo
                       you pass as `image` (longest side at most 1280 px), the face animated in place and nothing
                       else moving. "square": the 512 x 512 face only. Without an `image` (a preset face), or when
                       the face cannot be placed in the photo, the call is served "square": see
                       `call.ready.output` and `call.ready.output_reason`.
        realtime_settings  anything else of the realtime `session.update` (turn_detection, tools, ...).
        """
        _check_model(model)
        if video_format is not None:
            _check_video_format(video_format)
        if output is not None:
            _check_output(output)
        if voice is not None and voice not in VOICES:
            raise ValueError("voice must be one of %s, not %r" % (", ".join(VOICES), voice))
        if language not in (None, "auto", "ar", "en"):
            raise ValueError('language must be "ar", "en" or None (detect), not %r' % (language,))
        if isinstance(client_secret, ClientSecret):
            mode = mode or client_secret.mode
            video_format = video_format or client_secret.video_format
            output = output or client_secret.output
            transport = transport or client_secret.transport
            model, client_secret = client_secret.model, client_secret.value
        mode = mode or "conversational"
        _check_mode(mode)
        transport = transport or DEFAULT_TRANSPORT
        _check_transport(transport)
        if livekit_room is not None:
            if transport != "livekit":
                raise ValueError('livekit_room needs transport="livekit" (the avatar joins your room as WebRTC tracks)')
            livekit_room = _livekit_join(livekit_room)
        if transport == "livekit" and record:
            _need_socket_media(transport, "record=True")
        output = output or DEFAULT_OUTPUT
        if transport == "livekit":
            video_format = "webrtc"          # not a choice: the room carries its own H.264 track
        elif video_format is None:
            video_format = DEFAULT_VIDEO_FORMAT
            if not _has_av():                # PyAV is a dependency; a stripped-down install still gets video
                warnings.warn("PyAV (av) is not installed, so the call uses video_format=\"jpeg\"; "
                              "pip install av for H.264.", stacklevel=2)
                video_format = "jpeg"
        if transport == "websocket" and video_format == "h264":
            from .h264 import require_av
            require_av()                    # before any connection: a missing PyAV must not surface mid-call
        if mode == "conversational":
            session = build_session(avatar=avatar, voice=voice, language=language, image=image, face_crop=face_crop,
                                    idle_video=idle_video, instructions=instructions,
                                    input_sample_rate=input_sample_rate, settings=realtime_settings)
        else:
            if language or instructions or realtime_settings or model != "newera-s2s" or input_sample_rate != AUDIO_RATE:
                raise ValueError("language, instructions, model, input_sample_rate and realtime settings are for "
                                 "conversational mode; the render modes take only the face options.")
            session = build_session(avatar=avatar, voice=None, image=image, face_crop=face_crop, idle_video=idle_video)
        if livekit_room is not None:
            session["avatar_livekit"] = livekit_room.as_dict()
        return _Connect(self._client, session, model, record, client_secret, mode, video_format, output, transport)


def _has_av() -> bool:
    try:
        import av  # noqa: F401
    except ImportError:
        return False
    return True


def _check_transport(transport: str) -> None:
    if transport not in TRANSPORTS:
        raise ValueError("transport must be one of %s, not %r" % (", ".join(TRANSPORTS), transport))


def _check_output(output: str) -> None:
    if output not in OUTPUTS:
        raise ValueError("output must be one of %s, not %r" % (", ".join(OUTPUTS), output))


def _check_video_format(video_format: str) -> None:
    if video_format not in VIDEO_FORMATS:
        raise ValueError("video_format must be one of %s, not %r" % (", ".join(VIDEO_FORMATS), video_format))


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise ValueError("mode must be one of %s, not %r" % (", ".join(MODES), mode))


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
    def __init__(self, client, session: dict, model: str, record: bool, secret: str | None,
                 mode: str = "conversational", video_format: str = DEFAULT_VIDEO_FORMAT, output: str = DEFAULT_OUTPUT,
                 transport: str = DEFAULT_TRANSPORT):
        self.client, self.session, self.model, self.record, self.secret = client, session, model, record, secret
        self.mode = mode
        self.video_format = video_format
        self.output = output
        self.transport = transport
        self.call: AvatarCall | None = None

    def __await__(self):
        return self.__aenter__().__await__()

    async def __aenter__(self) -> "AvatarCall":
        kwargs: dict = {"max_size": None}       # a video event is a few hundred KB
        url = self.client.avatar_ws_url
        query: dict = {}
        if self.secret:
            kwargs["subprotocols"] = ["realtime", "newera-client-secret." + self.secret]
        else:
            kwargs["additional_headers"] = self.client._headers()
            if self.mode == "conversational":
                query["model"] = self.model
        if self.mode != "conversational":
            query["mode"] = self.mode          # the secret does not carry the mode: it is always in the URL
        if self.transport == "websocket":
            query["video_format"] = self.video_format  # always said, so a server's own defaults cannot surprise the call
        query["output"] = self.output
        query["transport"] = self.transport            # likewise: a secret's or a server's default must not surprise the call
        if query:
            url += "?" + urllib.parse.urlencode(query)
        ctx = self.client._ssl_context()
        if ctx is not None:
            kwargs["ssl"] = ctx
        try:
            ws = await websockets.connect(url, **kwargs)
        except websockets.InvalidStatus as exc:
            raise error_for(None, "The server refused the connection (HTTP %s)." % exc.response.status_code,
                            http_status=exc.response.status_code) from exc
        self.call = AvatarCall(ws, self.session, record=self.record, mode=self.mode, video_format=self.video_format,
                              transport=self.transport)
        try:
            await self.call._start()
        except BaseException:
            await self.call._abort()
            raise
        return self.call

    async def __aexit__(self, *exc) -> None:
        if self.call is not None:
            await self.call.close()


class Line:
    """One line of speech of a render-mode call, from `call.start_line()`, `call.send_line()` or `call.speak()`.

    Its state follows the server: `await line.wait_started()` (the first speaking frame is about to play) and
    `status = await line.wait()` ("completed", "interrupted" or "failed"). Feed audio with `append()` while
    you synthesise it, or hand `stream()` an iterator; leaving `async with` (or `end()`) closes the line."""

    def __init__(self, call: "AvatarCall", line_id: str, kind: str):
        self.call, self.id, self.kind = call, line_id, kind
        self.status: str | None = None              # None until the server says it is done
        self.error: AvatarError | None = None       # why it failed, when the server refused or the call dropped
        self._started = asyncio.Event()
        self._done = asyncio.Event()
        self._ended = kind == "text"                # nothing more to send
        self._cut = False                           # interrupted: no more audio is taken
        self._sent = kind == "text"                 # the server knows this line

    def __repr__(self) -> str:
        return "Line(%r, %s)" % (self.id, self.status or ("ended" if self._ended else "open"))

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def started(self) -> bool:
        return self._started.is_set()

    @property
    def interrupted(self) -> bool:
        return self._cut or self.status == "interrupted"

    def _finish(self, status: str, error: AvatarError | None = None) -> None:
        if self._done.is_set():
            return
        self.status, self.error = status, error
        self._done.set()

    async def append(self, pcm: bytes) -> bool:
        """Send a piece of the line's audio (PCM16 mono 24 kHz). True when it was sent, False when the line was
        interrupted (stop feeding it). The audio may be as small or as large as you like."""
        if self.kind != "audio":
            raise WrongMode("wrong_mode", "Only render-audio lines take audio.")
        if self._ended:
            raise RuntimeError("line %r has ended" % self.id)
        if self._cut or self._done.is_set():
            return False
        if not pcm:
            return True
        self._sent = True
        await self.call._line_event(self, "avatar.line.append", audio=base64.b64encode(pcm).decode())
        return True

    async def stream(self, chunks: "Iterable[bytes] | AsyncIterator[bytes]", *, end: bool = True) -> bool:
        """Append every chunk of a (async) iterator, e.g. the output of a streaming TTS, then end the line.
        Stops early, without error, when the line is interrupted; returns False in that case."""
        ok = True
        if hasattr(chunks, "__aiter__"):
            async for piece in chunks:          # type: ignore[union-attr]
                if not await self.append(piece):
                    ok = False
                    break
        else:
            for piece in chunks:                # type: ignore[union-attr]
                if not await self.append(piece):
                    ok = False
                    break
        if end and ok:
            await self.end()
        return ok

    async def end(self) -> None:
        """No more audio for this line. Safe to call twice, and after an interrupt."""
        if self._ended:
            return
        self._ended = True
        if self.kind == "audio" and not self._cut and not self._done.is_set():
            self._sent = True
            await self.call._line_event(self, "avatar.line.end")

    async def wait_started(self, timeout: float | None = None) -> bool:
        """True once `avatar.line.started` came; False if the line finished without ever speaking."""
        waiters = [asyncio.ensure_future(self._started.wait()), asyncio.ensure_future(self._done.wait())]
        try:
            await asyncio.wait_for(asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED), timeout)
        finally:
            for w in waiters:
                w.cancel()
        return self._started.is_set()

    async def wait(self, timeout: float | None = None, *, strict: bool = False) -> str:
        """Wait for `avatar.line.done` and return its status. With `strict=True` a "failed" line raises the
        typed exception that explains it (LineError, InsufficientQuota, ...). A dropped call fails open lines."""
        await asyncio.wait_for(self._done.wait(), timeout)
        if strict and self.status == "failed":
            raise self.error or self.call.last_error or AvatarError("line_failed", "The line failed.")
        return self.status or ""

    async def __aenter__(self) -> "Line":
        return self

    async def __aexit__(self, *exc) -> None:
        with contextlib.suppress(Exception):
            await self.end()


class AvatarCall:
    """An open call. Send audio in, iterate `events()` out; `close()` ends it (or leave the `async with`)."""

    def __init__(self, ws, session: dict, *, record: bool = False, mode: str = "conversational",
                 video_format: str = "jpeg", transport: str = "websocket"):
        self._ws = ws
        self.mode = mode
        self.transport = transport                    # "livekit" or "websocket"; what the server answers is in `ready.transport`
        self.livekit: LiveKitRoom | None = None       # the room to join (url, token, ...), from avatar.ready, with livekit
        self.video_format = video_format              # what the server answers in avatar.ready, once started
        self.lines: dict[str, Line] = {}              # the lines you started (render modes), by id
        self.last_error: AvatarError | None = None    # the last `error` event, as its exception
        self._line_callbacks: dict[str, list] = {"started": [], "done": [], "livekit_disconnected": []}
        self.livekit_disconnected: LiveKitDisconnected | None = None   # set when your own LiveKit room dropped the avatar
        self._seq = 0
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
            self._session.pop("avatar_livekit", None)      # your token is not kept past the send
            while True:
                try:
                    raw = await asyncio.wait_for(self._ws.recv(), 60)
                except asyncio.TimeoutError:
                    raise AvatarUnavailable("timeout", "The avatar did not start within 60 seconds.") from None
                ev = parse_event(json.loads(raw))
                if isinstance(ev, AvatarReady):
                    self.ready = ev
                    self.video_format = ev.video_format
                    self.transport = ev.transport
                    self.livekit = ev.livekit
                elif isinstance(ev, ErrorEvent):
                    await self._wait_close()
                    raise ev.exception(self.close_code)
                elif ev.type == "session.created":
                    self.session = ev.raw.get("session", {})
                    if self.livekit is None:             # render modes also carry the room here
                        self.livekit = parse_livekit(ev.raw.get("livekit") or self.session.get("livekit"))
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
                self._track(ev)
                await self._queue.put(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - connection broken
            if not isinstance(exc, websockets.ConnectionClosed):
                await self._queue.put(exc)
        finally:
            self._note_close()
            for line in self.lines.values():
                line._finish("failed", self.last_error or AvatarError(
                    "closed", "The call closed before the line was done.", self.close_code))
            await self._queue.put(None)

    def _track(self, ev: AvatarEvent) -> None:
        """Line bookkeeping and callbacks, done by the reader so it works whether or not you iterate events()."""
        if isinstance(ev, LineStarted):
            line = self.lines.get(ev.line_id)
            if line is not None:
                line._started.set()
            self._fire("started", ev)
        elif isinstance(ev, LineDone):
            line = self.lines.get(ev.line_id)
            if line is not None:
                line._started.set()
                line._finish(ev.status)
            self._fire("done", ev)
        elif isinstance(ev, LiveKitDisconnected):
            self.livekit_disconnected = ev
            self._fire("livekit_disconnected", ev)
        elif isinstance(ev, ErrorEvent):
            self.last_error = ev.exception(self.close_code)
            eid = ev.event_id or ""
            line = self.lines.get(eid[5:].rsplit(":", 1)[0]) if eid.startswith("line:") else None
            if line is not None:          # the server refused this line's event: no `done` will come
                line._finish("failed", self.last_error)

    def _fire(self, name: str, ev: AvatarEvent) -> None:
        for cb in self._line_callbacks[name]:
            try:
                out = cb(ev)
                if asyncio.iscoroutine(out):
                    asyncio.ensure_future(out)
            except Exception:  # noqa: BLE001 - a callback must not break the reader
                pass

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

    # ------------------------------------------------------------------ lines (render modes)

    def _need_mode(self, mode: str) -> None:
        if self.mode != mode:
            raise WrongMode("wrong_mode", "This is a %s call; that needs %s." % (self.mode, mode))

    def _new_line(self, line_id: str | None, kind: str) -> Line:
        if line_id is None:
            self._seq += 1
            line_id = "line-%d" % self._seq
        if not isinstance(line_id, str) or not 1 <= len(line_id) <= 64:
            raise LineError("invalid_line_id", "line_id must be a string of 1 to 64 characters.")
        old = self.lines.get(line_id)
        if old is not None and not old.done:
            raise LineError("duplicate_line", "line_id %r is already in use." % line_id)
        line = self.lines[line_id] = Line(self, line_id, kind)
        return line

    async def _line_event(self, line: Line, kind: str, **fields) -> None:
        self._seq += 1
        await self._send({"type": kind, "line_id": line.id, "event_id": "line:%s:%d" % (line.id, self._seq), **fields})

    def start_line(self, line_id: str | None = None) -> Line:
        """render-audio: begin a line (nothing is sent until its first `append`). `line_id` is yours (1 to 64
        characters; lines play in the order of their first audio) or generated. Use it as
        `async with call.start_line("a") as line: await line.append(pcm)`, or `await line.stream(iterator)`."""
        self._need_mode("render-audio")
        return self._new_line(line_id, "audio")

    async def send_line(self, pcm: bytes, line_id: str | None = None, *, chunk_ms: int = 500) -> Line:
        """render-audio: a whole line from audio you already have (PCM16 mono 24 kHz), sent in `chunk_ms`
        pieces and ended. Returns the Line; `await line.wait()` for its end."""
        line = self.start_line(line_id)
        step = max(2, AUDIO_RATE * chunk_ms // 1000 * 2)
        await line.stream(pcm[i:i + step] for i in range(0, len(pcm), step))
        return line

    async def speak(self, text: str, line_id: str | None = None, *, voice: str | None = None) -> Line:
        """render-text: the exact text of a line (up to 2000 characters); Newera voices it and lip-syncs the
        voice. Nothing is rewritten. `voice` picks the voice for this line (and the ones after it). Returns the Line."""
        self._need_mode("render-text")
        line = self._new_line(line_id, "text")
        fields: dict = {"text": text}
        if voice:
            fields["voice"] = voice
        await self._line_event(line, "avatar.line.text", **fields)
        return line

    async def request_keyframe(self) -> None:
        """h264 only: ask for a keyframe (`avatar.video.keyframe`), for when your decoder lost its picture. The
        server answers with an IDR unit (SPS and PPS in front) in the next video event. Harmless (sends nothing)
        on a JPEG call. `LipSyncPlayer(request_keyframe=call.request_keyframe)` calls it for you. Raises WrongTransport
        on a livekit call (the room's video is not decoded by you)."""
        _need_socket_media(self.transport, "request_keyframe()")
        if self.video_format == "h264" and not self._closed:
            await self._send({"type": "avatar.video.keyframe"})

    async def interrupt(self) -> None:
        """Stop now: queued audio is dropped, nothing rendered from it plays, and every open line ends with
        `LineDone(status="interrupted")` (and an `Interrupted` event). Audio you had not sent yet is dropped
        here: `line.append()` returns False and `line.stream()` stops. In conversational mode the reply is
        cancelled too. Lines you start after this are not affected."""
        for line in self.lines.values():
            if line.done:
                continue
            line._cut = True
            if not line._sent:                    # the server never heard of it: no `done` will come
                line._finish("interrupted")
        await self._send({"type": "avatar.interrupt"})

    def on_line_started(self, callback: Callable[[LineStarted], Any]) -> Callable:
        """Call `callback(LineStarted)` (a function or a coroutine function) at each `avatar.line.started`."""
        self._line_callbacks["started"].append(callback)
        return callback

    def on_line_done(self, callback: Callable[[LineDone], Any]) -> Callable:
        """Call `callback(LineDone)` at each `avatar.line.done` (completed, interrupted or failed)."""
        self._line_callbacks["done"].append(callback)
        return callback

    def on_livekit_disconnected(self, callback: Callable[[LiveKitDisconnected], Any]) -> Callable:
        """Call `callback(LiveKitDisconnected)` when your own LiveKit room (`livekit_room=`) dropped the avatar, so your
        server knows it left. `ev.reason` says why. The call ends normally right after (close code 1000,
        `call.close_reason == "livekit_room_disconnected"`)."""
        self._line_callbacks["livekit_disconnected"].append(callback)
        return callback

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
        """(frame_index, JPEG bytes) of every reply frame, in order, as they arrive. WrongTransport on a livekit call."""
        _need_socket_media(self.transport, "call.frames()")
        async for ev in self.events():
            if isinstance(ev, VideoChunk) and ev.speech:
                for i, jpeg in enumerate(ev.frames):
                    yield ev.frame_index + i, jpeg

    async def units(self) -> AsyncIterator[tuple[int, bytes, bool]]:
        """h264 calls: (frame_index, access unit, is_keyframe) of every reply frame, in order, as they arrive.
        WrongTransport on a livekit call."""
        _need_socket_media(self.transport, "call.units()")
        async for ev in self.events():
            if isinstance(ev, VideoChunk) and ev.speech and ev.video_format == "h264":
                for i, unit in enumerate(ev.units):
                    yield ev.frame_index + i, unit, i in ev.keyframes

    async def audio(self) -> AsyncIterator[bytes]:
        """The reply audio (PCM16 mono 24 kHz) in order, as it arrives. WrongTransport on a livekit call."""
        _need_socket_media(self.transport, "call.audio()")
        async for ev in self.events():
            if isinstance(ev, VideoChunk) and ev.speech and ev.audio:
                yield ev.audio
            elif isinstance(ev, AudioDelta):
                yield ev.audio

    async def wait_reply(self, timeout: float = 90.0, quiet: float = 3.0) -> AvatarReply:
        """Events up to the end of the next reply: `response.done`, then until no reply video has come for
        `quiet` seconds (the voice finishes before the video does). Raises the matching AvatarError on an
        `error` event. Collects text, video and audio."""
        reply = AvatarReply(transport=self.transport)
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

    def _recorded(self, what: str = "saving the call") -> Timeline:
        _need_socket_media(self.transport, what)
        if self.timeline is None:
            raise RuntimeError("This call was not recorded. Open it with connect(..., record=True).")
        return self.timeline

    def save_mp4(self, path: str | os.PathLike) -> str:
        """The recorded video and audio as an MP4 (needs `record=True`). A JPEG call is encoded with ffmpeg. An h264
        call is muxed without re-encoding the video and needs PyAV (the audio is encoded to AAC)."""
        return self._recorded("save_mp4()").save_mp4(path)

    def save_h264(self, path: str | os.PathLike) -> str:
        """An h264 call's video as the raw Annex-B stream that was received (needs `record=True`)."""
        return self._recorded("save_h264()").save_h264(path)

    def save_wav(self, path: str | os.PathLike) -> str:
        """The recorded reply audio as a WAV (needs `record=True`)."""
        return self._recorded("save_wav()").save_wav(path)

    def save_frames(self, directory: str | os.PathLike) -> str:
        """The recorded frames as numbered JPEGs (needs `record=True`; an h264 call is decoded, needs PyAV)."""
        return self._recorded("save_frames()").save_frames(directory)
