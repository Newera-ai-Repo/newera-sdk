"""Decode the avatar's H.264 video (`video_format="h264"`), and mux it into an MP4 without re-encoding.

    from newera.h264 import H264Decoder

    decoder = H264Decoder(on_lost=lambda: print("ask for a keyframe"))
    async for ev in call.events():
        if isinstance(ev, VideoChunk) and ev.video_format == "h264":
            for frame in decoder.decode(ev):          # DecodedFrame, in display order
                show(frame.rgb, frame.width, frame.height)

PyAV is a dependency of the SDK, imported only when a stream is really decoded or muxed: importing this module, and
every JPEG call, never loads it.

The stream rules (see `VideoChunk`): access units are decoded in stream order and none is skipped, except when the
decoder joins or recovers at a keyframe (SPS and PPS are in front of every keyframe). So `H264Decoder` drops units
until the first keyframe, starts clean at every later keyframe, and when one unit fails to decode it counts the error,
drops the following units until the next keyframe and calls `on_lost` (ask the server for one with
`await call.request_keyframe()`). Colours are BT.601 limited range, as signalled in the stream.
"""

from __future__ import annotations

import collections
import dataclasses
import io
import threading
import time
from fractions import Fraction
from typing import Any, Callable, Iterable

from .avatar import AvatarError

INSTALL_HINT = 'H.264 video needs PyAV: pip install av (it is a dependency of newera-sdk; reinstall it with pip install -U newera-sdk)'


def require_av():
    """The `av` module (PyAV), or an AvatarError("av_not_installed") that says how to install it."""
    try:
        import av
    except ImportError as exc:
        raise AvatarError("av_not_installed", INSTALL_HINT) from exc
    return av


# ------------------------------------------------------------------------------------------- frames

@dataclasses.dataclass
class DecodedFrame:
    """One decoded picture. `rgb` is RGB24 (`width * height * 3` bytes, no row padding). `index` is the frame number
    on the call's stream (`VideoChunk.frame_index + i`); `pts` is the same number (in 1/fps units)."""
    width: int
    height: int
    rgb: bytes = dataclasses.field(repr=False)
    pts: int = 0
    index: int = 0

    def to_ndarray(self):
        """A numpy array of shape (height, width, 3), dtype uint8. numpy is optional and imported here."""
        try:
            import numpy as np
        except ImportError as exc:
            raise ImportError("DecodedFrame.to_ndarray() needs numpy: pip install numpy") from exc
        return np.frombuffer(self.rgb, dtype=np.uint8).reshape(self.height, self.width, 3)

    def to_jpeg(self, quality: int = 90) -> bytes:
        """The picture as JPEG bytes. Uses Pillow when it is installed, otherwise PyAV's mjpeg encoder."""
        try:
            from PIL import Image
        except ImportError:
            return self._jpeg_with_av(quality)
        buf = io.BytesIO()
        Image.frombuffer("RGB", (self.width, self.height), self.rgb, "raw", "RGB", 0, 1).save(buf, "JPEG", quality=quality)
        return buf.getvalue()

    def _jpeg_with_av(self, quality: int) -> bytes:
        av = require_av()
        frame = av.VideoFrame(self.width, self.height, "rgb24")
        plane = frame.planes[0]
        stride, row = plane.line_size, self.width * 3
        data = bytearray(plane.buffer_size)
        for r in range(self.height):
            data[r * stride:r * stride + row] = self.rgb[r * row:(r + 1) * row]
        plane.update(bytes(data))
        frame = frame.reformat(format="yuvj420p")
        q = max(2, min(31, round(31 - (max(1, min(100, quality)) - 1) * 29 / 99)))     # mjpeg qscale: 2 best, 31 worst
        ctx = av.CodecContext.create("mjpeg", "w")
        ctx.width, ctx.height, ctx.pix_fmt = self.width, self.height, "yuvj420p"
        ctx.time_base = Fraction(1, 25)
        ctx.options = {"qmin": str(q), "qmax": str(q)}
        ctx.open()
        frame.pts = 0
        packets = list(ctx.encode(frame)) + list(ctx.encode(None))
        return b"".join(bytes(p) for p in packets)


class _PyAVCodec:
    """One H.264 decoder context of PyAV: `decode(unit) -> [DecodedFrame]`, `flush()`."""

    def __init__(self):
        self.av = require_av()
        self.ctx = self.av.CodecContext.create("h264", "r")
        try:
            self.ctx.options = {"flags": "low_delay"}            # a frame comes out as soon as its unit is decoded
        except Exception:  # noqa: BLE001 - an option the build does not know
            pass

    def decode(self, unit: bytes) -> list:
        return [self._convert(f) for f in self.ctx.decode(self.av.Packet(unit))]

    def flush(self) -> list:
        return [self._convert(f) for f in self.ctx.decode(None)]

    @staticmethod
    def _convert(frame) -> DecodedFrame:
        rgb = frame.reformat(format="rgb24")                    # BT.601 limited range in, as the stream signals
        plane, w, h = rgb.planes[0], rgb.width, rgb.height
        raw, row = bytes(plane), w * 3
        if plane.line_size != row:                               # drop the row padding
            raw = b"".join(raw[r * plane.line_size:r * plane.line_size + row] for r in range(h))
        return DecodedFrame(w, h, raw)


# ----------------------------------------------------------------------------------------- decoder

class H264Decoder:
    """Decodes `VideoChunk`s of an h264 call into `DecodedFrame`s, with the keyframe rules of the stream.

    on_lost          called (from whichever thread decodes) when the picture is lost or a join is waiting for a
                     keyframe: ask the server for one (`call.request_keyframe()`). Rate limited to one call per
                     `request_interval_s` while units keep being dropped.
    codec_factory    returns an object with `decode(unit) -> [DecodedFrame]` (and optionally `flush()`); the
                     default is a PyAV decoder, created on the first decode. For tests.

    Counters: `errors` (units that failed to decode), `dropped` (units skipped while waiting for a keyframe),
    `keyframes` (keyframes the decoder started from), `frames` (pictures produced). `synced` is False until the
    first keyframe and again after a failure. Safe to call from a worker thread; one decode at a time.
    """

    def __init__(self, *, on_lost: Callable[[], Any] | None = None, codec_factory: Callable[[], Any] | None = None,
                 request_interval_s: float = 1.0, clock: Callable[[], float] = time.monotonic):
        self.on_lost = on_lost
        self.errors = self.dropped = self.keyframes = self.frames = 0
        self.synced = False
        self._factory = codec_factory or _PyAVCodec
        self._clock, self._interval = clock, request_interval_s
        self._codec: Any = None
        self._pending: collections.deque = collections.deque()      # frame numbers of units whose picture is not out yet
        self._next_index: int | None = None
        self._last_ask: float | None = None
        self._last_index = -1
        self._lock = threading.RLock()

    # -- public -------------------------------------------------------------------------------------------------

    def decode(self, chunk) -> list:
        """The frames of one h264 `VideoChunk`, in order (fewer than its units when some were dropped)."""
        if getattr(chunk, "video_format", "jpeg") != "h264":
            raise ValueError("H264Decoder.decode() takes an h264 VideoChunk, not a %s one" % chunk.video_format)
        with self._lock:
            if self.synced and self._next_index is not None and chunk.frame_index != self._next_index \
                    and 0 not in chunk.keyframes:
                self._lose(ask=False)                    # a hole in the stream that no keyframe covers
            return self.decode_units(chunk.units, chunk.keyframes, chunk.frame_index)

    def decode_units(self, units: Iterable[bytes], keyframes: Iterable[int] = (), index0: int = 0) -> list:
        """Decode access units given with the indices (within `units`) of the keyframes among them; unit `i` is
        frame number `index0 + i`."""
        keys = set(keyframes)
        out: list = []
        with self._lock:
            count = 0
            for i, unit in enumerate(units):
                count = i + 1
                idx = index0 + i
                if i in keys:
                    out += self._resync()
                    self.synced = True
                    self.keyframes += 1
                elif not self.synced:
                    self.dropped += 1
                    self._ask()
                    continue
                try:
                    if self._codec is None:
                        self._codec = self._factory()
                    self._pending.append(idx)
                    frames = self._codec.decode(unit)
                except Exception:  # noqa: BLE001 - one bad unit: count it, wait for a keyframe
                    self.errors += 1
                    self._lose(ask=True)
                    continue
                out += self._label(frames)
            self._next_index = index0 + count
        return out

    def flush(self) -> list:
        """Pictures still inside the decoder (none for the server's streams, which have no reordering)."""
        with self._lock:
            return self._flush()

    def reset(self) -> None:
        """Forget everything: the next unit that decodes is the next keyframe (the server sends one after an
        interrupt). Does not ask for a keyframe."""
        with self._lock:
            self._drop_codec()
            self.synced = False
            self._next_index = None
            self._last_ask = None

    def close(self) -> None:
        self.reset()

    # -- internals ----------------------------------------------------------------------------------------------

    def _flush(self) -> list:
        fn = getattr(self._codec, "flush", None)
        if self._codec is None or fn is None or not self._pending:
            return []
        try:
            return self._label(fn())
        except Exception:  # noqa: BLE001
            return []

    def _resync(self) -> list:
        """A keyframe starts clean: finish what the old decoder still holds, then use a new one."""
        out = self._flush()
        self._drop_codec()
        return out

    def _drop_codec(self) -> None:
        codec, self._codec = self._codec, None
        self._pending.clear()
        close = getattr(codec, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001
                pass

    def _lose(self, *, ask: bool) -> None:
        self._drop_codec()
        self.synced = False
        if ask:
            self._ask(force=True)

    def _ask(self, force: bool = False) -> None:
        if self.on_lost is None:
            return
        now = self._clock()
        if not force and self._last_ask is not None and now - self._last_ask < self._interval:
            return
        self._last_ask = now
        try:
            self.on_lost()
        except Exception:  # noqa: BLE001 - a callback must not break decoding
            pass

    def _label(self, frames: list) -> list:
        for f in frames:
            idx = self._pending.popleft() if self._pending else self._last_index + 1
            f.index = f.pts = self._last_index = idx
            self.frames += 1
        return frames


# --------------------------------------------------------------------------------------------- mp4

def mux_mp4(units: list, pcm: bytes, fps: int, sample_rate: int, path: str) -> None:
    """Write an MP4 whose video is the given access units as they are (no re-encode) and whose audio is `pcm`
    (PCM16 mono) encoded to AAC. `units` is a list of `(access_unit, is_keyframe, slot)`: the unit is shown at
    `slot / fps` seconds and the first one must be a keyframe. The video timestamps are variable (a missing frame is
    a gap, none is repeated). Needs PyAV 14 or newer."""
    av = require_av()
    if not units or not units[0][1]:
        raise ValueError("the video must start at a keyframe")
    out_tb = Fraction(1, 90000)
    unit_ticks = round(90000 / fps)
    data = b"".join(u for u, _, _ in units)
    with av.open(io.BytesIO(data), format="h264") as src:
        vin = src.streams.video[0]
        if not getattr(vin.codec_context, "extradata", None):
            raise RuntimeError("PyAV did not find the SPS/PPS of the h264 stream (is the first unit a keyframe?)")
        parsed = [p for p in src.demux(vin) if p.size]
        out = av.open(path, "w", format="mp4")
        try:
            if not hasattr(out, "add_stream_from_template"):
                raise RuntimeError("Muxing without re-encoding needs PyAV 14 or newer: pip install -U \"av>=14\"")
            vout = out.add_stream_from_template(vin)
            if len(parsed) == len(units):
                packets = parsed                           # the parser split the stream exactly at the units
            else:                                          # defensive: one packet per received unit, by hand
                packets = []
                for u, key, _ in units:
                    p = av.Packet(u)
                    try:
                        p.is_keyframe = key
                    except (AttributeError, TypeError):
                        pass
                    packets.append(p)
            astream = out.add_stream("aac", rate=sample_rate, layout="mono") if pcm else None
            for p, (_, _, slot) in zip(packets, units):
                p.stream = vout
                p.time_base = out_tb
                p.pts = p.dts = round(slot * 90000 / fps)
                p.duration = unit_ticks
                out.mux(p)
            if astream is not None:
                step = sample_rate // 5 * 2                # 0.2 s per AudioFrame
                for off in range(0, len(pcm) - len(pcm) % 2, step):
                    piece = pcm[off:min(off + step, len(pcm) - len(pcm) % 2)]
                    frame = av.AudioFrame(format="s16", layout="mono", samples=len(piece) // 2)
                    frame.sample_rate = sample_rate
                    plane = frame.planes[0]
                    plane.update(piece[:plane.buffer_size].ljust(plane.buffer_size, b"\0"))
                    frame.pts = off // 2
                    frame.time_base = Fraction(1, sample_rate)
                    out.mux(astream.encode(frame))
                out.mux(astream.encode(None))
        finally:
            out.close()
