"""A small reference player that keeps the avatar lip-synced. No dependencies: you give it two functions.

    player = LipSyncPlayer(play_audio=speaker.write, show_frame=window.show, flush_audio=speaker.clear)
    await player.play(call.events())            # until the call ends

The audio is the master clock. Every `VideoChunk` says where its audio sits on the stream clock
(`audio_start_ms`); the player maps that clock to wall time once, `buffer_ms` ahead of the first chunk, writes each
chunk's audio just before it is due (`audio_lead_ms`), and shows frame `i` of the chunk exactly at
`audio_start_ms + i * 1000 / fps` on the same mapping. If the network delivers a chunk late the whole mapping
moves forward (audio is never skipped or cut), so sound and lips stay together. `Interrupted` (or
`interrupt()`) drops everything queued and calls `flush_audio` so your speaker stops at once.

`play_audio(pcm16_bytes)` is called with PCM16 mono audio (24 kHz, see `player.sample_rate`) for speaking chunks;
write it to a non-blocking audio stream. `show_frame(frame, frame_number)` shows one frame: for a JPEG call `frame` is
the JPEG bytes (unchanged); for an h264 call (`connect(video_format="h264")`) it is a `newera.h264.DecodedFrame`
(`.rgb`, `.width`, `.height`, `.to_jpeg()`, `.to_ndarray()`), or JPEG bytes too with `frame_format="jpeg"`. Both
may be plain functions or coroutine functions.

h264: the player decodes every chunk in stream order on one worker thread (the event loop is not blocked, and chunks
are decoded as soon as they are fed, ahead of their time), including idle chunks you do not show, because the decoder
may not skip a unit. The clock logic is the same as for JPEG. Pass `request_keyframe=call.request_keyframe` and the
player asks the server for a keyframe when its decoder loses the picture; on `Interrupted` it resets the decoder (the
server sends a keyframe next). Needs PyAV (installed with the SDK). For production use with a particular audio device, adapt it to that device's
own clock; `talk.Playback` is the device version for `newera avatar`.
"""

from __future__ import annotations

import asyncio
import collections
import concurrent.futures
import time
from typing import Any, AsyncIterator, Callable

from .avatar import AUDIO_RATE, AvatarEvent, Interrupted, VideoChunk, _need_socket_media


class LipSyncPlayer:
    def __init__(self, play_audio: Callable[[bytes], Any], show_frame: Callable[[bytes, int], Any], *,
                 flush_audio: Callable[[], Any] | None = None, buffer_ms: float = 250, audio_lead_ms: float = 60,
                 max_late_ms: float = 120, show_idle: bool = True, clock: Callable[[], float] = time.monotonic,
                 request_keyframe: Callable[[], Any] | None = None, frame_format: str = "auto", decoder: Any = None,
                 call: Any = None):
        # The player plays video chunks from the socket. A livekit call has none (its video is in the LiveKit room),
        # so say so now instead of playing nothing. `call` is the AvatarCall; `call.request_keyframe` also names it.
        call = call if call is not None else getattr(request_keyframe, "__self__", None)
        if getattr(call, "transport", None) == "livekit":
            _need_socket_media("livekit", "LipSyncPlayer")
        if frame_format not in ("auto", "jpeg"):
            raise ValueError('frame_format must be "auto" or "jpeg", not %r' % (frame_format,))
        self.play_audio, self.show_frame, self.flush_audio = play_audio, show_frame, flush_audio
        self.request_keyframe, self.frame_format = request_keyframe, frame_format
        self.decoder = decoder                         # an H264Decoder (made on the first h264 chunk when None)
        self.keyframe_requests = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self.buffer_ms, self.audio_lead_ms, self.max_late_ms, self.show_idle = buffer_ms, audio_lead_ms, max_late_ms, show_idle
        self.clock = clock
        self.sample_rate = AUDIO_RATE
        self.late_ms_total = 0.0                       # how far the clock had to move because chunks came late
        self._queue: collections.deque = collections.deque()
        self._wake = asyncio.Event()
        self._gen = 0
        self._t0: float | None = None                  # wall time (s) of stream time 0
        self._end_ms = 0.0                             # stream time up to which something was scheduled
        self._frame_tasks: set = set()
        self._closed = False

    # ----------------------------------------------------------------------------------------- input

    def feed(self, event: AvatarEvent) -> None:
        """Hand over a server event: video chunks are queued, `Interrupted` flushes. Others are ignored."""
        if isinstance(event, VideoChunk):
            fut = self._submit_decode(event) if event.video_format == "h264" else None
            if event.speech or self.show_idle:
                self.sample_rate = event.sample_rate
                self._queue.append((self._gen, event, fut))
                self._wake.set()
        elif isinstance(event, Interrupted):
            self.interrupt()

    def interrupt(self) -> None:
        """Drop everything queued and scheduled, stop the audio (`flush_audio`), and forget the clock mapping."""
        self._gen += 1
        self._queue.clear()
        if self._executor is not None and self.decoder is not None:
            self._executor.submit(self._safe, self.decoder.reset)      # after the decodes already queued, before new ones
        self._t0 = None
        for t in list(self._frame_tasks):
            t.cancel()
        self._frame_tasks.clear()
        if self.flush_audio is not None:
            out = self.flush_audio()
            if asyncio.iscoroutine(out):
                asyncio.ensure_future(out)

    def close(self) -> None:
        """Stop the playback loop after the queue is empty (see `play`)."""
        self._closed = True
        self._wake.set()

    # ---------------------------------------------------------------------------------------- output

    async def play(self, events: AsyncIterator[AvatarEvent] | None = None) -> None:
        """Run until `events` ends and everything has played (or `close()` was called). Without `events`, call
        `feed()` yourself from another task."""
        feeder = asyncio.ensure_future(self._feed_all(events)) if events is not None else None
        try:
            await self._run()
        finally:
            if feeder is not None:
                feeder.cancel()
            for t in list(self._frame_tasks):
                t.cancel()
            if self._executor is not None:
                self._executor.shutdown(wait=False)
                self._executor = None

    async def _feed_all(self, events) -> None:
        try:
            async for ev in events:
                self.feed(ev)
        finally:
            self.close()

    async def _run(self) -> None:
        while True:
            if not self._queue:
                if self._closed:
                    await self._drain_frames()
                    return
                self._wake.clear()
                await self._wake.wait()
                continue
            gen, chunk, fut = self._queue.popleft()
            if gen != self._gen:
                continue
            await self._schedule(gen, chunk, fut)

    async def _drain_frames(self) -> None:
        while self._frame_tasks:
            await asyncio.gather(*list(self._frame_tasks), return_exceptions=True)

    async def _schedule(self, gen: int, chunk: VideoChunk, fut: Any = None) -> None:
        shown = None
        if fut is not None:                                      # h264: the decode was started when the chunk was fed
            try:
                shown = await fut
            except Exception:  # noqa: BLE001 - the decode job does not raise; a cancelled one shows nothing
                shown = []
            if gen != self._gen:
                return
        now = self.clock()
        start = chunk.audio_start_ms
        if self._t0 is None or start > self._end_ms + 250:       # first chunk, or a gap in the stream
            self._t0 = now + self.buffer_ms / 1000 - start / 1000
        due = self._t0 + start / 1000
        late_ms = (now - due) * 1000
        if late_ms > self.max_late_ms:                           # arrived late: move the clock, keep the audio whole
            self._t0 += late_ms / 1000
            self.late_ms_total += late_ms
            due = self._t0 + start / 1000
        self._end_ms = max(self._end_ms, chunk.audio_end_ms)
        await self._sleep_until(due - self.audio_lead_ms / 1000)
        if gen != self._gen:
            return
        if chunk.speech and chunk.audio:
            await _call(self.play_audio, chunk.audio)
        if shown is None:
            shown = [(ms, chunk.frame_index + i, jpeg) for i, (jpeg, ms) in enumerate(zip(chunk.frames, chunk.frame_times_ms()))]
        task = asyncio.ensure_future(self._show(gen, shown, self._t0))
        self._frame_tasks.add(task)
        task.add_done_callback(self._frame_tasks.discard)

    async def _show(self, gen: int, frames: list, t0: float) -> None:
        for ms, n, frame in frames:
            await self._sleep_until(t0 + ms / 1000)
            if gen != self._gen:
                return
            await _call(self.show_frame, frame, n)

    # ------------------------------------------------------------------------------------------- h264

    def _submit_decode(self, chunk: VideoChunk):
        """Queue the decode of one h264 chunk on the single worker thread (so chunks are decoded strictly in order)
        and return its future: a list of `(due ms, frame number, frame)`. Every chunk goes through here, shown or not."""
        loop = self._loop = asyncio.get_running_loop()
        if self.decoder is None:
            from .h264 import H264Decoder
            self.decoder = H264Decoder()
        if getattr(self.decoder, "on_lost", None) is None:
            self.decoder.on_lost = self._on_lost
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="newera-h264")
        show = chunk.speech or self.show_idle
        return loop.run_in_executor(self._executor, self._decode_job, chunk, show)

    def _decode_job(self, chunk: VideoChunk, show: bool) -> list:
        try:
            frames = self.decoder.decode(chunk)
        except Exception:  # noqa: BLE001 - a broken decoder shows nothing; the next keyframe recovers
            return []
        if not show:
            return []
        out = []
        for f in frames:
            payload = f.to_jpeg() if self.frame_format == "jpeg" else f
            out.append((chunk.audio_start_ms + (f.index - chunk.frame_index) * 1000.0 / chunk.fps, f.index, payload))
        return out

    @staticmethod
    def _safe(fn: Callable) -> None:
        try:
            fn()
        except Exception:  # noqa: BLE001
            pass

    def _on_lost(self) -> None:
        """The decoder lost the picture (any thread): ask the server for a keyframe."""
        if self.request_keyframe is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(self._send_keyframe_request)
            except RuntimeError:                          # the loop is closed
                pass

    def _send_keyframe_request(self) -> None:
        self.keyframe_requests += 1
        out = self.request_keyframe()
        if asyncio.iscoroutine(out):
            asyncio.ensure_future(out)

    async def _sleep_until(self, wall: float) -> None:
        delay = wall - self.clock()
        if delay > 0:
            await asyncio.sleep(delay)


async def _call(fn: Callable, *args) -> None:
    out = fn(*args)
    if asyncio.iscoroutine(out):
        await out
