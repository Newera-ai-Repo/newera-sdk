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
write it to a non-blocking audio stream. `show_frame(jpeg_bytes, frame_number)` shows one JPEG. Both may be plain
functions or coroutine functions. For production use with a particular audio device, adapt it to that device's
own clock; `talk.Playback` is the device version for `newera avatar`.
"""

from __future__ import annotations

import asyncio
import collections
import time
from typing import Any, AsyncIterator, Callable

from .avatar import AUDIO_RATE, AvatarEvent, Interrupted, VideoChunk


class LipSyncPlayer:
    def __init__(self, play_audio: Callable[[bytes], Any], show_frame: Callable[[bytes, int], Any], *,
                 flush_audio: Callable[[], Any] | None = None, buffer_ms: float = 250, audio_lead_ms: float = 60,
                 max_late_ms: float = 120, show_idle: bool = True, clock: Callable[[], float] = time.monotonic):
        self.play_audio, self.show_frame, self.flush_audio = play_audio, show_frame, flush_audio
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
            if event.speech or self.show_idle:
                self.sample_rate = event.sample_rate
                self._queue.append((self._gen, event))
                self._wake.set()
        elif isinstance(event, Interrupted):
            self.interrupt()

    def interrupt(self) -> None:
        """Drop everything queued and scheduled, stop the audio (`flush_audio`), and forget the clock mapping."""
        self._gen += 1
        self._queue.clear()
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
            gen, chunk = self._queue.popleft()
            if gen != self._gen:
                continue
            await self._schedule(gen, chunk)

    async def _drain_frames(self) -> None:
        while self._frame_tasks:
            await asyncio.gather(*list(self._frame_tasks), return_exceptions=True)

    async def _schedule(self, gen: int, chunk: VideoChunk) -> None:
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
        task = asyncio.ensure_future(self._show(gen, chunk, self._t0))
        self._frame_tasks.add(task)
        task.add_done_callback(self._frame_tasks.discard)

    async def _show(self, gen: int, chunk: VideoChunk, t0: float) -> None:
        for i, (jpeg, ms) in enumerate(zip(chunk.frames, chunk.frame_times_ms())):
            await self._sleep_until(t0 + ms / 1000)
            if gen != self._gen:
                return
            await _call(self.show_frame, jpeg, chunk.frame_index + i)

    async def _sleep_until(self, wall: float) -> None:
        delay = wall - self.clock()
        if delay > 0:
            await asyncio.sleep(delay)


async def _call(fn: Callable, *args) -> None:
    out = fn(*args)
    if asyncio.iscoroutine(out):
        await out
