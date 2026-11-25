"""An interactive call with the avatar: microphone in, speaker out, the face in a window.

    pip install "newera-sdk[avatar]"
    export NEWERA_API_KEY=nwc_live_...
    newera avatar                    # or, from Python:  from newera import talk; talk()

Use headphones: the avatar's voice from the speakers would be picked up by the microphone and heard as
the caller interrupting. Press Ctrl+C (or close the window, or press Esc) to end the call.

The playback follows the browser player of the API docs: the audio is the clock. Each video event brings
its frames and the audio of exactly those frames; the audio is queued to the speaker, frames are shown
when the audio clock reaches their time, a reply starts after a short buffer (a short buffer
keeps playback smooth), and `avatar.interrupted` drops everything not yet played.
"""

from __future__ import annotations

import asyncio
import collections
import io
import os
import threading
import time
from typing import Any

from .avatar import (AUDIO_RATE, AudioDelta, AvatarBusy, AvatarError, ErrorEvent, Interrupted, NotEnabled,
                     ReplyTextDelta, ResponseDone, SpeechStarted, Unavailable, UserTranscript, VideoChunk)
from .client import Newera
from .h264 import H264Decoder

INSTALL_HINT = 'Install the call dependencies with:  pip install "newera-sdk[avatar]"'


class Playback:
    """The audio clock and the frame schedule of the avatar player. No audio or window code in here:
    `read()` is what the speaker callback pulls, `due_frame()` is what the window asks for."""

    def __init__(self, rate: int = AUDIO_RATE, prebuffer: float = 2.0):
        self.rate, self.prebuffer = rate, prebuffer
        self._lock = threading.Lock()
        self._buf = bytearray()                       # audio not yet played
        self._read_pos = 0                            # samples played so far: the clock
        self._frames: collections.deque = collections.deque()      # (at_sample, jpeg)
        self._next_frame = -1                         # frame_index that continues the current run
        self.interrupts = 0

    @property
    def clock(self) -> float:
        """Seconds of audio played (silence included)."""
        return self._read_pos / self.rate

    @property
    def buffered(self) -> float:
        """Seconds of audio queued and not yet played."""
        return len(self._buf) / 2 / self.rate

    def add(self, ev: VideoChunk, frames: list | None = None) -> None:
        """A `response.video.chunk`, in the order it arrived. For an h264 chunk pass its decoded `frames`
        (`H264Decoder.decode(ev)`); they are what `due_frame()` returns, in place of JPEG bytes."""
        with self._lock:
            end = self._read_pos + len(self._buf) // 2
            contiguous = ev.frame_index == self._next_frame
            if not contiguous or not self._buf:
                # The start of a reply (or of idle video), or the buffer ran dry: restart the clock with a cushion.
                lead = 0.4 if contiguous else (self.prebuffer if ev.speech else 0.2)
                at = max(end, self._read_pos + int(lead * self.rate))
                self._buf += bytes((at - end) * 2)
            else:
                at = end
            per = self.rate // ev.fps                                    # samples per frame
            count = ev.frame_count or len(ev.frames)
            want = per * count * 2
            audio = (ev.audio or b"")[:want]
            self._buf += audio + bytes(want - len(audio))                # no audio (idle face): silence keeps the clock
            if ev.video_format == "h264":
                for f in frames or []:
                    self._frames.append((at + (f.index - ev.frame_index) * per, f))
            else:
                for i, jpeg in enumerate(ev.frames):
                    self._frames.append((at + i * per, jpeg))
            self._next_frame = ev.frame_index + count

    def add_audio(self, pcm: bytes) -> None:
        """Voice-only audio (after `avatar.unavailable` the server sends response.audio.delta)."""
        with self._lock:
            if not self._buf:
                self._buf += bytes(int(0.2 * self.rate) * 2)
            self._buf += pcm

    def interrupt(self) -> None:
        """The caller started talking: stop the sound, drop every frame not shown yet."""
        with self._lock:
            self._buf.clear()
            self._frames.clear()
            self._next_frame = -1
            self.interrupts += 1

    def read(self, nbytes: int) -> bytes:
        """The next `nbytes` of audio for the speaker (silence when nothing is queued). Advances the clock."""
        with self._lock:
            out = bytes(self._buf[:nbytes])
            del self._buf[:nbytes]
            if len(out) < nbytes:
                out += bytes(nbytes - len(out))
            self._read_pos += nbytes // 2
            return out

    def due_frame(self):
        """The newest frame whose time has come on the audio clock (older ones are skipped), or None: JPEG bytes,
        or a `DecodedFrame` on an h264 call."""
        with self._lock:
            jpeg = None
            while self._frames and self._frames[0][0] <= self._read_pos:
                jpeg = self._frames.popleft()[1]
            return jpeg

    @property
    def idle(self) -> bool:
        """Nothing left to play or show."""
        with self._lock:
            return not self._buf and not self._frames


class Status:
    """What the window says under the face, set by the call task, read by the window."""

    def __init__(self):
        self.text = "Connecting..."


async def _drive(opts: dict, player: Playback, status: Status, stop: asyncio.Event, out) -> int:
    client = Newera(opts.get("api_key"), base_url=opts.get("base_url"))
    if not client.api_key:
        out("No API key. Set NEWERA_API_KEY (export NEWERA_API_KEY=nwc_live_...) or pass --api-key.")
        return 2
    try:
        avail = await client.avatar.availability()
        if avail.busy:
            out("The avatar can't take this call right now. Try again shortly.")
            return 3
        if not avail.available:
            out("The avatar is temporarily unavailable. Try again shortly.")
            return 3
    except AvatarError as exc:
        return _report(exc, out)

    mic_stream = out_stream = None
    input_file = opts.get("input_file")
    try:
        async with client.avatar.connect(
                opts.get("avatar") or "lama", voice=opts.get("voice") or "female", language=opts.get("language"),
                model=opts.get("model") or "newera-s2s", image=opts.get("image"), idle_video=bool(opts.get("idle_video")),
                instructions=opts.get("instructions"), record=bool(opts.get("record")),
                video_format=opts.get("video_format"), output=opts.get("output")) as call:
            loop = asyncio.get_running_loop()
            if not opts.get("mute"):
                out_stream = _open_output(player)
            mic_q: asyncio.Queue = asyncio.Queue()
            if input_file:
                out("Playing %s as the caller. Press Ctrl+C to stop." % input_file)
            else:
                mic_stream = _open_input(loop, mic_q, opts.get("input_device"))
                out("Connected. Speak now (Ctrl+C or Esc ends the call). Use headphones, or the avatar hears itself.")
            status.text = "Listening..." if not input_file else "Playing the question..."
            feeder = asyncio.create_task(_feed_file(call, input_file) if input_file else _feed_mic(call, mic_q))
            state = {"reply_done": False, "last_video": time.monotonic(), "in_reply": False, "frames": 0}
            consumer = asyncio.create_task(_events(call, player, status, state, out, _make_decoder(call, loop)))
            started = time.monotonic()
            try:
                while not stop.is_set():
                    await asyncio.sleep(0.1)
                    if consumer.done():
                        break
                    if input_file and state["reply_done"] and time.monotonic() - state["last_video"] > 3.0 and player.idle:
                        break
                    if input_file and time.monotonic() - started > float(opts.get("max_seconds") or 120):
                        out("No finished reply within %s s." % (opts.get("max_seconds") or 120))
                        state["failed"] = True
                        break
            finally:
                feeder.cancel()
                consumer.cancel()
                for t in (feeder, consumer):
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):  # noqa: BLE001
                        pass
            status.text = "Ending the call..."
            await call.close()
            if opts.get("record") and call.timeline is not None and len(call.timeline):
                out("Saved %s" % call.save_mp4(opts["record"]))
            summ = call.summary
            if summ:
                out("Call ended: %s s connected, billed %s s." % (summ.get("duration_s"),
                    (summ.get("billed") or {}).get("connected_seconds")))
            else:
                out("Call ended.")
            if state.get("error"):
                return 1
            return 1 if state.get("failed") else 0
    except AvatarError as exc:
        return _report(exc, out)
    finally:
        for s in (mic_stream, out_stream):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:  # noqa: BLE001
                    pass


def _report(exc: AvatarError, out) -> int:
    if isinstance(exc, AvatarBusy):
        out("The avatar can't take this call right now. Try again shortly.")
        return 3
    if isinstance(exc, NotEnabled):
        out("This key is not enabled for the avatar (%s). Ask your Newera account manager to enable it." % exc.code)
        return 4
    out("%s: %s" % (exc.code, exc.message))
    return 1


async def _feed_mic(call, q: asyncio.Queue) -> None:
    while True:
        await call.send_audio(await q.get())


async def _feed_file(call, path: str) -> None:
    await call.send_file(path)               # then silence in the background keeps turn detection running
    await asyncio.Event().wait()


def _make_decoder(call, loop) -> H264Decoder | None:
    """The h264 decoder of an h264 call (None for JPEG). When it loses the picture it asks for a keyframe."""
    if call.video_format != "h264":
        return None
    return H264Decoder(on_lost=lambda: loop.call_soon_threadsafe(lambda: asyncio.ensure_future(call.request_keyframe())))


async def _events(call, player: Playback, status: Status, state: dict, out, decoder: H264Decoder | None = None) -> None:
    loop = asyncio.get_running_loop()
    async for ev in call.events():
        if isinstance(ev, VideoChunk):
            if ev.video_format == "h264" and decoder is not None:
                # decode off the event loop, one chunk at a time (awaited here), so the units stay in order
                player.add(ev, await loop.run_in_executor(None, decoder.decode, ev))
            else:
                player.add(ev)
            if ev.speech:
                state["last_video"] = time.monotonic()
                status.text = "Speaking"
        elif isinstance(ev, AudioDelta):
            player.add_audio(ev.audio)
            state["last_video"] = time.monotonic()
        elif isinstance(ev, (Interrupted, SpeechStarted)):
            player.interrupt()
            if isinstance(ev, Interrupted) and decoder is not None:
                decoder.reset()                      # the server sends a keyframe next
            status.text = "Listening..."
            state["reply_done"] = False
        elif isinstance(ev, UserTranscript):
            if state["in_reply"]:
                print()
                state["in_reply"] = False
            if ev.text.strip():
                out("You:    " + ev.text.strip())
        elif isinstance(ev, ReplyTextDelta):
            if not state["in_reply"]:
                print("Avatar: ", end="", flush=True)
                state["in_reply"] = True
            print(ev.text, end="", flush=True)
        elif isinstance(ev, ResponseDone):
            if state["in_reply"]:
                print()
                state["in_reply"] = False
            if not ev.dropped:
                state["reply_done"] = True
                state["last_video"] = time.monotonic()
        elif isinstance(ev, Unavailable):
            out("(The face stopped; the voice goes on without it. %s)" % ev.message)
        elif isinstance(ev, ErrorEvent):
            out("Error: %s: %s" % (ev.code, ev.message))
            if ev.code not in ("avatar_locked",):
                state["error"] = True
                if call.close_code:       # the server closes after a fatal error
                    return


def _open_output(player: Playback):
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit(INSTALL_HINT)

    def callback(outdata, frames, time_info, status):
        outdata[:] = player.read(frames * 2)

    stream = sd.RawOutputStream(samplerate=AUDIO_RATE, channels=1, dtype="int16", blocksize=480, callback=callback)
    stream.start()
    return stream


def _open_input(loop, q: asyncio.Queue, device: Any = None):
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit(INSTALL_HINT)

    def callback(indata, frames, time_info, status):
        data = bytes(indata)
        loop.call_soon_threadsafe(_put, q, data)

    def _put(queue, data):
        if queue.qsize() < 150:        # 3 s: a stalled connection drops audio instead of growing
            queue.put_nowait(data)

    stream = sd.RawInputStream(samplerate=AUDIO_RATE, channels=1, dtype="int16", blocksize=480, device=device,
                               callback=callback)
    stream.start()
    return stream


def _fit(width: int, height: int, longer: int) -> tuple[int, int]:
    """The size of a window that shows a width x height picture with its longer side `longer` (same shape)."""
    if width <= 0 or height <= 0:
        return longer, longer
    scale = longer / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _run_window(done: threading.Event, request_stop, player: Playback, status: Status, opts: dict) -> None:
    """The window, on the main thread (macOS needs it there): show each frame when its time comes."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame
    except ImportError:
        raise SystemExit(INSTALL_HINT)
    pygame.init()
    size = int(opts.get("window_size") or 512)            # the longer side of the window
    win = (size, size)
    screen = pygame.display.set_mode(win)
    pygame.display.set_caption("Newera avatar (Esc or Ctrl+C to end)")
    font = pygame.font.Font(None, 26)
    clock = pygame.time.Clock()
    shot_dir = opts.get("screenshot_dir")
    shots, last_shot, shown = 0, 0.0, 0
    surface = None
    try:
        while not done.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)):
                    request_stop()
            if status.text == "Speaking" and player.idle:
                status.text = "Listening..."
            jpeg = player.due_frame()
            if jpeg is not None:
                try:
                    if isinstance(jpeg, bytes):
                        surface = pygame.image.load(io.BytesIO(jpeg), "frame.jpg")
                    else:                           # an h264 call: a DecodedFrame of RGB24
                        surface = pygame.image.frombuffer(jpeg.rgb, (jpeg.width, jpeg.height), "RGB").copy()
                    shown += 1
                    want = _fit(surface.get_width(), surface.get_height(), size)
                    if want != win:                 # the picture is not square (output="original"): the window is its shape
                        win = want
                        screen = pygame.display.set_mode(win)
                except pygame.error:
                    pass
            screen.fill((18, 18, 20))
            if surface is not None:
                screen.blit(pygame.transform.smoothscale(surface, win) if surface.get_size() != win else surface, (0, 0))
                label = font.render(status.text, True, (240, 240, 240), (0, 0, 0))
                screen.blit(label, (10, win[1] - label.get_height() - 8))
            else:
                label = font.render(status.text, True, (200, 200, 200))
                screen.blit(label, ((win[0] - label.get_width()) // 2, win[1] // 2))
            pygame.display.flip()
            if shot_dir and shown and shots < 4 and time.monotonic() - last_shot > 1.5 and status.text == "Speaking":
                os.makedirs(shot_dir, exist_ok=True)
                pygame.image.save(screen, os.path.join(shot_dir, "window-%d.png" % (shots + 1)))
                shots, last_shot = shots + 1, time.monotonic()
            clock.tick(60)
    except KeyboardInterrupt:
        request_stop()
    finally:
        pygame.quit()


def talk(*, api_key: str | None = None, avatar: str = "lama", image: str | None = None, voice: str = "female",
         language: str | None = None, model: str = "newera-s2s", instructions: str | None = None,
         idle_video: bool = False, input_file: str | None = None, window: bool = True, mute: bool = False,
         prebuffer: float = 2.0, record: str | None = None, base_url: str | None = None,
         input_device: Any = None, window_size: int = 512, screenshot_dir: str | None = None,
         max_seconds: float = 120.0, video_format: str | None = None, output: str | None = None) -> int:
    """An interactive call with the avatar: the microphone is the caller, the speaker plays the avatar, a window
    shows the face and the transcript is printed here. Returns the exit code (0 when it ended normally).

    `input_file` replays a WAV as the caller instead of the microphone, and ends the call after the reply.
    `video_format` is "h264" (the default) or "jpeg"; `output` is "original" (the default: the video has the size
    of your `image`) or "square" (512 x 512). The window takes the shape of the picture."""
    player, status = Playback(prebuffer=prebuffer), Status()
    result: dict = {"code": 1}
    holder: dict = {}
    opts = dict(api_key=api_key, avatar=avatar, image=image, voice=voice, language=language, model=model,
                instructions=instructions, idle_video=idle_video, input_file=input_file, mute=mute, record=record,
                base_url=base_url, input_device=input_device, window_size=window_size, screenshot_dir=screenshot_dir,
                max_seconds=max_seconds, video_format=video_format, output=output)
    print_lock = threading.Lock()

    def out(text: str) -> None:
        with print_lock:
            print(text, flush=True)

    def worker() -> None:
        async def main() -> None:
            holder["loop"], holder["stop"] = asyncio.get_running_loop(), asyncio.Event()
            holder["ready"].set()
            result["code"] = await _drive(opts, player, status, holder["stop"], out)
        try:
            asyncio.run(main())
        except Exception as exc:  # noqa: BLE001
            out("Stopped: %s" % exc)
            result["code"] = 1
        finally:
            holder["ready"].set()
            done.set()

    holder["ready"], done = threading.Event(), threading.Event()
    thread = threading.Thread(target=worker, name="newera-avatar", daemon=True)
    thread.start()
    holder["ready"].wait(5)

    def request_stop() -> None:
        loop = holder.get("loop")
        if loop is not None:
            loop.call_soon_threadsafe(holder["stop"].set)

    # The main thread waits on an Event, not on Thread.join(): a Ctrl+C that interrupts join() can leave
    # the thread looking finished while it still runs.
    stopping, deadline = False, None
    while not done.is_set():
        try:
            if not stopping:
                if window:
                    _run_window(done, request_stop, player, status, opts)
                else:
                    while not done.is_set():
                        done.wait(0.2)
                stopping = True
            else:
                done.wait(0.2)          # the call is ending after a stop: let it say goodbye
                if deadline is not None and time.monotonic() > deadline:
                    break
        except KeyboardInterrupt:
            stopping, deadline = True, deadline or time.monotonic() + 20
            request_stop()
        if stopping and deadline is None:
            deadline = time.monotonic() + 20
    return result["code"]
