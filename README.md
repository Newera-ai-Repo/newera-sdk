# newera-sdk

[![test](https://github.com/Newera-ai-Repo/newera-sdk/actions/workflows/test.yml/badge.svg)](https://github.com/Newera-ai-Repo/newera-sdk/actions/workflows/test.yml)

Speech-to-speech calls with tool calling, in Python. Newera listens, talks (Saudi
Arabic and English) and decides when to call **your functions**; you write the
functions. And a **talking avatar**: the same voice with a face, in one command
(`newera avatar`) or a few lines of Python. See [The avatar](#the-avatar).

```python
import asyncio
from newera import VoiceAgent

agent = VoiceAgent(api_key="nwc_live_...", instructions="You book clinic appointments.", voice="male")

@agent.tool
def get_free_slots(date: str) -> list[str]:
    """Free appointment times on a day.

    Args:
        date: The day, as YYYY-MM-DD.
    """
    return ["09:00", "10:30"]

@agent.tool(requires_confirmation=True, timeout_ms=20000)
async def book_appointment(patient_name: str, date: str, time: str) -> dict:
    """Book an appointment."""
    return {"booking_id": "B-1"}

async def main():
    async with agent.connect() as call:
        await call.send_text("I need an appointment on Saturday")
        reply = await call.wait_response()
        print(reply.transcript)          # tools ran while it waited

asyncio.run(main())
```

A complete agent with a mock clinic: [`examples/clinic_agent.py`](examples/clinic_agent.py).

## Install

```
pip install newera-sdk
```

The import name is `newera`: `from newera import VoiceAgent`.

It needs Python 3.9 or newer and depends on `websockets` (and `certifi`, for TLS certificates) and PyAV (`av`, a binary
wheel with FFmpeg inside, for macOS arm64 and x86_64, Linux and Windows, Python 3.9 to 3.13), which decodes the
default H.264 video.
For the live avatar call (microphone, speaker, window) install the extra:

```
pip install "newera-sdk[avatar]"
```

H.264 video (the default, see [Video format and output](#video-format-and-output)) needs nothing more: PyAV is installed with the SDK.

## The avatar

A talking face that answers by voice, in real time: you stream the caller's audio in and get the
assistant's speech back as **video with its sound** (512 x 512, 25 frames per second, lip-synced) plus the
words of both sides. It speaks Arabic and English. Access is on request: your organisation must be enabled
for the avatar (ask your Newera account manager).

### Try it: one command

```
pip install "newera-sdk[avatar]"
export NEWERA_API_KEY=nwc_live_...
newera avatar
```

Your microphone is the caller, your speaker plays the avatar, a window shows the face and the transcript is
printed in the terminal. **Use headphones**, or the avatar hears itself and interrupts. Press Ctrl+C (or Esc, or
close the window) to end the call. On macOS the terminal asks once for microphone access.

Options: `--voice male|female`, `--language ar|en` (default: detected on every turn), `--model newera-s2s-mini`,
`--instructions "..."`, `--image me.jpg` (your own face; only of a person who agreed to be animated),
`--idle-video`, `--record call.mp4`, `--no-window`, `--mute`, and `--input-file question.wav` to replay a WAV
as the caller instead of the microphone. The same from Python: `from newera import talk; talk(voice="male")`.

### In Python

```python
import asyncio
from newera import Newera

async def main():
    client = Newera()                                   # NEWERA_API_KEY from the environment
    async with client.avatar.connect(avatar="lama", voice="female", record=True) as call:
        await call.send_file("question.wav")            # streamed at real time, as a microphone would
        reply = await call.wait_reply()
        print(reply.user_text, "->", reply.text)
        call.save_mp4("reply.mp4")                      # frames and audio muxed (needs ffmpeg)

asyncio.run(main())
```

[`examples/avatar_quickstart.py`](examples/avatar_quickstart.py) is this, complete.

`client.avatar.connect(...)` takes `avatar` (a preset face, `"lama"`), `image` (your own photo: a path, bytes or
base64), `voice` (`"male"` or `"female"`, independent of the face), `language` (`"ar"`, `"en"`, or `None` to
detect), `model` (`"newera-s2s"` or `"newera-s2s-mini"`), `idle_video` (also stream the face while it listens),
`instructions`, `record` (keep the video so `save_mp4` works), and any other realtime `session.update`
setting as a keyword (`turn_detection=...`, `tools=...`).

The call object:

* `await call.send_audio(pcm16_bytes)`: PCM16 mono 24 kHz, in real time, about 20 ms at a time (keep sending
  silence too, so turn detection works). `await call.send_file("q.wav")` streams a WAV at real time for you,
  resampling it if needed. `await call.send_text("hello")` is a typed turn.
* `async for event in call.events():` every server event, typed: `UserTranscript(text)`,
  `ReplyTextDelta(text)`, `ReplyText(text)`, `VideoChunk` (`frames`: JPEG bytes, `audio`: the PCM16 24 kHz of exactly
  those frames, `frame_index`, `start_ms`, `fps`), `Interrupted` (the caller started talking: drop everything
  queued), `Unavailable`, `ResponseDone`, `ErrorEvent`, and `OtherEvent` for the rest of the realtime protocol.
  `call.frames()` yields `(frame_index, jpeg)` and `call.audio()` the reply audio, if that is all you need.
* `await call.wait_reply()` collects the next reply: `AvatarReply(user_text, text, frames, audio, chunks,
  first_text_s, first_video_s)`.
* `call.save_mp4(path)` (needs ffmpeg), `call.save_wav(path)`, `call.save_frames(dir)`: of the recorded call.
* `await call.close()` (also at the end of `async with`): sends `session.end`; `call.summary` then holds the
  connected and billed seconds.

Play the video with the audio as the clock: show frame `frame_index / fps` seconds into the audio of the same
event; buffer about 2 seconds before a reply starts.
`newera avatar` does exactly this (`newera.talk.Playback`); the docs have the same logic for browsers.

### Render modes: your voice or your text, Newera's face

The avatar has three modes. `conversational` is the call above and keeps working unchanged. In the two render
modes the avatar does not listen or think: you give it **lines** and it speaks them. They are a separate product
(request access for the render avatar), and only the face options apply (`avatar`, `image`, `idle_video`, `record`).

* `mode="render-audio"`: you stream your own PCM16 mono 24 kHz audio, line by line, and the face lip-syncs it exactly.
* `mode="render-text"`: you send the exact text of each line (up to 2000 characters); Newera voices it. Nothing is rewritten.

```python
async with client.avatar.connect(mode="render-audio") as call:
    async with call.start_line("greeting") as line:          # your id, 1 to 64 characters; lines play in order
        async for pcm in my_tts("Welcome!"):                  # append while it is still being synthesised
            await line.append(pcm)
    print(await line.wait())                                  # "completed" | "interrupted" | "failed"

async with client.avatar.connect(mode="render-text") as call:
    line = await call.speak("Welcome to the clinic.", voice="female")
    print(await line.wait(strict=True))                       # strict: a failed line raises its typed error
```

* `call.start_line(line_id=None) -> Line` (render-audio), then `await line.append(pcm) -> bool`,
  `await line.stream(iterable_or_async_iterator)` (appends every piece, then ends) and `await line.end()`.
  `await call.send_line(pcm, line_id=None)` sends a whole line you already have.
* `await call.speak(text, line_id=None, *, voice=None) -> Line` (render-text).
* `await line.wait_started()`, `await line.wait(timeout=None, strict=False) -> status`, `line.status`, `line.error`.
  Or callbacks for every line: `call.on_line_started(cb)` and `call.on_line_done(cb)` receive `LineStarted(line_id, at_ms)`
  and `LineDone(line_id, status)`; the same two arrive in `call.events()`.
* `await call.interrupt()`: stop now. Audio not yet sent is dropped here (`line.append()` returns `False`, `line.stream()`
  stops), the server drops what is queued, you get `Interrupted`, and every open line ends `LineDone("interrupted")`.
  Lines started afterwards play normally. It works in conversational mode too.

You must read `call.events()` (even to ignore it): the video chunks queue up until you do.

**Video with timestamps.** Each `VideoChunk` has `frames` (JPEG), `audio` (PCM16 24 kHz of exactly those frames),
`audio_start_ms` / `audio_end_ms` (the audio's place on one stream clock), `line_id`, and
`frame_times_ms()` (frame `i` is due at `audio_start_ms + i * 1000 / fps`).

**A reference player.** `LipSyncPlayer` keeps lip sync with no dependencies: you give it two functions.

```python
from newera import LipSyncPlayer

player = LipSyncPlayer(play_audio=speaker.write, show_frame=window.show, flush_audio=speaker.clear)
await player.play(call.events())      # audio is the clock: frames are shown by the same clock; Interrupted flushes
```

It buffers 250 ms (`buffer_ms`), writes each chunk's audio slightly before it is due, shows frame `i` at its
timestamp, moves the whole clock (never skipping audio) when a chunk arrives late, and drops everything on an
interrupt. It is a reference: adapt it to your audio device's own clock for production.

Browsers: `await client.avatar.create_session(mode="render-text")` checks your access, and the page opens
`wss://api.newera.ai/v1/avatar?mode=render-text` with the same subprotocols as before.

Examples: [`avatar_render_audio.py`](examples/avatar_render_audio.py), [`avatar_render_text.py`](examples/avatar_render_text.py).

### Video format and output

Two options, both with defaults you rarely need to change:

```python
# defaults: H.264 video, the size and framing of your photo
async with client.avatar.connect(image="me.jpg", mode="render-text", record=True) as call: ...
# the old behaviour: JPEG frames of the 512 x 512 face
async with client.avatar.connect(image="me.jpg", mode="render-text", output="square", video_format="jpeg") as call: ...
```

* `output="original"` (default): the video has the SAME size and framing as the photo you pass as `image` (a path, bytes,
  base64 or data URL). The face is animated in place with a thin soft edge, nothing else in the picture moves. The longer
  side is at most 1280 px (a 4800 x 6000 photo is served as 1024 x 1280). `output="square"`: the 512 x 512 face only.
  `original` needs your own photo: with a preset face (no `image`), or when the face cannot be placed in the photo, the
  call is served `square` and never fails for it: `call.ready.output` is what you got, `call.ready.output_reason` says
  why (`preset_face`, `no_face_found`, ...), and `call.ready.width` / `.height` is the picture size. Every `VideoChunk`
  carries `width` and `height`; `LipSyncPlayer`, `save_mp4()` and `newera avatar` handle any size.
* `video_format="h264"` (default): H.264, under 1 Mbit/s for the square and about the same for a photo-sized picture.
  `video_format="jpeg"`: JPEG frames, many times the data (a photo-sized JPEG stream is tens of Mbit/s), no decoder needed.
  The audio, timestamps, lines and interrupts are identical.
* On the command line: `newera avatar --image me.jpg`, `--square`, `--jpeg`.

H.264 details:

```python
async with client.avatar.connect(mode="render-audio", record=True) as call:
    ...                                    # VideoChunk.units: one Annex-B access unit per frame
    call.save_mp4("out.mp4")               # the received video muxed as it is, no re-encode
```

An unknown value is a `ValueError`; the server's `invalid_video_format` / `invalid_output` are an `InvalidRequest` and
`video_format_unavailable` an `AvatarUnavailable`. `create_session(video_format=..., output=...)` checks the values and
keeps them in the secret as the socket's defaults (a page may still pass `&video_format=` / `&output=` itself). If PyAV
was removed from the environment, `connect()` without `video_format` warns and uses JPEG, and `video_format="h264"` raises
`AvatarError("av_not_installed")` before any call is made.

What you get:

* `avatar.ready` has `call.ready.video` (`{"codec": "avc1.4D401F", "stream": "annexb", "gop_frames": 50, ...}`,
  `None` for JPEG) and `call.video_format`.
* A `VideoChunk` of an h264 call has `video_format == "h264"`, `units` (one access unit per frame, in stream
  order), `keyframes` (the indices within the chunk of the IDR units) and `frame_count`; `frames` is empty. The
  timestamps and `audio` are the same as for JPEG: frame `i` is due at `audio_start_ms + i * 1000 / fps`.
* `newera.h264.H264Decoder` turns chunks into `DecodedFrame`s (`rgb` as RGB24 bytes, `width`, `height`,
  `index`, `to_ndarray()` with numpy, `to_jpeg(quality=90)` with Pillow or else PyAV). It follows the stream's
  rules: it starts at a keyframe (units before the first one are dropped), starts clean at every later keyframe,
  counts a unit that fails to decode in `decoder.errors` and then waits for the next keyframe, and calls
  `on_lost` so you can ask for one. `decoder.reset()` forgets the picture (do it on `Interrupted`). Never skip a
  unit yourself: the server forces a keyframe on the first frame, after an interrupt, after any gap, every 50
  frames and on request, and only those are places to join.
* `await call.request_keyframe()` sends `avatar.video.keyframe`; it does nothing on a JPEG call.
* `LipSyncPlayer` plays h264 with the same clock as JPEG. Its `show_frame(frame, n)` receives a `DecodedFrame` (or
  JPEG bytes with `frame_format="jpeg"`; for a JPEG call it still receives JPEG bytes). The decode runs on a worker
  thread, in order, so the event loop is not blocked; on `Interrupted` it resets the decoder; pass
  `request_keyframe=call.request_keyframe` and it asks for a keyframe when the decoder loses the picture.
* `newera avatar` plays it in the pygame window (the window takes the shape of the picture).
* `call.save_mp4(path)` of an h264 call does not re-encode the video: the access units are muxed as received, with
  timestamps from the frame numbers (a frame that never came leaves a gap and nothing is repeated; gaps longer than a
  second are shortened), starting at the first keyframe; only the audio is encoded (PCM to AAC), with PyAV
  instead of ffmpeg. `call.save_h264(path)` writes the raw Annex-B stream, `call.save_frames(dir)` decodes to JPEG
  files. The JPEG behaviour of every `save_*` is unchanged.

Examples: [`avatar_h264.py`](examples/avatar_h264.py) (render-audio, prints the bandwidth, saves `out.mp4`), and
[`browser/avatar_h264.html`](examples/browser/avatar_h264.html), one file of plain HTML and JavaScript that decodes
the stream with the browser's WebCodecs `VideoDecoder` and plays the audio on the same clock.

### Errors

A call can be refused or ended; handle these errors, all `NeweraError`s:

| exception | when |
|---|---|
| `AvatarBusy` | the avatar can't take this call right now: try again after `.retry_after_s` seconds (may be None) |
| `AvatarUnavailable` | the face or Newera Core is temporarily unavailable (also `video_format_unavailable`: connect with `video_format="jpeg"`) |
| `NotEnabled` | the organisation is not enabled for this avatar product (conversational and render are separate) or the voice model |
| `AuthenticationError` | missing or wrong key, or a used or expired client secret |
| `InsufficientQuota` | the prepaid balance cannot cover the call |
| `RateLimited` | too many requests, or too many calls on this key right now |
| `InvalidRequest` | a bad session field or model (`invalid_avatar_image`: no usable face in the photo; `invalid_video_format`, `invalid_output`) |
| `CallTimeLimit` | the service ended the call |
| `WrongMode` | an event of another mode (audio line in render-text, a line in conversational) |
| `LineError` | a refused line: bad or duplicate `line_id`, too many queued lines, line too long, bad audio or text |

Each has `.code`, `.message`, `.close_code` and `.retry_after_s`. A refused call is never billed.

### A browser page

An API key must not sit in a web page. Your server trades it for a one-use client secret that is good for 60
seconds, and the page connects with that:

```python
secret = await client.avatar.create_session(model="newera-s2s")      # ClientSecret(value="avs_...", expires_at=...)
```

[`examples/avatar_server_token.py`](examples/avatar_server_token.py) is a FastAPI endpoint that returns it. The page
opens `wss://api.newera.ai/v1/avatar` with the subprotocols `["realtime", "newera-client-secret." + secret]`
(see the docs for the player code). Python can connect with a secret too:
`client.avatar.connect(client_secret=secret)`.

Only send a photo (`image=`) of a person who agreed to be animated, and tell them what it is for.

## Tools

`@agent.tool` turns a function into a tool. The schema comes from the function:

| from | becomes |
|---|---|
| the name | tool name (`name=` overrides) |
| docstring summary (text before a blank line or `Args:`) | description (`description=` overrides) |
| `Args:` section, Google style (`name: text` or `name (type): text`) | each parameter's description |
| `str`, `int`, `float`, `bool` | `string`, `integer`, `number`, `boolean` |
| `Literal["a", "b"]`, `Enum` | `enum` |
| `Optional[X]`, `X \| None`, parameters with a default | optional (not in `required`) |
| `list[X]`, `dict`, `Union[...]`, `Any` | `array`, `object`, `anyOf`, any value |

```python
@agent.tool(requires_confirmation=False, timeout_ms=15000)
def my_tool(...): ...
```

Both sync and async functions work (sync ones run in a thread, so a slow call
never blocks the audio). The return value is JSON encoded and given to the
model (dates, decimals, enums and dataclasses are converted). **If the function
raises, the model receives `{"error": "ValueError: ..."}`** and tells the caller;
the call carries on.

### What Newera does around your function

* **Validation.** The model's arguments are checked against the schema before
  your function is called. A wrong or missing argument never reaches you: the
  model is told and asks the caller.
* **Confirmation.** With `requires_confirmation=True`, your function is called
  only after the assistant read the details back and the caller said yes. A
  different argument is a new question. Declare every function that changes
  something this way (booking, paying, cancelling, sending).
* **Speech while you work.** If the model calls a tool without saying anything,
  Newera speaks a short line ("Sure, one moment while I check...") and, if your
  function takes more than 3 and 8 seconds, a progress line. You write nothing for this.
* **Timeouts.** `timeout_ms` (1000 to 60000, default 15000) is how long Newera
  waits for the result; after that the model tells the caller it is taking longer
  and offers to retry.
* **Barge-in.** If the caller interrupts while a read-only tool runs, it is
  cancelled (an `asyncio` cancel for async functions; a sync function in a thread
  cannot be stopped, but its result is ignored). A tool with `requires_confirmation`
  is never cancelled; its result is reported truthfully on the next turn.
* **Limits.** At most 32 tools; each name `^[a-zA-Z0-9_-]{1,64}$`; each schema
  16 KB, all of them 64 KB; a result is cut at 32 KB; at most 6 tool steps per reply
  (`VoiceAgent(max_tool_steps=...)`, up to 10). When the limit is reached the
  assistant tells the caller what is done and asks whether to continue.

Treat what your functions return as what the *model* will read to the caller,
and treat what comes back from a tool as data, never as instructions.

## Models

Choose the model when you create the agent. It is sent when the call connects
(`wss://api.newera.ai/v1/realtime?model=...`).

| model | use it for |
|---|---|
| `newera-s2s` (default) | conversations that need deeper reasoning: multi-step troubleshooting, consultative sales, bookings that balance several constraints, workflows that chain many tools, long calls with a lot of context, nuanced policy questions. |
| `newera-s2s-mini` | fast and efficient for high-volume, well-defined conversations: customer support and FAQs, appointment booking and rescheduling, order and delivery status, reservations, lead qualification and intake, surveys and feedback calls, reminders and confirmations, replacing IVR menus, and tool-driven flows such as looking up a record or opening a ticket. |

```python
agent = VoiceAgent(api_key="nwc_live_...", model="newera-s2s-mini")
```

An unknown model name raises `ValueError` before anything connects.

### Billing

Calls are billed per connected second. The clock starts when the connection
opens and stops when it closes. The total is rounded up to the next whole second,
with a minimum of 30 seconds per call.

* `newera-s2s`: 0.20 SAR per minute (0.0033333333 SAR per second)
* `newera-s2s-mini`: 0.10 SAR per minute (0.0016666667 SAR per second)

Examples: a 5-second call is billed as 30 seconds, which is 0.10 SAR on
`newera-s2s` or 0.05 SAR on `newera-s2s-mini`. A call of 2 minutes 3.4 seconds is
billed as 124 seconds, which is 0.41 SAR on `newera-s2s` or 0.21 SAR on
`newera-s2s-mini`.

Each API key can run up to 5 calls at the same time.

## Voice quality

`voice_quality` sets how the assistant sounds and how quickly it starts each reply.
It can also be changed during a call, and applies from the next reply.

| value | use it for |
|---|---|
| `instant` | the fastest start to every reply, for snappy back-and-forth: quick confirmations, short questions and answers, menu-style flows, high-volume reminder and notification calls |
| `balanced` | a natural voice with a quick start; the everyday choice for most support, booking and service calls |
| `studio` (default) | the most natural, polished voice, for conversations where how you sound matters most: sales, concierge and premium service, longer explanations |

```python
agent = VoiceAgent(api_key="nwc_live_...", voice_quality="balanced")
```

An unknown value raises `ValueError` before anything connects.

## The call

```python
async with agent.connect() as call:
    await call.send_audio(pcm16_bytes)      # 16-bit mono, 24 kHz by default (16 kHz: session={"newera": {"input_audio_sample_rate": 16000}}); real time, 20-100 ms chunks
    await call.send_text("hello")           # a typed turn (add respond=False to wait)
    await call.say("Welcome to the clinic") # the assistant speaks first
    async for event in call:                # every server event
        ...
    await call.end()                        # session.summary, then the call closes
```

`event.type` is the protocol's event name (protocol reference):
`response.audio.delta` (`event.audio` gives the PCM16 24 kHz bytes),
`response.audio_transcript.delta`, `input_audio_buffer.speech_started` (stop your
playback), `response.done`, `error`, and so on. The SDK adds one event,
`sdk.tool_result` (`name`, `arguments`, `output`, `error`, `ms`), after each of your
functions ran.

`await call.wait_response()` collects events up to the next `response.done` and
returns a `Reply(transcript, audio, status, usage, events)`; handy for text-only
scripts and tests.

`VoiceAgent(...)` takes `api_key`, `instructions`, `voice`, `model` (see
[Models](#models)), `tool_choice` (`"auto"`, `"none"`, `"required"`, or
`{"type": "function", "name": ...}`), `max_tool_steps`, `auto_continue`,
`voice_quality` (`"studio"` (default), `"balanced"` or `"instant"`, see [Voice quality](#voice-quality)),
`session` (extra `session.update` fields),
`url` and `headers`.

`auto_continue=False` is the OpenAI-style flow: after your outputs the SDK sends
`response.create` itself and the server does not continue on its own.

Browsers must not hold an API key: this SDK is for servers. Give browser clients
their own short-lived credentials instead.

## Tests

```
pip install -e '.[test]' && pytest
```

The tests run the SDK against a fake WebSocket server; no network, no key. `tests/test_avatar_e2e.py` talks to the
real avatar and is skipped unless `NEWERA_API_KEY` is set (use a key enabled for the avatar).

## Release notes

* 0.5.0: calls use H.264 video by default (about 0.7 instead of 5.6 Mbit/s) and the new `output="original"` by default: the
  video has the size and framing of the photo you pass as `image`. `connect(..., output="original"|"square",
  video_format="h264"|"jpeg")`, the same options on `create_session()`, `call.ready.output` / `output_reason`, the window
  of `newera avatar` and the examples take any picture size, `newera avatar --square --jpeg --image`. PyAV is now a
  dependency of the SDK (the `[h264]` extra stays as an empty alias). Also: `newera.h264.H264Decoder`, h264 in
  `LipSyncPlayer`, `call.request_keyframe()`, `save_mp4` without re-encoding, `save_h264`. Upgrade note: to keep the
  previous 512 x 512 JPEG calls pass `output="square", video_format="jpeg"`.
* 0.4.0: render modes (`mode="render-audio"` and `"render-text"`) with lines, `interrupt()`, line events, video
  timestamps (`audio_start_ms`, `audio_end_ms`, `line_id`), `LipSyncPlayer`, `retry_after_s`, new typed errors. Additive: conversational calls are unchanged.
* 0.3.1: documentation and message wording for the avatar. No change in behaviour.
* 0.3.0: the talking avatar (`client.avatar`, `newera avatar`).

## Releasing

1. Bump `version` in `pyproject.toml` (and `__version__` in `src/newera/__init__.py`).
2. Push to `main`. The `publish` workflow runs the tests. If that version is not yet on
   PyPI, it builds the package, uploads it through trusted publishing, and creates the
   `vX.Y.Z` tag and a GitHub Release. No API token is stored in the repository.
3. A push with an unchanged version only runs the tests.

## License

Apache License 2.0. See [LICENSE](LICENSE).
