# Changelog

## 0.8.0

Streamed text lines, per-line voice options, piece timing and streaming speech. Additive: nothing changes for existing
code. Needs the Newera Core release that has `avatar.line.text.delta`, `avatar.line.piece` and `stream` on
`/v1/audio/speech`.

Added
* render-text: `call.start_text_line(line_id=None, *, voice, quality, normalize, pronunciations) -> Line` with
  `await line.append_text(text)` / `await line.stream_text(iterable_or_async_iterator)` / `await line.end()`, and
  `await call.stream_text(chunks, line_id=None, **options)` for a whole line from a stream. The avatar starts speaking
  after the first clause, while the rest of the text is still arriving. Events `avatar.line.text.delta` and
  `avatar.line.text.end`.
* `speak(...)` and the streamed lines take `quality` ("instant", "balanced", "studio"), `normalize` (default on: numbers,
  times and amounts are read as words in the language of each sentence) and `pronunciations={"word": "respelling"}`.
  `connect(..., quality=, normalize=, pronunciations=)` sets the defaults of every line (render-text only; sent as
  `voice_quality`, `normalize`, `pronunciations` of the session).
* `<break time="500ms"/>` in the text is a pause of exactly that length.
* `LinePiece(line_id, index, text, start_ms, end_ms)` (`avatar.line.piece`): a piece of a line was voiced, with its times on
  the line's own clock. In `call.events()`, `call.on_line_piece(callback)` and `line.pieces`.
* `client.audio.speech(input, *, model="newera-tts", voice, quality, normalize, pronunciations, response_format, speed,
  language, stream=False)`: the audio as bytes, or with `stream=True` an async iterator of audio chunks that arrive as they
  are voiced (`pcm` raw, `wav` with an open-length streaming header). Stopping the iteration closes the connection.
  `AvatarError("stream_interrupted")` when a stream is cut after its first chunk.
* New error classes by code: `unknown_line`, `line_closed`, `invalid_quality`, `invalid_pronunciations` are `LineError`.
* README sections "Text that streams in, line options and following the speech" and "Text to speech"; examples
  `avatar_stream_text.py` and `speech_stream.py`.

## 0.7.0

Join your own LiveKit room. Additive, nothing changes for existing code. Needs the server side that accepts
`avatar_livekit` and your LiveKit host enabled by Newera for the account.

Added
* `connect(..., livekit_room=LiveKitJoin(url, token, publish_on_behalf=None) | dict)`: the avatar joins your room with the
  identity in your token and publishes its video and audio there. Sent in the first `session.update` as `avatar_livekit`.
  Needs `transport="livekit"` (`ValueError` otherwise). The token is hidden from `repr()`, never logged, and dropped after
  it is sent.
* `call.livekit` for such a call: `LiveKitRoom` with `byo=True`, the avatar's `identity`, and an empty `token` (no viewer
  token for a Newera room). New field `LiveKitRoom.byo`.
* `ByoRefused` (an `InvalidRequest`, with `.code`) for `byo_not_enabled`, `byo_host_not_allowed`, `invalid_livekit_url`,
  `invalid_livekit_token`, `invalid_livekit`, `avatar_locked`.
* `avatar.livekit.disconnected` as `LiveKitDisconnected` (`.reason`), `call.on_livekit_disconnected(callback)` and
  `call.livekit_disconnected`; the call then ends normally (close 1000 `livekit_room_disconnected`).
* README section "Join your own LiveKit room" (minting the token with `livekit-api`); `examples/avatar_own_room.py`.
* The SDK still has no dependency on the LiveKit packages.

## 0.6.0

**Behaviour change: the default transport of the avatar is now `livekit`.**

Until 0.5.0 a call sent the avatar's video and audio as `response.video.chunk` events on the WebSocket. From 0.6.0,
`client.avatar.connect(...)` asks for `transport="livekit"` by default: the video (H.264) and the voice are published in
a LiveKit (WebRTC) room, and nothing arrives as video on the socket. `call.livekit` has what a browser or app needs to join
(`url`, `token`, `room`, `identity`, `avatar_identity`, `video`, `audio`, `data_topics`) with any LiveKit client SDK. Lines,
audio, interrupts and the events (line started and done, errors) work as before on the socket.

**To keep the old behaviour, pass `transport="websocket"`:**

```python
async with client.avatar.connect(avatar="lama", mode="render-text", record=True, transport="websocket") as call:
    ...
```

Everything that needs the video on the socket raises `WrongTransport` (an `AvatarError`) on a livekit call, with a message
that says to use `transport="websocket"`: `record=True`, `save_mp4()`, `save_h264()`, `save_frames()`, `save_wav()`,
`call.frames()`, `call.units()`, `call.audio()`, `call.request_keyframe()`, `reply.frames`, `reply.units`, `reply.audio` and
`LipSyncPlayer`. `newera avatar` and `talk()` still play the frames, so they use the websocket transport.

Added
* `connect(..., transport="livekit"|"websocket")`, default `"livekit"`; `call.transport`; `call.livekit` (`LiveKitRoom`);
  `AvatarReady.transport` and `AvatarReady.livekit`; `video_format` is `"webrtc"` on a livekit call.
* `create_session(..., transport=...)`, kept in `ClientSecret.transport` and in `as_dict()`; `connect(client_secret=...)`
  uses the secret's transport. The socket URL always says the transport.
* `TransportUnavailable` (`transport_unavailable`: connect with `transport="websocket"` or retry), `WrongTransport`,
  `avatar_render_stalled` as an `AvatarUnavailable`, `invalid_transport` as an `InvalidRequest`.
* README: a Transport section with a Python example and a browser snippet; `examples/avatar_livekit.py`.

Changed
* The examples that save an MP4 or play frames pass `transport="websocket"`; `avatar_server_token.py` takes `?transport=`
  (default `livekit`) and `?mode=`.
* With livekit the SDK does not need PyAV to open a call, and sends no `video_format`.
* The SDK has no dependency on the LiveKit packages: it only passes the room's `url` and `token` on.

## 0.5.0

H.264 video by default, `output="original"|"square"`, PyAV as a dependency, `H264Decoder`, `save_mp4` without re-encoding.
See the README release notes and the GitHub releases for older versions.
