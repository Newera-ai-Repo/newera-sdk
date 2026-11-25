# Changelog

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
