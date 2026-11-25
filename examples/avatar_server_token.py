"""A web page that talks to the avatar, without an API key in the page.

    pip install newera-sdk fastapi uvicorn
    export NEWERA_API_KEY=nwc_live_...
    uvicorn avatar_server_token:app --port 8000

Your server trades its API key for a one-use client secret (60 seconds). The page asks for one when the
user clicks "talk", then opens the WebSocket with it. By default the avatar is shown with LiveKit (WebRTC):
the page opens the session socket, waits for `avatar.ready`, and joins the room it names with livekit-client:

    const {client_secret} = await (await fetch("/avatar-token?mode=render-text", {method: "POST"})).json();
    const ws = new WebSocket("wss://api.newera.ai/v1/avatar?mode=render-text&transport=livekit",
                             ["realtime", "newera-client-secret." + client_secret.value]);
    ws.onopen = () => ws.send(JSON.stringify({type: "session.update", session: {avatar: "lama"}}));
    ws.onmessage = async (m) => {
      const ev = JSON.parse(m.data);
      if (ev.type === "avatar.ready") await room.connect(ev.livekit.url, ev.livekit.token);   // see the README
    };

For a page that decodes the frames itself (examples/browser/avatar_h264.html), ask for `?transport=websocket` on the
endpoint below and open the socket with `&transport=websocket`.

The model of the call was fixed when the secret was made (`?model=` on the endpoint below). The secret is
good for one connection: for the next call the page asks again. Put your own login check on this
endpoint, or anyone could start calls on your account. The page code is in the docs: https://core.newera.ai/docs#avatar
"""

from fastapi import FastAPI, HTTPException

from newera import AvatarError, Newera

app = FastAPI()
client = Newera()                                        # NEWERA_API_KEY from the environment


@app.post("/avatar-token")
async def avatar_token(model: str = "newera-s2s", mode: str = "conversational", transport: str = "livekit"):
    try:
        secret = await client.avatar.create_session(model=model, mode=mode, transport=transport)
    except ValueError as exc:                            # an unknown model, mode or transport name
        raise HTTPException(400, str(exc))
    except AvatarError as exc:                           # not enabled, no quota, transport_unavailable, avatar unavailable ...
        raise HTTPException(exc.http_status or 502, {"code": exc.code, "message": exc.message})
    return secret.as_dict()                              # {"client_secret": {"value": "avs_...", "expires_at": ...}, "model": ..., "transport": ...}
