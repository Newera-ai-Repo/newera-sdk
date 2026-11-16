"""A web page that talks to the avatar, without an API key in the page.

    pip install newera-sdk fastapi uvicorn
    export NEWERA_API_KEY=nwc_live_...
    uvicorn avatar_server_token:app --port 8000

Your server trades its API key for a one-use client secret (60 seconds). The page asks for one when the
user clicks "talk", then opens the WebSocket with it:

    const {client_secret} = await (await fetch("/avatar-token", {method: "POST"})).json();
    const ws = new WebSocket("wss://api.newera.ai/v1/avatar",
                             ["realtime", "newera-client-secret." + client_secret.value]);
    ws.onopen = () => ws.send(JSON.stringify({type: "session.update", session: {avatar: "lama"}}));

The model of the call was fixed when the secret was made (`?model=` on the endpoint below). The secret is
good for one connection: for the next call the page asks again. Put your own login check on this
endpoint, or anyone could start calls on your account. The page code that plays the video is in the
docs: https://core.newera.ai/docs#avatar
"""

from fastapi import FastAPI, HTTPException

from newera import AvatarError, Newera

app = FastAPI()
client = Newera()                                        # NEWERA_API_KEY from the environment


@app.post("/avatar-token")
async def avatar_token(model: str = "newera-s2s"):
    try:
        secret = await client.avatar.create_session(model=model)
    except ValueError as exc:                            # an unknown model name
        raise HTTPException(400, str(exc))
    except AvatarError as exc:                           # not enabled, no quota, avatar unavailable ...
        raise HTTPException(exc.http_status or 502, {"code": exc.code, "message": exc.message})
    return secret.as_dict()                              # {"client_secret": {"value": "avs_...", "expires_at": ...}, "model": ...}
