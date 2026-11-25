"""Make the avatar join YOUR OWN LiveKit room (render-audio): it publishes its video and voice there.

    pip install newera-sdk livekit-api
    export NEWERA_API_KEY=nwc_live_...
    export LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=...

    python avatar_own_room.py --url wss://livekit.example.com --wav speech.wav
    python avatar_own_room.py --url wss://livekit.example.com --wav speech.wav --agent my-agent --room clinic-1

Newera must have enabled your LiveKit host for your account. This script mints a token for the avatar's participant
(identity with "avatar", kind "agent", join + publish for this one room), connects with `livekit_room=`, streams one line
of audio and waits for `avatar.line.done`. Your clients join the same room as usual and see the avatar publish there.
The token is never printed. The SDK itself does not depend on livekit: only this example imports `livekit.api`.
"""

import argparse
import asyncio
from datetime import timedelta
import os
import uuid

from newera import ByoRefused, InsufficientQuota, LiveKitJoin, Newera, NotEnabled
from newera.avatar import AUDIO_RATE, read_wav_24k


def mint_token(room: str, agent: str | None) -> str:
    from livekit import api                              # only here: pip install livekit-api

    grants = api.VideoGrants(room_join=True, room=room, can_publish=True, can_publish_data=True, can_subscribe=False)
    token = (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
             .with_identity("avatar-" + uuid.uuid4().hex[:8])
             .with_kind("agent")
             .with_grants(grants)
             .with_ttl(timedelta(minutes=10)))
    if agent:
        token = token.with_attributes({"lk.publish_on_behalf": agent})
    return token.to_jwt()


async def pieces(pcm: bytes, ms: int = 200):
    step = AUDIO_RATE * ms // 1000 * 2
    for i in range(0, len(pcm), step):
        yield pcm[i:i + step]
        await asyncio.sleep(ms / 1000 / 4)


async def main(args):
    client = Newera()                                    # NEWERA_API_KEY from the environment
    join = LiveKitJoin(args.url, mint_token(args.room, args.agent), publish_on_behalf=args.agent)
    async with client.avatar.connect(avatar="lama", mode="render-audio", image=args.image, livekit_room=join) as call:
        print("joined", args.url, "room", call.livekit.room or args.room, "as", call.livekit.identity, "byo:", call.livekit.byo)
        call.on_livekit_disconnected(lambda ev: print("the room dropped the avatar:", ev.reason))
        call.on_line_started(lambda ev: print("started", ev.line_id))
        call.on_line_done(lambda ev: print("done   ", ev.line_id, ev.status))
        line = call.start_line("line-1")
        await line.stream(pieces(read_wav_24k(args.wav)))
        print(line.id, await line.wait(timeout=120, strict=True))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", required=True, help="your LiveKit server, e.g. wss://livekit.example.com")
    p.add_argument("--wav", required=True, help="the 16-bit PCM WAV to speak")
    p.add_argument("--room", default="avatar-demo", help="the room to join (created by your LiveKit server on join)")
    p.add_argument("--agent", help="your agent's identity: the avatar is shown as this participant")
    p.add_argument("--image", help="your own face: a JPEG, PNG or WebP of one person who agreed to be animated")
    args = p.parse_args()
    try:
        asyncio.run(main(args))
    except ByoRefused as exc:
        raise SystemExit("Newera refused your room (%s): %s" % (exc.code, exc.message))
    except NotEnabled:
        raise SystemExit("This key is not enabled for the render avatar. Ask your Newera account manager.")
    except InsufficientQuota:
        raise SystemExit("Not enough prepaid balance for the call.")
