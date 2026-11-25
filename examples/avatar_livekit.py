"""Show the avatar in a browser or an app: the LiveKit (WebRTC) transport, the default.

    pip install newera-sdk
    export NEWERA_API_KEY=nwc_live_...

    python avatar_livekit.py "Welcome to the clinic."                   # render-text: you send the words
    python avatar_livekit.py --wav speech.wav                           # render-audio: you send the voice
    python avatar_livekit.py "Hello" --image me.jpg

The call opens as usual and the avatar's video and audio are published in a LiveKit room. This script prints the
`url` and `token` of the room: give them to your front end, which joins with a LiveKit client SDK
(`await room.connect(url, token)`) and plays the avatar's tracks (see the README). The token is for one viewer and has
to be used within 10 minutes, so open the page while this script waits. Here the script streams one line and waits for
`avatar.line.done`; line events (started, done) arrive on the socket, and also in the room.

Nothing arrives as video on this socket, so there is no `save_mp4()` here: pass `transport="websocket"` for that
(see avatar_render_text.py). Needs a key enabled for the render avatar.
"""

import argparse
import asyncio

from newera import AvatarBusy, InsufficientQuota, Newera, NotEnabled, TransportUnavailable
from newera.avatar import AUDIO_RATE, read_wav_24k


async def pieces(pcm: bytes, ms: int = 200):
    step = AUDIO_RATE * ms // 1000 * 2
    for i in range(0, len(pcm), step):
        yield pcm[i:i + step]
        await asyncio.sleep(ms / 1000 / 4)          # synthesis is a little faster than real time


async def main(args):
    client = Newera()                                # NEWERA_API_KEY from the environment
    mode = "render-audio" if args.wav else "render-text"
    async with client.avatar.connect(avatar="lama", mode=mode, image=args.image) as call:       # transport="livekit" is the default
        room = call.livekit
        print("transport:", call.transport)
        print("url:      ", room.url)
        print("room:     ", room.room)
        print("token:    ", room.token)                # a real server hands this to its front end and never logs it
        print("video:    ", room.video, " audio:", room.audio)
        call.on_line_started(lambda ev: print("started", ev.line_id, "at", ev.at_ms, "ms"))
        call.on_line_done(lambda ev: print("done   ", ev.line_id, ev.status))
        if args.wait:
            print("Waiting %s s for you to join the room ..." % args.wait)
            await asyncio.sleep(args.wait)
        if args.wav:
            line = call.start_line("line-1")
            await line.stream(pieces(read_wav_24k(args.wav)))
        else:
            line = await call.speak(args.text, "line-1", voice=args.voice)
        print(line.id, await line.wait(timeout=120, strict=True))    # strict: a failed line raises


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="?", default="Hello, this is the Newera avatar.", help="the words of the line (render-text)")
    p.add_argument("--wav", help="speak this 16-bit PCM WAV instead (render-audio)")
    p.add_argument("--voice", default=None, help="the voice for the line (render-text)")
    p.add_argument("--image", help="your own face: a JPEG, PNG or WebP of one person who agreed to be animated")
    p.add_argument("--wait", type=float, default=0, help="seconds to wait before the line, to join the room first")
    args = p.parse_args()
    try:
        asyncio.run(main(args))
    except AvatarBusy as exc:
        raise SystemExit("The avatar is busy. Try again in %s s." % (exc.retry_after_s or "a few"))
    except TransportUnavailable:
        raise SystemExit('The LiveKit transport is not available right now. Retry, or connect with transport="websocket".')
    except NotEnabled:
        raise SystemExit("This key is not enabled for the render avatar. Ask your Newera account manager.")
    except InsufficientQuota:
        raise SystemExit("Not enough prepaid balance for the call.")
