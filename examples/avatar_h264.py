"""Render-audio over H.264: about 0.7 Mbit/s of video instead of about 5.6 Mbit/s of JPEG.

    pip install "newera-sdk[h264]"            # PyAV decodes and muxes H.264
    export NEWERA_API_KEY=nwc_live_...

    python avatar_h264.py speech.wav          # any 16-bit PCM WAV; saved as out.mp4

The WAV is one line of speech, streamed in 0.2 s pieces. The server answers with H.264 access units (`VideoChunk.units`,
one per frame) and the audio of exactly those frames. The script counts the video bytes it receives, then
`save_mp4()` writes the MP4 *without re-encoding* the video: the received units are muxed as they are, and only
the audio is encoded (PCM to AAC). Needs a key enabled for the render avatar.
"""

import argparse
import asyncio
import time

from newera import AvatarBusy, InsufficientQuota, Newera, NotEnabled, VideoChunk
from newera.avatar import AUDIO_RATE, read_wav_24k


async def pieces(pcm: bytes, ms: int = 200):
    step = AUDIO_RATE * ms // 1000 * 2
    for i in range(0, len(pcm), step):
        yield pcm[i:i + step]
        await asyncio.sleep(ms / 1000 / 4)          # synthesis is a little faster than real time


async def main(args):
    client = Newera()                                # NEWERA_API_KEY from the environment
    # video_format="h264" fails right here, before any call, when PyAV is not installed.
    async with client.avatar.connect(avatar="lama", mode="render-audio", video_format="h264", record=True) as call:
        print("codec", call.ready.video["codec"], "keyframe every", call.ready.video["gop_frames"], "frames")
        stats = {"bytes": 0, "frames": 0, "keyframes": 0}

        async def count():
            async for ev in call.events():
                if isinstance(ev, VideoChunk) and ev.speech:
                    stats["bytes"] += sum(len(u) for u in ev.units)
                    stats["frames"] += ev.frame_count
                    stats["keyframes"] += len(ev.keyframes)
        reader = asyncio.ensure_future(count())

        started = time.monotonic()
        line = call.start_line("line-1")
        await line.stream(pieces(read_wav_24k(args.wav)))
        print("line", await line.wait(timeout=120, strict=True))
        await asyncio.sleep(0.5)                     # the last video events follow the line's end
        seconds = stats["frames"] / call.ready.fps
        if seconds:
            print("received %d frames (%d keyframes), %.0f KB of video: %.2f Mbit/s, in %.1f s"
                  % (stats["frames"], stats["keyframes"], stats["bytes"] / 1000, stats["bytes"] * 8 / seconds / 1e6,
                     time.monotonic() - started))
        print("saved", call.save_mp4(args.out))      # the units as received; only the audio is encoded
        reader.cancel()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav", help="the speech to lip-sync (16-bit PCM WAV)")
    p.add_argument("--out", default="out.mp4")
    args = p.parse_args()
    try:
        asyncio.run(main(args))
    except AvatarBusy as exc:
        raise SystemExit("The avatar is busy. Try again in %s s." % (exc.retry_after_s or "a few"))
    except NotEnabled:
        raise SystemExit("This key is not enabled for the render avatar. Ask your Newera account manager.")
    except InsufficientQuota:
        raise SystemExit("Not enough prepaid balance for the call.")
