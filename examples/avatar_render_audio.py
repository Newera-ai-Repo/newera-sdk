"""Render-audio: you bring the voice, the avatar brings the face.

    pip install newera-sdk
    export NEWERA_API_KEY=nwc_live_...

    python avatar_render_audio.py speech.wav                  # any 16-bit PCM WAV; saved as render_audio.mp4
    python avatar_render_audio.py a.wav b.wav --out talk.mp4  # two lines, played in order
    python avatar_render_audio.py speech.wav --image me.jpg   # your own face: the video has your photo's size
    python avatar_render_audio.py speech.wav --image me.jpg --square --jpeg   # the 512 x 512 face, JPEG frames

Each WAV is one line. It is streamed in 0.2 s pieces, the way a streaming text-to-speech would hand you audio
while it is still being synthesised (replace `pieces()` with your own generator). The face lip-syncs your audio
exactly; the video events carry that same audio with timestamps (`audio_start_ms` / `audio_end_ms`).
Needs a key enabled for the render avatar (ask your Newera account manager). By default the video is H.264 and,
with --image, has the size and framing of your photo (longest side at most 1280 px; a preset face is always
512 x 512): the MP4 is written with PyAV and needs no ffmpeg. --square gives the 512 x 512 face only, --jpeg JPEG
frames instead of H.264 (the MP4 then needs ffmpeg; without it frames and WAV are saved).
"""

import argparse
import asyncio
import shutil

from newera import AvatarBusy, Newera, NotEnabled, InsufficientQuota
from newera.avatar import AUDIO_RATE, read_wav_24k


async def pieces(pcm: bytes, ms: int = 200):
    step = AUDIO_RATE * ms // 1000 * 2
    for i in range(0, len(pcm), step):
        yield pcm[i:i + step]
        await asyncio.sleep(ms / 1000 / 4)          # synthesis is a little faster than real time


async def main(args):
    client = Newera()                                # NEWERA_API_KEY from the environment
    async with client.avatar.connect(avatar="lama", mode="render-audio", record=True, image=args.image,
                                     output="square" if args.square else "original",
                                     video_format="jpeg" if args.jpeg else "h264",
                                     transport="websocket") as call:     # the video chunks on the socket, to save the MP4
        r = call.ready
        print("video: %s, %s, %d x %d%s" % (r.video_format, r.output, r.width, r.height,
                                            " (%s)" % r.output_reason if r.output_reason else ""))
        call.on_line_started(lambda ev: print("started", ev.line_id, "at", ev.at_ms, "ms"))
        call.on_line_done(lambda ev: print("done   ", ev.line_id, ev.status))

        async def drain():                           # video chunks must be read, or they pile up in memory
            async for ev in call.events():
                pass                                 # a real app plays them: see newera.LipSyncPlayer
        reader = asyncio.ensure_future(drain())

        lines = []
        for n, wav in enumerate(args.wav, 1):
            line = call.start_line("line-%d" % n)
            await line.stream(pieces(read_wav_24k(wav)))      # appends the pieces, then ends the line
            lines.append(line)
        for line in lines:
            print(line.id, await line.wait(timeout=120, strict=True))   # strict: a failed line raises
        if call.video_format == "h264" or shutil.which("ffmpeg"):         # h264 is muxed with PyAV, JPEG with ffmpeg
            print("saved", call.save_mp4(args.out))
        else:
            print("saved", call.save_frames("render_frames"), "and", call.save_wav("render.wav"))
        reader.cancel()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav", nargs="+", help="one WAV per line of speech")
    p.add_argument("--out", default="render_audio.mp4")
    p.add_argument("--image", help="your own face: a JPEG, PNG or WebP of one person who agreed to be animated")
    p.add_argument("--square", action="store_true", help="only the 512 x 512 face, not your photo's size and framing")
    p.add_argument("--jpeg", action="store_true", help="JPEG frames instead of H.264 (several times the data)")
    args = p.parse_args()
    try:
        asyncio.run(main(args))
    except AvatarBusy as exc:
        raise SystemExit("The avatar is busy. Try again in %s s." % (exc.retry_after_s or "a few"))
    except NotEnabled:
        raise SystemExit("This key is not enabled for the render avatar. Ask your Newera account manager.")
    except InsufficientQuota:
        raise SystemExit("Not enough prepaid balance for the call.")
