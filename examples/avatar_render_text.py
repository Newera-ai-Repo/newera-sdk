"""Render-text: send the exact words, Newera voices them and the face speaks them.

    pip install newera-sdk
    export NEWERA_API_KEY=nwc_live_...

    python avatar_render_text.py "Welcome to the clinic." "How can I help you today?"
    python avatar_render_text.py "مرحبا بكم" --voice female --out hello.mp4
    python avatar_render_text.py "Hello" --image me.jpg          # your own face: the video has your photo's size
    python avatar_render_text.py "Hello" --image me.jpg --square --jpeg   # the 512 x 512 face, JPEG frames

Each argument is one line (up to 2000 characters). Nothing is rewritten. Lines play in the order you send them;
`line.wait()` says how each one ended ("completed", "interrupted" or "failed").

By default the video is H.264 and, with --image, has the size and framing of your photo (longest side at most 1280 px;
a preset face is always 512 x 512). --square gives the 512 x 512 face only, --jpeg JPEG frames instead of H.264.
"""

import argparse
import asyncio
import shutil

from newera import AvatarBusy, InsufficientQuota, Newera, NotEnabled, VideoChunk


async def main(args):
    client = Newera()
    async with client.avatar.connect(avatar="lama", mode="render-text", record=True, image=args.image,
                                     output="square" if args.square else "original",
                                     video_format="jpeg" if args.jpeg else "h264",
                                     transport="websocket") as call:     # the video chunks on the socket, to save the MP4
        r = call.ready
        print("video: %s, %s, %d x %d%s" % (r.video_format, r.output, r.width, r.height,
                                            " (%s)" % r.output_reason if r.output_reason else ""))
        async def watch():                           # each video chunk: its frames with the audio of those frames
            async for ev in call.events():
                if isinstance(ev, VideoChunk) and ev.speech:
                    print("line %s: %d frames, audio %.0f-%.0f ms" % (ev.line_id, ev.frame_count,
                                                                        ev.audio_start_ms, ev.audio_end_ms))
        reader = asyncio.ensure_future(watch())
        lines = [await call.speak(text, "line-%d" % n, voice=args.voice) for n, text in enumerate(args.text, 1)]
        for line in lines:
            print(line.id, await line.wait(timeout=120, strict=True))
        if call.video_format == "h264" or shutil.which("ffmpeg"):         # h264 is muxed with PyAV, JPEG with ffmpeg
            print("saved", call.save_mp4(args.out))
        else:
            print("saved", call.save_frames("render_frames"), "and", call.save_wav("render.wav"))
        reader.cancel()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="+", help="one argument per line of speech")
    p.add_argument("--voice", default=None, help="the voice for these lines (default: the account's)")
    p.add_argument("--out", default="render_text.mp4")
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
