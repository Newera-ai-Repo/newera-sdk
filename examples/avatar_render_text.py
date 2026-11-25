"""Render-text: send the exact words, Newera voices them and the face speaks them.

    pip install newera-sdk
    export NEWERA_API_KEY=nwc_live_...

    python avatar_render_text.py "Welcome to the clinic." "How can I help you today?"
    python avatar_render_text.py "مرحبا بكم" --voice female --out hello.mp4

Each argument is one line (up to 2000 characters). Nothing is rewritten. Lines play in the order you send them;
`line.wait()` says how each one ended ("completed", "interrupted" or "failed").
"""

import argparse
import asyncio
import shutil

from newera import AvatarBusy, InsufficientQuota, Newera, NotEnabled, VideoChunk


async def main(args):
    client = Newera()
    async with client.avatar.connect(avatar="lama", mode="render-text", record=True) as call:
        async def watch():                           # each video chunk: its frames with the audio of those frames
            async for ev in call.events():
                if isinstance(ev, VideoChunk) and ev.speech:
                    print("line %s: %d frames, audio %.0f-%.0f ms" % (ev.line_id, len(ev.frames),
                                                                        ev.audio_start_ms, ev.audio_end_ms))
        reader = asyncio.ensure_future(watch())
        lines = [await call.speak(text, "line-%d" % n, voice=args.voice) for n, text in enumerate(args.text, 1)]
        for line in lines:
            print(line.id, await line.wait(timeout=120, strict=True))
        if shutil.which("ffmpeg"):
            print("saved", call.save_mp4(args.out))
        else:
            print("saved", call.save_frames("render_frames"), "and", call.save_wav("render.wav"))
        reader.cancel()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="+", help="one argument per line of speech")
    p.add_argument("--voice", default=None, help="the voice for these lines (default: the account's)")
    p.add_argument("--out", default="render_text.mp4")
    args = p.parse_args()
    try:
        asyncio.run(main(args))
    except AvatarBusy as exc:
        raise SystemExit("The avatar is busy. Try again in %s s." % (exc.retry_after_s or "a few"))
    except NotEnabled:
        raise SystemExit("This key is not enabled for the render avatar. Ask your Newera account manager.")
    except InsufficientQuota:
        raise SystemExit("Not enough prepaid balance for the call.")
