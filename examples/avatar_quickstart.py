"""A talking face in ten lines: ask the avatar a question and save its answer as a video.

    pip install newera-sdk
    export NEWERA_API_KEY=nwc_live_...

    python avatar_quickstart.py question.wav            # a WAV of the caller's question (any rate, 16-bit PCM)
    python avatar_quickstart.py --text "What can you do?"
    python avatar_quickstart.py question.wav --out reply.mp4 --voice male

The reply is saved as reply.mp4 (JPEG frames and the reply audio, muxed with ffmpeg when it is installed;
otherwise as reply_frames/ and reply.wav). For a live call with a microphone, a speaker and a window, run
`newera avatar` instead (pip install "newera-sdk[avatar]").
"""

import argparse
import asyncio
import shutil

from newera import AvatarBusy, Newera, NotEnabled


async def main(args):
    client = Newera()                                    # NEWERA_API_KEY from the environment
    if not (await client.avatar.availability()).free:
        raise SystemExit("The avatar is in another call right now (one at a time). Try again in a few minutes.")
    async with client.avatar.connect(avatar="lama", voice=args.voice, record=True,
                                     instructions="You are a friendly assistant. Answer in one or two short sentences.") as call:
        if args.text:
            await call.send_text(args.text)
        else:
            await call.send_file(args.wav)               # streamed at real time, as a microphone would
        reply = await call.wait_reply()
        print("caller:", reply.user_text or args.text)
        print("avatar:", reply.text)
        if reply.first_video_s is not None:
            print("first reply video %.1f s after the question" % reply.first_video_s)
        print("%d frames, %.1f s of video" % (len(reply.frames), len(reply.frames) / 25))
        if shutil.which("ffmpeg"):
            print("saved", call.save_mp4(args.out))
        else:
            print("ffmpeg not found; saved", call.save_frames("reply_frames"), "and", call.save_wav("reply.wav"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav", nargs="?", help="the caller's question as a WAV file")
    p.add_argument("--text", help="type the question instead of sending a WAV")
    p.add_argument("--voice", default="female", choices=["male", "female"])
    p.add_argument("--out", default="reply.mp4")
    args = p.parse_args()
    if not (args.wav or args.text):
        p.error("give a WAV file or --text")
    try:
        asyncio.run(main(args))
    except AvatarBusy:
        raise SystemExit("The avatar is busy: it is in another call right now. Try again in a few minutes.")
    except NotEnabled:
        raise SystemExit("This key is not enabled for the avatar. Ask your Newera account manager.")
