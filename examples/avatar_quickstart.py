"""A talking face in ten lines: ask the avatar a question and save its answer as a video.

    pip install newera-sdk
    export NEWERA_API_KEY=nwc_live_...

    python avatar_quickstart.py question.wav            # a WAV of the caller's question (any rate, 16-bit PCM)
    python avatar_quickstart.py --text "What can you do?"
    python avatar_quickstart.py question.wav --out reply.mp4 --voice male
    python avatar_quickstart.py --text "Hello" --image me.jpg           # your own face, in your photo's size
    python avatar_quickstart.py --text "Hello" --image me.jpg --square --jpeg   # the 512 x 512 face, JPEG frames

The reply is saved as reply.mp4. By default the video is H.264 and, with --image, has the size and framing of your
photo (longest side at most 1280 px; a preset face is 512 x 512): it is muxed with PyAV, no ffmpeg needed. --square gives
the 512 x 512 face only; --jpeg JPEG frames (then ffmpeg makes the MP4, or the frames and a WAV are saved). For a live call with a microphone, a speaker and a window, run
`newera avatar` instead (pip install "newera-sdk[avatar]").
"""

import argparse
import asyncio
import shutil

from newera import AvatarBusy, Newera, NotEnabled


async def main(args):
    client = Newera()                                    # NEWERA_API_KEY from the environment
    async with client.avatar.connect(avatar="lama", voice=args.voice, record=True, image=args.image,
                                     output="square" if args.square else "original",
                                     video_format="jpeg" if args.jpeg else "h264",
                                     transport="websocket",      # the video chunks on the socket, to save the MP4
                                     instructions="You are a friendly assistant. Answer in one or two short sentences.") as call:
        if args.text:
            await call.send_text(args.text)
        else:
            await call.send_file(args.wav)               # streamed at real time, as a microphone would
        reply = await call.wait_reply()
        r = call.ready
        print("video: %s, %s, %d x %d%s" % (r.video_format, r.output, r.width, r.height,
                                            " (%s)" % r.output_reason if r.output_reason else ""))
        print("caller:", reply.user_text or args.text)
        print("avatar:", reply.text)
        if reply.first_video_s is not None:
            print("first reply video %.1f s after the question" % reply.first_video_s)
        frames = sum(c.frame_count for c in reply.chunks if c.speech)
        print("%d frames, %.1f s of video" % (frames, frames / 25))
        if call.video_format == "h264" or shutil.which("ffmpeg"):
            print("saved", call.save_mp4(args.out))
        else:
            print("ffmpeg not found; saved", call.save_frames("reply_frames"), "and", call.save_wav("reply.wav"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("wav", nargs="?", help="the caller's question as a WAV file")
    p.add_argument("--text", help="type the question instead of sending a WAV")
    p.add_argument("--voice", default="female", choices=["male", "female"])
    p.add_argument("--out", default="reply.mp4")
    p.add_argument("--image", help="your own face: a JPEG, PNG or WebP of one person who agreed to be animated")
    p.add_argument("--square", action="store_true", help="only the 512 x 512 face, not your photo's size and framing")
    p.add_argument("--jpeg", action="store_true", help="JPEG frames instead of H.264 (several times the data)")
    args = p.parse_args()
    if not (args.wav or args.text):
        p.error("give a WAV file or --text")
    try:
        asyncio.run(main(args))
    except AvatarBusy:
        raise SystemExit("The avatar can't take this call right now. Try again shortly.")
    except NotEnabled:
        raise SystemExit("This key is not enabled for the avatar. Ask your Newera account manager.")
