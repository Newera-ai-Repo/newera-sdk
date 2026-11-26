"""Render-text with text that streams in: the avatar starts speaking after the first clause while the rest is still being
written, and every voiced piece is reported with its times so a transcript can follow the speech.

    pip install newera-sdk
    export NEWERA_API_KEY=nwc_live_...

    python avatar_stream_text.py
    python avatar_stream_text.py --quality instant --pause 700

The "language model" here is a fake that types a sentence word by word; put your own stream in `llm_deltas()`.
"""

import argparse
import asyncio

from newera import AvatarBusy, InsufficientQuota, LineStarted, Newera, NotEnabled

ANSWER = ("Welcome to the clinic. Your appointment is at 10:30 tomorrow. {pause}"
          "Please bring your ID card, and arrive ten minutes early.")


async def llm_deltas(text):
    for word in text.split(" "):
        await asyncio.sleep(0.15)                       # a model's pace
        yield word + " "


async def main(args):
    client = Newera()
    pause = '<break time="%dms"/>' % args.pause if args.pause else ""
    async with client.avatar.connect(avatar="lama", mode="render-text", quality=args.quality) as call:
        started = {}

        @call.on_line_started
        def _started(ev: LineStarted):
            started[ev.line_id] = ev.at_ms

        @call.on_line_piece
        def _piece(p):                                  # show p.text when the clock reaches started_at + p.start_ms
            print("line %s piece %d  %6.0f - %6.0f ms (line clock)  %s" % (p.line_id, p.index, p.start_ms, p.end_ms, p.text))

        reader = asyncio.ensure_future(_drain(call))
        line = await call.stream_text(llm_deltas(ANSWER.format(pause=pause)), "answer-1",
                                      pronunciations={"clinic": "klinik"})
        print("status:", await line.wait(timeout=120, strict=True))
        print("stream clock at the first sound:", started.get("answer-1"), "ms")
        reader.cancel()


async def _drain(call):
    async for _ in call.events():                      # the video chunks queue up until they are read
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quality", default="balanced", choices=("instant", "balanced", "studio"))
    p.add_argument("--pause", type=int, default=600, help="milliseconds of pause in the middle (0 for none)")
    args = p.parse_args()
    try:
        asyncio.run(main(args))
    except AvatarBusy as exc:
        raise SystemExit("The avatar is busy. Try again in %s s." % (exc.retry_after_s or "a few"))
    except NotEnabled:
        raise SystemExit("This key is not enabled for the render avatar. Ask your Newera account manager.")
    except InsufficientQuota:
        raise SystemExit("Not enough prepaid balance for the call.")
