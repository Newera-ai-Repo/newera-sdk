"""Text to speech that streams: the first sentence arrives while the rest is still being voiced.

    pip install newera-sdk
    export NEWERA_API_KEY=nwc_live_...

    python speech_stream.py "Welcome to the clinic. Your appointment is at 10:30." --out reply.pcm
    # play it: ffplay -f s16le -ar 24000 -ac 1 reply.pcm

Pauses go in the text (`<break time="500ms"/>`); `--say Newera=نيو_إيرا` respells a word.
"""

import argparse
import asyncio
import time

from newera import Newera


async def main(args):
    client = Newera()
    pronunciations = dict(item.split("=", 1) for item in args.say) or None
    t0 = time.monotonic()
    total = 0
    with open(args.out, "wb") as out:
        async for chunk in client.audio.speech(args.text, response_format="pcm", quality=args.quality,
                                               pronunciations=pronunciations, stream=True):
            if not total:
                print("first audio after %.2f s" % (time.monotonic() - t0))
            total += len(chunk)
            out.write(chunk)                             # or hand it to a player as it arrives
    print("%.1f s of audio in %.2f s -> %s" % (total / 2 / 24000, time.monotonic() - t0, args.out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text")
    p.add_argument("--out", default="reply.pcm")
    p.add_argument("--quality", default="balanced", choices=("instant", "balanced", "studio"))
    p.add_argument("--say", action="append", default=[], metavar="WORD=RESPELLING")
    asyncio.run(main(p.parse_args()))
