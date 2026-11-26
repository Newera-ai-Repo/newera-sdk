"""Speech to text for a file.

    pip install newera-sdk
    export NEWERA_API_KEY=nwc_live_...

    python stt_file.py meeting.wav --language ar
    python stt_file.py silence.wav --no-vad        # send the clip to the model even if it has no speech
"""

import argparse
import asyncio

from newera import Newera


async def main(args):
    client = Newera()
    result = await client.audio.transcribe(args.file, language=args.language, prompt=args.prompt, vad=not args.no_vad)
    print(result.text or "(no speech)")
    print("billed audio: %s s" % result.audio_seconds)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", help="wav, mp3, m4a, ogg, flac or webm, up to 25 MB")
    p.add_argument("--language", default="ar", help="ar or en")
    p.add_argument("--prompt", default=None, help="names or terms that help the model")
    p.add_argument("--no-vad", action="store_true", help="skip the speech check (vad=False)")
    asyncio.run(main(p.parse_args()))
