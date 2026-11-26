"""Live speech to text: stream a WAV file at real-time pace, like a microphone, and print each segment as it ends.

    pip install newera-sdk
    export NEWERA_API_KEY=nwc_live_...

    python stt_stream.py clip.wav --language ar        # 16-bit mono PCM WAV; 8000, 16000, 24000 or 48000 Hz
    # convert any file: ffmpeg -i in.mp3 -ac 1 -ar 16000 clip.wav

With a microphone, call `await stream.send(pcm_chunk)` for each chunk you capture, then `await stream.commit()`.
"""

import argparse
import asyncio

from newera import Done, Newera, Segment, stt


async def main(args):
    rate = stt.wav_info(args.file)["sample_rate"]
    client = Newera()
    async with client.audio.transcribe_stream(language=args.language, sample_rate=rate) as stream:
        sender = asyncio.create_task(stream.send_wav(args.file, realtime=not args.fast))   # sends, then commits
        async for event in stream:
            if isinstance(event, stt.SpeechStarted):
                print("[%6.0f ms] speech starts" % event.audio_start_ms)
            elif isinstance(event, stt.SpeechStopped):
                print("[%6.0f ms] speech stops" % event.audio_end_ms)
            elif isinstance(event, Segment):
                print("[%6.0f - %6.0f ms] %s" % (event.start_ms, event.end_ms, event.text))
            elif isinstance(event, Done):
                print("\n%s\n(%.1f s of audio, billed %s s)" % (event.text, event.audio_ms / 1000, event.billed_seconds))
        await sender


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file")
    p.add_argument("--language", default="ar", help="ar or en")
    p.add_argument("--fast", action="store_true", help="send as fast as possible instead of at real-time pace")
    asyncio.run(main(p.parse_args()))
