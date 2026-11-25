"""The `newera` command.

    newera avatar                      # talk to the avatar: microphone, speaker and a window
    newera avatar --input-file q.wav   # replay a WAV as the caller (no microphone), end after the reply
    newera avatar --image me.jpg       # your own face: the video keeps your photo's size and framing
    newera avatar --square --jpeg      # the old output: a 512 x 512 face, JPEG frames
"""

from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="newera", description="Newera command line tools.")
    p.add_argument("--version", action="version", version="newera-sdk " + __version__)
    sub = p.add_subparsers(dest="command")
    a = sub.add_parser("avatar", help="talk to the avatar: microphone in, speaker out, the face in a window",
                       description="An interactive call with the Newera avatar. Use headphones. Ctrl+C ends the call. "
                                   "Needs: pip install \"newera-sdk[avatar]\" and NEWERA_API_KEY.")
    a.add_argument("--api-key", help="default: the NEWERA_API_KEY environment variable")
    a.add_argument("--avatar", default="lama", help="a preset face (default: lama)")
    a.add_argument("--image", help="your own face: a JPEG, PNG or WebP of one person who agreed to be animated")
    a.add_argument("--voice", default="female", choices=["male", "female"], help="the voice, independent of the face")
    a.add_argument("--language", choices=["ar", "en"], help="fix the call's language (default: detected on every turn)")
    a.add_argument("--model", default="newera-s2s", choices=["newera-s2s", "newera-s2s-mini"])
    a.add_argument("--instructions", help="what the avatar is and how it answers")
    a.add_argument("--idle-video", action="store_true", help="also show the face while it listens (blinking, small movements)")
    a.add_argument("--input-file", metavar="WAV", help="replay this WAV file as the caller instead of the microphone, "
                                                       "and end the call after the reply")
    a.add_argument("--jpeg", action="store_true", help="JPEG video frames instead of H.264 (several times the data)")
    a.add_argument("--square", action="store_true", help="only the 512 x 512 face, not your photo's size and framing")
    a.add_argument("--no-window", action="store_true", help="no window (transcript and sound only)")
    a.add_argument("--mute", action="store_true", help="do not play the avatar's voice")
    a.add_argument("--prebuffer", type=float, default=2.0, metavar="SECONDS",
                   help="video buffered before a reply starts (default 2; use 3.5 for long replies)")
    a.add_argument("--record", metavar="MP4", help="save the avatar's video and voice to this MP4 when the call ends (needs ffmpeg)")
    a.add_argument("--input-device", help="microphone: a device name or number (python -m sounddevice lists them)")
    a.add_argument("--screenshot-dir", metavar="DIR", help="save screenshots of the window to this folder")
    a.add_argument("--max-seconds", type=float, default=120.0, help="with --input-file: give up after this long (default 120)")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "avatar":
        parser.print_help()
        return 0
    from .talk import talk
    device = args.input_device
    if device is not None and device.isdigit():
        device = int(device)
    return talk(api_key=args.api_key, avatar=args.avatar, image=args.image, voice=args.voice, language=args.language,
                model=args.model, instructions=args.instructions, idle_video=args.idle_video, input_file=args.input_file,
                window=not args.no_window, mute=args.mute, prebuffer=args.prebuffer, record=args.record,
                input_device=device, screenshot_dir=args.screenshot_dir, max_seconds=args.max_seconds,
                video_format="jpeg" if args.jpeg else None, output="square" if args.square else None)


if __name__ == "__main__":
    sys.exit(main())
