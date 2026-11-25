"""Against the real service. Skipped unless NEWERA_API_KEY is set (and the key is enabled for the avatar)."""
import asyncio
import os

import pytest

from newera import Newera

pytestmark = pytest.mark.skipif(not os.environ.get("NEWERA_API_KEY"), reason="NEWERA_API_KEY is not set")


def test_avatar_replies_with_paired_video_and_audio(tmp_path):
    async def go():
        client = Newera()
        availability = await client.avatar.availability()
        if not availability.free:
            pytest.skip("the avatar is busy or unavailable right now")
        async with client.avatar.connect(voice="female", language="en", record=True,
                                         instructions="Answer in one short sentence.", transport="websocket") as call:
            assert call.ready.fps == 25
            await call.send_text("Hello, who are you?")
            reply = await call.wait_reply(timeout=90)
            assert reply.text
            assert len(reply.frames) > 0 and len(reply.audio) == len(reply.frames) * 1920
            call.save_wav(tmp_path / "reply.wav")
    asyncio.run(go())
