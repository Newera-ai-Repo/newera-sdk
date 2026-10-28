"""The SDK against a fake Newera server (a local WebSocket)."""
import asyncio
import json

import pytest
import websockets
from websockets.asyncio.server import serve

from newera import NeweraError, VoiceAgent
from newera.agent import _url_with_model


class FakeServer:
    """Speaks just enough of the protocol: session.created/updated, then for a
    user message makes `calls` function calls and waits for their outputs."""

    def __init__(self, calls=(), *, tools_error=None, cancel_first=False):
        self.calls, self.tools_error, self.cancel_first = list(calls), tools_error, cancel_first
        self.updates, self.outputs, self.received = [], {}, []
        self.paths = []
        self.server = None

    async def handler(self, ws):
        self.paths.append(ws.request.path)
        await ws.send(json.dumps({"type": "session.created", "session": {"id": "sess_1"}}))
        async for raw in ws:
            msg = json.loads(raw)
            self.received.append(msg)
            if msg["type"] == "session.update":
                self.updates.append(msg["session"])
                if self.tools_error:
                    await ws.send(json.dumps({"type": "error", "error": {"code": "invalid_tools", "message": self.tools_error}}))
                else:
                    await ws.send(json.dumps({"type": "session.updated", "session": msg["session"]}))
            elif msg["type"] == "response.create":
                if not any(m["type"] == "conversation.item.create" and m["item"]["type"] == "function_call_output"
                           for m in self.received):
                    await self.run_reply(ws)
                else:
                    await self.finish(ws)
            elif msg["type"] == "conversation.item.create" and msg["item"]["type"] == "function_call_output":
                self.outputs[msg["item"]["call_id"]] = msg["item"]["output"]
                if len(self.outputs) == len(self.calls) and not self.manual:
                    await self.finish(ws)

    manual = False

    async def run_reply(self, ws):
        await ws.send(json.dumps({"type": "response.created", "response": {"id": "resp_1"}}))
        for i, (name, args) in enumerate(self.calls):
            await ws.send(json.dumps({"type": "response.function_call_arguments.done", "response_id": "resp_1",
                                      "item_id": "item_%d" % i, "call_id": "call_%d" % i, "name": name,
                                      "arguments": json.dumps(args)}))
        if self.cancel_first:
            await ws.send(json.dumps({"type": "newera.function_call.cancelled", "call_id": "call_0"}))
        if not self.calls:
            await self.finish(ws)

    async def finish(self, ws):
        await ws.send(json.dumps({"type": "response.audio_transcript.delta", "delta": "done "}))
        await ws.send(json.dumps({"type": "response.audio.delta", "delta": "AAA="}))
        await ws.send(json.dumps({"type": "response.done", "response": {"id": "resp_1", "status": "completed", "usage": {"input_tokens": 5}}}))

    async def __aenter__(self):
        self.server = await serve(self.handler, "127.0.0.1", 0)
        self.url = "ws://127.0.0.1:%d" % self.server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *a):
        self.server.close()
        await self.server.wait_closed()


def make_agent(url, **kw):
    agent = VoiceAgent(api_key="nwc_test", instructions="be brief", voice="female", url=url, **kw)

    @agent.tool
    def get_slots(date: str) -> list:
        """Free times.

        Args:
            date: The day.
        """
        return ["10:00", "11:00"]

    @agent.tool(requires_confirmation=True, timeout_ms=20000)
    async def book(name: str, minutes: int = 30) -> dict:
        """Book it."""
        return {"booking": "B-1", "name": name, "minutes": minutes}

    @agent.tool
    def broken() -> str:
        """Always fails."""
        raise RuntimeError("database is down")

    return agent


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, 20))


def test_session_update_declares_tools_and_settings():
    async def go():
        async with FakeServer() as srv:
            agent = make_agent(srv.url, max_tool_steps=4)
            async with agent.connect() as call:
                assert call.session["voice"] == "female"
            s = srv.updates[0]
            assert s["instructions"] == "be brief" and s["tool_choice"] == "auto"
            by = {t["name"]: t for t in s["tools"]}
            assert by["get_slots"]["parameters"]["required"] == ["date"]
            assert by["get_slots"]["description"] == "Free times."
            assert "newera" not in by["get_slots"]
            assert by["book"]["newera"] == {"requires_confirmation": True, "timeout_ms": 20000}
            assert s["newera"] == {"max_tool_steps": 4}
    run(go())


def test_model_goes_into_the_url_not_the_session():
    async def go():
        async with FakeServer() as srv:
            async with make_agent(srv.url, model="newera-s2s-mini", max_tool_steps=4).connect():
                pass
            assert srv.paths[0] == "/?model=newera-s2s-mini"
            assert srv.updates[0]["newera"] == {"max_tool_steps": 4}
            assert "llm" not in srv.updates[0].get("newera", {})
            async with make_agent(srv.url).connect():
                pass
            assert srv.paths[1] == "/?model=newera-s2s"
            assert "newera" not in srv.updates[1]
    run(go())


def test_model_is_validated_client_side():
    with pytest.raises(ValueError) as info:
        VoiceAgent(model="newera-chat-large")
    assert "newera-s2s-mini" in str(info.value) and "newera-chat-large" in str(info.value)
    with pytest.raises(TypeError):
        VoiceAgent(llm="newera-chat-small")


def test_model_is_added_to_a_url_with_a_query_and_replaces_an_old_one():
    assert _url_with_model("wss://h/v1/realtime", "newera-s2s") == "wss://h/v1/realtime?model=newera-s2s"
    assert _url_with_model("wss://h/v1/realtime?region=ksa&model=old", "newera-s2s-mini") == \
        "wss://h/v1/realtime?region=ksa&model=newera-s2s-mini"


def test_tools_run_and_results_go_back():
    async def go():
        async with FakeServer(calls=[("get_slots", {"date": "2026-10-24"}), ("book", {"name": "Ahmed"}),
                                     ("broken", {})]) as srv:
            agent = make_agent(srv.url)
            async with agent.connect() as call:
                await call.send_text("hello")
                events = []
                async for ev in call:
                    events.append(ev)
                    if sum(e.type == "sdk.tool_result" for e in events) == 3 and any(
                            e.type == "response.done" for e in events):
                        break
            assert json.loads(srv.outputs["call_0"]) == ["10:00", "11:00"]
            assert json.loads(srv.outputs["call_1"]) == {"booking": "B-1", "name": "Ahmed", "minutes": 30}
            assert json.loads(srv.outputs["call_2"]) == {"error": "RuntimeError: database is down"}
            results = {e["name"]: e for e in events if e.type == "sdk.tool_result"}
            assert set(results) == {"get_slots", "book", "broken"}
            assert results["broken"]["error"] == "RuntimeError: database is down" and results["get_slots"]["error"] is None
    run(go())


def test_wait_response_collects_transcript_and_audio():
    async def go():
        async with FakeServer() as srv:
            async with make_agent(srv.url).connect() as call:
                await call.send_text("hello")
                reply = await call.wait_response()
            assert reply.transcript == "done " and reply.audio == b"\x00\x00" and reply.status == "completed"
            assert reply.usage == {"input_tokens": 5}
    run(go())


def test_unknown_tool_becomes_an_error_output():
    async def go():
        async with FakeServer(calls=[("nope", {})]) as srv:
            async with make_agent(srv.url).connect() as call:
                await call.send_text("x")
                await call.wait_response()
            assert "unknown tool" in json.loads(srv.outputs["call_0"])["error"]
    run(go())


def test_cancelled_call_is_not_answered():
    async def go():
        async with FakeServer(calls=[("book", {"name": "A"})], cancel_first=True) as srv:
            agent = VoiceAgent(url=srv.url)

            @agent.tool
            async def book(name: str) -> str:
                """Slow."""
                await asyncio.sleep(5)
                return "late"

            async with agent.connect() as call:
                await call.send_text("x")
                await asyncio.sleep(0.5)
            assert "call_0" not in srv.outputs
    run(go())


def test_manual_continue_sends_response_create_after_the_last_output():
    async def go():
        async with FakeServer(calls=[("get_slots", {"date": "d"}), ("get_slots", {"date": "e"})]) as srv:
            srv.manual = True
            agent = make_agent(srv.url, auto_continue=False)
            async with agent.connect() as call:
                await call.send_text("x")
                reply = await call.wait_response()
            assert srv.updates[0]["newera"] == {"tool_auto_continue": False}
            creates = [m for m in srv.received if m["type"] == "response.create"]
            assert len(creates) == 2          # the user's, then one after BOTH outputs
            assert reply.status == "completed"
    run(go())


def test_server_refusing_the_tools_raises():
    async def go():
        async with FakeServer(tools_error="tools[0].name must match ...") as srv:
            with pytest.raises(NeweraError) as info:
                async with make_agent(srv.url).connect():
                    pass
            assert info.value.code == "invalid_tools"
    run(go())


def test_duplicate_tool_name_is_refused():
    agent = VoiceAgent()

    @agent.tool
    def a() -> str: ...

    with pytest.raises(ValueError):
        agent.tool(lambda: None, name="a")


def test_audio_and_text_helpers_send_the_right_events():
    async def go():
        async with FakeServer() as srv:
            async with VoiceAgent(url=srv.url).connect() as call:
                await call.send_audio(b"\x01\x00\x02\x00")
                await call.send_text("hi", respond=False)
                await call.say("welcome")
                await call.end()
                await asyncio.sleep(0.2)
            kinds = [m["type"] for m in srv.received]
            assert kinds[1:] == ["input_audio_buffer.append", "conversation.item.create", "response.create",
                                 "session.end"]
            assert srv.received[1]["audio"] == "AQACAA=="
            assert srv.received[3]["response"] == {"instructions": "welcome"}
    run(go())


def test_voice_quality_is_sent_under_newera_when_set():
    agent = VoiceAgent(voice_quality="instant", max_tool_steps=3)
    assert agent.session_update()["session"]["newera"] == {"max_tool_steps": 3, "voice_quality": "instant"}
    assert "newera" not in VoiceAgent().session_update()["session"]


@pytest.mark.parametrize("value", ["instant", "balanced", "studio"])
def test_voice_quality_accepts_the_three_values(value):
    assert VoiceAgent(voice_quality=value).voice_quality == value


def test_voice_quality_rejects_anything_else():
    with pytest.raises(ValueError) as info:
        VoiceAgent(voice_quality="fast")
    assert "instant" in str(info.value) and "'fast'" in str(info.value)
    with pytest.raises(ValueError):
        VoiceAgent(voice_quality="")
