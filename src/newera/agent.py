"""VoiceAgent: a Newera S2S call with Python functions as tools.

    agent = VoiceAgent(api_key="nwc_live_...", instructions="You book clinic appointments.")

    @agent.tool
    def get_slots(date: str) -> list[str]:
        '''Free appointment times on a date.

        Args:
            date: The day, as YYYY-MM-DD.
        '''

    async with agent.connect() as call:
        await call.send_text("I need an appointment on Saturday")
        async for event in call:
            ...

The SDK declares the tools in session.update, runs each function the model
calls (sync functions in a thread, async ones on the loop), sends its result
back as a function_call_output, and the server carries on speaking. Results
are JSON encoded; an exception becomes {"error": "..."}, which the model
tells the caller about. For a tool declared requires_confirmation the SERVER
makes the model ask the caller first: your function is only ever called after
the caller said yes.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import inspect
import json
import time
from typing import Any, AsyncIterator, Callable

import websockets

from .schema import coerce_arguments, function_schema

DEFAULT_URL = "wss://api.newera.ai/v1/realtime"
MAX_OUTPUT_CHARS = 30000


class NeweraError(Exception):
    """An `error` event from the server (code and message as sent)."""

    def __init__(self, code: str, message: str):
        super().__init__("%s: %s" % (code, message))
        self.code, self.message = code, message


class Event(dict):
    """A server event: a dict with `.type` and a few helpers."""

    @property
    def type(self) -> str:
        return self.get("type", "")

    @property
    def audio(self) -> bytes | None:
        """PCM16 mono 24 kHz bytes of a response.audio.delta, else None."""
        if self.type == "response.audio.delta":
            return base64.b64decode(self["delta"])
        return None


@dataclasses.dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable
    requires_confirmation: bool = False
    timeout_ms: int = 15000

    def definition(self) -> dict:
        d = {"type": "function", "name": self.name, "description": self.description,
             "parameters": self.parameters}
        if self.requires_confirmation or self.timeout_ms != 15000:
            d["newera"] = {"requires_confirmation": self.requires_confirmation, "timeout_ms": self.timeout_ms}
        return d


@dataclasses.dataclass
class Reply:
    """One finished response, from Call.wait_response()."""
    transcript: str = ""
    audio: bytes = b""
    status: str = ""
    usage: dict = dataclasses.field(default_factory=dict)
    events: list = dataclasses.field(default_factory=list)


class VoiceAgent:
    def __init__(self, *, api_key: str | None = None, instructions: str | None = None, voice: str = "male",
                 url: str = DEFAULT_URL, tool_choice: Any = "auto", max_tool_steps: int | None = None,
                 auto_continue: bool = True, session: dict | None = None, headers: dict | None = None,
                 llm: str | None = None):
        self.api_key, self.instructions, self.voice, self.url = api_key, instructions, voice, url
        # The model behind the call: "newera-chat-large" (the server's default) or "newera-chat-small".
        # Sent as session.newera.llm; the server validates it.
        self.llm = llm
        self.tool_choice, self.max_tool_steps, self.auto_continue = tool_choice, max_tool_steps, auto_continue
        self.session = session or {}
        self.headers = headers or {}
        self.tools: dict[str, ToolSpec] = {}

    # --------------------------------------------------------------------- tools

    def tool(self, fn: Callable | None = None, *, name: str | None = None, description: str | None = None,
             requires_confirmation: bool = False, timeout_ms: int = 15000):
        """Register a function as a tool. Bare (`@agent.tool`) or with options:
        `@agent.tool(requires_confirmation=True, timeout_ms=20000)`. The
        function itself is returned unchanged."""
        def register(f: Callable) -> Callable:
            summary, params = function_schema(f)
            spec = ToolSpec(name or f.__name__, description or summary or f.__name__, params, f,
                            requires_confirmation, timeout_ms)
            if spec.name in self.tools:
                raise ValueError("A tool named %r is already registered." % spec.name)
            self.tools[spec.name] = spec
            return f
        return register(fn) if fn is not None else register

    def session_update(self) -> dict:
        s: dict = {"voice": self.voice, **self.session}
        if self.instructions:
            s["instructions"] = self.instructions
        if self.tools:
            s["tools"] = [t.definition() for t in self.tools.values()]
            s["tool_choice"] = self.tool_choice
        newera = dict(s.get("newera") or {})
        if self.max_tool_steps is not None:
            newera["max_tool_steps"] = self.max_tool_steps
        if not self.auto_continue:
            newera["tool_auto_continue"] = False
        if self.llm is not None:
            newera["llm"] = self.llm
        if newera:
            s["newera"] = newera
        return {"type": "session.update", "session": s}

    def connect(self) -> "_Connect":
        return _Connect(self)


class _Connect:
    def __init__(self, agent: VoiceAgent):
        self.agent, self._cm, self.call = agent, None, None

    async def __aenter__(self) -> "Call":
        headers = dict(self.agent.headers)
        if self.agent.api_key:
            headers["Authorization"] = "Bearer " + self.agent.api_key
        self._cm = websockets.connect(self.agent.url, additional_headers=headers, max_size=None)
        ws = await self._cm.__aenter__()
        self.call = Call(self.agent, ws)
        try:
            await self.call._start()
        except BaseException:
            await self.__aexit__(None, None, None)
            raise
        return self.call

    async def __aexit__(self, *exc) -> None:
        if self.call is not None:
            await self.call._stop()
        await self._cm.__aexit__(*exc)


class Call:
    """An open call. Iterate it for events; send audio and text with the methods."""

    def __init__(self, agent: VoiceAgent, ws):
        self.agent, self._ws = agent, ws
        self._queue: asyncio.Queue = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._tasks: dict[str, asyncio.Task] = {}
        self._outstanding: set[str] = set()
        self.session: dict = {}

    # ----------------------------------------------------------------- lifecycle

    async def _start(self) -> None:
        first = json.loads(await self._ws.recv())
        if first.get("type") == "error":
            raise NeweraError(first["error"].get("code", "error"), first["error"].get("message", ""))
        self.session = first.get("session", {})
        await self._send(self.agent.session_update())
        # session.updated, or an error saying why the tools were refused.
        while True:
            ev = json.loads(await asyncio.wait_for(self._ws.recv(), 15))
            if ev["type"] == "session.updated":
                self.session = ev.get("session", {})
                break
            if ev["type"] == "error":
                raise NeweraError(ev["error"].get("code", "error"), ev["error"].get("message", ""))
        self._reader = asyncio.create_task(self._read())

    async def _stop(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        if self._reader:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader

    async def _send(self, event: dict) -> None:
        await self._ws.send(json.dumps(event, ensure_ascii=False))

    # --------------------------------------------------------------------- send

    async def send_audio(self, pcm: bytes) -> None:
        """16-bit mono PCM at the session's input rate (24 kHz by default, as in OpenAI's API;
        `VoiceAgent(session={"newera": {"input_audio_sample_rate": 16000}})` for 16 kHz)."""
        await self._send({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()})

    async def send_text(self, text: str, *, respond: bool = True) -> None:
        await self._send({"type": "conversation.item.create", "item": {
            "type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}})
        if respond:
            await self._send({"type": "response.create"})

    async def say(self, instructions: str) -> None:
        """Make the assistant say something (a greeting, for example)."""
        await self._send({"type": "response.create", "response": {"instructions": instructions}})

    async def commit(self, *, respond: bool = True) -> None:
        """Manual turns (`turn_detection: null`): the audio sent so far is the caller's turn. As in
        OpenAI's API a commit alone starts no reply, so this also sends `response.create`;
        `respond=False` only adds the audio to the conversation."""
        await self._send({"type": "input_audio_buffer.commit"})
        if respond:
            await self._send({"type": "response.create"})

    async def cancel(self) -> None:
        await self._send({"type": "response.cancel"})

    async def update(self, **session) -> None:
        await self._send({"type": "session.update", "session": session})

    async def end(self) -> None:
        """Finish the call; the server sends session.summary, then closes."""
        await self._send({"type": "session.end"})

    # ------------------------------------------------------------------- receive

    def __aiter__(self) -> AsyncIterator[Event]:
        return self._events()

    async def _events(self) -> AsyncIterator[Event]:
        while True:
            item = await self._queue.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def wait_response(self, timeout: float = 120.0) -> Reply:
        """Events up to the next finished response (`response.done`), with its
        transcript and audio collected. Tool calls are handled meanwhile."""
        reply = Reply()

        async def collect() -> None:
            async for ev in self:
                reply.events.append(ev)
                if ev.type == "response.audio_transcript.delta":
                    reply.transcript += ev.get("delta", "")
                elif ev.type == "response.audio.delta":
                    reply.audio += ev.audio or b""
                elif ev.type == "response.done":
                    r = ev["response"]
                    reply.status, reply.usage = r.get("status", ""), r.get("usage", {})
                    return
                elif ev.type == "error":
                    raise NeweraError(ev["error"].get("code", "error"), ev["error"].get("message", ""))

        await asyncio.wait_for(collect(), timeout)
        return reply

    async def _read(self) -> None:
        try:
            async for raw in self._ws:
                ev = Event(json.loads(raw))
                t = ev.type
                if t == "response.function_call_arguments.done":
                    self._outstanding.add(ev["call_id"])
                    self._tasks[ev["call_id"]] = asyncio.create_task(self._run_tool(ev))
                elif t == "newera.function_call.cancelled":
                    task = self._tasks.pop(ev.get("call_id"), None)
                    self._outstanding.discard(ev.get("call_id"))
                    if task:
                        task.cancel()
                await self._queue.put(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - connection closed or broken
            if not isinstance(exc, websockets.ConnectionClosed):
                await self._queue.put(exc)
        finally:
            await self._queue.put(None)

    # --------------------------------------------------------------------- tools

    async def _run_tool(self, ev: Event) -> None:
        call_id, name = ev["call_id"], ev["name"]
        started = time.perf_counter()
        error = None
        spec = self.agent.tools.get(name)
        try:
            if spec is None:
                raise LookupError("unknown tool %r" % name)
            args = coerce_arguments(spec.fn, json.loads(ev.get("arguments") or "{}"))
            if inspect.iscoroutinefunction(spec.fn):
                result = await spec.fn(**args)
            else:
                result = await asyncio.to_thread(spec.fn, **args)
            output = json.dumps(result, default=_default, ensure_ascii=False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - the model hears about it, the call goes on
            error = "%s: %s" % (type(exc).__name__, exc)
            output = json.dumps({"error": error}, ensure_ascii=False)
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS]       # the server truncates at 32 KB and says so; stay under it
        self._tasks.pop(call_id, None)
        self._outstanding.discard(call_id)
        try:
            await self._send({"type": "conversation.item.create", "item": {
                "type": "function_call_output", "call_id": call_id, "output": output}})
            if not self.agent.auto_continue and not self._outstanding:
                await self._send({"type": "response.create"})
        except Exception:  # noqa: BLE001 - the call ended while the tool ran
            pass
        await self._queue.put(Event({
            "type": "sdk.tool_result", "call_id": call_id, "name": name, "arguments": ev.get("arguments"),
            "output": output, "error": error, "ms": int((time.perf_counter() - started) * 1000)}))


def _default(o: Any) -> Any:
    import datetime
    import decimal
    import enum
    if isinstance(o, (datetime.date, datetime.datetime, datetime.time)):
        return o.isoformat()
    if isinstance(o, decimal.Decimal):
        return float(o)
    if isinstance(o, enum.Enum):
        return o.value
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return dataclasses.asdict(o)
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)
