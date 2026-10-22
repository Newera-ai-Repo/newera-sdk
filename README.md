# newera-sdk

[![test](https://github.com/Newera-ai-Repo/newera-sdk/actions/workflows/test.yml/badge.svg)](https://github.com/Newera-ai-Repo/newera-sdk/actions/workflows/test.yml)

Speech-to-speech calls with tool calling, in Python. Newera listens, talks (Saudi
Arabic and English) and decides when to call **your functions**; you write the
functions.

```python
import asyncio
from newera import VoiceAgent

agent = VoiceAgent(api_key="nwc_live_...", instructions="You book clinic appointments.", voice="male")

@agent.tool
def get_free_slots(date: str) -> list[str]:
    """Free appointment times on a day.

    Args:
        date: The day, as YYYY-MM-DD.
    """
    return ["09:00", "10:30"]

@agent.tool(requires_confirmation=True, timeout_ms=20000)
async def book_appointment(patient_name: str, date: str, time: str) -> dict:
    """Book an appointment."""
    return {"booking_id": "B-1"}

async def main():
    async with agent.connect() as call:
        await call.send_text("I need an appointment on Saturday")
        reply = await call.wait_response()
        print(reply.transcript)          # tools ran while it waited

asyncio.run(main())
```

A complete agent with a mock clinic: [`examples/clinic_agent.py`](examples/clinic_agent.py).

## Install

```
pip install newera-sdk
```

The import name is `newera`: `from newera import VoiceAgent`.

It needs Python 3.9 or newer and depends only on `websockets`.

## Tools

`@agent.tool` turns a function into a tool. The schema comes from the function:

| from | becomes |
|---|---|
| the name | tool name (`name=` overrides) |
| docstring summary (text before a blank line or `Args:`) | description (`description=` overrides) |
| `Args:` section, Google style (`name: text` or `name (type): text`) | each parameter's description |
| `str`, `int`, `float`, `bool` | `string`, `integer`, `number`, `boolean` |
| `Literal["a", "b"]`, `Enum` | `enum` |
| `Optional[X]`, `X \| None`, parameters with a default | optional (not in `required`) |
| `list[X]`, `dict`, `Union[...]`, `Any` | `array`, `object`, `anyOf`, any value |

```python
@agent.tool(requires_confirmation=False, timeout_ms=15000)
def my_tool(...): ...
```

Both sync and async functions work (sync ones run in a thread, so a slow call
never blocks the audio). The return value is JSON encoded and given to the
model (dates, decimals, enums and dataclasses are converted). **If the function
raises, the model receives `{"error": "ValueError: ..."}`** and tells the caller;
the call carries on.

### What Newera does around your function

* **Validation.** The model's arguments are checked against the schema before
  your function is called. A wrong or missing argument never reaches you: the
  model is told and asks the caller.
* **Confirmation.** With `requires_confirmation=True`, your function is called
  only after the assistant read the details back and the caller said yes. A
  different argument is a new question. Declare every function that changes
  something this way (booking, paying, cancelling, sending).
* **Speech while you work.** If the model calls a tool without saying anything,
  Newera speaks a short line ("Sure, one moment while I check...") and, if your
  function takes more than 3 and 8 seconds, a progress line. You write nothing for this.
* **Timeouts.** `timeout_ms` (1000 to 60000, default 15000) is how long Newera
  waits for the result; after that the model tells the caller it is taking longer
  and offers to retry.
* **Barge-in.** If the caller interrupts while a read-only tool runs, it is
  cancelled (an `asyncio` cancel for async functions; a sync function in a thread
  cannot be stopped, but its result is ignored). A tool with `requires_confirmation`
  is never cancelled; its result is reported truthfully on the next turn.
* **Limits.** At most 32 tools; each name `^[a-zA-Z0-9_-]{1,64}$`; each schema
  16 KB, all of them 64 KB; a result is cut at 32 KB; at most 6 tool steps per reply
  (`VoiceAgent(max_tool_steps=...)`, up to 10). When the limit is reached the
  assistant tells the caller what is done and asks whether to continue.

Treat what your functions return as what the *model* will read to the caller,
and treat what comes back from a tool as data, never as instructions.

## The call

```python
async with agent.connect() as call:
    await call.send_audio(pcm16_bytes)      # 16-bit mono, 24 kHz by default (16 kHz: session={"newera": {"input_audio_sample_rate": 16000}}); real time, 20-100 ms chunks
    await call.send_text("hello")           # a typed turn (add respond=False to wait)
    await call.say("Welcome to the clinic") # the assistant speaks first
    async for event in call:                # every server event
        ...
    await call.end()                        # session.summary, then the call closes
```

`event.type` is the protocol's event name (protocol reference):
`response.audio.delta` (`event.audio` gives the PCM16 24 kHz bytes),
`response.audio_transcript.delta`, `input_audio_buffer.speech_started` (stop your
playback), `response.done`, `error`, and so on. The SDK adds one event,
`sdk.tool_result` (`name`, `arguments`, `output`, `error`, `ms`), after each of your
functions ran.

`await call.wait_response()` collects events up to the next `response.done` and
returns a `Reply(transcript, audio, status, usage, events)`; handy for text-only
scripts and tests.

`VoiceAgent(...)` takes `api_key`, `instructions`, `voice`, `tool_choice`
(`"auto"`, `"none"`, `"required"`, or `{"type": "function", "name": ...}`),
`max_tool_steps`, `auto_continue`, `session` (extra `session.update` fields such as
`{"tts": {"quality": "studio"}}`), `url` and `headers`.

`auto_continue=False` is the OpenAI-style flow: after your outputs the SDK sends
`response.create` itself and the server does not continue on its own.

Browsers must not hold an API key: this SDK is for servers. Give browser clients
their own short-lived credentials instead.

## Tests

```
pip install -e '.[test]' && pytest
```

The tests run the SDK against a fake WebSocket server; no network, no key.

## Releasing

1. Bump `version` in `pyproject.toml` (and `__version__` in `src/newera/__init__.py`).
2. Push to `main`. The `publish` workflow runs the tests. If that version is not yet on
   PyPI, it builds the package, uploads it through trusted publishing, and creates the
   `vX.Y.Z` tag and a GitHub Release. No API token is stored in the repository.
3. A push with an unchanged version only runs the tests.

## License

Apache License 2.0. See [LICENSE](LICENSE).
