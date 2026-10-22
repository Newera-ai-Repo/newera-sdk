"""A clinic receptionist that books appointments by voice.

    pip install newera-sdk
    export NEWERA_API_KEY=nwc_live_...

    python clinic_agent.py                                  # type to it
    python clinic_agent.py --wav question.wav               # 16 kHz mono 16-bit WAV
    python clinic_agent.py --text "أبي موعد يوم السبت"

The functions below are the "tools". The model decides when to call them; the
SDK runs them and hands the result back. `book_appointment` and
`cancel_appointment` change something, so they are declared with
requires_confirmation: Newera makes the assistant read the details back and
wait for the caller's yes before your function is ever called.

The "clinic" here is an in-memory dict: replace it with your own system.
"""

import argparse
import asyncio
import os
import wave
from enum import Enum
from typing import Literal, Optional

from newera import NeweraError, VoiceAgent

PATIENTS = {"0512345678": "Ahmed Al-Harbi", "0598765432": "Sara Al-Qahtani"}
SLOTS = {"2026-10-24": ["09:00", "10:30", "13:00"], "2026-10-25": ["11:00", "15:30"]}
BOOKINGS: dict[str, dict] = {}


class Department(str, Enum):
    GENERAL = "general"
    DENTAL = "dental"
    PEDIATRICS = "pediatrics"


agent = VoiceAgent(
    api_key=os.environ.get("NEWERA_API_KEY"),
    voice="male",
    session={"newera": {"input_audio_sample_rate": 16000}},      # the example sends a 16 kHz WAV
    instructions=(
        "أنت موظف استقبال في عيادة النخبة الطبية. ساعد المتصل في البحث عن مريض وحجز الموعد وإلغائه. "
        "لا تخترع مواعيد: استخدم الأدوات."
    ),
)


@agent.tool
def find_patient(phone: str) -> dict:
    """Look a patient up by mobile number.

    Args:
        phone: Saudi mobile number, 10 digits starting with 05.
    """
    name = PATIENTS.get(phone)
    return {"found": True, "name": name} if name else {"found": False}


@agent.tool
def get_free_slots(date: str, department: Department = Department.GENERAL) -> dict:
    """List the free appointment times on a day.

    Args:
        date: The day, as YYYY-MM-DD.
        department: Which department the visit is for.
    """
    return {"date": date, "department": department.value, "slots": SLOTS.get(date, [])}


@agent.tool(requires_confirmation=True, timeout_ms=20000)
async def book_appointment(patient_name: str, date: str, time: str,
                           department: Department = Department.GENERAL) -> dict:
    """Book an appointment. Changes the clinic's calendar.

    Args:
        patient_name: Full name as registered.
        date: The day, as YYYY-MM-DD.
        time: The slot, HH:MM, one of the free slots.
        department: Which department.
    """
    if time not in SLOTS.get(date, []):
        raise ValueError("that time is not free")       # the model tells the caller
    await asyncio.sleep(0.2)                            # your real system call
    booking_id = "B-%d" % (len(BOOKINGS) + 1)
    BOOKINGS[booking_id] = {"patient": patient_name, "date": date, "time": time, "department": department.value}
    SLOTS[date].remove(time)
    return {"booking_id": booking_id, **BOOKINGS[booking_id]}


@agent.tool(requires_confirmation=True)
def cancel_appointment(booking_id: str, reason: Optional[Literal["patient", "clinic"]] = None) -> dict:
    """Cancel an existing appointment.

    Args:
        booking_id: The booking number, for example B-1.
        reason: Who asked for the cancellation.
    """
    booking = BOOKINGS.pop(booking_id, None)
    if booking is None:
        raise KeyError("no booking %s" % booking_id)
    SLOTS[booking["date"]].append(booking["time"])
    return {"cancelled": booking_id}


async def converse(args) -> None:
    out = bytearray()
    async with agent.connect() as call:
        if args.wav:
            with wave.open(args.wav) as w:
                assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getsampwidth() == 2, \
                    "use a 16 kHz mono 16-bit WAV"
                pcm = w.readframes(w.getnframes())
            frame = 640                                         # 20 ms
            for i in range(0, len(pcm), frame):
                await call.send_audio(pcm[i:i + frame])
                await asyncio.sleep(0.02)
            for _ in range(60):                                 # silence: the server needs to hear it end
                await call.send_audio(bytes(frame))
                await asyncio.sleep(0.02)
        else:
            await call.send_text(args.text or input("you> "))

        async for event in call:
            if event.type == "response.audio_transcript.delta":
                print(event["delta"], end="", flush=True)
            elif event.type == "response.audio.delta":
                out += event.audio
            elif event.type == "sdk.tool_result":
                shown = event["error"] or event["output"]
                print("\n  [tool] %s(%s) -> %s  (%d ms)" % (event["name"], event["arguments"], shown, event["ms"]))
            elif event.type == "newera.function_call.blocked":
                print("\n  [held back until the caller confirms] %s" % event["name"])
            elif event.type == "response.done":
                print("\n[%s]" % event["response"]["status"])
                if args.once:
                    break
            elif event.type == "error":
                raise NeweraError(event["error"]["code"], event["error"]["message"])
    if out:
        with wave.open("reply.wav", "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(bytes(out))
        print("reply audio saved to reply.wav")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--text")
    ap.add_argument("--once", action="store_true", help="stop after the first finished reply")
    asyncio.run(converse(ap.parse_args()))
