"""
tools/gate_latency.py — does Gemini Live accept app-controlled turns, and what
does the voice gate cost in reply time?

Plays the same spoken command to Gemini two ways, in real time:
  server   today's path: audio streamed live, Gemini's own speech detection
  gate     the voice gate's path: automatic detection off; the first ~1.3 s is
           held (recognition), then sent in a burst, the rest live, and
           "activity end" after 0.7 s of silence
and measures end of speech → first reply audio, plus what Gemini transcribed.
Uses a few short Live sessions on the configured API key.

    .venv/bin/python tools/gate_latency.py [rounds]
"""

import asyncio
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import audio_bench as AB                                  # noqa: E402
from memory.config_manager import get_gemini_key          # noqa: E402

SR = 16000
CHUNK = SR // 50
PHRASES = ["What's the capital of France?", "Tell me a fun fact about the moon.",
           "How many days are there in a leap year?"]
SYSTEM = "Answer in one short sentence."


def live_model() -> str:
    m = re.search(r'LIVE_MODEL\s*=\s*"([^"]+)"', (ROOT / "main.py").read_text(encoding="utf-8"))
    return m.group(1)


MODES = {  # name: (manual turns, hold seconds, end-of-turn silence)
    "server": (False, 0, 0), "manual0.7": (True, 0, 0.7), "manual0.3": (True, 0, 0.3),
    "gate": (True, 1.3, 0.7), "gate0.4": (True, 1.3, 0.4),
    "hold0.5": (True, 0.5, 0.7), "holdall": (True, 9.0, 0.0),
    "final": (True, 0.7, 0.55),   # shipped gate: 0.15 s pre-roll + 0.8 s speech + model ≈ 1.0 s held
    "early": (True, 0.7, 0.55, True),   # turn opened at speech start; audio follows after the check
}


async def trial(client, model, pcm: np.ndarray, mode: str) -> dict:
    from google.genai import types
    cfg = dict(response_modalities=["AUDIO"], input_audio_transcription={}, system_instruction=SYSTEM)
    manual, hold_s, end_s, *rest = MODES[mode]
    early = bool(rest and rest[0])
    if manual:
        cfg["realtime_input_config"] = types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True))
    speech_end = None
    first_audio = None
    heard = []
    async with client.aio.live.connect(model=model, config=types.LiveConnectConfig(**cfg)) as s:
        async def send_audio(x):
            await s.send_realtime_input(audio=types.Blob(data=x.tobytes(), mime_type="audio/pcm;rate=16000"))

        async def sender():
            nonlocal speech_end
            lead = np.zeros(int(0.3 * SR), np.int16)
            tail = np.zeros(int(2.0 * SR), np.int16)
            data = np.concatenate([lead, pcm, tail])
            end_i = len(lead) + len(pcm)
            t0 = time.perf_counter()
            if not manual:
                for i in range(0, len(data), CHUNK):
                    await send_audio(data[i:i + CHUNK])
                    if speech_end is None and i + CHUNK >= end_i:
                        speech_end = time.perf_counter()
                    await asyncio.sleep(max(0, t0 + (i + CHUNK) / SR - time.perf_counter()))
                return
            hold = min(int(hold_s * SR) + len(lead), end_i) if hold_s else 0   # gate decides after ~1 s speech
            if early:
                await asyncio.sleep(len(lead) / SR)
                await s.send_realtime_input(activity_start=types.ActivityStart())
                await asyncio.sleep(max(0, t0 + hold / SR - time.perf_counter()))
            else:
                await asyncio.sleep(hold / SR)
                await s.send_realtime_input(activity_start=types.ActivityStart())
            for i in range(0, hold, SR // 10):               # burst of held audio
                await send_audio(data[i:min(hold, i + SR // 10)])
            stop = end_i + int(end_s * SR)                   # end after the silence
            for i in range(hold, stop, CHUNK):
                await asyncio.sleep(max(0, t0 + i / SR - time.perf_counter()))
                await send_audio(data[i:i + CHUNK])
                if speech_end is None and i + CHUNK >= end_i:
                    speech_end = t0 + end_i / SR
            await asyncio.sleep(max(0, t0 + stop / SR - time.perf_counter()))
            await s.send_realtime_input(activity_end=types.ActivityEnd())

        task = asyncio.create_task(sender())
        try:
            async def receiver():
                nonlocal first_audio
                async for r in s.receive():
                    sc = r.server_content
                    if sc and sc.input_transcription and sc.input_transcription.text:
                        heard.append(sc.input_transcription.text)
                    if r.data and first_audio is None:
                        first_audio = time.perf_counter()
                    if sc and sc.turn_complete:
                        return
            await asyncio.wait_for(receiver(), timeout=25)
        except asyncio.TimeoutError:
            pass
        task.cancel()
    lat = (first_audio - speech_end) if (first_audio and speech_end) else None
    return {"mode": mode, "latency_s": lat, "heard": "".join(heard).strip()}


async def empty_turn(client, model) -> dict:
    """What does Gemini do with a turn that has no audio (the gate's 'cancel')?"""
    from google.genai import types
    cfg = types.LiveConnectConfig(
        response_modalities=["AUDIO"], output_audio_transcription={}, system_instruction=SYSTEM,
        realtime_input_config=types.RealtimeInputConfig(
            automatic_activity_detection=types.AutomaticActivityDetection(disabled=True)))
    said, got_audio = [], False
    async with client.aio.live.connect(model=model, config=cfg) as s:
        await s.send_realtime_input(activity_start=types.ActivityStart())
        await asyncio.sleep(1.0)
        await s.send_realtime_input(activity_end=types.ActivityEnd())
        try:
            async def rx():
                nonlocal got_audio
                async for r in s.receive():
                    if r.data:
                        got_audio = True
                    sc = r.server_content
                    if sc and sc.output_transcription and sc.output_transcription.text:
                        said.append(sc.output_transcription.text)
                    if sc and sc.turn_complete:
                        return
            await asyncio.wait_for(rx(), timeout=8)
        except asyncio.TimeoutError:
            pass
    return {"replied": got_audio, "said": "".join(said).strip()}


async def main(rounds: int):
    from google import genai
    client = genai.Client(api_key=get_gemini_key(), http_options={"api_version": "v1beta"})
    model = live_model()
    if "empty" in MODE_LIST:
        for _ in range(rounds):
            print("empty turn:", await empty_turn(client, model), flush=True)
        return
    pcms = [(AB.set_speech_level(AB.tts(p, "Samantha"), -26) * 32767).astype(np.int16) for p in PHRASES]
    out = []
    for r in range(rounds):
        for i, pcm in enumerate(pcms):
            for mode in MODE_LIST:
                try:
                    res = await trial(client, model, pcm, mode)
                except Exception as e:
                    res = {"mode": mode, "latency_s": None, "heard": f"ERROR {type(e).__name__}: {e}"[:160]}
                res["phrase"] = i
                out.append(res)
                lat = f"{res['latency_s']:.2f}s" if res["latency_s"] else "  –  "
                print(f"{mode:<6} {lat:>7}  heard: {res['heard'][:70]}", flush=True)
    for mode in MODE_LIST:
        v = [r["latency_s"] for r in out if r["mode"] == mode and r["latency_s"]]
        if v:
            print(f"{mode}: median {np.median(v):.2f} s over {len(v)} (min {min(v):.2f}, max {max(v):.2f})")


MODE_LIST = ["server", "gate"]

if __name__ == "__main__":
    if len(sys.argv) > 2:
        MODE_LIST = sys.argv[2].split(",")
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 2))
