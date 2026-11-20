"""
tools/gate_room_check.py — check the voice gate through the real speaker and mic.

Plays Mac voices aloud and records them through the app's own mic processing:
  1. enrol a *test* voice (default Samantha) into a temporary profile
     (tools/bench_out/room_profile.npz — never your real profile),
  2. play commands from that voice, other voices and music, and run each
     recording through the voice gate exactly as the app would,
  3. print what the gate decided, its scores and timing.
Nothing is sent to Gemini. Turn the volume to a normal listening level.

    .venv/bin/python tools/gate_room_check.py [enrol_voice]
"""

import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
import audio_bench as AB                                          # noqa: E402
from audio.engine import AudioEngine, AudioSettings               # noqa: E402
from audio.speaker_id import ENROL_PHRASES, SpeakerIdentity       # noqa: E402
from audio.voice_gate import VoiceGate                            # noqa: E402

OUT = ROOT / "tools" / "bench_out"
COMMANDS = ["What's the weather like this evening?", "Open my calendar for next week.",
            "Play some relaxing music please.", "Remind me to call mum at six."]


def play(x: np.ndarray, sr=16000):
    with tempfile.TemporaryDirectory() as td:
        AB.write_wav(Path(td) / "a.wav", x)
        subprocess.run(["afplay", str(Path(td) / "a.wav")], check=True)


def record_while_playing(eng: AudioEngine, x: np.ndarray, extra=0.8) -> np.ndarray:
    got = {}
    done = threading.Event()
    eng.capture(len(x) / 16000 + extra, lambda pcm: (got.setdefault("x", pcm), done.set()))
    time.sleep(0.15)
    play(x)
    done.wait(15)
    return np.asarray(got.get("x", np.zeros(1)), np.float32)


def through_gate(sid, rec: np.ndarray, eng: AudioEngine) -> dict:
    """Feed a recording through the gate frame by frame, with the app's speech detection."""
    from audio.voice import VoiceConfig, VoiceProcessor
    events, acts = [], []
    g = VoiceGate(sid, on_audio=lambda b: None, on_activity=acts.append, on_event=events.append)
    det = VoiceProcessor(VoiceConfig(enabled=False))          # detection only (audio is already processed)
    pcm = (np.clip(rec, -1, 1) * 32767).astype(np.int16)
    pad = np.zeros(16000, np.int16)
    pcm = np.concatenate([pcm, pad])
    for i in range(len(pcm) // 160):
        f = pcm[i * 160:(i + 1) * 160]
        det.process(f)
        g.feed(f, det.speech, True)
        if g._pending is not None:
            g._pending.exception()
    ev = events[-1] if events else None
    return {"action": ev.action if ev else "nothing", "label": ev.label if ev else "-",
            "score": ev.score if ev else None, "decide_ms": ev.decide_ms if ev else 0,
            "turn": acts}


def main(voice="Samantha"):
    eng = AudioEngine(AudioSettings())
    eng.start()
    time.sleep(1.0)
    print(f"Enrolling the test voice '{voice}' through the speaker and mic…", flush=True)
    sid = SpeakerIdentity(profile_path=OUT / "room_profile.npz")
    sid.delete_profile()
    samples = []
    for p in ENROL_PHRASES:
        rec = record_while_playing(eng, AB.set_speech_level(AB.tts(p, voice), -20))
        chk = SpeakerIdentity.check_sample(rec)
        print(f"  sample: {'ok' if chk.ok else chk.reason}  speech {chk.speech_s:.1f}s  level {chk.level_dbfs:.0f} dBFS")
        if chk.ok:
            samples.append(rec)
    info = sid.enrol(samples)
    print(f"  profile: {info}\n")
    rows = []
    trials = [(voice, c) for c in COMMANDS] + [(o, COMMANDS[i % 4]) for i, o in
                                               enumerate(["Daniel", "Karen", "Rishi", "Moira", "Fred", "Tessa"])]
    for v, c in trials:
        rec = record_while_playing(eng, AB.set_speech_level(AB.tts(c, v), -20))
        r = through_gate(sid, rec, eng)
        rows.append((v, r))
        sc = f"{r['score']:.2f}" if r["score"] is not None else " –  "
        print(f"  {v:<9} {r['action']:<9} {r['label']:<10} score {sc}  decided {r['decide_ms']:5.0f} ms  turn {r['turn']}",
              flush=True)
    music = record_while_playing(eng, AB.music_bed(16000 * 4) * 0.25)
    r = through_gate(sid, music, eng)
    print(f"  {'music':<9} {r['action']:<9} {r['label']:<10}", flush=True)
    eng.stop()
    sid.delete_profile()
    you = [r for v, r in rows if v == voice]
    oth = [r for v, r in rows if v != voice]
    print(f"\nTest voice accepted {sum(r['action'] == 'accepted' for r in you)}/{len(you)}; "
          f"other voices accepted {sum(r['action'] == 'accepted' for r in oth)}/{len(oth)}. "
          "Temporary profile deleted.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Samantha")
