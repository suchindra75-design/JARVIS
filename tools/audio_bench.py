"""
tools/audio_bench.py — offline and acoustic benchmark for JARVIS voice processing.

Developer tool, not used by the app. Builds reproducible test scenes from
macOS `say` voices plus generated noise, runs candidate processing chains,
and reports noise reduction, echo reduction, speech preservation, level
consistency, VAD accuracy and per-frame cost.

    python tools/audio_bench.py scenes     # build test scenes (macOS only: uses `say`)
    python tools/audio_bench.py offline    # run candidates on the scenes
    python tools/audio_bench.py acoustic   # play through the speakers, record the mic

Everything is written to tools/bench_out/ (git-ignored).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tools" / "bench_out"
SCENES = OUT / "scenes"
SR = 16000
FRAME = SR // 100          # 10 ms
RNG = np.random.default_rng(1234)

NEAR_VOICES = ["Samantha", "Karen", "Rishi", "Moira", "Fred", "Tessa"]
FAR_VOICE = "Daniel"       # plays the part of JARVIS speaking
PHRASES = [
    "Open Spotify and play some relaxing jazz music.",
    "What will the weather be like in Bangalore tomorrow morning?",
    "Remind me to call the dentist at four thirty this afternoon.",
    "Search the web for the latest news about electric cars.",
    "Send a message to Priya saying I will be ten minutes late.",
    "How much free space is left on my hard drive right now?",
]
FAR_PHRASES = [
    "Certainly. I have opened the application and the music should begin playing shortly.",
    "Tomorrow will be partly cloudy with a high of twenty eight degrees and light winds.",
]
BABBLE = [
    "The quarterly numbers look much better than we expected last month.",
    "Did anyone see where I left the keys to the meeting room?",
    "I think we should order lunch before the afternoon session starts.",
]


# ── audio helpers ─────────────────────────────────────────────────────────────

def read_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def write_wav(p: Path, x: np.ndarray) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    y = np.clip(x, -1, 1)
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((y * 32767).astype(np.int16).tobytes())


def tts(text: str, voice: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        aiff, wav = Path(td) / "a.aiff", Path(td) / "a.wav"
        subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                        str(aiff), str(wav)], check=True, capture_output=True)
        return read_wav(wav)


def rms_db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)


def active_mask(clean: np.ndarray, frame: int = FRAME, thresh_db: float = -45) -> np.ndarray:
    n = len(clean) // frame
    e = np.array([rms_db(clean[i * frame:(i + 1) * frame]) for i in range(n)])
    return e > thresh_db


def set_speech_level(x: np.ndarray, target_db: float) -> np.ndarray:
    m = active_mask(x)
    act = np.concatenate([x[i * FRAME:(i + 1) * FRAME] for i in np.where(m)[0]]) if m.any() else x
    return x * 10 ** ((target_db - rms_db(act)) / 20)


def pad(x: np.ndarray, lead: float = 0.8, tail: float = 0.8) -> np.ndarray:
    return np.concatenate([np.zeros(int(lead * SR)), x, np.zeros(int(tail * SR))]).astype(np.float32)


def fit(x: np.ndarray, n: int) -> np.ndarray:
    return np.resize(x, n).astype(np.float32) if len(x) < n else x[:n].astype(np.float32)


# ── noise generators (synthetic, no third-party recordings) ──────────────────

def fan_noise(n: int) -> np.ndarray:
    w = RNG.standard_normal(n)
    brown = np.zeros(n); acc = 0.0
    for i in range(n):                      # leaky integrator → brown-ish rumble
        acc = 0.995 * acc + w[i]; brown[i] = acc
    brown /= np.max(np.abs(brown)) + 1e-9
    pink = np.convolve(RNG.standard_normal(n), np.ones(8) / 8, mode="same")
    t = np.arange(n) / SR
    hum = 0.3 * np.sin(2 * np.pi * 100 * t) + 0.15 * np.sin(2 * np.pi * 200 * t)
    blade = 0.2 * np.sin(2 * np.pi * 23 * t) * pink
    return (0.6 * brown + 0.4 * pink + hum * 0.2 + blade).astype(np.float32)


def music_bed(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    chords = [[220, 277.2, 329.6], [196, 246.9, 293.7], [174.6, 220, 261.6], [196, 246.9, 311.1]]
    y = np.zeros(n)
    beat = SR // 4
    for i in range(0, n, SR):
        c = chords[(i // SR) % 4]
        seg = slice(i, min(n, i + SR))
        tt = t[seg] - t[i]
        env = np.minimum(1, tt * 20) * np.exp(-tt * 1.5)
        for f in c:
            y[seg] += env * (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt))
        y[seg] += 0.8 * env * np.sin(2 * np.pi * c[0] / 2 * tt)          # bass
    for i in range(0, n, beat):                                          # hi-hat
        k = min(n - i, 800)
        y[i:i + k] += 0.25 * RNG.standard_normal(k) * np.exp(-np.arange(k) / 120)
    return (y / (np.max(np.abs(y)) + 1e-9)).astype(np.float32)


def room_ir(delay_ms: float = 35, rt60: float = 0.25, direct: float = 0.6) -> np.ndarray:
    n = int(0.3 * SR)
    t = np.arange(n) / SR
    ir = RNG.standard_normal(n) * np.exp(-6.9 * t / rt60) * 0.08
    d = int(delay_ms * SR / 1000)
    ir = np.concatenate([np.zeros(d), ir])
    ir[d] += direct
    return ir


def speaker_path(far: np.ndarray, ir: np.ndarray) -> np.ndarray:
    driven = np.tanh(1.6 * far) / np.tanh(1.6)           # mild loudspeaker nonlinearity
    return np.convolve(driven, ir)[: len(far)].astype(np.float32)


# ── scene builder ─────────────────────────────────────────────────────────────

def mix_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    m = active_mask(clean)
    act = np.concatenate([clean[i * FRAME:(i + 1) * FRAME] for i in np.where(m)[0]])
    noise = noise * 10 ** ((rms_db(act) - snr_db - rms_db(noise)) / 20)
    return clean + noise


def build_scenes() -> None:
    if sys.platform != "darwin":
        sys.exit("Scene building uses macOS `say`.")
    SCENES.mkdir(parents=True, exist_ok=True)
    manifest = []
    print("Synthesising speech…")
    near = [set_speech_level(tts(p, NEAR_VOICES[i % len(NEAR_VOICES)]), -26) for i, p in enumerate(PHRASES)]
    far = [set_speech_level(tts(p, FAR_VOICE), -20) for p in FAR_PHRASES]
    babble_src = [set_speech_level(tts(p, v), -26) for p, v in zip(BABBLE, ["Fred", "Karen", "Rishi"])]
    ir = room_ir()

    def add(name, kind, mic, clean=None, ref=None, text="", **extra):
        write_wav(SCENES / f"{name}.mic.wav", mic)
        if clean is not None:
            write_wav(SCENES / f"{name}.clean.wav", clean)
        if ref is not None:
            write_wav(SCENES / f"{name}.ref.wav", ref)
        manifest.append({"name": name, "kind": kind, "text": text,
                         "clean": clean is not None, "ref": ref is not None, **extra})

    for i, (x, txt) in enumerate(zip(near, PHRASES)):
        c = pad(x)
        add(f"quiet_{i}", "quiet", c + 3e-4 * RNG.standard_normal(len(c)).astype(np.float32), c, text=txt)
    for i in range(3):
        c = pad(set_speech_level(near[i], -42)); add(f"soft_{i}", "soft", c + 3e-4 * RNG.standard_normal(len(c)), c, text=PHRASES[i])
        c = pad(near[i + 3] * (0.89 / np.max(np.abs(near[i + 3])))); add(f"loud_{i}", "loud", c, c, text=PHRASES[i + 3])
    for snr in (0, 5, 10):
        for i in range(3):
            c = pad(near[i]); add(f"fan{snr}_{i}", f"fan_snr{snr}", mix_at_snr(c, fan_noise(len(c)), snr), c, text=PHRASES[i], snr=snr)
    for snr in (5, 10):
        for i in range(3):
            c = pad(near[i + 3]); add(f"music{snr}_{i}", f"music_snr{snr}", mix_at_snr(c, music_bed(len(c)), snr), c, text=PHRASES[i + 3], snr=snr)
    for i in range(3):
        c = pad(near[i])
        bab = sum(fit(b, len(c)) for b in babble_src)
        add(f"babble5_{i}", "babble_snr5", mix_at_snr(c, bab, 5), c, text=PHRASES[i], snr=5)
    for i, f in enumerate(far):
        r = pad(f, 0.3, 0.6)
        add(f"echo_{i}", "echo_only", speaker_path(r, ir), np.zeros_like(r), r)
    for i in range(3):
        nr = pad(near[i], 1.6, 0.6)
        fr = fit(pad(far[i % 2], 0.2, 0.0), len(nr))
        mic = nr + speaker_path(fr, ir)
        add(f"doubletalk_{i}", "doubletalk", mic, nr, fr, text=PHRASES[i])
    (SCENES / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"Built {len(manifest)} scenes in {SCENES}")


# ── candidates ────────────────────────────────────────────────────────────────

CANDIDATES = {
    # name: (echo_cancellation, noise_suppression, high_pass, auto_gain)
    "bypass":        None,
    "aec+agc":       (True, False, True, True),
    "aec+ns":        (True, True, True, False),
    "standard":      (True, True, True, True),
}


def run_apm(flags, mic: np.ndarray, ref: np.ndarray | None, delay_ms: int = 40):
    from livekit import rtc
    if flags is None:
        return mic.copy(), np.zeros(1)
    aec, ns, hpf, agc = flags
    apm = rtc.AudioProcessingModule(echo_cancellation=aec, noise_suppression=ns,
                                    high_pass_filter=hpf, auto_gain_control=agc)
    m16 = (np.clip(mic, -1, 1) * 32767).astype(np.int16)
    r16 = None if ref is None else (np.clip(ref, -1, 1) * 32767).astype(np.int16)
    out = np.zeros_like(m16)
    zeros = np.zeros(FRAME, np.int16).tobytes()
    times = []
    for i in range(len(m16) // FRAME):
        s = slice(i * FRAME, (i + 1) * FRAME)
        t0 = time.perf_counter()
        if aec:
            rb = r16[s].tobytes() if r16 is not None else zeros
            apm.process_reverse_stream(rtc.AudioFrame(rb, SR, 1, FRAME))
            apm.set_stream_delay_ms(delay_ms)
        fr = rtc.AudioFrame(m16[s].tobytes(), SR, 1, FRAME)
        apm.process_stream(fr)
        times.append(time.perf_counter() - t0)
        out[s] = np.frombuffer(fr.data, dtype=np.int16)
    return out.astype(np.float32) / 32768, np.array(times)


# ── metrics ───────────────────────────────────────────────────────────────────

def align(est: np.ndarray, ref: np.ndarray, max_lag: int = 640) -> np.ndarray:
    """Undo processing delay. Uses |correlation| because voiced speech is
    quasi-periodic, so neighbouring lags can correlate negatively."""
    n = min(len(est), len(ref))
    e, r = est[:n], ref[:n]
    size = 1 << (2 * n - 1).bit_length()
    xc = np.fft.irfft(np.fft.rfft(e, size) * np.conj(np.fft.rfft(r, size)), size)
    lag = int(np.argmax(np.abs(xc[: max_lag + 1])))
    return np.concatenate([e[lag:], np.zeros(lag)])


def si_sdr(est: np.ndarray, ref: np.ndarray) -> float:
    est = align(est, ref)
    n = min(len(est), len(ref))
    e, r = est[:n] - est[:n].mean(), ref[:n] - ref[:n].mean()
    a = np.dot(e, r) / (np.dot(r, r) + 1e-12)
    s = a * r
    return 10 * np.log10(np.dot(s, s) / (np.dot(e - s, e - s) + 1e-12))


def frame_power(x: np.ndarray, mask: np.ndarray) -> float:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return float("nan")
    return float(np.mean([np.mean(x[i * FRAME:(i + 1) * FRAME] ** 2) for i in idx]))


def vad_scores(x: np.ndarray, clean: np.ndarray, mode: int) -> tuple[float, float]:
    import webrtcvad
    v = webrtcvad.Vad(mode)
    f = 480                                         # 30 ms
    act = active_mask(clean, f, -40)
    x16 = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    det = np.array([v.is_speech(x16[i * f:(i + 1) * f].tobytes(), SR) for i in range(len(act))])
    hit = det[act].mean() if act.any() else float("nan")
    fa = det[~act].mean() if (~act).any() else float("nan")
    return float(hit), float(fa)


def evaluate() -> dict:
    manifest = json.loads((SCENES / "manifest.json").read_text())
    results = {c: {} for c in CANDIDATES}
    vad = {}
    for sc in manifest:
        mic = read_wav(SCENES / f"{sc['name']}.mic.wav")
        clean = read_wav(SCENES / f"{sc['name']}.clean.wav") if sc["clean"] else None
        ref = read_wav(SCENES / f"{sc['name']}.ref.wav") if sc["ref"] else None
        outs = {}
        for cname, flags in CANDIDATES.items():
            out, times = run_apm(flags, mic, ref)
            outs[cname] = out
            r = {"frame_ms_mean": float(times.mean() * 1000), "frame_ms_p99": float(np.percentile(times, 99) * 1000),
                 "clipped": int(np.sum(np.abs(out) >= 0.999))}
            if sc["kind"] == "echo_only":
                half = len(mic) // 2      # after convergence
                r["erle_db"] = rms_db(mic[half:]) - rms_db(out[half:])
                r["erle_first_second_db"] = rms_db(mic[: SR]) - rms_db(out[: SR])
            elif clean is not None:
                m = active_mask(clean)
                r["si_sdr"] = si_sdr(out, clean)
                r["si_sdr_in"] = si_sdr(mic, clean)
                n = min(len(m) * FRAME, len(out))
                r["noise_red_db"] = 10 * np.log10(frame_power(mic[:n], ~m) / (frame_power(out[:n], ~m) + 1e-12))
                r["speech_dbfs"] = 10 * np.log10(frame_power(out[:n], m) + 1e-12)
                if sc["kind"] == "doubletalk":
                    far_only = np.zeros_like(m); far_only[: int(1.4 * SR) // FRAME] = True
                    far_only &= ~m
                    r["erle_db"] = 10 * np.log10(frame_power(mic, far_only) / (frame_power(out, far_only) + 1e-12))
            results[cname][sc["name"]] = {"kind": sc["kind"], **r}
        if clean is not None and sc["kind"] in ("fan_snr5", "music_snr5", "babble_snr5", "quiet", "soft"):
            for cname in ("bypass", "standard"):
                for mode in (1, 2, 3):
                    hit, fa = vad_scores(outs[cname], clean, mode)
                    vad.setdefault(f"{cname}/mode{mode}", {}).setdefault(sc["kind"], []).append((hit, fa))
        # Light level = blend of noise-suppressed and echo-cancelled-only output
        if clean is not None and sc["kind"] != "echo_only":
            light = 0.6 * outs["standard"] + 0.4 * outs["aec+agc"]
            m = active_mask(clean)
            n = min(len(m) * FRAME, len(light))
            results.setdefault("light", {})[sc["name"]] = {
                "kind": sc["kind"], "si_sdr": si_sdr(light, clean),
                "noise_red_db": 10 * np.log10(frame_power(mic[:n], ~m) / (frame_power(light[:n], ~m) + 1e-12)),
                "speech_dbfs": 10 * np.log10(frame_power(light[:n], m) + 1e-12),
            }
        print(f"  {sc['name']:<14} done")
    return {"per_scene": results, "vad": {k: {kind: [float(np.nanmean([a for a, _ in v])), float(np.nanmean([b for _, b in v]))]
                                              for kind, v in d.items()} for k, d in vad.items()}}


def summarise(res: dict) -> str:
    lines = ["| candidate | scene type | SI-SDR in→out dB | noise reduction dB | echo reduction dB | speech dBFS | ms/frame |",
             "|---|---|---|---|---|---|---|"]
    for cname, scenes in res["per_scene"].items():
        kinds: dict[str, list] = {}
        for r in scenes.values():
            kinds.setdefault(r["kind"], []).append(r)
        for kind, rs in kinds.items():
            def avg(k):
                v = [r[k] for r in rs if k in r and np.isfinite(r[k])]
                return f"{np.mean(v):.1f}" if v else "–"
            sdr = f"{avg('si_sdr_in')}→{avg('si_sdr')}" if any("si_sdr" in r for r in rs) else "–"
            lines.append(f"| {cname} | {kind} | {sdr} | {avg('noise_red_db')} | {avg('erle_db')} | {avg('speech_dbfs')} | {avg('frame_ms_mean')} |")
    lines += ["", "| VAD | scene | speech hit rate | false alarm rate |", "|---|---|---|---|"]
    for k, d in res["vad"].items():
        for kind, (h, f) in d.items():
            lines.append(f"| {k} | {kind} | {h:.2f} | {f:.2f} |")
    return "\n".join(lines)


# ── acoustic test: real speakers → real mic ──────────────────────────────────

def acoustic() -> None:
    import sounddevice as sd
    from livekit import rtc  # noqa: F401  (fail early if missing)
    manifest = json.loads((SCENES / "manifest.json").read_text())
    far = read_wav(SCENES / "echo_0.ref.wav")
    far = np.concatenate([far, read_wav(SCENES / "echo_1.ref.wav")])
    noisy = read_wav(SCENES / "fan5_0.mic.wav")
    clean_noisy = read_wav(SCENES / "fan5_0.clean.wav")
    report = {"device_in": sd.query_devices(sd.default.device[0])["name"],
              "device_out": sd.query_devices(sd.default.device[1])["name"]}
    print(f"Playing {len(far)/SR:.1f}s of 'JARVIS' speech through {report['device_out']} and recording…")
    rec = sd.playrec((far * 0.5).reshape(-1, 1), SR, channels=1, dtype="float32", blocking=True)[:, 0]
    if rms_db(rec) < -75:
        sys.exit("The recording is silent — macOS probably blocked microphone access for this process.")
    # delay estimate from cross-correlation
    corr = np.correlate(rec[: 3 * SR], far[: 3 * SR] * 0.5, mode="full")[3 * SR - 1:]
    delay = int(np.argmax(np.abs(corr[: SR // 2])))
    report["measured_delay_ms"] = delay * 1000 / SR
    write_wav(OUT / "acoustic_echo_mic.wav", rec)
    for cname in ("bypass", "aec+agc", "standard"):
        out, times = run_apm(CANDIDATES[cname], rec, far * 0.5, delay_ms=int(report["measured_delay_ms"]))
        half = len(rec) // 2
        report[f"{cname}_erle_db"] = rms_db(rec[half:]) - rms_db(out[half:])
        write_wav(OUT / f"acoustic_echo_{cname}.wav", out)
    print("Playing fan noise + speech (not given as reference: tests noise suppression on the real mic)…")
    rec2 = sd.playrec((noisy * 0.5).reshape(-1, 1), SR, channels=1, dtype="float32", blocking=True)[:, 0]
    write_wav(OUT / "acoustic_noise_mic.wav", rec2)
    m = active_mask(clean_noisy)
    lag = int(report["measured_delay_ms"] * SR / 1000)
    rec2a = np.concatenate([rec2[lag:], np.zeros(lag)])
    for cname in ("bypass", "standard"):
        out, _ = run_apm(CANDIDATES[cname], rec2a, None)
        n = min(len(m) * FRAME, len(out))
        report[f"{cname}_noise_red_db"] = 10 * np.log10(frame_power(rec2a[:n], ~m) / (frame_power(out[:n], ~m) + 1e-12))
        write_wav(OUT / f"acoustic_noise_{cname}.wav", out)
    (OUT / "acoustic.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


# ── recognition quality through Gemini Live (uses the app's model and API key) ──

_NUM = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
        "sixteen seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _num_words(n: int) -> str:
    if n < 20:
        return _NUM[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _NUM[n % 10])
    return str(n)


def normalise(text: str) -> list[str]:
    import re
    t = text.lower()
    t = re.sub(r"(\d+):(\d+)", lambda m: f"{_num_words(int(m.group(1)))} {_num_words(int(m.group(2)))}", t)
    t = re.sub(r"\d+", lambda m: _num_words(int(m.group(0))), t)
    t = re.sub(r"[^a-z' ]+", " ", t).replace("'", "")
    return t.split()


def wer(ref: str, hyp: str) -> float:
    r, h = normalise(ref), normalise(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


ASR_CONFIGS = {
    "bypass":   dict(enabled=False),
    "standard": dict(),
    "light":    dict(noise_suppression="light"),
}
ASR_KINDS = ("quiet", "soft", "loud", "fan_snr0", "fan_snr5", "music_snr5", "babble_snr5")


def process_with_voice(mic: np.ndarray, cfg: dict) -> np.ndarray:
    sys.path.insert(0, str(ROOT))
    from audio.voice import VoiceProcessor, VoiceConfig
    vp = VoiceProcessor(VoiceConfig(**cfg))
    m16 = (np.clip(mic, -1, 1) * 32767).astype(np.int16)
    out = np.zeros_like(m16)
    for i in range(len(m16) // FRAME):
        s = slice(i * FRAME, (i + 1) * FRAME)
        vp.process_reverse(np.zeros(FRAME, np.int16))
        out[s] = vp.process(m16[s])
    if vp.error:
        raise RuntimeError(vp.error)
    return out


async def _transcribe(client, model, pcm16: np.ndarray) -> tuple[str, float | None]:
    from google.genai import types
    cfg = types.LiveConnectConfig(
        response_modalities=["AUDIO"], input_audio_transcription={},
        system_instruction="You are a test harness. Whatever you hear, reply with only the word: okay.",
    )
    text, first_audio = [], None
    async with client.aio.live.connect(model=model, config=cfg) as s:
        chunk = SR // 50                                  # 20 ms, sent in real time
        data = np.concatenate([pcm16, np.zeros(int(1.2 * SR), np.int16)])
        t_end = None
        import asyncio

        async def sender():
            nonlocal t_end
            for i in range(0, len(data), chunk):
                await s.send_realtime_input(audio=types.Blob(data=data[i:i + chunk].tobytes(),
                                                             mime_type="audio/pcm;rate=16000"))
                if i + chunk >= len(pcm16) and t_end is None:
                    t_end = time.perf_counter()
                await asyncio.sleep(chunk / SR)

        send_task = asyncio.create_task(sender())
        try:
            async def receiver():
                nonlocal first_audio
                async for r in s.receive():
                    sc = r.server_content
                    if sc and sc.input_transcription and sc.input_transcription.text:
                        text.append(sc.input_transcription.text)
                    if r.data and first_audio is None and t_end is not None:
                        first_audio = time.perf_counter() - t_end
                    if sc and sc.turn_complete:
                        return
            await asyncio.wait_for(receiver(), timeout=20)
        except asyncio.TimeoutError:
            pass
        finally:
            send_task.cancel()
    return "".join(text).strip(), first_audio


def asr() -> None:
    import asyncio
    sys.path.insert(0, str(ROOT))
    from google import genai
    import re
    live_model = re.search(r'LIVE_MODEL\s*=\s*"([^"]+)"', (ROOT / "main.py").read_text()).group(1)
    key = json.loads((ROOT / "config" / "api_keys.json").read_text())["gemini_api_key"]
    client = genai.Client(api_key=key, http_options={"api_version": "v1beta"})
    manifest = [m for m in json.loads((SCENES / "manifest.json").read_text()) if m["kind"] in ASR_KINDS]
    manifest = [m for m in manifest if not m["name"].endswith(("_3", "_4", "_5"))]   # 3 per kind
    rows = []

    async def run_all():
        for sc in manifest:
            mic = read_wav(SCENES / f"{sc['name']}.mic.wav")
            for cname, cfg in ASR_CONFIGS.items():
                pcm = process_with_voice(mic, cfg) if cfg.get("enabled", True) else (np.clip(mic, -1, 1) * 32767).astype(np.int16)
                try:
                    hyp, lat = await _transcribe(client, live_model, pcm)
                except Exception as e:
                    hyp, lat = f"<error {e}>", None
                w = wer(sc["text"], hyp)
                rows.append({"scene": sc["name"], "kind": sc["kind"], "config": cname, "wer": w,
                             "latency_s": lat, "hyp": hyp})
                print(f"  {sc['name']:<12} {cname:<9} WER {w:4.2f}  reply {lat if lat is None else round(lat, 2)}s  | {hyp[:70]}")

    asyncio.run(run_all())
    (OUT / "asr.json").write_text(json.dumps(rows, indent=1))
    print("\n| scene type | " + " | ".join(ASR_CONFIGS) + " |\n|---|" + "---|" * len(ASR_CONFIGS))
    for kind in ASR_KINDS:
        vals = []
        for c in ASR_CONFIGS:
            ws = [r["wer"] for r in rows if r["kind"] == kind and r["config"] == c]
            vals.append(f"{100 * np.mean(ws):.0f}%" if ws else "–")
        print(f"| {kind} | " + " | ".join(vals) + " |")
    for c in ASR_CONFIGS:
        ws = [r["wer"] for r in rows if r["config"] == c]
        ls = [r["latency_s"] for r in rows if r["config"] == c and r["latency_s"]]
        print(f"{c}: overall WER {100 * np.mean(ws):.1f}%   median reply latency {np.median(ls) if ls else float('nan'):.2f}s")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "offline"
    if cmd == "scenes":
        build_scenes()
    elif cmd == "offline":
        if not (SCENES / "manifest.json").exists():
            build_scenes()
        res = evaluate()
        (OUT / "offline.json").write_text(json.dumps(res, indent=1))
        md = summarise(res)
        (OUT / "offline.md").write_text(md)
        print(md)
    elif cmd == "acoustic":
        acoustic()
    elif cmd == "asr":
        asr()
    else:
        sys.exit(__doc__)
