"""
tools/speaker_bench.py — compare local speaker-recognition models for the voice gate.

Uses Mini LibriSpeech dev-clean-2 (26 real speakers, CC BY 4.0) in
tools/bench_out/libri/, and the candidate .onnx models in models/.
Everything runs locally; nothing is uploaded.

Each speaker is "the user" in turn: enrolled from 5 utterances (+ the 3 extra
condition samples the app asks for: quiet, far, with background sound), using
the app's own threshold rule. Then every other utterance is tried — the same
speaker's (should be accepted) and the other 25 speakers' (should be rejected),
cut to what the gate actually checks (1.0 / 1.5 / 2.5 s of speech) and under
clean / fan / music / TV-voice / reverberant conditions. Everything passes
through the same mic processing the app uses (noise suppression + gain).

    .venv/bin/python tools/speaker_bench.py models     # model comparison
    .venv/bin/python tools/speaker_bench.py vad        # speech-detector comparison
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from audio.speaker_id import UNCERTAIN_BAND, _speech_only, profile_from_embeddings   # noqa: E402
import audio_bench as AB                                                             # noqa: E402

SR = 16000
OUT = ROOT / "tools" / "bench_out"
LIBRI = OUT / "libri" / "LibriSpeech" / "dev-clean-2"
CACHE = OUT / "libri_wav"
MODELS = {
    "ResNet34 (current)": "wespeaker_en_voxceleb_resnet34_LM.onnx",
    "CAM++": "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
    "TitaNet-small": "nemo_en_titanet_small.onnx",
}
PER_SPK = 22                 # utterances per speaker: 5 enrol + 3 condition + 14 test
RNG = np.random.default_rng(11)


# ── data ─────────────────────────────────────────────────────────────────────

def load_libri() -> dict[str, list[np.ndarray]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    data = {}
    for spk in sorted(p.name for p in LIBRI.iterdir() if p.is_dir()):
        flacs = sorted(LIBRI.glob(f"{spk}/*/*.flac"))
        # prefer utterances of at least 3 s
        clips = []
        for f in flacs:
            w = CACHE / (f.stem + ".wav")
            if not w.exists():
                subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", str(f), str(w)],
                               check=True, capture_output=True)
            x = read_pcm16_wav(w)
            if len(x) >= 3 * SR:
                clips.append(x)
            if len(clips) >= PER_SPK:
                break
        if len(clips) >= PER_SPK:
            data[spk] = clips
    return data


def read_pcm16_wav(p: Path) -> np.ndarray:
    """16-bit mono WAV, including WAVE_FORMAT_EXTENSIBLE (what afconvert writes for FLAC)."""
    b = p.read_bytes()
    i = 12
    while i + 8 <= len(b):
        cid, size = b[i:i + 4], int.from_bytes(b[i + 4:i + 8], "little")
        if cid == b"data":
            return np.frombuffer(b[i + 8:i + 8 + size], np.int16).astype(np.float32) / 32768
        i += 8 + size + (size & 1)
    raise ValueError(f"no data chunk in {p}")


def tv_voices(n: int) -> list[np.ndarray]:
    """Synthetic 'TV' speech from the Mac's voices (stand-ins for other people)."""
    lines = ["And now the weather for the weekend ahead.", "Scientists say the discovery could change everything.",
             "Stay tuned, we'll be right back after the break.", "The match ended in a draw after extra time.",
             "Turn off the lights and lock the front door please.", "What's on my calendar for tomorrow?"]
    voices = ["Samantha", "Daniel", "Karen", "Moira", "Rishi", "Tessa", "Alex", "Fred"]
    out = []
    for i in range(n):
        try:
            out.append(AB.tts(lines[i % len(lines)], voices[i % len(voices)]))
        except Exception:
            pass
    return out


# ── conditions (applied before the app's mic processing) ─────────────────────

def level(x, db):
    return AB.set_speech_level(x, db)


def cond(x: np.ndarray, kind: str) -> np.ndarray:
    x = level(x, -26)
    n = len(x)
    if kind == "clean":
        y = x
    elif kind == "quiet":
        y = level(x, -40) + AB.fan_noise(n) * 10 ** (-62 / 20)
    elif kind == "far":
        y = level(AB.speaker_path(x, AB.room_ir(25, 0.55, 0.35)), -34)
    elif kind == "fan":
        y = AB.mix_at_snr(x, AB.fan_noise(n), 10)
    elif kind == "music":
        y = AB.mix_at_snr(x, AB.music_bed(n), 10)
    elif kind == "tv":
        tv = TV_BED[: n] if len(TV_BED) >= n else np.resize(TV_BED, n)
        y = AB.mix_at_snr(x, tv, 10)
    elif kind == "reverb":
        y = AB.speaker_path(x, AB.room_ir(30, 0.6, 0.4))
    else:
        raise ValueError(kind)
    return np.clip(y, -1, 1).astype(np.float32)


def app_mic(x: np.ndarray) -> np.ndarray:
    """The app's mic chain (noise suppression standard + gain), float32 out."""
    y = AB.process_with_voice(x, {"noise_suppression": "standard", "echo_cancellation": True,
                                  "auto_gain": True})
    return y.astype(np.float32) / 32768.0


def first_speech(x: np.ndarray, seconds: float) -> np.ndarray:
    """What the gate hands to the model: audio from the start until `seconds` of speech."""
    sp = _speech_only(x)
    if len(sp) <= seconds * SR:
        return x
    # walk forward until enough speech frames are included
    frame = 320
    n = len(x) // frame
    e = 10 * np.log10(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    keep = e > max(e.max() - 35.0, -60.0)
    need = int(seconds * SR / frame)
    idx = np.nonzero(np.cumsum(keep) >= need)[0]
    end = (idx[0] + 1) * frame if len(idx) else len(x)
    return x[: end + int(0.3 * SR)]


# ── models ───────────────────────────────────────────────────────────────────

class Model:
    def __init__(self, file: str):
        import sherpa_onnx
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(ROOT / "models" / file),
                                                          num_threads=1, provider="cpu")
        self.ext = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def embed(self, x: np.ndarray) -> np.ndarray:
        s = self.ext.create_stream()
        s.accept_waveform(SR, np.asarray(x, np.float32))
        s.input_finished()
        e = np.array(self.ext.compute(s), np.float32)
        return e / (np.linalg.norm(e) + 1e-9)


def eer(tgt: np.ndarray, non: np.ndarray) -> float:
    ts = np.sort(np.concatenate([tgt, non]))
    best = 1.0
    for t in ts[:: max(1, len(ts) // 2000)]:
        frr = np.mean(tgt < t); far = np.mean(non >= t)
        best = min(best, max(frr, far))
    return float(best)


def bench_models():
    global TV_BED
    print("Loading LibriSpeech…", flush=True)
    data = load_libri()
    spks = list(data)
    print(f"{len(spks)} speakers × {PER_SPK} utterances", flush=True)
    tv = tv_voices(12)
    TV_BED = np.concatenate([level(t, -26) for t in tv]) if tv else AB.music_bed(SR * 20)

    # Prepare processed audio once (shared by all models)
    print("Running the app's mic processing on every clip…", flush=True)
    t0 = time.time()
    enrol = {s: [app_mic(level(c[:4 * SR], -26)) for c in data[s][:5]] for s in spks}
    cond_samples = {s: [app_mic(cond(data[s][5][:4 * SR], "quiet")),
                        app_mic(cond(data[s][6][:4 * SR], "far")),
                        app_mic(cond(data[s][7][:4 * SR], "fan"))] for s in spks}
    test_conds = ["clean", "fan", "music", "tv", "reverb", "far", "quiet"]
    tests = {}                      # (spk, cond) -> list of processed clips
    for s in spks:
        for i, c in enumerate(test_conds):
            clips = data[s][8:]
            tests[(s, c)] = [app_mic(cond(x[:5 * SR], c)) for x in clips[i * 2:(i + 1) * 2]] if c != "clean" \
                else [app_mic(cond(x[:5 * SR], c)) for x in clips]
    nonspeech = [app_mic(AB.fan_noise(SR * 4) * 0.05), app_mic(AB.music_bed(SR * 4) * 0.1)] + \
                [app_mic(AB.mix_at_snr(AB.music_bed(SR * 4), AB.fan_noise(SR * 4), 0) * 0.08)]
    tvclips = [app_mic(level(t, -26)) for t in tv]
    print(f"  done in {time.time() - t0:.0f} s", flush=True)

    results = {}
    lengths = (0.5, 0.75, 1.0, 1.5)
    for name, file in MODELS.items():
        if not (ROOT / "models" / file).exists():
            print(f"skip {name}: not downloaded"); continue
        print(f"\n== {name} ==", flush=True)
        m = Model(file)
        x = first_speech(tests[(spks[0], "clean")][0], 1.0)
        m.embed(x)
        ts = []
        for _ in range(10):
            t = time.perf_counter(); m.embed(x); ts.append((time.perf_counter() - t) * 1000)
        speed = float(np.median(ts))

        cent = {}
        for s in spks:
            E5 = np.stack([m.embed(c) for c in enrol[s]])
            E8 = np.concatenate([E5, np.stack([m.embed(c) for c in cond_samples[s]])])
            cent[s] = {k: (E.mean(0) / np.linalg.norm(E.mean(0))) for k, E in (("5", E5), ("8", E8))}

        res = {"ms_per_1s": speed, "by": {}}
        for secs in lengths:
            emb = {k: [m.embed(first_speech(c, secs)) for c in v] for k, v in tests.items()}
            ns_emb = [m.embed(first_speech(c, secs)) for c in nonspeech]
            tv_emb = [m.embed(first_speech(c, secs)) for c in tvclips]
            for pk in ("5", "8"):
                tgt_by, non, tv_sc, ns_sc = {}, [], [], []
                for s in spks:
                    c = cent[s][pk]
                    for (s2, cn), es in emb.items():
                        for e in es:
                            (tgt_by.setdefault(cn, []) if s2 == s else non).append(float(e @ c))
                    tv_sc += [float(e @ c) for e in tv_emb]
                    ns_sc += [float(e @ c) for e in ns_emb]
                tgt_all = np.concatenate([np.array(v) for v in tgt_by.values()])
                non = np.array(non)
                # threshold that lets through 1 % / 0.5 % of other people
                thr1 = float(np.quantile(non, 0.99)); thr05 = float(np.quantile(non, 0.995))
                key = f"{secs}s/enrol{pk}"
                r = {"EER": eer(tgt_all, non), "thr_far1": thr1, "thr_far05": thr05,
                     "FRR_far1": float(np.mean(tgt_all < thr1)),
                     "FRR_far1_by_cond": {k: float(np.mean(np.array(v) < thr1)) for k, v in tgt_by.items()},
                     "TV_far1": float(np.mean(np.array(tv_sc) >= thr1)),
                     "noise_far1": float(np.mean(np.array(ns_sc) >= thr1)),
                     "non_median": float(np.median(non)), "tgt_median": float(np.median(tgt_all))}
                res["by"][key] = r
                worst = max(r["FRR_far1_by_cond"].items(), key=lambda kv: kv[1])
                print(f"  {key:<14} EER {r['EER']*100:5.1f}%  @FAR1%: thr {thr1:.2f} FRR {r['FRR_far1']*100:5.1f}% "
                      f"(clean {r['FRR_far1_by_cond']['clean']*100:4.1f}%, worst {worst[0]} {worst[1]*100:4.1f}%)  "
                      f"TV {r['TV_far1']*100:4.1f}%  noise {r['noise_far1']*100:4.1f}%", flush=True)
        print(f"  speed: {speed:.0f} ms per 1 s of speech")
        results[name] = res
    (OUT / "speaker_bench.json").write_text(json.dumps(results, indent=1))
    print(f"\nSaved {OUT / 'speaker_bench.json'}")


# ── VAD comparison ───────────────────────────────────────────────────────────

def bench_vad():
    """Current detector (min-statistics + webrtcvad) vs Silero (sherpa-onnx):
    speech recall on real speech in noise, false triggers on music/fan/TV-less noise,
    and how long after speech ends each one notices."""
    import sherpa_onnx
    from audio.voice import VoiceConfig, VoiceProcessor
    data = load_libri()
    clips = [c[:6 * SR] for s in list(data)[:12] for c in data[s][8:11]]

    def truth(clean):
        n = len(clean) // 160
        e = 10 * np.log10(np.mean(clean[: n * 160].reshape(n, 160) ** 2, axis=1) + 1e-12)
        return e > (e.max() - 30)

    def current(x):
        vp = VoiceProcessor(VoiceConfig(noise_suppression="standard", echo_cancellation=True, auto_gain=True))
        m = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        flags = []
        for i in range(len(m) // 160):
            vp.process_reverse(np.zeros(160, np.int16))
            vp.process(m[i * 160:(i + 1) * 160])
            flags.append(vp.speech)
        return np.array(flags)

    def silero(x):
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(ROOT / "models" / "silero_vad.onnx")
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = 0.2
        cfg.silero_vad.min_speech_duration = 0.1
        cfg.silero_vad.window_size = 512
        cfg.sample_rate = SR
        cfg.num_threads = 1
        vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=30)
        flags = []
        y = np.clip(x, -1, 1).astype(np.float32)
        for i in range(len(y) // 160):
            vad.accept_waveform(y[i * 160:(i + 1) * 160])
            flags.append(vad.is_speech_detected())
        return np.array(flags)

    rows = {}
    for name, fn in (("current", current), ("silero", silero)):
        rec, fa, lag, cost = {}, {}, [], []
        for c in ("clean", "fan", "music"):
            r = []
            for x in clips:
                x = level(x, -26)
                y = cond(x, c) if c != "clean" else x
                t = time.perf_counter(); f = fn(y); cost.append((time.perf_counter() - t) / (len(y) / SR))
                tr = truth(x)[: len(f)]
                r.append(np.mean(f[tr]) if tr.any() else 1.0)
                if c == "clean":
                    last = np.nonzero(tr)[0][-1]
                    after = np.nonzero(~f[last:])[0]
                    lag.append((after[0] if len(after) else len(f) - last) * 10)
            rec[c] = float(np.mean(r))
        for c, gen in (("fan only", AB.fan_noise), ("music only", AB.music_bed)):
            y = gen(SR * 20) * (0.05 if c == "fan only" else 0.12)
            fa[c] = float(np.mean(fn(y)))
        tvb = level(np.concatenate(tv_voices(4)), -30)
        fa["TV voices"] = float(np.mean(fn(tvb)))
        rows[name] = {"speech_recall": rec, "false_trigger": fa, "end_lag_ms": float(np.median(lag)),
                      "cpu_pct_realtime": float(np.median(cost) * 100)}
        print(name, json.dumps(rows[name], indent=1), flush=True)
    (OUT / "vad_bench.json").write_text(json.dumps(rows, indent=1))


TV_BED = np.zeros(1, np.float32)

if __name__ == "__main__":
    {"models": bench_models, "vad": bench_vad}[sys.argv[1] if len(sys.argv) > 1 else "models"]()
