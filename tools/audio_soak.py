"""
tools/audio_soak.py — accelerated long-run test of the AudioEngine (no devices).

Drives the engine's callbacks with fake streams as fast as it can process,
simulating a long conversation: mic audio continuously, JARVIS replies of
3–12 s, frequent interrupts, settings changes and device "unplugs".
Reports memory, buffer depths and processing cost per simulated 10 minutes.

    python tools/audio_soak.py 120      # simulate 120 minutes
"""

import os
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from audio.engine import AudioEngine, AudioSettings, IN_FRAME, OUT_BLOCK, OUT_RATE   # noqa: E402


class FakeStream:
    def __init__(self):
        self.active, self.latency = True, 0.02

    def abort(self, ignore_errors=True):
        self.active = False

    def close(self, ignore_errors=True):
        self.active = False


def main(minutes: float):
    rng = np.random.default_rng(7)
    sent = [0]
    eng = AudioEngine(AudioSettings(), on_chunk=lambda d: sent.__setitem__(0, sent[0] + 1))
    eng._open_streams = lambda: (setattr(eng, "_in_stream", FakeStream()), setattr(eng, "_out_stream", FakeStream()))
    eng.refresh_devices = eng._open_streams
    eng.start()
    proc = psutil.Process()
    total_frames = int(minutes * 60 * 100)
    mic = (rng.standard_normal(IN_FRAME * 500) * 1500).astype(np.int16).reshape(-1, 1)
    reply = (np.sin(2 * np.pi * 220 * np.arange(OUT_RATE * 12) / OUT_RATE) * 8000).astype(np.int16)
    outbuf = np.zeros((OUT_BLOCK, 1), np.float32)
    next_reply = 500
    rss0 = proc.memory_info().rss
    t0 = time.perf_counter()
    print(f"{'sim min':>7} {'RSS MB':>7} {'ΔMB':>6} {'cap':>4} {'play ms':>8} {'raw':>4} {'dsp ms':>7} {'dropped':>7} {'sent':>8}")
    for f in range(total_frames):
        eng._in_cb(mic[(f % 500) * IN_FRAME:((f % 500) + 1) * IN_FRAME], IN_FRAME, None, None)
        eng._out_cb(outbuf, OUT_BLOCK, None, None)
        if f == next_reply:
            n = int(OUT_RATE * rng.uniform(3, 12))
            for i in range(0, n, 4800):                       # Gemini sends ~200 ms chunks
                eng.play(reply[i:i + 4800].tobytes())
            next_reply = f + int(rng.uniform(800, 2500))
        if f % 3000 == 1500 and rng.random() < 0.5:
            eng.flush()                                       # interrupt
        if f % 60000 == 30000:
            eng.apply_settings(AudioSettings(noise_suppression=rng.choice(["off", "light", "standard"])))
        if f % 90000 == 45000:
            eng._in_stream.active = False                     # simulated unplug
        if f % 20 == 0:
            while len(eng._cap) > 40:                         # let the worker keep up (real time would)
                time.sleep(0.0005)
        if f % 60000 == 0 and f:
            m = f / 6000
            rss = proc.memory_info().rss
            st = eng.stats()
            print(f"{m:7.0f} {rss / 1e6:7.1f} {(rss - rss0) / 1e6:6.1f} {len(eng._cap):4d} "
                  f"{st['playback_buffer_ms']:8.0f} {len(eng._raw_out):4d} {st['dsp_ms_per_10ms']:7.2f} "
                  f"{st['dropped_mic_blocks']:7d} {sent[0]:8d}", flush=True)
    eng.stop()
    wall = time.perf_counter() - t0
    print(f"Simulated {minutes:.0f} min in {wall / 60:.1f} min wall time. "
          f"Final RSS growth {(proc.memory_info().rss - rss0) / 1e6:.1f} MB.")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 120)
