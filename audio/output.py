"""
audio/output.py — speaker-side enhancement for Raya's voice (24 kHz mono).

This evens out and protects what Raya says. It does not remove noise from
your room — that only happens on the microphone side (audio/voice.py).

Chain per 10 ms block:  loudness normaliser → volume/duck ramp → look-ahead limiter
The voice itself (Gemini's native audio) is not changed, only its level.

Also: optional ducking of other apps' audio while Raya speaks
(Windows via pycaw, Linux via pactl; not available on macOS).
"""

from __future__ import annotations

import math
import os
import platform
import subprocess
import threading
import time

import numpy as np

OUT_RATE = 24000
BLOCK = OUT_RATE // 100             # 10 ms


class OutputEnhancer:
    TARGET_DBFS = -18.0             # speech loudness target
    MAX_BOOST_DB = 12.0
    MAX_CUT_DB = 12.0
    SPEECH_GATE_DBFS = -50.0        # quieter blocks don't move the normaliser
    LOOKAHEAD = 48                  # 2 ms at 24 kHz
    CEILING = 10 ** (-1.0 / 20)     # −1 dBFS

    def __init__(self):
        self.enabled = True
        self.normalize = True
        self.volume = 1.0           # user volume, 0.0–1.5
        self._level_db: float | None = None
        self._norm_gain = 1.0       # linear, smoothed
        self._vol_gain = 1.0        # linear, smoothed (volume × duck)
        self._lim_gain = 1.0
        self._delay = np.zeros(self.LOOKAHEAD, dtype=np.float32)
        self.gain_reduction_db = 0.0
        self.clipped_blocks = 0

    def reset(self) -> None:
        """Forget the look-ahead tail (after a flush). Keeps learned loudness."""
        self._delay[:] = 0
        self._lim_gain = 1.0

    def process(self, x: np.ndarray) -> np.ndarray:
        """x: float32 in [-1, 1], any length (normally 240 samples)."""
        if not self.enabled:
            y = x * self.volume
            np.clip(y, -1.0, 1.0, out=y)
            return y.astype(np.float32)

        # 1) loudness normaliser: speech-gated level → slow gain toward target
        rms = float(np.sqrt(np.mean(x * x))) if len(x) else 0.0
        lvl = 20 * math.log10(rms + 1e-9)
        if self.normalize and lvl > self.SPEECH_GATE_DBFS:
            self._level_db = lvl if self._level_db is None else 0.97 * self._level_db + 0.03 * lvl
        want_db = 0.0
        if self.normalize and self._level_db is not None:
            want_db = max(-self.MAX_CUT_DB, min(self.MAX_BOOST_DB, self.TARGET_DBFS - self._level_db))
        want = 10 ** (want_db / 20)
        coef = 0.82 if want < self._norm_gain else 0.98      # ≈50 ms attack, ≈500 ms release
        start = self._norm_gain
        self._norm_gain = coef * self._norm_gain + (1 - coef) * want
        y = x * np.linspace(start, self._norm_gain, len(x), dtype=np.float32)

        # 2) volume ramp (user volume; ducking of Raya itself is never applied)
        vstart = self._vol_gain
        self._vol_gain += (self.volume - self._vol_gain) * 0.35            # ≈30 ms ramp
        y *= np.linspace(vstart, self._vol_gain, len(y), dtype=np.float32)

        # 3) look-ahead limiter (2 ms): gain is computed from samples not yet output
        buf = np.concatenate([self._delay, y])
        out = buf[: len(y)].copy()
        self._delay = buf[len(y):].astype(np.float32)
        peak = float(np.max(np.abs(buf))) if len(buf) else 0.0
        target = min(1.0, self.CEILING / peak) if peak > 0 else 1.0
        lstart = self._lim_gain
        self._lim_gain = target if target < self._lim_gain else self._lim_gain + (1.0 - self._lim_gain) * 0.08
        self._lim_gain = min(self._lim_gain, target) if target < 1.0 else self._lim_gain
        out *= np.linspace(lstart, self._lim_gain, len(out), dtype=np.float32)
        if np.max(np.abs(out)) > self.CEILING:
            self.clipped_blocks += 1
        np.clip(out, -self.CEILING, self.CEILING, out=out)
        self.gain_reduction_db = -20 * math.log10(max(self._lim_gain, 1e-6))
        return out

    def drain(self) -> np.ndarray:
        """Return the 2 ms still held in the look-ahead (end of a reply) and clear it."""
        tail = self._delay.copy()
        self._delay[:] = 0
        return tail

    @property
    def holding(self) -> bool:
        return bool(np.any(self._delay))


class Ducker:
    """Lower other apps' audio while Raya speaks. Opt-in; best effort; never raises."""

    DUCK_TO = 0.3           # other apps go to 30 % volume
    START_AFTER = 0.15      # s of Raya speech before ducking
    RESTORE_AFTER = 0.8     # s of silence before restoring

    def __init__(self):
        self.enabled = False
        self.platform = platform.system()
        self.available = self.platform in ("Windows", "Linux")
        self.reason = "" if self.available else "Not available on macOS (it has no public way to lower other apps)."
        self._ducked = False
        self._saved: dict = {}
        self._speaking_since: float | None = None
        self._silent_since: float | None = None
        self._lock = threading.Lock()

    def update(self, speaking: bool) -> None:
        """Call regularly (e.g. every 10 ms) from the audio worker."""
        if not (self.enabled and self.available):
            if self._ducked:
                self._restore_async()
            return
        now = time.monotonic()
        if speaking:
            self._silent_since = None
            self._speaking_since = self._speaking_since or now
            if not self._ducked and now - self._speaking_since >= self.START_AFTER:
                self._ducked = True
                threading.Thread(target=self._duck, daemon=True).start()
        else:
            self._speaking_since = None
            self._silent_since = self._silent_since or now
            if self._ducked and now - self._silent_since >= self.RESTORE_AFTER:
                self._restore_async()

    def _restore_async(self) -> None:
        self._ducked = False
        threading.Thread(target=self.restore, daemon=True).start()

    def _duck(self) -> None:
        with self._lock:
            try:
                if self.platform == "Windows":
                    from pycaw.pycaw import AudioUtilities
                    me = os.getpid()
                    for s in AudioUtilities.GetAllSessions():
                        if s.Process and s.Process.pid != me and s.SimpleAudioVolume:
                            key = ("win", s.Process.pid)
                            if key not in self._saved:
                                self._saved[key] = s.SimpleAudioVolume.GetMasterVolume()
                            s.SimpleAudioVolume.SetMasterVolume(self._saved[key] * self.DUCK_TO, None)
                elif self.platform == "Linux":
                    out = subprocess.run(["pactl", "-f", "json", "list", "sink-inputs"],
                                         capture_output=True, text=True, timeout=2).stdout
                    import json
                    me = str(os.getpid())
                    for si in json.loads(out or "[]"):
                        if si.get("properties", {}).get("application.process.id") == me:
                            continue
                        idx = si["index"]
                        vols = [int(v["value_percent"].rstrip("%")) for v in si["volume"].values()]
                        self._saved[("pa", idx)] = max(vols) if vols else 100
                        subprocess.run(["pactl", "set-sink-input-volume", str(idx),
                                        f"{int(self._saved[('pa', idx)] * self.DUCK_TO)}%"],
                                       capture_output=True, timeout=2)
            except Exception as e:
                print(f"[Audio] Ducking failed: {e}")

    def restore(self) -> None:
        with self._lock:
            saved, self._saved = self._saved, {}
            for key, vol in saved.items():
                try:
                    if key[0] == "win":
                        from pycaw.pycaw import AudioUtilities
                        for s in AudioUtilities.GetAllSessions():
                            if s.Process and s.Process.pid == key[1] and s.SimpleAudioVolume:
                                s.SimpleAudioVolume.SetMasterVolume(vol, None)
                    elif key[0] == "pa":
                        subprocess.run(["pactl", "set-sink-input-volume", str(key[1]), f"{vol}%"],
                                       capture_output=True, timeout=2)
                except Exception:
                    pass
