"""
audio/voice.py — microphone voice processing (16 kHz mono, 10 ms frames).

Chain:  WebRTC APM (high-pass, echo cancellation, noise suppression)
        → speech-gated digital gain (our own AGC) → peak limiter → VAD / meters

Why our own AGC instead of WebRTC's: measured on this Mac's speakers, WebRTC's
AGC raises the residual echo by ~12 dB while JARVIS talks (echo reduction fell
from 17 dB to ~5 dB). Ours only adapts while the user is speaking and holds
still while JARVIS is speaking.

Noise suppression levels (WebRTC's suppressor is on/off only):
    off       no suppression (echo cancellation can still run)
    light     60 % suppressed + 40 % unsuppressed signal
    standard  WebRTC noise suppression

Nothing here touches devices or threads; AudioEngine feeds frames in.
Any processing error switches the processor to bypass and records why.
"""

from __future__ import annotations

import collections
import math
import time
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
FRAME = SAMPLE_RATE // 100          # 10 ms
NS_LEVELS = ("off", "light", "standard")


@dataclass(frozen=True)
class VoiceConfig:
    enabled: bool = True            # False = bypass everything
    noise_suppression: str = "standard"
    echo_cancellation: bool = True
    auto_gain: bool = True
    agc_target_dbfs: float = -23.0
    high_pass: bool = True


def _db(x: float) -> float:
    return 10.0 * math.log10(x + 1e-12)


class SpeechGatedAGC:
    """Slow digital gain toward a speech target, frozen outside the user's speech."""

    MAX_GAIN_DB = 24.0
    MIN_GAIN_DB = -6.0
    UP_DB_PER_S = 6.0               # rise slowly (no pumping)
    DOWN_DB_PER_S = 40.0            # fall fast (no blasting)

    def __init__(self, target_dbfs: float = -23.0):
        self.target = target_dbfs
        self.gain_db = 0.0
        self._level_db = None

    NOISE_CEILING_DBFS = -55.0      # never lift the background above this

    def process(self, x: np.ndarray, speech: bool, frozen: bool,
                noise_floor_db: float | None = None) -> np.ndarray:
        start = 10 ** (self.gain_db / 20)
        if speech and not frozen:
            lvl = _db(float(np.mean(x * x)))
            self._level_db = lvl if self._level_db is None else 0.9 * self._level_db + 0.1 * lvl
            ceiling = self.MAX_GAIN_DB
            if noise_floor_db is not None:
                ceiling = min(ceiling, max(0.0, self.NOISE_CEILING_DBFS - noise_floor_db))
            want = min(ceiling, max(self.MIN_GAIN_DB, self.target - self._level_db))
            step = (self.UP_DB_PER_S if want > self.gain_db else self.DOWN_DB_PER_S) * FRAME / SAMPLE_RATE
            self.gain_db += max(-step, min(step, want - self.gain_db))
        end = 10 ** (self.gain_db / 20)
        return x * np.linspace(start, end, len(x), endpoint=False, dtype=np.float32)


class PeakLimiter:
    """Instant-attack, smooth-release limiter (no look-ahead: zero added latency)."""

    def __init__(self, ceiling_dbfs: float = -1.0, release_ms: float = 80.0, sample_rate: int = SAMPLE_RATE):
        self.ceiling = 10 ** (ceiling_dbfs / 20)
        self.gain = 1.0
        self._rel = math.exp(-1.0 / (release_ms / 1000 * sample_rate / 160))   # per 160-sample block

    def process(self, x: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(x))) if len(x) else 0.0
        target = min(1.0, self.ceiling / peak) if peak > 0 else 1.0
        start = self.gain
        self.gain = target if target < self.gain else 1.0 - (1.0 - self.gain) * self._rel
        self.gain = min(self.gain, target)
        g = np.linspace(start, self.gain, len(x), dtype=np.float32)
        y = x * g
        np.clip(y, -self.ceiling, self.ceiling, out=y)      # belt and braces
        return y


class VoiceProcessor:
    def __init__(self, config: VoiceConfig | None = None):
        self.config = config or VoiceConfig()
        self.error: str | None = None          # set when processing fell back to bypass
        self.frame_ms = 0.0                    # moving average processing cost
        self.level_dbfs = -120.0               # output level of the last frame
        self.speech = False                    # VAD decision (with hangover)
        self._vad_hist = 0
        self._floor_db = -60.0
        self._smooth_db = -60.0
        self._floor_hist = collections.deque(maxlen=150)     # 1.5 s
        self._recent = collections.deque(maxlen=5)
        self._apm = None
        self._apm_dry = None
        self._vad = None
        self._agc = SpeechGatedAGC(self.config.agc_target_dbfs)
        self._limiter = PeakLimiter()
        self._delay_ms = 60
        self._build()

    # ── configuration ────────────────────────────────────────────────────

    def _build(self) -> None:
        c = self.config
        self.error = None
        self._apm = self._apm_dry = None
        self._agc.target = c.agc_target_dbfs
        try:
            import webrtcvad
            self._vad = webrtcvad.Vad(2)
        except Exception as e:                 # VAD is optional
            self._vad = None
            print(f"[Voice] VAD unavailable: {e}")
        if not c.enabled:
            return
        needs_apm = c.echo_cancellation or c.noise_suppression != "off" or c.high_pass
        if not needs_apm:
            return
        try:
            from livekit import rtc
            self._rtc = rtc
            self._apm = rtc.AudioProcessingModule(
                echo_cancellation=c.echo_cancellation,
                noise_suppression=c.noise_suppression != "off",
                high_pass_filter=c.high_pass,
                auto_gain_control=False,
            )
            if c.noise_suppression == "light":
                self._apm_dry = rtc.AudioProcessingModule(
                    echo_cancellation=c.echo_cancellation, noise_suppression=False,
                    high_pass_filter=c.high_pass, auto_gain_control=False,
                )
        except Exception as e:
            self._fail(f"WebRTC audio processing unavailable ({e})")

    def configure(self, config: VoiceConfig) -> None:
        """Apply new settings. Call from the thread that calls process()."""
        if config != self.config:
            self.config = config
            self._build()

    def set_stream_delay_ms(self, ms: int) -> None:
        self._delay_ms = max(0, min(500, int(ms)))

    @property
    def active(self) -> bool:
        return self.config.enabled and self.error is None

    def _fail(self, why: str) -> None:
        if self.error is None:
            print(f"[Voice] Falling back to bypass: {why}")
        self.error = why
        self._apm = self._apm_dry = None

    # ── processing ───────────────────────────────────────────────────────

    def process_reverse(self, frame: np.ndarray) -> None:
        """Feed 10 ms of what the speaker is playing (int16, 16 kHz) for echo cancellation."""
        if self._apm is None or not self.config.echo_cancellation:
            return
        try:
            data = frame.tobytes()
            self._apm.process_reverse_stream(self._rtc.AudioFrame(data, SAMPLE_RATE, 1, FRAME))
            if self._apm_dry is not None:
                self._apm_dry.process_reverse_stream(self._rtc.AudioFrame(data, SAMPLE_RATE, 1, FRAME))
        except Exception as e:
            self._fail(f"echo reference failed ({e})")

    def process(self, frame: np.ndarray, far_active: bool = False) -> np.ndarray:
        """Process 10 ms of mic audio (int16, 16 kHz). Returns int16."""
        t0 = time.perf_counter()
        out = frame
        if self.config.enabled and self.error is None:
            try:
                x = frame
                if self._apm is not None:
                    x = self._run_apm(self._apm, frame)
                    if self._apm_dry is not None:
                        dry = self._run_apm(self._apm_dry, frame)
                        x = (0.6 * x.astype(np.float32) + 0.4 * dry.astype(np.float32)).astype(np.int16)
                xf = x.astype(np.float32) / 32768.0
                speech = self._detect(x)
                if self.config.auto_gain:
                    xf = self._agc.process(xf, speech, frozen=far_active,
                                           noise_floor_db=self._floor_db)
                xf = self._limiter.process(xf)
                out = (xf * 32767.0).astype(np.int16)
            except Exception as e:
                self._fail(f"processing error ({e})")
                out = frame
        else:
            self._detect(frame)
        self.level_dbfs = _db(float(np.mean((out.astype(np.float32) / 32768.0) ** 2)))
        dt = (time.perf_counter() - t0) * 1000
        self.frame_ms = 0.98 * self.frame_ms + 0.02 * dt
        return out

    def _run_apm(self, apm, frame: np.ndarray) -> np.ndarray:
        if self.config.echo_cancellation:
            apm.set_stream_delay_ms(self._delay_ms)
        f = self._rtc.AudioFrame(frame.tobytes(), SAMPLE_RATE, 1, FRAME)
        apm.process_stream(f)
        # Polarity is preserved (verified with broadband signals for every
        # stage combination). Added delay: 8 ms echo canceller + 6 ms suppressor.
        return np.frombuffer(f.data, dtype=np.int16).copy()

    def _detect(self, x: np.ndarray) -> bool:
        """Speech = WebRTC VAD agreeing with a level gate above the noise floor.

        webrtcvad alone flags steady room noise and music as speech (measured
        false-alarm rates of 30–100 %, and ~50 % in a quiet room on this Mac),
        which would make the AGC turn noise up. The noise floor is the minimum
        of the smoothed level over the last 1.5 s (minimum statistics).
        """
        lvl = _db(float(np.mean((x.astype(np.float32) / 32768.0) ** 2)))
        self._smooth_db = 0.8 * self._smooth_db + 0.2 * lvl
        self._floor_hist.append(self._smooth_db)
        floor = min(self._floor_hist)
        self._floor_db = floor
        raw = False
        if self._vad is not None and self._smooth_db > floor + 12.0 and self._smooth_db > -55.0:
            try:
                raw = self._vad.is_speech(x.tobytes(), SAMPLE_RATE)
            except Exception:
                raw = False
        self._recent.append(raw)
        onset = sum(self._recent) >= 3                   # 3 of the last 5 frames
        # hangover: stay "speech" for 200 ms after the last confirmed frame
        self._vad_hist = 20 if onset else max(0, self._vad_hist - 1)
        self.speech = self._vad_hist > 0
        return onset
