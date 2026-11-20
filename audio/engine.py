"""
audio/engine.py — the one place JARVIS captures and plays audio.

Threads
  PortAudio input callback   copies 10 ms mic blocks into a bounded ring
  PortAudio output callback  pulls 10 ms from the playback ring, taps what it played
  DSP worker (this module)   echo reference → voice processing → 20 ms chunks to Gemini;
                             JARVIS audio → output enhancement → playback ring;
                             meters, latency stats, device watchdog
Nothing here runs on the Qt thread or the asyncio loop. The loop hands audio in
with play() (non-blocking) and receives mic chunks through `on_chunk`.

Formats: mic 16 kHz mono int16 (what Gemini Live expects), speaker 24 kHz mono
int16 (what Gemini Live sends). PortAudio/CoreAudio converts to the device rate.
"""

from __future__ import annotations

import collections
import math
import threading
import time
from dataclasses import asdict, dataclass, fields

import numpy as np

from audio.output import BLOCK as OUT_BLOCK, OUT_RATE, Ducker, OutputEnhancer
from audio.voice import FRAME as IN_FRAME, SAMPLE_RATE as IN_RATE, VoiceConfig, VoiceProcessor

SEND_FRAMES = 2                      # 2 × 10 ms = 20 ms chunks to Gemini
MAX_PLAYBACK_S = 120                 # cap on queued JARVIS audio (a very long answer)
PREBUFFER_MS = 60                    # start playing once this much is queued
HALF_DUPLEX_TAIL_MS = 300            # keep the mic closed this long after the speaker goes quiet
SPECTRUM_BANDS = 24


@dataclass
class AudioSettings:
    processing_enabled: bool = True      # master switch; False = full bypass
    voice_isolation: bool = True
    noise_suppression: str = "standard"  # off | light | standard
    echo_cancellation: bool = True
    auto_gain: bool = True
    input_device: str | None = None      # device name; None = system default
    output_enhancement: bool = True
    normalization: bool = True
    ducking: bool = False
    output_device: str | None = None
    output_volume: float = 1.0
    full_duplex: bool = False            # experimental: keep the mic open while JARVIS speaks

    @classmethod
    def from_dict(cls, d: dict | None) -> "AudioSettings":
        names = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in (d or {}).items() if k in names})
        if s.noise_suppression not in ("off", "light", "standard"):
            s.noise_suppression = "standard"
        s.output_volume = float(min(1.5, max(0.0, s.output_volume)))
        return s

    def to_dict(self) -> dict:
        return asdict(self)

    def voice_config(self) -> VoiceConfig:
        on = self.processing_enabled and self.voice_isolation
        return VoiceConfig(enabled=on, noise_suppression=self.noise_suppression,
                           echo_cancellation=self.echo_cancellation, auto_gain=self.auto_gain)


def list_devices() -> dict:
    """{'input': [names], 'output': [names], 'default_input': name, 'default_output': name}"""
    import sounddevice as sd
    devs = sd.query_devices()
    di, do = sd.default.device
    res = {"input": [], "output": [], "default_input": None, "default_output": None}
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0:
            res["input"].append(d["name"])
            if i == di:
                res["default_input"] = d["name"]
        if d["max_output_channels"] > 0:
            res["output"].append(d["name"])
            if i == do:
                res["default_output"] = d["name"]
    return res


def _find_device(name: str | None, kind: str):
    """Index of the named device, or None (= system default) if missing."""
    if not name:
        return None
    import sounddevice as sd
    for i, d in enumerate(sd.query_devices()):
        if d["name"] == name and d[f"max_{kind}_channels"] > 0:
            return i
    return None


class AudioEngine:
    def __init__(self, settings: AudioSettings | None = None, on_chunk=None, should_send=None):
        self.settings = settings or AudioSettings()
        self.on_chunk = on_chunk                  # fn(bytes) — 20 ms of 16 kHz int16 for Gemini
        self.should_send = should_send or (lambda: True)
        self.voice = VoiceProcessor(self.settings.voice_config())
        self.enhancer = OutputEnhancer()
        self.ducker = Ducker()
        self._apply_output_settings()

        # rings (bounded)
        self._cap = collections.deque(maxlen=100)            # 1 s of 10 ms mic blocks
        self._raw_out = collections.deque()                  # JARVIS audio not yet enhanced (bytes)
        self._raw_out_samples = 0
        self._raw_lock = threading.Lock()
        self._gen = 0                                        # bumped by flush(); stale audio is dropped
        self._play = collections.deque()                     # enhanced float32 blocks
        self._play_samples = 0
        self._play_lock = threading.Lock()
        self._ref = collections.deque(maxlen=200)            # played blocks for the echo canceller
        self._wake = threading.Event()

        self._in_stream = None
        self._out_stream = None
        self._worker = None
        self._running = False
        self._prebuffering = True
        self._last_played_audio = 0.0                        # monotonic time of last non-silent output
        self._send_buf: list[np.ndarray] = []
        self._resample_ref = None
        self._pending_settings: AudioSettings | None = None
        self._stream_lock = threading.Lock()

        # status for UI
        self.input_error: str | None = None
        self.output_error: str | None = None
        self.dropped_in_blocks = 0
        self.underruns = 0
        self._levels = {"mic_db": -120.0, "out_db": -120.0, "speech": False, "playing": False,
                        "spectrum": [0.0] * SPECTRUM_BANDS, "sending": False}
        self._reply_lat = collections.deque(maxlen=20)
        self._speech_end_t: float | None = None
        self._was_playing = False
        self._mic_test: dict | None = None
        self.on_utterance = None                             # fn(np.ndarray int16) — finished user speech
        self._utt: list[np.ndarray] = []
        self._utt_prev_speech = False
        self._spec_buf: list[np.ndarray] = []
        self.round_trip_ms: float | None = None
        self._click: dict | None = None
        self._click_fire = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._open_streams()
        self._worker = threading.Thread(target=self._run, name="AudioDSP", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        self._close_streams()
        self.ducker.restore()

    def _open_streams(self) -> None:
        import sounddevice as sd
        with self._stream_lock:
            self._close_streams_locked()
            s = self.settings
            # input
            try:
                dev = _find_device(s.input_device, "input")
                self._in_stream = sd.InputStream(
                    samplerate=IN_RATE, channels=1, dtype="int16", blocksize=IN_FRAME,
                    device=dev, latency="low", callback=self._in_cb,
                )
                self._in_stream.start()
                self.input_error = None
                if s.input_device and dev is None:
                    self.input_error = f"'{s.input_device}' not found — using the system default"
            except Exception as e:
                self._in_stream = None
                self.input_error = f"Microphone unavailable: {e}"
                print(f"[Audio] {self.input_error}")
            # output
            try:
                dev = _find_device(s.output_device, "output")
                self._out_stream = sd.OutputStream(
                    samplerate=OUT_RATE, channels=1, dtype="float32", blocksize=OUT_BLOCK,
                    device=dev, latency="low", callback=self._out_cb,
                )
                self._out_stream.start()
                self.output_error = None
                if s.output_device and dev is None:
                    self.output_error = f"'{s.output_device}' not found — using the system default"
            except Exception as e:
                self._out_stream = None
                self.output_error = f"Speaker unavailable: {e}"
                print(f"[Audio] {self.output_error}")
            self._update_delay()

    def _close_streams(self) -> None:
        with self._stream_lock:
            self._close_streams_locked()

    def _close_streams_locked(self) -> None:
        for st in (self._in_stream, self._out_stream):
            if st is not None:
                try:
                    st.abort(ignore_errors=True)
                    st.close(ignore_errors=True)
                except Exception:
                    pass
        self._in_stream = self._out_stream = None

    def refresh_devices(self) -> None:
        """Re-scan devices (PortAudio only sees new devices after a restart) and reopen."""
        import sounddevice as sd
        with self._stream_lock:
            self._close_streams_locked()
            try:
                sd._terminate()
                sd._initialize()
            except Exception as e:
                print(f"[Audio] Device rescan failed: {e}")
        self._open_streams()

    def _update_delay(self) -> None:
        """Tell the echo canceller roughly how late the echo arrives.

        PortAudio's *input* latency is not trustworthy here (it reports ~300 ms
        for 16 kHz capture on a MacBook; a click test measures ~64 ms round
        trip), so use a measured round trip when we have one, otherwise the
        output latency plus a typical input path. WebRTC refines it itself.
        """
        try:
            if self.round_trip_ms:
                hint = self.round_trip_ms
            else:
                lat_out = self._out_stream.latency * 1000 if self._out_stream else 20
                hint = lat_out + 30
            self.voice.set_stream_delay_ms(int(min(200, max(10, hint))))
        except Exception:
            pass

    def measure_round_trip(self, clicks: int = 3) -> float | None:
        """Play short clicks and time them back through the mic (speaker → room → mic).

        Blocks the calling thread for ~1.5 s (call it off the UI thread).
        Returns the median in ms, or None if the mic never heard a click.
        """
        if not self._running or self._in_stream is None or self._out_stream is None:
            return None
        results = []
        for _ in range(clicks):
            time.sleep(0.35)                              # let the previous click die away
            self._click = {"t_out": None, "t_in": None,
                           "noise": float(10 ** (self.voice._floor_db / 20)) if self.voice else 0.001}
            self._click_fire = True
            t0 = time.monotonic()
            while time.monotonic() - t0 < 1.0:
                c = self._click
                if c["t_in"] and c["t_out"]:
                    results.append((c["t_in"] - c["t_out"]) * 1000)
                    break
                time.sleep(0.005)
        self._click = None
        if not results:
            return None
        results.sort()
        self.round_trip_ms = results[len(results) // 2]
        self._update_delay()
        return self.round_trip_ms

    # ── settings ─────────────────────────────────────────────────────────

    def apply_settings(self, s: AudioSettings) -> None:
        """Apply settings from any thread. Device changes reopen streams; the rest is instant."""
        devices_changed = (s.input_device != self.settings.input_device
                           or s.output_device != self.settings.output_device)
        self._pending_settings = s
        self._wake.set()
        if devices_changed and self._running:
            self.settings = s
            threading.Thread(target=self._open_streams, daemon=True).start()

    def _apply_output_settings(self) -> None:
        s = self.settings
        on = s.processing_enabled and s.output_enhancement
        self.enhancer.enabled = on
        self.enhancer.normalize = on and s.normalization
        self.enhancer.volume = s.output_volume
        self.ducker.enabled = s.processing_enabled and s.ducking

    # ── API for the asyncio side ─────────────────────────────────────────

    def play(self, pcm24: bytes) -> None:
        """Queue JARVIS audio (24 kHz int16). Never blocks."""
        if not pcm24:
            return
        n = len(pcm24) // 2
        if (self._raw_out_samples + self._play_samples + n) > MAX_PLAYBACK_S * OUT_RATE:
            return                                   # absurdly long backlog: drop, don't grow forever
        with self._raw_lock:
            self._raw_out.append(pcm24)
            self._raw_out_samples += n
        self._wake.set()

    def flush(self) -> None:
        """Stop JARVIS immediately (interrupt). Fades the last 10 ms to avoid a click."""
        with self._raw_lock:
            self._raw_out.clear()
            self._raw_out_samples = 0
            self._gen += 1
        with self._play_lock:
            self._gen += 1
            self.enhancer.reset()                       # drop the look-ahead (future audio)
            # fade the very next 10 ms that would have played, so there is no click
            nxt = self._play[0][:OUT_BLOCK] if self._play else np.zeros(0, np.float32)
            fade = nxt * np.linspace(1.0, 0.0, len(nxt), dtype=np.float32) if len(nxt) else nxt
            self._play.clear()
            self._play_samples = 0
            if len(fade):
                self._play.append(fade)
                self._play_samples = len(fade)
        self._prebuffering = False

    @property
    def is_playing(self) -> bool:
        """True while JARVIS audio is queued or the speaker played it within the last 80 ms."""
        return (self._raw_out_samples + self._play_samples) > 0 or \
            (time.monotonic() - self._last_played_audio) < 0.08

    def levels(self) -> dict:
        """Latest meter snapshot (cheap; safe to call at 60 fps from the UI)."""
        return self._levels

    def stats(self) -> dict:
        lat_out = None
        try:
            lat_out = self._out_stream.latency * 1000 if self._out_stream else None
        except Exception:
            pass
        rl = sorted(self._reply_lat)
        return {
            "round_trip_ms": self.round_trip_ms,          # measured speaker → mic
            "output_latency_ms": lat_out,
            "processing_delay_ms": 14.0 if self.voice.active and self.settings.noise_suppression != "off"
                                   else (8.0 if self.voice.active else 0.0),
            "dsp_ms_per_10ms": self.voice.frame_ms,
            "dsp_cpu_pct": self.voice.frame_ms / 10 * 100,
            "playback_buffer_ms": self._play_samples / OUT_RATE * 1000,
            "reply_latency_ms": rl[len(rl) // 2] * 1000 if rl else None,
            "processing_active": self.voice.active,
            "processing_error": self.voice.error,
            "input_error": self.input_error,
            "output_error": self.output_error,
            "dropped_mic_blocks": self.dropped_in_blocks,
            "underruns": self.underruns,
            "limiter_db": self.enhancer.gain_reduction_db,
            "ducking_available": self.ducker.available,
            "ducking_reason": self.ducker.reason,
        }

    # ── tests from the settings panel ────────────────────────────────────

    def capture(self, seconds: float, on_done) -> bool:
        """Record `seconds` of processed mic audio for local use (e.g. voice enrolment).

        Nothing captured is sent to Gemini or played back. on_done(float32 array,
        16 kHz) is called on the audio worker thread. Returns False if busy/no mic.
        """
        if self._mic_test is not None or self._in_stream is None:
            return False
        self._mic_test = {"frames": [], "need": int(seconds * 100), "on_done": on_done, "playback": False}
        return True

    def start_mic_test(self, seconds: float = 5.0, on_done=None) -> None:
        """Record `seconds` of processed mic (not sent to Gemini), then play it back."""
        self._mic_test = {"frames": [], "need": int(seconds * 100), "on_done": on_done}

    @property
    def mic_test_running(self) -> bool:
        return self._mic_test is not None

    def play_test_tone(self) -> None:
        """A short two-note chime through the full output chain."""
        t = np.arange(int(0.35 * OUT_RATE)) / OUT_RATE
        env = np.minimum(1, t * 60) * np.exp(-t * 7)
        a = 0.25 * env * np.sin(2 * np.pi * 660 * t)
        b = 0.25 * env * np.sin(2 * np.pi * 880 * t)
        pcm = (np.concatenate([a, np.zeros(1200), b]) * 32767).astype(np.int16)
        self.play(pcm.tobytes())

    # ── PortAudio callbacks (keep them tiny) ─────────────────────────────

    def _in_cb(self, indata, frames, time_info, status) -> None:
        c = self._click
        if c is not None and c["t_out"] is not None and c["t_in"] is None:
            now = time.monotonic()
            peak = float(np.max(np.abs(indata[:, 0]))) / 32768.0
            # nothing can come back faster than ~15 ms (device buffers + air)
            if now - c["t_out"] > 0.015 and peak > max(0.05, 6 * c["noise"]):
                c["t_in"] = now
        if len(self._cap) == self._cap.maxlen:
            self.dropped_in_blocks += 1                # deque drops the oldest itself
        self._cap.append(indata[:, 0].copy())
        self._wake.set()

    def _out_cb(self, outdata, frames, time_info, status) -> None:
        out = outdata[:, 0]
        filled = 0
        if not self._prebuffering:
            with self._play_lock:
                while filled < frames and self._play:
                    blk = self._play[0]
                    take = min(frames - filled, len(blk))
                    out[filled:filled + take] = blk[:take]
                    if take == len(blk):
                        self._play.popleft()
                    else:
                        self._play[0] = blk[take:]
                    self._play_samples -= take
                    filled += take
            if filled < frames and filled > 0:
                self.underruns += 1
        out[filled:] = 0.0
        if self._click_fire and self._click is not None:
            self._click_fire = False
            out[:48] += 0.6
            self._click["t_out"] = time.monotonic()
        if filled:
            self._last_played_audio = time.monotonic()
        if self._play_samples == 0 and filled < frames:
            self._prebuffering = True
        self._ref.append(out.copy())                   # what the speaker actually played

    # ── worker ───────────────────────────────────────────────────────────

    def _run(self) -> None:
        import soxr
        self._resample_ref = soxr.ResampleStream(OUT_RATE, IN_RATE, 1, dtype="float32")
        ref_pending = np.zeros(0, dtype=np.float32)
        last_watch = time.monotonic()
        backoff = 1.0
        while self._running:
            if not self._cap:
                self._wake.wait(0.02)
            self._wake.clear()
            try:
                if self._pending_settings is not None:
                    s, self._pending_settings = self._pending_settings, None
                    self.settings = s
                    self.voice.configure(s.voice_config())
                    self._apply_output_settings()
                    self._update_delay()

                # No speaker: drop JARVIS audio rather than let it pile up and
                # keep the half-duplex gate closed forever.
                if self._out_stream is None:
                    with self._raw_lock:
                        self._raw_out.clear()
                        self._raw_out_samples = 0
                    with self._play_lock:
                        self._play.clear()
                        self._play_samples = 0

                # JARVIS audio → enhancer → playback ring
                while True:
                    with self._raw_lock:
                        if not self._raw_out:
                            break
                        raw = self._raw_out.popleft()
                        self._raw_out_samples -= len(raw) // 2
                        gen = self._gen
                    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    for i in range(0, len(x), OUT_BLOCK):
                        y = self.enhancer.process(x[i:i + OUT_BLOCK])
                        with self._play_lock:
                            if gen != self._gen:          # interrupted while we were processing
                                break
                            self._play.append(y)
                            self._play_samples += len(y)
                # end of a reply: push out the last 2 ms held by the limiter's look-ahead
                if not self._raw_out and self._play_samples < OUT_BLOCK and self.enhancer.holding:
                    with self._play_lock:
                        tail = self.enhancer.drain()
                        self._play.append(tail)
                        self._play_samples += len(tail)
                    self._prebuffering = False
                if self._prebuffering and self._play_samples >= PREBUFFER_MS * OUT_RATE / 1000:
                    self._prebuffering = False

                # played audio → 16 kHz echo reference (+ output meters)
                while self._ref:
                    blk = self._ref.popleft()
                    self._update_out_meter(blk)
                    r = self._resample_ref.resample_chunk(blk)
                    ref_pending = np.concatenate([ref_pending, r])
                playing = self.is_playing
                self.ducker.update(playing)

                # mic blocks → (reference frame) → voice processing → Gemini.
                # At most 10 per pass so speaker output and the watchdog are
                # never starved if capture ever gets ahead (e.g. after a stall).
                for _ in range(min(len(self._cap), 10)):
                    blk = self._cap.popleft()
                    playing = self.is_playing
                    if len(ref_pending) >= IN_FRAME:
                        ref = ref_pending[:IN_FRAME]
                        ref_pending = ref_pending[IN_FRAME:]
                    else:
                        ref = np.zeros(IN_FRAME, np.float32)
                    if len(ref_pending) > IN_RATE // 2:        # never let the reference run away
                        ref_pending = ref_pending[-IN_RATE // 10:]
                    self.voice.process_reverse((np.clip(ref, -1, 1) * 32767).astype(np.int16))
                    out = self.voice.process(blk, far_active=playing)
                    self._route_mic(out, playing)
            except Exception as e:
                print(f"[Audio] Worker error: {e}")
                time.sleep(0.05)

            now = time.monotonic()
            if now - last_watch > 1.0:
                last_watch = now
                if self._streams_dead():
                    print("[Audio] Audio device lost — reopening…")
                    self.input_error = self.input_error or "Device disconnected — reconnecting…"
                    time.sleep(min(backoff, 10))
                    backoff = min(backoff * 2, 10)
                    try:
                        self.refresh_devices()
                    except Exception as e:
                        print(f"[Audio] Reopen failed: {e}")
                else:
                    backoff = 1.0

    def _streams_dead(self) -> bool:
        if not self._running:
            return False
        for st in (self._in_stream, self._out_stream):
            if st is None:
                return True
            try:
                if not st.active:
                    return True
            except Exception:
                return True
        return False

    def _route_mic(self, out: np.ndarray, playing: bool) -> None:
        mic_db = self.voice.level_dbfs
        speech = self.voice.speech
        self._update_mic_spectrum(out)
        # mic test: capture processed audio locally, never send it
        if self._mic_test is not None:
            t = self._mic_test
            t["frames"].append(out)
            if len(t["frames"]) >= t["need"]:
                self._mic_test = None
                pcm16 = np.concatenate(t["frames"]).astype(np.float32) / 32768.0
                if t.get("playback", True):
                    import soxr
                    pcm24 = soxr.resample(pcm16, IN_RATE, OUT_RATE)
                    self.play((np.clip(pcm24, -1, 1) * 32767).astype(np.int16).tobytes())
                if t["on_done"]:
                    try:
                        t["on_done"](pcm16) if not t.get("playback", True) else t["on_done"]()
                    except Exception as e:
                        print(f"[Audio] capture callback failed: {e}")
            self._levels = {**self._levels, "mic_db": mic_db, "speech": speech, "sending": False}
            return

        # half-duplex: keep the mic closed while JARVIS is audible, plus a short tail
        recently = (time.monotonic() - self._last_played_audio) < HALF_DUPLEX_TAIL_MS / 1000
        gate_closed = (not self.settings.full_duplex or not self.voice.active
                       or not self.settings.echo_cancellation) and (playing or recently)
        sending = (not gate_closed) and self.should_send()

        # reply-latency bookkeeping: end of user speech → first JARVIS audio
        if sending and not speech and self._levels.get("speech"):
            self._speech_end_t = time.monotonic()
        if playing and not self._was_playing and self._speech_end_t:
            dt = time.monotonic() - self._speech_end_t
            if dt < 15:
                self._reply_lat.append(dt)
            self._speech_end_t = None
        self._was_playing = playing

        self._levels = {**self._levels, "mic_db": mic_db, "speech": speech, "sending": sending}
        self._collect_utterance(out, speech, sending)
        if not sending:
            self._send_buf.clear()
            return
        self._send_buf.append(out)
        if len(self._send_buf) >= SEND_FRAMES and self.on_chunk:
            data = np.concatenate(self._send_buf).tobytes()
            self._send_buf.clear()
            try:
                self.on_chunk(data)
            except Exception:
                pass

    UTTERANCE_MIN_S = 1.0
    UTTERANCE_MAX_S = 10.0

    def _collect_utterance(self, out: np.ndarray, speech: bool, sending: bool) -> None:
        """Hand each finished stretch of the user's speech to on_utterance (speaker check).

        Only audio that is actually being sent to Gemini counts, so mic tests,
        muted time and JARVIS's own voice are never checked.
        """
        if self.on_utterance is None:
            return
        if sending and speech:
            self._utt.append(out.copy())
        ended = self._utt and ((self._utt_prev_speech and not speech) or not sending
                               or len(self._utt) >= self.UTTERANCE_MAX_S * 100)
        self._utt_prev_speech = speech and sending
        if ended:
            frames, self._utt = self._utt, []
            if len(frames) >= self.UTTERANCE_MIN_S * 100:
                try:
                    self.on_utterance(np.concatenate(frames))
                except Exception as e:
                    print(f"[Audio] utterance callback failed: {e}")

    def _update_mic_spectrum(self, out: np.ndarray) -> None:
        """24-band mic spectrum every 20 ms for the voice ribbon (cheap: one 320-point FFT)."""
        self._spec_buf.append(out)
        if len(self._spec_buf) < 2:
            return
        x = np.concatenate(self._spec_buf).astype(np.float32) / 32768.0
        self._spec_buf.clear()
        prev = self._levels.get("mic_spectrum") or [0.0] * SPECTRUM_BANDS
        if float(np.max(np.abs(x))) < 1e-4:
            spec = [0.85 * o for o in prev]
        else:
            mag = np.abs(np.fft.rfft(x * np.hanning(len(x))))
            edges = np.geomspace(2, len(mag) - 1, SPECTRUM_BANDS + 1).astype(int)
            bands = [float(np.mean(mag[edges[i]:max(edges[i] + 1, edges[i + 1])])) for i in range(SPECTRUM_BANDS)]
            new = [min(1.0, max(0.0, (20 * math.log10(b + 1e-9) + 30) / 45)) for b in bands]
            spec = [max(n, 0.85 * o) for n, o in zip(new, prev)]
        self._levels = {**self._levels, "mic_spectrum": spec}

    def _update_out_meter(self, blk: np.ndarray) -> None:
        rms = float(np.sqrt(np.mean(blk * blk))) if len(blk) else 0.0
        out_db = 20 * math.log10(rms + 1e-9)
        spec = self._levels["spectrum"]
        if rms > 1e-5:
            mag = np.abs(np.fft.rfft(blk * np.hanning(len(blk))))
            edges = np.geomspace(2, len(mag) - 1, SPECTRUM_BANDS + 1).astype(int)
            bands = [float(np.mean(mag[edges[i]:max(edges[i] + 1, edges[i + 1])])) for i in range(SPECTRUM_BANDS)]
            new = [min(1.0, max(0.0, (20 * math.log10(b + 1e-9) + 30) / 45)) for b in bands]
            spec = [max(n, 0.85 * o) for n, o in zip(new, spec)]       # fast attack, smooth fall
        else:
            spec = [0.85 * o for o in spec]
        self._levels = {**self._levels, "out_db": out_db, "spectrum": spec, "playing": self.is_playing}
