"""
audio/voice_gate.py — only the enrolled user's speech reaches Gemini.

Sits between the microphone processing (audio/voice.py: echo cancellation,
noise suppression, gain, speech detection) and Gemini Live. Noise suppression
cleans the audio; this decides *whose* speech it is. They are separate steps.

Per utterance:
  idle        no speech → nothing is sent
  candidate   speech started → hold it locally (with 300 ms before the start)
              until there is enough speech to recognise the voice, then check
              it on a background thread
  verified    it's you → tell Gemini a turn started, send what was held (fast),
              then stream live until 0.7 s of silence → tell Gemini it ended
  rejected    someone/something else → drop it; while the sound goes on, keep
              re-checking the most recent second, so you can talk over the
              TV or right after someone else without waiting for a pause
  Uncertain   listen longer and re-check; still unsure → drop it, show a hint
  Short       "yes", "stop" (0.3–0.5 s): too short to recognise; accepted only
              within `follow_up_s` of an utterance that WAS recognised, and only
              while Gemini isn't busy answering. 0.5 s or more is checked.

Gemini must be told turns start/end itself (automatic activity detection off),
because otherwise its own server-side speech detection decides turns from
whatever audio it gets. main.py sets that up when the gate is active.

Opening the turn early (open_early): Gemini listens in real time from the
moment a turn starts, so held audio costs reply time. When Gemini is idle,
the turn is opened as soon as speech starts and the audio follows once it's
recognised; if it isn't you, the turn is closed empty ("cancel") — Gemini
does not answer an empty turn (checked with tools/gate_latency.py). Measured
0.4–0.6 s faster replies. While Gemini is busy answering, opening a turn would
interrupt it, so then the turn opens only after recognition.

This is not security: a recording of the user's voice gets through. Approvals
for risky actions are unchanged and never depend on it.

Threads: feed() runs on the audio DSP thread and never blocks; recognition
runs on one worker thread. Callbacks are invoked on the DSP thread.
"""

from __future__ import annotations

import collections
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

FRAME_MS = 10
RATE = 16000


@dataclass
class GateEvent:
    """What the gate decided about one utterance (for the UI and the decision log)."""
    action: str                 # "accepted" | "ignored" | "fallback"
    label: str                  # Verdict label: you | uncertain | unknown | too_short | follow_up | unavailable | not_enrolled
    score: float | None = None
    threshold: float | None = None
    speech_s: float = 0.0
    decide_ms: float = 0.0      # speech start → decision
    verify_ms: float = 0.0      # time spent in the speaker model
    at: float = 0.0


class VoiceGate:
    # Gemini listens in real time from the moment a turn starts, so every bit of
    # audio held for the check adds that much to the reply time (measured with
    # tools/gate_latency.py). Hence a short pre-roll and an early first check.
    PREROLL_MS = 150            # audio kept from before speech was detected
    VERIFY_AT_S = 0.8           # speech needed before the first check
    RECHECK_STEP_S = 0.5        # an uncertain result is re-checked with this much more speech
    MAX_WAIT_S = 2.3            # give up deciding after this much speech
    END_SILENCE_MS = 550        # silence that ends a turn
    MAX_TURN_S = 30.0
    MAX_HOLD_S = 6.0            # never hold more than this locally
    SEND_CHUNK_MS = 20
    FOLLOW_UP_MIN_S = 0.3       # a "short reply" must have at least this much speech…
    SHORT_CHECK_S = 0.5         # …and from this much on it is checked by the model instead
    ROLL_WINDOW_S = 1.0         # while rejecting: re-check this much of the latest speech…
    ROLL_STEP_S = 0.6           # …after every this much more of it…
    ROLL_HITS = 2               # …and accept only after this many matches in a row
    # Every extra check is another chance for a similar voice to scrape past the
    # threshold, so a second or later check must clear it by this margin.
    RECHECK_MARGIN = 0.03

    def __init__(self, identity, on_audio, on_activity, on_event=None, policy: str = "prefer",
                 follow_up_s: float = 20.0, threshold_offset: float = 0.0, open_early=None):
        """
        identity     SpeakerIdentity (or anything with .status and .verify(np.ndarray) -> Verdict)
        on_audio     fn(bytes) — int16 PCM for Gemini
        on_activity  fn("start" | "end" | "cancel")   cancel = close a turn that had no audio
        open_early   fn() -> bool: may a turn be opened before recognition (Gemini idle)?
        on_event     fn(GateEvent) — optional, for the UI / log
        policy       "prefer": if recognition is unavailable, pass speech through (and say so)
                     "only":   if recognition is unavailable, ignore voice input
        threshold_offset  added to the profile's threshold (+ stricter, − more lenient)
        """
        self.identity = identity
        self.on_audio = on_audio
        self.on_activity = on_activity
        self.on_event = on_event
        self.policy = policy
        self.follow_up_s = follow_up_s
        self.threshold_offset = threshold_offset
        self.open_early = open_early
        self._opened = False                    # a turn is open at Gemini
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="VoiceGate")
        self._preroll: collections.deque = collections.deque(maxlen=self.PREROLL_MS // FRAME_MS)
        self._recent: collections.deque = collections.deque(
            maxlen=int((self.PREROLL_MS / 1000 + self.ROLL_WINDOW_S) * 1000 / FRAME_MS))
        self._reset()
        self._last_accept_t = -1e9
        self.last_event: GateEvent | None = None
        self.stats = collections.Counter()

    # ── public ───────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        return self._state

    def feed(self, frame: np.ndarray, speech: bool, allowed: bool) -> None:
        """One 10 ms int16 frame of processed mic audio. allowed=False while JARVIS
        is speaking, the mic is muted, or the phone is talking."""
        if not allowed:
            if self._state == "verified":
                self._end_turn()
            elif self._state != "idle":
                self._cancel_open()
                self._reset()
            self._preroll.clear()
            return

        st = self._state
        if st == "idle":
            if speech:
                self._state = "candidate"
                self._t0 = time.monotonic()
                self._hold = list(self._preroll)
                self._preroll.clear()
                self._hold.append(frame)
                self._recent.clear(); self._recent.extend(self._hold)
                self._speech_frames = 1
                self._silence = 0
                try:
                    early = bool(self.open_early and self.open_early())
                except Exception:
                    early = False
                if early and getattr(self.identity, "status", "") == "ready":
                    self._activity("start")
                    self._opened = True
            else:
                self._preroll.append(frame)
            return

        self._recent.append(frame)

        self._silence = 0 if speech else self._silence + 1
        if speech:
            self._speech_frames += 1
        ended = self._silence * FRAME_MS >= self.END_SILENCE_MS

        if st == "candidate":
            self._hold.append(frame)
            if len(self._hold) * FRAME_MS > self.MAX_HOLD_S * 1000:
                self._hold.pop(1)                       # keep the start, drop oldest middle
            self._advance_candidate(ended)
        elif st == "verified":
            self._send(frame)
            if ended or (time.monotonic() - self._turn_t) > self.MAX_TURN_S:
                self._end_turn()
        elif st == "rejected":
            if ended:
                self._reset()
            else:
                self._roll(speech)

    def warm_up(self) -> None:
        """Load the speaker model now (first use otherwise costs ~1 s mid-sentence)."""
        def work():
            try:
                self.identity.embed(np.zeros(RATE, np.float32))
            except Exception:
                pass
        self._pool.submit(work)

    def reset(self) -> None:
        """Forget any utterance in progress (e.g. a new Gemini session started)."""
        self._reset()
        self._preroll.clear()
        self._opened = False                         # a new session has no turn open

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ── candidate → decision ─────────────────────────────────────────────

    @property
    def _speech_s(self) -> float:
        return self._speech_frames * FRAME_MS / 1000

    def _advance_candidate(self, ended: bool) -> None:
        status = getattr(self.identity, "status", "unavailable")
        if status != "ready":
            if self.policy == "prefer":
                self._accept(GateEvent("fallback", status))
            else:
                self._reject(GateEvent("ignored", status))
            return

        fut = self._pending
        if fut is not None:
            if not fut.done():
                return                                   # decided on a later frame
            self._pending = None
            try:
                v, ms = fut.result()
            except Exception as e:                       # recognition crashed
                print(f"[VoiceGate] recognition failed: {e}")
                v, ms = None, 0.0
            if v is None or v.label == "unavailable":
                ev = GateEvent("fallback" if self.policy == "prefer" else "ignored", "unavailable")
                (self._accept if self.policy == "prefer" else self._reject)(ev)
                return
            ev = GateEvent("", v.label, v.score, v.threshold, v.speech_s, verify_ms=ms)
            if (v.label == "you" and self._checked_at is not None and v.score is not None
                    and v.threshold is not None and v.score < v.threshold + self.RECHECK_MARGIN):
                v.label = ev.label = "uncertain"          # a re-check must clear the bar with room to spare
            if v.label == "you":
                ev.action = "accepted"; self._accept(ev); return
            if v.label == "unknown":
                ev.action = "ignored"; self._reject(ev); return
            # uncertain / too_short: keep listening if there's still time
            self._checked_at = self._speech_s
            if self._speech_s >= self.MAX_WAIT_S or ended:
                ev.action = "ignored"; self._reject(ev); return

        if ended and self._pending is None:
            # utterance finished before we had enough speech to recognise
            if self._speech_s < self.SHORT_CHECK_S:
                try:
                    idle = self.open_early() if self.open_early else True
                except Exception:
                    idle = False
                if (time.monotonic() - self._last_accept_t <= self.follow_up_s
                        and self._speech_s >= self.FOLLOW_UP_MIN_S and idle):
                    self._accept(GateEvent("accepted", "follow_up", speech_s=self._speech_s))
                else:
                    self._reject(GateEvent("ignored", "too_short", speech_s=self._speech_s), end_now=True)
                return
            self._submit()                               # enough speech: check it now
            return

        due = (self._checked_at is None and self._speech_s >= self.VERIFY_AT_S) or \
              (self._checked_at is not None and self._speech_s >= self._checked_at + self.RECHECK_STEP_S)
        if due and self._pending is None:
            self._submit()

    def _roll(self, speech: bool) -> None:
        """Rejected, but the sound continues: has the user started talking?"""
        if getattr(self.identity, "status", "unavailable") != "ready":
            return                                       # nothing to re-check with ("only" mode stays closed)
        if speech:
            self._roll_frames += 1
        fut = self._pending
        if fut is not None:
            if not fut.done():
                return
            self._pending = None
            try:
                v, ms = fut.result()
            except Exception:
                return
            ok = (v is not None and v.label == "you" and v.score is not None and v.threshold is not None
                  and v.score >= v.threshold + self.RECHECK_MARGIN)
            self._roll_hits = self._roll_hits + 1 if ok else 0
            if ok and self._roll_hits >= self.ROLL_HITS:
                self._hold = list(self._recent)
                self._speech_frames = int(self.ROLL_WINDOW_S * 1000 / FRAME_MS)
                self._accept(GateEvent("accepted", "you", v.score, v.threshold, v.speech_s, verify_ms=ms))
            return
        if self._roll_frames * FRAME_MS >= self.ROLL_STEP_S * 1000:
            self._roll_frames = 0
            self._submit(list(self._recent))

    def _submit(self, frames: list | None = None) -> None:
        audio = np.concatenate(frames if frames is not None else self._hold).astype(np.float32) / 32768.0
        off = self.threshold_offset
        ident = self.identity

        def work():
            t = time.perf_counter()
            v = ident.verify(audio)
            if v.score is not None and v.threshold is not None and off:
                v.threshold = v.threshold + off
                from audio.speaker_id import UNCERTAIN_BAND as band
                v.label = "you" if v.score >= v.threshold else (
                    "uncertain" if v.score >= v.threshold - band else "unknown")
            try:
                ident.last = v                              # Settings shows the latest reading
            except Exception:
                pass
            return v, (time.perf_counter() - t) * 1000

        self._pending: Future | None = self._pool.submit(work)

    # ── outcomes ─────────────────────────────────────────────────────────

    def _accept(self, ev: GateEvent) -> None:
        self._state = "verified"
        self._turn_t = time.monotonic()
        self._emit_event(ev)
        if not self._opened:
            self._activity("start")
            self._opened = True
        held = np.concatenate(self._hold) if self._hold else np.zeros(0, np.int16)
        self._hold = []
        self._out = []
        # held audio goes out in 100 ms pieces (fewer messages, faster than real time)
        step = RATE // 10
        for i in range(0, len(held), step):
            self._emit_audio(held[i:i + step].tobytes())
        if self._silence * FRAME_MS >= self.END_SILENCE_MS:
            self._end_turn()                             # it had already finished

    def _reject(self, ev: GateEvent, end_now: bool = False) -> None:
        self._emit_event(ev)
        self._cancel_open()
        self._hold = []
        if end_now or self._silence * FRAME_MS >= self.END_SILENCE_MS:
            self._reset()
        else:
            self._state = "rejected"
            self._roll_frames = 0
            self._roll_hits = 0

    def _end_turn(self) -> None:
        if self._out:
            self._emit_audio(np.concatenate(self._out).tobytes())
            self._out = []
        self._activity("end")
        self._opened = False
        self._last_accept_t = time.monotonic()
        self._reset()

    def _activity(self, kind: str) -> None:
        try:
            self.on_activity(kind)
        except Exception as e:
            print(f"[VoiceGate] {kind} failed: {e}")

    def _cancel_open(self) -> None:
        if self._opened:
            self._activity("cancel")
            self._opened = False

    def _send(self, frame: np.ndarray) -> None:
        self._out.append(frame)
        if len(self._out) * FRAME_MS >= self.SEND_CHUNK_MS:
            self._emit_audio(np.concatenate(self._out).tobytes())
            self._out = []

    def _emit_audio(self, data: bytes) -> None:
        try:
            self.on_audio(data)
        except Exception:
            pass

    def _emit_event(self, ev: GateEvent) -> None:
        ev.at = time.time()
        ev.decide_ms = (time.monotonic() - self._t0) * 1000 if self._t0 else 0.0
        if not ev.speech_s:
            ev.speech_s = self._speech_s
        self.last_event = ev
        self.stats[f"{ev.action}:{ev.label}"] += 1
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception as e:
                print(f"[VoiceGate] event callback failed: {e}")

    def _reset(self) -> None:
        self._state = "idle"
        self._hold: list[np.ndarray] = []
        self._out: list[np.ndarray] = []
        self._speech_frames = 0
        self._silence = 0
        self._t0 = 0.0
        self._turn_t = 0.0
        self._checked_at: float | None = None
        self._roll_frames = 0
        self._roll_hits = 0
        if hasattr(self, "_recent"):
            self._recent.clear()
        p = getattr(self, "_pending", None)
        if p is not None:
            p.cancel()
        self._pending = None


CHECK_STEPS = [
    ("normal", "Open my calendar for tomorrow.", "Say this normally.", 5.0, "you"),
    ("normal", "What's the weather like this weekend?", "Say this normally.", 5.0, "you"),
    ("normal", "Remind me to call home at seven.", "Say this normally.", 5.0, "you"),
    ("normal", "Play some music I like.", "Say this a little faster than usual.", 5.0, "you"),
    ("quiet", "Turn the volume down a bit.", "Say this quietly.", 5.0, "you"),
    ("quiet", "What time is it?", "Say this quietly.", 5.0, "you"),
    ("far", "How long is my drive to work?", "Press Record, then step about 3 metres away and say this.", 8.0, "you"),
    ("far", "Search the web for cricket scores.", "From across the room again.", 8.0, "you"),
    ("over_tv", "Pause the music please.", "Start a TV show or video with people talking, then say this over it.", 6.0, "you"),
    ("over_tv", "Read me my last message.", "With the TV still on, say this.", 6.0, "you"),
    ("tv_only", "Stay silent", "Keep the TV / video playing and say nothing.", 6.0, "other"),
    ("tv_only", "Stay silent", "Again: TV only, say nothing.", 6.0, "other"),
    ("music_only", "Stay silent", "Play music with singing, and say nothing.", 6.0, "other"),
    ("other_person", "Open my calendar for tomorrow.", "Ask someone else to say this (or press Skip).", 5.0, "other"),
    ("other_person", "What's the weather like this weekend?", "The other person again (or Skip).", 5.0, "other"),
]


def replay(identity, pcm: np.ndarray, **kw) -> GateEvent | None:
    """Run a recording (float32 or int16, 16 kHz, already mic-processed) through
    the gate exactly as live audio would go, and return its final decision.
    Used by the accuracy check in Settings and by tools/gate_room_check.py."""
    from audio.voice import VoiceConfig, VoiceProcessor
    x = np.asarray(pcm)
    if x.dtype != np.int16:
        x = (np.clip(x.astype(np.float32), -1, 1) * 32767).astype(np.int16)
    x = np.concatenate([x, np.zeros(RATE, np.int16)])           # trailing silence ends the turn
    events = []
    g = VoiceGate(identity, on_audio=lambda b: None, on_activity=lambda k: None, on_event=events.append, **kw)
    det = VoiceProcessor(VoiceConfig(enabled=False))            # speech detection only
    try:
        for i in range(len(x) // 160):
            f = x[i * 160:(i + 1) * 160]
            det.process(f)
            g.feed(f, det.speech, True)
            if g._pending is not None:
                g._pending.exception()                          # wait for the model, in order
    finally:
        g.close()
    accepted = [e for e in events if e.action in ("accepted", "fallback")]
    return accepted[0] if accepted else (events[-1] if events else None)
