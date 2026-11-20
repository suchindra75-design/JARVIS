"""
Tests for the personal voice gate (audio/voice_gate.py) with a fake recogniser.

    .venv/bin/python -m unittest tests.test_voice_gate -v
"""

import os
import sys
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from audio.speaker_id import Verdict          # noqa: E402
from audio.voice_gate import VoiceGate        # noqa: E402

FRAME = 160


class FakeID:
    def __init__(self, label="you", score=0.8, status="ready", delay=0.0, labels=None):
        self.label, self.score, self.status, self.delay = label, score, status, delay
        self.labels = list(labels or [])
        self.calls = []

    def verify(self, audio):
        self.calls.append(len(audio) / 16000)
        time.sleep(self.delay)
        lab = self.labels.pop(0) if self.labels else self.label
        return Verdict(lab, self.score, 0.6, len(audio) / 16000)


class Harness:
    def __init__(self, ident, **kw):
        self.sent = bytearray()
        self.log = []                       # ("start"|"end"|"audio", n)
        self.events = []
        self.gate = VoiceGate(ident, on_audio=self._audio, on_activity=lambda k: self.log.append((k, 0)),
                              on_event=self.events.append, **kw)

    def _audio(self, b):
        self.sent += b
        self.log.append(("audio", len(b)))

    def run(self, pattern, allowed=True, settle=True):
        """pattern: list of (seconds, is_speech)."""
        for secs, sp in pattern:
            for _ in range(int(secs * 100)):
                v = 3000 if sp else 20
                self.gate.feed(np.full(FRAME, v, np.int16), sp, allowed)
                if settle and self.gate._pending is not None:
                    self.gate._pending.exception()  # wait for it ("instant" but ordered); errors stay in the future
        return self

    @property
    def kinds(self):
        return [k for k, _ in self.log if k != "audio"]


class EarlyOpenTests(unittest.TestCase):
    """Turn opened at speech start (Gemini idle), audio after recognition."""

    def test_accepted_turn_is_opened_at_speech_start(self):
        h = Harness(FakeID("you"), open_early=lambda: True)
        h.run([(0.05, True)])
        self.assertEqual(h.kinds, ["start"], "opened before any recognition")
        self.assertEqual(len(h.sent), 0, "but no audio yet")
        h.run([(2.0, True), (1.0, False)])
        self.assertEqual(h.kinds, ["start", "end"])
        self.assertGreater(len(h.sent), 0)

    def test_someone_else_gets_an_empty_cancelled_turn(self):
        h = Harness(FakeID("unknown", 0.1), open_early=lambda: True).run([(2.0, True), (1.0, False)])
        self.assertEqual(h.kinds, ["start", "cancel"])
        self.assertEqual(len(h.sent), 0, "none of their audio is sent")

    def test_not_opened_early_while_gemini_is_busy(self):
        h = Harness(FakeID("unknown", 0.1), open_early=lambda: False).run([(2.0, True), (1.0, False)])
        self.assertEqual(h.log, [])
        h = Harness(FakeID("you"), open_early=lambda: False).run([(2.0, True), (1.0, False)])
        self.assertEqual(h.kinds, ["start", "end"])

    def test_mic_closing_mid_check_cancels_the_open_turn(self):
        h = Harness(FakeID("you", delay=0.5), open_early=lambda: True)
        h.run([(0.3, True)], settle=False)
        h.run([(0.2, True)], allowed=False, settle=False)   # muted / JARVIS starts talking
        self.assertEqual(h.kinds, ["start", "cancel"])

    def test_short_noise_blip_is_cancelled(self):
        h = Harness(FakeID("you"), open_early=lambda: True).run([(0.2, True), (1.0, False)])
        self.assertEqual(h.kinds, ["start", "cancel"])
        self.assertEqual(len(h.sent), 0)

    def test_every_open_is_closed(self):
        rng = np.random.default_rng(3)
        ident = FakeID(labels=list(rng.choice(["you", "unknown", "uncertain"], 400)))
        h = Harness(ident, open_early=lambda: True)
        for _ in range(60):
            h.run([(float(rng.uniform(0.1, 3.0)), True), (float(rng.uniform(0.0, 1.2)), False)],
                  allowed=bool(rng.random() > 0.1))
        h.run([(2.0, False)])
        opens = h.kinds.count("start"); closes = h.kinds.count("end") + h.kinds.count("cancel")
        self.assertEqual(opens, closes)
        for a, b in zip(h.kinds[::2], h.kinds[1::2]):
            self.assertEqual(a, "start"); self.assertIn(b, ("end", "cancel"))


class VoiceGateTests(unittest.TestCase):
    def test_silence_sends_nothing(self):
        h = Harness(FakeID()).run([(3, False)])
        self.assertEqual(h.log, [])
        self.assertEqual(h.gate.state, "idle")

    def test_your_speech_is_forwarded_whole_with_preroll(self):
        h = Harness(FakeID("you")).run([(0.5, False), (2.0, True), (1.0, False)])
        self.assertEqual(h.kinds, ["start", "end"])
        sent_s = len(h.sent) / 2 / 16000
        # pre-roll + 2.0 s speech + trailing silence up to the end of turn
        want = VoiceGate.PREROLL_MS / 1000 + 2.0 + VoiceGate.END_SILENCE_MS / 1000
        self.assertAlmostEqual(sent_s, want, delta=0.05)
        self.assertEqual(h.events[0].action, "accepted")
        self.assertEqual(h.gate.state, "idle")

    def test_start_comes_before_audio_and_end_after(self):
        h = Harness(FakeID("you")).run([(2.0, True), (1.0, False)])
        idx = [i for i, (k, _) in enumerate(h.log) if k == "audio"]
        self.assertLess(h.log.index(("start", 0)), idx[0])
        self.assertGreater(h.log.index(("end", 0)), idx[-1])

    def test_unknown_speaker_is_dropped(self):
        h = Harness(FakeID("unknown", 0.2)).run([(2.5, True), (1.0, False)])
        self.assertEqual(h.log, [])
        self.assertEqual(h.events[0].action, "ignored")
        self.assertEqual(h.events[0].label, "unknown")
        self.assertEqual(h.gate.state, "idle")

    def test_uncertain_is_rechecked_then_accepted(self):
        ident = FakeID(labels=["uncertain", "you"])
        h = Harness(ident).run([(2.5, True), (1.0, False)])
        self.assertEqual(len(ident.calls), 2)
        self.assertGreater(ident.calls[1], ident.calls[0])
        self.assertEqual(h.kinds, ["start", "end"])

    def test_uncertain_until_deadline_is_dropped(self):
        h = Harness(FakeID("uncertain", 0.55)).run([(4.0, True), (1.0, False)])
        self.assertEqual(h.log, [])
        self.assertEqual(h.events[-1].label, "uncertain")

    def test_short_reply_needs_recent_recognised_speech(self):
        h = Harness(FakeID("you"))
        h.run([(0.4, True), (1.0, False)])              # "yes" out of nowhere → ignored
        self.assertEqual(h.log, [])
        self.assertEqual(h.events[-1].label, "too_short")
        h.run([(2.0, True), (1.0, False)])              # a recognised command
        h.log.clear()
        h.run([(0.4, True), (1.0, False)])              # "yes" right after → accepted
        self.assertEqual(h.kinds, ["start", "end"])
        self.assertEqual(h.events[-1].label, "follow_up")
        h.log.clear()
        h.run([(0.15, True), (1.0, False)])             # a click or blip → not a reply
        self.assertEqual(h.log, [])

    def test_half_second_or_more_is_checked_not_waved_through(self):
        ident = FakeID(labels=["you", "unknown"])
        h = Harness(ident)
        h.run([(2.0, True), (1.0, False)])              # recognised command
        h.log.clear()
        h.run([(0.6, True), (1.0, False)])              # 0.6 s right after → checked → not you
        self.assertEqual(h.log, [])
        self.assertEqual(len(ident.calls), 2)

    def test_no_short_reply_while_gemini_is_busy(self):
        busy = {"v": False}
        h = Harness(FakeID("you"), open_early=lambda: not busy["v"])
        h.run([(2.0, True), (1.0, False)])
        busy["v"] = True                                # Gemini is still answering
        h.log.clear()
        h.run([(0.4, True), (1.0, False)])
        self.assertEqual([k for k in h.kinds if k != "cancel"], [])
        busy["v"] = False
        h.log.clear()
        h.run([(0.4, True), (1.0, False)])
        self.assertEqual(h.kinds, ["start", "end"])

    def test_short_reply_window_expires(self):
        h = Harness(FakeID("you"), follow_up_s=0.0)
        h.run([(2.0, True), (1.0, False)])
        h.log.clear()
        h.run([(0.4, True), (1.0, False)])
        self.assertEqual(h.log, [])

    def test_prefer_falls_back_to_open_when_recognition_unavailable(self):
        h = Harness(FakeID(status="unavailable"), policy="prefer").run([(1.5, True), (1.0, False)])
        self.assertEqual(h.kinds, ["start", "end"])
        self.assertEqual(h.events[0].action, "fallback")

    def test_only_mode_ignores_when_recognition_unavailable(self):
        h = Harness(FakeID(status="unavailable"), policy="only").run([(1.5, True), (1.0, False)])
        self.assertEqual(h.log, [])

    def test_recogniser_crash_is_safe(self):
        class Boom(FakeID):
            def verify(self, audio):
                raise RuntimeError("model gone")
        h = Harness(Boom(), policy="only").run([(2.0, True), (1.0, False)])
        self.assertEqual(h.log, [])
        h = Harness(Boom(), policy="prefer").run([(2.0, True), (1.0, False)])
        self.assertEqual(h.kinds, ["start", "end"])

    def test_jarvis_speaking_ends_turn_and_blocks(self):
        h = Harness(FakeID("you")).run([(2.0, True)])
        self.assertEqual(h.kinds, ["start"])
        h.run([(1.0, True)], allowed=False)             # JARVIS starts talking / mic muted
        self.assertEqual(h.kinds, ["start", "end"])
        n = len(h.log)
        h.run([(2.0, True)], allowed=False)             # its own voice is never forwarded
        self.assertEqual(len(h.log), n)

    def test_strictness_offset_can_reject(self):
        h = Harness(FakeID("you", score=0.65), threshold_offset=0.10).run([(2.5, True), (1.0, False)])
        self.assertEqual(h.log, [])                     # 0.65 < 0.60 + 0.10
        self.assertEqual(h.events[-1].label, "uncertain")

    def test_feed_never_blocks_on_slow_recogniser(self):
        h = Harness(FakeID("you", delay=0.3))
        t = time.perf_counter()
        h.run([(1.5, True)], settle=False)
        self.assertLess(time.perf_counter() - t, 0.15, "feed() must not wait for the model")
        time.sleep(0.4)
        h.run([(0.5, True), (1.0, False)], settle=False)
        self.assertEqual(h.kinds, ["start", "end"])

    def test_hold_is_bounded(self):
        h = Harness(FakeID("uncertain", delay=0.0), policy="prefer")
        h.gate.MAX_WAIT_S = 1e9                         # never decide
        h.run([(20.0, True)])
        self.assertLessEqual(len(h.gate._hold), VoiceGate.MAX_HOLD_S * 100 + 1)

    def test_you_talking_straight_after_the_tv_is_caught(self):
        """TV speech, then you, with no pause in between."""
        ident = FakeID(labels=["unknown", "unknown", "you"])
        h = Harness(ident).run([(1.0, True)])          # TV: first check → unknown → rejected
        self.assertEqual(h.gate.state, "rejected")
        h.run([(2.5, True), (1.0, False)])              # sound continues (now you) → rolling re-check
        self.assertEqual(h.kinds, ["start", "end"])
        sent_s = len(h.sent) / 2 / 16000
        everything = VoiceGate.PREROLL_MS / 1000 + 3.5 + VoiceGate.END_SILENCE_MS / 1000
        self.assertLess(sent_s, everything - 0.7, "only the recent window, not the TV part, is forwarded")
        self.assertEqual([e.action for e in h.events], ["ignored", "accepted"])

    def test_continuous_other_speech_never_gets_through_and_stays_quiet(self):
        ident = FakeID("unknown", 0.1)
        h = Harness(ident).run([(20.0, True), (1.0, False)])
        self.assertEqual(h.log, [])
        self.assertEqual(len(h.events), 1, "one notice per stretch, not one per re-check")
        self.assertGreater(len(ident.calls), 20)        # it kept checking (~every 0.6 s)
        self.assertTrue(all(c <= 1.2 for c in ident.calls[1:]), "re-checks use only the recent window")

    def test_long_turn_is_capped(self):
        h = Harness(FakeID("you"))
        h.gate.MAX_TURN_S = 0.5
        h.run([(1.2, True)])
        time.sleep(0.6)
        h.run([(0.1, True)])
        self.assertIn("end", h.kinds)


if __name__ == "__main__":
    unittest.main()
