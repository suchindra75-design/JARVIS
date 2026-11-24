"""
Unit tests for the audio pipeline. No real devices: PortAudio streams are faked.

    .venv/bin/python -m unittest tests.test_audio -v
"""

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.engine import AudioEngine, AudioSettings, OUT_BLOCK, OUT_RATE, IN_FRAME   # noqa: E402
from audio.output import OutputEnhancer                                             # noqa: E402
from audio.voice import VoiceConfig, VoiceProcessor, SpeechGatedAGC                 # noqa: E402


def tone(freq, secs, rate, amp):
    t = np.arange(int(secs * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class OutputEnhancerTests(unittest.TestCase):
    def run_blocks(self, enh, x):
        return np.concatenate([enh.process(x[i:i + OUT_BLOCK]) for i in range(0, len(x), OUT_BLOCK)])

    def test_limiter_never_exceeds_ceiling(self):
        enh = OutputEnhancer()
        x = tone(220, 2, OUT_RATE, 0.9)
        x[::997] = 1.0                                   # spikes
        enh.volume = 1.5                                 # push it harder
        y = self.run_blocks(enh, x)
        self.assertLessEqual(float(np.max(np.abs(y))), OutputEnhancer.CEILING + 1e-6)

    def test_normaliser_brings_quiet_voice_up_within_limits(self):
        enh = OutputEnhancer()
        x = tone(300, 4, OUT_RATE, 10 ** (-35 / 20) * np.sqrt(2))
        y = self.run_blocks(enh, x)
        last = y[-OUT_RATE // 2:]
        lvl = 20 * np.log10(np.sqrt(np.mean(last ** 2)))
        self.assertGreater(lvl, -35 + 9)                 # boosted…
        self.assertLessEqual(lvl, -35 + 12 + 0.5)        # …but never more than +12 dB

    def test_volume_change_is_ramped_not_stepped(self):
        enh = OutputEnhancer()
        enh.normalize = False
        x = np.full(OUT_RATE, 0.2, dtype=np.float32)
        y1 = self.run_blocks(enh, x[: OUT_RATE // 2])
        enh.volume = 0.2
        y2 = self.run_blocks(enh, x[OUT_RATE // 2:])
        y = np.concatenate([y1, y2])[200:]          # skip the start-up edge from silence
        self.assertLess(float(np.max(np.abs(np.diff(y)))), 0.01)   # no audible click
        self.assertAlmostEqual(float(y[-1]), 0.2 * 0.2, places=3)  # reached the new volume

    def test_bypass_only_applies_volume(self):
        enh = OutputEnhancer()
        enh.enabled = False
        x = tone(440, 0.1, OUT_RATE, 0.5)
        np.testing.assert_allclose(self.run_blocks(enh, x), x, atol=1e-6)


class VoiceProcessorTests(unittest.TestCase):
    def frames(self, x):
        x16 = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        return [x16[i:i + IN_FRAME] for i in range(0, len(x16) - IN_FRAME + 1, IN_FRAME)]

    def test_bypass_is_bit_exact(self):
        vp = VoiceProcessor(VoiceConfig(enabled=False))
        for f in self.frames(tone(300, 0.5, 16000, 0.3)):
            np.testing.assert_array_equal(vp.process(f), f)

    def test_processing_error_falls_back_to_bypass(self):
        vp = VoiceProcessor(VoiceConfig())
        f = self.frames(tone(300, 0.1, 16000, 0.3))[0]
        with mock.patch.object(vp, "_run_apm", side_effect=RuntimeError("boom")):
            out = vp.process(f)
        np.testing.assert_array_equal(out, f)
        self.assertIn("boom", vp.error)
        self.assertFalse(vp.active)
        vp.configure(VoiceConfig(noise_suppression="light"))      # new settings retry
        self.assertIsNone(vp.error)

    def test_all_levels_run_and_keep_polarity(self):
        # broadband, speech-like bursts (a pure tone makes the lag ambiguous)
        rng = np.random.default_rng(0)
        x = np.convolve(rng.standard_normal(16000), np.ones(4) / 4, "same").astype(np.float32) * 0.15
        x *= (np.sin(2 * np.pi * 3 * np.arange(16000) / 16000) > 0)
        for lvl in ("off", "light", "standard"):
            vp = VoiceProcessor(VoiceConfig(noise_suppression=lvl, auto_gain=False))
            out = np.concatenate([vp.process(f) for f in self.frames(x)]).astype(np.float32)
            self.assertIsNone(vp.error, lvl)
            xc = np.correlate(out[:12000], x[:12000], "full")[11999:11999 + 800]
            k = int(np.argmax(np.abs(xc)))
            self.assertGreater(xc[k], 0, f"{lvl}: output phase-inverted (lag {k})")

    def test_agc_holds_still_while_raya_speaks(self):
        agc = SpeechGatedAGC(-23)
        x = tone(300, 0.01, 16000, 0.01)
        for _ in range(100):
            agc.process(x, speech=True, frozen=True)
        self.assertEqual(agc.gain_db, 0.0)
        for _ in range(100):
            agc.process(x, speech=True, frozen=False)
        self.assertGreater(agc.gain_db, 3.0)

    def test_limiter_on_mic_path(self):
        vp = VoiceProcessor(VoiceConfig(echo_cancellation=False, noise_suppression="off", high_pass=False))
        out = np.concatenate([vp.process(f) for f in self.frames(tone(300, 1, 16000, 1.0))])
        self.assertLessEqual(int(np.max(np.abs(out))), int(32767 * 10 ** (-1 / 20)) + 2)


class FakeStream:
    def __init__(self, latency=0.02):
        self.active = True
        self.latency = latency

    def abort(self, ignore_errors=True):
        self.active = False

    def close(self, ignore_errors=True):
        self.active = False


class _EngineBase(unittest.TestCase):
    """Helpers to drive the engine's callbacks by hand; the DSP worker thread runs for real."""

    def make(self, **settings):
        chunks = []
        allowed = {"v": True}
        eng = AudioEngine(AudioSettings(**settings), on_chunk=chunks.append, should_send=lambda: allowed["v"])
        opened = []

        def fake_open():
            eng._in_stream, eng._out_stream = FakeStream(), FakeStream()
            opened.append(time.monotonic())
        eng._open_streams = fake_open
        eng.refresh_devices = fake_open
        eng.start()
        self.addCleanup(eng.stop)
        return eng, chunks, allowed, opened

    def feed_mic(self, eng, secs, amp=0.05):
        x = (tone(250, secs, 16000, amp) * 32767).astype(np.int16).reshape(-1, 1)
        for i in range(0, len(x) - IN_FRAME + 1, IN_FRAME):
            eng._in_cb(x[i:i + IN_FRAME], IN_FRAME, None, None)
            if i % (IN_FRAME * 10) == 0:
                time.sleep(0.002)

    def pull_speaker(self, eng, blocks):
        out = []
        for _ in range(blocks):
            buf = np.zeros((OUT_BLOCK, 1), np.float32)
            eng._out_cb(buf, OUT_BLOCK, None, None)
            out.append(buf[:, 0].copy())
        return np.concatenate(out)

    def wait(self, cond, timeout=2.0):
        t = time.monotonic()
        while time.monotonic() - t < timeout:
            if cond():
                return True
            time.sleep(0.01)
        return False


class EngineTests(_EngineBase):
    def test_mic_chunks_are_20ms(self):
        eng, chunks, _, _ = self.make()
        self.feed_mic(eng, 0.5)
        self.assertTrue(self.wait(lambda: len(chunks) >= 20))
        self.assertTrue(all(len(c) == 2 * 320 for c in chunks))

    def test_half_duplex_closes_mic_while_speaking_and_for_the_tail(self):
        eng, chunks, _, _ = self.make()
        eng.play((tone(300, 0.5, OUT_RATE, 0.3) * 32767).astype(np.int16).tobytes())
        self.assertTrue(self.wait(lambda: eng._play_samples > 0))
        self.pull_speaker(eng, 10)
        before = len(chunks)
        self.feed_mic(eng, 0.3)
        time.sleep(0.1)
        self.assertEqual(len(chunks), before, "mic must be closed while Raya plays")
        self.pull_speaker(eng, 60)                  # drain everything
        # a real output device keeps pulling; keep the fake one running too
        self.assertTrue(self.wait(lambda: (self.pull_speaker(eng, 1) is not None) and not eng.is_playing))
        time.sleep(0.35)                            # past the tail
        self.feed_mic(eng, 0.2)
        self.assertTrue(self.wait(lambda: len(chunks) > before), "mic must reopen after the tail")

    def test_should_send_gate(self):
        eng, chunks, allowed, _ = self.make()
        allowed["v"] = False
        self.feed_mic(eng, 0.3)
        time.sleep(0.1)
        self.assertEqual(chunks, [])

    def test_flush_stops_quickly_and_without_a_click(self):
        eng, _, _, _ = self.make()
        eng.play((tone(300, 2.0, OUT_RATE, 0.4) * 32767).astype(np.int16).tobytes())
        self.assertTrue(self.wait(lambda: eng._play_samples > OUT_RATE))
        before = self.pull_speaker(eng, 10)
        eng.flush()
        after = self.pull_speaker(eng, 5)
        self.assertLess(float(np.max(np.abs(after[-OUT_BLOCK:]))), 1e-6, "silent within 50 ms")
        joined = np.concatenate([before[-10:], after[:40]])
        self.assertLess(float(np.max(np.abs(np.diff(joined)))), 0.1)
        self.assertTrue(self.wait(lambda: (self.pull_speaker(eng, 1) is not None) and not eng.is_playing, 0.5))

    def test_capture_ring_is_bounded_when_worker_is_starved(self):
        eng, _, _, _ = self.make()
        eng._running = False                       # freeze the worker
        eng._wake.set()
        time.sleep(0.05)
        self.feed_mic(eng, 3.0)
        self.assertLessEqual(len(eng._cap), eng._cap.maxlen)
        self.assertGreater(eng.dropped_in_blocks, 0)

    def test_device_loss_triggers_reopen(self):
        eng, _, _, opened = self.make()
        n = len(opened)
        eng._in_stream.active = False              # simulate unplugged mic
        self.assertTrue(self.wait(lambda: len(opened) > n, 4.0))

    def test_no_speaker_does_not_lock_the_mic(self):
        eng, chunks, _, _ = self.make()
        eng._out_stream = None
        eng.play((tone(300, 1.0, OUT_RATE, 0.3) * 32767).astype(np.int16).tobytes())
        self.assertTrue(self.wait(lambda: not eng.is_playing, 1.0))

    def test_settings_apply_without_restart_and_bypass(self):
        eng, chunks, _, _ = self.make()
        eng.apply_settings(AudioSettings(processing_enabled=False))
        self.assertTrue(self.wait(lambda: not eng.voice.config.enabled))
        self.assertFalse(eng.enhancer.enabled)
        x = (tone(250, 0.1, 16000, 0.2) * 32767).astype(np.int16)
        start = len(chunks)
        self.feed_mic(eng, 0.2, amp=0.2)
        self.assertTrue(self.wait(lambda: len(chunks) > start))

    def test_mic_test_is_never_sent(self):
        eng, chunks, _, _ = self.make()
        done = threading.Event()
        eng.start_mic_test(0.3, on_done=done.set)
        self.feed_mic(eng, 0.4)
        self.assertTrue(done.wait(2))
        self.assertTrue(self.wait(lambda: eng._play_samples > 0 or eng._raw_out_samples > 0))
        self.assertLessEqual(len(chunks), 5)        # only frames after the test ended


class EngineHookTests(_EngineBase):
    """capture() for voice enrolment and the utterance tap for speaker checks."""

    def test_capture_returns_audio_and_sends_nothing(self):
        eng, chunks, _, _ = self.make()
        got = {}
        self.assertTrue(eng.capture(0.3, lambda a: got.setdefault("a", a)))
        self.assertFalse(eng.capture(0.3, lambda a: None), "second capture while busy must be refused")
        self.feed_mic(eng, 0.5)
        self.assertTrue(self.wait(lambda: "a" in got))
        self.assertEqual(got["a"].dtype, np.float32)
        self.assertEqual(len(got["a"]), 30 * IN_FRAME)
        self.assertLessEqual(len(chunks), 10)               # only frames after the capture ended
        self.assertEqual(eng._play_samples + eng._raw_out_samples, 0, "capture must not play back")

    def test_utterances_only_from_audio_sent_to_gemini(self):
        eng, chunks, allowed, _ = self.make()
        utts = []
        eng.on_utterance = utts.append
        eng.voice._detect = lambda x: True                # treat everything as speech
        eng.voice.speech = True
        allowed["v"] = False                              # muted: nothing is sent → nothing checked
        self.feed_mic(eng, 1.5)
        time.sleep(0.2)
        self.assertEqual(utts, [])
        allowed["v"] = True
        x = (tone(250, 1.5, 16000, 0.05) * 32767).astype(np.int16).reshape(-1, 1)
        for i in range(0, len(x) - IN_FRAME + 1, IN_FRAME):   # paced like a real device (4× real time)
            eng._in_cb(x[i:i + IN_FRAME], IN_FRAME, None, None)
            if (i // IN_FRAME) % 10 == 9:
                time.sleep(0.025)
        time.sleep(0.1)
        allowed["v"] = False                              # stop sending → utterance ends
        self.feed_mic(eng, 0.1)
        self.assertTrue(self.wait(lambda: len(utts) == 1))
        self.assertGreaterEqual(len(utts[0]), 100 * IN_FRAME)

    def test_mic_spectrum_in_levels(self):
        eng, _, _, _ = self.make()
        self.feed_mic(eng, 0.3, amp=0.2)
        self.assertTrue(self.wait(lambda: len(eng.levels().get("mic_spectrum") or []) == 24))


class SettingsTests(unittest.TestCase):
    def test_from_dict_ignores_junk_and_clamps(self):
        s = AudioSettings.from_dict({"noise_suppression": "max", "output_volume": 9, "bogus": 1})
        self.assertEqual(s.noise_suppression, "standard")
        self.assertEqual(s.output_volume, 1.5)

    def test_update_config_merges_and_refuses_corrupt(self):
        import memory.config_manager as cm
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / "api_keys.json"
            f.write_text(json.dumps({"gemini_api_key": "KEEP", "assistant_name": "Raya"}))
            with mock.patch.object(cm, "CONFIG_FILE", f), mock.patch.object(cm, "CONFIG_DIR", Path(td)):
                cm.save_audio_settings({"full_duplex": True})
                d = json.loads(f.read_text())
                self.assertEqual(d["gemini_api_key"], "KEEP")
                self.assertTrue(d["audio"]["full_duplex"])
                self.assertEqual(f.stat().st_mode & 0o777, 0o600)
                f.write_text("{not json")
                with self.assertRaises(Exception):
                    cm.save_brief_enabled(False)
                self.assertEqual(f.read_text(), "{not json")


if __name__ == "__main__":
    unittest.main()
