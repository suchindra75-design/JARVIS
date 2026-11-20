"""
Tests for audio/speaker_id.py. Uses macOS `say` voices; skipped where unavailable.

    .venv/bin/python -m unittest tests.test_speaker_id -v
"""

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audio.speaker_id import ENROL_PHRASES, SpeakerIdentity   # noqa: E402

HAVE_SAY = shutil.which("say") is not None and shutil.which("afconvert") is not None
_CACHE: dict = {}


def tts(text: str, voice: str) -> np.ndarray:
    key = (text, voice)
    if key not in _CACHE:
        with tempfile.TemporaryDirectory() as td:
            subprocess.run(["say", "-v", voice, "-o", f"{td}/a.aiff", text], check=True)
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                            f"{td}/a.aiff", f"{td}/a.wav"], check=True, capture_output=True)
            with wave.open(f"{td}/a.wav") as w:
                _CACHE[key] = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return _CACHE[key]


@unittest.skipUnless(HAVE_SAY, "needs macOS say/afconvert")
class SpeakerIdTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.sid = SpeakerIdentity(model_dir=ROOT / "models", profile_path=self.tmp / "voice_profile.npz")
        if self.sid.status in ("unavailable", "no_model"):
            self.skipTest(f"speaker model not available ({self.sid.status})")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def enrol(self, voice="Samantha"):
        return self.sid.enrol([tts(p, voice) for p in ENROL_PHRASES])

    def test_status_flow_and_private_profile_file(self):
        self.assertEqual(self.sid.status, "not_enrolled")
        info = self.enrol()
        self.assertEqual(self.sid.status, "ready")
        self.assertEqual(info["samples"], 5)
        mode = self.sid.profile_path.stat().st_mode & 0o777
        self.assertEqual(mode, 0o600)
        again = SpeakerIdentity(model_dir=ROOT / "models", profile_path=self.sid.profile_path)
        self.assertEqual(again.status, "ready")                 # survives a restart
        self.sid.delete_profile()
        self.assertFalse(self.sid.profile_path.exists())
        self.assertEqual(self.sid.status, "not_enrolled")

    def test_recognises_enrolled_voice_and_rejects_others(self):
        self.enrol("Samantha")
        me = self.sid.verify(tts("What time is the meeting with the design team tomorrow?", "Samantha"))
        self.assertEqual(me.label, "you", me)
        for other in ("Karen", "Rishi", "Fred", "Daniel", "Moira"):
            v = self.sid.verify(tts("What time is the meeting with the design team tomorrow?", other))
            self.assertNotEqual(v.label, "you", f"{other} accepted as the user: {v}")

    def test_short_speech_is_not_guessed(self):
        self.enrol()
        v = self.sid.verify(tts("Yes.", "Samantha")[: 8000])
        self.assertEqual(v.label, "too_short")

    def test_append_and_retrain(self):
        self.enrol()
        info = self.sid.enrol([tts(ENROL_PHRASES[0], "Samantha"), tts(ENROL_PHRASES[1], "Samantha")], append=True)
        self.assertEqual(info["samples"], 7)
        self.sid.delete_profile()
        self.assertEqual(self.enrol("Karen")["samples"], 5)

    def test_sample_quality_checks(self):
        good = tts(ENROL_PHRASES[0], "Samantha")
        self.assertTrue(SpeakerIdentity.check_sample(good).ok)
        self.assertIn("quiet", SpeakerIdentity.check_sample(good * 0.001).reason)
        self.assertIn("loud", SpeakerIdentity.check_sample(np.clip(good * 40, -1, 1)).reason)
        self.assertIn("speech", SpeakerIdentity.check_sample(good[:16000]).reason)
        noisy = good + np.random.default_rng(0).standard_normal(len(good)).astype(np.float32) * 0.2
        self.assertIn("noise", SpeakerIdentity.check_sample(noisy).reason)

    def test_background_verify_runs_off_the_calling_thread(self):
        self.enrol()
        done = threading.Event()
        seen = {}

        def cb(v):
            seen["thread"] = threading.current_thread().name
            seen["label"] = v.label
            done.set()
        self.sid.submit(tts("Turn the volume down a little.", "Samantha"), cb)
        self.assertTrue(done.wait(10))
        self.assertEqual(seen["thread"], "SpeakerID")
        self.assertEqual(seen["label"], "you")

    def test_corrupt_profile_is_ignored_not_fatal(self):
        self.sid.profile_path.write_bytes(b"not a profile")
        s = SpeakerIdentity(model_dir=ROOT / "models", profile_path=self.sid.profile_path)
        self.assertEqual(s.status, "not_enrolled")


if __name__ == "__main__":
    unittest.main()
