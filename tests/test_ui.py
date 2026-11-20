"""
UI integration tests (off-screen Qt). Real widgets, real AudioEngine (fake devices),
real SpeakerIdentity; config is redirected to a temporary file so your settings
are never touched.

    .venv/bin/python -m unittest tests.test_ui -v
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import wave
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                                                        # noqa: E402
from PyQt6.QtCore import QTimer                                           # noqa: E402
from PyQt6.QtWidgets import QAbstractButton, QApplication                 # noqa: E402

import memory.config_manager as cm                                        # noqa: E402
import ui as U                                                            # noqa: E402
from audio.engine import IN_FRAME, AudioEngine, AudioSettings             # noqa: E402
from audio.speaker_id import ENROL_PHRASES, SpeakerIdentity               # noqa: E402

HAVE_SAY = shutil.which("say") is not None


class FakeStream:
    active, latency = True, 0.02

    def abort(self, ignore_errors=True): self.active = False
    def close(self, ignore_errors=True): self.active = False


def pump(ms: int = 50):
    """Run the Qt event loop for `ms` milliseconds."""
    app = QApplication.instance()
    end = time.monotonic() + ms / 1000
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.005)


def wait_for(cond, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        pump(20)
    return False


def tts16(text: str, voice: str = "Samantha") -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["say", "-v", voice, "-o", f"{td}/a.aiff", text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1", f"{td}/a.aiff", f"{td}/a.wav"],
                       check=True, capture_output=True)
        with wave.open(f"{td}/a.wav") as w:
            return np.frombuffer(w.readframes(w.getnframes()), np.int16)


class UITestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cfg = cls.tmp / "api_keys.json"
        cfg.write_text(json.dumps({"gemini_api_key": "TEST-KEY-not-real-0000000", "os_system": "mac",
                                   "assistant_name": "JARVIS"}))
        cls.patches = [mock.patch.object(cm, "CONFIG_FILE", cfg), mock.patch.object(cm, "CONFIG_DIR", cls.tmp),
                       mock.patch.object(U, "API_FILE", cfg), mock.patch.object(U, "CONFIG_DIR", cls.tmp)]
        for p in cls.patches:
            p.start()
        cls.cfg = cfg

    @classmethod
    def tearDownClass(cls):
        for p in cls.patches:
            p.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.eng = AudioEngine(AudioSettings(), on_chunk=lambda d: None)
        self.eng._open_streams = lambda: (setattr(self.eng, "_in_stream", FakeStream()),
                                          setattr(self.eng, "_out_stream", FakeStream()))
        self.eng.start()
        self.sid = SpeakerIdentity(profile_path=self.tmp / f"voice_{id(self)}.npz")
        self.j = U.JarvisUI("face.png")
        self.w = self.j._win
        self.w.resize(1180, 760)
        self.j.attach_audio(self.eng)
        self.j.attach_speaker_id(self.sid)
        self.j.set_remote_providers(lambda: {"available": True, "running": False, "phones": 0, "url": ""},
                                    lambda: 2, lambda: None)
        pump(100)

    def tearDown(self):
        self.eng.stop()
        self.w.close()
        self.w.deleteLater()
        pump(50)

    def feed(self, pcm16: np.ndarray, speed: float = 4.0):
        """Deliver mic audio like a device would: 10 ms blocks, `speed`× real time.

        (The engine's mic ring holds 1 s; flooding it faster than the DSP thread
        runs drops audio by design, exactly as a real overloaded device would.)
        """
        x = pcm16.reshape(-1, 1)
        for i in range(0, len(x) - IN_FRAME + 1, IN_FRAME):
            self.eng._in_cb(x[i:i + IN_FRAME], IN_FRAME, None, None)
            if (i // IN_FRAME) % 10 == 9:
                time.sleep(0.1 / speed)
                QApplication.instance().processEvents()


class WiringTests(UITestBase):
    def test_no_decorative_buttons(self):
        """Every button in the window and in every Settings section is connected to something."""
        self.w.open_settings("General")
        for sec in ("Audio", "Voice identity", "Remote", "About"):
            self.w._settings.show_section(sec)
        pump(100)
        unwired = []
        for b in self.w.findChildren(QAbstractButton):
            sigs = [b.clicked, b.toggled, b.pressed]
            if not any(b.receivers(s) for s in sigs) and type(b).__name__ not in ("QScrollBar",):
                unwired.append(f"{type(b).__name__}:{b.text() or b.toolTip() or b.objectName()}")
        self.assertEqual(unwired, [], f"buttons with no handler: {unwired}")

    def test_settings_audio_controls_change_the_engine_and_config(self):
        self.w.open_settings("Audio")
        pane = self.w._settings.panes["Audio"]
        pane.t_aec.click()
        self.assertTrue(wait_for(lambda: self.eng.settings.echo_cancellation is False))
        pane.seg._btns["light"].click()
        self.assertTrue(wait_for(lambda: self.eng.settings.noise_suppression == "light"))
        saved = json.loads(self.cfg.read_text())["audio"]
        self.assertFalse(saved["echo_cancellation"])
        self.assertEqual(saved["noise_suppression"], "light")
        self.assertEqual(json.loads(self.cfg.read_text())["gemini_api_key"], "TEST-KEY-not-real-0000000")
        pane.t_master.click()                                  # bypass disables dependent controls
        self.assertFalse(pane.t_aec.isEnabled())
        self.assertFalse(pane.seg._btns["standard"].isEnabled())

    def test_general_toggles_reflect_real_state(self):
        self.w.open_settings("General")
        g = self.w._settings.panes["General"]
        g.brief.click()
        self.assertFalse(json.loads(self.cfg.read_text())["morning_brief_enabled"])
        with mock.patch.object(type(self.w), "_toggle_autostart") as tog, \
                mock.patch.object(type(self.w), "_check_autostart", return_value=False):
            g.auto.set_quietly(False)
            g.auto.click()                         # OS refused → switch shows the real (off) state
            tog.assert_called_once()
            self.assertFalse(g.auto.isChecked())
        g.name.setText("Friday")
        with mock.patch.object(U, "apply_ui_accent", return_value=False):
            g._save()
        self.assertEqual(json.loads(self.cfg.read_text())["assistant_name"], "Friday")
        self.assertEqual(self.w._brand.text(), "FRIDAY")

    def test_remote_section_uses_backend(self):
        self.w.open_settings("Remote")
        r = self.w._settings.panes["Remote"]
        self.assertEqual(r.state.text(), "Off")
        self.assertFalse(r.forget.isEnabled())
        self.j.update_remote(True, 1)
        pump(50)
        self.assertIn("Phone connected", self.w.chip_phone.text())


class ApprovalTests(UITestBase):
    def ask(self, timeout=30):
        res = {}
        threading.Thread(target=lambda: res.setdefault("v", self.j.confirm_action("Send a message to Priya",
                                                                                   "To: Priya\nMessage: hi", timeout)),
                         daemon=True).start()
        self.assertTrue(wait_for(lambda: self.w._confirm_panel.isVisible()))
        return res

    def test_card_is_above_settings_and_escape_denies(self):
        self.w.open_settings("Audio")
        res = self.ask()
        cw = self.w.centralWidget()
        kids = [c for c in cw.children() if isinstance(c, U.QWidget) and c.isVisible()]
        self.assertGreater(kids.index(self.w._confirm_panel), kids.index(self.w._settings), "card under settings")
        self.assertTrue(self.w.chip_approval.isVisible())
        self.w._on_escape()                        # Esc = Deny, Settings stays open
        self.assertTrue(wait_for(lambda: "v" in res))
        self.assertFalse(res["v"])
        self.assertIsNotNone(self.w._settings)
        self.assertFalse(self.w.chip_approval.isVisible())

    def test_allow_and_scrim_click_never_answers(self):
        res = self.ask()
        self.w._scrim_clicked()
        pump(50)
        self.assertNotIn("v", res)
        self.w._confirm_panel._allow_btn.click()
        self.assertTrue(wait_for(lambda: res.get("v") is True))

    def test_no_default_button(self):
        self.ask()
        for b in (self.w._confirm_panel._allow_btn, self.w._confirm_panel._deny_btn):
            self.assertFalse(b.isDefault() or b.autoDefault())
        self.w._confirm_panel._finish(False)


class ConversationTests(UITestBase):
    def kinds(self):
        return [r.kind for r in self.w.convo.convo.rows()]

    def test_messages_tools_and_legacy_lines(self):
        j = self.j
        j.add_message("user", "hello", "voice")
        j.tool_event("t1", "web_search", "started", "cats")
        j.write_log("[Code] Running x.py")
        j.tool_event("t1", "web_search", "done")
        j.add_message("assistant", "Here you go.")
        j.write_log("SYS: Phone connected via Remote Dashboard.")
        j.write_log("ERR: Something broke")
        j.write_log("[Web]: from phone")
        pump(100)
        self.assertEqual(self.kinds(), ["you", "tool", "note", "assistant", "note", "error", "you"])
        tool = [r for r in self.w.convo.convo.rows() if r.kind == "tool"][0]
        self.assertEqual(tool.status, "done")

    def test_tool_approval_states_and_stage_activity(self):
        self.j.set_state("THINKING")
        self.j.tool_event("t2", "send_message", "started", "Priya")
        pump(50)
        self.assertIn("Send message", self.w.stage.caption.text())
        self.j.tool_event("t2", "send_message", "approval")
        pump(50)
        self.assertIn("approval", self.w.stage.caption.text())
        self.j.tool_event("t2", "send_message", "declined")
        pump(50)
        row = [r for r in self.w.convo.convo.rows() if r.kind == "tool"][-1]
        self.assertEqual(row.status, "declined")

    def test_results_and_typed_messages(self):
        sent = []
        self.j.on_text_command = sent.append
        self.w.convo.input.setText("open notes")
        self.w.convo.send_btn.click()
        self.assertTrue(wait_for(lambda: sent == ["open notes"]))
        self.j.show_content("NEWS", "headline")
        pump(50)
        self.assertEqual(self.w.convo.tabs[1].text(), "Results  1")

    def test_history_is_capped(self):
        for i in range(450):
            self.w.convo.add_note(f"n{i}")
        self.assertLessEqual(len(self.w.convo.convo.rows()), 400)


class StatusTests(UITestBase):
    def test_states_connection_mute_and_speaker(self):
        self.j.set_connection("reconnecting", "retrying")
        pump(30)
        self.assertEqual(self.w.chip_conn.tone, "warn")
        self.j.set_connection("offline", "Your Gemini API key was rejected.")
        pump(30)
        self.assertEqual(self.w.stage.visual_state, "error")
        self.j.set_state("LISTENING")
        pump(30)
        self.assertEqual(self.w.chip_conn.text(), "Connected")
        self.assertNotEqual(self.w.stage.visual_state, "error")
        self.j.set_state("SPEAKING"); pump(30)
        self.assertEqual(self.w.stage.visual_state, "speaking")
        self.assertTrue(self.w.stage.interrupt_btn.isVisible())
        self.w.stage.mic.click(); pump(30)
        self.assertTrue(self.j.muted)
        self.assertEqual(self.w.chip_mic.text(), "Muted")
        self.assertEqual(self.w.stage.visual_state, "muted")
        # "not set up" when the speaker model isn't downloaded (e.g. a fresh checkout), else "not enrolled"
        self.assertIn(self.w.chip_voice.text(), ("Voice: not enrolled", "Voice: not set up"))

    def test_window_sizes(self):
        for wd, ht in ((820, 580), (1100, 700), (1440, 900)):
            self.w.resize(wd, ht); pump(80)
            self.assertEqual(self.w.convo.isVisible(), wd >= 1100, wd)
            self.assertEqual(self.w._convo_btn.isVisible(), wd < 1100, wd)
        self.w.resize(820, 580); pump(50)
        self.w._convo_btn.click(); pump(50)
        self.assertTrue(self.w.convo.isVisible())


@unittest.skipUnless(HAVE_SAY, "needs macOS say")
class VoiceIdentityFlowTests(UITestBase):
    def test_enrol_through_the_settings_ui_then_recognise(self):
        if self.sid.status == "no_model":
            self.skipTest("speaker model not downloaded")
        self.w.open_settings("Voice identity")
        vp = self.w._settings.panes["Voice identity"]
        self.assertEqual(vp.state_lbl.text(), "No voice profile yet")
        vp.b_primary.click()
        self.assertTrue(vp.rec.isVisible())
        self.assertTrue(getattr(self.w, "_voice_setup", False), "phrases read aloud must not reach Gemini")
        for i, phrase in enumerate(ENROL_PHRASES):
            vp.rec_go.click()
            self.assertTrue(self.eng.mic_test_running)
            pcm = tts16(phrase)
            pcm = np.concatenate([pcm, np.zeros(16000 * 5, np.int16)])[: 16000 * 5]
            self.feed(pcm)
            self.assertTrue(wait_for(lambda: not self.eng.mic_test_running, 10))
            pump(200)
        # the three "other conditions" samples are optional
        from audio.speaker_id import ENROL_CONDITIONS
        for _ in ENROL_CONDITIONS:
            self.assertTrue(vp.rec_skip.isVisible())
            vp.rec_skip.click()
            pump(50)
        self.assertTrue(wait_for(lambda: self.sid.status == "ready", 30), vp.rec_msg.text())
        pump(100)
        self.assertFalse(self.w._voice_setup, "the mic goes back to Gemini after enrolment")
        self.assertEqual(vp.state_lbl.text(), "Profile ready")
        self.assertTrue(os.path.exists(self.sid.profile_path))
        # live recognition: an utterance being sent to Gemini is checked in the background
        got = {}
        self.eng.on_utterance = lambda pcm: self.sid.submit(pcm, lambda v: (got.setdefault("v", v),
                                                                             self.j.set_speaker(v.label, v.score)))
        self.j.add_message("user", "test", "voice")
        self.feed(np.concatenate([tts16("What time is the meeting tomorrow afternoon?"), np.zeros(8000, np.int16)]))
        self.assertTrue(wait_for(lambda: "v" in got, 15))
        pump(100)
        self.assertEqual(got["v"].label, "you")
        self.assertTrue(self.w.chip_voice.text().startswith("You"))
        # accuracy check: one spoken step, one skipped, then stopped — the result is still saved
        import json, tempfile, ui_settings
        out = os.path.join(tempfile.mkdtemp(), "check.json")
        with mock.patch.object(ui_settings, "check_results_path", return_value=out):
            vp.b_check.click(); pump(50)
            self.assertTrue(vp.rec.isVisible())
            self.assertTrue(self.w._voice_setup, "check phrases must not reach Gemini")
            vp.rec_go.click()
            self.assertTrue(self.eng.mic_test_running)
            pcm = np.concatenate([tts16("Open my calendar for tomorrow."), np.zeros(16000 * 5, np.int16)])[: 16000 * 5]
            self.feed(pcm)
            self.assertTrue(wait_for(lambda: vp._idx == 1, 20), vp.rec_msg.text())
            pump(50)
            vp.rec_skip.click(); pump(50)
            vp.rec_cancel.click(); pump(50)
            self.assertFalse(self.w._voice_setup)
            with open(out, encoding="utf-8") as f:
                saved = json.load(f)
        self.assertEqual(len(saved), 1)
        self.assertFalse(saved[0]["complete"])
        self.assertEqual([(r["step"], r["expect"], r["accepted"]) for r in saved[0]["results"]],
                         [("normal", "you", True)])
        # delete needs two clicks
        vp.b_delete.click()
        self.assertTrue(os.path.exists(self.sid.profile_path))
        vp.b_delete.click()
        self.assertFalse(os.path.exists(self.sid.profile_path))


class ResponsivenessTests(UITestBase):
    def test_gui_thread_never_stalls(self):
        """While speaking, with messages and tool events arriving from a worker thread."""
        self.j.set_state("SPEAKING")
        gaps, last = [], [time.monotonic()]

        def beat():
            now = time.monotonic(); gaps.append(now - last[0]); last[0] = now
        t = QTimer(); t.timeout.connect(beat); t.start(10)

        def worker():
            for i in range(60):
                self.j.add_message("assistant", f"reply {i} " * 8)
                self.j.tool_event(f"x{i}", "web_search", "started", "q")
                self.j.tool_event(f"x{i}", "web_search", "done")
                self.j.set_speaking_text(f"partial {i}")
                time.sleep(0.05)
        th = threading.Thread(target=worker); th.start()
        end = time.monotonic() + 4
        while time.monotonic() < end:
            QApplication.instance().processEvents(); time.sleep(0.002)
        th.join(); t.stop()
        worst = max(gaps[5:]) * 1000
        self.assertLess(worst, 100, f"GUI thread stalled {worst:.0f} ms")


if __name__ == "__main__":
    unittest.main()
