"""
ui_settings.py — the Settings sheet: General · Audio · Voice identity · Remote · About.

One window replaces the old ⚙ drop-down and its separate pop-ups. Every
control reads and writes real state through a small controller interface that
MainWindow implements (see `SettingsController` below); nothing here is a
placeholder. Slow work (device rescans, round-trip measurement, recording,
voiceprint computation) runs on worker threads and reports back via signals.
"""

from __future__ import annotations

import dataclasses
import os
import platform
import threading
import time

import numpy as np
from PyQt6.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPainter, QPen
from PyQt6.QtWidgets import (QComboBox, QFrame, QGridLayout, QHBoxLayout, QLineEdit, QPushButton,
                             QScrollArea, QSizePolicy, QSlider, QStackedWidget, QVBoxLayout, QWidget)

from ui_theme import (C, LevelMeter, Segmented, Toggle, UIFont, button, caps, fade_in, label, qcol,
                      reduce_motion)


class SettingsController:
    """What the Settings sheet needs from the app (MainWindow implements this)."""
    audio = None                 # AudioEngine | None
    speaker_id = None            # SpeakerIdentity | None

    def read_config(self) -> dict: return {}
    def save_identity(self, name: str, user: str, color: str) -> None: ...
    def preview_accent(self, hex_color: str) -> None: ...
    def default_accent(self) -> str: return C.PRI
    def brief_enabled(self) -> bool: return True
    def set_brief(self, on: bool) -> None: ...
    def autostart_enabled(self) -> bool: return False
    def set_autostart(self, on: bool) -> None: ...
    def create_shortcut(self) -> None: ...
    def save_audio(self, settings: dict) -> None: ...
    def show_speaker(self) -> bool: return True
    def set_show_speaker(self, on: bool) -> None: ...
    def record_audio(self, seconds: float, on_done) -> bool: return False
    def remote_status(self) -> dict: return {}
    def open_pairing(self) -> None: ...
    def forget_phones(self) -> int: return 0
    def stop_remote(self) -> None: ...
    def toast(self, title: str, body: str = "", tone: str = "neutral") -> None: ...


# ── pane scaffolding ─────────────────────────────────────────────────────────

class Pane(QScrollArea):
    """Scrollable settings page with consistent rows."""

    def __init__(self, title: str, lead: str = ""):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(f"""
            QScrollArea {{ background: transparent; border: none; }}
            QScrollBar:vertical {{ background: transparent; width: 6px; margin: 2px; }}
            QScrollBar::handle:vertical {{ background: {C.BORDER_B}; border-radius: 3px; min-height: 30px; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
        """)
        self.body = QWidget(); self.body.setStyleSheet("background: transparent;")
        self.lay = QVBoxLayout(self.body)
        self.lay.setContentsMargins(26, 22, 22, 22)
        self.lay.setSpacing(4)
        self.setWidget(self.body)
        self.lay.addWidget(label(title, 17, C.TEXT, QFont.Weight.Medium))
        if lead:
            self.lay.addSpacing(2)
            self.lay.addWidget(label(lead, 11.5, C.TEXT_MED, wrap=True))
        self.lay.addSpacing(8)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.body.setFixedWidth(self.viewport().width())

    def finish(self):
        self.lay.addStretch(1)

    def section(self, title: str):
        self.lay.addSpacing(14)
        self.lay.addWidget(caps(title))
        self.lay.addSpacing(4)

    def _row(self, title: str, hint: str, control: QWidget | None, align_top=False) -> QWidget:
        w = QWidget(); w.setObjectName("Row")
        w.setStyleSheet(f"#Row {{ border-bottom: 1px solid #2A2D32; }}")
        h = QHBoxLayout(w); h.setContentsMargins(0, 9, 0, 9); h.setSpacing(14)
        col = QVBoxLayout(); col.setSpacing(2)
        col.addWidget(label(title, 12.5))
        if hint:
            col.addWidget(label(hint, 11, C.TEXT_DIM, wrap=True))
        h.addLayout(col, 1)
        if control is not None:
            h.addWidget(control, 0, Qt.AlignmentFlag.AlignTop if align_top else Qt.AlignmentFlag.AlignVCenter)
        self.lay.addWidget(w)
        return w

    def toggle(self, title: str, hint: str, value: bool, on_change) -> Toggle:
        t = Toggle(bool(value))
        t.toggled.connect(on_change)
        self._row(title, hint, t)
        return t

    def control(self, title: str, hint: str, widget: QWidget) -> QWidget:
        return self._row(title, hint, widget)


def _combo() -> QComboBox:
    cb = QComboBox(); cb.setFont(UIFont(11.5))
    cb.setMinimumWidth(210)
    cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    cb.setMinimumContentsLength(16)
    cb.setStyleSheet(f"""
        QComboBox {{ background: {C.INPUT}; color: {C.TEXT}; border: 1px solid {C.BORDER};
                     border-radius: 7px; padding: 4px 10px; min-height: 18px; }}
        QComboBox:hover {{ border-color: {C.BORDER_B}; }}
        QComboBox::drop-down {{ border: none; width: 18px; }}
        QComboBox QAbstractItemView {{ background: {C.PANEL2}; color: {C.TEXT};
                     selection-background-color: {C.HOVER}; border: 1px solid {C.BORDER}; }}
    """)
    return cb


def _field(text: str = "", placeholder: str = "") -> QLineEdit:
    f = QLineEdit(text); f.setPlaceholderText(placeholder); f.setFont(UIFont(12)); f.setFixedHeight(30)
    f.setMinimumWidth(210)
    f.setStyleSheet(f"""
        QLineEdit {{ background: {C.INPUT}; color: {C.TEXT}; border: 1px solid {C.BORDER};
                     border-radius: 7px; padding: 0 10px; }}
        QLineEdit:focus {{ border-color: {C.BORDER_B}; }}
    """)
    return f


def _slider() -> QSlider:
    s = QSlider(Qt.Orientation.Horizontal)
    s.setStyleSheet(f"""
        QSlider::groove:horizontal {{ height: 4px; background: {C.HOVER}; border-radius: 2px; }}
        QSlider::sub-page:horizontal {{ background: {C.PRI}; border-radius: 2px; }}
        QSlider::handle:horizontal {{ background: {C.TEXT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
    """)
    return s


def _rowbox(*widgets, stretch_first=False) -> QWidget:
    w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(8)
    for i, x in enumerate(widgets):
        h.addWidget(x, 1 if (stretch_first and i == 0) else 0)
    return w


# ── General ──────────────────────────────────────────────────────────────────

class GeneralPane(Pane):
    def __init__(self, ctl: SettingsController):
        super().__init__("General", "How the assistant introduces itself, its look, and how it starts.")
        self.ctl = ctl
        cfg = ctl.read_config()
        self.section("Identity")
        self.name = _field(cfg.get("assistant_name") or "JARVIS", "JARVIS")
        self.control("Assistant name", "Shown in the window and used by the assistant to refer to itself.", self.name)
        self.user = _field(cfg.get("user_name") or "", "Leave blank for “sir”")
        self.control("What it calls you", "Leave blank to be addressed as “sir”.", self.user)

        self.section("Appearance")
        from ui import HueWheel                           # reuse the existing colour wheel
        self._color = (cfg.get("ui_color") or ctl.default_accent()).lower()
        self.wheel = HueWheel(self._color)
        self.wheel.setFixedSize(150, 150)
        self.wheel.hue_picked.connect(self._pick)
        self.wheel.hue_committed.connect(self._pick)
        default = button("Pure silver", "ghost", 28)
        default.clicked.connect(lambda: (self.wheel.set_color(ctl.default_accent()), self._pick(ctl.default_accent())))
        box = QWidget(); bl = QVBoxLayout(box); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(6)
        bl.addWidget(self.wheel, 0, Qt.AlignmentFlag.AlignRight); bl.addWidget(default, 0, Qt.AlignmentFlag.AlignRight)
        self._row("Accent tint", "A subtle tint on the silver accent. Saturation is capped so the interface "
                  "stays graphite.", box, align_top=True)

        save = button("Save name and look", "primary", 30)
        save.clicked.connect(self._save)
        self.lay.addSpacing(8); self.lay.addWidget(save, 0, Qt.AlignmentFlag.AlignLeft)

        self.section("Startup")
        self.brief = self.toggle("Morning briefing", "Greets you and reads the top headlines when it first connects.",
                                 ctl.brief_enabled(), ctl.set_brief)
        self.auto = self.toggle("Open at login", "Starts the assistant automatically when you log in.",
                                ctl.autostart_enabled(), self._autostart)
        sc = button("Create shortcut", "secondary", 28)
        sc.clicked.connect(ctl.create_shortcut)
        self.control("Desktop shortcut", "Adds a launcher to your desktop.", sc)
        self.finish()

    def _pick(self, hx: str):
        self._color = hx.lower()
        self.ctl.preview_accent(self._color)

    def _autostart(self, on: bool):
        self.ctl.set_autostart(on)
        self.auto.set_quietly(self.ctl.autostart_enabled())       # show what the OS really did

    def _save(self):
        self.ctl.save_identity(self.name.text().strip(), self.user.text().strip(), self._color)
        self.ctl.toast("Saved", "Name and look updated.", "ok")

    def refresh(self):
        self.brief.set_quietly(self.ctl.brief_enabled())
        self.auto.set_quietly(self.ctl.autostart_enabled())


# ── Audio ────────────────────────────────────────────────────────────────────

class AudioPane(Pane):
    _devices = pyqtSignal(dict)
    _rtt = pyqtSignal(object)

    def __init__(self, ctl: SettingsController):
        super().__init__("Audio", "Your microphone and JARVIS's voice. Changes apply immediately.")
        self.ctl = ctl
        eng = ctl.audio
        self.eng = eng
        if eng is None:
            self.lay.addWidget(label("The audio engine isn't running, so these settings are unavailable. "
                                     "The activity log has the reason.", 12, C.RED, wrap=True))
            self.finish()
            return
        s = eng.settings

        self.section("Microphone")
        self.in_combo = _combo()
        rescan = button("↻", "ghost", 30); rescan.setFixedWidth(34); rescan.setToolTip("Rescan devices")
        rescan.clicked.connect(lambda: self._load_devices(True))
        self.control("Input device", "", _rowbox(self.in_combo, rescan))
        self.in_combo.activated.connect(lambda _i: self._set(input_device=self.in_combo.currentData()))
        self.t_iso = self.toggle("Voice isolation", "Cleans up your voice before it's sent: echo cancellation, "
                                 "noise suppression and gain.", s.voice_isolation, lambda on: self._set(voice_isolation=on))
        self.seg = Segmented([("off", "Off"), ("light", "Light"), ("standard", "Standard")], s.noise_suppression)
        self.seg.changed.connect(lambda k: self._set(noise_suppression=k))
        self.control("Noise suppression", "Standard removes fans and hum. Light keeps more of the room.", self.seg)
        self.t_aec = self.toggle("Echo cancellation", "Removes JARVIS's own voice from your microphone.",
                                 s.echo_cancellation, lambda on: self._set(echo_cancellation=on))
        self.t_agc = self.toggle("Automatic gain", "Evens out quiet and loud speech. Holds still while JARVIS talks.",
                                 s.auto_gain, lambda on: self._set(auto_gain=on))
        self.meter = LevelMeter()
        self.vad = label("", 11, C.TEXT_DIM)
        self.mic_btn = button("Test microphone", "secondary", 28)
        self.mic_btn.clicked.connect(self._mic_test)
        mt = QWidget(); ml = QVBoxLayout(mt); ml.setContentsMargins(0, 0, 0, 0); ml.setSpacing(6)
        ml.addWidget(self.meter); ml.addWidget(_rowbox(self.vad, self.mic_btn, stretch_first=True))
        mt.setMinimumWidth(260)
        self.control("Live input", "Level after processing. The dot lights when speech is detected. "
                     "The test records 5 s and plays it back; nothing is sent to Gemini.", mt)

        self.section("JARVIS speaker")
        self.out_combo = _combo()
        rescan2 = button("↻", "ghost", 30); rescan2.setFixedWidth(34); rescan2.setToolTip("Rescan devices")
        rescan2.clicked.connect(lambda: self._load_devices(True))
        self.control("Output device", "", _rowbox(self.out_combo, rescan2))
        self.out_combo.activated.connect(lambda _i: self._set(output_device=self.out_combo.currentData()))
        self.t_enh = self.toggle("Output enhancement", "Keeps JARVIS's voice even and prevents clipping. This shapes "
                                 "what you hear; it doesn't remove noise in your room.",
                                 s.output_enhancement, lambda on: self._set(output_enhancement=on))
        self.t_norm = self.toggle("Volume normalisation", "Brings quiet and loud replies to the same level.",
                                  s.normalization, lambda on: self._set(normalization=on))
        self.t_duck = self.toggle("Lower other apps while JARVIS talks",
                                  eng.ducker.reason or "Other apps drop to 30 % and come back afterwards.",
                                  s.ducking, lambda on: self._set(ducking=on))
        self.vol = _slider(); self.vol.setRange(0, 150); self.vol.setValue(int(round(s.output_volume * 100)))
        self.vol.setMinimumWidth(200)
        self.vol_lbl = label(f"{self.vol.value()} %", 11, C.TEXT_MED, mono=True)
        self.vol.valueChanged.connect(self._volume)
        self.control("Volume", "", _rowbox(self.vol, self.vol_lbl, stretch_first=True))
        tone = button("Play test sound", "secondary", 28)
        tone.clicked.connect(eng.play_test_tone)
        self.control("Speaker test", "Plays a short chime through the full output chain.", tone)

        self.section("Advanced")
        self.t_master = self.toggle("Audio processing", "Off = bypass: raw microphone and speaker. Use this if "
                                    "anything sounds wrong.", s.processing_enabled,
                                    lambda on: self._set(processing_enabled=on))
        self.t_duplex = self.toggle("Talk over JARVIS (experimental)",
                                    "Keeps the mic open while JARVIS speaks so you can interrupt by voice. Needs echo "
                                    "cancellation; on laptop speakers your voice is turned down while JARVIS talks.",
                                    s.full_duplex, lambda on: self._set(full_duplex=on))
        grid = QGridLayout(); grid.setHorizontalSpacing(16); grid.setVerticalSpacing(5)
        self._stat = {}
        rows = [("round_trip_ms", "Speaker → mic round trip"), ("processing_delay_ms", "Processing delay"),
                ("output_latency_ms", "Speaker buffer"), ("dsp", "Processing load"),
                ("playback_buffer_ms", "Queued speech"), ("reply_latency_ms", "Reply time (median)"),
                ("health", "Dropouts")]
        for i, (k, name) in enumerate(rows):
            grid.addWidget(label(name, 11, C.TEXT_MED), i, 0)
            v = label("—", 11, C.TEXT, mono=True); grid.addWidget(v, i, 1); self._stat[k] = v
        self.rtt_btn = button("Measure", "ghost", 24)
        self.rtt_btn.clicked.connect(self._measure)
        grid.addWidget(self.rtt_btn, 0, 2)
        grid.setColumnStretch(1, 1)
        gw = QWidget(); gw.setLayout(grid)
        self.lay.addSpacing(6); self.lay.addWidget(gw)
        self.status = label("", 11, C.TEXT_DIM, wrap=True)
        self.lay.addSpacing(6); self.lay.addWidget(self.status)
        self.finish()

        self._devices.connect(self._fill_devices)
        self._rtt.connect(self._rtt_done)
        self._load_devices(False)
        self._sync_enabled()
        self._fast = QTimer(self); self._fast.timeout.connect(self._tick_meter); self._fast.start(33)
        self._slow = QTimer(self); self._slow.timeout.connect(self._tick_stats); self._slow.start(300)
        self._tick_stats()

    # behaviour
    def _set(self, **changes):
        s = dataclasses.replace(self.eng._pending_settings or self.eng.settings, **changes)
        self.eng.apply_settings(s)
        try:
            self.ctl.save_audio(s.to_dict())
        except Exception as e:
            self.ctl.toast("Couldn't save audio settings", str(e), "bad")
        self._sync_enabled(s)

    def _sync_enabled(self, s=None):
        s = s or self.eng._pending_settings or self.eng.settings
        on = s.processing_enabled
        iso = on and s.voice_isolation
        self.t_iso.setEnabled(on)
        self.seg.setEnabled(iso)
        self.t_aec.setEnabled(iso)
        self.t_agc.setEnabled(iso)
        self.t_enh.setEnabled(on)
        self.t_norm.setEnabled(on and s.output_enhancement)
        self.t_duck.setEnabled(on and self.eng.ducker.available)
        self.t_duplex.setEnabled(iso and s.echo_cancellation)

    def _volume(self, v: int):
        self.vol_lbl.setText(f"{v} %")
        self._set(output_volume=v / 100.0)

    def _load_devices(self, rescan: bool):
        def work():
            from audio.engine import list_devices
            try:
                if rescan:
                    self.eng.refresh_devices()
                self._devices.emit(list_devices())
            except Exception as e:
                self._devices.emit({"error": str(e)})
        threading.Thread(target=work, daemon=True).start()

    def _fill_devices(self, d: dict):
        if "error" in d:
            self.ctl.toast("Couldn't list audio devices", d["error"], "bad")
            return
        for cb, kind, cur in ((self.in_combo, "input", self.eng.settings.input_device),
                              (self.out_combo, "output", self.eng.settings.output_device)):
            cb.blockSignals(True); cb.clear()
            dflt = d.get(f"default_{kind}")
            cb.addItem(f"System default{f' — {dflt}' if dflt else ''}", None)
            for n in d.get(kind, []):
                cb.addItem(n, n)
            if cur and cur not in d.get(kind, []):
                cb.addItem(f"{cur} (not connected)", cur)
            i = cb.findData(cur); cb.setCurrentIndex(i if i >= 0 else 0)
            cb.blockSignals(False)

    def _mic_test(self):
        if self.eng.mic_test_running:
            return
        self.mic_btn.setEnabled(False); self.mic_btn.setText("Recording… speak now")
        self.eng.start_mic_test(5.0)

    def _measure(self):
        self.rtt_btn.setEnabled(False)
        self._stat["round_trip_ms"].setText("measuring…")
        threading.Thread(target=lambda: self._rtt.emit(self.eng.measure_round_trip()), daemon=True).start()

    def _rtt_done(self, v):
        self.rtt_btn.setEnabled(True)
        if v is None:
            self.ctl.toast("Round trip not measured", "The mic didn't hear the click. Turn the volume up or check "
                           "the devices.", "warn")
        self._tick_stats()

    def _tick_meter(self):
        lv = self.eng.levels()
        self.meter.set_level(lv.get("mic_db", -120.0), lv.get("speech", False))
        self.vad.setText("Speech detected" if lv.get("speech") else ("Sending to Gemini" if lv.get("sending") else
                                                                       "Not sending (muted or JARVIS speaking)"))
        if not self.mic_btn.isEnabled() and not self.eng.mic_test_running:
            if self.eng.is_playing:
                self.mic_btn.setText("Playing back…")
            else:
                self.mic_btn.setText("Test microphone"); self.mic_btn.setEnabled(True)

    def _tick_stats(self):
        st = self.eng.stats()
        ms = lambda v: "—" if v is None else f"{v:.0f} ms"        # noqa: E731
        if self.rtt_btn.isEnabled():
            self._stat["round_trip_ms"].setText(ms(st["round_trip_ms"]))
        self._stat["processing_delay_ms"].setText(ms(st["processing_delay_ms"]))
        self._stat["output_latency_ms"].setText(ms(st["output_latency_ms"]))
        self._stat["dsp"].setText(f"{st['dsp_cpu_pct']:.1f} % of one core")
        self._stat["playback_buffer_ms"].setText(ms(st["playback_buffer_ms"]))
        self._stat["reply_latency_ms"].setText(ms(st["reply_latency_ms"]))
        self._stat["health"].setText(f"{st['dropped_mic_blocks']} mic · {st['underruns']} speaker")
        probs = [x for x in (st["input_error"], st["output_error"]) if x]
        if st["processing_error"]:
            probs.insert(0, f"Processing switched to bypass: {st['processing_error']}")
        if probs:
            self.status.setStyleSheet(f"color: {C.RED}; background: transparent;")
            self.status.setText("  ·  ".join(probs))
        else:
            self.status.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent;")
            s = self.eng.settings
            self.status.setText("Processing active." if (s.processing_enabled and st["processing_active"])
                                else "Bypass: raw microphone and speaker.")

    def stop(self):
        for t in ("_fast", "_slow"):
            if hasattr(self, t):
                getattr(self, t).stop()


# ── Voice identity ───────────────────────────────────────────────────────────

class ConfidenceRing(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(112, 112)
        self.score: float | None = None
        self.label = "No reading yet"
        self.tone = C.TEXT_DIM

    def set(self, score, text, tone):
        self.score, self.label, self.tone = score, text, tone
        self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(8, 8, 96, 96)
        p.setPen(QPen(qcol(C.HOVER), 7)); p.drawEllipse(r)
        if self.score is not None:
            p.setPen(QPen(qcol(self.tone), 7, cap=Qt.PenCapStyle.RoundCap))
            p.drawArc(r, 90 * 16, -int(max(0.0, min(1.0, self.score)) * 360 * 16))
        p.setPen(QPen(qcol(C.TEXT)))
        p.setFont(UIFont(18, QFont.Weight.Light))
        p.drawText(QRectF(0, 30, 112, 30), Qt.AlignmentFlag.AlignCenter,
                   "—" if self.score is None else f"{self.score:.2f}")
        p.setPen(QPen(qcol(C.TEXT_MED))); p.setFont(UIFont(9.5))
        p.drawText(QRectF(10, 60, 92, 20), Qt.AlignmentFlag.AlignCenter, self.label)


class VoicePane(Pane):
    _recorded = pyqtSignal(object)
    _job_done = pyqtSignal(str, object)

    TONE = {"you": C.GREEN, "uncertain": C.ACC2, "unknown": C.ACC2, "too_short": C.TEXT_DIM}

    def __init__(self, ctl: SettingsController):
        super().__init__("Voice identity",
                         "The assistant learns what your voice sounds like so it can show when it's you speaking. "
                         "This is separate from transcription: it says who is speaking, not what was said.")
        self.ctl = ctl
        self.sid = ctl.speaker_id
        self._mode = None              # None | "enrol" | "add" | "test"
        self._samples: list = []
        self._idx = 0

        card = QFrame(); card.setObjectName("VCard")
        card.setStyleSheet(f"#VCard {{ background: #1D1F23; border: 1px solid {C.BORDER}; border-radius: 12px; }}")
        cl = QHBoxLayout(card); cl.setContentsMargins(16, 14, 16, 14); cl.setSpacing(18)
        self.ring = ConfidenceRing()
        cl.addWidget(self.ring, 0, Qt.AlignmentFlag.AlignTop)
        info = QVBoxLayout(); info.setSpacing(6)
        self.state_lbl = label("", 13.5, C.TEXT, QFont.Weight.Medium)
        self.detail_lbl = label("", 11.5, C.TEXT_MED, wrap=True)
        info.addWidget(self.state_lbl); info.addWidget(self.detail_lbl)
        self.bar = QWidget(); self.bar.setFixedHeight(18)
        self.bar.paintEvent = self._paint_bar
        info.addWidget(self.bar)
        cl.addLayout(info, 1)
        self.lay.addWidget(card)

        # recording area (used for enrolment, adding samples and testing)
        self.rec = QFrame(); self.rec.setObjectName("Rec")
        self.rec.setStyleSheet(f"#Rec {{ background: {C.INPUT}; border: 1px solid {C.BORDER}; border-radius: 12px; }}")
        rl = QVBoxLayout(self.rec); rl.setContentsMargins(16, 14, 16, 14); rl.setSpacing(8)
        self.rec_step = caps("")
        self.rec_phrase = label("", 15, C.TEXT, QFont.Weight.Light, wrap=True)
        self.rec_meter = LevelMeter(show_dot=True)
        self.rec_msg = label("", 11.5, C.TEXT_MED, wrap=True)
        rb = QHBoxLayout(); rb.setSpacing(8)
        self.rec_go = button("Record", "primary", 30)
        self.rec_go.clicked.connect(self._record)
        self.rec_cancel = button("Cancel", "ghost", 30)
        self.rec_cancel.clicked.connect(self._cancel)
        rb.addWidget(self.rec_go); rb.addWidget(self.rec_cancel); rb.addStretch()
        for w in (self.rec_step, self.rec_phrase, self.rec_meter, self.rec_msg):
            rl.addWidget(w)
        rl.addLayout(rb)
        self.rec.hide()
        self.lay.addSpacing(10); self.lay.addWidget(self.rec)

        self.actions = QWidget(); al = QHBoxLayout(self.actions); al.setContentsMargins(0, 10, 0, 0); al.setSpacing(8)
        self.b_primary = button("Enrol my voice", "primary", 30)
        self.b_primary.clicked.connect(self._primary)
        self.b_test = button("Test recognition", "secondary", 30); self.b_test.clicked.connect(lambda: self._start("test"))
        self.b_add = button("Add samples", "secondary", 30); self.b_add.clicked.connect(lambda: self._start("add"))
        self.b_retrain = button("Retrain from scratch", "secondary", 30); self.b_retrain.clicked.connect(self._retrain)
        self.b_delete = button("Delete profile", "danger", 30); self.b_delete.clicked.connect(self._delete)
        for b in (self.b_primary, self.b_test, self.b_add, self.b_retrain):
            al.addWidget(b)
        al.addStretch(); al.addWidget(self.b_delete)
        self.lay.addWidget(self.actions)

        self.section("Display")
        self.t_show = self.toggle("Show who's speaking", "Adds the “You / Unknown speaker” status chip and a label "
                                  "on your messages.", ctl.show_speaker(), ctl.set_show_speaker)
        self.lay.addSpacing(12)
        note = label("Voice recognition is a convenience, not security. Recordings and people with similar voices can "
                     "fool it, so it never approves actions — approvals always need your click. Your voice profile "
                     "(numbers describing your voice, no recordings) stays on this Mac and is never sent to Gemini.",
                     11, C.TEXT_DIM, wrap=True)
        note.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent; border-left: 2px solid {C.BORDER_B};"
                           f" padding-left: 10px;")
        self.lay.addWidget(note)
        self.finish()

        self._recorded.connect(self._on_recorded)
        self._job_done.connect(self._on_job)
        self._meter_t = QTimer(self); self._meter_t.timeout.connect(self._tick); self._meter_t.start(33)
        self.refresh()

    # status
    def refresh(self):
        sid = self.sid
        st = sid.status if sid else "unavailable"
        ready = st == "ready"
        self.b_primary.setText({"no_model": "Download voice model (26 MB)", "ready": "Enrol again"}.get(st, "Enrol my voice"))
        self.b_primary.setVisible(st in ("no_model", "not_enrolled"))
        self.b_primary.setEnabled(st in ("no_model", "not_enrolled") and self._mode is None)
        for b in (self.b_test, self.b_add, self.b_retrain, self.b_delete):
            b.setVisible(ready)
            b.setEnabled(self._mode is None)
        if st == "unavailable":
            self.state_lbl.setText("Unavailable")
            self.detail_lbl.setText(f"The speaker model couldn't be loaded{': ' + sid.error if sid and sid.error else ''}.")
        elif st == "no_model":
            self.state_lbl.setText("Not set up")
            self.detail_lbl.setText("Voice identity needs a 26 MB speaker model, downloaded once and checked against "
                                    "a known checksum.")
        elif st == "not_enrolled":
            self.state_lbl.setText("No voice profile yet")
            self.detail_lbl.setText("Enrolment takes about a minute: read 5 short phrases out loud.")
        else:
            s = sid.summary
            when = time.strftime("%d %b %Y", time.localtime(s["created"]))
            self.state_lbl.setText("Profile ready")
            self.detail_lbl.setText(f"{s['samples']} samples · enrolled {when} · match threshold {s['threshold']:.2f}")
        v = sid.last if sid else None
        if v and v.score is not None:
            self.ring.set(v.score, v.text, self.TONE.get(v.label, C.TEXT_DIM))
        elif v:
            self.ring.set(None, v.text, C.TEXT_DIM)
        else:
            self.ring.set(None, "No reading yet", C.TEXT_DIM)
        self.bar.update()

    def _paint_bar(self, _):
        sid = self.sid
        if not (sid and sid.status == "ready"):
            return
        p = QPainter(self.bar); p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.bar.width(), self.bar.height()
        p.setPen(Qt.PenStyle.NoPen); p.setBrush(qcol(C.HOVER)); p.drawRoundedRect(QRectF(0, 10, w, 5), 2.5, 2.5)
        v = sid.last
        if v and v.score is not None:
            p.setBrush(qcol(self.TONE.get(v.label, C.TEXT_DIM)))
            p.drawRoundedRect(QRectF(0, 10, w * max(0.0, min(1.0, v.score)), 5), 2.5, 2.5)
        thr = sid.summary["threshold"]
        p.setPen(QPen(qcol(C.TEXT_MED), 1.5)); p.drawLine(int(w * thr), 6, int(w * thr), 18)
        p.setFont(UIFont(9)); p.setPen(QPen(qcol(C.TEXT_DIM)))
        p.drawText(QRectF(0, 0, w, 9), Qt.AlignmentFlag.AlignLeft, "Unknown")
        p.drawText(QRectF(0, 0, w, 9), Qt.AlignmentFlag.AlignRight, "You")

    def _tick(self):
        if self.rec.isVisible() and self.ctl.audio is not None:
            lv = self.ctl.audio.levels()
            self.rec_meter.set_level(lv.get("mic_db", -120.0), lv.get("speech", False))

    # flows
    def _primary(self):
        if self.sid.status == "no_model":
            self._download()
        else:
            self._start("enrol")

    def _download(self):
        self._mode = "download"; self.refresh()
        self.detail_lbl.setText("Downloading the speaker model…")
        def work():
            try:
                self.sid.download_model(); self._job_done.emit("download", None)
            except Exception as e:
                self._job_done.emit("download", e)
        threading.Thread(target=work, daemon=True).start()

    def _start(self, mode: str):
        from audio.speaker_id import ENROL_PHRASES, ENROL_SAMPLES
        self._mode, self._samples, self._idx = mode, [], 0
        self._phrases = list(ENROL_PHRASES[:ENROL_SAMPLES]) if mode == "enrol" else \
            (list(ENROL_PHRASES[:2]) if mode == "add" else ["Say anything for a few seconds — for example, what you "
                                                             "plan to do today."])
        self.rec.show()
        self._show_step()
        self.refresh()

    def _show_step(self, msg: str = "", tone: str = ""):
        n = len(self._phrases)
        title = {"enrol": "Enrolment", "add": "Add samples", "test": "Test recognition"}[self._mode]
        self.rec_step.setText(f"{title} · {self._idx + 1} of {n}" if n > 1 else title)
        self.rec_phrase.setText(f"“{self._phrases[self._idx]}”" if self._mode != "test" else self._phrases[0])
        self.rec_msg.setText(msg or "Press Record, then read the phrase at a normal speaking volume.")
        self.rec_msg.setStyleSheet(f"color: {tone or C.TEXT_MED}; background: transparent;")
        self.rec_go.setEnabled(True); self.rec_go.setText("Record")

    def _record(self):
        secs = 5.0 if self._mode != "test" else 4.0
        if not self.ctl.record_audio(secs, lambda x: self._recorded.emit(x)):
            self.rec_msg.setText("The microphone isn't available. Check Audio settings.")
            return
        self.rec_go.setEnabled(False); self.rec_go.setText("Recording…")
        self.rec_msg.setText("Speak now.")

    def _on_recorded(self, audio):
        from audio.speaker_id import SpeakerIdentity
        x = np.asarray(audio, dtype=np.float32)
        if self._mode == "test":
            self.rec_msg.setText("Checking…")
            def work():
                self._job_done.emit("test", self.sid.verify(x))
            threading.Thread(target=work, daemon=True).start()
            return
        chk = SpeakerIdentity.check_sample(x)
        if not chk.ok:
            self._show_step(chk.reason, C.ACC2)
            return
        self._samples.append(x)
        self._idx += 1
        if self._idx < len(self._phrases):
            self._show_step("Got it.", C.GREEN)
            return
        self.rec_go.setEnabled(False); self.rec_msg.setText("Building your voice profile…")
        samples, append = list(self._samples), self._mode == "add"
        def work():
            try:
                self._job_done.emit("enrol", self.sid.enrol(samples, append=append))
            except Exception as e:
                self._job_done.emit("enrol", e)
        threading.Thread(target=work, daemon=True).start()

    def _on_job(self, kind: str, result):
        if kind == "download":
            self._mode = None
            if isinstance(result, Exception):
                self.ctl.toast("Download failed", str(result), "bad")
        elif kind == "enrol":
            self._mode = None; self.rec.hide()
            if isinstance(result, Exception):
                self.ctl.toast("Enrolment failed", str(result), "bad")
            else:
                self.ctl.toast("Voice profile saved",
                               f"{result['samples']} samples · threshold {result['threshold']:.2f}", "ok")
        elif kind == "test":
            self.sid.last = result
            self._mode = None; self.rec.hide()
            self.ctl.toast(f"Recognition: {result.text}",
                           "" if result.score is None else f"Score {result.score:.2f} (threshold {result.threshold:.2f})",
                           "ok" if result.label == "you" else "warn")
        self.refresh()

    def _cancel(self):
        self._mode = None; self.rec.hide(); self.refresh()

    def _retrain(self):
        self.sid.delete_profile()
        self._start("enrol")

    def _delete(self):
        if self.b_delete.text() != "Click again to delete":
            self.b_delete.setText("Click again to delete")
            QTimer.singleShot(4000, lambda: self.b_delete.setText("Delete profile"))
            return
        self.sid.delete_profile()
        self.b_delete.setText("Delete profile")
        self.ctl.toast("Voice profile deleted", "The file has been removed from this Mac.", "neutral")
        self.refresh()

    def stop(self):
        self._meter_t.stop()


# ── Remote ───────────────────────────────────────────────────────────────────

class RemotePane(Pane):
    def __init__(self, ctl: SettingsController):
        super().__init__("Remote", "Control the assistant from your phone on the same Wi-Fi. The server only runs "
                         "while you use it and turns itself off after 15 minutes without a phone.")
        self.ctl = ctl
        self.state = label("", 13.5, C.TEXT, QFont.Weight.Medium)
        self.detail = label("", 11.5, C.TEXT_MED, wrap=True)
        self.lay.addWidget(self.state); self.lay.addWidget(self.detail)
        row = QHBoxLayout(); row.setSpacing(8)
        self.pair = button("Show pairing code", "primary", 30); self.pair.clicked.connect(ctl.open_pairing)
        self.forget = button("Forget all phones", "secondary", 30); self.forget.clicked.connect(self._forget)
        self.off = button("Turn off now", "danger", 30); self.off.clicked.connect(self._off)
        for b in (self.pair, self.forget, self.off):
            row.addWidget(b)
        row.addStretch()
        w = QWidget(); w.setLayout(row); self.lay.addSpacing(10); self.lay.addWidget(w)
        self.lay.addSpacing(14)
        self.lay.addWidget(label("Pairing uses a one-time code (valid 10 minutes). The connection is encrypted with a "
                                 "certificate made on this Mac; your phone will ask you to accept it once.",
                                 11, C.TEXT_DIM, wrap=True))
        self.finish()
        self._t = QTimer(self); self._t.timeout.connect(self.refresh); self._t.start(1000)
        self.refresh()

    def refresh(self):
        st = self.ctl.remote_status()
        if not st.get("available", True):
            self.state.setText("Unavailable"); self.detail.setText(st.get("reason", ""))
            for b in (self.pair, self.forget, self.off):
                b.setEnabled(False)
            return
        if st.get("running"):
            n = st.get("phones", 0)
            self.state.setText("On" + (f" · {n} phone{'s' if n != 1 else ''} connected" if n else " · waiting for a phone"))
            self.detail.setText(f"Address: {st.get('url', '')}")
        else:
            self.state.setText("Off")
            self.detail.setText("Show a pairing code to turn it on.")
        self.forget.setEnabled(bool(st.get("running")))
        self.off.setEnabled(bool(st.get("running")))

    def _forget(self):
        n = self.ctl.forget_phones()
        self.ctl.toast("Phones forgotten", f"{n} remembered phone{'s' if n != 1 else ''} removed. "
                       "They'll need a new pairing code.", "neutral")
        self.refresh()

    def _off(self):
        self.ctl.stop_remote()
        self.refresh()

    def stop(self):
        self._t.stop()


# ── About ────────────────────────────────────────────────────────────────────

class AboutPane(Pane):
    def __init__(self, ctl: SettingsController):
        super().__init__("About")
        root = os.path.dirname(os.path.abspath(__file__))
        rows = [
            ("Version", "Mark XLIX"),
            ("System", f"macOS {platform.mac_ver()[0]}" if platform.system() == "Darwin" else platform.platform()),
            ("Reduce motion", "On (animations simplified)" if reduce_motion() else "Off"),
            ("Settings file", os.path.join(root, "config", "api_keys.json")),
            ("Voice profile", os.path.join(root, "memory", "voice_profile.npz")),
            ("Licence", "Creative Commons BY-NC 4.0 — personal, non-commercial use"),
        ]
        g = QGridLayout(); g.setHorizontalSpacing(18); g.setVerticalSpacing(8)
        for i, (k, v) in enumerate(rows):
            g.addWidget(label(k, 11.5, C.TEXT_MED), i, 0, Qt.AlignmentFlag.AlignTop)
            vl = label(v, 11.5, C.TEXT, wrap=True); vl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            g.addWidget(vl, i, 1)
        g.setColumnStretch(1, 1)
        w = QWidget(); w.setLayout(g); self.lay.addWidget(w)
        self.lay.addSpacing(14)
        self.lay.addWidget(label("Shortcuts: F4 mute · Esc interrupt · F11 full screen · ⌘, settings", 11, C.TEXT_DIM))
        self.finish()


# ── the sheet ────────────────────────────────────────────────────────────────

class _NavButton(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(UIFont(12.5))
        self.setFixedHeight(32)
        self.setStyleSheet(f"""
            QPushButton {{ text-align: left; padding: 0 12px; border: none; border-radius: 7px;
                           background: transparent; color: {C.TEXT_MED}; }}
            QPushButton:hover {{ background: {C.PANEL2}; color: {C.TEXT}; }}
            QPushButton:checked {{ background: {C.HOVER}; color: {C.TEXT}; }}
        """)


class SettingsSheet(QWidget):
    """Modal settings sheet centred over the main window."""
    closed = pyqtSignal()
    SECTIONS = ("General", "Audio", "Voice identity", "Remote", "About")
    _W, _H = 820, 600

    def __init__(self, ctl: SettingsController, section: str = "General", parent=None):
        super().__init__(parent)
        self.ctl = ctl
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("Settings")
        self.setStyleSheet(f"#Settings {{ background: {C.GLASS}; border: 1px solid {C.BORDER_B}; border-radius: 14px; }}")
        h = QHBoxLayout(self); h.setContentsMargins(0, 0, 0, 0); h.setSpacing(0)
        nav = QWidget(); nav.setObjectName("Nav"); nav.setFixedWidth(196)
        nav.setStyleSheet(f"#Nav {{ background: rgba(25,27,31,230); border-right: 1px solid {C.BORDER};"
                          f" border-top-left-radius: 14px; border-bottom-left-radius: 14px; }}")
        nl = QVBoxLayout(nav); nl.setContentsMargins(12, 18, 12, 14); nl.setSpacing(2)
        nl.addWidget(caps("Settings")); nl.addSpacing(8)
        self.nav: dict[str, _NavButton] = {}
        for s in self.SECTIONS:
            b = _NavButton(s); b.clicked.connect(lambda _=False, s=s: self.show_section(s))
            nl.addWidget(b); self.nav[s] = b
        nl.addStretch()
        done = button("Done", "secondary", 30); done.clicked.connect(self.close_sheet)
        nl.addWidget(done)
        h.addWidget(nav)
        self.stack = QStackedWidget()
        h.addWidget(self.stack, 1)
        self.panes: dict[str, QWidget] = {}
        self.show_section(section)

    def show_section(self, name: str):
        if name not in self.panes:
            cls = {"General": GeneralPane, "Audio": AudioPane, "Voice identity": VoicePane,
                   "Remote": RemotePane, "About": AboutPane}[name]
            p = cls(self.ctl)
            self.panes[name] = p
            self.stack.addWidget(p)
        p = self.panes[name]
        if hasattr(p, "refresh"):
            p.refresh()
        self.stack.setCurrentWidget(p)
        for k, b in self.nav.items():
            b.setChecked(k == name)

    def close_sheet(self):
        for p in self.panes.values():
            if hasattr(p, "stop"):
                p.stop()
        self.hide()
        self.closed.emit()
        self.deleteLater()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close_sheet(); return
        super().keyPressEvent(e)
