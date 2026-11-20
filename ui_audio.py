"""
ui_audio.py — Audio Settings panel (Graphite & Silver).

Opened from the ⚙ drawer. Every control applies to the running AudioEngine
immediately (no restart, Gemini session untouched) and is saved to config.
Nothing here runs audio work on the UI thread: device rescans and the
round-trip measurement run in short worker threads; meters are polled from
the engine's snapshot 4–30 times a second.
"""

from __future__ import annotations

import dataclasses
import threading

from PyQt6.QtCore import QEasingCurve, QPointF, QRectF, Qt, QTimer, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractButton, QButtonGroup, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from ui import C, UIFont, fade_in, qcol, reduce_motion


# ── small controls ────────────────────────────────────────────────────────────

class Toggle(QAbstractButton):
    """A pill switch with an eased knob."""

    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(38, 22)
        self._pos = 1.0 if checked else 0.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(0 if reduce_motion() else 140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(self._set_pos)
        self.toggled.connect(self._animate)

    def _set_pos(self, v):
        self._pos = float(v)
        self.update()

    def _animate(self, on: bool):
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def setChecked(self, on: bool):          # keep the knob in sync without animating
        super().setChecked(on)
        self._pos = 1.0 if on else 0.0
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        en = self.isEnabled()
        track_off, track_on = QColor(C.HOVER), QColor(C.PRI)
        t = self._pos
        tr = QColor(int(track_off.red() + (track_on.red() - track_off.red()) * t),
                    int(track_off.green() + (track_on.green() - track_off.green()) * t),
                    int(track_off.blue() + (track_on.blue() - track_off.blue()) * t))
        if not en:
            tr.setAlpha(90)
        p.setPen(QPen(qcol(C.BORDER_B if t < 0.5 else C.PRI), 1))
        p.setBrush(tr)
        p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 11, 11)
        x = 3 + t * (self.width() - 22)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcol(C.BG if t > 0.5 else C.TEXT_MED, 255 if en else 120))
        p.drawEllipse(QRectF(x, 3, 16, 16))


class LevelMeter(QWidget):
    """Horizontal dBFS meter with a peak hold and a speech dot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(14)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._v = 0.0
        self._peak = 0.0
        self._speech = False

    def set_level(self, db: float, speech: bool):
        v = max(0.0, min(1.0, (db + 60.0) / 60.0))
        self._v += (v - self._v) * (0.6 if v > self._v else 0.2)
        self._peak = max(self._v, self._peak - 0.01)
        self._speech = speech
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width() - 18, self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcol(C.INPUT))
        p.drawRoundedRect(QRectF(0, 3, w, h - 6), 4, 4)
        fill = qcol(C.RED if self._v > 0.92 else (C.ACC2 if self._v > 0.75 else C.PRI))
        p.setBrush(fill)
        p.drawRoundedRect(QRectF(0, 3, max(2.0, w * self._v), h - 6), 4, 4)
        p.setPen(QPen(qcol(C.TEXT), 1.5))
        px = w * self._peak
        p.drawLine(QPointF(px, 2), QPointF(px, h - 2))
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcol(C.GREEN if self._speech else C.HOVER))
        p.drawEllipse(QRectF(w + 6, h / 2 - 4, 8, 8))


def _label(text: str, size: float = 10, color: str | None = None, weight=None, mono=False, spacing=0.0):
    l = QLabel(text)
    l.setFont(UIFont(size, weight, mono=mono, spacing=spacing))
    l.setStyleSheet(f"color: {color or C.TEXT}; background: transparent;")
    return l


def _button(text: str, primary: bool = False) -> QPushButton:
    b = QPushButton(text)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setFont(UIFont(9, None, spacing=0.4))
    b.setFixedHeight(28)
    fg = C.BG if primary else C.TEXT
    bg = C.PRI if primary else C.PANEL2
    b.setStyleSheet(f"""
        QPushButton {{ background: {bg}; color: {fg}; border: 1px solid {C.BORDER_B if not primary else C.PRI};
                       border-radius: 6px; padding: 0 12px; }}
        QPushButton:hover {{ background: {C.WHITE if primary else C.HOVER}; border-color: {C.PRI}; }}
        QPushButton:disabled {{ color: {C.TEXT_DIM}; border-color: {C.BORDER}; }}
    """)
    return b


_COMBO_QSS = f"""
    QComboBox {{ background: {C.INPUT}; color: {C.TEXT}; border: 1px solid {C.BORDER};
                 border-radius: 6px; padding: 4px 10px; min-height: 18px; }}
    QComboBox:hover {{ border-color: {C.BORDER_B}; }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{ background: {C.PANEL2}; color: {C.TEXT};
                 selection-background-color: {C.HOVER}; border: 1px solid {C.BORDER}; }}
"""


# ── the panel ─────────────────────────────────────────────────────────────────

class AudioSettingsOverlay(QWidget):
    closed = pyqtSignal()
    _devices_ready = pyqtSignal(dict)
    _rtt_ready = pyqtSignal(object)
    _OW, _OH = 520, 640

    def __init__(self, engine, save_fn, parent=None):
        super().__init__(parent)
        self.engine = engine
        self._save = save_fn
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("AudioSettings")
        self.setStyleSheet(f"""
            #AudioSettings {{ background: {C.GLASS}; border: 1px solid {C.BORDER_B};
                              border-top: 1px solid {C.HOVER}; border-radius: 12px; }}
            QScrollArea, #AudioBody {{ background: transparent; border: none; }}
            QScrollBar:vertical {{ background: transparent; width: 6px; }}
            QScrollBar::handle:vertical {{ background: {C.BORDER_B}; border-radius: 3px; min-height: 24px; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
        """)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 12, 16)
        outer.setSpacing(10)

        head = QHBoxLayout()
        head.addWidget(_label("AUDIO", 11, C.TEXT, weight=_DEMI(), spacing=3))
        head.addStretch()
        x = QPushButton("✕")
        x.setFixedSize(26, 26)
        x.setCursor(Qt.CursorShape.PointingHandCursor)
        x.setStyleSheet(f"QPushButton {{ color: {C.TEXT_MED}; background: transparent; border: none; font-size: 14px; }}"
                        f"QPushButton:hover {{ color: {C.TEXT}; }}")
        x.clicked.connect(self.close_panel)
        head.addWidget(x)
        outer.addLayout(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget(); body.setObjectName("AudioBody")
        self._lay = QVBoxLayout(body)
        self._lay.setContentsMargins(0, 0, 8, 0)
        self._lay.setSpacing(6)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        self._scroll, self._body = scroll, body

        if engine is None:
            self._lay.addWidget(_label("The audio engine isn't running, so audio settings are unavailable.\n"
                                       "Check the activity log for the reason.", 10, C.TEXT_MED))
            self._lay.addStretch()
            return

        s = engine.settings
        # MICROPHONE
        self._section("Microphone")
        self.t_iso = self._toggle_row("Voice isolation", "Echo cancellation, noise suppression and gain on your mic.",
                                      s.voice_isolation, "voice_isolation")
        self.seg = self._segmented_row("Noise suppression", ("off", "light", "standard"),
                                       ("Off", "Light", "Standard"), s.noise_suppression)
        self.t_aec = self._toggle_row("Echo cancellation", "Removes JARVIS's own voice from your mic.",
                                      s.echo_cancellation, "echo_cancellation")
        self.t_agc = self._toggle_row("Automatic gain", "Evens out quiet and loud speech. Holds still while JARVIS talks.",
                                      s.auto_gain, "auto_gain")
        self.in_combo = self._device_row("Input device", "input_device")
        self.mic_meter = LevelMeter()
        mrow = QHBoxLayout(); mrow.setSpacing(10)
        mrow.addWidget(self.mic_meter, 1)
        self.mic_test_btn = _button("Test mic (5 s)")
        self.mic_test_btn.clicked.connect(self._mic_test)
        mrow.addWidget(self.mic_test_btn)
        self._lay.addWidget(_label("Live microphone (after processing)", 9, C.TEXT_MED))
        self._lay.addLayout(mrow)

        # SPEAKER
        self._section("Speaker")
        self.t_enh = self._toggle_row("Output enhancement", "Keeps JARVIS's voice even and stops clipping. "
                                      "Doesn't remove room noise — that's the mic side.",
                                      s.output_enhancement, "output_enhancement")
        self.t_norm = self._toggle_row("Volume normalisation", "Brings quiet and loud replies to the same level.",
                                       s.normalization, "normalization")
        self.t_duck = self._toggle_row("Lower other apps while JARVIS talks", engine.ducker.reason or
                                       "Other apps drop to 30 % and come back after.", s.ducking, "ducking")
        if not engine.ducker.available:
            self.t_duck.setEnabled(False)
        self.out_combo = self._device_row("Output device", "output_device")
        vrow = QHBoxLayout(); vrow.setSpacing(10)
        vrow.addWidget(_label("Volume", 10))
        self.vol = QSlider(Qt.Orientation.Horizontal)
        self.vol.setRange(0, 150)
        self.vol.setValue(int(round(s.output_volume * 100)))
        self.vol.setStyleSheet(f"""
            QSlider::groove:horizontal {{ height: 4px; background: {C.HOVER}; border-radius: 2px; }}
            QSlider::sub-page:horizontal {{ background: {C.PRI}; border-radius: 2px; }}
            QSlider::handle:horizontal {{ background: {C.TEXT}; width: 14px; margin: -6px 0; border-radius: 7px; }}
        """)
        self.vol_lbl = _label(f"{self.vol.value()} %", 9, C.TEXT_MED, mono=True)
        self.vol.valueChanged.connect(self._volume_changed)
        vrow.addWidget(self.vol, 1); vrow.addWidget(self.vol_lbl)
        self._lay.addLayout(vrow)
        tone = _button("Play test sound")
        tone.clicked.connect(engine.play_test_tone)
        self._lay.addWidget(tone, 0, Qt.AlignmentFlag.AlignLeft)

        # GENERAL
        self._section("General")
        self.t_master = self._toggle_row("Audio processing", "Off = bypass: raw mic and speaker, exactly as before.",
                                         s.processing_enabled, "processing_enabled")
        self.t_duplex = self._toggle_row("Talk over JARVIS (experimental)",
                                         "Keeps the mic open while JARVIS speaks so you can interrupt by voice. "
                                         "Needs echo cancellation; on laptop speakers your voice is turned down "
                                         "while JARVIS talks, so speak clearly.", s.full_duplex, "full_duplex")
        grid = QGridLayout(); grid.setHorizontalSpacing(14); grid.setVerticalSpacing(4)
        self._stat = {}
        rows = [("round_trip_ms", "Speaker → mic round trip"), ("processing_delay_ms", "Processing delay"),
                ("output_latency_ms", "Speaker buffer"), ("dsp_ms_per_10ms", "Processing load"),
                ("playback_buffer_ms", "Queued speech"), ("reply_latency_ms", "Reply time (median)"),
                ("health", "Dropouts")]
        for i, (k, name) in enumerate(rows):
            grid.addWidget(_label(name, 9, C.TEXT_MED), i, 0)
            v = _label("—", 9, C.TEXT, mono=True)
            grid.addWidget(v, i, 1)
            self._stat[k] = v
        self.rtt_btn = _button("Measure")
        self.rtt_btn.setFixedHeight(22)
        self.rtt_btn.clicked.connect(self._measure_rtt)
        grid.addWidget(self.rtt_btn, 0, 2)
        grid.setColumnStretch(1, 1)
        self._lay.addLayout(grid)
        self.status = _label("", 9, C.TEXT_MED)
        self.status.setWordWrap(True)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self._lay.addWidget(self.status)
        self._lay.addStretch()

        self._devices_ready.connect(self._fill_devices)
        self._rtt_ready.connect(self._rtt_done)
        self._load_devices(rescan=False)
        self._sync_enabled()

        self._fast = QTimer(self); self._fast.timeout.connect(self._tick_meter); self._fast.start(33)
        self._slow = QTimer(self); self._slow.timeout.connect(self._tick_stats); self._slow.start(250)
        self._tick_stats()

    # ── layout helpers ────────────────────────────────────────────────────

    def _section(self, title: str):
        self._lay.addSpacing(10)
        self._lay.addWidget(_label(title.upper(), 9, C.TEXT_MED, weight=_DEMI(), spacing=2))
        line = QFrame(); line.setFixedHeight(1)
        line.setStyleSheet(f"background: {C.BORDER}; border: none;")
        self._lay.addWidget(line)
        self._lay.addSpacing(2)

    def _toggle_row(self, title, hint, value, key) -> Toggle:
        row = QHBoxLayout(); row.setSpacing(12)
        col = QVBoxLayout(); col.setSpacing(1)
        col.addWidget(_label(title, 10))
        h = _label(hint, 8.5, C.TEXT_DIM); h.setWordWrap(True)
        h.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        col.addWidget(h)
        row.addLayout(col, 1)
        t = Toggle(bool(value))
        t.toggled.connect(lambda on, k=key: self._set(**{k: on}))
        row.addWidget(t, 0, Qt.AlignmentFlag.AlignTop)
        self._lay.addLayout(row)
        return t

    def _segmented_row(self, title, keys, labels, current) -> QButtonGroup:
        row = QHBoxLayout(); row.setSpacing(12)
        row.addWidget(_label(title, 10), 1)
        box = QHBoxLayout(); box.setSpacing(0)
        grp = QButtonGroup(self); grp.setExclusive(True)
        for i, (k, l) in enumerate(zip(keys, labels)):
            b = QPushButton(l); b.setCheckable(True); b.setChecked(k == current)
            b.setCursor(Qt.CursorShape.PointingHandCursor); b.setFont(UIFont(9))
            b.setFixedHeight(26)
            left = "6px" if i == 0 else "0"; right = "6px" if i == len(keys) - 1 else "0"
            b.setStyleSheet(f"""
                QPushButton {{ background: {C.INPUT}; color: {C.TEXT_MED}; border: 1px solid {C.BORDER};
                    border-top-left-radius: {left}; border-bottom-left-radius: {left};
                    border-top-right-radius: {right}; border-bottom-right-radius: {right}; padding: 0 12px; }}
                QPushButton:hover {{ color: {C.TEXT}; }}
                QPushButton:checked {{ background: {C.PRI}; color: {C.BG}; border-color: {C.PRI}; }}
            """)
            b.toggled.connect(lambda on, k=k: on and self._set(noise_suppression=k))
            grp.addButton(b); box.addWidget(b)
        row.addLayout(box)
        self._lay.addLayout(row)
        return grp

    def _device_row(self, title, key) -> QComboBox:
        row = QHBoxLayout(); row.setSpacing(8)
        row.addWidget(_label(title, 10))
        cb = QComboBox(); cb.setFont(UIFont(9)); cb.setStyleSheet(_COMBO_QSS)
        cb.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        cb.setMinimumContentsLength(12)
        cb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        cb.activated.connect(lambda _i, k=key, c=cb: self._set(**{k: c.currentData()}))
        row.addWidget(cb, 1)
        rb = QPushButton("↻"); rb.setFixedSize(28, 28); rb.setToolTip("Rescan devices")
        rb.setCursor(Qt.CursorShape.PointingHandCursor)
        rb.setStyleSheet(f"QPushButton {{ background: {C.PANEL2}; color: {C.TEXT_MED}; border: 1px solid {C.BORDER};"
                         f" border-radius: 6px; }} QPushButton:hover {{ color: {C.TEXT}; border-color: {C.BORDER_B}; }}")
        rb.clicked.connect(lambda: self._load_devices(rescan=True))
        row.addWidget(rb)
        self._lay.addLayout(row)
        return cb

    # ── behaviour ─────────────────────────────────────────────────────────

    def _set(self, **changes):
        s = dataclasses.replace(self.engine.settings, **changes)
        self.engine.apply_settings(s)
        try:
            self._save(s.to_dict())
        except Exception as e:
            self.status.setText(f"Couldn't save settings: {e}")
        self._sync_enabled()

    def _sync_enabled(self):
        s = self.engine.settings if self.engine._pending_settings is None else self.engine._pending_settings
        on = s.processing_enabled
        for w in (self.t_iso, self.t_enh, self.t_duplex):
            w.setEnabled(on)
        iso = on and s.voice_isolation
        for w in (self.t_aec, self.t_agc):
            w.setEnabled(iso)
        for b in self.seg.buttons():
            b.setEnabled(iso)
        self.t_norm.setEnabled(on and s.output_enhancement)
        self.t_duck.setEnabled(on and self.engine.ducker.available)
        self.t_duplex.setEnabled(on and iso and s.echo_cancellation)

    def _volume_changed(self, v: int):
        self.vol_lbl.setText(f"{v} %")
        self._set(output_volume=v / 100.0)

    def _load_devices(self, rescan: bool):
        def work():
            from audio.engine import list_devices
            if rescan:
                self.engine.refresh_devices()
            try:
                self._devices_ready.emit(list_devices())
            except Exception as e:
                self._devices_ready.emit({"error": str(e)})
        threading.Thread(target=work, daemon=True).start()

    def _fill_devices(self, d: dict):
        if "error" in d:
            self.status.setText(f"Couldn't list audio devices: {d['error']}")
            return
        for cb, kind, cur in ((self.in_combo, "input", self.engine.settings.input_device),
                              (self.out_combo, "output", self.engine.settings.output_device)):
            cb.blockSignals(True); cb.clear()
            default = d.get(f"default_{kind}")
            cb.addItem(f"System default{f'  ({default})' if default else ''}", None)
            for name in d.get(kind, []):
                cb.addItem(name, name)
            if cur and cur not in d.get(kind, []):
                cb.addItem(f"{cur}  (not connected)", cur)
            idx = cb.findData(cur)
            cb.setCurrentIndex(idx if idx >= 0 else 0)
            cb.blockSignals(False)

    def _mic_test(self):
        if self.engine.mic_test_running:
            return
        self.mic_test_btn.setEnabled(False)
        self.mic_test_btn.setText("Recording… speak now")
        self.engine.start_mic_test(5.0)

    def _measure_rtt(self):
        self.rtt_btn.setEnabled(False)
        self._stat["round_trip_ms"].setText("measuring…")
        threading.Thread(target=lambda: self._rtt_ready.emit(self.engine.measure_round_trip()), daemon=True).start()

    def _rtt_done(self, v):
        self.rtt_btn.setEnabled(True)
        if v is None:
            self.status.setText("Round trip: the mic didn't hear the click. Turn the volume up or check the devices.")
        self._tick_stats()

    def _tick_meter(self):
        lv = self.engine.levels()
        self.mic_meter.set_level(lv.get("mic_db", -120.0), lv.get("speech", False))
        if not self.mic_test_btn.isEnabled() and not self.engine.mic_test_running:
            if self.engine.is_playing:
                self.mic_test_btn.setText("Playing back…")
            else:
                self.mic_test_btn.setText("Test mic (5 s)")
                self.mic_test_btn.setEnabled(True)

    def _tick_stats(self):
        st = self.engine.stats()

        def ms(v, digits=0):
            return "—" if v is None else f"{v:.{digits}f} ms"
        self._stat["round_trip_ms"].setText(ms(st["round_trip_ms"]) if self.rtt_btn.isEnabled() else "measuring…")
        self._stat["processing_delay_ms"].setText(ms(st["processing_delay_ms"]))
        self._stat["output_latency_ms"].setText(ms(st["output_latency_ms"]))
        self._stat["dsp_ms_per_10ms"].setText(f"{st['dsp_cpu_pct']:.1f} % of one core")
        self._stat["playback_buffer_ms"].setText(ms(st["playback_buffer_ms"]))
        self._stat["reply_latency_ms"].setText(ms(st["reply_latency_ms"]))
        self._stat["health"].setText(f"{st['dropped_mic_blocks']} mic · {st['underruns']} speaker")
        problems = [x for x in (st["input_error"], st["output_error"]) if x]
        if st["processing_error"]:
            problems.insert(0, f"Processing bypassed — {st['processing_error']}")
        if problems:
            self.status.setStyleSheet(f"color: {C.RED}; background: transparent;")
            self.status.setText("  ·  ".join(problems))
        else:
            self.status.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent;")
            s = self.engine.settings
            self.status.setText("Processing active." if (s.processing_enabled and st["processing_active"])
                                else "Bypass: raw mic and speaker.")

    def close_panel(self):
        for t in ("_fast", "_slow"):
            if hasattr(self, t):
                getattr(self, t).stop()
        self.hide()
        self.closed.emit()
        self.deleteLater()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        # Keep the content exactly as wide as the visible area: long wrapped
        # hints must wrap, never push controls off the right edge.
        if hasattr(self, "_scroll"):
            self._body.setFixedWidth(self._scroll.viewport().width())

    def showEvent(self, e):
        super().showEvent(e)
        if hasattr(self, "_scroll"):
            QTimer.singleShot(0, lambda: self._body.setFixedWidth(self._scroll.viewport().width()))

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            self.close_panel()
            return
        super().keyPressEvent(e)


def _DEMI():
    from PyQt6.QtGui import QFont
    return QFont.Weight.DemiBold
