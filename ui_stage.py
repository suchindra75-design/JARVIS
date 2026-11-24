"""
ui_stage.py — the central "stage": what Raya is doing, in words and one voice ribbon.

States (all driven by real app state; nothing here is decorative-only):
  ready       connected, mic open, nobody talking        faint ribbon drifting slowly
  listening   your speech detected on the mic (VAD)       ribbon follows your mic level
  processing  Gemini is working / a tool is running       flat line, amber light travelling
  speaking    Raya audio is playing                     ribbon follows the output spectrum
  muted       microphone muted                            dim flat line
  connecting  connecting / reconnecting to Gemini         slow pulse
  error       something needs attention                   muted-red dashed line + one action

The ribbon is one QPainterPath per frame. It repaints at 60 fps only while it
is moving with sound and drops to 15 fps otherwise; under the OS
"reduce motion" setting it shows levels without drift or travelling light.
"""

from __future__ import annotations

import math
import time

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient
from PyQt6.QtWidgets import QAbstractButton, QHBoxLayout, QSizePolicy, QVBoxLayout, QWidget

from ui_theme import C, UIFont, button, label, mix, qcol, reduce_motion

STATE_TEXT = {
    "ready":      ("Ready", "Speak or type a message."),
    "listening":  ("Listening", "Go ahead — I'm listening."),
    "processing": ("Thinking", "Working on it…"),
    "speaking":   ("Speaking", ""),
    "muted":      ("Muted", "Microphone muted. Press F4 or the mic button to unmute."),
    "connecting": ("Connecting", "Connecting to Gemini…"),
    "error":      ("Something's wrong", ""),
}


def _state_colour(state: str) -> str:
    return {"ready": C.TEXT_DIM, "listening": C.PRI, "processing": C.ACC2, "speaking": C.TEXT,
            "muted": C.TEXT_DIM, "connecting": C.TEXT_MED, "error": C.RED}.get(state, C.TEXT_MED)


class Ribbon(QWidget):
    """The voice ribbon. Reads levels from the AudioEngine snapshot each frame."""

    N = 120

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.audio = None
        self.state = "connecting"
        self._t0 = time.monotonic()
        self._lvl = 0.0                       # smoothed 0..1 amplitude
        self._bands = [0.0] * 24
        self._col = QColor(_state_colour(self.state))
        self._sweep = 0.0
        self.speech_seen_at = 0.0             # last time VAD heard speech (used by the stage)
        self.background = None                # callable(QRect in our coords) -> QPixmap, set by the stage
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._tmr = QTimer(self)
        self._tmr.timeout.connect(self._step)
        self._tmr.start(66)

    def sizeHint(self):
        return QSize(560, 180)

    def _levels(self) -> tuple[float, list[float]]:
        lv = self.audio.levels() if self.audio is not None else None
        if not lv:
            return 0.0, [0.0] * 24
        if lv.get("speech"):
            self.speech_seen_at = time.monotonic()
        if self.state == "speaking":
            db, spec = lv.get("out_db", -120.0), lv.get("spectrum") or [0.0] * 24
            return max(0.0, min(1.0, (db + 50.0) / 40.0)), list(spec)
        if self.state in ("listening", "ready"):
            db = lv.get("mic_db", -120.0)
            spec = lv.get("mic_spectrum") or []
            return max(0.0, min(1.0, (db + 58.0) / 40.0)), list(spec) or [0.0] * 24
        return 0.0, [0.0] * 24

    def _step(self):
        target, bands = self._levels()
        moving = self.state in ("listening", "speaking", "processing")
        self._lvl += (target - self._lvl) * (0.45 if target > self._lvl else 0.12)
        self._bands = [o + (n - o) * (0.5 if n > o else 0.18) for o, n in zip(self._bands, bands)]
        k = 1.0 if reduce_motion() else 0.14
        self._col = mix(self._col, _state_colour(self.state), k)
        want = 33 if (moving and not reduce_motion()) else 66      # 30 fps moving, 15 fps otherwise
        if self._tmr.interval() != want:
            self._tmr.setInterval(want)
        self.update()

    def _band_at(self, u: float) -> float:
        if not any(self._bands):
            return 1.0
        i = u * (len(self._bands) - 1)
        a, f = int(i), i - int(i)
        b = min(a + 1, len(self._bands) - 1)
        return 0.35 + 0.65 * (self._bands[a] * (1 - f) + self._bands[b] * f)

    def paintEvent(self, _):
        p = QPainter(self)
        if self.background is not None:
            p.drawPixmap(0, 0, self.background(self.geometry()))
        else:
            p.fillRect(self.rect(), qcol(C.BG))
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        W, H = self.width(), self.height()
        x0, x1 = W * 0.04, W * 0.96
        cy = H / 2
        t = 0.0 if reduce_motion() else time.monotonic() - self._t0
        st = self.state
        col = self._col

        if st == "processing":
            p.setPen(QPen(qcol(C.BORDER), 1.4))
            p.drawLine(QPointF(x0, cy), QPointF(x1, cy))
            span = (x1 - x0)
            pos = 0.5 if reduce_motion() else (t / 1.8) % 1.0
            cx = x0 + span * (-0.15 + pos * 1.3)
            g = QLinearGradient(cx - span * 0.18, 0, cx + span * 0.18, 0)
            g.setColorAt(0.0, qcol(C.ACC2, 0)); g.setColorAt(0.5, qcol(C.ACC2, 235)); g.setColorAt(1.0, qcol(C.ACC2, 0))
            p.setPen(QPen(g, 2.4, cap=Qt.PenCapStyle.RoundCap))
            p.drawLine(QPointF(max(x0, cx - span * 0.18), cy), QPointF(min(x1, cx + span * 0.18), cy))
            return
        if st == "error":
            pen = QPen(qcol(C.RED, 220), 1.6, Qt.PenStyle.CustomDashLine)
            pen.setDashPattern([2, 4])
            p.setPen(pen)
            p.drawLine(QPointF(x0, cy), QPointF(x1, cy))
            return

        amp_px = {"listening": H * 0.42, "speaking": H * 0.44}.get(st, H * 0.05)
        level = self._lvl if st in ("listening", "speaking") else 0.0
        base = {"ready": 0.55, "connecting": 0.35 + 0.35 * math.sin(t * 1.2) ** 2,
                "muted": 0.0}.get(st, 0.08)
        A = amp_px * max(level, base * 0.25 if st in ("listening", "speaking") else base)

        def path(phase: float, scale: float) -> QPainterPath:
            pp = QPainterPath()
            for i in range(self.N + 1):
                u = i / self.N
                env = math.sin(math.pi * u) ** 1.6
                wave = (math.sin(u * 9.0 + phase) * 0.55 + math.sin(u * 17.0 - phase * 1.3) * 0.30
                        + math.sin(u * 29.0 + phase * 0.7) * 0.15)
                y = cy - A * scale * env * wave * self._band_at(u)
                x = x0 + (x1 - x0) * u
                (pp.moveTo if i == 0 else pp.lineTo)(x, y)
            return pp

        speed = {"listening": 2.6, "speaking": 3.2}.get(st, 0.35)
        ph = t * speed
        # depth line (dim) then main line with faded ends
        p.setPen(QPen(qcol(C.TEXT_MED, 70 if st != "muted" else 30), 1.1))
        p.drawPath(path(ph + 1.9, 0.6))
        g = QLinearGradient(x0, 0, x1, 0)
        a = 230 if st in ("listening", "speaking") else (150 if st == "ready" else 90)
        if st == "muted":
            a = 70
        g.setColorAt(0.0, qcol(col.name(), 0)); g.setColorAt(0.18, qcol(col.name(), a))
        g.setColorAt(0.82, qcol(col.name(), a)); g.setColorAt(1.0, qcol(col.name(), 0))
        p.setPen(QPen(g, 2.0 if st in ("listening", "speaking") else 1.5, cap=Qt.PenCapStyle.RoundCap))
        p.drawPath(path(ph, 1.0))


class MicButton(QAbstractButton):
    """Round microphone button. Checked = muted."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(56, 56)
        self.setToolTip("Mute / unmute microphone (F4)")
        self._hover = False
        self.level = 0.0

    def enterEvent(self, e):
        self._hover = True; self.update()

    def leaveEvent(self, e):
        self._hover = False; self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        muted = self.isChecked()
        r = QRectF(2, 2, self.width() - 4, self.height() - 4)
        g = QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0, qcol(C.HOVER if self._hover else "#26292E")); g.setColorAt(1, qcol(C.PANEL))
        p.setBrush(g)
        ring = C.RED if muted else (C.PRI if self.level > 0.05 else C.BORDER_B)
        p.setPen(QPen(qcol(ring, 220), 1.4 + (0 if muted else 1.6 * self.level)))
        p.drawEllipse(r)
        cx, cy = self.width() / 2, self.height() / 2
        p.setPen(QPen(qcol(C.RED if muted else C.TEXT), 1.7, cap=Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(cx - 4.5, cy - 11, 9, 14), 4.5, 4.5)
        arc = QRectF(cx - 8, cy - 6, 16, 13)
        p.drawArc(arc, 200 * 16, 140 * 16)
        p.drawLine(QPointF(cx, cy + 7), QPointF(cx, cy + 11))
        if muted:
            p.drawLine(QPointF(cx - 11, cy + 11), QPointF(cx + 11, cy - 11))


class VoiceStage(QWidget):
    """Title + ribbon + caption + control dock. Public API mirrors app state."""
    mute_toggled = pyqtSignal(bool)
    interrupt_requested = pyqtSignal()
    error_action = pyqtSignal()

    def __init__(self, assistant_name: str = "Raya", parent=None):
        super().__init__(parent)
        self.setObjectName("VoiceStage")
        self._bg: QPixmap | None = None        # gradient rendered once per size, reused every frame
        self._name = assistant_name
        self._backend_state = "INITIALISING"
        self._muted = False
        self._activity = ""
        self._error: tuple[str, str, str | None] | None = None
        self._speaking_caption = ""
        self._hearing_caption = ""      # live transcript of the user
        self._hearing_at = 0.0
        self._gate_hint = ""            # e.g. "Ignored — not your voice"
        self._gate_hint_at = 0.0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 20, 28, 18)
        lay.setSpacing(10)
        lay.addStretch(3)
        self.title = label("", 24, C.TEXT, QFont.Weight.Light)
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.title)
        self.ribbon = Ribbon()
        self.ribbon.background = self._bg_region
        lay.addWidget(self.ribbon, 0)
        self.caption = label("", 12.5, C.TEXT_MED, wrap=True)
        self.caption.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.caption.setMaximumHeight(60)
        lay.addWidget(self.caption)
        row = QHBoxLayout(); row.addStretch()
        self.error_btn = button("Open settings", "secondary", 28)
        self.error_btn.clicked.connect(self.error_action.emit)
        self.error_btn.hide()
        row.addWidget(self.error_btn); row.addStretch()
        lay.addLayout(row)
        lay.addStretch(4)

        dock = QHBoxLayout(); dock.setSpacing(14); dock.addStretch()
        self.interrupt_btn = button("Interrupt   esc", "ghost", 32)
        self.interrupt_btn.setToolTip("Stop Raya speaking (Esc)")
        self.interrupt_btn.clicked.connect(self.interrupt_requested.emit)
        self.mic = MicButton()
        self.mic.toggled.connect(self._on_mic)
        self._spacer = QWidget(); self._spacer.setFixedSize(self.interrupt_btn.sizeHint())
        self._spacer.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._spacer.setStyleSheet("background: transparent;")
        dock.addWidget(self.interrupt_btn); dock.addWidget(self.mic); dock.addWidget(self._spacer)
        dock.addStretch()
        lay.addLayout(dock)
        self.hint = label("F4 mutes  ·  Esc interrupts  ·  ⌘, settings", 10.5, C.TEXT_DIM)
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.hint)

        self._tick = QTimer(self)                  # re-evaluate ready vs listening
        self._tick.timeout.connect(self._refresh)
        self._tick.start(200)
        self._refresh()

    # ── public API (called on the GUI thread) ─────────────────────────────

    def attach_audio(self, engine) -> None:
        self.ribbon.audio = engine

    def set_backend_state(self, state: str) -> None:
        """LISTENING | THINKING | PROCESSING | SPEAKING | SLEEPING | INITIALISING"""
        self._backend_state = state
        if state in ("LISTENING", "SPEAKING", "THINKING", "PROCESSING"):
            self.clear_error(only_connection=True)
        if state != "SPEAKING":
            self._speaking_caption = ""
        else:
            self._hearing_caption = ""
        if state == "LISTENING":
            self._activity = ""
        self._refresh()

    def set_muted(self, muted: bool) -> None:
        self._muted = muted
        self.mic.blockSignals(True); self.mic.setChecked(muted); self.mic.blockSignals(False)
        self.mic.update()
        self._refresh()

    def set_activity(self, text: str) -> None:
        """What Raya is doing right now (e.g. 'Searching the web for …')."""
        self._activity = text
        self._refresh()

    def set_speaking_text(self, text: str) -> None:
        self._speaking_caption = text
        self._refresh()

    def set_gate_hint(self, text: str) -> None:
        """Brief, quiet note from the voice filter (shown for ~2.5 s while idle)."""
        self._gate_hint = text
        self._gate_hint_at = time.monotonic()
        self._refresh()
        QTimer.singleShot(2600, self._refresh)

    def set_hearing_text(self, text: str) -> None:
        """Live transcript of what the user is saying (shown until Raya answers)."""
        self._hearing_caption = text
        self._hearing_at = time.monotonic()
        self._refresh()

    def set_error(self, title: str, detail: str = "", action: str | None = "Open settings",
                  connection: bool = False) -> None:
        self._error = (title, detail, action, connection)
        self._refresh()

    def clear_error(self, only_connection: bool = False) -> None:
        if self._error and (not only_connection or self._error[3]):
            self._error = None
            self._refresh()

    def set_name(self, name: str) -> None:
        self._name = name
        self._refresh()

    @property
    def visual_state(self) -> str:
        return self.ribbon.state

    # ── background (rendered once per size) ──────────────────────────────

    def _background(self) -> QPixmap:
        if self._bg is None or self._bg.size() != self.size():
            pm = QPixmap(self.size())
            pm.fill(qcol(C.BG))
            p = QPainter(pm)
            g = QRadialGradient(QPointF(self.width() * 0.5, self.height() * 0.42), max(self.width(), self.height()) * 0.6)
            g.setColorAt(0.0, qcol(C.PANEL)); g.setColorAt(1.0, qcol(C.BG))
            p.fillRect(pm.rect(), g)
            p.end()
            self._bg = pm
        return self._bg

    def _bg_region(self, rect):
        return self._background().copy(rect)

    def paintEvent(self, _):
        p = QPainter(self)
        p.drawPixmap(0, 0, self._background())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._bg = None

    # ── internals ────────────────────────────────────────────────────────

    def _on_mic(self, muted: bool):
        self._muted = muted
        self.mute_toggled.emit(muted)
        self._refresh()

    def _derive(self) -> str:
        if self._error:
            return "error"
        if self._muted:
            return "muted"
        b = self._backend_state
        if b == "SPEAKING":
            return "speaking"
        if b in ("THINKING", "PROCESSING"):
            return "processing"
        if b in ("SLEEPING", "INITIALISING"):
            return "connecting"
        heard = time.monotonic() - self.ribbon.speech_seen_at < 1.2
        return "listening" if heard else "ready"

    def _refresh(self):
        st = self._derive()
        self.ribbon.state = st
        title, cap = STATE_TEXT[st]
        if st == "ready":
            title = self._name
        if st == "error":
            title, cap, action, _ = self._error
            self.error_btn.setText(action or "")
            self.error_btn.setVisible(bool(action))
        else:
            self.error_btn.hide()
        if st == "processing" and self._activity:
            cap = self._activity
        if st == "speaking" and self._speaking_caption:
            cap = self._speaking_caption
        if (self._gate_hint and st in ("listening", "ready")
                and time.monotonic() - self._gate_hint_at < 2.5):
            cap = self._gate_hint
        if (self._hearing_caption and st in ("listening", "ready", "processing")
                and not (st == "processing" and self._activity)
                and time.monotonic() - self._hearing_at < 6.0):
            cap = f"“{self._hearing_caption}”"
        if self.title.text() != title:
            self.title.setText(title)
        if self.caption.text() != cap:
            self.caption.setText(cap)
        self.title.setStyleSheet(f"color: {C.RED if st == 'error' else C.TEXT}; background: transparent;")
        busy = st in ("speaking", "processing")
        self.interrupt_btn.setVisible(busy)
        self._spacer.setVisible(busy)
        lv = self.ribbon.audio.levels() if self.ribbon.audio is not None else {}
        self.mic.level = 0.0 if self._muted else max(0.0, min(1.0, (lv.get("mic_db", -120) + 55) / 40))
        self.mic.update()
