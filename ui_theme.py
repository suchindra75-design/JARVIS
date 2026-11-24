"""
ui_theme.py — Graphite & Silver design system for Raya.

One place for tokens, type roles, motion helpers and the small components every
screen shares (chip, buttons, toggle, segmented control, level meter, toast).
ui.py re-exports C / UIFont / fade_in / reduce_motion / qcol from here so
existing code keeps working.

Type scale (points): display 24 light · title 15 · body 12.5 · secondary 11.5 ·
caption 10.5 · label 9.5 caps. Numbers use the mono role with tabular figures.
Motion: hover 120 ms, sheets/toasts 180 ms, state cross-fades 250 ms; all
skipped under the OS "reduce motion" setting.
"""

from __future__ import annotations

import platform
import time

from PyQt6.QtCore import (QEasingCurve, QPointF, QPropertyAnimation, QRectF, Qt, QTimer,
                          QVariantAnimation, pyqtSignal)
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (QAbstractButton, QButtonGroup, QGraphicsOpacityEffect, QHBoxLayout,
                             QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget)

_OS = platform.system()


class C:
    """Graphite & Silver tokens. Names are kept from the original theme."""
    # surfaces
    BG        = "#101113"     # window background
    DARK      = "#15171A"     # status bar, deep panels
    PANEL     = "#191B1F"     # secondary background (sidebar)
    PANEL2    = "#22252A"     # elevated surfaces
    BAR_BG    = "#22252A"
    INPUT     = "#141518"     # text fields
    HOVER     = "#2A2D33"
    GLASS     = "rgba(34, 37, 42, 240)"   # sheets: elevated surface at 94 %
    SCRIM     = "rgba(8, 9, 10, 150)"     # dims what's behind a modal sheet
    # lines
    BORDER    = "#30343A"
    BORDER_A  = "#34383F"
    BORDER_B  = "#454A52"
    # accent (silver)
    PRI       = "#C2C7CE"
    PRI_DIM   = "#7E838B"
    PRI_GHO   = "#1E2024"
    # text
    WHITE     = "#F2F3F4"
    TEXT      = "#E6E7E9"
    TEXT_MED  = "#92969D"
    TEXT_DIM  = "#6B6F76"
    # status (quiet, no neon)
    ACC       = "#D9DCE0"     # speaking
    ACC2      = "#C9A35A"     # processing / warning / approval
    GREEN     = "#7FB08F"     # success / connected / recognised
    GREEN_D   = "#4F6B58"
    RED       = "#C2675F"     # error / danger / muted
    MUTED_C   = "#C2675F"
    OK_BG     = "#18211C"
    WARN_BG   = "#211D15"
    BAD_BG    = "#231716"
    BAD_BG2   = "#2E1C1A"
    WARN_LINE = "#5B4A2A"
    BAD_LINE  = "#5A3532"


_UI_FAMILIES   = [".AppleSystemUIFont", "Helvetica Neue", "Segoe UI Variable Text", "Segoe UI",
                  "Inter", "Cantarell", "Arial"]
_MONO_FAMILIES = ["SF Mono", "Menlo", "Cascadia Mono", "Consolas", "DejaVu Sans Mono", "Courier New"]


def UIFont(size: float, weight=None, mono: bool = False, spacing: float = 0.0) -> QFont:
    """Font roles: UI text (system sans) or data (monospace). Sizes in points."""
    f = QFont()
    f.setFamilies(_MONO_FAMILIES if mono else _UI_FAMILIES)
    f.setPointSizeF(float(size))
    if weight is not None:
        if weight == QFont.Weight.Bold and not mono:
            weight = QFont.Weight.DemiBold          # elegant, not heavy
        f.setWeight(weight)
    if spacing:
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    if mono:
        f.setStyleHint(QFont.StyleHint.Monospace)
    return f


def qcol(h: str, a: int = 255) -> QColor:
    c = QColor(h); c.setAlpha(a); return c


def mix(a: str | QColor, b: str | QColor, t: float) -> QColor:
    a, b = QColor(a), QColor(b)
    return QColor(int(a.red() + (b.red() - a.red()) * t), int(a.green() + (b.green() - a.green()) * t),
                  int(a.blue() + (b.blue() - a.blue()) * t), int(a.alpha() + (b.alpha() - a.alpha()) * t))


_REDUCE_MOTION: bool | None = None


def reduce_motion() -> bool:
    """Honour the OS 'reduce motion' setting (macOS); False elsewhere or on error."""
    global _REDUCE_MOTION
    if _REDUCE_MOTION is None:
        _REDUCE_MOTION = False
        if _OS == "Darwin":
            try:
                from AppKit import NSWorkspace
                _REDUCE_MOTION = bool(NSWorkspace.sharedWorkspace().accessibilityDisplayShouldReduceMotion())
            except Exception:
                pass
    return _REDUCE_MOTION


def fade_in(w: QWidget, ms: int = 180) -> None:
    """Show a widget with a short opacity fade (skipped under reduce-motion)."""
    w.show(); w.raise_()
    if reduce_motion():
        return
    eff = QGraphicsOpacityEffect(w)
    w.setGraphicsEffect(eff)
    anim = QPropertyAnimation(eff, b"opacity", w)
    anim.setDuration(ms)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.finished.connect(lambda: w.setGraphicsEffect(None))
    anim.start()
    w._fade_anim = anim


def label(text: str = "", size: float = 12.5, color: str | None = None, weight=None,
          mono: bool = False, spacing: float = 0.0, wrap: bool = False) -> QLabel:
    l = QLabel(text)
    l.setFont(UIFont(size, weight, mono=mono, spacing=spacing))
    l.setStyleSheet(f"color: {color or C.TEXT}; background: transparent; border: none;")
    if wrap:
        l.setWordWrap(True)
        l.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    return l


def caps(text: str, color: str | None = None) -> QLabel:
    """Small uppercase section label."""
    return label(text.upper(), 9.5, color or C.TEXT_MED, QFont.Weight.DemiBold, spacing=1.6)


# ── buttons ──────────────────────────────────────────────────────────────────

def button(text: str, kind: str = "secondary", height: int = 30) -> QPushButton:
    """kind: primary | secondary | danger | ghost"""
    b = QPushButton(text)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    b.setFont(UIFont(11.5, QFont.Weight.Medium if kind == "primary" else None))
    b.setFixedHeight(height)
    styles = {
        "primary":   (C.PRI, C.BG, C.PRI, C.WHITE, C.WHITE),
        "secondary": (C.PANEL2, C.TEXT, C.BORDER_B, C.HOVER, C.PRI),
        "danger":    ("transparent", C.RED, C.BAD_LINE, C.BAD_BG, C.RED),
        "ghost":     ("transparent", C.TEXT_MED, C.BORDER, C.PANEL2, C.BORDER_B),
    }
    bg, fg, line, hbg, hline = styles[kind]
    b.setStyleSheet(f"""
        QPushButton {{ background: {bg}; color: {fg}; border: 1px solid {line};
                       border-radius: 8px; padding: 0 14px; }}
        QPushButton:hover {{ background: {hbg}; border-color: {hline}; {'color: ' + C.TEXT + ';' if kind == 'ghost' else ''} }}
        QPushButton:pressed {{ background: {C.PANEL}; }}
        QPushButton:disabled {{ color: {C.TEXT_DIM}; border-color: {C.BORDER}; background: transparent; }}
        QPushButton:focus {{ outline: none; border-color: {C.PRI}; }}
    """)
    return b


# ── status chip ──────────────────────────────────────────────────────────────

class Chip(QAbstractButton):
    """Rounded status pill: coloured dot + text. Clickable (opens the related setting)."""

    @staticmethod
    def _tones() -> dict:
        # read at paint time so an accent change is picked up immediately
        return {
            "neutral": (C.TEXT_DIM, C.TEXT_MED, C.BORDER),
            "ok":      (C.GREEN, C.TEXT_MED, C.BORDER),
            "live":    (C.PRI, C.TEXT, C.BORDER_B),
            "warn":    (C.ACC2, C.ACC2, C.WARN_LINE),
            "bad":     (C.RED, C.RED, C.BAD_LINE),
        }

    def __init__(self, text: str = "", tone: str = "neutral", parent=None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(UIFont(11))
        self._tone = tone
        self._hover = 0.0
        self.setText(text)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(0 if reduce_motion() else 120)
        self._anim.valueChanged.connect(lambda v: (setattr(self, "_hover", float(v)), self.update()))

    def set(self, text: str, tone: str | None = None, tooltip: str | None = None) -> None:
        changed = text != self.text() or (tone and tone != self._tone)
        if tone:
            self._tone = tone
        self.setText(text)
        if tooltip is not None:
            self.setToolTip(tooltip)
        if changed:
            self.updateGeometry()
            self.update()

    @property
    def tone(self) -> str:
        return self._tone

    def sizeHint(self):
        from PyQt6.QtCore import QSize
        w = self.fontMetrics().horizontalAdvance(self.text()) + 32
        return QSize(w, 24)

    def enterEvent(self, e):
        self._anim.stop(); self._anim.setStartValue(self._hover); self._anim.setEndValue(1.0); self._anim.start()

    def leaveEvent(self, e):
        self._anim.stop(); self._anim.setStartValue(self._hover); self._anim.setEndValue(0.0); self._anim.start()

    def paintEvent(self, _):
        tones = self._tones()
        dot, fg, line = tones.get(self._tone, tones["neutral"])
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        p.setPen(QPen(qcol(line), 1))
        p.setBrush(mix(C.PANEL, C.HOVER, self._hover))
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcol(dot))
        cy = self.height() / 2
        if self._tone == "live":
            p.setBrush(qcol(dot, 40)); p.drawEllipse(QPointF(13, cy), 6, 6)
            p.setBrush(qcol(dot))
        p.drawEllipse(QPointF(13, cy), 3, 3)
        p.setPen(QPen(qcol(fg)))
        p.setFont(self.font())
        p.drawText(QRectF(22, 0, self.width() - 28, self.height()),
                   Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, self.text())


# ── toggle, segmented control, level meter ───────────────────────────────────

class Toggle(QAbstractButton):
    """Pill switch with an eased knob."""

    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(36, 20)
        self._pos = 1.0 if checked else 0.0
        super().setChecked(checked)
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(0 if reduce_motion() else 140)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.valueChanged.connect(lambda v: (setattr(self, "_pos", float(v)), self.update()))
        self.toggled.connect(self._animate)

    def _animate(self, on: bool):
        self._anim.stop()
        self._anim.setStartValue(self._pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()

    def set_quietly(self, on: bool) -> None:
        """Reflect backend state without emitting toggled()."""
        self.blockSignals(True)
        super().setChecked(on)
        self.blockSignals(False)
        self._pos = 1.0 if on else 0.0
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        t, en = self._pos, self.isEnabled()
        track = mix(C.HOVER, C.PRI, t)
        if not en:
            track.setAlpha(90)
        p.setPen(QPen(qcol(C.BORDER_B if t < 0.5 else C.PRI, 255 if en else 90), 1))
        p.setBrush(track)
        p.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 10, 10)
        x = 3 + t * (self.width() - 20)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcol(C.BG if t > 0.5 else C.TEXT_MED, 255 if en else 110))
        p.drawEllipse(QRectF(x, 3, 14, 14))


class Segmented(QWidget):
    """Exclusive segmented control. changed(key) when the user picks a segment."""
    changed = pyqtSignal(str)

    def __init__(self, options: list[tuple[str, str]], current: str, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        self._grp = QButtonGroup(self); self._grp.setExclusive(True)
        self._btns: dict[str, QPushButton] = {}
        for i, (key, text) in enumerate(options):
            b = QPushButton(text); b.setCheckable(True); b.setChecked(key == current)
            b.setCursor(Qt.CursorShape.PointingHandCursor); b.setFont(UIFont(11)); b.setFixedHeight(26)
            l = "7px" if i == 0 else "0"; r = "7px" if i == len(options) - 1 else "0"
            b.setStyleSheet(f"""
                QPushButton {{ background: {C.INPUT}; color: {C.TEXT_MED}; border: 1px solid {C.BORDER};
                    border-top-left-radius: {l}; border-bottom-left-radius: {l};
                    border-top-right-radius: {r}; border-bottom-right-radius: {r}; padding: 0 12px; }}
                QPushButton:hover {{ color: {C.TEXT}; background: {C.PANEL2}; }}
                QPushButton:checked {{ background: {C.PRI}; color: {C.BG}; border-color: {C.PRI}; }}
                QPushButton:disabled {{ color: {C.TEXT_DIM}; }}
            """)
            b.clicked.connect(lambda _=False, k=key: self.changed.emit(k))
            self._grp.addButton(b); lay.addWidget(b); self._btns[key] = b

    def set_quietly(self, key: str) -> None:
        b = self._btns.get(key)
        if b:
            b.setChecked(True)

    def setEnabled(self, on: bool) -> None:          # noqa: N802
        super().setEnabled(on)
        for b in self._btns.values():
            b.setEnabled(on)


class LevelMeter(QWidget):
    """Horizontal dBFS meter with peak hold and an optional speech dot."""

    def __init__(self, show_dot: bool = True, parent=None):
        super().__init__(parent)
        self.setFixedHeight(14)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._v = self._peak = 0.0
        self._speech = False
        self._dot = show_dot

    def set_level(self, db: float, speech: bool = False) -> None:
        v = max(0.0, min(1.0, (db + 60.0) / 60.0))
        self._v += (v - self._v) * (0.6 if v > self._v else 0.2)
        self._peak = max(self._v, self._peak - 0.01)
        self._speech = speech
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width() - (18 if self._dot else 0), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(qcol(C.INPUT)); p.drawRoundedRect(QRectF(0, 4, w, h - 8), 3, 3)
        p.setBrush(qcol(C.RED if self._v > 0.92 else (C.ACC2 if self._v > 0.78 else C.PRI)))
        p.drawRoundedRect(QRectF(0, 4, max(2.0, w * self._v), h - 8), 3, 3)
        p.setPen(QPen(qcol(C.TEXT), 1.4))
        p.drawLine(QPointF(w * self._peak, 2), QPointF(w * self._peak, h - 2))
        if self._dot:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(qcol(C.GREEN if self._speech else C.HOVER))
            p.drawEllipse(QRectF(w + 6, h / 2 - 4, 8, 8))


# ── toasts ───────────────────────────────────────────────────────────────────

class Toast(QWidget):
    """One notification card. Errors stay until dismissed; others auto-hide."""
    closed = pyqtSignal()

    def __init__(self, title: str, body: str = "", tone: str = "neutral",
                 action: tuple[str, object] | None = None, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("Toast")
        line, fg, bg = {"bad": (C.BAD_LINE, C.RED, C.BAD_BG), "warn": (C.WARN_LINE, C.ACC2, C.WARN_BG),
                        "ok": (C.GREEN_D, C.GREEN, C.OK_BG)}.get(tone, (C.BORDER_B, C.TEXT, C.PANEL2))
        self.setStyleSheet(f"#Toast {{ background: {bg}; border: 1px solid {line}; border-radius: 10px; }}")
        self.setFixedWidth(300)
        lay = QVBoxLayout(self); lay.setContentsMargins(12, 9, 8, 10); lay.setSpacing(2)
        top = QHBoxLayout(); top.setSpacing(6)
        t = label(title, 11.5, fg, QFont.Weight.DemiBold); t.setWordWrap(True)      # Preferred policy:
        top.addWidget(t, 1)                                                           # height follows text
        x = QPushButton("✕"); x.setFixedSize(20, 20); x.setCursor(Qt.CursorShape.PointingHandCursor)
        x.setStyleSheet(f"QPushButton {{ color: {C.TEXT_DIM}; background: transparent; border: none; }}"
                        f"QPushButton:hover {{ color: {C.TEXT}; }}")
        x.clicked.connect(self.dismiss)
        top.addWidget(x, 0, Qt.AlignmentFlag.AlignTop)
        lay.addLayout(top)
        if body:
            b_ = label(body, 11, C.TEXT_MED); b_.setWordWrap(True)
            lay.addWidget(b_)
        if action:
            b = button(action[0], "secondary", 26)
            b.clicked.connect(lambda: (action[1](), self.dismiss()))
            lay.addSpacing(4); lay.addWidget(b, 0, Qt.AlignmentFlag.AlignLeft)
        self.tone = tone
        if tone != "bad":
            QTimer.singleShot(6000, self.dismiss)

    def dismiss(self):
        if self.isVisible():
            self.hide()
            self.closed.emit()
            self.deleteLater()


class ToastHost(QWidget):
    """Stacks toasts in a corner of its parent. Never steals focus or clicks elsewhere."""
    MAX = 4

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self._toasts: list[Toast] = []
        self.right_margin = 16
        self.top_margin = 12
        self.hide()

    def show_toast(self, title: str, body: str = "", tone: str = "neutral", action=None,
                   key: str | None = None) -> Toast:
        # same key replaces the previous toast (e.g. repeated reconnect errors)
        if key:
            for t in list(self._toasts):
                if getattr(t, "key", None) == key:
                    t.dismiss()
        t = Toast(title, body, tone, action, self.parentWidget())
        t.key = key
        t.closed.connect(lambda t=t: self._remove(t))
        self._toasts.append(t)
        while len(self._toasts) > self.MAX:
            self._toasts[0].dismiss()
        self.relayout()
        fade_in(t)
        return t

    def _remove(self, t):
        if t in self._toasts:
            self._toasts.remove(t)
        self.relayout()

    def relayout(self):
        par = self.parentWidget()
        y = self.top_margin
        for t in self._toasts:
            t.adjustSize()
            t.move(par.width() - t.width() - self.right_margin, y)
            t.raise_()
            y += t.height() + 8

    def clear(self):
        for t in list(self._toasts):
            t.dismiss()


def relative_time(ts: float) -> str:
    d = time.time() - ts
    if d < 60:
        return "just now"
    if d < 3600:
        return f"{int(d // 60)} min ago"
    return time.strftime("%H:%M", time.localtime(ts))
