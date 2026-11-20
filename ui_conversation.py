"""
ui_conversation.py — conversation sidebar: messages, tool activity, results, files, composer.

Replaces the single typewriter log. Rows are real widgets with selectable text:
  you          your message (voice transcript, typed, or from the phone)
  assistant    JARVIS's reply
  tool         a tool call: running → done / failed / declined / waiting for approval, with duration
  note         small system notes (connected, phone paired, …)
  error        a failure you should notice

Session-only: nothing is written to disk. The list is capped (oldest rows
removed) so a long session can't grow memory without bound.
"""

from __future__ import annotations

import re
import time

from PyQt6.QtCore import QEvent, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QGuiApplication
from PyQt6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QLineEdit, QPushButton, QScrollArea,
                             QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

from ui_theme import C, UIFont, button, caps, label, qcol

MAX_ROWS = 400

TOOL_NAMES = {
    "open_app": "Open app", "web_search": "Web search", "weather_report": "Weather",
    "send_message": "Send message", "reminder": "Reminder", "youtube_video": "YouTube",
    "screen_process": "Look", "close_camera": "Close camera", "computer_settings": "Computer settings",
    "browser_control": "Browser", "file_controller": "Files", "desktop_control": "Desktop",
    "code_helper": "Code", "dev_agent": "Build project", "computer_control": "Keyboard & mouse",
    "game_updater": "Games", "flight_finder": "Flights", "system_status": "System status",
    "file_processor": "Process file", "save_memory": "Remember", "shutdown_jarvis": "Shut down",
}

# log tags written by actions → tool names (for routing progress lines)
_TAG_TOOL = {"code": "code_helper", "youtube": "youtube_video", "msg": "send_message",
             "gameupdater": "game_updater", "file": "file_controller", "desktop": "desktop_control",
             "browser": "browser_control", "settings": "computer_settings", "reminder": "reminder"}


def _sel(l: QLabel) -> QLabel:
    l.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    l.setCursor(Qt.CursorShape.IBeamCursor)
    return l


def _hhmm(ts: float) -> str:
    return time.strftime("%H:%M", time.localtime(ts))


class _Pinned:
    """Mixin: once the list has pinned our height, don't make Qt re-measure wrapped text.

    QBoxLayout re-runs heightForWidth for every wrapped label in the whole list on
    each insertion (measured: ≈180 ms per message at 200 rows). Pinned rows report
    no height-for-width, so an insertion costs the same at 10 or 400 rows.
    """
    def hasHeightForWidth(self) -> bool:          # noqa: N802 (Qt override)
        return False


class _Row(_Pinned, QWidget):
    kind = "row"

    def refit(self) -> None:
        """Ask the list to re-measure this row after its content changed."""
        p = self.parentWidget()
        while p is not None and not isinstance(p, _List):
            p = p.parentWidget()
        if p is not None:
            p.freeze(self)


class UserRow(_Row):
    kind = "you"

    def __init__(self, text: str, ts: float, source: str = ""):
        super().__init__()
        # Full width minus a left indent: the measured height always matches the
        # drawn height (rows are measured once and pinned — see _Pinned).
        lay = QHBoxLayout(self); lay.setContentsMargins(40, 0, 0, 0)
        bubble = QFrame(); bubble.setObjectName("Bubble")
        bubble.setStyleSheet(f"#Bubble {{ background: {C.PANEL2}; border: 1px solid {C.BORDER}; "
                             f"border-radius: 12px; border-bottom-right-radius: 4px; }}")
        b = QVBoxLayout(bubble); b.setContentsMargins(11, 7, 11, 8); b.setSpacing(2)
        head = QHBoxLayout(); head.setSpacing(6)
        self.who = label(f"YOU{' · ' + source.upper() if source else ''} · {_hhmm(ts)}", 9.5, C.TEXT_DIM, spacing=0.6)
        head.addWidget(self.who)
        self.badge = label("", 9.5, C.TEXT_DIM)
        head.addWidget(self.badge); head.addStretch()
        b.addLayout(head)
        b.addWidget(_sel(label(text, 12.5, C.TEXT, wrap=True)))
        lay.addWidget(bubble, 1)
        self.bubble = bubble

    def set_speaker(self, verdict_label: str, score: float | None) -> None:
        """Show the voice-identity result for this utterance (display only)."""
        txt, col = {"you": ("✓ your voice", C.GREEN), "uncertain": ("? not sure it's you", C.ACC2),
                    "unknown": ("unknown speaker", C.ACC2)}.get(verdict_label, ("", C.TEXT_DIM))
        if score is not None and txt:
            txt += f" · {score:.2f}"
        self.badge.setText(txt)
        self.badge.setStyleSheet(f"color: {col}; background: transparent;")
        self.refit()


class AssistantRow(_Row):
    kind = "assistant"

    def __init__(self, name: str, text: str, ts: float):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 36, 0); lay.setSpacing(2)
        lay.addWidget(label(f"{name.upper()} · {_hhmm(ts)}", 9.5, C.TEXT_DIM, spacing=0.6))
        self.body = _sel(label(text, 12.5, C.TEXT, wrap=True))
        lay.addWidget(self.body)


class NoteRow(_Row):
    kind = "note"

    def __init__(self, text: str, ts: float):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(6)
        lay.addWidget(label("·", 12, C.TEXT_DIM))
        lay.addWidget(_sel(label(text, 11, C.TEXT_DIM, wrap=True)), 1)
        lay.addWidget(label(_hhmm(ts), 9.5, C.TEXT_DIM, mono=True))


class ErrorRow(_Row):
    kind = "error"

    def __init__(self, text: str, ts: float):
        super().__init__()
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        box = QFrame(); box.setObjectName("Err")
        box.setStyleSheet(f"#Err {{ background: {C.BAD_BG}; border: 1px solid {C.BAD_LINE}; border-radius: 9px; }}")
        b = QHBoxLayout(box); b.setContentsMargins(10, 7, 10, 7); b.setSpacing(8)
        b.addWidget(label("!", 12, C.RED, QFont.Weight.DemiBold), 0, Qt.AlignmentFlag.AlignTop)
        b.addWidget(_sel(label(text, 11.5, C.TEXT, wrap=True)), 1)
        b.addWidget(label(_hhmm(ts), 9.5, C.TEXT_DIM, mono=True), 0, Qt.AlignmentFlag.AlignTop)
        lay.addWidget(box)


class ToolRow(_Row):
    """A tool call and its live status."""
    kind = "tool"

    STATES = {  # icon, colour, word
        "running":  ("●", "ACC2", "running"),
        "approval": ("●", "ACC2", "waiting for your approval"),
        "done":     ("✓", "GREEN", "done"),
        "failed":   ("✕", "RED", "failed"),
        "declined": ("⊘", "TEXT_DIM", "declined"),
    }

    def __init__(self, call_id: str, tool: str, detail: str, ts: float):
        super().__init__()
        self.call_id, self.tool, self.started = call_id, tool, ts
        self.status = "running"
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        box = QFrame(); box.setObjectName("Tool")
        box.setStyleSheet(f"#Tool {{ background: #1D1F23; border: 1px solid {C.BORDER}; border-radius: 8px; }}")
        b = QVBoxLayout(box); b.setContentsMargins(9, 5, 10, 6); b.setSpacing(2)
        top = QHBoxLayout(); top.setSpacing(7)
        self.icon = label("", 11, C.ACC2)
        top.addWidget(self.icon)
        top.addWidget(label(TOOL_NAMES.get(tool, tool.replace("_", " ").capitalize()), 11.5, C.TEXT,
                            QFont.Weight.Medium))
        self.state_lbl = label("", 11, C.TEXT_MED)
        top.addWidget(self.state_lbl, 1)
        self.dur = label("", 10, C.TEXT_DIM, mono=True)
        top.addWidget(self.dur)
        b.addLayout(top)
        self.detail = _sel(label(detail, 10.5, C.TEXT_DIM, wrap=True))
        self.detail.setVisible(bool(detail))
        b.addWidget(self.detail)
        self.progress = label("", 10.5, C.TEXT_MED, wrap=True)
        self.progress.hide()
        b.addWidget(self.progress)
        lay.addWidget(box, 1)
        self.set_status("running")

    def set_status(self, status: str, summary: str = "") -> None:
        self.status = status
        icon, col, word = self.STATES.get(status, self.STATES["running"])
        self.icon.setText(icon)
        self.icon.setStyleSheet(f"color: {getattr(C, col)}; background: transparent;")
        self.state_lbl.setText(word)
        if status in ("done", "failed", "declined"):
            self.dur.setText(f"{time.time() - self.started:.1f} s")
        if summary:
            self.set_progress(summary)
        else:
            self.refit()

    def set_progress(self, text: str) -> None:
        self.progress.setText(text[:300])
        self.progress.show()
        self.refit()


class ResultCard(_Pinned, QFrame):
    def __init__(self, title: str, text: str, ts: float):
        super().__init__()
        self.setObjectName("Res")
        self.setStyleSheet(f"#Res {{ background: {C.PANEL2}; border: 1px solid {C.BORDER}; border-radius: 10px; }}")
        lay = QVBoxLayout(self); lay.setContentsMargins(12, 9, 12, 10); lay.setSpacing(6)
        top = QHBoxLayout()
        top.addWidget(label(title, 11.5, C.TEXT, QFont.Weight.Medium, wrap=True), 1)
        top.addWidget(label(_hhmm(ts), 9.5, C.TEXT_DIM, mono=True))
        lay.addLayout(top)
        body = _sel(label(text[:4000], 11.5, C.TEXT_MED, wrap=True))
        lay.addWidget(body)
        cp = button("Copy", "ghost", 24)
        cp.clicked.connect(lambda: (QGuiApplication.clipboard().setText(text), cp.setText("Copied")))
        lay.addWidget(cp, 0, Qt.AlignmentFlag.AlignLeft)


class _List(QScrollArea):
    """Vertical list that sticks to the bottom unless you've scrolled up."""
    new_below = pyqtSignal(bool)

    def __init__(self, spacing: int = 10):
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
        self.lay = QVBoxLayout(self.body); self.lay.setContentsMargins(14, 12, 12, 12); self.lay.setSpacing(spacing)
        self.lay.addStretch(1)
        self.setWidget(self.body)
        self._stick = True
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        self.verticalScrollBar().rangeChanged.connect(self._on_range)

    def _on_scroll(self, v):
        sb = self.verticalScrollBar()
        self._stick = v >= sb.maximum() - 24
        if self._stick:
            self.new_below.emit(False)

    def _on_range(self, _lo, hi):
        if self._stick:
            self.verticalScrollBar().setValue(hi)
        else:
            self.new_below.emit(True)

    def _row_width(self) -> int:
        m = self.lay.contentsMargins()
        return max(120, self.viewport().width() - m.left() - m.right())

    def freeze(self, w: QWidget) -> None:
        """Pin a row's height for the current width.

        Wrapped labels make Qt re-measure every row on every insertion, so an
        unpinned list costs O(rows) per message (≈200 ms at 200 rows). Pinned
        rows make each insertion cheap; we re-measure only when the width or
        that row's content changes.
        """
        w.setMinimumHeight(0); w.setMaximumHeight(16_777_215)
        lay = w.layout()
        h = lay.totalHeightForWidth(self._row_width()) if lay and lay.hasHeightForWidth() else w.sizeHint().height()
        w.setFixedHeight(max(h, w.minimumSizeHint().height() if not (lay and lay.hasHeightForWidth()) else h))

    def add(self, w: QWidget) -> None:
        self.lay.insertWidget(self.lay.count(), w)
        self.freeze(w)
        while self.lay.count() - 1 > MAX_ROWS:
            old = self.lay.takeAt(1).widget()
            if old:
                old.deleteLater()

    def rows(self) -> list[QWidget]:
        return [self.lay.itemAt(i).widget() for i in range(1, self.lay.count()) if self.lay.itemAt(i).widget()]

    def scroll_to_end(self):
        self._stick = True
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.body.setFixedWidth(self.viewport().width())
        if not hasattr(self, "_refit"):
            self._refit = QTimer(self); self._refit.setSingleShot(True)
            self._refit.timeout.connect(self._refreeze_all)
        self._refit.start(120)                       # debounce: re-measure once after resizing stops

    def _refreeze_all(self):
        for r in self.rows():
            self.freeze(r)


class _Tab(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(UIFont(11.5))
        self.base = text
        self.setStyleSheet(f"""
            QPushButton {{ background: transparent; color: {C.TEXT_DIM}; border: none;
                           border-bottom: 2px solid transparent; padding: 10px 2px 8px 2px; }}
            QPushButton:hover {{ color: {C.TEXT_MED}; }}
            QPushButton:checked {{ color: {C.TEXT}; border-bottom: 2px solid {C.PRI}; }}
        """)

    def set_count(self, n: int):
        self.setText(f"{self.base}  {n}" if n else self.base)


class ConversationPanel(QWidget):
    """Sidebar with Conversation / Results / Files tabs and the composer."""
    send_text = pyqtSignal(str)
    attach_requested = pyqtSignal()

    def __init__(self, assistant_name: str = "JARVIS", parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("ConvPanel")
        self.setStyleSheet(f"#ConvPanel {{ background: {C.PANEL}; border-left: 1px solid {C.BORDER}; }}")
        self.name = assistant_name
        self._tools: dict[str, ToolRow] = {}
        self._last_user: UserRow | None = None
        self._unseen_results = 0

        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(0)
        bar = QWidget(); bar.setStyleSheet(f"border-bottom: 1px solid {C.BORDER}; background: transparent;")
        tb = QHBoxLayout(bar); tb.setContentsMargins(14, 0, 14, 0); tb.setSpacing(18)
        self.tabs = [_Tab("Conversation"), _Tab("Results"), _Tab("Files")]
        for i, t in enumerate(self.tabs):
            t.clicked.connect(lambda _=False, i=i: self.show_tab(i))
            tb.addWidget(t)
        tb.addStretch()
        lay.addWidget(bar)

        self.stack = QStackedWidget()
        self.convo = _List(10)
        self.results = _List(10)
        self.results_empty = label("Web results and answers JARVIS shows on screen will appear here.",
                                   11.5, C.TEXT_DIM, wrap=True)
        self.results.lay.insertWidget(0, self.results_empty)
        self.files = QWidget()
        fl = QVBoxLayout(self.files); fl.setContentsMargins(14, 14, 14, 14); fl.setSpacing(10)
        fl.addWidget(caps("Current file"))
        self.files_host = QVBoxLayout(); fl.addLayout(self.files_host)
        fl.addWidget(label("Drop a file here or use ＋ in the message box. JARVIS can summarise, convert, "
                           "transcribe or answer questions about it.", 11, C.TEXT_DIM, wrap=True))
        fl.addStretch()
        for w in (self.convo, self.results, self.files):
            self.stack.addWidget(w)
        lay.addWidget(self.stack, 1)

        self.jump = button("↓  New messages", "secondary", 26)
        self.jump.setParent(self)
        self.jump.hide()
        self.jump.clicked.connect(lambda: (self.convo.scroll_to_end(), self.jump.hide()))
        self.convo.new_below.connect(lambda on: self.jump.setVisible(on and self.stack.currentIndex() == 0))

        comp = QWidget(); comp.setStyleSheet(f"border-top: 1px solid {C.BORDER}; background: transparent;")
        cl = QHBoxLayout(comp); cl.setContentsMargins(12, 10, 12, 12); cl.setSpacing(8)
        self.attach = QPushButton("＋"); self.attach.setFixedSize(34, 34)
        self.attach.setToolTip("Attach a file")
        self.attach.setCursor(Qt.CursorShape.PointingHandCursor)
        self.attach.setStyleSheet(f"QPushButton {{ background: {C.PANEL2}; color: {C.TEXT_MED}; border: 1px solid {C.BORDER};"
                                  f" border-radius: 9px; font-size: 15px; }} QPushButton:hover {{ color: {C.TEXT};"
                                  f" border-color: {C.BORDER_B}; }}")
        self.attach.clicked.connect(self.attach_requested.emit)
        self.input = QLineEdit(); self.input.setFont(UIFont(12.5))
        self.input.setPlaceholderText(f"Message {assistant_name}…")
        self.input.setFixedHeight(34)
        self.input.setStyleSheet(f"""
            QLineEdit {{ background: {C.INPUT}; color: {C.TEXT}; border: 1px solid {C.BORDER};
                         border-radius: 9px; padding: 0 11px; selection-background-color: {C.HOVER}; }}
            QLineEdit:focus {{ border-color: {C.BORDER_B}; }}
        """)
        self.input.returnPressed.connect(self._send)
        self.send_btn = QPushButton("↑"); self.send_btn.setFixedSize(34, 34)
        self.send_btn.setToolTip("Send (Return)")
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setStyleSheet(f"QPushButton {{ background: {C.PRI}; color: {C.BG}; border: none; border-radius: 9px;"
                                    f" font-size: 15px; font-weight: 600; }} QPushButton:hover {{ background: {C.WHITE}; }}"
                                    f" QPushButton:disabled {{ background: {C.HOVER}; color: {C.TEXT_DIM}; }}")
        self.send_btn.clicked.connect(self._send)
        self.input.textChanged.connect(lambda t: self.send_btn.setEnabled(bool(t.strip())))
        self.send_btn.setEnabled(False)
        cl.addWidget(self.attach); cl.addWidget(self.input, 1); cl.addWidget(self.send_btn)
        lay.addWidget(comp)
        self.show_tab(0)

    # ── tabs ──────────────────────────────────────────────────────────────

    def show_tab(self, i: int) -> None:
        self.stack.setCurrentIndex(i)
        for j, t in enumerate(self.tabs):
            t.setChecked(i == j)
        if i == 1:
            self._unseen_results = 0
            self.tabs[1].set_count(0)
        self.jump.hide()

    def set_files_widget(self, w: QWidget) -> None:
        self.files_host.addWidget(w)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.jump.adjustSize()
        self.jump.move((self.width() - self.jump.width()) // 2, self.height() - 96)

    # ── composer ──────────────────────────────────────────────────────────

    def _send(self):
        t = self.input.text().strip()
        if not t:
            return
        self.input.clear()
        self.send_text.emit(t)

    def focus_input(self):
        self.input.setFocus()

    # ── content API (GUI thread) ─────────────────────────────────────────

    def add_user(self, text: str, source: str = "", ts: float | None = None) -> UserRow:
        r = UserRow(text, ts or time.time(), source)
        self.convo.add(r)
        self._last_user = r
        return r

    def add_assistant(self, text: str, ts: float | None = None) -> None:
        self.convo.add(AssistantRow(self.name, text, ts or time.time()))

    def add_note(self, text: str) -> None:
        self.convo.add(NoteRow(text, time.time()))

    def add_error(self, text: str) -> None:
        self.convo.add(ErrorRow(text, time.time()))

    def tool_started(self, call_id: str, tool: str, detail: str = "") -> None:
        if tool == "save_memory":
            return                                    # silent by design; shown as nothing
        r = ToolRow(call_id, tool, detail, time.time())
        self._tools[call_id] = r
        self.convo.add(r)
        if len(self._tools) > 50:
            for k in list(self._tools)[:-50]:
                self._tools.pop(k, None)

    def tool_status(self, call_id: str, status: str, summary: str = "") -> None:
        r = self._tools.get(call_id)
        if r is not None:
            r.set_status(status, summary)

    def tool_progress_for(self, tool: str, text: str) -> bool:
        """Attach a progress line to the newest running call of `tool`."""
        for r in reversed(list(self._tools.values())):
            if r.tool == tool and r.status in ("running", "approval"):
                r.set_progress(text)
                return True
        return False

    def set_last_user_speaker(self, verdict_label: str, score: float | None, max_age_s: float = 20) -> None:
        r = self._last_user
        if r is not None and time.time() - getattr(r, "_ts", time.time()) <= max_age_s:
            r.set_speaker(verdict_label, score)

    def add_result(self, title: str, text: str) -> None:
        self.results_empty.hide()
        self.results.add(ResultCard(title, text, time.time()))
        if self.stack.currentIndex() != 1:
            self._unseen_results += 1
            self.tabs[1].set_count(self._unseen_results)

    def set_name(self, name: str) -> None:
        self.name = name
        self.input.setPlaceholderText(f"Message {name}…")

    # ── legacy log lines (write_log) ─────────────────────────────────────

    _TAG = re.compile(r"^\[([A-Za-z]+)\]:?\s*(.*)$", re.S)

    def route_log(self, text: str) -> str:
        """Turn an old-style log line into the right row. Returns the row kind used."""
        t = (text or "").strip()
        if not t:
            return "none"
        low = t.lower()
        if low.startswith("you:"):
            self.add_user(t[4:].strip()); return "you"
        if low.startswith(f"{self.name.lower()}:"):
            self.add_assistant(t[len(self.name) + 1:].strip()); return "assistant"
        if t.startswith("[Web]:"):
            self.add_user(t[6:].strip(), source="phone"); return "you"
        if low.startswith("err:"):
            self.add_error(t[4:].strip()); return "error"
        if low.startswith("sys:"):
            self.add_note(t[4:].strip()); return "note"
        m = self._TAG.match(t)
        if m:
            tool = _TAG_TOOL.get(m.group(1).lower())
            if tool and self.tool_progress_for(tool, m.group(2)):
                return "progress"
            self.add_note(m.group(2) or t); return "note"
        self.add_note(t)
        return "note"
