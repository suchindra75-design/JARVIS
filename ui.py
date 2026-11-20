from __future__ import annotations

import collections
import json
import math
import os
import platform
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

if platform.system() == "Windows":
    _WIN_HIDE: dict = {"creationflags": subprocess.CREATE_NO_WINDOW}
else:
    _WIN_HIDE: dict = {}

from PyQt6.QtCore import (
    QEasingCurve, QMimeData, QObject, QPointF, QRectF, QSize, Qt,
    QTimer, QUrl, pyqtSignal,
)
from PyQt6.QtGui import (
    QBrush, QColor, QConicalGradient, QDragEnterEvent, QDropEvent, QFont,
    QFontDatabase, QKeySequence, QLinearGradient, QPainter, QPainterPath,
    QPen, QPixmap, QRadialGradient, QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QScrollArea, QSizePolicy, QSplitter,
    QStackedWidget, QTextEdit, QVBoxLayout, QWidget, QProgressBar,
)
from ui_theme import (C, UIFont, reduce_motion, fade_in, qcol, Chip, ToastHost, button, caps,
                      label)                                    # Graphite & Silver design system
from ui_stage import VoiceStage
from ui_conversation import ConversationPanel

def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

BASE_DIR   = _base_dir()
CONFIG_DIR = BASE_DIR / "config"
API_FILE   = CONFIG_DIR / "api_keys.json"


def _read_full_config() -> dict:
    """Read api_keys.json config dict. Returns {} on any error."""
    try:
        return json.loads(API_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


_DEFAULT_W, _DEFAULT_H = 1180, 760
_MIN_W,     _MIN_H     = 820, 580
_CONVO_W = 360          # conversation sidebar
_NARROW_W = 1100        # below this the sidebar becomes a toggle

_OS = platform.system()  # "Windows" | "Darwin" | "Linux"


# Ana renge (accent) bağlı anahtarlar — durum renkleri (ACC, GREEN, RED…) sabit kalır
_HUE_LINKED = (
    "BG", "PANEL", "PANEL2", "BORDER", "BORDER_B", "BORDER_A",
    "PRI", "PRI_DIM", "PRI_GHO", "TEXT", "TEXT_DIM", "TEXT_MED",
    "WHITE", "DARK", "BAR_BG",
)
_PALETTE_DEFAULTS: dict[str, str] = {k: getattr(C, k) for k in _HUE_LINKED}

DEFAULT_UI_COLOR = _PALETTE_DEFAULTS["PRI"]


def apply_ui_accent(accent_hex: str) -> bool:
    """Tint the silver accent toward the chosen hue, keeping it low-saturation.

    Graphite & Silver keeps its neutrals fixed; only the accent family (PRI,
    PRI_DIM, PRI_GHO, BORDER_B) picks up a subtle tint. Saturation is capped
    so no choice can turn the UI neon. Grey or the old cyan default = pure silver.
    """
    import colorsys

    accent_hex = (accent_hex or "").strip().lower()
    if not (accent_hex.startswith("#") and len(accent_hex) == 7):
        return False
    try:
        int(accent_hex[1:], 16)
    except ValueError:
        return False

    for k, v in _PALETTE_DEFAULTS.items():          # start from pure graphite
        setattr(C, k, v)
    r, g, b = (int(accent_hex[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, s, _v = colorsys.rgb_to_hsv(r, g, b)
    if s < 0.08 or accent_hex in (_LEGACY_DEFAULT_ACCENT, DEFAULT_UI_COLOR.lower()):
        return True

    def tint(key: str, sat_cap: float) -> None:
        _h0, _s0, v0 = colorsys.rgb_to_hsv(*(int(_PALETTE_DEFAULTS[key][i:i + 2], 16) / 255 for i in (1, 3, 5)))
        rr, gg, bb = colorsys.hsv_to_rgb(h, min(s, sat_cap), v0)
        setattr(C, key, "#{:02x}{:02x}{:02x}".format(int(rr * 255 + .5), int(gg * 255 + .5), int(bb * 255 + .5)))

    tint("PRI", 0.08)
    tint("PRI_DIM", 0.07)
    tint("PRI_GHO", 0.12)
    tint("BORDER_B", 0.06)
    return True


_LEGACY_DEFAULT_ACCENT = "#00d4ff"


def current_palette() -> dict[str, str]:
    """C sınıfındaki accent'e bağlı renklerin anlık kopyası."""
    return {k: getattr(C, k) for k in _HUE_LINKED}


def retheme_all_widgets(old: dict[str, str], new: dict[str, str]) -> None:
    """
    CANLI tam tema değişimi. Uygulamadaki HER widget'ın stylesheet'inde eski
    palet renklerini yenileriyle değiştirir ve yeniden çizdirir. Böylece renk
    değişimi yalnızca boyanan öğelerde değil, panel/buton/kenarlık dahil tüm
    arayüzde ANINDA uygulanır — yeniden başlatma gerekmez.
    """
    mapping = {old[k].lower(): new[k].lower()
               for k in old if old[k].lower() != new.get(k, old[k]).lower()}
    if not mapping:
        return
    app = QApplication.instance()
    if app is None:
        return
    for w in app.allWidgets():
        try:
            ss = w.styleSheet()
            if ss:
                s2 = ss
                for o, n in mapping.items():
                    if o in s2:
                        s2 = s2.replace(o, n)
                if s2 != ss:
                    w.setStyleSheet(s2)
            w.update()
        except Exception:
            pass


def _nvml_gpu_windows() -> float:
    """Return NVIDIA GPU utilisation % using nvml.dll directly — zero subprocess."""
    global _nvml_lib, _nvml_ok
    if _nvml_ok is False:
        return -1.0
    try:
        import ctypes

        class _Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

        if _nvml_lib is None:
            for dll_name in ("nvml", r"C:\Windows\System32\nvml.dll"):
                try:
                    lib = ctypes.WinDLL(dll_name)
                    lib.nvmlInit_v2()
                    _nvml_lib = lib
                    break
                except Exception:
                    continue

        if _nvml_lib is None:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            _nvml_ok = True
            return float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)

        dev = ctypes.c_void_p()
        _nvml_lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev))
        util = _Util()
        _nvml_lib.nvmlDeviceGetUtilizationRates(dev, ctypes.byref(util))
        _nvml_ok = True
        return float(util.gpu)
    except Exception:
        _nvml_ok = False
        return -1.0


class _SysMetrics:
    def __init__(self):
        self.cpu  = 0.0
        self.mem  = 0.0
        self.net  = 0.0   
        self.gpu  = -1.0  
        self.tmp  = -1.0  
        self._lock = threading.Lock()
        self._last_net = psutil.net_io_counters()
        self._last_net_t = time.time()
        self._running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def _loop(self):
        while self._running:
            try:
                self._update()
            except Exception:
                pass
            time.sleep(1.5)

    def _update(self):
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent

        nc  = psutil.net_io_counters()
        now = time.time()
        dt  = now - self._last_net_t
        if dt > 0:
            sent = (nc.bytes_sent - self._last_net.bytes_sent) / dt
            recv = (nc.bytes_recv - self._last_net.bytes_recv) / dt
            net  = (sent + recv) / (1024 * 1024)
        else:
            net = 0.0
        self._last_net   = nc
        self._last_net_t = now

        gpu = self._get_gpu()

        tmp = self._get_temp()

        with self._lock:
            self.cpu = cpu
            self.mem = mem
            self.net = net
            self.gpu = gpu
            self.tmp = tmp

    def _get_gpu(self) -> float:
        # pynvml — subprocess-free, works on all platforms if installed
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            return float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
        except Exception:
            pass

        # Windows: nvml.dll via ctypes (already cached in _nvml_gpu_windows)
        if _OS == "Windows":
            return _nvml_gpu_windows()

        # Linux / macOS: libnvidia-ml shared lib via ctypes
        try:
            import ctypes
            _lib = "libnvidia-ml.so.1" if _OS == "Linux" else "libnvidia-ml.dylib"

            class _Util(ctypes.Structure):
                _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

            nv = ctypes.CDLL(_lib)
            nv.nvmlInit_v2()
            dev = ctypes.c_void_p()
            nv.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev))
            u = _Util()
            nv.nvmlDeviceGetUtilizationRates(dev, ctypes.byref(u))
            return float(u.gpu)
        except Exception:
            pass

        return -1.0   # N/A — zero subprocess on all platforms

    def _get_temp(self) -> float:
        # psutil — works on Linux; occasionally Windows with driver support
        try:
            temps = psutil.sensors_temperatures()
            for name in ["coretemp", "k10temp", "cpu_thermal", "acpitz",
                         "cpu-thermal", "zenpower", "it8688"]:
                if name in temps and temps[name]:
                    return temps[name][0].current
            for entries in temps.values():
                if entries:
                    return entries[0].current
        except Exception:
            pass

        # Windows: wmi module (pure Python COM, zero subprocess)
        if _OS == "Windows":
            try:
                import wmi  # type: ignore
                w = wmi.WMI(namespace="root/wmi")
                tz = w.MSAcpi_ThermalZoneTemperature()
                if tz:
                    return (tz[0].CurrentTemperature / 10.0) - 273.15
            except Exception:
                pass

        return -1.0   # N/A — zero subprocess on all platforms

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "cpu": self.cpu,
                "mem": self.mem,
                "net": self.net,
                "gpu": self.gpu,
                "tmp": self.tmp,
            }


_metrics = _SysMetrics()

_FILE_ICONS = {
    "image":   ("🖼", C.PRI), "video":   ("🎬", C.PRI),
    "audio":   ("🎵", C.PRI), "pdf":     ("📄", C.PRI),
    "word":    ("📝", C.PRI), "excel":   ("📊", C.PRI),
    "code":    ("💻", C.PRI), "archive": ("📦", C.PRI),
    "pptx":    ("📊", C.PRI), "text":    ("📃", C.PRI),
    "data":    ("🔧", C.PRI), "unknown": ("📎", C.PRI),
}
_EXT_TO_CAT = {
    **dict.fromkeys(["jpg","jpeg","png","gif","webp","bmp","tiff","svg","ico"], "image"),
    **dict.fromkeys(["mp4","avi","mov","mkv","wmv","flv","webm","m4v"],         "video"),
    **dict.fromkeys(["mp3","wav","ogg","m4a","aac","flac","wma","opus"],        "audio"),
    **dict.fromkeys(["pdf"],                                                     "pdf"),
    **dict.fromkeys(["doc","docx"],                                              "word"),
    **dict.fromkeys(["xls","xlsx","ods"],                                        "excel"),
    **dict.fromkeys(["ppt","pptx"],                                              "pptx"),
    **dict.fromkeys(["py","js","ts","jsx","tsx","html","css","java","c","cpp",
                     "cs","go","rs","rb","php","swift","kt","sh","sql","lua"],   "code"),
    **dict.fromkeys(["zip","rar","tar","gz","7z","bz2","xz"],                   "archive"),
    **dict.fromkeys(["txt","md","rst","log"],                                    "text"),
    **dict.fromkeys(["csv","tsv","json","xml"],                                  "data"),
}

def _file_category(path: Path) -> str:
    return _EXT_TO_CAT.get(path.suffix.lower().lstrip("."), "unknown")

def _fmt_size(size: int) -> str:
    if   size < 1024:    return f"{size} B"
    elif size < 1024**2: return f"{size/1024:.1f} KB"
    elif size < 1024**3: return f"{size/1024**2:.1f} MB"
    else:                return f"{size/1024**3:.1f} GB"


class FileDropZone(QWidget):
    file_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(100)
        self._current_file: str | None = None
        self._hovering  = False
        self._drag_over = False
        self._dash_offset = 0.0
        self._anim_tmr = QTimer(self)
        self._anim_tmr.timeout.connect(self._animate)
        self._anim_tmr.start(40)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._canvas = _DropCanvas(self)
        layout.addWidget(self._canvas)

    def _animate(self):
        self._dash_offset = (self._dash_offset + 0.8) % 20
        self._canvas.update()

    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
            self._drag_over = True; self._canvas.update()

    def dragLeaveEvent(self, e):
        self._drag_over = False; self._canvas.update()

    def dropEvent(self, e: QDropEvent):
        self._drag_over = False
        urls = e.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if Path(path).is_file():
                self._set_file(path)
        self._canvas.update()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._browse()

    def enterEvent(self, e):
        self._hovering = True; self._canvas.update()

    def leaveEvent(self, e):
        self._hovering = False; self._canvas.update()

    def current_file(self) -> str | None:
        return self._current_file

    def clear_file(self):
        self._current_file = None; self._canvas.update()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a file for JARVIS", str(Path.home()),
            "All Files (*.*);;"
            "Images (*.jpg *.jpeg *.png *.gif *.webp *.bmp *.svg);;"
            "Documents (*.pdf *.docx *.txt *.md *.pptx);;"
            "Data (*.csv *.xlsx *.json *.xml);;"
            "Code (*.py *.js *.ts *.html *.css *.java *.cpp *.go);;"
            "Audio (*.mp3 *.wav *.ogg *.m4a *.aac *.flac);;"
            "Video (*.mp4 *.avi *.mov *.mkv *.wmv *.webm);;"
            "Archives (*.zip *.rar *.tar *.gz *.7z)",
        )
        if path:
            self._set_file(path)

    def _set_file(self, path: str):
        self._current_file = path
        self._canvas.update()
        self.file_selected.emit(path)


class _DropCanvas(QWidget):
    def __init__(self, zone: FileDropZone):
        super().__init__(zone)
        self._z = zone

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        z    = self._z
        W, H = self.width(), self.height()
        pad  = 6
        rect = QRectF(pad, pad, W - pad * 2, H - pad * 2)

        bg_col = qcol(C.HOVER if z._drag_over else (C.PANEL2 if z._hovering else C.PANEL))
        p.setBrush(QBrush(bg_col)); p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(rect, 6, 6)

        if z._current_file:   border_col = qcol(C.GREEN, 200)
        elif z._drag_over:    border_col = qcol(C.PRI, 230)
        elif z._hovering:     border_col = qcol(C.BORDER_B, 200)
        else:                 border_col = qcol(C.BORDER, 160)

        pen = QPen(border_col, 1.5, Qt.PenStyle.DashLine)
        pen.setDashOffset(z._dash_offset)
        p.setPen(pen); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(rect, 6, 6)

        if z._current_file:   self._paint_file(p, W, H)
        elif z._drag_over:    self._paint_drag_over(p, W, H)
        else:                 self._paint_idle(p, W, H, z._hovering)

    def _paint_idle(self, p, W, H, hover):
        cx, cy = W / 2, H / 2
        col = qcol(C.PRI_DIM if not hover else C.PRI)
        p.setPen(QPen(col, 2)); p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawLine(QPointF(cx, cy - 14), QPointF(cx, cy + 4))
        p.drawLine(QPointF(cx - 8, cy - 6), QPointF(cx, cy - 14))
        p.drawLine(QPointF(cx + 8, cy - 6), QPointF(cx, cy - 14))
        p.drawLine(QPointF(cx - 14, cy + 4), QPointF(cx + 14, cy + 4))
        p.setFont(UIFont(9))
        p.setPen(QPen(qcol(C.PRI_DIM if not hover else C.TEXT), 1))
        p.drawText(QRectF(0, cy + 8, W, 16), Qt.AlignmentFlag.AlignCenter,
                   "Drop file here  or  Click to Browse")
        p.setFont(UIFont(8))
        p.setPen(QPen(qcol(C.TEXT_DIM), 1))
        p.drawText(QRectF(0, cy + 24, W, 14), Qt.AlignmentFlag.AlignCenter,
                   "Images · Video · Audio · PDF · Docs · Code · Data")

    def _paint_drag_over(self, p, W, H):
        cx, cy = W / 2, H / 2
        p.setFont(UIFont(20))
        p.setPen(QPen(qcol(C.PRI), 1))
        p.drawText(QRectF(0, cy - 24, W, 32), Qt.AlignmentFlag.AlignCenter, "⬇")
        p.setFont(UIFont(9, QFont.Weight.Bold))
        p.setPen(QPen(qcol(C.PRI), 1))
        p.drawText(QRectF(0, cy + 12, W, 16), Qt.AlignmentFlag.AlignCenter, "Release to load")

    def _paint_file(self, p, W, H):
        path = Path(self._z._current_file)
        cat  = _file_category(path)
        icon, icon_col = _FILE_ICONS.get(cat, _FILE_ICONS["unknown"])
        size_str = _fmt_size(path.stat().st_size)
        ext_str  = path.suffix.upper().lstrip(".") or "FILE"

        block_x, block_w = 10, 60
        p.setFont(QFont("Segoe UI Emoji", 22) if _OS == "Windows" else QFont("Arial", 22))
        p.setPen(QPen(qcol(icon_col), 1))
        p.drawText(QRectF(block_x, 0, block_w, H), Qt.AlignmentFlag.AlignCenter, icon)

        tx = block_x + block_w + 6
        tw = W - tx - 38

        p.setFont(UIFont(9, QFont.Weight.Bold))
        p.setPen(QPen(qcol(C.WHITE), 1))
        name = path.name if len(path.name) <= 34 else path.name[:31] + "..."
        p.drawText(QRectF(tx, H * 0.18, tw, 16),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, name)

        p.setFont(UIFont(8))
        p.setPen(QPen(qcol(C.TEXT_DIM), 1))
        p.drawText(QRectF(tx, H * 0.18 + 18, tw, 14),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                   f"{ext_str}  ·  {size_str}")

        p.setFont(UIFont(7))
        p.setPen(QPen(qcol(C.TEXT_DIM), 1))
        par = str(path.parent)
        if len(par) > 42: par = "…" + par[-41:]
        p.drawText(QRectF(tx, H * 0.18 + 34, tw, 12),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, par)

        p.setFont(UIFont(9, QFont.Weight.Bold))
        p.setPen(QPen(qcol(C.RED, 180), 1))
        p.drawText(QRectF(W - 34, 0, 28, H), Qt.AlignmentFlag.AlignCenter, "✕")

    def mousePressEvent(self, e):
        z = self._z
        if z._current_file and e.pos().x() > self.width() - 34:
            z.clear_file()
        else:
            z.mousePressEvent(e)


class _CameraPreview(QWidget):
    """Floating overlay that briefly shows what the camera captured."""

    _W, _H = 244, 188

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            _CameraPreview {{
                background: {C.GLASS};
                border: 1px solid {C.PRI};
                border-radius: 10px;
            }}
        """)
        self.setFixedWidth(self._W)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 5, 6, 6)
        lay.setSpacing(4)

        hdr = QHBoxLayout()
        title = QLabel("◈  VISUAL INPUT")
        title.setFont(UIFont(8, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {C.PRI}; background: transparent;")
        hdr.addWidget(title)
        hdr.addStretch()
        close_btn = QPushButton("✕")
        close_btn.setFixedSize(16, 16)
        close_btn.setFont(UIFont(9))
        close_btn.setStyleSheet(
            f"color: {C.TEXT_DIM}; background: transparent; border: none;"
        )
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.clicked.connect(self.hide)
        hdr.addWidget(close_btn)
        lay.addLayout(hdr)

        self._img_lbl = QLabel()
        self._img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._img_lbl.setStyleSheet("background: transparent;")
        lay.addWidget(self._img_lbl)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

        self.hide()

    def show_frame(self, img_bytes: bytes) -> None:
        px = QPixmap()
        px.loadFromData(img_bytes)
        if not px.isNull():
            max_w = self._W - 12
            scaled = px.scaled(
                max_w, 160,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._img_lbl.setPixmap(scaled)
            self._img_lbl.setFixedSize(scaled.width(), scaled.height())
            self.adjustSize()
        self.show()
        self.raise_()
        self._timer.start(6_000)   # auto-dismiss after 6 s


class SetupOverlay(QWidget):
    done = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            SetupOverlay {{
                background: {C.GLASS};
                border: 1px solid {C.BORDER_B};
                border-radius: 10px;
            }}
        """)

        detected = {"darwin": "mac", "windows": "windows"}.get(
            _OS.lower(), "linux"
        )
        self._sel_os = detected

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 22, 30, 22)
        layout.setSpacing(8)

        def _lbl(txt, font_size=9, bold=False, color=C.PRI,
                 align=Qt.AlignmentFlag.AlignCenter):
            w = QLabel(txt)
            w.setAlignment(align)
            w.setFont(UIFont(font_size,
                            QFont.Weight.Bold if bold else QFont.Weight.Normal))
            w.setStyleSheet(f"color: {color}; background: transparent;")
            return w

        layout.addWidget(_lbl("◈  INITIALISATION REQUIRED", 13, True))
        layout.addWidget(_lbl("Configure J.A.R.V.I.S. before first boot.", 9, color=C.PRI_DIM))
        layout.addSpacing(6)

        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {C.BORDER};"); layout.addWidget(sep)
        layout.addSpacing(4)

        layout.addWidget(_lbl("GEMINI API KEY", 8, color=C.TEXT_DIM,
                               align=Qt.AlignmentFlag.AlignLeft))
        self._key_input = QLineEdit()
        self._key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_input.setPlaceholderText("AIza…")
        self._key_input.setFont(UIFont(10))
        self._key_input.setFixedHeight(32)
        self._key_input.setStyleSheet(f"""
            QLineEdit {{
                background: {C.INPUT}; color: {C.TEXT};
                border: 1px solid {C.BORDER}; border-radius: 6px; padding: 4px 8px;
            }}
            QLineEdit:focus {{ border: 1px solid {C.PRI}; }}
        """)
        layout.addWidget(self._key_input)
        layout.addSpacing(12)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color: {C.BORDER};"); layout.addWidget(sep2)
        layout.addSpacing(4)

        layout.addWidget(_lbl("OPERATING SYSTEM", 8, color=C.TEXT_DIM,
                               align=Qt.AlignmentFlag.AlignLeft))
        det_name = {"windows": "Windows", "mac": "macOS", "linux": "Linux"}[detected]
        layout.addWidget(_lbl(f"Auto-detected: {det_name}", 8, color=C.ACC2,
                               align=Qt.AlignmentFlag.AlignLeft))

        os_row = QHBoxLayout(); os_row.setSpacing(6)
        self._os_btns: dict[str, QPushButton] = {}
        for key, label in [("windows","⊞  Windows"),("mac","  macOS"),("linux","🐧  Linux")]:
            btn = QPushButton(label)
            btn.setFont(UIFont(9, QFont.Weight.Bold))
            btn.setFixedHeight(32)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _, k=key: self._sel(k))
            os_row.addWidget(btn)
            self._os_btns[key] = btn
        layout.addLayout(os_row)
        self._sel(detected)
        layout.addSpacing(12)

        init_btn = QPushButton("▸  INITIALISE SYSTEMS")
        init_btn.setFont(UIFont(10, QFont.Weight.Bold))
        init_btn.setFixedHeight(36)
        init_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        init_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {C.PRI};
                border: 1px solid {C.PRI_DIM}; border-radius: 6px;
            }}
            QPushButton:hover {{
                background: {C.PRI_GHO}; border: 1px solid {C.PRI};
            }}
        """)
        init_btn.clicked.connect(self._submit)
        layout.addWidget(init_btn)

    def _sel(self, key: str):
        self._sel_os = key
        pal = {"windows":(C.PRI,C.PRI_GHO),"mac":(C.PRI,C.PRI_GHO),"linux":(C.PRI,C.PRI_GHO)}
        for k, btn in self._os_btns.items():
            if k == key:
                fg, bg = pal[k]
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {fg}; color: {bg};
                        border: none; border-radius: 6px; font-weight: bold;
                    }}
                """)
            else:
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: {C.INPUT}; color: {C.TEXT_DIM};
                        border: 1px solid {C.BORDER}; border-radius: 6px;
                    }}
                    QPushButton:hover {{ color: {C.TEXT}; border: 1px solid {C.BORDER_B}; }}
                """)

    def _submit(self):
        key = self._key_input.text().strip()
        if not key:
            self._key_input.setStyleSheet(
                self._key_input.styleSheet() +
                f" QLineEdit {{ border: 1px solid {C.RED}; }}"
            )
            return
        self.done.emit(key, self._sel_os)


class HueWheel(QWidget):
    """
    Dairesel renk seçici. Kullanıcı tutamacı (küçük beyaz daire) çarkın
    çevresinde sürükleyerek TÜM renk tonları arasından seçim yapar.
    Merkezdeki dolu daire seçilen rengin canlı önizlemesidir.
    """

    hue_picked    = pyqtSignal(str)   # sürükleme sırasında (canlı)
    hue_committed = pyqtSignal(str)   # tutamaç bırakıldığında

    _RING = 16   # halka kalınlığı (px)

    def __init__(self, initial_hex: str = DEFAULT_UI_COLOR, parent=None):
        super().__init__(parent)
        self.setFixedSize(148, 148)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hue  = 0.53
        self._drag = False
        self.set_color(initial_hex)

    # ── API ──────────────────────────────────────────────────────────────────
    def color(self) -> str:
        return QColor.fromHsvF(self._hue, 1.0, 1.0).name()

    def set_color(self, hex_str: str):
        c = QColor((hex_str or "").strip())
        if c.isValid() and c.hsvHueF() >= 0:
            self._hue = c.hsvHueF()
            self.update()

    # ── geometri yardımcıları ────────────────────────────────────────────────
    def _ring_rect(self) -> QRectF:
        m = self._RING / 2 + 3
        return QRectF(self.rect()).adjusted(m, m, -m, -m)

    def _hue_from_pos(self, pos: QPointF) -> float:
        c  = QRectF(self.rect()).center()
        dx = pos.x() - c.x()
        dy = c.y() - pos.y()          # ekran y'si aşağı — matematiksel eksene çevir
        ang = math.atan2(dy, dx)      # [-π, π], saat yönünün tersi
        return (ang / (2 * math.pi)) % 1.0

    # ── çizim ────────────────────────────────────────────────────────────────
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect   = self._ring_rect()
        center = rect.center()

        grad = QConicalGradient(center, 0)
        for i in range(0, 361, 20):
            grad.setColorAt(i / 360.0, QColor.fromHsvF((i % 360) / 360.0, 1.0, 1.0))
        p.setPen(QPen(QBrush(grad), self._RING))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(rect)

        # merkez önizleme dairesi
        preview = QColor.fromHsvF(self._hue, 1.0, 1.0)
        inner   = rect.adjusted(30, 30, -30, -30)
        p.setPen(QPen(qcol(C.BORDER_B), 1))
        p.setBrush(QBrush(preview))
        p.drawEllipse(inner)

        # sürüklenen tutamaç
        r   = rect.width() / 2
        ang = self._hue * 2 * math.pi
        hx  = center.x() + r * math.cos(ang)
        hy  = center.y() - r * math.sin(ang)
        p.setPen(QPen(QColor(C.BG), 2))
        p.setBrush(QBrush(QColor(C.WHITE)))
        p.drawEllipse(QPointF(hx, hy), 7.5, 7.5)

    # ── fare ─────────────────────────────────────────────────────────────────
    def mousePressEvent(self, e):
        self._drag = True
        self._hue  = self._hue_from_pos(e.position())
        self.update()
        self.hue_picked.emit(self.color())

    def mouseMoveEvent(self, e):
        if self._drag:
            self._hue = self._hue_from_pos(e.position())
            self.update()
            self.hue_picked.emit(self.color())

    def mouseReleaseEvent(self, e):
        if self._drag:
            self._drag = False
            self.hue_committed.emit(self.color())


class ClipboardPanel(QWidget):
    """Floating panel shown when text is copied — offers quick Jarvis actions."""

    action_requested = pyqtSignal(str)
    _W, _H = 326, 112

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            ClipboardPanel {{
                background: {C.GLASS};
                border: 1px solid {C.BORDER_B};
                border-radius: 10px;
            }}
        """)
        self.setFixedWidth(self._W)
        self._clip_text = ""

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 7)
        lay.setSpacing(4)

        hdr = QHBoxLayout(); hdr.setSpacing(4)
        icon_lbl = QLabel("◈  CLIPBOARD DETECTED")
        icon_lbl.setFont(UIFont(8, QFont.Weight.Bold))
        icon_lbl.setStyleSheet(f"color: {C.ACC2}; background: transparent;")
        hdr.addWidget(icon_lbl); hdr.addStretch()
        x_btn = QPushButton("✕")
        x_btn.setFixedSize(16, 16)
        x_btn.setFont(UIFont(9))
        x_btn.setStyleSheet(f"color: {C.TEXT_DIM}; background: transparent; border: none;")
        x_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        x_btn.clicked.connect(self.hide)
        hdr.addWidget(x_btn)
        lay.addLayout(hdr)

        self._preview = QLabel()
        self._preview.setFont(UIFont(9))
        self._preview.setStyleSheet(f"""
            color: {C.TEXT}; background: {C.PANEL2};
            border: 1px solid {C.BORDER}; border-radius: 6px; padding: 4px 6px;
        """)
        self._preview.setWordWrap(False)
        self._preview.setFixedHeight(28)
        lay.addWidget(self._preview)

        btn_row = QHBoxLayout(); btn_row.setSpacing(4)
        _bs = (f"QPushButton {{ background: {C.PANEL2}; color: {C.TEXT_MED}; "
               f"border: 1px solid {C.BORDER}; border-radius: 6px; }}"
               f"QPushButton:hover {{ color: {C.PRI}; border-color: {C.BORDER_B}; }}")
        for label, cmd_fmt in [
            ("TRANSLATE", "Translate this text to English: {text}"),
            ("SUMMARISE", "Summarise this: {text}"),
            ("EXPLAIN",   "Explain this: {text}"),
            ("FIX",       "Fix grammar and spelling: {text}"),
        ]:
            b = QPushButton(label)
            b.setFixedHeight(22)
            b.setFont(UIFont(8, QFont.Weight.Bold))
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(_bs)
            b.clicked.connect(lambda _, c=cmd_fmt: self._trigger(c))
            btn_row.addWidget(b)
        lay.addLayout(btn_row)

        self._dismiss_timer = QTimer(self)
        self._dismiss_timer.setSingleShot(True)
        self._dismiss_timer.timeout.connect(self.hide)
        self.hide()

    def _trigger(self, cmd_fmt: str):
        if self._clip_text:
            self.action_requested.emit(cmd_fmt.format(text=self._clip_text[:800]))
        self.hide()

    def show_clipboard(self, text: str):
        self._clip_text = text
        preview = text[:58].replace('\n', ' ')
        if len(text) > 58:
            preview += "…"
        self._preview.setText(f'"{preview}"')
        self.show(); self.raise_()
        self._dismiss_timer.start(8000)


class _Scrim(QWidget):
    """Dims the window behind a modal card. Clicking it asks the card to close."""
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"background: {C.SCRIM};")
        self.hide()

    def mousePressEvent(self, e):
        self.clicked.emit()


class _ConfirmRequest:
    """One pending approval. The worker thread waits on `event`; the GUI thread answers."""

    def __init__(self, title: str, detail: str, timeout: int):
        self.title    = title
        self.detail   = detail
        self.timeout  = max(5, int(timeout))
        self.approved = False
        self.event    = threading.Event()

    def answer(self, approved: bool) -> None:
        if not self.event.is_set():
            self.approved = approved
            self.event.set()


class ConfirmPanel(QWidget):
    """Centered approval card for risky actions. Allow must be clicked; Esc or timeout denies.

    Presentation only differs from the original: an icon for the kind of action,
    details as labelled fields, larger buttons. The decision logic is unchanged.
    """

    answered = pyqtSignal(bool)
    _W, _H = 460, 320

    _ICONS = (("send", "✉"), ("message", "✉"), ("delete", "⌫"), ("shut", "⏻"), ("restart", "⏻"),
              ("run", "▶"), ("build", "▶"), ("code", "{ }"), ("file", "▤"), ("change", "✎"),
              ("write", "✎"), ("desktop", "▦"), ("schedule", "◷"), ("extract", "▤"))

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setObjectName("Approval")
        self.setStyleSheet(f"#Approval {{ background: rgba(34,37,42,250); border: 1px solid {C.WARN_LINE};"
                           f" border-radius: 14px; }}")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._remaining = 0
        self._open      = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 14)
        lay.setSpacing(12)

        head = QHBoxLayout(); head.setSpacing(12)
        self._icon = QLabel("!")
        self._icon.setFixedSize(34, 34)
        self._icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon.setFont(UIFont(14))
        self._icon.setStyleSheet(f"color: {C.ACC2}; background: {C.WARN_BG}; border: 1px solid {C.WARN_LINE};"
                                 f" border-radius: 9px;")
        head.addWidget(self._icon, 0, Qt.AlignmentFlag.AlignTop)
        tcol = QVBoxLayout(); tcol.setSpacing(2)
        tcol.addWidget(label("APPROVAL NEEDED", 9.5, C.ACC2, QFont.Weight.DemiBold, spacing=1.8))
        self._title = label("", 14.5, C.TEXT, QFont.Weight.Medium, wrap=True)
        tcol.addWidget(self._title)
        head.addLayout(tcol, 1)
        lay.addLayout(head)

        box = QFrame(); box.setObjectName("ApprDet")
        box.setStyleSheet(f"#ApprDet {{ background: {C.INPUT}; border: 1px solid {C.BORDER}; border-radius: 9px; }}")
        bl = QVBoxLayout(box); bl.setContentsMargins(12, 10, 12, 10); bl.setSpacing(4)
        self._detail = QLabel()
        self._detail.setFont(UIFont(12))
        self._detail.setWordWrap(True)
        self._detail.setTextFormat(Qt.TextFormat.RichText)
        self._detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._detail.setStyleSheet(f"color: {C.TEXT}; background: transparent; border: none;")
        self._detail.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.setWidget(self._detail)
        bl.addWidget(scroll)
        lay.addWidget(box, 1)

        row = QHBoxLayout(); row.setSpacing(10)
        self._deny_btn = button("Deny", "danger", 38)
        self._allow_btn = button("Allow", "primary", 38)
        for b in (self._deny_btn, self._allow_btn):
            b.setFont(UIFont(13, QFont.Weight.Medium))
            b.setAutoDefault(False)      # No default button: Enter must never approve by accident.
            b.setDefault(False)
        self._deny_btn.clicked.connect(lambda: self._finish(False))
        self._allow_btn.clicked.connect(lambda: self._finish(True))
        row.addWidget(self._deny_btn, 1)
        row.addWidget(self._allow_btn, 1)
        lay.addLayout(row)

        foot = QHBoxLayout()
        self._timer_lbl = label("", 10.5, C.TEXT_DIM)
        foot.addWidget(self._timer_lbl); foot.addStretch()
        foot.addWidget(label("Esc denies · Enter does nothing", 10.5, C.TEXT_DIM))
        lay.addLayout(foot)

        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._on_tick)
        self.hide()

    @staticmethod
    def _detail_html(detail: str) -> str:
        import html
        rows = []
        for line in (detail or "(no details)").split("\n"):
            if ":" in line and len(line.split(":", 1)[0]) <= 18 and not line.startswith(" "):
                k, v = line.split(":", 1)
                rows.append(f"<span style='color:{C.TEXT_DIM}'>{html.escape(k.strip())}</span>&nbsp;&nbsp;"
                            f"{html.escape(v.strip())}")
            else:
                rows.append(html.escape(line))
        return "<br>".join(rows)

    def show_request(self, title: str, detail: str, timeout: int) -> None:
        self._open = True
        self._title.setText(title)
        low = title.lower()
        self._icon.setText(next((g for k, g in self._ICONS if k in low), "!"))
        self._detail.setText(self._detail_html(detail))
        self._remaining = timeout
        self._update_timer_label()
        self._tick.start()
        fade_in(self)
        self._deny_btn.setFocus()

    def _update_timer_label(self) -> None:
        self._timer_lbl.setText(f"Denied automatically in {self._remaining} s")

    def _on_tick(self) -> None:
        self._remaining -= 1
        if self._remaining <= 0:
            self._finish(False)
        else:
            self._update_timer_label()

    def _finish(self, approved: bool) -> None:
        # Track our own open state: isVisible() is False whenever the main
        # window is hidden, and the request must still be answered then.
        if not self._open:
            return
        self._open = False
        self._tick.stop()
        self.hide()
        self.answered.emit(approved)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self._finish(False)
            return
        super().keyPressEvent(event)


class RemoteKeyOverlay(QWidget):
    """Floating overlay — QR code for instant phone pairing + manual key fallback."""

    closed = pyqtSignal()

    _OW, _OH = 400, 465

    def __init__(self, url: str, key: str, auto_login_url: str = "",
                 manual_url: str = "", expiry_secs: int = 600, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(f"""
            RemoteKeyOverlay {{
                background: {C.GLASS};
                border: 1px solid {C.BORDER_B};
                border-radius: 14px;
            }}
        """)
        self._expiry          = time.time() + expiry_secs
        self._on_new_key      = None
        self._auto_login_url  = auto_login_url
        self._manual_url      = manual_url or url

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 16, 24, 16)
        lay.setSpacing(5)

        def _lbl(txt, fs=9, bold=False, color=C.PRI,
                 align=Qt.AlignmentFlag.AlignCenter):
            w = QLabel(txt)
            w.setAlignment(align)
            w.setFont(UIFont(fs,
                            QFont.Weight.Bold if bold else QFont.Weight.Normal))
            w.setStyleSheet(f"color: {color}; background: transparent;")
            w.setWordWrap(True)
            return w

        lay.addWidget(_lbl("◈  REMOTE ACCESS", 12, True))
        sep = QFrame(); sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {C.BORDER}; margin: 1px 0;")
        lay.addWidget(sep)

        # ── QR code ───────────────────────────────────────────────────────────
        self._qr_label = QLabel()
        self._qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._qr_label.setFixedSize(176, 176)
        self._qr_label.setStyleSheet(
            "background: white; border-radius: 10px; padding: 4px;"
        )
        qr_row = QHBoxLayout()
        qr_row.addStretch()
        qr_row.addWidget(self._qr_label)
        qr_row.addStretch()
        lay.addLayout(qr_row)

        self._update_qr(auto_login_url)

        lay.addWidget(_lbl("Scan with phone camera to connect instantly", 8, color=C.TEXT_DIM))

        sep2 = QFrame(); sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet(f"color: {C.BORDER}; margin: 1px 0;")
        lay.addWidget(sep2)

        lay.addWidget(_lbl("Or enter manually:", 7, color=C.TEXT_DIM,
                           align=Qt.AlignmentFlag.AlignLeft))

        self._url_lbl = QLabel(self._manual_url)
        self._url_lbl.setFont(UIFont(9))
        self._url_lbl.setStyleSheet(f"color: {C.PRI_DIM}; background: transparent;")
        self._url_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._url_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self._url_lbl)

        self._key_lbl = QLabel(key)
        self._key_lbl.setFont(UIFont(28, QFont.Weight.Bold))
        self._key_lbl.setStyleSheet(f"""
            color: {C.ACC};
            background: {C.PANEL2};
            border: 1px solid {C.BORDER_B};
            border-radius: 8px;
            padding: 6px 4px;
            letter-spacing: 10px;
        """)
        self._key_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._key_lbl)

        self._timer_lbl = QLabel()
        self._timer_lbl.setFont(UIFont(9))
        self._timer_lbl.setStyleSheet(f"color: {C.TEXT_MED}; background: transparent;")
        self._timer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._timer_lbl)

        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        new_btn = QPushButton("NEW KEY")
        new_btn.setFixedHeight(32)
        new_btn.setFont(UIFont(9, QFont.Weight.Bold))
        new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        new_btn.setStyleSheet(f"""
            QPushButton {{
                background: {C.PANEL}; color: {C.PRI};
                border: 1px solid {C.PRI_DIM}; border-radius: 5px;
            }}
            QPushButton:hover {{ background: {C.PRI_GHO}; border: 1px solid {C.PRI}; }}
        """)
        new_btn.clicked.connect(self._refresh_key)
        btn_row.addWidget(new_btn)

        close_btn = QPushButton("DISMISS")
        close_btn.setFixedHeight(32)
        close_btn.setFont(UIFont(9, QFont.Weight.Bold))
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent; color: {C.TEXT_MED};
                border: 1px solid {C.BORDER}; border-radius: 5px;
            }}
            QPushButton:hover {{ color: {C.TEXT}; border: 1px solid {C.BORDER_B}; }}
        """)
        close_btn.clicked.connect(self._do_close)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

        self._ctimer = QTimer(self)
        self._ctimer.timeout.connect(self._tick)
        self._ctimer.start(1000)
        self._tick()

    def set_new_key_callback(self, fn) -> None:
        self._on_new_key = fn

    def _update_qr(self, url: str) -> None:
        if not url:
            self._qr_label.setText("—")
            return
        try:
            import qrcode as _qrmod
            from io import BytesIO
            qr = _qrmod.QRCode(
                box_size=5, border=2,
                error_correction=_qrmod.constants.ERROR_CORRECT_M,
            )
            qr.add_data(url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            buf = BytesIO()
            img.save(buf, format="PNG")
            px = QPixmap()
            px.loadFromData(buf.getvalue())
            self._qr_label.setPixmap(
                px.scaled(170, 170,
                          Qt.AspectRatioMode.KeepAspectRatio,
                          Qt.TransformationMode.SmoothTransformation)
            )
        except ImportError:
            self._qr_label.setText("pip install\nqrcode[pil]")
            self._qr_label.setFont(UIFont(9))
            self._qr_label.setStyleSheet(
                "color: #888; background: white; border-radius: 10px; padding: 4px;"
            )
        except Exception:
            self._qr_label.setText(url[:28])
            self._qr_label.setFont(UIFont(8))
            self._qr_label.setStyleSheet(
                f"color: {C.PRI}; background: white; border-radius: 10px; padding: 4px;"
            )

    def _tick(self):
        remaining = max(0, int(self._expiry - time.time()))
        m, s = divmod(remaining, 60)
        self._timer_lbl.setText(f"Key expires in  {m:02d}:{s:02d}")
        if remaining == 0:
            self._do_close()

    def mark_connected(self) -> None:
        """Call from any thread when a phone successfully connects."""
        self._ctimer.stop()
        self._key_lbl.setText("CONNECTED")
        self._key_lbl.setStyleSheet(f"""
            color: {C.GREEN};
            background: {C.OK_BG};
            border: 2px solid {C.GREEN_D};
            border-radius: 8px;
            padding: 6px 4px;
            letter-spacing: 4px;
        """)
        self._qr_label.setText("✓")
        self._qr_label.setFont(UIFont(54, QFont.Weight.Bold))
        self._qr_label.setStyleSheet(
            f"color: {C.GREEN}; background: {C.OK_BG}; border-radius: 10px;"
        )
        self._timer_lbl.setText("Phone connected — JARVIS ready")
        self._timer_lbl.setStyleSheet(f"color: {C.GREEN}; background: transparent;")

    def _refresh_key(self):
        if self._on_new_key:
            result = self._on_new_key()
            if result:
                url    = result[0]
                key    = result[1]
                auto   = result[2] if len(result) >= 3 else ""
                manual = result[3] if len(result) >= 4 else url
                self._manual_url     = manual or url
                self._url_lbl.setText(self._manual_url)
                self._key_lbl.setText(key)
                self._auto_login_url = auto
                self._update_qr(auto or url)
                self._expiry = time.time() + 600
                self._key_lbl.setStyleSheet(f"""
                    color: {C.ACC};
                    background: {C.PANEL2};
                    border: 1px solid {C.BORDER_B};
                    border-radius: 8px;
                    padding: 6px 4px;
                    letter-spacing: 10px;
                """)
                self._timer_lbl.setStyleSheet(
                    f"color: {C.TEXT_MED}; background: transparent;"
                )
                self._ctimer.start(1000)
                self._tick()

    def _do_close(self):
        self._ctimer.stop()
        self.hide()
        self.closed.emit()


class MainWindow(QMainWindow):
    _log_sig        = pyqtSignal(str)
    _state_sig      = pyqtSignal(str)
    _content_sig    = pyqtSignal(str, str)   # (title, text) — thread-safe content display
    _reconfig_sig   = pyqtSignal()           # trigger setup overlay from any thread
    _camera_sig     = pyqtSignal(bytes)      # show camera frame preview (small overlay)
    _cam_stream_sig = pyqtSignal(bool)       # True=start live stream, False=stop
    _cam_frame_sig  = pyqtSignal(bytes)      # live camera frame → HUD area
    _clipboard_sig  = pyqtSignal(str)        # clipboard text changed (thread-safe)
    _confirm_sig    = pyqtSignal(object)     # _ConfirmRequest from a worker thread
    _phone_sig      = pyqtSignal()           # phone paired (from the dashboard thread)
    _msg_sig        = pyqtSignal(str, str, str)       # role, text, source
    _tool_sig       = pyqtSignal(str, str, str, str)  # call id, tool, status, detail
    _conn_sig       = pyqtSignal(str, str)            # state, detail
    _speaker_sig    = pyqtSignal(str, object)         # verdict label, score
    _toast_sig      = pyqtSignal(str, str, str)       # title, body, tone
    _caption_sig    = pyqtSignal(str)                 # reply text while speaking
    _hearing_sig    = pyqtSignal(str)                 # live transcript of the user
    _gate_sig       = pyqtSignal(str, str, object)    # voice filter: action, label, score
    _remote_sig     = pyqtSignal(bool, int)           # dashboard running, phones

    def __init__(self, face_path: str):
        super().__init__()
        self._face_path = face_path

        # Load customization from config
        _cfg = _read_full_config()
        self._assistant_name: str = (_cfg.get("assistant_name") or "JARVIS").strip()
        _display = self._assistant_name.upper()

        # Kayıtlı UI rengini panel/stylesheet'ler kurulmadan ÖNCE uygula
        _ui_color = (_cfg.get("ui_color") or "").strip()
        if _ui_color and _ui_color.lower() != DEFAULT_UI_COLOR:
            apply_ui_accent(_ui_color)

        self.setWindowTitle(f"{_display} — MARK XLIX")
        self.setMinimumSize(_MIN_W, _MIN_H)
        self.resize(_DEFAULT_W, _DEFAULT_H)

        screen = QApplication.primaryScreen().availableGeometry()
        self.move(
            (screen.width()  - _DEFAULT_W) // 2,
            (screen.height() - _DEFAULT_H) // 2,
        )

        self.on_text_command   = None
        self.on_remote_clicked = None   # callable: () -> (url, key) | None
        self.audio = None               # AudioEngine (set by JarvisUI.attach_audio)
        self.on_interrupt      = None   # callable: () -> None — stop JARVIS mid-speech
        self._muted            = False
        self._current_file: str | None = None
        self._remote_overlay: RemoteKeyOverlay | None = None

        self.speaker_id = None                 # SpeakerIdentity (set by JarvisUI.attach_speaker_id)
        self._show_speaker = bool(_cfg.get("voice_identity_display", True))
        self._settings = None

        central = QWidget()
        central.setStyleSheet(f"background: {C.BG};")
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_titlebar())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Stage: what JARVIS is doing (voice ribbon, title, caption, mic dock)
        self.stage = VoiceStage(self._assistant_name)
        self.stage.mute_toggled.connect(self._set_muted)
        self.stage.interrupt_requested.connect(self._do_interrupt)
        self.stage.error_action.connect(lambda: self.open_settings("Audio"))

        # Live camera container — replaces the stage while the camera stream is active
        _cam_cont = QWidget()
        _cam_cont.setStyleSheet(f"background: {C.BG};")
        _cam_v = QVBoxLayout(_cam_cont)
        _cam_v.setContentsMargins(0, 0, 0, 0)
        _cam_v.setSpacing(0)
        _cam_hdr = QHBoxLayout()
        _cam_hdr.setContentsMargins(14, 8, 14, 8)
        _cam_hdr.addWidget(caps("Camera"))
        _cam_hdr.addStretch()
        _cam_x = button("Close camera", "ghost", 26)
        _cam_x.clicked.connect(self.stop_camera_stream)
        _cam_hdr.addWidget(_cam_x)
        _cam_v.addLayout(_cam_hdr)
        self._cam_live_lbl = QLabel()
        self._cam_live_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_live_lbl.setStyleSheet("background: transparent;")
        self._cam_live_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        _cam_v.addWidget(self._cam_live_lbl, stretch=1)

        # Stack: 0 = stage, 1 = live camera
        self._hud_cam_stack = QStackedWidget()
        self._hud_cam_stack.addWidget(self.stage)
        self._hud_cam_stack.addWidget(_cam_cont)
        body.addWidget(self._hud_cam_stack, stretch=1)

        # Conversation sidebar (messages, tool activity, results, files, composer)
        self.convo = ConversationPanel(self._assistant_name)
        self.convo.setFixedWidth(_CONVO_W)
        self.convo.send_text.connect(self._send_text)
        self._drop_zone = FileDropZone()
        self._drop_zone.file_selected.connect(self._on_file_selected)
        self.convo.set_files_widget(self._drop_zone)
        self.convo.attach_requested.connect(self._attach_file)
        self._file_hint = label("", 11, C.TEXT_MED, wrap=True)
        self.convo.set_files_widget(self._file_hint)
        body.addWidget(self.convo, stretch=0)
        self._convo_forced = False             # user opened the sidebar in a narrow window

        root.addLayout(body, stretch=1)
        root.addWidget(self._build_statusbar())

        self.toasts = ToastHost(central)
        self.toasts.top_margin = 58
        self.toasts.right_margin = _CONVO_W + 16
        self._settings: "SettingsSheet | None" = None
        self.on_remote_status = None           # () -> dict, set by main
        self.on_forget_phones = None           # () -> int
        self.on_stop_remote = None             # () -> None
        self._last_err_toast = 0.0

        self._clock_tmr = QTimer(self)
        self._clock_tmr.timeout.connect(self._tick_clock)
        self._clock_tmr.start(1000)
        self._tick_clock()

        self._metric_tmr = QTimer(self)
        self._metric_tmr.timeout.connect(self._update_metrics)
        self._metric_tmr.start(2000)
        self._update_metrics()

        self._log_sig.connect(self._route_log)
        self._state_sig.connect(self._apply_state)
        self._content_sig.connect(self._show_content)
        self._reconfig_sig.connect(self._show_setup)
        self._camera_sig.connect(self._show_camera_frame)
        self._cam_stream_sig.connect(self._on_cam_stream)
        self._cam_frame_sig.connect(self._on_cam_frame)
        self._clipboard_sig.connect(self._show_clipboard_panel)
        self._msg_sig.connect(self._on_message)
        self._tool_sig.connect(self._on_tool_event)
        self._conn_sig.connect(self.set_connection)
        self._speaker_sig.connect(self._on_speaker)
        self._toast_sig.connect(lambda t, b, tone: self.toasts.show_toast(t, b, tone))
        self._caption_sig.connect(self.stage.set_speaking_text)
        self._hearing_sig.connect(self.stage.set_hearing_text)
        self._gate_sig.connect(self._on_gate)
        self._cam_stop = threading.Event()

        # Camera preview overlay (child of central widget, positioned in resizeEvent)
        self._cam_preview = _CameraPreview(self.centralWidget())

        # Clipboard panel (child of central widget, bottom-center)
        self._clipboard_panel = ClipboardPanel(self.centralWidget())
        self._clipboard_panel.action_requested.connect(self._on_clipboard_action)
        QApplication.clipboard().dataChanged.connect(self._on_clipboard_changed)

        # Approval card for risky tool actions (child of central widget, centered)
        self._scrim = _Scrim(self.centralWidget())
        self._scrim.clicked.connect(self._scrim_clicked)
        self._confirm_panel = ConfirmPanel(self.centralWidget())
        self._confirm_panel.answered.connect(self._on_confirm_answered)
        self._confirm_queue: list[_ConfirmRequest] = []
        self._confirm_active: _ConfirmRequest | None = None
        self._confirm_sig.connect(self._enqueue_confirm)
        self._phone_sig.connect(self.notify_phone_connected)

        self._overlay: SetupOverlay | None = None
        self._ready = self._check_config()
        if not self._ready:
            self._show_setup()

        sc_mute = QShortcut(QKeySequence("F4"), self)
        sc_mute.activated.connect(self._toggle_mute)
        sc_full = QShortcut(QKeySequence("F11"), self)
        sc_full.activated.connect(self._toggle_fullscreen)
        sc_intr = QShortcut(QKeySequence("Escape"), self)
        sc_intr.activated.connect(self._on_escape)
        sc_pref = QShortcut(QKeySequence(QKeySequence.StandardKey.Preferences), self)
        sc_pref.activated.connect(lambda: self.open_settings())
        self._remote_sig.connect(self.set_remote_chip)
        self._apply_responsive()

    def _show_camera_frame(self, img_bytes: bytes):
        """Slot — display camera preview overlay (main thread)."""
        self._cam_preview.show_frame(img_bytes)
        cw = self.centralWidget()
        pw = _CameraPreview._W
        ph = self._cam_preview.height()
        side = self.convo.width() if self.convo.isVisible() else 0
        self._cam_preview.setGeometry(cw.width() - side - pw - 12, cw.height() - ph - 40, pw, ph)

    # --- Live camera stream in HUD area ------------------------------------
    def _on_cam_stream(self, start: bool) -> None:
        if start:
            self._hud_cam_stack.setCurrentIndex(1)
        else:
            self._hud_cam_stack.setCurrentIndex(0)
            self._cam_live_lbl.clear()

    def _on_cam_frame(self, data: bytes) -> None:
        px = QPixmap()
        px.loadFromData(data)
        if not px.isNull():
            w, h = self._cam_live_lbl.width(), self._cam_live_lbl.height()
            if w > 1 and h > 1:
                self._cam_live_lbl.setPixmap(
                    px.scaled(w, h,
                              Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
                )

    def start_camera_stream(self) -> None:
        self._cam_stop.clear()
        self._cam_stream_sig.emit(True)
        t = threading.Thread(target=self._cam_loop, daemon=True, name="cam-stream")
        t.start()

    def _cam_loop(self) -> None:
        try:
            import cv2
            # Reuse camera index detected by screen_processor (cached in api_keys.json)
            cam_idx = 0
            try:
                import json as _j
                cfg = _j.loads((CONFIG_DIR / "api_keys.json").read_text())
                cam_idx = int(cfg.get("camera_index", 0))
            except Exception:
                pass
            try:
                backend = cv2.CAP_DSHOW if _OS == "Windows" else cv2.CAP_ANY
            except AttributeError:
                backend = 0
            cap = cv2.VideoCapture(cam_idx, backend)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return
            # warm-up frames
            for _ in range(5):
                cap.read()
            while not self._cam_stop.wait(0.033) and cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                    self._cam_frame_sig.emit(buf.tobytes())
            cap.release()
        except Exception as e:
            print(f"[Camera] Stream error: {e}")
        finally:
            self._cam_stream_sig.emit(False)

    def stop_camera_stream(self) -> None:
        self._cam_stop.set()

    # ------------------------------------------------------------------
    # Icon generation — arc-reactor style, rendered with Pillow
    # ------------------------------------------------------------------
    @staticmethod
    def _build_jarvis_icon(out_path: Path) -> bool:
        """
        Render a JARVIS arc-reactor icon at 4× resolution and downsample
        for crisp results at all sizes. Saves a multi-res .ico to out_path.
        Returns True on success.
        """
        try:
            import math
            import PIL.Image
            import PIL.ImageDraw
            import PIL.ImageFilter
        except ImportError:
            return False

        CYAN   = (0, 212, 255)
        DIM    = (0, 100, 140)
        DARK   = (0, 6, 10)
        GLOW   = (0, 160, 200)
        WHITE  = (220, 240, 255)

        def _render(sz: int) -> PIL.Image.Image:
            S  = sz * 4                     # draw at 4× then downscale
            img = PIL.Image.new("RGBA", (S, S), (0, 0, 0, 0))
            d   = PIL.ImageDraw.Draw(img)
            cx = cy = S // 2

            # ── filled background circle ──────────────────────────────────
            R = S // 2 - 2
            d.ellipse([cx-R, cy-R, cx+R, cy+R], fill=(*DARK, 255))

            # ── outer border ring ─────────────────────────────────────────
            lw = max(2, S // 40)
            d.ellipse([cx-R, cy-R, cx+R, cy+R],
                      outline=(*CYAN, 220), width=lw)

            # ── mid decorative ring ───────────────────────────────────────
            R2 = int(R * 0.72)
            d.ellipse([cx-R2, cy-R2, cx+R2, cy+R2],
                      outline=(*DIM, 180), width=max(1, lw // 2))

            # ── 6 radial spokes (hex bolt) ────────────────────────────────
            R_inner = int(R * 0.30)
            R_outer = int(R * 0.62)
            spoke_w = max(1, S // 80)
            for i in range(6):
                angle = math.radians(i * 60 - 30)
                x1 = cx + int(R_inner * math.cos(angle))
                y1 = cy + int(R_inner * math.sin(angle))
                x2 = cx + int(R_outer * math.cos(angle))
                y2 = cy + int(R_outer * math.sin(angle))
                d.line([x1, y1, x2, y2], fill=(*GLOW, 200), width=spoke_w)

            # ── 6 tick marks on outer ring ────────────────────────────────
            for i in range(6):
                angle = math.radians(i * 60)
                for dr in range(lw * 2):
                    rx = (R - lw - dr)
                    d.point(
                        [cx + int(rx * math.cos(angle)),
                         cy + int(rx * math.sin(angle))],
                        fill=(*WHITE, 220),
                    )

            # ── inner glowing ring ────────────────────────────────────────
            Ri = int(R * 0.26)
            d.ellipse([cx-Ri, cy-Ri, cx+Ri, cy+Ri],
                      outline=(*CYAN, 255), width=max(2, lw))

            # ── bright glow soft blur applied before core ─────────────────
            # (draw a slightly larger cyan circle on a separate layer)
            glow_layer = PIL.Image.new("RGBA", (S, S), (0, 0, 0, 0))
            gd = PIL.ImageDraw.Draw(glow_layer)
            Rc = int(R * 0.13)
            gd.ellipse([cx-Rc*2, cy-Rc*2, cx+Rc*2, cy+Rc*2],
                       fill=(*CYAN, 110))
            glow_layer = glow_layer.filter(PIL.ImageFilter.GaussianBlur(S // 14))
            img = PIL.Image.alpha_composite(img, glow_layer)
            d   = PIL.ImageDraw.Draw(img)

            # ── core dot ──────────────────────────────────────────────────
            d.ellipse([cx-Rc, cy-Rc, cx+Rc, cy+Rc], fill=(*WHITE, 255))

            # ── downscale to target size ──────────────────────────────────
            return img.resize((sz, sz), PIL.Image.LANCZOS)

        try:
            sizes  = [256, 128, 64, 48, 32, 16]
            frames = [_render(s) for s in sizes]
            frames[0].save(
                out_path,
                format="ICO",
                append_images=frames[1:],
                sizes=[(s, s) for s in sizes],
            )
            return True
        except Exception as e:
            print(f"[Shortcut] ⚠️  Icon generation failed: {e}")
            return False

    @staticmethod
    def _get_desktop_dir() -> Path:
        """
        Resolve the user's REAL desktop directory instead of assuming
        ~/Desktop, which breaks when:
          • OneDrive "Known Folder Move" relocates the desktop
            (C:/Users/x/OneDrive/Desktop) — very common on Win 10/11;
          • the XDG desktop is localized on Linux (~/Masaüstü,
            ~/Schreibtisch, ~/Bureau, …).
        Falls back to ~/Desktop only as a last resort.
        """
        home = Path.home()
        _os = platform.system()

        if _os == "Windows":
            # ── 1) SHGetKnownFolderPath(FOLDERID_Desktop) — the canonical
            #       answer; follows OneDrive redirection. No dependencies. ──
            try:
                import ctypes
                from ctypes import wintypes

                class _GUID(ctypes.Structure):
                    _fields_ = [("Data1", wintypes.DWORD),
                                ("Data2", wintypes.WORD),
                                ("Data3", wintypes.WORD),
                                ("Data4", ctypes.c_ubyte * 8)]

                # FOLDERID_Desktop {B4BFCC3A-DB2C-424C-B029-7FE99A87C641}
                fid = _GUID(0xB4BFCC3A, 0xDB2C, 0x424C,
                            (ctypes.c_ubyte * 8)(0xB0, 0x29, 0x7F, 0xE9,
                                                 0x9A, 0x87, 0xC6, 0x41))
                buf = ctypes.c_wchar_p()
                if ctypes.windll.shell32.SHGetKnownFolderPath(
                        ctypes.byref(fid), 0, None, ctypes.byref(buf)) == 0:
                    p = Path(buf.value)
                    ctypes.windll.ole32.CoTaskMemFree(buf)
                    if p.is_dir():
                        return p
            except Exception:
                pass

            # ── 2) Registry: User Shell Folders (may contain %VARS%) ──────
            try:
                import winreg
                with winreg.OpenKey(
                        winreg.HKEY_CURRENT_USER,
                        r"Software\Microsoft\Windows\CurrentVersion"
                        r"\Explorer\User Shell Folders") as key:
                    val, _t = winreg.QueryValueEx(key, "Desktop")
                p = Path(os.path.expandvars(val))
                if p.is_dir():
                    return p
            except Exception:
                pass

        elif _os == "Linux":
            # ── xdg-user-dir honours localized names (~/Masaüstü, …) ──────
            try:
                out = subprocess.run(["xdg-user-dir", "DESKTOP"],
                                     capture_output=True, text=True, timeout=5)
                p = Path(out.stdout.strip())
                if out.stdout.strip() and p != home and p.is_dir():
                    return p
            except Exception:
                pass
            try:
                cfg = home / ".config" / "user-dirs.dirs"
                for line in cfg.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("XDG_DESKTOP_DIR"):
                        val = line.split("=", 1)[1].strip().strip('"')
                        p = Path(val.replace("$HOME", str(home)))
                        if p != home and p.is_dir():
                            return p
            except Exception:
                pass

        # macOS: ~/Desktop is always the real path (localization is
        # display-only). Everything else lands here as a last resort.
        return home / "Desktop"

    @staticmethod
    def _create_lnk_windows(lnk: str, target: str, args: str,
                             work_dir: str, icon_loc: str) -> None:
        """
        Create a Windows .lnk shortcut WITHOUT launching PowerShell or cmd.
        Tries win32com (pywin32) first; falls back to wscript.exe + VBScript.
        wscript.exe is a GUI-mode host — it never opens a console window.
        Raises on failure so the caller can log a useful error.
        """
        # ── Option 1: pywin32 (pure Python COM, zero subprocess) ──────────
        com_err: Exception | None = None
        try:
            from win32com.client import Dispatch   # type: ignore
            sh = Dispatch("WScript.Shell")
            sc = sh.CreateShortCut(lnk)
            sc.TargetPath       = target
            sc.Arguments        = f'"{args}"'
            sc.WorkingDirectory = work_dir
            sc.Description      = "J.A.R.V.I.S AI Assistant"
            sc.IconLocation     = icon_loc
            sc.save()
            return
        except ImportError:
            pass
        except Exception as e:            # COM error — still try VBScript
            com_err = e

        # ── Option 2: wscript.exe + VBScript (always available on Windows,
        #    GUI-mode executable — never opens a console window) ────────────
        def q(s: str) -> str:              # escape for a VBScript string literal
            return s.replace('"', '""')

        vbs = "\n".join([
            'On Error Resume Next',
            'Set ws = CreateObject("WScript.Shell")',
            f'Set sc = ws.CreateShortcut("{q(lnk)}")',
            f'sc.TargetPath = "{q(target)}"',
            f'sc.Arguments = Chr(34) & "{q(args)}" & Chr(34)',
            f'sc.WorkingDirectory = "{q(work_dir)}"',
            'sc.Description = "J.A.R.V.I.S AI Assistant"',
            f'sc.IconLocation = "{q(icon_loc)}"',
            'sc.Save',
            'If Err.Number <> 0 Then WScript.Quit 1',
        ])
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".vbs")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(vbs)
            proc = subprocess.Popen(
                ["wscript.exe", "/nologo", tmp],
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW,
            )
            proc.wait(timeout=10)
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

        if not Path(lnk).exists():
            raise RuntimeError(
                f"could not create '{lnk}'"
                + (f" ({com_err})" if com_err else "")
            )

    def _create_desktop_shortcut(self):
        """
        Create a desktop shortcut on Windows / macOS / Linux.
        Never opens a terminal, console, or PowerShell window on any platform.
        """
        import stat as _stat
        script  = Path(__file__).resolve().parent / "main.py"
        python  = Path(sys.executable)
        desktop = self._get_desktop_dir()

        # Arc-reactor icon (.ico — also exported as .png for Linux/macOS)
        ico_path = Path(__file__).resolve().parent / "config" / "jarvis.ico"
        if not ico_path.exists():
            self._build_jarvis_icon(ico_path)

        try:
            _os = platform.system()
            desktop.mkdir(parents=True, exist_ok=True)

            # ── Windows ───────────────────────────────────────────────────────
            if _os == "Windows":
                pythonw  = python.parent / "pythonw.exe"
                target   = str(pythonw if pythonw.exists() else python)
                lnk      = str(desktop / "J.A.R.V.I.S.lnk")
                icon_loc = str(ico_path) if ico_path.exists() else f"{target},0"
                self._create_lnk_windows(lnk, target, str(script),
                                         str(script.parent), icon_loc)

            # ── macOS — proper .app bundle (no Terminal window) ───────────────
            elif _os == "Darwin":
                app     = desktop / "J.A.R.V.I.S.app"
                mac_dir = app / "Contents" / "MacOS"
                res_dir = app / "Contents" / "Resources"
                mac_dir.mkdir(parents=True, exist_ok=True)
                res_dir.mkdir(exist_ok=True)

                # Launcher executable (bash — runs as background process,
                # macOS does NOT open Terminal for executables inside .app bundles)
                launcher = mac_dir / "JARVIS"
                launcher.write_text(
                    "#!/usr/bin/env bash\n"
                    f'cd "{script.parent}"\n'
                    f'exec "{python}" "{script}"\n'
                )
                launcher.chmod(launcher.stat().st_mode
                               | _stat.S_IEXEC | _stat.S_IXGRP | _stat.S_IXOTH)

                # Minimal Info.plist (required for .app recognition)
                (app / "Contents" / "Info.plist").write_text(
                    '<?xml version="1.0" encoding="UTF-8"?>\n'
                    '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                    '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
                    '<plist version="1.0"><dict>\n'
                    '  <key>CFBundleExecutable</key><string>JARVIS</string>\n'
                    '  <key>CFBundleIdentifier</key>'
                    '<string>com.jarvis.assistant</string>\n'
                    '  <key>CFBundleName</key><string>J.A.R.V.I.S</string>\n'
                    '  <key>CFBundlePackageType</key><string>APPL</string>\n'
                    '  <key>CFBundleVersion</key><string>1.0</string>\n'
                    '</dict></plist>\n'
                )

                # Optional: copy icon as .icns (skip silently if Pillow is missing)
                try:
                    import PIL.Image
                    icns = res_dir / "AppIcon.icns"
                    PIL.Image.open(ico_path).save(icns, format="ICNS")
                    # Inject icon reference into plist
                    plist = app / "Contents" / "Info.plist"
                    txt = plist.read_text()
                    plist.write_text(
                        txt.replace(
                            '</dict></plist>',
                            '  <key>CFBundleIconFile</key>'
                            '<string>AppIcon</string>\n</dict></plist>\n',
                        )
                    )
                except Exception:
                    pass  # icon is optional

            # ── Linux — .desktop file (Terminal=false, no console) ────────────
            else:
                # Export .ico → .png for better desktop integration
                png_path = ico_path.with_suffix(".png")
                if not png_path.exists() and ico_path.exists():
                    try:
                        import PIL.Image
                        PIL.Image.open(ico_path).resize(
                            (256, 256), PIL.Image.LANCZOS
                        ).save(png_path, format="PNG")
                    except Exception:
                        png_path = ico_path  # fallback to .ico

                icon_line = f"Icon={png_path}\n" if png_path.exists() else ""
                desk = desktop / "J.A.R.V.I.S.desktop"
                desk.write_text(
                    "[Desktop Entry]\n"
                    "Name=J.A.R.V.I.S\n"
                    f'Exec="{python}" "{script}"\n'
                    f"Path={script.parent}\n"
                    "Type=Application\n"
                    "Terminal=false\n"
                    "Categories=Utility;\n"
                    + icon_line
                )
                desk.chmod(desk.stat().st_mode | 0o755)
                # GNOME refuses to launch desktop files until they are
                # marked trusted ("Allow Launching") — do it automatically.
                try:
                    subprocess.run(
                        ["gio", "set", str(desk),
                         "metadata::trusted", "true"],
                        capture_output=True, timeout=5,
                    )
                except Exception:
                    pass  # non-GNOME desktops don't need (or have) gio

            self._route_log(f"SYS: Desktop shortcut created in '{desktop}'.")
        except Exception as e:
            self._route_log(
                f"ERR: Shortcut failed — {e} (desktop dir: '{desktop}')"
            )

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cw = self.centralWidget()
        self._apply_responsive()
        for ov, (ow, oh) in ((self._overlay, (460, 390)),
                             (self._remote_overlay, (RemoteKeyOverlay._OW, RemoteKeyOverlay._OH))):
            if ov and ov.isVisible():
                ov.setGeometry((cw.width() - ow) // 2, (cw.height() - oh) // 2, ow, oh)
        side = self.convo.width() if self.convo.isVisible() else 0
        pw = _CameraPreview._W
        ph = self._cam_preview.height() or _CameraPreview._H
        self._cam_preview.setGeometry(cw.width() - side - pw - 12, cw.height() - ph - 40, pw, ph)
        if self._clipboard_panel.isVisible():
            self._position_clipboard_panel()
        if self._scrim.isVisible():
            self._scrim.setGeometry(0, 0, cw.width(), cw.height())
        if self._settings is not None:
            self._position_settings()
        if self._confirm_panel.isVisible():
            self._position_confirm_panel()


    def _on_file_selected(self, path: str):
        self._current_file = path
        p    = Path(path)
        cat  = _file_category(p)
        icon, _ = _FILE_ICONS.get(cat, _FILE_ICONS["unknown"])
        size = _fmt_size(p.stat().st_size)
        self._file_hint.setText(f"{icon}  {p.name}  ·  {size}  ·  Tell {self._assistant_name} what to do with it")
        self._route_log(f"FILE: {p.name} ({size}) loaded")
        if self.on_text_command:
            msg = (
                f"[FILE_UPLOADED] path={path} | name={p.name} | "
                f"type={p.suffix.lstrip('.')} | size={size} | "
                f"Briefly tell the user you can see the file '{p.name}' "
                f"({size}) has been uploaded and ask what they'd like to do with it."
            )
            threading.Thread(target=self.on_text_command, args=(msg,), daemon=True).start()

    def notify_phone_connected(self) -> None:
        if self._remote_overlay and self._remote_overlay.isVisible():
            self._remote_overlay.mark_connected()

    def _open_remote(self):
        if not self.on_remote_clicked:
            self._route_log("SYS: Dashboard not running — remote unavailable.")
            return
        result = self.on_remote_clicked()
        if not result:
            self._route_log("SYS: Could not generate remote key.")
            return
        url    = result[0]
        key    = result[1]
        auto   = result[2] if len(result) >= 3 else ""
        manual = result[3] if len(result) >= 4 else url
        if self._remote_overlay:
            self._remote_overlay._do_close()
        cw  = self.centralWidget()
        ow, oh = RemoteKeyOverlay._OW, RemoteKeyOverlay._OH
        ov  = RemoteKeyOverlay(url, key, auto_login_url=auto, manual_url=manual,
                               expiry_secs=600, parent=cw)
        ov.set_new_key_callback(self.on_remote_clicked)
        ov.setGeometry(
            (cw.width()  - ow) // 2,
            (cw.height() - oh) // 2,
            ow, oh,
        )
        ov.closed.connect(lambda: setattr(self, '_remote_overlay', None))
        ov.show()
        self._remote_overlay = ov
        self._route_log(f"SYS: Remote key generated — manual: {manual or url}")

    # ── Auto-start ──────────────────────────────────────────────────────────────

    def _check_autostart(self) -> bool:
        """Returns True if auto-start is currently registered on this OS."""
        try:
            if _OS == "Windows":
                import winreg
                key = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_READ)
                try:
                    winreg.QueryValueEx(key, "JARVIS_AI")
                    return True
                except FileNotFoundError:
                    return False
                finally:
                    winreg.CloseKey(key)
            elif _OS == "Darwin":
                return (Path.home() / "Library" / "LaunchAgents"
                        / "com.jarvis.assistant.plist").exists()
            else:
                return (Path.home() / ".config" / "autostart" / "jarvis.desktop").exists()
        except Exception:
            return False

    def _toggle_autostart(self):
        currently_on = self._check_autostart()
        try:
            script = str(Path(__file__).resolve().parent / "main.py")
            if _OS == "Windows":
                import winreg
                reg = winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Run", 0, winreg.KEY_ALL_ACCESS)
                if currently_on:
                    winreg.DeleteValue(reg, "JARVIS_AI")
                else:
                    pythonw = Path(sys.executable).parent / "pythonw.exe"
                    exe = str(pythonw if pythonw.exists() else sys.executable)
                    winreg.SetValueEx(reg, "JARVIS_AI", 0, winreg.REG_SZ,
                                      f'"{exe}" "{script}"')
                winreg.CloseKey(reg)
            elif _OS == "Darwin":
                plist_dir = Path.home() / "Library" / "LaunchAgents"
                plist_dir.mkdir(parents=True, exist_ok=True)
                plist = plist_dir / "com.jarvis.assistant.plist"
                if currently_on:
                    plist.unlink(missing_ok=True)
                else:
                    plist.write_text(
                        '<?xml version="1.0" encoding="UTF-8"?>\n'
                        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
                        '<plist version="1.0"><dict>\n'
                        '  <key>Label</key><string>com.jarvis.assistant</string>\n'
                        '  <key>ProgramArguments</key><array>\n'
                        f'    <string>{sys.executable}</string>\n'
                        f'    <string>{script}</string>\n'
                        '  </array>\n'
                        '  <key>RunAtLoad</key><true/>\n'
                        '</dict></plist>\n'
                    )
            else:
                desk_dir = Path.home() / ".config" / "autostart"
                desk_dir.mkdir(parents=True, exist_ok=True)
                desk = desk_dir / "jarvis.desktop"
                if currently_on:
                    desk.unlink(missing_ok=True)
                else:
                    desk.write_text(
                        "[Desktop Entry]\n"
                        f"Name={self._assistant_name}\n"
                        f"Exec={sys.executable} {script}\n"
                        "Type=Application\nTerminal=false\n"
                        "X-GNOME-Autostart-enabled=true\n"
                    )
            enabled = not currently_on
            self._update_autostart_btn(enabled)
            self._route_log(
                f"SYS: Auto-start {'enabled' if enabled else 'disabled'}.")
        except Exception as e:
            self._route_log(f"ERR: Auto-start failed — {e}")

    def _update_autostart_btn(self, enabled: bool):
        """Settings reads the real login-item state directly; nothing to restyle here."""
        return

    def _preview_ui_color(self, hex_color: str):
        """Canlı önizleme — tüm arayüzü yeni renge boyar (config'e YAZMAZ)."""
        old = current_palette()
        if apply_ui_accent(hex_color):
            retheme_all_widgets(old, current_palette())

    def _apply_name_update(self, name: str, user_name: str, ui_color: str = ""):
        """Update everything that shows the name/accent and save it (merge, atomic)."""
        from memory.config_manager import update_config
        self._assistant_name = name.strip() or "JARVIS"
        display = self._assistant_name.upper()
        self.setWindowTitle(f"{display} — MARK XLIX")
        self._brand.setText(display)
        self.stage.set_name(self._assistant_name)
        self.convo.set_name(self._assistant_name)
        if ui_color:
            old = current_palette()
            if apply_ui_accent(ui_color):
                retheme_all_widgets(old, current_palette())
        try:
            changes = {"assistant_name": self._assistant_name, "user_name": user_name.strip()}
            if ui_color:
                changes["ui_color"] = ui_color.strip().lower()
            update_config(changes)
            self._route_log(f"SYS: Identity updated — {display}")
        except Exception as e:
            self._route_log(f"ERR: Couldn't save settings — {e}")

    def _on_clipboard_changed(self):
        try:
            text = QApplication.clipboard().text().strip()
            if len(text) >= 10:
                self._clipboard_sig.emit(text)
        except Exception:
            pass

    def _show_clipboard_panel(self, text: str):
        self._clipboard_panel.show_clipboard(text)
        self._position_clipboard_panel()

    def _position_clipboard_panel(self):
        cw = self.centralWidget()
        pw = ClipboardPanel._W
        ph = self._clipboard_panel.sizeHint().height() or ClipboardPanel._H
        x = (cw.width() - pw) // 2
        y = cw.height() - ph - 6
        self._clipboard_panel.setGeometry(x, y, pw, ph)
        self._clipboard_panel.raise_()

    def _on_clipboard_action(self, cmd: str):
        if self.on_text_command:
            threading.Thread(target=self.on_text_command, args=(cmd,), daemon=True).start()

    # ── Action approval ──────────────────────────────────────────────────────────

    def _enqueue_confirm(self, req: "_ConfirmRequest"):
        self._confirm_queue.append(req)
        if self._confirm_active is None:
            self._show_next_confirm()

    def _show_next_confirm(self):
        while self._confirm_queue:
            req = self._confirm_queue.pop(0)
            if req.event.is_set():        # worker already gave up waiting
                continue
            self._confirm_active = req
            self._route_log(f"SYS: Approval needed — {req.title}")
            self.chip_approval.show()
            if not self.isActiveWindow():
                QApplication.alert(self)              # bounce the Dock icon; no extra permission
            self._show_scrim()
            self._confirm_panel.show_request(req.title, req.detail, req.timeout)
            self._position_confirm_panel()
            self.raise_(); self.activateWindow()
            return
        self._confirm_active = None
        self.chip_approval.hide()
        self._update_scrim()

    def _on_confirm_answered(self, approved: bool):
        req = self._confirm_active
        self._confirm_active = None
        if req is not None:
            req.answer(approved)
            self._route_log(
                f"SYS: {'Allowed' if approved else 'Denied'} — {req.title}"
            )
        self._show_next_confirm()

    # ── Audio settings + scrim ───────────────────────────────────────────────────

    def _position_confirm_panel(self):
        cw = self.centralWidget()
        pw = min(ConfirmPanel._W, cw.width() - 24)
        ph = min(ConfirmPanel._H, cw.height() - 24)
        self._confirm_panel.setGeometry((cw.width() - pw) // 2, (cw.height() - ph) // 2, pw, ph)
        self._confirm_panel.raise_()

    # ────────────────────────────────────────────────────────────────────────────

    def _do_interrupt(self):
        if self.on_interrupt:
            self.on_interrupt()

    # ── layout pieces ───────────────────────────────────────────────────────────

    def _build_titlebar(self) -> QWidget:
        w = QWidget(); w.setObjectName("TitleBar"); w.setFixedHeight(46)
        w.setStyleSheet(f"#TitleBar {{ background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #17191C, stop:1 #141618);"
                        f" border-bottom: 1px solid {C.BORDER}; }}")
        lay = QHBoxLayout(w); lay.setContentsMargins(16, 0, 12, 0); lay.setSpacing(8)
        self._brand = label(self._assistant_name.upper(), 11.5, C.TEXT, QFont.Weight.DemiBold, spacing=2.4)
        lay.addWidget(self._brand); lay.addSpacing(8)
        self.chip_conn = Chip("Connecting…", "neutral")
        self.chip_conn.setToolTip("Connection to Gemini Live")
        self.chip_conn.clicked.connect(self._conn_chip_clicked)
        self.chip_mic = Chip("Mic live", "live")
        self.chip_mic.setToolTip("Microphone — click for audio settings")
        self.chip_mic.clicked.connect(lambda: self.open_settings("Audio"))
        self.chip_voice = Chip("Voice: not set up", "neutral")
        self.chip_voice.setToolTip("Voice identity — who is speaking. Click for enrolment and the accuracy check")
        self.chip_voice.clicked.connect(lambda: self.open_settings("Voice identity"))
        self.chip_phone = Chip("Phone off", "neutral")
        self.chip_phone.setToolTip("Phone remote control")
        self.chip_phone.clicked.connect(lambda: self.open_settings("Remote"))
        self.chip_approval = Chip("Approval needed", "warn")
        self.chip_approval.clicked.connect(self._focus_approval)
        self.chip_approval.hide()
        for c in (self.chip_conn, self.chip_mic, self.chip_voice, self.chip_phone, self.chip_approval):
            lay.addWidget(c)
        lay.addStretch()
        self._convo_btn = button("Conversation", "ghost", 28)
        self._convo_btn.setCheckable(True)
        self._convo_btn.clicked.connect(self._toggle_convo)
        self._convo_btn.hide()
        lay.addWidget(self._convo_btn)
        self._clock_lbl = label("--:--", 12, C.TEXT_MED, mono=True)
        self._clock_lbl.setToolTip("")
        lay.addWidget(self._clock_lbl)
        gear = QPushButton("⚙"); gear.setFixedSize(30, 30); gear.setCursor(Qt.CursorShape.PointingHandCursor)
        gear.setToolTip("Settings (⌘,)")
        gear.setStyleSheet(f"QPushButton {{ background: {C.PANEL}; color: {C.TEXT_MED}; border: 1px solid {C.BORDER};"
                           f" border-radius: 8px; font-size: 14px; }} QPushButton:hover {{ color: {C.TEXT};"
                           f" border-color: {C.BORDER_B}; background: {C.PANEL2}; }}")
        gear.clicked.connect(lambda: self.open_settings())
        lay.addWidget(gear)
        self._refresh_voice_chip()
        return w

    def _build_statusbar(self) -> QWidget:
        w = QWidget(); w.setObjectName("StatusBar"); w.setFixedHeight(26)
        w.setStyleSheet(f"#StatusBar {{ background: {C.DARK}; border-top: 1px solid {C.BORDER}; }}")
        lay = QHBoxLayout(w); lay.setContentsMargins(14, 0, 14, 0); lay.setSpacing(18)
        self._st = {}
        for k in ("cpu", "mem", "gpu", "tmp", "audio", "rtt"):
            l = label("", 10.5, C.TEXT_DIM, mono=k in ("cpu", "mem", "gpu", "tmp", "rtt"))
            self._st[k] = l; lay.addWidget(l)
        lay.addStretch()
        lay.addWidget(label("F4 mute   ·   esc interrupt   ·   F11 full screen   ·   ⌘, settings", 10.5, C.TEXT_DIM))
        return w

    def _tick_clock(self):
        self._clock_lbl.setText(time.strftime("%H:%M"))
        self._clock_lbl.setToolTip(time.strftime("%A %d %B %Y"))

    def _update_metrics(self):
        snap = _metrics.snapshot()
        self._st["cpu"].setText(f"CPU {snap['cpu']:.0f}%")
        self._st["mem"].setText(f"Memory {snap['mem']:.0f}%")
        self._st["gpu"].setText(f"GPU {snap['gpu']:.0f}%" if snap["gpu"] >= 0 else "")
        self._st["gpu"].setVisible(snap["gpu"] >= 0)
        self._st["tmp"].setText(f"{snap['tmp']:.0f}°C" if snap["tmp"] >= 0 else "")
        self._st["tmp"].setVisible(snap["tmp"] >= 0)
        a = self.audio
        if a is None:
            self._st["audio"].setText("Audio unavailable")
            self._st["audio"].setStyleSheet(f"color: {C.RED}; background: transparent;")
            self._st["rtt"].setText("")
            return
        st = a.stats()
        if st["input_error"] or st["output_error"] or st["processing_error"]:
            txt, col = "Audio: needs attention", C.ACC2
        elif a.settings.processing_enabled and st["processing_active"]:
            txt, col = "Audio processing on", C.TEXT_DIM
        else:
            txt, col = "Audio bypass", C.TEXT_DIM
        self._st["audio"].setText(txt)
        self._st["audio"].setStyleSheet(f"color: {col}; background: transparent;")
        rtt = st.get("round_trip_ms")
        self._st["rtt"].setText(f"Round trip {rtt:.0f} ms" if rtt else "")
        self._refresh_mic_chip()

    # ── responsive layout ───────────────────────────────────────────────────────

    def _apply_responsive(self):
        narrow = self.width() < _NARROW_W
        self._convo_btn.setVisible(narrow)
        self.convo.setVisible((not narrow) or self._convo_forced)
        self._convo_btn.setChecked(self.convo.isVisible())
        self.chip_phone.setVisible(self.width() >= 900)
        if hasattr(self, "toasts"):
            # toasts sit under the title bar, beside (never over) the conversation
            self.toasts.top_margin = 58
            self.toasts.right_margin = (_CONVO_W if self.convo.isVisible() else 0) + 16
            self.toasts.relayout()

    def _toggle_convo(self):
        self._convo_forced = not self.convo.isVisible()
        self._apply_responsive()

    # ── state from the backend ──────────────────────────────────────────────────

    def _apply_state(self, state: str):
        self.stage.set_backend_state(state)
        if state in ("LISTENING", "SPEAKING", "THINKING", "PROCESSING"):
            if self.chip_conn.tone != "ok":
                self.set_connection("connected")
            if state == "LISTENING":
                self.stage.set_activity("")
        elif state == "SLEEPING" and self.chip_conn.tone == "ok":
            self.set_connection("reconnecting")

    def set_connection(self, state: str, detail: str = ""):
        """connected | connecting | reconnecting | offline (with a reason)."""
        text, tone = {"connected": ("Connected", "ok"), "connecting": ("Connecting…", "neutral"),
                      "reconnecting": ("Reconnecting…", "warn"), "offline": ("Offline", "bad")
                      }.get(state, (state, "neutral"))
        self.chip_conn.set(text, tone, detail or text)
        self._conn_detail = detail
        if state == "offline":
            self.stage.set_error("Can't reach Gemini", detail or "Check your connection or API key.",
                                 "Try again later" if "key" not in detail.lower() else None, connection=True)
        elif state == "connected":
            self.stage.clear_error(only_connection=True)

    def _conn_chip_clicked(self):
        d = getattr(self, "_conn_detail", "")
        self.toasts.show_toast(self.chip_conn.text(), d or "Connection to Gemini Live.", self.chip_conn.tone
                               if self.chip_conn.tone in ("warn", "bad", "ok") else "neutral", key="conn")

    def _refresh_mic_chip(self):
        a = self.audio
        if self._muted:
            self.chip_mic.set("Muted", "bad", "Microphone muted (F4)")
        elif a is None or a.stats().get("input_error"):
            err = a.stats().get("input_error") if a else "Audio engine not running"
            self.chip_mic.set("Mic problem", "bad", err or "")
        elif a.levels().get("sending") is False and a.is_playing:
            self.chip_mic.set("Mic paused", "neutral", "Paused while JARVIS speaks")
        else:
            self.chip_mic.set("Mic live", "live", "Microphone is live")

    # ── messages, tools, speaker ───────────────────────────────────────────────

    def _route_log(self, text: str):
        kind = self.convo.route_log(text)
        if kind == "error":
            body = text.split(":", 1)[-1].strip()
            if time.time() - self._last_err_toast > 4:              # don't stack identical bursts
                self._last_err_toast = time.time()
                self.toasts.show_toast("Something went wrong", body, "bad", key="err:" + body[:40])

    def _on_message(self, role: str, text: str, source: str):
        if role == "user":
            r = self.convo.add_user(text, source)
            r._ts = time.time()
            self._pending_user_row = r
            ps = getattr(self, "_pending_speaker", None)
            if source == "voice" and ps and time.time() - ps[2] < 30 and self._show_speaker:
                try:
                    r.set_speaker(ps[0], ps[1])
                except Exception:
                    pass
            self._pending_speaker = None
        elif role == "assistant":
            self.convo.add_assistant(text)
            self.stage.set_speaking_text("")
        elif role == "note":
            self.convo.add_note(text)
        elif role == "error":
            self._route_log("ERR: " + text)

    def _on_tool_event(self, call_id: str, tool: str, status: str, detail: str):
        from ui_conversation import TOOL_NAMES
        name = TOOL_NAMES.get(tool, tool.replace("_", " ").capitalize())
        if status == "started":
            self.convo.tool_started(call_id, tool, detail)
            if tool != "save_memory":
                self.stage.set_activity(f"{name}{' — ' + detail if detail else ''}…")
        else:
            self.convo.tool_status(call_id, status, "" if status != "failed" else detail)
            if status == "approval":
                self.stage.set_activity(f"{name} — waiting for your approval")
            elif status in ("done", "failed", "declined"):
                self.stage.set_activity("")

    def _on_speaker(self, label_: str, score):
        self._last_verdict = (label_, score)
        self._refresh_voice_chip()
        # With the voice filter on, the verdict arrives *before* the message it
        # belongs to; _on_gate/_on_message attach it instead.
        if self._show_speaker and not self._gate_on():
            self.convo.set_last_user_speaker(label_, score)

    def _gate_on(self) -> bool:
        try:
            from memory.config_manager import get_voice_gate
            sid = self.speaker_id
            return get_voice_gate()["mode"] != "off" and sid is not None and sid.status == "ready"
        except Exception:
            return False

    def _on_gate(self, action: str, label_: str, score):
        """The voice filter decided about something it heard."""
        if action == "ignored":
            hint = {"unknown": "Ignored — that didn't sound like you.",
                    "uncertain": "Not sure that was you — say it again?"}.get(label_)
            if hint:
                self.stage.set_gate_hint(hint)
        elif action == "accepted":
            self._pending_speaker = ("you", score if label_ == "you" else None, time.time())
        elif action == "fallback" and label_ in ("unavailable", "no_model"):
            self.stage.set_gate_hint("Voice check unavailable — listening to everyone.")
        self._refresh_voice_chip()

    def _refresh_voice_chip(self):
        if not hasattr(self, "chip_voice"):
            return
        sid = self.speaker_id
        st = sid.status if sid else "unavailable"
        if not self._show_speaker and st == "ready":
            self.chip_voice.hide(); return
        self.chip_voice.show()
        if st != "ready":
            self.chip_voice.set({"no_model": "Voice: not set up", "not_enrolled": "Voice: not enrolled",
                                 "unavailable": "Voice: unavailable"}.get(st, "Voice"), "neutral",
                                "Voice identity — click to set up")
            return
        gate = self._gate_on()
        v = getattr(self, "_last_verdict", None)
        if not v:
            self.chip_voice.set("Listening for you" if gate else "Voice ready", "neutral",
                                "Voice filter on: only your voice goes to Gemini" if gate else "Waiting for you to speak")
            return
        lab, score = v
        text = {"you": "You", "uncertain": "Not sure", "unknown": "Ignored" if gate else "Unknown speaker",
                "too_short": "Too short to tell"}.get(lab, lab)
        if score is not None:
            text += f" · {score:.2f}"
        self.chip_voice.set(text, {"you": "ok", "uncertain": "warn", "unknown": "warn"}.get(lab, "neutral"),
                            "Recognition is a display aid, not security.")

    def _show_content(self, title: str, text: str):
        self.convo.add_result(title, text)

    # ── composer, files, mute, interrupt ────────────────────────────────────────

    def _send_text(self, txt: str):
        self.convo.add_user(txt, "typed")
        if self.on_text_command:
            threading.Thread(target=self.on_text_command, args=(txt,), daemon=True).start()

    def _attach_file(self):
        self.convo.show_tab(2)
        self._drop_zone._browse()

    def _set_muted(self, muted: bool):
        if muted != self._muted:
            self._toggle_mute()

    def _toggle_mute(self):
        self._muted = not self._muted
        self.stage.set_muted(self._muted)
        self._refresh_mic_chip()
        self.convo.add_note("Microphone muted." if self._muted else "Microphone on.")

    def _on_escape(self):
        if self._confirm_active is not None:
            self._confirm_panel._finish(False)          # Esc always means Deny on an approval
        elif self._settings is not None:
            self._settings.close_sheet()
        else:
            self._do_interrupt()

    # ── settings sheet ─────────────────────────────────────────────────────────

    def open_settings(self, section: str = "General"):
        from ui_settings import SettingsSheet
        if self._settings is not None:
            self._settings.show_section(section)
            return
        sh = SettingsSheet(self, section, parent=self.centralWidget())
        sh.closed.connect(self._settings_closed)
        self._settings = sh
        self._show_scrim()
        self._position_settings()
        fade_in(sh)
        sh.setFocus()
        if self._confirm_panel.isVisible():           # approvals always stay on top
            self._confirm_panel.raise_()

    def _settings_closed(self):
        self._settings = None
        self._refresh_voice_chip()
        self._update_scrim()

    def _position_settings(self):
        if self._settings is None:
            return
        from ui_settings import SettingsSheet
        cw = self.centralWidget()
        w = min(SettingsSheet._W, cw.width() - 32)
        h = min(SettingsSheet._H, cw.height() - 32)
        self._settings.setGeometry((cw.width() - w) // 2, (cw.height() - h) // 2, w, h)
        self._settings.raise_()
        if self._confirm_panel.isVisible():
            self._confirm_panel.raise_()

    def _show_scrim(self):
        cw = self.centralWidget()
        self._scrim.setGeometry(0, 0, cw.width(), cw.height())
        self._scrim.show()
        self._scrim.raise_()
        if self._settings is not None:
            self._settings.raise_()
        if self._confirm_panel.isVisible():
            self._confirm_panel.raise_()

    def _update_scrim(self):
        if self._settings is None and self._confirm_active is None:
            self._scrim.hide()

    def _scrim_clicked(self):
        # A click outside closes Settings; it never answers an approval.
        if self._confirm_active is None and self._settings is not None:
            self._settings.close_sheet()

    def _focus_approval(self):
        if self._confirm_active is not None:
            self._confirm_panel.raise_(); self._confirm_panel.setFocus()

    # SettingsController implementation (used by ui_settings) ────────────────────

    def read_config(self) -> dict:
        return _read_full_config()

    def save_identity(self, name: str, user: str, color: str) -> None:
        self._apply_name_update(name, user, color)

    def preview_accent(self, hex_color: str) -> None:
        self._preview_ui_color(hex_color)

    def default_accent(self) -> str:
        return DEFAULT_UI_COLOR

    def brief_enabled(self) -> bool:
        from memory.config_manager import get_brief_enabled
        return get_brief_enabled()

    def set_brief(self, on: bool) -> None:
        from memory.config_manager import save_brief_enabled
        save_brief_enabled(bool(on))

    def autostart_enabled(self) -> bool:
        return self._check_autostart()

    def set_autostart(self, on: bool) -> None:
        if bool(on) != self._check_autostart():
            self._toggle_autostart()

    def create_shortcut(self) -> None:
        self._create_desktop_shortcut()

    def save_audio(self, settings: dict) -> None:
        from memory.config_manager import save_audio_settings
        save_audio_settings(settings)
        self._update_metrics()

    def show_speaker(self) -> bool:
        return self._show_speaker

    def set_show_speaker(self, on: bool) -> None:
        from memory.config_manager import update_config
        self._show_speaker = bool(on)
        update_config({"voice_identity_display": self._show_speaker})
        self._refresh_voice_chip()

    def set_voice_setup(self, active: bool) -> None:
        self._voice_setup = bool(active)

    def record_audio(self, seconds: float, on_done) -> bool:
        return bool(self.audio and self.audio.capture(seconds, on_done))

    def remote_status(self) -> dict:
        if not self.on_remote_status:
            return {"available": False, "reason": "The phone dashboard isn't available (missing packages?)."}
        try:
            return self.on_remote_status()
        except Exception as e:
            return {"available": False, "reason": str(e)}

    def open_pairing(self) -> None:
        if self._settings is not None:
            self._settings.close_sheet()
        self._open_remote()

    def forget_phones(self) -> int:
        return int(self.on_forget_phones()) if self.on_forget_phones else 0

    def stop_remote(self) -> None:
        if self.on_stop_remote:
            self.on_stop_remote()

    def toast(self, title: str, body: str = "", tone: str = "neutral") -> None:
        self.toasts.show_toast(title, body, tone)

    def set_remote_chip(self, running: bool, phones: int) -> None:
        if phones:
            self.chip_phone.set(f"Phone connected" + (f" ×{phones}" if phones > 1 else ""), "ok")
        elif running:
            self.chip_phone.set("Remote on", "live", "Waiting for a phone to connect")
        else:
            self.chip_phone.set("Phone off", "neutral", "Phone remote control is off")

    def _check_config(self) -> bool:
        if not API_FILE.exists(): return False
        try:
            d = json.loads(API_FILE.read_text(encoding="utf-8"))
            return bool(d.get("gemini_api_key")) and bool(d.get("os_system"))
        except Exception:
            return False

    def _show_setup(self):
        ov = SetupOverlay(self.centralWidget())
        cw = self.centralWidget()
        ow, oh = 460, 390
        ov.setGeometry(
            (cw.width()  - ow) // 2,
            (cw.height() - oh) // 2,
            ow, oh,
        )
        ov.done.connect(self._on_setup_done)
        ov.show()
        self._overlay = ov

    def _on_setup_done(self, key: str, os_name: str):
        from memory.config_manager import update_config
        update_config({"gemini_api_key": key, "os_system": os_name})     # keeps your other settings
        self._ready = True
        if self._overlay:
            self._overlay.hide()
            self._overlay = None
        self._apply_state("LISTENING")
        self._assistant_name = _read_full_config().get("assistant_name", "JARVIS") or "JARVIS"
        self._route_log(f"SYS: Initialised. {self._assistant_name} online.")

class _RootShim:
    def __init__(self, app: QApplication):
        self._app = app
    def mainloop(self):
        self._app.exec()
    def protocol(self, *_):
        pass


class JarvisUI:
    def __init__(self, face_path: str, size=None):
        self._app = QApplication.instance() or QApplication(sys.argv)
        self._app.setStyle("Fusion")
        self._win = MainWindow(face_path)
        self._win.show()
        self.root = _RootShim(self._app)

    @property
    def muted(self) -> bool:
        return self._win._muted

    @muted.setter
    def muted(self, v: bool):
        if v != self._win._muted:
            self._win._toggle_mute()

    @property
    def current_file(self) -> str | None:
        return self._win._drop_zone.current_file()

    @property
    def on_text_command(self):
        return self._win.on_text_command

    @on_text_command.setter
    def on_text_command(self, cb):
        self._win.on_text_command = cb

    @property
    def on_remote_clicked(self):
        return self._win.on_remote_clicked

    @on_remote_clicked.setter
    def on_remote_clicked(self, cb):
        self._win.on_remote_clicked = cb

    @property
    def on_interrupt(self):
        return self._win.on_interrupt

    @on_interrupt.setter
    def on_interrupt(self, cb):
        self._win.on_interrupt = cb

    def notify_phone_connected(self) -> None:
        """Thread-safe: the dashboard calls this from its asyncio thread."""
        self._win._phone_sig.emit()

    def attach_audio(self, engine) -> None:
        """Give the UI the AudioEngine: live levels for the stage, status and Audio settings."""
        self._win.audio = engine
        self._win.stage.attach_audio(engine)
        QTimer.singleShot(0, self._win._update_metrics)

    def attach_speaker_id(self, sid) -> None:
        """Give the UI the SpeakerIdentity (voice identity status and settings)."""
        self._win.speaker_id = sid
        QTimer.singleShot(0, self._win._refresh_voice_chip)

    # ── thread-safe structured updates (call from any thread) ─────────────────

    def add_message(self, role: str, text: str, source: str = "") -> None:
        """role: user | assistant | note | error"""
        self._win._msg_sig.emit(role, text, source)

    def tool_event(self, call_id: str, tool: str, status: str, detail: str = "") -> None:
        """status: started | approval | done | failed | declined"""
        self._win._tool_sig.emit(call_id, tool, status, detail)

    def set_connection(self, state: str, detail: str = "") -> None:
        self._win._conn_sig.emit(state, detail)

    def set_speaker(self, label: str, score) -> None:
        self._win._speaker_sig.emit(label, score)

    def notify(self, title: str, body: str = "", tone: str = "neutral") -> None:
        self._win._toast_sig.emit(title, body, tone)

    def set_speaking_text(self, text: str) -> None:
        self._win._caption_sig.emit(text[-220:])

    def set_hearing_text(self, text: str) -> None:
        self._win._hearing_sig.emit(text[-220:])

    def gate_event(self, action: str, label: str, score) -> None:
        self._win._gate_sig.emit(action, label, score)

    def set_remote_providers(self, status_fn, forget_fn, stop_fn) -> None:
        self._win.on_remote_status = status_fn
        self._win.on_forget_phones = forget_fn
        self._win.on_stop_remote = stop_fn

    def update_remote(self, running: bool, phones: int) -> None:
        self._win._remote_sig.emit(bool(running), int(phones))

    def confirm_action(self, title: str, detail: str = "", timeout: int = 60) -> bool:
        """Ask the user to approve a risky action. Call from a worker thread; blocks
        until Allow/Deny is clicked or `timeout` seconds pass (which counts as Deny)."""
        if threading.current_thread() is threading.main_thread():
            # Blocking here would freeze the GUI and could never be answered.
            print(f"[Confirm] Called on the GUI thread — refusing: {title}")
            return False
        req = _ConfirmRequest(title, detail, timeout)
        self._win._confirm_sig.emit(req)
        # Small grace period beyond the visible countdown.
        if not req.event.wait(req.timeout + 3):
            req.answer(False)
        return req.approved

    def set_state(self, state: str):
        self._win._state_sig.emit(state)

    def write_log(self, text: str):
        self._win._log_sig.emit(text)

    def wait_for_api_key(self):
        while not self._win._ready:
            time.sleep(0.1)

    def show_content(self, title: str, text: str):
        """Thread-safe: display content in the panel below the HUD."""
        self._win._content_sig.emit(title[:48], text[:4000])

    def prompt_reconfig(self):
        """Thread-safe: show the API key setup overlay (e.g. after an auth error)."""
        self._win._ready = False
        self._win._reconfig_sig.emit()

    def show_camera_frame(self, img_bytes: bytes):
        """Thread-safe: show a webcam frame in the small overlay (screen captures)."""
        self._win._camera_sig.emit(img_bytes)

    def start_camera_stream(self) -> None:
        """Thread-safe: start live camera feed in the full HUD area."""
        self._win.start_camera_stream()

    def stop_camera_stream(self) -> None:
        """Thread-safe: stop the live camera feed."""
        self._win.stop_camera_stream()

    @property
    def assistant_name(self) -> str:
        return self._win._assistant_name

    def start_speaking(self):
        self.set_state("SPEAKING")

    def stop_speaking(self):
        if not self.muted:
            self.set_state("LISTENING")