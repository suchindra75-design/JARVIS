"""
core/rename_migration.py — move what the app made under its old name (JARVIS /
Mark XLIX) over to Raya's names, so nothing is lost by the rename.

    ~/.jarvis                     → ~/.raya              reminder scripts
    ~/.jarvis_profiles            → ~/.raya_profiles     browser-automation logins
    ~/Desktop/JarvisProjects      → RayaProjects
    ~/{Downloads,Documents}/JARVIS Uploads → Raya Uploads
    config/certs/jarvis.{key,crt} → raya.{key,crt}       paired phones keep trusting it
    saved assistant name "JARVIS" → "Raya"
    macOS   LaunchAgents com.jarvis.{assistant,gameupdater,reminder.*} → com.raya.*
    Windows Run-key value JARVIS_AI → Raya; task JARVIS_GameUpdater → Raya_GameUpdater
    Linux   autostart jarvis.desktop → raya.desktop; crontab marker

The desktop shortcut is redone by the UI (RayaUI.migrate_legacy_shortcut).

Pending reminders on Windows and Linux are left where they are: the OS
scheduler runs them by their full path, and each script deletes itself after
firing. The old folder is removed once it is empty.

Every step checks before it acts, so a second run changes nothing, and a step
that fails is logged without stopping the others. Nothing here raises.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable

BASE_DIR = Path(__file__).resolve().parent.parent
_DONE_MARKER = ".renamed-from-jarvis"      # in ~/.raya; skips the slow OS-scheduler checks


def _log(msg: str) -> None:
    print(f"[Rename] {msg}")


def _merge_move(old: Path, new: Path) -> bool:
    """Move `old` to `new`. If both exist, move over what `new` lacks, then drop
    `old` if nothing is left in it. Returns True if anything moved."""
    if not old.exists():
        return False
    if not new.exists():
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old), str(new))
        return True
    moved = False
    if old.is_dir() and new.is_dir():
        for child in list(old.iterdir()):
            target = new / child.name
            if child.is_dir() and target.is_dir():
                moved |= _merge_move(child, target)
            elif not target.exists():
                shutil.move(str(child), str(target))
                moved = True
        try:
            old.rmdir()
        except OSError:
            pass            # something in both places with the same name: keep the old copy
    return moved


def _has_files(d: Path) -> bool:
    return d.is_dir() and any(p.is_file() for p in d.rglob("*"))


def _remove_empty_tree(d: Path) -> None:
    if not d.is_dir() or _has_files(d):
        return
    for p in sorted(d.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        try:
            p.rmdir()
        except OSError:
            pass
    try:
        d.rmdir()
    except OSError:
        pass


class _Migration:
    def __init__(self, home: Path, base_dir: Path, os_name: str, run: Callable):
        self.home, self.base, self.os, self.run = home, base_dir, os_name, run
        self.done: list[str] = []

    def step(self, fn: Callable, *args) -> None:
        try:
            fn(*args)
        except Exception as e:                    # never let a rename stop the app
            _log(f"⚠️ {getattr(fn, '__name__', 'step')} failed: {e}")

    def note(self, msg: str) -> None:
        self.done.append(msg)
        _log(msg)

    # ── folders and files ─────────────────────────────────────────────────────
    def folders(self) -> None:
        h = self.home
        pairs = [
            (h / ".jarvis_profiles", h / ".raya_profiles"),
            (h / "Desktop" / "JarvisProjects", h / "Desktop" / "RayaProjects"),
            (h / "Downloads" / "JARVIS Uploads", h / "Downloads" / "Raya Uploads"),
            (h / "Documents" / "JARVIS Uploads", h / "Documents" / "Raya Uploads"),
        ]
        for old, new in pairs:
            if _merge_move(old, new):
                self.note(f"moved {old} → {new}")
        if _merge_move(h / ".raya_profiles" / "firefox_jarvis", h / ".raya_profiles" / "firefox_raya"):
            self.note("renamed the Firefox automation profile")

    def reminders_folder(self) -> None:
        old, new = self.home / ".jarvis", self.home / ".raya"
        if not old.exists():
            return
        if self.os == "mac":
            if _merge_move(old, new):
                self.note(f"moved {old} → {new}")
        elif _has_files(old):
            _log(f"keeping {old} until its pending reminders have fired")
        else:
            _remove_empty_tree(old)

    def certs(self) -> None:
        d = self.base / "config" / "certs"
        for ext in ("key", "crt"):
            old, new = d / f"jarvis.{ext}", d / f"raya.{ext}"
            if old.exists() and not new.exists():
                old.rename(new)
                self.note(f"renamed dashboard certificate file {old.name} → {new.name}")

    def assistant_name(self) -> None:
        import memory.config_manager as cm
        if not cm.CONFIG_FILE.exists():
            return
        name = json.loads(cm.CONFIG_FILE.read_text(encoding="utf-8")).get("assistant_name")
        if isinstance(name, str) and name.strip().upper() == "JARVIS":
            cm.update_config({"assistant_name": "Raya"})
            self.note('saved assistant name "JARVIS" → "Raya"')

    # ── macOS LaunchAgents ────────────────────────────────────────────────────
    def _agents(self) -> Path:
        return self.home / "Library" / "LaunchAgents"

    def _rewrite_agent(self, old_plist: Path, new_plist: Path, replace: list[tuple[str, str]],
                       reload: bool) -> None:
        text = old_plist.read_text(encoding="utf-8")
        for a, b in replace:
            text = text.replace(a, b)
        if reload:
            self.run(["launchctl", "unload", str(old_plist)], capture_output=True)
        new_plist.write_text(text, encoding="utf-8")
        new_plist.chmod(0o644)
        old_plist.unlink()
        if reload:
            self.run(["launchctl", "load", str(new_plist)], capture_output=True)
        self.note(f"LaunchAgent {old_plist.name} → {new_plist.name}")

    def mac_agents(self) -> None:
        agents = self._agents()
        if not agents.is_dir():
            return

        # Auto-start: RunAtLoad, so loading it now would start a second copy and
        # unloading the old one could stop this one. The new file takes over at
        # the next login.
        old, new = agents / "com.jarvis.assistant.plist", agents / "com.raya.assistant.plist"
        if old.exists():
            if new.exists():
                old.unlink()
            else:
                self._rewrite_agent(old, new, [("com.jarvis.assistant", "com.raya.assistant")], reload=False)

        old, new = agents / "com.jarvis.gameupdater.plist", agents / "com.raya.gameupdater.plist"
        if old.exists() and not new.exists():
            self._rewrite_agent(old, new, [("com.jarvis.gameupdater", "com.raya.gameupdater")], reload=True)

        old_dir, new_dir = self.home / ".jarvis" / "reminders", self.home / ".raya" / "reminders"
        for old in sorted(agents.glob("com.jarvis.reminder.*.plist")):
            old_label = old.name[:-len(".plist")]
            old_task = old_label[len("com.jarvis.reminder."):]
            new_task = old_task.replace("JARVISReminder_", "RayaReminder_", 1)
            new_label = f"com.raya.reminder.{new_task}"
            new = agents / f"{new_label}.plist"
            if new.exists():
                continue
            # the script itself has already moved with ~/.jarvis → ~/.raya
            script_old = new_dir / f"{old_task}.py"
            script_new = new_dir / f"{new_task}.py"
            if script_old.exists() and not script_new.exists():
                body = script_old.read_text(encoding="utf-8").replace("J.A.R.V.I.S", "Raya")
                script_new.write_text(body, encoding="utf-8")
                script_new.chmod(0o600)
                script_old.unlink()
            self._rewrite_agent(old, new, [
                (old_label, new_label),
                (str(old_dir / f"{old_task}.py"), str(script_new)),
            ], reload=True)

    # ── Windows ───────────────────────────────────────────────────────────────
    def windows_autostart(self) -> None:
        import winreg
        path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_ALL_ACCESS) as reg:
            try:
                value, kind = winreg.QueryValueEx(reg, "JARVIS_AI")
            except FileNotFoundError:
                return
            try:
                winreg.QueryValueEx(reg, "Raya")
            except FileNotFoundError:
                winreg.SetValueEx(reg, "Raya", 0, kind, value)
            winreg.DeleteValue(reg, "JARVIS_AI")
            self.note("auto-start entry JARVIS_AI → Raya")

    def windows_game_task(self) -> None:
        flags = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
        q = self.run(["schtasks", "/Query", "/TN", "JARVIS_GameUpdater", "/XML"],
                     capture_output=True, text=True, **flags)
        if q.returncode != 0 or not q.stdout.strip():
            return
        fd, xml = tempfile.mkstemp(suffix=".xml", prefix="raya_task_")
        os.close(fd)
        try:
            Path(xml).write_text(q.stdout, encoding="utf-16")
            c = self.run(["schtasks", "/Create", "/TN", "Raya_GameUpdater", "/XML", xml, "/F"],
                         capture_output=True, text=True, **flags)
            if c.returncode != 0:
                raise RuntimeError((c.stderr or c.stdout).strip() or "schtasks /Create failed")
            self.run(["schtasks", "/Delete", "/TN", "JARVIS_GameUpdater", "/F"],
                     capture_output=True, text=True, **flags)
            self.note("scheduled task JARVIS_GameUpdater → Raya_GameUpdater")
        finally:
            Path(xml).unlink(missing_ok=True)

    # ── Linux ─────────────────────────────────────────────────────────────────
    def linux_autostart(self) -> None:
        d = self.home / ".config" / "autostart"
        if _merge_move(d / "jarvis.desktop", d / "raya.desktop"):
            self.note("autostart jarvis.desktop → raya.desktop")

    def linux_crontab(self) -> None:
        if not shutil.which("crontab"):
            return
        cur = self.run(["crontab", "-l"], capture_output=True, text=True)
        if cur.returncode != 0 or "JARVIS_GameUpdater" not in cur.stdout:
            return
        new = cur.stdout.replace("# JARVIS_GameUpdater", "# Raya_GameUpdater")
        w = self.run(["crontab", "-"], input=new, capture_output=True, text=True)
        if w.returncode != 0:
            raise RuntimeError(w.stderr.strip() or "crontab write failed")
        self.note("crontab game-update entry renamed")

    # ── all ───────────────────────────────────────────────────────────────────
    def run_all(self) -> list[str]:
        self.step(self.folders)
        self.step(self.reminders_folder)          # before mac_agents: it rewrites paths into ~/.raya
        if self.os == "mac":
            self.step(self.mac_agents)
        self.step(self.certs)
        self.step(self.assistant_name)

        marker = self.home / ".raya" / _DONE_MARKER
        if self.os in ("windows", "linux") and not marker.exists():
            failed = False
            for fn in ((self.windows_autostart, self.windows_game_task) if self.os == "windows"
                       else (self.linux_autostart, self.linux_crontab)):
                try:
                    fn()
                except Exception as e:
                    failed = True
                    _log(f"⚠️ {fn.__name__} failed: {e}")
            if not failed:
                try:
                    marker.parent.mkdir(parents=True, exist_ok=True)
                    marker.write_text("done\n")
                except OSError:
                    pass
        return self.done


def _os_name() -> str:
    s = platform.system()
    return "mac" if s == "Darwin" else "windows" if s == "Windows" else "linux"


def migrate_legacy_names(home: Path | None = None, base_dir: Path | None = None,
                         os_name: str | None = None, run: Callable = subprocess.run) -> list[str]:
    """Rename what the old JARVIS build left on this computer. Returns what changed."""
    try:
        return _Migration(home or Path.home(), base_dir or BASE_DIR,
                          os_name or _os_name(), run).run_all()
    except Exception as e:
        _log(f"⚠️ migration stopped: {e}")
        return []
