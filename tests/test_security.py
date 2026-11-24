"""
Regression tests for the safety features: tool approvals and the phone dashboard.

    .venv/bin/python -m unittest tests.test_security -v

The dashboard tests start real HTTPS listeners on spare ports (18200/18201)
with a throwaway certificate; nothing touches config/certs or ports 8000/8001.
"""

import asyncio
import json
import os
import shutil
import ssl
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class ApprovalTests(unittest.TestCase):
    def test_risky_tools_need_approval(self):
        from main import _approval_needed
        risky = [
            ("send_message", {"receiver": "A", "message_text": "hi", "platform": "WhatsApp"}),
            ("dev_agent", {"description": "x"}),
            ("file_controller", {"action": "delete", "path": "~/a"}),
            ("file_controller", {"action": "write", "path": "~/a", "content": "x"}),
            ("file_controller", {"action": "rename", "path": "~/a", "new_name": "b"}),
            ("file_processor", {"action": "run"}),
            ("file_processor", {"action": "extract"}),
            ("game_updater", {"shutdown_when_done": True}),
            ("game_updater", {"action": "schedule", "hour": "3", "minute": 5}),
        ]
        for name, args in risky:
            self.assertIsNotNone(_approval_needed(name, args), name)
        for name, args in [("file_controller", {"action": "read"}), ("web_search", {"query": "x"}),
                           ("file_processor", {"action": "summarize"})]:
            self.assertIsNone(_approval_needed(name, args), name)

    def test_fail_closed_without_ui(self):
        from actions.confirm import DECLINED, request_confirmation
        from actions.code_helper import code_helper
        from actions.desktop import desktop_control
        self.assertFalse(request_confirmation(None, "x"))
        self.assertEqual(code_helper({"action": "run", "file_path": "/bin/echo"}, player=None), DECLINED)
        self.assertEqual(desktop_control({"action": "clean"}, player=None), DECLINED)

    def test_desktop_never_executes_generated_code(self):
        from actions import desktop
        self.assertFalse(hasattr(desktop, "_execute_generated_code"))
        r = desktop.desktop_control({"action": "task", "task": "delete everything"})
        self.assertIn("can't do", r)


class DashboardTests(unittest.TestCase):
    PORT = 18200

    def test_dashboard_security(self):
        asyncio.run(self._run())

    async def _run(self):
        import dashboard.server as S
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        S.CERT_DIR, S.KEY_FILE, S.CERT_FILE = tmp, tmp / "raya.key", tmp / "raya.crt"
        S.PORT = self.PORT
        S._ensure_network_access = lambda port: None

        def call(path, data=None, token=None):
            ctx = ssl.create_default_context(cafile=str(S.CERT_FILE))
            h = {"Content-Type": "application/json"}
            if token:
                h["Authorization"] = f"Bearer {token}"
            req = urllib.request.Request(f"https://127.0.0.1:{S.PORT}{path}",
                                         data=json.dumps(data).encode() if data is not None else None, headers=h)
            try:
                with urllib.request.urlopen(req, context=ctx, timeout=5) as r:
                    return r.status, r.read().decode()
            except urllib.error.HTTPError as e:
                return e.code, e.read().decode()

        loop = asyncio.get_running_loop()
        acall = lambda *a, **k: loop.run_in_executor(None, lambda: call(*a, **k))

        d = S.DashboardServer()
        self.assertFalse(d.running, "must not listen until Remote Control is pressed")
        ok, msg = await d.start()
        self.assertTrue(ok, msg)
        try:
            self.assertEqual(os.stat(S.KEY_FILE).st_mode & 0o777, 0o600)
            codes = [(await acall("/login", {"pin": "ZZZZZZ"}))[0] for _ in range(6)]
            self.assertEqual(codes[:5], [401] * 5)
            self.assertEqual(codes[5], 429, "pairing must be rate-limited")
            d._fails.clear()
            key = d.new_key()
            st, body = await acall("/login", {"pin": key})
            tok = json.loads(body)["token"]
            self.assertEqual((await acall("/login", {"pin": key}))[0], 401, "codes are single-use")
            self.assertEqual((await acall("/api/files", token=tok))[0], 200)
            d._tokens[tok] = 0
            self.assertEqual((await acall("/api/files", token=tok))[0], 401, "tokens expire")
        finally:
            await d.stop("test")
        self.assertFalse(d.running)
        self.assertFalse(d._tokens)


if __name__ == "__main__":
    unittest.main()
