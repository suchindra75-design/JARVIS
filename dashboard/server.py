"""
dashboard/server.py — JARVIS phone remote-control dashboard

HTTPS on ports 8000 (QR link) and 8001 (typed-in address), LAN only.

Security model:
  - The server is NOT started with the app. It starts when you press
    Remote Control and stops by itself after IDLE_STOP_SECS with no phone
    connected and no activity.
  - TLS uses a certificate generated on this computer (config/certs/), never
    the one that shipped with the download. If no certificate can be made,
    the dashboard refuses to start rather than falling back to plain HTTP.
  - Pairing keys are single-use, expire in 10 minutes, and failed attempts
    are rate-limited per address. Session tokens expire after TOKEN_TTL_SECS,
    remembered-device tokens after DEVICE_TTL_SECS.
  - Commands are additionally AES-256-CBC encrypted by the phone page
    (CryptoJS, served locally after a one-time download).

Install deps:  pip install fastapi "uvicorn[standard]" cryptography
"""

import asyncio
import base64
import datetime as _dt
import hashlib
import ipaddress
import os
import re
import secrets
import socket
import string
import time
from pathlib import Path

_DEPS_OK = False
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
    import uvicorn
    _DEPS_OK = True
except ImportError:
    pass

# python-multipart is required for file uploads — optional dependency
_UPLOAD_OK = False
try:
    from fastapi import UploadFile, File as FastAPIFile
    _UPLOAD_OK = True
except Exception:
    pass

BASE_DIR    = Path(__file__).resolve().parent.parent
STATIC_DIR  = Path(__file__).parent / "static"
PORT        = 8000
MAX_UPLOAD_MB = 500

CERT_DIR    = BASE_DIR / "config" / "certs"
KEY_FILE    = CERT_DIR / "jarvis.key"
CERT_FILE   = CERT_DIR / "jarvis.crt"
# SHA-256 of the certificate that shipped with the public download. Its private
# key is public, so it is always replaced by one generated on this computer.
_SHIPPED_CERT_SHA256 = bytes.fromhex(
    "60CF6EB16C2F234041D03B85E8315D521486463FACBBE824B7E0D985A3F1B654"
)

TOKEN_TTL_SECS   = 12 * 3600        # a phone browser session
DEVICE_TTL_SECS  = 7 * 24 * 3600    # "remember this phone" (in memory, until restart)
IDLE_STOP_SECS   = 15 * 60          # stop the server after this long with no phone/activity
FAIL_WINDOW_SECS = 10 * 60          # pairing failures are counted over this window
MAX_FAILS_PER_IP = 5                # then that address is refused until the window passes
MAX_FAILS_TOTAL  = 30               # then every outstanding pairing key is cancelled


def _make_uploads_dir() -> Path:
    """Return (and create) the cross-platform uploads folder."""
    for candidate in [
        Path.home() / "Downloads" / "JARVIS Uploads",
        Path.home() / "Documents" / "JARVIS Uploads",
        BASE_DIR / "uploads",
    ]:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except Exception:
            pass
    return BASE_DIR / "uploads"


UPLOADS_DIR = _make_uploads_dir()

def _get_gemini_key() -> str | None:
    try:
        import json as _json
        with open(BASE_DIR / "config" / "api_keys.json", "r", encoding="utf-8") as f:
            return _json.load(f).get("gemini_api_key")
    except Exception:
        return None

_KEY_CHARS = [c for c in (string.ascii_uppercase + string.digits)
              if c not in ('O', 'I', 'L', '0', '1')]

# ── AES-256-CBC ───────────────────────────────────────────────────────────────
_AES_SALT = b'JARVIS-DASHBOARD-v1'


def _derive_key(session_key: str) -> bytes:
    """SHA-256(sessionKey‖salt) → 32-byte AES-256 key (microseconds, no PBKDF2 needed)."""
    return hashlib.sha256(session_key.encode('utf-8') + _AES_SALT).digest()


def _decrypt_cbc(aes_key: bytes, enc_b64: str) -> str:
    """Decrypt base64(IV[16] ‖ ciphertext) with AES-256-CBC + PKCS7."""
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    from cryptography.hazmat.primitives import padding as sym_pad
    raw      = base64.b64decode(enc_b64)
    iv, ct   = raw[:16], raw[16:]
    dec      = Cipher(algorithms.AES(aes_key), modes.CBC(iv)).decryptor()
    padded   = dec.update(ct) + dec.finalize()
    unpadder = sym_pad.PKCS7(128).unpadder()
    return (unpadder.update(padded) + unpadder.finalize()).decode('utf-8')


# ── CryptoJS (auto-download once, served locally) ─────────────────────────────
_CRYPTOJS_CDN  = ("https://cdnjs.cloudflare.com/ajax/libs/"
                  "crypto-js/4.2.0/crypto-js.min.js")
_CRYPTOJS_FILE = STATIC_DIR / "crypto-js.min.js"


def _ensure_network_access(port: int) -> None:
    """Cross-platform, best-effort: open port in the OS firewall for LAN access.

    Runs in a background thread — never blocks uvicorn startup.

    Windows : writes a .bat file, runs it elevated via Windows ShellExecuteW
              (native UAC dialog, guaranteed to appear). One-time setup.
    macOS   : osascript admin dialog if the Application Firewall is on.
    Linux   : pkexec GUI → sudo -n → prints manual command as fallback.
    """
    import sys, subprocess, os, tempfile, threading

    # ── Windows ──────────────────────────────────────────────────────────────
    if sys.platform == "win32":
        import ctypes, time

        port_rule = f"JARVIS Dashboard Port {port}"
        prog_rule  = "JARVIS Dashboard Python"
        py_exe     = sys.executable

        def _netsh_rule_exists(name: str) -> bool:
            try:
                r = subprocess.run(
                    ["netsh", "advfirewall", "firewall", "show", "rule", f"name={name}"],
                    capture_output=True, text=True, timeout=5,
                )
                return r.returncode == 0 and "No rules match" not in r.stdout
            except Exception:
                return False

        def _network_is_public() -> bool:
            try:
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                     "(Get-NetConnectionProfile | "
                     "Where-Object {$_.NetworkCategory -eq 'Public'} | "
                     "Measure-Object).Count"],
                    capture_output=True, text=True, timeout=6,
                )
                return r.stdout.strip() not in ("", "0")
            except Exception:
                return False

        need_port    = not _netsh_rule_exists(port_rule)
        need_prog    = not _netsh_rule_exists(prog_rule)

        # Networks marked "Public" are left alone on purpose: silently switching
        # them to "Private" would relax Windows' protections on café/hotel Wi-Fi.
        if _network_is_public():
            print("[Dashboard] This network is marked Public in Windows, so phones may not be "
                  "able to connect. Only change it to Private in Windows Settings if you trust it.")

        if not need_port and not need_prog:
            return  # already fully configured

        # Build a .bat file — netsh only, runs fast when elevated
        bat_lines = ["@echo off"]
        if need_port:
            bat_lines.append(
                f'netsh advfirewall firewall add rule '
                f'name="{port_rule}" protocol=TCP dir=in '
                f'localport={port} action=allow'
            )
        if need_prog:
            bat_lines.append(
                f'netsh advfirewall firewall add rule '
                f'name="{prog_rule}" dir=in action=allow '
                f'program="{py_exe}" enable=yes'
            )

        bat_body = "\r\n".join(bat_lines) + "\r\n"
        fd, bat_path = tempfile.mkstemp(suffix=".bat", prefix="jarvis_fw_")
        try:
            os.write(fd, bat_body.encode("mbcs"))   # Windows cmd.exe expects ANSI
            os.close(fd)
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            return

        # ── Try running directly (succeeds when already admin) ────────────────
        try:
            r = subprocess.run(
                [bat_path], capture_output=True, timeout=8, shell=True
            )
            if r.returncode == 0:
                print(f"[Dashboard] Firewall configured for port {port}.")
                try:
                    os.unlink(bat_path)
                except Exception:
                    pass
                return
        except Exception:
            pass

        # ── ShellExecuteW: native UAC elevation (most reliable on Windows) ────
        # ShellExecuteW with verb "runas" always shows the UAC dialog regardless
        # of UAC level settings. Non-blocking — uvicorn is already running.
        print("[Dashboard] One-time network setup required.")
        print("[Dashboard] >>> A Windows security dialog will appear — click 'Yes' <<<")
        try:
            ret = ctypes.windll.shell32.ShellExecuteW(
                None,       # hwnd  (no parent window)
                "runas",    # verb  (request elevation)
                bat_path,   # file  (our .bat)
                None,       # params
                None,       # working dir
                0,          # SW_HIDE (run without a visible cmd window)
            )
            if int(ret) > 32:
                # ShellExecuteW returns immediately; bat finishes in ~1 second.
                # Sleep briefly so the rules are in place before the first retry.
                time.sleep(2)
                print(f"[Dashboard] Network setup complete — port {port} is open.")
                print("[Dashboard] Refresh your phone browser to connect.")
            else:
                print("[Dashboard] Setup was not allowed.")
                print("[Dashboard] Phone connections may fail until JARVIS is run as Administrator.")
        except Exception as e:
            print(f"[Dashboard] Firewall setup error: {e}")
        finally:
            # Cleanup after the bat has had time to run
            def _cleanup(path: str) -> None:
                time.sleep(5)
                try:
                    os.unlink(path)
                except Exception:
                    pass
            threading.Thread(target=_cleanup, args=(bat_path,), daemon=True).start()
        return

    # ── macOS ─────────────────────────────────────────────────────────────────
    if sys.platform == "darwin":
        fw_ctl = "/usr/libexec/ApplicationFirewall/socketfilterfw"
        try:
            r = subprocess.run(
                [fw_ctl, "--getglobalstate"], capture_output=True, text=True, timeout=5,
            )
            if "disabled" in r.stdout.lower():
                return  # firewall off — nothing to do

            py = sys.executable
            listed = subprocess.run(
                [fw_ctl, "--listapps"], capture_output=True, text=True, timeout=5,
            )
            if py in listed.stdout:
                return  # already allowed

            print("[Dashboard] One-time network setup — enter your password in the macOS dialog.")
            subprocess.run(
                ["osascript", "-e",
                 f'do shell script "{fw_ctl} --add {py} && {fw_ctl} --unblockapp {py}"'
                 f' with administrator privileges'],
                timeout=60,
            )
        except Exception:
            pass  # macOS firewall is off by default — silent failure is fine
        return

    # ── Linux ─────────────────────────────────────────────────────────────────
    def _privileged(cmd: list[str]) -> bool:
        for prefix in (["pkexec"], ["sudo", "-n"]):
            try:
                r = subprocess.run(prefix + cmd, capture_output=True, timeout=30)
                if r.returncode == 0:
                    return True
            except Exception:
                pass
        return False

    try:  # ufw
        r = subprocess.run(["ufw", "status"], capture_output=True, text=True, timeout=5)
        if "active" in r.stdout.lower():
            if _privileged(["ufw", "allow", f"{port}/tcp"]):
                print(f"[Dashboard] ufw: port {port} allowed.")
            else:
                print(f"[Dashboard] Run manually:  sudo ufw allow {port}/tcp")
            return
    except FileNotFoundError:
        pass

    try:  # firewalld
        r = subprocess.run(
            ["firewall-cmd", "--state"], capture_output=True, text=True, timeout=5,
        )
        if "running" in r.stdout.lower():
            ok = (_privileged(["firewall-cmd", "--add-port", f"{port}/tcp", "--permanent"])
                  and _privileged(["firewall-cmd", "--reload"]))
            if ok:
                print(f"[Dashboard] firewalld: port {port} allowed.")
            else:
                print(f"[Dashboard] Run manually:  sudo firewall-cmd --add-port={port}/tcp --permanent && sudo firewall-cmd --reload")
            return
    except FileNotFoundError:
        pass

    try:  # iptables (not persistent but works until reboot)
        r = subprocess.run(["iptables", "-L", "INPUT", "-n"], capture_output=True, timeout=5)
        if r.returncode == 0:
            if _privileged(["iptables", "-A", "INPUT", "-p", "tcp", "--dport", str(port), "-j", "ACCEPT"]):
                print(f"[Dashboard] iptables: port {port} opened.")
            else:
                print(f"[Dashboard] Run manually:  sudo iptables -A INPUT -p tcp --dport {port} -j ACCEPT")
    except FileNotFoundError:
        pass  # no iptables means firewall is probably off — nothing to do


def _ensure_crypto_js() -> None:
    if _CRYPTOJS_FILE.exists():
        return
    try:
        import urllib.request
        print("[Dashboard] Downloading CryptoJS (one-time setup)…")
        urllib.request.urlretrieve(_CRYPTOJS_CDN, str(_CRYPTOJS_FILE))
        print("[Dashboard] CryptoJS cached — will serve locally from now on.")
    except Exception as e:
        print(f"[Dashboard] CryptoJS download failed: {e}")
        print(f"[Dashboard] Encryption will fall back to CDN load on client.")


_ensure_crypto_js()


# ── Per-computer TLS certificate ──────────────────────────────────────────────

def _cert_needs_renewal(ip: str) -> str | None:
    """Return why the certificate must be (re)generated, or None if it's fine."""
    if not (KEY_FILE.exists() and CERT_FILE.exists()):
        return "no certificate yet"
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes
        cert = x509.load_pem_x509_certificate(CERT_FILE.read_bytes())
        if cert.fingerprint(hashes.SHA256()) == _SHIPPED_CERT_SHA256:
            return "replacing the certificate that shipped with the download"
        not_after = cert.not_valid_after_utc
        if not_after - _dt.datetime.now(_dt.timezone.utc) < _dt.timedelta(days=30):
            return "certificate expires soon"
        try:
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            ips = {str(a) for a in san.get_values_for_type(x509.IPAddress)}
        except x509.ExtensionNotFound:
            ips = set()
        if ip not in ips:
            return f"this computer's address changed to {ip}"
    except Exception as e:
        return f"certificate unreadable ({e})"
    return None


def _write_private(path: Path, data: bytes) -> None:
    """Write a file readable only by the current user, replacing it atomically."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def _ensure_cert(ip: str) -> bool:
    """Make sure a certificate generated on this computer exists for `ip`."""
    reason = _cert_needs_renewal(ip)
    if reason is None:
        try:
            os.chmod(KEY_FILE, 0o600)
        except Exception:
            pass
        return True
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        print(f"[Dashboard] Creating a new HTTPS certificate ({reason}).")
        key  = ec.generate_private_key(ec.SECP256R1())
        name = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "JARVIS (this computer)"),
            x509.NameAttribute(NameOID.COMMON_NAME, f"jarvis-{socket.gethostname()}"[:64]),
        ])
        alt = [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
        if ip and ip != "127.0.0.1":
            alt.append(x509.IPAddress(ipaddress.ip_address(ip)))
        now  = _dt.datetime.now(_dt.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(name).issuer_name(name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _dt.timedelta(minutes=5))
            .not_valid_after(now + _dt.timedelta(days=397))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .sign(key, hashes.SHA256())
        )
        CERT_DIR.mkdir(parents=True, exist_ok=True)
        _write_private(KEY_FILE, key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ))
        CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
        print("[Dashboard] New certificate ready. Your phone will ask you to accept it once.")
        return True
    except Exception as e:
        print(f"[Dashboard] Could not create an HTTPS certificate: {e}")
        return False


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


# ── helpers ───────────────────────────────────────────────────────────────────

def _local_ip() -> str:
    """Return the best LAN-facing IPv4 address, no internet required."""
    # Method 1: route trick (fast, works when internet is available)
    for probe in ("8.8.8.8", "1.1.1.1", "192.168.1.1"):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.settimeout(0.5)
            s.connect((probe, 80))
            ip = s.getsockname()[0]
            s.close()
            if not ip.startswith("127."):
                return ip
        except Exception:
            pass

    # Method 2: hostname resolution (works offline on most systems)
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if not ip.startswith("127."):
            return ip
    except Exception:
        pass

    # Method 3: enumerate all interfaces (fully offline, no external deps)
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and not ip.startswith("169.254."):
                return ip
    except Exception:
        pass

    return "127.0.0.1"


def _read(name: str) -> str:
    return (STATIC_DIR / name).read_text(encoding="utf-8")


# ── DashboardServer ───────────────────────────────────────────────────────────

class DashboardServer:

    def __init__(self):
        self._ip                          = _local_ip()
        self._tokens: dict[str, float]    = {}   # auth_token → expiry (epoch secs)
        self._token_keys: dict[str, str]  = {}   # auth_token → session_key
        self._fails: dict[str, list[float]] = {} # client IP → failed pairing times
        self._servers: list               = []   # running uvicorn.Server objects
        self._server_tasks: list          = []
        self._watchdog_task               = None
        self._start_lock                  = asyncio.Lock()
        self._last_activity               = time.time()
        self._log_callback                = None
        self._aes_cache:  dict[str, bytes]= {}   # session_key → AES bytes
        self._clients: set[WebSocket]     = set()
        self._history: list[dict]         = []
        self._command_queue               = asyncio.Queue()
        self._wake_callback               = None
        self._connect_callback            = None
        self._pending_keys: dict[str, float] = {}
        self._device_sessions: dict[str, dict] = {}  # device_token → {session_key}
        self._phone_audio_queue: asyncio.Queue    = asyncio.Queue(maxsize=200)
        self._uploads_dir                 = UPLOADS_DIR
        self._login_html                  = _read("login.html")
        self._app_html                    = _read("app.html")
        self.app                          = self._build_app()

    # ── one-time key management ───────────────────────────────────────────

    def new_key(self, expiry_secs: int = 600) -> str:
        now = time.time()
        self._pending_keys = {k: v for k, v in self._pending_keys.items() if v > now}
        key = ''.join(secrets.choice(_KEY_CHARS) for _ in range(6))
        self._pending_keys[key] = now + expiry_secs
        self._touch()
        return key

    def get_url(self) -> str:
        return f"https://{self._ip}:{PORT}"

    def get_manual_url(self) -> str:
        """Address to type into a phone browser (alias port, also HTTPS)."""
        return f"{self._ip}:{PORT + 1}"

    @property
    def running(self) -> bool:
        return bool(self._servers)

    # ── tokens, activity and rate limiting ───────────────────────────────

    def _touch(self) -> None:
        self._last_activity = time.time()

    def _token_valid(self, tok: str) -> bool:
        exp = self._tokens.get(tok)
        if exp is None:
            return False
        if exp <= time.time():
            self._tokens.pop(tok, None)
            self._token_keys.pop(tok, None)
            return False
        self._touch()
        return True

    def _issue_token(self, session_key: str) -> str:
        tok = secrets.token_urlsafe(32)
        self._tokens[tok] = time.time() + TOKEN_TTL_SECS
        self._token_keys[tok] = session_key
        self._aes_key(session_key)
        self._touch()
        return tok

    def _recent_fails(self, ip: str) -> list[float]:
        cutoff = time.time() - FAIL_WINDOW_SECS
        lst = [t for t in self._fails.get(ip, []) if t > cutoff]
        if lst:
            self._fails[ip] = lst
        else:
            self._fails.pop(ip, None)
        return lst

    def _blocked(self, ip: str) -> bool:
        return len(self._recent_fails(ip)) >= MAX_FAILS_PER_IP

    def _record_fail(self, ip: str) -> None:
        self._fails.setdefault(ip, []).append(time.time())
        total = sum(len(self._recent_fails(k)) for k in list(self._fails))
        if total >= MAX_FAILS_TOTAL and self._pending_keys:
            self._pending_keys.clear()
            self._log("Remote: too many wrong pairing attempts — all pairing codes cancelled.")
        elif len(self._fails.get(ip, [])) == MAX_FAILS_PER_IP:
            self._log(f"Remote: blocked {ip} for 10 minutes after repeated wrong pairing codes.")

    def _purge_expired(self) -> None:
        now = time.time()
        for tok in [t for t, exp in self._tokens.items() if exp <= now]:
            self._tokens.pop(tok, None)
            self._token_keys.pop(tok, None)
        for dev in [d for d, s in self._device_sessions.items() if s.get("expires", 0) <= now]:
            self._device_sessions.pop(dev, None)
        self._pending_keys = {k: v for k, v in self._pending_keys.items() if v > now}
        for ip in list(self._fails):
            self._recent_fails(ip)

    def _log(self, text: str) -> None:
        print(f"[Dashboard] {text}")
        if self._log_callback:
            try:
                self._log_callback(f"SYS: {text}")
            except Exception:
                pass

    def _aes_key(self, session_key: str) -> bytes:
        if session_key not in self._aes_cache:
            self._aes_cache[session_key] = _derive_key(session_key)
        return self._aes_cache[session_key]

    def _decrypt(self, token: str, enc_b64: str) -> str | None:
        sk = self._token_keys.get(token)
        if not sk:
            return None
        try:
            return _decrypt_cbc(self._aes_key(sk), enc_b64)
        except Exception:
            return None

    # ── callbacks ────────────────────────────────────────────────────────

    def set_wake_callback(self, fn) -> None:
        self._wake_callback = fn

    def set_connect_callback(self, fn) -> None:
        self._connect_callback = fn

    def set_log_callback(self, fn) -> None:
        self._log_callback = fn

    # ── broadcast ────────────────────────────────────────────────────────

    async def broadcast(self, msg: dict) -> None:
        self._history.append(msg)
        if len(self._history) > 300:
            self._history = self._history[-300:]
        dead: set[WebSocket] = set()
        for ws in list(self._clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.add(ws)
        self._clients -= dead

    # ── FastAPI app ───────────────────────────────────────────────────────

    def _build_app(self) -> "FastAPI":
        app = FastAPI(docs_url=None, redoc_url=None)

        def _auth(req: Request) -> bool:
            tok = req.headers.get("authorization", "").removeprefix("Bearer ").strip()
            return bool(tok) and self._token_valid(tok)

        def _client_ip(req) -> str:
            return req.client.host if req.client else "?"

        _too_many = {"ok": False, "error": "Too many wrong codes. Wait 10 minutes and scan a new QR code."}

        # serve CryptoJS from local cache, fallback to CDN redirect
        @app.get("/static/crypto.js")
        async def serve_crypto():
            if _CRYPTOJS_FILE.exists():
                return FileResponse(str(_CRYPTOJS_FILE),
                                    media_type="application/javascript")
            from fastapi.responses import RedirectResponse
            return RedirectResponse(_CRYPTOJS_CDN)

        @app.get("/login", response_class=HTMLResponse)
        async def login_page():
            return HTMLResponse(self._login_html)

        @app.get("/", response_class=HTMLResponse)
        async def index():
            # Auth is handled client-side via sessionStorage bearer token.
            # Server-side header auth can't work here because browser navigations
            # don't send custom headers (location.href doesn't carry Authorization).
            html = (self._app_html
                    .replace("__IP__", self._ip)
                    .replace("__PORT__", str(PORT)))
            return HTMLResponse(html)

        @app.post("/login")
        async def login(req: Request):
            ip = _client_ip(req)
            if self._blocked(ip):
                return JSONResponse(_too_many, status_code=429)
            try:
                body = await req.json()
            except Exception:
                return JSONResponse({"ok": False}, status_code=400)
            entered = str(body.get("pin", "")).strip().upper()
            now     = time.time()
            if entered in self._pending_keys and self._pending_keys[entered] > now:
                del self._pending_keys[entered]          # one-time use
                tok = self._issue_token(entered)
                if self._connect_callback:
                    self._connect_callback()
                asyncio.create_task(self.broadcast(
                    {"type": "sys", "text": "Remote connection established."}
                ))
                # Bearer token in response body — no cookies needed (works on any browser/HTTP)
                return JSONResponse({"ok": True, "token": tok})
            self._record_fail(ip)
            return JSONResponse({"ok": False, "error": "Invalid or expired key"},
                                status_code=401)

        @app.get("/auto-login")
        async def auto_login(req: Request, key: str = ""):
            """QR code target — validates one-time key, creates session, redirects phone."""
            now = time.time()
            ip  = _client_ip(req)
            if self._blocked(ip):
                return JSONResponse(_too_many, status_code=429)
            key = key.strip().upper()
            if not key or key not in self._pending_keys or self._pending_keys[key] <= now:
                if key:
                    self._record_fail(ip)
                return HTMLResponse("""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width">
<style>
  body{background:#07090f;color:#dde3ed;font-family:sans-serif;
       display:flex;align-items:center;justify-content:center;height:100vh;margin:0;text-align:center}
  h2{color:#f87171;margin-bottom:12px}p{color:#5e6a7e;font-size:14px}
</style></head>
<body><div><h2>Link Expired</h2>
<p>Press <strong style="color:#dde3ed">Remote Control</strong> in JARVIS to get a new QR code.</p>
</div></body></html>""")

            del self._pending_keys[key]
            tok     = self._issue_token(key)
            dev_tok = secrets.token_urlsafe(32)
            self._device_sessions[dev_tok] = {
                "session_key": key,
                "expires": time.time() + DEVICE_TTL_SECS,
            }

            if self._connect_callback:
                self._connect_callback()
            asyncio.create_task(self.broadcast(
                {"type": "sys", "text": "Remote connection established via QR code."}
            ))

            return HTMLResponse(f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width">
<style>
  body{{background:#07090f;color:#dde3ed;font-family:sans-serif;
       display:flex;align-items:center;justify-content:center;height:100vh;margin:0;text-align:center}}
  p{{color:#5e6a7e;font-size:14px}}
</style></head>
<body>
<script>
  sessionStorage.setItem('jarvis_token','{tok}');
  sessionStorage.setItem('jarvis_key','{key}');
  localStorage.setItem('jarvis_device_token','{dev_tok}');
  setTimeout(function(){{location.replace('/')}},400);
</script>
<p>Connecting to JARVIS…</p>
</body></html>""")

        @app.post("/api/device-login")
        async def device_login_ep(req: Request):
            """Return a fresh auth token for a previously paired device token."""
            ip = _client_ip(req)
            if self._blocked(ip):
                return JSONResponse(_too_many, status_code=429)
            try:
                body = await req.json()
            except Exception:
                return JSONResponse({"ok": False}, status_code=400)
            dev_tok = (body.get("device_token") or "").strip()
            dev     = self._device_sessions.get(dev_tok) if dev_tok else None
            if dev and dev.get("expires", 0) <= time.time():
                self._device_sessions.pop(dev_tok, None)
                dev = None
            if not dev:
                if dev_tok:
                    self._record_fail(ip)
                return JSONResponse({"ok": False}, status_code=401)
            session_key = dev["session_key"]
            tok = self._issue_token(session_key)
            if self._connect_callback:
                self._connect_callback()
            asyncio.create_task(self.broadcast(
                {"type": "sys", "text": "Known device reconnected automatically."}
            ))
            return JSONResponse({"ok": True, "token": tok, "key": session_key})

        @app.post("/api/revoke-devices")
        async def revoke_devices(req: Request):
            """Invalidate all persistent device tokens (admin action)."""
            if not _auth(req):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            count = len(self._device_sessions)
            self._device_sessions.clear()
            return JSONResponse({"ok": True, "revoked": count})

        @app.post("/api/command")
        async def command(req: Request):
            if not _auth(req):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            body  = await req.json()
            token = req.headers.get("authorization", "").removeprefix("Bearer ").strip()
            enc   = body.get("enc", "")
            if enc:
                text = self._decrypt(token, enc)
                if text is None:
                    return JSONResponse({"error": "Decryption failed"}, status_code=400)
            else:
                text = (body.get("text") or "").strip()
            if text:
                await self._command_queue.put(text)
                if self._wake_callback:
                    self._wake_callback()
            return JSONResponse({"ok": True})

        @app.post("/api/wake")
        async def wake_ep(req: Request):
            if not _auth(req):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            if self._wake_callback:
                self._wake_callback()
            return JSONResponse({"ok": True})

        # ── Phone mic real-time audio → Gemini Live ──────────────────────────

        @app.websocket("/ws/phone-audio")
        async def phone_audio_ws(websocket: WebSocket, token: str = ""):
            tok = token.strip()
            if not tok or not self._token_valid(tok):
                await websocket.close(code=4001)
                return
            await websocket.accept()
            asyncio.create_task(self.broadcast(
                {"type": "sys", "text": "Phone microphone live."}
            ))
            try:
                while True:
                    data = await websocket.receive_bytes()
                    try:
                        self._phone_audio_queue.put_nowait(
                            {"data": data, "mime_type": "audio/pcm"}
                        )
                    except asyncio.QueueFull:
                        pass  # drop frame rather than block
            except WebSocketDisconnect:
                pass
            finally:
                asyncio.create_task(self.broadcast(
                    {"type": "sys", "text": "Phone microphone stopped."}
                ))

        # ── File sharing ──────────────────────────────────────────────────────

        def _safe_filename(raw: str) -> str:
            name = Path(raw).name                          # strip path components
            name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(". ")
            return name or "upload"

        if _UPLOAD_OK:
            @app.post("/api/upload")
            async def upload_file(req: Request, file: UploadFile = FastAPIFile(...)):
                if not _auth(req):
                    return JSONResponse({"error": "Unauthorized"}, status_code=401)

                safe = _safe_filename(file.filename or "upload")
                dest = self._uploads_dir / safe
                stem, suffix = Path(safe).stem, Path(safe).suffix
                counter = 1
                while dest.exists():
                    dest = self._uploads_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

                size = 0
                max_bytes = MAX_UPLOAD_MB * 1024 * 1024
                try:
                    with open(dest, "wb") as fout:
                        while True:
                            chunk = await file.read(65536)
                            if not chunk:
                                break
                            size += len(chunk)
                            if size > max_bytes:
                                fout.close()
                                dest.unlink(missing_ok=True)
                                return JSONResponse(
                                    {"error": f"File too large (max {MAX_UPLOAD_MB} MB)"},
                                    status_code=413,
                                )
                            fout.write(chunk)
                except Exception as exc:
                    try:
                        dest.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return JSONResponse({"error": str(exc)}, status_code=500)

                asyncio.create_task(self.broadcast({
                    "type": "file_received",
                    "name": dest.name,
                    "size": size,
                    "saved_to": str(self._uploads_dir),
                }))
                return JSONResponse({"ok": True, "name": dest.name, "size": size})
        else:
            @app.post("/api/upload")
            async def upload_unavailable(req: Request):
                return JSONResponse(
                    {"error": "File uploads require: pip install python-multipart"},
                    status_code=503,
                )

        @app.get("/api/files")
        async def list_files(req: Request):
            if not _auth(req):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            files = []
            try:
                for f in sorted(
                    (p for p in self._uploads_dir.iterdir() if p.is_file()),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                ):
                    files.append({"name": f.name, "size": f.stat().st_size})
            except Exception:
                pass
            return JSONResponse({"files": files})

        @app.get("/uploads/{filename}")
        async def download_file(filename: str, token: str = ""):
            # Auth via query param — browser <a download> can't send custom headers
            tok = token.strip()
            if not tok or not self._token_valid(tok):
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
            safe = re.sub(r'[/\\]', '', filename)
            path = self._uploads_dir / safe
            if not path.exists() or not path.is_file():
                return JSONResponse({"error": "Not found"}, status_code=404)
            return FileResponse(str(path), filename=safe)

        @app.websocket("/ws")
        async def ws_ep(websocket: WebSocket, token: str = ""):
            tok = token.strip()
            if not tok or not self._token_valid(tok):
                await websocket.close(code=4001)
                return
            await websocket.accept()
            self._clients.add(websocket)
            for entry in self._history[-50:]:
                try:
                    await websocket.send_json(entry)
                except Exception:
                    break
            try:
                while True:
                    data = await websocket.receive_json()
                    self._touch()
                    if data.get("type") == "command":
                        enc = data.get("enc", "")
                        t   = self._decrypt(tok, enc) if enc else (data.get("text") or "").strip()
                        if t:
                            await self._command_queue.put(t)
                            if self._wake_callback:
                                self._wake_callback()
            except WebSocketDisconnect:
                pass
            finally:
                self._clients.discard(websocket)

        return app

    # ── start / stop ──────────────────────────────────────────────────────

    async def start(self) -> tuple[bool, str]:
        """Start both HTTPS listeners if they aren't running. Returns (ok, message).

        Port 8000 serves the QR link; 8001 is the same app for typing the address
        by hand (Chrome upgrades a bare IP:PORT to https, so it needs TLS too).
        """
        async with self._start_lock:
            if self.running:
                self._touch()
                return True, "already running"
            if not _DEPS_OK:
                return False, "fastapi/uvicorn not installed. Run: pip install fastapi \"uvicorn[standard]\" cryptography"

            self._ip = _local_ip()
            loop = asyncio.get_running_loop()
            if not await loop.run_in_executor(None, _ensure_cert, self._ip):
                return False, "could not create an HTTPS certificate, so the dashboard was not started"

            busy = [p for p in (PORT, PORT + 1) if not _port_free(p)]
            if busy:
                return False, f"port {busy[0]} is already in use by another program"

            for port in (PORT, PORT + 1):
                loop.run_in_executor(None, _ensure_network_access, port)
                cfg = uvicorn.Config(
                    self.app, host="0.0.0.0", port=port, log_level="warning",
                    ssl_keyfile=str(KEY_FILE), ssl_certfile=str(CERT_FILE),
                )
                server = uvicorn.Server(cfg)
                self._servers.append(server)
                self._server_tasks.append(asyncio.create_task(self._run_server(server)))

            # Wait (briefly) until both listeners are accepting connections.
            # A listener that fails removes itself from self._servers.
            def _all_up() -> bool:
                return len(self._servers) == 2 and all(s.started for s in self._servers)
            for _ in range(100):
                if _all_up() or len(self._servers) < 2:
                    break
                await asyncio.sleep(0.05)
            if not _all_up():
                await self.stop("failed to start")
                return False, "the HTTPS server did not start"

            self._touch()
            self._watchdog_task = asyncio.create_task(self._watchdog())
            self._log(f"Remote dashboard on — https://{self._ip}:{PORT} "
                      f"(stops after {IDLE_STOP_SECS // 60} min with no phone connected).")
            return True, "started"

    async def _run_server(self, server) -> None:
        try:
            await server.serve()
        except (SystemExit, Exception) as e:
            # uvicorn calls sys.exit() on startup errors; never let that end JARVIS.
            print(f"[Dashboard] Server stopped with an error: {e!r}")
        finally:
            if server in self._servers:
                self._servers.remove(server)

    async def stop(self, reason: str = "") -> None:
        """Stop the listeners and forget every session and paired phone."""
        servers, tasks = list(self._servers), list(self._server_tasks)
        self._servers.clear()
        self._server_tasks.clear()
        # Revoke every credential first, so nothing is accepted while shutting down.
        self._tokens.clear()
        self._token_keys.clear()
        self._aes_cache.clear()
        self._device_sessions.clear()
        self._pending_keys.clear()
        for s in servers:
            s.should_exit = True
        for ws in list(self._clients):
            try:
                await ws.close(code=4002)   # phone page shows "turned off on the computer"
            except Exception:
                pass
        self._clients.clear()
        if tasks:
            await asyncio.wait(tasks, timeout=5)
        wd, self._watchdog_task = self._watchdog_task, None
        if wd and wd is not asyncio.current_task():
            wd.cancel()
        if servers:
            self._log(f"Remote dashboard off{f' ({reason})' if reason else ''}. "
                      "Press Remote Control to turn it on again.")

    async def _watchdog(self) -> None:
        """Expire old tokens and stop the server when nobody has used it for a while."""
        while self.running:
            await asyncio.sleep(30)
            self._purge_expired()
            if self._clients:
                self._touch()          # a phone is connected — keep running
                continue
            if time.time() - self._last_activity > IDLE_STOP_SECS:
                await self.stop(f"no phone connected for {IDLE_STOP_SECS // 60} minutes")
                return
