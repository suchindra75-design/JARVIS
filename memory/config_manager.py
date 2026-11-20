import json
import os
import sys
import threading
from pathlib import Path

def get_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent

BASE_DIR    = get_base_dir()
CONFIG_DIR  = BASE_DIR / "config"
CONFIG_FILE = CONFIG_DIR / "api_keys.json"

_LOCK = threading.Lock()


def ensure_config_dir() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

def config_exists() -> bool:
    return CONFIG_FILE.exists()


def update_config(changes: dict) -> dict:
    """Merge `changes` into the config file and write it atomically.

    Never drops keys it didn't change. If the existing file can't be parsed it
    is left untouched and the update is refused, so a bad read can't wipe the
    API key. The file is readable only by the current user.
    """
    ensure_config_dir()
    with _LOCK:
        data: dict = {}
        if CONFIG_FILE.exists():
            data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))   # raises on corrupt file
        data.update(changes)
        tmp = CONFIG_FILE.with_suffix(".json.tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        os.replace(tmp, CONFIG_FILE)
        try:
            os.chmod(CONFIG_FILE, 0o600)
        except OSError:
            pass
        return data


def save_api_keys(gemini_api_key: str) -> None:
    update_config({"gemini_api_key": gemini_api_key.strip()})

def load_api_keys() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"❌ Failed to load api_keys.json: {e}")
        return {}

def get_gemini_key() -> str | None:
    return load_api_keys().get("gemini_api_key")

def is_configured() -> bool:
    key = get_gemini_key()
    return bool(key and len(key) > 15)


def get_assistant_name() -> str:
    """Return the configured assistant name, or 'JARVIS' if not set."""
    return load_api_keys().get("assistant_name", "JARVIS") or "JARVIS"


def get_user_name() -> str:
    """Return the configured user name for addressing."""
    return load_api_keys().get("user_name", "")


def save_assistant_config(assistant_name: str, user_name: str) -> None:
    """Persist assistant name and user name to config."""
    update_config({
        "assistant_name": assistant_name.strip() or "JARVIS",
        "user_name": user_name.strip(),
    })


def get_brief_enabled() -> bool:
    return load_api_keys().get("morning_brief_enabled", True)


def save_brief_enabled(enabled: bool) -> None:
    update_config({"morning_brief_enabled": enabled})


def get_audio_settings() -> dict:
    """Saved audio settings (may be empty; AudioSettings fills defaults)."""
    s = load_api_keys().get("audio", {})
    return s if isinstance(s, dict) else {}


def save_audio_settings(settings: dict) -> None:
    update_config({"audio": settings})
