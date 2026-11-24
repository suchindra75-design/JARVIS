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
    """Return the configured assistant name, or 'Raya' if not set."""
    return load_api_keys().get("assistant_name", "Raya") or "Raya"


def get_user_name() -> str:
    """Return the configured user name for addressing."""
    return load_api_keys().get("user_name", "")


def save_assistant_config(assistant_name: str, user_name: str) -> None:
    """Persist assistant name and user name to config."""
    update_config({
        "assistant_name": assistant_name.strip() or "Raya",
        "user_name": user_name.strip(),
    })


# Gemini text models used by the action modules (not the Live voice model,
# which lives in main.py). Google retires model IDs over time, so they are kept
# here in one place; `text_model` / `text_model_lite` in config/api_keys.json
# override them without a code change.
DEFAULT_TEXT_MODEL      = "gemini-3-flash-preview"
DEFAULT_TEXT_MODEL_LITE = "gemini-3.5-flash-lite"


def get_text_model(lite: bool = False) -> str:
    """Model ID for generate_content calls.

    lite=True is for small, latency-sensitive jobs (intent parsing, date
    parsing, locating a click target); the default is for heavier work
    (search-grounded answers, code generation, summaries).
    """
    cfg = load_api_keys()
    key, default = (("text_model_lite", DEFAULT_TEXT_MODEL_LITE) if lite
                    else ("text_model", DEFAULT_TEXT_MODEL))
    value = cfg.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else default


# Gemini Live prebuilt voices: (id, voice, description). The voice is fixed
# when a session connects, so a change takes effect on the next start.
VOICES = [
    ("Kore", "female", "firm"),        ("Aoede", "female", "breezy"),
    ("Zephyr", "female", "bright"),    ("Leda", "female", "youthful"),
    ("Despina", "female", "smooth"),   ("Sulafat", "female", "warm"),
    ("Achernar", "female", "soft"),    ("Vindemiatrix", "female", "gentle"),
    ("Gacrux", "female", "mature"),    ("Erinome", "female", "clear"),
    ("Charon", "male", "informative"), ("Puck", "male", "upbeat"),
    ("Orus", "male", "firm"),          ("Fenrir", "male", "excitable"),
    ("Iapetus", "male", "clear"),      ("Algieba", "male", "smooth"),
]
DEFAULT_VOICE = "Charon"


def get_voice_name() -> str:
    v = load_api_keys().get("voice_name")
    return v if isinstance(v, str) and v in {n for n, _, _ in VOICES} else DEFAULT_VOICE


def save_voice_name(name: str) -> None:
    if name not in {n for n, _, _ in VOICES}:
        raise ValueError(f"Unknown voice: {name}")
    update_config({"voice_name": name})


def get_speaking_style() -> str:
    """Free-text delivery instructions (tone, pace). Empty = the prompt's default."""
    v = load_api_keys().get("speaking_style", "")
    return v.strip()[:400] if isinstance(v, str) else ""


def save_speaking_style(style: str) -> None:
    update_config({"speaking_style": (style or "").strip()[:400]})


# Personal voice gate (audio/voice_gate.py). mode: "off" | "prefer" | "only".
# "prefer" passes speech through if voice recognition isn't available;
# "only" ignores voice input then. strictness is added to the profile's
# match threshold (+ stricter, − more lenient).
VOICE_GATE_DEFAULTS = {"mode": "prefer", "follow_up_s": 20.0, "strictness": 0.0}


def get_voice_gate() -> dict:
    g = dict(VOICE_GATE_DEFAULTS)
    saved = load_api_keys().get("voice_gate")
    if isinstance(saved, dict):
        if saved.get("mode") in ("off", "prefer", "only"):
            g["mode"] = saved["mode"]
        try:
            g["follow_up_s"] = float(min(60.0, max(0.0, float(saved.get("follow_up_s", g["follow_up_s"])))))
            g["strictness"] = float(min(0.15, max(-0.15, float(saved.get("strictness", g["strictness"])))))
        except (TypeError, ValueError):
            pass
    return g


def save_voice_gate(changes: dict) -> dict:
    g = get_voice_gate()
    g.update({k: v for k, v in changes.items() if k in VOICE_GATE_DEFAULTS})
    update_config({"voice_gate": g})
    return get_voice_gate()


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
