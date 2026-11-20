import platform as _platform
import subprocess as _subprocess

# ── Nuclear: force CREATE_NO_WINDOW on EVERY subprocess call on Windows ───────
# This patches Popen itself, so no per-file flag is needed anywhere.
if _platform.system() == "Windows":
    _OrigPopen = _subprocess.Popen
    _CREATE_NO_WINDOW = getattr(_subprocess, "CREATE_NO_WINDOW", 0x08000000)

    class _Popen(_OrigPopen):
        def __init__(self, args, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _CREATE_NO_WINDOW
            kw.pop("startupinfo", None)   # drop any stale/shared STARTUPINFO
            super().__init__(args, **kw)

    _subprocess.Popen = _Popen
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import re
import threading
import time
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.genai.live import AsyncSession
    from dashboard.server import DashboardServer

from google import genai
from google.genai import types
from ui import JarvisUI
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
)

from actions.file_processor import file_processor
from actions.flight_finder     import flight_finder
from actions.open_app          import open_app
from actions.weather_report    import weather_action
from actions.send_message      import send_message
from actions.reminder          import reminder
from actions.computer_settings import computer_settings
from actions.screen_processor  import _capture_camera, _capture_screen
from actions.youtube_video     import youtube_video
from actions.desktop           import desktop_control
from actions.browser_control   import browser_control
from actions.file_controller   import file_controller
from actions.code_helper       import code_helper
from actions.dev_agent         import dev_agent
from actions.web_search        import web_search as web_search_action
from actions.computer_control  import computer_control
from actions.game_updater      import game_updater
from actions.system_monitor    import SystemMonitor, get_system_status
from actions.proactive         import ProactiveEngine
from actions.web_search        import _news as _fetch_news_sync
from memory.config_manager     import get_brief_enabled, get_audio_settings, get_voice_name, get_speaking_style, get_voice_gate
from audio.engine              import AudioEngine, AudioSettings
from actions.confirm           import DECLINED, preview, request_confirmation


# File-changing actions that need on-screen approval. Read-only actions don't.
_FILE_WRITE_ACTIONS = {
    "create_file", "create_folder", "delete", "move", "copy",
    "rename", "write", "organize_desktop",
}


def _approval_needed(name: str, args: dict) -> tuple[str, str] | None:
    """Return (title, detail) when a tool call must be approved by the user first.

    code_helper, computer_settings and desktop_control ask inside the action
    itself, because their real action is only known after intent detection.
    """
    action = str(args.get("action", "")).lower().strip()

    if name == "send_message":
        return (
            f"Send a {args.get('platform') or 'message'} message to {args.get('receiver') or '?'}",
            preview(args.get("message_text"), 800),
        )

    if name == "dev_agent":
        return (
            "Build and run a new project",
            f"{preview(args.get('description'), 500)}\n\n"
            f"Language: {args.get('language') or 'python'}\n"
            "This writes files, installs packages and runs the generated code.",
        )

    if name == "file_controller" and action in _FILE_WRITE_ACTIONS:
        lines = [f"Action: {action}", f"Path: {args.get('path') or '?'}"]
        if args.get("destination"):
            lines.append(f"Destination: {args['destination']}")
        if args.get("new_name"):
            lines.append(f"New name: {args['new_name']}")
        if args.get("content"):
            lines.append(f"Content:\n{preview(args['content'], 400)}")
        verb = "Delete" if action == "delete" else "Change"
        return (f"{verb} files: {action.replace('_', ' ')}", "\n".join(lines))

    if name == "file_processor" and action in ("run", "extract"):
        what = "Run this code file" if action == "run" else "Extract this archive"
        detail = f"File: {args.get('file_path') or '(the uploaded file)'}"
        if action == "extract":
            detail += f"\nDestination: {args.get('destination') or '(next to the archive)'}"
        return (what, detail)

    if name == "game_updater":
        if args.get("shutdown_when_done"):
            return ("Shut down the computer when the download finishes",
                    f"Game: {args.get('game_name') or 'all'} · Platform: {args.get('platform') or 'both'}")
        if action == "schedule":
            return ("Schedule a daily game update",
                    f"Every day at {args.get('hour', 3)}:{str(args.get('minute', 0)).zfill(2)}")

    return None


def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent


BASE_DIR        = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
LIVE_MODEL          = "models/gemini-2.5-flash-native-audio-preview-12-2025"

def _get_api_key() -> str:
    with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are JARVIS, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )

_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)

def _clean_transcript(text: str) -> str:    
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()


def _clean_piece(text: str) -> str:
    """A streamed transcript piece. Gemini sends word fragments whose leading
    space marks a new word ("He", "y", " Ja", "rvis"), so spaces are kept."""
    text = _CTRL_RE.sub("", text)
    return re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)


def _join_pieces(pieces: list[str]) -> str:
    return re.sub(r"\s+", " ", "".join(pieces)).strip()

TOOL_DECLARATIONS = [
    {
        "name": "open_app",
        "description": (
            "Opens any application on the computer. "
            "Use this whenever the user asks to open, launch, or start any app, "
            "website, or program. Always call this tool — never just say you opened it."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "app_name": {
                    "type": "STRING",
                    "description": "Exact name of the application (e.g. 'WhatsApp', 'Chrome', 'Spotify')"
                }
            },
            "required": ["app_name"]
        }
    },
    {
        "name": "web_search",
        "description": (
            "Searches the web. Use for ANY question about current facts, events, prices, "
            "or topics — always prefer this over guessing. "
            "Modes: 'search' (default), 'news' (latest headlines on a topic), "
            "'research' (deep comprehensive answer), 'price' (product cost lookup), "
            "'compare' (side-by-side comparison of items)."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query":  {"type": "STRING", "description": "Search query or topic"},
                "mode":   {"type": "STRING", "description": "search | news | research | price | compare"},
                "items":  {"type": "ARRAY",  "items": {"type": "STRING"}, "description": "Items to compare (compare mode)"},
                "aspect": {"type": "STRING", "description": "Comparison aspect: price | specs | reviews | features"},
            },
            "required": ["query"]
        }
    },
    {
        "name": "system_status",
        "description": (
            "Returns real-time system metrics: CPU usage, RAM, GPU load, CPU temperature, "
            "uptime, and process count. Use when the user asks about computer performance, "
            "temperature, memory, or resource usage."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "weather_report",
        "description": "Gives the weather report to user",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "city": {"type": "STRING", "description": "City name"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "send_message",
        "description": "Sends a text message via WhatsApp, Telegram, or other messaging platform.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "receiver":     {"type": "STRING", "description": "Recipient contact name"},
                "message_text": {"type": "STRING", "description": "The message to send"},
                "platform":     {"type": "STRING", "description": "Platform: WhatsApp, Telegram, etc."}
            },
            "required": ["receiver", "message_text", "platform"]
        }
    },
    {
        "name": "reminder",
        "description": "Sets a timed reminder using Task Scheduler.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "date":    {"type": "STRING", "description": "Date in YYYY-MM-DD format"},
                "time":    {"type": "STRING", "description": "Time in HH:MM format (24h)"},
                "message": {"type": "STRING", "description": "Reminder message text"}
            },
            "required": ["date", "time", "message"]
        }
    },
    {
        "name": "youtube_video",
        "description": (
            "Controls YouTube. Use for: playing videos, summarizing a video's content, "
            "getting video info, or showing trending videos."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "play | summarize | get_info | trending (default: play)"},
                "query":  {"type": "STRING", "description": "Search query for play action"},
                "save":   {"type": "BOOLEAN", "description": "Save summary to Notepad (summarize only)"},
                "region": {"type": "STRING", "description": "Country code for trending e.g. TR, US"},
                "url":    {"type": "STRING", "description": "Video URL for get_info action"},
            },
            "required": []
        }
    },
    {
        "name": "screen_process",
        "description": (
            "Captures the screen or webcam image and lets you analyze it. "
            "MUST be called when user asks what is on screen, what you see, "
            "look at camera, analyze my screen, etc. "
            "You have NO visual ability without this tool. "
            "After the image is captured it is sent directly to you — describe what you see and answer the user's question. "
            "When using camera: the live view stays open until user says close it or calls close_camera."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {"type": "STRING", "description": "'screen' to capture display, 'camera' for webcam. Default: 'screen'"},
                "text":  {"type": "STRING", "description": "The question or instruction about the captured image"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "close_camera",
        "description": (
            "Closes the live camera view shown on screen. "
            "Call when user says: close camera, stop camera, turn off camera, "
            "kamerayı kapat, kapat, creepy, etc."
        ),
        "parameters": {"type": "OBJECT", "properties": {}, "required": []}
    },
    {
        "name": "computer_settings",
        "description": (
            "Controls the computer: volume, brightness, window management, keyboard shortcuts, "
            "typing text on screen, closing apps, fullscreen, dark mode, WiFi, restart, shutdown, "
            "scrolling, tab management, zoom, screenshots, lock screen, refresh/reload page. "
            "Use for ANY single computer control command."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "The action to perform"},
                "description": {"type": "STRING", "description": "Natural language description of what to do"},
                "value":       {"type": "STRING", "description": "Optional value: volume level, text to type, etc."}
            },
            "required": []
        }
    },
    {
        "name": "browser_control",
        "description": (
            "Controls any web browser. Use for: opening websites, searching the web, "
            "clicking elements, filling forms, scrolling, screenshots, navigation, any web-based task. "
            "Simple open/search requests launch the user's own browser normally (their real profile "
            "and logged-in accounts); interactive actions (click, type, fill_form...) attach an "
            "automation browser. "
            "Always pass the 'browser' parameter when the user specifies a browser (e.g. 'open in Edge', "
            "'use Firefox', 'open Chrome'). Multiple browsers can run simultaneously."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "go_to | search | click | type | scroll | fill_form | smart_click | smart_type | get_text | get_url | press | new_tab | close_tab | screenshot | back | forward | reload | switch | list_browsers | close | close_all"},
                "browser":     {"type": "STRING", "description": "Target browser: chrome | edge | firefox | opera | operagx | brave | vivaldi | safari. Omit to use the currently active browser."},
                "url":         {"type": "STRING", "description": "URL for go_to / new_tab action"},
                "query":       {"type": "STRING", "description": "Search query for search action"},
                "engine":      {"type": "STRING", "description": "Search engine: google | bing | duckduckgo | yandex (default: google)"},
                "selector":    {"type": "STRING", "description": "CSS selector for click/type"},
                "text":        {"type": "STRING", "description": "Text to click or type"},
                "description": {"type": "STRING", "description": "Element description for smart_click/smart_type"},
                "direction":   {"type": "STRING", "description": "up | down for scroll"},
                "amount":      {"type": "INTEGER", "description": "Scroll amount in pixels (default: 500)"},
                "key":         {"type": "STRING", "description": "Key name for press action (e.g. Enter, Escape, F5)"},
                "path":        {"type": "STRING", "description": "Save path for screenshot"},
                "incognito":   {"type": "BOOLEAN", "description": "Open in private/incognito mode"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear field before typing (default: true)"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "file_controller",
        "description": "Manages files and folders: list, create, delete, move, copy, rename, read, write, find, disk usage.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "list | create_file | create_folder | delete | move | copy | rename | read | write | find | largest | disk_usage | organize_desktop | info"},
                "path":        {"type": "STRING", "description": "File/folder path or shortcut: desktop, downloads, documents, home"},
                "destination": {"type": "STRING", "description": "Destination path for move/copy"},
                "new_name":    {"type": "STRING", "description": "New name for rename"},
                "content":     {"type": "STRING", "description": "Content for create_file/write"},
                "name":        {"type": "STRING", "description": "File name to search for"},
                "extension":   {"type": "STRING", "description": "File extension to search (e.g. .pdf)"},
                "count":       {"type": "INTEGER", "description": "Number of results for largest"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "desktop_control",
        "description": "Controls the desktop: wallpaper, organize, clean, list, stats.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {"type": "STRING", "description": "wallpaper | wallpaper_url | current_wallpaper | organize | clean | list | stats"},
                "path":   {"type": "STRING", "description": "Image path for wallpaper"},
                "url":    {"type": "STRING", "description": "Image URL for wallpaper_url"},
                "mode":   {"type": "STRING", "description": "by_type or by_date for organize"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "code_helper",
        "description": "Writes, edits, explains, runs, or builds code files.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "write | edit | explain | run | build | auto (default: auto)"},
                "description": {"type": "STRING", "description": "What the code should do or what change to make"},
                "language":    {"type": "STRING", "description": "Programming language (default: python)"},
                "output_path": {"type": "STRING", "description": "Where to save the file"},
                "file_path":   {"type": "STRING", "description": "Path to existing file for edit/explain/run/build"},
                "code":        {"type": "STRING", "description": "Raw code string for explain"},
                "args":        {"type": "STRING", "description": "CLI arguments for run/build"},
                "timeout":     {"type": "INTEGER", "description": "Execution timeout in seconds (default: 30)"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "dev_agent",
        "description": "Builds complete multi-file projects from scratch: plans, writes files, installs deps, opens VSCode, runs and fixes errors.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "description":  {"type": "STRING", "description": "What the project should do"},
                "language":     {"type": "STRING", "description": "Programming language (default: python)"},
                "project_name": {"type": "STRING", "description": "Optional project folder name"},
                "timeout":      {"type": "INTEGER", "description": "Run timeout in seconds (default: 30)"},
            },
            "required": ["description"]
        }
    },
    {
        "name": "computer_control",
        "description": "Direct computer control: type, click, hotkeys, scroll, move mouse, screenshots, find elements on screen.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":      {"type": "STRING", "description": "type | smart_type | click | double_click | right_click | hotkey | press | scroll | move | copy | paste | screenshot | wait | clear_field | focus_window | screen_find | screen_click | random_data | user_data"},
                "text":        {"type": "STRING", "description": "Text to type or paste"},
                "x":           {"type": "INTEGER", "description": "X coordinate"},
                "y":           {"type": "INTEGER", "description": "Y coordinate"},
                "keys":        {"type": "STRING", "description": "Key combination e.g. 'ctrl+c'"},
                "key":         {"type": "STRING", "description": "Single key e.g. 'enter'"},
                "direction":   {"type": "STRING", "description": "up | down | left | right"},
                "amount":      {"type": "INTEGER", "description": "Scroll amount (default: 3)"},
                "seconds":     {"type": "NUMBER",  "description": "Seconds to wait"},
                "title":       {"type": "STRING",  "description": "Window title for focus_window"},
                "description": {"type": "STRING",  "description": "Element description for screen_find/screen_click"},
                "type":        {"type": "STRING",  "description": "Data type for random_data"},
                "field":       {"type": "STRING",  "description": "Field for user_data: name|email|city"},
                "clear_first": {"type": "BOOLEAN", "description": "Clear field before typing (default: true)"},
                "path":        {"type": "STRING",  "description": "Save path for screenshot"},
            },
            "required": ["action"]
        }
    },
    {
        "name": "game_updater",
        "description": (
            "THE ONLY tool for ANY Steam or Epic Games request. "
            "Use for: installing, downloading, updating games, listing installed games, "
            "checking download status, scheduling updates. "
            "ALWAYS call directly for any Steam/Epic/game request. "
            "NEVER use browser_control or web_search for Steam/Epic."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action":    {"type": "STRING",  "description": "update | install | list | download_status | schedule | cancel_schedule | schedule_status (default: update)"},
                "platform":  {"type": "STRING",  "description": "steam | epic | both (default: both)"},
                "game_name": {"type": "STRING",  "description": "Game name (partial match supported)"},
                "app_id":    {"type": "STRING",  "description": "Steam AppID for install (optional)"},
                "hour":      {"type": "INTEGER", "description": "Hour for scheduled update 0-23 (default: 3)"},
                "minute":    {"type": "INTEGER", "description": "Minute for scheduled update 0-59 (default: 0)"},
                "shutdown_when_done": {"type": "BOOLEAN", "description": "Shut down PC when download finishes"},
            },
            "required": []
        }
    },
    {
        "name": "flight_finder",
        "description": "Searches Google Flights and speaks the best options.",
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "origin":      {"type": "STRING",  "description": "Departure city or airport code"},
                "destination": {"type": "STRING",  "description": "Arrival city or airport code"},
                "date":        {"type": "STRING",  "description": "Departure date (any format)"},
                "return_date": {"type": "STRING",  "description": "Return date for round trips"},
                "passengers":  {"type": "INTEGER", "description": "Number of passengers (default: 1)"},
                "cabin":       {"type": "STRING",  "description": "economy | premium | business | first"},
                "save":        {"type": "BOOLEAN", "description": "Save results to Notepad"},
            },
            "required": ["origin", "destination", "date"]
        }
    },
    {
        "name": "shutdown_jarvis",
        "description": (
            "Shuts down the assistant completely. "
            "Call this when the user expresses intent to end the conversation, "
            "close the assistant, say goodbye, or stop Jarvis. "
            "The user can say this in ANY language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
    "name": "file_processor",
    "description": (
        "Processes any file that the user has uploaded or dropped onto the interface. "
        "Use this when the user refers to an uploaded file and wants an action on it. "
        "Supports: images (describe/ocr/resize/compress/convert), "
        "PDFs (summarize/extract_text/to_word), "
        "Word docs & text files (summarize/fix/reformat/translate), "
        "CSV/Excel (analyze/stats/filter/sort/convert), "
        "JSON/XML (validate/format/analyze), "
        "code files (explain/review/fix/optimize/run/document/test), "
        "audio (transcribe/trim/convert/info), "
        "video (trim/extract_audio/extract_frame/compress/transcribe/info), "
        "archives (list/extract), "
        "presentations (summarize/extract_text). "
        "ALWAYS call this tool when a file has been uploaded and the user gives a command about it. "
        "If the user's command is ambiguous, pick the most logical action for that file type."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "file_path": {
                "type": "STRING",
                "description": "Full path to the uploaded file. Leave empty to use the currently uploaded file."
            },
            "action": {
                "type": "STRING",
                "description": (
                    "What to do with the file. Examples by type:\n"
                    "image: describe | ocr | resize | compress | convert | info\n"
                    "pdf: summarize | extract_text | to_word | info\n"
                    "docx/txt: summarize | fix | reformat | translate_hint | word_count | to_bullet\n"
                    "csv/excel: analyze | stats | filter | sort | convert | info\n"
                    "json: validate | format | analyze | to_csv\n"
                    "code: explain | review | fix | optimize | run | document | test\n"
                    "audio: transcribe | trim | convert | info\n"
                    "video: trim | extract_audio | extract_frame | compress | transcribe | info | convert\n"
                    "archive: list | extract\n"
                    "pptx: summarize | extract_text | analyze"
                )
            },
            "instruction": {
                "type": "STRING",
                "description": "Free-form instruction if action doesn't cover it. E.g. 'translate this to Turkish', 'find all email addresses'"
            },
            "format": {
                "type": "STRING",
                "description": "Target format for conversion. E.g. 'mp3', 'pdf', 'csv', 'png'"
            },
            "width":     {"type": "INTEGER", "description": "Target width for image resize"},
            "height":    {"type": "INTEGER", "description": "Target height for image resize"},
            "scale":     {"type": "NUMBER",  "description": "Scale factor for image resize (e.g. 0.5)"},
            "quality":   {"type": "INTEGER", "description": "Quality 1-100 for image/video compress"},
            "start":     {"type": "STRING",  "description": "Start time for trim: seconds or HH:MM:SS"},
            "end":       {"type": "STRING",  "description": "End time for trim: seconds or HH:MM:SS"},
            "timestamp": {"type": "STRING",  "description": "Timestamp for video frame extraction HH:MM:SS"},
            "column":    {"type": "STRING",  "description": "Column name for CSV filter/sort"},
            "value":     {"type": "STRING",  "description": "Filter value for CSV filter"},
            "condition": {"type": "STRING",  "description": "Filter condition: equals|contains|gt|lt"},
            "ascending": {"type": "BOOLEAN", "description": "Sort order for CSV sort (default: true)"},
            "save":      {"type": "BOOLEAN", "description": "Save result to file (default: true)"},
            "destination": {"type": "STRING", "description": "Output folder for archive extract"},
        },
        "required": []
    }
},
    {
        "name": "save_memory",
        "description": (
            "Save an important personal fact about the user to long-term memory. "
            "Call this silently whenever the user reveals something worth remembering: "
            "name, age, city, job, preferences, hobbies, relationships, projects, or future plans. "
            "Do NOT call for: weather, reminders, searches, or one-time commands. "
            "Do NOT announce that you are saving — just call it silently. "
            "Values must be in English regardless of the conversation language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": (
                        "identity — name, age, birthday, city, job, language, nationality | "
                        "preferences — favorite food/color/music/film/game/sport, hobbies | "
                        "projects — active projects, goals, things being built | "
                        "relationships — friends, family, partner, colleagues | "
                        "wishes — future plans, things to buy, travel dreams | "
                        "notes — habits, schedule, anything else worth remembering"
                    )
                },
                "key":   {"type": "STRING", "description": "Short snake_case key (e.g. name, favorite_food, sister_name)"},
                "value": {"type": "STRING", "description": "Concise value in English (e.g. Fatih, pizza, older sister)"},
            },
            "required": ["category", "key", "value"]
        }
    },
]

# --- Plugin system ---


def _tool_detail(name: str, args: dict) -> str:
    """One short phrase describing a tool call, for the conversation and the stage caption."""
    for k in ("query", "app_name", "receiver", "game_name", "city", "url", "description", "task",
              "file_path", "path", "destination", "action"):
        v = args.get(k)
        if isinstance(v, str) and v.strip():
            v = v.strip().replace("\n", " ")
            return (v[:60] + "…") if len(v) > 60 else v
    return ""


class _GateReconnect(Exception):
    """Raised inside the session to reconnect with a different voice-gate mode."""


class JarvisLive:

    def __init__(self, ui: JarvisUI):
        self.ui             = ui
        self._asst_name     = "JARVIS"   # updated each session from config
        self.session: "AsyncSession | None" = None
        self.audio: "AudioEngine | None" = None     # created in run()
        self.speaker_id = None                       # SpeakerIdentity, created in run()
        self.gate = None                             # VoiceGate (audio/voice_gate.py)
        self._gate_manual = False                    # this session: we tell Gemini when turns start/end
        self._phone_turn = False
        self._quick_reconnect = False
        self._awaiting_reply = False                 # a user turn ended; Gemini hasn't finished answering
        self._gate_log_lock = threading.Lock()
        self.out_queue: "asyncio.Queue | None" = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._is_speaking         = False
        self._speaking_lock       = threading.Lock()
        self._phone_active        = False   # True while phone mic is streaming; pauses PC mic
        self._pending_vision       = None    # (img_bytes, mime_type, question, angle) to inject after tool response
        self._vision_cam_active    = False   # True if camera was opened for vision → auto-close after response
        self._vision_close_pending = False   # True after vision injected; next turn_complete closes camera
        self._vision_last_time     = 0.0     # monotonic time of last screen_process call (cooldown guard)
        self._vision_busy          = False   # True while a vision capture/inject cycle is in flight
        self._interrupted          = False   # True while draining audio after user interrupt
        self.ui.on_text_command   = self._on_text_command
        self.ui.on_remote_clicked = self._make_remote_key
        self.ui.on_interrupt      = self.interrupt
        self._turn_done_event: asyncio.Event | None = None
        self._dashboard: "DashboardServer | None" = None
        self._briefing_sent    = False          # morning briefing fires once per process
        self._sys_monitor      = SystemMonitor()  # persistent cooldown state
        self._proactive        = ProactiveEngine()
        self._last_user_speech = time.monotonic()  # updated on every user utterance

    def _make_remote_key(self):
        """Called from Qt main thread when user presses Remote Control."""
        if self._dashboard is None:
            self.ui.write_log(
                "ERR: Dashboard unavailable. "
                "Run: pip install fastapi \"uvicorn[standard]\" cryptography"
            )
            return None
        if not self._dashboard.running:
            # The server only runs while you're using it: start it now.
            if not self._loop:
                self.ui.write_log("SYS: Still starting up — try Remote Control again in a moment.")
                return None
            try:
                ok, msg = asyncio.run_coroutine_threadsafe(
                    self._dashboard.start(), self._loop
                ).result(timeout=15)
            except Exception as e:
                ok, msg = False, str(e) or type(e).__name__
            if not ok:
                self.ui.write_log(f"ERR: Remote dashboard not started — {msg}")
                return None
        key    = self._dashboard.new_key()
        url    = self._dashboard.get_url()
        manual = self._dashboard.get_manual_url()
        return url, key, f"{url}/auto-login?key={key}", manual

    def _on_text_command(self, text: str):
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns=types.Content(parts=[types.Part(text=text)]),
                turn_complete=True
            ),
            self._loop
        )

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            self._is_speaking = value
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")

    def interrupt(self) -> None:
        """Stop JARVIS mid-speech: flush the speaker within ~10 ms and open the mic."""
        self._interrupted = True
        if self.audio:
            self.audio.flush()
            print("[JARVIS] ✋ Interrupted — playback flushed")
        self.set_speaking(False)
        if self._turn_done_event:
            self._turn_done_event.clear()
        self.ui.write_log("SYS: Interrupted — listening...")

    def speak(self, text: str):
        if not self._loop or not self.session:
            return
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns=types.Content(parts=[types.Part(text=text)]),
                turn_complete=True
            ),
            self._loop
        )

    def speak_error(self, tool_name: str, error: str | Exception):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Sir, {tool_name} encountered an error. {short}")

    def _build_config(self) -> types.LiveConnectConfig:
        from datetime import datetime

        # Load customization from config
        try:
            _cfg = json.loads(open(API_CONFIG_PATH, encoding="utf-8").read())
            self._asst_name = (_cfg.get("assistant_name") or "JARVIS").strip()
            _user_name = (_cfg.get("user_name") or "").strip()
        except Exception:
            self._asst_name = "JARVIS"
            _user_name = ""

        memory     = load_memory()
        mem_str    = format_memory_for_prompt(memory)
        sys_prompt = _load_system_prompt()

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {time_str}\n"
            f"Use this to calculate exact times for reminders.\n\n"
        )

        # Identity injection — overrides any hardcoded name in prompt.txt
        _addr = (f"ADDRESS: Always call the user '{_user_name}'."
                 if _user_name
                 else "ADDRESS: When speaking Turkish → always say \"efendim\". "
                      "When speaking English → say \"sir\". Never mix languages.")
        identity_ctx = (
            f"[IDENTITY]\n"
            f"Your name is {self._asst_name}. "
            f"Always refer to yourself as {self._asst_name}.\n"
            f"{_addr}\n\n"
        )

        parts = [time_ctx, identity_ctx]
        _style = get_speaking_style()
        if _style:
            parts.append(
                "[VOICE & TONE — overrides any tone guidance further down]\n"
                f"{_style}\n"
                "Always speak clearly and at a relaxed pace, finishing each word.\n\n"
            )
        if mem_str:
            parts.append(mem_str)
        parts.append(sys_prompt)

        manual_turns = self._configure_gate()
        extra = {}
        if manual_turns:
            # The voice gate decides when the user is talking; Gemini's own
            # server-side speech detection would react to everything it hears.
            extra["realtime_input_config"] = types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(disabled=True))

        return types.LiveConnectConfig(
            **extra,
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            system_instruction="\n".join(parts),
            tools=[{"function_declarations": TOOL_DECLARATIONS}],
            session_resumption=types.SessionResumptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=get_voice_name()
                    )
                )
            ),
        )

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})

        print(f"[JARVIS] 🔧 {name}  {args}")
        self.ui.set_state("THINKING")
        call_id = str(getattr(fc, "id", "") or f"{name}-{time.monotonic_ns()}")
        if hasattr(self.ui, "tool_event"):
            self.ui.tool_event(call_id, name, "started", _tool_detail(name, args))

        if name == "save_memory":
            if hasattr(self.ui, "tool_event"):
                self.ui.tool_event(call_id, name, "done")
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 save_memory: {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True}
            )

        loop   = asyncio.get_event_loop()
        result = "Done."

        # Risky actions wait for the user to click Allow on screen.
        approval = _approval_needed(name, args)
        if approval:
            title, detail = approval
            if hasattr(self.ui, "tool_event"):
                self.ui.tool_event(call_id, name, "approval")
            approved = await loop.run_in_executor(
                None, lambda: request_confirmation(self.ui, title, detail)
            )
            if not approved:
                print(f"[JARVIS] 🚫 {name} declined by user")
                if hasattr(self.ui, "tool_event"):
                    self.ui.tool_event(call_id, name, "declined")
                if not self.ui.muted:
                    self.ui.set_state("LISTENING")
                return types.FunctionResponse(
                    id=fc.id, name=name, response={"result": DECLINED}
                )

        try:
            if name == "open_app":
                r = await loop.run_in_executor(None, lambda: open_app(parameters=args, response=None, player=self.ui))
                result = r or f"Opened {args.get('app_name')}."

            elif name == "weather_report":
                r = await loop.run_in_executor(None, lambda: weather_action(parameters=args, player=self.ui))
                result = r or "Weather delivered."

            elif name == "browser_control":
                r = await loop.run_in_executor(None, lambda: browser_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "file_controller":
                r = await loop.run_in_executor(None, lambda: file_controller(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "send_message":
                r = await loop.run_in_executor(None, lambda: send_message(parameters=args, response=None, player=self.ui, session_memory=None))
                result = r or f"Message sent to {args.get('receiver')}."

            elif name == "reminder":
                r = await loop.run_in_executor(None, lambda: reminder(parameters=args, response=None, player=self.ui))
                result = r or "Reminder set."

            elif name == "youtube_video":
                r = await loop.run_in_executor(None, lambda: youtube_video(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "screen_process":
                import time as _t_mod
                _now = _t_mod.monotonic()
                _cooldown = 4.0  # seconds — covers echo window after speaking ends
                if self._vision_busy or (_now - self._vision_last_time) < _cooldown:
                    _wait = max(0, _cooldown - (_now - self._vision_last_time))
                    print(f"[Vision] ⏳ Cooldown active ({_wait:.1f}s remaining) — ignoring duplicate call")
                    result = "Vision is still processing the previous request. I will not call this again."
                else:
                    self._vision_busy      = True
                    self._vision_last_time = _now
                    angle     = args.get("angle", "screen").lower()
                    user_text = args.get("text", "What do you see?")
                    if angle == "camera":
                        img_b, mime_t = await loop.run_in_executor(None, _capture_camera)
                        self.ui.start_camera_stream()
                        self._vision_cam_active = True
                        print(f"[Vision] 📷 Camera: {len(img_b):,} bytes")
                        _stall = "camera"
                    else:
                        img_b, mime_t = await loop.run_in_executor(None, _capture_screen)
                        print(f"[Vision] 🖥️  Screen: {len(img_b):,} bytes")
                        _stall = "screen"
                    self._pending_vision = (img_b, mime_t, user_text, angle)
                    result = (
                        f"[VISION_ACTIVE] {_stall.capitalize()} captured. "
                        f"Immediately say ONE short natural sentence in the user's own language, "
                        f"telling them you are looking at their {_stall} right now. "
                        f"Do NOT describe or guess content — the actual image arrives in the NEXT message."
                    )

            elif name == "close_camera":
                self.ui.stop_camera_stream()
                result = "Camera closed."

            elif name == "computer_settings":
                r = await loop.run_in_executor(None, lambda: computer_settings(parameters=args, response=None, player=self.ui))
                result = r or "Done."

            elif name == "desktop_control":
                r = await loop.run_in_executor(None, lambda: desktop_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "code_helper":
                r = await loop.run_in_executor(None, lambda: code_helper(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "dev_agent":
                r = await loop.run_in_executor(None, lambda: dev_agent(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "web_search":
                r = await loop.run_in_executor(None, lambda: web_search_action(parameters=args, player=self.ui))
                result = r or "Done."
                # Mirror results to the on-screen content panel
                _mode = args.get("mode", "search")
                if r and not r.startswith("No results") and not r.startswith("Search failed"):
                    _query = args.get("query") or ", ".join(args.get("items", []))
                    _label = f"{_mode.upper()} — {_query[:38]}" if _query else _mode.upper()
                    self.ui.show_content(_label, r)
            elif name == "file_processor":
                if not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                r = await loop.run_in_executor(
                    None,
                    lambda: file_processor(parameters=args, player=self.ui, speak=self.speak)
                )
                result = r or "Done."

            elif name == "computer_control":
                r = await loop.run_in_executor(None, lambda: computer_control(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "game_updater":
                r = await loop.run_in_executor(None, lambda: game_updater(parameters=args, player=self.ui, speak=self.speak))
                result = r or "Done."

            elif name == "flight_finder":
                r = await loop.run_in_executor(None, lambda: flight_finder(parameters=args, player=self.ui))
                result = r or "Done."

            elif name == "system_status":
                r = await loop.run_in_executor(None, get_system_status)
                result = str(r)

            elif name == "shutdown_jarvis":
                self.ui.write_log("SYS: Shutdown requested.")
                self.speak("Goodbye, sir.")
                def _shutdown():
                    import time, os
                    time.sleep(1)
                    os._exit(0)
                threading.Thread(target=_shutdown, daemon=True).start()

            else:
                result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if hasattr(self.ui, "tool_event"):
            failed = result.startswith(f"Tool '{name}' failed") or result == DECLINED
            self.ui.tool_event(call_id, name, "failed" if failed else "done",
                               result[:200] if failed else "")

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        print(f"[JARVIS] 📤 {name} → {result[:80]}")
        return types.FunctionResponse(
            id=fc.id, name=name,
            response={"result": result}
        )

    async def _send_realtime(self):
        while True:
            out_queue = self.out_queue
            session = self.session
            if out_queue and session:
                msg = await out_queue.get()
                act = msg.get("activity") if isinstance(msg, dict) else None
                if act == "start":
                    await session.send_realtime_input(activity_start=types.ActivityStart())
                elif act in ("end", "cancel"):
                    # "cancel" closes a turn opened early that turned out not to be
                    # the user: it has no audio, and Gemini doesn't answer it.
                    await session.send_realtime_input(activity_end=types.ActivityEnd())
                    if act == "end":
                        self._awaiting_reply = True
                        self._awaiting_since = time.monotonic()
                else:
                    await session.send_realtime_input(media=msg)
            else:
                await asyncio.sleep(0.1)

    # ── audio engine glue ───────────────────────────────────────────────────
    # The AudioEngine (audio/engine.py) owns the mic, speaker and all
    # processing on its own threads. These hooks only move bytes between it
    # and the Gemini session.

    def _mic_allowed(self) -> bool:
        """Called on the audio thread: may the PC mic be sent to Gemini right now?
        Not while muted, while the phone is talking, or while the user is reading
        enrolment / test phrases in Settings (they'd be taken as commands)."""
        return (not self.ui.muted and not self._phone_active
                and not getattr(self.ui._win, "_voice_setup", False))

    def _on_mic_chunk(self, data: bytes) -> None:
        """Called on the audio thread with 20 ms of processed 16 kHz mic audio."""
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._enqueue_mic, data)

    def _enqueue_mic(self, data: bytes) -> None:
        self._put_out({"data": data, "mime_type": "audio/pcm"})

    def _put_out(self, item: dict) -> None:
        """Queue for Gemini. When full, drop the oldest audio — never a turn marker."""
        q = self.out_queue
        if q is None:
            return
        if q.full():
            items = []
            while not q.empty():
                items.append(q.get_nowait())
            for i, it in enumerate(items):
                if not (isinstance(it, dict) and "activity" in it):
                    del items[i]
                    break
            for it in items:
                q.put_nowait(it)
        try:
            q.put_nowait(item)
        except asyncio.QueueFull:
            pass

    # ── personal voice gate ─────────────────────────────────────────────────

    def _gate_wanted(self) -> bool:
        if self.audio is None or self.speaker_id is None:
            return False
        mode = get_voice_gate()["mode"]
        if mode == "off":
            return False
        return self.speaker_id.status == "ready" or mode == "only"

    def _configure_gate(self) -> bool:
        """Install or remove the gate for the session about to start.
        Returns True when Gemini must be told when turns start and end."""
        want = self._gate_wanted()
        if want:
            if self.gate is None:
                from audio.voice_gate import VoiceGate
                self.gate = VoiceGate(self.speaker_id, on_audio=self._on_mic_chunk,
                                      on_activity=self._on_mic_activity, on_event=self._on_gate_event,
                                      open_early=self._gate_may_open_early)
                self.gate.warm_up()
            self._tune_gate()
            self.gate.reset()
            self.audio.gate = self.gate
            print(f"[VoiceGate] On ({self.gate.policy}): only your voice is sent to Gemini.")
        elif self.audio is not None:
            self.audio.gate = None
        self._gate_manual = want
        self._phone_turn = False
        self._awaiting_reply = False
        return want

    def _gate_may_open_early(self) -> bool:
        """Audio thread: opening a turn now must not interrupt an answer in progress."""
        if self._awaiting_reply and time.monotonic() - getattr(self, "_awaiting_since", 0) > 30:
            self._awaiting_reply = False                 # no turn_complete came; don't stay late forever
        return not self._awaiting_reply and not self._is_speaking

    def _tune_gate(self) -> None:
        g = self.gate
        if g is None:
            return
        cfg = get_voice_gate()
        g.policy = cfg["mode"] if cfg["mode"] in ("prefer", "only") else "prefer"
        g.follow_up_s = cfg["follow_up_s"]
        g.threshold_offset = cfg["strictness"]

    def _on_mic_activity(self, kind: str) -> None:
        """Audio thread: the gate says a turn of the user's speech started/ended."""
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._put_out, {"activity": kind})

    def _on_gate_event(self, ev) -> None:
        """Audio thread: the gate decided about an utterance."""
        score = f"{ev.score:.2f}" if ev.score is not None else "–"
        thr = f"{ev.threshold:.2f}" if ev.threshold is not None else "–"
        print(f"[VoiceGate] {ev.action:<8} {ev.label:<12} score {score} / {thr}  "
              f"speech {ev.speech_s:.1f}s  decided {ev.decide_ms:.0f} ms (model {ev.verify_ms:.0f} ms)")
        if ev.label in ("you", "unknown", "uncertain"):
            self.ui.set_speaker(ev.label, ev.score)
        if hasattr(self.ui, "gate_event"):
            self.ui.gate_event(ev.action, ev.label, ev.score)
        self._log_gate_event(ev)

    def _log_gate_event(self, ev) -> None:
        """Decision log for measuring accuracy (numbers only — no audio, no words)."""
        try:
            path = BASE_DIR / "memory" / "voice_gate_log.csv"
            new = not path.exists()
            with self._gate_log_lock:
                fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
                with os.fdopen(fd, "a", encoding="utf-8") as f:
                    if new:
                        f.write("time,action,label,score,threshold,speech_s,decide_ms,verify_ms\n")
                    f.write(f"{datetime.now().isoformat(timespec='seconds')},{ev.action},{ev.label},"
                            f"{'' if ev.score is None else round(ev.score, 4)},"
                            f"{'' if ev.threshold is None else round(ev.threshold, 4)},"
                            f"{ev.speech_s:.2f},{ev.decide_ms:.0f},{ev.verify_ms:.0f}\n")
        except Exception as e:
            print(f"[VoiceGate] log failed: {e}")

    async def _watch_gate(self):
        """Reconnect when the gate should switch on/off (enrolled, deleted, mode
        changed); apply strictness / follow-up changes live."""
        while True:
            await asyncio.sleep(1.0)
            if self._gate_wanted() != self._gate_manual:
                print("[VoiceGate] Voice filter setting changed — reconnecting to apply it.")
                self.ui.write_log("SYS: Voice filter changed — reconnecting.")
                self._quick_reconnect = True
                raise _GateReconnect()
            if self._gate_manual:
                self._tune_gate()

    async def _listen_audio(self):
        """Keeps the session's task layout; capture itself runs in the AudioEngine."""
        print("[JARVIS] 🎤 Mic via AudioEngine")
        while True:
            await asyncio.sleep(3600)

    async def _receive_audio(self):
        print("[JARVIS] 👂 Recv started")
        out_buf, in_buf = [], []

        try:
            while True:
                session = self.session
                if not session:
                    await asyncio.sleep(0.1)
                    continue

                def _post_user_turn():
                    """Show what the user said in the conversation and on the phone."""
                    full = _join_pieces(in_buf)
                    in_buf.clear()
                    if not full:
                        return
                    print(f"[JARVIS] Heard: {full[:120]}")
                    if hasattr(self.ui, "add_message"):
                        self.ui.add_message("user", full, "voice")
                    else:
                        self.ui.write_log(f"You: {full}")
                    if self._dashboard:
                        asyncio.create_task(self._dashboard.broadcast({
                            "type": "log", "speaker": "user",
                            "text": full,
                            "ts": datetime.now().isoformat(),
                        }))

                async for response in session.receive():

                    if response.data:
                        if self._interrupted:
                            pass  # discard: interrupted
                        else:
                            if self._turn_done_event and self._turn_done_event.is_set():
                                self._turn_done_event.clear()
                            if in_buf:
                                _post_user_turn()              # reply started: show what was heard now
                            if self.audio:
                                self.audio.play(response.data)     # non-blocking

                    if response.server_content:
                        sc = response.server_content

                        # Gemini stopped its answer because it heard the user:
                        # stop the speaker now instead of finishing the queue.
                        if getattr(sc, "interrupted", False) and self.audio:
                            self.audio.flush()
                            self.set_speaking(False)

                        if sc.output_transcription and sc.output_transcription.text:
                            txt = _clean_piece(sc.output_transcription.text)
                            if txt and txt != (out_buf[-1] if out_buf else ""):
                                if in_buf:
                                    _post_user_turn()
                                out_buf.append(txt)
                                if hasattr(self.ui, "set_speaking_text"):
                                    self.ui.set_speaking_text(_join_pieces(out_buf))

                        if sc.input_transcription and sc.input_transcription.text:
                            txt = _clean_piece(sc.input_transcription.text)
                            if txt:
                                in_buf.append(txt)
                                self._last_user_speech = time.monotonic()
                                if hasattr(self.ui, "set_hearing_text"):
                                    self.ui.set_hearing_text(_join_pieces(in_buf))

                        if sc.turn_complete:
                            self._awaiting_reply = False
                            if self._turn_done_event:
                                self._turn_done_event.set()

                            # If this turn_complete ends an interrupted response, clear the
                            # flag and skip all further processing for that turn.
                            if self._interrupted:
                                self._interrupted = False
                                in_buf  = []
                                out_buf = []
                                continue

                            _post_user_turn()

                            full_out = _join_pieces(out_buf)
                            if full_out:
                                if hasattr(self.ui, "add_message"):
                                    self.ui.add_message("assistant", full_out)
                                else:
                                    self.ui.write_log(f"{self._asst_name}: {full_out}")
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "jarvis",
                                        "text": full_out,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            out_buf = []

                            # Vision injection: model finished tool-response turn → now send the image
                            if self._pending_vision and session:
                                img_b, mime_t, question, angle = self._pending_vision
                                self._pending_vision = None
                                print(f"[Vision] 📤 {len(img_b):,} bytes (angle={angle}) → main session")
                                await session.send_client_content(
                                    turns=types.Content(
                                        parts=[
                                            types.Part(
                                                inline_data=types.Blob(
                                                    mime_type=mime_t,
                                                    data=img_b
                                                )
                                            ),
                                            types.Part(text=question),
                                        ]
                                    ),
                                    turn_complete=True,
                                )
                                # Mark next turn_complete behaviour depending on angle
                                if self._vision_cam_active:
                                    # Camera: keep busy until JARVIS finishes speaking the answer
                                    self._vision_cam_active    = False
                                    self._vision_close_pending = True
                                else:
                                    # Screen-only: no camera to close; release busy flag now
                                    self._vision_busy = False
                            elif self._vision_close_pending:
                                # This turn_complete IS the vision answer — close camera + release busy flag
                                self._vision_close_pending = False
                                self._vision_busy = False
                                async def _cam_close():
                                    await asyncio.sleep(2.0)
                                    self.ui.stop_camera_stream()
                                asyncio.create_task(_cam_close())

                    if response.tool_call:
                        fn_responses = []
                        for fc in (response.tool_call.function_calls or []):
                            print(f"[JARVIS] 📞 {fc.name}")
                            fr = await self._execute_tool(fc)
                            fn_responses.append(fr)
                        await session.send_tool_response(
                            function_responses=fn_responses
                        )
        except Exception as e:
            print(f"[JARVIS] ❌ Recv: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        """Mirror the AudioEngine's real playback state into the UI (SPEAKING/LISTENING).

        Playback itself happens on the engine's audio thread; this only watches it.
        """
        print("[JARVIS] 🔊 Playback via AudioEngine")
        speaking = False
        try:
            while True:
                await asyncio.sleep(0.05)
                playing = bool(self.audio and self.audio.is_playing)
                if playing and not speaking and not self._interrupted:
                    speaking = True
                    self.set_speaking(True)
                elif not playing and speaking:
                    speaking = False
                    self.set_speaking(False)
                if (not playing and self._turn_done_event
                        and self._turn_done_event.is_set()):
                    self._turn_done_event.clear()
        finally:
            self.set_speaking(False)

    # ── Morning briefing ────────────────────────────────────────────────────────

    async def _send_startup_briefing(self) -> None:
        """
        Two-phase briefing optimized for speed:
          Phase 1 — instant greeting (no tools) → speech starts in <1s
          Phase 2 — news pre-fetched in a background thread while Phase 1 plays,
                    delivered as ready text (no Gemini tool-call round-trip) and
                    shown on the UI content panel. Waits for turn_complete event
                    instead of a fixed sleep so there is no unnecessary gap.
        """
        memory   = load_memory()
        identity = memory.get("identity", {})

        def _val(k: str) -> str:
            e = identity.get(k, {})
            return (e.get("value", "") if isinstance(e, dict) else str(e)).strip()

        lang = _val("language")
        name = _val("name")
        time_str = datetime.now().strftime("%H:%M")

        # Start fetching news immediately — runs in parallel while phase 1 plays
        loop = asyncio.get_event_loop()
        news_future = loop.run_in_executor(None, _fetch_news_sync, "top world news today")

        await asyncio.sleep(0.3)
        if not self.session:
            return

        # ── Phase 1: instant greeting ─────────────────────────────────────────
        lang_clause = f" Respond in {lang}." if lang else ""
        name_clause = f" Address the user as {name}." if name else ""
        p1 = (
            f"Greet the user, mention it is {time_str}, and say you are fetching today's news now. "
            f"One short sentence only. Do not call any tools.{lang_clause}{name_clause}"
        )

        # Clear the turn-done event so we can wait for Phase 1 to finish
        if self._turn_done_event:
            self._turn_done_event.clear()

        await self.session.send_client_content(
            turns=types.Content(parts=[types.Part(text=p1)]),
            turn_complete=True,
        )
        self.ui.write_log("SYS: Briefing phase 1 (greeting) sent.")

        # ── Phase 2: fire as soon as Phase 1 audio is done ───────────────────
        async def _deliver_news():
            try:
                lang_str = f" Respond in {lang}." if lang else ""

                # Wait for news fetch (already running) and Phase 1 turn-complete
                # in parallel — whichever takes longer determines the wait time
                news_done   = asyncio.wrap_future(news_future)
                turn_waited = False
                if self._turn_done_event:
                    try:
                        await asyncio.wait_for(self._turn_done_event.wait(), timeout=6.0)
                        turn_waited = True
                    except asyncio.TimeoutError:
                        pass

                # If turn_complete didn't fire (timeout), give a small buffer
                if not turn_waited:
                    await asyncio.sleep(1.0)

                try:
                    news_text = await asyncio.wait_for(news_done, timeout=4.0)
                except Exception:
                    news_text = ""

                if not self.session:
                    return

                if news_text and len(news_text) > 60:
                    # Show on UI content panel immediately
                    self.ui.show_content("NEWS — top world news today", news_text)

                    p2 = (
                        f"[BRIEFING] Here are today's top news headlines:\n{news_text}\n\n"
                        "Pick ONE headline, summarise it in one sentence, then say the full list "
                        f"is displayed on screen. Do not call any tools.{lang_str}"
                    )
                else:
                    p2 = (
                        "News headlines could not be fetched right now. "
                        f"Let the user know briefly.{lang_str}"
                    )

                await self.session.send_client_content(
                    turns=types.Content(parts=[types.Part(text=p2)]),
                    turn_complete=True,
                )
                self.ui.write_log("SYS: Briefing phase 2 (news) sent.")
            except Exception as e:
                print(f"[Briefing] Phase 2 error: {e}")
                self.ui.write_log(f"SYS: Briefing phase 2 failed: {e}")

        asyncio.create_task(_deliver_news())

    # ── System monitor ──────────────────────────────────────────────────────────

    async def _run_system_monitor(self) -> None:
        """Background task: voice alerts when metrics exceed thresholds."""
        while True:
            await asyncio.sleep(10)
            alert = await asyncio.to_thread(self._sys_monitor.check)
            if alert and self.session:
                try:
                    await self.session.send_client_content(
                        turns=types.Content(parts=[types.Part(text=alert)]),
                        turn_complete=True,
                    )
                except Exception as e:
                    print(f"[Monitor] ⚠️ Could not send alert: {e}")

    # ── Proactive mode ──────────────────────────────────────────────────────────

    async def _run_proactive_mode(self) -> None:
        """
        Background task: periodically checks if the user has been silent long enough,
        then hands time + memory context to Gemini so it can decide what (if anything)
        to say proactively. No hardcoded rules — Gemini makes the call.
        """
        while True:
            await asyncio.sleep(60)   # evaluate once per minute

            if not self.session:
                continue

            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking:
                continue

            if not self._proactive.should_trigger(self._last_user_speech):
                continue

            self._proactive.mark_triggered()

            try:
                memory = await asyncio.to_thread(load_memory)
                prompt = self._proactive.build_prompt(memory)
                await self.session.send_client_content(
                    turns=types.Content(parts=[types.Part(text=prompt)]),
                    turn_complete=True,
                )
                self.ui.write_log("SYS: Proactive check-in.")
            except Exception as e:
                print(f"[Proactive] ⚠️ {e}")

    # ── Phone audio relay ────────────────────────────────────────────────────────

    async def _relay_phone_audio(self) -> None:
        """Forward phone mic PCM chunks from dashboard queue into the Gemini Live session."""
        while True:
            dashboard = self._dashboard
            if not dashboard:
                await asyncio.sleep(0.5)
                continue
            q = dashboard._phone_audio_queue
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No audio for 1 s → phone mic inactive, give PC mic back
                self._phone_active = False
                if self._phone_turn:
                    self._phone_turn = False
                    self._put_out({"activity": "end"})
                continue
            self._phone_active = True   # phone is streaming — silence PC mic
            with self._speaking_lock:
                speaking = self._is_speaking
            if not speaking and not self.ui.muted and self.out_queue:
                # The phone is a signed-in device, so its audio skips the voice
                # check; with the gate on, Gemini still needs turn markers.
                if self._gate_manual and not self._phone_turn:
                    self._phone_turn = True
                    self._put_out({"activity": "start"})
                self._put_out(chunk)

    def _on_phone_connected(self) -> None:
        self.ui.write_log("SYS: Phone connected via Remote Dashboard.")
        self.ui.notify_phone_connected()

    # ── voice identity + remote status (UI support) ────────────────────────

    def _on_utterance(self, pcm) -> None:
        """Audio thread: a finished stretch of the user's speech. Check who it was, off-thread."""
        sid = getattr(self, "speaker_id", None)
        if sid is None or sid.status != "ready":
            return
        if not getattr(self.ui._win, "_show_speaker", True):
            return
        sid.submit(pcm, lambda v: self.ui.set_speaker(v.label, v.score))

    def _remote_status(self) -> dict:
        d = self._dashboard
        if d is None:
            return {"available": False, "reason": "The phone dashboard isn't available (fastapi/uvicorn missing)."}
        return {"available": True, "running": d.running, "phones": len(d._clients),
                "remembered": len(d._device_sessions), "url": d.get_url() if d.running else ""}

    def _forget_phones(self) -> int:
        d = self._dashboard
        if d is None:
            return 0
        n = len(d._device_sessions)
        d._device_sessions.clear()
        return n

    def _stop_remote(self) -> None:
        d = self._dashboard
        if d is not None and self._loop is not None:
            asyncio.run_coroutine_threadsafe(d.stop("turned off from Settings"), self._loop)

    async def _watch_remote(self) -> None:
        last = None
        while True:
            await asyncio.sleep(2)
            d = self._dashboard
            if d is None:
                continue
            cur = (d.running, len(d._clients))
            if cur != last:
                last = cur
                self.ui.update_remote(*cur)

    # ── dashboard command relay ─────────────────────────────────────────────

    async def _process_dashboard_commands(self) -> None:
        while True:
            dashboard = self._dashboard
            if not dashboard:
                await asyncio.sleep(0.5)
                continue
            try:
                text = await asyncio.wait_for(
                    dashboard._command_queue.get(), timeout=0.5
                )
                if not text:
                    continue
                # Wait up to 8s for session to become ready after a wake
                for _ in range(80):
                    if self.session:
                        break
                    await asyncio.sleep(0.1)
                if self.session:
                    await self.session.send_client_content(
                        turns=types.Content(parts=[types.Part(text=text)]),
                        turn_complete=True,
                    )
                    self.ui.write_log(f"[Web]: {text}")
                else:
                    print(f"[Dashboard] Dropped command (no session): {text}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"[Dashboard] Command error: {e}")
                await asyncio.sleep(0.5)

    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self):
        self._loop = asyncio.get_event_loop()

        # Phone dashboard (optional — needs: pip install fastapi "uvicorn[standard]" cryptography).
        # Prepared here but NOT listening: it starts when you press Remote Control
        # and stops itself when no phone has used it for a while.
        self._bg_tasks: set[asyncio.Task] = set()

        # Audio: one engine for the whole process (survives Gemini reconnects).
        # If it can't start, JARVIS still runs text-only rather than crashing.
        try:
            self.audio = AudioEngine(
                AudioSettings.from_dict(get_audio_settings()),
                on_chunk=self._on_mic_chunk,
                should_send=self._mic_allowed,
            )
            self.audio.start()
            if hasattr(self.ui, "attach_audio"):
                self.ui.attach_audio(self.audio)
            for err in (self.audio.input_error, self.audio.output_error):
                if err:
                    self.ui.write_log(f"ERR: Audio — {err}")
        except Exception as e:
            print(f"[Audio] Engine failed to start: {e}")
            traceback.print_exc()
            self.ui.write_log(f"ERR: Audio engine failed to start — {e}")
        # Voice identity (display only): check each finished utterance in the background.
        try:
            from audio.speaker_id import SpeakerIdentity
            self.speaker_id = SpeakerIdentity()
            if hasattr(self.ui, "attach_speaker_id"):
                self.ui.attach_speaker_id(self.speaker_id)
            if self.audio:
                self.audio.on_utterance = self._on_utterance
        except Exception as e:
            print(f"[SpeakerID] Disabled: {e}")
            self.speaker_id = None

        try:
            from dashboard.server import DashboardServer
            self._dashboard = DashboardServer()
            self._dashboard.set_connect_callback(self._on_phone_connected)
            self._dashboard.set_log_callback(self.ui.write_log)
            # Runs for the whole lifetime, not just inside an active session
            t = asyncio.create_task(self._process_dashboard_commands())
            self._bg_tasks.add(t)
            t.add_done_callback(self._bg_tasks.discard)
            if hasattr(self.ui, "set_remote_providers"):
                self.ui.set_remote_providers(self._remote_status, self._forget_phones, self._stop_remote)
                t2 = asyncio.create_task(self._watch_remote())
                self._bg_tasks.add(t2)
                t2.add_done_callback(self._bg_tasks.discard)
        except Exception as e:
            print(f"[Dashboard] Disabled: {e}")
            self._dashboard = None

        while True:
            try:
                print("[JARVIS] Connecting...")
                if hasattr(self.ui, "set_connection"):
                    self.ui.set_connection("connecting")
                self.ui.set_state("THINKING")
                config = self._build_config()

                # Fresh client on every reconnect — avoids stale HTTP session state
                client = genai.Client(
                    api_key=_get_api_key(),
                    http_options={"api_version": "v1beta"}
                )

                async with (
                    client.aio.live.connect(model=LIVE_MODEL, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session          = session
                    self.out_queue        = asyncio.Queue(maxsize=200)
                    self._turn_done_event = asyncio.Event()

                    # Reset transient state that must not carry over from a previous session
                    self._pending_vision       = None
                    self._vision_cam_active    = False
                    self._vision_close_pending = False
                    self._vision_busy          = False
                    self._vision_last_time     = 0.0
                    self._interrupted          = False

                    print("[JARVIS] Connected.")
                    if hasattr(self.ui, "set_connection"):
                        self.ui.set_connection("connected")
                    self.ui.set_state("LISTENING")
                    self.ui.write_log("SYS: JARVIS online.")

                    if self._dashboard:
                        await self._dashboard.broadcast({"type": "status", "state": "active"})

                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())
                    tg.create_task(self._run_system_monitor())
                    tg.create_task(self._watch_gate())
                    tg.create_task(self._run_proactive_mode())
                    if self._dashboard:
                        tg.create_task(self._relay_phone_audio())

                    # Morning briefing — fires once per process launch (if enabled)
                    if not self._briefing_sent and get_brief_enabled():
                        self._briefing_sent = True
                        tg.create_task(self._send_startup_briefing())

            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except BaseException as e:
                # Catches both Exception and BaseExceptionGroup (Python 3.11+
                # TaskGroup raises BaseExceptionGroup when tasks are cancelled
                # externally, which `except Exception` would miss, letting the
                # exception escape the while-loop and causing asyncio.run() to
                # start shutdown — resulting in "executor after shutdown" errors).
                err_str = str(e)
                if self._quick_reconnect:
                    err_str = ""                          # planned: voice-gate mode change
                else:
                    print(f"[JARVIS] Error ({type(e).__name__}): {e}")
                    traceback.print_exc()

                # Invalid API key — stop hammering the API, prompt re-configuration
                if "API key not valid" in err_str or "1007" in err_str:
                    self.ui.write_log("ERR: API key invalid — please re-enter your key.")
                    if hasattr(self.ui, "set_connection"):
                        self.ui.set_connection("offline", "Your Gemini API key was rejected. Enter a new key.")
                    self.ui.set_state("SLEEPING")
                    self.ui.prompt_reconfig()
                    while not self.ui._win._ready:
                        await asyncio.sleep(1)
                    print("[JARVIS] New API key saved — reconnecting...")
                    _conn_backoff = 3
                    continue

                # Network / timeout errors — log clearly and back off
                is_net_err = any(k in err_str for k in (
                    "TimeoutError", "timed out", "getaddrinfo", "CancelledError",
                    "ConnectionRefusedError", "OSError", "Cannot connect",
                ))
                if is_net_err:
                    _conn_backoff = min(getattr(self, "_conn_backoff", 3) * 2, 60)
                    self._conn_backoff = _conn_backoff
                    self.ui.write_log(
                        f"SYS: Can't reach Gemini — retrying in {_conn_backoff} s. "
                        "Check your internet connection (a VPN may be needed in some regions)."
                    )
                    if hasattr(self.ui, "set_connection"):
                        self.ui.set_connection("reconnecting", f"Can't reach Gemini — retrying in {_conn_backoff} s.")
                else:
                    self._conn_backoff = 3
            finally:
                self.session = None

            self.set_speaking(False)
            self.ui.set_state("SLEEPING")

            if self._dashboard:
                await self._dashboard.broadcast({"type": "status", "state": "sleeping"})

            delay = getattr(self, "_conn_backoff", 3)
            if self._quick_reconnect:
                self._quick_reconnect = False
                delay = 0.3
            print(f"[JARVIS] Reconnecting in {delay}s...")
            await asyncio.sleep(delay)

def _single_instance_lock():
    """Only one JARVIS at a time: two would share the mic and both answer."""
    path = BASE_DIR / "config" / ".jarvis.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return None
    return f                                   # keep open for the life of the process


def main():
    lock = _single_instance_lock()
    if lock is None:
        print("[JARVIS] Already running — not starting a second copy.")
        return
    ui = JarvisUI("face.png")
    live: dict = {}

    def runner():
        ui.wait_for_api_key()
        jarvis = JarvisLive(ui)
        live["jarvis"] = jarvis
        try:
            asyncio.run(jarvis.run())
        except KeyboardInterrupt:
            print("\n🔴 Shutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

    # Window closed: stop the audio thread before Python tears down its
    # libraries, and give other apps their volume back if they were ducked.
    audio = getattr(live.get("jarvis"), "audio", None)
    if audio is not None:
        try:
            audio.stop()
        except Exception as e:
            print(f"[JARVIS] Audio shutdown: {e}")

if __name__ == "__main__":
    main()