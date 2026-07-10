"""Immutable configuration — constants loaded from config.toml / env vars."""

import os
import sys
from pathlib import Path

DATA_DIR = Path.home() / ".qwen-cli"
CONFIG_FILE = DATA_DIR / "config.toml"
DATA_DIR.mkdir(exist_ok=True)


def _load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError:
            return {}
    try:
        return tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _validate_config(cfg: dict) -> None:
    KNOWN_KEYS = {
        "base_url",
        "model",
        "token_limit",
        "max_tool_depth",
        "max_auto_continue",
        "editor",
        "openai_api_key",
        "fallback_model",
        "google_api_key",
        "google_cse_id",
        "brave_api_key",
        "auto_search",
        "preset",
    }
    for key in cfg:
        if key not in KNOWN_KEYS:
            suggestions = sorted(KNOWN_KEYS, key=lambda k: len(set(k) ^ set(key)))[:3]
            print(
                f"[config] Warning: unknown key '{key}' in config.toml — did you mean: {', '.join(suggestions)}?",
                file=sys.stderr,
            )
    if "token_limit" in cfg and (not isinstance(cfg["token_limit"], int) or cfg["token_limit"] <= 0):
        print(
            f"[config] Warning: token_limit={cfg['token_limit']} is invalid — must be a positive integer",
            file=sys.stderr,
        )
    if "max_tool_depth" in cfg and (not isinstance(cfg["max_tool_depth"], int) or cfg["max_tool_depth"] <= 0):
        print(
            f"[config] Warning: max_tool_depth={cfg['max_tool_depth']} is invalid — must be a positive integer",
            file=sys.stderr,
        )
    if "max_auto_continue" in cfg and (not isinstance(cfg["max_auto_continue"], int) or cfg["max_auto_continue"] < 0):
        print(
            f"[config] Warning: max_auto_continue={cfg['max_auto_continue']} is invalid"
            " — must be a non-negative integer",
            file=sys.stderr,
        )
    if "auto_search" in cfg and cfg["auto_search"] not in ("off", "smart", "aggressive"):
        print(
            f"[config] Warning: auto_search='{cfg['auto_search']}' is invalid — use off, smart, or aggressive",
            file=sys.stderr,
        )
    if "preset" in cfg and cfg["preset"] not in ("thinking", "code", "instruct"):
        print(
            f"[config] Warning: preset='{cfg['preset']}' is invalid — use thinking, code, or instruct", file=sys.stderr
        )


_CFG = _load_config()
_validate_config(_CFG)


def _cfg(key: str, env: str, default: str) -> str:
    return str(_CFG.get(key, os.environ.get(env, default)))


BASE_URL = _cfg("base_url", "QWEN_BASE_URL", "http://localhost:8080/v1")
MODEL = _cfg("model", "QWEN_MODEL", "Qwen3.6-27B")
TOKEN_LIMIT = int(_cfg("token_limit", "QWEN_TOKEN_LIMIT", "32000"))
MAX_TOOL_DEPTH = int(_cfg("max_tool_depth", "QWEN_MAX_TOOL_DEPTH", "20"))
MAX_AUTO_CONTINUE = int(_cfg("max_auto_continue", "QWEN_MAX_AUTO_CONTINUE", "4"))
DEFAULT_EDITOR = _cfg("editor", "EDITOR", "notepad" if sys.platform == "win32" else "nano")
OPENAI_API_KEY = _cfg("openai_api_key", "OPENAI_API_KEY", "")
FALLBACK_MODEL = _cfg("fallback_model", "QWEN_FALLBACK_MODEL", "gpt-4o-mini")
GOOGLE_API_KEY = _cfg("google_api_key", "GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = _cfg("google_cse_id", "GOOGLE_CSE_ID", "")
BRAVE_API_KEY = _cfg("brave_api_key", "BRAVE_API_KEY", "")

MEMORY_FILE = DATA_DIR / "memory.md"
SESSIONS_DIR = DATA_DIR / "sessions"
EXPORTS_DIR = DATA_DIR / "exports"
BACKUPS_DIR = DATA_DIR / "backups"
HISTORY_FILE = DATA_DIR / "input_history.txt"
PINS_FILE = DATA_DIR / "pins.json"
AUTOSAVE_FILE = SESSIONS_DIR / "autosave.json"
HANDOFF_FILE = DATA_DIR / "handoff.json"
INDEX_DIR = DATA_DIR / "project_indices"
CT_DIR = DATA_DIR / ".clawteam"
INTEL_DIR = DATA_DIR / "intelligence"
INTEL_FEED = INTEL_DIR / "feed.md"
INTEL_QUEUE = INTEL_DIR / "queue.json"
INTEL_TOPICS = INTEL_DIR / "topics.json"
PT_HISTORY_FILE = DATA_DIR / "pt_history.txt"

TOOL_RESULT_LIMIT = 16_000
AUTO_SAVE_INTERVAL = 5
AUX_LLM_TIMEOUT = float(_cfg("aux_timeout", "QWEN_AUX_TIMEOUT", "120"))

AUTO_SEARCH_MODE = _cfg("auto_search", "QWEN_AUTO_SEARCH", "smart").lower()
if AUTO_SEARCH_MODE not in ("off", "smart", "aggressive"):
    AUTO_SEARCH_MODE = "aggressive"

for _d in (SESSIONS_DIR, EXPORTS_DIR, BACKUPS_DIR, INDEX_DIR, CT_DIR, INTEL_DIR):
    _d.mkdir(exist_ok=True)

_LONG_OUTPUT = 81920
_MAX_BACKUP_STACK = 10
_INTEL_INTERVAL = 240
_INTEL_INJECT_N = 6

_TOOL_RETRY_MAX = 2
_TOOL_RETRY_BASE = 1.0
_TOOL_TIMEOUT_SLOW = 60
_TOOL_TIMEOUT_FAST = 15
_TOOL_TIMEOUT_NET = 30
_TOOL_RETRYABLE_TOOLS = frozenset(
    {
        "web_search",
        "search_news",
        "fetch_url",
        "fetch_rendered",
        "describe_image",
        "get_video_transcript",
        "team_inbox_receive",
        "team_task_list",
    }
)

SAMPLING_PRESETS: dict[str, dict] = {
    "thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "max_tokens": 32768,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "chat_template_kwargs": {"preserve_thinking": True},
        },
    },
    "code": {
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "max_tokens": 32768,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repeat_penalty": 1.0,
            "chat_template_kwargs": {"preserve_thinking": True},
        },
    },
    "instruct": {
        "temperature": 0.7,
        "top_p": 0.80,
        "presence_penalty": 1.5,
        "max_tokens": 32768,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0},
    },
}
