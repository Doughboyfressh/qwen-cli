"""Immutable configuration — constants loaded from config.toml / env vars."""

import os
import sys
from pathlib import Path

# QWEN_DATA_DIR redirects ALL user data (memory.md, sessions, backups, ...) —
# set by tests/conftest.py so the suite can never pollute the real data dir
# (test runs used to write 'modified code.txt' entries into real memory.md).
DATA_DIR = Path(os.environ.get("QWEN_DATA_DIR") or (Path.home() / ".qwen-cli"))
CONFIG_FILE = DATA_DIR / "config.toml"
DATA_DIR.mkdir(parents=True, exist_ok=True)


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


_VALIDATORS: dict[str, tuple[type | tuple[type, ...], str, bool]] = {
    "token_limit": (int, "must be a positive integer", True),
    "max_tool_depth": (int, "must be a positive integer", True),
    "max_auto_continue": (int, "must be a non-negative integer", False),
    "auto_search": (str, "use off, smart, or aggressive", False),
    "preset": (str, "use thinking, code, or instruct", False),
}


def _validate_config(cfg: dict) -> None:
    known_keys = {
        "base_url",
        "model",
        "aux_base_url",
        "aux_model",
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
        "browser_proxy",
        "aux_timeout",
    }
    for key in cfg:
        if key not in known_keys:
            suggestions = sorted(known_keys, key=lambda k: len(set(k) ^ set(key)))[:3]
            print(
                f"[config] Warning: unknown key '{key}' in config.toml — did you mean: {', '.join(suggestions)}?",
                file=sys.stderr,
            )
    for key, (typ, msg, positive) in _VALIDATORS.items():
        if key not in cfg:
            continue
        val = cfg[key]
        if not isinstance(val, typ):
            print(f"[config] Warning: {key}={val!r} is invalid — {msg}", file=sys.stderr)
            continue
        if typ is int and positive and val <= 0:
            print(f"[config] Warning: {key}={val} is invalid — {msg}", file=sys.stderr)
        if typ is int and not positive and val < 0:
            print(f"[config] Warning: {key}={val} is invalid — {msg}", file=sys.stderr)
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
# Optional second llama-server: fast MoE model that absorbs background LLM work
# (memory extraction, summaries, intel, titles) so it never queues behind the
# main conversation's single slot. Empty aux_base_url disables the aux backend.
AUX_BASE_URL = _cfg("aux_base_url", "QWEN_AUX_BASE_URL", "http://localhost:8081/v1")
AUX_MODEL = _cfg("aux_model", "QWEN_AUX_MODEL", "Qwen3.6-35B-A3B")
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
COMMANDS_DIR = DATA_DIR / "commands"  # user-defined slash commands: <name>.md -> /name
AUDIT_LOG_FILE = DATA_DIR / "audit.log"  # append-only record of every run_command/run_script execution

TOOL_RESULT_LIMIT = 16_000
AUTO_SAVE_INTERVAL = 1  # save after every turn — lose at most one on crash
AUX_LLM_TIMEOUT = float(_cfg("aux_timeout", "QWEN_AUX_TIMEOUT", "120"))

AUTO_SEARCH_MODE = _cfg("auto_search", "QWEN_AUTO_SEARCH", "smart").lower()
if AUTO_SEARCH_MODE not in ("off", "smart", "aggressive"):
    AUTO_SEARCH_MODE = "smart"  # match the documented default on invalid values

# Live Intelligence background crawlers (3 browser threads + a feed injected
# into the system prompt) are opt-in: they cost tokens and background work
# that a coding session rarely wants. "on" starts them at launch; /intel on
# starts them mid-session regardless.
INTEL_MODE = _cfg("intel", "QWEN_INTEL", "off").lower()
if INTEL_MODE not in ("on", "off"):
    INTEL_MODE = "off"

# "auto": optional tool groups (browser/media/team) are sent only after the
# model enables them via enable_tools — their schemas are context the 28k
# window can't spare on turns that never use them. "all": always send every
# tool (pre-gating behavior; also set for spawned team agents via env).
TOOL_GROUPS_MODE = _cfg("tool_groups", "QWEN_TOOL_GROUPS", "auto").lower()
if TOOL_GROUPS_MODE not in ("auto", "all"):
    TOOL_GROUPS_MODE = "auto"

for _d in (SESSIONS_DIR, EXPORTS_DIR, BACKUPS_DIR, INDEX_DIR, CT_DIR, INTEL_DIR, COMMANDS_DIR):
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

# max_tokens=16384 pairs with the server's -c 49152 (start-qwen.bat): the
# input ceiling is 49152 - 16384 = 32768, so TOKEN_LIMIT=28000 keeps the same
# ~4.7k tokenizer-drift headroom as the old 65536/32768 split. The context was
# traded down to afford a q8_0 KV cache (q4_0 K measurably hurts long-context
# quality). NOTE: K and V cache types must match under flash-attn — a mixed
# q8_0/q4_0 cache fell off the FA CUDA fast path and prompt processing
# collapsed 2500 -> 69 t/s (benchmarked live; see start-qwen.bat). /long mode
# still raises output to 81,920 on servers with a bigger -c.
SAMPLING_PRESETS: dict[str, dict] = {
    "thinking": {
        "temperature": 1.0,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "max_tokens": 16384,
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
        "max_tokens": 16384,
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
        "max_tokens": 16384,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0},
    },
}
