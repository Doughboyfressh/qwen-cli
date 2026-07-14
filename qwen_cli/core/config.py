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
        "provider",  # name of the active [providers.<name>] profile
        "providers",  # table of profiles
        "api_key",  # auth for base_url (any OpenAI-compatible provider)
        "sampler_extras",  # send llama.cpp-only samplers (top_k/min_p/repeat_penalty)
        "mcp",
        "intel",
        "tool_groups",
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


# --- Providers ---------------------------------------------------------------
# Any OpenAI-compatible endpoint can drive this CLI — the client is already an
# OpenAI client, so llama.cpp, vLLM, Ollama, OpenAI, Anthropic (its
# OpenAI-compatible endpoint), OpenRouter, DeepSeek, Groq and Together all work
# through the same code path. What was missing was the ability to *name* them:
#
#   provider = "claude"
#
#   [providers.local]
#   base_url = "http://localhost:8080/v1"
#   model    = "Qwen3.6-27B"
#   sampler_extras = true          # llama.cpp top_k/min_p/repeat_penalty
#
#   [providers.claude]
#   base_url = "https://api.anthropic.com/v1/"
#   api_key  = "sk-ant-..."
#   model    = "claude-sonnet-4-5"
#   token_limit = 180000
#
# Top-level base_url/model/token_limit still work and become the implicit
# default profile, so existing config.toml files keep running unchanged.
def _providers() -> dict:
    raw = _CFG.get("providers")
    return raw if isinstance(raw, dict) else {}


def _provider_cfg(name: str) -> dict:
    p = _providers().get(name)
    return p if isinstance(p, dict) else {}


ACTIVE_PROVIDER = _cfg("provider", "QWEN_PROVIDER", "")
_P = _provider_cfg(ACTIVE_PROVIDER)


def _pcfg(key: str, env: str, default: str) -> str:
    """Provider profile first, then the top-level key, then env, then default."""
    if key in _P:
        return str(_P[key])
    return _cfg(key, env, default)


BASE_URL = _pcfg("base_url", "QWEN_BASE_URL", "http://localhost:8080/v1")
MODEL = _pcfg("model", "QWEN_MODEL", "Qwen3.6-27B")
# Auth for BASE_URL. "no-key" keeps local llama.cpp/Ollama happy (they ignore it)
# while letting a real provider be the PRIMARY backend rather than only a
# fallback — previously the primary client was hardcoded to api_key="no-key", so
# no authenticated endpoint could ever be the main model.
API_KEY = _pcfg("api_key", "QWEN_API_KEY", "") or "no-key"
# Optional second llama-server: fast MoE model that absorbs background LLM work
# (memory extraction, summaries, intel, titles) so it never queues behind the
# main conversation's single slot. Empty aux_base_url disables the aux backend.
AUX_BASE_URL = _cfg("aux_base_url", "QWEN_AUX_BASE_URL", "http://localhost:8081/v1")
AUX_MODEL = _cfg("aux_model", "QWEN_AUX_MODEL", "Qwen3.6-35B-A3B")
# 28000, not 32000: the server runs -c 49152 with 16384 reserved for output, so
# the input ceiling is 32768 — a 32000 default left ~768 tokens for tokenizer
# drift and tool schemas instead of the ~4.7k the comment on SAMPLING_PRESETS
# (and the README) assume. Only ever bit a fresh install with no config.toml.
TOKEN_LIMIT = int(_pcfg("token_limit", "QWEN_TOKEN_LIMIT", "28000"))


def _is_local(url: str) -> bool:
    """Is this endpoint on this machine? Decides the sampler default and whether a
    connection failure is worth falling back from."""
    # S104 reads "0.0.0.0" as binding to all interfaces. Nothing is bound here —
    # it is a substring test against a URL the user configured.
    return any(h in url for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))  # noqa: S104


# top_k / min_p / repeat_penalty and Qwen's chat_template_kwargs are llama.cpp
# extensions. Cloud APIs reject unknown fields outright, so default this to ON
# only for a local endpoint. Override per profile with sampler_extras = true/false.
_TRUTHY = ("1", "true", "yes", "on")
_SAMPLER_DEFAULT = "true" if _is_local(BASE_URL) else "false"
SAMPLER_EXTRAS = _pcfg("sampler_extras", "QWEN_SAMPLER_EXTRAS", _SAMPLER_DEFAULT).strip().lower() in _TRUTHY

# chat_template_kwargs={"preserve_thinking": ...} is an argument to QWEN'S chat
# template. Hand it to a Llama/Mistral/Gemma GGUF whose template has no such
# parameter and llama.cpp can reject the request outright — so this CLI could
# only ever really drive Qwen. Default it from the model name and let a profile
# say otherwise; any thinking-style model (Qwen, DeepSeek-R1, QwQ) can opt in.
def _looks_like_thinking_model(name: str) -> bool:
    n = name.lower()
    return any(tag in n for tag in ("qwen", "qwq", "deepseek-r1", "r1-", "thinking"))


PRESERVE_THINKING = _pcfg(
    "preserve_thinking",
    "QWEN_PRESERVE_THINKING",
    "true" if _looks_like_thinking_model(MODEL) else "false",
).strip().lower() in _TRUTHY
# Per-provider: rounds are only useful if the findings survive them. On a 28k
# window more rounds mean more mid-run compaction, which is what shreds a long
# analysis; on a 200k model they are nearly free. So this belongs in the profile,
# not in a single global default.
MAX_TOOL_DEPTH = int(_pcfg("max_tool_depth", "QWEN_MAX_TOOL_DEPTH", "20"))
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
