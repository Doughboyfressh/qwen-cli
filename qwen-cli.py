#!/usr/bin/env python3
"""Qwen CLI v8 — <think> strip, parallel tools, result cap, -- no-tools, auto-compact, auto-save, health check, JSON retry"""

import sys
import json
import atexit
import os
import shlex
import subprocess
import difflib
import fnmatch
import time
import threading
import re
import ast
import hashlib
import concurrent.futures
import uuid
import tempfile
import asyncio

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from datetime import datetime
from pathlib import Path
from openai import OpenAI

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.rule import Rule
from rich.table import Table
from rich.syntax import Syntax
from rich.text import Text

import logging as _logging
import logging.handlers as _logging_handlers

_logging.basicConfig(
    filename=str(Path.home() / ".qwen-cli" / "qwen.log"),
    level=_logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
_logger = _logging.getLogger("qwen")
try:
    _h = _logging_handlers.RotatingFileHandler(str(Path.home() / ".qwen-cli" / "qwen.log"), maxBytes=1_048_576, backupCount=3, encoding="utf-8")
    _logger.handlers = [_h]
except Exception:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR    = Path.home() / ".qwen-cli"
CONFIG_FILE = DATA_DIR / "config.toml"
DATA_DIR.mkdir(exist_ok=True)

def _load_config() -> dict:
    """Read and parse config.toml. Falls back to tomli on Python <3.11."""
    if not CONFIG_FILE.exists():
        return {}
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            print("[config] tomli not installed — run: pip install tomli  (needed on Python <3.11)", file=sys.stderr)
            return {}
    try:
        return tomllib.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[config] Could not parse {CONFIG_FILE}: {e}", file=sys.stderr)
        return {}

def _validate_config(cfg: dict) -> None:
    """Validate config.toml values. Warn on unknown keys or invalid types."""
    KNOWN_KEYS = {
        "base_url", "model", "token_limit", "max_tool_depth", "max_auto_continue",
        "editor", "openai_api_key", "fallback_model",
        "google_api_key", "google_cse_id", "brave_api_key",
        "auto_search", "preset",
    }
    for key in cfg:
        if key not in KNOWN_KEYS:
            near_miss = sorted(KNOWN_KEYS - {key}, key=lambda k: len(set(k) & set(key)), reverse=True)
            did_you_mean = f" (did you mean '{near_miss[0]}'?)" if near_miss else ""
            print(f"[config] Unknown key '{key}' in config.toml{did_you_mean}", file=sys.stderr)

    if "token_limit" in cfg:
        val = cfg["token_limit"]
        if not isinstance(val, int) or val <= 0:
            print(f"[config] 'token_limit' must be a positive integer, got: {val!r}", file=sys.stderr)
    if "max_tool_depth" in cfg:
        val = cfg["max_tool_depth"]
        if not isinstance(val, int) or val <= 0:
            print(f"[config] 'max_tool_depth' must be a positive integer, got: {val!r}", file=sys.stderr)
    if "max_auto_continue" in cfg:
        val = cfg["max_auto_continue"]
        if not isinstance(val, int) or val < 0:
            print(f"[config] 'max_auto_continue' must be a non-negative integer, got: {val!r}", file=sys.stderr)
    if "auto_search" in cfg:
        if cfg["auto_search"] not in ("off", "smart", "aggressive"):
            print(f"[config] 'auto_search' must be 'off', 'smart', or 'aggressive', got: {cfg['auto_search']!r}", file=sys.stderr)
    if "preset" in cfg:
        if cfg["preset"] not in ("thinking", "code", "instruct"):
            print(f"[config] 'preset' must be 'thinking', 'code', or 'instruct', got: {cfg['preset']!r}", file=sys.stderr)

_CFG = _load_config()
_validate_config(_CFG)

def _cfg(key: str, env: str, default: str) -> str:
    """Resolve a config value: file setting > environment variable > hardcoded default."""
    return str(_CFG.get(key, os.environ.get(env, default)))

BASE_URL         = _cfg("base_url",         "QWEN_BASE_URL",         "http://localhost:8080/v1")
MODEL            = _cfg("model",            "QWEN_MODEL",            "Qwen3.6-27B")
TOKEN_LIMIT      = int(_cfg("token_limit",      "QWEN_TOKEN_LIMIT",      "32000"))
MAX_TOOL_DEPTH   = int(_cfg("max_tool_depth",   "QWEN_MAX_TOOL_DEPTH",   "20"))
# How many times a single turn may auto-continue when the model is cut off
# mid-output (hit max_tokens / stream dropped) before its answer is complete.
MAX_AUTO_CONTINUE = int(_cfg("max_auto_continue", "QWEN_MAX_AUTO_CONTINUE", "4"))
DEFAULT_EDITOR   = _cfg("editor", "EDITOR", "notepad" if sys.platform == "win32" else "nano")
OPENAI_API_KEY   = _cfg("openai_api_key",   "OPENAI_API_KEY",   "")
FALLBACK_MODEL   = _cfg("fallback_model",   "QWEN_FALLBACK_MODEL",   "gpt-4o-mini")
ACTIVE_BACKEND   = "llama.cpp"  # updated at startup if fallback activates
GOOGLE_API_KEY   = _cfg("google_api_key",   "GOOGLE_API_KEY",   "")   # Google Custom Search API
GOOGLE_CSE_ID    = _cfg("google_cse_id",    "GOOGLE_CSE_ID",    "")   # Google Custom Search Engine ID
BRAVE_API_KEY    = _cfg("brave_api_key",    "BRAVE_API_KEY",    "")   # Brave Search API key

MEMORY_FILE  = DATA_DIR / "memory.md"
SESSIONS_DIR = DATA_DIR / "sessions"
EXPORTS_DIR  = DATA_DIR / "exports"
BACKUPS_DIR  = DATA_DIR / "backups"
HISTORY_FILE = DATA_DIR / "input_history.txt"
PINS_FILE    = DATA_DIR / "pins.json"
AUTOSAVE_FILE = SESSIONS_DIR / "autosave.json"
HANDOFF_FILE = DATA_DIR / "handoff.json"
INDEX_DIR     = DATA_DIR / "project_indices"
CT_DIR        = DATA_DIR / ".clawteam"
INTEL_DIR     = DATA_DIR / "intelligence"
INTEL_FEED    = INTEL_DIR / "feed.md"
INTEL_QUEUE   = INTEL_DIR / "queue.json"
INTEL_TOPICS  = INTEL_DIR / "topics.json"

TOOL_RESULT_LIMIT  = 16_000   # max chars returned to model per tool call
AUTO_SAVE_INTERVAL = 5        # silent autosave every N turns
# Short, non-streaming helper calls (summarize tool output, generate title /
# commit msg, extract memory) must fail fast instead of inheriting the client's
# 1-hour streaming read timeout — a stalled server on _smart_cap's critical path
# would otherwise hang the whole turn. Override via QWEN_AUX_TIMEOUT.
AUX_LLM_TIMEOUT = float(_cfg("aux_timeout", "QWEN_AUX_TIMEOUT", "120"))
# How eagerly to auto-search the web before the model answers:
#   off | smart (only question-shaped / factual messages) | aggressive (almost every turn).
# Default smart: grounds real questions in live data while skipping the search
# latency on chit-chat and transform tasks. Toggle at runtime with /autosearch.
AUTO_SEARCH_MODE = _cfg("auto_search", "QWEN_AUTO_SEARCH", "smart").lower()
if AUTO_SEARCH_MODE not in ("off", "smart", "aggressive"):
    AUTO_SEARCH_MODE = "aggressive"

for _d in (SESSIONS_DIR, EXPORTS_DIR, BACKUPS_DIR, INDEX_DIR, CT_DIR, INTEL_DIR):
    _d.mkdir(exist_ok=True)

# Shared tool implementations — configured once after paths are defined
_qt = __import__('qwen_tools')  # noqa: E402
_qt.GOOGLE_API_KEY = GOOGLE_API_KEY
_qt.GOOGLE_CSE_ID  = GOOGLE_CSE_ID
_qt.BRAVE_API_KEY  = BRAVE_API_KEY
_qt.BACKUPS_DIR    = BACKUPS_DIR
_apply_diff, do_web_search, do_fetch_url, do_get_video_transcript, do_search_news, _html_to_text, presearch_decision = (  # noqa: E402
    _qt._apply_diff, _qt.do_web_search, _qt.do_fetch_url, _qt.do_get_video_transcript,
    _qt.do_search_news, _qt._html_to_text, _qt.presearch_decision,
)

console = Console(force_terminal=True, legacy_windows=False)

try:
    import readline as _rl
    try:
        _rl.read_history_file(str(HISTORY_FILE))
    except FileNotFoundError:
        pass
    atexit.register(_rl.write_history_file, str(HISTORY_FILE))
except ImportError:
    _rl = None  # type: ignore

try:
    from prompt_toolkit import PromptSession as _PtSession
    from prompt_toolkit.history import FileHistory as _PtFileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory as _PtAutoSuggest
    from prompt_toolkit.completion import Completer as _PtCompleter, Completion as _PtCompletion
    _HAS_PT = True
except ImportError:
    _HAS_PT = False

_pt_session = None
PT_HISTORY_FILE = DATA_DIR / "pt_history.txt"

_cli_client: "OpenAI | None" = None  # set in main(); allows tool fns to make direct API calls

_backup_stack: list[dict] = []  # stack of backups; /undo pops the most recent
_MAX_BACKUP_STACK = 10

# LSP client module — imported lazily
_lsp_client_mod: object | None = None
_last_user_input: str = ""
_branches: dict[str, list] = {}          # in-memory named conversation snapshots
_last_turn_tokens: dict = {"prompt": 0, "completion": 0}
_real_ctx_tokens: int = 0  # real prompt_tokens from last API response — used for compact check
_turn_count: int = 0
_session_start: float = 0.0
_current_mode: str = ""
_session_title: str = ""
_auto_memory_count: int = 0
_BG_LLM_SEM = threading.Semaphore(2)   # at most 2 background LLM calls at a time
_main_llm_busy = False                  # True while run_turn() holds the LLM slot
_intel_memory_written: dict[str, str] = {}  # topic_name → date; prevents duplicate entries
_cached_index: dict | None = None
_main_llm_busy_lock = threading.Lock()  # Protects reads/writes of _main_llm_busy
_cached_index_root: Path | None = None
_watched_files: dict[str, float] = {}   # path → last mtime
_watch_pending: list[str]        = []   # paths changed since last turn
_watch_stop   = threading.Event()
_watch_thread: threading.Thread | None = None
_focus_set: list[str] = []              # files loaded via /focus this session
_session_changes: dict[str, str] = {}  # path → original content before first edit
_tool_call_retry_log: dict[int, list] = {}  # depth → list of (tool, error) for retry context

# Error recovery: retry config for failed tool calls
_TOOL_RETRY_MAX    = 2          # max retries per failed tool call
_TOOL_RETRY_BASE   = 1.0        # base backoff in seconds
_TOOL_TIMEOUT_SLOW = 60         # seconds for I/O-heavy tools (web_search, fetch_url, etc.)
_TOOL_TIMEOUT_FAST = 15         # seconds for local tools (read_file, list_dir, team_*, etc.)
_TOOL_TIMEOUT_NET  = 30         # seconds for network tools (fetch_url, describe_image, etc.)
_TOOL_TIMEOUT_LLM  = 120        # seconds for LLM-dependent tools (fetch_rendered, browser_action)
_TOOL_RETRYABLE_TOOLS = frozenset({
    "web_search", "search_news", "fetch_url", "fetch_rendered",
    "describe_image", "get_video_transcript",
    "team_inbox_receive", "team_task_list",
})

_model_params: dict = {}               # runtime overrides — layered on top of active preset

# Qwen3.6 official sampling presets (https://qwen.readthedocs.io/best-practices)
SAMPLING_PRESETS: dict[str, dict] = {
    "thinking": {        # general tasks — thinking mode
        "temperature": 1.0, "top_p": 0.95, "presence_penalty": 0.0, "max_tokens": 32768,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
                       "chat_template_kwargs": {"preserve_thinking": True}},
    },
    "code": {            # precise coding / WebDev — thinking mode
        "temperature": 0.6, "top_p": 0.95, "presence_penalty": 0.0, "max_tokens": 32768,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
                       "chat_template_kwargs": {"preserve_thinking": True}},
    },
    "instruct": {        # non-thinking / instruct mode
        "temperature": 0.7, "top_p": 0.80, "presence_penalty": 1.5, "max_tokens": 32768,
        "extra_body": {"top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0},
    },
}
_active_preset: str = _CFG.get("preset", "thinking") if _CFG.get("preset") in SAMPLING_PRESETS else "thinking"

# High-output mode: Qwen3's recommended 81,920-token ceiling for hard math/coding
# competition problems (vs. 32,768 for general use). Toggled at runtime via /long.
_LONG_OUTPUT      = 81920
_long_mode        = False
_TOKEN_LIMIT_BASE = TOKEN_LIMIT     # input budget to restore when /long is turned off
_memory_lock    = threading.Lock()     # serializes all memory.md read-modify-write
_intel_stop     = threading.Event()
_intel_lock     = threading.Lock()
_intel_enabled   = threading.Event()  # thread-safe flag for intel crawlers
_INTEL_INTERVAL = 240      # seconds between each crawler's crawl cycles
_INTEL_INJECT_N = 6        # recent feed entries injected into system prompt
_INTEL_CRAWLERS = 3        # number of parallel background browser threads

def _intel_default_topics() -> list[dict]:
    year = datetime.now().year
    return [
        {"name": "AI & LLM news",           "query": "latest AI LLM model releases news today",          "last_checked": 0},
        {"name": "Python ecosystem",         "query": f"Python new libraries tools releases {year}",       "last_checked": 0},
        {"name": "Security vulnerabilities", "query": "critical security vulnerabilities CVE this week",   "last_checked": 0},
        {"name": "Tech industry news",       "query": "technology industry news today",                    "last_checked": 0},
        {"name": "Open source trending",     "query": "trending open source projects GitHub today",        "last_checked": 0},
        {"name": "Developer APIs",           "query": f"new developer APIs web services released {year}",  "last_checked": 0},
    ]

_INTEL_DEFAULT_TOPICS: list[dict] = _intel_default_topics()

MEMORY_CURATE_INTERVAL = 10   # consolidate memory.md every N auto-extractions

_THINK_RE      = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
# Qwen3 sometimes falls back to XML tool calls in plain text instead of API tool_calls
_XML_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_XML_PARAM_RE     = re.compile(r'<parameter\s+name=["\']([^"\']+)["\']>(.*?)</parameter>', re.DOTALL)


def _parse_xml_tool_calls(text: str) -> tuple[str, list]:
    """
    Extract <tool_call>...</tool_call> blocks from text.
    Handles both JSON-body and XML-param formats that Qwen3 emits.
    Returns (cleaned_text, synthetic_tool_calls).
    """
    tool_calls = []
    clean = text
    for m in _XML_TOOL_CALL_RE.finditer(text):
        body = m.group(1).strip()
        tc = None
        # Format A: JSON body {"name": ..., "arguments": {...}}
        try:
            data = json.loads(body)
            name = data.get("name") or data.get("tool") or ""
            args = data.get("arguments") or data.get("parameters") or {}
            if name:
                tc = {"id": f"xml_{len(tool_calls)}", "type": "function",
                      "function": {"name": name,
                                   "arguments": json.dumps(args) if isinstance(args, dict) else args}}
        except (json.JSONDecodeError, AttributeError):
            pass
        # Format B: <function_name>x</function_name><parameter name="k">v</parameter>
        if tc is None:
            fn_m = re.search(r"<function_name>(.*?)</function_name>", body, re.DOTALL)
            if fn_m:
                name = fn_m.group(1).strip()
                args = {k: v for k, v in _XML_PARAM_RE.findall(body)}
                tc = {"id": f"xml_{len(tool_calls)}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(args)}}
        if tc:
            tool_calls.append(tc)
            clean = clean.replace(m.group(0), "")
    return clean.strip(), tool_calls
_PARALLEL_TOOLS = frozenset({
    "web_search", "search_news", "fetch_url", "read_file",
    "list_directory", "find_files", "search_files",
    "describe_image", "get_video_transcript",
    "team_task_list", "team_inbox_receive",
    "lsp_query",
})
_STATEFUL_TOOLS = frozenset({
    "browser_action", "run_command", "run_script",
    "write_file", "patch_file", "move_file", "delete_file",
    "team_task_add", "team_task_update", "team_inbox_send", "team_spawn_agent",
})
_SERIAL_BY_DOMAIN = frozenset({"fetch_url", "fetch_rendered", "describe_image", "get_video_transcript"})


def _extract_domain(url: str) -> str:
    """Extract domain from a URL for contention detection."""
    import re
    m = re.search(r'https?://([^/:\d]+)', url)
    return m.group(1) if m else ""


def _classify_tool_batch(tool_calls: list, parsed_args: list) -> list[list[int]]:
    """Group tool calls into batches respecting:
    1. Stateful tools run alone (never parallelized)
    2. Fetches to the same domain are serialized (rate-limit safety)
    3. All other parallel-safe tools can run together

    Returns a list of batches (each batch is a list of indices)."""
    n = len(tool_calls)
    if n <= 1:
        return [[i] for i in range(n)]

    batches = []
    assigned = [False] * n
    domain_groups: dict[str, list[int]] = {}

    for i in range(n):
        name = tool_calls[i]["function"]["name"]
        args = parsed_args[i] or {}
        if name in _STATEFUL_TOOLS:
            batches.append([i])
            assigned[i] = True
        elif name in _SERIAL_BY_DOMAIN:
            url = args.get("url", "")
            domain = _extract_domain(url)
            if domain:
                domain_groups.setdefault(domain, []).append(i)
                assigned[i] = True

    # Domain groups: each domain gets its own batch
    for indices in domain_groups.values():
        batches.append(indices)

    # Remaining unassigned (parallel-safe, no domain contention)
    remaining = [i for i in range(n) if not assigned[i]]
    if remaining:
        batches.append(remaining)

    return batches
_SUMMARIZE_TOOLS = frozenset({"web_search", "search_news", "fetch_url", "fetch_rendered", "describe_image", "get_video_transcript"})
_RUNNABLE_LANGS  = {"bash", "sh", "shell", "powershell", "ps1", "python", "py", "cmd", "batch", "bat"}

_PLAN_RE  = re.compile(r"^\s*\d+[\.\)]\s+[^\n]+", re.MULTILINE)
_HEDGE_RE = re.compile(
    r"\b(?:i(?:'m| am) not (?:sure|certain)|i (?:think|believe|suspect)|"
    r"might|may|could|probably|possibly|perhaps|unclear|uncertain|"
    r"i don't know|not certain|i'm unsure)\b",
    re.IGNORECASE,
)

_MODE_PROMPTS: dict[str, str] = {
    "code":     "Mode: code — write minimal, correct, idiomatic code; no explanations unless asked; prefer patch_file over write_file; show diffs.",
    "debug":    "Mode: debug — trace root causes systematically; use run_command to check live state; read error output before suggesting fixes.",
    "explain":  "Mode: explain — explain clearly with concrete examples; break complex ideas into steps; match depth to the question.",
    "creative": "Mode: creative — explore unusual approaches; don't be constrained by convention; think expansively before converging.",
}

_DANGEROUS_CMD_RE = re.compile(
    r"\b(rm\s+-[^-]*r|"
    r"rmdir\s+/s|rd\s+/s|del\s+/[a-z/\s]*[sqf]\b|"
    r"(remove-item|ri)\b[^|;\n]*-rec|"
    r"format\s+[a-z]:|"
    r"git\s+reset\s+--hard|git\s+clean\s+-[^-]*f|"
    r"drop\s+table|drop\s+database|truncate\s+table|"
    r"dd\s+if=|mkfs\.|shred\s+|"
    r"sudo\s+rm|sudo\s+mkfs|sudo\s+dd|sudo\s+chmod\s+777|"
    r"chmod\s+777)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# ClawTeam coordination layer  (ClawTeam-compatible file format)
# ---------------------------------------------------------------------------

def _ct_team_dir(team: str) -> Path:
    """Internal helper: ct team dir."""
    d = CT_DIR / "teams" / team
    d.mkdir(parents=True, exist_ok=True)
    return d

def _ct_tasks_dir(team: str) -> Path:
    """Internal helper: ct tasks dir."""
    d = CT_DIR / "tasks" / team
    d.mkdir(parents=True, exist_ok=True)
    return d

def _ct_inbox_dir(team: str, agent: str) -> Path:
    """Internal helper: ct inbox dir."""
    d = _ct_team_dir(team) / "inboxes" / agent
    d.mkdir(parents=True, exist_ok=True)
    return d

def _ct_now() -> str:
    """Internal helper: ct now."""
    from datetime import timezone
    return datetime.now(timezone.utc).isoformat()

def _ct_atomic_write(path: Path, data: str) -> None:
    """Internal helper: ct atomic write."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem + "-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, str(path))
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise

def _ct_load_team(team: str) -> dict | None:
    """Internal helper: ct load team."""
    cfg = _ct_team_dir(team) / "config.json"
    if not cfg.exists():
        return None
    try:
        return json.loads(cfg.read_text(encoding="utf-8"))
    except Exception:
        return None

def _ct_save_team(team: str, config: dict) -> None:
    """Internal helper: ct save team."""
    cfg = _ct_team_dir(team) / "config.json"
    _ct_atomic_write(cfg, json.dumps(config, indent=2, ensure_ascii=False))

def _ct_team_create(team: str, description: str = "") -> dict:
    """Internal helper: ct team create."""
    existing = _ct_load_team(team)
    if existing:
        return existing
    config = {
        "name": team,
        "description": description,
        "leadAgentId": "",
        "createdAt": _ct_now(),
        "members": [],
        "budgetCents": 0.0,
    }
    _ct_save_team(team, config)
    return config

def _ct_team_list() -> list[str]:
    """Internal helper: ct team list."""
    teams_dir = CT_DIR / "teams"
    if not teams_dir.exists():
        return []
    return sorted(d.name for d in teams_dir.iterdir() if d.is_dir() and (d / "config.json").exists())

def _ct_team_join(team: str, agent_name: str, agent_type: str = "general-purpose") -> dict:
    """Internal helper: ct team join."""
    config = _ct_load_team(team) or _ct_team_create(team)
    for m in config["members"]:
        if m["name"] == agent_name:
            return m
    member = {
        "name": agent_name,
        "user": "",
        "agentId": uuid.uuid4().hex[:12],
        "agentType": agent_type,
        "joinedAt": _ct_now(),
    }
    config["members"].append(member)
    _ct_save_team(team, config)
    return member

def _ct_task_add(team: str, subject: str, owner: str = "", description: str = "", priority: str = "medium") -> dict:
    """Internal helper: ct task add."""
    task_id = uuid.uuid4().hex[:8]
    task = {
        "id": task_id,
        "subject": subject,
        "description": description,
        "status": "pending",
        "owner": owner,
        "priority": priority,
        "blocks": [],
        "blocked_by": [],
        "metadata": {},
        "locked_by": "",
        "locked_at": "",
        "started_at": "",
        "created_at": _ct_now(),
        "updated_at": _ct_now(),
    }
    _ct_atomic_write(_ct_tasks_dir(team) / f"task-{task_id}.json",
                     json.dumps(task, indent=2, ensure_ascii=False))
    return task

def _ct_task_list(team: str, owner: str = "", status: str = "") -> list[dict]:
    """Internal helper: ct task list."""
    tasks = []
    for f in sorted(_ct_tasks_dir(team).glob("task-*.json")):
        try:
            t = json.loads(f.read_text(encoding="utf-8"))
            if owner and t.get("owner", "") != owner:
                continue
            if status and t.get("status", "") != status:
                continue
            tasks.append(t)
        except Exception:
            continue
    return tasks

def _ct_task_update(team: str, task_id: str, status: str | None = None,
                    owner: str | None = None, note: str = "") -> dict | None:
    """Internal helper: ct task update."""
    tasks_dir = _ct_tasks_dir(team)
    path = tasks_dir / f"task-{task_id}.json"
    if not path.exists():
        matches = list(tasks_dir.glob(f"task-{task_id}*.json"))
        if not matches:
            return None
        path = matches[0]
    try:
        task = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if status:
        task["status"] = status
        if status == "in_progress" and not task.get("started_at"):
            task["started_at"] = _ct_now()
    if owner is not None:
        task["owner"] = owner
    if note:
        task.setdefault("notes", []).append({"text": note, "at": _ct_now()})
    task["updated_at"] = _ct_now()
    _ct_atomic_write(path, json.dumps(task, indent=2, ensure_ascii=False))
    return task

def _ct_inbox_send(team: str, to_agent: str, message: str, from_agent: str = "user") -> str:
    """Internal helper: ct inbox send."""
    msg_id = uuid.uuid4().hex[:12]
    msg = {
        "id": msg_id,
        "type": "message",
        "from": from_agent,
        "to": to_agent,
        "subject": message[:80],
        "body": message,
        "team": team,
        "sentAt": _ct_now(),
        "read": False,
    }
    _ct_atomic_write(_ct_inbox_dir(team, to_agent) / f"msg-{msg_id}.json",
                     json.dumps(msg, indent=2, ensure_ascii=False))
    return msg_id

def _ct_inbox_receive(team: str, agent: str, peek: bool = False) -> list[dict]:
    """Internal helper: ct inbox receive."""
    messages = []
    for f in sorted(_ct_inbox_dir(team, agent).glob("msg-*.json")):
        try:
            if peek:
                msg = json.loads(f.read_text(encoding="utf-8"))
                messages.append(msg)
            else:
                # Atomic rename-to-consume: if two readers race, only one rename wins;
                # the loser gets FileNotFoundError and skips this message safely.
                claimed = f.with_suffix(".consuming")
                f.rename(claimed)
                try:
                    msg = json.loads(claimed.read_text(encoding="utf-8"))
                    messages.append(msg)
                finally:
                    claimed.unlink(missing_ok=True)
        except (FileNotFoundError, FileExistsError):
            continue  # another consumer already claimed this message
        except Exception:
            continue
    return messages

def _ct_board_render(team: str) -> str:
    """Internal helper: ct board render."""
    config = _ct_load_team(team)
    if not config:
        return f"[team not found: {team}]"
    tasks = _ct_task_list(team)
    lines = [f"Team: {config.get('name', team)}"]
    if config.get("description"):
        lines.append(f"  {config['description']}")
    members = config.get("members", [])
    lines.append(f"\nMembers ({len(members)}):")
    for m in members:
        lines.append(f"  {m.get('name', '?')}  ({m.get('agentType', 'agent')})")
    lines.append(f"\nTasks ({len(tasks)}):")
    by_status: dict[str, list] = {}
    for t in tasks:
        by_status.setdefault(t.get("status", "pending"), []).append(t)
    for st in ("in_progress", "pending", "blocked", "completed"):
        group = by_status.get(st, [])
        if not group:
            continue
        lines.append(f"\n  {st.upper()} ({len(group)}):")
        for t in group[:30]:
            owner_tag = f"  @{t['owner']}" if t.get("owner") else ""
            pri = t.get("priority", "medium")
            pri_tag = f"  [{pri}]" if pri != "medium" else ""
            nn = len(t.get("notes", []))
            notes_tag = f"  ({nn} note{'s' if nn != 1 else ''})" if nn else ""
            lines.append(f"    [{t['id'][:6]}] {t.get('subject', '?')}{owner_tag}{pri_tag}{notes_tag}")
    return "\n".join(lines)

def _ct_spawn(team: str, agent_name: str, task: str, cwd: str = "") -> str:
    """Internal helper: ct spawn."""
    import shutil
    qwen_bin = (
        shutil.which("qwen")
        or shutil.which("qwen.bat")
        or str(Path(sys.executable).parent / "qwen.bat")
    )
    if not _ct_load_team(team):
        _ct_team_create(team)
    _ct_team_join(team, agent_name)
    members = (_ct_load_team(team) or {}).get("members", [])
    leader = members[0]["name"] if members else "leader"

    # Create a tracked task for this agent
    new_task = _ct_task_add(team, task, owner=agent_name, priority="medium")
    task_id = new_task["id"] if new_task else "?"

    prompt = (
        f"# Agent Brief\n\n"
        f"**Name:** {agent_name}\n"
        f"**Team:** {team}\n"
        f"**Leader:** {leader}\n"
        f"**Your task ID:** {task_id[:6]}\n\n"
        f"## Task\n\n{task}\n\n"
        f"## How to operate\n\n"
        f"You are running in autonomous agent mode. Work through your task using your tools. "
        f"Verify the result of each action before moving on. "
        f"When done, send your findings/results back to the leader via team_inbox_send, then mark the task completed.\n\n"
        f"## Your tools (you have full access to all of these)\n\n"
        f"- **web_search** — search the web for information\n"
        f"- **fetch_url** — fetch raw HTTP content of a URL\n"
        f"- **fetch_rendered** — fetch JS-rendered page content via browser\n"
        f"- **browser_action** — full browser automation (navigate, click, fill, screenshot, etc.)\n"
        f"- **run_command** — run shell commands, git, pip, npm, etc.\n"
        f"- **run_script** — execute inline Python/JS/bash/PowerShell/etc.\n"
        f"- **read_file** — read a file from disk\n"
        f"- **patch_file** — apply a unified diff patch to a file (preferred for edits)\n"
        f"- **write_file** — write/create a file\n"
        f"- **move_file** — move or rename a file\n"
        f"- **delete_file** — delete a file\n"
        f"- **list_directory** — list directory contents\n"
        f"- **find_files** — find files matching a glob pattern\n"
        f"- **search_files** — search file contents with regex\n"
        f"- **ask_user** — ask the user a clarifying question\n"
        f"- **team_spawn_agent** — spawn your own sub-subagents for parallel work\n"
        f"- **team_task_update**, **team_inbox_send**, **team_inbox_receive**, **team_board** — coordination\n\n"
        f"## Coordination protocol\n\n"
        f"1. Mark your task in progress immediately: `team_task_update` team={team} task_id={task_id[:6]} status=in_progress\n"
        f"2. Check your inbox for any extra context: `team_inbox_receive` team={team} agent={agent_name}\n"
        f"3. Log major progress milestones: `team_task_update` team={team} task_id={task_id[:6]} note=...\n"
        f"4. When done: send results to leader with `team_inbox_send` team={team} to={leader} message=<your full results>\n"
        f"5. Mark complete: `team_task_update` team={team} task_id={task_id[:6]} status=completed\n\n"
        f"Team files: {CT_DIR / 'teams' / team}\n"
        f"Task files: {CT_DIR / 'tasks' / team}\n"
    )

    spawn_dir = CT_DIR / "spawn"
    spawn_dir.mkdir(exist_ok=True)
    task_file = spawn_dir / f"{team}-{agent_name}-task.md"
    # Atomic write ensures the file is fully flushed before the spawned process reads it
    _ct_atomic_write(task_file, prompt)

    work_dir = cwd or str(Path.cwd())
    safe_name = re.sub(r"[^\w-]", "", agent_name)[:32]
    cmd = f'start "qwen-{safe_name}" cmd /k "{qwen_bin}" --task "@{task_file}"'
    try:
        subprocess.Popen(cmd, shell=True, cwd=work_dir)
        return (
            f"Spawned agent '{agent_name}' for team '{team}'.\n"
            f"Task ID: {task_id[:6]}\n"
            f"Brief: {task_file}"
        )
    except Exception as e:
        return f"[spawn error: {e}]"


def do_team_task_add(team: str, subject: str, owner: str = "", priority: str = "medium") -> str:
    """Handle team task add operation."""
    task = _ct_task_add(team, subject, owner=owner, priority=priority)
    if not task:
        return f"[failed to add task to team '{team}']"
    owner_tag = f"  @{task['owner']}" if task.get("owner") else ""
    return f"[{task['id'][:6]}] {task['subject']}{owner_tag}  [{task['priority']}]  status: {task['status']}"


def do_team_board(team: str) -> str:
    """Handle team board operation."""
    return _ct_board_render(team)


def do_team_list() -> str:
    """Handle team list operation."""
    teams = _ct_team_list()
    if not teams:
        return "[no teams — create one with /team create <name>]"
    lines = [f"Teams ({len(teams)}):"]
    for t in teams:
        cfg = _ct_load_team(t) or {}
        members = cfg.get("members", [])
        tasks = _ct_task_list(t)
        pending = sum(1 for x in tasks if x.get("status") != "completed")
        lines.append(f"  {t}  ({len(members)} members, {pending} open tasks)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web. Use this FIRST before answering any factual question, "
                "how-to question, or request for information — even if you think you know "
                "the answer. Current events, prices, docs, tutorials, software releases, "
                "troubleshooting, and general knowledge all benefit from a fresh search. "
                "Prefer searching over relying on training data."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "The search query"}},
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Search specifically for recent news articles. "
                "Use this instead of web_search when the question is about current events, "
                "recent releases, breaking news, or anything time-sensitive. "
                "Results are pulled from DDG news, Brave news, and Google, merged and ranked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "News search query"},
                    "max_results": {"type": "integer", "description": "Max results (default 8)"},
                },
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the raw text content of a URL (plain HTTP, no JavaScript rendering).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url":       {"type": "string",  "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Max characters to return (default 20000)"},
                },
                "required": ["url"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_rendered",
            "description": (
                "Fetch a URL with full JavaScript rendering via a real browser. "
                "Use this instead of fetch_url when the page loads content dynamically (SPAs, React, etc.), "
                "but you don't need to click or fill forms. Much faster than browser_action for read-only pages."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url":       {"type": "string",  "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Max characters to return (default 15000)"},
                },
                "required": ["url"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_image",
            "description": (
                "Download and describe an image from a URL. "
                "Returns metadata (size, format) and a visual description if the model supports vision. "
                "Use this when the user shares an image URL or when search results reference an image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Direct URL of the image to describe"},
                },
                "required": ["url"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_video_transcript",
            "description": (
                "Get the transcript or captions of a video. "
                "Works best with YouTube URLs — retrieves the full spoken text. "
                "Falls back to fetching the page description for non-YouTube videos. "
                "Use this whenever the user shares or asks about a video."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url":  {"type": "string", "description": "URL of the YouTube video or video page"},
                    "lang": {"type": "string", "description": "Preferred transcript language code (default: en)"},
                },
                "required": ["url"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "browser_action",
            "description": (
                "Control a real Chromium browser to automate web interactions: "
                "navigate to pages, fill forms, click buttons, select dropdowns, submit forms, "
                "scroll pages, hover over elements, press keyboard shortcuts, take screenshots, "
                "read page text, extract all links, or get the current URL. "
                "Use this when a page requires JavaScript interaction. "
                "Actions: navigate, fill, type, click, select, submit, wait_for, "
                "scroll (value=pixels or selector=element), hover, "
                "press_key (value=key e.g. Enter/Tab/Escape/Control+A), "
                "screenshot, get_text (selector optional), get_url, get_links, close."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "fill", "type", "click", "select", "submit",
                                 "wait_for", "scroll", "hover", "press_key",
                                 "screenshot", "get_text", "get_url", "get_links", "close"],
                        "description": "The browser action to perform.",
                    },
                    "url": {"type": "string", "description": "URL to navigate to (for 'navigate')."},
                    "selector": {
                        "type": "string",
                        "description": (
                            "CSS selector or semantic shorthand: 'label:Text', 'button:Name', 'link:Name', 'text:Text'. "
                            "Used by fill, type, click, select, wait_for, hover, press_key, get_text, scroll."
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": (
                            "Text to fill/type, option label for select, "
                            "pixels to scroll (negative = up), or key for press_key (e.g. 'Enter', 'Control+A')."
                        ),
                    },
                    "screenshot_path": {"type": "string", "description": "File path to save screenshot. Defaults to ~/screenshot.png."},
                },
                "required": ["action"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command and stream its stdout + stderr in real-time. "
                "Use for git, pip, npm, system checks, package installs, and anything that runs in a terminal. "
                "Dangerous commands (rm -rf, format, etc.) require explicit user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string",  "description": "Shell command to run"},
                    "cwd":     {"type": "string",  "description": "Working directory (default: current)"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30, max 300)"},
                    "env":     {"type": "object",  "description": "Extra environment variables to set, e.g. {\"DEBUG\": \"1\", \"NODE_ENV\": \"production\"}",
                                "additionalProperties": {"type": "string"}},
                    "stdin":   {"type": "string",  "description": "Text to pipe into the command's stdin"},
                },
                "required": ["command"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_script",
            "description": (
                "Execute inline code in a specific language — writes to a temp file and runs it. "
                "No need to create a file manually. "
                "Supported: python, javascript/node, bash, sh, powershell/ps1, ruby, php, lua, perl, go, typescript/ts, r."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {"type": "string",  "description": "Language name: python, javascript, bash, powershell, ruby, php, go, etc."},
                    "code":     {"type": "string",  "description": "The source code to execute"},
                    "cwd":      {"type": "string",  "description": "Working directory (default: current)"},
                    "timeout":  {"type": "integer", "description": "Timeout in seconds (default 30)"},
                },
                "required": ["language", "code"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file with syntax highlighting. Use offset/limit for a range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":   {"type": "string",  "description": "Absolute or relative file path"},
                    "offset": {"type": "integer", "description": "1-based line number to start from (optional)"},
                    "limit":  {"type": "integer", "description": "Max lines to read (optional)"},
                },
                "required": ["path"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": (
                "Apply a unified diff patch to a file. "
                "PREFER this over write_file for targeted edits — only the changed lines are needed, "
                "so it uses far less context and is safer. "
                "Generate a standard unified diff (--- a/... +++ b/... @@ ... @@) and pass it here."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to patch"},
                    "diff": {"type": "string", "description": "Unified diff to apply"},
                },
                "required": ["path", "diff"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write the FULL content of a file. Use patch_file instead for targeted edits. "
                "Creates file if new; shows diff and asks confirmation before overwriting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Full content to write"},
                },
                "required": ["path", "content"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the contents of a directory. Set recursive=true for a full tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":      {"type": "string",  "description": "Directory path"},
                    "recursive": {"type": "boolean", "description": "Show full tree (default false)"},
                },
                "required": ["path"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files matching a glob pattern inside a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string", "description": "Root directory to search"},
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"},
                },
                "required": ["path", "pattern"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": (
                "Search for a text string or regex across files in a directory. "
                "Results are grouped by file with line numbers. "
                "Use context to show surrounding lines (like grep -C)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path":    {"type": "string",  "description": "Root directory to search"},
                    "query":   {"type": "string",  "description": "Text or regex pattern to find"},
                    "pattern": {"type": "string",  "description": "Glob to filter files (default '**/*')"},
                    "context": {"type": "integer", "description": "Lines of context before/after each match (default 0)"},
                },
                "required": ["path", "query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_file",
            "description": "Move or rename a file. If dst is a directory, the file is placed inside it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "src": {"type": "string", "description": "Source file path"},
                    "dst": {"type": "string", "description": "Destination path (file or directory)"},
                },
                "required": ["src", "dst"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file permanently (creates a backup first, then asks confirmation).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to delete"},
                },
                "required": ["path"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Ask the user a clarifying question and wait for their answer. "
                "Use when the request is ambiguous and guessing would likely be wrong. "
                "Do NOT use for simple yes/no confirmations — just proceed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The clarifying question to ask"},
                },
                "required": ["question"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "team_task_list",
            "description": "List tasks for a team. Filter by owner or status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team":   {"type": "string", "description": "Team name"},
                    "owner":  {"type": "string", "description": "Filter by owner agent name (optional)"},
                    "status": {"type": "string", "description": "Filter by status: pending, in_progress, completed, blocked (optional)"},
                },
                "required": ["team"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "team_task_add",
            "description": "Add a new task to a team's task board.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team":     {"type": "string", "description": "Team name"},
                    "subject":  {"type": "string", "description": "Task description"},
                    "owner":    {"type": "string", "description": "Assign to this agent (optional)"},
                    "priority": {"type": "string", "description": "Priority: low, medium, high (default: medium)"},
                },
                "required": ["team", "subject"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "team_task_update",
            "description": "Update a task's status, owner, or add a progress note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team":    {"type": "string", "description": "Team name"},
                    "task_id": {"type": "string", "description": "Task ID (first 6+ chars ok)"},
                    "status":  {"type": "string", "description": "New status: pending, in_progress, completed, blocked"},
                    "owner":   {"type": "string", "description": "Assign to this agent (optional)"},
                    "note":    {"type": "string", "description": "Add a timestamped note to the task (optional)"},
                },
                "required": ["team", "task_id"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "team_inbox_send",
            "description": "Send a message to a team member's inbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team":       {"type": "string", "description": "Team name"},
                    "to":         {"type": "string", "description": "Recipient agent name"},
                    "message":    {"type": "string", "description": "Message body"},
                    "from_agent": {"type": "string", "description": "Sender name (default: user)"},
                },
                "required": ["team", "to", "message"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "team_inbox_receive",
            "description": "Read messages from a team member's inbox. By default consumes them; use peek=true to read without removing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team":  {"type": "string", "description": "Team name"},
                    "agent": {"type": "string", "description": "Your agent name"},
                    "peek":  {"type": "boolean", "description": "If true, read without consuming messages (default: false)"},
                },
                "required": ["team", "agent"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "team_board",
            "description": "Show the full task board for a team — members, tasks by status, owners, priorities, and notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team name"},
                },
                "required": ["team"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "team_list",
            "description": "List all teams with member counts and open task counts.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "team_spawn_agent",
            "description": (
                "Spawn a new qwen agent in a new terminal to handle a subtask autonomously in parallel. "
                "USE THIS PROACTIVELY — whenever a task has 3+ independent parts, a subtask would take you "
                "5+ tool calls, or you're iterating over a list, spawn an agent instead of doing it yourself. "
                "Spawned agents have full access to ALL tools: web_search, fetch_url, fetch_rendered, "
                "browser_action, run_command, run_script, read_file, patch_file, write_file, move_file, "
                "delete_file, list_directory, find_files, search_files, ask_user, and all team tools. "
                "After spawning, use team_board to track progress and team_inbox_receive to collect results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "team":       {"type": "string", "description": "Team name"},
                    "agent_name": {"type": "string", "description": "Name for the new agent"},
                    "task":       {"type": "string", "description": "Task description for the agent"},
                    "cwd":        {"type": "string", "description": "Working directory (default: current)"},
                },
                "required": ["team", "agent_name", "task"],
            },
        }
    },
]

BASE_SYSTEM = (
    "You are a helpful AI assistant running locally via llama.cpp. "
    "You DO have persistent memory across sessions: facts from conversations are automatically "
    "extracted and stored in memory.md, then injected into every session under "
    "'=== Persistent Memory ==='. Users can also add facts with /remember. "
    "Never claim you have no memory — you do. "
    "You have tools for web search, fetching URLs, running shell commands, and full filesystem access. "
    "For file edits, PREFER patch_file over write_file — patches are smaller and safer. "
    "Only use write_file when creating a new file or replacing the entire content. "
    "Use run_command to check real state; use fetch_url to read a full page after finding a URL via search. "
    "You also have a '=== Live Intelligence ===' section in your context, populated by "
    "background crawlers that continuously browse the web and summarize recent news, releases, "
    "vulnerabilities, and trends across tracked topics. Treat this as fresh, verified data — "
    "use it to ground answers about current events without needing an extra web search. "
    "IMPORTANT: Web search results are automatically injected before every response under "
    "'[Auto web search results...]'. You MUST read and use these results in your answer. "
    "If the auto-results are not detailed enough, call web_search again with a more specific query, "
    "then fetch_url on the most relevant result to get full page content. "
    "Never answer factual questions from training data alone — always verify with the injected results "
    "or additional searches. Cite sources (URLs) when using web information. "
    "For reading web pages, use the fastest tool that works: "
    "fetch_url (fastest, no JS) → fetch_rendered (Playwright, JS-rendered, read-only) → browser_action (slowest, use only for forms/clicks/interaction). "
    "Use describe_image for any image URL the user shares or that appears in search results. "
    "Use get_video_transcript for any YouTube or video URL — never say you cannot watch videos. "
    "You also have browser_action to fill forms, click buttons, and sign up for websites. "
    "If a request is genuinely ambiguous, use ask_user to clarify before acting. "
    "Do not use emoji characters in your responses — plain text only.\n\n"
    "You also have built-in multi-agent team coordination tools: "
    "team_list, team_board, team_task_add, team_task_list, team_task_update, "
    "team_inbox_send, team_inbox_receive, team_spawn_agent. "
    "DELEGATION RULE — use team_spawn_agent often and proactively. Spawn subagents whenever: "
    "(1) a task has 3 or more independent subtasks — spawn one agent per subtask; "
    "(2) research and implementation can happen in parallel — spawn a researcher and a coder; "
    "(3) you need to process a list (check N files, summarize N pages, test N endpoints) — spawn N agents; "
    "(4) any subtask would take you more than 4-5 tool calls to complete on its own. "
    "YOU are the coordinator. Subagents run in full autonomous mode and have access to ALL the same tools: "
    "web_search, fetch_url, fetch_rendered, browser_action, run_command, run_script, "
    "read_file, patch_file, write_file, move_file, delete_file, list_directory, find_files, search_files, ask_user, "
    "and all team coordination tools. "
    "After spawning, track progress with team_board, collect results via team_inbox_receive. "
    "team_task_update accepts a 'note' param to log progress. team_inbox_receive accepts 'peek' to read without consuming.\n\n"
    "IMPORTANT — when asked about your own capabilities, limitations, or what you can/cannot do, "
    "describe yourself accurately based on what you actually have, NOT like a generic cloud LLM:\n"
    "- Memory: you DO have persistent memory. Facts are saved to memory.md and injected every session.\n"
    "- Current information: you search the web in real time — your knowledge is NOT limited to a training cutoff.\n"
    "- Math and logic: use run_script (Python) for reliable computation — don't say you're unreliable at math.\n"
    "- File access: you have full read/write access to the local filesystem via your file tools.\n"
    "- Shell/system: you can run any shell command, install packages, manage processes.\n"
    "- Videos: you can get YouTube transcripts via get_video_transcript.\n"
    "- Images: you can describe images from URLs via describe_image.\n"
    "- Browser: you can automate a real browser (navigate, click, fill forms, screenshot) via browser_action.\n"
    "Real limitations to be honest about: "
    "you cannot make phone calls or send emails independently; "
    "you cannot access accounts or services unless the user provides credentials; "
    "browser automation may fail on sites with strong anti-bot measures or heavy JavaScript — "
    "say so and offer fetch_url as a fallback; "
    "your context window is finite — very long tasks may summarize or drop old history; "
    "for multi-step plans, always verify the result of each step before declaring it done.\n\n"
    "ANTI-HALLUCINATION RULE: Before stating any specific fact — a name, date, version number, "
    "statistic, quote, or technical detail — if you have any uncertainty at all, call web_search first. "
    "Do not say 'I think' or 'I believe' and then state an unverified fact. Search and verify first. "
    "If you search and still cannot find a reliable source, say explicitly 'I couldn't verify this' "
    "rather than guessing. It is always better to say 'let me check' than to state something wrong confidently."
)

HELP_TEXT = """
## Commands

| Command | Description |
|---|---|
| `/help [query]` | Show this help; with a query, show only matching commands |
| `/clear` | Clear conversation history |
| `/retry` | Re-send the last message |
| `/run <cmd>` | Run a shell command directly (streaming, Ctrl+C cancels) |
| `/edit <file>` | Open a file in your editor without leaving the session |
| `/system [msg]` | Show or set system prompt |
| `/file <path>` | Load a file into context |
| `/project [path]` | Load directory tree + key files (defaults to cwd) |
| `/undo` | Restore last overwritten file (multi-level stack, up to 10 levels) |
| `/export [name]` | Export conversation to a markdown file |
| `/copy` | Copy last response to clipboard |
| `/paste` | Paste clipboard content into context |
| `/remember <fact>` | Persist a fact across sessions |
| `/memory` | Show persistent memory |
| `/forget` | Clear persistent memory |
| `/pin <text>` | Add a persistent note injected into every system prompt |
| `/pin list` | List pinned notes |
| `/pin remove <n>` | Remove pin by number |
| `/branch <name>` | Snapshot current conversation (in-memory) |
| `/branch list` | List saved branches |
| `/branch restore <name>` | Restore a branch |
| `/search <query>` | Manual web search |
| `/autosearch [mode]` | Auto web-search before answering: `off` · `smart` (default) · `aggressive` |
| `/save [name]` | Save session to JSON |
| `/load [name]` | Load a session (fuzzy matches name or topic) |
| `/sessions` | List saved sessions with topic and turn count |
| `/trim` | Sliding-window summarize old history to free context |
| `/context` | Show token usage breakdown |
| `/model` | Show current model |
| `/model list` | List models available on llama.cpp server |
| `/model <name>` | Switch to a different model |
| `/config` | Show current configuration |
| `/exit` `/quit` | Exit (auto-saves session) |
| `-- <message>` | Send message without tool use (faster, no tool spin-up) |
| `/mode [name]` | Set response style: `code` · `debug` · `explain` · `creative` · `off` |
| `/preset [name]` | Switch sampling preset: `thinking` · `code` · `instruct` (no arg = show current) |
| `/index [force]` | Build/refresh symbol index for the current project |
| `/focus <files…>` | Load specific files into context (space-separated; glob patterns ok) |
| `/focus` | List files focused this session |
| `/focus off` | Clear focus tracking |
| `/changes` | List files modified by the model this session |
| `/changes diff` | Show full diffs of all session modifications |
| `/rollback` | Restore ALL session-modified files to their pre-session state |
| `/search-sessions <q>` | Full-text search across all saved session history |
| `/review [file]` | Structured code review (Issues / Suggestions / Security / Style) |
| `/params [name] [val]` | Set runtime model params: `temp` · `top_p` · `max` · `reset` |
| `/long [on/off]` | High-output mode (max 81,920 tokens) for hard math/coding problems; input budget shrinks to match |
| `/error` | Paste clipboard error and immediately diagnose it |
| `/task <goal>` | Plan-approve-execute-test agentic workflow with retry on failure |
| `/agent <goal>` | Autonomous loop — model iterates with tools until done or Ctrl+C |
| `/git commit [msg]` | AI-generated commit message from staged diff, then commit |
| `/git pr` | AI-generated PR description from commits ahead of main |
| `/watch <file>` | Re-inject file into context whenever it changes on disk |
| `/watch off` | Stop all file watchers |
| `/cd [path]` | Change working directory (no arg = show cwd) |
| `/history [n]` | Show last N conversation turns as a transcript (default: all) |
| `/note <text>` | Inject a note into history without calling the model |
| `/compact` | Alias for /trim — summarize old history to free context |
| `/stats` | Show session stats: turns, tokens, uptime, memory, preset |
| `/cleanup [what]` | Purge stale data: `teams` · `backups` · `tasks` · `all` (default) · `dry-run` |
| `/intel` | Show live intelligence feed from background crawlers |
| `/intel topics` | List tracked topics and last crawl time |
| `/intel add <name>|<query>` | Track a new topic |
| `/intel remove <name>` | Stop tracking a topic |
| /intel on/off | Pause/resume background crawlers |

## LSP / Code Intelligence

| Command | Description |
|---|---|
| /lsp status | Show LSP server status |
| /lsp diagnose <file> | Run diagnostics on a file |
| /lsp define <file> <line> <col> | Go to definition |
| /lsp refs <file> <line> <col> | Find all references |
| /lsp hover <file> <line> <col> | Get hover info / docstring |
| /lsp symbols <file> | List all symbols in a file |
| /lsp rename <file> <line> <col> <new_name> | Find rename locations |
| /lsp shutdown | Shut down LSP server |
| `/intel on/off` | Pause/resume background crawlers |

## Team coordination (multi-agent)

| Command | Description |
|---|---|
| `/team create <name> [desc]` | Create a new team |
| `/team list` | List all teams |
| `/team show <name>` | Show team members and task board |
| `/team join <team> [agent]` | Join a team as an agent |
| `/spawn <team> <agent> <task>` | Spawn a new qwen worker in a new terminal |
| `/inbox <team> [agent]` | Read and consume inbox messages |
| `/inbox send <team> <to> <msg>` | Send a message to a team member |
| `/board [team]` | Show team board (all teams if no name given) |

## Autonomous tools (Qwen calls these itself)

`web_search` · `fetch_url` · `browser_action` · `run_command` · `run_script` · `read_file` · **`patch_file`** · `write_file` · `move_file` · `delete_file` · `list_directory` · `find_files` · `search_files` · `ask_user`

**Team tools:** `team_list` · `team_board` · `team_task_add` · `team_task_list` · `team_task_update` · `team_inbox_send` · `team_inbox_receive` · `team_spawn_agent`

Team data lives in `~/.clawteam/` and is compatible with the ClawTeam CLI if installed.

## Tips

- Type `@path/to/file.py` anywhere in your message to inject that file for the turn
- Use `@file.py::function_name` to inject only a specific function or class (saves context)
- End a line with `\\` to continue on the next line (multiline input)
- Press **Ctrl+C** during a response or command to cancel and return to the prompt
- `run_command` streams output in real-time; Ctrl+C kills the process
- After a response containing code, press `r` to run the block or `c` to copy it to the clipboard
- Tab completes `/commands` and `@file` paths
- Create `.qwen-system.md` in a project for project-specific instructions
- Edit `~/.qwen-cli/config.toml` for persistent settings
- Pipe input: `echo "explain this" | qwen` or `cat file.py | qwen "what does this do?"`
"""

# ---------------------------------------------------------------------------
# Pins (persistent per-session reminders injected into every system prompt)
# ---------------------------------------------------------------------------

def load_pins() -> list[str]:
    """Load Pins"""
    if not PINS_FILE.exists():
        return []
    try:
        return json.loads(PINS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

def save_pins(pins: list[str]) -> None:
    """Save Pins"""
    PINS_FILE.write_text(json.dumps(pins, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------------------------------------------------------------------
# Memory & system prompt
# ---------------------------------------------------------------------------

def load_memory() -> str:
    """Load Memory"""
    return MEMORY_FILE.read_text(encoding="utf-8").strip() if MEMORY_FILE.exists() else ""

def save_memory(text: str):
    """Save Memory"""
    MEMORY_FILE.write_text(text, encoding="utf-8")

# ---------------------------------------------------------------------------
# Codebase symbol indexer
# ---------------------------------------------------------------------------

_INDEX_EXTS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs",
    ".go", ".rs", ".java", ".cs", ".cpp", ".c", ".rb", ".php",
}

def _project_index_path(root: Path) -> Path:
    """Internal helper: project index path."""
    slug = hashlib.md5(str(root.resolve()).encode()).hexdigest()[:12]
    return INDEX_DIR / f"{slug}.json"

def _extract_py_symbols(filepath: Path) -> dict:
    """Extract top-level functions and classes from a Python file via ast."""
    try:
        tree = ast.parse(filepath.read_bytes(), filename=str(filepath))
    except Exception:
        return {}
    functions: list[str] = []
    classes: list[dict] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            methods = [
                n.name for n in ast.iter_child_nodes(node)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            classes.append({"name": node.name, "methods": methods})
    return {"functions": functions, "classes": classes}

def _extract_generic_symbols(filepath: Path, ext: str) -> dict:
    """Extract top-level symbols from non-Python files via regex."""
    functions: list[str] = []
    classes: list[str] = []
    try:
        lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return {}
    for line in lines:
        s = line.strip()
        if ext in (".js", ".ts", ".tsx", ".jsx", ".mjs"):
            m = re.match(r"(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(\w+)", s)
            if m:
                functions.append(m.group(1))
            m = re.match(r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(", s)
            if m:
                functions.append(m.group(1))
            m = re.match(r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)", s)
            if m:
                classes.append(m.group(1))
        elif ext == ".go":
            m = re.match(r"func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(", s)
            if m:
                functions.append(m.group(1))
            m = re.match(r"type\s+(\w+)\s+struct", s)
            if m:
                classes.append(m.group(1))
        elif ext == ".rs":
            m = re.match(r"(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", s)
            if m:
                functions.append(m.group(1))
            m = re.match(r"(?:pub\s+)?(?:struct|enum|trait)\s+(\w+)", s)
            if m:
                classes.append(m.group(1))
        elif ext in (".java", ".cs"):
            # Java/C# methods: match access modifiers, static, return type, then method name
            m = re.match(
                r"(?:(?:public|private|protected|static|final|abstract|synchronized|async|void|int|long|float|double|bool|boolean|string|char|byte|short|var|auto|Task<\w+>|List<\w+>|Map<\w+,\s*\w+>|Set<\w+>|Dictionary<\w+,\s*\w+>)\s+)*"
                r"(\w+)\s*\(", s
            )
            if m and m.group(1) not in ("if", "for", "while", "switch", "catch", "using", "new", "return", "throw", "else", "try"):
                functions.append(m.group(1))
            # Classes and structs
            m = re.match(r"(?:(?:public|private|protected|abstract|static|sealed|partial|internal|friend)\s+)*"
                         r"(?:class|struct|interface|enum)\s+(\w+)", s)
            if m:
                classes.append(m.group(1))
    return {
        "functions": list(dict.fromkeys(functions)),
        "classes":   list(dict.fromkeys(classes)),
    }

def _extract_symbol_from_file(p: Path, symbol: str) -> str | None:
    """Extract a named function or class from a source file."""
    ext = p.suffix.lower()
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if ext == ".py":
        try:
            tree = ast.parse(source.encode(), filename=str(p))
            lines = source.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if node.name == symbol and hasattr(node, "end_lineno"):
                        return "\n".join(lines[node.lineno - 1:node.end_lineno])
        except Exception:
            pass
        return None
    # Generic: find start line by pattern, grab until next unindented line
    lines = source.splitlines()
    patterns = [
        re.compile(rf"(?:async\s+)?(?:def|function|func|fn)\s+{re.escape(symbol)}\s*[\(\{{<]"),
        re.compile(rf"(?:class|struct|enum|trait|interface)\s+{re.escape(symbol)}\b"),
        re.compile(rf"\b{re.escape(symbol)}\s*[:=]\s*(?:async\s+)?\(?(?:function)?\s*\(?"),
    ]
    start = None
    for i, line in enumerate(lines):
        if any(rx.search(line) for rx in patterns):
            start = i
            break
    if start is None:
        return None
    end = start + 1
    while end < len(lines) and end - start < 150:
        line = lines[end]
        if line and not line[0].isspace() and end > start + 2:
            break
        end += 1
    return "\n".join(lines[start:end])


def build_symbol_index(root: Path, max_files: int = 500) -> dict:
    """Walk a project directory and build a {rel_path: {functions, classes}} index."""
    extra_ignore = load_qwenignore(root)
    index: dict[str, dict] = {}
    for fpath in sorted(root.rglob("*")):
        if len(index) >= max_files:
            break
        if not fpath.is_file():
            continue
        if any(part in IGNORE_DIRS for part in fpath.parts):
            continue
        if _qwen_ignored(fpath, root, extra_ignore):
            continue
        ext = fpath.suffix.lower()
        if ext not in _INDEX_EXTS or fpath.stat().st_size > 500_000:
            continue
        rel = str(fpath.relative_to(root)).replace("\\", "/")
        syms = _extract_py_symbols(fpath) if ext == ".py" else _extract_generic_symbols(fpath, ext)
        if syms.get("functions") or syms.get("classes"):
            index[rel] = syms
    return index

def _save_symbol_index(root: Path, index: dict) -> None:
    """Internal helper: save symbol index."""
    data = {"root": str(root), "built_at": datetime.now().isoformat(), "index": index}
    _project_index_path(root).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

def _load_symbol_index(root: Path, max_age_hours: int = 24) -> dict | None:
    """Internal helper: load symbol index."""
    p = _project_index_path(root)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        age_h = (datetime.now() - datetime.fromisoformat(data["built_at"])).total_seconds() / 3600
        return data["index"] if age_h <= max_age_hours else None
    except Exception:
        return None

def _get_index(root: Path) -> dict | None:
    """Return in-memory cached index, falling back to disk, then None."""
    global _cached_index, _cached_index_root
    if _cached_index_root == root and _cached_index is not None:
        return _cached_index
    _cached_index = _load_symbol_index(root)
    _cached_index_root = root
    return _cached_index

def _format_symbol_index(index: dict) -> str:
    """Compact text representation: one line per file."""
    lines = []
    for rel, syms in sorted(index.items()):
        parts = [f"class {c['name']}" if isinstance(c, dict) else f"class {c}"
                 for c in syms.get("classes", [])]
        parts += [f"{f}()" for f in syms.get("functions", [])]
        if parts:
            lines.append(f"{rel}: {', '.join(parts)}")
    return "\n".join(lines)

def cmd_index(root: Path, force: bool = False) -> dict:
    """Build or load the project symbol index; update in-memory cache."""
    global _cached_index, _cached_index_root
    if not force:
        cached = _load_symbol_index(root)
        if cached:
            sym_count = sum(len(s.get("functions", [])) + len(s.get("classes", []))
                            for s in cached.values())
            console.print(f"[dim][index: {len(cached)} files, {sym_count} symbols — '/index force' to rebuild][/dim]")
            _cached_index, _cached_index_root = cached, root
            return cached
    console.print(f"[dim]Indexing {root}...[/dim]")
    index = build_symbol_index(root)
    _save_symbol_index(root, index)
    _cached_index, _cached_index_root = index, root
    sym_count = sum(len(s.get("functions", [])) + len(s.get("classes", []))
                    for s in index.values())
    console.print(f"[green][indexed: {len(index)} files, {sym_count} symbols][/green]")
    return index

def _detect_test_command(root: Path) -> str | None:
    """Detect the project's test runner."""
    if (root / "pytest.ini").exists():
        return "pytest"
    if (root / "pyproject.toml").exists():
        try:
            cfg = (root / "pyproject.toml").read_text(encoding="utf-8")
            if "pytest" in cfg:
                return "pytest"
        except Exception:
            pass
    if list(root.glob("test_*.py")) or (root / "tests").is_dir():
        return "pytest"
    if (root / "package.json").exists():
        try:
            pkg = json.loads((root / "package.json").read_text(encoding="utf-8"))
            if "test" in pkg.get("scripts", {}):
                return "npm test"
        except Exception:
            pass
    if (root / "Cargo.toml").exists():
        return "cargo test"
    if (root / "go.mod").exists():
        return "go test ./..."
    return None


# build_system_prompt() runs every turn and used to spawn up to 4 `shell=True`
# git subprocesses each time (~100ms apiece on Windows) before the LLM call even
# started. Cache the result per-cwd with a short TTL so back-to-back turns (and
# the rapid /agent and /task loops especially) reuse it. The TTL is short enough
# that interactive edits show up promptly; _invalidate_git_cache() forces a
# refresh the instant we know state changed (e.g. after committing).
_GIT_CTX_CACHE: dict[str, tuple[float, str]] = {}   # cwd -> (monotonic_ts, context)
_GIT_CTX_TTL = 3.0

def _invalidate_git_cache() -> None:
    """Internal helper: invalidate git cache."""
    _GIT_CTX_CACHE.clear()

def get_git_context() -> str:
    """Get Git Context"""
    key = str(Path.cwd())
    now = time.monotonic()
    cached = _GIT_CTX_CACHE.get(key)
    if cached is not None and now - cached[0] < _GIT_CTX_TTL:
        return cached[1]
    ctx = _compute_git_context()
    _GIT_CTX_CACHE[key] = (now, ctx)
    return ctx

def _compute_git_context() -> str:
    """Internal helper: compute git context."""
    try:
        check = subprocess.run(
            "git rev-parse --is-inside-work-tree",
            shell=True, capture_output=True, timeout=5,
        )
        if check.returncode != 0:
            return ""
        branch = subprocess.run(
            "git branch --show-current", shell=True,
            capture_output=True, text=True, timeout=5, encoding="utf-8",
        ).stdout.strip()
        status = subprocess.run(
            "git status --short", shell=True,
            capture_output=True, text=True, timeout=5, encoding="utf-8",
        ).stdout.strip()
        diff_stat = subprocess.run(
            "git diff --stat HEAD", shell=True,
            capture_output=True, text=True, timeout=5, encoding="utf-8",
        ).stdout.strip()
        parts = [f"Branch: {branch}"]
        if status:
            parts.append(f"Status:\n{status}")
        if diff_stat:
            parts.append(f"Diff stat:\n{diff_stat}")
        return "\n".join(parts)
    except Exception:
        return ""

def load_project_system(cwd: Path) -> str:
    """Load Project System"""
    f = cwd / ".qwen-system.md"
    return f.read_text(encoding="utf-8").strip() if f.exists() else ""

def build_system_prompt(base: str) -> str:
    """Build System Prompt"""
    cwd = Path.cwd()
    # Date only (no minute-precision time): a timestamp that changes every minute
    # at the very top of the prompt defeats llama-server's automatic prefix-cache
    # reuse across turns, forcing the whole (large) system prompt to reprocess.
    # Date-only keeps the prefix byte-stable all day. For the exact time the model
    # can web-search; minute precision was never reliable from a static prompt anyway.
    today = datetime.now().strftime("%A, %B %d, %Y")
    parts = [f"Current date: {today}", f"Current directory: {cwd}", "", base]
    if _current_mode and _current_mode in _MODE_PROMPTS:
        parts += ["", "=== Mode ===", _MODE_PROMPTS[_current_mode], "=== End Mode ==="]
    proj_sys = load_project_system(cwd)
    if proj_sys:
        parts += ["", "=== Project Instructions ===", proj_sys, "=== End Project Instructions ==="]
    git_ctx = get_git_context()
    if git_ctx:
        parts += ["", "=== Git State ===", git_ctx, "=== End Git State ==="]
    idx = _get_index(cwd)
    if idx:
        idx_text = _format_symbol_index(idx)
        if idx_text:
            parts += ["", "=== Symbol Index ===", idx_text, "=== End Symbol Index ==="]
    mem = load_memory()
    if mem:
        parts += ["", "=== Persistent Memory ===", mem, "=== End Memory ==="]
    intel = intel_get_recent()
    if intel:
        parts += ["", "=== Live Intelligence (background web feed) ===", intel,
                  "=== End Live Intelligence ==="]
    pins = load_pins()
    if pins:
        parts += ["", "=== Pinned Notes ==="]
        parts += [f"- {p}" for p in pins]
        parts += ["=== End Pinned Notes ==="]
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# @file expansion
# ---------------------------------------------------------------------------

_AT_REF_RE = re.compile(r'@([\S]+)')

def expand_at_refs(text: str) -> str:
    """Expand @path and @path::symbol tokens in message text."""
    injections: list[str] = []
    for m in _AT_REF_RE.finditer(text):
        raw = m.group(1)
        symbol: str | None = None
        if "::" in raw:
            file_part, sym_raw = raw.split("::", 1)
            raw    = file_part.rstrip(".,;:)\"'")
            symbol = sym_raw.rstrip(".,;:)\"'") or None
        else:
            raw = raw.rstrip(".,;:)\"'")
        p = _resolve(raw)
        if not p.exists() or not p.is_file():
            continue
        lang = LANG_MAP.get(p.suffix.lower(), "")
        try:
            if symbol:
                extracted = _extract_symbol_from_file(p, symbol)
                if extracted is None:
                    console.print(f"[yellow]  @{raw}::{symbol}: symbol not found[/yellow]")
                    continue
                injections.append(f"[File: {p}, symbol: {symbol}]\n```{lang}\n{extracted}\n```")
                console.print(f"[dim green]  @{raw}::{symbol} → {len(extracted):,} chars[/dim green]")
            else:
                raw_bytes = p.read_bytes()
                if b"\x00" in raw_bytes[:8192]:
                    console.print(f"[yellow]  @{raw}: binary file skipped[/yellow]")
                    continue
                content = raw_bytes.decode("utf-8", errors="replace")
                if len(content) > 100_000:
                    content = content[:100_000] + "\n... [truncated]"
                injections.append(f"[File: {p}]\n```{lang}\n{content}\n```")
                console.print(f"[dim green]  @{raw} → {p.name} ({len(content):,} chars)[/dim green]")
        except Exception as e:
            console.print(f"[yellow]  @{raw}: {e}[/yellow]")

    if injections:
        return text + "\n\n" + "\n\n".join(injections)
    return text

# ---------------------------------------------------------------------------
# Tab completion
# ---------------------------------------------------------------------------

_COMMANDS = [
    "/help", "/clear", "/retry", "/run", "/edit", "/system", "/file",
    "/project", "/focus", "/review", "/undo", "/export", "/copy", "/paste",
    "/remember", "/memory", "/forget", "/pin", "/branch", "/search",
    "/search-sessions", "/save", "/load", "/sessions", "/trim", "/compact", "/context",
    "/model", "/mode", "/preset", "/params", "/long", "/autosearch", "/agent", "/task", "/index", "/git",
    "/watch", "/changes", "/rollback", "/error", "/config", "/cd",
    "/history", "/note", "/stats", "/exit", "/quit",
    "/team", "/spawn", "/inbox", "/board",
    "/intel",
]

def _setup_tab_completion() -> None:
    """Internal helper: setup tab completion."""
    if _rl is None:
        return
    try:
        def completer(text: str, state: int) -> str | None:
            """Completer"""
            if text.startswith("/"):
                options = [c + " " for c in _COMMANDS if c.startswith(text)]
            elif text.startswith("@") or "/" in text or "\\" in text:
                raw  = text.lstrip("@")
                pre  = "@" if text.startswith("@") else ""
                p    = Path(raw).expanduser()
                base = p if (raw.endswith("/") or raw.endswith("\\")) else p.parent
                try:
                    options = []
                    for child in sorted(base.iterdir()):
                        candidate = str(child) + ("/" if child.is_dir() else "")
                        if candidate.startswith(str(p)):
                            options.append(pre + candidate)
                except Exception:
                    options = []
            else:
                options = []
            try:
                return options[state]
            except IndexError:
                return None

        _rl.set_completer(completer)
        _rl.set_completer_delims(" \t\n")
        _rl.parse_and_bind("tab: complete")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Web search — multi-engine chain: Google CSE → Brave → DDG → Bing scrape
# ---------------------------------------------------------------------------

def do_describe_image(url: str) -> str:
    """Handle describe image operation."""
    return _qt.do_describe_image(url, llm_client=_cli_client, llm_model=MODEL)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def _session_meta(path: Path) -> dict:
    """Internal helper: session meta."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data.get("meta", {})
        if not meta:
            history = data.get("history", [])
            user_msgs = [m for m in history if m.get("role") == "user"
                         and not (m.get("content") or "").startswith("# Project Context")]
            topic = (user_msgs[0].get("content") or "")[:60].replace("\n", " ") if user_msgs else ""
            turns = sum(1 for m in history if m.get("role") == "assistant")
            meta  = {"topic": topic, "turns": turns, "saved_at": ""}
        return meta
    except Exception:
        return {"topic": "", "turns": 0, "saved_at": ""}

def save_session(history: list, system_prompt: str, name: str | None = None):
    """Save Session"""
    user_msgs = [m for m in history if m.get("role") == "user"
                 and not (m.get("content") or "").startswith("# Project Context")]
    topic = (user_msgs[0].get("content") or "")[:80].replace("\n", " ") if user_msgs else ""
    turns = sum(1 for m in history if m.get("role") == "assistant")
    fname = (name or datetime.now().strftime("%Y%m%d_%H%M%S")) + ".json"
    path  = SESSIONS_DIR / fname
    data  = {
        "meta": {"topic": topic, "turns": turns, "saved_at": datetime.now().isoformat()},
        "system_prompt": system_prompt,
        "history": history,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[dim][saved: {path.name}  ({turns} turn{'s' if turns!=1 else ''})][/dim]")

def _silent_autosave(history: list, system_prompt: str) -> None:
    """Rolling checkpoint to autosave.json — no console output, never raises."""
    try:
        user_msgs = [m for m in history if m.get("role") == "user"
                     and not (m.get("content") or "").startswith("# Project Context")]
        raw_topic = (user_msgs[0].get("content") or "")[:80].replace("\n", " ") if user_msgs else ""
        topic = _session_title or raw_topic
        turns = sum(1 for m in history if m.get("role") == "assistant")
        data  = {
            "meta":          {"topic": topic, "turns": turns, "saved_at": datetime.now().isoformat()},
            "system_prompt": system_prompt,
            "history":       history,
        }
        AUTOSAVE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


HANDOFF_PROMPT_TEMPLATE = """You are resuming a session that ran out of context window. Here is what happened:

{summary}

Use this context to pick up where you left off. Do NOT ask the user to repeat themselves. Check memory.md for additional persistent facts."""


def _generate_handoff(client, history, base_system):
    """Generate a compact handoff summary for session resumption."""
    try:
        chat = [m for m in history if m.get("role") in ("user", "assistant")]
        recent = chat[-16:]
        recent_text = "\n".join(
            f"{m['role'].upper()}: {(m.get('content') or '')[:800]}"
            for m in recent
        )
        prompt = [
            {"role": "system", "content": (
                "You are a session summarizer. Produce a handoff document for an AI assistant "
                "resuming this conversation. Include: (1) what the session goal is, "
                "(2) what has been accomplished, (3) what files were created/modified, "
                "(4) the exact task being worked on and the immediate next step, "
                "(5) any open questions or blockers. Be precise. Keep it under 800 words."
            )},
            {"role": "user", "content": f"Summarize this session for handoff:\n\n{recent_text}"},
        ]
        summary, _, _ = stream_once(client, prompt, use_tools=False)
        return summary or "(summarization failed - see autosave for raw history)"
    except Exception as e:
        return f"(handoff summary failed: {e})"


def _write_handoff(summary, history):
    """Write a handoff file for the next session to pick up."""
    try:
        chat = [m for m in history if m.get("role") in ("user", "assistant")]
        last_user = ""
        for m in reversed(chat):
            if m.get("role") == "user":
                last_user = m.get("content", "")[:400]
                break
        data = {
            "summary": summary,
            "last_user_message": last_user,
            "turns": sum(1 for m in history if m.get("role") == "assistant"),
            "timestamp": datetime.now().isoformat(),
        }
        HANDOFF_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _save_exit_handoff(history):
    """Save a lightweight handoff on normal exit so next session has context."""
    try:
        chat = [m for m in history if m.get("role") in ("user", "assistant")]
        last_user = ""
        for m in reversed(chat):
            if m.get("role") == "user":
                last_user = m.get("content", "")[:400]
                break
        last_assistant = ""
        for m in reversed(chat):
            if m.get("role") == "assistant":
                last_assistant = m.get("content", "")[:600]
                break
        data = {
            "summary": (f"Session had {sum(1 for m in history if m.get('role') == 'assistant')} turns. "
                        f"Last user: {last_user[:200]}. Last assistant: {last_assistant[:200]}"),
            "last_user_message": last_user,
            "last_assistant_message": last_assistant,
            "turns": sum(1 for m in history if m.get("role") == "assistant"),
            "timestamp": datetime.now().isoformat(),
        }
        HANDOFF_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def _consume_handoff():
    """Read and remove the handoff file. Returns dict or None."""
    try:
        if not HANDOFF_FILE.exists():
            return None
        data = json.loads(HANDOFF_FILE.read_text(encoding="utf-8"))
        summary = data.get("summary", "")
        last_user = data.get("last_user_message", "")
        turns = data.get("turns", 0)
        # Also carry forward any explicit next_step from overflow handoffs
        next_step = data.get("next_step", "")
        HANDOFF_FILE.unlink(missing_ok=True)
        if not summary:
            return None
        prompt = HANDOFF_PROMPT_TEMPLATE.format(summary=summary)
        if next_step:
            prompt += f"\n\nThe immediate next step was: {next_step}"
        return {"prompt": prompt, "last_user": last_user, "turns": turns, "next_step": next_step}
    except Exception:
        return None

def load_session(name: str) -> tuple[list | None, str | None]:
    """Load Session"""
    if not name.endswith(".json"):
        name += ".json"
    path = SESSIONS_DIR / name
    if not path.exists():
        return None, None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("history", []), data.get("system_prompt", "")

def _fuzzy_find_session(query: str) -> list[Path]:
    """Internal helper: fuzzy find session."""
    q = query.lower().removesuffix(".json")
    matches: list[Path] = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        if q in f.stem.lower():
            matches.append(f)
            continue
        if q in _session_meta(f).get("topic", "").lower():
            matches.append(f)
    return matches

def list_sessions() -> None:
    """List Sessions"""
    files = sorted(SESSIONS_DIR.glob("*.json"), reverse=True)
    if not files:
        console.print("[dim][no saved sessions][/dim]")
        return
    t = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
    t.add_column("Name",  style="cyan",    no_wrap=True)
    t.add_column("Turns", justify="right", style="dim")
    t.add_column("Topic", style="white")
    t.add_column("Saved", style="dim",     no_wrap=True)
    for f in files:
        meta  = _session_meta(f)
        saved = meta.get("saved_at", "")[:16].replace("T", " ") if meta.get("saved_at") else ""
        t.add_row(f.stem, str(meta.get("turns", "?")), meta.get("topic", "")[:60], saved)
    console.print(t)

def cmd_focus(arg: str, history: list) -> None:
    """Load specific files into context and track them for this session."""
    global _focus_set
    if not arg or arg == "list":
        if not _focus_set:
            console.print("[dim][no focused files][/dim]")
        else:
            for f in _focus_set:
                console.print(f"  [cyan]{f}[/cyan]")
        return
    if arg in ("off", "clear"):
        _focus_set.clear()
        console.print("[dim][focus tracking cleared (files remain in context)][/dim]")
        return
    try:
        tokens = shlex.split(arg)
    except Exception:
        tokens = arg.split()
    loaded = 0
    for token in tokens:
        if any(c in token for c in ("*", "?", "[")):
            hits = sorted(Path.cwd().glob(token))
            if not hits:
                console.print(f"[yellow][no files matched: {token}][/yellow]")
            for hit in hits:
                if hit.is_file() and load_file_into_context(str(hit), history):
                    _focus_set.append(str(hit))
                    loaded += 1
        else:
            p = _resolve(token)
            if not p.exists():
                console.print(f"[red][not found: {token}][/red]")
            elif p.is_dir():
                console.print(f"[yellow][{token} is a directory — use /project][/yellow]")
            elif load_file_into_context(str(p), history):
                _focus_set.append(str(p))
                loaded += 1
    if loaded:
        console.print(f"[dim][{loaded} file(s) focused — /focus to list, /focus off to clear][/dim]")


def cmd_changes(arg: str = "") -> None:
    """Show files modified by the model during this session."""
    if not _session_changes:
        console.print("[dim][no files modified this session][/dim]")
        return
    if arg == "diff":
        for path_str, original in _session_changes.items():
            p = Path(path_str)
            try:
                current = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
                diff = list(difflib.unified_diff(
                    original.splitlines(keepends=True),
                    current.splitlines(keepends=True),
                    fromfile=f"a/{p.name}", tofile=f"b/{p.name}", lineterm="",
                ))
                if diff:
                    console.print(f"\n[bold]{p.name}[/bold]  [dim]{path_str}[/dim]")
                    preview = "".join(diff[:100])
                    console.print(Syntax(preview, "diff", theme="monokai"))
                    if len(diff) > 100:
                        console.print(f"[dim]  ... ({len(diff)-100} more lines)[/dim]")
                else:
                    console.print(f"  [dim]{p.name} — no changes (possibly reverted)[/dim]")
            except Exception as e:
                console.print(f"[red][error reading {p.name}: {e}][/red]")
    else:
        console.print(f"[bold]Files modified this session ({len(_session_changes)}):[/bold]")
        for path_str, original in _session_changes.items():
            p = Path(path_str)
            try:
                current = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
                orig_lines    = len(original.splitlines())
                current_lines = len(current.splitlines())
                delta = current_lines - orig_lines
                sign  = "+" if delta >= 0 else ""
                console.print(
                    f"  [cyan]{p.name}[/cyan]  [dim]{sign}{delta} lines[/dim]  [dim]{path_str}[/dim]"
                )
            except Exception:
                console.print(f"  [cyan]{p.name}[/cyan]  [dim]{path_str}[/dim]")
        console.print("[dim]  /changes diff for full unified diffs[/dim]")


def cmd_history(arg: str, history: list) -> None:
    """Show the last N conversation turns as a readable transcript."""
    try:
        n = int(arg) if arg else 0
    except ValueError:
        n = 0
    chat = [m for m in history if m.get("role") in ("user", "assistant")]
    if not chat:
        console.print("[dim][no conversation history][/dim]")
        return
    # Pair up user/assistant turns
    pairs: list[tuple[dict, dict | None]] = []
    i = 0
    while i < len(chat):
        if chat[i].get("role") == "user":
            nxt = chat[i + 1] if i + 1 < len(chat) and chat[i + 1].get("role") == "assistant" else None
            pairs.append((chat[i], nxt))
            i += 2 if nxt else 1
        else:
            i += 1
    shown = pairs[-n:] if n else pairs
    console.print(f"[dim]Showing {len(shown)} of {len(pairs)} turn(s):[/dim]\n")
    for idx, (u, a) in enumerate(shown, len(pairs) - len(shown) + 1):
        u_text = (u.get("content") or "").strip()
        if u_text.startswith("# Project Context") or u_text.startswith("[File:") or u_text.startswith("File:"):
            u_text = "[context injection]"
        console.print(Rule(f"[dim]Turn {idx}[/dim]", style="dim"))
        console.print(f"[bold cyan]You:[/bold cyan] {u_text[:400]}{'…' if len(u_text) > 400 else ''}")
        if a:
            a_text = (a.get("content") or "").strip()
            preview = a_text[:800] + ("…" if len(a_text) > 800 else "")
            console.print(Markdown(preview))


def cmd_search_sessions(query: str) -> None:
    """Full-text search across all saved session JSON files."""
    if not query:
        console.print("[yellow][usage: /search-sessions <query>][/yellow]")
        return
    q_lower = query.lower()
    files   = sorted(SESSIONS_DIR.glob("*.json"), reverse=True)
    matches: list[tuple[str, str, str, str]] = []

    for f in files:
        if f.name == "autosave.json":
            continue
        try:
            data    = json.loads(f.read_text(encoding="utf-8"))
            meta    = data.get("meta", {})
            topic   = meta.get("topic", "")
            hist    = data.get("history", [])
            for msg in hist:
                content = msg.get("content") or ""
                if q_lower in content.lower():
                    idx     = content.lower().find(q_lower)
                    start   = max(0, idx - 40)
                    excerpt = content[start:idx + 120].replace("\n", " ").strip()
                    matches.append((f.stem, topic[:40], msg.get("role", ""), excerpt[:120]))
                    break
        except Exception:
            continue

    if not matches:
        console.print(f"[dim][no sessions matching '{query}'][/dim]")
        return

    console.print(f"[bold]{len(matches)} session(s) matching '{query}':[/bold]\n")
    t = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
    t.add_column("Session",  style="cyan", no_wrap=True, max_width=24)
    t.add_column("Topic",    style="dim",  max_width=30)
    t.add_column("Role",     style="dim",  no_wrap=True)
    t.add_column("Excerpt",  style="white")
    for stem, topic, role, excerpt in matches[:20]:
        t.add_row(stem, topic, role, excerpt)
    console.print(t)
    if len(matches) > 20:
        console.print(f"[dim]  ... and {len(matches)-20} more[/dim]")
    console.print("\n[dim]Load with: /load <session-name>[/dim]")


def cmd_params(arg: str) -> None:
    """Get or set runtime model parameters (temperature, top_p, max_tokens, presence_penalty)."""
    global _model_params, _active_preset
    _PARAM_ALIASES = {"temp": "temperature", "topp": "top_p", "max": "max_tokens", "pp": "presence_penalty"}
    _VALID = ("temperature", "top_p", "max_tokens", "presence_penalty")
    if not arg or arg in ("list", "show", ""):
        preset = SAMPLING_PRESETS.get(_active_preset, {})
        console.print(f"[bold]Preset:[/bold] [cyan]{_active_preset}[/cyan]  (thinking · code · instruct)")
        console.print(f"  temp={preset.get('temperature')}  top_p={preset.get('top_p')}  "
                      f"presence_penalty={preset.get('presence_penalty')}  max_tokens={preset.get('max_tokens')}")
        eb = preset.get("extra_body", {})
        console.print(f"  top_k={eb.get('top_k')}  min_p={eb.get('min_p')}  "
                      f"repeat_penalty={eb.get('repeat_penalty')}  "
                      f"preserve_thinking={eb.get('chat_template_kwargs', {}).get('preserve_thinking', False)}")
        if _model_params:
            console.print("[bold]Overrides:[/bold]")
            for k, v in _model_params.items():
                console.print(f"  [yellow]{k}[/yellow] = [bold]{v}[/bold]")
        console.print("[dim]  /params temp 0.6 · /params max 16384 · /params reset[/dim]")
        return
    if arg in ("reset", "clear", "off"):
        _model_params.clear()
        console.print("[dim][param overrides cleared — preset values restored][/dim]")
        return
    parts = arg.split()
    if len(parts) < 2:
        console.print("[yellow][usage: /params <name> <value>  or  /params reset][/yellow]")
        return
    name, val_str = parts[0].lower(), parts[1]
    name = _PARAM_ALIASES.get(name, name)
    if name not in _VALID:
        console.print(f"[yellow][unknown param: {name} — valid: {', '.join(_VALID)}][/yellow]")
        return
    try:
        val = int(val_str) if name == "max_tokens" else float(val_str)
        _model_params[name] = val
        console.print(f"[green][{name} = {val}][/green]")
    except ValueError:
        console.print(f"[yellow][invalid value: {val_str}][/yellow]")


def cmd_long(arg: str) -> None:
    """Toggle high-output mode (max_tokens=81,920) for hard math/programming problems.

    Qwen3 recommends 32,768 output tokens for general use and 81,920 for difficult
    competition-style problems. The larger output reservation reduces the input
    budget (context_window - 81,920 - buffer), so TOKEN_LIMIT is adjusted to match.
    """
    global _long_mode, TOKEN_LIMIT, _TOKEN_LIMIT_BASE
    a = arg.strip().lower()
    if a in ("off", "0", "no"):
        want = False
    elif a in ("on", "1", "yes"):
        want = True
    else:
        want = not _long_mode            # bare /long toggles
    if want == _long_mode:
        console.print(f"[dim][long mode already {'on' if want else 'off'}][/dim]")
        return
    if want:
        _TOKEN_LIMIT_BASE = TOKEN_LIMIT
        default_out = SAMPLING_PRESETS.get(_active_preset, {}).get("max_tokens", 32768)
        _model_params["max_tokens"] = _LONG_OUTPUT
        TOKEN_LIMIT = max(8192, _TOKEN_LIMIT_BASE + default_out - _LONG_OUTPUT)
        _long_mode = True
        console.print(
            f"[green][long mode ON][/green] max output → [bold]{_LONG_OUTPUT:,}[/bold] tokens "
            f"(input budget → ~{TOKEN_LIMIT:,}). Use for hard math/coding; [dim]/long off[/dim] to restore."
        )
    else:
        _model_params.pop("max_tokens", None)
        TOKEN_LIMIT = _TOKEN_LIMIT_BASE
        _long_mode = False
        console.print(
            f"[green][long mode OFF][/green] output restored to preset default; input budget → ~{TOKEN_LIMIT:,}."
        )


def cmd_autosearch(arg: str) -> None:
    """Show or set how eagerly the web is searched before the model answers."""
    global AUTO_SEARCH_MODE
    a = arg.strip().lower()
    if not a:
        console.print(f"[cyan]auto-search:[/cyan] [bold]{AUTO_SEARCH_MODE}[/bold]")
        console.print("[dim]  off        - never auto-search the web[/dim]")
        console.print("[dim]  smart      - search only question-shaped / factual messages (default)[/dim]")
        console.print("[dim]  aggressive - search almost every turn[/dim]")
        console.print("[dim]  usage: /autosearch off|smart|aggressive[/dim]")
        return
    if a in ("on", "max", "always", "more"):
        a = "aggressive"
    if a in ("0", "none"):
        a = "off"
    if a not in ("off", "smart", "aggressive"):
        console.print(f"[yellow][unknown mode '{a}' - use off | smart | aggressive][/yellow]")
        return
    AUTO_SEARCH_MODE = a
    console.print(f"[green]auto-search -> {a}[/green]")
    # Persist to config.toml
    try:
        text = CONFIG_FILE.read_text(encoding="utf-8")
        text = re.sub(r'auto_search\s*=\s*"[^"]*"', f'auto_search = "{a}"', text)
        CONFIG_FILE.write_text(text, encoding="utf-8")
        console.print("[dim]Saved to config.toml — persists across sessions[/dim]")
    except Exception:
        pass


def cmd_preset(arg: str) -> None:
    """Switch between Qwen3.6 official sampling presets."""
    global _active_preset
    arg = arg.strip().lower()
    aliases = {"thinking_code": "code", "think": "thinking", "tc": "code", "i": "instruct", "t": "thinking", "c": "code"}
    name = aliases.get(arg, arg)
    if not arg:
        console.print(f"[bold]Active preset:[/bold] [cyan]{_active_preset}[/cyan]")
        for n, p in SAMPLING_PRESETS.items():
            marker = "[green]✓[/green]" if n == _active_preset else " "
            eb = p.get("extra_body", {})
            pt = "[dim]preserve_thinking[/dim]" if eb.get("chat_template_kwargs", {}).get("preserve_thinking") else ""
            console.print(f"  {marker} [bold]{n}[/bold]  temp={p['temperature']}  top_p={p['top_p']}  pp={p['presence_penalty']}  {pt}")
        console.print("[dim]  /preset thinking · /preset code · /preset instruct[/dim]")
        return
    if name not in SAMPLING_PRESETS:
        console.print(f"[yellow][unknown preset '{arg}' — use: {' · '.join(SAMPLING_PRESETS)}][/yellow]")
        return
    _active_preset = name
    p = SAMPLING_PRESETS[name]
    eb = p.get("extra_body", {})
    pt = " + preserve_thinking" if eb.get("chat_template_kwargs", {}).get("preserve_thinking") else ""
    console.print(f"[green][preset → {name}][/green]  temp={p['temperature']}  top_p={p['top_p']}  max={p['max_tokens']}{pt}")
    console.print(f"[dim]  (to persist: add preset = \"{name}\" to {CONFIG_FILE})[/dim]")


def cmd_rollback() -> None:
    """Restore all files modified this session to their pre-session content."""
    if not _session_changes:
        console.print("[dim][no session changes to roll back][/dim]")
        return
    console.print(f"[bold yellow]  Roll back {len(_session_changes)} file(s) to session-start state:[/bold yellow]")
    for path_str in _session_changes:
        console.print(f"    [cyan]{Path(path_str).name}[/cyan]  [dim]{path_str}[/dim]")
    try:
        answer = console.input("[bold red]  Rollback all? [y/N]: [/bold red]").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return
    if answer != "y":
        console.print("[dim][rollback cancelled][/dim]")
        return
    restored = 0
    for path_str, original in list(_session_changes.items()):
        try:
            Path(path_str).write_text(original, encoding="utf-8")
            console.print(f"  [green]restored:[/green] {Path(path_str).name}")
            restored += 1
        except Exception as e:
            console.print(f"  [red]error: {Path(path_str).name}: {e}[/red]")
    _session_changes.clear()
    console.print(f"[green][rolled back {restored} file(s) — _session_changes cleared][/green]")


def cmd_stats(history: list) -> None:
    """Show session statistics."""
    turns = sum(1 for m in history if m.get("role") == "assistant")
    mem   = load_memory()
    mem_lines = len([ln for ln in mem.splitlines() if ln.strip()]) if mem else 0
    tok   = _last_turn_tokens
    uptime_s = time.monotonic() - _session_start if _session_start else 0
    mins, secs = int(uptime_s // 60), int(uptime_s % 60)

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="cyan")
    t.add_row("Title",          _session_title or "(untitled)")
    t.add_row("Model",          f"{MODEL}  [{_active_preset} preset]")
    ctx_tok = _real_ctx_tokens
    ctx_pct = f"{ctx_tok * 100 // TOKEN_LIMIT}%" if ctx_tok else "n/a"
    t.add_row("Turns",          str(turns))
    t.add_row("Last turn",      f"{tok.get('prompt', 0):,} in / {tok.get('completion', 0):,} out tok")
    t.add_row("Context",        f"{ctx_tok:,} / {TOKEN_LIMIT:,} tok ({ctx_pct})" if ctx_tok else "n/a (no turn yet)")
    t.add_row("Memory entries", str(mem_lines))
    t.add_row("Modified files", str(len(_session_changes)))
    t.add_row("Branches",       str(len(_branches)))
    t.add_row("Uptime",         f"{mins}m {secs:02d}s")
    t.add_row("Auto-search",    AUTO_SEARCH_MODE)
    t.add_row("Mode",           _current_mode or "default")
    console.print(t)


def cmd_review(arg: str, history: list, base_system: str, client: OpenAI) -> None:
    """Structured code review targeting a file or the current /focus set."""
    targets: list[Path] = []
    if arg:
        p = _resolve(arg)
        if not p.exists() or not p.is_file():
            console.print(f"[red][not found: {arg}][/red]")
            return
        targets = [p]
    elif _focus_set:
        targets = [Path(f) for f in _focus_set if Path(f).is_file()]
    else:
        console.print("[yellow][usage: /review <file>  or  /focus <files> then /review][/yellow]")
        return

    all_content: list[str] = []
    for p in targets:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            lang = LANG_MAP.get(p.suffix.lower(), "")
            all_content.append(f"### {p.name}\n```{lang}\n{text[:40_000]}\n```")
        except Exception as e:
            console.print(f"[red][error reading {p.name}: {e}][/red]")
    if not all_content:
        return

    review_prompt = (
        "Perform a structured code review. Use exactly these four sections:\n\n"
        "## Issues\n(Bugs, logic errors, correctness problems — ordered by severity)\n\n"
        "## Suggestions\n(Architecture, readability, performance improvements)\n\n"
        "## Security\n(Input validation, injection risks, credential handling — 'None found' if clean)\n\n"
        "## Style\n(Naming, formatting, documentation gaps)\n\n"
        "Cite file names and line numbers or function names where relevant. Be specific and concise."
    )
    msgs = [
        {"role": "system", "content": build_system_prompt(base_system)},
        *history,
        {"role": "user", "content": f"{review_prompt}\n\nFiles:\n\n" + "\n\n".join(all_content)},
    ]
    names = ", ".join(p.name for p in targets)
    console.print(Rule(f"[bold green]Review: {names}[/bold green]", style="dim green"))
    t0    = time.monotonic()
    reply = run_turn(client, msgs, allow_tools=False)
    if reply:
        history.append({"role": "user",      "content": f"[Code review: {names}]"})
        history.append({"role": "assistant",  "content": reply})
        _print_turn_footer(time.monotonic() - t0)
        try:
            save_path = targets[0].parent / (targets[0].stem + ".review.md")
            ans = console.input(f"[dim]  Save to {save_path.name}? [y/N]: [/dim]").strip().lower()
            if ans == "y":
                header = f"# Code Review: {names}\n\n*{datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n"
                save_path.write_text(header + reply, encoding="utf-8")
                console.print(f"[green][saved: {save_path}][/green]")
        except (KeyboardInterrupt, EOFError):
            pass


def cmd_error(history: list, base_system: str, client: OpenAI) -> None:
    """Paste clipboard content as an error and immediately diagnose it."""
    global _turn_count
    error_text = ""
    try:
        import pyperclip
        error_text = pyperclip.paste().strip()
        if not error_text:
            console.print("[yellow][clipboard is empty — paste error manually:][/yellow]")
    except ImportError:
        console.print("[dim][pyperclip not installed — paste error (blank line to finish):][/dim]")
    except Exception as e:
        console.print(f"[yellow][clipboard read failed: {e} — paste manually:][/yellow]")

    if not error_text:
        try:
            lines: list[str] = []
            while True:
                ln = (console.input("  ") if _pt_session is None
                      else _pt_session.prompt("  ")).rstrip()
                if not ln and lines:
                    break
                lines.append(ln)
            error_text = "\n".join(lines).strip()
        except (KeyboardInterrupt, EOFError):
            return

    if not error_text:
        console.print("[yellow][no error text provided][/yellow]")
        return

    console.print(f"[dim]  Diagnosing {len(error_text):,} chars of error output...[/dim]")
    user_content = (
        "I got this error. Diagnose it — explain the root cause and tell me exactly how to fix it:\n\n"
        f"```\n{error_text[:8_000]}\n```"
    )
    msgs = [{"role": "system", "content": build_system_prompt(base_system)}] + history
    msgs.append({"role": "user", "content": user_content})
    t0    = time.monotonic()
    reply = run_turn(client, msgs, allow_tools=True)
    if reply:
        history.append({"role": "user",      "content": user_content})
        history.append({"role": "assistant",  "content": reply})
        _turn_count += 1
        _print_turn_footer(time.monotonic() - t0)


def cmd_load_session(arg: str, history: list, base_system: str) -> tuple[list, str]:
    """Command: load session."""
    if not arg:
        list_sessions()
        return history, base_system
    exact = SESSIONS_DIR / (arg if arg.endswith(".json") else arg + ".json")
    if exact.exists():
        h, s = load_session(arg)
        if h is not None:
            turns = sum(1 for m in h if m.get("role") == "assistant")
            console.print(f"[green][loaded {exact.stem} — {turns} turns][/green]")
            return h, (s or base_system)
        return history, base_system
    matches = _fuzzy_find_session(arg)
    if not matches:
        console.print(f"[red][no session found matching '{arg}'][/red]")
        return history, base_system
    if len(matches) == 1:
        h, s = load_session(matches[0].stem)
        if h is not None:
            turns = sum(1 for m in h if m.get("role") == "assistant")
            console.print(f"[green][loaded {matches[0].stem} — {turns} turns][/green]")
            return h, (s or base_system)
        return history, base_system
    console.print(f"[yellow][multiple matches for '{arg}':][/yellow]")
    for i, f in enumerate(matches[:8], 1):
        meta = _session_meta(f)
        console.print(f"  [cyan]{i}.[/cyan] {f.stem}  [dim]{meta.get('topic','')[:50]}[/dim]")
    try:
        pick = console.input("Pick [1]: ").strip()
        idx  = (int(pick) - 1) if pick else 0
        if 0 <= idx < len(matches):
            h, s = load_session(matches[idx].stem)
            if h is not None:
                console.print(f"[green][loaded {matches[idx].stem}][/green]")
                return h, (s or base_system)
    except (ValueError, KeyboardInterrupt):
        pass
    return history, base_system

def export_session(history: list, name: str = "") -> None:
    """Export Session"""
    fname = (name or datetime.now().strftime("%Y%m%d_%H%M%S")) + ".md"
    path  = EXPORTS_DIR / fname
    lines = [
        "# Qwen Chat Export\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n",
        f"**Model:** {MODEL}\n\n---\n",
    ]
    for msg in history:
        role    = msg.get("role", "")
        content = (msg.get("content") or "").strip()
        if role == "user" and not content.startswith("# Project Context"):
            lines.append(f"\n**You:** {content}\n")
        elif role == "assistant":
            lines.append(f"\n**Qwen:**\n\n{content}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green][exported: {path}][/green]")

# ---------------------------------------------------------------------------
# Token estimate & context breakdown
# ---------------------------------------------------------------------------

def approx_tokens(messages: list) -> int:
    """Approx Tokens"""
    return sum(len((m.get("content") or "")) for m in messages) // 4

def show_context_breakdown(base_system: str, history: list) -> None:
    """Show Context Breakdown"""
    sys_text = build_system_prompt(base_system)
    sys_tok  = len(sys_text) // 4
    mem_tok  = len(load_memory()) // 4
    git_tok  = len(get_git_context()) // 4
    pin_tok  = sum(len(p) for p in load_pins()) // 4

    proj_msgs = [m for m in history if (m.get("content") or "").startswith("# Project Context")]
    proj_tok  = sum(len(m.get("content") or "") for m in proj_msgs) // 4
    chat_msgs = [m for m in history if not (m.get("content") or "").startswith("# Project Context")]
    chat_tok  = sum(len(m.get("content") or "") for m in chat_msgs) // 4
    total     = sys_tok + chat_tok
    pct       = total * 100 // TOKEN_LIMIT

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(justify="right", style="cyan")
    t.add_row("System prompt",   f"{sys_tok:,} tok")
    if mem_tok:
        t.add_row("  └ memory",  f"{mem_tok:,} tok")
    if git_tok:
        t.add_row("  └ git",     f"{git_tok:,} tok")
    if pin_tok:
        t.add_row("  └ pins",    f"{pin_tok:,} tok")
    if proj_tok:
        t.add_row("Project ctx", f"{proj_tok:,} tok")
    chat_turns = sum(1 for m in history if m.get("role") == "assistant")
    t.add_row("Chat history",    f"{chat_tok:,} tok  ({chat_turns} turns)")
    if _model_params:
        params_str = "  ".join(f"{k}={v}" for k, v in _model_params.items())
        t.add_row("Params", f"[yellow]{params_str}[/yellow]")
    if _session_changes:
        t.add_row("Modified files", f"[yellow]{len(_session_changes)}[/yellow]  /changes for details")
    t.add_row("─" * 20,         "─" * 14)
    color = "yellow" if pct >= 75 else "green"
    t.add_row("[bold]Total[/bold]",
              f"[bold {color}]~{total:,} / {TOKEN_LIMIT:,} ({pct}%)[/bold {color}]")
    console.print(t)

def warn_if_context_high(messages: list) -> None:
    """Warn If Context High"""
    tok = approx_tokens(messages)
    pct = tok * 100 // TOKEN_LIMIT
    if pct >= 75:
        console.print(
            f"[yellow]  context: ~{tok:,} / {TOKEN_LIMIT:,} tokens ({pct}%) — consider /trim[/yellow]"
        )

# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

IGNORE_DIRS = {
    ".git", "node_modules", ".venv", "venv", "__pycache__", ".next",
    "dist", "build", ".idea", ".mypy_cache", ".pytest_cache", "coverage",
    ".tox", "env", ".eggs", "target", "out", ".nuxt", ".output",
    "models",  # skip large model weight directories
}
KEY_FILES = {
    "README.md", "README.txt", "README.rst",
    "package.json", "pyproject.toml", "setup.py", "setup.cfg",
    "CLAUDE.md", "requirements.txt", "Cargo.toml", "go.mod",
    "composer.json", "Makefile", ".env.example",
}
LANG_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".jsx": "jsx", ".json": "json", ".md": "markdown", ".html": "html",
    ".css": "css", ".sh": "bash", ".ps1": "powershell", ".bat": "batch",
    ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml",
    ".rs": "rust", ".go": "go", ".cpp": "cpp", ".c": "c", ".cs": "csharp",
    ".java": "java", ".rb": "ruby", ".php": "php", ".txt": "",
}

def load_qwenignore(root: Path) -> set[str]:
    """Load Qwenignore"""
    f = root / ".qwenignore"
    if not f.exists():
        return set()
    patterns: set[str] = set()
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.add(line)
    return patterns

def _qwen_ignored(entry: Path, root: Path, extra: set[str]) -> bool:
    """Internal helper: qwen ignored."""
    name = entry.name
    rel  = str(entry.relative_to(root)).replace("\\", "/")
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(rel, p) for p in extra)

def _resolve(path: str) -> Path:
    """Internal helper: resolve."""
    p = Path(path).expanduser()
    return p.resolve(strict=False) if p.is_absolute() else (Path.cwd() / p).resolve(strict=False)

def build_project_tree(root: Path, max_depth: int = 3, max_files: int = 200) -> str:
    """Build Project Tree"""
    extra_ignore = load_qwenignore(root)
    lines: list[str] = []
    count = [0]

    def walk(path: Path, depth: int, prefix: str):
        """Walk"""
        if count[0] >= max_files:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return
        visible = [
            e for e in entries
            if not (e.is_dir()  and e.name in IGNORE_DIRS)
            and not (e.is_file() and e.suffix in {".pyc", ".pyo"})
            and not _qwen_ignored(e, root, extra_ignore)
        ]
        for i, entry in enumerate(visible):
            if count[0] >= max_files:
                lines.append(f"{prefix}  ... (truncated at {max_files} entries)")
                break
            connector = "└── " if i == len(visible) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            count[0] += 1
            if entry.is_dir() and depth < max_depth:
                ext = "    " if i == len(visible) - 1 else "│   "
                walk(entry, depth + 1, prefix + ext)

    walk(root, 0, "")
    return "\n".join(lines)

def load_project_context(arg: str, history: list) -> bool:
    """Load Project Context"""
    root = Path(arg).expanduser().resolve() if arg else Path.cwd()
    if not root.is_dir():
        console.print(f"[red][not a directory: {root}][/red]")
        return False
    console.print(f"[dim]Loading project: {root}[/dim]")
    tree = build_project_tree(root)
    key_sections: list[str] = []
    for fpath in sorted(root.iterdir(), key=lambda p: p.name):
        if fpath.name in KEY_FILES and fpath.is_file():
            try:
                text = fpath.read_text(encoding="utf-8", errors="replace")
                if len(text) > 12_000:
                    text = text[:12_000] + "\n... [truncated]"
                lang = LANG_MAP.get(fpath.suffix.lower(), "")
                key_sections.append(f"### {fpath.name}\n```{lang}\n{text}\n```")
            except Exception:
                pass
    parts = [f"# Project Context — {root}\n\n## Directory Tree\n```\n{tree}\n```"]
    if key_sections:
        parts.append("## Key Files\n\n" + "\n\n".join(key_sections))
    history.append({"role": "user", "content": "\n\n".join(parts)})
    console.print(
        f"[green][project loaded: {root.name} — {len(tree.splitlines())} tree lines"
        f"{f', {len(key_sections)} key file(s)' if key_sections else ''}][/green]"
    )
    # Build or load symbol index in background
    def _bg_index():
        """Internal helper: bg index."""
        idx = _load_symbol_index(root)
        if idx is None:
            idx = build_symbol_index(root)
            _save_symbol_index(root, idx)
        global _cached_index, _cached_index_root
        _cached_index, _cached_index_root = idx, root
        sym_count = sum(len(s.get("functions", [])) + len(s.get("classes", []))
                        for s in idx.values())
        console.print(f"[dim][indexed: {len(idx)} files, {sym_count} symbols — /index force to refresh][/dim]")
    threading.Thread(target=_bg_index, daemon=True).start()
    return True

def load_file_into_context(arg: str, history: list) -> bool:
    """Load File Into Context"""
    path = Path(arg.strip('"').strip("'").strip())
    if not path.is_absolute() and not path.exists():
        path = Path.home() / path
    if not path.exists():
        console.print(f"[red][file not found: {arg}][/red]")
        return False
    try:
        raw = path.read_bytes()
        if b"\x00" in raw[:8192]:
            console.print(f"[red][binary file not supported: {path.name}][/red]")
            return False
        text   = raw.decode("utf-8", errors="replace")
        size   = len(raw)
        nlines = text.count("\n") + 1
        if size > 150_000:
            console.print(f"[yellow][large file ({size:,} bytes) — truncating to 150 KB][/yellow]")
            text = text[:150_000]
        lang    = LANG_MAP.get(path.suffix.lower(), "")
        content = f"File: `{path.name}` ({nlines:,} lines, {size:,} bytes)\n\n```{lang}\n{text}\n```"
        history.append({"role": "user", "content": content})
        console.print(f"[green][loaded: {path.name} — {nlines:,} lines, {size:,} bytes][/green]")
        return True
    except Exception as e:
        console.print(f"[red][error reading file: {e}][/red]")
        return False

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _is_dangerous(command: str) -> bool:
    """Check if a command is dangerous, including expanded forms."""
    # Check the raw command
    if _DANGEROUS_CMD_RE.search(command):
        return True
    # Check for variable expansion patterns
    import re as _re
    # Detect: CMD=rm; $CMD -rf /
    if _re.search(r"\$[A-Z_]+\s+(-[a-zA-Z]*r|rm|mkfs|dd|chmod)", command):
        return True
    # Detect: base64 encoded commands
    if _re.search(r"base64\s+(-d|--decode)", command):
        return True
    # Detect: eval/exec with suspicious content
    if _re.search(r"eval\s+.*\$\(", command):
        return True
    # Detect: command substitution with dangerous commands
    if _re.search(r"\$\(.*\b(rm|mkfs|dd|chmod)\b", command):
        return True
    # Detect: pipe to shell (curl/wget ... | bash/sh/python)
    if _re.search(r"\|\s*(bash|sh|zsh|fish|python\d*|perl|ruby)\b", command):
        return True
    return False

# ---------------------------------------------------------------------------
# Browser automation (Playwright) — stealth, smart waits, CAPTCHA pause, cookies
# ---------------------------------------------------------------------------

_browser_state: dict = {}
_render_state:  dict = {}   # dedicated headless page for fetch_rendered (separate from browser_action)
COOKIE_FILE = DATA_DIR / "browser_cookies.json"

_STEALTH_JS = """// === Comprehensive Anti-Detection ===

// 0. Block Playwright-specific Function.toString leak
(function() {
  const _origToString = Function.prototype.toString;
  Function.prototype.toString = function() {
    if (typeof this === 'function' && this.name) {
      return 'function ' + this.name + '() { [native code] }';
    }
    return _origToString.apply(this, arguments);
  };
  Function.prototype.toString.toString = () => 'function toString() { [native code] }';
})();

// 1. Hide webdriver flag — multiple detection vectors
Object.defineProperty(navigator, 'webdriver', {
  get: () => undefined,
  configurable: true,
  enumerable: true,
});

// 1b. hasOwnProperty('webdriver') detection bypass
(function() {
  const _origHas = Object.prototype.hasOwnProperty;
  Object.prototype.hasOwnProperty = function(prop) {
    if (this === navigator && prop === 'webdriver') return false;
    return _origHas.apply(this, arguments);
  };
})();

// 2. Full chrome object (with missing app, support, runtime)
window.chrome = {
  runtime: {
    onMessage: { addListener: () => {}, removeListener: () => {}, hasListener: () => false },
    connect: () => ({ onMessage: { addListener: () => {}, removeListener: () => {} } }),
    sendMessage: () => {},
    executionContext: 1,
  },
  app: {
    isInstalled: false,
    InstallState: { disabled: 'disabled', installed: 'installed', not_installed: 'not_installed' },
    RunningState: { running: 'running', not_running: 'not_running' },
    getDetails: () => ({ id: '' }),
  },
  loadTimes: function() {},
  csi: function() {},
  support: { createScript: () => {}, removeScript: () => {} },
};

// 3. Plugin and MimeType arrays
const _plugins = [
  {name:'Chrome PDF Plugin', filename:'internal-pdf-viewer', description:'Portable Document Format', length:1},
  {name:'Chrome PDF Viewer', filename:'mhjfbmdgcfjbbpaeojofohoefgiehjai', description:'Chrome PDF Plugin', length:1},
  {name:'Widevine Content Decryption Module', filename:'widevinecdm.dll', description:'Widevine', length:1},
];
Object.defineProperty(navigator, 'plugins', {
  get: () => {
    const arr = Object.values(_plugins).map(p => Object.assign({}, p, { enabled: true }));
    arr.length = _plugins.length;
    return arr;
  },
  configurable: true,
});

const _mimeTypes = [
  {type:'application/pdf', suffixes:'pdf', description:'Portable Document Format'},
  {type:'application/x-google-chrome-pdf', suffixes:'pdf', description:'Portable Document Format'},
];
Object.defineProperty(navigator, 'mimeTypes', {
  get: () => {
    const arr = Object.values(_mimeTypes).map(m => Object.assign({}, m));
    arr.length = _mimeTypes.length;
    return arr;
  },
  configurable: true,
});

// 4. Standard navigator properties
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'], configurable: true });
Object.defineProperty(navigator, 'hardwareConcurrency', { get: () => 4, configurable: true });
Object.defineProperty(navigator, 'maxTouchPoints', { get: () => 1, configurable: true });
Object.defineProperty(navigator, 'deviceMemory', { get: () => 8, configurable: true });
Object.defineProperty(navigator, 'vendor', { get: () => 'Google Inc.', configurable: true });
Object.defineProperty(navigator, 'platform', { get: () => 'Win32', configurable: true });
Object.defineProperty(navigator, 'pdfViewer', { get: () => true, configurable: true });

// 4b. WebRTC IP leak prevention
try {
  const _origRTCPeer = window.RTCPeerConnection;
  if (_origRTCPeer) {
    window.RTCPeerConnection = function(config) {
      if (config && config.iceServers) config.iceServers = [];
      return new _origRTCPeer(config);
    };
    window.RTCPeerConnection.prototype = _origRTCPeer.prototype;
  }
} catch(e) {}

// 5. Screen properties and outer dimensions (headless browsers leak outerWidth=0)
const _screenProps = { colorDepth: 24, pixelDepth: 24 };
Object.defineProperty(window, 'screen', {
  get: () => Object.assign({}, screen, _screenProps, {
    orientation: { type: 'landscape-primary', angle: 0, onchange: null },
  }),
  configurable: true,
  enumerable: true,
});
if (window.outerWidth === 0 || window.outerHeight === 0) {
  Object.defineProperty(window, 'outerWidth', { get: () => window.innerWidth + 16, configurable: true });
  Object.defineProperty(window, 'outerHeight', { get: () => window.innerHeight + 78, configurable: true });
}

// 5b. Screen position offsets (real browsers are rarely exactly 0,0)
if (window.screenLeft === 0 && window.screenTop === 0) {
  Object.defineProperty(window, 'screenLeft', { value: 20, configurable: true });
  Object.defineProperty(window, 'screenTop', { value: 20, configurable: true });
  Object.defineProperty(window, 'screenX', { value: 20, configurable: true });
  Object.defineProperty(window, 'screenY', { value: 20, configurable: true });
}

// 6. navigator.permissions override
const origQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (params) =>
    params.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : origQuery(params);

// 6b. TrustedTypes — present in real Chrome, absent in headless
try {
  if (!window.trustedTypes) {
    window.trustedTypes = {
      createPolicy: (name, config) => ({
        createScript: (s) => s,
        createScriptUrl: (s) => s,
        createScriptElement: (s) => null,
        createStyle: (s) => s,
        createURL: (s) => s,
      }),
      isHTML: () => false,
      isScriptURL: () => false,
    };
  }
} catch(e) {}

// 7. WebGL 1.0 and 2.0 vendor/renderer spoofing
const glParam = WebGLRenderingContext.prototype.getParameter;
WebGLRenderingContext.prototype.getParameter = function(param) {
  if (param === 37445) return 'Google Inc. (NVIDIA)';
  if (param === 37446) return 'ANGLE (NVIDIA, NVIDIA GeForce, Direct3D)';
  return glParam.apply(this, arguments);
};
try {
  const gl2Param = WebGL2RenderingContext.prototype.getParameter;
  WebGL2RenderingContext.prototype.getParameter = function(param) {
    if (param === 37445) return 'Google Inc. (NVIDIA)';
    if (param === 37446) return 'ANGLE (NVIDIA, NVIDIA GeForce, Direct3D12)';
    return gl2Param.apply(this, arguments);
  };
} catch(e) {}

// 8. Canvas fingerprint noise — perturb sub-pixel rendering
const _origGetContext = HTMLCanvasElement.prototype.getContext;
HTMLCanvasElement.prototype.getContext = function(type) {
  const ctx = _origGetContext.apply(this, arguments);
  if (!ctx || type !== '2d') return ctx;
  const _fillText = ctx.fillText.bind(ctx);
  ctx.fillText = function(...args) {
    ctx.imageSmoothingEnabled = false;
    return _fillText(...args);
  };
  return ctx;
};

// 9. AudioContext fingerprint noise
try {
  const _origGetFloat = AnalyserNode.prototype.getFloatFrequencyData;
  AnalyserNode.prototype.getFloatFrequencyData = function(freqData) {
    _origGetFloat.call(this, freqData);
    for (let i = 0; i < freqData.length; i++) {
      if (!isNaN(freqData[i])) {
        freqData[i] += (Math.random() - 0.5) * 1e-30;
      }
    }
  };
} catch(e) {}

// 10. navigator.connection
Object.defineProperty(navigator, 'connection', {
  get: () => ({ downlink: 10, effectiveType: '4g', rtt: 50, saveData: false }),
  configurable: true,
});

// 11. Remove iframe sandbox detection
try { delete HTMLIFrameElement.prototype.sandbox; } catch(e) {}

// 12. Performance entries — filter out devtools/CDP URLs
const _origPerfEntries = performance.getEntriesByType.bind(performance);
performance.getEntriesByType = function(type) {
  if (type === 'resource') {
    return _origPerfEntries(type).filter(e =>
      !e.name.includes('/devtools/') && !e.name.includes('chrome-devtools')
    );
  }
  return _origPerfEntries(type);
};

// 13. Deep toString override for patched objects
[
  [window.navigator, ['plugins', 'mimeTypes', 'languages', 'permissions', 'connection', 'hardwareConcurrency', 'deviceMemory']],
  [window, ['chrome', 'screen']],
].forEach(([obj, keys]) => {
  keys.forEach(key => {
    try {
      const desc = Object.getOwnPropertyDescriptor(obj, key);
      if (desc && desc.get) {
        Object.defineProperty(desc.get, 'toString', {
          value: () => `function ${key}() { [native code] }`,
          configurable: true,
        });
      }
    } catch(e) {}
  });
});

// 14. Constructor toString override
[window.navigator, window].forEach(obj => {
  Object.getOwnPropertyNames(obj.constructor).forEach(key => {
    try {
      Object.defineProperty(obj.constructor[key], 'toString', {
        value: () => `function ${key}() { [native code] }`,
        configurable: true,
      });
    } catch(e) {}
  });
});

// 15. navigator.serviceWorker — suppress errors gracefully
if (navigator.serviceWorker) {
  const _swReg = navigator.serviceWorker.register.bind(navigator.serviceWorker);
  navigator.serviceWorker.register = function(...args) {
    return _swReg(...args).catch(() => {});
  };
}

// 16. IntersectionObserver timing — headless fires immediately, real browsers defer
try {
  const _origIO = window.IntersectionObserver;
  window.IntersectionObserver = function(callback, options) {
    const observer = new _origIO(callback, options);
    const _origObserve = observer.observe.bind(observer);
    observer.observe = (el) => setTimeout(() => _origObserve(el), 50);
    return observer;
  };
  window.IntersectionObserver.prototype = _origIO.prototype;
} catch(e) {}
"""

_CAPTCHA_SIGNALS = [
    "captcha", "verify you are human", "i am not a robot",
    "hcaptcha", "recaptcha", "cf-challenge", "cloudflare",
    "bot verification", "security check",
]


_ANTIBOT_RESPONSE_PATTERNS = [
    "challenge", "blocked", "forbidden", "access denied", "rate limit",
    "too many requests", "please try again later", "suspicious activity",
    "cf-turnstile", "cloudflare turnstile", "checking your browser",
    "under attack mode", "ddos protection", "waf block",
    "perimeter x", "datadome", "imperva", "shape security",
    "you need to enable javascript", "blocked by firewall",
]


def _browser_detect_antibot(page) -> str:
    """Check if the page is showing an anti-bot block page. Returns hint string or empty."""
    try:
        title = page.title().lower()
        body = ""
        try:
            body = page.inner_text("body", timeout=3000).lower()
        except Exception:
            pass
        content = title + " " + body
        for pat in _ANTIBOT_RESPONSE_PATTERNS:
            if pat in content:
                return pat
    except Exception:
        pass
    return ""


def _browser_random_viewport() -> dict:
    """Return a randomized viewport to avoid fingerprint matching."""
    import random as _r
    widths = [1280, 1366, 1440, 1536, 1600, 1920]
    heights = [720, 768, 800, 900, 1024, 1080, 1200]
    return {"width": _r.choice(widths), "height": _r.choice(heights)}


def _browser_random_ua() -> str:
    """Return a randomized but realistic user-agent string."""
    import random as _r
    major = _r.choice(["131", "130", "129", "128", "127"])
    minor = _r.randint(0, 999)
    return (
        f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        f"AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{major}.{minor}.0.0 Safari/537.36"
    )


def _browser_random_delay(min_ms: float = 100, max_ms: float = 600) -> None:
    """Introduce a small random delay to simulate human timing."""
    import random as _r
    import time as _t
    _t.sleep(_r.uniform(min_ms / 1000, max_ms / 1000))



def _browser_jitter_mouse(page) -> None:
    """Move mouse slightly in a natural-looking pattern before interactions."""
    import random as _r
    _steps = _r.randint(2, 5)
    for _ in range(_steps):
        _dx = _r.randint(-30, 30)
        _dy = _r.randint(-10, 10)
        page.mouse.move(_dx, _dy, steps=_r.randint(3, 8))

def _browser_has_captcha(page) -> bool:
    """Internal helper: browser has captcha."""
    try:
        content = (page.title() + " " + page.inner_text("body", timeout=3000)).lower()
        return any(kw in content for kw in _CAPTCHA_SIGNALS)
    except Exception:
        return False


def _browser_load_cookies(context) -> None:
    """Internal helper: browser load cookies."""
    if COOKIE_FILE.exists():
        try:
            context.add_cookies(json.loads(COOKIE_FILE.read_text()))
        except Exception:
            pass


def _browser_save_cookies(page) -> None:
    """Internal helper: browser save cookies."""
    try:
        COOKIE_FILE.write_text(json.dumps(page.context.cookies()))
    except Exception:
        pass


def _get_page():
    """Internal helper: get page."""
    if "page" not in _browser_state or _browser_state.get("closed"):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError("playwright not installed — run: pip install playwright && playwright install chromium")
        pw = sync_playwright().start()
        _ua = _browser_random_ua()
        _major = _ua.split("Chrome/")[1].split(".")[0] if "Chrome/" in _ua else "131"
        browser = pw.chromium.launch(
            headless=False,  # headed mode is harder to detect
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-site-isolation-trials",
                "--disable-infobars",
                "--disable-extensions",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-accelerated-2d-canvas",
                "--disable-dev-shm-usage",
                "--lang=en-US",
            ],
        )
        ctx = browser.new_context(
            user_agent=_ua,
            viewport=_browser_random_viewport(),
            locale="en-US",
            timezone_id="America/New_York",
            geolocation={"latitude": 40.7128, "longitude": -74.0060},
            permissions=["geolocation"],
            extra_http_headers={
                "sec-ch-ua": f'"Chromium";v="{_major}", "Google Chrome";v="{_major}"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "document",
                "sec-fetch-mode": "navigate",
                "sec-fetch-site": "none",
                "sec-fetch-user": "?1",
                "accept-language": "en-US,en;q=0.9",
            },
            color_scheme="light",
        )
        # Proxy from config (optional)
        _cfg_data = _cfg()
        _proxy_url = _cfg_data.get("browser_proxy", "")
        if _proxy_url:
            console.print(f"[dim]  [browser] using proxy: {_proxy_url}[/dim]")
        ctx.add_init_script(_STEALTH_JS)
        _browser_load_cookies(ctx)
        page = ctx.new_page()
        _browser_state.update(playwright=pw, browser=browser, context=ctx, page=page, closed=False)
    return _browser_state["page"]


def _browser_resolve_selector(page, selector: str):
    """Internal helper: browser resolve selector."""
    if selector.startswith("label:"):
        return page.get_by_label(selector[6:])
    if selector.startswith("button:"):
        return page.get_by_role("button", name=selector[7:])
    if selector.startswith("link:"):
        return page.get_by_role("link", name=selector[5:])
    if selector.startswith("text:"):
        return page.get_by_text(selector[5:])
    return page.locator(selector)


def _browser_smart_wait(page, timeout: int = 8000) -> None:
    """Wait for the page to settle — networkidle with domcontentloaded fallback."""
    try:
        page.wait_for_load_state("networkidle", timeout=timeout)
    except Exception:
        try:
            page.wait_for_load_state("domcontentloaded", timeout=timeout)
        except Exception:
            pass


def _browser_check_captcha_pause(page) -> str:
    """If a CAPTCHA is detected, pause and let the user solve it. Returns status string."""
    if not _browser_has_captcha(page):
        return ""
    console.print("[bold yellow]  [browser] CAPTCHA / human-verification detected.[/bold yellow]")
    console.print("[bold yellow]  Solve it in the browser window, then press Enter to continue.[/bold yellow]")
    try:
        console.input("")
    except (KeyboardInterrupt, EOFError):
        pass
    _browser_save_cookies(page)
    return "[captcha-paused: user resolved]"


def do_browser_action(action: str, url: str = "", selector: str = "",
                      value: str = "", screenshot_path: str = "") -> str:
    """Control a real Chromium browser via playwright automation.

    Actions: navigate, fill, type, click, select, submit, wait_for, scroll,
    hover, press_key, screenshot, get_text, get_url, get_links, close.
    Returns output text or status messages.
    """
    try:
        if action == "close":
            if "browser" in _browser_state:
                _browser_save_cookies(_browser_state["page"])
                _browser_state["browser"].close()
            if "playwright" in _browser_state:
                _browser_state["playwright"].stop()
            _browser_state.clear()
            _browser_state["closed"] = True
            return "[browser closed — cookies saved]"

        page = _get_page()

        if action == "navigate":
            if not url:
                return "[navigate requires a url]"
            console.print(f"[bold cyan]  [browser][/bold cyan] navigate → {url}")
            last_nav_err = None
            for _attempt in range(3):
                if _attempt > 0:
                    import time as _time
                    _time.sleep(1.5 * _attempt)
                    console.print(f"[dim]  [browser] retry {_attempt}/2...[/dim]")
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    last_nav_err = None
                    break
                except Exception as _nav_err:
                    last_nav_err = _nav_err
                    _e = str(_nav_err)
                    if "net::" not in _e and "ERR_" not in _e and "timeout" not in _e.lower():
                        break
            if last_nav_err:
                return f"[browser navigation failed: {last_nav_err}]"
            _browser_smart_wait(page, timeout=12000)
            # Check for anti-bot pages
            antibot_pat = _browser_detect_antibot(page)
            if antibot_pat:
                console.print(f"[bold yellow]  [browser] anti-bot signal detected: {antibot_pat}[/bold yellow]")
                _browser_random_delay()
            captcha_note = _browser_check_captcha_pause(page)
            _browser_save_cookies(page)
            title = page.title()
            result = f"[navigated to: {url}]\nPage title: {title}"
            if antibot_pat:
                result += f"\n[anti-bot signal: {antibot_pat} — proceeding with caution]"
            if captcha_note:
                result += f"\n{captcha_note}"
            return result

        if action == "fill":
            if not selector or value is None:
                return "[fill requires selector and value]"
            console.print(f"[bold cyan]  [browser][/bold cyan] fill {selector!r} = {value!r}")
            loc = _browser_resolve_selector(page, selector)
            loc.first.wait_for(state="visible", timeout=8000)
            _browser_random_delay(200, 500)
            _browser_jitter_mouse(page)
            loc.first.fill(value)
            return f"[filled {selector!r} with {value!r}]"

        if action == "type":
            # Human-like character-by-character typing (helps bypass keystroke detection)
            if not selector or value is None:
                return "[type requires selector and value]"
            console.print(f"[bold cyan]  [browser][/bold cyan] type {selector!r} = {value!r}")
            loc = _browser_resolve_selector(page, selector)
            loc.first.wait_for(state="visible", timeout=8000)
            loc.first.click()
            loc.first.type(value, delay=60)
            return f"[typed into {selector!r}]"

        if action == "click":
            if not selector:
                return "[click requires a selector]"
            console.print(f"[bold cyan]  [browser][/bold cyan] click {selector!r}")
            loc = _browser_resolve_selector(page, selector)
            loc.first.wait_for(state="visible", timeout=8000)
            _browser_random_delay(150, 400)
            _browser_jitter_mouse(page)
            loc.first.click()
            _browser_smart_wait(page)
            captcha_note = _browser_check_captcha_pause(page)
            _browser_save_cookies(page)
            result = f"[clicked {selector!r} — now at: {page.url}]"
            if captcha_note:
                result += f"\n{captcha_note}"
            return result

        if action == "select":
            if not selector or not value:
                return "[select requires selector and value]"
            console.print(f"[bold cyan]  [browser][/bold cyan] select {selector!r} = {value!r}")
            loc = _browser_resolve_selector(page, selector)
            loc.first.wait_for(state="visible", timeout=8000)
            loc.first.select_option(label=value)
            return f"[selected {value!r} in {selector!r}]"

        if action == "submit":
            target = selector or "form"
            console.print(f"[bold cyan]  [browser][/bold cyan] submit {target!r}")
            if selector:
                _browser_resolve_selector(page, selector).first.press("Enter")
            else:
                page.locator("form").first.evaluate("f => f.submit()")
            _browser_smart_wait(page, timeout=15000)
            captcha_note = _browser_check_captcha_pause(page)
            _browser_save_cookies(page)
            result = f"[form submitted — now at: {page.url}]"
            if captcha_note:
                result += f"\n{captcha_note}"
            return result

        if action == "wait_for":
            if not selector:
                return "[wait_for requires a selector]"
            console.print(f"[bold cyan]  [browser][/bold cyan] wait_for {selector!r}")
            _browser_resolve_selector(page, selector).first.wait_for(state="visible", timeout=15000)
            return f"[element visible: {selector!r}]"

        if action == "scroll":
            if selector:
                console.print(f"[bold cyan]  [browser][/bold cyan] scroll to {selector!r}")
                _browser_resolve_selector(page, selector).first.scroll_into_view_if_needed()
                return f"[scrolled to {selector!r}]"
            pixels = int(value) if value and value.lstrip("-").isdigit() else 0
            if pixels:
                console.print(f"[bold cyan]  [browser][/bold cyan] scroll {pixels}px")
                page.evaluate(f"window.scrollBy(0, {pixels})")
                return f"[scrolled by {pixels}px]"
            console.print("[bold cyan]  [browser][/bold cyan] scroll down")
            page.evaluate("window.scrollBy(0, window.innerHeight)")
            return "[scrolled down one page]"

        if action == "hover":
            if not selector:
                return "[hover requires a selector]"
            console.print(f"[bold cyan]  [browser][/bold cyan] hover {selector!r}")
            loc = _browser_resolve_selector(page, selector)
            loc.first.wait_for(state="visible", timeout=8000)
            _browser_random_delay()
            loc.first.hover()
            return f"[hovered over {selector!r}]"

        if action == "press_key":
            if not value:
                return "[press_key requires a value (e.g. 'Enter', 'Tab', 'Escape', 'Control+A')]"
            console.print(f"[bold cyan]  [browser][/bold cyan] press_key {value!r}")
            if selector:
                _browser_resolve_selector(page, selector).first.press(value)
            else:
                page.keyboard.press(value)
            return f"[pressed {value!r}]"

        if action == "get_url":
            return f"[current URL: {page.url}]"

        if action == "get_links":
            console.print("[bold cyan]  [browser][/bold cyan] get_links")
            links = page.evaluate(
                "() => Array.from(document.querySelectorAll('a[href]'))"
                ".map(a => ({text: a.innerText.trim().slice(0,100), href: a.href}))"
                ".filter(l => l.href.startsWith('http')).slice(0, 50)"
            )
            if not links:
                return "[no links found on page]"
            lines = [f"[links on {page.url}]"]
            for lnk in links:
                lines.append(f"  {lnk['text'][:40]!r:42} → {lnk['href']}")
            return "\n".join(lines)

        if action == "screenshot":
            path = screenshot_path or str(Path.home() / "screenshot.png")
            console.print(f"[bold cyan]  [browser][/bold cyan] screenshot → {path}")
            page.screenshot(path=path, full_page=True)
            return f"[screenshot saved: {path}]\nURL: {page.url}\nTitle: {page.title()}"

        if action == "get_text":
            console.print("[bold cyan]  [browser][/bold cyan] get_text")
            if selector:
                text = _browser_resolve_selector(page, selector).first.inner_text()
            else:
                text = page.inner_text("body")
            if len(text) > 16000:
                text = text[:16000] + "\n...[truncated]"
            return f"[page text — {page.url}]\n{text}"

        return f"[unknown browser action: {action}]"

    except Exception as e:
        _emsg = str(e)
        if "net::" in _emsg or "ERR_" in _emsg:
            return (f"[browser network error: {e}]\n"
                    "Tip: Check the URL or your connection. Try a different URL or use fetch_url instead.")
        if "timeout" in _emsg.lower():
            return (f"[browser timeout: {e}]\n"
                    "Tip: The page may be JS-heavy or slow — try increasing wait time or use get_text after waiting.")
        if "closed" in _emsg.lower() or "Target page" in _emsg:
            _browser_state.clear()
            _browser_state["closed"] = True
            return (f"[browser closed unexpectedly: {e}]\n"
                    "Tip: Use navigate action to reopen the browser.")
        if "captcha" in _emsg.lower() or "challenge" in _emsg.lower():
            return (f"[browser blocked (anti-bot): {e}]\n"
                    "Tip: The site may require manual CAPTCHA solving — navigate there first.")
        return f"[browser error: {e}]"


def _get_render_page():
    """Get (or create) the dedicated headless Playwright page used by fetch_rendered."""
    if "page" not in _render_state or _render_state.get("closed"):
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError("playwright not installed — run: pip install playwright && playwright install chromium")
        pw  = sync_playwright().start()
        _ua = _browser_random_ua()
        _major = _ua.split("Chrome/")[1].split(".")[0] if "Chrome/" in _ua else "131"
        bro = pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-features=IsolateOrigins,site-per-process",
                "--disable-site-isolation-trials",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-accelerated-2d-canvas",
                "--disable-dev-shm-usage",
                "--lang=en-US",
            ],
        )
        ctx = bro.new_context(
            user_agent=_ua,
            viewport=_browser_random_viewport(),
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={
                "sec-ch-ua": f'"Chromium";v="{_major}", "Google Chrome";v="{_major}"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "document",
                "sec-fetch-mode": "navigate",
                "sec-fetch-site": "none",
                "sec-fetch-user": "?1",
                "accept-language": "en-US,en;q=0.9",
            },
            color_scheme="light",
        )
        ctx.add_init_script(_STEALTH_JS)
        page = ctx.new_page()
        _render_state.update(playwright=pw, browser=bro, context=ctx, page=page, closed=False)
    return _render_state["page"]


def do_fetch_rendered(url: str, max_chars: int = 15000) -> str:
    """
    Fetch a URL with full JS rendering via a dedicated headless Playwright instance.
    Separate from browser_action so it never shares state or interferes.
    Uses trafilatura/readability on the rendered HTML for clean text extraction.
    """
    try:
        page = _get_render_page()
        console.print(f"[dim cyan]  fetch_rendered: {url}[/dim cyan]")

        # Navigate with retry for transient network errors
        last_err = None
        for _attempt in range(3):
            if _attempt:
                import time as _t
                _t.sleep(1.5 * _attempt)
                console.print(f"[dim]  [fetch_rendered] retry {_attempt}/2...[/dim]")
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
                last_err = None
                break
            except Exception as _nav_err:
                last_err = _nav_err
                _e = str(_nav_err)
                if "net::" not in _e and "ERR_" not in _e and "timeout" not in _e.lower():
                    break
        if last_err:
            return f"[fetch_rendered navigation failed: {last_err}]"

        # Wait for JS content to settle
        _browser_smart_wait(page, timeout=10000)

        # Check for CAPTCHA (headless triggers it less, but still possible)
        captcha_note = _browser_check_captcha_pause(page)

        # Extract text: prefer trafilatura/readability on full rendered HTML
        try:
            html = page.content()
            text = _html_to_text(html, url=url)
        except Exception:
            text = page.inner_text("body")

        if len(text) > max_chars:
            # Smart truncate at sentence boundary
            window = text[max(0, max_chars - 500): max_chars]
            last_break = max(window.rfind(". "), window.rfind("\n\n"))
            cut = (max(0, max_chars - 500) + last_break + 1) if last_break > 0 else max_chars
            text = text[:cut] + "\n...[truncated]"

        title  = page.title()
        result = f"[Rendered: {url}]\nTitle: {title}\n\n{text}"
        if captcha_note:
            result += f"\n{captcha_note}"
        return result

    except Exception as e:
        _emsg = str(e)
        if "closed" in _emsg.lower() or "Target page" in _emsg:
            _render_state.clear()
            _render_state["closed"] = True
        return f"[fetch_rendered error: {e}]"


_MAX_CMD_OUTPUT = 25_000

_SCRIPT_INTERP: dict[str, tuple[str, str]] = {
    "python":     (sys.executable, ".py"),
    "python3":    (sys.executable, ".py"),
    "py":         (sys.executable, ".py"),
    "javascript": ("node", ".js"),
    "js":         ("node", ".js"),
    "node":       ("node", ".js"),
    "bash":       ("bash", ".sh"),
    "sh":         ("sh", ".sh"),
    "powershell": ("pwsh -NoProfile -ExecutionPolicy Bypass -File", ".ps1"),
    "ps1":        ("pwsh -NoProfile -ExecutionPolicy Bypass -File", ".ps1"),
    "ruby":       ("ruby", ".rb"),
    "php":        ("php", ".php"),
    "lua":        ("lua", ".lua"),
    "perl":       ("perl", ".pl"),
    "go":         ("go run", ".go"),
    "r":          ("Rscript", ".r"),
    "typescript": ("npx ts-node", ".ts"),
    "ts":         ("npx ts-node", ".ts"),
}


def do_run_command(command: str, cwd: str = "", timeout: int = 30,
                   env: dict | None = None, stdin: str = "", quiet: bool = False) -> str:
    """Run a shell command and return its stdout + stderr in real-time.

    Dangerous commands (rm -rf, format, etc.) require explicit confirmation.
    """
    work_dir = _resolve(cwd) if cwd else Path.cwd()
    if _is_dangerous(command):
        console.print(f"[bold red]  [dangerous][/bold red] {command}")
        answer = console.input("[bold red]  Run anyway? [y/N]:[/bold red] ").strip().lower()
        if answer != "y":
            return "[command cancelled by user]"
    elif not quiet:
        console.print(f"[bold yellow]  [run_command][/bold yellow] {command}")

    proc_env  = {**os.environ, **env} if env else None
    stdout_buf: list[str] = []
    stderr_buf: list[str] = []
    cancelled  = threading.Event()
    t0 = time.monotonic()

    try:
        proc = subprocess.Popen(
            command, shell=True, cwd=work_dir,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
            env=proc_env,
        )
        if stdin:
            try:
                proc.stdin.write(stdin)
                proc.stdin.close()
            except BrokenPipeError:
                pass

        def _pipe(stream, buf: list[str], style: str):
            """Internal helper: pipe."""
            for line in stream:
                if cancelled.is_set():
                    break
                buf.append(line)
                console.print(f"[{style}]{line.rstrip()}[/{style}]")

        t_out = threading.Thread(target=_pipe, args=(proc.stdout, stdout_buf, "dim"),        daemon=True)
        t_err = threading.Thread(target=_pipe, args=(proc.stderr, stderr_buf, "dim yellow"), daemon=True)
        t_out.start()
        t_err.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            cancelled.set()
            proc.kill()
            t_out.join(timeout=1)
            t_err.join(timeout=1)
            partial = "".join(stdout_buf).strip()
            return f"[timed out after {timeout}s]\n\n{partial}" if partial else f"[timed out after {timeout}s]"
        except KeyboardInterrupt:
            cancelled.set()
            proc.kill()
            t_out.join(timeout=1)
            t_err.join(timeout=1)
            console.print("\n[dim][command cancelled][/dim]")
            partial = "".join(stdout_buf).strip()
            return f"[cancelled]\n\n{partial}" if partial else "[cancelled]"

        t_out.join(timeout=2)
        t_err.join(timeout=2)
        elapsed = time.monotonic() - t0

        full_out = "".join(stdout_buf).strip()
        full_err = "".join(stderr_buf).strip()
        if len(full_out) > _MAX_CMD_OUTPUT:
            full_out = full_out[:_MAX_CMD_OUTPUT] + f"\n... [stdout truncated at {_MAX_CMD_OUTPUT} chars]"
        if len(full_err) > _MAX_CMD_OUTPUT:
            full_err = full_err[:_MAX_CMD_OUTPUT] + f"\n... [stderr truncated at {_MAX_CMD_OUTPUT} chars]"

        parts: list[str] = []
        if full_out:
            parts.append("stdout:\n" + full_out)
        if full_err:
            parts.append("stderr:\n" + full_err)
        rc_note = "✓" if proc.returncode == 0 else "✗"
        parts.append(f"exit code: {proc.returncode} {rc_note}  ({elapsed:.1f}s)")
        return "\n\n".join(parts)
    except Exception as e:
        return f"[error: {e}]"


def do_run_script(language: str, code: str, cwd: str = "", timeout: int = 30) -> str:
    """Execute inline code by writing to a temp file and running with the right interpreter."""
    import tempfile as _tmpmod
    lang_key = language.lower().strip()
    if lang_key not in _SCRIPT_INTERP:
        supported = ", ".join(sorted(_SCRIPT_INTERP))
        return f"[unsupported language: {language!r} — supported: {supported}]"
    interp, ext = _SCRIPT_INTERP[lang_key]
    if _DANGEROUS_CMD_RE.search(code):
        return "[script blocked: contains dangerous operations]"
    n_lines = len(code.splitlines())
    console.print(f"[bold yellow]  [run_script][/bold yellow] {lang_key}  ({n_lines} lines)")

    # Feature 4: Pre-flight type check for Python scripts
    if lang_key == "python":
        try:
            global _lsp_client_mod
            if _lsp_client_mod is None:
                import lsp_client as _mod
                _lsp_client_mod = _mod
            result = _lsp_client_mod.lsp_preflight_check(code, "python")
            if not result["clean"]:
                console.print(f"[dim yellow]  Pre-flight: {result['errors']} error(s), {result['warnings']} warning(s) in script[/dim yellow]")
        except Exception:
            pass

    with _tmpmod.NamedTemporaryFile(
        suffix=ext, mode="w", delete=False, encoding="utf-8", prefix="qwen_script_"
    ) as tf:
        tf.write(code)
        tmp = tf.name
    interp_parts = interp.split()
    quoted = [f'"{p}"' if " " in p else p for p in interp_parts]
    cmd = " ".join(quoted) + f' "{tmp}"'
    try:
        return do_run_command(cmd, cwd=cwd, timeout=timeout, quiet=True)
    finally:
        Path(tmp).unlink(missing_ok=True)

def do_read_file(path: str, offset: int = 0, limit: int = 0) -> str:
    """Handle read file operation."""
    try:
        p = _resolve(path)
        if not p.exists():
            return f"[file not found: {p}]"
        raw = p.read_bytes()
        if b"\x00" in raw[:8192]:
            return f"[binary file not supported: {p.name}]"
        text  = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)

        if offset or limit:
            start      = max(0, offset - 1)
            end        = (start + limit) if limit else total
            snippet    = "\n".join(lines[start:end])
            header     = f"{p}  (lines {start+1}–{min(end, total)} of {total})"
            start_line = start + 1
        else:
            if len(raw) > 150_000:
                text  = text[:150_000] + "\n... [truncated at 150 KB]"
                lines = text.splitlines()
            snippet    = "\n".join(lines)
            header     = f"{p}  ({total} lines)"
            start_line = 1

        lang = LANG_MAP.get(p.suffix.lower(), "")
        console.print(f"[dim cyan]  {header}[/dim cyan]")
        if lang:
            try:
                console.print(Syntax(snippet, lang, line_numbers=True,
                                     start_line=start_line, theme="monokai", word_wrap=False))
            except Exception:
                console.print(snippet)
        else:
            for i, ln in enumerate(snippet.splitlines(), start=start_line):
                console.print(f"[dim]{i:>4}[/dim]  {ln}")

        return f"{header}\n\n{snippet}"
    except Exception as e:
        return f"[error: {e}]"

def do_patch_file(path: str, diff: str) -> str:
    """Handle patch file operation."""
    try:
        p = _resolve(path)

        # Feature 1: Pre-edit diagnostics + Feature 9: Conflict detection
        try:
            global _lsp_client_mod
            if _lsp_client_mod is None:
                import lsp_client as _mod
                _lsp_client_mod = _mod
            if _lsp_client_mod._is_code_file(str(p)):
                pre = _lsp_client_mod.lsp_pre_edit_check(str(p))
                if not pre["clean"]:
                    console.print(
                        f"[dim yellow]  Pre-edit: {pre['error_count']} error(s), "
                        f"{pre['warning_count']} warning(s) already in file[/dim yellow]"
                    )
                impact = _lsp_client_mod.lsp_check_patch_impact(str(p), diff)
                if impact["conflicts"]:
                    console.print(f"[dim red]  Conflict: {len(impact['conflicts'])} error(s) on lines about to be changed[/dim red]")
        except Exception:
            pass

        # Feature 5: Rename safety -- check renamed symbols have all refs updated
        try:
            if _lsp_client_mod is not None and _lsp_client_mod._is_code_file(str(p)):
                for dline in diff.split("\n"):
                    if dline.startswith("+") and not dline.startswith("+++"):
                        stripped = dline.lstrip("+").strip()
                        if any(kw in stripped for kw in ("class ", "def ", " = ", "import ")):
                            words = stripped.split()
                            for w in words:
                                if w.isidentifier() and len(w) > 2:
                                    refs = _lsp_client_mod.lsp_references(str(p), w)
                                    if refs.get("count", 0) > 1:
                                        console.print(
                                            f'[dim yellow]  Rename safety: "{w}" has {refs["count"]} references -- verify all updated[/dim yellow]'
                                        )
                                    break
        except Exception:
            pass

        # Feature 6+10: Post-edit trend tracking + import check
        try:
            if _lsp_client_mod is not None and _lsp_client_mod._is_code_file(str(p)):
                post = _lsp_client_mod.lsp_post_edit_check(str(p))
                if post["new_errors"] > 0:
                    console.print(f"[dim red]  Post-edit: {post['new_errors']} new error(s) introduced[/dim red]")
                if post["fixed_errors"] > 0:
                    console.print(f"[dim green]  Post-edit: {post['fixed_errors']} error(s) fixed[/dim green]")
                imports = _lsp_client_mod.lsp_check_imports(str(p))
                if imports["broken"]:
                    console.print(f"[dim red]  Imports: {len(imports['broken'])} broken import(s)[/dim red]")
        except Exception:
            pass

        if not p.exists():
            return f"[file not found: {p}]"
        original = p.read_text(encoding="utf-8", errors="replace")
        if str(p) not in _session_changes:
            _session_changes[str(p)] = original
        try:
            patched = _apply_diff(original, diff)
        except ValueError as e:
            return f"[patch failed: {e}]"

        if patched == original:
            return f"[no changes after applying patch: {p}]"

        # Show diff preview
        preview_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/{p.name}", tofile=f"b/{p.name}", lineterm="",
        ))
        preview = "".join(preview_lines[:60])
        if len(preview_lines) > 60:
            preview += f"\n... ({len(preview_lines)-60} more lines)"
        console.print(Syntax(preview, "diff", theme="monokai"))
        answer = console.input("[bold yellow]  Apply patch? [y/N]:[/bold yellow] ").strip().lower()
        if answer != "y":
            return "[patch cancelled by user]"

        # Backup and write
        global _backup_stack
        stamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = BACKUPS_DIR / f"{p.name}.{stamp}.bak"
        backup.write_text(original, encoding="utf-8")
        _backup_stack.append({"original": p, "backup": backup, "content": original})
        if len(_backup_stack) > _MAX_BACKUP_STACK:
            _backup_stack.pop(0)

        p.write_text(patched, encoding="utf-8")
        lines_changed = sum(
            1 for ln in preview_lines if ln.startswith(("+", "-")) and not ln.startswith(("---", "+++"))
        )
        console.print(f"[bold yellow]  [patch_file][/bold yellow] applied to {p}  ({lines_changed} lines changed)")
        return f"[patched: {p}  ({lines_changed} lines changed)]"
    except Exception as e:
        return f"[error: {e}]"

def _fmt_size(n: int) -> str:
    """Internal helper: fmt size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}"
        n //= 1024
    return f"{n} TB"


def do_list_directory(path: str, recursive: bool = False) -> str:
    """Handle list directory operation."""
    try:
        p = _resolve(path)
        if not p.exists():
            return f"[not found: {p}]"
        if not p.is_dir():
            return f"[not a directory: {p}]"
        if recursive:
            return f"Tree: {p}\n\n{build_project_tree(p, max_depth=4, max_files=300)}"
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        dirs  = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        nd, nf = len(dirs), len(files)
        lines = [f"{p}:  ({nd} dir{'s' if nd!=1 else ''}, {nf} file{'s' if nf!=1 else ''})\n"]
        for e in dirs:
            lines.append(f"  [dir ]  {e.name}/")
        for e in files:
            st    = e.stat()
            size  = _fmt_size(st.st_size)
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  [file]  {e.name:<45}  {size:>8}  {mtime}")
        return "\n".join(lines)
    except Exception as e:
        return f"[error: {e}]"


def do_find_files(path: str, pattern: str) -> str:
    """Handle find files operation."""
    try:
        import os as _os
        import fnmatch as _fnmatch
        p = _resolve(path)
        if not p.exists():
            return f"[not found: {p}]"
        _name_pat = pattern.rstrip("/").split("/")[-1] if pattern else "*"
        results = []
        for _dp_str, _dns, _fns in _os.walk(p):
            _dns[:] = sorted(d for d in _dns if d not in IGNORE_DIRS)
            _dp = Path(_dp_str)
            for _fn in sorted(_fns):
                if _fnmatch.fnmatch(_fn, _name_pat):
                    results.append(_dp / _fn)
        if not results:
            return f"No files matched '{pattern}' in {p}"
        lines = [f"Matches for '{pattern}' in {p}  ({len(results)} found):\n"]
        for r in results[:500]:
            rel = r.relative_to(p)
            if r.is_file():
                st    = r.stat()
                size  = _fmt_size(st.st_size)
                mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                lines.append(f"  {str(rel):<60}  {size:>8}  {mtime}")
            else:
                lines.append(f"  {rel}/")
        if len(results) > 500:
            lines.append(f"  ... and {len(results)-500} more")
        return "\n".join(lines)
    except Exception as e:
        return f"[error: {e}]"


def do_search_files(path: str, query: str, pattern: str = "**/*", context: int = 0) -> str:
    """Handle search files operation."""
    try:
        p = _resolve(path)
        if not p.exists():
            return f"[not found: {p}]"
        try:
            rx = re.compile(query, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(query), re.IGNORECASE)

        file_hits: dict[str, tuple[list[int], list[str]]] = {}
        total = 0

        # Use os.walk with in-place directory pruning so ignored dirs (e.g. .venv)
        # are never descended into, avoiding the 10,000+ file enumeration hang.
        import os as _os
        import fnmatch as _fnmatch
        _name_pat = pattern.rstrip("/").split("/")[-1] if pattern else "*"

        def _walk_files(root: Path):
            if root.is_file():
                yield root
                return
            for _dp_str, _dns, _fns in _os.walk(root):
                _dns[:] = sorted(d for d in _dns if d not in IGNORE_DIRS)
                _dp = Path(_dp_str)
                for _fn in sorted(_fns):
                    if _fnmatch.fnmatch(_fn, _name_pat):
                        yield _dp / _fn

        for fpath in _walk_files(p):
            if not fpath.is_file():
                continue
            try:
                if fpath.stat().st_size > 5_000_000:
                    continue
                raw = fpath.read_bytes()
                if b"\x00" in raw[:4096]:
                    continue
                file_lines = raw.decode("utf-8", errors="replace").splitlines()
                match_idx  = [i for i, ln in enumerate(file_lines) if rx.search(ln)]
                if not match_idx:
                    continue
                lines_out: list[str] = []
                if context:
                    match_set = set(match_idx)
                    ranges: list[tuple[int, int]] = []
                    for mi in match_idx:
                        lo, hi = max(0, mi - context), min(len(file_lines) - 1, mi + context)
                        if ranges and lo <= ranges[-1][1] + 1:
                            ranges[-1] = (ranges[-1][0], max(ranges[-1][1], hi))
                        else:
                            ranges.append((lo, hi))
                    for ri, (lo, hi) in enumerate(ranges):
                        if ri:
                            lines_out.append("  ···")
                        for ci in range(lo, hi + 1):
                            marker = ">" if ci in match_set else " "
                            lines_out.append(f"  {marker} L{ci+1:>4}: {file_lines[ci].rstrip()}")
                else:
                    for mi in match_idx:
                        lines_out.append(f"  L{mi+1:>4}: {file_lines[mi].rstrip()}")
                rel = str(fpath.relative_to(p))
                file_hits[rel] = (match_idx, lines_out)
                total += len(match_idx)
                if total >= 500:
                    break
            except Exception:
                continue

        if not file_hits:
            return f"No matches for '{query}' in {p}"
        out = [f"Matches for '{query}' in {p} — {total} match(es) in {len(file_hits)} file(s):\n"]
        for fname, (indices, hit_lines) in file_hits.items():
            n = len(indices)
            out.append(f"{fname}  ({n} match{'es' if n!=1 else ''}):")
            out.extend(hit_lines[:250])
            if len(hit_lines) > 250:
                out.append(f"  ... ({len(hit_lines)-250} more lines)")
            out.append("")
        if total >= 500:
            out.append("... (stopped at 500 — narrow your query or glob pattern)")
        return "\n".join(out)
    except Exception as e:
        return f"[error: {e}]"

def _cleanup_backups(keep: int = 50) -> None:
    """Keep only the most recent N backup files, deleting the rest."""
    if not BACKUPS_DIR.exists():
        return
    files = sorted(BACKUPS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in files[keep:]:
        try:
            old.unlink()
        except Exception:
            pass

def _backup_file(p: Path) -> None:
    """Internal helper: backup file."""
    global _backup_stack
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup  = BACKUPS_DIR / f"{p.name}.{stamp}.bak"
    content = p.read_text(encoding="utf-8", errors="replace")
    backup.write_text(content, encoding="utf-8")
    _backup_stack.append({"original": p, "backup": backup, "content": content})
    if len(_backup_stack) > _MAX_BACKUP_STACK:
        _backup_stack.pop(0)
    _cleanup_backups()

def do_write_file(path: str, content: str) -> str:
    """Handle write file operation."""
    try:
        p = _resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        existed = p.exists()
        if existed:
            old = p.read_text(encoding="utf-8", errors="replace")
            if str(p) not in _session_changes:
                _session_changes[str(p)] = old
            if old == content:
                return f"[no changes: {p}]"
            _backup_file(p)
            diff = list(difflib.unified_diff(
                old.splitlines(keepends=True),
                content.splitlines(keepends=True),
                fromfile=f"a/{p.name}", tofile=f"b/{p.name}", lineterm="",
            ))
            preview = "".join(diff[:80])
            if len(diff) > 80:
                preview += f"\n... ({len(diff)-80} more diff lines)"
            console.print(Syntax(preview, "diff", theme="monokai"))
            answer = console.input("[bold yellow]  Overwrite? [y/N]:[/bold yellow] ").strip().lower()
            if answer != "y":
                return "[write cancelled by user]"
        p.write_text(content, encoding="utf-8")
        action = "updated" if existed else "created"
        console.print(f"[bold yellow]  [write_file][/bold yellow] {action}: {p}")

        # Feature 6+10: Post-edit trend tracking + import check
        try:
            global _lsp_client_mod
            if _lsp_client_mod is None:
                import lsp_client as _mod
                _lsp_client_mod = _mod
            if _lsp_client_mod._is_code_file(str(p)):
                post = _lsp_client_mod.lsp_post_edit_check(str(p))
                if post["new_errors"] > 0:
                    console.print(f"[dim red]  Post-edit: {post['new_errors']} new error(s) introduced[/dim red]")
                if post["fixed_errors"] > 0:
                    console.print(f"[dim green]  Post-edit: {post['fixed_errors']} error(s) fixed[/dim green]")
                imports = _lsp_client_mod.lsp_check_imports(str(p))
                if imports["broken"]:
                    console.print(f"[dim red]  Imports: {len(imports['broken'])} broken import(s)[/dim red]")
        except Exception:
            pass

        return f"[{action}: {p}  ({len(content):,} chars)]"
    except Exception as e:
        return f"[error: {e}]"


def do_move_file(src: str, dst: str) -> str:
    """Handle move file operation."""
    try:
        s = _resolve(src)
        d = _resolve(dst)
        if not s.exists():
            return f"[not found: {s}]"
        if d.is_dir():
            d = d / s.name
        if d.exists():
            answer = console.input(
                f"[bold yellow]  {d} already exists. Overwrite? [y/N]:[/bold yellow] "
            ).strip().lower()
            if answer != "y":
                return "[move cancelled by user]"
            _backup_file(d)
        d.parent.mkdir(parents=True, exist_ok=True)
        s.replace(d)  # replace() overwrites dst atomically on Windows; rename() does not
        console.print(f"[bold yellow]  [move_file][/bold yellow] {s} → {d}")
        return f"[moved: {s} → {d}]"
    except Exception as e:
        return f"[error: {e}]"


def do_delete_file(path: str) -> str:
    """Handle delete file operation."""
    try:
        p = _resolve(path)
        if not p.exists():
            return f"[not found: {p}]"
        if p.is_dir():
            return f"[{p} is a directory — use run_command to delete directories]"
        console.print(
            f"[bold red]  [delete_file][/bold red] {p}  ({_fmt_size(p.stat().st_size)})"
        )
        answer = console.input("[bold yellow]  Permanently delete? [y/N]:[/bold yellow] ").strip().lower()
        if answer != "y":
            return "[delete cancelled by user]"
        _backup_file(p)
        p.unlink()
        return f"[deleted: {p}]"
    except Exception as e:
        return f"[error: {e}]"


# ---------------------------------------------------------------------------
# Model management / clipboard
# ---------------------------------------------------------------------------

def list_models(client: OpenAI) -> list[str]:
    """List Models"""
    try:
        return [m.id for m in client.models.list().data]
    except Exception as e:
        console.print(f"[red][error fetching models: {e}][/red]")
        return []

def cmd_copy(history: list) -> None:
    """Command: copy."""
    try:
        import pyperclip
        last = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), None)
        if last:
            pyperclip.copy(last)
            console.print("[green][copied last response to clipboard][/green]")
        else:
            console.print("[yellow][no response to copy][/yellow]")
    except ImportError:
        console.print("[yellow][pip install pyperclip to enable clipboard support][/yellow]")
    except Exception as e:
        console.print(f"[red][error: {e}][/red]")

def cmd_paste(history: list) -> None:
    """Command: paste."""
    try:
        import pyperclip
        content = pyperclip.paste()
        if content:
            history.append({"role": "user", "content": f"[Clipboard content]\n{content}"})
            console.print(f"[green][pasted {len(content):,} chars from clipboard][/green]")
        else:
            console.print("[yellow][clipboard is empty][/yellow]")
    except ImportError:
        console.print("[yellow][pip install pyperclip to enable clipboard support][/yellow]")
    except Exception as e:
        console.print(f"[red][error: {e}][/red]")

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

def _make_pt_session() -> None:
    """Set up prompt_toolkit session for enhanced input (if available)."""
    global _pt_session
    if not _HAS_PT:
        return

    class _Completer(_PtCompleter):
        def get_completions(self, document, complete_event):
            """Get Completions"""
            text = document.text_before_cursor
            word = (text.split()[-1] if text.split() else text)
            # Command completion
            if text.lstrip().startswith("/"):
                for cmd in _COMMANDS:
                    if cmd.startswith(word):
                        yield _PtCompletion(cmd[len(word):], start_position=0, display=cmd)
                return
            # @file path completion
            at_pos = text.rfind("@")
            if at_pos >= 0:
                raw = text[at_pos + 1:]
                if "::" not in raw:
                    p    = Path(raw).expanduser()
                    base = p if (raw.endswith("/") or raw.endswith("\\")) else p.parent
                    stem = p.name if not (raw.endswith("/") or raw.endswith("\\")) else ""
                    try:
                        for child in sorted(base.iterdir()):
                            name = child.name + ("/" if child.is_dir() else "")
                            if child.name.lower().startswith(stem.lower()):
                                yield _PtCompletion(name, start_position=-len(stem))
                    except Exception:
                        pass

    _pt_session = _PtSession(
        history=_PtFileHistory(str(PT_HISTORY_FILE)),
        auto_suggest=_PtAutoSuggest(),
        completer=_Completer(),
        complete_while_typing=False,
        enable_history_search=True,
    )


def _close_loitering_event_loop():
    """Close a running event loop left behind by async HTTP clients (Python 3.14 issue)."""
    try:
        loop = asyncio.get_running_loop()
        try:
            loop.close()
        except RuntimeError:
            pass  # Loop is still running, can't close from same thread
    except RuntimeError:
        pass  # No running loop
    # Reset the event loop policy to clear any lingering state (suppress deprecation warnings)
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception:
        pass


def _read_input_inline() -> str:
    """Read input inline - assumes no running event loop."""
    if _pt_session is not None:
        line = _pt_session.prompt("\nYou: ").strip()
    else:
        line = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
    if not line.endswith("\\"):
        return line
    lines = [line[:-1]]
    while True:
        if _pt_session is not None:
            cont = _pt_session.prompt("... ").strip()
        else:
            cont = console.input("[bold cyan]...[/bold cyan] ").strip()
        if cont.endswith("\\"):
            lines.append(cont[:-1])
        else:
            lines.append(cont)
            break
    return "\n".join(lines)


def _read_input_in_thread() -> str:
    """Read input in a sub-thread to avoid asyncio event loop conflicts."""
    result = []
    error = []
    def _prompt_thread():
        """Internal helper: prompt thread."""
        try:
            result.append(_read_input_inline())
        except Exception as e:
            error.append(str(e))
    t = threading.Thread(target=_prompt_thread, daemon=True)
    t.start()
    # No fixed timeout — poll every 0.1s so KeyboardInterrupt stays responsive.
    # The daemon thread will be killed on process exit; we never block forever.
    try:
        while t.is_alive():
            t.join(timeout=0.1)
    except KeyboardInterrupt:
        raise
    if error:
        console.print(f"[yellow]  [warning] input error: {error[0]}[/yellow]")
        return ""
    return result[0] if result else ""


def read_input() -> str:
    """Read user input, handling asyncio event loop conflicts on Python 3.14+."""
    _close_loitering_event_loop()
    # If a loop is still running, prompt_toolkit's asyncio.run() will fail.
    # Fall back to a sub-thread which has its own clean event loop.
    try:
        asyncio.get_running_loop()
        return _read_input_in_thread()
    except RuntimeError:
        return _read_input_inline()

# ---------------------------------------------------------------------------
# LLM streaming — with auto-reconnect and real token counts
# ---------------------------------------------------------------------------

_LIVE_PREVIEW_LINES  = 18
_stream_usage_supported = True   # flipped to False if server rejects stream_options

def _live_updater(live: Live, max_lines: int = _LIVE_PREVIEW_LINES):
    """Internal helper: live updater."""
    def update(text: str):
        """Update"""
        clean = _THINK_RE.sub("", text).strip()
        if not clean:
            live.update(Text("⟳ thinking…", style="dim italic"))
            return
        split = clean.split("\n")
        preview = ("…\n" + "\n".join(split[-max_lines:])) if len(split) > max_lines else clean
        live.update(Markdown(preview))
    return update

def _create_with_retry(client: OpenAI, **kwargs):
    """Create a chat completion with exponential back-off on connection errors."""
    global _stream_usage_supported

    # Try without stream_options if previously unsupported
    if not _stream_usage_supported:
        kwargs.pop("stream_options", None)

    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            err = str(e).lower()
            is_conn = any(s in err for s in ("connection", "timeout", "refused", "unreachable", "reset by peer", "eof"))
            is_param = any(s in err for s in ("unknown field", "extra inputs", "unrecognized", "invalid param", "stream_options", "extra_body"))
            if is_param:
                stripped = False
                if "stream_options" in kwargs:
                    _stream_usage_supported = False
                    kwargs.pop("stream_options")
                    stripped = True
                if "extra_body" in kwargs:
                    kwargs.pop("extra_body")
                    stripped = True
                if stripped:
                    continue
            if is_conn and attempt < 3:
                wait = 2 ** attempt
                console.print(f"[yellow]  [connection error — retry {attempt+1}/3 in {wait}s: {e}][/yellow]")
                time.sleep(wait)
            else:
                raise

def stream_once(client: OpenAI, messages: list, use_tools: bool,
                update_fn=None) -> tuple[str, list, dict]:
    """Returns (text, tool_calls, usage_dict)."""
    global _stream_usage_supported
    content_parts: list[str] = []
    tc_buf: dict[int, dict]  = {}
    usage: dict = {}
    finish_reason: str | None = None

    # Build kwargs from active preset, then layer manual _model_params on top
    preset = SAMPLING_PRESETS.get(_active_preset, SAMPLING_PRESETS["thinking"])
    kwargs: dict = dict(
        model=MODEL, messages=messages, stream=True,
        temperature=preset["temperature"],
        top_p=preset["top_p"],
        presence_penalty=preset["presence_penalty"],
        max_tokens=preset["max_tokens"],
        extra_body=dict(preset["extra_body"]),   # copy so retries can pop it safely
    )
    # Manual overrides win over preset
    for k in ("temperature", "top_p", "max_tokens", "presence_penalty"):
        if k in _model_params:
            kwargs[k] = _model_params[k]
    if _stream_usage_supported:
        kwargs["stream_options"] = {"include_usage": True}
    if use_tools:
        kwargs["tools"]       = TOOLS
        kwargs["tool_choice"] = "auto"

    stream = _create_with_retry(client, **kwargs)
    interrupted = False
    try:
        for chunk in stream:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = {
                    "prompt":     getattr(chunk.usage, "prompt_tokens",     0) or 0,
                    "completion": getattr(chunk.usage, "completion_tokens", 0) or 0,
                }
            if not chunk.choices:
                continue
            fr = getattr(chunk.choices[0], "finish_reason", None)
            if fr:
                finish_reason = fr
            delta = chunk.choices[0].delta
            if delta.content:
                content_parts.append(delta.content)
                if update_fn is not None:
                    update_fn("".join(content_parts))
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tc_buf:
                        tc_buf[idx] = {"id": "", "type": "function",
                                       "function": {"name": "", "arguments": ""}}
                    if tc.id:
                        tc_buf[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tc_buf[idx]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            tc_buf[idx]["function"]["arguments"] += tc.function.arguments
    except Exception as e:
        # Connection dropped mid-stream (e.g. the model server restarted). Salvage
        # whatever text we have rather than losing the whole turn; drop any partial
        # tool call since its JSON arguments would be truncated and unusable.
        if not content_parts:
            raise
        interrupted = True
        console.print(f"\n[yellow]  [stream interrupted — keeping partial reply ({type(e).__name__})][/yellow]")

    full_text  = "".join(content_parts)
    api_calls  = [] if interrupted else list(tc_buf.values())

    # Record how the generation ended so callers can tell a *complete* answer
    # apart from one the model was cut off in the middle of. "length" means the
    # token cap stopped it; an interrupted stream is likewise incomplete.
    usage["finish_reason"] = "interrupted" if interrupted else finish_reason
    usage["truncated"]     = interrupted or finish_reason == "length"

    # Qwen3 fallback: model emitted <tool_call> XML in text instead of API tool_calls
    if not api_calls and use_tools and "<tool_call>" in full_text.lower():
        clean_text, xml_calls = _parse_xml_tool_calls(full_text)
        if xml_calls:
            return clean_text, xml_calls, usage

    return full_text, api_calls, usage



def _short_args(name: str, args: dict) -> str:
    """Internal helper: short args."""
    if name in ("web_search", "search_news"):
        return repr(args.get("query", "")[:40])
    if name == "fetch_url":
        return repr(args.get("url", "")[:50])
    if name in ("read_file", "write_file", "patch_file"):
        return repr(args.get("path", "")[:40])
    if name == "run_command":
        return repr(args.get("command", "")[:40])
    if name in ("list_directory", "find_files", "search_files"):
        return repr(args.get("path", "")[:30])
    return ""


def _auto_extract_memory(client: OpenAI, user_msg: str, assistant_msg: str) -> None:
    """Background: pull memorable facts from this exchange and append to memory.md."""
    global _auto_memory_count
    with _main_llm_busy_lock:
        if _main_llm_busy:
            return
        return
    if not _BG_LLM_SEM.acquire(blocking=False):
        return
    try:
        excerpt = f"User: {user_msg[:600]}\n\nAssistant: {assistant_msg[:1000]}"
        prompt = [
            {"role": "system", "content": (
                "Extract facts worth remembering long-term from this exchange: "
                "concrete preferences, decisions, names, configs, or constraints. "
                "Skip pleasantries and transient state. "
                "Format as short bullet points. "
                "If nothing is notable, reply with exactly: NONE"
            )},
            {"role": "user", "content": excerpt},
        ]
        resp = client.chat.completions.create(
            model=MODEL, messages=prompt, stream=False, max_tokens=200,
            timeout=AUX_LLM_TIMEOUT,
        )
        facts = (resp.choices[0].message.content or "").strip()
        if facts and facts.upper() != "NONE" and len(facts) > 10:
            with _memory_lock:
                mem = load_memory()
                entry = f"\n\n<!-- auto {datetime.now().strftime('%Y-%m-%d')} -->\n{facts}"
                save_memory((mem + entry).strip())
                _auto_memory_count += 1
                if _auto_memory_count % MEMORY_CURATE_INTERVAL == 0:
                    _curate_memory(client, locked=True)
    except Exception:
        pass
    finally:
        _BG_LLM_SEM.release()


def _curate_memory(client: OpenAI, locked: bool = False) -> None:
    """Background: deduplicate and consolidate memory.md when it grows noisy."""
    if not locked:
        lock_ctx = _memory_lock
    else:
        import contextlib
        lock_ctx = contextlib.nullcontext()
    with lock_ctx:
        mem = load_memory()
        if not mem or len(mem) < 400:
            return
        try:
            prompt = [
                {"role": "system", "content": "Consolidate this memory into clean, non-redundant bullet points. Remove duplicates. Preserve all unique facts. Be concise."},
                {"role": "user",   "content": mem},
            ]
            resp = client.chat.completions.create(
                model=MODEL, messages=prompt, stream=False, max_tokens=600,
                timeout=AUX_LLM_TIMEOUT,
            )
            consolidated = (resp.choices[0].message.content or "").strip()
            if consolidated and len(consolidated) > 20:
                save_memory(consolidated)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Background intelligence crawlers — web browsing + memory training
# ---------------------------------------------------------------------------

def _intel_load_topics() -> list[dict]:
    """Internal helper: intel load topics."""
    if INTEL_TOPICS.exists():
        try:
            return json.loads(INTEL_TOPICS.read_text(encoding="utf-8"))
        except Exception:
            pass
    return [dict(t) for t in _INTEL_DEFAULT_TOPICS]


def _intel_save_topics(topics: list[dict]) -> None:
    """Internal helper: intel save topics."""
    INTEL_DIR.mkdir(exist_ok=True)
    INTEL_TOPICS.write_text(json.dumps(topics, indent=2, ensure_ascii=False), encoding="utf-8")


def _intel_enqueue(topic_name: str, query: str, raw: str) -> None:
    """Internal helper: intel enqueue."""
    with _intel_lock:
        INTEL_DIR.mkdir(exist_ok=True)
        try:
            items = json.loads(INTEL_QUEUE.read_text(encoding="utf-8")) if INTEL_QUEUE.exists() else []
        except Exception:
            items = []
        items.append({"topic": topic_name, "query": query, "raw": raw[:3000], "ts": time.time()})
        INTEL_QUEUE.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")


def _intel_dequeue_all() -> list[dict]:
    """Internal helper: intel dequeue all."""
    with _intel_lock:
        if not INTEL_QUEUE.exists():
            return []
        try:
            items = json.loads(INTEL_QUEUE.read_text(encoding="utf-8"))
            INTEL_QUEUE.write_text("[]", encoding="utf-8")
            return items
        except Exception:
            return []


def _intel_load_feed() -> str:
    """Internal helper: intel load feed."""
    if INTEL_FEED.exists():
        try:
            return INTEL_FEED.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return ""


def _intel_prepend_entry(topic_name: str, summary: str) -> None:
    """Internal helper: intel prepend entry."""
    ts    = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"<!-- {ts} | {topic_name} -->\n{summary.strip()}"
    with _intel_lock:
        existing = _intel_load_feed()
        chunks   = [c.strip() for c in existing.split("\n\n") if c.strip()] if existing else []
        chunks.insert(0, entry)
        chunks = chunks[:40]           # cap at 40 entries
        INTEL_FEED.write_text("\n\n".join(chunks), encoding="utf-8")


def intel_get_recent(n: int = _INTEL_INJECT_N) -> str:
    """Return the N most recent intel entries for injection into system prompt."""
    feed = _intel_load_feed()
    if not feed:
        return ""
    chunks = [c.strip() for c in feed.split("\n\n") if c.strip()]
    return "\n\n".join(chunks[:n])


def _intel_crawl_once() -> None:
    """Pick the least-recently-crawled topic, do a web search, enqueue the raw result."""
    if not _intel_enabled.is_set():
        return
    try:
        topics = _intel_load_topics()
        if not topics:
            return
        topic = min(topics, key=lambda t: t.get("last_checked", 0))
        raw = do_web_search(topic["query"], max_results=5)
        if raw and "error" not in raw.lower()[:40]:
            _intel_enqueue(topic["name"], topic["query"], raw)
        for t in topics:
            if t["name"] == topic["name"]:
                t["last_checked"] = time.time()
                break
        _intel_save_topics(topics)
    except Exception:
        pass


def _intel_process_queue(client: OpenAI) -> None:
    """Post-turn: LLM-summarize queued raw results, update feed, train memory."""
    items = _intel_dequeue_all()
    if not items:
        return
    with _main_llm_busy_lock:
        if _main_llm_busy:
            return
    if not _BG_LLM_SEM.acquire(blocking=False):
        return
    try:
        for item in items:
            try:
                prompt = [
                    {"role": "system", "content": (
                        "Summarize these web search results into 3-5 concise bullet points "
                        "(max 90 chars each). Focus on concrete facts, releases, and updates. "
                        "No preamble, just the bullets."
                    )},
                    {"role": "user", "content":
                        f"Topic: {item['topic']}\n\n{item['raw'][:3000]}"},
                ]
                resp = client.chat.completions.create(
                    model=MODEL, messages=prompt, stream=False,
                    max_tokens=250, timeout=AUX_LLM_TIMEOUT,
                )
                summary = (resp.choices[0].message.content or "").strip()
                if not summary or len(summary) < 20:
                    continue
                _intel_prepend_entry(item["topic"], summary)
                # Training: persist durable facts (versions, releases, alerts) to memory
                _intel_train_memory(client, item["topic"], summary)
            except Exception:
                pass
    finally:
        _BG_LLM_SEM.release()


def _intel_train_memory(client: OpenAI, topic_name: str, summary: str) -> None:
    """If the intel summary contains durable facts, add them to persistent memory."""
    today = datetime.now().strftime('%Y-%m-%d')
    if _intel_memory_written.get(topic_name) == today:
        return  # already wrote facts for this topic today
    try:
        prompt = [
            {"role": "system", "content": (
                "Decide if any of these facts should be saved to a persistent memory file "
                "(version numbers, critical releases, security alerts, key API changes). "
                "If yes, output 1-2 short bullet lines starting with '- '. "
                "If nothing is worth persisting, reply exactly: SKIP"
            )},
            {"role": "user", "content": f"Topic: {topic_name}\n{summary}"},
        ]
        resp = client.chat.completions.create(
            model=MODEL, messages=prompt, stream=False,
            max_tokens=120, timeout=AUX_LLM_TIMEOUT,
        )
        facts = (resp.choices[0].message.content or "").strip()
        if facts and facts.upper() != "SKIP" and facts.startswith("-"):
            with _memory_lock:
                mem = load_memory()
                tag = f"\n\n<!-- intel {today} -->\n{facts}"
                save_memory((mem + tag).strip())
            _intel_memory_written[topic_name] = today
    except Exception:
        pass


def _intel_extract_topics(client: OpenAI, user_msg: str, reply: str) -> None:
    """Post-turn: extract new search-worthy topics from this exchange and track them."""
    with _main_llm_busy_lock:
        if _main_llm_busy:
            return
    if not _BG_LLM_SEM.acquire(blocking=False):
        return
    try:
        prompt = [
            {"role": "system", "content": (
                "Extract up to 2 web-searchable topics from this conversation worth monitoring "
                "(new technologies, frameworks, tools, or domains the user cares about). "
                "Reply one per line as: NAME|search query\n"
                "Example: FastAPI 1.0|FastAPI 1.0 release features changelog\n"
                "If nothing new to track, reply: NONE"
            )},
            {"role": "user", "content":
                f"User: {user_msg[:400]}\nAssistant: {reply[:400]}"},
        ]
        resp = client.chat.completions.create(
            model=MODEL, messages=prompt, stream=False,
            max_tokens=80, timeout=AUX_LLM_TIMEOUT,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text or text.upper() == "NONE":
            return
        topics = _intel_load_topics()
        existing = {t["name"].lower() for t in topics}
        for line in text.splitlines():
            if "|" not in line:
                continue
            name, query = line.split("|", 1)
            name, query = name.strip(), query.strip()
            if name and query and name.lower() not in existing:
                topics.append({"name": name, "query": query, "last_checked": 0})
                existing.add(name.lower())
        if len(topics) > 25:
            topics = sorted(topics, key=lambda t: t.get("last_checked", 0), reverse=True)[:25]
        _intel_save_topics(topics)
    except Exception:
        pass
    finally:
        _BG_LLM_SEM.release()


def _intel_crawler_thread(delay_s: int) -> None:
    """Background daemon: crawl one topic every _INTEL_INTERVAL seconds."""
    INTEL_DIR.mkdir(exist_ok=True)
    _intel_stop.wait(timeout=delay_s)  # stagger startup
    while not _intel_stop.is_set():
        _intel_crawl_once()
        _intel_stop.wait(timeout=_INTEL_INTERVAL)


def start_intel_crawlers() -> None:
    """Start _INTEL_CRAWLERS background crawler threads, staggered."""
    if not INTEL_TOPICS.exists():
        _intel_save_topics([dict(t) for t in _INTEL_DEFAULT_TOPICS])
    stagger = max(15, _INTEL_INTERVAL // _INTEL_CRAWLERS)
    for i in range(_INTEL_CRAWLERS):
        t = threading.Thread(
            target=_intel_crawler_thread,
            args=(15 + i * stagger,),
            daemon=True,
            name=f"intel-crawler-{i}",
        )
        t.start()


def _watch_worker(mtimes: dict[str, float]) -> None:
    """Poll watched files every 2s and queue changed paths into _watch_pending."""
    while not _watch_stop.is_set():
        for path in list(mtimes.keys()):
            try:
                mtime = Path(path).stat().st_mtime
                old   = mtimes.get(path)
                if old is not None and mtime != old:
                    _watch_pending.append(path)
                mtimes[path] = mtime
            except Exception:
                pass
        _watch_stop.wait(timeout=2)


_NOT_PLAN_RE = re.compile(
    r"\b(question|clarif|let\s+me\s+know|could\s+you|can\s+you|please\s+(share|tell)|"
    r"what\s+(kind|type|should|do|are|is)|any\s+(preference|design|tech|stack)|"
    r"a\s+(few|couple\s*of)\s+(quick\s+)?question)\b",
    re.IGNORECASE,
)

_IMPERATIVE_RE = re.compile(
    r"^\s*\d+[\.\)]\s+(?:install|run|execute|edit|write|create|update|delete|"
    r"rename|move|open|close|copy|download|upload|configure|set(?:\s+up)?|add|"
    r"remove|check|verify|test|build|deploy|start|stop|restart|enable|disable|"
    r"import|export|fix|patch|modify|change|replace|generate|launch|init(?:ialize)?|"
    r"clone|push|pull|commit|merge|revert|reset|refactor|implement|define|"
    r"register|apply|read|parse|format|convert|send|connect|save|load|clear|"
    r"refresh|reload|navigate|click|select|enter|type|press|switch|toggle)\b",
    re.MULTILINE | re.IGNORECASE,
)

def _looks_like_plan(text: str) -> bool:
    """True if the response has 3+ numbered imperative action steps."""
    steps = _PLAN_RE.findall(text)
    if len(steps) < 3:
        return False
    # Numbered items that end with '?' are a Q&A list, not a plan
    question_items = re.findall(r"^\s*\d+[\.\)]\s+[^\n]*\?", text, re.MULTILINE)
    if len(question_items) >= 2:
        return False
    # Context words that indicate clarification rather than a plan
    numbered_text = "\n".join(steps)
    if _NOT_PLAN_RE.search(numbered_text):
        return False
    # Require a majority of steps to start with an imperative action verb
    imperative_count = len(_IMPERATIVE_RE.findall(text))
    if imperative_count < max(2, len(steps) // 2):
        return False
    return True


def _confidence_warning(text: str) -> None:
    """Print a soft warning if the response contains many hedging phrases."""
    count = len(_HEDGE_RE.findall(text))
    if count >= 3:
        console.print(
            f"[dim yellow]  [low confidence — {count} hedge phrases — consider /retry with more context][/dim yellow]"
        )


def cmd_agent(goal: str, history: list, base_system: str, client: OpenAI, max_iter: int = 20) -> None:
    """Autonomous agent loop: model iterates with tools until it emits AGENT_DONE."""
    agent_suffix = (
        "\n\nYou are running in autonomous agent mode. Work toward the goal using tools. "
        "After each action, briefly verify the result is correct before moving on — "
        "do not assume success; check the output or re-read the file. "
        "DELEGATION: if the goal has 3+ independent subtasks, or any subtask would take 5+ tool calls, "
        "spawn subagents via team_spawn_agent instead of doing everything yourself. "
        "Subagents have full access to all the same tools. You coordinate, they execute in parallel. "
        "When the goal is fully achieved and you have confirmed the result, end your response with exactly: AGENT_DONE"
    )
    console.print(f"[bold cyan]  Agent goal:[/bold cyan] {goal}")
    console.print(f"[dim]  (max {max_iter} iterations — Ctrl+C to stop)[/dim]")

    working = list(history)
    working.append({"role": "user", "content": f"[Agent task] {goal}"})

    for iteration in range(1, max_iter + 1):
        console.print(Rule(f"[dim]Agent {iteration}/{max_iter}[/dim]", style="dim"))
        # Keep the agent going across many iterations — summarize+preserve the task
        # if the accumulated working set is nearing the context limit.
        working = _maybe_autocompact(working, base_system, client)
        msgs = [{"role": "system", "content": build_system_prompt(base_system) + agent_suffix}] + working
        try:
            reply = run_turn(client, msgs, allow_tools=True)
        except KeyboardInterrupt:
            console.print("\n[dim]  \\[agent stopped][/dim]")
            break

        if not reply:
            console.print("[red]  \\[agent error or cancelled — stopping][/red]")
            break

        working.append({"role": "assistant", "content": reply})
        history.append({"role": "user",      "content": f"[Agent task] {goal}" if iteration == 1 else "[continue]"})
        history.append({"role": "assistant",  "content": reply})

        if re.search(r'\bAGENT_DONE\b', reply):
            console.print(f"[green]  \\[agent done in {iteration} iteration{'s' if iteration != 1 else ''}][/green]")
            break

        working.append({"role": "user", "content": "Continue working toward the goal."})
    else:
        console.print(f"[yellow]  \\[agent reached max iterations ({max_iter}) — goal may be unfinished][/yellow]")


def cmd_git_commit(client: OpenAI, msg: str = "") -> None:
    """Generate a commit message from staged diff and commit."""
    diff = subprocess.run(
        "git diff --staged", shell=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()
    if not diff:
        console.print("[yellow][no staged changes — git add first][/yellow]")
        return

    if not msg:
        try:
            prompt = [
                {"role": "system", "content": "Write a concise conventional commit message (type: subject). Reply with only the message, no quotes."},
                {"role": "user",   "content": f"```diff\n{diff[:6000]}\n```"},
            ]
            resp = client.chat.completions.create(model=MODEL, messages=prompt, stream=False, max_tokens=120, timeout=AUX_LLM_TIMEOUT)
            msg  = (resp.choices[0].message.content or "").strip().strip('"\'')
        except Exception as e:
            console.print(f"[red][error generating message: {e}][/red]")
            return

    console.print(f"\n[bold]  Commit message:[/bold] {msg}\n")
    try:
        answer = console.input("[bold yellow]  Commit? [y/N]: [/bold yellow]").strip().lower()
        if answer == "y":
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False,
                                            encoding="utf-8") as tf:
                tf.write(msg)
                tf_path = tf.name
            try:
                do_run_command(f'git commit -F "{tf_path}"')
                _invalidate_git_cache()   # state changed — drop stale git context
            finally:
                Path(tf_path).unlink(missing_ok=True)
    except (KeyboardInterrupt, EOFError):
        pass


def cmd_git_pr(client: OpenAI) -> None:
    """Generate a PR description from commits ahead of main/origin/main."""
    for base in ("main", "origin/main", "master", "origin/master"):
        log = subprocess.run(
            f"git log {base}..HEAD --oneline", shell=True,
            capture_output=True, text=True, encoding="utf-8",
        ).stdout.strip()
        if log:
            stat = subprocess.run(
                f"git diff {base}...HEAD --stat", shell=True,
                capture_output=True, text=True, encoding="utf-8",
            ).stdout.strip()
            break
    else:
        console.print("[yellow][no commits ahead of main/master][/yellow]")
        return

    try:
        prompt = [
            {"role": "system", "content": "Write a GitHub PR description with ## Summary (3-5 bullets) and ## Changes. Be concise and specific."},
            {"role": "user",   "content": f"Commits:\n{log}\n\nFile changes:\n{stat}"},
        ]
        resp = client.chat.completions.create(model=MODEL, messages=prompt, stream=False, max_tokens=400, timeout=AUX_LLM_TIMEOUT)
        pr   = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        console.print(f"[red][error: {e}][/red]")
        return

    console.print(Markdown(pr))
    try:
        if console.input("[dim]  Copy to clipboard? [y/N]: [/dim]").strip().lower() == "y":
            try:
                import pyperclip
                pyperclip.copy(pr)
                console.print("[green][copied][/green]")
            except ImportError:
                console.print("[yellow][pip install pyperclip][/yellow]")
    except (KeyboardInterrupt, EOFError):
        pass


def cmd_task(goal: str, history: list, base_system: str, client: OpenAI) -> None:
    """Plan-approve-execute-test agentic task loop."""
    root     = Path.cwd()
    test_cmd = _detect_test_command(root)

    console.print(f"[bold cyan]  Task:[/bold cyan] {goal}")
    if test_cmd:
        console.print(f"[dim]  Tests: {test_cmd}[/dim]")

    # Generate plan (no tools, fast)
    console.print("[dim]  Planning...[/dim]")
    plan_msgs = [
        {"role": "system", "content": build_system_prompt(base_system)},
        *history,
        {"role": "user", "content": (
            f"Create a numbered step-by-step plan to accomplish: {goal}\n\n"
            "List only the steps. Be specific about which files to read or modify."
        )},
    ]
    try:
        plan_resp = client.chat.completions.create(
            model=MODEL, messages=plan_msgs, stream=False, max_tokens=400,
            timeout=AUX_LLM_TIMEOUT,
        )
        plan_text = (plan_resp.choices[0].message.content or "").strip()
    except Exception as e:
        console.print(f"[red][error generating plan: {e}][/red]")
        return

    console.print(Markdown(plan_text))

    # Approve plan
    try:
        approval = console.input("\n[bold yellow]  Execute? [y/edit/N]: [/bold yellow]").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return
    if approval in ("n", ""):
        console.print("[dim][task cancelled][/dim]")
        return
    if approval == "edit":
        try:
            console.print("[dim]  Enter revised plan (blank line to finish):[/dim]")
            lines: list[str] = []
            while True:
                ln = console.input("  ").rstrip()
                if not ln:
                    break
                lines.append(ln)
            if lines:
                plan_text = "\n".join(lines)
        except (KeyboardInterrupt, EOFError):
            return

    # Parse numbered steps
    step_lines = [ln.strip() for ln in plan_text.splitlines()
                  if re.match(r"^\d+[\.\)]\s+\S", ln.strip())]
    if not step_lines:
        step_lines = [plan_text]

    total = len(step_lines)
    working = list(history)
    working.append({"role": "user", "content": f"[Task] {goal}\n\nPlan:\n{plan_text}"})
    working.append({"role": "assistant", "content": "Understood. I will execute each step now."})

    for i, step in enumerate(step_lines, 1):
        label = step[:70] + ("..." if len(step) > 70 else "")
        console.print(Rule(f"[dim]Step {i}/{total}: {label}[/dim]", style="dim"))

        # Keep multi-step tasks within the context window — summarize+preserve the
        # task so far if the accumulated working set is nearing the limit.
        working = _maybe_autocompact(working, base_system, client)

        retries = 0
        extra_ctx = ""
        while retries <= 2:
            step_msgs = [{"role": "system", "content": build_system_prompt(base_system)}] + working
            step_msgs.append({"role": "user", "content":
                f"Execute step {i}/{total}: {step}{extra_ctx}"})
            try:
                reply = run_turn(client, step_msgs, allow_tools=True)
            except KeyboardInterrupt:
                console.print("\n[dim]  \\[task stopped][/dim]")
                return
            if not reply:
                console.print(f"[red]  \\[step {i} failed — stopping][/red]")
                return

            working.append({"role": "user",      "content": f"Step {i}: {step}"})
            working.append({"role": "assistant",  "content": reply})
            history.append({"role": "user",      "content": f"[Task {i}/{total}] {step}"})
            history.append({"role": "assistant",  "content": reply})

            # Run tests if code was written/patched
            modified = any(kw in reply for kw in ("[patched:", "[created:", "[updated:"))
            if test_cmd and modified:
                console.print(f"[dim]  Running: {test_cmd}[/dim]")
                test_out = do_run_command(test_cmd, timeout=60)
                # Check for actual test failures — require non-zero counts to avoid
                # "0 failed" or "0 errors" in a passing summary line being a false positive
                has_fail = re.search(r'\b(FAILED|ERROR)\b', test_out)
                has_count = re.search(r'\b[1-9]\d*\s+(?:failed|error)\b', test_out, re.IGNORECASE)
                has_fail_test = re.search(r'\b(FAIL|ERROR)\b.*\b(test|spec|suite)\b', test_out)
                passed = not (has_fail or has_count or has_fail_test)
                if not passed and retries < 2:
                    console.print(f"[yellow]  Tests failed — retrying step {i} (attempt {retries+2}/3)[/yellow]")
                    extra_ctx = (f"\n\nPrevious attempt failed tests:\n{test_out[:1500]}"
                                 f"\nFix the issue then complete step {i}.")
                    retries += 1
                    continue
                elif not passed:
                    console.print("[yellow]  Tests still failing — moving on[/yellow]")
            break

        if i < total:
            try:
                fb = console.input(
                    f"[dim]  [Step {i}/{total} done — Enter=next  f=feedback  s=stop]: [/dim]"
                ).strip().lower()
            except (KeyboardInterrupt, EOFError):
                fb = "s"
            if fb == "s":
                console.print("[dim][task stopped by user][/dim]")
                return
            elif fb.startswith("f"):
                try:
                    feedback = console.input("  Feedback: ").strip()
                except (KeyboardInterrupt, EOFError):
                    feedback = ""
                if feedback:
                    note = f"[User feedback after step {i}]: {feedback}"
                    working.append({"role": "user", "content": note})
                    history.append({"role": "user", "content": note})

    console.print("[bold green]  Task complete.[/bold green]")


_title_lock = threading.Lock()  # prevents duplicate title generation

def _generate_session_title(client: OpenAI, history: list) -> None:
    """Background: generate a short session title after the first exchange."""
    global _session_title
    with _main_llm_busy_lock:
        if _main_llm_busy:
            return
    with _title_lock:
        if _session_title:
            return
        _session_title = "..."  # sentinel: prevent duplicate LLM calls
    try:
        excerpt = "\n".join(
            f"{m['role'].upper()}: {(m.get('content') or '')[:300]}"
            for m in history[:4]
            if m.get("role") in ("user", "assistant")
        )
        prompt = [
            {"role": "system", "content": "Generate a short 3-5 word title for this conversation. Reply with only the title."},
            {"role": "user",   "content": excerpt},
        ]
        resp = client.chat.completions.create(
            model=MODEL, messages=prompt, stream=False, max_tokens=15,
            timeout=AUX_LLM_TIMEOUT,
        )
        title = (resp.choices[0].message.content or "").strip().strip('"\'').rstrip(".")
        if title and 3 <= len(title) <= 60:
            with _title_lock:
                if _session_title == "...":
                    _session_title = title
            console.print(f"[dim]  [{title}][/dim]")
    except Exception:
        pass


def _extract_runnable_code(text: str) -> tuple[str, str] | None:
    """Return (lang, code) for the first executable code block in text, or None."""
    for m in _CODE_BLOCK_RE.finditer(text):
        lang = (m.group(1) or "").lower().strip()
        code = m.group(2).strip()
        if lang in _RUNNABLE_LANGS and code:
            return lang, code
    return None


def _extract_first_code_block(text: str) -> tuple[str, str] | None:
    """Return (lang, code) for the first fenced code block of any language, or None."""
    m = _CODE_BLOCK_RE.search(text)
    if not m:
        return None
    lang = (m.group(1) or "").lower().strip()
    code = m.group(2).strip()
    return (lang, code) if code else None


def _copy_to_clipboard(text: str) -> bool:
    """Copy text to the system clipboard. Returns False if pyperclip is unavailable."""
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        return False


def _run_code_block(lang: str, code: str) -> None:
    """Execute a fenced code block extracted from a model response."""
    if _DANGEROUS_CMD_RE.search(code):
        console.print("[red][code block contains dangerous commands — blocked][/red]")
        return
    n = len(code.splitlines())
    console.print(f"[bold yellow]  [code_block][/bold yellow] {lang} ({n} lines)")
    answer = console.input("[bold yellow]  Execute? [y/N]:[/bold yellow] ").strip().lower()
    if answer != "y":
        console.print("[dim]  Code execution cancelled[/dim]")
        return
    result = do_run_script(lang, code)
    console.print(result)


def _smart_cap(client: OpenAI, result: str, name: str, context: str = "") -> str:
    """Cap a tool result. Summarizes web/fetch results via LLM; hard-truncates others."""
    if len(result) <= TOOL_RESULT_LIMIT:
        return result
    if name not in _SUMMARIZE_TOOLS:
        return _cap_result(result, name)
    try:
        # Text-based key-fact extraction first (fast, no LLM call)
        key_lines = []
        url_re = re.compile(r'https?://\S+')
        number_re = re.compile(r'\b\d{2,}(?:,\d{0,3})*(?:\.\d+)?\b')
        seen_urls = set()
        for _line in result.strip().split('\n'):
            _line = _line.strip()
            if not _line:
                continue
            urls = url_re.findall(_line)
            new_urls = [u for u in urls if u not in seen_urls]
            if new_urls or number_re.search(_line) or len(_line) < 120:
                seen_urls.update(new_urls)
                key_lines.append(_line)
        if len(key_lines) <= 80:
            return f'[key facts extracted {len(result):,}->{len(key_lines):,} lines]\n\n' + '\n'.join(key_lines)
        ctx_hint = f' relevant to: "{context[:200]}"' if context else ""
        prompt = [
            {"role": "system", "content": "Summarize precisely. Preserve all URLs, numbers, and key facts."},
            {"role": "user",   "content": (
                f"Summarize the following{ctx_hint} in under 1500 words, "
                f"keeping all URLs and critical data:\n\n{result[:40_000]}"
            )},
        ]
        resp = client.chat.completions.create(
            model=MODEL, messages=prompt, stream=False, max_tokens=1500,
            timeout=AUX_LLM_TIMEOUT,
        )
        summary = (resp.choices[0].message.content or "").strip()
        return f"[summarized {len(result):,}→{len(summary):,} chars]\n\n{summary}"
    except Exception:
        return _cap_result(result, name)


_context_growth_history: list[int] = []  # token counts per turn for growth tracking

def _track_context_growth(token_count: int) -> None:
    """Track token count after each turn for predictive compaction."""
    _context_growth_history.append(token_count)
    if len(_context_growth_history) > 10:
        _context_growth_history.pop(0)


def _estimate_turns_remaining(current_tokens: int, threshold_pct: int = 80) -> int:
    """Estimate how many turns until context reaches the threshold."""
    if len(_context_growth_history) < 2:
        return -1
    recent = _context_growth_history[-5:] if len(_context_growth_history) >= 5 else _context_growth_history
    growths = [recent[i] - recent[i-1] for i in range(1, len(recent)) if recent[i] > recent[i-1]]
    if not growths:
        return -1
    avg_growth = sum(growths) // len(growths)
    if avg_growth <= 0:
        return -1
    threshold_tokens = TOKEN_LIMIT * threshold_pct // 100
    remaining = max(0, threshold_tokens - current_tokens)
    return max(1, remaining // avg_growth)


def _detect_session_type(history: list) -> str:
    """Classify the session as 'chatty', 'heavy', or 'normal' based on message sizes."""
    if not history:
        return "normal"
    sizes = [len(m.get("content", "") or "") for m in history if m.get("role") in ("user", "assistant")]
    if not sizes:
        return "normal"
    avg_size = sum(sizes) / len(sizes)
    if avg_size > 5000:
        return "heavy"
    if len(sizes) > 20 and avg_size < 500:
        return "chatty"
    return "normal"


def _adaptive_compaction_threshold(session_type: str) -> int:
    """Return the compaction threshold percentage based on session type."""
    if session_type == "heavy":
        return 70
    elif session_type == "chatty":
        return 85
    return 80


def _cap_result(result: str, name: str = "") -> str:
    """Internal helper: cap result."""
    if len(result) <= TOOL_RESULT_LIMIT:
        return result
    return (result[:TOOL_RESULT_LIMIT]
            + f"\n\n... [truncated: result was {len(result):,} chars; showing first {TOOL_RESULT_LIMIT:,}]")


def _call_with_retry(name: str, args: dict, dispatch_fn, max_retries: int = _TOOL_RETRY_MAX) -> str:
    """Execute a tool call with retry logic and exponential backoff.

    Only retries retryable tools (network/file reads). Write operations and
    interactive tools are never retried to avoid side effects.

    On each retry the error from the previous attempt is appended to the
    returned result so the model can see what failed and adapt.
    """
    if name not in _TOOL_RETRYABLE_TOOLS:
        return dispatch_fn(name, args)

    errors: list[str] = []
    for attempt in range(1 + max_retries):
        try:
            return dispatch_fn(name, args)
        except Exception as e:
            errors.append(f"[attempt {attempt + 1}/{1 + max_retries}] {type(e).__name__}: {e}")
            if attempt < max_retries:
                backoff = _TOOL_RETRY_BASE * (2 ** attempt)
                console.print(f"[dim yellow]  [retry {attempt + 1}/{max_retries}] {name} failed — "
                              f"retrying in {backoff:.0f}s: {e}[/dim yellow]")
                time.sleep(backoff)

    # All retries exhausted — return a structured error so the model can reason about it
    err_summary = "; ".join(errors)
    hint_map = {
        "web_search": "Try a different query or use fetch_url on a specific URL.",
        "search_news": "Try rewording the query or use web_search instead.",
        "fetch_url": "Check the URL is correct. Try fetch_rendered for JavaScript pages.",
        "fetch_rendered": "The page may require interaction. Try browser_action to navigate and fill forms.",
        "describe_image": "Verify the image URL is publicly accessible.",
        "get_video_transcript": "The video may not have captions. Try searching for a transcript via web_search.",
    }
    hint = hint_map.get(name, "Consider using a different tool to achieve the same goal.")
    return f"[tool_call_failed: {name} after {1 + max_retries} attempts\nerrors: {err_summary}\nhint: {hint}]"


def _call_with_timeout(name: str, fn, *args, timeout: int, **kwargs) -> str:
    """Call a tool function with a timeout. Returns the result or a timeout error string."""
    result_holder = []
    error_holder = []

    def _target():
        try:
            result_holder.append(fn(*args, **kwargs))
        except Exception as e:
            error_holder.append(e)

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        return f"[{name}] timed out after {timeout}s. The operation took too long to complete."
    if error_holder:
        raise error_holder[0]
    if result_holder:
        return result_holder[0]
    # Shouldn't happen, but be safe
    return f"[{name}] returned empty result"


def _call_tool_safe(name: str, args: dict) -> str:
    """Dispatch a parallel-safe (non-interactive) tool."""
    if name == "web_search":
        q = args.get("query", "")
        console.print(f"[dim cyan]  web_search: {q}[/dim cyan]")
        result = _call_with_timeout(name, do_web_search, q, timeout=_TOOL_TIMEOUT_SLOW)
        console.print(f"[dim cyan]  → {len(result.splitlines())} lines[/dim cyan]")
        return result
    if name == "search_news":
        q = args.get("query", "")
        console.print(f"[dim cyan]  search_news: {q}[/dim cyan]")
        result = _call_with_timeout(name, do_search_news, q, args.get("max_results", 8), timeout=_TOOL_TIMEOUT_SLOW)
        console.print(f"[dim cyan]  → {len(result.splitlines())} lines[/dim cyan]")
        return result
    if name == "fetch_url":
        u = args.get("url", "")
        console.print(f"[dim cyan]  fetch_url: {u}[/dim cyan]")
        result = _call_with_timeout(name, do_fetch_url, u, args.get("max_chars", 20_000), timeout=_TOOL_TIMEOUT_NET)
        console.print(f"[dim cyan]  → {len(result):,} chars[/dim cyan]")
        return result
    if name == "describe_image":
        u = args.get("url", "")
        console.print(f"[dim cyan]  describe_image: {u}[/dim cyan]")
        return _call_with_timeout(name, do_describe_image, u, timeout=_TOOL_TIMEOUT_NET)
    if name == "get_video_transcript":
        u = args.get("url", "")
        console.print(f"[dim cyan]  get_video_transcript: {u}[/dim cyan]")
        return _call_with_timeout(name, do_get_video_transcript, u, args.get("lang", "en"), timeout=_TOOL_TIMEOUT_SLOW)
    if name == "read_file":
        return _call_with_timeout(name, do_read_file, args.get("path", ""), args.get("offset", 0), args.get("limit", 0), timeout=_TOOL_TIMEOUT_FAST)
    if name == "list_directory":
        p = args.get("path", "")
        console.print(f"[dim cyan]  list_directory: {p}[/dim cyan]")
        return _call_with_timeout(name, do_list_directory, p, args.get("recursive", False), timeout=_TOOL_TIMEOUT_FAST)
    if name == "find_files":
        p, pat = args.get("path", ""), args.get("pattern", "*")
        console.print(f"[dim cyan]  find_files: {pat} in {p}[/dim cyan]")
        return _call_with_timeout(name, do_find_files, p, pat, timeout=_TOOL_TIMEOUT_FAST)
    if name == "search_files":
        p, q2 = args.get("path", ""), args.get("query", "")
        console.print(f"[dim cyan]  search_files: '{q2}' in {p}[/dim cyan]")
        return _call_with_timeout(name, do_search_files, p, q2, args.get("pattern", "**/*"), args.get("context", 0), timeout=_TOOL_TIMEOUT_FAST)
    if name == "team_task_list":
        team = args.get("team", "")
        console.print(f"[dim cyan]  team_task_list: {team}[/dim cyan]")
        tasks = _ct_task_list(team, args.get("owner", ""), args.get("status", ""))
        if not tasks:
            return f"[no tasks in team '{team}']"
        lines = [f"Tasks for team '{team}' ({len(tasks)} total):"]
        for t in tasks:
            owner_tag = f" [{t['owner']}]" if t.get("owner") else ""
            lines.append(f"  [{t['id'][:6]}] [{t['status']}] {t['subject']}{owner_tag}")
        return "\n".join(lines)
    if name == "team_task_add":
        team, subject = args.get("team", ""), args.get("subject", "")
        console.print(f"[dim cyan]  team_task_add: {team}[/dim cyan]")
        return do_team_task_add(team, subject, args.get("owner", ""), args.get("priority", "medium"))
    if name == "team_board":
        team = args.get("team", "")
        console.print(f"[dim cyan]  team_board: {team}[/dim cyan]")
        return do_team_board(team)
    if name == "team_list":
        console.print("[dim cyan]  team_list[/dim cyan]")
        return do_team_list()
    if name == "team_inbox_receive":
        team, agent = args.get("team", ""), args.get("agent", "")
        peek = args.get("peek", False)
        console.print(f"[dim cyan]  team_inbox_receive: {team}/{agent}{'  (peek)' if peek else ''}[/dim cyan]")
        msgs = _ct_inbox_receive(team, agent, peek=peek)
        if not msgs:
            return f"[inbox empty for {agent} in team '{team}']"
        lines = [f"Messages for {agent} in team '{team}':"]
        for m in msgs:
            lines.append(f"  From: {m.get('from', '?')}  |  {m.get('body', '')}")
        return "\n".join(lines)
    return f"[unknown tool: {name}]"


def do_ask_user(question: str) -> str:
    """Handle ask user operation."""
    console.print(f"\n[bold cyan]  [ask_user][/bold cyan] {question}")
    try:
        answer = console.input("[bold cyan]  Your answer: [/bold cyan]").strip()
        return answer if answer else "[no answer provided]"
    except (KeyboardInterrupt, EOFError):
        return "[user cancelled]"


def _dispatch_interactive(name: str, args: dict) -> str:
    """Dispatch an interactive tool (may prompt user — must run on main thread)."""
    if name == "fetch_rendered":
        u = args.get("url", "")
        console.print(f"[bold cyan]  [fetch_rendered][/bold cyan] {u}")
        return do_fetch_rendered(u, args.get("max_chars", 15000))
    if name == "browser_action":
        action = args.get("action", "")
        console.print(f"[bold cyan]  [browser_action][/bold cyan] {action}")
        return do_browser_action(
            action=action,
            url=args.get("url", ""),
            selector=args.get("selector", ""),
            value=args.get("value", ""),
            screenshot_path=args.get("screenshot_path", ""),
        )
    if name == "run_command":
        return do_run_command(
            args.get("command", ""), args.get("cwd", ""), args.get("timeout", 30),
            env=args.get("env") or None, stdin=args.get("stdin", ""),
        )
    if name == "run_script":
        return do_run_script(
            args.get("language", ""), args.get("code", ""),
            cwd=args.get("cwd", ""), timeout=args.get("timeout", 30),
        )
    if name == "patch_file":
        return do_patch_file(args.get("path", ""), args.get("diff", ""))
    if name == "write_file":
        return do_write_file(args.get("path", ""), args.get("content", ""))
    if name == "move_file":
        return do_move_file(args.get("src", ""), args.get("dst", ""))
    if name == "delete_file":
        return do_delete_file(args.get("path", ""))
    if name == "ask_user":
        return do_ask_user(args.get("question", ""))
    if name == "team_task_update":
        team, task_id = args.get("team", ""), args.get("task_id", "")
        console.print(f"[dim cyan]  team_task_update: {team}/{task_id}[/dim cyan]")
        task = _ct_task_update(team, task_id, args.get("status"), args.get("owner"), args.get("note", ""))
        if not task:
            return f"[task '{task_id}' not found in team '{team}']"
        nn = len(task.get("notes", []))
        notes_tag = f"  ({nn} note{'s' if nn != 1 else ''})" if nn else ""
        return f"[updated: {task['id'][:6]}] {task['subject']} → {task['status']}{notes_tag}"
    if name == "team_inbox_send":
        team, to = args.get("team", ""), args.get("to", "")
        msg = args.get("message", "")
        from_agent = args.get("from_agent", "user")
        console.print(f"[dim cyan]  team_inbox_send: {team}/{to}[/dim cyan]")
        msg_id = _ct_inbox_send(team, to, msg, from_agent)
        return f"[sent message {msg_id[:8]} to {to} in team '{team}']"
    if name == "team_spawn_agent":
        team = args.get("team", "")
        agent_name = args.get("agent_name", "")
        task = args.get("task", "")
        cwd = args.get("cwd", "")
        console.print(f"[dim cyan]  team_spawn_agent: {team}/{agent_name}[/dim cyan]")
        return _ct_spawn(team, agent_name, task, cwd)
    return f"[unknown tool: {name}]"


def _execute_tool_call(client: "OpenAI", name: str, args: dict) -> str:
    """Dispatch a tool call in pipe mode (no interactive prompts)."""
    if name in _PARALLEL_TOOLS:
        return _call_tool_safe(name, args)
    return _dispatch_interactive(name, args)


def _strip_think(text: str) -> tuple[str, str]:
    """Extract <think>…</think> blocks. Returns (clean_text, think_content)."""
    parts = _THINK_RE.findall(text)
    if not parts:
        return text, ""
    clean = _THINK_RE.sub("", text).strip()
    think = "\n\n---\n\n".join(p.strip() for p in parts if p.strip())
    return clean, think


def _auto_presearch(working: list) -> list:
    """
    Before the first LLM call, auto-run web_search on the user's message and
    inject the results as grounding context. Whether it fires is governed by
    AUTO_SEARCH_MODE (off | smart | aggressive); see presearch_decision().
    """
    last_user = next(
        (m.get("content", "") for m in reversed(working) if m.get("role") == "user"), ""
    )
    do_search, query = presearch_decision(last_user, AUTO_SEARCH_MODE)
    if not do_search:
        return working

    console.print(f"[dim cyan]  [auto-search] {query[:70]}[/dim cyan]")
    results = do_web_search(query, max_results=5)
    suffix = (
        f"\n\n[Auto web search results — use these to ground your answer. "
        f"Search again with web_search if you need more detail.]\n\n{results}"
    )

    # Append results to the last user message content (avoids mid-conversation system messages)
    new_working = list(working)
    for i in range(len(new_working) - 1, -1, -1):
        if new_working[i].get("role") == "user":
            msg = dict(new_working[i])
            msg["content"] = (msg.get("content") or "") + suffix
            new_working[i] = msg
            return new_working

    return new_working


def _compact_tool_loop(working: list, keep_recent_tools: int = 4, head_chars: int = 240) -> list:
    """Shrink older tool results during an in-progress tool loop so a long agentic
    run doesn't overflow the context window before it finishes.

    Only the *content* of older `tool` messages is truncated to a short preview — no
    message is removed, so every assistant `tool_calls` keeps its matching `tool`
    result (the API requires that pairing; dropping messages would orphan a call).
    The most recent `keep_recent_tools` results, all assistant reasoning, and the
    task/user messages are left intact, so the model keeps its recent findings and
    still knows what it is working on.
    """
    tool_idxs = [i for i, m in enumerate(working) if m.get("role") == "tool"]
    if len(tool_idxs) <= keep_recent_tools:
        return working
    new    = list(working)
    shrunk = 0
    for i in tool_idxs[:-keep_recent_tools]:
        content = new[i].get("content") or ""
        if len(content) > head_chars + 80:
            new[i] = {**new[i], "content":
                      content[:head_chars] + f"\n[... {len(content) - head_chars:,} chars condensed to fit context]"}
            shrunk += 1
    if shrunk:
        console.print(f"[dim]  [mid-run compact] condensed {shrunk} older tool result(s) to free context[/dim]")
    return new


def run_turn(client: OpenAI, messages: list, allow_tools: bool = True) -> str | None:
    """Full turn with tool-use loop. Returns reply, '' on cancel, None on error."""
    global _last_turn_tokens, _real_ctx_tokens
    working    = _auto_presearch(list(messages)) if allow_tools else list(messages)
    # Feature 10: Surface unresolved errors from prior turn
    try:
        global _lsp_client_mod
        if _lsp_client_mod is None:
            import lsp_client as _mod
            _lsp_client_mod = _mod
        trend = _lsp_client_mod.lsp_trend_report()
        if trend.get("unresolved_errors"):
            console.print("[dim red]  Unresolved errors from prior edit:[/dim red]")
            for e in trend["unresolved_errors"][:5]:
                console.print(f"    {e}")
    except Exception:
        pass


    use_tools  = allow_tools
    first_call = True
    depth      = 0
    tool_chain: list[str] = []
    total_prompt     = 0
    total_completion = 0
    final_segments: list[str] = []   # final-answer pieces, joined across auto-continues
    auto_continue    = 0             # how many times we've resumed a cut-off answer

    while True:
        # Mid-run compaction: a long tool loop can pile up large tool results and
        # overflow the context window before the task finishes. When the working set
        # nears the limit, shrink older tool results in place so the run keeps going.
        if depth > 0 and approx_tokens(working) >= TOKEN_LIMIT * 0.85:
            working = _compact_tool_loop(working)

        if depth >= MAX_TOOL_DEPTH:
            console.print(f"[yellow][max tool depth ({MAX_TOOL_DEPTH}) reached — synthesizing with gathered data][/yellow]")
            # Force one final no-tools call so the model can answer with what it found
            synth_msgs = list(working)
            synth_msgs.append({"role": "user", "content":
                "Based on all the information gathered above, provide a comprehensive answer now. "
                "Do not search for more — synthesize what you have."})
            try:
                with Live("", console=console, vertical_overflow="crop",
                          refresh_per_second=15, transient=True) as live:
                    final_text, _, final_usage = stream_once(
                        client, synth_msgs, use_tools=False, update_fn=_live_updater(live)
                    )
                final_text, think_c = _strip_think(final_text)
                if think_c:
                    console.print(Panel(Markdown(think_c), title="[dim]thinking[/dim]",
                                        border_style="dim", padding=(0, 1)))
                console.print(Markdown(final_text) if final_text else Markdown("*(no synthesis)*"))
                total_prompt     += final_usage.get("prompt", 0)
                total_completion += final_usage.get("completion", 0)
            except Exception:
                final_text = ""
            if tool_chain:
                console.print(f"[dim]  ↳ {' → '.join(tool_chain)}[/dim]")
            _last_turn_tokens = {"prompt": total_prompt, "completion": total_completion}
            return ("".join(final_segments) + (final_text or "")) or None

        depth_tag = f" [dim][{depth}/{MAX_TOOL_DEPTH}][/dim]" if depth > 0 else ""
        console.print(Rule(f"[bold green]Qwen[/bold green]{depth_tag}", style="dim green"))

        try:
            with Live("", console=console, vertical_overflow="crop",
                      refresh_per_second=15, transient=True) as live:
                text, tool_calls, usage = stream_once(
                    client, working, use_tools, _live_updater(live)
                )
            text, think_content = _strip_think(text)
            if think_content:
                console.print(Panel(
                    Markdown(think_content),
                    title="[dim]thinking[/dim]", border_style="dim", padding=(0, 1),
                ))
            console.print(Markdown(text) if text else Markdown("*(no response)*"))
        except KeyboardInterrupt:
            console.print("\n[dim][cancelled][/dim]")
            return ""
        except Exception as e:
            if use_tools and first_call:
                console.print(f"[yellow][retrying without tool use — {e}][/yellow]")
                use_tools  = False
                first_call = False
                try:
                    with Live("", console=console, vertical_overflow="crop",
                              refresh_per_second=15, transient=True) as live:
                        text, tool_calls, usage = stream_once(
                            client, working, False, _live_updater(live)
                        )
                    text, think_content = _strip_think(text)
                    if think_content:
                        console.print(Panel(
                            Markdown(think_content),
                            title="[dim]thinking[/dim]", border_style="dim", padding=(0, 1),
                        ))
                    console.print(Markdown(text) if text else Markdown("*(no response)*"))
                except KeyboardInterrupt:
                    console.print("\n[dim][cancelled][/dim]")
                    return ""
                except Exception as e2:
                    console.print(f"[red][error] {e2}[/red]")
                    return None
            else:
                console.print(f"[red][error] {e}[/red]")
                return None

        total_prompt     += usage.get("prompt", 0)
        total_completion += usage.get("completion", 0)
        # Keep the most recent real prompt token count so auto-compact uses it
        if usage.get("prompt", 0):
            _real_ctx_tokens = usage["prompt"]

        if not tool_calls:
            final_segments.append(text)
            # The model produced a final answer with no tool calls. If it was cut
            # off mid-output (token cap hit, or the stream dropped), it hasn't
            # actually finished the project — resume it instead of stopping.
            if usage.get("truncated") and text.strip() and auto_continue < MAX_AUTO_CONTINUE:
                auto_continue += 1
                fr = usage.get("finish_reason") or "stream ended"
                console.print(
                    f"[yellow]  \\[response cut off ({fr}) — continuing "
                    f"{auto_continue}/{MAX_AUTO_CONTINUE}][/yellow]"
                )
                working.append({"role": "assistant", "content": text})
                working.append({"role": "user", "content": (
                    "Your previous message was cut off before you finished. "
                    "Continue from exactly where you stopped — do not repeat anything "
                    "you already wrote, and do not start over."
                )})
                continue
            if usage.get("truncated") and auto_continue >= MAX_AUTO_CONTINUE:
                console.print(
                    f"[yellow]  \\[still cut off after {MAX_AUTO_CONTINUE} continuations "
                    f"— stopping][/yellow]"
                )
            if tool_chain:
                console.print(f"[dim]  ↳ {' → '.join(tool_chain)}[/dim]")
            _last_turn_tokens = {"prompt": total_prompt, "completion": total_completion}
            return "".join(final_segments)

        working.append({
            "role": "assistant",
            "content": text or None,
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["function"]["name"],
                              "arguments": tc["function"]["arguments"]}}
                for tc in tool_calls
            ],
        })

        # Parse args upfront; handle malformed JSON per-call
        parsed_args: list[dict | None] = []
        for tc in tool_calls:
            try:
                parsed_args.append(json.loads(tc["function"]["arguments"]))
            except json.JSONDecodeError:
                parsed_args.append(None)

        # Extract user query for summarization context
        last_user_msg = next(
            (m.get("content", "") for m in reversed(working) if m.get("role") == "user"), ""
        )

        # Collect results indexed by position
        tool_results: dict[int, tuple[str, str]] = {}  # idx -> (tc_id, result)

        # Classify tools into sequential batches to avoid conflicts
        batches = _classify_tool_batch(tool_calls, parsed_args)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for batch in batches:
                safe_futures: dict[int, concurrent.futures.Future] = {}

                for i in batch:
                    tc, args = tool_calls[i], parsed_args[i]
                    name = tc["function"]["name"]
                    if args is None:
                        err = (f"[tool_call_error: malformed JSON arguments for '{name}'. "
                               f"Please retry the call with valid JSON.]")
                        tool_chain.append(f"{name}(!json)")
                        tool_results[i] = (tc["id"], err)
                        continue
                    if name in _PARALLEL_TOOLS:
                        tool_chain.append(f"{name}({_short_args(name, args)})")
                        safe_futures[i] = pool.submit(
                            _call_with_retry, name, args, _call_tool_safe, max_retries=_TOOL_RETRY_MAX,
                        )
                    elif name in _TOOL_RETRYABLE_TOOLS and name not in _PARALLEL_TOOLS:
                        tool_chain.append(f"{name}({_short_args(name, args)})")
                        result = _call_with_retry(name, args, _dispatch_interactive)
                        tool_results[i] = (tc["id"], _smart_cap(client, result, name, last_user_msg))
                    else:
                        tool_chain.append(f"{name}({_short_args(name, args)})")
                        result = _dispatch_interactive(name, args)
                        tool_results[i] = (tc["id"], _smart_cap(client, result, name, last_user_msg))

                for i, fut in safe_futures.items():
                    try:
                        result = fut.result(timeout=60)
                    except Exception as exc:
                        idx = i
                        rname = tool_calls[idx]["function"]["name"]
                        try:
                            result = _call_with_retry(rname, parsed_args[idx], _call_tool_safe, max_retries=1)
                        except Exception as exc2:
                            result = f"[tool error: {exc}; retry: {exc2}]"
                    tool_results[i] = (tool_calls[i]["id"],
                                       _smart_cap(client, result, tool_calls[i]["function"]["name"], last_user_msg))

        for i in range(len(tool_calls)):
            tc_id, result = tool_results[i]
            working.append({"role": "tool", "tool_call_id": tc_id, "content": result})

        first_call = False
        depth     += 1

# ---------------------------------------------------------------------------
# Truncate-middle — fast, no LLM call, used by auto-compact
# ---------------------------------------------------------------------------

def truncate_middle(history: list, keep_first: int = 6, keep_last: int = 20) -> list:
    """Drop middle messages to recover context space. Keeps system msgs + first K + last K chat turns."""
    sys_msgs  = [m for m in history if m.get("role") == "system"]
    chat_msgs = [m for m in history if m.get("role") not in ("system",)]

    if len(chat_msgs) <= keep_first + keep_last:
        console.print("[dim][truncate-middle: history too short to trim][/dim]")
        return history

    first   = chat_msgs[:keep_first]
    last    = chat_msgs[-keep_last:]
    dropped = len(chat_msgs) - keep_first - keep_last

    marker = {
        "role": "system",
        "content": (
            f"[{dropped} messages from the middle of this conversation were removed "
            f"to fit the context window. The first {keep_first} and most recent "
            f"{keep_last} messages are retained below.]"
        ),
    }
    new_history = sys_msgs + first + [marker] + last
    console.print(
        f"[dim][truncate-middle: dropped {dropped} middle messages — "
        f"kept {keep_first} oldest + {keep_last} most recent][/dim]"
    )
    return new_history


# ---------------------------------------------------------------------------
# /trim — sliding-window summarization (manual, LLM-based)
# ---------------------------------------------------------------------------

def cmd_trim(history: list, client: OpenAI) -> list:
    """Command: trim."""
    CHUNK = 8
    KEEP  = 4
    if len(history) < KEEP + CHUNK:
        console.print("[dim][history too short to trim (need ≥12 turns)][/dim]")
        return history

    # Separate "work" turns (file writes/patches) from pure chat — keep work turns verbatim
    _WORK_MARKERS = ("[patched:", "[created:", "[updated:", "[write_file]", "[patch_file]")
    work_pairs: list[tuple[int, dict, dict | None]] = []  # (orig_idx, user_msg, asst_msg)
    chat_only: list[dict] = []

    to_process_full = history[:-KEEP]
    keep            = history[-KEEP:]

    i = 0
    while i < len(to_process_full):
        msg = to_process_full[i]
        if msg.get("role") == "user":
            asst = to_process_full[i + 1] if i + 1 < len(to_process_full) else None
            asst_content = (asst.get("content") or "") if asst else ""
            if any(m in asst_content for m in _WORK_MARKERS):
                work_pairs.append((i, msg, asst))
                i += 2
                continue
        chat_only.append(msg)
        i += 1

    to_process = chat_only
    if not to_process:
        console.print("[dim][nothing to summarize — all turns contain file modifications][/dim]")
        return history

    console.print(
        f"[dim]Summarizing {len(to_process)} chat messages"
        f"{f' (preserving {len(work_pairs)} work turn(s) verbatim)' if work_pairs else ''}"
        f" in chunks of {CHUNK}...[/dim]"
    )

    rolling_summary = ""
    i = 0
    chunk_num = 0
    while i < len(to_process):
        chunk      = to_process[i:i + CHUNK]
        chunk_num += 1
        chunk_text = "\n".join(
            f"{m['role'].upper()}: {(m.get('content') or '')[:600]}"
            for m in chunk if m.get("role") in ("user", "assistant")
        )
        user_content = (
            (f"Previous summary: {rolling_summary}\n\n"
             f"New exchanges:\n{chunk_text}\n\n"
             "Write an updated running summary in 3-5 sentences.")
            if rolling_summary else
            f"Summarize this conversation in 3-5 sentences:\n\n{chunk_text}"
        )
        prompt = [
            {"role": "system", "content": (
                "You are a concise summarizer for an ongoing work session. Preserve key "
                "decisions, facts, outcomes, code changes made, files modified, commands run, "
                "and their results — note what was changed and why. Then ALWAYS end with a "
                "line exactly of the form 'CURRENT TASK: <the objective being actively worked "
                "on right now and the immediate next step>' so the work can continue "
                "seamlessly after older messages are dropped from context."
            )},
            {"role": "user",   "content": user_content},
        ]
        console.print(Rule(f"[dim]Chunk {chunk_num}[/dim]", style="dim"))
        try:
            with Live("", console=console, vertical_overflow="crop",
                      refresh_per_second=15, transient=True) as live:
                summary, _, _ = stream_once(client, prompt, use_tools=False,
                                            update_fn=_live_updater(live))
            console.print(Markdown(summary))
            rolling_summary = summary
        except KeyboardInterrupt:
            console.print("[dim][trim cancelled — keeping partial progress][/dim]")
            break
        i += CHUNK

    if rolling_summary:
        # Prepend session-changes note so the model knows what was modified
        if _session_changes:
            modified = ", ".join(Path(p).name for p in _session_changes)
            rolling_summary = f"[Files modified this session: {modified}]\n\n{rolling_summary}"
        summary_msg = {"role": "system", "content": (
            "[Earlier messages were compacted to free context. Summary of the work so far "
            "below — keep going on the CURRENT TASK noted at its end without asking the user "
            f"to repeat themselves:\n{rolling_summary}]"
        )}
        # Re-insert preserved work turns in their original order
        preserved = [msg for _, u, a in sorted(work_pairs, key=lambda x: x[0])
                     for msg in ([u] + ([a] if a else []))]
        new_history = [summary_msg] + preserved + keep
        console.print(
            f"[dim][trimmed → 1 summary + {len(preserved)} preserved work turn(s) + {len(keep)} recent][/dim]"
        )
        return new_history
    return history


def _maybe_autocompact(history: list, base_system: str, client: "OpenAI") -> list:
    """Keep the session going when the context window fills up.

    When usage crosses the limit we SUMMARIZE old turns (preserving file-editing
    turns verbatim and an explicit 'CURRENT TASK' note) so the model remembers
    what it was working on and continues seamlessly. Blunt truncate-middle is only
    a last resort if summarization is unavailable or didn't free enough space.

    Called both before and after each turn so a single large exchange can't blow
    past the window unhandled. Returns the (possibly compacted) history.
    """
    global _real_ctx_tokens
    if _real_ctx_tokens:
        tok = _real_ctx_tokens
    else:
        full_msgs = [{"role": "system", "content": build_system_prompt(base_system)}] + history
        tok = approx_tokens(full_msgs)
    pct = tok * 100 // TOKEN_LIMIT
    _track_context_growth(tok)

    compaction_threshold = _adaptive_compaction_threshold(_detect_session_type(history))
    warning_threshold = max(60, compaction_threshold - 15)

    if pct >= compaction_threshold:
        label = "auto-compact" if pct >= 90 else "auto-trim"
        console.print(
            f"[yellow]  [{label}] context at {pct}% ({tok:,} tokens) — "
            f"summarizing & preserving current task so work continues...[/yellow]"
        )
        before = len(history)
        try:
            history = cmd_trim(history, client)
        except Exception as _trim_err:
            console.print(f"[dim][summarize failed: {_trim_err} — truncating middle as fallback][/dim]")
            history = truncate_middle(history)
        # If still over after trim, escalate with progressively smaller keep values
        after_tok = approx_tokens([{"role": "system", "content": build_system_prompt(base_system)}] + history)
        if after_tok * 100 // TOKEN_LIMIT >= 85:
            history = truncate_middle(history, keep_first=4, keep_last=16)
            after_tok = approx_tokens([{"role": "system", "content": build_system_prompt(base_system)}] + history)
        # Emergency: if still critically full, aggressive truncation
        if after_tok * 100 // TOKEN_LIMIT >= 92:
            console.print("[yellow]  [emergency-trim] still over — aggressive middle truncation[/yellow]")
            history = truncate_middle(history, keep_first=2, keep_last=12)
            _real_ctx_tokens = 0
            return history
        # If summarization couldn't shrink (too few but huge turns) and we're
        # critically full, fall back to a hard middle-truncation as a safety net.
        if len(history) >= before and pct >= 90:
            history = truncate_middle(history)
        _real_ctx_tokens = 0
        after_tok = approx_tokens([{"role": "system", "content": build_system_prompt(base_system)}] + history)
        console.print(
            f"[green]  [{label} done] freed context to {after_tok * 100 // TOKEN_LIMIT}% ({after_tok:,} tokens)[/green]"
        )
    elif pct >= warning_threshold:
        turns_left = _estimate_turns_remaining(tok, compaction_threshold)
        turns_hint = f"~{turns_left} turns until auto-trim, " if turns_left > 0 else ""
        console.print(
            f"[yellow]  context: ~{tok:,} / {TOKEN_LIMIT:,} tokens ({pct}%) — "
            f"{turns_hint}auto-trim at {compaction_threshold}%[/yellow]"
        )
    return history


# ---------------------------------------------------------------------------
# Pipe mode
# ---------------------------------------------------------------------------

def run_piped(client: OpenAI) -> None:
    """Run in non-interactive pipe mode with full multi-round tool loop.

    stdout → clean final answer only (safe to pipe / redirect to file)
    stderr → tool status, intermediate chain-of-thought, errors
    """
    piped  = sys.stdin.read().strip()
    prefix = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else ""
    prompt = f"{prefix}\n\n{piped}".strip() if prefix else piped
    if not prompt:
        sys.exit(0)

    global console
    # Redirect Rich console to stderr so tool-activity lines don't pollute stdout
    _saved_console = console
    console = Console(stderr=True, legacy_windows=False)
    try:
        messages: list = [
            {"role": "system", "content": build_system_prompt(BASE_SYSTEM)},
            {"role": "user",   "content": prompt},
        ]
        depth = 0
        max_depth = 10
        while depth < max_depth:
            stream = client.chat.completions.create(
                model=MODEL, messages=messages, stream=True,
                tools=TOOLS, tool_choice="auto",
            )
            tc_buf: dict[int, dict] = {}
            text_parts: list[str] = []
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    text_parts.append(delta.content)  # buffer; don't write yet
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tc_buf:
                            tc_buf[idx] = {"id": "", "type": "function",
                                           "function": {"name": "", "arguments": ""}}
                        if tc.id:
                            tc_buf[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tc_buf[idx]["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                tc_buf[idx]["function"]["arguments"] += tc.function.arguments

            tool_calls = list(tc_buf.values())
            if not tool_calls:
                # Final round — strip <think> blocks then write clean answer to stdout
                final_text = "".join(text_parts)
                clean, _ = _strip_think(final_text)
                sys.stdout.write(clean)
                break

            # Intermediate round — chain-of-thought goes to stderr so it's visible
            # on the terminal but doesn't contaminate piped output
            if text_parts:
                sys.stderr.write("".join(text_parts))
                sys.stderr.flush()

            assistant_content = "".join(text_parts) or None
            messages.append({"role": "assistant", "content": assistant_content,
                             "tool_calls": tool_calls})
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    fn_args = {}
                result = _execute_tool_call(client, fn_name, fn_args)
                messages.append({
                    "role": "tool", "tool_call_id": tc["id"],
                    "name": fn_name, "content": str(result),
                })
            depth += 1

        sys.stdout.write("\n")
    except Exception as e:
        sys.stderr.write(f"error: {e}\n")
        sys.exit(1)
    finally:
        console = _saved_console

# ---------------------------------------------------------------------------
# Config display
# ---------------------------------------------------------------------------


def show_config() -> None:
    """Show Config"""
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="cyan")
    t.add_row("Config file",    str(CONFIG_FILE) + (" ✓" if CONFIG_FILE.exists() else " (not found)"))
    t.add_row("base_url",       BASE_URL)
    t.add_row("model",          MODEL)
    t.add_row("token_limit",    str(TOKEN_LIMIT))
    t.add_row("max_tool_depth", str(MAX_TOOL_DEPTH))
    t.add_row("auto_search",    AUTO_SEARCH_MODE)
    t.add_row("editor",         DEFAULT_EDITOR)
    t.add_row("data_dir",       str(DATA_DIR))
    console.print(t)
    if not CONFIG_FILE.exists():
        console.print(f"[dim]Create {CONFIG_FILE} to set persistent defaults.[/dim]")
        console.print('[dim]Example:\n  base_url = "http://localhost:8000/v1"\n  model = "local-model"\n  editor = "code --wait"[/dim]')

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def make_client() -> "OpenAI":
    """Make Client"""
    global MODEL, ACTIVE_BACKEND
    import httpx
    _timeout = httpx.Timeout(connect=30.0, read=3600.0, write=60.0, pool=10.0)
    lm_client = OpenAI(base_url=BASE_URL, api_key="no-key", timeout=_timeout)
    try:
        lm_client.models.list()
        ACTIVE_BACKEND = "llama.cpp"
        return lm_client
    except Exception:
        pass

    if OPENAI_API_KEY:
        console.print(f"[yellow]  llama.cpp server unreachable — falling back to OpenAI ({FALLBACK_MODEL})[/yellow]")
        MODEL = FALLBACK_MODEL
        ACTIVE_BACKEND = "openai"
        return OpenAI(api_key=OPENAI_API_KEY, timeout=_timeout)

    console.print(f"[yellow]  [warning] llama.cpp server not reachable at {BASE_URL}[/yellow]")
    console.print("[dim]  Set openai_api_key in config.toml or OPENAI_API_KEY env var to enable cloud fallback.[/dim]")
    ACTIVE_BACKEND = "llama.cpp"
    return lm_client

# ==============================================================================
# REPL Context & Command Dispatch System
# ==============================================================================

class _ReplContext:
    """Holds shared mutable state for the REPL loop.

    Encapsulates history, base_system, and client so command handlers
    don't need to access globals directly.
    """
    def __init__(self, history: list[dict], base_system: str, client):
        self.history = history
        self.base_system = base_system
        self.client = client


def _run_turn_and_handle_reply(ctx: _ReplContext, user_input: str,
                                allow_tools: bool = True) -> None:
    """Run a single LLM turn and handle the reply side-effects.

    This is the shared logic between the normal REPL flow and /retry.
    Appends to ctx.history, increments _turn_count, handles autosave,
    background tasks, and code block prompts.
    """
    global _turn_count

    # Pre-turn safety: compact if near limit
    ctx.history = _maybe_autocompact(ctx.history, ctx.base_system, ctx.client)

    messages = [{"role": "system", "content": build_system_prompt(ctx.base_system)}] + ctx.history
    messages.append({"role": "user", "content": user_input})

    t0 = time.monotonic()
    with _main_llm_busy_lock:
        _main_llm_busy = True
    try:
        reply = run_turn(ctx.client, messages, allow_tools=allow_tools)
    finally:
        with _main_llm_busy_lock:
            _main_llm_busy = False
    elapsed = time.monotonic() - t0

    if not reply:
        return

    ctx.history.append({"role": "user", "content": user_input})
    ctx.history.append({"role": "assistant", "content": reply})
    _turn_count += 1
    _print_turn_footer(elapsed)
    _confidence_warning(reply)

    # Plan detection
    if allow_tools and _looks_like_plan(reply):
        try:
            ans = console.input("[dim]  Plan detected — execute? [y/N]: [/dim]").strip().lower()
            if ans == "y":
                exec_msgs = [{"role": "system", "content": build_system_prompt(ctx.base_system)}] + ctx.history
                exec_msgs.append({"role": "user", "content": "Please execute this plan step by step now."})
                t1 = time.monotonic()
                exec_reply = run_turn(ctx.client, exec_msgs, allow_tools=True)
                if exec_reply:
                    ctx.history.append({"role": "user", "content": "Please execute this plan step by step now."})
                    ctx.history.append({"role": "assistant", "content": exec_reply})
                    _turn_count += 1
                    _print_turn_footer(time.monotonic() - t1)
        except (KeyboardInterrupt, EOFError):
            pass

    # Auto-compact
    ctx.history = _maybe_autocompact(ctx.history, ctx.base_system, ctx.client)

    # Silent autosave
    if _turn_count % AUTO_SAVE_INTERVAL == 0:
        _silent_autosave(ctx.history, ctx.base_system)

    # Background: auto-extract memory
    threading.Thread(
        target=_auto_extract_memory,
        args=(ctx.client, user_input, reply),
        daemon=True,
    ).start()

    # Background: intel processing
    threading.Thread(
        target=_intel_process_queue,
        args=(ctx.client,),
        daemon=True,
    ).start()
    threading.Thread(
        target=_intel_extract_topics,
        args=(ctx.client, user_input, reply),
        daemon=True,
    ).start()

    # Background: session title
    if _turn_count == 1:
        threading.Thread(
            target=_generate_session_title,
            args=(ctx.client, list(ctx.history)),
            daemon=True,
        ).start()

    # Offer to run / copy runnable code blocks
    runnable = _extract_runnable_code(reply)
    if runnable:
        lang, code = runnable
        label = lang or "code"
        try:
            answer = console.input(f"[dim]  [{label}] [r] run · [c] copy · [Enter] skip: [/dim]").strip().lower()
            if answer == "r":
                _run_code_block(lang, code)
            elif answer == "c":
                if _copy_to_clipboard(code):
                    console.print("[green]  [copied code block to clipboard][/green]")
                else:
                    console.print("[yellow]  [pip install pyperclip to enable copy][/yellow]")
        except (KeyboardInterrupt, EOFError):
            pass


# ==============================================================================
# Command Handler Functions
# ==============================================================================
# Signature: handler(ctx, arg) -> None
# To break the REPL loop, handlers raise StopIteration.
# ==============================================================================


def _cmd_exit(ctx: _ReplContext, arg: str) -> None:
    if ctx.history:
        save_session(ctx.history, ctx.base_system)
    console.print("[dim]Bye.[/dim]")
    raise StopIteration


def _cmd_help(ctx: _ReplContext, arg: str) -> None:
    if arg:
        q = arg.lower()
        rows = []
        for ln in HELP_TEXT.splitlines():
            s = ln.strip()
            if not s.startswith("|") or "---" in s or s == "| Command | Description |":
                continue
            if q in s.lower():
                rows.append(s)
        if rows:
            console.print(Markdown(
                "| Command | Description |\n|---|---|\n" + "\n".join(rows)
            ))
        else:
            console.print(f"[yellow][no help entries match '{arg}' — try /help with no argument][/yellow]")
    else:
        console.print(Markdown(HELP_TEXT))


def _cmd_clear(ctx: _ReplContext, arg: str) -> None:
    if arg.isdigit():
        n = int(arg)
        drop = min(n * 2, len(ctx.history))
        del ctx.history[-drop:]
        console.print(f"[dim][dropped {drop // 2} turn(s)][/dim]")
    else:
        ctx.history.clear()
        console.print("[dim][history cleared][/dim]")


def _cmd_retry(ctx: _ReplContext, arg: str) -> None:
    global _last_user_input, _turn_count
    if not _last_user_input:
        console.print("[yellow][nothing to retry][/yellow]")
        return
    user_input = _last_user_input
    if ctx.history and ctx.history[-1]["role"] == "assistant":
        ctx.history.pop()
    if ctx.history and ctx.history[-1]["role"] == "user" and ctx.history[-1]["content"] == user_input:
        ctx.history.pop()
    console.print(f"[dim][retrying: {user_input[:60]}{'…' if len(user_input)>60 else ''}][/dim]")
    _run_turn_and_handle_reply(ctx, user_input)


def _cmd_run(ctx: _ReplContext, arg: str) -> None:
    if not arg:
        console.print("[yellow][usage: /run <command>][/yellow]")
        return
    if _is_dangerous(arg):
        console.print(f"[bold red]  dangerous command:[/bold red] {arg}")
        answer = console.input("[bold red]  Run anyway? [y/N]:[/bold red] ").strip().lower()
        if answer != "y":
            console.print("[dim][cancelled][/dim]")
            return
    do_run_command(arg)


def _cmd_edit(ctx: _ReplContext, arg: str) -> None:
    if not arg:
        console.print("[yellow][usage: /edit <file>][/yellow]")
        return
    if ".." in Path(arg).parts:
        console.print("[red][edit blocked: path traversal (.. ) detected][/red]")
        return
    p = _resolve(arg)
    try:
        subprocess.run(f'{DEFAULT_EDITOR} "{p}"', shell=True)
        console.print(f"[dim][editor closed: {p.name}][/dim]")
    except Exception as e:
        console.print(f"[red][error opening editor: {e}][/red]")


def _cmd_system(ctx: _ReplContext, arg: str) -> None:
    if not arg:
        console.print(f"[dim]{ctx.base_system}[/dim]")
    else:
        ctx.base_system = arg
        ctx.history.clear()
        console.print("[dim][system prompt updated, history cleared][/dim]")


def _cmd_file(ctx: _ReplContext, arg: str) -> None:
    if not arg:
        console.print("[yellow][usage: /file <path>][/yellow]")
    else:
        load_file_into_context(arg, ctx.history)


def _cmd_project(ctx: _ReplContext, arg: str) -> None:
    load_project_context(arg, ctx.history)


def _cmd_focus(ctx: _ReplContext, arg: str) -> None:
    cmd_focus(arg, ctx.history)


def _cmd_changes(ctx: _ReplContext, arg: str) -> None:
    cmd_changes(arg)


def _cmd_search_sessions(ctx: _ReplContext, arg: str) -> None:
    cmd_search_sessions(arg)


def _cmd_undo(ctx: _ReplContext, arg: str) -> None:
    global _backup_stack
    if not _backup_stack:
        console.print("[yellow][no backup available][/yellow]")
    else:
        entry = _backup_stack.pop()
        orig: Path = entry["original"]
        bak: Path = entry["backup"]
        orig.write_text(entry["content"], encoding="utf-8")
        console.print(f"[green][restored: {orig.name} from {bak.name}][/green]")
        remaining = len(_backup_stack)
        if remaining:
            console.print(f"[dim]  ({remaining} more undo level{'s' if remaining != 1 else ''} available)[/dim]")


def _cmd_export(ctx: _ReplContext, arg: str) -> None:
    export_session(ctx.history, arg)


def _cmd_copy(ctx: _ReplContext, arg: str) -> None:
    cmd_copy(ctx.history)


def _cmd_paste(ctx: _ReplContext, arg: str) -> None:
    cmd_paste(ctx.history)


def _cmd_remember(ctx: _ReplContext, arg: str) -> None:
    if not arg:
        console.print("[yellow][usage: /remember <fact>][/yellow]")
    else:
        with _memory_lock:
            mem = load_memory()
            save_memory((mem + f"\n- {arg}").strip())
        console.print("[green][remembered — persists across sessions][/green]")


def _cmd_memory(ctx: _ReplContext, arg: str) -> None:
    mem = load_memory()
    console.print(Markdown(mem) if mem else "[dim][memory is empty][/dim]")


def _cmd_forget(ctx: _ReplContext, arg: str) -> None:
    try:
        confirm = console.input("[red]Clear ALL persistent memory? [y/N]: [/red]").strip().lower()
    except (KeyboardInterrupt, EOFError):
        confirm = ""
    if confirm == "y":
        with _memory_lock:
            save_memory("")
        console.print("[dim][persistent memory cleared][/dim]")
    else:
        console.print("[dim][cancelled][/dim]")


def _cmd_pin(ctx: _ReplContext, arg: str) -> None:
    pins = load_pins()
    if not arg or arg == "list":
        if not pins:
            console.print("[dim][no pins][/dim]")
        else:
            for i, p in enumerate(pins, 1):
                console.print(f"  [cyan]{i}.[/cyan] {p}")
    elif arg.startswith("remove "):
        try:
            idx = int(arg.split(None, 1)[1]) - 1
            removed = pins.pop(idx)
            save_pins(pins)
            console.print(f"[dim][removed pin: {removed}][/dim]")
        except (ValueError, IndexError):
            console.print("[yellow][usage: /pin remove <number>][/yellow]")
    else:
        pins.append(arg)
        save_pins(pins)
        console.print(f"[green][pinned: {arg}][/green]")


def _cmd_branch(ctx: _ReplContext, arg: str) -> None:
    global _branches
    sub_parts = arg.split(None, 1)
    sub = sub_parts[0] if sub_parts else ""
    sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""

    if sub == "list" or not sub:
        if not _branches:
            console.print("[dim][no branches][/dim]")
        else:
            for name, h in _branches.items():
                turns = sum(1 for m in h if m.get("role") == "assistant")
                console.print(f"  [cyan]{name}[/cyan]  [dim]{turns} turns[/dim]")
    elif sub == "restore":
        if not sub_arg:
            console.print("[yellow][usage: /branch restore <name>][/yellow]")
        elif sub_arg not in _branches:
            console.print(f"[red][no branch: {sub_arg}][/red]")
        else:
            ctx.history = list(_branches[sub_arg])
            turns = sum(1 for m in ctx.history if m.get("role") == "assistant")
            console.print(f"[green][restored branch '{sub_arg}' — {turns} turns][/green]")
    else:
        _branches[sub] = list(ctx.history)
        turns = sum(1 for m in ctx.history if m.get("role") == "assistant")
        console.print(f"[green][branch '{sub}' saved — {turns} turns][/green]")
        console.print(f"[dim]  restore with: /branch restore {sub}[/dim]")


def _cmd_search(ctx: _ReplContext, arg: str) -> None:
    global _turn_count
    if not arg:
        console.print("[yellow][usage: /search <query>][/yellow]")
        return
    console.print(f"[dim cyan][searching: {arg}][/dim cyan]")
    result = do_web_search(arg)
    console.print(Markdown(f"```\n{result}\n```"))
    search_input = f"[Web search results for: {arg}]\n{result}\n\nBased on these results, answer the query: {arg}"
    _run_turn_and_handle_reply(ctx, search_input)


def _cmd_save(ctx: _ReplContext, arg: str) -> None:
    save_session(ctx.history, ctx.base_system, arg or None)


def _cmd_load(ctx: _ReplContext, arg: str) -> None:
    result = cmd_load_session(arg, ctx.history, ctx.base_system)
    ctx.history, ctx.base_system = result


def _cmd_sessions(ctx: _ReplContext, arg: str) -> None:
    list_sessions()


def _cmd_context(ctx: _ReplContext, arg: str) -> None:
    show_context_breakdown(ctx.base_system, ctx.history)


def _cmd_trim(ctx: _ReplContext, arg: str) -> None:
    ctx.history = cmd_trim(ctx.history, ctx.client)


def _cmd_mode(ctx: _ReplContext, arg: str) -> None:
    global _current_mode
    if not arg:
        modes_list = " / ".join(_MODE_PROMPTS.keys())
        if _current_mode:
            console.print(f"[dim][mode: {_current_mode} — {_MODE_PROMPTS[_current_mode]}][/dim]")
        else:
            console.print(f"[dim][mode: default — available: {modes_list} / off][/dim]")
    elif arg in _MODE_PROMPTS:
        _current_mode = arg
        console.print(f"[green][mode: {arg}][/green]")
    elif arg in ("off", "none", "reset", "default"):
        _current_mode = ""
        console.print("[dim][mode cleared][/dim]")
    else:
        console.print(f"[yellow][unknown mode — try: {' / '.join(_MODE_PROMPTS)} / off][/yellow]")


def _cmd_model(ctx: _ReplContext, arg: str) -> None:
    global MODEL
    if not arg:
        console.print(f"[dim][model: {MODEL}][/dim]")
    elif arg == "list":
        models = list_models(ctx.client)
        for m in models:
            mark = "  [bold green]<- current[/bold green]" if m == MODEL else ""
            console.print(f"  [cyan]{m}[/cyan]{mark}")
        if not models:
            console.print("[dim][no models found][/dim]")
    else:
        MODEL = arg
        console.print(f"[green][model switched to: {MODEL}][/green]")


def _cmd_index(ctx: _ReplContext, arg: str) -> None:
    cmd_index(Path.cwd(), force=(arg.lower() == "force"))


def _cmd_task(ctx: _ReplContext, arg: str) -> None:
    if not arg:
        console.print("[yellow][usage: /task <goal>][/yellow]")
    else:
        cmd_task(arg, ctx.history, ctx.base_system, ctx.client)


def _cmd_agent(ctx: _ReplContext, arg: str) -> None:
    if not arg:
        console.print("[yellow][usage: /agent <goal>][/yellow]")
    else:
        cmd_agent(arg, ctx.history, ctx.base_system, ctx.client)


def _cmd_git(ctx: _ReplContext, arg: str) -> None:
    sub_parts = arg.split(None, 1)
    sub = sub_parts[0].lower() if sub_parts else ""
    sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""
    if sub == "commit":
        cmd_git_commit(ctx.client, sub_arg)
    elif sub == "pr":
        cmd_git_pr(ctx.client)
    else:
        console.print("[yellow][usage: /git commit [msg] · /git pr][/yellow]")


def _cmd_lsp(ctx: _ReplContext, arg: str) -> None:
    """LSP command handler: /lsp <subcommand> [args...]."""
    global _lsp_client_mod
    parts = arg.split()
    sub = parts[0].lower() if parts else ""
    sub_args = parts[1:] if len(parts) > 1 else []

    def _ensure_lsp():
        global _lsp_client_mod
        if _lsp_client_mod is None:
            import lsp_client as _mod
            _lsp_client_mod = _mod

    _ensure_lsp()

    if sub == "status":
        console.print(_lsp_client_mod.lsp_status())
    elif sub == "diagnose" or sub == "diag":
        if not sub_args:
            console.print("[yellow]usage: /lsp diagnose <file>][/yellow]")
        else:
            console.print(_lsp_client_mod.lsp_diagnostics(sub_args[0]))
    elif sub == "define" or sub == "def":
        if len(sub_args) < 3:
            console.print("[yellow]usage: /lsp define <file> <line> <col>][/yellow]")
        else:
            console.print(_lsp_client_mod.lsp_definition(sub_args[0], int(sub_args[1]), int(sub_args[2])))
    elif sub == "refs" or sub == "references":
        if len(sub_args) < 3:
            console.print("[yellow]usage: /lsp refs <file> <line> <col>][/yellow]")
        else:
            console.print(_lsp_client_mod.lsp_references(sub_args[0], int(sub_args[1]), int(sub_args[2])))
    elif sub == "hover":
        if len(sub_args) < 3:
            console.print("[yellow]usage: /lsp hover <file> <line> <col>][/yellow]")
        else:
            console.print(_lsp_client_mod.lsp_hover(sub_args[0], int(sub_args[1]), int(sub_args[2])))
    elif sub == "symbols" or sub == "sym":
        if not sub_args:
            console.print("[yellow]usage: /lsp symbols <file>][/yellow]")
        else:
            console.print(_lsp_client_mod.lsp_symbols(sub_args[0]))
    elif sub == "rename":
        if len(sub_args) < 4:
            console.print("[yellow]usage: /lsp rename <file> <line> <col> <new_name>][/yellow]")
        else:
            console.print(_lsp_client_mod.lsp_rename(sub_args[0], int(sub_args[1]), int(sub_args[2]), sub_args[3]))
    else:
        console.print("[yellow]usage: /lsp status | diagnose <file> | define <file> <line> <col> | refs <file> <line> <col> | hover <file> <line> <col> | symbols <file> | rename <file> <line> <col> <new_name>][/yellow]")


def _cmd_watch(ctx: _ReplContext, arg: str) -> None:
    global _watch_thread
    if not arg or arg == "list":
        if _watched_files:
            for _wf in _watched_files:
                console.print(f"  [cyan]{_wf}[/cyan]")
        else:
            console.print("[dim][no files watched][/dim]")
    elif arg == "off":
        _watch_stop.set()
        if _watch_thread:
            _watch_thread.join(timeout=3)
        _watched_files.clear()
        _watch_pending.clear()
        _watch_stop.clear()
        _watch_thread = None
        console.print("[dim][watch stopped][/dim]")
    else:
        _wp = _resolve(arg)
        if not _wp.exists():
            console.print(f"[red][file not found: {_wp}][/red]")
        else:
            _watched_files[str(_wp)] = _wp.stat().st_mtime
            console.print(f"[green][watching: {_wp}][/green]")
            if _watch_thread is None or not _watch_thread.is_alive():
                _watch_stop.clear()
                _watch_thread = threading.Thread(
                    target=_watch_worker, args=(_watched_files,), daemon=True
                )
                _watch_thread.start()


def _cmd_preset(ctx: _ReplContext, arg: str) -> None:
    cmd_preset(arg)


def _cmd_params(ctx: _ReplContext, arg: str) -> None:
    cmd_params(arg)


def _cmd_long(ctx: _ReplContext, arg: str) -> None:
    cmd_long(arg)


def _cmd_autosearch(ctx: _ReplContext, arg: str) -> None:
    cmd_autosearch(arg)


def _cmd_rollback(ctx: _ReplContext, arg: str) -> None:
    cmd_rollback()


def _cmd_review(ctx: _ReplContext, arg: str) -> None:
    cmd_review(arg, ctx.history, ctx.base_system, ctx.client)


def _cmd_error(ctx: _ReplContext, arg: str) -> None:
    cmd_error(ctx.history, ctx.base_system, ctx.client)


def _cmd_config(ctx: _ReplContext, arg: str) -> None:
    show_config()


def _cmd_team(ctx: _ReplContext, arg: str) -> None:
    sub_parts = arg.split(None, 2)
    sub = sub_parts[0].lower() if sub_parts else "list"
    sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""
    desc_arg = sub_parts[2].strip() if len(sub_parts) > 2 else ""
    if sub == "create":
        if not sub_arg:
            console.print("[yellow][usage: /team create <name> [description]][/yellow]")
        else:
            cfg = _ct_team_create(sub_arg, desc_arg)
            console.print(f"[green][team '{sub_arg}' created][/green]")
    elif sub == "list" or not sub_arg:
        teams = _ct_team_list()
        if not teams:
            console.print("[dim][no teams — create one with /team create <name>][/dim]")
        else:
            for t in teams:
                cfg = _ct_load_team(t) or {}
                n = len(cfg.get("members", []))
                console.print(f"  [cyan]{t}[/cyan]  [dim]{n} member(s)[/dim]")
    elif sub == "show":
        if not sub_arg:
            console.print("[yellow][usage: /team show <name>][/yellow]")
        else:
            console.print(Markdown(_ct_board_render(sub_arg)))
    elif sub == "join":
        if not sub_arg:
            console.print("[yellow][usage: /team join <team> [agent_name]][/yellow]")
        else:
            agent = desc_arg or "user"
            _ct_team_join(sub_arg, agent)
            console.print(f"[green][joined team '{sub_arg}' as '{agent}'][/green]")
    else:
        console.print("[yellow][usage: /team create|list|show|join][/yellow]")


def _cmd_spawn(ctx: _ReplContext, arg: str) -> None:
    spawn_parts = arg.split(None, 2)
    if len(spawn_parts) < 3:
        console.print("[yellow][usage: /spawn <team> <agent_name> <task>][/yellow]")
    else:
        s_team, s_agent, s_task = spawn_parts
        result = _ct_spawn(s_team.strip(), s_agent.strip(), s_task.strip())
        console.print(f"[green]{result}[/green]")


def _cmd_inbox(ctx: _ReplContext, arg: str) -> None:
    inbox_parts = arg.split(None, 3)
    sub = inbox_parts[0].lower() if inbox_parts else ""
    if sub == "send":
        if len(inbox_parts) < 4:
            console.print("[yellow][usage: /inbox send <team> <to_agent> <message>][/yellow]")
        else:
            _, i_team, i_to, i_msg = inbox_parts[0], inbox_parts[1], inbox_parts[2], inbox_parts[3]
            msg_id = _ct_inbox_send(i_team, i_to, i_msg)
            console.print(f"[green][sent message {msg_id[:8]} to {i_to} in team '{i_team}'][/green]")
    elif sub and sub not in ("receive", "read"):
        i_team = sub
        i_agent = inbox_parts[1] if len(inbox_parts) > 1 else "user"
        msgs = _ct_inbox_receive(i_team, i_agent)
        if not msgs:
            console.print(f"[dim][inbox empty for {i_agent} in team '{i_team}'][/dim]")
        else:
            for m in msgs:
                console.print(Panel(
                    m.get("body", ""),
                    title=f"From: {m.get('from', '?')}",
                    border_style="cyan",
                ))
    else:
        console.print("[yellow][usage: /inbox <team> [agent] | /inbox send <team> <to> <message>][/yellow]")


def _cmd_board(ctx: _ReplContext, arg: str) -> None:
    if not arg:
        teams = _ct_team_list()
        if not teams:
            console.print("[dim][no teams][/dim]")
        else:
            for t in teams:
                console.print(Markdown(_ct_board_render(t)))
                console.print(Rule())
    else:
        console.print(Markdown(_ct_board_render(arg.strip())))


def _cmd_history(ctx: _ReplContext, arg: str) -> None:
    cmd_history(arg, ctx.history)


def _cmd_note(ctx: _ReplContext, arg: str) -> None:
    if not arg:
        console.print("[yellow][usage: /note <text>][/yellow]")
    else:
        ctx.history.append({"role": "user", "content": f"[Note] {arg}"})
        console.print("[dim][note added to history][/dim]")


def _cmd_compact(ctx: _ReplContext, arg: str) -> None:
    ctx.history = cmd_trim(ctx.history, ctx.client)


def _cmd_stats(ctx: _ReplContext, arg: str) -> None:
    cmd_stats(ctx.history)


def _cmd_cleanup(ctx: _ReplContext, arg: str) -> None:
    """Clean up stale teams, backups, and completed tasks."""

    sub = arg.strip().lower() if arg else "all"
    dry_run = sub == "dry-run"
    what = sub if sub in ("teams", "backups", "tasks", "all") else "all"
    if dry_run:
        what = "all"

    removed = 0

    # --- Teams ---
    if what in ("teams", "all"):
        ct_dir = CT_DIR / "teams"
        if ct_dir.is_dir():
            now_ts = time.time()
            max_age = 7 * 86400  # 7 days
            for team_dir in sorted(ct_dir.iterdir()):
                if not team_dir.is_dir():
                    continue
                tasks_dir = team_dir / "tasks"
                has_active = False
                if tasks_dir.is_dir():
                    try:
                        for tf in tasks_dir.iterdir():
                            if tf.suffix == ".json":
                                raw = tf.read_text(encoding="utf-8", errors="replace")
                                td = json.loads(raw)
                                status = td.get("status", "pending").lower()
                                if status not in ("completed", "blocked"):
                                    has_active = True
                                    break
                                # even completed/blocked tasks count as recent if modified within max_age
                                if tf.stat().st_mtime > now_ts - max_age:
                                    has_active = True
                                    break
                    except (json.JSONDecodeError, OSError):
                        has_active = True
                if has_active:
                    continue
                # Remove the team directory
                if not dry_run:
                    try:
                        import shutil
                        shutil.rmtree(team_dir)
                    except Exception:
                        pass
                removed += 1
                if dry_run:
                    console.print(f"  [dim]would remove team:[/dim] {team_dir.name}")

    # --- Backups ---
    if what in ("backups", "all"):
        if BACKUPS_DIR.is_dir():
            now_ts = time.time()
            max_age = 7 * 86400  # 7 days
            for bf in list(BACKUPS_DIR.iterdir()):
                if bf.stat().st_mtime < now_ts - max_age:
                    if not dry_run:
                        try:
                            bf.unlink()
                        except Exception:
                            pass
                    removed += 1
                    if dry_run:
                        console.print(f"  [dim]would remove backup:[/dim] {bf.name}")

    # --- Completed tasks ---
    if what in ("tasks", "all"):
        ct_dir = CT_DIR / "teams"
        if ct_dir.is_dir():
            now_ts = time.time()
            max_age = 3 * 86400  # 3 days for completed tasks
            for team_dir in list(ct_dir.iterdir()):
                tasks_dir = team_dir / "tasks"
                if not tasks_dir.is_dir():
                    continue
                for tf in list(tasks_dir.iterdir()):
                    if tf.suffix != ".json":
                        continue
                    try:
                        raw = tf.read_text(encoding="utf-8", errors="replace")
                        td = json.loads(raw)
                        status = td.get("status", "pending").lower()
                        if status == "completed" and tf.stat().st_mtime < now_ts - max_age:
                            if not dry_run:
                                tf.unlink()
                            removed += 1
                            if dry_run:
                                console.print(f"  [dim]would remove task:[/dim] {team_dir.name}/{tf.name}")
                    except (json.JSONDecodeError, OSError):
                        pass

    if dry_run:
        console.print(f"[dim][dry-run: {removed} item(s) would be removed][/dim]")
    else:
        console.print(f"[dim][cleanup done: {removed} item(s) removed][/dim]")


def _cmd_intel(ctx: _ReplContext, arg: str) -> None:
    global _intel_enabled
    sub = arg.strip().lower() if arg else ""
    if sub == "off":
        _intel_enabled = False
        console.print("[dim][intel crawlers paused — /intel on to resume][/dim]")
    elif sub == "on":
        _intel_enabled = True
        console.print("[dim][intel crawlers resumed][/dim]")
    elif sub.startswith("topics"):
        topics = _intel_load_topics()
        from rich.table import Table as _T
        tbl = _T(title="Intel topics", box=None)
        tbl.add_column("Name", style="cyan")
        tbl.add_column("Query", style="dim")
        tbl.add_column("Last crawled", style="dim")
        for tp in sorted(topics, key=lambda t: -t.get("last_checked", 0)):
            lc = datetime.fromtimestamp(tp["last_checked"]).strftime("%m-%d %H:%M") if tp.get("last_checked") else "never"
            tbl.add_row(tp["name"], tp["query"][:55], lc)
        console.print(tbl)
    elif sub.startswith("add "):
        rest = arg[4:].strip()
        if "|" in rest:
            tname, tquery = rest.split("|", 1)
            tname, tquery = tname.strip(), tquery.strip()
        else:
            tname, tquery = rest, rest + " news today"
        topics = _intel_load_topics()
        if any(t["name"].lower() == tname.lower() for t in topics):
            console.print(f"[yellow][topic '{tname}' already tracked][/yellow]")
        else:
            topics.append({"name": tname, "query": tquery, "last_checked": 0})
            _intel_save_topics(topics)
            console.print(f"[green][added topic '{tname}'][/green]")
    elif sub.startswith("remove "):
        tname = arg[7:].strip()
        topics = _intel_load_topics()
        before = len(topics)
        topics = [t for t in topics if t["name"].lower() != tname.lower()]
        if len(topics) < before:
            _intel_save_topics(topics)
            console.print(f"[green][removed topic '{tname}'][/green]")
        else:
            console.print(f"[yellow][topic '{tname}' not found][/yellow]")
    else:
        feed = intel_get_recent(10)
        if feed:
            console.print(Markdown(feed))
        else:
            console.print("[dim][no intel yet — crawlers are warming up][/dim]")
        status = "[green]running[/green]" if _intel_enabled else "[yellow]paused[/yellow]"
        console.print(f"[dim]Crawlers: {status} · {_INTEL_CRAWLERS} threads · {_INTEL_INTERVAL}s interval · /intel topics · /intel add <name>|<query> · /intel on/off[/dim]")


def _cmd_cd(ctx: _ReplContext, arg: str) -> None:
    if not arg:
        console.print(f"[dim][cwd: {Path.cwd()}][/dim]")
    else:
        try:
            new_cwd = _resolve(arg)
            if not new_cwd.is_dir():
                console.print(f"[red][not a directory: {new_cwd}][/red]")
            else:
                os.chdir(new_cwd)
                _invalidate_git_cache()
                console.print(f"[green][cwd → {new_cwd}][/green]")
        except Exception as e:
            console.print(f"[red][error: {e}][/red]")


def _cmd_unknown(ctx: _ReplContext, directive: str) -> None:
    console.print(f"[yellow][unknown command: {directive} — try /help][/yellow]")


# Dispatch dictionary: command name -> handler function
# Handlers that need special treatment (like /exit raising StopIteration) are handled in the dispatch loop.
_REPL_COMMANDS: dict[str, callable] = {
    "/exit": _cmd_exit,
    "/quit": _cmd_exit,
    "/help": _cmd_help,
    "/clear": _cmd_clear,
    "/retry": _cmd_retry,
    "/run": _cmd_run,
    "/edit": _cmd_edit,
    "/system": _cmd_system,
    "/file": _cmd_file,
    "/project": _cmd_project,
    "/focus": _cmd_focus,
    "/changes": _cmd_changes,
    "/search-sessions": _cmd_search_sessions,
    "/undo": _cmd_undo,
    "/export": _cmd_export,
    "/copy": _cmd_copy,
    "/paste": _cmd_paste,
    "/remember": _cmd_remember,
    "/memory": _cmd_memory,
    "/forget": _cmd_forget,
    "/pin": _cmd_pin,
    "/branch": _cmd_branch,
    "/search": _cmd_search,
    "/save": _cmd_save,
    "/load": _cmd_load,
    "/sessions": _cmd_sessions,
    "/context": _cmd_context,
    "/trim": _cmd_trim,
    "/mode": _cmd_mode,
    "/model": _cmd_model,
    "/index": _cmd_index,
    "/task": _cmd_task,
    "/agent": _cmd_agent,
    "/git": _cmd_git,
    "/lsp": _cmd_lsp,
    "/watch": _cmd_watch,
    "/preset": _cmd_preset,
    "/params": _cmd_params,
    "/long": _cmd_long,
    "/autosearch": _cmd_autosearch,
    "/rollback": _cmd_rollback,
    "/review": _cmd_review,
    "/error": _cmd_error,
    "/config": _cmd_config,
    "/team": _cmd_team,
    "/spawn": _cmd_spawn,
    "/inbox": _cmd_inbox,
    "/board": _cmd_board,
    "/history": _cmd_history,
    "/note": _cmd_note,
    "/compact": _cmd_compact,
    "/stats": _cmd_stats,
    "/cleanup": _cmd_cleanup,
    "/intel": _cmd_intel,
    "/cd": _cmd_cd,
}


def _dispatch_command(ctx: _ReplContext, directive: str, arg: str) -> bool:
    """Dispatch a /command to its handler.

    Returns True if the REPL loop should break (e.g. /exit).
    """
    handler = _REPL_COMMANDS.get(directive)
    if handler:
        try:
            handler(ctx, arg)
        except StopIteration:
            return True
    else:
        _cmd_unknown(ctx, directive)
    return False


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main entry point for qwen-cli.

    # Feature 7+8: Auto-index project symbols + start LSP watch on startup
    try:
        global _lsp_client_mod
        if _lsp_client_mod is None:
            import lsp_client as _mod
            _lsp_client_mod = _mod
        code_files = []
        for ext in (".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".cpp", ".c"):
            for f in Path.cwd().rglob(f"*{ext}"):
                rel = str(f.relative_to(Path.cwd()))
                if not rel.startswith(".") and "/." not in rel:
                    code_files.append(str(f))
        if code_files:
            _lsp_client_mod.lsp_index_symbols(code_files)
            _lsp_client_mod.lsp_start_watch(code_files)
    except Exception:
        pass


    Initializes the client, prints the welcome banner, and enters the REPL loop.
    Command dispatch is handled by _REPL_COMMANDS dispatch dictionary.
    """
    global MODEL, _backup_stack, _last_user_input, _branches, _turn_count, _current_mode, _session_title, _watch_thread, _cli_client, _real_ctx_tokens, _session_start

    _session_start = time.monotonic()
    client = make_client()
    _cli_client = client

    # --task flag: spawn as autonomous worker with a pre-assigned task
    _auto_task: str | None = None
    _cli_args = sys.argv[1:]
    if "--task" in _cli_args:
        _idx = _cli_args.index("--task")
        if _idx + 1 < len(_cli_args):
            _auto_task = _cli_args[_idx + 1]
            sys.argv = [sys.argv[0]] + _cli_args[:_idx] + _cli_args[_idx + 2:]

    if not sys.stdin.isatty():
        run_piped(client)
        return

    # Auto-task mode: skip REPL setup, run agent immediately
    if _auto_task:
        base_system = BASE_SYSTEM
        history: list[dict] = []
        _auto_task = expand_at_refs(_auto_task)
        cmd_agent(_auto_task, history, base_system, client)
        return

    _setup_tab_completion()
    _make_pt_session()

    base_system = BASE_SYSTEM
    history: list[dict] = []
    ctx = _ReplContext(history, base_system, client)

    pt_note = "prompt_toolkit" if _HAS_PT else "readline"
    _backend_label = f"[cyan]{BASE_URL}[/cyan]" if ACTIVE_BACKEND == "llama.cpp" else "[bold yellow]OpenAI (cloud fallback)[/bold yellow]"
    console.print(Panel(
        f"[bold]Model[/bold]   [cyan]{MODEL}[/cyan]  [dim]({_active_preset} preset)[/dim]\n"
        f"[bold]Server[/bold]  {_backend_label}\n"
        f"[bold]Search[/bold]  [dim]{AUTO_SEARCH_MODE} (/autosearch to change)[/dim]\n"
        f"[bold]Data[/bold]    [cyan]{DATA_DIR}[/cyan]\n\n"
        f"[dim]@file · @file::symbol · /team · /spawn · /inbox · /board · /help · Ctrl+C cancels · /exit[/dim]\n"
        f"[dim]Input: {pt_note}{'  (Ctrl+R history, Tab completion)' if _HAS_PT else ''}  · {_INTEL_CRAWLERS} intel crawlers (/intel)[/dim]",
        title="[bold green]Qwen CLI[/bold green]",
        border_style="green",
    ))

    # Start background intelligence crawlers
    start_intel_crawlers()

    cwd = Path.cwd()
    if any((cwd / m).exists() for m in KEY_FILES | {".git"}):
        console.print("[dim]Project detected — loading context automatically (/project to reload)[/dim]")
        load_project_context("", history)

    while True:
        # Inject any watched-file updates that arrived since last turn
        if _watch_pending:
            for _wp in _watch_pending[:]:
                try:
                    _p = Path(_wp)
                    _lang = LANG_MAP.get(_p.suffix.lower(), "")
                    _content = _p.read_text(encoding="utf-8", errors="replace")
                    history.append({"role": "user", "content": f"[File updated: {_p.name}]\n```{_lang}\n{_content[:10_000]}\n```"})
                    console.print(f"[dim cyan]  [watch] {_p.name} updated — injected[/dim cyan]")
                except Exception as _we:
                    console.print(f"[yellow]  [watch] error reading {_wp}: {_we}[/yellow]")
            _watch_pending.clear()

        try:
            user_input = read_input().strip()
        except KeyboardInterrupt:
            console.print()
            continue
        except EOFError:
            console.print()
            if history:
                save_session(history, base_system)
            console.print("[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        # Slash command dispatch
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            directive = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if _dispatch_command(ctx, directive, arg):
                break
            continue

        # -- prefix: skip tool use for this turn
        no_tools = False
        if user_input.startswith("-- "):
            no_tools = True
            user_input = user_input[3:].strip()

        # Normal chat turn — expand @file refs before sending
        user_input = expand_at_refs(user_input)
        _last_user_input = user_input

        _run_turn_and_handle_reply(ctx, user_input, allow_tools=not no_tools)

def _print_turn_footer(elapsed: float) -> None:
    """Internal helper: print turn footer."""
    tok = _last_turn_tokens
    p, c = tok.get("prompt", 0), tok.get("completion", 0)
    if p or c:
        console.print(f"[dim]  {elapsed:.1f}s · {p:,} in / {c:,} out tok[/dim]")
    else:
        console.print(f"[dim]  {elapsed:.1f}s[/dim]")


# Cleanup watch thread on exit
def _cleanup_watch():
    """Internal helper: cleanup watch."""
    global _watch_thread
    _watch_stop.set()
    if _watch_thread and _watch_thread.is_alive():
        _watch_thread.join(timeout=2)
    _watch_stop.clear()
    _watched_files.clear()
    _watch_pending.clear()
    _watch_thread = None

atexit.register(_cleanup_watch)

if __name__ == "__main__":
    main()
