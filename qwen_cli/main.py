#!/usr/bin/env python3
"""Qwen CLI v8 — <think> strip, parallel tools, result cap, -- no-tools, auto-compact, auto-save, health check, JSON retry."""

import atexit
import concurrent.futures
import difflib
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import contextlib
import logging as _logging
import logging.handlers as _logging_handlers
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table


def _get_openai():
    """Lazy-load openai module (heavy import: httpx, pydantic, etc.)."""
    import openai as _openai_mod
    return _openai_mod.OpenAI


_logging.basicConfig(
    filename=str(Path.home() / ".qwen-cli" / "qwen.log"),
    level=_logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
_logger = _logging.getLogger("qwen")
try:
    _h = _logging_handlers.RotatingFileHandler(
        str(Path.home() / ".qwen-cli" / "qwen.log"), maxBytes=1_048_576, backupCount=3, encoding="utf-8"
    )
    _logger.handlers = [_h]
except Exception:
    _logger.handlers = []  # logging unavailable, continue without file logging

# ---------------------------------------------------------------------------
# Config — constants from config.toml / env vars (qwen_cli/core/config.py)
# ---------------------------------------------------------------------------

import qwen_cli.core.config as _config  # noqa: E402
from qwen_cli.core.config import (  # noqa: E402
    _TOOL_RETRY_BASE,
    _TOOL_RETRY_MAX,
    _TOOL_RETRYABLE_TOOLS,
    _TOOL_TIMEOUT_FAST,
    _TOOL_TIMEOUT_NET,
    _TOOL_TIMEOUT_SLOW,
    AUTOSAVE_FILE,
    AUX_LLM_TIMEOUT,
    BACKUPS_DIR,
    BASE_URL,
    BRAVE_API_KEY,
    CONFIG_FILE,
    CT_DIR,  # noqa: F401 — accessed via _main. in commands.py
    DATA_DIR,
    DEFAULT_EDITOR,
    EXPORTS_DIR,
    FALLBACK_MODEL,
    GOOGLE_API_KEY,
    GOOGLE_CSE_ID,
    HANDOFF_FILE,
    HISTORY_FILE,
    INTEL_DIR,
    INTEL_FEED,
    INTEL_QUEUE,
    INTEL_TOPICS,
    MAX_AUTO_CONTINUE,
    MAX_TOOL_DEPTH,
    MEMORY_FILE,
    MODEL,
    OPENAI_API_KEY,
    PINS_FILE,
    SAMPLING_PRESETS,
    SESSIONS_DIR,
    TOKEN_LIMIT,
    TOOL_RESULT_LIMIT,
)

_CFG = _config._load_config()
ACTIVE_BACKEND = "llama.cpp"  # updated at startup if fallback activates
AUTO_SEARCH_MODE = _config.AUTO_SEARCH_MODE

from qwen_cli.core.indexer import (  # noqa: E402
    IGNORE_DIRS,
    _extract_symbol_from_file,
    _format_symbol_index,
    _load_symbol_index,
    _qwen_ignored,
    _save_symbol_index,
    build_symbol_index,
    load_qwenignore,
)

# Shared tool implementations — configured once after paths are defined
from qwen_cli.tools import shared as _qt  # noqa: E402

_qt.GOOGLE_API_KEY = GOOGLE_API_KEY
_qt.GOOGLE_CSE_ID = GOOGLE_CSE_ID
_qt.BRAVE_API_KEY = BRAVE_API_KEY
_qt.BACKUPS_DIR = BACKUPS_DIR
_apply_diff, do_web_search, do_fetch_url, do_get_video_transcript, do_search_news, _html_to_text, presearch_decision = (
    _qt._apply_diff,
    _qt.do_web_search,
    _qt.do_fetch_url,
    _qt.do_get_video_transcript,
    _qt.do_search_news,
    _qt._html_to_text,
    _qt.presearch_decision,
)

from qwen_cli.core.repl import (  # noqa: E402
    _repl_loop,
    _repl_setup,
    _run_turn_and_handle_reply,  # noqa: F401 — accessed via _main. in commands.py
    _watch_worker,  # noqa: F401 — accessed via _main. in commands.py
)
from qwen_cli.core.stream import (  # noqa: E402
    TOOLS,
    _live_updater,
    _short_args,
    _strip_think,
    stream_once,
)
from qwen_cli.tools.browser import do_browser_action, do_fetch_rendered  # noqa: E402

console = Console(force_terminal=True, legacy_windows=False)

_stream_usage_supported: bool = True  # set to False on first 4xx from usage endpoint

try:
    import readline as _rl

    with contextlib.suppress(FileNotFoundError):
        _rl.read_history_file(str(HISTORY_FILE))
    atexit.register(_rl.write_history_file, str(HISTORY_FILE))
except ImportError:
    _rl = None  # type: ignore

try:
    import importlib.util as _importlib_util

    _HAS_PT = _importlib_util.find_spec("prompt_toolkit") is not None
except Exception:
    _HAS_PT = False

_pt_session = None

_cli_client: object | None = None  # set in main(); allows tool fns to make direct API calls

_backup_stack: list[dict] = []  # stack of backups; /undo pops the most recent
_MAX_BACKUP_STACK = 10

# LSP client module — imported lazily
_lsp_client_mod: object | None = None


def _get_lsp():
    """Lazy-load and cache the lsp_client module."""
    global _lsp_client_mod
    if _lsp_client_mod is None:
        from qwen_cli.tools import lsp as _mod

        _lsp_client_mod = _mod
    return _lsp_client_mod


_last_user_input: str = ""
_branches: dict[str, list] = {}  # in-memory named conversation snapshots
_last_turn_tokens: dict = {"prompt": 0, "completion": 0}
_real_ctx_tokens: int = 0  # real prompt_tokens from last API response — used for compact check
_turn_count: int = 0
_session_start: float = 0.0
_current_mode: str = ""
_session_title: str = ""
_auto_memory_count: int = 0
_BG_LLM_SEM = threading.Semaphore(2)  # at most 2 background LLM calls at a time
_main_llm_busy = False  # True while run_turn() holds the LLM slot
_intel_memory_written: dict[str, str] = {}  # topic_name → date; prevents duplicate entries
_cached_index: dict | None = None
_main_llm_busy_lock = threading.Lock()  # Protects reads/writes of _main_llm_busy
_cached_index_root: Path | None = None
_watched_files: dict[str, float] = {}  # path → last mtime
_watch_pending: list[str] = []  # paths changed since last turn
_watch_stop = threading.Event()
_watch_thread: threading.Thread | None = None
_focus_set: list[str] = []  # files loaded via /focus this session
_session_changes: dict[str, str] = {}  # path → original content before first edit
_tool_call_retry_log: dict[int, list] = {}  # depth → list of (tool, error) for retry context

# --- Enforced verification (agent mode) -------------------------------------
# Tools that mutate files; after any of these, /agent will not accept AGENT_DONE
# until at least one verifying tool has run.
_MUTATING_FILE_TOOLS = frozenset({"patch_file", "write_file", "move_file", "delete_file"})
# Tools that count as verification (checking real state after a change).
_VERIFYING_TOOLS = frozenset(
    {"read_file", "run_command", "run_script", "lsp_query", "search_files", "find_files", "list_directory"}
)
_last_turn_tool_names: list[str] = []  # tool names executed during the current run_turn, in order

_model_params: dict = {}  # runtime overrides — layered on top of active preset

_active_preset: str = _CFG.get("preset", "thinking") if _CFG.get("preset") in SAMPLING_PRESETS else "thinking"

# High-output mode: Qwen3's recommended 81,920-token ceiling for hard math/coding
# competition problems (vs. 32,768 for general use). Toggled at runtime via /long.
_LONG_OUTPUT = 81920
_long_mode = False
_TOKEN_LIMIT_BASE = TOKEN_LIMIT  # input budget to restore when /long is turned off
_memory_lock = threading.Lock()  # serializes all memory.md read-modify-write
_intel_stop = threading.Event()
_intel_lock = threading.Lock()
_intel_enabled = threading.Event()  # thread-safe flag for intel crawlers
_INTEL_INTERVAL = 240  # seconds between each crawler's crawl cycles
_INTEL_INJECT_N = 6  # recent feed entries injected into system prompt
_INTEL_CRAWLERS = 3  # number of parallel background browser threads

# Module-level thread pool for parallel tool execution — avoids per-turn pool creation
_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="qwen-tool")


def _get_pool() -> concurrent.futures.ThreadPoolExecutor:
    return _POOL


atexit.register(lambda: _POOL.shutdown(wait=False))


def _intel_default_topics() -> list[dict]:
    """Return the default set of topics for the Live Intelligence background crawlers."""
    year = datetime.now().year
    return [
        {"name": "AI & LLM news", "query": "latest AI LLM model releases news today", "last_checked": 0},
        {"name": "Python ecosystem", "query": f"Python new libraries tools releases {year}", "last_checked": 0},
        {
            "name": "Security vulnerabilities",
            "query": "critical security vulnerabilities CVE this week",
            "last_checked": 0,
        },
        {"name": "Tech industry news", "query": "technology industry news today", "last_checked": 0},
        {"name": "Open source trending", "query": "trending open source projects GitHub today", "last_checked": 0},
        {"name": "Developer APIs", "query": f"new developer APIs web services released {year}", "last_checked": 0},
    ]


_INTEL_DEFAULT_TOPICS: list[dict] = _intel_default_topics()

MEMORY_CURATE_INTERVAL = 10  # consolidate memory.md every N auto-extractions


_PARALLEL_TOOLS = frozenset(
    {
        "web_search",
        "search_news",
        "fetch_url",
        "read_file",
        "list_directory",
        "find_files",
        "search_files",
        "describe_image",
        "get_video_transcript",
        "team_task_list",
        "team_inbox_receive",
        "lsp_query",
    }
)
_STATEFUL_TOOLS = frozenset(
    {
        "browser_action",
        "run_command",
        "run_script",
        "write_file",
        "patch_file",
        "move_file",
        "delete_file",
        "team_task_add",
        "team_task_update",
        "team_inbox_send",
        "team_spawn_agent",
    }
)
_SERIAL_BY_DOMAIN = frozenset({"fetch_url", "fetch_rendered", "describe_image", "get_video_transcript"})


def _extract_domain(url: str) -> str:
    """Extract domain from a URL for contention detection."""
    m = re.search(r"https?://([^/:\d]+)", url)
    return m.group(1) if m else ""


def _classify_tool_batch(tool_calls: list, parsed_args: list) -> list[list[int]]:
    """Group tool calls into batches respecting:
    1. Stateful tools run alone (never parallelized)
    2. Fetches to the same domain are serialized (rate-limit safety)
    3. All other parallel-safe tools can run together.

    Returns a list of batches (each batch is a list of indices).
    """
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


# Tool-set constants used by the dispatch engine
_SUMMARIZE_TOOLS = frozenset(
    {"web_search", "search_news", "fetch_url", "fetch_rendered", "describe_image", "get_video_transcript"}
)
_RUNNABLE_LANGS = {"bash", "sh", "shell", "powershell", "ps1", "python", "py", "cmd", "batch", "bat"}

_PLAN_RE = re.compile(r"^\s*\d+[\.\)]\s+[^\n]+", re.MULTILINE)
_HEDGE_RE = re.compile(
    r"\b(?:i(?:'m| am) not (?:sure|certain)|i (?:think|believe|suspect)|"
    r"might|may|could|probably|possibly|perhaps|unclear|uncertain|"
    r"i don't know|not certain|i'm unsure)\b",
    re.IGNORECASE,
)

_MODE_PROMPTS: dict[str, str] = {
    "code": "Mode: code — write minimal, correct, idiomatic code; no explanations unless asked; prefer patch_file over write_file; show diffs.",
    "debug": "Mode: debug — trace root causes systematically; use run_command to check live state; read error output before suggesting fixes.",
    "explain": "Mode: explain — explain clearly with concrete examples; break complex ideas into steps; match depth to the question.",
    "creative": "Mode: creative — explore unusual approaches; don't be constrained by convention; think expansively before converging.",
}

_CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)

_DANGEROUS_CMD_RE = re.compile(
    r"\b(rm\s+-[^-]*r|"
    r"rmdir\s+/s|rd\s+/s|del\s+/[a-z/\s]*[sqf]\b|"
    r"(remove-item|ri)\b[^|;\n]*-rec|"
    r"format\s+[a-z]:|"
    r"git\s+reset\s+--hard|git\s+clean\s+-[^-]*f|"
    r"drop\s+table|drop\s+database|truncate\s+table|"
    r"dd\s+if=|mkfs[.\s]|shred\s+|"
    r"sudo\s+rm|sudo\s+mkfs|sudo\s+dd|sudo\s+chmod\s+777|"
    r"chmod\s+777|"
    r"no-preserve-root|"
    r"shutil\.rmtree|os\.remove|os\.unlink|os\.rmdir|"
    r"invoke-expression|\biex\b|downloadstring)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Team coordination layer  (qwen_cli/tools/team.py)
# ---------------------------------------------------------------------------

from qwen_cli.tools.team import (  # noqa: E402, F401  — re-exported for tests / qwen_cli package
    _ct_atomic_write,
    _ct_board_render,
    _ct_inbox_dir,
    _ct_inbox_receive,
    _ct_inbox_send,
    _ct_load_team,
    _ct_now,
    _ct_save_team,
    _ct_spawn,
    _ct_task_add,
    _ct_task_list,
    _ct_task_update,
    _ct_tasks_dir,
    _ct_team_create,
    _ct_team_dir,
    _ct_team_join,
    _ct_team_list,
    do_team_board,
    do_team_list,
    do_team_task_add,
)

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
    "rather than guessing. It is always better to say 'let me check' than to state something wrong confidently.\n\n"
    "WORKING DISCIPLINE — how to execute any non-trivial task:\n"
    "1. PLAN: state a short plan (2-5 steps) before acting. Revise it if you learn something that changes it.\n"
    "2. GROUND: read a file before editing it; check real state with run_command instead of assuming. "
    "Never edit code you have not read this session.\n"
    "3. ACT: make the smallest change that solves the problem. One logical change at a time.\n"
    "4. VERIFY: after every change, prove it worked — run the tests, run the code, re-read the file, "
    "or check the output. A step is not done until verified. Never claim success without evidence.\n"
    "5. REPORT: end with 1-3 sentences on what changed and how you verified it. No filler, "
    "no restating the plan, no apologizing.\n"
    "If a step fails: read the actual error, form a hypothesis, fix, re-verify. "
    "If still stuck after 3 distinct attempts, stop and tell the user what you tried, what you observed, "
    "and what you would try next — do not loop on the same failing approach.\n"
    "Style: be direct and concise. Lead with the answer, then supporting detail. "
    "Do not narrate every tool call. Do not pad responses with caveats or summaries of what you are about to do."
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
    """Load Pins."""
    if not PINS_FILE.exists():
        return []
    try:
        return json.loads(PINS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_pins(pins: list[str]) -> None:
    """Save Pins."""
    PINS_FILE.write_text(json.dumps(pins, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Memory & system prompt
# ---------------------------------------------------------------------------


def load_memory() -> str:
    """Load Memory."""
    return MEMORY_FILE.read_text(encoding="utf-8").strip() if MEMORY_FILE.exists() else ""


def save_memory(text: str) -> None:
    """Save Memory."""
    MEMORY_FILE.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Symbol index helpers
# ---------------------------------------------------------------------------


def _get_index(root: Path) -> dict | None:
    """Return in-memory cached index, falling back to disk, then None."""
    global _cached_index, _cached_index_root
    if _cached_index_root == root and _cached_index is not None:
        return _cached_index
    _cached_index = _load_symbol_index(root)
    _cached_index_root = root
    return _cached_index


def cmd_index(root: Path, force: bool = False) -> dict:
    """Build or load the project symbol index; update in-memory cache."""
    global _cached_index, _cached_index_root
    if not force:
        cached = _load_symbol_index(root)
        if cached:
            sym_count = sum(len(s.get("functions", [])) + len(s.get("classes", [])) for s in cached.values())
            console.print(f"[dim][index: {len(cached)} files, {sym_count} symbols — '/index force' to rebuild][/dim]")
            _cached_index, _cached_index_root = cached, root
            return cached
    console.print(f"[dim]Indexing {root}...[/dim]")
    index = build_symbol_index(root)
    _save_symbol_index(root, index)
    _cached_index, _cached_index_root = index, root
    sym_count = sum(len(s.get("functions", [])) + len(s.get("classes", [])) for s in index.values())
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
            _logger.debug("Could not read pyproject.toml in %s", root)
    if list(root.glob("test_*.py")) or (root / "tests").is_dir():
        return "pytest"
    if (root / "package.json").exists():
        try:
            pkg = json.loads((root / "package.json").read_text(encoding="utf-8"))
            if "test" in pkg.get("scripts", {}):
                return "npm test"
        except Exception:
            _logger.debug("Could not read package.json in %s", root)
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
_GIT_CTX_CACHE: dict[str, tuple[float, str]] = {}  # cwd -> (monotonic_ts, context)
_GIT_CTX_TTL = 5.0


def _invalidate_git_cache() -> None:
    """Internal helper: invalidate git cache."""
    _GIT_CTX_CACHE.clear()


def get_git_context() -> str:
    """Get Git Context."""
    key = str(Path.cwd())
    now = time.monotonic()
    cached = _GIT_CTX_CACHE.get(key)
    if cached is not None and now - cached[0] < _GIT_CTX_TTL:
        return cached[1]
    ctx = _compute_git_context()
    _GIT_CTX_CACHE[key] = (now, ctx)
    return ctx


def _compute_git_context() -> str:
    try:
        result = subprocess.run(
            ["git", "status", "-b", "--porcelain"],
            capture_output=True, text=True, timeout=5, encoding="utf-8",
        )
        if result.returncode != 0:
            return ""
        lines = result.stdout.strip().splitlines()
        parts = []
        branch_line = lines[0] if lines else ""
        if branch_line.startswith("## "):
            branch = branch_line[3:].split("...")[0]
            parts.append(f"Branch: {branch}")
        status_lines = [line for line in lines[1:] if line.strip()] if len(lines) > 1 else []
        if status_lines:
            parts.append("Status:\n" + "\n".join(status_lines))
        diff_stat = subprocess.run(
            ["git", "diff", "--stat", "HEAD"],
            capture_output=True, text=True, timeout=5, encoding="utf-8",
        ).stdout.strip()
        if diff_stat:
            parts.append(f"Diff stat:\n{diff_stat}")
        return "\n".join(parts)
    except Exception:
        return ""


def load_project_system(cwd: Path) -> str:
    """Load Project System."""
    f = cwd / ".qwen-system.md"
    return f.read_text(encoding="utf-8").strip() if f.exists() else ""


def build_system_prompt(base: str) -> str:
    """Build System Prompt."""
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
        parts += ["", "=== Live Intelligence (background web feed) ===", intel, "=== End Live Intelligence ==="]
    pins = load_pins()
    if pins:
        parts += ["", "=== Pinned Notes ==="]
        parts += [f"- {p}" for p in pins]
        parts += ["=== End Pinned Notes ==="]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# @file expansion
# ---------------------------------------------------------------------------

_AT_REF_RE = re.compile(r"@([\S]+)")


def expand_at_refs(text: str) -> str:
    """Expand @path and @path::symbol tokens in message text."""
    injections: list[str] = []
    for m in _AT_REF_RE.finditer(text):
        raw = m.group(1)
        symbol: str | None = None
        if "::" in raw:
            file_part, sym_raw = raw.split("::", 1)
            raw = file_part.rstrip(".,;:)\"'")
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
    "/help",
    "/clear",
    "/retry",
    "/run",
    "/edit",
    "/system",
    "/file",
    "/project",
    "/focus",
    "/review",
    "/undo",
    "/export",
    "/copy",
    "/paste",
    "/remember",
    "/memory",
    "/forget",
    "/pin",
    "/branch",
    "/search",
    "/search-sessions",
    "/save",
    "/load",
    "/sessions",
    "/trim",
    "/compact",
    "/context",
    "/model",
    "/mode",
    "/preset",
    "/params",
    "/long",
    "/autosearch",
    "/agent",
    "/task",
    "/index",
    "/git",
    "/watch",
    "/changes",
    "/rollback",
    "/error",
    "/config",
    "/cd",
    "/history",
    "/note",
    "/stats",
    "/exit",
    "/quit",
    "/team",
    "/spawn",
    "/inbox",
    "/board",
    "/intel",
]


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
            user_msgs = [
                m
                for m in history
                if m.get("role") == "user" and not (m.get("content") or "").startswith("# Project Context")
            ]
            topic = (user_msgs[0].get("content") or "")[:60].replace("\n", " ") if user_msgs else ""
            turns = sum(1 for m in history if m.get("role") == "assistant")
            meta = {"topic": topic, "turns": turns, "saved_at": ""}
        return meta
    except Exception:
        return {"topic": "", "turns": 0, "saved_at": ""}


def save_session(history: list, system_prompt: str, name: str | None = None) -> None:
    """Save Session."""
    user_msgs = [
        m for m in history if m.get("role") == "user" and not (m.get("content") or "").startswith("# Project Context")
    ]
    topic = (user_msgs[0].get("content") or "")[:80].replace("\n", " ") if user_msgs else ""
    turns = sum(1 for m in history if m.get("role") == "assistant")
    fname = (name or datetime.now().strftime("%Y%m%d_%H%M%S")) + ".json"
    path = SESSIONS_DIR / fname
    data = {
        "meta": {"topic": topic, "turns": turns, "saved_at": datetime.now().isoformat()},
        "system_prompt": system_prompt,
        "history": history,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    console.print(f"[dim][saved: {path.name}  ({turns} turn{'s' if turns != 1 else ''})][/dim]")


def _silent_autosave(history: list, system_prompt: str) -> None:
    """Rolling checkpoint to autosave.json — no console output, never raises."""
    try:
        user_msgs = [
            m
            for m in history
            if m.get("role") == "user" and not (m.get("content") or "").startswith("# Project Context")
        ]
        raw_topic = (user_msgs[0].get("content") or "")[:80].replace("\n", " ") if user_msgs else ""
        topic = _session_title or raw_topic
        turns = sum(1 for m in history if m.get("role") == "assistant")
        data = {
            "meta": {"topic": topic, "turns": turns, "saved_at": datetime.now().isoformat()},
            "system_prompt": system_prompt,
            "history": history,
        }
        AUTOSAVE_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        _logger.exception("Autosave failed — session data may be lost on crash")


HANDOFF_PROMPT_TEMPLATE = """You are resuming a session that ran out of context window. Here is what happened:

{summary}

Use this context to pick up where you left off. Do NOT ask the user to repeat themselves. Check memory.md for additional persistent facts."""


def _generate_handoff(client, history, base_system) -> str:
    """Generate a compact handoff summary for session resumption."""
    try:
        chat = [m for m in history if m.get("role") in ("user", "assistant")]
        recent = chat[-16:]
        recent_text = "\n".join(f"{m['role'].upper()}: {(m.get('content') or '')[:800]}" for m in recent)
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a session summarizer. Produce a handoff document for an AI assistant "
                    "resuming this conversation. Include: (1) what the session goal is, "
                    "(2) what has been accomplished, (3) what files were created/modified, "
                    "(4) the exact task being worked on and the immediate next step, "
                    "(5) any open questions or blockers. Be precise. Keep it under 800 words."
                ),
            },
            {"role": "user", "content": f"Summarize this session for handoff:\n\n{recent_text}"},
        ]
        summary, _, _ = stream_once(client, prompt, use_tools=False)
        return summary or "(summarization failed - see autosave for raw history)"
    except Exception as e:
        return f"(handoff summary failed: {e})"


def _write_handoff(summary, history) -> None:
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
        _logger.exception("Overflow handoff write failed — session resumption may be degraded")


def _save_exit_handoff(history) -> None:
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
            "summary": (
                f"Session had {sum(1 for m in history if m.get('role') == 'assistant')} turns. "
                f"Last user: {last_user[:200]}. Last assistant: {last_assistant[:200]}"
            ),
            "last_user_message": last_user,
            "last_assistant_message": last_assistant,
            "turns": sum(1 for m in history if m.get("role") == "assistant"),
            "timestamp": datetime.now().isoformat(),
        }
        HANDOFF_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        _logger.exception("Exit handoff write failed — next session will lack context")


def _consume_handoff() -> dict | None:
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
    """Load Session."""
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
    """List Sessions."""
    files = sorted(SESSIONS_DIR.glob("*.json"), reverse=True)
    if not files:
        console.print("[dim][no saved sessions][/dim]")
        return
    t = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
    t.add_column("Name", style="cyan", no_wrap=True)
    t.add_column("Turns", justify="right", style="dim")
    t.add_column("Topic", style="white")
    t.add_column("Saved", style="dim", no_wrap=True)
    for f in files:
        meta = _session_meta(f)
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
                diff = list(
                    difflib.unified_diff(
                        original.splitlines(keepends=True),
                        current.splitlines(keepends=True),
                        fromfile=f"a/{p.name}",
                        tofile=f"b/{p.name}",
                        lineterm="",
                    )
                )
                if diff:
                    console.print(f"\n[bold]{p.name}[/bold]  [dim]{path_str}[/dim]")
                    preview = "".join(diff[:100])
                    console.print(Syntax(preview, "diff", theme="monokai"))
                    if len(diff) > 100:
                        console.print(f"[dim]  ... ({len(diff) - 100} more lines)[/dim]")
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
                orig_lines = len(original.splitlines())
                current_lines = len(current.splitlines())
                delta = current_lines - orig_lines
                sign = "+" if delta >= 0 else ""
                console.print(
                    f"  [cyan]{p.name}[/cyan]  [dim]{sign}{delta} lines[/dim]  [dim]{path_str}[/dim]",
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
        if u_text.startswith(("# Project Context", "[File:", "File:")):
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
    files = sorted(SESSIONS_DIR.glob("*.json"), reverse=True)
    matches: list[tuple[str, str, str, str]] = []

    for f in files:
        if f.name == "autosave.json":
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            meta = data.get("meta", {})
            topic = meta.get("topic", "")
            hist = data.get("history", [])
            for msg in hist:
                content = msg.get("content") or ""
                if q_lower in content.lower():
                    idx = content.lower().find(q_lower)
                    start = max(0, idx - 40)
                    excerpt = content[start : idx + 120].replace("\n", " ").strip()
                    matches.append((f.stem, topic[:40], msg.get("role", ""), excerpt[:120]))
                    break
        except Exception:
            _logger.debug("Skipping unreadable session file %s", f)
            continue

    if not matches:
        console.print(f"[dim][no sessions matching '{query}'][/dim]")
        return

    console.print(f"[bold]{len(matches)} session(s) matching '{query}':[/bold]\n")
    t = Table(show_header=True, header_style="bold dim", box=None, padding=(0, 2))
    t.add_column("Session", style="cyan", no_wrap=True, max_width=24)
    t.add_column("Topic", style="dim", max_width=30)
    t.add_column("Role", style="dim", no_wrap=True)
    t.add_column("Excerpt", style="white")
    for stem, topic, role, excerpt in matches[:20]:
        t.add_row(stem, topic, role, excerpt)
    console.print(t)
    if len(matches) > 20:
        console.print(f"[dim]  ... and {len(matches) - 20} more[/dim]")
    console.print("\n[dim]Load with: /load <session-name>[/dim]")


def cmd_params(arg: str) -> None:
    """Get or set runtime model parameters (temperature, top_p, max_tokens, presence_penalty)."""
    global _model_params, _active_preset
    _param_aliases = {"temp": "temperature", "topp": "top_p", "max": "max_tokens", "pp": "presence_penalty"}
    _valid = ("temperature", "top_p", "max_tokens", "presence_penalty")
    if not arg or arg in ("list", "show", ""):
        preset = SAMPLING_PRESETS.get(_active_preset, {})
        console.print(f"[bold]Preset:[/bold] [cyan]{_active_preset}[/cyan]  (thinking · code · instruct)")
        console.print(
            f"  temp={preset.get('temperature')}  top_p={preset.get('top_p')}  "
            f"presence_penalty={preset.get('presence_penalty')}  max_tokens={preset.get('max_tokens')}"
        )
        eb = preset.get("extra_body", {})
        console.print(
            f"  top_k={eb.get('top_k')}  min_p={eb.get('min_p')}  "
            f"repeat_penalty={eb.get('repeat_penalty')}  "
            f"preserve_thinking={eb.get('chat_template_kwargs', {}).get('preserve_thinking', False)}"
        )
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
    name = _param_aliases.get(name, name)
    if name not in _valid:
        console.print(f"[yellow][unknown param: {name} — valid: {', '.join(_valid)}][/yellow]")
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
        want = not _long_mode  # bare /long toggles
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
            f"(input budget → ~{TOKEN_LIMIT:,}). Use for hard math/coding; [dim]/long off[/dim] to restore.",
        )
    else:
        _model_params.pop("max_tokens", None)
        TOKEN_LIMIT = _TOKEN_LIMIT_BASE
        _long_mode = False
        console.print(
            f"[green][long mode OFF][/green] output restored to preset default; input budget → ~{TOKEN_LIMIT:,}.",
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
        _logger.exception("Failed to persist auto_search setting to config.toml")


def cmd_preset(arg: str) -> None:
    """Switch between Qwen3.6 official sampling presets."""
    global _active_preset
    arg = arg.strip().lower()
    aliases = {
        "thinking_code": "code",
        "think": "thinking",
        "tc": "code",
        "i": "instruct",
        "t": "thinking",
        "c": "code",
    }
    name = aliases.get(arg, arg)
    if not arg:
        console.print(f"[bold]Active preset:[/bold] [cyan]{_active_preset}[/cyan]")
        for n, p in SAMPLING_PRESETS.items():
            marker = "[green]✓[/green]" if n == _active_preset else " "
            eb = p.get("extra_body", {})
            pt = "[dim]preserve_thinking[/dim]" if eb.get("chat_template_kwargs", {}).get("preserve_thinking") else ""
            console.print(
                f"  {marker} [bold]{n}[/bold]  temp={p['temperature']}  top_p={p['top_p']}  pp={p['presence_penalty']}  {pt}"
            )
        console.print("[dim]  /preset thinking · /preset code · /preset instruct[/dim]")
        return
    if name not in SAMPLING_PRESETS:
        console.print(f"[yellow][unknown preset '{arg}' — use: {' · '.join(SAMPLING_PRESETS)}][/yellow]")
        return
    _active_preset = name
    p = SAMPLING_PRESETS[name]
    eb = p.get("extra_body", {})
    pt = " + preserve_thinking" if eb.get("chat_template_kwargs", {}).get("preserve_thinking") else ""
    console.print(
        f"[green][preset → {name}][/green]  temp={p['temperature']}  top_p={p['top_p']}  max={p['max_tokens']}{pt}"
    )
    console.print(f'[dim]  (to persist: add preset = "{name}" to {CONFIG_FILE})[/dim]')


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
    mem = load_memory()
    mem_lines = len([ln for ln in mem.splitlines() if ln.strip()]) if mem else 0
    tok = _last_turn_tokens
    uptime_s = time.monotonic() - _session_start if _session_start else 0
    mins, secs = int(uptime_s // 60), int(uptime_s % 60)

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="cyan")
    t.add_row("Title", _session_title or "(untitled)")
    t.add_row("Model", f"{MODEL}  [{_active_preset} preset]")
    ctx_tok = _real_ctx_tokens
    ctx_pct = f"{ctx_tok * 100 // TOKEN_LIMIT}%" if ctx_tok else "n/a"
    t.add_row("Turns", str(turns))
    t.add_row("Last turn", f"{tok.get('prompt', 0):,} in / {tok.get('completion', 0):,} out tok")
    t.add_row("Context", f"{ctx_tok:,} / {TOKEN_LIMIT:,} tok ({ctx_pct})" if ctx_tok else "n/a (no turn yet)")
    t.add_row("Memory entries", str(mem_lines))
    t.add_row("Modified files", str(len(_session_changes)))
    t.add_row("Branches", str(len(_branches)))
    t.add_row("Uptime", f"{mins}m {secs:02d}s")
    t.add_row("Auto-search", AUTO_SEARCH_MODE)
    t.add_row("Mode", _current_mode or "default")
    console.print(t)


def cmd_review(arg: str, history: list, base_system: str, client: object) -> None:
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
    t0 = time.monotonic()
    reply = run_turn(client, msgs, allow_tools=False)
    if reply:
        history.append({"role": "user", "content": f"[Code review: {names}]"})
        history.append({"role": "assistant", "content": reply})
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


def cmd_error(history: list, base_system: str, client: object) -> None:
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
                ln = (console.input("  ") if _pt_session is None else _pt_session.prompt("  ")).rstrip()
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
    msgs = [{"role": "system", "content": build_system_prompt(base_system)}, *history]
    msgs.append({"role": "user", "content": user_content})
    t0 = time.monotonic()
    reply = run_turn(client, msgs, allow_tools=True)
    if reply:
        history.append({"role": "user", "content": user_content})
        history.append({"role": "assistant", "content": reply})
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
        console.print(f"  [cyan]{i}.[/cyan] {f.stem}  [dim]{meta.get('topic', '')[:50]}[/dim]")
    try:
        pick = console.input("Pick [1]: ").strip()
        idx = (int(pick) - 1) if pick else 0
        if 0 <= idx < len(matches):
            h, s = load_session(matches[idx].stem)
            if h is not None:
                console.print(f"[green][loaded {matches[idx].stem}][/green]")
                return h, (s or base_system)
    except (ValueError, KeyboardInterrupt):
        pass
    return history, base_system


def export_session(history: list, name: str = "") -> None:
    """Export Session."""
    fname = (name or datetime.now().strftime("%Y%m%d_%H%M%S")) + ".md"
    path = EXPORTS_DIR / fname
    lines = [
        "# Qwen Chat Export\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  \n",
        f"**Model:** {MODEL}\n\n---\n",
    ]
    for msg in history:
        role = msg.get("role", "")
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
    """Approx Tokens."""
    return sum(len(m.get("content") or "") for m in messages) // 4


def show_context_breakdown(base_system: str, history: list) -> None:
    """Show Context Breakdown."""
    sys_text = build_system_prompt(base_system)
    sys_tok = len(sys_text) // 4
    mem_tok = len(load_memory()) // 4
    git_tok = len(get_git_context()) // 4
    pin_tok = sum(len(p) for p in load_pins()) // 4

    proj_msgs = [m for m in history if (m.get("content") or "").startswith("# Project Context")]
    proj_tok = sum(len(m.get("content") or "") for m in proj_msgs) // 4
    chat_msgs = [m for m in history if not (m.get("content") or "").startswith("# Project Context")]
    chat_tok = sum(len(m.get("content") or "") for m in chat_msgs) // 4
    total = sys_tok + chat_tok
    pct = total * 100 // TOKEN_LIMIT

    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(justify="right", style="cyan")
    t.add_row("System prompt", f"{sys_tok:,} tok")
    if mem_tok:
        t.add_row("  └ memory", f"{mem_tok:,} tok")
    if git_tok:
        t.add_row("  └ git", f"{git_tok:,} tok")
    if pin_tok:
        t.add_row("  └ pins", f"{pin_tok:,} tok")
    if proj_tok:
        t.add_row("Project ctx", f"{proj_tok:,} tok")
    chat_turns = sum(1 for m in history if m.get("role") == "assistant")
    t.add_row("Chat history", f"{chat_tok:,} tok  ({chat_turns} turns)")
    if _model_params:
        params_str = "  ".join(f"{k}={v}" for k, v in _model_params.items())
        t.add_row("Params", f"[yellow]{params_str}[/yellow]")
    if _session_changes:
        t.add_row("Modified files", f"[yellow]{len(_session_changes)}[/yellow]  /changes for details")
    t.add_row("─" * 20, "─" * 14)
    color = "yellow" if pct >= 75 else "green"
    t.add_row("[bold]Total[/bold]", f"[bold {color}]~{total:,} / {TOKEN_LIMIT:,} ({pct}%)[/bold {color}]")
    console.print(t)


def warn_if_context_high(messages: list) -> None:
    """Warn If Context High."""
    tok = approx_tokens(messages)
    pct = tok * 100 // TOKEN_LIMIT
    if pct >= 75:
        console.print(
            f"[yellow]  context: ~{tok:,} / {TOKEN_LIMIT:,} tokens ({pct}%) — consider /trim[/yellow]",
        )


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

KEY_FILES = {
    "README.md",
    "README.txt",
    "README.rst",
    "package.json",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "CLAUDE.md",
    "requirements.txt",
    "Cargo.toml",
    "go.mod",
    "composer.json",
    "Makefile",
    ".env.example",
}
LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "jsx",
    ".json": "json",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".sh": "bash",
    ".ps1": "powershell",
    ".bat": "batch",
    ".sql": "sql",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".rs": "rust",
    ".go": "go",
    ".cpp": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".txt": "",
}


def _resolve(path: str) -> Path:
    """Internal helper: resolve."""
    p = Path(path).expanduser()
    return p.resolve(strict=False) if p.is_absolute() else (Path.cwd() / p).resolve(strict=False)


def build_project_tree(root: Path, max_depth: int = 3, max_files: int = 200) -> str:
    """Build Project Tree."""
    extra_ignore = load_qwenignore(root)
    lines: list[str] = []
    count = [0]

    def walk(path: Path, depth: int, prefix: str) -> None:
        """Walk."""
        if count[0] >= max_files:
            return
        try:
            entries = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
        except PermissionError:
            return
        visible = [
            e
            for e in entries
            if not (e.is_dir() and e.name in IGNORE_DIRS)
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
    """Load Project Context."""
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
            except Exception as e:
                _logger.warning("Could not read key file %s: %s", fpath, e)
                console.print(f"[dim yellow]  [project: skipped unreadable {fpath.name} — {e}][/dim yellow]")
    parts = [f"# Project Context — {root}\n\n## Directory Tree\n```\n{tree}\n```"]
    if key_sections:
        parts.append("## Key Files\n\n" + "\n\n".join(key_sections))
    history.append({"role": "user", "content": "\n\n".join(parts)})
    console.print(
        f"[green][project loaded: {root.name} — {len(tree.splitlines())} tree lines"
        f"{f', {len(key_sections)} key file(s)' if key_sections else ''}][/green]",
    )

    # Build or load symbol index in background
    def _bg_index() -> None:
        """Internal helper: bg index."""
        idx = _load_symbol_index(root)
        if idx is None:
            idx = build_symbol_index(root)
            _save_symbol_index(root, idx)
        global _cached_index, _cached_index_root
        _cached_index, _cached_index_root = idx, root
        sym_count = sum(len(s.get("functions", [])) + len(s.get("classes", [])) for s in idx.values())
        console.print(f"[dim][indexed: {len(idx)} files, {sym_count} symbols — /index force to refresh][/dim]")

    threading.Thread(target=_bg_index, daemon=True).start()
    return True


def load_file_into_context(arg: str, history: list) -> bool:
    """Load File Into Context."""
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
        text = raw.decode("utf-8", errors="replace")
        size = len(raw)
        nlines = text.count("\n") + 1
        if size > 150_000:
            console.print(f"[yellow][large file ({size:,} bytes) — truncating to 150 KB][/yellow]")
            text = text[:150_000]
        lang = LANG_MAP.get(path.suffix.lower(), "")
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
    # Detect: CMD=rm; $CMD -rf /
    if re.search(r"\$[A-Z_]+\s+(-[a-zA-Z]*r|rm|mkfs|dd|chmod)", command):
        return True
    # Detect: base64 encoded commands
    if re.search(r"base64\s+(-d|--decode)", command):
        return True
    # Detect: eval/exec with suspicious content
    if re.search(r"eval\s+.*\$\(", command):
        return True
    # Detect: command substitution with dangerous commands
    if re.search(r"\$\(.*\b(rm|mkfs|dd|chmod)\b", command):
        return True
    # Detect: pipe to shell/interpreter (curl/wget/echo ... | bash/sh/python/powershell/iex)
    return bool(re.search(r"\|\s*(bash|sh|zsh|fish|python\d*|perl|ruby|pwsh|powershell|iex|cmd)\b", command))


_MAX_CMD_OUTPUT = 25_000

_SCRIPT_INTERP: dict[str, tuple[str, str]] = {
    "python": (sys.executable, ".py"),
    "python3": (sys.executable, ".py"),
    "py": (sys.executable, ".py"),
    "javascript": ("node", ".js"),
    "js": ("node", ".js"),
    "node": ("node", ".js"),
    "bash": ("bash", ".sh"),
    "sh": ("sh", ".sh"),
    "powershell": ("pwsh -NoProfile -ExecutionPolicy Bypass -File", ".ps1"),
    "ps1": ("pwsh -NoProfile -ExecutionPolicy Bypass -File", ".ps1"),
    "ruby": ("ruby", ".rb"),
    "php": ("php", ".php"),
    "lua": ("lua", ".lua"),
    "perl": ("perl", ".pl"),
    "go": ("go run", ".go"),
    "r": ("Rscript", ".r"),
    "typescript": ("npx ts-node", ".ts"),
    "ts": ("npx ts-node", ".ts"),
}


def do_run_command(
    command: str, cwd: str = "", timeout: int = 30, env: dict | None = None, stdin: str = "", quiet: bool = False
) -> str:
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

    proc_env = {**os.environ, **env} if env else None
    stdout_buf: list[str] = []
    stderr_buf: list[str] = []
    cancelled = threading.Event()
    t0 = time.monotonic()

    _shell_meta = re.compile(r"[|;&$`<>()\[\]{}!\\\n]")
    try:
        if _shell_meta.search(command):
            _logger.warning("shell=True for command with metacharacters: %.120s", command)
            proc = subprocess.Popen(
                command,
                shell=True,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=proc_env,
            )
        else:
            cmd_parts = shlex.split(command)
            proc = subprocess.Popen(
                cmd_parts,
                shell=False,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=proc_env,
            )
        if stdin:
            try:
                proc.stdin.write(stdin)
                proc.stdin.close()
            except BrokenPipeError:
                pass

        def _pipe(stream, buf: list[str], style: str) -> None:
            """Internal helper: pipe."""
            for line in stream:
                if cancelled.is_set():
                    break
                buf.append(line)
                console.print(f"[{style}]{line.rstrip()}[/{style}]")

        t_out = threading.Thread(target=_pipe, args=(proc.stdout, stdout_buf, "dim"), daemon=True)
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
            result = _get_lsp().lsp_preflight_check(code, "python")
            if not result["clean"]:
                console.print(
                    f"[dim yellow]  Pre-flight: {result['errors']} error(s), {result['warnings']} warning(s) in script[/dim yellow]"
                )
        except Exception:
            _logger.debug("LSP pre-flight check unavailable", exc_info=True)

    with _tmpmod.NamedTemporaryFile(
        suffix=ext,
        mode="w",
        delete=False,
        encoding="utf-8",
        prefix="qwen_script_",
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
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)

        if offset or limit:
            start = max(0, offset - 1)
            end = (start + limit) if limit else total
            snippet = "\n".join(lines[start:end])
            header = f"{p}  (lines {start + 1}–{min(end, total)} of {total})"
            start_line = start + 1
        else:
            if len(raw) > 150_000:
                text = text[:150_000] + "\n... [truncated at 150 KB]"
                lines = text.splitlines()
            snippet = "\n".join(lines)
            header = f"{p}  ({total} lines)"
            start_line = 1

        lang = LANG_MAP.get(p.suffix.lower(), "")
        console.print(f"[dim cyan]  {header}[/dim cyan]")
        if lang:
            try:
                console.print(
                    Syntax(snippet, lang, line_numbers=True, start_line=start_line, theme="monokai", word_wrap=False)
                )
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
            _lsp = _get_lsp()
            if _lsp._is_code_file(str(p)):
                pre = _lsp.lsp_pre_edit_check(str(p))
                if not pre["clean"]:
                    console.print(
                        f"[dim yellow]  Pre-edit: {pre['error_count']} error(s), "
                        f"{pre['warning_count']} warning(s) already in file[/dim yellow]",
                    )
                impact = _get_lsp().lsp_check_patch_impact(str(p), diff)
                if impact["conflicts"]:
                    console.print(
                        f"[dim red]  Conflict: {len(impact['conflicts'])} error(s) on lines about to be changed[/dim red]"
                    )
        except Exception:
            _logger.debug("LSP pre-edit check failed for %s (non-critical)", p)

        # Feature 5: Rename safety -- check renamed symbols have all refs updated
        try:
            _lsp = _get_lsp()
            if _lsp._is_code_file(str(p)):
                for dline in diff.split("\n"):
                    if dline.startswith("+") and not dline.startswith("+++"):
                        stripped = dline.lstrip("+").strip()
                        if any(kw in stripped for kw in ("class ", "def ", " = ", "import ")):
                            words = stripped.split()
                            for w in words:
                                if w.isidentifier() and len(w) > 2:
                                    refs = _lsp.lsp_references(str(p), w)
                                    if refs.get("count", 0) > 1:
                                        console.print(
                                            f'[dim yellow]  Rename safety: "{w}" has {refs["count"]} references -- verify all updated[/dim yellow]',
                                        )
                                    break
        except Exception:
            _logger.debug("LSP rename-safety check failed for %s (non-critical)", p)

        # Feature 6+10: Post-edit trend tracking + import check
        try:
            _lsp = _get_lsp()
            if _lsp._is_code_file(str(p)):
                post = _lsp.lsp_post_edit_check(str(p))
                if post["new_errors"] > 0:
                    console.print(f"[dim red]  Post-edit: {post['new_errors']} new error(s) introduced[/dim red]")
                if post["fixed_errors"] > 0:
                    console.print(f"[dim green]  Post-edit: {post['fixed_errors']} error(s) fixed[/dim green]")
                imports = _lsp.lsp_check_imports(str(p))
                if imports["broken"]:
                    console.print(f"[dim red]  Imports: {len(imports['broken'])} broken import(s)[/dim red]")
        except Exception:
            _logger.debug("LSP post-edit check failed for %s (non-critical)", p)

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
        preview_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                patched.splitlines(keepends=True),
                fromfile=f"a/{p.name}",
                tofile=f"b/{p.name}",
                lineterm="",
            )
        )
        preview = "".join(preview_lines[:60])
        if len(preview_lines) > 60:
            preview += f"\n... ({len(preview_lines) - 60} more lines)"
        console.print(Syntax(preview, "diff", theme="monokai"))
        answer = console.input("[bold yellow]  Apply patch? [y/N]:[/bold yellow] ").strip().lower()
        if answer != "y":
            return "[patch cancelled by user]"

        # Backup and write
        global _backup_stack
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        dirs = [e for e in entries if e.is_dir()]
        files = [e for e in entries if e.is_file()]
        nd, nf = len(dirs), len(files)
        lines = [f"{p}:  ({nd} dir{'s' if nd != 1 else ''}, {nf} file{'s' if nf != 1 else ''})\n"]
        for e in dirs:
            lines.append(f"  [dir ]  {e.name}/")
        for e in files:
            st = e.stat()
            size = _fmt_size(st.st_size)
            mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
            lines.append(f"  [file]  {e.name:<45}  {size:>8}  {mtime}")
        return "\n".join(lines)
    except Exception as e:
        return f"[error: {e}]"


def do_find_files(path: str, pattern: str) -> str:
    """Handle find files operation."""
    try:
        import fnmatch as _fnmatch
        import os as _os

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
                st = r.stat()
                size = _fmt_size(st.st_size)
                mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
                lines.append(f"  {rel!s:<60}  {size:>8}  {mtime}")
            else:
                lines.append(f"  {rel}/")
        if len(results) > 500:
            lines.append(f"  ... and {len(results) - 500} more")
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
        import fnmatch as _fnmatch
        import os as _os

        _name_pat = pattern.rstrip("/").split("/")[-1] if pattern else "*"

        def _walk_files(root: Path) -> Iterator[Path]:
            """Recursively walk a directory, yielding file paths matching a name pattern."""
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
                match_idx = [i for i, ln in enumerate(file_lines) if rx.search(ln)]
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
                            lines_out.append(f"  {marker} L{ci + 1:>4}: {file_lines[ci].rstrip()}")
                else:
                    for mi in match_idx:
                        lines_out.append(f"  L{mi + 1:>4}: {file_lines[mi].rstrip()}")
                rel = str(fpath.relative_to(p))
                file_hits[rel] = (match_idx, lines_out)
                total += len(match_idx)
                if total >= 500:
                    break
            except Exception:
                _logger.debug("Skipping unreadable file %s during search", fpath)
                continue

        if not file_hits:
            return f"No matches for '{query}' in {p}"
        out = [f"Matches for '{query}' in {p} — {total} match(es) in {len(file_hits)} file(s):\n"]
        for fname, (indices, hit_lines) in file_hits.items():
            n = len(indices)
            out.append(f"{fname}  ({n} match{'es' if n != 1 else ''}):")
            out.extend(hit_lines[:250])
            if len(hit_lines) > 250:
                out.append(f"  ... ({len(hit_lines) - 250} more lines)")
            out.append("")
        if total >= 500:
            out.append("... (stopped at 500 — narrow your query or glob pattern)")
        return "\n".join(out)
    except Exception as e:
        return f"[error: {e}]"


def _cleanup_backups(keep: int = 50, keep_per_file: int = 10) -> None:
    """Keep the most recent N backups overall, and at most M per original file.

    The per-file cap stops one hot file (e.g. a test fixture edited hundreds of
    times) from evicting the only backups of everything else.
    """
    if not BACKUPS_DIR.exists():
        return
    files = sorted(BACKUPS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    per_file: dict[str, int] = {}
    for i, f in enumerate(files):
        orig = f.name.rsplit(".", 2)[0]  # "<original>.<stamp>.bak" → "<original>"
        per_file[orig] = per_file.get(orig, 0) + 1
        if i >= keep or per_file[orig] > keep_per_file:
            with contextlib.suppress(Exception):
                f.unlink()


def _backup_file(p: Path) -> None:
    """Internal helper: backup file."""
    global _backup_stack
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = BACKUPS_DIR / f"{p.name}.{stamp}.bak"
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
            diff = list(
                difflib.unified_diff(
                    old.splitlines(keepends=True),
                    content.splitlines(keepends=True),
                    fromfile=f"a/{p.name}",
                    tofile=f"b/{p.name}",
                    lineterm="",
                )
            )
            preview = "".join(diff[:80])
            if len(diff) > 80:
                preview += f"\n... ({len(diff) - 80} more diff lines)"
            console.print(Syntax(preview, "diff", theme="monokai"))
            answer = console.input("[bold yellow]  Overwrite? [y/N]:[/bold yellow] ").strip().lower()
            if answer != "y":
                return "[write cancelled by user]"
        p.write_text(content, encoding="utf-8")
        action = "updated" if existed else "created"
        console.print(f"[bold yellow]  [write_file][/bold yellow] {action}: {p}")

        # Feature 6+10: Post-edit trend tracking + import check
        try:
            _lsp = _get_lsp()
            if _lsp._is_code_file(str(p)):
                post = _lsp.lsp_post_edit_check(str(p))
                if post["new_errors"] > 0:
                    console.print(f"[dim red]  Post-edit: {post['new_errors']} new error(s) introduced[/dim red]")
                if post["fixed_errors"] > 0:
                    console.print(f"[dim green]  Post-edit: {post['fixed_errors']} error(s) fixed[/dim green]")
                imports = _lsp.lsp_check_imports(str(p))
                if imports["broken"]:
                    console.print(f"[dim red]  Imports: {len(imports['broken'])} broken import(s)[/dim red]")
        except Exception:
            _logger.debug("LSP post-edit/imports check failed for %s (non-critical)", p)

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
            answer = (
                console.input(
                    f"[bold yellow]  {d} already exists. Overwrite? [y/N]:[/bold yellow] ",
                )
                .strip()
                .lower()
            )
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
            f"[bold red]  [delete_file][/bold red] {p}  ({_fmt_size(p.stat().st_size)})",
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


def list_models(client: object) -> list[str]:
    """List Models."""
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


def _run_background_tasks(client, user_msg: str, reply: str, history_snapshot: list, turn_count: int) -> None:
    with _BG_LLM_SEM:
        _auto_extract_memory(client, user_msg, reply)
        _intel_process_queue(client)
        if turn_count == 1:
            _generate_session_title(client, history_snapshot)
    _intel_extract_topics(client, user_msg, reply)


def record_session_changes_memory(client: object | None = None) -> None:
    """Session-end hook: if files were modified this session, log a dated entry to memory.md.

    Always writes a deterministic entry (file list). If the model is reachable, the
    entry is upgraded to a one-line summary of what actually changed, built from the
    real diffs (original content is kept in _session_changes).
    """
    if not _session_changes:
        return
    names = sorted({Path(p).name for p in _session_changes})
    shown = ", ".join(names[:12]) + (f" (+{len(names) - 12} more)" if len(names) > 12 else "")
    date = datetime.now().strftime("%Y-%m-%d")
    line = f"- {date}: modified {shown}"

    if client is not None:
        try:
            import difflib

            chunks = []
            for path_str, original in list(_session_changes.items())[:8]:
                try:
                    current = Path(path_str).read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                diff = "\n".join(
                    difflib.unified_diff(
                        original.splitlines(),
                        current.splitlines(),
                        fromfile=Path(path_str).name,
                        tofile=Path(path_str).name,
                        lineterm="",
                        n=1,
                    )
                )
                if diff:
                    chunks.append(diff[:1200])
            if chunks:
                prompt = [
                    {
                        "role": "system",
                        "content": (
                            "Summarize these code diffs in ONE line under 120 characters: "
                            "what changed, past tense, no filler. Reply with only the line."
                        ),
                    },
                    {"role": "user", "content": "\n\n".join(chunks)[:6000]},
                ]
                resp = client.chat.completions.create(
                    model=MODEL, messages=prompt, stream=False, max_tokens=80, timeout=AUX_LLM_TIMEOUT
                )
                summary = (resp.choices[0].message.content or "").strip().splitlines()[0].strip()
                if 10 < len(summary) <= 200:
                    line = f"- {date}: {summary} (files: {shown})"
        except Exception:
            _logger.debug("Session-change summary via model failed; using file list")

    marker = "# Recent Changes"
    with _memory_lock:
        mem = load_memory()
        # Dedup: skip if an entry for the same date and file set is already logged
        # (prevents repeated identical entries from tool-test or short sessions).
        prefix = f"- {date}:"
        for existing in mem.splitlines():
            if existing.startswith(prefix) and shown in existing:
                return
        if marker in mem:
            idx = mem.index(marker) + len(marker)
            mem = mem[:idx] + "\n\n" + line + mem[idx:]
        else:
            mem = (mem + f"\n\n{marker}\n\n{line}").strip()
        save_memory(mem.strip())
    console.print(f"[dim][memory: logged session changes — {shown}][/dim]")


def _auto_extract_memory(client, user_msg: str, assistant_msg: str) -> None:
    """Pull memorable facts from this exchange and append to memory.md."""
    global _auto_memory_count
    with _main_llm_busy_lock:
        if _main_llm_busy:
            return
    try:
        excerpt = f"User: {user_msg[:600]}\n\nAssistant: {assistant_msg[:1000]}"
        prompt = [
            {
                "role": "system",
                "content": (
                    "Extract facts worth remembering long-term from this exchange: "
                    "concrete preferences, decisions, names, configs, or constraints. "
                    "Skip pleasantries and transient state. "
                    "Format as short bullet points. "
                    "If nothing is notable, reply with exactly: NONE"
                ),
            },
            {"role": "user", "content": excerpt},
        ]
        resp = client.chat.completions.create(
            model=MODEL,
            messages=prompt,
            stream=False,
            max_tokens=200,
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
        _logger.debug("Auto-memory extraction failed")


def _curate_memory(client: object, locked: bool = False) -> None:
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
                {
                    "role": "system",
                    "content": "Consolidate this memory into clean, non-redundant bullet points. Remove duplicates. Preserve all unique facts. Be concise.",
                },
                {"role": "user", "content": mem},
            ]
            resp = client.chat.completions.create(
                model=MODEL,
                messages=prompt,
                stream=False,
                max_tokens=600,
                timeout=AUX_LLM_TIMEOUT,
            )
            consolidated = (resp.choices[0].message.content or "").strip()
            if consolidated and len(consolidated) > 20:
                save_memory(consolidated)
        except Exception:
            _logger.debug("Memory curation failed")


# ---------------------------------------------------------------------------
# Background intelligence crawlers — web browsing + memory training
# ---------------------------------------------------------------------------


def _intel_load_topics() -> list[dict]:
    """Internal helper: intel load topics."""
    if INTEL_TOPICS.exists():
        try:
            return json.loads(INTEL_TOPICS.read_text(encoding="utf-8"))
        except Exception:
            _logger.debug("Failed to load intel topics from %s", INTEL_TOPICS)
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
            _logger.debug("Failed to load intel feed from %s", INTEL_FEED)
    return ""


def _intel_prepend_entry(topic_name: str, summary: str) -> None:
    """Internal helper: intel prepend entry."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"<!-- {ts} | {topic_name} -->\n{summary.strip()}"
    with _intel_lock:
        existing = _intel_load_feed()
        chunks = [c.strip() for c in existing.split("\n\n") if c.strip()] if existing else []
        chunks.insert(0, entry)
        chunks = chunks[:40]  # cap at 40 entries
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
    topic: dict | None = None
    try:
        # _INTEL_CRAWLERS runs multiple copies of this function concurrently.
        # Claim the topic (mark last_checked, save) under the lock *before*
        # doing the slow web search, so two threads never pick the same
        # least-recently-crawled topic and _intel_extract_topics()'s topic
        # additions can't be lost to an overlapping read-modify-write.
        with _intel_lock:
            topics = _intel_load_topics()
            if not topics:
                return
            topic = min(topics, key=lambda t: t.get("last_checked", 0))
            for t in topics:
                if t["name"] == topic["name"]:
                    t["last_checked"] = time.time()
                    break
            _intel_save_topics(topics)
        raw = do_web_search(topic["query"], max_results=5)
        if raw and "error" not in raw.lower()[:40]:
            _intel_enqueue(topic["name"], topic["query"], raw)
    except Exception:
        _logger.debug("Intel background crawl failed for topic '%s'", topic.get("name", "?") if topic else "?")


def _intel_process_queue(client) -> None:
    """Post-turn: LLM-summarize queued raw results, update feed, train memory."""
    items = _intel_dequeue_all()
    if not items:
        return
    with _main_llm_busy_lock:
        if _main_llm_busy:
            return
    for item in items:
        try:
            prompt = [
                {
                    "role": "system",
                    "content": (
                        "Summarize these web search results into 3-5 concise bullet points "
                        "(max 90 chars each). Focus on concrete facts, releases, and updates. "
                        "No preamble, just the bullets."
                    ),
                },
                {"role": "user", "content": f"Topic: {item['topic']}\n\n{item['raw'][:3000]}"},
            ]
            resp = client.chat.completions.create(
                model=MODEL,
                messages=prompt,
                stream=False,
                max_tokens=250,
                timeout=AUX_LLM_TIMEOUT,
            )
            summary = (resp.choices[0].message.content or "").strip()
            if not summary or len(summary) < 20:
                continue
            _intel_prepend_entry(item["topic"], summary)
            _intel_train_memory(client, item["topic"], summary)
        except Exception:
            _logger.debug("Intel queue item '%s' failed", item.get("topic", "?"))


def _intel_train_memory(client: object, topic_name: str, summary: str) -> None:
    """If the intel summary contains durable facts, add them to persistent memory."""
    today = datetime.now().strftime("%Y-%m-%d")
    if _intel_memory_written.get(topic_name) == today:
        return  # already wrote facts for this topic today
    try:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Decide if any of these facts should be saved to a persistent memory file "
                    "(version numbers, critical releases, security alerts, key API changes). "
                    "If yes, output 1-2 short bullet lines starting with '- '. "
                    "If nothing is worth persisting, reply exactly: SKIP"
                ),
            },
            {"role": "user", "content": f"Topic: {topic_name}\n{summary}"},
        ]
        resp = client.chat.completions.create(
            model=MODEL,
            messages=prompt,
            stream=False,
            max_tokens=120,
            timeout=AUX_LLM_TIMEOUT,
        )
        facts = (resp.choices[0].message.content or "").strip()
        if facts and facts.upper() != "SKIP" and facts.startswith("-"):
            with _memory_lock:
                mem = load_memory()
                tag = f"\n\n<!-- intel {today} -->\n{facts}"
                save_memory((mem + tag).strip())
            _intel_memory_written[topic_name] = today
    except Exception:
        _logger.debug("Intel memory training failed for '%s'", topic_name)


def _intel_extract_topics(client, user_msg: str, reply: str) -> None:
    """Post-turn: extract new search-worthy topics from this exchange and track them."""
    with _main_llm_busy_lock:
        if _main_llm_busy:
            return
    try:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Extract up to 2 web-searchable topics from this conversation worth monitoring "
                    "(new technologies, frameworks, tools, or domains the user cares about). "
                    "Reply one per line as: NAME|search query\n"
                    "Example: FastAPI 1.0|FastAPI 1.0 release features changelog\n"
                    "If nothing new to track, reply: NONE"
                ),
            },
            {"role": "user", "content": f"User: {user_msg[:400]}\nAssistant: {reply[:400]}"},
        ]
        resp = client.chat.completions.create(
            model=MODEL,
            messages=prompt,
            stream=False,
            max_tokens=80,
            timeout=AUX_LLM_TIMEOUT,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text or text.upper() == "NONE":
            return
        # See _intel_crawl_once(): topics.json is shared with the background
        # crawler threads, so the read-modify-write must hold _intel_lock too
        # or a concurrent crawl can silently overwrite these additions.
        with _intel_lock:
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
        _logger.debug("Intel topic extraction failed")


def _intel_crawler_thread(delay_s: int) -> None:
    """Background daemon: crawl one topic every _INTEL_INTERVAL seconds."""
    INTEL_DIR.mkdir(exist_ok=True)
    _intel_stop.wait(timeout=delay_s)  # stagger startup
    while not _intel_stop.is_set():
        try:
            _intel_crawl_once()
        except Exception:
            # _intel_crawl_once() already wraps its own body in try/except;
            # this outer guard exists so a bug there can never silently kill
            # this thread forever (as `_intel_enabled = False` used to, by
            # replacing the Event with a plain bool — see _cmd_intel).
            _logger.exception("Intel crawler thread tick failed")
        _intel_stop.wait(timeout=_INTEL_INTERVAL)


def start_intel_crawlers() -> None:
    """Start _INTEL_CRAWLERS background crawler threads, staggered."""
    if not INTEL_TOPICS.exists():
        _intel_save_topics([dict(t) for t in _INTEL_DEFAULT_TOPICS])
    # threading.Event() starts unset; without this, _intel_crawl_once()'s
    # `if not _intel_enabled.is_set(): return` guard is true forever and no
    # crawler thread ever does real work.
    _intel_enabled.set()
    stagger = max(15, _INTEL_INTERVAL // _INTEL_CRAWLERS)
    for i in range(_INTEL_CRAWLERS):
        t = threading.Thread(
            target=_intel_crawler_thread,
            args=(15 + i * stagger,),
            daemon=True,
            name=f"intel-crawler-{i}",
        )
        t.start()


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
    return not imperative_count < max(2, len(steps) // 2)


def _confidence_warning(text: str) -> None:
    """Print a soft warning if the response contains many hedging phrases."""
    count = len(_HEDGE_RE.findall(text))
    if count >= 3:
        console.print(
            f"[dim yellow]  [low confidence — {count} hedge phrases — consider /retry with more context][/dim yellow]",
        )


def _verification_pending(tool_names: list, pending: bool = False) -> bool:
    """Track whether a file mutation happened without a subsequent verifying tool call.

    Scans tool names in order: a mutating tool sets the pending flag; any verifying
    tool that runs AFTER it clears the flag. Carries state across turns via `pending`.
    """
    for name in tool_names:
        if name in _MUTATING_FILE_TOOLS:
            pending = True
        elif pending and name in _VERIFYING_TOOLS:
            pending = False
    return pending


def cmd_agent(goal: str, history: list, base_system: str, client: object, max_iter: int = 20) -> None:
    """Autonomous agent loop: model iterates with tools until it emits AGENT_DONE.

    Completion is ENFORCED, not just prompted: if the agent modified files, AGENT_DONE
    is rejected until at least one verifying tool (tests, execution, re-read) has run
    after the last mutation.
    """
    agent_suffix = (
        "\n\nYou are running in autonomous agent mode. Work toward the goal using tools. "
        "After each action, briefly verify the result is correct before moving on — "
        "do not assume success; check the output or re-read the file. "
        "ENFORCEMENT: if you modify any file, AGENT_DONE will be REJECTED until you verify "
        "the change afterward (run the tests, execute the code, or re-read the file and check it). "
        "DELEGATION: if the goal has 3+ independent subtasks, or any subtask would take 5+ tool calls, "
        "spawn subagents via team_spawn_agent instead of doing everything yourself. "
        "Subagents have full access to all the same tools. You coordinate, they execute in parallel. "
        "When the goal is fully achieved and you have confirmed the result, end your response with exactly: AGENT_DONE"
    )
    console.print(f"[bold cyan]  Agent goal:[/bold cyan] {goal}")
    console.print(f"[dim]  (max {max_iter} iterations — Ctrl+C to stop)[/dim]")

    working = list(history)
    working.append({"role": "user", "content": f"[Agent task] {goal}"})

    pending_verify = False
    verify_rejections = 0
    for iteration in range(1, max_iter + 1):
        console.print(Rule(f"[dim]Agent {iteration}/{max_iter}[/dim]", style="dim"))
        # Keep the agent going across many iterations — summarize+preserve the task
        # if the accumulated working set is nearing the context limit.
        working = _maybe_autocompact(working, base_system, client)
        msgs = [{"role": "system", "content": build_system_prompt(base_system) + agent_suffix}, *working]
        try:
            reply = run_turn(client, msgs, allow_tools=True)
        except KeyboardInterrupt:
            console.print("\n[dim]  \\[agent stopped][/dim]")
            break

        if not reply:
            console.print("[red]  \\[agent error or cancelled — stopping][/red]")
            break

        working.append({"role": "assistant", "content": reply})
        history.append({"role": "user", "content": f"[Agent task] {goal}" if iteration == 1 else "[continue]"})
        history.append({"role": "assistant", "content": reply})

        pending_verify = _verification_pending(list(_last_turn_tool_names), pending_verify)

        if re.search(r"\bAGENT_DONE\b", reply):
            if pending_verify and verify_rejections < 3:
                verify_rejections += 1
                console.print(
                    f"[yellow]  \\[AGENT_DONE rejected ({verify_rejections}/3) — "
                    f"files were modified but never verified][/yellow]"
                )
                working.append(
                    {
                        "role": "user",
                        "content": (
                            "STOP — your AGENT_DONE was REJECTED. You modified files but never verified "
                            "the changes. Verify now: run the tests, execute the code, or re-read the "
                            "modified file(s) and confirm they are correct. State the evidence you found, "
                            "then emit AGENT_DONE again."
                        ),
                    }
                )
                continue
            if pending_verify:
                console.print(
                    "[yellow]  \\[agent finished WITHOUT verification after 3 rejections — review changes manually][/yellow]"
                )
            console.print(f"[green]  \\[agent done in {iteration} iteration{'s' if iteration != 1 else ''}][/green]")
            break

        working.append({"role": "user", "content": "Continue working toward the goal."})
    else:
        console.print(f"[yellow]  \\[agent reached max iterations ({max_iter}) — goal may be unfinished][/yellow]")


def cmd_git_commit(client: object, msg: str = "") -> None:
    """Generate a commit message from staged diff and commit."""
    try:
        diff = subprocess.run(
            ["git", "diff", "--staged"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        ).stdout.strip()
    except subprocess.TimeoutExpired:
        console.print("[red][git diff timed out][/red]")
        return
    if not diff:
        console.print("[yellow][no staged changes — git add first][/yellow]")
        return

    if not msg:
        try:
            prompt = [
                {
                    "role": "system",
                    "content": "Write a concise conventional commit message (type: subject). Reply with only the message, no quotes.",
                },
                {"role": "user", "content": f"```diff\n{diff[:6000]}\n```"},
            ]
            resp = client.chat.completions.create(
                model=MODEL, messages=prompt, stream=False, max_tokens=120, timeout=AUX_LLM_TIMEOUT
            )
            msg = (resp.choices[0].message.content or "").strip().strip("\"'")
        except Exception as e:
            console.print(f"[red][error generating message: {e}][/red]")
            return

    console.print(f"\n[bold]  Commit message:[/bold] {msg}\n")
    try:
        answer = console.input("[bold yellow]  Commit? [y/N]: [/bold yellow]").strip().lower()
        if answer == "y":
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as tf:
                tf.write(msg)
                tf_path = tf.name
            try:
                do_run_command(f'git commit -F "{tf_path}"')
                _invalidate_git_cache()  # state changed — drop stale git context
            finally:
                Path(tf_path).unlink(missing_ok=True)
    except (KeyboardInterrupt, EOFError):
        pass


def cmd_git_pr(client: object) -> None:
    """Generate a PR description from commits ahead of main/origin/main."""
    for base in ("main", "origin/main", "master", "origin/master"):
        try:
            log = subprocess.run(
                ["git", "log", f"{base}..HEAD", "--oneline"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=10,
            ).stdout.strip()
        except subprocess.TimeoutExpired:
            continue
        if log:
            try:
                stat = subprocess.run(
                    ["git", "diff", f"{base}...HEAD", "--stat"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=10,
                ).stdout.strip()
            except subprocess.TimeoutExpired:
                stat = ""
            break
    else:
        console.print("[yellow][no commits ahead of main/master][/yellow]")
        return

    try:
        prompt = [
            {
                "role": "system",
                "content": "Write a GitHub PR description with ## Summary (3-5 bullets) and ## Changes. Be concise and specific.",
            },
            {"role": "user", "content": f"Commits:\n{log}\n\nFile changes:\n{stat}"},
        ]
        resp = client.chat.completions.create(
            model=MODEL, messages=prompt, stream=False, max_tokens=400, timeout=AUX_LLM_TIMEOUT
        )
        pr = (resp.choices[0].message.content or "").strip()
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


def cmd_task(goal: str, history: list, base_system: str, client: object) -> None:
    """Plan-approve-execute-test agentic task loop."""
    root = Path.cwd()
    test_cmd = _detect_test_command(root)

    console.print(f"[bold cyan]  Task:[/bold cyan] {goal}")
    if test_cmd:
        console.print(f"[dim]  Tests: {test_cmd}[/dim]")

    # Generate plan (no tools, fast)
    console.print("[dim]  Planning...[/dim]")
    plan_msgs = [
        {"role": "system", "content": build_system_prompt(base_system)},
        *history,
        {
            "role": "user",
            "content": (
                f"Create a numbered step-by-step plan to accomplish: {goal}\n\n"
                "List only the steps. Be specific about which files to read or modify."
            ),
        },
    ]
    try:
        plan_resp = client.chat.completions.create(
            model=MODEL,
            messages=plan_msgs,
            stream=False,
            max_tokens=400,
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
    step_lines = [ln.strip() for ln in plan_text.splitlines() if re.match(r"^\d+[\.\)]\s+\S", ln.strip())]
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
            step_msgs = [{"role": "system", "content": build_system_prompt(base_system)}, *working]
            step_msgs.append({"role": "user", "content": f"Execute step {i}/{total}: {step}{extra_ctx}"})
            try:
                reply = run_turn(client, step_msgs, allow_tools=True)
            except KeyboardInterrupt:
                console.print("\n[dim]  \\[task stopped][/dim]")
                return
            if not reply:
                console.print(f"[red]  \\[step {i} failed — stopping][/red]")
                return

            working.append({"role": "user", "content": f"Step {i}: {step}"})
            working.append({"role": "assistant", "content": reply})
            history.append({"role": "user", "content": f"[Task {i}/{total}] {step}"})
            history.append({"role": "assistant", "content": reply})

            # Run tests if code was written/patched
            modified = any(kw in reply for kw in ("[patched:", "[created:", "[updated:"))
            if test_cmd and modified:
                console.print(f"[dim]  Running: {test_cmd}[/dim]")
                test_out = do_run_command(test_cmd, timeout=60)
                # Check for actual test failures — require non-zero counts to avoid
                # "0 failed" or "0 errors" in a passing summary line being a false positive
                has_fail = re.search(r"\b(FAILED|ERROR)\b", test_out)
                has_count = re.search(r"\b[1-9]\d*\s+(?:failed|error)\b", test_out, re.IGNORECASE)
                has_fail_test = re.search(r"\b(FAIL|ERROR)\b.*\b(test|spec|suite)\b", test_out)
                passed = not (has_fail or has_count or has_fail_test)
                if not passed and retries < 2:
                    console.print(f"[yellow]  Tests failed — retrying step {i} (attempt {retries + 2}/3)[/yellow]")
                    extra_ctx = (
                        f"\n\nPrevious attempt failed tests:\n{test_out[:1500]}\nFix the issue then complete step {i}."
                    )
                    retries += 1
                    continue
                if not passed:
                    console.print("[yellow]  Tests still failing — moving on[/yellow]")
            break

        if i < total:
            try:
                fb = (
                    console.input(
                        f"[dim]  [Step {i}/{total} done — Enter=next  f=feedback  s=stop]: [/dim]",
                    )
                    .strip()
                    .lower()
                )
            except (KeyboardInterrupt, EOFError):
                fb = "s"
            if fb == "s":
                console.print("[dim][task stopped by user][/dim]")
                return
            if fb.startswith("f"):
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


def _generate_session_title(client: object, history: list) -> None:
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
            {
                "role": "system",
                "content": "Generate a short 3-5 word title for this conversation. Reply with only the title.",
            },
            {"role": "user", "content": excerpt},
        ]
        resp = client.chat.completions.create(
            model=MODEL,
            messages=prompt,
            stream=False,
            max_tokens=15,
            timeout=AUX_LLM_TIMEOUT,
        )
        title = (resp.choices[0].message.content or "").strip().strip("\"'").rstrip(".")
        if title and 3 <= len(title) <= 60:
            with _title_lock:
                if _session_title == "...":
                    _session_title = title
            console.print(f"[dim]  [{title}][/dim]")
    except Exception:
        _logger.debug("Session title generation failed")


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


def _smart_cap(client: object, result: str, name: str, context: str = "") -> str:
    """Cap a tool result. Summarizes web/fetch results via LLM; hard-truncates others."""
    if len(result) <= TOOL_RESULT_LIMIT:
        return result
    if name not in _SUMMARIZE_TOOLS:
        return _cap_result(result, name)
    try:
        # Text-based key-fact extraction first (fast, no LLM call)
        key_lines = []
        url_re = re.compile(r"https?://\S+")
        number_re = re.compile(r"\b\d{2,}(?:,\d{0,3})*(?:\.\d+)?\b")
        seen_urls = set()
        for _line in result.strip().split("\n"):
            _line = _line.strip()
            if not _line:
                continue
            urls = url_re.findall(_line)
            new_urls = [u for u in urls if u not in seen_urls]
            if new_urls or number_re.search(_line) or len(_line) < 120:
                seen_urls.update(new_urls)
                key_lines.append(_line)
        if len(key_lines) <= 80:
            return f"[key facts extracted {len(result):,}->{len(key_lines):,} lines]\n\n" + "\n".join(key_lines)
        ctx_hint = f' relevant to: "{context[:200]}"' if context else ""
        prompt = [
            {"role": "system", "content": "Summarize precisely. Preserve all URLs, numbers, and key facts."},
            {
                "role": "user",
                "content": (
                    f"Summarize the following{ctx_hint} in under 1500 words, "
                    f"keeping all URLs and critical data:\n\n{result[:40_000]}"
                ),
            },
        ]
        resp = client.chat.completions.create(
            model=MODEL,
            messages=prompt,
            stream=False,
            max_tokens=1500,
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
    growths = [recent[i] - recent[i - 1] for i in range(1, len(recent)) if recent[i] > recent[i - 1]]
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
    if session_type == "chatty":
        return 85
    return 80


def _cap_result(result: str, name: str = "") -> str:
    """Internal helper: cap result."""
    if len(result) <= TOOL_RESULT_LIMIT:
        return result
    return (
        result[:TOOL_RESULT_LIMIT]
        + f"\n\n... [truncated: result was {len(result):,} chars; showing first {TOOL_RESULT_LIMIT:,}]"
    )


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
                backoff = _TOOL_RETRY_BASE * (2**attempt)
                console.print(
                    f"[dim yellow]  [retry {attempt + 1}/{max_retries}] {name} failed — "
                    f"retrying in {backoff:.0f}s: {e}[/dim yellow]"
                )
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


_TIMEOUT_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=8, thread_name_prefix="qwen-timeout")
atexit.register(lambda: _TIMEOUT_POOL.shutdown(wait=False))


def _call_with_timeout(name: str, fn, *args, timeout: int, **kwargs) -> str:
    """Call a tool function with a timeout using a shared thread pool."""
    fut = _TIMEOUT_POOL.submit(fn, *args, **kwargs)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        fut.cancel()
        return f"[{name}] timed out after {timeout}s. The operation took too long to complete."
    except Exception:
        raise


_TOOL_HANDLERS_SAFE: dict[str, Callable[[dict], str]] = {
    "web_search": lambda a: _call_with_timeout(
        "web_search", do_web_search, a.get("query", ""), timeout=_TOOL_TIMEOUT_SLOW
    ),
    "search_news": lambda a: _call_with_timeout(
        "search_news", do_search_news, a.get("query", ""), a.get("max_results", 8), timeout=_TOOL_TIMEOUT_SLOW
    ),
    "fetch_url": lambda a: _call_with_timeout(
        "fetch_url", do_fetch_url, a.get("url", ""), a.get("max_chars", 20_000), timeout=_TOOL_TIMEOUT_NET
    ),
    "describe_image": lambda a: _call_with_timeout(
        "describe_image", do_describe_image, a.get("url", ""), timeout=_TOOL_TIMEOUT_NET
    ),
    "get_video_transcript": lambda a: _call_with_timeout(
        "get_video_transcript", do_get_video_transcript, a.get("url", ""), a.get("lang", "en"),
        timeout=_TOOL_TIMEOUT_SLOW,
    ),
    "read_file": lambda a: _call_with_timeout(
        "read_file", do_read_file, a.get("path", ""), a.get("offset", 0), a.get("limit", 0),
        timeout=_TOOL_TIMEOUT_FAST,
    ),
    "list_directory": lambda a: _call_with_timeout(
        "list_directory", do_list_directory, a.get("path", ""), a.get("recursive", False),
        timeout=_TOOL_TIMEOUT_FAST,
    ),
    "find_files": lambda a: _call_with_timeout(
        "find_files", do_find_files, a.get("path", ""), a.get("pattern", "*"), timeout=_TOOL_TIMEOUT_FAST
    ),
    "search_files": lambda a: _call_with_timeout(
        "search_files", do_search_files, a.get("path", ""), a.get("query", ""),
        a.get("pattern", "**/*"), a.get("context", 0), timeout=_TOOL_TIMEOUT_FAST,
    ),
    "team_task_list": lambda a: _ct_render_task_list(
        a.get("team", ""), a.get("owner", ""), a.get("status", "")
    ),
    "team_task_add": lambda a: do_team_task_add(
        a.get("team", ""), a.get("subject", ""), a.get("owner", ""), a.get("priority", "medium")
    ),
    "team_board": lambda a: do_team_board(a.get("team", "")),
    "team_list": lambda _: do_team_list(),
    "team_inbox_receive": lambda a: _ct_render_inbox(
        a.get("team", ""), a.get("agent", ""), a.get("peek", False)
    ),
}


def _ct_render_task_list(team: str, owner: str, status: str) -> str:
    console.print(f"[dim cyan]  team_task_list: {team}[/dim cyan]")
    tasks = _ct_task_list(team, owner, status)
    if not tasks:
        return f"[no tasks in team '{team}']"
    lines = [f"Tasks for team '{team}' ({len(tasks)} total):"]
    for t in tasks:
        owner_tag = f" [{t['owner']}]" if t.get("owner") else ""
        lines.append(f"  [{t['id'][:6]}] [{t['status']}] {t['subject']}{owner_tag}")
    return "\n".join(lines)


def _ct_render_inbox(team: str, agent: str, peek: bool = False) -> str:
    console.print(f"[dim cyan]  team_inbox_receive: {team}/{agent}{'  (peek)' if peek else ''}[/dim cyan]")
    msgs = _ct_inbox_receive(team, agent, peek=peek)
    if not msgs:
        return f"[inbox empty for {agent} in team '{team}']"
    lines = [f"Messages for {agent} in team '{team}':"]
    for m in msgs:
        lines.append(f"  From: {m.get('from', '?')}  |  {m.get('body', '')}")
    return "\n".join(lines)


def _call_tool_safe(name: str, args: dict) -> str:
    """Dispatch a parallel-safe (non-interactive) tool."""
    handler = _TOOL_HANDLERS_SAFE.get(name)
    if handler is None:
        return f"[unknown tool: {name}]"
    _last_turn_tool_names.append(name)
    return handler(args)


def do_ask_user(question: str) -> str:
    """Handle ask user operation."""
    console.print(f"\n[bold cyan]  [ask_user][/bold cyan] {question}")
    try:
        answer = console.input("[bold cyan]  Your answer: [/bold cyan]").strip()
        return answer or "[no answer provided]"
    except (KeyboardInterrupt, EOFError):
        return "[user cancelled]"


def _ct_do_task_update(args: dict) -> str:
    team, task_id = args.get("team", ""), args.get("task_id", "")
    console.print(f"[dim cyan]  team_task_update: {team}/{task_id}[/dim cyan]")
    task = _ct_task_update(team, task_id, args.get("status"), args.get("owner"), args.get("note", ""))
    if not task:
        return f"[task '{task_id}' not found in team '{team}']"
    nn = len(task.get("notes", []))
    notes_tag = f"  ({nn} note{'s' if nn != 1 else ''})" if nn else ""
    return f"[updated: {task['id'][:6]}] {task['subject']} → {task['status']}{notes_tag}"


def _ct_do_inbox_send(args: dict) -> str:
    team, to = args.get("team", ""), args.get("to", "")
    msg = args.get("message", "")
    from_agent = args.get("from_agent", "user")
    console.print(f"[dim cyan]  team_inbox_send: {team}/{to}[/dim cyan]")
    msg_id = _ct_inbox_send(team, to, msg, from_agent)
    return f"[sent message {msg_id[:8]} to {to} in team '{team}']"


def _ct_do_spawn_agent(args: dict) -> str:
    team, agent_name = args.get("team", ""), args.get("agent_name", "")
    task = args.get("task", "")
    cwd = args.get("cwd", "")
    console.print(f"[dim cyan]  team_spawn_agent: {team}/{agent_name}[/dim cyan]")
    return _ct_spawn(team, agent_name, task, cwd)


_TOOL_HANDLERS_INTERACTIVE: dict[str, Callable[[dict], str]] = {
    "fetch_rendered": lambda a: do_fetch_rendered(a.get("url", ""), a.get("max_chars", 15000)),
    "browser_action": lambda a: do_browser_action(
        action=a.get("action", ""),
        url=a.get("url", ""),
        selector=a.get("selector", ""),
        value=a.get("value", ""),
        screenshot_path=a.get("screenshot_path", ""),
    ),
    "run_command": lambda a: do_run_command(
        a.get("command", ""), a.get("cwd", ""), a.get("timeout", 30),
        env=a.get("env") or None, stdin=a.get("stdin", ""),
    ),
    "run_script": lambda a: do_run_script(
        a.get("language", ""), a.get("code", ""), cwd=a.get("cwd", ""), timeout=a.get("timeout", 30),
    ),
    "patch_file": lambda a: do_patch_file(a.get("path", ""), a.get("diff", "")),
    "write_file": lambda a: do_write_file(a.get("path", ""), a.get("content", "")),
    "move_file": lambda a: do_move_file(a.get("src", ""), a.get("dst", "")),
    "delete_file": lambda a: do_delete_file(a.get("path", "")),
    "ask_user": lambda a: do_ask_user(a.get("question", "")),
    "team_task_update": _ct_do_task_update,
    "team_inbox_send": _ct_do_inbox_send,
    "team_spawn_agent": _ct_do_spawn_agent,
}


def _dispatch_interactive(name: str, args: dict) -> str:
    """Dispatch an interactive tool (may prompt user — must run on main thread)."""
    handler = _TOOL_HANDLERS_INTERACTIVE.get(name)
    if handler is None:
        return f"[unknown tool: {name}]"
    _last_turn_tool_names.append(name)
    return handler(args)


def _execute_tool_call(client, name: str, args: dict) -> str:
    """Dispatch a tool call in pipe mode (no interactive prompts)."""
    if name in _PARALLEL_TOOLS:
        return _call_tool_safe(name, args)
    return _dispatch_interactive(name, args)


def _auto_presearch(working: list) -> list:
    """Before the first LLM call, auto-run web_search on the user's message and
    inject the results as grounding context. Whether it fires is governed by
    AUTO_SEARCH_MODE (off | smart | aggressive); see presearch_decision().
    """
    last_user = next(
        (m.get("content", "") for m in reversed(working) if m.get("role") == "user"),
        "",
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
    new = list(working)
    shrunk = 0
    for i in tool_idxs[:-keep_recent_tools]:
        content = new[i].get("content") or ""
        if len(content) > head_chars + 80:
            new[i] = {
                **new[i],
                "content": content[:head_chars]
                + f"\n[... {len(content) - head_chars:,} chars condensed to fit context]",
            }
            shrunk += 1
    if shrunk:
        console.print(f"[dim]  [mid-run compact] condensed {shrunk} older tool result(s) to free context[/dim]")
    return new


def run_turn(client: object, messages: list, allow_tools: bool = True) -> str | None:
    """Full turn with tool-use loop. Returns reply, '' on cancel, None on error."""
    global _last_turn_tokens, _real_ctx_tokens
    del _last_turn_tool_names[:]  # fresh tool log for this turn (read by /agent verification)
    working = _auto_presearch(list(messages)) if allow_tools else list(messages)
    # Feature 10: Surface unresolved errors from prior turn
    try:
        trend = _get_lsp().lsp_trend_report()
        if trend.get("unresolved_errors"):
            console.print("[dim red]  Unresolved errors from prior edit:[/dim red]")
            for e in trend["unresolved_errors"][:5]:
                console.print(f"    {e}")
    except Exception:
        _logger.debug("LSP trend report unavailable", exc_info=True)

    use_tools = allow_tools
    first_call = True
    depth = 0
    tool_chain: list[str] = []
    total_prompt = 0
    total_completion = 0
    final_segments: list[str] = []  # final-answer pieces, joined across auto-continues
    auto_continue = 0  # how many times we've resumed a cut-off answer

    while True:
        # Mid-run compaction: a long tool loop can pile up large tool results and
        # overflow the context window before the task finishes. When the working set
        # nears the limit, shrink older tool results in place so the run keeps going.
        if depth > 0 and approx_tokens(working) >= TOKEN_LIMIT * 0.85:
            working = _compact_tool_loop(working)

        if depth >= MAX_TOOL_DEPTH:
            console.print(
                f"[yellow][max tool depth ({MAX_TOOL_DEPTH}) reached — synthesizing with gathered data][/yellow]"
            )
            # Force one final no-tools call so the model can answer with what it found
            synth_msgs = list(working)
            synth_msgs.append(
                {
                    "role": "user",
                    "content": "Based on all the information gathered above, provide a comprehensive answer now. "
                    "Do not search for more — synthesize what you have.",
                }
            )
            try:
                with Live("", console=console, vertical_overflow="crop", refresh_per_second=15, transient=True) as live:
                    final_text, _, final_usage = stream_once(
                        client,
                        synth_msgs,
                        use_tools=False,
                        update_fn=_live_updater(live),
                    )
                final_text, think_c = _strip_think(final_text)
                if think_c:
                    console.print(
                        Panel(Markdown(think_c), title="[dim]thinking[/dim]", border_style="dim", padding=(0, 1))
                    )
                console.print(Markdown(final_text) if final_text else Markdown("*(no synthesis)*"))
                total_prompt += final_usage.get("prompt", 0)
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
            with Live("", console=console, vertical_overflow="crop", refresh_per_second=15, transient=True) as live:
                text, tool_calls, usage = stream_once(
                    client,
                    working,
                    use_tools,
                    _live_updater(live),
                )
            text, think_content = _strip_think(text)
            if think_content:
                console.print(
                    Panel(
                        Markdown(think_content),
                        title="[dim]thinking[/dim]",
                        border_style="dim",
                        padding=(0, 1),
                    )
                )
            console.print(Markdown(text) if text else Markdown("*(no response)*"))
        except KeyboardInterrupt:
            console.print("\n[dim][cancelled][/dim]")
            return ""
        except Exception as e:
            if use_tools and first_call:
                console.print(f"[yellow][retrying without tool use — {e}][/yellow]")
                use_tools = False
                first_call = False
                try:
                    with Live(
                        "", console=console, vertical_overflow="crop", refresh_per_second=15, transient=True
                    ) as live:
                        text, tool_calls, usage = stream_once(
                            client,
                            working,
                            False,
                            _live_updater(live),
                        )
                    text, think_content = _strip_think(text)
                    if think_content:
                        console.print(
                            Panel(
                                Markdown(think_content),
                                title="[dim]thinking[/dim]",
                                border_style="dim",
                                padding=(0, 1),
                            )
                        )
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

        total_prompt += usage.get("prompt", 0)
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
                    f"[yellow]  \\[response cut off ({fr}) — continuing {auto_continue}/{MAX_AUTO_CONTINUE}][/yellow]",
                )
                working.append({"role": "assistant", "content": text})
                working.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous message was cut off before you finished. "
                            "Continue from exactly where you stopped — do not repeat anything "
                            "you already wrote, and do not start over."
                        ),
                    }
                )
                continue
            if usage.get("truncated") and auto_continue >= MAX_AUTO_CONTINUE:
                console.print(
                    f"[yellow]  \\[still cut off after {MAX_AUTO_CONTINUE} continuations — stopping][/yellow]",
                )
            if tool_chain:
                console.print(f"[dim]  ↳ {' → '.join(tool_chain)}[/dim]")
            _last_turn_tokens = {"prompt": total_prompt, "completion": total_completion}
            return "".join(final_segments)

        working.append(
            {
                "role": "assistant",
                "content": text or None,
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                    }
                    for tc in tool_calls
                ],
            }
        )

        # Parse args upfront; handle malformed JSON per-call
        parsed_args: list[dict | None] = []
        for tc in tool_calls:
            try:
                parsed_args.append(json.loads(tc["function"]["arguments"]))
            except json.JSONDecodeError:
                parsed_args.append(None)

        # Extract user query for summarization context
        last_user_msg = next(
            (m.get("content", "") for m in reversed(working) if m.get("role") == "user"),
            "",
        )

        # Collect results indexed by position
        tool_results: dict[int, tuple[str, str]] = {}  # idx -> (tc_id, result)

        # Classify tools into sequential batches to avoid conflicts
        batches = _classify_tool_batch(tool_calls, parsed_args)

        try:
            pool = _get_pool()
            for batch in batches:
                safe_futures: dict[int, concurrent.futures.Future] = {}

                for i in batch:
                    tc, args = tool_calls[i], parsed_args[i]
                    name = tc["function"]["name"]
                    if args is None:
                        err = (
                            f"[tool_call_error: malformed JSON arguments for '{name}'. "
                            f"Please retry the call with valid JSON.]"
                        )
                        tool_chain.append(f"{name}(!json)")
                        tool_results[i] = (tc["id"], err)
                        continue
                    if name in _PARALLEL_TOOLS:
                        tool_chain.append(f"{name}({_short_args(name, args)})")
                        safe_futures[i] = pool.submit(
                            _call_with_retry,
                            name,
                            args,
                            _call_tool_safe,
                            max_retries=_TOOL_RETRY_MAX,
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
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        idx = i
                        rname = tool_calls[idx]["function"]["name"]
                        try:
                            result = _call_with_retry(rname, parsed_args[idx], _call_tool_safe, max_retries=1)
                        except Exception as exc2:
                            result = f"[tool error: {exc}; retry: {exc2}]"
                    tool_results[i] = (
                        tool_calls[i]["id"],
                        _smart_cap(client, result, tool_calls[i]["function"]["name"], last_user_msg),
                    )

            for i in range(len(tool_calls)):
                tc_id, result = tool_results[i]
                working.append({"role": "tool", "tool_call_id": tc_id, "content": result})

            first_call = False
            depth += 1
        except KeyboardInterrupt:
            console.print("\n[dim][tools cancelled — returning to prompt][/dim]")
            for i in range(len(tool_calls)):
                working.append(
                    {"role": "tool", "tool_call_id": tool_calls[i]["id"], "content": "[user cancelled tool execution]"}
                )


# ---------------------------------------------------------------------------
# Truncate-middle — fast, no LLM call, used by auto-compact
# ---------------------------------------------------------------------------


def truncate_middle(history: list, keep_first: int = 6, keep_last: int = 20) -> list:
    """Drop middle messages to recover context space. Keeps system msgs + first K + last K chat turns."""
    sys_msgs = [m for m in history if m.get("role") == "system"]
    chat_msgs = [m for m in history if m.get("role") != "system"]

    if len(chat_msgs) <= keep_first + keep_last:
        console.print("[dim][truncate-middle: history too short to trim][/dim]")
        return history

    first = chat_msgs[:keep_first]
    last = chat_msgs[-keep_last:]
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
        f"kept {keep_first} oldest + {keep_last} most recent][/dim]",
    )
    return new_history


# ---------------------------------------------------------------------------
# /trim — sliding-window summarization (manual, LLM-based)
# ---------------------------------------------------------------------------


def cmd_trim(history: list, client: object) -> list:
    """Command: trim."""
    CHUNK = 8
    keep_count = 4
    if len(history) < keep_count + CHUNK:
        console.print("[dim][history too short to trim (need ≥12 turns)][/dim]")
        return history

    # Separate "work" turns (file writes/patches) from pure chat — keep work turns verbatim
    _work_markers = ("[patched:", "[created:", "[updated:", "[write_file]", "[patch_file]")
    work_pairs: list[tuple[int, dict, dict | None]] = []  # (orig_idx, user_msg, asst_msg)
    chat_only: list[dict] = []

    to_process_full = history[:-keep_count]
    keep = history[-keep_count:]

    i = 0
    while i < len(to_process_full):
        msg = to_process_full[i]
        if msg.get("role") == "user":
            asst = to_process_full[i + 1] if i + 1 < len(to_process_full) else None
            asst_content = (asst.get("content") or "") if asst else ""
            if any(m in asst_content for m in _work_markers):
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
        f" in chunks of {CHUNK}...[/dim]",
    )

    rolling_summary = ""
    i = 0
    chunk_num = 0
    while i < len(to_process):
        chunk = to_process[i : i + CHUNK]
        chunk_num += 1
        chunk_text = "\n".join(
            f"{m['role'].upper()}: {(m.get('content') or '')[:600]}"
            for m in chunk
            if m.get("role") in ("user", "assistant")
        )
        user_content = (
            (
                f"Previous summary: {rolling_summary}\n\n"
                f"New exchanges:\n{chunk_text}\n\n"
                "Write an updated running summary in 3-5 sentences."
            )
            if rolling_summary
            else f"Summarize this conversation in 3-5 sentences:\n\n{chunk_text}"
        )
        prompt = [
            {
                "role": "system",
                "content": (
                    "You are a concise summarizer for an ongoing work session. Preserve key "
                    "decisions, facts, outcomes, code changes made, files modified, commands run, "
                    "and their results — note what was changed and why. Then ALWAYS end with a "
                    "line exactly of the form 'CURRENT TASK: <the objective being actively worked "
                    "on right now and the immediate next step>' so the work can continue "
                    "seamlessly after older messages are dropped from context."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        console.print(Rule(f"[dim]Chunk {chunk_num}[/dim]", style="dim"))
        try:
            with Live("", console=console, vertical_overflow="crop", refresh_per_second=15, transient=True) as live:
                summary, _, _ = stream_once(client, prompt, use_tools=False, update_fn=_live_updater(live))
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
        summary_msg = {
            "role": "system",
            "content": (
                "[Earlier messages were compacted to free context. Summary of the work so far "
                "below — keep going on the CURRENT TASK noted at its end without asking the user "
                f"to repeat themselves:\n{rolling_summary}]"
            ),
        }
        # Re-insert preserved work turns in their original order
        preserved = [msg for _, u, a in sorted(work_pairs, key=lambda x: x[0]) for msg in ([u] + ([a] if a else []))]
        new_history = [summary_msg, *preserved, *keep]
        console.print(
            f"[dim][trimmed → 1 summary + {len(preserved)} preserved work turn(s) + {len(keep)} recent][/dim]",
        )
        return new_history
    return history


def _maybe_autocompact(history: list, base_system: str, client) -> list:
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
        full_msgs = [{"role": "system", "content": build_system_prompt(base_system)}, *history]
        tok = approx_tokens(full_msgs)
    pct = tok * 100 // TOKEN_LIMIT
    _track_context_growth(tok)

    compaction_threshold = _adaptive_compaction_threshold(_detect_session_type(history))
    warning_threshold = max(60, compaction_threshold - 15)

    if pct >= compaction_threshold:
        label = "auto-compact" if pct >= 90 else "auto-trim"
        console.print(
            f"[yellow]  [{label}] context at {pct}% ({tok:,} tokens) — "
            f"summarizing & preserving current task so work continues...[/yellow]",
        )
        before = len(history)
        try:
            history = cmd_trim(history, client)
        except Exception as _trim_err:
            console.print(f"[dim][summarize failed: {_trim_err} — truncating middle as fallback][/dim]")
            history = truncate_middle(history)
        # If still over after trim, escalate with progressively smaller keep values
        after_tok = approx_tokens([{"role": "system", "content": build_system_prompt(base_system)}, *history])
        if after_tok * 100 // TOKEN_LIMIT >= 85:
            history = truncate_middle(history, keep_first=4, keep_last=16)
            after_tok = approx_tokens([{"role": "system", "content": build_system_prompt(base_system)}, *history])
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
        after_tok = approx_tokens([{"role": "system", "content": build_system_prompt(base_system)}, *history])
        console.print(
            f"[green]  [{label} done] freed context to {after_tok * 100 // TOKEN_LIMIT}% ({after_tok:,} tokens)[/green]",
        )
    elif pct >= warning_threshold:
        turns_left = _estimate_turns_remaining(tok, compaction_threshold)
        turns_hint = f"~{turns_left} turns until auto-trim, " if turns_left > 0 else ""
        console.print(
            f"[yellow]  context: ~{tok:,} / {TOKEN_LIMIT:,} tokens ({pct}%) — "
            f"{turns_hint}auto-trim at {compaction_threshold}%[/yellow]",
        )
    return history


# ---------------------------------------------------------------------------
# Pipe mode
# ---------------------------------------------------------------------------


def run_piped(client: object) -> None:
    """Run in non-interactive pipe mode with full multi-round tool loop.

    stdout → clean final answer only (safe to pipe / redirect to file)
    stderr → tool status, intermediate chain-of-thought, errors
    """
    piped = sys.stdin.read().strip()
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
            {"role": "user", "content": prompt},
        ]
        depth = 0
        max_depth = 10
        while depth < max_depth:
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
                tools=TOOLS,
                tool_choice="auto",
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
                            tc_buf[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
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
            messages.append({"role": "assistant", "content": assistant_content, "tool_calls": tool_calls})
            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"] or "{}")
                except json.JSONDecodeError:
                    fn_args = {}
                result = _execute_tool_call(client, fn_name, fn_args)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": fn_name,
                        "content": str(result),
                    }
                )
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
    """Show Config."""
    t = Table(show_header=False, box=None, padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="cyan")
    t.add_row("Config file", str(CONFIG_FILE) + (" ✓" if CONFIG_FILE.exists() else " (not found)"))
    t.add_row("base_url", BASE_URL)
    t.add_row("model", MODEL)
    t.add_row("token_limit", str(TOKEN_LIMIT))
    t.add_row("max_tool_depth", str(MAX_TOOL_DEPTH))
    t.add_row("auto_search", AUTO_SEARCH_MODE)
    t.add_row("editor", DEFAULT_EDITOR)
    t.add_row("data_dir", str(DATA_DIR))
    console.print(t)
    if not CONFIG_FILE.exists():
        console.print(f"[dim]Create {CONFIG_FILE} to set persistent defaults.[/dim]")
        console.print(
            '[dim]Example:\n  base_url = "http://localhost:8000/v1"\n  model = "local-model"\n  editor = "code --wait"[/dim]'
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def make_client():  # returns OpenAI instance
    """Make Client."""
    global MODEL, ACTIVE_BACKEND
    import httpx

    _timeout = httpx.Timeout(connect=30.0, read=120.0, write=60.0, pool=10.0)
    lm_client = _get_openai()(base_url=BASE_URL, api_key="no-key", timeout=_timeout)
    try:
        lm_client.models.list()
        ACTIVE_BACKEND = "llama.cpp"
        return lm_client
    except Exception:
        _logger.debug("llama.cpp server unreachable, checking fallback")

    if OPENAI_API_KEY:
        console.print(f"[yellow]  llama.cpp server unreachable — falling back to OpenAI ({FALLBACK_MODEL})[/yellow]")
        MODEL = FALLBACK_MODEL
        ACTIVE_BACKEND = "openai"
        return _get_openai()(api_key=OPENAI_API_KEY, timeout=_timeout)

    console.print(f"[yellow]  [warning] llama.cpp server not reachable at {BASE_URL}[/yellow]")
    console.print("[dim]  Set openai_api_key in config.toml or OPENAI_API_KEY env var to enable cloud fallback.[/dim]")
    ACTIVE_BACKEND = "llama.cpp"
    return lm_client


# ==============================================================================
# Main Entry Point
# ==============================================================================


def main() -> None:
    """Main entry point for qwen-cli."""
    global \
        MODEL, \
        _backup_stack, \
        _last_user_input, \
        _branches, \
        _turn_count, \
        _current_mode, \
        _session_title, \
        _watch_thread, \
        _cli_client, \
        _real_ctx_tokens, \
        _session_start

    _session_start = time.monotonic()

    def _lsp_prewarm() -> None:
        try:
            from qwen_cli.tools import lsp as _lsp_mod

            _lsp_mod._ensure_server("")
            _lsp_mod.lsp_query("diagnostics", "")
        except Exception:
            _logger.debug("LSP server startup pre-warm failed (non-critical)")

    # Run off the main thread: even with _create_server()'s internal timeout,
    # nothing should ever block the user from reaching the prompt at startup.
    threading.Thread(target=_lsp_prewarm, daemon=True, name="lsp-prewarm").start()

    from qwen_cli.core.context import clean_old_snapshots as _clean_snaps

    _removed = _clean_snaps(keep=5)
    if _removed:
        _logger.debug("Cleaned %d old context snapshots", _removed)

    client = make_client()
    _cli_client = client

    _auto_task: str | None = None
    _cli_args = sys.argv[1:]
    if "--task" in _cli_args:
        _idx = _cli_args.index("--task")
        if _idx + 1 < len(_cli_args):
            _auto_task = _cli_args[_idx + 1]
            sys.argv = [sys.argv[0], *_cli_args[:_idx], *_cli_args[_idx + 2 :]]

    if not sys.stdin.isatty():
        run_piped(client)
        return

    if _auto_task:
        base_system = BASE_SYSTEM
        history: list[dict] = []
        _auto_task = expand_at_refs(_auto_task)
        cmd_agent(_auto_task, history, base_system, client)
        return

    base_system, history, ctx = _repl_setup(client)
    try:
        _repl_loop(ctx, history, base_system)
    except Exception as _fatal:
        _logger.exception("Fatal error in REPL loop")
        _silent_autosave(history, base_system)
        console.print(f"\n[red][fatal error] {_fatal}[/red]")
        console.print("[dim]Session autosaved. Check qwen.log for details.[/dim]")


def _print_turn_footer(elapsed: float) -> None:
    """Internal helper: print turn footer."""
    tok = _last_turn_tokens
    p, c = tok.get("prompt", 0), tok.get("completion", 0)
    if p or c:
        console.print(f"[dim]  {elapsed:.1f}s · {p:,} in / {c:,} out tok[/dim]")
    else:
        console.print(f"[dim]  {elapsed:.1f}s[/dim]")


if __name__ == "__main__":
    main()
