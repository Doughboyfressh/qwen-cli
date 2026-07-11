"""Streaming module — LLM streaming, retry, tool call parsing, live preview."""

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from qwen_cli.core.config import MODEL, SAMPLING_PRESETS

console = Console(force_terminal=True, legacy_windows=False)
_logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_XML_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_XML_PARAM_RE = re.compile(r'<parameter\s+name=["\']([^"\']+)["\']>(.*?)</parameter>', re.DOTALL)

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
        },
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
                    "query": {"type": "string", "description": "News search query"},
                    "max_results": {"type": "integer", "description": "Max results (default 8)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch the raw text content of a URL (plain HTTP, no JavaScript rendering).",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Max characters to return (default 20000)"},
                },
                "required": ["url"],
            },
        },
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
                    "url": {"type": "string", "description": "URL to fetch"},
                    "max_chars": {"type": "integer", "description": "Max characters to return (default 15000)"},
                },
                "required": ["url"],
            },
        },
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
        },
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
                    "url": {"type": "string", "description": "URL of the YouTube video or video page"},
                    "lang": {"type": "string", "description": "Preferred transcript language code (default: en)"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browser_action",
            "description": (
                "Control a real Chromium browser to automate web interactions: "
                "navigate to pages, fill forms, click buttons, select dropdowns, submit forms, "
                "scroll pages, hover over elements, press keyboard shortcuts, take screenshots, "
                "read page text, extract all links, upload files, run JS, or get the current URL. "
                "Use this when a page requires JavaScript interaction. "
                "If a click/submit opens a new tab (target=_blank link, OAuth popup, window.open), "
                "that tab automatically becomes the active page for subsequent calls. "
                "Actions: navigate, fill, type, click, select, submit, wait_for, "
                "scroll (value=pixels or selector=element), hover, "
                "press_key (value=key e.g. Enter/Tab/Escape/Control+A), "
                "go_back, go_forward, "
                "evaluate (value=JS expression, returns JSON-serialized result), "
                "upload_file (selector=file input, value=local file path), "
                "screenshot, get_text (selector optional), get_url, get_links, close."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "navigate",
                            "fill",
                            "type",
                            "click",
                            "select",
                            "submit",
                            "wait_for",
                            "scroll",
                            "hover",
                            "press_key",
                            "screenshot",
                            "get_text",
                            "get_url",
                            "get_links",
                            "go_back",
                            "go_forward",
                            "evaluate",
                            "upload_file",
                        ],
                        "description": "The browser action to perform.",
                    },
                    "url": {"type": "string", "description": "URL to navigate to (for 'navigate')."},
                    "selector": {
                        "type": "string",
                        "description": (
                            "CSS selector or semantic shorthand: 'label:Text', 'button:Name', 'link:Name', 'text:Text'. "
                            "Used by fill, type, click, select, wait_for, hover, press_key, get_text, scroll, upload_file "
                            "(file input element)."
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": (
                            "Text to fill/type, option label for select, "
                            "pixels to scroll (negative = up), key for press_key (e.g. 'Enter', 'Control+A'), "
                            "JS expression for evaluate, or local file path for upload_file."
                        ),
                    },
                    "screenshot_path": {
                        "type": "string",
                        "description": "File path to save screenshot. Defaults to ~/screenshot.png.",
                    },
                },
                "required": ["action"],
            },
        },
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
                    "command": {"type": "string", "description": "Shell command to run"},
                    "cwd": {"type": "string", "description": "Working directory (default: current)"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30, max 300)"},
                    "env": {
                        "type": "object",
                        "description": 'Extra environment variables to set, e.g. {"DEBUG": "1", "NODE_ENV": "production"}',
                        "additionalProperties": {"type": "string"},
                    },
                    "stdin": {"type": "string", "description": "Text to pipe into the command's stdin"},
                },
                "required": ["command"],
            },
        },
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
                    "language": {
                        "type": "string",
                        "description": "Language name: python, javascript, bash, powershell, ruby, php, go, etc.",
                    },
                    "code": {"type": "string", "description": "The source code to execute"},
                    "cwd": {"type": "string", "description": "Working directory (default: current)"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
                },
                "required": ["language", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file with syntax highlighting. Use offset/limit for a range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path"},
                    "offset": {"type": "integer", "description": "1-based line number to start from (optional)"},
                    "limit": {"type": "integer", "description": "Max lines to read (optional)"},
                },
                "required": ["path"],
            },
        },
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
        },
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
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Full content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the contents of a directory. Set recursive=true for a full tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "recursive": {"type": "boolean", "description": "Show full tree (default false)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_files",
            "description": "Find files matching a glob pattern inside a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Root directory to search"},
                    "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'"},
                },
                "required": ["path", "pattern"],
            },
        },
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
                    "path": {"type": "string", "description": "Root directory to search"},
                    "query": {"type": "string", "description": "Text or regex pattern to find"},
                    "pattern": {"type": "string", "description": "Glob to filter files (default '**/*')"},
                    "context": {
                        "type": "integer",
                        "description": "Lines of context before/after each match (default 0)",
                    },
                },
                "required": ["path", "query"],
            },
        },
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
        },
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
        },
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
        },
    },
    {
        "type": "function",
        "function": {
            "name": "team_task_list",
            "description": "List tasks for a team. Filter by owner or status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team name"},
                    "owner": {"type": "string", "description": "Filter by owner agent name (optional)"},
                    "status": {
                        "type": "string",
                        "description": "Filter by status: pending, in_progress, completed, blocked (optional)",
                    },
                },
                "required": ["team"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "team_task_add",
            "description": "Add a new task to a team's task board.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team name"},
                    "subject": {"type": "string", "description": "Task description"},
                    "owner": {"type": "string", "description": "Assign to this agent (optional)"},
                    "priority": {"type": "string", "description": "Priority: low, medium, high (default: medium)"},
                },
                "required": ["team", "subject"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "team_task_update",
            "description": "Update a task's status, owner, or add a progress note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team name"},
                    "task_id": {"type": "string", "description": "Task ID (first 6+ chars ok)"},
                    "status": {"type": "string", "description": "New status: pending, in_progress, completed, blocked"},
                    "owner": {"type": "string", "description": "Assign to this agent (optional)"},
                    "note": {"type": "string", "description": "Add a timestamped note to the task (optional)"},
                },
                "required": ["team", "task_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "team_inbox_send",
            "description": "Send a message to a team member's inbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team name"},
                    "to": {"type": "string", "description": "Recipient agent name"},
                    "message": {"type": "string", "description": "Message body"},
                    "from_agent": {"type": "string", "description": "Sender name (default: user)"},
                },
                "required": ["team", "to", "message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "team_inbox_receive",
            "description": "Read messages from a team member's inbox. By default consumes them; use peek=true to read without removing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team": {"type": "string", "description": "Team name"},
                    "agent": {"type": "string", "description": "Your agent name"},
                    "peek": {
                        "type": "boolean",
                        "description": "If true, read without consuming messages (default: false)",
                    },
                },
                "required": ["team", "agent"],
            },
        },
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
        },
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
        },
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
                    "team": {"type": "string", "description": "Team name"},
                    "agent_name": {"type": "string", "description": "Name for the new agent"},
                    "task": {"type": "string", "description": "Task description for the agent"},
                    "cwd": {"type": "string", "description": "Working directory (default: current)"},
                },
                "required": ["team", "agent_name", "task"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": (
                "Create or update your visible step-by-step plan for the current task. "
                "Call this FIRST for any non-trivial multi-step task, listing every step with "
                "status 'pending'. Call it again whenever a step's status changes — mark a step "
                "'in_progress' right before you start it, and 'completed' only after you've verified "
                "it actually worked. Always pass the FULL list of steps (not just the one that "
                "changed) — each call replaces the whole plan. Keep steps short (one line each)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "description": "The complete ordered list of steps for the current task.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string", "description": "Short description of the step"},
                                "status": {
                                    "type": "string",
                                    "description": "One of: pending, in_progress, completed",
                                },
                            },
                            "required": ["text", "status"],
                        },
                    },
                },
                "required": ["steps"],
            },
        },
    },
]

_HEARTBEAT_SEC = 30  # print "still waiting" after this many seconds with no token
_LIVE_PREVIEW_LINES = 18


def _parse_xml_tool_calls(text: str) -> tuple[str, list]:
    """Extract <tool_call>...</tool_call> blocks from text.
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
                tc = {
                    "id": f"xml_{len(tool_calls)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args) if isinstance(args, dict) else args},
                }
        except (json.JSONDecodeError, AttributeError):
            pass
        # Format B: <function_name>x</function_name><parameter name="k">v</parameter>
        if tc is None:
            fn_m = re.search(r"<function_name>(.*?)</function_name>", body, re.DOTALL)
            if fn_m:
                name = fn_m.group(1).strip()
                args = dict(_XML_PARAM_RE.findall(body))
                tc = {
                    "id": f"xml_{len(tool_calls)}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
        if tc:
            tool_calls.append(tc)
            clean = clean.replace(m.group(0), "")
    return clean.strip(), tool_calls


def _live_updater(live: Live, max_lines: int = _LIVE_PREVIEW_LINES) -> Callable[[str], None]:
    """Internal helper: live updater."""

    def update(text: str) -> None:
        """Update."""
        clean = _THINK_RE.sub("", text).strip()
        if not clean:
            live.update(Text("\u27b3 thinking\u2026", style="dim italic"))
            return
        split = clean.split("\n")
        preview = ("\u2026\n" + "\n".join(split[-max_lines:])) if len(split) > max_lines else clean
        live.update(Markdown(preview))

    return update


def _create_with_retry(client: object, **kwargs) -> Any:
    """Create a chat completion with exponential back-off on connection errors."""
    import qwen_cli.main as _main

    # Try without stream_options if previously unsupported
    if not _main._stream_usage_supported:
        kwargs.pop("stream_options", None)

    for attempt in range(4):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            err = str(e).lower()
            is_conn = any(s in err for s in ("connection", "timeout", "refused", "unreachable", "reset by peer", "eof"))
            is_param = any(
                s in err
                for s in (
                    "unknown field",
                    "extra inputs",
                    "unrecognized",
                    "invalid param",
                    "stream_options",
                    "extra_body",
                )
            )
            if is_param:
                stripped = False
                if "stream_options" in kwargs:
                    _main._stream_usage_supported = False
                    kwargs.pop("stream_options")
                    stripped = True
                if "extra_body" in kwargs:
                    kwargs.pop("extra_body")
                    stripped = True
                if stripped:
                    continue
            if is_conn and attempt < 3:
                wait = 2**attempt
                console.print(f"[yellow]  [connection error \u2014 retry {attempt + 1}/3 in {wait}s: {e}][/yellow]")
                time.sleep(wait)
            else:
                raise
    return None


def stream_once(client: object, messages: list, use_tools: bool, update_fn=None) -> tuple[str, list, dict]:
    """Returns (text, tool_calls, usage_dict)."""
    import qwen_cli.main as _main

    content_parts: list[str] = []
    tc_buf: dict[int, dict] = {}
    usage: dict = {}
    finish_reason: str | None = None

    # Build kwargs from active preset, then layer manual _model_params on top
    preset = SAMPLING_PRESETS.get(_main._active_preset, SAMPLING_PRESETS["thinking"])
    kwargs: dict = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "temperature": preset["temperature"],
        "top_p": preset["top_p"],
        "presence_penalty": preset["presence_penalty"],
        "max_tokens": preset["max_tokens"],
        "extra_body": dict(preset["extra_body"]),  # copy so retries can pop it safely
    }
    # Manual overrides win over preset
    for k in ("temperature", "top_p", "max_tokens", "presence_penalty"):
        if k in _main._model_params:
            kwargs[k] = _main._model_params[k]
    if _main._stream_usage_supported:
        kwargs["stream_options"] = {"include_usage": True}
    if use_tools:
        kwargs["tools"] = TOOLS
        kwargs["tool_choice"] = "auto"

    interrupted = False
    try:
        stream = _create_with_retry(client, **kwargs)
    except Exception as e:
        if not content_parts:
            return "", [], {"prompt": 0, "completion": 0, "finish_reason": "error", "truncated": False}
        interrupted = True
        console.print(f"\n[yellow]  [stream error \u2014 {type(e).__name__}: {e}][/yellow]")
        full_text = "".join(content_parts)
        api_calls = [] if interrupted else list(tc_buf.values())
        usage["finish_reason"] = "interrupted" if interrupted else finish_reason
        usage["truncated"] = interrupted or finish_reason == "length"
        if not api_calls and use_tools and "<tool_call>" in full_text.lower():
            clean_text, xml_calls = _parse_xml_tool_calls(full_text)
            if xml_calls:
                return clean_text, xml_calls, usage
        return full_text, api_calls, usage
    if stream is None:
        return "", [], {"prompt": 0, "completion": 0, "finish_reason": "error", "truncated": False}

    _hb_stop = threading.Event()
    _last_chunk_time = time.monotonic()
    _last_hb_print = 0.0

    def _hb_loop() -> None:
        nonlocal _last_hb_print
        while not _hb_stop.wait(timeout=5):
            elapsed = time.monotonic() - _last_chunk_time
            if elapsed > _HEARTBEAT_SEC and time.monotonic() - _last_hb_print > _HEARTBEAT_SEC:
                _last_hb_print = time.monotonic()
                console.print(f"\n[dim]  still waiting for response ({int(elapsed)}s)...[/dim]")

    threading.Thread(target=_hb_loop, daemon=True).start()

    try:
        for chunk in stream:
            _last_chunk_time = time.monotonic()
            if hasattr(chunk, "usage") and chunk.usage:
                usage = {
                    "prompt": getattr(chunk.usage, "prompt_tokens", 0) or 0,
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
                        tc_buf[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                    if tc.id:
                        tc_buf[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tc_buf[idx]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            tc_buf[idx]["function"]["arguments"] += tc.function.arguments
    except Exception as e:
        if not content_parts:
            return "", [], {"prompt": 0, "completion": 0, "finish_reason": "error", "truncated": False}
        interrupted = True
        console.print(f"\n[yellow]  [stream interrupted \u2014 keeping partial reply ({type(e).__name__})][/yellow]")
    finally:
        _hb_stop.set()

    full_text = "".join(content_parts)
    api_calls = [] if interrupted else list(tc_buf.values())

    # Record how the generation ended so callers can tell a *complete* answer
    # apart from one the model was cut off in the middle of. "length" means the
    # token cap stopped it; an interrupted stream is likewise incomplete.
    usage["finish_reason"] = "interrupted" if interrupted else finish_reason
    usage["truncated"] = interrupted or finish_reason == "length"

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


def _strip_think(text: str) -> tuple[str, str]:
    """Extract <think>\u2026</think> blocks. Returns (clean_text, think_content)."""
    parts = _THINK_RE.findall(text)
    if not parts:
        return text, ""
    clean = _THINK_RE.sub("", text).strip()
    think = "\n\n---\n\n".join(p.strip() for p in parts if p.strip())
    return clean, think
