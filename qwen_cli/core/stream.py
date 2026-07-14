"""Streaming module — LLM streaming, retry, tool call parsing, live preview."""

import contextlib
import json
import logging
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

from qwen_cli.core.config import SAMPLING_PRESETS

console = Console(force_terminal=True, legacy_windows=False)
_logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_XML_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)
_XML_PARAM_RE = re.compile(r'<parameter\s+name=["\']([^"\']+)["\']>(.*?)</parameter>', re.DOTALL)
# Format C (Llama-style, Qwen drifts into it on long contexts). Both the
# unclosed and the closed spelling occur in the wild:
#   <function=name> <parameter=key> value <parameter=key2> value2 </function>
#   <function=name><parameter=key>value</parameter></function>
# The value lookahead MUST include </parameter>. Without it the closing tag was
# swallowed into the value itself — read_file(path="main.py</parameter>") is a
# file-not-found, and update_plan's steps array stopped being valid JSON, so it
# decoded to a string, failed the tool's isinstance(steps, list) check, and the
# plan was silently dropped.
_FN_EQ_RE = re.compile(r"<function=([^>\s]+)>", re.IGNORECASE)
_PARAM_EQ_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*(?=</parameter>|<parameter=|</function|$)",
    re.DOTALL | re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Tools
#
# CORE_TOOLS is always sent. TOOL_GROUPS ("browser", "media", "team") are only
# sent after the model enables them via enable_tools — their schemas cost real
# context (the full set is ~4k tokens of a 28k window) and most turns never
# touch them. Enabling is sticky for the session so the prompt prefix stays
# cache-stable. QWEN_TOOL_GROUPS=all (or config tool_groups="all") restores
# the old always-on behavior; spawned team agents run with it set.
# ---------------------------------------------------------------------------

_ALL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web. Use FIRST for anything factual — current events, prices, docs, "
                "versions, troubleshooting — rather than trusting training data."
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
                "Search recent news articles. Use instead of web_search for current events, "
                "recent releases, and anything time-sensitive."
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
                + (
                    " Runs via cmd.exe: Unix tools (grep, tail, head, wc) and bare PowerShell cmdlets "
                    "(Select-Object, Select-String) are NOT available in pipelines — use "
                    'findstr /C:"text", wrap the whole pipeline in powershell -Command "...", '
                    "or use run_script for filtering."
                    if sys.platform == "win32"
                    else ""
                )
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
            "description": (
                "Read a file. Returns at most 400 lines per call; the header states the exact "
                "range returned (e.g. 'lines 1–400 of 1263') and how to continue with offset. "
                "For large files, read only the ranges you need instead of paging through everything."
            ),
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
            "name": "edit_file",
            "description": (
                "Replace an exact string in a file. PREFER this over patch_file and write_file for "
                "targeted edits — no line numbers or diff syntax needed. Read the file first, then "
                "pass the text to change EXACTLY as it appears (same whitespace and indentation). "
                "old_string must occur exactly once in the file; include enough surrounding lines to "
                "make it unique, or set replace_all=true to change every occurrence."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path of the file to edit"},
                    "old_string": {
                        "type": "string",
                        "description": "Exact existing text to replace, copied verbatim from the file",
                    },
                    "new_string": {"type": "string", "description": "The replacement text"},
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every occurrence of old_string (default false)",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "patch_file",
            "description": (
                "Apply a unified diff patch to a file. Use edit_file for simple text replacements; "
                "use this when a single edit spans many separate locations (multi-hunk). "
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
                "Write the FULL content of a file. Use edit_file instead for targeted edits. "
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
                "Spawn a qwen agent in a new terminal for a LARGE independent subtask. All agents "
                "share one inference slot on the local server, so they run serially, not in "
                "parallel — spawn only when a subtask is substantial (10+ tool calls) and can "
                "proceed unattended; do small work inline yourself. Spawned agents get every tool. "
                "Track progress with team_board; collect results with team_inbox_receive."
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
            "name": "lsp_query",
            "description": (
                "Query a language server about code: jump to a definition, find all references to a "
                "symbol, read hover docs, or list a file's symbols. Use this instead of grepping when "
                "you need to know where something is DEFINED or everything that USES it — it resolves "
                "symbols properly rather than matching text. Line and column are 1-based. "
                "Actions: definition, references, hover, symbols, diagnostics, completion, status."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": [
                            "definition",
                            "references",
                            "hover",
                            "symbols",
                            "diagnostics",
                            "completion",
                            "status",
                        ],
                        "description": "The query to run.",
                    },
                    "file_path": {"type": "string", "description": "Path of the file to query"},
                    "line": {"type": "integer", "description": "1-based line number of the symbol"},
                    "column": {"type": "integer", "description": "1-based column number of the symbol"},
                    "new_name": {"type": "string", "description": "Unused; reserved for rename"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_plan",
            "description": (
                "Create or update your visible step plan. Call FIRST for any multi-step task; "
                "call again as statuses change (in_progress before starting a step, completed "
                "only once verified). Each call replaces the whole plan — always pass the FULL "
                "step list, one short line per step."
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
    {
        "type": "function",
        "function": {
            "name": "enable_tools",
            "description": (
                "Enable an optional tool group for the rest of this session, then call its tools "
                "directly. Groups: 'browser' (browser_action — clicks/forms/screenshots; "
                "fetch_rendered — JS-rendered read-only page fetch), 'media' (describe_image, "
                "get_video_transcript), 'lsp' (lsp_query — go-to-definition, find-references, "
                "hover docs, file symbols), 'team' (multi-agent task boards, inboxes, spawning "
                "subagents), 'mcp' (external MCP servers from config.toml, if any are configured). "
                "Call this as soon as a task needs one of those capabilities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group": {
                        "type": "string",
                        "enum": ["browser", "media", "lsp", "team", "mcp", "all"],
                        "description": "Tool group to enable ('all' for every group)",
                    },
                },
                "required": ["group"],
            },
        },
    },
]

# Partition into the always-sent core and on-demand groups (see header comment).
_GROUP_MEMBERS: dict[str, tuple[str, ...]] = {
    "browser": ("fetch_rendered", "browser_action"),
    "media": ("describe_image", "get_video_transcript"),
    "lsp": ("lsp_query",),
    "team": (
        "team_task_list",
        "team_task_add",
        "team_task_update",
        "team_inbox_send",
        "team_inbox_receive",
        "team_board",
        "team_list",
        "team_spawn_agent",
    ),
}
_GATED_NAMES = {name for members in _GROUP_MEMBERS.values() for name in members}
CORE_TOOLS = [t for t in _ALL_TOOLS if t["function"]["name"] not in _GATED_NAMES]
TOOL_GROUPS: dict[str, list] = {
    group: [t for t in _ALL_TOOLS if t["function"]["name"] in members]
    for group, members in _GROUP_MEMBERS.items()
}
# Full flat list — used by "all" mode, spawned agents, and anything that needs
# every schema regardless of gating.
TOOLS = _ALL_TOOLS

_HEARTBEAT_SEC = 30  # print "still waiting" after this many seconds with no token
_LIVE_PREVIEW_LINES = 18


def _tool_call_blocks(text: str) -> list[tuple[str, str]]:
    """Every <tool_call> block as (raw_span, body).

    Includes an UNCLOSED trailing block. Qwen truncates mid-call under long
    contexts — seen live: "<tool_call> function=update_plan> <parameter=steps>
    [{...}" with no closing tag and the JSON cut off. _XML_TOOL_CALL_RE needs a
    closing tag, so such a block was invisible here: nothing parsed it, and the
    raw markup went to the user and into history.
    """
    blocks = [(m.group(0), m.group(1).strip()) for m in _XML_TOOL_CALL_RE.finditer(text)]
    consumed = sum(len(raw) for raw, _ in blocks)
    tail = text[consumed:] if consumed else text
    lower = tail.lower()
    if "<tool_call>" in lower and "</tool_call>" not in lower:
        start = lower.index("<tool_call>")
        blocks.append((tail[start:], tail[start + len("<tool_call>") :].strip()))
    return blocks


# Qwen drops the opening angle bracket under long contexts: "function=name>" and
# "parameter=key>" instead of "<function=name>" / "<parameter=key>". Put it back
# so the format-C regexes can do their job, rather than discarding a real call.
_BARE_TAG_RE = re.compile(r"(?<!<)\b(function|parameter)=", re.IGNORECASE)
# Barest spelling of all, also seen live: "<tool_call> update_plan> steps> [...]"
# — no "function=" or "parameter=" at all, just name> key> value.
_BARE_NAME_RE = re.compile(r"^\s*<?([A-Za-z_]\w*)>\s*(.*)$", re.DOTALL)
_BARE_PARAM_RE = re.compile(r"<?([A-Za-z_]\w*)>\s*(.*?)\s*(?=<?[A-Za-z_]\w*>|</|$)", re.DOTALL)


def _decode_arg(raw: str) -> object:
    """JSON-typed values (arrays, objects, numbers) arrive as bare text — decode
    them so e.g. update_plan's steps array isn't handed over as a string."""
    val = raw.strip().rstrip("<").strip()
    try:
        return json.loads(val)
    except (json.JSONDecodeError, ValueError):
        return val


def _parse_bare_call(body: str) -> dict | None:
    """Last-resort parse of 'name> key> value key2> value2' with no tags at all."""
    m = _BARE_NAME_RE.match(body)
    if not m:
        return None
    name, rest = m.group(1).strip(), m.group(2)
    if not name or name.lower() in ("function", "parameter"):
        return None
    args = {key: _decode_arg(val) for key, val in _BARE_PARAM_RE.findall(rest)}
    return {"name": name, "arguments": args}


def _parse_xml_tool_calls(text: str) -> tuple[str, list]:
    """Extract <tool_call>...</tool_call> blocks from text.
    Handles both JSON-body and XML-param formats that Qwen3 emits.
    Returns (cleaned_text, synthetic_tool_calls).
    """
    tool_calls = []
    clean = text
    for raw_span, raw_body in _tool_call_blocks(text):
        body = _BARE_TAG_RE.sub(r"<\1=", raw_body)  # repair dropped '<' before function=/parameter=
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
        # Format C: <function=name> <parameter=key> value ...
        if tc is None:
            fn_m = _FN_EQ_RE.search(body)
            if fn_m:
                args = {key: _decode_arg(val) for key, val in _PARAM_EQ_RE.findall(body)}
                tc = {
                    "id": f"xml_{len(tool_calls)}",
                    "type": "function",
                    "function": {"name": fn_m.group(1).strip(), "arguments": json.dumps(args)},
                }
        # Format D: no tags at all — "update_plan> steps> [...]". Seen live under
        # long contexts, and the call was simply lost.
        if tc is None:
            bare = _parse_bare_call(body)
            if bare:
                tc = {
                    "id": f"xml_{len(tool_calls)}",
                    "type": "function",
                    "function": {"name": bare["name"], "arguments": json.dumps(bare["arguments"])},
                }
        if tc:
            tool_calls.append(tc)
            clean = clean.replace(raw_span, "")
    return clean.strip(), tool_calls


def _live_updater(live: Live, max_lines: int = _LIVE_PREVIEW_LINES) -> Callable[[str], None]:
    """Internal helper: live updater."""

    def update(text: str) -> None:
        """Update."""
        clean = _THINK_RE.sub("", text)
        # Tool-call markup is machinery, not prose — it should never flash up in the
        # live preview, closed or (mid-stream) not yet closed.
        clean = _XML_TOOL_CALL_RE.sub("", clean)
        if "<tool_call>" in clean.lower():
            clean = clean[: clean.lower().index("<tool_call>")]
        clean = clean.strip()
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


def _failed_usage() -> dict:
    """The usage dict for a call that produced nothing at all."""
    return {"prompt": 0, "completion": 0, "finish_reason": "error", "truncated": False}


def _build_stream_kwargs(messages: list, use_tools: bool) -> dict:
    """Request kwargs: the active preset, with manual /params overrides on top."""
    import qwen_cli.main as _main

    preset = SAMPLING_PRESETS.get(_main._active_preset, SAMPLING_PRESETS["thinking"])
    kwargs: dict = {
        # _main.MODEL, NOT the config constant: `from ...config import MODEL` binds
        # the value at import, so every request carried the configured model no
        # matter what. /model was cosmetic, and the cloud fallback was broken —
        # make_client set MODEL = "gpt-4o-mini" while the request still said
        # "Qwen3.6-27B". It only looked fine locally because llama.cpp ignores the
        # model field and serves whatever is loaded.
        "model": _main.MODEL,
        "messages": messages,
        "stream": True,
        "temperature": preset["temperature"],
        "top_p": preset["top_p"],
        "presence_penalty": preset["presence_penalty"],
        "max_tokens": preset["max_tokens"],
    }
    # llama.cpp-only samplers (top_k / min_p / repeat_penalty) and Qwen's
    # chat_template_kwargs. Cloud APIs reject unknown fields, so only send them to
    # a backend that wants them; _create_with_retry can still strip them if a
    # server we thought was local turns out to be picky.
    if _main.SAMPLER_EXTRAS:
        extra = dict(preset["extra_body"])  # copy so retries can pop it safely
        if not _main.PRESERVE_THINKING:
            # A Qwen chat-template argument. A Llama/Mistral/Gemma template has no
            # such parameter and llama.cpp can reject the whole request for it.
            extra.pop("chat_template_kwargs", None)
        kwargs["extra_body"] = extra
    for k in ("temperature", "top_p", "max_tokens", "presence_penalty"):
        if k in _main._model_params:
            kwargs[k] = _main._model_params[k]
    if _main._stream_usage_supported:
        kwargs["stream_options"] = {"include_usage": True}
    if use_tools:
        kwargs["tools"] = _main.active_tools()
        kwargs["tool_choice"] = "auto"
    return kwargs


@contextlib.contextmanager
def _heartbeat() -> Iterator[Callable[[], None]]:
    """Print a "still waiting" line when the server goes quiet mid-generation.

    Yields a tick() the caller invokes on each chunk; without it the thread has
    no way to tell a slow stream from a hung one.
    """
    stop = threading.Event()
    state = {"last_chunk": time.monotonic(), "last_print": 0.0}

    def tick() -> None:
        state["last_chunk"] = time.monotonic()

    def loop() -> None:
        while not stop.wait(timeout=5):
            now = time.monotonic()
            silent_for = now - state["last_chunk"]
            if silent_for > _HEARTBEAT_SEC and now - state["last_print"] > _HEARTBEAT_SEC:
                state["last_print"] = now
                console.print(f"\n[dim]  still waiting for response ({int(silent_for)}s)...[/dim]")

    threading.Thread(target=loop, daemon=True).start()
    try:
        yield tick
    finally:
        stop.set()


def _accumulate_tool_calls(tc_buf: dict[int, dict], deltas) -> None:
    """Fold one chunk's tool_call deltas into the buffer.

    Names and JSON argument strings arrive split across chunks, so both are
    concatenated, never assigned.
    """
    for tc in deltas:
        slot = tc_buf.setdefault(
            tc.index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if tc.id:
            slot["id"] = tc.id
        if tc.function:
            if tc.function.name:
                slot["function"]["name"] += tc.function.name
            if tc.function.arguments:
                slot["function"]["arguments"] += tc.function.arguments


def _drain_stream(stream, update_fn) -> tuple[str, list, dict]:
    """Consume the stream. Returns (text, tool_calls, usage).

    A stream that dies mid-flight keeps whatever it had already produced: the
    partial reply is worth more than nothing, and usage["truncated"] tells the
    caller the answer is incomplete so run_turn can resume it. Tool calls are
    dropped in that case — a half-streamed call has a truncated JSON argument
    string and is unusable.
    """
    content_parts: list[str] = []
    tc_buf: dict[int, dict] = {}
    usage: dict = {}
    finish_reason: str | None = None
    interrupted = False

    with _heartbeat() as tick:
        try:
            for chunk in stream:
                tick()
                if getattr(chunk, "usage", None):
                    usage = {
                        "prompt": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                        "completion": getattr(chunk.usage, "completion_tokens", 0) or 0,
                    }
                if not chunk.choices:
                    continue
                finish_reason = getattr(chunk.choices[0], "finish_reason", None) or finish_reason
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                    if update_fn is not None:
                        update_fn("".join(content_parts))
                if delta.tool_calls:
                    _accumulate_tool_calls(tc_buf, delta.tool_calls)
        except Exception as e:
            if not content_parts:
                return "", [], _failed_usage()
            interrupted = True
            console.print(
                f"\n[yellow]  [stream interrupted — keeping partial reply ({type(e).__name__})][/yellow]"
            )

    # Record how the generation ended so callers can tell a *complete* answer
    # apart from one the model was cut off in the middle of. "length" means the
    # token cap stopped it; an interrupted stream is likewise incomplete.
    usage["finish_reason"] = "interrupted" if interrupted else finish_reason
    usage["truncated"] = interrupted or finish_reason == "length"
    return "".join(content_parts), ([] if interrupted else list(tc_buf.values())), usage


def _recover_xml_tool_calls(text: str, api_calls: list, use_tools: bool) -> tuple[str, list]:
    """Handle the model writing tool calls as text instead of using the API field.

    With tools enabled that's a Qwen3 quirk worth honouring: parse them out and
    run them. With tools DISABLED (the forced no-tools synthesis round after the
    tool budget is spent) there is nothing to run, so strip the markup rather
    than leak raw <tool_call> tags to the user and into history.
    """
    if "<tool_call>" not in text.lower():
        return text, api_calls
    if api_calls and use_tools:
        return text, api_calls

    clean_text, xml_calls = _parse_xml_tool_calls(text)
    if not xml_calls:
        # Tool-call markup none of formats A-D could parse. Strip it rather than
        # render raw tags to the user and store them in history.
        stripped = _XML_TOOL_CALL_RE.sub("", text)
        if "<tool_call>" in stripped.lower():  # unclosed block — cut from the tag on
            stripped = stripped[: stripped.lower().index("<tool_call>")]
        stripped = stripped.strip()
        if stripped == text.strip():
            return stripped, api_calls
        console.print("[dim]  [dropped an unparseable tool call][/dim]")
        if stripped or use_tools:
            # With tools live, an empty reply is fine: run_turn's empty-reply nudge
            # asks the model to answer or call a tool properly.
            return stripped, api_calls
        # Tools are OFF (the forced synthesis after the tool budget is spent) and
        # the markup was the ENTIRE reply. Returning "" here loses the whole turn:
        # run_turn's depth-cap path yields None and the REPL drops it, so the user
        # gets nothing at all — which is exactly what happened on a live audit that
        # spent all 20 rounds gathering data and then printed "(no synthesis)".
        # Say something, so the turn survives and the user learns why.
        return (
            "[I ran out of tool rounds and my final answer came back as a malformed tool call, so "
            "there is no report here. Ask me to continue and I will summarize what I already found.]",
            api_calls,
        )
    if use_tools:
        return clean_text, xml_calls

    names = ", ".join(tc["function"]["name"] for tc in xml_calls)
    console.print(
        f"[dim]  [dropped {len(xml_calls)} tool call(s) ({names}) — tool use unavailable for this reply][/dim]"
    )
    return clean_text or f"[ran out of tool rounds before finishing — wanted to call: {names}]", api_calls


def stream_once(client: object, messages: list, use_tools: bool, update_fn=None) -> tuple[str, list, dict]:
    """Returns (text, tool_calls, usage_dict)."""
    kwargs = _build_stream_kwargs(messages, use_tools)
    try:
        stream = _create_with_retry(client, **kwargs)
    except Exception as e:
        # Nothing has streamed yet, so there is no partial reply to salvage. The
        # old code had a recovery branch here that rebuilt a reply from
        # content_parts — dead code: content_parts cannot be non-empty before the
        # first chunk arrives, so its guard never passed.
        console.print(f"\n[yellow]  [stream error — {type(e).__name__}: {e}][/yellow]")
        return "", [], _failed_usage()
    if stream is None:
        return "", [], _failed_usage()

    text, api_calls, usage = _drain_stream(stream, update_fn)
    text, api_calls = _recover_xml_tool_calls(text, api_calls, use_tools)
    return text, api_calls, usage


def _short_args(name: str, args: dict) -> str:
    """Internal helper: short args."""
    if name in ("web_search", "search_news"):
        return repr(args.get("query", "")[:40])
    if name == "fetch_url":
        return repr(args.get("url", "")[:50])
    if name in ("read_file", "write_file", "patch_file", "edit_file"):
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
