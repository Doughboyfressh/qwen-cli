# qwen-cli

An agentic coding CLI that runs entirely on your own hardware. It drives a local
[llama.cpp](https://github.com/ggml-org/llama.cpp) server (Qwen3.6-27B by default) with a full tool
loop: file editing, shell, web search, browser automation, LSP code intelligence, persistent memory,
and multi-agent subagents.

## Requirements

- **Python 3.10+**
- **[LM Studio](https://lmstudio.ai/)** (only for its bundled CUDA `llama-server.exe` — the CLI talks
  to the server directly, not to LM Studio)
- An NVIDIA GPU for the default 27B model. Any OpenAI-compatible endpoint works instead — see
  [Configuration](#configuration).
- Models in `~/.qwen-cli/models/`:
  - `Qwen3.6-27B-UD-Q6_K_XL.gguf` + `mmproj-F32.gguf` (main, with vision)
  - `Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf` (optional aux — see [Two servers](#two-servers))

## Install

```bash
git clone <repo> ~/.qwen-cli
cd ~/.qwen-cli
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows
# .venv/bin/python -m pip install -r requirements.txt     # Linux/macOS
```

Optional extras, each degrades gracefully if absent:

```bash
playwright install chrome     # browser_action / fetch_rendered
pip install patchright        # stronger anti-bot resistance than plain playwright
pip install pyright ruff      # post-edit diagnostics
```

## Run

```bash
start-qwen.bat        # Windows: starts the llama.cpp server(s), then the CLI
```

`start-qwen.bat` kills anything on ports 8080/8081, auto-detects the newest CUDA `llama-server.exe`,
waits for `/health`, then launches the REPL. To run the CLI against a server you started yourself:

```bash
.venv/Scripts/python qwen-cli.py
```

Non-interactive use:

```bash
echo "explain this" | qwen                 # pipe mode — clean answer on stdout
cat main.py | qwen "what does this do?"
qwen -c                                    # continue the last session
qwen -r <name>                             # fuzzy-load a saved session
```

## Two servers

The launcher starts **two** llama.cpp servers:

| | port | role |
|---|---|---|
| **main** | 8080 | your conversation. Qwen3.6-27B, vision-capable. |
| **aux** | 8081 | background work — memory extraction, summaries, session titles, tool-result condensing. |

They exist because llama.cpp serves **one request at a time**. Without the aux server, every
background memory write queues behind your next message. Aux is strictly optional: if it's down or
`aux_base_url` is empty, background work silently falls back to the main model.

## Configuration

`~/.qwen-cli/config.toml` (gitignored — it holds API keys). Every key is also settable by env var.

```toml
base_url     = "http://localhost:8080/v1"
model        = "Qwen3.6-27B"
aux_base_url = "http://localhost:8081/v1"   # "" disables the aux backend
token_limit  = 28000                        # input budget; auto-compact fires at 80%
auto_search  = "smart"                      # off | smart | aggressive
preset       = "thinking"                   # thinking | code | instruct
tool_groups  = "auto"                       # auto | all  (see Tool groups)
intel        = "off"                        # background web crawlers

# Optional: cloud fallback if the local server is unreachable
openai_api_key = "..."

# Optional: better search than the scraped default
brave_api_key  = "..."
google_api_key = "..."
google_cse_id  = "..."

# Optional: external MCP servers
[mcp.servers.github]
command = "npx"
args    = ["-y", "@modelcontextprotocol/server-github"]
```

`token_limit` must stay well under the server's `-c` minus the preset's `max_tokens` output
reservation. The defaults (`-c 49152`, 16384 out, 28000 in) leave ~4.7k of headroom for
tokenizer-estimate drift and tool schemas. **K and V cache types must match under `--flash-attn`** —
a mixed q8_0/q4_0 cache falls off the CUDA fast path and prompt processing collapses (2500 → 69 t/s,
benchmarked). See the comments in `start-qwen.bat`.

## Tool groups

The model's context is finite, and tool schemas cost real tokens (~4k of a 28k window for the full
set). So only **core** tools — files, shell, search, planning — are sent every turn. The rest load on
demand when the model calls `enable_tools(group)`, and stay loaded for the session:

| group | tools |
|---|---|
| `browser` | `browser_action` (clicks, forms, screenshots), `fetch_rendered` (JS-rendered fetch) |
| `media` | `describe_image`, `get_video_transcript` |
| `lsp` | `lsp_query` — go-to-definition, find-references, hover, symbols |
| `team` | task boards, inboxes, `team_spawn_agent` |
| `mcp` | whatever your configured MCP servers expose |

Set `tool_groups = "all"` to send everything every turn.

## Commands

`/help` lists all of them. The ones worth knowing on day one:

| | |
|---|---|
| `/agent <goal>` | autonomous loop — iterates with tools until done. Won't accept completion until it has *verified* any file it changed. |
| `/task <goal>` | plan → approve → execute → test, with retry on test failure |
| `@file.py` · `@file.py::func` | inject a file (or one symbol) into the turn |
| `/changes` · `/rollback` | what the model touched this session; undo all of it |
| `/undo` | restore the last overwritten file (10-deep) |
| `/context` · `/compact` | token breakdown; summarize old turns to free space |
| `/auto on` | apply file edits without a y/N each time (dangerous shell commands still ask) |
| `/model <name>` | switch between the main and aux backends mid-session |

Define your own: `~/.qwen-cli/commands/<name>.md` becomes `/name`, with `$ARGUMENTS` substituted.

## Memory

Facts are extracted from each exchange into `memory.md` and injected into every system prompt.
`/remember <fact>` adds one by hand, `/memory` shows the file, `/forget` clears it. It's capped and
periodically consolidated, so it can't grow without bound.

Memory and the intel feed are *snapshotted* into the system prompt rather than read live — a prompt
prefix that changes every turn defeats llama.cpp's prefix cache and forces a full re-eval of the
whole prompt. They refresh at natural cache-break points (session start, `/remember`, `/clear`,
compaction).

## Safety

- Destructive shell commands (`rm -rf`, `DROP TABLE`, `git reset --hard`, pipe-to-shell, …) always
  prompt, even under `/auto`. Both `run_command` and `run_script` use the same gate.
- Every shell execution is appended to `audit.log` with its cwd and outcome.
- File writes are backed up to `backups/` before being overwritten.
- Content arriving through a tool (web pages, command output) is treated as data, never as
  instructions — the system prompt draws that trust boundary explicitly.

## Development

```bash
.venv/Scripts/python -m pytest tests/ -q     # 632 tests
.venv/Scripts/python -m ruff check qwen_cli tests
```

Layout:

```
qwen_cli/
  main.py            entry point, tool implementations, turn loop, memory, agent modes
  core/
    config.py        constants from config.toml / env
    stream.py        tool schemas, streaming, tool-call parsing
    repl.py          interactive loop, input, file watching
    commands.py      slash-command handlers
    indexer.py       project symbol index
  tools/
    shared.py        web search, URL fetch, unified-diff application
    browser.py       Playwright automation + stealth
    lsp.py           language-server client, diagnostics
    mcp.py           MCP stdio client
    team.py          multi-agent teams, task boards, spawning
```

Runtime data (sessions, backups, memory, logs, cookies) lives in `~/.qwen-cli/` and is gitignored.
Set `QWEN_DATA_DIR` to relocate all of it — the test suite does this so it can never touch your real
memory or sessions.
