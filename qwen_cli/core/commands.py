"""Slash command handlers for the REPL loop."""

import contextlib
import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

_logger = logging.getLogger(__name__)

# Import ReplContext from repl (used in function signatures)
from qwen_cli.core.repl import _ReplContext  # noqa: E402

# ==============================================================================
# Command Handler Functions
# ==============================================================================
# Signature: handler(ctx, arg) -> None
# To break the REPL loop, handlers raise StopIteration.
# ==============================================================================


def _cmd_exit(ctx: _ReplContext, arg: str) -> None:
    """Save the current session and exit the REPL loop."""
    import qwen_cli.main as _main

    if ctx.history:
        _main.save_session(ctx.history, ctx.base_system)
    with contextlib.suppress(Exception):
        _main.record_session_changes_memory(ctx.client)
    _main.console.print("[dim]Bye.[/dim]")
    raise StopIteration


def _cmd_help(ctx: _ReplContext, arg: str) -> None:
    """Display available commands, or search the help text for a keyword."""
    import qwen_cli.main as _main

    if arg:
        q = arg.lower()
        rows = []
        for ln in _main.HELP_TEXT.splitlines():
            s = ln.strip()
            if not s.startswith("|") or "---" in s or s == "| Command | Description |":
                continue
            if q in s.lower():
                rows.append(s)
        if rows:
            _main.console.print(
                Markdown(
                    "| Command | Description |\n|---|---|\n" + "\n".join(rows),
                )
            )
        else:
            _main.console.print(f"[yellow][no help entries match '{arg}' — try /help with no argument][/yellow]")
    else:
        _main.console.print(Markdown(_main.HELP_TEXT))


def _cmd_clear(ctx: _ReplContext, arg: str) -> None:
    """Clear the conversation history, optionally dropping only the last N turns."""
    import qwen_cli.main as _main

    if arg.isdigit():
        n = int(arg)
        drop = min(n * 2, len(ctx.history))
        del ctx.history[-drop:]
        _main.console.print(f"[dim][dropped {drop // 2} turn(s)][/dim]")
    else:
        ctx.history.clear()
        _main._current_plan.clear()
        _main.console.print("[dim][history cleared][/dim]")


def _cmd_retry(ctx: _ReplContext, arg: str) -> None:
    """Remove the last assistant reply and re-run the last user input."""
    import qwen_cli.main as _main

    if not _main._last_user_input:
        _main.console.print("[yellow][nothing to retry][/yellow]")
        return
    user_input = _main._last_user_input
    if ctx.history and ctx.history[-1]["role"] == "assistant":
        ctx.history.pop()
    if ctx.history and ctx.history[-1]["role"] == "user" and ctx.history[-1]["content"] == user_input:
        ctx.history.pop()
    _main.console.print(f"[dim][retrying: {user_input[:60]}{'…' if len(user_input) > 60 else ''}][/dim]")
    _main._run_turn_and_handle_reply(ctx, user_input)


def _cmd_run(ctx: _ReplContext, arg: str) -> None:
    """Execute a shell command, with a safety prompt for dangerous commands."""
    import qwen_cli.main as _main

    if not arg:
        _main.console.print("[yellow][usage: /run <command>][/yellow]")
        return
    if _main._is_dangerous(arg):
        _main.console.print(f"[bold red]  dangerous command:[/bold red] {arg}")
        answer = _main.console.input("[bold red]  Run anyway? [y/N]:[/bold red] ").strip().lower()
        if answer != "y":
            _main.console.print("[dim][cancelled][/dim]")
            return
    _main.do_run_command(arg)


def _cmd_edit(ctx: _ReplContext, arg: str) -> None:
    """Open a file in the default editor ($EDITOR)."""
    import qwen_cli.main as _main

    if not arg:
        _main.console.print("[yellow][usage: /edit <file>][/yellow]")
        return
    if ".." in Path(arg).parts:
        _main.console.print("[red][edit blocked: path traversal (.. ) detected][/red]")
        return
    p = _main._resolve(arg)
    try:
        subprocess.run([_main.DEFAULT_EDITOR, str(p)], shell=False)
        _main.console.print(f"[dim][editor closed: {p.name}][/dim]")
    except Exception as e:
        _main.console.print(f"[red][error opening editor: {e}][/red]")


def _cmd_system(ctx: _ReplContext, arg: str) -> None:
    """Display or replace the system prompt; clears history when changed."""
    import qwen_cli.main as _main

    if not arg:
        _main.console.print(f"[dim]{ctx.base_system}[/dim]")
    else:
        ctx.base_system = arg
        ctx.history.clear()
        _main.console.print("[dim][system prompt updated, history cleared][/dim]")


def _cmd_file(ctx: _ReplContext, arg: str) -> None:
    """Load the contents of a file into the conversation context."""
    import qwen_cli.main as _main

    if not arg:
        _main.console.print("[yellow][usage: /file <path>][/yellow]")
    else:
        _main.load_file_into_context(arg, ctx.history)


def _cmd_project(ctx: _ReplContext, arg: str) -> None:
    """Load project context (project_system.md, git status, file tree) into the conversation."""
    import qwen_cli.main as _main

    _main.load_project_context(arg, ctx.history)


def _cmd_focus(ctx: _ReplContext, arg: str) -> None:
    """Narrow project context to a specific subdirectory or file pattern."""
    import qwen_cli.main as _main

    _main.cmd_focus(arg, ctx.history)


def _cmd_changes(ctx: _ReplContext, arg: str) -> None:
    """Show recent file changes in the current project (via git diff or filesystem)."""
    import qwen_cli.main as _main

    _main.cmd_changes(arg)


def _cmd_search_sessions(ctx: _ReplContext, arg: str) -> None:
    """Search saved session files for matching text."""
    import qwen_cli.main as _main

    _main.cmd_search_sessions(arg)


def _cmd_undo(ctx: _ReplContext, arg: str) -> None:
    """Restore the most recently backed-up file from the undo stack."""
    import qwen_cli.main as _main

    if not _main._backup_stack:
        _main.console.print("[yellow][no backup available][/yellow]")
    else:
        entry = _main._backup_stack.pop()
        orig: Path = entry["original"]
        bak: Path = entry["backup"]
        orig.write_text(entry["content"], encoding="utf-8")
        _main.console.print(f"[green][restored: {orig.name} from {bak.name}][/green]")
        remaining = len(_main._backup_stack)
        if remaining:
            _main.console.print(f"[dim]  ({remaining} more undo level{'s' if remaining != 1 else ''} available)[/dim]")


def _cmd_export(ctx: _ReplContext, arg: str) -> None:
    """Export the current session to a Markdown file."""
    import qwen_cli.main as _main

    _main.export_session(ctx.history, arg)


def _cmd_copy(ctx: _ReplContext, arg: str) -> None:
    """Copy the last assistant message to the clipboard."""
    import qwen_cli.main as _main

    _main.cmd_copy(ctx.history)


def _cmd_paste(ctx: _ReplContext, arg: str) -> None:
    """Read text from the clipboard and append it to the conversation history."""
    import qwen_cli.main as _main

    _main.cmd_paste(ctx.history)


def _cmd_remember(ctx: _ReplContext, arg: str) -> None:
    """Append a fact to persistent memory (memory.md)."""
    import qwen_cli.main as _main

    if not arg:
        _main.console.print("[yellow][usage: /remember <fact>][/yellow]")
    else:
        with _main._memory_lock:
            mem = _main.load_memory()
            _main.save_memory((mem + f"\n- {arg}").strip())
        _main.console.print("[green][remembered — persists across sessions][/green]")


def _cmd_memory(ctx: _ReplContext, arg: str) -> None:
    """Display the current persistent memory contents."""
    import qwen_cli.main as _main

    mem = _main.load_memory()
    _main.console.print(Markdown(mem) if mem else "[dim][memory is empty][/dim]")


def _cmd_forget(ctx: _ReplContext, arg: str) -> None:
    """Clear all persistent memory after user confirmation."""
    import qwen_cli.main as _main

    try:
        confirm = _main.console.input("[red]Clear ALL persistent memory? [y/N]: [/red]").strip().lower()
    except (KeyboardInterrupt, EOFError):
        confirm = ""
    if confirm == "y":
        with _main._memory_lock:
            _main.save_memory("")
        _main.console.print("[dim][persistent memory cleared][/dim]")
    else:
        _main.console.print("[dim][cancelled][/dim]")


def _cmd_pin(ctx: _ReplContext, arg: str) -> None:
    """Manage pinned items: list, add, or remove pins that persist across sessions."""
    import qwen_cli.main as _main

    pins = _main.load_pins()
    if not arg or arg == "list":
        if not pins:
            _main.console.print("[dim][no pins][/dim]")
        else:
            for i, p in enumerate(pins, 1):
                _main.console.print(f"  [cyan]{i}.[/cyan] {p}")
    elif arg.startswith("remove "):
        try:
            idx = int(arg.split(None, 1)[1]) - 1
            removed = pins.pop(idx)
            _main.save_pins(pins)
            _main.console.print(f"[dim][removed pin: {removed}][/dim]")
        except (ValueError, IndexError):
            _main.console.print("[yellow][usage: /pin remove <number>][/yellow]")
    else:
        pins.append(arg)
        _main.save_pins(pins)
        _main.console.print(f"[green][pinned: {arg}][/green]")


def _cmd_branch(ctx: _ReplContext, arg: str) -> None:
    """Manage conversation branches: list, save current state, or restore a saved branch."""
    import qwen_cli.main as _main

    sub_parts = arg.split(None, 1)
    sub = sub_parts[0] if sub_parts else ""
    sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""

    if sub == "list" or not sub:
        if not _main._branches:
            _main.console.print("[dim][no branches][/dim]")
        else:
            for name, h in _main._branches.items():
                turns = sum(1 for m in h if m.get("role") == "assistant")
                _main.console.print(f"  [cyan]{name}[/cyan]  [dim]{turns} turns[/dim]")
    elif sub == "restore":
        if not sub_arg:
            _main.console.print("[yellow][usage: /branch restore <name>][/yellow]")
        elif sub_arg not in _main._branches:
            _main.console.print(f"[red][no branch: {sub_arg}][/red]")
        else:
            ctx.history = list(_main._branches[sub_arg])
            turns = sum(1 for m in ctx.history if m.get("role") == "assistant")
            _main.console.print(f"[green][restored branch '{sub_arg}' — {turns} turns][/green]")
    else:
        _main._branches[sub] = list(ctx.history)
        turns = sum(1 for m in ctx.history if m.get("role") == "assistant")
        _main.console.print(f"[green][branch '{sub}' saved — {turns} turns][/green]")
        _main.console.print(f"[dim]  restore with: /branch restore {sub}[/dim]")


def _cmd_search(ctx: _ReplContext, arg: str) -> None:
    """Perform a web search and feed the results into the model for a response."""
    import qwen_cli.main as _main

    if not arg:
        _main.console.print("[yellow][usage: /search <query>][/yellow]")
        return
    _main.console.print(f"[dim cyan][searching: {arg}][/dim cyan]")
    result = _main.do_web_search(arg)
    _main.console.print(Markdown(f"```\n{result}\n```"))
    search_input = f"[Web search results for: {arg}]\n{result}\n\nBased on these results, answer the query: {arg}"
    _main._run_turn_and_handle_reply(ctx, search_input)


def _cmd_save(ctx: _ReplContext, arg: str) -> None:
    """Save the current session to a JSON file."""
    import qwen_cli.main as _main

    _main.save_session(ctx.history, ctx.base_system, arg or None)


def _cmd_load(ctx: _ReplContext, arg: str) -> None:
    """Load a previously saved session into the current context."""
    import qwen_cli.main as _main

    result = _main.cmd_load_session(arg, ctx.history, ctx.base_system)
    ctx.history, ctx.base_system = result


def _cmd_sessions(ctx: _ReplContext, arg: str) -> None:
    """List all saved sessions."""
    import qwen_cli.main as _main

    _main.list_sessions()


def _cmd_context(ctx: _ReplContext, arg: str) -> None:
    """Show a token-count breakdown of the current context (system prompt + history)."""
    import qwen_cli.main as _main

    _main.show_context_breakdown(ctx.base_system, ctx.history)


def _cmd_trim(ctx: _ReplContext, arg: str) -> None:
    """Trim the conversation history using importance-aware truncation."""
    import qwen_cli.main as _main

    ctx.history = _main.cmd_trim(ctx.history, ctx.client)


def _cmd_mode(ctx: _ReplContext, arg: str) -> None:
    """Set or clear an operational mode (e.g., coding, writing, analysis)."""
    import qwen_cli.main as _main

    if not arg:
        modes_list = " / ".join(_main._MODE_PROMPTS.keys())
        if _main._current_mode:
            _main.console.print(
                f"[dim][mode: {_main._current_mode} — {_main._MODE_PROMPTS[_main._current_mode]}][/dim]"
            )
        else:
            _main.console.print(f"[dim][mode: default — available: {modes_list} / off][/dim]")
    elif arg in _main._MODE_PROMPTS:
        _main._current_mode = arg
        _main.console.print(f"[green][mode: {arg}][/green]")
    elif arg in ("off", "none", "reset", "default"):
        _main._current_mode = ""
        _main.console.print("[dim][mode cleared][/dim]")
    else:
        _main.console.print(f"[yellow][unknown mode — try: {' / '.join(_main._MODE_PROMPTS)} / off][/yellow]")


def _cmd_model(ctx: _ReplContext, arg: str) -> None:
    """Display the current model, list available models, or switch to a different model."""
    import qwen_cli.main as _main

    if not arg:
        _main.console.print(f"[dim][model: {_main.MODEL}][/dim]")
    elif arg == "list":
        models = _main.list_models(ctx.client)
        for m in models:
            mark = "  [bold green]<- current[/bold green]" if m == _main.MODEL else ""
            _main.console.print(f"  [cyan]{m}[/cyan]{mark}")
        if not models:
            _main.console.print("[dim][no models found][/dim]")
    else:
        _main.MODEL = arg
        _main.console.print(f"[green][model switched to: {_main.MODEL}][/green]")


def _cmd_index(ctx: _ReplContext, arg: str) -> None:
    """Build or rebuild the project file index for faster searching."""
    import qwen_cli.main as _main

    _main.cmd_index(Path.cwd(), force=(arg.lower() == "force"))


def _cmd_task(ctx: _ReplContext, arg: str) -> None:
    """Create a multi-step task plan using the LLM and execute it step by step."""
    import qwen_cli.main as _main

    if not arg:
        _main.console.print("[yellow][usage: /task <goal>][/yellow]")
    else:
        _main.cmd_task(arg, ctx.history, ctx.base_system, ctx.client)


def _cmd_agent(ctx: _ReplContext, arg: str) -> None:
    """Spawn a sub-agent to autonomously execute a goal."""
    import qwen_cli.main as _main

    if not arg:
        _main.console.print("[yellow][usage: /agent <goal>][/yellow]")
    else:
        _main.cmd_agent(arg, ctx.history, ctx.base_system, ctx.client)


def _cmd_plan(ctx: _ReplContext, arg: str) -> None:
    """Show the current /agent or /task progress plan, if one is active."""
    import qwen_cli.main as _main

    if not _main._current_plan:
        _main.console.print("[dim][no active plan — run /agent or /task to start one][/dim]")
    else:
        _main._render_plan_panel()


def _cmd_git(ctx: _ReplContext, arg: str) -> None:
    """Handle git subcommands: commit and PR creation."""
    import qwen_cli.main as _main

    sub_parts = arg.split(None, 1)
    sub = sub_parts[0].lower() if sub_parts else ""
    sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""
    if sub == "commit":
        _main.cmd_git_commit(ctx.client, sub_arg)
    elif sub == "pr":
        _main.cmd_git_pr(ctx.client)
    else:
        _main.console.print("[yellow][usage: /git commit [msg] · /git pr][/yellow]")


def _cmd_lsp(ctx: _ReplContext, arg: str) -> None:
    """LSP command handler: /lsp <subcommand> [args...]."""
    import qwen_cli.main as _main

    parts = arg.split()
    sub = parts[0].lower() if parts else ""
    sub_args = parts[1:] if len(parts) > 1 else []

    _lsp = _main._get_lsp()

    _lsp_dispatch = {
        "status": lambda _a: _lsp.lsp_status(),
        "diagnose": lambda a: _lsp.lsp_diagnostics(a[0]) if a else "usage: /lsp diagnose <file>",
        "diag": lambda a: _lsp.lsp_diagnostics(a[0]) if a else "usage: /lsp diagnose <file>",
        "define": lambda a: (
            _lsp.lsp_definition(a[0], int(a[1]), int(a[2])) if len(a) >= 3 else "usage: /lsp define <file> <line> <col>"
        ),
        "def": lambda a: (
            _lsp.lsp_definition(a[0], int(a[1]), int(a[2])) if len(a) >= 3 else "usage: /lsp define <file> <line> <col>"
        ),
        "refs": lambda a: (
            _lsp.lsp_references(a[0], int(a[1]), int(a[2])) if len(a) >= 3 else "usage: /lsp refs <file> <line> <col>"
        ),
        "references": lambda a: (
            _lsp.lsp_references(a[0], int(a[1]), int(a[2])) if len(a) >= 3 else "usage: /lsp refs <file> <line> <col>"
        ),
        "hover": lambda a: (
            _lsp.lsp_hover(a[0], int(a[1]), int(a[2])) if len(a) >= 3 else "usage: /lsp hover <file> <line> <col>"
        ),
        "symbols": lambda a: _lsp.lsp_symbols(a[0]) if a else "usage: /lsp symbols <file>",
        "sym": lambda a: _lsp.lsp_symbols(a[0]) if a else "usage: /lsp symbols <file>",
        "rename": lambda a: (
            _lsp.lsp_rename(a[0], int(a[1]), int(a[2]), a[3])
            if len(a) >= 4
            else "usage: /lsp rename <file> <line> <col> <new_name>"
        ),
    }

    handler = _lsp_dispatch.get(sub)
    if handler:
        result = handler(sub_args)
        if isinstance(result, str) and result.startswith("usage:"):
            _main.console.print(f"[yellow]{result}[/yellow]")
        else:
            _main.console.print(result)
    else:
        _main.console.print(
            "[yellow]/lsp: status | diagnose <file> | define <file> <line> <col> | refs <file> <line> <col> | hover <file> <line> <col> | symbols <file> | rename <file> <line> <col> <new_name>[/yellow]"
        )


def _cmd_watch(ctx: _ReplContext, arg: str) -> None:
    """Watch files for changes; list watched files, stop watching, or add a file."""
    import qwen_cli.main as _main

    if not arg or arg == "list":
        if _main._watched_files:
            for _wf in _main._watched_files:
                _main.console.print(f"  [cyan]{_wf}[/cyan]")
        else:
            _main.console.print("[dim][no files watched][/dim]")
    elif arg == "off":
        _main._watch_stop.set()
        if _main._watch_thread:
            _main._watch_thread.join(timeout=3)
        _main._watched_files.clear()
        _main._watch_pending.clear()
        _main._watch_stop.clear()
        _main._watch_thread = None
        _main.console.print("[dim][watch stopped][/dim]")
    else:
        _wp = _main._resolve(arg)
        if not _wp.exists():
            _main.console.print(f"[red][file not found: {_wp}][/red]")
        else:
            _main._watched_files[str(_wp)] = _wp.stat().st_mtime
            _main.console.print(f"[green][watching: {_wp}][/green]")
            if _main._watch_thread is None or not _main._watch_thread.is_alive():
                _main._watch_stop.clear()
                _main._watch_thread = threading.Thread(
                    target=_main._watch_worker,
                    args=(_main._watched_files,),
                    daemon=True,
                )
                _main._watch_thread.start()


def _cmd_preset(ctx: _ReplContext, arg: str) -> None:
    """Apply a conversation preset (predefined system prompt template)."""
    import qwen_cli.main as _main

    _main.cmd_preset(arg)


def _cmd_params(ctx: _ReplContext, arg: str) -> None:
    """Display or modify model parameters (temperature, top_p, etc.)."""
    import qwen_cli.main as _main

    _main.cmd_params(arg)


def _cmd_long(ctx: _ReplContext, arg: str) -> None:
    """Toggle long mode for extended, detailed responses."""
    import qwen_cli.main as _main

    _main.cmd_long(arg)


def _cmd_autosearch(ctx: _ReplContext, arg: str) -> None:
    """Toggle automatic web search before answering questions."""
    import qwen_cli.main as _main

    _main.cmd_autosearch(arg)


def _cmd_rollback(ctx: _ReplContext, arg: str) -> None:
    """Rollback to a previous version of a modified file."""
    import qwen_cli.main as _main

    _main.cmd_rollback()


def _cmd_review(ctx: _ReplContext, arg: str) -> None:
    """Ask the LLM to review code or text in the conversation."""
    import qwen_cli.main as _main

    _main.cmd_review(arg, ctx.history, ctx.base_system, ctx.client)


def _cmd_error(ctx: _ReplContext, arg: str) -> None:
    """Diagnose and explain the last error that occurred."""
    import qwen_cli.main as _main

    _main.cmd_error(ctx.history, ctx.base_system, ctx.client)


def _cmd_config(ctx: _ReplContext, arg: str) -> None:
    """Display the current configuration settings."""
    import qwen_cli.main as _main

    _main.show_config()


def _cmd_team(ctx: _ReplContext, arg: str) -> None:
    """Manage teams: create, list, show board, or join a team."""
    import qwen_cli.main as _main

    sub_parts = arg.split(None, 2)
    sub = sub_parts[0].lower() if sub_parts else "list"
    sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""
    desc_arg = sub_parts[2].strip() if len(sub_parts) > 2 else ""
    if sub == "create":
        if not sub_arg:
            _main.console.print("[yellow][usage: /team create <name> [description]][/yellow]")
        else:
            _main._ct_team_create(sub_arg, desc_arg)
            _main.console.print(f"[green][team '{sub_arg}' created][/green]")
    elif sub == "list" or not sub_arg:
        teams = _main._ct_team_list()
        if not teams:
            _main.console.print("[dim][no teams — create one with /team create <name>][/dim]")
        else:
            for t in teams:
                cfg = _main._ct_load_team(t) or {}
                n = len(cfg.get("members", []))
                _main.console.print(f"  [cyan]{t}[/cyan]  [dim]{n} member(s)[/dim]")
    elif sub == "show":
        if not sub_arg:
            _main.console.print("[yellow][usage: /team show <name>][/yellow]")
        else:
            _main.console.print(Markdown(_main._ct_board_render(sub_arg)))
    elif sub == "join":
        if not sub_arg:
            _main.console.print("[yellow][usage: /team join <team> [agent_name]][/yellow]")
        else:
            agent = desc_arg or "user"
            _main._ct_team_join(sub_arg, agent)
            _main.console.print(f"[green][joined team '{sub_arg}' as '{agent}'][/green]")
    else:
        _main.console.print("[yellow][usage: /team create|list|show|join][/yellow]")


def _cmd_spawn(ctx: _ReplContext, arg: str) -> None:
    """Spawn a sub-agent on a team to execute a task autonomously."""
    import qwen_cli.main as _main

    spawn_parts = arg.split(None, 2)
    if len(spawn_parts) < 3:
        _main.console.print("[yellow][usage: /spawn <team> <agent_name> <task>][/yellow]")
    else:
        s_team, s_agent, s_task = spawn_parts
        result = _main._ct_spawn(s_team.strip(), s_agent.strip(), s_task.strip())
        _main.console.print(f"[green]{result}[/green]")


def _cmd_inbox(ctx: _ReplContext, arg: str) -> None:
    """Read or send messages in a team inbox."""
    import qwen_cli.main as _main

    inbox_parts = arg.split(None, 3)
    sub = inbox_parts[0].lower() if inbox_parts else ""
    if sub == "send":
        if len(inbox_parts) < 4:
            _main.console.print("[yellow][usage: /inbox send <team> <to_agent> <message>][/yellow]")
        else:
            _, i_team, i_to, i_msg = inbox_parts[0], inbox_parts[1], inbox_parts[2], inbox_parts[3]
            msg_id = _main._ct_inbox_send(i_team, i_to, i_msg)
            _main.console.print(f"[green][sent message {msg_id[:8]} to {i_to} in team '{i_team}'][/green]")
    elif sub and sub not in ("receive", "read"):
        i_team = sub
        i_agent = inbox_parts[1] if len(inbox_parts) > 1 else "user"
        msgs = _main._ct_inbox_receive(i_team, i_agent)
        if not msgs:
            _main.console.print(f"[dim][inbox empty for {i_agent} in team '{i_team}'][/dim]")
        else:
            for m in msgs:
                _main.console.print(
                    Panel(
                        m.get("body", ""),
                        title=f"From: {m.get('from', '?')}",
                        border_style="cyan",
                    )
                )
    else:
        _main.console.print("[yellow][usage: /inbox <team> [agent] | /inbox send <team> <to> <message>][/yellow]")


def _cmd_board(ctx: _ReplContext, arg: str) -> None:
    """Display the task board for a team (or all teams)."""
    import qwen_cli.main as _main

    if not arg:
        teams = _main._ct_team_list()
        if not teams:
            _main.console.print("[dim][no teams][/dim]")
        else:
            for t in teams:
                _main.console.print(Markdown(_main._ct_board_render(t)))
                _main.console.print(Rule())
    else:
        _main.console.print(Markdown(_main._ct_board_render(arg.strip())))


def _cmd_history(ctx: _ReplContext, arg: str) -> None:
    """Search or display the conversation history."""
    import qwen_cli.main as _main

    _main.cmd_history(arg, ctx.history)


def _cmd_note(ctx: _ReplContext, arg: str) -> None:
    """Add a timestamped note to the conversation or a task."""
    import qwen_cli.main as _main

    if not arg:
        _main.console.print("[yellow][usage: /note <text>][/yellow]")
    else:
        ctx.history.append({"role": "user", "content": f"[Note] {arg}"})
        _main.console.print("[dim][note added to history][/dim]")


def _cmd_compact(ctx: _ReplContext, arg: str) -> None:
    """Manually trigger context compaction to reduce token usage."""
    import qwen_cli.main as _main

    ctx.history = _main.cmd_trim(ctx.history, ctx.client)


def _cmd_stats(ctx: _ReplContext, arg: str) -> None:
    """Display session statistics: token counts, turn count, timing."""
    import qwen_cli.main as _main

    _main.cmd_stats(ctx.history)


def _cmd_cleanup(ctx: _ReplContext, arg: str) -> None:
    """Clean up stale teams, backups, and completed tasks."""
    import qwen_cli.main as _main

    sub = arg.strip().lower() if arg else "all"
    dry_run = sub == "dry-run"
    what = sub if sub in ("teams", "backups", "tasks", "all") else "all"
    if dry_run:
        what = "all"

    removed = 0

    # --- Teams ---
    if what in ("teams", "all"):
        ct_dir = _main.CT_DIR / "teams"
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
                        _logger.exception("Failed to remove team dir: %s", team_dir)
                removed += 1
                if dry_run:
                    _main.console.print(f"  [dim]would remove team:[/dim] {team_dir.name}")

    # --- Backups ---
    if what in ("backups", "all") and _main.BACKUPS_DIR.is_dir():
        now_ts = time.time()
        max_age = 7 * 86400  # 7 days
        for bf in list(_main.BACKUPS_DIR.iterdir()):
            if bf.stat().st_mtime < now_ts - max_age:
                if not dry_run:
                    with contextlib.suppress(Exception):
                        bf.unlink()
                removed += 1
                if dry_run:
                    _main.console.print(f"  [dim]would remove backup:[/dim] {bf.name}")

    # --- Completed tasks ---
    if what in ("tasks", "all"):
        ct_dir = _main.CT_DIR / "teams"
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
                                _main.console.print(f"  [dim]would remove task:[/dim] {team_dir.name}/{tf.name}")
                    except (json.JSONDecodeError, OSError):
                        pass

    if dry_run:
        _main.console.print(f"[dim][dry-run: {removed} item(s) would be removed][/dim]")
    else:
        _main.console.print(f"[dim][cleanup done: {removed} item(s) removed][/dim]")


def _cmd_intel(ctx: _ReplContext, arg: str) -> None:
    """Manage Live Intelligence: toggle crawlers, add/remove topics, view feed."""
    import qwen_cli.main as _main

    sub = arg.strip().lower() if arg else ""
    if sub == "off":
        # .clear()/.set(), never rebind — _intel_enabled is a threading.Event
        # shared with the crawler threads; assigning a bool here used to
        # replace the Event object itself, crashing every crawler thread with
        # AttributeError on its next `.is_set()` call.
        _main._intel_enabled.clear()
        _main.console.print("[dim][intel crawlers paused — /intel on to resume][/dim]")
    elif sub == "on":
        _main._intel_enabled.set()
        _main.console.print("[dim][intel crawlers resumed][/dim]")
    elif sub.startswith("topics"):
        topics = _main._intel_load_topics()
        from rich.table import Table as _T

        tbl = _T(title="Intel topics", box=None)
        tbl.add_column("Name", style="cyan")
        tbl.add_column("Query", style="dim")
        tbl.add_column("Last crawled", style="dim")
        for tp in sorted(topics, key=lambda t: -t.get("last_checked", 0)):
            lc = (
                datetime.fromtimestamp(tp["last_checked"]).strftime("%m-%d %H:%M")
                if tp.get("last_checked")
                else "never"
            )
            tbl.add_row(tp["name"], tp["query"][:55], lc)
        _main.console.print(tbl)
    elif sub.startswith("add "):
        rest = arg[4:].strip()
        if "|" in rest:
            tname, tquery = rest.split("|", 1)
            tname, tquery = tname.strip(), tquery.strip()
        else:
            tname, tquery = rest, rest + " news today"
        with _main._intel_lock:
            topics = _main._intel_load_topics()
            if any(t["name"].lower() == tname.lower() for t in topics):
                added = False
            else:
                topics.append({"name": tname, "query": tquery, "last_checked": 0})
                _main._intel_save_topics(topics)
                added = True
        if added:
            _main.console.print(f"[green][added topic '{tname}'][/green]")
        else:
            _main.console.print(f"[yellow][topic '{tname}' already tracked][/yellow]")
    elif sub.startswith("remove "):
        tname = arg[7:].strip()
        with _main._intel_lock:
            topics = _main._intel_load_topics()
            before = len(topics)
            topics = [t for t in topics if t["name"].lower() != tname.lower()]
            removed = len(topics) < before
            if removed:
                _main._intel_save_topics(topics)
        if removed:
            _main.console.print(f"[green][removed topic '{tname}'][/green]")
        else:
            _main.console.print(f"[yellow][topic '{tname}' not found][/yellow]")
    else:
        feed = _main.intel_get_recent(10)
        if feed:
            _main.console.print(Markdown(feed))
        else:
            _main.console.print("[dim][no intel yet — crawlers are warming up][/dim]")
        status = "[green]running[/green]" if _main._intel_enabled.is_set() else "[yellow]paused[/yellow]"
        _main.console.print(
            f"[dim]Crawlers: {status} · {_main._INTEL_CRAWLERS} threads · {_main._INTEL_INTERVAL}s interval · /intel topics · /intel add <name>|<query> · /intel on/off[/dim]"
        )


def _cmd_cd(ctx: _ReplContext, arg: str) -> None:
    """Change the working directory; display current directory if no argument given."""
    import qwen_cli.main as _main

    if not arg:
        _main.console.print(f"[dim][cwd: {Path.cwd()}][/dim]")
    else:
        try:
            new_cwd = _main._resolve(arg)
            if not new_cwd.is_dir():
                _main.console.print(f"[red][not a directory: {new_cwd}][/red]")
            else:
                os.chdir(new_cwd)
                _main._invalidate_git_cache()
                _main.console.print(f"[green][cwd → {new_cwd}][/green]")
        except Exception as e:
            _main.console.print(f"[red][error: {e}][/red]")


def _cmd_unknown(ctx: _ReplContext, directive: str) -> None:
    """Handle unknown commands by printing an error and suggesting /help."""
    import qwen_cli.main as _main

    _main.console.print(f"[yellow][unknown command: {directive} — try /help][/yellow]")


# ==============================================================================
# Command Dispatch Table
# ==============================================================================

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
    "/plan": _cmd_plan,
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
