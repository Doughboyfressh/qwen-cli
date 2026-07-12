"""REPL module — interactive loop, input, watch, setup, turn execution."""

import asyncio
import atexit
import contextlib
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

_logger = logging.getLogger(__name__)
console = Console(force_terminal=True, legacy_windows=False)


class _ReplContext:
    def __init__(self, history: list[dict], base_system: str, client: object) -> None:
        self.history = history
        self.base_system = base_system
        self.client = client


def _setup_tab_completion() -> None:
    import qwen_cli.main as _main

    if _main._rl is None:
        return
    try:

        def completer(text: str, state: int) -> str | None:
            if text.startswith("/"):
                options = [c + " " for c in _main._all_commands() if c.startswith(text)]
            elif text.startswith("@") or "/" in text or "\\" in text:
                raw = text.lstrip("@")
                pre = "@" if text.startswith("@") else ""
                p = Path(raw).expanduser()
                base = p if (raw.endswith(("/", "\\"))) else p.parent
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

        _main._rl.set_completer(completer)
        _main._rl.parse_and_bind("tab: complete")
    except Exception:
        pass


def _make_pt_session() -> None:
    import qwen_cli.main as _main

    if not _main._HAS_PT:
        return

    from prompt_toolkit import PromptSession as _PtSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory as _PtAutoSuggest
    from prompt_toolkit.completion import Completer as _PtCompleter
    from prompt_toolkit.completion import Completion as _PtCompletion
    from prompt_toolkit.history import FileHistory as _PtFileHistory

    from qwen_cli.core.config import PT_HISTORY_FILE

    class _Completer(_PtCompleter):
        def get_completions(self, document, complete_event) -> Iterator[_PtCompletion]:
            text = document.text_before_cursor
            word = text.split()[-1] if text.split() else text
            if text.lstrip().startswith("/"):
                for cmd in _main._all_commands():
                    if cmd.startswith(word):
                        yield _PtCompletion(cmd[len(word) :], start_position=0, display=cmd)
                return
            at_pos = text.rfind("@")
            if at_pos >= 0:
                raw = text[at_pos + 1 :]
                if "::" not in raw:
                    p = Path(raw).expanduser()
                    base = p if (raw.endswith(("/", "\\"))) else p.parent
                    stem = p.name if not (raw.endswith(("/", "\\"))) else ""
                    try:
                        for child in sorted(base.iterdir()):
                            name = child.name + ("/" if child.is_dir() else "")
                            if child.name.lower().startswith(stem.lower()):
                                yield _PtCompletion(name, start_position=-len(stem))
                    except Exception:
                        pass

    _main._pt_session = _PtSession(
        history=_PtFileHistory(str(PT_HISTORY_FILE)),
        auto_suggest=_PtAutoSuggest(),
        completer=_Completer(),
        complete_while_typing=False,
        enable_history_search=True,
    )


def _close_loitering_event_loop() -> None:
    """Close a leftover running asyncio loop on this thread, if any.

    Background work (LSP handshakes, browser automation, intelligence
    threads) can leave the main thread's asyncio state marked as "running"
    without actually returning control cleanly. prompt_toolkit's
    PromptSession.prompt() calls asyncio.run() internally, which raises
    "cannot be called from a running event loop" if that happens — this
    clears it defensively before every prompt.
    """
    try:
        loop = asyncio.get_running_loop()
        try:
            loop.close()
        except RuntimeError:
            pass
    except RuntimeError:
        pass
    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    except Exception:
        pass


def _read_input_inline() -> str:
    import qwen_cli.main as _main

    if _main._pt_session is not None:
        line = _main._pt_session.prompt("\nYou: ").strip()
    else:
        line = console.input("\n[bold cyan]You:[/bold cyan] ").strip()
    if not line.endswith("\\"):
        return line
    lines = [line[:-1]]
    while True:
        if _main._pt_session is not None:
            cont = _main._pt_session.prompt("... ").strip()
        else:
            cont = console.input("[bold cyan]...[/bold cyan] ").strip()
        if cont.endswith("\\"):
            lines.append(cont[:-1])
        else:
            lines.append(cont)
            break
    return "\n".join(lines)


def _read_input_in_thread() -> str:
    result = []
    error = []

    def _prompt_thread() -> None:
        try:
            result.append(_read_input_inline())
        except Exception as e:
            error.append(str(e))

    t = threading.Thread(target=_prompt_thread, daemon=True)
    t.start()
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
    _close_loitering_event_loop()
    try:
        asyncio.get_running_loop()
        return _read_input_in_thread()
    except RuntimeError:
        pass
    try:
        return _read_input_inline()
    except RuntimeError as e:
        # prompt_toolkit's prompt() calls asyncio.run(), which raises if a
        # background thread's loop state landed on this thread between the
        # get_running_loop() check above and here. Observed killing a live
        # session (fatal REPL crash) — a fresh thread has no running loop,
        # so the thread path always works. Fall back instead of dying.
        if "running event loop" not in str(e).lower():
            raise
        _logger.warning("prompt blocked by a loitering event loop — retrying input in a thread")
        return _read_input_in_thread()


def _watch_worker(mtimes: dict[str, float]) -> None:
    import qwen_cli.main as _main

    while not _main._watch_stop.is_set():
        for path in list(mtimes.keys()):
            try:
                mtime = Path(path).stat().st_mtime
                old = mtimes.get(path)
                if old is not None and mtime != old:
                    _main._watch_pending.append(path)
                mtimes[path] = mtime
            except Exception:
                _logger.debug("File watcher: stat failed for %s", path)
        _main._watch_stop.wait(timeout=2)


def _run_turn_and_handle_reply(ctx: _ReplContext, user_input: str, allow_tools: bool = True) -> None:
    import qwen_cli.main as _main

    ctx.history = _main._maybe_autocompact(ctx.history, ctx.base_system, ctx.client)

    messages = [{"role": "system", "content": _main.build_system_prompt(ctx.base_system)}, *ctx.history]
    messages.append({"role": "user", "content": user_input})

    t0 = time.monotonic()
    with _main._main_llm_busy_lock:
        _main._main_llm_busy = True
    try:
        reply = _main.run_turn(ctx.client, messages, allow_tools=allow_tools)
    finally:
        with _main._main_llm_busy_lock:
            _main._main_llm_busy = False
    elapsed = time.monotonic() - t0

    if not reply:
        return

    # Low-confidence self-correction: _confidence_warning() below only *tells*
    # the user a reply hedges heavily — it doesn't do anything about it. If the
    # model hedged (multiple "I think"/"might"/"probably"-style phrases) and
    # never actually searched to verify, force one grounding pass instead of
    # handing back an unverified answer. Scoped to plain chat only (not
    # /agent or /task, via allow_tools) since "might"/"could" are often
    # legitimate engineering judgment there, not factual uncertainty.
    if (
        allow_tools
        and _main.AUTO_SEARCH_MODE != "off"
        and len(_main._HEDGE_RE.findall(reply)) >= 3
        and not any(name in ("web_search", "search_news") for name in _main._last_turn_tool_names)
    ):
        console.print("[dim yellow]  [low confidence — verifying with a search before answering][/dim yellow]")
        messages.append({"role": "assistant", "content": reply})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your last answer used a lot of hedging language (e.g. 'I think', 'might', 'probably') "
                    "without searching to verify. Use web_search now to check the specific facts you were "
                    "unsure about, then give a corrected, grounded answer."
                ),
            }
        )
        with _main._main_llm_busy_lock:
            _main._main_llm_busy = True
        try:
            # presearch=False: this synthetic message already tells the model
            # to search; auto-presearching it just web-searches the critique
            # text itself ("hedging language..."), which is noise.
            revised = _main.run_turn(ctx.client, messages, allow_tools=allow_tools, presearch=False)
        finally:
            with _main._main_llm_busy_lock:
                _main._main_llm_busy = False
        if revised:
            reply = revised

    ctx.history.append({"role": "user", "content": user_input})
    ctx.history.append({"role": "assistant", "content": reply})
    _main._turn_count += 1
    _main._print_turn_footer(elapsed)
    _main._confidence_warning(reply)

    if allow_tools and _main._looks_like_plan(reply):
        try:
            ans = console.input("[dim]  Plan detected — execute? [y/N]: [/dim]").strip().lower()
            if ans == "y":
                exec_msgs = [{"role": "system", "content": _main.build_system_prompt(ctx.base_system)}, *ctx.history]
                exec_msgs.append({"role": "user", "content": "Please execute this plan step by step now."})
                t1 = time.monotonic()
                exec_reply = _main.run_turn(ctx.client, exec_msgs, allow_tools=True)
                if exec_reply:
                    ctx.history.append({"role": "user", "content": "Please execute this plan step by step now."})
                    ctx.history.append({"role": "assistant", "content": exec_reply})
                    _main._turn_count += 1
                    _main._print_turn_footer(time.monotonic() - t1)
        except (KeyboardInterrupt, EOFError):
            pass

    ctx.history = _main._maybe_autocompact(ctx.history, ctx.base_system, ctx.client)

    from qwen_cli.core.config import AUTO_SAVE_INTERVAL

    if _main._turn_count % AUTO_SAVE_INTERVAL == 0:
        _main._silent_autosave(ctx.history, ctx.base_system)

    threading.Thread(
        target=_main._run_background_tasks,
        args=(ctx.client, user_input, reply, list(ctx.history), _main._turn_count),
        daemon=True,
    ).start()

    runnable = _main._extract_runnable_code(reply)
    if runnable:
        lang, code = runnable
        label = lang or "code"
        try:
            answer = console.input(f"[dim]  [{label}] [r] run \xb7 [c] copy \xb7 [Enter] skip: [/dim]").strip().lower()
            if answer == "r":
                _main._run_code_block(lang, code)
            elif answer == "c":
                if _main._copy_to_clipboard(code):
                    console.print("[green]  [copied code block to clipboard][/green]")
                else:
                    console.print("[yellow]  [pip install pyperclip to enable copy][/yellow]")
        except (KeyboardInterrupt, EOFError):
            pass


def _dispatch_command(ctx: _ReplContext, directive: str, arg: str) -> bool:
    from qwen_cli.core.commands import _REPL_COMMANDS, _cmd_unknown, _run_custom_command

    handler = _REPL_COMMANDS.get(directive)
    try:
        if handler:
            handler(ctx, arg)
        elif not _run_custom_command(ctx, directive, arg):
            _cmd_unknown(ctx, directive)
    except StopIteration:
        return True
    except KeyboardInterrupt:
        console.print("\n[dim][cancelled][/dim]")
    except Exception as e:
        # A bug in any single command must not end the whole session —
        # mirror the recovery the plain-chat turn path already gets.
        _logger.exception("Unhandled error in command %s", directive)
        console.print(Panel(str(e), title=f"[bold red]Error running {directive}[/bold red]", border_style="red"))
    return False


def _repl_setup(client: object) -> tuple[str, list, _ReplContext]:
    import qwen_cli.main as _main

    _setup_tab_completion()
    _make_pt_session()

    base_system = _main.BASE_SYSTEM
    history: list[dict] = []
    ctx = _ReplContext(history, base_system, client)

    pt_note = "prompt_toolkit" if _main._HAS_PT else "readline"
    _backend_label = (
        f"[cyan]{_main.BASE_URL}[/cyan]"
        if _main.ACTIVE_BACKEND == "llama.cpp"
        else "[bold yellow]OpenAI (cloud fallback)[/bold yellow]"
    )
    console.print(
        Panel(
            f"[bold]Model[/bold]   [cyan]{_main.MODEL}[/cyan]  [dim]({_main._active_preset} preset)[/dim]\n"
            f"[bold]Server[/bold]  {_backend_label}\n"
            f"[bold]Search[/bold]  [dim]{_main.AUTO_SEARCH_MODE} (/autosearch to change)[/dim]\n"
            f"[bold]Data[/bold]    [cyan]{_main.DATA_DIR}[/cyan]\n\n"
            f"[dim]@file \xb7 @file::symbol \xb7 /team \xb7 /spawn \xb7 /inbox \xb7 /board \xb7 /help \xb7 Ctrl+C cancels \xb7 /exit[/dim]\n"
            f"[dim]Input: {pt_note}{'  (Ctrl+R history, Tab completion)' if _main._HAS_PT else ''}  \xb7 {_main._INTEL_CRAWLERS} intel crawlers (/intel)[/dim]",
            title="[bold green]Qwen CLI[/bold green]",
            border_style="green",
        )
    )
    _main.start_intel_crawlers()

    cwd = Path.cwd()
    if any((cwd / m).exists() for m in _main.KEY_FILES | {".git"}):
        console.print("[dim]Project detected — loading context automatically (/project to reload)[/dim]")
        _main.load_project_context("", history)

    # _consume_handoff() reads (and deletes) whatever the previous session left
    # behind on exit or crash — see _save_exit_handoff(). Injected the same way
    # load_project_context() injects the project tree: as a "user"-role context
    # message the model sees on the next turn, not something requiring an
    # immediate reply.
    handoff = _main._consume_handoff()
    if handoff:
        turns_note = f" ({handoff['turns']} turns)" if handoff.get("turns") else ""
        console.print(f"[dim]Resuming from previous session{turns_note}...[/dim]")
        history.append({"role": "user", "content": handoff["prompt"]})

    return base_system, history, ctx


def _repl_loop(ctx: _ReplContext, history: list, base_system: str) -> None:
    import qwen_cli.main as _main

    while True:
        if _main._watch_pending:
            for _wp in _main._watch_pending[:]:
                try:
                    _p = Path(_wp)
                    _lang = _main.LANG_MAP.get(_p.suffix.lower(), "")
                    _content = _p.read_text(encoding="utf-8", errors="replace")
                    history.append(
                        {"role": "user", "content": f"[File updated: {_p.name}]\n```{_lang}\n{_content[:10_000]}\n```"}
                    )
                    console.print(f"[dim cyan]  [watch] {_p.name} updated — injected[/dim cyan]")
                except Exception as _we:
                    console.print(f"[yellow]  [watch] error reading {_wp}: {_we}[/yellow]")
            _main._watch_pending.clear()

        try:
            user_input = read_input().strip()
        except KeyboardInterrupt:
            console.print()
            continue
        except EOFError:
            console.print()
            if history:
                _main.save_session(history, base_system)
                _main._save_exit_handoff(history)
            with contextlib.suppress(Exception):
                _main.record_session_changes_memory(ctx.client)
            _cleanup_watch()
            console.print("[dim]Bye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            directive = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if _dispatch_command(ctx, directive, arg):
                _cleanup_watch()
                break
            continue

        no_tools = False
        if user_input.startswith("-- "):
            no_tools = True
            user_input = user_input[3:].strip()

        user_input = _main.expand_at_refs(user_input)
        _main._last_user_input = user_input
        try:
            _run_turn_and_handle_reply(ctx, user_input, allow_tools=not no_tools)
        except Exception as _repl_err:
            _logger.exception("Unhandled error in turn")
            console.print(
                Panel(
                    f"{_repl_err}\n\n[dim]Type /retry to try again.[/dim]",
                    title="[bold red]Error[/bold red]",
                    border_style="red",
                )
            )


def _cleanup_watch() -> None:
    import qwen_cli.main as _main

    if not hasattr(_main, "_watch_stop"):
        return
    _main._watch_stop.set()
    if _main._watch_thread and _main._watch_thread.is_alive():
        _main._watch_thread.join(timeout=2)
    _main._watch_stop.clear()
    if hasattr(_main, "_watched_files"):
        _main._watched_files.clear()
    if hasattr(_main, "_watch_pending"):
        _main._watch_pending.clear()
    _main._watch_thread = None


atexit.register(_cleanup_watch)
