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
            m = _ct_team_join(sub_arg, agent)
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
        console.print(f"[dim][note added to history][/dim]")


def _cmd_compact(ctx: _ReplContext, arg: str) -> None:
    ctx.history = cmd_trim(ctx.history, ctx.client)


def _cmd_stats(ctx: _ReplContext, arg: str) -> None:
    cmd_stats(ctx.history)


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
