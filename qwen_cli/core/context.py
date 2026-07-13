"""Context-window management — token estimates, compaction, trim, task anchor.

Extracted from main.py as part of the monolith split. Owns the growth-tracking
state and the compaction policy (adaptive thresholds, summarize-then-truncate
escalation). run_turn (still in main) calls _compact_tool_loop and
_refresh_task_anchor mid-loop; the REPL calls _maybe_autocompact around turns.

Cross-subsystem access goes through a lazy `import qwen_cli.main as _main` at
call time — the project convention for split-out modules. That matters doubly
here: TOKEN_LIMIT is rebound at runtime by /long, _real_ctx_tokens is written
by run_turn after every call, and tests monkeypatch cmd_trim/_generate_handoff
on main — all of which must keep resolving through main's namespace.
"""

import logging
import re

_logger = logging.getLogger(__name__)


def approx_tokens(messages: list) -> int:
    """Approx Tokens."""
    return sum(len(m.get("content") or "") for m in messages) // 4


# ---------------------------------------------------------------------------
# Context growth tracking & adaptive compaction thresholds
# ---------------------------------------------------------------------------

_context_growth_history: list[int] = []  # token counts per turn for growth tracking


def _track_context_growth(token_count: int) -> None:
    """Track token count after each turn for predictive compaction."""
    _context_growth_history.append(token_count)
    if len(_context_growth_history) > 10:
        _context_growth_history.pop(0)


def _estimate_turns_remaining(current_tokens: int, threshold_pct: int = 80) -> int:
    """Estimate how many turns until context reaches the threshold."""
    import qwen_cli.main as _main

    if len(_context_growth_history) < 2:
        return -1
    recent = _context_growth_history[-5:] if len(_context_growth_history) >= 5 else _context_growth_history
    growths = [recent[i] - recent[i - 1] for i in range(1, len(recent)) if recent[i] > recent[i - 1]]
    if not growths:
        return -1
    avg_growth = sum(growths) // len(growths)
    if avg_growth <= 0:
        return -1
    threshold_tokens = _main.TOKEN_LIMIT * threshold_pct // 100
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


# ---------------------------------------------------------------------------
# Mid-run (in-turn) compaction
# ---------------------------------------------------------------------------


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
    import qwen_cli.main as _main

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
        _main.console.print(f"[dim]  [mid-run compact] condensed {shrunk} older tool result(s) to free context[/dim]")
    return new


_TASK_ANCHOR_PREFIX = "[Context note — older tool results above were condensed to save space."


def _refresh_task_anchor(working: list, task: str) -> list:
    """Re-state the active request at the tail after mid-run compaction.

    Once older tool results are condensed, the strongest remaining signal near
    the top of the window can be stale chit-chat ("hey" → greeting), and the
    model may answer *that* instead of the task — seen in practice as a
    mid-audit "Hey! What's up?" final reply. Deterministically re-anchoring the
    request keeps the loop on target. Idempotent: any previous anchor is
    removed so at most one exists, always at the tail (never between an
    assistant tool_calls message and its tool results — at the point this runs,
    the working list always ends on completed tool results or user nudges).
    """
    if not task.strip():
        return working
    new = [
        m
        for m in working
        if not (m.get("role") == "user" and (m.get("content") or "").startswith(_TASK_ANCHOR_PREFIX))
    ]
    new.append(
        {
            "role": "user",
            "content": (
                f"{_TASK_ANCHOR_PREFIX} The task below is unchanged and still in progress — "
                "continue it using the findings above. Do not greet, restart, or re-plan.]\n\n"
                f"Active request:\n{task[:600]}"
            ),
        }
    )
    return new


# ---------------------------------------------------------------------------
# Between-turn compaction: truncate-middle, /trim summarization, auto-compact
# ---------------------------------------------------------------------------


def truncate_middle(history: list, keep_first: int = 6, keep_last: int = 20) -> list:
    """Drop middle messages to recover context space. Keeps system msgs + first K + last K chat turns."""
    import qwen_cli.main as _main

    sys_msgs = [m for m in history if m.get("role") == "system"]
    chat_msgs = [m for m in history if m.get("role") != "system"]

    if len(chat_msgs) <= keep_first + keep_last:
        _main.console.print("[dim][truncate-middle: history too short to trim][/dim]")
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
    _main.console.print(
        f"[dim][truncate-middle: dropped {dropped} middle messages — "
        f"kept {keep_first} oldest + {keep_last} most recent][/dim]",
    )
    return new_history


def cmd_trim(history: list, client: object) -> list:
    """Command: trim."""
    import qwen_cli.main as _main

    from rich.live import Live
    from rich.markdown import Markdown
    from rich.rule import Rule

    console = _main.console
    # Whatever this function returns, the last real prompt-token count from the
    # API no longer describes the (possibly shrunk) history — _maybe_autocompact
    # must recompute rather than reuse a now-stale figure.
    _main._real_ctx_tokens = 0
    CHUNK = 8
    keep_count = 4
    if len(history) < keep_count + CHUNK:
        console.print("[dim][history too short to trim (need ≥12 turns)][/dim]")
        return history

    # Separate "work" turns from pure chat, keyed on the turn-actions ledger the
    # REPL appends to assistant messages (tool results themselves never enter
    # history, so markers like "[patched:" only appear if the model happens to
    # quote them — the ledger tag is the reliable signal). Work turns are kept
    # in condensed form: trimmed prose + the full ledger line.
    _work_markers = ("[turn actions:", "[patched:", "[created:", "[updated:", "[write_file]", "[patch_file]")
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
                summary, _, _ = _main.stream_once(client, prompt, use_tools=False, update_fn=_main._live_updater(live))
            console.print(Markdown(summary))
            rolling_summary = summary
        except KeyboardInterrupt:
            console.print("[dim][trim cancelled — keeping partial progress][/dim]")
            break
        i += CHUNK

    if rolling_summary:
        # Prepend session-changes note so the model knows what was modified
        if _main._session_changes:
            from pathlib import Path

            modified = ", ".join(Path(p).name for p in _main._session_changes)
            rolling_summary = f"[Files modified this session: {modified}]\n\n{rolling_summary}"
        # Carry the visible plan across compaction deterministically — the LLM
        # summary alone loses it, and the model then re-plans from scratch and
        # redoes already-completed steps after every trim.
        if _main._current_plan:
            icon = {"completed": "x", "in_progress": "~", "pending": " "}
            plan_txt = "\n".join(f"  [{icon[s['status']]}] {s['text']}" for s in _main._current_plan)
            rolling_summary += (
                "\n\nPlan state (already agreed with the user — do NOT re-plan or redo "
                "completed steps; continue from the first unfinished step):\n" + plan_txt
            )
        summary_msg = {
            "role": "system",
            "content": (
                "[Earlier messages were compacted to free context. Summary of the work so far "
                "below — keep going on the CURRENT TASK noted at its end without asking the user "
                f"to repeat themselves:\n{rolling_summary}]"
            ),
        }

        # Re-insert preserved work turns in their original order, condensed to
        # trimmed prose + the full turn-actions ledger (verbatim preservation
        # would defeat the point of trimming).
        def _condense_work_pair(u: dict, a: dict | None) -> list[dict]:
            uc = u.get("content") or ""
            out = [{"role": "user", "content": uc[:200] + ("…" if len(uc) > 200 else "")}]
            if a is not None:
                ac = a.get("content") or ""
                m = re.search(r"\[turn actions:.*\]\s*$", ac, re.DOTALL)
                ledger = m.group(0) if m else ""
                prose = (ac[: m.start()] if m else ac).strip()
                body = prose[:300] + ("…" if len(prose) > 300 else "")
                out.append({"role": "assistant", "content": body + ("\n\n" + ledger if ledger else "")})
            return out

        preserved = [
            msg for _, u, a in sorted(work_pairs, key=lambda x: x[0]) for msg in _condense_work_pair(u, a)
        ]
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
    import qwen_cli.main as _main

    console = _main.console
    if _main._real_ctx_tokens:
        tok = _main._real_ctx_tokens
    else:
        full_msgs = [{"role": "system", "content": _main.build_system_prompt(base_system)}, *history]
        tok = approx_tokens(full_msgs)
    pct = tok * 100 // _main.TOKEN_LIMIT
    _track_context_growth(tok)

    compaction_threshold = _adaptive_compaction_threshold(_detect_session_type(history))
    warning_threshold = max(60, compaction_threshold - 15)

    if pct >= compaction_threshold:
        label = "auto-compact" if pct >= 90 else "auto-trim"
        console.print(
            f"[yellow]  [{label}] context at {pct}% ({tok:,} tokens) — "
            f"summarizing & preserving current task so work continues...[/yellow]",
        )
        # Overflow handoff: capture a resumable summary BEFORE history is
        # rewritten, so a crash or exit right after overflow leaves the next
        # session more than the lightweight exit note. On summary failure this
        # still writes a pointer at the autosave. _consume_handoff carries the
        # next_step field forward on resume.
        try:
            next_step = next((s["text"] for s in _main._current_plan if s.get("status") != "completed"), "")
            _main._write_handoff(_main._generate_handoff(client, history, base_system), history, next_step=next_step)
        except Exception:
            _logger.debug("Overflow handoff write failed", exc_info=True)
        before = len(history)
        try:
            history = _main.cmd_trim(history, client)
        except Exception as _trim_err:
            console.print(f"[dim][summarize failed: {_trim_err} — truncating middle as fallback][/dim]")
            history = _main.truncate_middle(history)
        # If still over after trim, escalate with progressively smaller keep values
        after_tok = approx_tokens([{"role": "system", "content": _main.build_system_prompt(base_system)}, *history])
        if after_tok * 100 // _main.TOKEN_LIMIT >= 85:
            history = _main.truncate_middle(history, keep_first=4, keep_last=16)
            after_tok = approx_tokens(
                [{"role": "system", "content": _main.build_system_prompt(base_system)}, *history]
            )
        # Emergency: if still critically full, aggressive truncation
        if after_tok * 100 // _main.TOKEN_LIMIT >= 92:
            console.print("[yellow]  [emergency-trim] still over — aggressive middle truncation[/yellow]")
            history = _main.truncate_middle(history, keep_first=2, keep_last=12)
            _main._real_ctx_tokens = 0
            return history
        # If summarization couldn't shrink (too few but huge turns) and we're
        # critically full, fall back to a hard middle-truncation as a safety net.
        if len(history) >= before and pct >= 90:
            history = _main.truncate_middle(history)
        _main._real_ctx_tokens = 0
        # Compaction rewrites history, so the prefix cache is lost anyway — the
        # one cheap moment to fold in memory/intel updates accumulated since the
        # last snapshot (see refresh_system_snapshot).
        _main.refresh_system_snapshot()
        after_tok = approx_tokens([{"role": "system", "content": _main.build_system_prompt(base_system)}, *history])
        console.print(
            f"[green]  [{label} done] freed context to "
            f"{after_tok * 100 // _main.TOKEN_LIMIT}% ({after_tok:,} tokens)[/green]",
        )
    elif pct >= warning_threshold:
        turns_left = _estimate_turns_remaining(tok, compaction_threshold)
        turns_hint = f"~{turns_left} turns until auto-trim, " if turns_left > 0 else ""
        console.print(
            f"[yellow]  context: ~{tok:,} / {_main.TOKEN_LIMIT:,} tokens ({pct}%) — "
            f"{turns_hint}auto-trim at {compaction_threshold}%[/yellow]",
        )
    return history
