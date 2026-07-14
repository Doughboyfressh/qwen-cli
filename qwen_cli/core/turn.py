"""The core tool loop — run_turn and its per-turn helpers.

Extracted from main.py as part of the monolith split. This is the
highest-traffic code in the project: every model turn flows through run_turn,
which drives presearch grounding, the volatile git tail, tool batching and
dispatch, retries, mid-run compaction + task re-anchoring, auto-continue on
cut-off answers, the empty-reply nudge, and the turn ledger.

All turn state (_turn_ledger, _last_turn_tool_names, _turn_read_cache,
_last_turn_tokens, _real_ctx_tokens, _turn_hit_round_cap) stays OWNED BY MAIN
and is read/written here as _main.<name> at call time: the REPL and the agent
loops consume it via main, tests rebind it on main, and TOKEN_LIMIT /
MAX_TOOL_DEPTH / AUTO_SEARCH_MODE are runtime-mutable there too. The tool
implementations (_dispatch_interactive, _call_tool_safe, do_*) also still live
in main and are reached the same way — the project convention for split-out
modules (see core/repl.py, core/context.py).
"""

import concurrent.futures  # noqa: TC003 — pool.submit futures are runtime objects here
import json
import logging
import re
import time
from pathlib import Path

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule

from qwen_cli.core.config import (
    _TOOL_RETRY_BASE,
    _TOOL_RETRY_MAX,
    _TOOL_RETRYABLE_TOOLS,
    AUX_LLM_TIMEOUT,
    TOOL_RESULT_LIMIT,
)
from qwen_cli.core.context import _compact_tool_loop, _refresh_task_anchor, approx_tokens

_logger = logging.getLogger(__name__)


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
    import qwen_cli.main as _main

    n = len(tool_calls)
    if n <= 1:
        return [[i] for i in range(n)]

    batches = []
    assigned = [False] * n
    domain_groups: dict[str, list[int]] = {}

    for i in range(n):
        name = tool_calls[i]["function"]["name"]
        args = parsed_args[i] or {}
        if name in _main._STATEFUL_TOOLS:
            batches.append([i])
            assigned[i] = True
        elif name in _main._SERIAL_BY_DOMAIN:
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


def _smart_cap(client: object, result: str, name: str, context: str = "") -> str:
    """Cap a tool result. Summarizes web/fetch results via LLM; hard-truncates others."""
    import qwen_cli.main as _main

    if name == "read_file":
        # read_file paginates itself with honest range headers and continuation
        # hints; a blind head-truncation here would cut the footer and reintroduce
        # the "model thinks it read the whole file" bug.
        return result
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
        bg_client, bg_model = _main._bg_llm(client)
        resp = bg_client.chat.completions.create(
            model=bg_model,
            messages=prompt,
            stream=False,
            max_tokens=1500,
            timeout=AUX_LLM_TIMEOUT,
        )
        summary = (resp.choices[0].message.content or "").strip()
        return f"[summarized {len(result):,}→{len(summary):,} chars]\n\n{summary}"
    except Exception:
        return _cap_result(result, name)


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
    import qwen_cli.main as _main

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
                _main.console.print(
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



def _auto_presearch(working: list) -> list:
    """Before the first LLM call, auto-run web_search on the user's message and
    inject the results as grounding context. Whether it fires is governed by
    _main.AUTO_SEARCH_MODE (off | smart | aggressive); see _main.presearch_decision().
    """
    import qwen_cli.main as _main

    last_user = next(
        (m.get("content", "") for m in reversed(working) if m.get("role") == "user"),
        "",
    )
    do_search, query = _main.presearch_decision(last_user, _main.AUTO_SEARCH_MODE)
    if not do_search:
        return working

    _main.console.print(f"[dim cyan]  [auto-search] {query[:70]}[/dim cyan]")
    results = _main.do_web_search(query, max_results=5)
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


def _inject_volatile_tail(working: list) -> list:
    """Append per-turn volatile context (git state) to the outgoing user message.

    Lives at the tail of the conversation, not in the system prompt: git status
    changes after every file edit, and a changed system prompt costs a full
    prefix re-eval, while a changed tail costs only the last few hundred tokens.
    Only the messages copy sent to the API is modified — the raw user input is
    what lands in the saved history, so old turns stay byte-stable too.
    """
    import qwen_cli.main as _main

    git_ctx = _main.get_git_context()
    if not git_ctx:
        return working
    suffix = f"\n\n[Current git state — for reference]\n{git_ctx}"
    new_working = list(working)
    for i in range(len(new_working) - 1, -1, -1):
        if new_working[i].get("role") == "user":
            msg = dict(new_working[i])
            msg["content"] = (msg.get("content") or "") + suffix
            new_working[i] = msg
            break
    return new_working


def _ledger_entry(name: str, args: dict, result: str) -> str | None:
    """One compact line for the turn ledger — only tools whose effects the next
    turn needs to know about (reads, file mutations, commands)."""
    import qwen_cli.main as _main

    first = (result or "").strip().splitlines()[0] if result else ""
    if name == "read_file":
        fname = Path(str(args.get("path", ""))).name
        m = re.search(r"\((\d[\d,]* lines|lines [\d,]+[–-][\d,]+ of [\d,]+)\)", first)
        return f"read {fname} ({m.group(1)})" if m else f"read {fname}"
    if name in _main._MUTATING_FILE_TOOLS:
        if first.startswith("["):
            return first[:120]
        return f"{name} {Path(str(args.get('path', ''))).name}"
    if name in ("run_command", "run_script"):
        cmd = str(args.get("command") or args.get("code") or "").replace("\n", " ")[:60]
        failed = first.lower().startswith(("[error", "[tool_call", "[timeout", "[blocked", "[cancelled"))
        return f"{name} {cmd!r}" + (f" -> {first[:60]}" if failed else "")
    return None


def _format_turn_ledger() -> str:
    """Render main._turn_ledger as a tag for the stored assistant message ('' if empty)."""
    import qwen_cli.main as _main

    if not _main._turn_ledger:
        return ""
    entries = list(_main._turn_ledger)
    if len(entries) > 14:
        entries = entries[:3] + [f"...{len(entries) - 13} more..."] + entries[-10:]
    return f"\n\n[turn actions: {'; '.join(entries)[:900]}]"


def _prepare_turn(messages: list, allow_tools: bool, presearch: bool) -> list:
    """Reset per-turn state and build the outgoing working list."""
    import qwen_cli.main as _main

    _main._turn_hit_round_cap = False  # set when this turn ends via the MAX_TOOL_DEPTH forced synthesis
    del _main._last_turn_tool_names[:]  # fresh tool log for this turn (read by /agent verification)
    del _main._turn_ledger[:]  # fresh action ledger for this turn (read by the REPL history append)
    _main._turn_read_cache.clear()  # fresh read-dedup window for this turn
    _main._turn_seen_lines.clear()  # fresh citation evidence (see main._unverified_citations)
    del _main._turn_written[:]  # fresh record of what this turn wrote (citation guard)
    working = _auto_presearch(list(messages)) if (allow_tools and presearch) else list(messages)
    if allow_tools:
        working = _inject_volatile_tail(working)
    # Surface unresolved errors from the prior turn's edits
    try:
        trend = _main._get_lsp().lsp_trend_report()
        if trend.get("unresolved_errors"):
            _main.console.print("[dim red]  Unresolved errors from prior edit:[/dim red]")
            for e in trend["unresolved_errors"][:5]:
                _main.console.print(f"    {e}")
    except Exception:
        _logger.debug("LSP trend report unavailable", exc_info=True)
    return working


def _stream_live(client, msgs: list, use_tools: bool):
    """One streaming call rendered through a transient Live region."""
    import qwen_cli.main as _main

    with Live("", console=_main.console, vertical_overflow="crop", refresh_per_second=15, transient=True) as live:
        return _main.stream_once(client, msgs, use_tools, _main._live_updater(live))


def _strip_and_show_thinking(text: str) -> str:
    """Split out <think> content, render it as a dim panel, return the reply text."""
    import qwen_cli.main as _main

    text, think = _main._strip_think(text)
    if think:
        _main.console.print(
            Panel(Markdown(think), title="[dim]thinking[/dim]", border_style="dim", padding=(0, 1))
        )
    return text


def _synthesize_at_depth_cap(client, working: list) -> tuple[str, dict]:
    """MAX_TOOL_DEPTH reached — force one final no-tools call so the model can
    answer with what it found. Returns (final_text, usage)."""
    import qwen_cli.main as _main

    _main.console.print(
        f"[yellow][max tool depth ({_main.MAX_TOOL_DEPTH}) reached — synthesizing with gathered data][/yellow]"
    )
    synth_msgs = list(working)
    synth_msgs.append(
        {
            "role": "user",
            "content": "Based on all the information gathered above, provide a comprehensive answer now. "
            "Do not search for more — synthesize what you have.",
        }
    )
    try:
        final_text, _, final_usage = _stream_live(client, synth_msgs, False)
        final_text = _strip_and_show_thinking(final_text)
        _main.console.print(Markdown(final_text) if final_text else Markdown("*(no synthesis)*"))
        return final_text, final_usage
    except Exception:
        return "", {}


def _reply_action(
    text: str, usage: dict, working: list, final_segments: list, auto_continue: int, empty_retries: int
) -> tuple[str, int, int]:
    """Decide how a no-tool-calls reply ends (or continues) the turn.

    Returns (action, auto_continue, empty_retries) where action is one of:
    'nudge'    — the reply was empty; a re-ask was appended to working;
    'continue' — the reply was cut off; a resume request was appended;
    'done'     — the turn is finished (final_segments holds the full reply).
    Mutates working and final_segments in place.
    """
    import qwen_cli.main as _main

    # An empty final reply (no text, no tools) burns the whole turn — seen
    # in practice as "(no response)" iterations. Nudge the model once
    # before giving up.
    if not text.strip() and not final_segments and empty_retries < 1:
        empty_retries += 1
        _main.console.print("[yellow]  \\[empty response — asking the model to answer][/yellow]")
        working.append(
            {
                "role": "user",
                "content": (
                    "Your last response was empty. Answer the request now — if you need "
                    "information, call a tool; otherwise state your answer directly."
                ),
            }
        )
        return "nudge", auto_continue, empty_retries
    final_segments.append(text)
    # The model produced a final answer with no tool calls. If it was cut
    # off mid-output (token cap hit, or the stream dropped), it hasn't
    # actually finished — resume it instead of stopping.
    if usage.get("truncated") and text.strip() and auto_continue < _main.MAX_AUTO_CONTINUE:
        auto_continue += 1
        fr = usage.get("finish_reason") or "stream ended"
        _main.console.print(
            f"[yellow]  \\[response cut off ({fr}) — continuing {auto_continue}/{_main.MAX_AUTO_CONTINUE}][/yellow]",
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
        return "continue", auto_continue, empty_retries
    if usage.get("truncated") and auto_continue >= _main.MAX_AUTO_CONTINUE:
        _main.console.print(
            f"[yellow]  \\[still cut off after {_main.MAX_AUTO_CONTINUE} continuations — stopping][/yellow]",
        )
    return "done", auto_continue, empty_retries


def _parse_tool_args(tool_calls: list) -> list[dict | None]:
    """Parse each call's JSON arguments upfront; None marks malformed JSON."""
    parsed: list[dict | None] = []
    for tc in tool_calls:
        try:
            parsed.append(json.loads(tc["function"]["arguments"]))
        except json.JSONDecodeError:
            parsed.append(None)
    return parsed


def _execute_batches(
    client, tool_calls: list, parsed_args: list, tool_chain: list, last_user_msg: str
) -> dict[int, tuple[str, str]]:
    """Run one round of tool calls in conflict-free batches.

    Parallel-safe tools go through the thread pool with retries; stateful and
    interactive tools run serially in order. Returns {index: (tool_call_id,
    capped_result)}. KeyboardInterrupt propagates to the caller, which
    substitutes cancellation results to keep the tool_calls pairing intact.
    """
    import qwen_cli.main as _main

    tool_results: dict[int, tuple[str, str]] = {}  # idx -> (tc_id, result)
    batches = _classify_tool_batch(tool_calls, parsed_args)
    pool = _main._get_pool()
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
            if name in _main._PARALLEL_TOOLS:
                tool_chain.append(f"{name}({_main._short_args(name, args)})")
                safe_futures[i] = pool.submit(
                    _call_with_retry,
                    name,
                    args,
                    _main._call_tool_safe,
                    max_retries=_TOOL_RETRY_MAX,
                )
            elif name in _TOOL_RETRYABLE_TOOLS and name not in _main._PARALLEL_TOOLS:
                tool_chain.append(f"{name}({_main._short_args(name, args)})")
                result = _call_with_retry(name, args, _main._dispatch_interactive)
                tool_results[i] = (tc["id"], _smart_cap(client, result, name, last_user_msg))
            else:
                tool_chain.append(f"{name}({_main._short_args(name, args)})")
                result = _main._dispatch_interactive(name, args)
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
                    result = _call_with_retry(rname, parsed_args[idx], _main._call_tool_safe, max_retries=1)
                except Exception as exc2:
                    result = f"[tool error: {exc}; retry: {exc2}]"
            tool_results[i] = (
                tool_calls[i]["id"],
                _smart_cap(client, result, tool_calls[i]["function"]["name"], last_user_msg),
            )
    return tool_results


def reground_citations(client: object, messages: list, reply: str) -> str:
    """Force one grounding pass if `reply` cites file:line it never actually read.

    The system prompt forbids inventing line numbers; this checks. A live
    self-audit confidently placed _session_title at main.py:1210 (it is at 340)
    — it had grepped, not read. Shared by the REPL and both agent loops.

    Returns the corrected reply, or the original if there was nothing to fix.
    Mutates `messages` (the caller's working copy) as run_turn does.
    """
    import qwen_cli.main as _main

    bad = _main._unverified_citations(reply)
    # Files written this turn get the same scrutiny. A fabricated file:line that
    # reaches disk is worse than one merely spoken: it outlives the turn, and the
    # next reader — human or model — treats it as established fact. A live
    # self-audit wrote a report inventing five functions and their complexity
    # scores; every claim went through write_file, so checking only the chat reply
    # caught none of it.
    written_bad: list[tuple[str, list[str]]] = []
    for path, content in _main._turn_written:
        wb = _main._unverified_citations(content)
        if wb:
            written_bad.append((path, wb))

    if not bad and not written_bad:
        return reply

    all_bad = list(bad) + [c for _, cites in written_bad for c in cites]
    shown = ", ".join(all_bad[:6]) + (f" (+{len(all_bad) - 6} more)" if len(all_bad) > 6 else "")
    _main.console.print(f"[dim yellow]  [unverified citation(s): {shown} — checking before answering][/dim yellow]")

    # The correction is a whole extra turn, and _prepare_turn resets the per-turn
    # record of what happened. That record is not bookkeeping: /agent refuses
    # AGENT_DONE until _last_turn_tool_names shows a mutation was verified, and
    # the REPL writes _turn_ledger into history as the only trace of the turn's
    # tool work. Letting the correction pass clear them would erase the very
    # edits being verified, so carry both across.
    prior_tools = list(_main._last_turn_tool_names)
    prior_ledger = list(_main._turn_ledger)

    demand = (
        f"You cited {shown}, but nothing you read this turn actually showed you "
        f"{'those lines' if len(all_bad) > 1 else 'that line'}. Do not guess at line numbers, and do "
        "not invent function or symbol names. Use read_file (or search_files) to look at the real "
        "content now, then give your answer again with citations you have actually verified — "
        "correcting or dropping any that were wrong."
    )
    if written_bad:
        files = ", ".join(path for path, _ in written_bad)
        demand += (
            f"\n\nUnverified citations also went into {files}, which you WROTE this turn. A wrong "
            "citation on disk outlives this conversation and the next reader will trust it. After "
            "you have checked the real lines, correct that file with edit_file too — or delete the "
            "claims you cannot support."
        )

    messages.append({"role": "assistant", "content": reply})
    messages.append({"role": "user", "content": demand})
    with _main._main_llm_busy_lock:
        _main._main_llm_busy = True
    try:
        # presearch=False: this is a code-grounding correction, not a web question.
        corrected = _main.run_turn(client, messages, allow_tools=True, presearch=False)
    finally:
        with _main._main_llm_busy_lock:
            _main._main_llm_busy = False

    _main._last_turn_tool_names[:0] = prior_tools
    _main._turn_ledger[:0] = prior_ledger
    return corrected or reply


def run_turn(client: object, messages: list, allow_tools: bool = True, presearch: bool = True) -> str | None:
    """Full turn with tool-use loop. Returns reply, '' on cancel, None on error.

    presearch=False skips the auto-web-search grounding pass — used for
    synthetic follow-up messages (e.g. the hedging re-check) that already
    instruct the model to search itself.

    The loop skeleton lives here; each concern is a helper: _prepare_turn
    (state reset + grounding), _stream_live/_strip_and_show_thinking
    (streaming + rendering), _synthesize_at_depth_cap (round-cap exit),
    _reply_action (empty-nudge / auto-continue / done), _parse_tool_args and
    _execute_batches (tool dispatch).
    """
    import qwen_cli.main as _main

    working = _prepare_turn(messages, allow_tools, presearch)
    # The request this turn must keep answering — re-anchored after mid-run
    # compaction. Taken from the original messages so the volatile git tail and
    # presearch additions don't leak into the anchor text.
    turn_task = next(
        (m.get("content") or "" for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    use_tools = allow_tools
    first_call = True
    depth = 0
    tool_chain: list[str] = []
    total_prompt = 0
    total_completion = 0
    final_segments: list[str] = []  # final-answer pieces, joined across auto-continues
    auto_continue = 0  # how many times we've resumed a cut-off answer
    empty_retries = 0  # nudges sent after an empty final reply (no text, no tools)

    def _finish(reply):
        # Shared successful-exit path: show the tool chain, record token totals.
        # Cancel ('') and error (None) exits deliberately skip this, as before.
        if tool_chain:
            _main.console.print(f"[dim]  ↳ {' → '.join(tool_chain)}[/dim]")
        _main._last_turn_tokens = {"prompt": total_prompt, "completion": total_completion}
        return reply

    while True:
        # Mid-run compaction: a long tool loop can pile up large tool results and
        # overflow the context window before the task finishes. When the working set
        # nears the limit, shrink older tool results in place so the run keeps going.
        # approx_tokens underestimates when tool results are token-dense, so also
        # trust the real prompt count the server reported for the previous call.
        if depth > 0 and max(approx_tokens(working), _main._real_ctx_tokens) >= _main.TOKEN_LIMIT * 0.85:
            working = _compact_tool_loop(working)
            working = _refresh_task_anchor(working, turn_task)

        if depth >= _main.MAX_TOOL_DEPTH:
            _main._turn_hit_round_cap = True  # REPL may auto-continue if the plan is unfinished
            final_text, final_usage = _synthesize_at_depth_cap(client, working)
            total_prompt += final_usage.get("prompt", 0)
            total_completion += final_usage.get("completion", 0)
            return _finish(("".join(final_segments) + (final_text or "")) or None)

        depth_tag = f" [dim][{depth}/{_main.MAX_TOOL_DEPTH}][/dim]" if depth > 0 else ""
        _main.console.print(Rule(f"[bold green]Qwen[/bold green]{depth_tag}", style="dim green"))

        try:
            text, tool_calls, usage = _stream_live(client, working, use_tools)
            text = _strip_and_show_thinking(text)
            if text:
                _main.console.print(Markdown(text))
            elif not tool_calls:
                _main.console.print(Markdown("*(no response)*"))
        except KeyboardInterrupt:
            _main.console.print("\n[dim][cancelled][/dim]")
            return ""
        except Exception as e:
            if not (use_tools and first_call):
                _main.console.print(f"[red][error] {e}[/red]")
                return None
            # First call failed with tools enabled — retry once without tools.
            _main.console.print(f"[yellow][retrying without tool use — {e}][/yellow]")
            use_tools = False
            first_call = False
            try:
                text, tool_calls, usage = _stream_live(client, working, False)
                text = _strip_and_show_thinking(text)
                _main.console.print(Markdown(text) if text else Markdown("*(no response)*"))
            except KeyboardInterrupt:
                _main.console.print("\n[dim][cancelled][/dim]")
                return ""
            except Exception as e2:
                _main.console.print(f"[red][error] {e2}[/red]")
                return None

        total_prompt += usage.get("prompt", 0)
        total_completion += usage.get("completion", 0)
        # Keep the most recent real prompt token count so auto-compact uses it
        if usage.get("prompt", 0):
            _main._real_ctx_tokens = usage["prompt"]

        if not tool_calls:
            action, auto_continue, empty_retries = _reply_action(
                text, usage, working, final_segments, auto_continue, empty_retries
            )
            if action == "done":
                return _finish("".join(final_segments))
            continue  # 'nudge' and 'continue' both go another round

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

        parsed_args = _parse_tool_args(tool_calls)
        # Extract user query for summarization context
        last_user_msg = next(
            (m.get("content", "") for m in reversed(working) if m.get("role") == "user"),
            "",
        )

        try:
            tool_results = _execute_batches(client, tool_calls, parsed_args, tool_chain, last_user_msg)
            for i in range(len(tool_calls)):
                tc_id, result = tool_results[i]
                working.append({"role": "tool", "tool_call_id": tc_id, "content": result})
                args_i = parsed_args[i] if isinstance(parsed_args[i], dict) else {}
                entry = _ledger_entry(tool_calls[i]["function"]["name"], args_i, result)
                if entry:
                    _main._turn_ledger.append(entry)

            first_call = False
            depth += 1
        except KeyboardInterrupt:
            # Ctrl+C here means "stop", so actually stop: this used to fall
            # through to the next loop iteration and fire another LLM call
            # despite printing "returning to prompt", forcing a second Ctrl+C.
            # Pair every outstanding tool_call with a result first — the API
            # rejects an assistant tool_calls message with no matching tool
            # message, and `working` is the list a caller may still hold.
            _main.console.print("\n[dim][tools cancelled — returning to prompt][/dim]")
            for i in range(len(tool_calls)):
                working.append(
                    {"role": "tool", "tool_call_id": tool_calls[i]["id"], "content": "[user cancelled tool execution]"}
                )
            return ""


