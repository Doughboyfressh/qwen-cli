"""Autonomous agent loops — /agent and /task.

Extracted from main.py as part of the monolith split. cmd_agent runs the
verification-gated goal loop (AGENT_DONE is rejected until a mutating change
has been verified); cmd_task runs the plan-first stepwise loop. Both drive
run_turn and the visible plan.

Loop state (_current_plan, _auto_approve, _last_turn_tool_names) stays OWNED
BY MAIN and is read/rebound as _main.<name> at call time — the REPL renders
the plan from main, /auto toggles approval there, and tests script run_turn
and patch the collaborators on main. Same convention as core/turn.py.
"""

import re
from pathlib import Path

from rich.markdown import Markdown
from rich.rule import Rule

from qwen_cli.core.config import AUX_LLM_TIMEOUT

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
    import qwen_cli.main as _main

    steps = _main._PLAN_RE.findall(text)
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
    import qwen_cli.main as _main

    count = len(_main._HEDGE_RE.findall(text))
    if count >= 3:
        _main.console.print(
            f"[dim yellow]  [low confidence — {count} hedge phrases — consider /retry with more context][/dim yellow]",
        )


def _verification_pending(tool_names: list, pending: bool = False) -> bool:
    """Track whether a file mutation happened without a subsequent verifying tool call.

    Scans tool names in order: a mutating tool sets the pending flag; any verifying
    tool that runs AFTER it clears the flag. Carries state across turns via `pending`.
    """
    import qwen_cli.main as _main

    for name in tool_names:
        if name in _main._MUTATING_FILE_TOOLS:
            pending = True
        elif pending and name in _main._VERIFYING_TOOLS:
            pending = False
    return pending


def cmd_agent(goal: str, history: list, base_system: str, client: object, max_iter: int = 20) -> None:
    """Autonomous agent loop: model iterates with tools until it emits AGENT_DONE.

    Completion is ENFORCED, not just prompted: if the agent modified files, AGENT_DONE
    is rejected until at least one verifying tool (tests, execution, re-read) has run
    after the last mutation.
    """
    import qwen_cli.main as _main

    agent_suffix = (
        "\n\nYou are running in autonomous agent mode. Work toward the goal using tools. "
        "PLAN: call update_plan FIRST with your full step breakdown (status 'pending' for all). "
        "Before starting each step call update_plan again marking it 'in_progress'; after you've "
        "verified it, call update_plan marking it 'completed'. Always pass the complete step list. "
        "After each action, briefly verify the result is correct before moving on — "
        "do not assume success; check the output or re-read the file. "
        "ENFORCEMENT: if you modify any file, AGENT_DONE will be REJECTED until you verify "
        "the change afterward (run the tests, execute the code, or re-read the file and check it). "
        "DELEGATION: if the goal has 3+ independent subtasks, or any subtask would take 5+ tool calls, "
        "spawn subagents via team_spawn_agent instead of doing everything yourself. "
        "Subagents have full access to all the same tools. You coordinate, they execute in parallel. "
        "When the goal is fully achieved and you have confirmed the result, end your response with exactly: AGENT_DONE"
    )
    _main._current_plan = []
    _main.console.print(f"[bold cyan]  Agent goal:[/bold cyan] {goal}")
    _main.console.print(f"[dim]  (max {max_iter} iterations — Ctrl+C to stop)[/dim]")

    # An "autonomous" loop that blocks on y/N for every file edit isn't autonomous —
    # offer to auto-approve edits up front for the duration of this run only.
    restore_auto = _main._auto_approve
    if not _main._auto_approve:
        try:
            ans = _main.console.input("[bold yellow]  Auto-approve file edits for this run? [Y/n]: [/bold yellow]").strip().lower()
            if ans in ("", "y", "yes"):
                _main._auto_approve = True
        except (KeyboardInterrupt, EOFError):
            pass

    working = list(history)
    working.append({"role": "user", "content": f"[Agent task] {goal}"})

    pending_verify = False
    verify_rejections = 0
    agent_replies: list[str] = []
    try:
        for iteration in range(1, max_iter + 1):
            _main.console.print(Rule(f"[dim]Agent {iteration}/{max_iter}[/dim]", style="dim"))
            # Keep the agent going across many iterations — summarize+preserve the task
            # if the accumulated working set is nearing the context limit.
            working = _main._maybe_autocompact(working, base_system, client)
            msgs = [{"role": "system", "content": _main.build_system_prompt(base_system) + agent_suffix}, *working]
            try:
                reply = _main.run_turn(client, msgs, allow_tools=True)
            except KeyboardInterrupt:
                _main.console.print("\n[dim]  \\[agent stopped][/dim]")
                break

            if not reply:
                _main.console.print("[red]  \\[agent error or cancelled — stopping][/red]")
                break

            # An agent reports its own progress; a fabricated file:line in that
            # report is how a wrong claim becomes the next iteration's premise.
            # reground_citations preserves _last_turn_tool_names, so the
            # verification gate below still sees this turn's real mutations.
            reply = _main.reground_citations(client, msgs, reply)

            working.append({"role": "assistant", "content": reply})
            agent_replies.append(reply)

            pending_verify = _verification_pending(list(_main._last_turn_tool_names), pending_verify)

            if re.search(r"\bAGENT_DONE\b", reply):
                if pending_verify and verify_rejections < 3:
                    verify_rejections += 1
                    _main.console.print(
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
                    _main.console.print(
                        "[yellow]  \\[agent finished WITHOUT verification after 3 rejections — review changes manually][/yellow]"
                    )
                _main.console.print(
                    f"[green]  \\[agent done in {iteration} iteration{'s' if iteration != 1 else ''}][/green]"
                )
                break

            working.append({"role": "user", "content": "Continue working toward the goal."})
        else:
            _main.console.print(f"[yellow]  \\[agent reached max iterations ({max_iter}) — goal may be unfinished][/yellow]")
    finally:
        # Auto-approve granted for this run must not leak into the interactive session.
        _main._auto_approve = restore_auto
        # Record ONE consolidated turn instead of a "[continue]"/reply pair per
        # iteration — saved sessions used to replay agent scaffolding as real
        # conversation. All iteration replies are preserved, joined in order.
        if agent_replies:
            history.append({"role": "user", "content": f"[Agent task] {goal}"})
            history.append({"role": "assistant", "content": "\n\n---\n\n".join(agent_replies)})



def cmd_task(goal: str, history: list, base_system: str, client: object) -> None:
    """Plan-approve-execute-test agentic task loop."""
    import qwen_cli.main as _main

    root = Path.cwd()
    test_cmd = _main._detect_test_command(root)

    _main.console.print(f"[bold cyan]  Task:[/bold cyan] {goal}")
    if test_cmd:
        _main.console.print(f"[dim]  Tests: {test_cmd}[/dim]")

    # Generate plan (no tools, fast)
    _main.console.print("[dim]  Planning...[/dim]")
    plan_msgs = [
        {"role": "system", "content": _main.build_system_prompt(base_system)},
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
            model=_main.MODEL,
            messages=plan_msgs,
            stream=False,
            max_tokens=400,
            timeout=AUX_LLM_TIMEOUT,
        )
        plan_text = (plan_resp.choices[0].message.content or "").strip()
    except Exception as e:
        _main.console.print(f"[red][error generating plan: {e}][/red]")
        return

    _main.console.print(Markdown(plan_text))

    # Approve plan
    try:
        approval = _main.console.input("\n[bold yellow]  Execute? [y/edit/N]: [/bold yellow]").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return
    if approval in ("n", ""):
        _main.console.print("[dim][task cancelled][/dim]")
        return
    if approval == "edit":
        try:
            _main.console.print("[dim]  Enter revised plan (blank line to finish):[/dim]")
            lines: list[str] = []
            while True:
                ln = _main.console.input("  ").rstrip()
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

    # /task already has a deterministic, parsed step list — seed the visible
    # plan from it directly rather than asking the model to call update_plan,
    # so the checklist always exactly matches what will actually execute.
    _main._current_plan = [{"text": s, "status": "pending"} for s in step_lines]

    for i, step in enumerate(step_lines, 1):
        label = step[:70] + ("..." if len(step) > 70 else "")
        _main.console.print(Rule(f"[dim]Step {i}/{total}: {label}[/dim]", style="dim"))
        _main._current_plan[i - 1]["status"] = "in_progress"
        _main._render_plan_panel()

        # Keep multi-step tasks within the context window — summarize+preserve the
        # task so far if the accumulated working set is nearing the limit.
        working = _main._maybe_autocompact(working, base_system, client)

        retries = 0
        extra_ctx = ""
        while retries <= 2:
            step_msgs = [{"role": "system", "content": _main.build_system_prompt(base_system)}, *working]
            step_msgs.append({"role": "user", "content": f"Execute step {i}/{total}: {step}{extra_ctx}"})
            try:
                reply = _main.run_turn(client, step_msgs, allow_tools=True)
            except KeyboardInterrupt:
                _main.console.print("\n[dim]  \\[task stopped][/dim]")
                return
            if not reply:
                _main.console.print(f"[red]  \\[step {i} failed — stopping][/red]")
                return

            # Same guard as /agent: the step report must not cite lines it never
            # read. Preserves _last_turn_tool_names, which the `modified` check
            # below reads to decide whether to run the tests.
            reply = _main.reground_citations(client, step_msgs, reply)

            working.append({"role": "user", "content": f"Step {i}: {step}"})
            working.append({"role": "assistant", "content": reply})
            history.append({"role": "user", "content": f"[Task {i}/{total}] {step}"})
            history.append({"role": "assistant", "content": reply})

            # Run tests if code was written/patched. Checks which tools actually ran
            # this step (same mechanism /agent's verification enforcement uses) rather
            # than string-matching the model's free-text reply for bracket tags it
            # isn't guaranteed to echo back verbatim.
            modified = any(name in _main._MUTATING_FILE_TOOLS for name in _main._last_turn_tool_names)
            if test_cmd and modified:
                _main.console.print(f"[dim]  Running: {test_cmd}[/dim]")
                test_out = _main.do_run_command(test_cmd, timeout=60)
                # Check for actual test failures — require non-zero counts to avoid
                # "0 failed" or "0 errors" in a passing summary line being a false positive
                has_fail = re.search(r"\b(FAILED|ERROR)\b", test_out)
                has_count = re.search(r"\b[1-9]\d*\s+(?:failed|error)\b", test_out, re.IGNORECASE)
                has_fail_test = re.search(r"\b(FAIL|ERROR)\b.*\b(test|spec|suite)\b", test_out)
                passed = not (has_fail or has_count or has_fail_test)
                if not passed and retries < 2:
                    _main.console.print(f"[yellow]  Tests failed — retrying step {i} (attempt {retries + 2}/3)[/yellow]")
                    extra_ctx = (
                        f"\n\nPrevious attempt failed tests:\n{test_out[:1500]}\nFix the issue then complete step {i}."
                    )
                    retries += 1
                    continue
                if not passed:
                    _main.console.print("[yellow]  Tests still failing — moving on[/yellow]")
            break

        _main._current_plan[i - 1]["status"] = "completed"
        _main._render_plan_panel()

        if i < total:
            try:
                fb = (
                    _main.console.input(
                        f"[dim]  [Step {i}/{total} done — Enter=next  f=feedback  s=stop]: [/dim]",
                    )
                    .strip()
                    .lower()
                )
            except (KeyboardInterrupt, EOFError):
                fb = "s"
            if fb == "s":
                _main.console.print("[dim][task stopped by user][/dim]")
                return
            if fb.startswith("f"):
                try:
                    feedback = _main.console.input("  Feedback: ").strip()
                except (KeyboardInterrupt, EOFError):
                    feedback = ""
                if feedback:
                    note = f"[User feedback after step {i}]: {feedback}"
                    working.append({"role": "user", "content": note})
                    history.append({"role": "user", "content": note})

    _main.console.print("[bold green]  Task complete.[/bold green]")


