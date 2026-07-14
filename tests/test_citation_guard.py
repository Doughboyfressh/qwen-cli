"""The citation guard: file:line claims must be backed by something the model read.

The system prompt already forbids inventing line numbers. A live self-audit
obeyed the letter of that rule and still placed _session_title at main.py:1210
(it is at 340) — it had grepped, not read. Asking nicely is not the same as
checking, so the REPL now verifies citations against an evidence ledger of the
lines actually displayed this turn, and forces one grounding pass if any claim
is unsupported.

The false-positive tests matter most: a wrong accusation costs a whole
round-trip, so the guard must stay quiet on anything it cannot adjudicate.
"""

from unittest.mock import patch

import pytest

from qwen_cli.tools.files import do_read_file, do_search_files


@pytest.fixture(autouse=True)
def _clean_ledger(qwen_cli):
    """Reset every piece of per-turn state these tests touch, on the way in AND out.

    _last_turn_tool_names and _turn_ledger are module-level lists mutated in
    place. Leaving a stray ledger entry behind makes the REPL append a
    '[turn actions: ...]' tag to history in an unrelated test, which fails there
    and passes in isolation — the worst kind of test bug.
    """
    for attr in ("_turn_seen_lines", "_injected_files"):
        getattr(qwen_cli, attr).clear()
    qwen_cli._last_turn_tool_names[:] = []
    qwen_cli._turn_ledger[:] = []
    yield
    for attr in ("_turn_seen_lines", "_injected_files"):
        getattr(qwen_cli, attr).clear()
    qwen_cli._last_turn_tool_names[:] = []
    qwen_cli._turn_ledger[:] = []


@pytest.fixture
def code_file(tmp_path, monkeypatch):
    # chdir: the guard only adjudicates files it can place — one it read this turn,
    # or one that exists relative to the cwd. In a real session the cwd IS the
    # project being discussed, so the test has to stand somewhere too.
    p = tmp_path / "sample.py"
    p.write_text("\n".join(f"line_{i}" for i in range(1, 61)) + "\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return p


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See main.py:340 for the definition.", [("main.py", 340)]),
        ("It is on line 340 of main.py.", [("main.py", 340)]),
        ("Look at `core/repl.py` line 12.", [("core/repl.py", 12)]),
        ("qwen_cli/tools/files.py:99 and main.py:1", [("qwen_cli/tools/files.py", 99), ("main.py", 1)]),
        ("No citations here at all.", []),
    ],
)
def test_extracts_citations(qwen_cli, text, expected):
    assert qwen_cli._extract_citations(text) == expected


def test_ignores_citations_inside_code_blocks(qwen_cli):
    """A line number inside a quoted traceback or diff is data, not a claim."""
    text = 'Here is the error:\n```\n  File "app.py", line 42, in main\n```\nThat is the trace.'
    assert qwen_cli._extract_citations(text) == []


# ---------------------------------------------------------------------------
# The guard fires on fabrication
# ---------------------------------------------------------------------------


def test_citation_to_a_line_never_read_is_unverified(qwen_cli, code_file):
    do_read_file(str(code_file), offset=1, limit=10)  # saw lines 1-10 only
    assert qwen_cli._unverified_citations(f"The bug is at {code_file.name}:55") == [f"{code_file.name}:55"]


def test_citation_to_an_unread_file_is_unverified(qwen_cli, code_file):
    assert qwen_cli._unverified_citations(f"See {code_file.name}:3") == [f"{code_file.name}:3"]


def test_the_real_world_case(qwen_cli, code_file):
    """Grepping and then citing a line you never looked at — the exact failure."""
    do_search_files(str(code_file.parent), "line_7", "*.py", 0)  # shows line 7
    assert qwen_cli._unverified_citations(f"{code_file.name}:1210 defines it") == [f"{code_file.name}:1210"]


# ---------------------------------------------------------------------------
# The guard stays quiet when the model really did look (no false positives)
# ---------------------------------------------------------------------------


def test_read_lines_are_verified(qwen_cli, code_file):
    do_read_file(str(code_file), offset=1, limit=20)
    assert qwen_cli._unverified_citations(f"See {code_file.name}:15") == []


def test_full_read_verifies_any_line(qwen_cli, code_file):
    do_read_file(str(code_file))
    assert qwen_cli._unverified_citations(f"See {code_file.name}:60") == []


def test_search_hit_verifies_that_line(qwen_cli, code_file):
    """A grep DOES prove the line number, even though it proves nothing about behavior."""
    do_search_files(str(code_file.parent), "line_42", "*.py", 0)
    assert qwen_cli._unverified_citations(f"Defined at {code_file.name}:42") == []


def test_search_context_lines_are_verified(qwen_cli, code_file):
    do_search_files(str(code_file.parent), "line_42", "*.py", 2)  # shows 40-44
    assert qwen_cli._unverified_citations(f"See {code_file.name}:44") == []


def test_injected_file_verifies_any_line(qwen_cli, code_file):
    """@file / /file paste the whole file into context — the model can see all of it."""
    qwen_cli._injected_files.add(str(code_file))
    assert qwen_cli._unverified_citations(f"See {code_file.name}:57") == []


def test_unknown_file_is_not_flagged(qwen_cli):
    """A traceback from a library, or a file that isn't ours, can't be adjudicated."""
    assert qwen_cli._unverified_citations("django/db/models.py:1204 raises it") == []
    assert qwen_cli._unverified_citations("their example was at foo_not_real.py:9") == []


def test_relative_path_matches_the_absolute_read(qwen_cli, code_file):
    do_read_file(str(code_file), offset=5, limit=3)  # lines 5-7
    assert qwen_cli._unverified_citations(f"see ./{code_file.name}:6") == []


# ---------------------------------------------------------------------------
# Enforcement: the REPL forces a grounding pass and keeps the corrected answer
# ---------------------------------------------------------------------------


def test_repl_forces_a_grounding_pass_on_a_bad_citation(qwen_cli, code_file):
    from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

    replies = iter([f"It is at {code_file.name}:999.", f"Corrected: it is at {code_file.name}:3."])
    calls: list[bool] = []

    def fake_run_turn(client, messages, allow_tools=True, presearch=True):
        calls.append(presearch)
        return next(replies)

    ctx = _ReplContext([], "sys", object())
    with (
        patch.object(qwen_cli, "run_turn", side_effect=fake_run_turn),
        patch.object(qwen_cli, "build_system_prompt", return_value="sys"),
        patch.object(qwen_cli, "_maybe_autocompact", side_effect=lambda h, b, c: h),
        patch.object(qwen_cli, "_silent_autosave"),
        patch.object(qwen_cli, "_looks_like_plan", return_value=False),
        patch.object(qwen_cli, "_extract_runnable_code", return_value=None),
        patch.object(qwen_cli, "_run_background_tasks"),
        patch.object(qwen_cli, "_HEDGE_RE", __import__("re").compile(r"(?!x)x")),  # no hedging
    ):
        _run_turn_and_handle_reply(ctx, "where is it?", allow_tools=True)

    assert len(calls) == 2, "an unverified citation must force a second, grounding turn"
    assert calls[1] is False, "the grounding pass must not web-presearch its own critique"
    assert "Corrected" in ctx.history[-1]["content"]


def test_reground_preserves_the_turn_record(qwen_cli, code_file):
    """The correction pass must not erase what the main turn actually did.

    run_turn resets _last_turn_tool_names and _turn_ledger, and /agent refuses
    AGENT_DONE until the tool names show a mutation was verified. If regrounding
    wiped them, the guard would destroy the evidence the verification gate reads.
    """
    qwen_cli._last_turn_tool_names[:] = ["edit_file", "read_file"]
    qwen_cli._turn_ledger[:] = ["edit_file sample.py"]

    def fake_run_turn(client, messages, allow_tools=True, presearch=True):
        # what a real correction turn does: clears the record, then logs its own
        qwen_cli._last_turn_tool_names[:] = ["search_files"]
        qwen_cli._turn_ledger[:] = ["read sample.py (60 lines)"]
        return f"Corrected: {code_file.name}:3."

    with patch.object(qwen_cli, "run_turn", side_effect=fake_run_turn):
        out = qwen_cli.reground_citations(object(), [], f"It is at {code_file.name}:999.")

    assert out == f"Corrected: {code_file.name}:3."
    assert "edit_file" in qwen_cli._last_turn_tool_names, "the mutation record was destroyed"
    assert qwen_cli._last_turn_tool_names == ["edit_file", "read_file", "search_files"]
    assert qwen_cli._turn_ledger == ["edit_file sample.py", "read sample.py (60 lines)"]


def test_agent_loop_regrounds_a_bad_citation(qwen_cli, code_file):
    """/agent reports its own progress — a fabricated line there becomes the
    next iteration's premise, so the loop gets the guard too."""
    from qwen_cli.core.agent import cmd_agent

    replies = iter([f"Done. See {code_file.name}:999.", f"Corrected: {code_file.name}:4. AGENT_DONE"])

    def fake_run_turn(client, messages, allow_tools=True, presearch=True):
        return next(replies)

    history: list = []
    with (
        patch.object(qwen_cli, "run_turn", side_effect=fake_run_turn),
        patch.object(qwen_cli, "build_system_prompt", return_value="sys"),
        patch.object(qwen_cli, "_maybe_autocompact", side_effect=lambda w, b, c: w),
        patch.object(qwen_cli, "_auto_approve", True),
    ):
        cmd_agent("do a thing", history, "sys", object(), max_iter=3)

    final = history[-1]["content"]
    assert "Corrected" in final, "the agent recorded the fabricated reply instead of the corrected one"
    assert "999" not in final


def test_repl_does_not_fire_on_a_verified_citation(qwen_cli, code_file):
    from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

    do_read_file(str(code_file), offset=1, limit=20)
    calls: list[bool] = []

    def fake_run_turn(client, messages, allow_tools=True, presearch=True):
        calls.append(presearch)
        return f"It is at {code_file.name}:5."

    ctx = _ReplContext([], "sys", object())
    with (
        patch.object(qwen_cli, "run_turn", side_effect=fake_run_turn),
        patch.object(qwen_cli, "build_system_prompt", return_value="sys"),
        patch.object(qwen_cli, "_maybe_autocompact", side_effect=lambda h, b, c: h),
        patch.object(qwen_cli, "_silent_autosave"),
        patch.object(qwen_cli, "_looks_like_plan", return_value=False),
        patch.object(qwen_cli, "_extract_runnable_code", return_value=None),
        patch.object(qwen_cli, "_run_background_tasks"),
        patch.object(qwen_cli, "_HEDGE_RE", __import__("re").compile(r"(?!x)x")),
    ):
        _run_turn_and_handle_reply(ctx, "where is it?", allow_tools=True)

    assert len(calls) == 1, "a verified citation must not cost an extra round-trip"
