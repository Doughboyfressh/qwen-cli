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
    def _reset() -> None:
        for attr in ("_turn_seen_lines", "_injected_files"):
            getattr(qwen_cli, attr).clear()
        qwen_cli._last_turn_tool_names[:] = []
        qwen_cli._turn_ledger[:] = []
        qwen_cli._turn_written[:] = []

    _reset()
    yield
    _reset()


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


def test_fabricated_citation_written_to_a_file_is_caught(qwen_cli, code_file, tmp_path):
    """The laundering case: the claim goes to disk, not into the chat reply.

    A live self-audit wrote a report inventing five functions and their complexity
    scores. Every claim went through write_file, so a guard that only inspected
    the spoken reply caught none of it — and the bad citations outlived the turn.
    """
    from qwen_cli.tools.files import do_write_file

    with patch.object(qwen_cli, "_auto_approve", True):
        do_write_file(str(tmp_path / "report.md"), f"The bug lives at {code_file.name}:999.")

    demands: list[str] = []

    def fake_run_turn(client, messages, allow_tools=True, presearch=True):
        demands.append(messages[-1]["content"])
        return "Corrected."

    with patch.object(qwen_cli, "run_turn", side_effect=fake_run_turn):
        out = qwen_cli.reground_citations(object(), [], "Wrote the report.")  # reply itself is clean

    assert out == "Corrected."
    assert demands, "a bad citation inside a written file must still force a grounding pass"
    assert f"{code_file.name}:999" in demands[0]
    assert "report.md" in demands[0], "the model must be told which file to correct"


def test_clean_written_file_does_not_fire(qwen_cli, code_file, tmp_path):
    from qwen_cli.tools.files import do_read_file, do_write_file

    do_read_file(str(code_file), offset=1, limit=20)
    with patch.object(qwen_cli, "_auto_approve", True):
        do_write_file(str(tmp_path / "report.md"), f"See {code_file.name}:5 — verified.")

    calls = []
    with patch.object(qwen_cli, "run_turn", side_effect=lambda *a, **k: calls.append(1)):
        out = qwen_cli.reground_citations(object(), [], "Wrote the report.")

    assert out == "Wrote the report."
    assert not calls, "a verified citation in a written file must not cost a round-trip"


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


# ---------------------------------------------------------------------------
# Symbol guard: invented function names attributed to real files
#
# The citation guard checks file:line. It did not catch what actually did the
# damage in a live self-audit: cmd_repl, cmd_browser, cmd_fetch, _run_agent_loop
# and _browser_init were reported with complexity scores and line counts. None of
# them exist. Two P0 recommendations were built on them.
#
# The false-negative bias is deliberate. A name found ANYWHERE in the source is
# accepted, and a false accusation costs a full round-trip — so most of these
# tests pin down when the guard must stay silent.
# ---------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "core").mkdir()
    (tmp_path / "core" / "repl.py").write_text(
        "def read_input():\n    pass\n\ndef _repl_loop(ctx):\n    pass\n", encoding="utf-8"
    )
    (tmp_path / "main.py").write_text("import subprocess\n\ndef main():\n    subprocess.run(['git'])\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_invented_symbol_attributed_to_a_real_file_is_caught(qwen_cli, project):
    """The exact failure: a table row about a function nobody ever wrote."""
    text = "| 29 | `core/repl.py` | `cmd_repl` | 311 | CRITICAL |"
    assert qwen_cli._unverified_symbols(text) == ["cmd_repl"]


def test_multiple_invented_symbols(qwen_cli, project):
    text = "In `main.py`, `cmd_browser` and `cmd_fetch` are the worst offenders."
    assert qwen_cli._unverified_symbols(text) == ["cmd_browser", "cmd_fetch"]


def test_real_symbol_is_not_flagged(qwen_cli, project):
    assert qwen_cli._unverified_symbols("`core/repl.py` defines `_repl_loop`.") == []


def test_symbol_that_only_appears_as_a_call_is_not_flagged(qwen_cli, project):
    """Existence is a raw substring search — a call site is proof enough that the
    name is real. The index only holds top-level defs; trusting it would accuse
    every method and every import."""
    assert qwen_cli._unverified_symbols("`main.py` shells out via `subprocess`.") == []


def test_proposals_are_not_flagged(qwen_cli, project):
    """Code that does not exist YET is the whole point of a suggestion."""
    for line in [
        "In `core/repl.py` you should extract a `handle_input` helper.",
        "Consider adding `_dispatch_command` to `main.py`.",
        "Split `main.py` and create `parse_args` there.",
    ]:
        assert qwen_cli._unverified_symbols(line) == [], line


def test_symbol_without_a_file_reference_is_not_flagged(qwen_cli, project):
    """No file attribution means no claim about THIS project — could be anything."""
    assert qwen_cli._unverified_symbols("You could use `functools.lru_cache` or `some_helper`.") == []


def test_symbols_in_code_blocks_are_not_flagged(qwen_cli, project):
    text = "Example for `main.py`:\n```python\ndef totally_made_up():\n    pass\n```"
    assert qwen_cli._unverified_symbols(text) == []


def test_short_identifiers_are_ignored(qwen_cli, project):
    """`ctx`, `p`, `os` — too short to be a confident claim, too common to risk."""
    assert qwen_cli._unverified_symbols("`core/repl.py` takes `ctx`.") == []


def test_symbol_guard_fires_through_reground(qwen_cli, project):
    demands: list[str] = []

    def fake_run_turn(client, messages, allow_tools=True, presearch=True):
        demands.append(messages[-1]["content"])
        return "Corrected: no such function."

    with patch.object(qwen_cli, "run_turn", side_effect=fake_run_turn):
        out = qwen_cli.reground_citations(object(), [], "`core/repl.py` defines `cmd_repl` (CC 29).")

    assert out == "Corrected: no such function."
    assert "cmd_repl" in demands[0]
    assert "invented" in demands[0].lower()


def test_symbol_guard_silent_on_an_honest_reply(qwen_cli, project):
    calls = []
    with patch.object(qwen_cli, "run_turn", side_effect=lambda *a, **k: calls.append(1)):
        out = qwen_cli.reground_citations(object(), [], "`core/repl.py` defines `_repl_loop` and `read_input`.")
    assert out == "`core/repl.py` defines `_repl_loop` and `read_input`."
    assert not calls, "an honest reply must not cost an extra round-trip"


def test_a_name_only_in_comments_or_strings_does_not_count_as_existing(qwen_cli, tmp_path, monkeypatch):
    """A hallucinated name must not launder itself into existence by being written down.

    The first cut searched raw text, so the comment in main.py documenting
    `cmd_repl` as a fabrication — and these very tests asserting it is one — made
    `cmd_repl` look real, and the guard went silent. Existence is now decided from
    code with comments and string literals stripped.
    """
    (tmp_path / "m.py").write_text(
        '# cmd_repl was reported by the audit but does not exist\n'
        'NOTE = "see cmd_repl"\n'
        '"""cmd_repl appears in this docstring too."""\n'
        'def real_function():\n    pass\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    qwen_cli._SYMBOL_EXISTS_MEMO.clear()
    qwen_cli._PROJECT_FILES_MEMO.clear()

    assert not qwen_cli._symbol_exists_in_project("cmd_repl"), "prose must not prove existence"
    assert qwen_cli._symbol_exists_in_project("real_function")
    assert qwen_cli._unverified_symbols("`m.py` defines `cmd_repl`.") == ["cmd_repl"]


def test_package_relative_paths_are_judgeable(qwen_cli, tmp_path, monkeypatch):
    """A model describing this codebase writes "core/repl.py", but the file is at
    qwen_cli/core/repl.py — resolving from the project root finds nothing. Without
    suffix matching the whole line was unjudgeable and the guard skipped it, which
    is exactly why it missed every fabrication in the real audit."""
    (tmp_path / "pkg" / "core").mkdir(parents=True)
    (tmp_path / "pkg" / "core" / "repl.py").write_text("def _repl_loop():\n    pass\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    qwen_cli._SYMBOL_EXISTS_MEMO.clear()
    qwen_cli._PROJECT_FILES_MEMO.clear()

    assert qwen_cli._citation_is_judgeable("core/repl.py"), "package-relative path must be placeable"
    assert qwen_cli._unverified_symbols("`core/repl.py` defines `cmd_repl`.") == ["cmd_repl"]
    assert qwen_cli._unverified_symbols("`core/repl.py` defines `_repl_loop`.") == []
