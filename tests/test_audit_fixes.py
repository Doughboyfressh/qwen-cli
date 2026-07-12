"""Tests for the 2026-07 audit fixes: read_file pagination, turn ledger,
auto-continue at the round cap, edit_file tolerant matching, and diff preview
before edit approval."""

import json
from contextlib import ExitStack
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# #1 read_file pagination — honest headers, per-call caps, continuation hints
# ---------------------------------------------------------------------------


class TestReadPagination:
    def _read(self, qwen_cli, monkeypatch, path, **kw):
        monkeypatch.setattr(qwen_cli, "_turn_read_cache", set())
        monkeypatch.setattr(qwen_cli.console, "print", lambda *a, **k: None)
        return qwen_cli.do_read_file(str(path), **kw)

    def test_small_file_returned_whole(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "small.txt"
        p.write_text("\n".join(f"line {i}" for i in range(1, 51)), encoding="utf-8")
        result = self._read(qwen_cli, monkeypatch, p)
        assert "(50 lines)" in result
        assert "line 50" in result
        assert "Read the next part" not in result

    def test_large_file_capped_with_honest_header(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "big.txt"
        p.write_text("\n".join(f"line {i}" for i in range(1, 1264)), encoding="utf-8")
        result = self._read(qwen_cli, monkeypatch, p)
        assert "lines 1–400 of 1263" in result
        assert "line 400" in result
        assert "line 401" not in result
        assert "offset=401" in result  # continuation hint

    def test_offset_continues_where_hint_says(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "big.txt"
        p.write_text("\n".join(f"line {i}" for i in range(1, 1264)), encoding="utf-8")
        result = self._read(qwen_cli, monkeypatch, p, offset=401)
        assert "lines 401–800 of 1263" in result
        assert "line 401" in result and "line 800" in result
        assert "offset=801" in result

    def test_explicit_limit_over_cap_is_capped(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "big.txt"
        p.write_text("\n".join(f"line {i}" for i in range(1, 1264)), encoding="utf-8")
        result = self._read(qwen_cli, monkeypatch, p, offset=1, limit=1000)
        assert "lines 1–400 of 1263" in result

    def test_offset_beyond_eof(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "small.txt"
        p.write_text("one\ntwo\n", encoding="utf-8")
        result = self._read(qwen_cli, monkeypatch, p, offset=99)
        assert "beyond the end" in result

    def test_long_line_file_char_guard(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "minified.js"
        p.write_text("\n".join("x" * 5000 for _ in range(20)), encoding="utf-8")
        result = self._read(qwen_cli, monkeypatch, p)
        # window shrinks to stay under the char cap, header range stays truthful
        assert len(result) < qwen_cli._MAX_READ_CHARS + 500
        assert "of 20" in result

    def test_smart_cap_exempts_read_file(self, qwen_cli):
        huge = "z" * 100_000
        assert qwen_cli._smart_cap(None, huge, "read_file") == huge
        capped = qwen_cli._smart_cap(None, huge, "run_command")
        assert len(capped) < 100_000 and "truncated" in capped


# ---------------------------------------------------------------------------
# #3 turn ledger — tool activity recorded into stored history
# ---------------------------------------------------------------------------


class TestTurnLedger:
    def test_read_entry_carries_range(self, qwen_cli):
        result = "C:/x/big.txt  (lines 1–400 of 1263)\n\ncontent"
        entry = qwen_cli._ledger_entry("read_file", {"path": "C:/x/big.txt"}, result)
        assert entry == "read big.txt (lines 1–400 of 1263)"

    def test_edit_entry_uses_result_summary(self, qwen_cli):
        result = "[edited: C:/x/app.py  (1 replacement(s), 6 lines changed)]"
        entry = qwen_cli._ledger_entry("edit_file", {"path": "C:/x/app.py"}, result)
        assert entry.startswith("[edited:") and "6 lines changed" in entry

    def test_command_entry_notes_failure(self, qwen_cli):
        ok = qwen_cli._ledger_entry("run_command", {"command": "git status"}, "clean")
        bad = qwen_cli._ledger_entry("run_command", {"command": "git push"}, "[error: rejected]")
        assert ok == "run_command 'git status'"
        assert "[error: rejected]" in bad

    def test_uninteresting_tools_skipped(self, qwen_cli):
        assert qwen_cli._ledger_entry("web_search", {"query": "x"}, "results") is None
        assert qwen_cli._ledger_entry("update_plan", {"steps": []}, "[plan updated]") is None

    def test_format_caps_length(self, qwen_cli, monkeypatch):
        monkeypatch.setattr(qwen_cli, "_turn_ledger", [f"read f{i}.py (10 lines)" for i in range(30)])
        tag = qwen_cli._format_turn_ledger()
        assert tag.startswith("\n\n[turn actions: ")
        assert "more..." in tag and len(tag) < 1000
        monkeypatch.setattr(qwen_cli, "_turn_ledger", [])
        assert qwen_cli._format_turn_ledger() == ""

    def test_trim_preserves_ledger_turns_condensed(self, qwen_cli, monkeypatch):
        monkeypatch.setattr(
            qwen_cli, "stream_once", lambda *a, **k: ("Summary. CURRENT TASK: continue.", [], {})
        )
        monkeypatch.setattr(qwen_cli, "_current_plan", [])
        history = []
        for i in range(9):
            history.append({"role": "user", "content": f"chat {i}"})
            history.append({"role": "assistant", "content": f"reply {i}"})
        # one work turn with a ledger tag and long prose, early enough to be trimmed
        history.insert(2, {"role": "user", "content": "fix the countdown"})
        history.insert(
            3,
            {
                "role": "assistant",
                "content": "p" * 800 + "\n\n[turn actions: read game.html (lines 1–400 of 1263); [edited: game.html (1 replacement(s), 6 lines changed)]]",
            },
        )
        new_history = qwen_cli.cmd_trim(history, client=None)
        joined = json.dumps(new_history)
        assert "turn actions" in joined  # ledger survived
        assert "p" * 800 not in joined  # prose was condensed
        preserved = [m for m in new_history if "[turn actions:" in (m.get("content") or "")]
        assert preserved and len(preserved[0]["content"]) < 1300


# ---------------------------------------------------------------------------
# #5 auto-continue when the tool-round cap ends a turn mid-plan
# ---------------------------------------------------------------------------


def _repl_patches(stack, qwen_cli, run_turn):
    stack.enter_context(patch("qwen_cli.main.AUTO_SAVE_INTERVAL", 5, create=True))
    stack.enter_context(patch.object(qwen_cli, "_maybe_autocompact", side_effect=lambda h, bs, c: h))
    stack.enter_context(patch.object(qwen_cli, "build_system_prompt", return_value="sys"))
    stack.enter_context(patch.object(qwen_cli, "run_turn", side_effect=run_turn))
    for fn in (
        "_print_turn_footer",
        "_confidence_warning",
        "_silent_autosave",
        "_auto_extract_memory",
        "_intel_process_queue",
        "_intel_extract_topics",
        "_generate_session_title",
    ):
        stack.enter_context(patch.object(qwen_cli, fn))
    stack.enter_context(patch.object(qwen_cli, "_extract_runnable_code", return_value=None))
    stack.enter_context(patch.object(qwen_cli, "_looks_like_plan", return_value=False))
    stack.enter_context(patch.object(qwen_cli, "_main_llm_busy_lock", MagicMock()))


class TestAutoContinue:
    def _run(self, qwen_cli, run_turn, plan):
        from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

        ctx = _ReplContext([], "sys", "client")
        with ExitStack() as stack:
            _repl_patches(stack, qwen_cli, run_turn)
            stack.enter_context(patch.object(qwen_cli, "_current_plan", plan))
            qwen_cli._turn_hit_round_cap = False
            _run_turn_and_handle_reply(ctx, "improve my game")
        qwen_cli._turn_hit_round_cap = False
        return ctx

    def test_continues_after_round_cap_with_unfinished_plan(self, qwen_cli):
        calls = []

        def fake_run_turn(client, messages, allow_tools=True, presearch=True):
            calls.append(messages[-1]["content"])
            qwen_cli._turn_hit_round_cap = len(calls) == 1  # cap only on the first turn
            return "ran out of budget" if len(calls) == 1 else "plan finished"

        plan = [{"text": "fix countdown", "status": "in_progress"}]
        ctx = self._run(qwen_cli, fake_run_turn, plan)
        assert len(calls) == 2
        assert "Continue working on the current plan" in calls[1]
        assert len(ctx.history) == 4  # original pair + continuation pair
        assert ctx.history[3]["content"].startswith("plan finished")

    def test_no_continue_when_plan_complete(self, qwen_cli):
        calls = []

        def fake_run_turn(client, messages, allow_tools=True, presearch=True):
            calls.append(1)
            qwen_cli._turn_hit_round_cap = True
            return "done"

        plan = [{"text": "fix countdown", "status": "completed"}]
        ctx = self._run(qwen_cli, fake_run_turn, plan)
        assert len(calls) == 1
        assert len(ctx.history) == 2

    def test_continuations_are_bounded(self, qwen_cli):
        calls = []

        def fake_run_turn(client, messages, allow_tools=True, presearch=True):
            calls.append(1)
            qwen_cli._turn_hit_round_cap = True  # never finishes
            return "still going"

        plan = [{"text": "endless step", "status": "pending"}]
        ctx = self._run(qwen_cli, fake_run_turn, plan)
        assert len(calls) == 4  # initial + 3 auto-continues, then stop
        assert len(ctx.history) == 8

    def test_cancelled_continuation_stops(self, qwen_cli):
        calls = []

        def fake_run_turn(client, messages, allow_tools=True, presearch=True):
            calls.append(1)
            qwen_cli._turn_hit_round_cap = True
            return "budget hit" if len(calls) == 1 else ""  # Ctrl+C on continuation

        plan = [{"text": "step", "status": "pending"}]
        ctx = self._run(qwen_cli, fake_run_turn, plan)
        assert len(calls) == 2
        assert len(ctx.history) == 2  # cancelled continuation stored nothing


# ---------------------------------------------------------------------------
# #6 edit_file tolerant matching — CRLF and trailing-whitespace recovery
# ---------------------------------------------------------------------------


class TestEditTolerantMatch:
    def _edit(self, qwen_cli, monkeypatch, path, old, new, **kw):
        monkeypatch.setattr(qwen_cli, "_confirm_action", lambda _p: True)
        monkeypatch.setattr(qwen_cli, "_backup_file", lambda _p: None)
        monkeypatch.setattr(qwen_cli, "_lsp_post_edit_report", lambda _p: None)
        monkeypatch.setattr(qwen_cli.console, "print", lambda *a, **k: None)
        monkeypatch.setattr(qwen_cli, "_session_changes", {})
        return qwen_cli.do_edit_file(str(path), old, new, **kw)

    def test_crlf_file_lf_old_string_keeps_crlf(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "crlf.py"
        p.write_bytes(b"def a():\r\n    return 1\r\n\r\ndef b():\r\n    return 2\r\n")
        result = self._edit(qwen_cli, monkeypatch, p, "def a():\n    return 1", "def a():\n    return 10")
        assert result.startswith("[edited:")
        raw = p.read_bytes()
        assert b"return 10\r\n" in raw  # edit applied, CRLF style preserved
        assert b"\r\r" not in raw

    def test_lf_file_stays_lf(self, qwen_cli, tmp_path, monkeypatch):
        # regression: write_text used to convert every \n to os.linesep, so any
        # edit on Windows silently rewrote LF files as CRLF
        p = tmp_path / "lf.py"
        p.write_bytes(b"alpha = 1\nbeta = 2\ngamma = 3\n")
        result = self._edit(qwen_cli, monkeypatch, p, "beta = 2", "beta = 20")
        assert result.startswith("[edited:")
        raw = p.read_bytes()
        assert b"\r" not in raw
        assert b"beta = 20\n" in raw

    def test_trailing_whitespace_in_file(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "ws.py"
        p.write_text("x = 1   \ny = 2\nz = 3\n", encoding="utf-8")  # trailing spaces on line 1
        result = self._edit(qwen_cli, monkeypatch, p, "x = 1\ny = 2", "x = 9\ny = 2")
        assert result.startswith("[edited:") and "trailing whitespace" in result
        assert "x = 9" in p.read_text(encoding="utf-8")

    def test_ambiguous_window_still_errors(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "dup.py"
        p.write_text("a = 1  \nb = 2\n\na = 1\nb = 2\n", encoding="utf-8")
        result = self._edit(qwen_cli, monkeypatch, p, "a = 1\nb = 2 ", "a = 5\nb = 2")
        assert "not found" in result or "occurs" in result
        assert "a = 5" not in p.read_text(encoding="utf-8")

    def test_exact_match_has_no_note(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "exact.py"
        p.write_text("value = 42\n", encoding="utf-8")
        result = self._edit(qwen_cli, monkeypatch, p, "value = 42", "value = 43")
        assert result.startswith("[edited:")
        assert "converted" not in result and "ignoring" not in result

    def test_genuinely_absent_still_errors(self, qwen_cli, tmp_path, monkeypatch):
        p = tmp_path / "nope.py"
        p.write_text("real content\n", encoding="utf-8")
        result = self._edit(qwen_cli, monkeypatch, p, "imaginary content", "whatever")
        assert "not found" in result


# ---------------------------------------------------------------------------
# #7 diff preview before the Apply edit? prompt
# ---------------------------------------------------------------------------


class TestEditDiffPreview:
    def _printed_syntax(self, qwen_cli, tmp_path, monkeypatch, auto: bool, answer: bool):
        from rich.syntax import Syntax

        p = tmp_path / "app.py"
        p.write_text("a = 1\nb = 2\n", encoding="utf-8")
        printed = []
        order = []
        monkeypatch.setattr(qwen_cli.console, "print", lambda *a, **k: (printed.append(a[0] if a else None), order.append("print")))
        monkeypatch.setattr(qwen_cli, "_confirm_action", lambda _p: (order.append("confirm"), answer)[1])
        monkeypatch.setattr(qwen_cli, "_backup_file", lambda _p: None)
        monkeypatch.setattr(qwen_cli, "_lsp_post_edit_report", lambda _p: None)
        monkeypatch.setattr(qwen_cli, "_session_changes", {})
        monkeypatch.setattr(qwen_cli, "_auto_approve", auto)
        result = qwen_cli.do_edit_file(str(p), "a = 1", "a = 100")
        syntaxes = [x for x in printed if isinstance(x, Syntax)]
        return result, syntaxes, order, p

    def test_diff_shown_before_confirm(self, qwen_cli, tmp_path, monkeypatch):
        result, syntaxes, order, p = self._printed_syntax(qwen_cli, tmp_path, monkeypatch, auto=False, answer=True)
        assert result.startswith("[edited:")
        assert syntaxes, "no diff preview printed"
        assert "+a = 100" in syntaxes[0].code and "-a = 1" in syntaxes[0].code
        assert order.index("print") < order.index("confirm")  # preview BEFORE the y/N prompt

    def test_declined_edit_leaves_file_untouched(self, qwen_cli, tmp_path, monkeypatch):
        result, syntaxes, _order, p = self._printed_syntax(qwen_cli, tmp_path, monkeypatch, auto=False, answer=False)
        assert "cancelled" in result
        assert syntaxes  # user saw what they were declining
        assert p.read_text(encoding="utf-8") == "a = 1\nb = 2\n"

    def test_auto_mode_prints_no_preview(self, qwen_cli, tmp_path, monkeypatch):
        result, syntaxes, _order, p = self._printed_syntax(qwen_cli, tmp_path, monkeypatch, auto=True, answer=True)
        assert result.startswith("[edited:")
        assert not syntaxes  # /auto stays quiet
        assert "a = 100" in p.read_text(encoding="utf-8")
