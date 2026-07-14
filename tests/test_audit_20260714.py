"""Regression tests for the 2026-07-14 audit.

Four latent bugs, each verified live before the fix:

1. write_file / patch_file / _backup_file / undo / rollback wrote user files via
   Path.write_text(), which translates "\\n" to os.linesep — on Windows that
   silently rewrote every LF file as CRLF. do_edit_file already round-tripped
   endings correctly; nothing else did.
2. BASE_SYSTEM promises the model that `git push --force` requires confirmation,
   but _DANGEROUS_CMD_RE had no git-push pattern, so it ran unprompted.
3. _repl_loop held history/base_system as locals while /load and /branch restore
   rebind ctx.history — exiting with Ctrl+D after a /load saved the pre-load
   history and discarded the loaded session.
4. Ctrl+C during tool execution printed "returning to prompt" but fell through
   to another LLM call instead of returning.
"""

from unittest.mock import patch

import pytest

from qwen_cli.core.repl import _repl_loop, _ReplContext
from qwen_cli.tools.files import do_edit_file, do_patch_file, do_write_file

LF_BODY = b"line one\nline two\nline three\n"
CRLF_BODY = b"line one\r\nline two\r\nline three\r\n"


@pytest.fixture
def lf_file(tmp_path):
    p = tmp_path / "sample.py"
    p.write_bytes(LF_BODY)
    return p


@pytest.fixture
def crlf_file(tmp_path):
    p = tmp_path / "sample_crlf.py"
    p.write_bytes(CRLF_BODY)
    return p


@pytest.fixture
def auto_approve(qwen_cli):
    with patch.object(qwen_cli, "_auto_approve", True):
        yield


@pytest.fixture
def ctx(qwen_cli):
    return _ReplContext([], "sys", object())


# ---------------------------------------------------------------------------
# 1. Line endings survive every write path
# ---------------------------------------------------------------------------


def test_edit_file_keeps_lf(qwen_cli, lf_file, auto_approve):
    do_edit_file(str(lf_file), "line two", "line 2")
    assert b"\r\n" not in lf_file.read_bytes()


def test_write_file_keeps_lf(qwen_cli, lf_file, auto_approve):
    do_write_file(str(lf_file), "line one\nline two\nline 3\n")
    assert b"\r\n" not in lf_file.read_bytes()


def test_patch_file_keeps_lf(qwen_cli, lf_file, auto_approve):
    diff = "--- a/sample.py\n+++ b/sample.py\n@@ -1,3 +1,3 @@\n line one\n-line two\n+line 2\n line three\n"
    do_patch_file(str(lf_file), diff)
    body = lf_file.read_bytes()
    assert b"\r\n" not in body
    assert b"line 2\n" in body


def test_write_file_preserves_existing_crlf(qwen_cli, crlf_file, auto_approve):
    """A CRLF file stays CRLF — the fix must not convert in the other direction."""
    do_write_file(str(crlf_file), "line one\nline two\nline 3\n")
    body = crlf_file.read_bytes()
    assert b"\r\n" in body
    assert b"line 3\r\n" in body


def test_patch_file_preserves_existing_crlf(qwen_cli, crlf_file, auto_approve):
    diff = (
        "--- a/sample_crlf.py\n+++ b/sample_crlf.py\n@@ -1,3 +1,3 @@\n"
        " line one\n-line two\n+line 2\n line three\n"
    )
    do_patch_file(str(crlf_file), diff)
    body = crlf_file.read_bytes()
    assert b"line 2\r\n" in body
    assert b"\n\n" not in body.replace(b"\r\n", b"")  # no bare LF left behind


def test_new_file_is_written_verbatim(qwen_cli, tmp_path, auto_approve):
    p = tmp_path / "brand_new.py"
    do_write_file(str(p), "a\nb\n")
    assert p.read_bytes() == b"a\nb\n"


def test_undo_restores_lf_file_byte_for_byte(qwen_cli, lf_file, auto_approve, ctx):
    from qwen_cli.core.commands import _cmd_undo

    do_edit_file(str(lf_file), "line two", "line 2")
    assert lf_file.read_bytes() != LF_BODY

    _cmd_undo(ctx, "")
    assert lf_file.read_bytes() == LF_BODY


def test_rollback_restores_lf_file_byte_for_byte(qwen_cli, lf_file, auto_approve):
    do_write_file(str(lf_file), "totally\ndifferent\n")
    assert lf_file.read_bytes() != LF_BODY

    with patch.object(qwen_cli.console, "input", return_value="y"):
        qwen_cli.cmd_rollback()
    assert lf_file.read_bytes() == LF_BODY


def test_backup_copy_is_byte_faithful(qwen_cli, lf_file, auto_approve):
    """The .bak is the only recovery path after a crash — it must not be rewritten."""
    do_edit_file(str(lf_file), "line two", "line 2")
    backup = qwen_cli._backup_stack[-1]["backup"]
    assert backup.read_bytes() == LF_BODY


# ---------------------------------------------------------------------------
# 2. The dangerous-command gate matches what BASE_SYSTEM promises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        "git push --force",
        "git push -f origin main",
        "git push --force-with-lease",
        "git push --force origin feature",
    ],
)
def test_force_push_is_gated(qwen_cli, command):
    assert qwen_cli._is_dangerous(command), f"{command!r} must prompt — BASE_SYSTEM says it does"


@pytest.mark.parametrize(
    "command",
    ["git push", "git push origin main", "git push --tags", "git fetch origin"],
)
def test_ordinary_push_is_not_gated(qwen_cli, command):
    assert not qwen_cli._is_dangerous(command), f"{command!r} is not destructive"


def test_base_system_promise_matches_the_gate(qwen_cli):
    """The prompt and the regex must not drift apart again."""
    assert "git push --force" in qwen_cli.BASE_SYSTEM
    assert qwen_cli._is_dangerous("git push --force")


# ---------------------------------------------------------------------------
# 3. The REPL follows ctx.history across a /load or /branch restore
# ---------------------------------------------------------------------------


def test_exit_saves_the_loaded_session_not_the_replaced_one(qwen_cli):
    """After /load rebinds ctx.history, Ctrl+D must save the LOADED history."""
    original = [{"role": "user", "content": "OLD"}]
    ctx = _ReplContext(original, "sys", object())
    saved: dict = {}

    def fake_read_input():
        # First call: rebind ctx.history the way _cmd_load / _cmd_branch do.
        ctx.history = [{"role": "user", "content": "LOADED"}]
        raise EOFError

    with (
        patch("qwen_cli.core.repl.read_input", side_effect=fake_read_input),
        patch("qwen_cli.core.repl._cleanup_watch"),
        patch.object(qwen_cli, "save_session", side_effect=lambda h, s, n=None: saved.update(history=h, system=s)),
        patch.object(qwen_cli, "_save_exit_handoff"),
        patch.object(qwen_cli, "record_session_changes_memory"),
    ):
        _repl_loop(ctx, original, "sys")

    assert saved["history"][0]["content"] == "LOADED"


def test_exit_saves_the_updated_base_system(qwen_cli):
    """/system rebinds ctx.base_system; the exit save must not use the stale one."""
    history = [{"role": "user", "content": "hi"}]
    ctx = _ReplContext(history, "old system", object())
    saved: dict = {}

    def fake_read_input():
        ctx.base_system = "new system"
        raise EOFError

    with (
        patch("qwen_cli.core.repl.read_input", side_effect=fake_read_input),
        patch("qwen_cli.core.repl._cleanup_watch"),
        patch.object(qwen_cli, "save_session", side_effect=lambda h, s, n=None: saved.update(system=s)),
        patch.object(qwen_cli, "_save_exit_handoff"),
        patch.object(qwen_cli, "record_session_changes_memory"),
    ):
        _repl_loop(ctx, history, "old system")

    assert saved["system"] == "new system"


def test_watch_injection_lands_in_the_live_history(qwen_cli, tmp_path):
    """A watched-file injection after a /load must reach the list turns read."""
    watched = tmp_path / "w.py"
    watched.write_text("print(1)\n", encoding="utf-8")

    stale = [{"role": "user", "content": "OLD"}]
    ctx = _ReplContext(stale, "sys", object())
    loaded = [{"role": "user", "content": "LOADED"}]

    steps = iter(["rebind", "eof"])

    def fake_read_input():
        if next(steps) == "rebind":
            ctx.history = loaded  # what /load and /branch restore do
            qwen_cli._watch_pending.append(str(watched))
            return ""  # empty input → loop continues to the watch block
        raise EOFError

    with (
        patch("qwen_cli.core.repl.read_input", side_effect=fake_read_input),
        patch("qwen_cli.core.repl._cleanup_watch"),
        patch.object(qwen_cli, "save_session"),
        patch.object(qwen_cli, "_save_exit_handoff"),
        patch.object(qwen_cli, "record_session_changes_memory"),
    ):
        _repl_loop(ctx, stale, "sys")

    assert any("[File updated: w.py]" in (m.get("content") or "") for m in loaded)
    assert not any("[File updated" in (m.get("content") or "") for m in stale)


# ---------------------------------------------------------------------------
# 4. Ctrl+C during tool execution returns to the prompt
# ---------------------------------------------------------------------------


def test_ctrl_c_during_tools_returns_immediately(qwen_cli):
    """Cancelling tools must stop the turn, not fire another LLM call."""
    tool_call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "read_file", "arguments": '{"path": "x.py"}'},
    }
    calls: list[int] = []

    def fake_stream_once(client, msgs, use_tools, update_fn=None):
        calls.append(1)
        return "", [dict(tool_call)], {"prompt": 1, "completion": 1}

    with (
        patch.object(qwen_cli, "stream_once", side_effect=fake_stream_once),
        patch("qwen_cli.core.turn._execute_batches", side_effect=KeyboardInterrupt),
    ):
        reply = qwen_cli.run_turn(object(), [{"role": "user", "content": "read x"}], allow_tools=True)

    assert reply == ""  # cancelled, same contract as Ctrl+C during streaming
    assert len(calls) == 1, "a second LLM call means the cancel did not return to the prompt"
