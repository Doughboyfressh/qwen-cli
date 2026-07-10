"""Tests for slash command handlers in qwen_cli/core/commands.py."""

import io
import os
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rich.console import Console


def _mock_ctx():
    """Create a minimal mock _ReplContext for testing commands."""
    ctx = SimpleNamespace()
    ctx.history = []
    ctx.base_system = ""
    ctx.client = "mock-client"
    return ctx


def _run_cmd(cmd_fn, qwen_cli, ctx, arg):
    """Run a command and capture console output by replacing console.print."""
    buf = io.StringIO()
    real_console = qwen_cli.console
    qwen_cli.console = Console(file=buf, force_terminal=True, legacy_windows=False, no_color=True)
    try:
        cmd_fn(ctx, arg)
    except StopIteration:
        pass
    val = buf.getvalue()
    qwen_cli.console = real_console
    return val


def _run_and_capture_print(cmd_fn, qwen_cli, ctx, arg):
    """Run a command and capture the raw strings passed to console.print."""
    calls = []
    original_print = qwen_cli.console.print
    qwen_cli.console.print = lambda *args, **kwargs: calls.append(str(args[0]) if args else "")
    try:
        cmd_fn(ctx, arg)
    except StopIteration:
        pass
    qwen_cli.console.print = original_print
    return "\n".join(calls)


def _run_and_render(cmd_fn, qwen_cli, ctx, arg):
    """Run a command and capture rendered console output (handles Markdown objects)."""
    buf = io.StringIO()
    real_console = qwen_cli.console
    qwen_cli.console = Console(file=buf, force_terminal=False, no_color=True, width=200, legacy_windows=False)
    try:
        cmd_fn(ctx, arg)
    except StopIteration:
        pass
    val = buf.getvalue()
    qwen_cli.console = real_console
    return val


def _resolve_cmd(name):
    from qwen_cli.core.commands import _REPL_COMMANDS
    return _REPL_COMMANDS[name]


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')


def _strip_ansi(text):
    return _ANSI_RE.sub('', text)


# --- _cmd_exit ---


def test_cmd_exit_raises_stop_iteration(qwen_cli):
    from qwen_cli.core.commands import _cmd_exit
    ctx = _mock_ctx()
    import pytest
    with pytest.raises(StopIteration):
        _cmd_exit(ctx, "")


# --- _cmd_help ---


def test_cmd_help_shows_full_text(qwen_cli):
    ctx = _mock_ctx()
    output = _run_and_render(_resolve_cmd("/help"), qwen_cli, ctx, "")
    assert "/help" in output or "help" in output


def test_cmd_help_search_matches(qwen_cli):
    ctx = _mock_ctx()
    output = _run_and_render(_resolve_cmd("/help"), qwen_cli, ctx, "clear")
    assert "/clear" in output or "clear" in output


def test_cmd_help_search_no_match(qwen_cli):
    ctx = _mock_ctx()
    output = _run_and_capture_print(_resolve_cmd("/help"), qwen_cli, ctx, "zzznotacommand")
    assert "no help entries" in output


# --- _cmd_clear ---


def test_cmd_clear_history(qwen_cli):
    from qwen_cli.core.commands import _cmd_clear
    ctx = _mock_ctx()
    ctx.history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    output = _run_and_capture_print(_cmd_clear, qwen_cli, ctx, "")
    assert ctx.history == []
    assert "history cleared" in output


def test_cmd_clear_drop_last_n(qwen_cli):
    from qwen_cli.core.commands import _cmd_clear
    ctx = _mock_ctx()
    ctx.history = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    output = _run_and_capture_print(_cmd_clear, qwen_cli, ctx, "1")
    assert len(ctx.history) == 2
    assert "dropped 1 turn" in output


# --- _cmd_retry ---


def test_cmd_retry_nothing_to_retry(qwen_cli):
    from qwen_cli.core.commands import _cmd_retry
    ctx = _mock_ctx()
    qwen_cli._last_user_input = ""
    output = _run_and_capture_print(_cmd_retry, qwen_cli, ctx, "")
    assert "nothing to retry" in output


def test_cmd_retry_delegates(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_retry
    ctx = _mock_ctx()
    qwen_cli._last_user_input = "hello"
    ctx.history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "world"},
    ]
    called = []
    monkeypatch.setattr(qwen_cli, "_run_turn_and_handle_reply", lambda _ctx, inp: called.append(inp))
    output = _run_and_capture_print(_cmd_retry, qwen_cli, ctx, "")
    assert called == ["hello"]
    assert ctx.history == []
    assert "retrying" in output.lower()


# --- _cmd_system ---


def test_cmd_system_displays_current(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_system
    ctx = _mock_ctx()
    ctx.base_system = "You are a test bot"
    output = _run_and_capture_print(_cmd_system, qwen_cli, ctx, "")
    assert "You are a test bot" in output


def test_cmd_system_sets_new_prompt(qwen_cli):
    from qwen_cli.core.commands import _cmd_system
    ctx = _mock_ctx()
    ctx.history = [{"role": "user", "content": "hello"}]
    output = _run_and_capture_print(_cmd_system, qwen_cli, ctx, "New prompt")
    assert ctx.base_system == "New prompt"
    assert ctx.history == []
    assert "system prompt updated" in output


# --- _cmd_export ---


def test_cmd_export_calls_export_session(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_export
    ctx = _mock_ctx()
    called = []
    monkeypatch.setattr(qwen_cli, "export_session", lambda h, a: called.append((list(h), a)))
    _run_and_capture_print(_cmd_export, qwen_cli, ctx, "myfile")
    assert called == [([], "myfile")]


# --- _cmd_copy ---


def test_cmd_copy_calls_cmd_copy(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_copy
    ctx = _mock_ctx()
    called = []
    monkeypatch.setattr(qwen_cli, "cmd_copy", lambda h: called.append(h))
    _run_and_capture_print(_cmd_copy, qwen_cli, ctx, "")
    assert called == [[]]


# --- _cmd_paste ---


def test_cmd_paste_calls_cmd_paste(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_paste
    ctx = _mock_ctx()
    called = []
    monkeypatch.setattr(qwen_cli, "cmd_paste", lambda h: called.append(h))
    _run_and_capture_print(_cmd_paste, qwen_cli, ctx, "")
    assert called == [[]]


# --- _cmd_remember ---


def test_cmd_remember_no_arg(qwen_cli):
    from qwen_cli.core.commands import _cmd_remember
    ctx = _mock_ctx()
    output = _run_and_capture_print(_cmd_remember, qwen_cli, ctx, "")
    assert "usage:" in output


def test_cmd_remember_adds_fact(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_remember
    ctx = _mock_ctx()
    monkeypatch.setattr(qwen_cli, "load_memory", lambda: "existing memory")
    saved = []
    monkeypatch.setattr(qwen_cli, "save_memory", lambda t: saved.append(t))
    output = _run_and_capture_print(_cmd_remember, qwen_cli, ctx, "my fact")
    assert saved == ["existing memory\n- my fact"]
    assert "remembered" in output


# --- _cmd_memory ---


def test_cmd_memory_empty(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_memory
    ctx = _mock_ctx()
    monkeypatch.setattr(qwen_cli, "load_memory", lambda: "")
    output = _run_and_capture_print(_cmd_memory, qwen_cli, ctx, "")
    assert "memory is empty" in output


def test_cmd_memory_shows_content(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_memory
    ctx = _mock_ctx()
    monkeypatch.setattr(qwen_cli, "load_memory", lambda: "Some memory content")
    output = _run_and_render(_cmd_memory, qwen_cli, ctx, "")
    assert "Some memory content" in output


# --- _cmd_mode ---


def test_cmd_mode_no_arg_default(qwen_cli):
    from qwen_cli.core.commands import _cmd_mode
    ctx = _mock_ctx()
    qwen_cli._current_mode = ""
    output = _run_and_capture_print(_cmd_mode, qwen_cli, ctx, "")
    assert "mode:" in output and "default" in output.lower()


def test_cmd_mode_set_known(qwen_cli):
    from qwen_cli.core.commands import _cmd_mode
    ctx = _mock_ctx()
    qwen_cli._current_mode = ""
    output = _run_and_capture_print(_cmd_mode, qwen_cli, ctx, "code")
    assert qwen_cli._current_mode == "code"
    assert "mode: code" in output.lower()


def test_cmd_mode_off(qwen_cli):
    from qwen_cli.core.commands import _cmd_mode
    ctx = _mock_ctx()
    qwen_cli._current_mode = "code"
    output = _run_and_capture_print(_cmd_mode, qwen_cli, ctx, "off")
    assert qwen_cli._current_mode == ""
    assert "mode cleared" in output


def test_cmd_mode_unknown(qwen_cli):
    from qwen_cli.core.commands import _cmd_mode
    ctx = _mock_ctx()
    output = _run_and_capture_print(_cmd_mode, qwen_cli, ctx, "bogus")
    assert "unknown mode" in output


# --- _cmd_compact ---


def test_cmd_compact_calls_cmd_trim(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_compact
    ctx = _mock_ctx()
    ctx.history = [{"role": "user", "content": "hi"}]
    called = []
    monkeypatch.setattr(qwen_cli, "cmd_trim", lambda h, c: called.append((list(h), c)) or [])
    _run_and_capture_print(_cmd_compact, qwen_cli, ctx, "")
    assert called == [([{"role": "user", "content": "hi"}], "mock-client")]


# --- _cmd_stats ---


def test_cmd_stats_calls_cmd_stats(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_stats
    ctx = _mock_ctx()
    called = []
    monkeypatch.setattr(qwen_cli, "cmd_stats", lambda h: called.append(h))
    _run_and_capture_print(_cmd_stats, qwen_cli, ctx, "")
    assert called == [[]]


# --- _cmd_cd ---


def test_cmd_cd_no_arg_shows_cwd(qwen_cli):
    from qwen_cli.core.commands import _cmd_cd
    ctx = _mock_ctx()
    output = _run_and_capture_print(_cmd_cd, qwen_cli, ctx, "")
    assert "cwd:" in output


def test_cmd_cd_valid_dir(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_cd
    ctx = _mock_ctx()
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(qwen_cli, "_resolve", lambda p: Path(tmpdir))
        monkeypatch.setattr(qwen_cli, "_invalidate_git_cache", lambda: None)
        chdir_called = []
        monkeypatch.setattr(os, "chdir", lambda p: chdir_called.append(str(p)))
        output = _run_and_capture_print(_cmd_cd, qwen_cli, ctx, tmpdir)
        assert chdir_called == [tmpdir]
        assert "cwd" in output.lower()


def test_cmd_cd_not_a_directory(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_cd
    ctx = _mock_ctx()
    p = Path("/nonexistent_dir_xyz")
    monkeypatch.setattr(qwen_cli, "_resolve", lambda arg: p)
    output = _run_and_capture_print(_cmd_cd, qwen_cli, ctx, "/nonexistent_dir_xyz")
    assert "not a directory" in output


# --- _cmd_watch ---


def test_cmd_watch_list_empty(qwen_cli):
    from qwen_cli.core.commands import _cmd_watch
    ctx = _mock_ctx()
    qwen_cli._watched_files = {}
    qwen_cli._watch_stop = SimpleNamespace()
    qwen_cli._watch_stop.is_set = lambda: False
    qwen_cli._watch_stop.set = lambda: None
    qwen_cli._watch_stop.clear = lambda: None
    qwen_cli._watch_stop.wait = lambda **kw: None
    qwen_cli._watch_thread = None
    qwen_cli._watch_pending = []
    output = _run_and_capture_print(_cmd_watch, qwen_cli, ctx, "list")
    assert "no files watched" in output


def test_cmd_watch_off(qwen_cli):
    from qwen_cli.core.commands import _cmd_watch
    ctx = _mock_ctx()
    stop_calls = []
    stop = SimpleNamespace()
    stop.is_set = lambda: False
    stop.set = lambda: stop_calls.append("set")
    stop.clear = lambda: stop_calls.append("clear")
    stop.wait = lambda **kw: None
    thread = SimpleNamespace()
    thread.is_alive = lambda: False
    thread.join = lambda **kw: None
    qwen_cli._watched_files = {"somefile": 1.0}
    qwen_cli._watch_stop = stop
    qwen_cli._watch_thread = thread
    qwen_cli._watch_pending = ["somefile"]
    output = _run_and_capture_print(_cmd_watch, qwen_cli, ctx, "off")
    assert "watch stopped" in output
    assert qwen_cli._watched_files == {}
    assert qwen_cli._watch_pending == []
    assert qwen_cli._watch_thread is None


def test_cmd_watch_add_file(qwen_cli, monkeypatch, tmp_path):
    from qwen_cli.core.commands import _cmd_watch
    ctx = _mock_ctx()
    f = tmp_path / "test.txt"
    f.write_text("hello")
    monkeypatch.setattr(qwen_cli, "_resolve", lambda p: f)
    qwen_cli._watched_files = {}
    stop = SimpleNamespace()
    stop.is_set = lambda: False
    stop.set = lambda: None
    stop.clear = lambda: None
    stop.wait = lambda **kw: None
    qwen_cli._watch_stop = stop
    qwen_cli._watch_thread = None
    qwen_cli._watch_pending = []
    output = _run_and_capture_print(_cmd_watch, qwen_cli, ctx, str(f))
    assert "watching:" in output
    assert str(f) in qwen_cli._watched_files


def test_cmd_watch_add_file_not_found(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_watch
    ctx = _mock_ctx()
    missing = SimpleNamespace()
    missing.exists = lambda: False
    missing.__str__ = lambda s: "/missing.txt"
    monkeypatch.setattr(qwen_cli, "_resolve", lambda p: missing)
    qwen_cli._watched_files = {}
    stop = SimpleNamespace()
    stop.is_set = lambda: False
    stop.set = lambda: None
    stop.clear = lambda: None
    stop.wait = lambda **kw: None
    qwen_cli._watch_stop = stop
    qwen_cli._watch_thread = None
    qwen_cli._watch_pending = []
    output = _run_and_capture_print(_cmd_watch, qwen_cli, ctx, "/missing.txt")
    assert "file not found" in output


# --- _cmd_pin ---


def test_cmd_pin_list_empty(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_pin
    ctx = _mock_ctx()
    monkeypatch.setattr(qwen_cli, "load_pins", lambda: [])
    monkeypatch.setattr(qwen_cli, "save_pins", lambda p: None)
    output = _run_and_capture_print(_cmd_pin, qwen_cli, ctx, "list")
    assert "no pins" in output


def test_cmd_pin_add(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_pin
    ctx = _mock_ctx()
    pins = []
    monkeypatch.setattr(qwen_cli, "load_pins", lambda: pins)
    saved = []
    monkeypatch.setattr(qwen_cli, "save_pins", lambda p: saved.append(list(p)))
    output = _run_and_capture_print(_cmd_pin, qwen_cli, ctx, "my important note")
    assert "pinned:" in output
    assert saved == [["my important note"]]


def test_cmd_pin_remove(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_pin
    ctx = _mock_ctx()
    pins = ["pin1", "pin2"]
    monkeypatch.setattr(qwen_cli, "load_pins", lambda: pins)
    saved = []
    monkeypatch.setattr(qwen_cli, "save_pins", lambda p: saved.append(list(p)))
    output = _run_and_capture_print(_cmd_pin, qwen_cli, ctx, "remove 2")
    assert "removed pin:" in output
    assert saved == [["pin1"]]


def test_cmd_pin_remove_invalid(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_pin
    ctx = _mock_ctx()
    monkeypatch.setattr(qwen_cli, "load_pins", lambda: [])
    output = _run_and_capture_print(_cmd_pin, qwen_cli, ctx, "remove 99")
    assert "usage:" in output.lower()


# --- _cmd_branch ---


def test_cmd_branch_list_empty(qwen_cli):
    from qwen_cli.core.commands import _cmd_branch
    ctx = _mock_ctx()
    qwen_cli._branches = {}
    output = _run_and_capture_print(_cmd_branch, qwen_cli, ctx, "list")
    assert "no branches" in output


def test_cmd_branch_save(qwen_cli):
    from qwen_cli.core.commands import _cmd_branch
    ctx = _mock_ctx()
    ctx.history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    qwen_cli._branches = {}
    output = _run_and_capture_print(_cmd_branch, qwen_cli, ctx, "mybranch")
    assert "mybranch" in qwen_cli._branches
    assert "branch" in output and "saved" in output


def test_cmd_branch_restore(qwen_cli):
    from qwen_cli.core.commands import _cmd_branch
    ctx = _mock_ctx()
    qwen_cli._branches = {"saved": [{"role": "user", "content": "hey"}]}
    output = _run_and_capture_print(_cmd_branch, qwen_cli, ctx, "restore saved")
    assert ctx.history == qwen_cli._branches["saved"]
    assert "restored branch" in output


def test_cmd_branch_restore_unknown(qwen_cli):
    from qwen_cli.core.commands import _cmd_branch
    ctx = _mock_ctx()
    qwen_cli._branches = {"foo": []}
    output = _run_and_capture_print(_cmd_branch, qwen_cli, ctx, "restore nonexistent")
    assert "no branch:" in output.lower()


# --- _cmd_note ---


def test_cmd_note_no_arg(qwen_cli):
    from qwen_cli.core.commands import _cmd_note
    ctx = _mock_ctx()
    output = _run_and_capture_print(_cmd_note, qwen_cli, ctx, "")
    assert "usage:" in output


def test_cmd_note_adds_to_history(qwen_cli):
    from qwen_cli.core.commands import _cmd_note
    ctx = _mock_ctx()
    output = _run_and_capture_print(_cmd_note, qwen_cli, ctx, "my note")
    assert ctx.history == [{"role": "user", "content": "[Note] my note"}]
    assert "note added" in output


# --- _cmd_unknown ---


def test_cmd_unknown_shows_error(qwen_cli):
    from qwen_cli.core.commands import _cmd_unknown
    ctx = _mock_ctx()
    output = _run_and_capture_print(_cmd_unknown, qwen_cli, ctx, "/foobar")
    assert "unknown command" in output
    assert "/foobar" in output


# --- _cmd_forget ---


def test_cmd_forget_cancelled(monkeypatch, qwen_cli):
    from qwen_cli.core.commands import _cmd_forget
    ctx = _mock_ctx()
    monkeypatch.setattr(qwen_cli.console, "input", lambda _prompt="": "n")
    output = _run_and_capture_print(_cmd_forget, qwen_cli, ctx, "")
    assert "cancelled" in output


def test_cmd_forget_confirmed(monkeypatch, qwen_cli):
    from qwen_cli.core.commands import _cmd_forget
    ctx = _mock_ctx()
    monkeypatch.setattr(qwen_cli.console, "input", lambda _prompt="": "y")
    saved = []
    monkeypatch.setattr(qwen_cli, "save_memory", lambda t: saved.append(t))
    output = _run_and_capture_print(_cmd_forget, qwen_cli, ctx, "")
    assert saved == [""]
    assert "cleared" in output


# --- _cmd_history ---


def test_cmd_history_calls_cmd_history(qwen_cli, monkeypatch):
    from qwen_cli.core.commands import _cmd_history
    ctx = _mock_ctx()
    called = []
    monkeypatch.setattr(qwen_cli, "cmd_history", lambda a, h: called.append((a, h)))
    _run_and_capture_print(_cmd_history, qwen_cli, ctx, "5")
    assert called == [("5", [])]


# --- _cmd_model ---


def test_cmd_model_no_arg_shows_current(qwen_cli):
    from qwen_cli.core.commands import _cmd_model
    ctx = _mock_ctx()
    output = _run_and_capture_print(_cmd_model, qwen_cli, ctx, "")
    assert "model:" in output.lower()


def test_cmd_model_switch(qwen_cli):
    from qwen_cli.core.commands import _cmd_model
    ctx = _mock_ctx()
    output = _run_and_capture_print(_cmd_model, qwen_cli, ctx, "gpt-4")
    assert qwen_cli.MODEL == "gpt-4"
    assert "model switched" in output.lower()
