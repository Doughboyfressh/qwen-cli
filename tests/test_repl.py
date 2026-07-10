"""Tests for the REPL module — context, input, dispatch, watch, and loop setup."""

import io
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console


# ==============================================================================
# _ReplContext
# ==============================================================================


def test_repl_context_construction():
    from qwen_cli.core.repl import _ReplContext

    history = [{"role": "user", "content": "hello"}]
    ctx = _ReplContext(history, "system prompt", "fake_client")
    assert ctx.history is history
    assert ctx.base_system == "system prompt"
    assert ctx.client == "fake_client"


# ==============================================================================
# _close_loitering_event_loop
# ==============================================================================


def test_close_loitering_event_loop_no_running_loop():
    from qwen_cli.core.repl import _close_loitering_event_loop

    _close_loitering_event_loop()


# ==============================================================================
# read_input
# ==============================================================================


def test_read_input_with_pt_session(qwen_cli):
    from qwen_cli.core.repl import read_input

    qwen_cli._pt_session = MagicMock()
    qwen_cli._pt_session.prompt.return_value = "  hello world  "
    result = read_input()
    assert result == "hello world"
    qwen_cli._pt_session.prompt.assert_called()


def test_read_input_inline_no_pt_session(qwen_cli):
    from qwen_cli.core.repl import _read_input_inline

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, legacy_windows=False)

    with patch("qwen_cli.core.repl.console", console):
        with patch.object(console, "input", return_value="  hello  "):
            qwen_cli._pt_session = None
            result = _read_input_inline()
    assert result == "hello"


def test_read_input_inline_with_continuation(qwen_cli):
    from qwen_cli.core.repl import _read_input_inline

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, legacy_windows=False)
    inputs = iter(["line one\\", "line two"])

    with patch("qwen_cli.core.repl.console", console):
        with patch.object(console, "input", side_effect=lambda prompt: next(inputs)):
            qwen_cli._pt_session = None
            result = _read_input_inline()
    assert result == "line one\nline two"


# ==============================================================================
# _watch_worker
# ==============================================================================


def test_watch_worker_exits_when_event_set(tmp_path):
    """_watch_worker returns immediately when _watch_stop is already set."""
    from qwen_cli.core.repl import _watch_worker
    import qwen_cli.main as _main

    test_file = tmp_path / "watchme.txt"
    test_file.write_text("content", encoding="utf-8")

    mtimes = {str(test_file): 0.0}
    _main._watch_pending.clear()
    _main._watch_stop.clear()
    _main._watch_stop.set()

    # Call synchronously — it should return immediately without looping
    _watch_worker(mtimes)

    # The function should not have added anything to watch_pending
    assert _main._watch_pending == []


def test_watch_worker_detects_file_change(tmp_path):
    from qwen_cli.core.repl import _watch_worker
    import qwen_cli.main as _main

    test_file = tmp_path / "watchme.txt"
    test_file.write_text("v1", encoding="utf-8")

    # Use a low baseline mtime so any real file stat will differ
    mtimes = {str(test_file): 0.0}
    _main._watch_pending.clear()
    _main._watch_stop.clear()

    t = threading.Thread(target=_watch_worker, args=(mtimes,), daemon=True)
    t.start()

    time.sleep(0.5)

    test_file.write_text("v2", encoding="utf-8")

    time.sleep(1.0)

    _main._watch_stop.set()
    t.join(timeout=3)

    assert str(test_file) in _main._watch_pending


# ==============================================================================
# _run_turn_and_handle_reply
# ==============================================================================


def test_run_turn_and_handle_reply_basic(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

    ctx = _ReplContext([], "You are a helpful assistant.", "fake_client")
    user_input = "hello"

    with (
        patch("qwen_cli.main.AUTO_SAVE_INTERVAL", 5, create=True),
        patch.object(qwen_cli, "_maybe_autocompact", side_effect=lambda h, bs, c: h),
        patch.object(qwen_cli, "build_system_prompt", return_value="system prompt"),
        patch.object(qwen_cli, "run_turn", return_value="Hi there!"),
        patch.object(qwen_cli, "_print_turn_footer"),
        patch.object(qwen_cli, "_confidence_warning"),
        patch.object(qwen_cli, "_silent_autosave"),
        patch.object(qwen_cli, "_auto_extract_memory"),
        patch.object(qwen_cli, "_intel_process_queue"),
        patch.object(qwen_cli, "_intel_extract_topics"),
        patch.object(qwen_cli, "_generate_session_title"),
        patch.object(qwen_cli, "_extract_runnable_code", return_value=None),
        patch.object(qwen_cli, "_looks_like_plan", return_value=False),
        patch.object(qwen_cli, "_main_llm_busy_lock", MagicMock()),
    ):
        _run_turn_and_handle_reply(ctx, user_input)

    assert len(ctx.history) == 2
    assert ctx.history[0]["role"] == "user"
    assert ctx.history[0]["content"] == "hello"
    assert ctx.history[1]["role"] == "assistant"
    assert ctx.history[1]["content"] == "Hi there!"


def test_run_turn_and_handle_reply_empty_reply(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

    ctx = _ReplContext([], "system", "client")

    with (
        patch("qwen_cli.main.AUTO_SAVE_INTERVAL", 5, create=True),
        patch.object(qwen_cli, "_maybe_autocompact", side_effect=lambda h, bs, c: h),
        patch.object(qwen_cli, "build_system_prompt", return_value="sys"),
        patch.object(qwen_cli, "run_turn", return_value=None),
        patch.object(qwen_cli, "_main_llm_busy_lock", MagicMock()),
    ):
        _run_turn_and_handle_reply(ctx, "hi")

    assert ctx.history == []


def test_run_turn_and_handle_reply_with_plan(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, legacy_windows=False)
    ctx = _ReplContext([{"role": "user", "content": "prev"}], "system", "client")

    with (
        patch("qwen_cli.main.AUTO_SAVE_INTERVAL", 5, create=True),
        patch.object(qwen_cli, "_maybe_autocompact", side_effect=lambda h, bs, c: h),
        patch.object(qwen_cli, "build_system_prompt", return_value="sys"),
        patch.object(qwen_cli, "run_turn", return_value="Here is my plan..."),
        patch.object(qwen_cli, "_print_turn_footer"),
        patch.object(qwen_cli, "_confidence_warning"),
        patch.object(qwen_cli, "_silent_autosave"),
        patch.object(qwen_cli, "_auto_extract_memory"),
        patch.object(qwen_cli, "_intel_process_queue"),
        patch.object(qwen_cli, "_intel_extract_topics"),
        patch.object(qwen_cli, "_generate_session_title"),
        patch.object(qwen_cli, "_extract_runnable_code", return_value=None),
        patch.object(qwen_cli, "_looks_like_plan", return_value=True),
        patch.object(qwen_cli, "_main_llm_busy_lock", MagicMock()),
        patch("qwen_cli.core.repl.console", console),
    ):
        with patch.object(console, "input", return_value="n"):
            _run_turn_and_handle_reply(ctx, "make a plan")

    assert len(ctx.history) == 3
    assert ctx.history[-1]["role"] == "assistant"


# ==============================================================================
# _dispatch_command
# ==============================================================================


def test_dispatch_command_exit_returns_true():
    from qwen_cli.core.repl import _ReplContext, _dispatch_command

    ctx = _ReplContext([], "system", "client")
    result = _dispatch_command(ctx, "/exit", "")
    assert result is True


def test_dispatch_command_unknown_returns_false(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _dispatch_command

    ctx = _ReplContext([], "system", "client")
    result = _dispatch_command(ctx, "/nonexistent", "")
    assert result is False


def test_dispatch_command_help_returns_false(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _dispatch_command

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, legacy_windows=False)

    ctx = _ReplContext([], "system", "client")

    with patch.object(qwen_cli, "console", console):
        result = _dispatch_command(ctx, "/help", "")

    assert result is False


def test_dispatch_command_known_returns_false(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _dispatch_command

    ctx = _ReplContext([{"role": "user", "content": "hello"}], "system", "client")
    result = _dispatch_command(ctx, "/clear", "")
    assert result is False
    assert ctx.history == []


# ==============================================================================
# _repl_setup
# ==============================================================================


def test_repl_setup_returns_tuple(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _repl_setup

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, legacy_windows=False)

    with (
        patch("qwen_cli.core.repl._setup_tab_completion"),
        patch("qwen_cli.core.repl._make_pt_session"),
        patch.object(qwen_cli, "start_intel_crawlers"),
        patch.object(qwen_cli, "load_project_context"),
        patch.object(qwen_cli, "ACTIVE_BACKEND", "llama.cpp"),
        patch.object(qwen_cli, "BASE_URL", "http://localhost:8080"),
        patch.object(qwen_cli, "MODEL", "qwen2.5"),
        patch.object(qwen_cli, "_active_preset", "thinking"),
        patch.object(qwen_cli, "AUTO_SEARCH_MODE", "off"),
        patch.object(qwen_cli, "DATA_DIR", "/tmp/data"),
        patch.object(qwen_cli, "_HAS_PT", False),
        patch.object(qwen_cli, "_INTEL_CRAWLERS", 3),
        patch.object(qwen_cli, "BASE_SYSTEM", "test system"),
        patch.object(qwen_cli, "KEY_FILES", frozenset()),
        patch("qwen_cli.core.repl.console", console),
    ):
        base_system, history, ctx = _repl_setup("fake_client")

    assert base_system == "test system"
    assert history == []
    assert isinstance(ctx, _ReplContext)
    output = buf.getvalue()
    assert "Qwen CLI" in output or "Model" in output


# ==============================================================================
# _repl_loop
# ==============================================================================


def test_repl_loop_exit_on_slash_exit(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _repl_loop

    ctx = _ReplContext([], "system", "client")

    with (
        patch("qwen_cli.core.repl.read_input", return_value="/exit"),
        patch.object(qwen_cli, "_watch_pending", []),
        patch.object(qwen_cli, "save_session"),
    ):
        _repl_loop(ctx, ctx.history, ctx.base_system)


def test_repl_loop_empty_input_continues(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _repl_loop

    ctx = _ReplContext([], "system", "client")
    inputs = iter(["", "/exit"])

    with (
        patch("qwen_cli.core.repl.read_input", side_effect=lambda: next(inputs)),
        patch.object(qwen_cli, "_watch_pending", []),
        patch.object(qwen_cli, "save_session"),
    ):
        _repl_loop(ctx, ctx.history, ctx.base_system)


def test_repl_loop_unknown_command_continues(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _repl_loop

    ctx = _ReplContext([], "system", "client")
    inputs = iter(["/undefined_cmd_test", "/exit"])

    with (
        patch("qwen_cli.core.repl.read_input", side_effect=lambda: next(inputs)),
        patch.object(qwen_cli, "_watch_pending", []),
        patch.object(qwen_cli, "save_session"),
    ):
        _repl_loop(ctx, ctx.history, ctx.base_system)


# ==============================================================================
# _cleanup_watch
# ==============================================================================


def test_cleanup_watch_sets_and_clears(qwen_cli):
    """_cleanup_watch calls set() then clear() on _watch_stop."""
    from qwen_cli.core.repl import _cleanup_watch

    qwen_cli._watch_stop.clear()
    qwen_cli._watched_files["/tmp/fake"] = 1.0
    qwen_cli._watch_pending.append("/tmp/fake")
    qwen_cli._watch_thread = None

    _cleanup_watch()

    assert not qwen_cli._watch_stop.is_set()
    assert qwen_cli._watched_files == {}
    assert qwen_cli._watch_pending == []
    assert qwen_cli._watch_thread is None


def test_cleanup_watch_with_alive_thread(qwen_cli):
    from qwen_cli.core.repl import _cleanup_watch

    qwen_cli._watch_stop.clear()
    qwen_cli._watched_files.clear()
    qwen_cli._watch_pending.clear()

    def _dummy_thread():
        while not qwen_cli._watch_stop.is_set():
            qwen_cli._watch_stop.wait(timeout=0.1)

    t = threading.Thread(target=_dummy_thread, daemon=True)
    t.start()
    qwen_cli._watch_thread = t

    _cleanup_watch()

    assert not qwen_cli._watch_stop.is_set()
    assert qwen_cli._watched_files == {}
    assert qwen_cli._watch_pending == []
    assert qwen_cli._watch_thread is None


# ==============================================================================
# _setup_tab_completion
# ==============================================================================


def test_setup_tab_completion_noop_when_rl_none(qwen_cli):
    from qwen_cli.core.repl import _setup_tab_completion

    with patch.object(qwen_cli, "_rl", None):
        _setup_tab_completion()


def test_setup_tab_completion_sets_completer(qwen_cli):
    from qwen_cli.core.repl import _setup_tab_completion

    mock_rl = MagicMock()
    with (
        patch.object(qwen_cli, "_rl", mock_rl),
        patch.object(qwen_cli, "_COMMANDS", ["/help", "/exit", "/clear"]),
    ):
        _setup_tab_completion()

    mock_rl.set_completer.assert_called_once()
    mock_rl.parse_and_bind.assert_called_once_with("tab: complete")


# ==============================================================================
# _make_pt_session
# ==============================================================================


def test_make_pt_session_noop_when_no_pt(qwen_cli):
    from qwen_cli.core.repl import _make_pt_session

    with patch.object(qwen_cli, "_HAS_PT", False):
        _make_pt_session()

    assert True


# ==============================================================================
# _ReplContext as SimpleNamespace (matching test_lsp.py pattern)
# ==============================================================================


def test_dispatch_command_via_simple_namespace_ctx():
    from qwen_cli.core.repl import _dispatch_command

    ctx = SimpleNamespace()
    ctx.history = []
    ctx.base_system = "system"
    ctx.client = "fake"

    result = _dispatch_command(ctx, "/exit", "")
    assert result is True


def test_dispatch_command_clear_via_simple_namespace():
    from qwen_cli.core.repl import _dispatch_command

    ctx = SimpleNamespace()
    ctx.history = [{"role": "user", "content": "test"}]
    ctx.base_system = "system"
    ctx.client = "fake"

    result = _dispatch_command(ctx, "/clear", "")

    assert result is False
    assert ctx.history == []


# ==============================================================================
# _repl_loop — watch_pending injection
# ==============================================================================


def test_repl_loop_watch_pending_injected(qwen_cli, tmp_path):
    from qwen_cli.core.repl import _ReplContext, _repl_loop

    test_file = tmp_path / "inject.py"
    test_file.write_text("print('hello')", encoding="utf-8")

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, legacy_windows=False)

    ctx = _ReplContext([], "system", "client")
    inputs = iter(["/exit"])

    with (
        patch("qwen_cli.core.repl.read_input", side_effect=lambda: next(inputs)),
        patch.object(qwen_cli, "_watch_pending", [str(test_file)]),
        patch.object(qwen_cli, "save_session"),
        patch.object(qwen_cli, "LANG_MAP", {".py": "python"}),
        patch("qwen_cli.core.repl.console", console),
    ):
        _repl_loop(ctx, ctx.history, ctx.base_system)

    output = buf.getvalue()
    assert "inject.py" in output
    assert "updated" in output or "watch" in output.lower()


# ==============================================================================
# _run_turn_and_handle_reply — runnable code detection
# ==============================================================================


def test_run_turn_and_handle_reply_runnable_code(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, legacy_windows=False)

    ctx = _ReplContext([], "system", "client")

    with (
        patch("qwen_cli.main.AUTO_SAVE_INTERVAL", 5, create=True),
        patch.object(qwen_cli, "_maybe_autocompact", side_effect=lambda h, bs, c: h),
        patch.object(qwen_cli, "build_system_prompt", return_value="sys"),
        patch.object(qwen_cli, "run_turn", return_value="```python\nprint(1)\n```"),
        patch.object(qwen_cli, "_print_turn_footer"),
        patch.object(qwen_cli, "_confidence_warning"),
        patch.object(qwen_cli, "_silent_autosave"),
        patch.object(qwen_cli, "_auto_extract_memory"),
        patch.object(qwen_cli, "_intel_process_queue"),
        patch.object(qwen_cli, "_intel_extract_topics"),
        patch.object(qwen_cli, "_generate_session_title"),
        patch.object(qwen_cli, "_extract_runnable_code", return_value=("python", "print(1)")),
        patch.object(qwen_cli, "_looks_like_plan", return_value=False),
        patch.object(qwen_cli, "_main_llm_busy_lock", MagicMock()),
        patch("qwen_cli.core.repl.console", console),
    ):
        with patch.object(console, "input", return_value="c"):
            _run_turn_and_handle_reply(ctx, "write code")

    assert len(ctx.history) == 2


# ==============================================================================
# _run_turn_and_handle_reply — autosave on turn_count % interval
# ==============================================================================


def test_run_turn_and_handle_reply_autosave(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

    ctx = _ReplContext([], "system", "client")
    # After _turn_count += 1 inside the function, 1 + 1 = 2, and 2 % 2 == 0
    qwen_cli._turn_count = 1

    with (
        patch("qwen_cli.main.AUTO_SAVE_INTERVAL", 2, create=True),
        patch.object(qwen_cli, "_maybe_autocompact", side_effect=lambda h, bs, c: h),
        patch.object(qwen_cli, "build_system_prompt", return_value="sys"),
        patch.object(qwen_cli, "run_turn", return_value="hello"),
        patch.object(qwen_cli, "_print_turn_footer"),
        patch.object(qwen_cli, "_confidence_warning"),
        patch.object(qwen_cli, "_silent_autosave") as mock_autosave,
        patch.object(qwen_cli, "_auto_extract_memory"),
        patch.object(qwen_cli, "_intel_process_queue"),
        patch.object(qwen_cli, "_intel_extract_topics"),
        patch.object(qwen_cli, "_generate_session_title"),
        patch.object(qwen_cli, "_extract_runnable_code", return_value=None),
        patch.object(qwen_cli, "_looks_like_plan", return_value=False),
        patch.object(qwen_cli, "_main_llm_busy_lock", MagicMock()),
    ):
        _run_turn_and_handle_reply(ctx, "test")

    mock_autosave.assert_called_once()
