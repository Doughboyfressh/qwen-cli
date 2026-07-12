"""Tests for the REPL module — context, input, dispatch, watch, and loop setup."""

import io
import threading
import time
from contextlib import ExitStack
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
# _run_turn_and_handle_reply — low-confidence self-correction
#
# _confidence_warning() only *tells* the user a reply hedged heavily; it
# doesn't do anything about it. If the model hedged and never searched to
# verify, one grounding pass should be forced automatically instead of
# handing back an unverified answer.
# ==============================================================================

_HEDGED_REPLY = "It might be v3. It's probably outdated. Perhaps check again. I think that's right."


def _patch_common(qwen_cli, run_turn_side_effect):
    """Returns an already-populated ExitStack so callers can do `with _patch_common(...):`."""
    stack = ExitStack()
    for cm in (
        patch("qwen_cli.main.AUTO_SAVE_INTERVAL", 5, create=True),
        patch.object(qwen_cli, "_maybe_autocompact", side_effect=lambda h, bs, c: h),
        patch.object(qwen_cli, "build_system_prompt", return_value="sys"),
        patch.object(qwen_cli, "run_turn", side_effect=run_turn_side_effect),
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
        patch.object(qwen_cli, "AUTO_SEARCH_MODE", "aggressive"),
    ):
        stack.enter_context(cm)
    return stack


def test_reverify_triggers_on_hedged_reply_without_search(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

    ctx = _ReplContext([], "system", "client")
    calls = []

    def fake_run_turn(client, messages, allow_tools=True, presearch=True):
        calls.append(list(messages))
        if len(calls) == 1:
            qwen_cli._last_turn_tool_names[:] = []  # no search performed
            return _HEDGED_REPLY
        return "It's version 3.2, confirmed via the changelog."

    with _patch_common(qwen_cli, fake_run_turn):
        _run_turn_and_handle_reply(ctx, "what version is it")

    assert len(calls) == 2  # original + one forced reverify pass
    assert ctx.history[-1]["content"] == "It's version 3.2, confirmed via the changelog."
    # The reverify prompt appended to the LLM messages, not to permanent history
    assert len(ctx.history) == 2
    assert any("hedging language" in (m.get("content") or "") for m in calls[1])


def test_reverify_skipped_when_search_already_ran(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

    ctx = _ReplContext([], "system", "client")
    calls = []

    def fake_run_turn(client, messages, allow_tools=True, presearch=True):
        calls.append(messages)
        qwen_cli._last_turn_tool_names[:] = ["web_search"]
        return _HEDGED_REPLY

    with _patch_common(qwen_cli, fake_run_turn):
        _run_turn_and_handle_reply(ctx, "what version is it")

    assert len(calls) == 1  # already searched — don't loop again
    assert ctx.history[-1]["content"] == _HEDGED_REPLY


def test_reverify_skipped_when_tools_disabled(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

    ctx = _ReplContext([], "system", "client")
    calls = []

    def fake_run_turn(client, messages, allow_tools=True, presearch=True):
        calls.append(messages)
        qwen_cli._last_turn_tool_names[:] = []
        return _HEDGED_REPLY

    with _patch_common(qwen_cli, fake_run_turn):
        _run_turn_and_handle_reply(ctx, "what version is it", allow_tools=False)

    assert len(calls) == 1  # -- no-tools mode: never force a search
    assert ctx.history[-1]["content"] == _HEDGED_REPLY


def test_reverify_skipped_when_auto_search_off(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

    ctx = _ReplContext([], "system", "client")
    calls = []

    def fake_run_turn(client, messages, allow_tools=True, presearch=True):
        calls.append(messages)
        qwen_cli._last_turn_tool_names[:] = []
        return _HEDGED_REPLY

    with _patch_common(qwen_cli, fake_run_turn) as stack:
        stack.enter_context(patch.object(qwen_cli, "AUTO_SEARCH_MODE", "off"))
        _run_turn_and_handle_reply(ctx, "what version is it")

    assert len(calls) == 1
    assert ctx.history[-1]["content"] == _HEDGED_REPLY


def test_reverify_falls_back_to_original_when_revised_is_empty(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

    ctx = _ReplContext([], "system", "client")
    calls = []

    def fake_run_turn(client, messages, allow_tools=True, presearch=True):
        calls.append(messages)
        qwen_cli._last_turn_tool_names[:] = []
        return _HEDGED_REPLY if len(calls) == 1 else None

    with _patch_common(qwen_cli, fake_run_turn):
        _run_turn_and_handle_reply(ctx, "what version is it")

    assert len(calls) == 2
    assert ctx.history[-1]["content"] == _HEDGED_REPLY  # kept the original, not lost it


def test_reverify_not_triggered_when_reply_is_confident(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _run_turn_and_handle_reply

    ctx = _ReplContext([], "system", "client")
    calls = []

    def fake_run_turn(client, messages, allow_tools=True, presearch=True):
        calls.append(messages)
        qwen_cli._last_turn_tool_names[:] = []
        return "It's version 3.2."

    with _patch_common(qwen_cli, fake_run_turn):
        _run_turn_and_handle_reply(ctx, "what version is it")

    assert len(calls) == 1


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
        patch.object(qwen_cli, "_consume_handoff", return_value=None),
        patch("qwen_cli.core.repl.console", console),
    ):
        base_system, history, ctx = _repl_setup("fake_client")

    assert base_system == "test system"
    assert history == []
    assert isinstance(ctx, _ReplContext)
    output = buf.getvalue()
    assert "Qwen CLI" in output or "Model" in output


def test_repl_setup_injects_handoff_when_present(qwen_cli):
    """_consume_handoff() reads (and deletes) whatever the previous session left
    behind on exit or crash. _repl_setup() must inject it into history the same
    way load_project_context() injects the project tree — as a "user"-role
    message the model sees on the next turn."""
    from qwen_cli.core.repl import _repl_setup

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, legacy_windows=False)
    handoff = {"prompt": "Resuming: you were mid-refactor of foo.py", "last_user": "refactor foo.py", "turns": 4}

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
        patch.object(qwen_cli, "_consume_handoff", return_value=handoff),
        patch("qwen_cli.core.repl.console", console),
    ):
        _base_system, history, _ctx = _repl_setup("fake_client")

    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Resuming: you were mid-refactor of foo.py"
    assert "Resuming from previous session" in buf.getvalue()


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


def test_cmd_exit_writes_handoff_with_nonempty_history(qwen_cli):
    from qwen_cli.core.commands import _cmd_exit
    from qwen_cli.core.repl import _ReplContext

    ctx = _ReplContext([{"role": "user", "content": "hi"}], "system", "client")
    with (
        patch.object(qwen_cli, "save_session"),
        patch.object(qwen_cli, "_save_exit_handoff") as mock_handoff,
        patch.object(qwen_cli, "record_session_changes_memory"),
        pytest.raises(StopIteration),
    ):
        _cmd_exit(ctx, "")
    mock_handoff.assert_called_once_with(ctx.history)


def test_cmd_exit_skips_handoff_with_empty_history(qwen_cli):
    from qwen_cli.core.commands import _cmd_exit
    from qwen_cli.core.repl import _ReplContext

    ctx = _ReplContext([], "system", "client")
    with (
        patch.object(qwen_cli, "save_session"),
        patch.object(qwen_cli, "_save_exit_handoff") as mock_handoff,
        patch.object(qwen_cli, "record_session_changes_memory"),
        pytest.raises(StopIteration),
    ):
        _cmd_exit(ctx, "")
    mock_handoff.assert_not_called()


def test_repl_loop_eof_writes_handoff_with_nonempty_history(qwen_cli):
    from qwen_cli.core.repl import _ReplContext, _repl_loop

    ctx = _ReplContext([{"role": "user", "content": "hi"}], "system", "client")

    def _raise_eof():
        raise EOFError

    with (
        patch("qwen_cli.core.repl.read_input", side_effect=_raise_eof),
        patch.object(qwen_cli, "_watch_pending", []),
        patch.object(qwen_cli, "save_session"),
        patch.object(qwen_cli, "_save_exit_handoff") as mock_handoff,
        patch.object(qwen_cli, "record_session_changes_memory"),
    ):
        _repl_loop(ctx, ctx.history, ctx.base_system)

    mock_handoff.assert_called_once_with(ctx.history)


def test_main_crash_path_writes_handoff(qwen_cli):
    """main()'s fatal-error handler already autosaves on a crash; it must also
    write a handoff so the *next* session starts with context instead of
    silently losing it."""
    history_marker = [{"role": "user", "content": "mid-task"}]

    with (
        patch.object(qwen_cli, "make_client", return_value="fake_client"),
        patch("sys.stdin") as mock_stdin,
        patch.object(qwen_cli, "_repl_setup", return_value=("system", history_marker, "fake_ctx")),
        patch.object(qwen_cli, "_repl_loop", side_effect=RuntimeError("boom")),
        patch.object(qwen_cli, "_silent_autosave") as mock_autosave,
        patch.object(qwen_cli, "_save_exit_handoff") as mock_handoff,
        patch("threading.Thread"),
        patch("qwen_cli.core.context.clean_old_snapshots", return_value=0),
        # Hermetic: a real qwen-cli session on the dev box holds qwen-cli.lock
        # and the guard would prompt for input mid-test.
        patch.object(qwen_cli, "_acquire_repl_lock", return_value=True),
    ):
        mock_stdin.isatty.return_value = True
        qwen_cli.main()

    mock_autosave.assert_called_once()
    mock_handoff.assert_called_once_with(history_marker)


def test_main_skips_lsp_prewarm_for_task_mode(qwen_cli):
    """--task (team_spawn_agent's spawned agents) essentially never touch /lsp
    diagnostics — starting the prewarm thread for every spawned agent in a
    team is pure overhead, and one that compounds badly if the LSP handshake
    is broken on the host machine (each spawned agent would otherwise burn a
    thread on a doomed ~20s jedi-language-server attempt)."""
    with (
        patch.object(qwen_cli, "make_client", return_value="fake_client"),
        patch("sys.stdin") as mock_stdin,
        patch("sys.argv", ["qwen-cli.py", "--task", "do something"]),
        patch.object(qwen_cli, "expand_at_refs", side_effect=lambda t: t),
        patch.object(qwen_cli, "cmd_agent"),
        patch("threading.Thread") as mock_thread,
        patch("qwen_cli.core.context.clean_old_snapshots", return_value=0),
    ):
        mock_stdin.isatty.return_value = True
        qwen_cli.main()

    prewarm_calls = [c for c in mock_thread.call_args_list if c.kwargs.get("name") == "lsp-prewarm"]
    assert prewarm_calls == []


def test_main_starts_lsp_prewarm_for_interactive_mode(qwen_cli):
    with (
        patch.object(qwen_cli, "make_client", return_value="fake_client"),
        patch("sys.stdin") as mock_stdin,
        patch("sys.argv", ["qwen-cli.py"]),
        patch.object(qwen_cli, "_repl_setup", return_value=("system", [], "fake_ctx")),
        patch.object(qwen_cli, "_repl_loop"),
        patch("threading.Thread") as mock_thread,
        patch("qwen_cli.core.context.clean_old_snapshots", return_value=0),
        # Hermetic: without this, a REAL qwen-cli session running on the dev
        # box holds qwen-cli.lock and the guard prompts for input mid-test.
        patch.object(qwen_cli, "_acquire_repl_lock", return_value=True),
    ):
        mock_stdin.isatty.return_value = True
        qwen_cli.main()

    prewarm_calls = [c for c in mock_thread.call_args_list if c.kwargs.get("name") == "lsp-prewarm"]
    assert len(prewarm_calls) == 1


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
        patch.object(qwen_cli, "_save_exit_handoff"),
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
        patch("qwen_cli.core.config.AUTO_SAVE_INTERVAL", 2),
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


class TestHedgeRegex:
    """Recommendation modals must not count as hedging: a live session saw an
    improvement-list answer ("could add X", "could use more tests") trip the
    forced re-search. Only modal+be/have reads as factual uncertainty."""

    def test_recommendations_are_not_hedging(self, qwen_cli):
        text = (
            "You could add rate limiting. The error messages could use more detail. "
            "Splitting main.py would help; you may want to start with tool dispatch. "
            "I could not find any dead code."
        )
        assert len(qwen_cli._HEDGE_RE.findall(text)) == 0

    def test_factual_uncertainty_still_counts(self, qwen_cli):
        text = "It might be v3. It could have changed. This may be outdated. Probably fine."
        assert len(qwen_cli._HEDGE_RE.findall(text)) >= 3
